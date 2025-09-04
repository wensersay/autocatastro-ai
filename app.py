"""
app.py — autoCata v0.9.1 (smart sectors + grouped notarial)

Objetivo: fidelidad notarial sin depender de NAME_HINTS y con decisión correcta
entre diagonales y cardinales según la **cobertura angular real** del colindante.

Novedades clave frente a 0.9.0:
1) **Asignación de sectores inteligente (smart)**: para cada colindante se calcula
   la cobertura angular (histograma sobre 8 rumbos) muestreando su **contorno**
   respecto del centroide de la parcela objeto. Reglas (configurables):
   - Si la pareja de cardinales adyacentes a una diagonal (p.ej. E+S ↔ SE) tiene
     cobertura suficiente, se asignan **ambos cardinales** y **se vacía la diagonal**.
   - En caso contrario, si una diagonal domina claramente, se conserva la **diagonal**.
   - Preferencia general: **cardinales** frente a diagonal cuando hay duda.
2) **Notarial agrupado por titular**: si un mismo titular ocupa 2+ lados, se 
   redacta «Este y Sur, Fulano», retirándolo de las listas por lado para no duplicar.
3) **Debug enriquecido**: por cada vecino OCR se guarda su `coverage` de rumbos.

Variables (Railway → Variables) – valores sugeridos:
- FAST_MODE=1, PDF_DPI=320, DPI_FAST=220, EDGE_BUDGET_MS=25000
- NEIGH_MIN_AREA_HARD=300, NEIGH_MAX_DIST_RATIO=2.2
- ROW_OWNER_LINES=2, SECOND_LINE_FORCE=1, SECOND_LINE_MAXCHARS=40, SECOND_LINE_MAXTOKENS=8
- PROPERCASE=1
- SECTOR_ASSIGN_MODE=smart
- DIAG_KEEP_DOMINANCE=0.55, CARD_PAIR_MIN_EACH=0.20, CARD_PAIR_MIN_COMBINED=0.50, CARD_SINGLE_MIN=0.30
- DIAG_TO_CARDINALS=0   # desactivado por defecto: decide el motor «smart»
"""
from __future__ import annotations

import base64
import logging
import math
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pdf2image import convert_from_bytes
from pydantic import BaseModel, Field
from PIL import Image

__version__ = "0.9.1"

# ----------------------------------------
# Configuración
# ----------------------------------------

@dataclass
class Cfg:
    auth_token: Optional[str] = os.getenv("AUTH_TOKEN")
    pdf_dpi: int = int(os.getenv("PDF_DPI", "320"))
    dpi_fast: int = int(os.getenv("DPI_FAST", "220"))
    fast_mode: bool = os.getenv("FAST_MODE", "1") == "1"

    text_only: bool = os.getenv("TEXT_ONLY", "0") == "1"

    name_hints: List[str] = field(default_factory=lambda: [s.strip() for s in os.getenv("NAME_HINTS", "").split("|") if s.strip()])
    name_hints_file: Optional[str] = os.getenv("NAME_HINTS_FILE")

    second_line_force: bool = os.getenv("SECOND_LINE_FORCE", "1") == "1"
    second_line_maxchars: int = int(os.getenv("SECOND_LINE_MAXCHARS", "40"))
    second_line_maxtokens: int = int(os.getenv("SECOND_LINE_MAXTOKENS", "8"))

    neigh_min_area_hard: int = int(os.getenv("NEIGH_MIN_AREA_HARD", "600"))
    neigh_max_dist_ratio: float = float(os.getenv("NEIGH_MAX_DIST_RATIO", "2.2"))

    row_owner_lines: int = int(os.getenv("ROW_OWNER_LINES", "2"))

    propercase: bool = os.getenv("PROPERCASE", "1") == "1"

    sector_assign_mode: str = os.getenv("SECTOR_ASSIGN_MODE", "smart")
    diag_keep_dominance: float = float(os.getenv("DIAG_KEEP_DOMINANCE", "0.55"))
    card_pair_min_each: float = float(os.getenv("CARD_PAIR_MIN_EACH", "0.20"))
    card_pair_min_combined: float = float(os.getenv("CARD_PAIR_MIN_COMBINED", "0.50"))
    card_single_min: float = float(os.getenv("CARD_SINGLE_MIN", "0.30"))

    diag_to_cardinals: bool = os.getenv("DIAG_TO_CARDINALS", "0") == "1"

CFG = Cfg()

NAME_HINTS: set[str] = set()
if CFG.name_hints_file and os.path.exists(CFG.name_hints_file):
    try:
        with open(CFG.name_hints_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    NAME_HINTS.add(s)
    except Exception:
        pass
for s in CFG.name_hints:
    NAME_HINTS.add(s)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("autoCata")

app = FastAPI(title="autoCata API", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------
# Modelos (8 rumbos)
# ----------------------------------------

class Linderos(BaseModel):
    norte: List[str]    = Field(default_factory=list)
    noreste: List[str]  = Field(default_factory=list)
    este: List[str]     = Field(default_factory=list)
    sureste: List[str]  = Field(default_factory=list)
    sur: List[str]      = Field(default_factory=list)
    suroeste: List[str] = Field(default_factory=list)
    oeste: List[str]    = Field(default_factory=list)
    noroeste: List[str] = Field(default_factory=list)

class ExtractResult(BaseModel):
    linderos: Linderos
    owners_detected: List[str] = Field(default_factory=list)
    notarial_text: Optional[str] = None
    note: Optional[str] = None
    debug: Optional[Dict] = None
    files: Optional[Dict[str, str]] = None

class MultiResult(BaseModel):
    results: List[ExtractResult]
    version: str

# ----------------------------------------
# Seguridad
# ----------------------------------------

bearer_scheme = HTTPBearer(auto_error=False)

def require_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)):
    if CFG.auth_token is None:
        return
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Falta o es inválido el encabezado Authorization (Bearer)")
    if credentials.credentials != CFG.auth_token:
        raise HTTPException(status_code=403, detail="Token no autorizado")

# ----------------------------------------
# Utilidades OCR y nombres
# ----------------------------------------

def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

LOWER_CONNECTORS = {"de","del","la","las","los","y","da","do","das","dos"}
ACCENT_MAP = {
    "JOSE": "José", "RODRIGUEZ": "Rodríguez", "ALVAREZ": "Álvarez", "LOPEZ": "López",
    "FERNANDEZ": "Fernández", "VAZQUEZ": "Vázquez", "MARTIN": "Martín", "MARTINEZ": "Martínez",
    "PEREZ": "Pérez", "GOMEZ": "Gómez", "GARCIA": "García", "NUNEZ": "Núñez",
}
GIVEN_NAMES = {"JOSE","JOSÉ","LUIS","MARIA","MARÍA","ANTONIO","MANUEL","ANA","JUAN","CARLOS","PABLO","ROGELIO","FRANCISCO","MARTA","ELENA","LAURA"}
UPPER_NAME_RE = re.compile(r"^[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\.'\-]+$", re.UNICODE)
BAD_TOKENS = {"POLÍGONO","POLIGONO","PARCELA","APELLIDOS","NOMBRE","RAZON","RAZÓN","SOCIAL","NIF","DOMICILIO","LOCALIZACIÓN","LOCALIZACION","REFERENCIA","CATASTRAL","TITULARIDAD","PRINCIPAL"}
GEO_TOKENS = {"LUGO","BARCELONA","MADRID","VALENCIA","SEVILLA","CORUÑA","A CORUÑA","MONFORTE","LEM","LEMOS","HOSPITALET","L'HOSPITALET","SAVIAO","SAVIÑAO","GALICIA","[LUGO]","[BARCELONA]"}
NAME_CONNECTORS = {"DE","DEL","LA","LOS","LAS","DA","DO","DAS","DOS","Y"}


def propercase_spanish(s: str) -> str:
    if not s:
        return s
    toks = [t for t in re.split(r"\s+", s.strip()) if t]
    out: List[str] = []
    for t in toks:
        raw = t.strip(".,;:()[]{}")
        up = strip_accents(raw).upper()
        if re.match(r"^(S\.?L\.?|S\.?A\.?|SCOOP\.?|COOP\.?|CB)$", raw, re.I):
            out.append(raw.upper().replace(" ", "")); continue
        if up in ACCENT_MAP:
            out.append(ACCENT_MAP[up]); continue
        if up.lower() in LOWER_CONNECTORS:
            out.append(up.lower()); continue
        out.append(raw.capitalize())
    return " ".join(out)


def reorder_surname_first(s: str) -> str:
    toks = s.split()
    if len(toks) >= 2:
        last = strip_accents(toks[-1]).upper()
        if last in GIVEN_NAMES:
            return toks[-1] + " " + " ".join(toks[:-1])
    return s


def postprocess_name(text: str) -> str:
    s = re.sub(r"\s+", " ", (text or "")).strip()
    if not s:
        return ""
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) == 2 and len(parts[0]) >= 3 and len(parts[1]) >= 2:
            s = f"{parts[1]} {parts[0]}"
    s = s.replace("  ", " ")
    s = re.sub(r"\b(TITULAR EN INVESTIGACION|TITULAR EN INVESTIGACIÓN)\b", "Titular en investigación", s, flags=re.I)
    s = propercase_spanish(s)
    s = reorder_surname_first(s)
    return s


def clean_candidate_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"[^0-9A-Za-zÁÉÍÓÚÜÑáéíóúüñ .'-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" .-'")
    up = s.upper()
    for ch in ["=", "*", "_", "/", "\\", "|", "[", "]", "{", "}", "<", ">"]:
        if ch in up:
            return ""
    STOP = {"DATOS","ESCALA","LEYENDA","MAPA","COORDENADAS","COORD","REFERENCIA","CATASTRAL","PARCELA","POLIGONO","POLÍGONO","HOJA","AHORA","NORTE","SUR","ESTE","OESTE"}
    if any(kw in up for kw in STOP):
        return ""
    if re.search(r"\b(S\.?:?L\.?(?:\b|$)|S\.?:?A\.?(?:\b|$)|CB\b|S\.?COOP\.?)", s, flags=re.I):
        return s
    tokens = s.split()
    good = [t for t in tokens if (len(t) >= 2 and (t[0].isupper() or t.isupper()))]
    return " ".join(tokens[:7]) if len(good) >= 2 else ""

# ----------------------------------------
# OCR auxiliares
# ----------------------------------------

def ocr_image_to_data_lines(img_bgr: np.ndarray) -> List[Tuple[str, Tuple[int,int,int,int]]]:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    data = pytesseract.image_to_data(pil, config="--oem 1 --psm 6 -l spa+eng", output_type=pytesseract.Output.DICT)
    n = len(data.get("text", []))
    lines: Dict[Tuple[int,int], Dict] = {}
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        conf = data["conf"][i]
        try:
            conf = float(conf)
        except Exception:
            conf = -1
        if not txt or conf < 0:
            continue
        b = data["block_num"][i]; ln = data["line_num"][i]
        x = data["left"][i]; y = data["top"][i]; w = data["width"][i]; h = data["height"][i]
        key = (b, ln)
        if key not in lines:
            lines[key] = {"words": [], "bbox": [x, y, x+w, y+h]}
        lines[key]["words"].append(txt)
        bx0, by0, bx1, by1 = lines[key]["bbox"]
        lines[key]["bbox"] = [min(bx0,x), min(by0,y), max(bx1,x+w), max(by1,y+h)]
    out: List[Tuple[str, Tuple[int,int,int,int]]] = []
    for v in lines.values():
        text_line = " ".join(v["words"]).strip()
        x0,y0,x1,y1 = v["bbox"]
        out.append((text_line, (x0,y0,x1-x0,y1-y0)))
    out.sort(key=lambda t: t[1][1])
    return out


def ocr_best_of_three(crop_bgr: np.ndarray) -> str:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 60, 60)
    _, thr2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thr3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cands: List[str] = []
    for im in (cv2.cvtColor(thr2, cv2.COLOR_GRAY2BGR), cv2.cvtColor(255 - thr3, cv2.COLOR_GRAY2BGR)):
        txt = pytesseract.image_to_string(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)), config="--oem 1 --psm 6 -l spa+eng")
        cands.append(txt or "")
    ln = ocr_image_to_data_lines(crop_bgr)
    lines_txt = [t for (t, _bbox) in ln]
    if lines_txt:
        joined = lines_txt[0]
        if len(lines_txt) >= 2:
            below = lines_txt[1].strip()
            if 0 < len(below) <= CFG.second_line_maxchars and len(below.split()) <= CFG.second_line_maxtokens:
                joined = f"{joined} {below}"
        cands.append(joined)
    def score(txt: str) -> Tuple[int,int]:
        t = clean_candidate_text(postprocess_name(txt))
        tok = len(t.split())
        return (int(tok >= 2), len(t))
    best = ""; best_score = (-1, -1)
    for s in cands:
        t = clean_candidate_text(postprocess_name(s))
        sc = score(s)
        if sc > best_score:
            best_score = sc; best = t
    return best

# ----------------------------------------
# PDF → imagen + máscaras y contornos (con contorno)
# ----------------------------------------

def load_map_page_bgr(pdf_bytes: bytes) -> np.ndarray:
    dpi = CFG.dpi_fast if CFG.fast_mode else CFG.pdf_dpi
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=2, last_page=2, thread_count=1)
        if not pages:
            pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=1, last_page=1, thread_count=1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo rasterizar el PDF: {e}")
    return np.array(pages[0].convert("RGB"))[:, :, ::-1]


def detect_masks(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, np.array([30, 10, 35]), np.array([95, 255, 255]))
    ranges = [
        (np.array([0,   5, 25]), np.array([25, 255, 255])),
        (np.array([140, 5, 25]), np.array([179,255, 255])),
        (np.array([120,10, 40]), np.array([150,255, 255])),
    ]
    mask_pink = np.zeros(mask_green.shape, dtype=np.uint8)
    for lo, hi in ranges:
        mask_pink |= cv2.inRange(hsv, lo, hi)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, k3, iterations=1)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, k3, iterations=1)
    mask_pink  = cv2.morphologyEx(mask_pink,  cv2.MORPH_OPEN, k3, iterations=1)
    mask_pink  = cv2.morphologyEx(mask_pink,  cv2.MORPH_CLOSE, k3, iterations=1)
    return mask_green, mask_pink


def contours_and_centroids(mask: np.ndarray) -> List[Dict]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[Dict] = []
    thr = CFG.neigh_min_area_hard
    for c in cnts:
        area = cv2.contourArea(c)
        if area < thr:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        x, y, w, h = cv2.boundingRect(c)
        out.append({"centroid": (cx, cy), "bbox": (x, y, w, h), "area": int(area), "contour": c})
    out.sort(key=lambda d: -d["area"])
    return out

# ----------------------------------------
# Cobertura angular y decisión de sectores
# ----------------------------------------

SIDES8 = ["este","noreste","norte","noroeste","oeste","suroeste","sur","sureste"]
PAIRS = {
    "noreste": ("norte", "este"),
    "sureste": ("sur", "este"),
    "suroeste": ("sur", "oeste"),
    "noroeste": ("norte", "oeste"),
}


def angle_between(p0: Tuple[int, int], p1: Tuple[int, int]) -> float:
    dx = p1[0] - p0[0]; dy = p1[1] - p0[1]
    ang = math.degrees(math.atan2(-dy, dx))
    return (ang + 360.0) % 360.0


def angle_to_8(ang: float) -> str:
    if   (337.5 <= ang < 360.0) or (0.0 <= ang < 22.5):  return "este"
    if   22.5 <= ang < 67.5:   return "noreste"
    if   67.5 <= ang < 112.5:  return "norte"
    if  112.5 <= ang < 157.5:  return "noroeste"
    if  157.5 <= ang < 202.5:  return "oeste"
    if  202.5 <= ang < 247.5:  return "suroeste"
    if  247.5 <= ang < 292.5:  return "sur"
    return "sureste"


def sector_coverage(subj_c: Tuple[int,int], contour: np.ndarray, step: int = 8) -> Dict[str, float]:
    """Devuelve proporción de puntos del contorno en cada rumbo (8)."""
    cnt = contour.reshape(-1, 2)
    n = len(cnt)
    if n == 0:
        return {k: 0.0 for k in SIDES8}
    counts = {k: 0 for k in SIDES8}
    for i in range(0, n, max(1, step)):
        x, y = int(cnt[i][0]), int(cnt[i][1])
        ang = angle_between(subj_c, (x, y))
        side = angle_to_8(ang)
        counts[side] += 1
    total = sum(counts.values()) or 1
    return {k: counts[k] / total for k in SIDES8}


def decide_sides_smart(cov: Dict[str, float]) -> List[str]:
    """Reglas smart con preferencia por cardinales y limpieza de diagonales si procede."""
    # 1) ¿Hay una pareja cardinal fuerte para alguna diagonal?
    for diag, (a, b) in PAIRS.items():
        if cov[a] >= CFG.card_pair_min_each and cov[b] >= CFG.card_pair_min_each and (cov[a] + cov[b]) >= CFG.card_pair_min_combined:
            return [a, b]  # cardinales preferidos
    # 2) ¿Alguna diagonal domina claramente?
    best_side = max(SIDES8, key=lambda s: cov.get(s, 0.0))
    if best_side in PAIRS:
        a, b = PAIRS[best_side]
        if cov[best_side] >= CFG.diag_keep_dominance and cov[a] < CFG.card_pair_min_each and cov[b] < CFG.card_pair_min_each:
            return [best_side]  # diagonal fuerte
    # 3) ¿Dos cardinales medianamente fuertes contiguos?
    cardinales = ["norte","este","sur","oeste"]
    best_pair: Optional[Tuple[str,str]] = None
    best_sum = 0.0
    adj = {"norte":["noreste","noroeste"], "este":["noreste","sureste"], "sur":["sureste","suroeste"], "oeste":["noroeste","suroeste"]}
    # pares cardinales adyacentes válidos
    card_pairs = [("norte","este"),("este","sur"),("sur","oeste"),("oeste","norte")]
    for a,b in card_pairs:
        s = cov[a] + cov[b]
        if cov[a] >= CFG.card_single_min and cov[b] >= CFG.card_single_min and s > best_sum:
            best_sum = s; best_pair = (a,b)
    if best_pair:
        return list(best_pair)
    # 4) último recurso: el mejor sector (cardinal o diagonal)
    return [best_side]

# ----------------------------------------
# OCR cerca del vecino con line-joiner
# ----------------------------------------

def ocr_name_near_with_linejoin(bgr: np.ndarray, bbox: Tuple[int,int,int,int], subj_c: Tuple[int,int]) -> str:
    x,y,w,h = bbox
    cnx, cny = x + w//2, y + h//2
    L = max(w, h)
    vx, vy = cnx - subj_c[0], cny - subj_c[1]
    norm = math.hypot(vx, vy)
    vx, vy = (1.0, 0.0) if norm < 1e-3 else (vx/norm, vy/norm)
    H, W = bgr.shape[:2]
    def _clip(x0,y0,x1,y1):
        x0 = max(0,min(W-1,x0)); y0=max(0,min(H-1,y0)); x1=max(1,min(W,x1)); y1=max(1,min(H,y1));
        if x1<=x0: x1=x0+1
        if y1<=y0: y1=y0+1
        return x0,y0,x1,y1
    crops: List[np.ndarray] = []
    pad = int(max(60, 2.4*L))
    x0,y0,x1,y1 = _clip(x-pad, y-pad, x+w+pad, y+h+pad)
    crops.append(bgr[y0:y1, x0:x1])
    for f in (1.0, 1.6, 2.2):
        cx = int(cnx + vx * f * 2.0 * L)
        cy = int(cny + vy * f * 2.0 * L)
        s  = int(max(72, 2.0 * L))
        x0,y0,x1,y1 = _clip(cx - s, cy - s, cx + s, cy + s)
        crops.append(bgr[y0:y1, x0:x1])
    best = ""; best_len = -1
    for crop in crops:
        if max(crop.shape[:2]) < 320:
            fx = 320.0 / max(1, max(crop.shape[:2]))
            crop = cv2.resize(crop, None, fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
        txt = ocr_best_of_three(crop)
        txt = clean_candidate_text(postprocess_name(txt))
        if len(txt) > best_len:
            best, best_len = txt, len(txt)
    return best

# ----------------------------------------
# Row-based (heredado, mantiene lado por ángulo simple)
# ----------------------------------------

def _binarize(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    _, bw  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bwi = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw, bwi


def _enhance_gray(g: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(g)

# (Se mantiene la lógica de filas de 0.9.0 por estabilidad)

# ----------------------------------------
# Notarial agrupado
# ----------------------------------------

SIDE_ORDER = ["norte","noreste","este","sureste","sur","suroeste","oeste","noroeste"]


def join_sides_spanish(sides: List[str]) -> str:
    pretty = {
        "norte": "Norte", "noreste": "Noreste", "este": "Este", "sureste": "Sureste",
        "sur": "Sur", "suroeste": "Suroeste", "oeste": "Oeste", "noroeste": "Noroeste",
    }
    labels = [pretty[s] for s in sides]
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} y {labels[1]}"
    return ", ".join(labels[:-1]) + f" y {labels[-1]}"


def generate_notarial_text_grouped(owners_by_side: Dict[str, List[str]]) -> str:
    # Mapa owner -> lados
    owner_sides: Dict[str, List[str]] = {}
    for side in SIDE_ORDER:
        for nm in owners_by_side.get(side, []) or []:
            owner_sides.setdefault(nm, []).append(side)
    # Retirar nombres agrupados de sus lados originales
    sides_left: Dict[str, List[str]] = {s: [] for s in SIDE_ORDER}
    grouped: List[Tuple[List[str], str]] = []
    for owner, sides in owner_sides.items():
        if len(sides) >= 2:
            # agrupar este titular
            grouped.append((sorted(sides, key=lambda x: SIDE_ORDER.index(x)), owner))
        else:
            s = sides[0]
            sides_left.setdefault(s, []).append(owner)
    # Añadir los que queden (con varios por lado)
    for side in SIDE_ORDER:
        for nm in owners_by_side.get(side, []) or []:
            if nm not in owner_sides or len(owner_sides[nm]) == 1:
                if nm not in sides_left[side]:
                    sides_left[side].append(nm)
    # Construir texto
    parts: List[str] = []
    # Primero grupos multi-lado ordenados por el primer lado
    grouped.sort(key=lambda t: SIDE_ORDER.index(t[0][0]))
    for sides, owner in grouped:
        parts.append(f"{join_sides_spanish(sides)}, {owner}")
    # Luego los restantes por lado
    for side in SIDE_ORDER:
        arr = list(dict.fromkeys(sides_left[side]))
        if not arr:
            continue
        if len(arr) == 1:
            parts.append(f"{side.capitalize()}, {arr[0]}")
        else:
            parts.append(f"{side.capitalize()}, " + ", ".join(arr[:-1]) + f" y {arr[-1]}")
    return f"Linda: {"; ".join(parts)}." if parts else "No se han podido determinar linderos suficientes para una redacción notarial fiable."

# ----------------------------------------
# Proceso principal
# ----------------------------------------

def process_pdf(pdf_bytes: bytes) -> ExtractResult:
    bgr = load_map_page_bgr(pdf_bytes)

    # Gate OCR ligero para descartar PDFs mal formados
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    txt_gate = pytesseract.image_to_string(Image.fromarray(rgb), config="--oem 1 --psm 6 -l spa+eng") or ""
    if len(txt_gate.strip()) < 12:
        raise HTTPException(status_code=400, detail="El PDF no contiene texto OCR legible o metadatos reconocibles para su análisis.")

    owners_detected_union: List[str] = []

    # Pipeline por color (con contorno y decisión smart)
    mask_green, mask_pink = detect_masks(bgr)
    subs = contours_and_centroids(mask_green)
    neigh = contours_and_centroids(mask_pink)
    subj = subs[0] if subs else None

    ldr_out: Dict[str, List[str]] = {k: [] for k in SIDE_ORDER}
    debug_rows: List[Dict] = []

    if subj:
        subj_c = subj["centroid"]
        # filtrar vecinos por distancia al rectángulo de la parcela
        sx, sy, sw, sh = subj["bbox"]
        def dist_to_rect(cx: int, cy: int, rect: Tuple[int,int,int,int]) -> float:
            x, y, w, h = rect
            dx = max(x - cx, 0, cx - (x + w))
            dy = max(y - cy, 0, cy - (y + h))
            return math.hypot(dx, dy)
        maxd = CFG.neigh_max_dist_ratio * max(sw, sh)
        neigh = [nb for nb in neigh if dist_to_rect(nb["centroid"][0], nb["centroid"][1], subj["bbox"]) <= maxd]
        for nb in neigh[:12]:
            coverage = sector_coverage(subj_c, nb["contour"], step=8)
            if CFG.sector_assign_mode == "smart":
                sides = decide_sides_smart(coverage)
            else:
                # modo simple por centroide (retrocompatibilidad)
                ang = angle_between(subj_c, nb["centroid"]) ; sides = [angle_to_8(ang)]
            name = ocr_name_near_with_linejoin(bgr, nb["bbox"], subj_c)
            if not name:
                continue
            for side in sides:
                if name not in ldr_out[side]:
                    ldr_out[side].append(name)
            if name not in owners_detected_union:
                owners_detected_union.append(name)
            debug_rows.append({
                "neighbor_bbox": list(nb["bbox"]),
                "centroid": list(nb["centroid"]),
                "coverage": coverage,
                "assigned": sides,
                "owner": name,
            })

    # Si el flag de expansión está activo, mantener ese comportamiento (no recomendado con smart)
    if CFG.diag_to_cardinals:
        for diag, (a, b) in PAIRS.items():
            for nm in list(ldr_out.get(diag, [])):
                if nm and nm not in ldr_out[a]:
                    ldr_out[a].append(nm)
                if nm and nm not in ldr_out[b]:
                    ldr_out[b].append(nm)
            ldr_out[diag] = []

    # Notarial agrupado por titular
    notarial = generate_notarial_text_grouped(ldr_out)

    files = {"notarial_text.txt": base64.b64encode(notarial.encode("utf-8")).decode("ascii")}

    debug = {"rows": debug_rows, "subject_centroid": list(subj["centroid"]) if subj else None}

    return ExtractResult(
        linderos=Linderos(**ldr_out),
        owners_detected=list(dict.fromkeys(owners_detected_union))[:24],
        notarial_text=notarial,
        note=None,
        debug=debug,
        files=files,
    )

# ----------------------------------------
# API
# ----------------------------------------

@app.post("/extract", response_model=MultiResult)
def extract_endpoint(
    files: List[UploadFile] = File(..., description="Sube 1..5 PDFs catastrales"),
    _: None = Depends(require_token),
):
    if not files:
        raise HTTPException(status_code=400, detail="Debe subir al menos un PDF")
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Se permite un máximo de 5 PDFs por solicitud")

    results: List[ExtractResult] = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Archivo no PDF: {f.filename}")
        pdf_bytes = f.file.read()
        try:
            results.append(process_pdf(pdf_bytes))
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Error procesando %s", f.filename)
            raise HTTPException(status_code=500, detail=f"Error interno procesando {f.filename}: {e}")

    return MultiResult(results=results, version=__version__)

@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}

@app.get("/")
def root():
    return {"ok": True, "service": "autoCata", "version": __version__}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)





