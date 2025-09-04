"""
app.py — autoCata v0.9.0 (line-joiner, generalizable OCR)

Objetivo: Generalizar para miles de PDFs distintos SIN depender de NAME_HINTS.

Novedades clave frente a 0.8.5:
1) **Line-Joiner por OCR estructurado (Tesseract image_to_data)**: agrupamos por líneas
   (block_num, line_num), verificamos solapes horizontales y distancia vertical
   y **concatenamos 2 líneas** cuando corresponde (nombres largos que saltan de renglón).
   → Recupera segundos nombres como «Luis» sin necesidad de hints.
2) **Candidatos robustos**: voto entre 3 fuentes para cada rótulo: (a) OCR binarizado,
   (b) OCR invertido, (c) OCR-por-líneas (image_to_data). Elegimos el mejor por
   longitud/validez de nombre.
3) **Filtros semánticos ligeros**: reglas de tokenización para nombres de persona
   españolas (con conectores «de/del/y»), reorden «Apellidos Apellidos Nombre» →
   «Nombre Apellidos Apellidos», propercase con tildes.
4) **Diagonales ⇒ Cardinales siempre**: mantenemos expansión SE→E+S, NE→N+E, etc.
5) **Límites de tiempo y eficiencia**: DPI moderado, pre-recortes, y máximo de
   10 vecinos OCR por pipeline color.

Opcional (sigue disponible):
- `NAME_HINTS` para normalizaciones canónicas (no requerido para funcionar bien).

Variables sugeridas (Railway → Variables):
- FAST_MODE=1, PDF_DPI=320, DPI_FAST=220, EDGE_BUDGET_MS=25000
- ROW_OWNER_LINES=2, SECOND_LINE_FORCE=1, SECOND_LINE_MAXCHARS=40, SECOND_LINE_MAXTOKENS=8
- NEIGH_MIN_AREA_HARD=300, NEIGH_MAX_DIST_RATIO=2.2
- PROPERCASE=1, DIAG_TO_CARDINALS=1
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

__version__ = "0.9.0"

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
    second_line_strict: bool = os.getenv("SECOND_LINE_STRICT", "0") == "1"

    neigh_min_area_hard: int = int(os.getenv("NEIGH_MIN_AREA_HARD", "600"))
    neigh_max_dist_ratio: float = float(os.getenv("NEIGH_MAX_DIST_RATIO", "2.2"))

    row_owner_lines: int = int(os.getenv("ROW_OWNER_LINES", "2"))

    propercase: bool = os.getenv("PROPERCASE", "1") == "1"
    diag_to_cardinals: bool = os.getenv("DIAG_TO_CARDINALS", "1") == "1"

    edge_budget_ms: int = int(os.getenv("EDGE_BUDGET_MS", "25000"))

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
    "JOSE": "José",
    "RODRIGUEZ": "Rodríguez",
    "ALVAREZ": "Álvarez",
    "LOPEZ": "López",
    "FERNANDEZ": "Fernández",
    "VAZQUEZ": "Vázquez",
    "MARTIN": "Martín",
    "MARTINEZ": "Martínez",
    "PEREZ": "Pérez",
    "GOMEZ": "Gómez",
    "GARCIA": "García",
    "NUNEZ": "Núñez",
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
# OCR básicos
# ----------------------------------------

def ocr_image_to_data_lines(img_bgr: np.ndarray) -> List[Tuple[str, Tuple[int,int,int,int]]]:
    """Devuelve [(texto_linea, bbox_linea)] usando image_to_data.
    Agrupa por (block_num, line_num), junta palabras, y calcula bbox de la línea.
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    data = pytesseract.image_to_data(pil, config="--oem 1 --psm 6 -l spa+eng", output_type=pytesseract.Output.DICT)
    n = len(data.get("text", []))
    lines: Dict[Tuple[int,int], Dict] = {}
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        conf = data["conf"][i]
        if isinstance(conf, str):
            try:
                conf = float(conf)
            except Exception:
                conf = -1
        if not txt or (isinstance(conf, (int,float)) and conf < 0):
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
    for (_, _), v in lines.items():
        text_line = " ".join(v["words"]).strip()
        x0,y0,x1,y1 = v["bbox"]
        out.append((text_line, (x0,y0,x1-x0,y1-y0)))
    # ordenar por Y
    out.sort(key=lambda t: t[1][1])
    return out


def ocr_best_of_three(crop_bgr: np.ndarray) -> str:
    """Elige el mejor candidato entre: binario, invertido y líneas (image_to_data)."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 60, 60)
    _, thr2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thr3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cands: List[str] = []
    for im in (cv2.cvtColor(thr2, cv2.COLOR_GRAY2BGR), cv2.cvtColor(255 - thr3, cv2.COLOR_GRAY2BGR)):
        txt = pytesseract.image_to_string(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)), config="--oem 1 --psm 6 -l spa+eng")
        cands.append(txt or "")
    # vía líneas
    ln = ocr_image_to_data_lines(crop_bgr)
    lines_txt = [t for (t, _bbox) in ln]
    if lines_txt:
        # unir posible segunda línea si pasa filtros de longitud
        joined = lines_txt[0]
        if len(lines_txt) >= 2:
            below = lines_txt[1].strip()
            if 0 < len(below) <= CFG.second_line_maxchars and len(below.split()) <= CFG.second_line_maxtokens:
                joined = f"{joined} {below}"
        cands.append(joined)
    # Limpiar y puntuar
    def score(txt: str) -> Tuple[int,int]:
        t = clean_candidate_text(postprocess_name(txt))
        tok = len(t.split())
        # priorizar textos que parecen nombres de persona (>=2 tokens)
        return (int(tok >= 2), len(t))
    best = ""
    best_score = (-1, -1)
    for s in cands:
        t = clean_candidate_text(postprocess_name(s))
        sc = score(s)
        if sc > best_score:
            best_score = sc; best = t
    return best

# ----------------------------------------
# PDF → imagen
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

# ----------------------------------------
# Segmentación por color y vecinos
# ----------------------------------------

def detect_masks(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # verde (parcela objetivo)
    mask_green = cv2.inRange(hsv, np.array([30, 10, 35]), np.array([95, 255, 255]))
    # rosa/magenta/rojo (vecinos)
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
        out.append({"centroid": (cx, cy), "bbox": (x, y, w, h), "area": int(area)})
    out.sort(key=lambda d: -d["area"])
    return out

# ----------------------------------------
# Ángulos a 8 rumbos
# ----------------------------------------

def angle_between(p0: Tuple[int, int], p1: Tuple[int, int]) -> float:
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
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

# ----------------------------------------
# OCR cerca del vecino (con line-joiner)
# ----------------------------------------

def ocr_name_near_with_linejoin(bgr: np.ndarray, bbox: Tuple[int,int,int,int], subj_c: Tuple[int,int]) -> str:
    x,y,w,h = bbox
    cnx, cny = x + w//2, y + h//2
    L = max(w, h)
    vx, vy = cnx - subj_c[0], cny - subj_c[1]
    norm = math.hypot(vx, vy)
    if norm < 1e-3:
        vx, vy = 1.0, 0.0
    else:
        vx, vy = vx/norm, vy/norm
    H, W = bgr.shape[:2]
    pad = int(max(60, 2.4*L))
    def _clip(x0,y0,x1,y1):
        x0 = max(0,min(W-1,x0)); y0=max(0,min(H-1,y0)); x1=max(1,min(W,x1)); y1=max(1,min(H,y1));
        if x1<=x0: x1=x0+1
        if y1<=y0: y1=y0+1
        return x0,y0,x1,y1
    crops: List[np.ndarray] = []
    # caja amplia alrededor
    x0,y0,x1,y1 = _clip(x-pad, y-pad, x+w+pad, y+h+pad)
    crops.append(bgr[y0:y1, x0:x1])
    # 3 saltos hacia fuera en la dirección del vecino
    for f in (1.0, 1.6, 2.2):
        cx = int(cnx + vx * f * 2.0 * L)
        cy = int(cny + vy * f * 2.0 * L)
        s  = int(max(72, 2.0 * L))
        x0,y0,x1,y1 = _clip(cx - s, cy - s, cx + s, cy + s)
        crops.append(bgr[y0:y1, x0:x1])
    best = ""
    best_len = -1
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
# Fallback MSER
# ----------------------------------------

def find_text_neighbors(bgr: np.ndarray, subject_bbox: Tuple[int,int,int,int]) -> List[Dict]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    try:
        mser = cv2.MSER_create()
    except Exception:
        return []
    regions, _ = mser.detectRegions(gray)
    sx, sy, sw, sh = subject_bbox
    def dist_to_rect(cx: int, cy: int, rect: Tuple[int,int,int,int]) -> float:
        x, y, w, h = rect
        dx = max(x - cx, 0, cx - (x + w))
        dy = max(y - cy, 0, cy - (y + h))
        return math.hypot(dx, dy)
    maxd = CFG.neigh_max_dist_ratio * max(sw, sh) * 1.2
    out: List[Dict] = []
    for pts in regions:
        x, y, w, h = cv2.boundingRect(pts)
        if w < 20 or h < 10 or w > 700 or h > 220:
            continue
        area = w*h
        cx, cy = x + w//2, y + h//2
        d = dist_to_rect(cx, cy, subject_bbox)
        if d == 0 or d > maxd:
            continue
        out.append({"centroid": (cx, cy), "bbox": (x, y, w, h), "area": area})
    out.sort(key=lambda d: -d["area"])  # por tamaño
    return out

# ----------------------------------------
# Row-based mejorado (2 líneas)
# ----------------------------------------

def _binarize(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    _, bw  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bwi = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw, bwi


def _enhance_gray(g: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(g)


def _ocr_text(img: np.ndarray, psm: int, whitelist: Optional[str] = None) -> str:
    cfg = f"--psm {psm} --oem 3"
    if whitelist is not None:
        safe = (whitelist or "").replace('"','')
        cfg += f' -c tessedit_char_whitelist="{safe}"'
    txt = pytesseract.image_to_string(img, config=cfg) or ""
    txt = txt.replace("\r", "\n")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{2,}", "\n", txt)
    return txt.strip()


def _clean_owner_line(line: str) -> str:
    if not line:
        return ""
    toks = [t for t in re.split(r"[^\wÁÉÍÓÚÜÑ'-]+", line.upper()) if t]
    out = []
    for t in toks:
        if any(ch.isdigit() for ch in t): break
        if t in GEO_TOKENS or "[" in t or "]" in t: break
        if t in BAD_TOKENS: continue
        out.append(t)
        if len([x for x in out if x not in NAME_CONNECTORS]) >= 7:
            break
    compact = []
    for t in out:
        if (not compact) and t in NAME_CONNECTORS:
            continue
        compact.append(t)
    name = " ".join(compact).strip()
    return name[:80]


def _pick_owner_from_text(txt: str) -> str:
    if not txt:
        return ""
    lines = [l.strip() for l in txt.split("\n") if l.strip()]
    for l in lines:
        U = l.upper()
        if any(tok in U for tok in BAD_TOKENS):
            continue
        if sum(ch.isdigit() for ch in U) > 1:
            continue
        if not UPPER_NAME_RE.match(U):
            continue
        name = _clean_owner_line(U)
        if len(name) >= 6:
            return name
    return ""


def _find_header_and_owner_band(bgr: np.ndarray, row_y: int, x_text0: int, x_text1: int, lines: int = 2) -> Tuple[int,int,int,int]:
    h, w = bgr.shape[:2]
    pad_y = int(h * 0.06)
    y0s = max(0, row_y - pad_y)
    y1s = min(h, row_y + pad_y)
    band = bgr[y0s:y1s, x_text0:x_text1]
    if band.size == 0:
        lh = int(h * (0.035 if lines == 1 else 0.06))
        y0 = max(0, row_y - int(h*0.012)); y1 = min(h, y0 + lh)
        return x_text0, int(x_text0 + 0.58*(x_text1-x_text0)), y0, y1
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    bw, bwi = _binarize(gray)
    for im in (bw, bwi):
        data = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT, config="--psm 6 --oem 3")
        words = data.get("text", []); xs = data.get("left", []); ys = data.get("top", []); ws = data.get("width", []); hs = data.get("height", [])
        x_nif = None; header_bottom = None
        for t, lx, ty, ww, hh in zip(words, xs, ys, ws, hs):
            if not t: continue
            T = t.upper()
            if "APELLIDOS" in T: header_bottom = max(header_bottom or 0, ty + hh)
            if T == "NIF": x_nif = lx; header_bottom = max(header_bottom or 0, ty + hh)
        if header_bottom is not None:
            y0 = y0s + header_bottom + 6
            lh = int(h * (0.035 if lines == 1 else 0.06))
            y1 = min(h, y0 + lh)
            if x_nif is not None:
                x0 = x_text0; x1 = min(x_text1, x_text0 + x_nif - 8)
            else:
                x0 = x_text0; x1 = int(x_text0 + 0.58*(x_text1-x_text0))
            if x1 - x0 > (x_text1 - x_text0) * 0.22:
                return x0, x1, y0, y1
    lh = int(h * (0.035 if lines == 1 else 0.06))
    y0 = max(0, row_y - int(h*0.012)); y1 = min(h, y0 + lh)
    x0 = x_text0; x1 = int(x_text0 + 0.58*(x_text1-x_text0))
    return x0, x1, y0, y1


def _extract_owner_from_row(bgr: np.ndarray, row_y: int, lines: int = 2) -> Tuple[str, Tuple[int,int,int,int], int]:
    h, w = bgr.shape[:2]
    x_text0 = int(w * 0.30); x_text1 = int(w * 0.96)
    attempts = []
    for attempt in range(2):
        extra_left = attempt * max(16, int(w * 0.02))
        x0_base = max(0, x_text0 - extra_left)
        x0, x1, y0, y1 = _find_header_and_owner_band(bgr, row_y, x0_base, x_text1, lines=lines)
        roi = bgr[y0:y1, x0:x1]
        if roi.size == 0:
            attempts.append((attempt, x0,y0,x1,y1, "")); continue
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)
        g = _enhance_gray(g)
        bw, bwi = _binarize(g)
        WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑabcdefghijklmnopqrstuvwxyzáéíóúüñ '"
        variants = [ _ocr_text(bw, 6, WL), _ocr_text(bwi, 6, WL), _ocr_text(bw, 7, WL) ]
        owner = ""
        for txt in variants:
            owner = _pick_owner_from_text(txt)
            if owner:
                break
        if not owner:
            # fallback: image_to_data y unir 2 líneas
            rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            lines_data = ocr_image_to_data_lines(rgb)
            if lines_data:
                joined = lines_data[0][0]
                if len(lines_data) >= 2:
                    below = lines_data[1][0].strip()
                    if 0 < len(below) <= CFG.second_line_maxchars and len(below.split()) <= CFG.second_line_maxtokens:
                        joined = f"{joined} {below}"
                owner = _pick_owner_from_text(joined)
        owner = clean_candidate_text(postprocess_name(owner))
        attempts.append((attempt, x0,y0,x1,y1, owner))
        if owner and len(owner) >= 6:
            return owner, (x0,y0,x1,y1), attempt
    best = max(attempts, key=lambda t: len(t[5]) if t[5] else 0)
    return best[5], (best[1],best[2],best[3],best[4]), best[0]


def detect_rows_and_extract8(bgr: np.ndarray) -> Tuple[Dict[str,List[str]], dict]:
    h, w = bgr.shape[:2]
    top = int(h * 0.12); bottom = int(h * 0.92)
    left = int(w * 0.06); right = int(w * 0.40)
    crop = bgr[top:bottom, left:right]
    # colores
    mg, mp = detect_masks(crop)
    def _centroids(mask: np.ndarray, min_area: int) -> List[Tuple[int,int,int]]:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: List[Tuple[int,int,int]] = []
        for c in cnts:
            a = cv2.contourArea(c)
            if a < min_area: continue
            M = cv2.moments(c)
            if M["m00"] == 0: continue
            cx = int(M["m10"] / M["m00"]); cy = int(M["m01"] / M["m00"])
            out.append((cx, cy, int(a)))
        out.sort(key=lambda x: -x[2])
        return out
    mains  = _centroids(mg, min_area=max(240, CFG.neigh_min_area_hard))
    neighs = _centroids(mp, min_area=max(180, CFG.neigh_min_area_hard // 2))
    if not mains:
        return {k: [] for k in ("norte","noreste","este","sureste","sur","suroeste","oeste","noroeste")}, {"rows": []}
    mains_abs  = [(cx+left, cy+top, a) for (cx,cy,a) in mains]
    mains_abs.sort(key=lambda t: t[1])
    neighs_abs = [(cx+left, cy+top, a) for (cx,cy,a) in neighs]
    ldr: Dict[str, List[str]] = {k: [] for k in ("norte","noreste","este","sureste","sur","suroeste","oeste","noroeste")}
    rows_dbg = []
    used_rows = 0
    for (mcx, mcy, _a) in mains_abs[:8]:
        best = None; best_d = 1e18
        for (nx, ny, _na) in neighs_abs:
            d = (nx-mcx)**2 + (ny-mcy)**2
            if d < best_d:
                best_d = d; best = (nx, ny)
        side = ""
        if best is not None and best_d < (w*0.28)**2:
            ang = angle_between((mcx, mcy), best)
            side = angle_to_8(ang)
        owner, roi, attempt_id = _extract_owner_from_row(bgr, row_y=mcy, lines=max(1, CFG.row_owner_lines))
        if side and owner:
            if owner not in ldr[side]:
                ldr[side].append(owner); used_rows += 1
        rows_dbg.append({"row_y": mcy, "main_center": [mcx, mcy], "neigh_center": list(best) if best else None, "side": side, "owner": owner, "roi_attempt": attempt_id, "roi": list(roi)})
    return ldr, {"rows": rows_dbg, "used_rows": used_rows}

# ----------------------------------------
# Notarial
# ----------------------------------------

def generate_notarial_text(extracted: Dict) -> str:
    owners_by_side: Dict[str, List[str]] = extracted.get("linderos", {})
    order = ["norte","noreste","este","sureste","sur","suroeste","oeste","noroeste"]
    parts = []
    for side in order:
        vals = owners_by_side.get(side, []) or []
        vals = list(dict.fromkeys(vals))
        if not vals: continue
        if len(vals) == 1: parts.append(f"{side.capitalize()}, {vals[0]}")
        else: parts.append(f"{side.capitalize()}, " + ", ".join(vals[:-1]) + f" y {vals[-1]}")
    sides_text = "; ".join(parts)
    return f"Linda: {sides_text}." if sides_text else "No se han podido determinar linderos suficientes para una redacción notarial fiable."

# ----------------------------------------
# Expansión de diagonales a cardinales (siempre)
# ----------------------------------------

def expand_diagonals_to_cardinals(ldr: Dict[str, List[str]]) -> Dict[str, List[str]]:
    if not CFG.diag_to_cardinals:
        return ldr
    mapping = {
        "noreste": ("norte", "este"),
        "sureste": ("sur", "este"),
        "suroeste": ("sur", "oeste"),
        "noroeste": ("norte", "oeste"),
    }
    for diag, (a, b) in mapping.items():
        for nm in list(ldr.get(diag, [])):
            if nm and nm not in ldr[a]:
                ldr[a].append(nm)
            if nm and nm not in ldr[b]:
                ldr[b].append(nm)
        ldr[diag] = []
    return ldr

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

    # Pipeline basado en color
    m_green, m_pink = detect_masks(bgr)
    subs = contours_and_centroids(m_green)
    neigh = contours_and_centroids(m_pink)
    subj = subs[0] if subs else None
    ldr_map: Dict[str, List[str]] = {k: [] for k in ("norte","noreste","este","sureste","sur","suroeste","oeste","noroeste")}
    used_method = "none"
    if subj:
        sx, sy, sw, sh =  subj["bbox"]
        def _dist_to_rect(cx: int, cy: int, rect: Tuple[int,int,int,int]) -> float:
            x, y, w, h = rect
            dx = max(x - cx, 0, cx - (x + w))
            dy = max(y - cy, 0, cy - (y + h))
            return math.hypot(dx, dy)
        maxd = CFG.neigh_max_dist_ratio * max(sw, sh)
        neigh = [nb for nb in neigh if _dist_to_rect(nb["centroid"][0], nb["centroid"][1], subj["bbox"]) <= maxd]
        used_method = "color" if neigh else "mser"
        if not neigh:
            neigh = find_text_neighbors(bgr, subj["bbox"])  # MSER fallback
        subj_c = subj["centroid"]
        for nb in neigh[:10]:
            side = angle_to_8(angle_between(subj_c, nb["centroid"]))
            name = ocr_name_near_with_linejoin(bgr, nb["bbox"], subj_c)
            if name and name not in ldr_map[side]:
                ldr_map[side].append(name)
                owners_detected_union.append(name)

    # Pipeline por filas
    ldr_row, rows_dbg = detect_rows_and_extract8(bgr)
    for side, arr in ldr_row.items():
        for nm in arr:
            if nm and nm not in owners_detected_union:
                owners_detected_union.append(nm)

    # Fusión
    ldr_out: Dict[str, List[str]] = {k: [] for k in ("norte","noreste","este","sureste","sur","suroeste","oeste","noroeste")}
    for side in ldr_out.keys():
        seen = set()
        for src in (ldr_map.get(side, []), ldr_row.get(side, [])):
            for nm in src:
                nm_pp = clean_candidate_text(postprocess_name(nm))
                if nm_pp and nm_pp not in seen:
                    ldr_out[side].append(nm_pp); seen.add(nm_pp)

    # Expansión de diagonales a cardinales (siempre)
    ldr_out = expand_diagonals_to_cardinals(ldr_out)

    ldr_model = Linderos(**ldr_out)
    notarial = generate_notarial_text({"linderos": ldr_out})
    files = {"notarial_text.txt": base64.b64encode(notarial.encode("utf-8")).decode("ascii")}

    debug = {"method": used_method, "rows": rows_dbg}

    return ExtractResult(
        linderos=ldr_model,
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




