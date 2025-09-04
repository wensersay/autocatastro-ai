"""
app.py — autoCata v0.8.0 (integrado 8 rumbos + fila)

✔ Integra lo mejor de 0.7.x (color + OCR cercano + limpieza + fallbacks) con tu lógica LEGACY por filas
  (detección de filas, búsqueda de cabecera APELLIDOS/NIF y extracción del titular alineado al y de la fila)

Novedades clave
- 8 rumbos: norte, noreste, este, sureste, sur, suroeste, oeste, noroeste.
- Dos pipelines combinados:
  • map_based → color (verde/rosa), vecinos y OCR cercano a cada vecino.
  • row_based → filas + lectura de titular (1–2 líneas) en la tabla APELLIDOS/NOMBRE/RAZÓN SOCIAL.
  Se fusionan los resultados (únicos por lado), priorizando nombres más "limpios".
- Mejor lectura de segunda línea del titular (configurable): ROW_OWNER_LINES=2, SECOND_LINE_FORCE.
- HSV permisivo, filtro de distancia anti-cartela, fallback MSER y "barrido direccional" para rótulos separados.

Endpoints
- POST /extract  (files[] PDF, 1..5) → MultiResult
- GET  /health

Variables útiles (todas opcionales)
- AUTH_TOKEN (Bearer)
- PDF_DPI=500          NEIGH_MIN_AREA_HARD=300..1200   NEIGH_MAX_DIST_RATIO=1.2..2.5
- SECOND_LINE_FORCE=1  SECOND_LINE_MAXCHARS=28         SECOND_LINE_MAXTOKENS=5
- ROW_BAND_FRAC=0.25   ROW_OWNER_LINES=2               TEXT_ONLY=0/1
- NAME_HINTS="GARCÍA|PÉREZ|S.L.|S.A."                 DIAG_MODE=0/1
- OPENAI_API_KEY (si quieres GPT para redacción)       MODEL_NOTARIAL=gpt-4o-mini
"""
from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
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

# OpenAI opcional (solo para redacción notarial)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

__version__ = "0.8.0"

# ----------------------------------------
# Configuración
# ----------------------------------------

@dataclass
class Cfg:
    auth_token: Optional[str] = os.getenv("AUTH_TOKEN")
    openai_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    model_notarial: str = os.getenv("MODEL_NOTARIAL", "gpt-4o-mini")

    pdf_dpi: int = int(os.getenv("PDF_DPI", "500"))
    fast_mode: bool = os.getenv("FAST_MODE", "0") == "1"

    text_only: bool = os.getenv("TEXT_ONLY", "0") == "1"

    name_hints: List[str] = field(default_factory=lambda: [s.strip() for s in os.getenv("NAME_HINTS", "").split("|") if s.strip()])
    name_hints_file: Optional[str] = os.getenv("NAME_HINTS_FILE")

    second_line_force: bool = os.getenv("SECOND_LINE_FORCE", "0") == "1"
    second_line_maxchars: int = int(os.getenv("SECOND_LINE_MAXCHARS", "28"))
    second_line_maxtokens: int = int(os.getenv("SECOND_LINE_MAXTOKENS", "5"))
    second_line_strict: bool = os.getenv("SECOND_LINE_STRICT", "0") == "1"

    neigh_min_area_hard: int = int(os.getenv("NEIGH_MIN_AREA_HARD", "600"))
    row_band_frac: float = float(os.getenv("ROW_BAND_FRAC", "0.25"))
    neigh_max_dist_ratio: float = float(os.getenv("NEIGH_MAX_DIST_RATIO", "1.8"))

    row_owner_lines: int = int(os.getenv("ROW_OWNER_LINES", "2"))  # leer 1 o 2 líneas en la tabla

    diag_mode: bool = os.getenv("DIAG_MODE", "0") == "1"

    reorder_to_nombre_apellidos: bool = os.getenv("REORDER_TO_NOMBRE_APELLIDOS", "1") == "1"
    reorder_min_conf: int = int(os.getenv("REORDER_MIN_CONF", "70"))


def load_name_hints(cfg: Cfg) -> List[str]:
    hints = list(cfg.name_hints)
    if cfg.name_hints_file and os.path.exists(cfg.name_hints_file):
        try:
            with open(cfg.name_hints_file, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        hints.append(s)
        except Exception:
            pass
    return hints

CFG = Cfg()
NAME_HINTS = set(load_name_hints(CFG))

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
# Utilidades OCR
# ----------------------------------------

def run_ocr(img_bgr: np.ndarray, psm: int = 6, lang: str = "spa+eng") -> Tuple[str, float]:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    cfg = f"--oem 1 --psm {psm} -l {lang}"
    try:
        data = pytesseract.image_to_data(pil, config=cfg, output_type=pytesseract.Output.DICT)
        text = " ".join([w for w in data["text"] if w.strip()])
        confs = [c for c in data["conf"] if isinstance(c, (int, float)) and c >= 0]
        conf = float(np.mean(confs)) if confs else 0.0
        return text.strip(), conf
    except Exception as e:
        logger.warning(f"OCR error: {e}")
        return "", 0.0

def _prepro_variants(gray: np.ndarray) -> List[np.ndarray]:
    outs: List[np.ndarray] = []
    thr1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 11)
    outs.append(255 - thr1)
    _, thr2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    outs.append(cv2.cvtColor(thr2, cv2.COLOR_GRAY2BGR))
    _, thr3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    outs.append(cv2.cvtColor(255 - thr3, cv2.COLOR_GRAY2BGR))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cimg = clahe.apply(gray)
    _, thr4 = cv2.threshold(cimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    outs.append(cv2.cvtColor(thr4, cv2.COLOR_GRAY2BGR))
    return outs

def run_ocr_multi(img_bgr: np.ndarray) -> Tuple[str, float]:
    variants: List[np.ndarray] = []
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 60, 60)
    for scale in (1.0, 1.5, 2.0):
        g = gray if scale == 1.0 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        variants.extend(_prepro_variants(g))
    best_text, best_conf = "", 0.0
    for v in variants:
        for psm in (7, 6, 11, 13):
            t, c = run_ocr(v, psm=psm)
            if (c > best_conf + 1.0) or (abs(c - best_conf) < 1.0 and len(t) > len(best_text)):
                best_text, best_conf = t, c
    return best_text, best_conf

# ----------------------------------------
# PDF → imagen de mapa (prefiere página 2)
# ----------------------------------------

def load_map_page_bgr(pdf_bytes: bytes, dpi: int) -> np.ndarray:
    # intenta pág. 2 (mapa), si no existe cae a pág. 1
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=2, last_page=2)
        if pages:
            arr = np.array(pages[0].convert("RGB"))[:, :, ::-1]
            return arr
    except Exception:
        pass
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=1, last_page=1)
    if not pages:
        raise HTTPException(status_code=400, detail="No se pudo rasterizar el PDF")
    return np.array(pages[0].convert("RGB"))[:, :, ::-1]

# ----------------------------------------
# Segmentación por color (verde sujeto / rosa vecinos)
# ----------------------------------------

def detect_masks(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Verde permisivo
    lower_green = np.array([30, 10, 35])
    upper_green = np.array([95, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    # Rosa/rojo/magenta permisivo
    ranges = [
        (np.array([0,   5, 25]), np.array([25, 255, 255])),
        (np.array([140, 5, 25]), np.array([179,255, 255])),
        (np.array([120,10, 40]), np.array([150,255, 255])),
    ]
    mask_pink = np.zeros(mask_green.shape, dtype=np.uint8)
    for lo, hi in ranges:
        mask_pink |= cv2.inRange(hsv, lo, hi)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, k5, iterations=1)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, k5, iterations=2)
    mask_green = cv2.dilate(mask_green, k3, iterations=1)
    mask_pink  = cv2.morphologyEx(mask_pink,  cv2.MORPH_OPEN, k5, iterations=1)
    mask_pink  = cv2.morphologyEx(mask_pink,  cv2.MORPH_CLOSE, k5, iterations=2)
    mask_pink  = cv2.dilate(mask_pink,  k3, iterations=1)
    return mask_green, mask_pink


def contours_and_centroids(mask: np.ndarray) -> List[Dict]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[Dict] = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < CFG.neigh_min_area_hard:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        x, y, w, h = cv2.boundingRect(c)
        out.append({"contour": c, "centroid": (cx, cy), "bbox": (x, y, w, h), "area": int(area)})
    out.sort(key=lambda d: -d["area"])  # área desc
    return out

# ----------------------------------------
# Rumbos (8)
# ----------------------------------------

def angle_between(p0: Tuple[int, int], p1: Tuple[int, int]) -> float:
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    ang = math.degrees(math.atan2(-dy, dx))  # 0°→Este, 90°→Norte
    return (ang + 360.0) % 360.0


def angle_to_8(ang: float) -> str:
    # sectores de 45° centrados en E(0), NE(45), N(90), NO(135), O(180), SO(225), S(270), SE(315)
    # Usamos intervalos [center-22.5, center+22.5)
    if   (337.5 <= ang < 360.0) or (0.0 <= ang < 22.5):  return "este"
    if   22.5 <= ang < 67.5:   return "noreste"
    if   67.5 <= ang < 112.5:  return "norte"
    if  112.5 <= ang < 157.5:  return "noroeste"
    if  157.5 <= ang < 202.5:  return "oeste"
    if  202.5 <= ang < 247.5:  return "suroeste"
    if  247.5 <= ang < 292.5:  return "sur"
    # 292.5 .. 337.5
    return "sureste"

# ----------------------------------------
# Limpieza y helpers de nombres
# ----------------------------------------

def postprocess_name(text: str) -> str:
    s = re.sub(r"\s+", " ", (text or "")).strip()
    if not s:
        return ""
    if CFG.reorder_to_nombre_apellidos and "," in s:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) == 2 and len(parts[0]) >= 3 and len(parts[1]) >= 2:
            s = f"{parts[1]} {parts[0]}"
    s = s.replace("  ", " ")
    s = re.sub(r"\b(TITULAR EN INVESTIGACION|TITULAR EN INVESTIGACIÓN)\b", "Titular en investigación", s, flags=re.I)
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
    for hint in NAME_HINTS:
        if hint and hint.lower() in s.lower():
            return s
    tokens = s.split()
    good = [t for t in tokens if (len(t) >= 2 and (t.isupper() or t.istitle()))]
    return " ".join(tokens[:6]) if len(good) >= 2 else ""


def maybe_concat_second_line(lines: List[str]) -> str:
    if not lines:
        return ""
    if len(lines) == 1:
        return lines[0]
    top = lines[0].strip()
    below = lines[1].strip()
    if not CFG.second_line_force:
        return top
    if len(below) <= CFG.second_line_maxchars and len(below.split()) <= CFG.second_line_maxtokens:
        if CFG.second_line_strict and not re.search(r"[A-ZÁÉÍÓÚÑ]{2,}", below):
            return top
        return f"{top} {below}"
    return top

# ----------------------------------------
# OCR cercano a vecino + barrido direccional + MSER (map_based)
# ----------------------------------------

def ocr_name_near(bgr: np.ndarray, bbox: Tuple[int,int,int,int]) -> Tuple[str, float]:
    x, y, w, h = bbox
    pad = int(max(12, 0.20 * max(w, h)))
    H, W = bgr.shape[:2]
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
    crop = bgr[y0:y1, x0:x1]
    text, conf = run_ocr_multi(crop)
    text = clean_candidate_text(postprocess_name(text))
    return text, conf


def _clip_rect(x0:int,y0:int,x1:int,y1:int,W:int,H:int) -> Tuple[int,int,int,int]:
    x0 = max(0, min(W-1, x0)); y0 = max(0, min(H-1, y0))
    x1 = max(0, min(W,   x1)); y1 = max(0, min(H,   y1))
    if x1 <= x0: x1 = min(W, x0+1)
    if y1 <= y0: y1 = min(H, y0+1)
    return x0,y0,x1,y1


def _crop(bgr: np.ndarray, rect: Tuple[int,int,int,int]) -> np.ndarray:
    x,y,w,h = rect
    H,W = bgr.shape[:2]
    x0,y0,x1,y1 = _clip_rect(x, y, x+w, y+h, W, H)
    return bgr[y0:y1, x0:x1]


def ocr_label_around_neighbor(bgr: np.ndarray, nb_bbox: Tuple[int,int,int,int], subj_c: Tuple[int,int]) -> Tuple[str,float]:
    x,y,w,h = nb_bbox
    cnx, cny = x + w//2, y + h//2
    H,W = bgr.shape[:2]
    L = max(w, h)
    vx, vy = cnx - subj_c[0], cny - subj_c[1]
    norm = math.hypot(vx, vy)
    if norm < 1e-3:
        vx, vy = 1.0, 0.0
    else:
        vx, vy = vx/norm, vy/norm
    candidates: List[np.ndarray] = []
    pad = int(max(80, 3.0*L))
    candidates.append(_crop(bgr, (x - pad, y - pad, w + 2*pad, h + 2*pad)))
    for f in (1.0, 1.6, 2.2, 2.8):
        cx = int(cnx + vx * f * 2.2 * L)
        cy = int(cny + vy * f * 2.2 * L)
        s  = int(max(90, 2.6 * L))
        candidates.append(_crop(bgr, (cx - s, cy - s, 2*s, 2*s)))
    # ortogonales
    ox, oy = -vy, vx
    for f in (1.2, 1.8):
        for sign in (-1, +1):
            cx = int(cnx + sign * ox * f * 2.0 * L)
            cy = int(cny + sign * oy * f * 2.0 * L)
            s  = int(max(80, 2.2 * L))
            candidates.append(_crop(bgr, (cx - s, cy - s, 2*s, 2*s)))
    best_text, best_conf = "", 0.0
    for crop in candidates:
        t, c = run_ocr_multi(crop)
        t = clean_candidate_text(postprocess_name(t))
        if not t:
            continue
        if (c > best_conf + 1.0) or (abs(c - best_conf) < 1.0 and len(t) > len(best_text)):
            best_text, best_conf = t, c
    return best_text, best_conf


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
# LEGACY ROW-BASED: detectar filas y extraer titular en tabla (mejorado 2 líneas)
# ----------------------------------------

UPPER_NAME_RE = re.compile(r"^[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\.'\-]+$", re.UNICODE)
BAD_TOKENS = {"POLÍGONO","POLIGONO","PARCELA","APELLIDOS","NOMBRE","RAZON","RAZÓN","SOCIAL","NIF","DOMICILIO","LOCALIZACIÓN","LOCALIZACION","REFERENCIA","CATASTRAL","TITULARIDAD","PRINCIPAL"}
GEO_TOKENS = {"LUGO","BARCELONA","MADRID","VALENCIA","SEVILLA","CORUÑA","A CORUÑA","MONFORTE","LEM","LEMOS","HOSPITALET","L'HOSPITALET","SAVIAO","SAVIÑAO","GALICIA","[LUGO]","[BARCELONA]"}
NAME_CONNECTORS = {"DE","DEL","LA","LOS","LAS","DA","DO","DAS","DOS","Y"}


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
        if len([x for x in out if x not in NAME_CONNECTORS]) >= 6:
            break
    compact = []
    for t in out:
        if (not compact) and t in NAME_CONNECTORS:
            continue
        compact.append(t)
    name = " ".join(compact).strip()
    return name[:72]


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
    """Busca 'APELLIDOS'/'NIF' y devuelve ROI que cubre 1–2 líneas de titular."""
    h, w = bgr.shape[:2]
    pad_y = int(h * 0.06)
    y0s = max(0, row_y - pad_y)
    y1s = min(h, row_y + pad_y)
    band = bgr[y0s:y1s, x_text0:x_text1]
    if band.size == 0:
        lh = int(h * (0.035 if lines == 1 else 0.06))
        y0 = max(0, row_y - int(h*0.012))
        y1 = min(h, y0 + lh)
        return x_text0, int(x_text0 + 0.58*(x_text1-x_text0)), y0, y1
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    bw, bwi = _binarize(gray)
    for im in (bw, bwi):
        data = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT, config="--psm 6 --oem 3")
        words = data.get("text", [])
        xs = data.get("left", [])
        ys = data.get("top", [])
        ws = data.get("width", [])
        hs = data.get("height", [])
        x_nif = None
        header_bottom = None
        for t, lx, ty, ww, hh in zip(words, xs, ys, ws, hs):
            if not t: continue
            T = t.upper()
            if "APELLIDOS" in T:
                header_bottom = max(header_bottom or 0, ty + hh)
            if T == "NIF":
                x_nif = lx
                header_bottom = max(header_bottom or 0, ty + hh)
        if header_bottom is not None:
            y0 = y0s + header_bottom + 6
            lh = int(h * (0.035 if lines == 1 else 0.06))
            y1 = min(h, y0 + lh)
            if x_nif is not None:
                x0 = x_text0
                x1 = min(x_text1, x_text0 + x_nif - 8)
            else:
                x0 = x_text0
                x1 = int(x_text0 + 0.58*(x_text1-x_text0))
            if x1 - x0 > (x_text1 - x_text0) * 0.22:
                return x0, x1, y0, y1
    # fallback si no se detecta cabecera
    lh = int(h * (0.035 if lines == 1 else 0.06))
    y0 = max(0, row_y - int(h*0.012))
    y1 = min(h, y0 + lh)
    x0 = x_text0
    x1 = int(x_text0 + 0.58*(x_text1-x_text0))
    return x0, x1, y0, y1


def _extract_owner_from_row(bgr: np.ndarray, row_y: int, lines: int = 2) -> Tuple[str, Tuple[int,int,int,int], int]:
    h, w = bgr.shape[:2]
    x_text0 = int(w * 0.30)  # más a la izquierda que legacy para no "morder"
    x_text1 = int(w * 0.96)
    attempts = []
    for attempt in range(3):
        extra_left = attempt * max(18, int(w * 0.03))
        x0_base = max(0, x_text0 - extra_left)
        x0, x1, y0, y1 = _find_header_and_owner_band(bgr, row_y, x0_base, x_text1, lines=lines)
        roi = bgr[y0:y1, x0:x1]
        if roi.size == 0:
            attempts.append((attempt, x0,y0,x1,y1, ""))
            continue
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
        g = _enhance_gray(g)
        bw, bwi = _binarize(g)
        WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '"
        variants = [
            _ocr_text(bw,  psm=6,  whitelist=WL),
            _ocr_text(bwi, psm=6,  whitelist=WL),
            _ocr_text(bw,  psm=7,  whitelist=WL),
            _ocr_text(bwi, psm=7,  whitelist=WL),
            _ocr_text(bw,  psm=13, whitelist=WL),
        ]
        owner = ""
        for txt in variants:
            owner = _pick_owner_from_text(txt)
            if owner:
                break
        attempts.append((attempt, x0,y0,x1,y1, owner))
        if owner and len(owner) >= 8:
            return owner, (x0,y0,x1,y1), attempt
    best = max(attempts, key=lambda t: len(t[5]) if t[5] else 0)
    return best[5], (best[1],best[2],best[3],best[4]), best[0]


def detect_rows_and_extract8(bgr: np.ndarray, annotate: bool = False) -> Tuple[Dict[str,List[str]], dict]:
    h, w = bgr.shape[:2]
    # recorte de zona mapa (como legacy): izq 6..40%, alto 12..92%
    top = int(h * 0.12); bottom = int(h * 0.92)
    left = int(w * 0.06); right = int(w * 0.40)
    crop = bgr[top:bottom, left:right]
    mg, mp = detect_masks(crop)
    # contornos
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
    # traslada a absolutos
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
            side = angle_to_8(angle_between((mcx, mcy), best))
        owner, roi, attempt_id = _extract_owner_from_row(bgr, row_y=mcy, lines=max(1, CFG.row_owner_lines))
        if side and owner:
            if owner not in ldr[side]:
                ldr[side].append(owner)
                used_rows += 1
        rows_dbg.append({
            "row_y": mcy,
            "main_center": [mcx, mcy],
            "neigh_center": list(best) if best is not None else None,
            "side": side,
            "owner": owner,
            "roi_attempt": attempt_id,
            "roi": list(roi),
        })
    return ldr, {"rows": rows_dbg, "used_rows": used_rows}

# ----------------------------------------
# Redacción notarial (usa GPT si hay API key, si no plantilla)
# ----------------------------------------

def generate_notarial_text(extracted: Dict) -> str:
    owners_by_side: Dict[str, List[str]] = extracted.get("linderos", {})
    order = ["norte","noreste","este","sureste","sur","suroeste","oeste","noroeste"]
    parts = []
    for side in order:
        vals = owners_by_side.get(side, []) or []
        vals = list(dict.fromkeys(vals))
        if not vals:
            continue
        if len(vals) == 1:
            parts.append(f"{side.capitalize()}, {vals[0]}")
        else:
            parts.append(f"{side.capitalize()}, " + ", ".join(vals[:-1]) + f" y {vals[-1]}")
    sides_text = "; ".join(parts)
    prompt = (
        "Redacta en estilo notarial, claro y conciso, un párrafo de linderos en español, "
        "usando los ocho rumbos si hay datos (N, NE, E, SE, S, SO, O, NO). "
        f"Linderos detectados: {json.dumps(owners_by_side, ensure_ascii=False)}\n\n"
        "Devuelve solo un párrafo, sin encabezados, sin comillas, evitando redundancias."
    )
    if CFG.openai_key and OpenAI is not None:
        try:
            client = OpenAI(api_key=CFG.openai_key)
            resp = client.chat.completions.create(
                model=CFG.model_notarial,
                messages=[
                    {"role": "system", "content": "Eres un notario español que redacta linderos con precisión."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=250,
            )
            content = resp.choices[0].message.content.strip()
            if content:
                return content
        except Exception as e:
            logger.warning(f"OpenAI fallback por error: {e}")
    return f"Linda: {sides_text}." if sides_text else "No se han podido determinar linderos suficientes para una redacción notarial fiable."

# ----------------------------------------
# Proceso por PDF
# ----------------------------------------

def process_pdf(pdf_bytes: bytes) -> ExtractResult:
    bgr = load_map_page_bgr(pdf_bytes, dpi=CFG.pdf_dpi)
    # Gate de legibilidad mínima sobre esta página
    sample_text, conf = run_ocr(bgr, psm=6, lang="spa+eng")
    if len(sample_text) < 20:
        raise HTTPException(status_code=400, detail="El PDF no contiene texto OCR legible o metadatos reconocibles para su análisis.")

    if CFG.text_only:
        ldr_empty = Linderos()
        notarial = generate_notarial_text({"linderos": ldr_empty.model_dump()})
        files = {"notarial_text.txt": base64.b64encode(notarial.encode("utf-8")).decode("ascii")}
        return ExtractResult(linderos=ldr_empty, owners_detected=[], notarial_text=notarial, note="TEXT_ONLY activo", debug={"ocr_conf": conf}, files=files)

    owners_detected_union: List[str] = []

    # 1) Pipeline map_based (color + OCR cercano)
    m_green, m_pink = detect_masks(bgr)
    subs = contours_and_centroids(m_green)
    neigh = contours_and_centroids(m_pink)
    subj = subs[0] if subs else None
    ldr_map: Dict[str, List[str]] = {k: [] for k in ("norte","noreste","este","sureste","sur","suroeste","oeste","noroeste")}
    used_method = "none"
    if subj:
        sx, sy, sw, sh = subj["bbox"]
        def _dist_to_rect(cx: int, cy: int, rect: Tuple[int,int,int,int]) -> float:
            x, y, w, h = rect
            dx = max(x - cx, 0, cx - (x + w))
            dy = max(y - cy, 0, cy - (y + h))
            return math.hypot(dx, dy)
        maxd = CFG.neigh_max_dist_ratio * max(sw, sh)
        neigh = [nb for nb in neigh if _dist_to_rect(nb["centroid"][0], nb["centroid"][1], subj["bbox"]) <= maxd]
        if not neigh:
            neigh = find_text_neighbors(bgr, subj["bbox"])  # MSER fallback
            used_method = "mser" if neigh else "color"
        else:
            used_method = "color"
        subj_c = subj["centroid"]
        for i, nb in enumerate(neigh[:12]):
            side = angle_to_8(angle_between(subj_c, nb["centroid"]))
            name, conf = ocr_name_near(bgr, nb["bbox"])
            if not name or len(name) < 4:
                name2, conf2 = ocr_label_around_neighbor(bgr, nb["bbox"], subj_c)
                if len(name2) > len(name):
                    name, conf = name2, conf2
            # También intenta una "línea 2" justo debajo de la caja
            x, y, w, h = nb["bbox"]
            line2_box = (x - int(0.2*w), y + h, int(w*1.4), int(h * 1.0))
            l2_name, _ = ocr_name_near(bgr, line2_box)
            final = maybe_concat_second_line([name, l2_name]) if name else l2_name
            final = clean_candidate_text(postprocess_name(final))
            if final and final not in ldr_map[side]:
                ldr_map[side].append(final)
                owners_detected_union.append(final)

    # 2) Pipeline row_based (LEGACY mejorado)
    ldr_row, rows_dbg = detect_rows_and_extract8(bgr)
    for side, arr in ldr_row.items():
        for nm in arr:
            if nm and nm not in owners_detected_union:
                owners_detected_union.append(nm)

    # 3) Fusiona
    ldr_out: Dict[str, List[str]] = {k: [] for k in ("norte","noreste","este","sureste","sur","suroeste","oeste","noroeste")}
    # prioridad: nombres limpios por map_based; añade row_based si faltan
    for side in ldr_out.keys():
        seen = set()
        for src in (ldr_map.get(side, []), ldr_row.get(side, [])):
            for nm in src:
                if nm and nm not in seen:
                    ldr_out[side].append(nm); seen.add(nm)

    ldr_model = Linderos(**ldr_out)
    notarial = generate_notarial_text({"linderos": ldr_out})
    files = {"notarial_text.txt": base64.b64encode(notarial.encode("utf-8")).decode("ascii")}

    debug = None
    if CFG.diag_mode:
        debug = {
            "method": used_method,
            "rows": rows_dbg,
            "subject_centroid": (subj["centroid"] if subj else None),
        }

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)



