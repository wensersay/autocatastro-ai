"""
app.py — autoCata v0.7.3 (todo-en-uno)
OCR con Tesseract + extracción visual (color y fallback por texto) + redacción notarial (GPT-4o)

▶ Endpoints
- POST /extract → procesa 1..5 PDFs de certificaciones catastrales (secuencial)

▶ Dependencias (pip)
  fastapi, uvicorn, pydantic, python-multipart, pillow, pdf2image, pytesseract,
  opencv-python-headless, numpy, openai (>=1.40), python-dotenv (opcional)

▶ Sistema
- tesseract-ocr (+ idioma spa)
- poppler-utils

▶ Variables de entorno más útiles
- AUTH_TOKEN                        (Bearer)
- OPENAI_API_KEY                    (GPT-4o)
- MODEL_NOTARIAL                    (por defecto "gpt-4o-mini")
- PDF_DPI                           (por defecto 300; prueba 450–500 si texto pequeño)
- TEXT_ONLY                         (1 → desactiva CV y genera texto base)
- NAME_HINTS                        ("GARCÍA|PÉREZ|S.L.|S.A.")
- SECOND_LINE_FORCE                 (1 → concatena 2ª línea del nombre)
- NEIGH_MIN_AREA_HARD               (área mínima para vecinos por color; p.ej. 300–1200)
- NEIGH_MAX_DIST_RATIO              (filtrado por distancia 1.0–2.0)
- ROW_BAND_FRAC                     (fallback por bandas si no hay sujeto)
- DIAG_MODE                         (1 → adjunta debug)
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
from pydantic import BaseModel
from PIL import Image

# OpenAI opcional (solo para redacción notarial)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

__version__ = "0.7.3"

# ----------------------------------------
# Configuración
# ----------------------------------------

@dataclass
class Cfg:
    auth_token: Optional[str] = os.getenv("AUTH_TOKEN")
    openai_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    model_notarial: str = os.getenv("MODEL_NOTARIAL", "gpt-4o-mini")

    pdf_dpi: int = int(os.getenv("PDF_DPI", "300"))
    auto_dpi: Optional[int] = int(os.getenv("AUTO_DPI")) if os.getenv("AUTO_DPI") else None
    fast_dpi: int = int(os.getenv("FAST_DPI", "220"))
    slow_dpi: int = int(os.getenv("SLOW_DPI", "400"))
    fast_mode: bool = os.getenv("FAST_MODE", "0") == "1"

    text_only: bool = os.getenv("TEXT_ONLY", "0") == "1"

    name_hints: List[str] = field(default_factory=lambda: [s.strip() for s in os.getenv("NAME_HINTS", "").split("|") if s.strip()])
    name_hints_file: Optional[str] = os.getenv("NAME_HINTS_FILE")

    second_line_force: bool = os.getenv("SECOND_LINE_FORCE", "0") == "1"
    second_line_maxchars: int = int(os.getenv("SECOND_LINE_MAXCHARS", "28"))
    second_line_maxtokens: int = int(os.getenv("SECOND_LINE_MAXTOKENS", "5"))
    second_line_strict: bool = os.getenv("SECOND_LINE_STRICT", "0") == "1"

    neigh_min_area_hard: int = int(os.getenv("NEIGH_MIN_AREA_HARD", "1200"))
    side_max_dist_frac: float = float(os.getenv("SIDE_MAX_DIST_FRAC", "0.65"))
    row_band_frac: float = float(os.getenv("ROW_BAND_FRAC", "0.25"))
    neigh_max_dist_ratio: float = float(os.getenv("NEIGH_MAX_DIST_RATIO", "1.4"))

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
# Modelos
# ----------------------------------------

class Linderos(BaseModel):
    norte: List[str] = []
    este: List[str] = []
    sur: List[str] = []
    oeste: List[str] = []


class ExtractResult(BaseModel):
    linderos: Linderos
    owners_detected: List[str]
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
# OCR utilidades
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
        for psm in (7, 6, 11):
            t, c = run_ocr(v, psm=psm)
            if (c > best_conf + 1.0) or (abs(c - best_conf) < 1.0 and len(t) > len(best_text)):
                best_text, best_conf = t, c
    return best_text, best_conf

# ----------------------------------------
# PDF → imágenes
# ----------------------------------------

def pdf_to_images(pdf_bytes: bytes, dpi: Optional[int] = None) -> List[np.ndarray]:
    use_dpi = dpi or CFG.pdf_dpi
    images = convert_from_bytes(pdf_bytes, dpi=use_dpi, fmt="png")
    out: List[np.ndarray] = []
    for im in images:
        arr = np.array(im)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        out.append(arr)
    return out

# ----------------------------------------
# Segmentación por color (verde sujeto / rosa vecinos)
# ----------------------------------------

def detect_masks(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Verde (parcela objeto) — permisivo
    lower_green = np.array([30, 10, 35])
    upper_green = np.array([95, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Rosa/rojo/magenta (vecinos) — MUY permisivo (varios rangos)
    ranges = [
        (np.array([0,   5, 25]), np.array([25, 255, 255])),   # rojo-anaranjado pálido
        (np.array([140, 5, 25]), np.array([179,255, 255])),   # rojo/magenta alto
        (np.array([120,10, 40]), np.array([150,255, 255])),   # púrpura
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
    out = []
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


def choose_subject(subjects: List[Dict]) -> Optional[Dict]:
    return subjects[0] if subjects else None

# ----------------------------------------
# Asignación de orientaciones
# ----------------------------------------

def angle_between(p0: Tuple[int, int], p1: Tuple[int, int]) -> float:
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    ang = math.degrees(math.atan2(-dy, dx))  # 0°→E, 90°→N
    return (ang + 360.0) % 360.0


def angle_to_cardinals(ang: float) -> List[str]:
    bands = [
        (337.5, 360.0, ["este"]), (0.0, 22.5, ["este"]),
        (22.5, 67.5, ["norte", "este"]),
        (67.5, 112.5, ["norte"]),
        (112.5, 157.5, ["norte", "oeste"]),
        (157.5, 202.5, ["oeste"]),
        (202.5, 247.5, ["sur", "oeste"]),
        (247.5, 292.5, ["sur"]),
        (292.5, 337.5, ["sur", "este"]),
    ]
    for lo, hi, labels in bands:
        if lo < hi and (lo <= ang < hi):
            return labels
        if lo > hi and (ang >= lo or ang < hi):
            return labels
    return ["este"]


def assign_orientations(subject_c: Tuple[int, int], neighbors: List[Dict]) -> Dict[str, List[int]]:
    idx_by_side: Dict[str, List[int]] = {"norte": [], "este": [], "sur": [], "oeste": []}
    for i, nb in enumerate(neighbors):
        ang = angle_between(subject_c, nb["centroid"])  # 0°→E, 90°→N
        for s in angle_to_cardinals(ang):
            idx_by_side[s].append(i)
    return idx_by_side

# ----------------------------------------
# OCR de nombres
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


def ocr_name_near(bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[str, float]:
    x, y, w, h = bbox
    pad = int(max(10, 0.10 * max(w, h)))
    H, W = bgr.shape[:2]
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)
    crop = bgr[y0:y1, x0:x1]
    text, conf = run_ocr_multi(crop)
    text = clean_candidate_text(postprocess_name(text))
    return text, conf


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
# Fallback por texto (MSER)
# ----------------------------------------

def find_text_neighbors(bgr: np.ndarray, subject_bbox: Tuple[int,int,int,int]) -> List[Dict]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    try:
        mser = cv2.MSER_create()
    except Exception:
        return []
    regions, _ = mser.detectRegions(gray)

    sx, sy, sw, sh = subject_bbox

    def dist_to_rect(cx: int, cy: int, rect: Tuple[int, int, int, int]) -> float:
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
        area = w * h
        cx, cy = x + w // 2, y + h // 2
        d = dist_to_rect(cx, cy, subject_bbox)
        if d == 0 or d > maxd:
            continue
        out.append({"centroid": (cx, cy), "bbox": (x, y, w, h), "area": area})
    out.sort(key=lambda d: -d["area"])  # por tamaño
    return out

# ----------------------------------------
# Redacción notarial
# ----------------------------------------

def generate_notarial_text(extracted: Dict) -> str:
    owners_by_side = extracted.get("linderos", {})

    def join_side(side: str) -> str:
        vals = owners_by_side.get(side, []) or []
        vals = list(dict.fromkeys(vals))
        if not vals:
            return ""
        if len(vals) == 1:
            return f"{side.capitalize()}, {vals[0]}"
        return f"{side.capitalize()}, " + ", ".join(vals[:-1]) + f" y {vals[-1]}"

    sides_text = "; ".join([s for s in [join_side("norte"), join_side("sur"), join_side("este"), join_side("oeste")] if s])

    prompt = (
        "Redacta en estilo notarial, claro y conciso, un párrafo de linderos en español, "
        "a partir de la siguiente información de una certificación catastral. "
        "Respeta que un mismo lado puede incluir varios colindantes, y usa coma y 'y' adecuadamente.\n\n"
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
# Proceso principal
# ----------------------------------------

def process_pdf(pdf_bytes: bytes) -> ExtractResult:
    dpi = CFG.fast_dpi if CFG.fast_mode else CFG.pdf_dpi
    pages = pdf_to_images(pdf_bytes, dpi=dpi)
    if not pages:
        raise HTTPException(status_code=400, detail="No se pudo renderizar el PDF a imágenes")

    # Gate de legibilidad mínima
    sample_text, conf = run_ocr(pages[0], psm=6, lang="spa+eng")
    if len(sample_text) < 20:
        raise HTTPException(status_code=400, detail="El PDF no contiene texto OCR legible o metadatos reconocibles para su análisis.")

    if CFG.text_only:
        ldr = Linderos(norte=[], sur=[], este=[], oeste=[])
        notarial = generate_notarial_text({"linderos": ldr.dict()})
        files = {"notarial_text.txt": base64.b64encode(notarial.encode("utf-8")).decode("ascii")}
        return ExtractResult(linderos=ldr, owners_detected=[], notarial_text=notarial, note="TEXT_ONLY activo", debug={"ocr_conf": conf}, files=files)

    bgr = pages[0]
    m_green, m_pink = detect_masks(bgr)
    subs = contours_and_centroids(m_green)
    neigh = contours_and_centroids(m_pink)

    subj = choose_subject(subs)
    if not subj:
        # Fallback por bandas (Norte/Sur)
        H, W = bgr.shape[:2]
        band = int(H * CFG.row_band_frac)
        top_txt, _ = run_ocr(bgr[0:band, :, :], psm=6)
        bot_txt, _ = run_ocr(bgr[H - band:H, :, :], psm=6)
        ldr = Linderos(norte=[top_txt[:60]] if top_txt else [], sur=[bot_txt[:60]] if bot_txt else [], este=[], oeste=[])
        notarial = generate_notarial_text({"linderos": ldr.dict()})
        files = {"notarial_text.txt": base64.b64encode(notarial.encode("utf-8")).decode("ascii")}
        return ExtractResult(linderos=ldr, owners_detected=[t for t in [top_txt, bot_txt] if t], notarial_text=notarial, note="Fallback por bandas", debug={"bands": {"row_band_frac": CFG.row_band_frac}}, files=files)

    # Filtro por distancia (evitar cartelas)
    sx, sy, sw, sh = subj["bbox"]

    def _dist_to_rect(cx: int, cy: int, rect: Tuple[int, int, int, int]) -> float:
        x, y, w, h = rect
        dx = max(x - cx, 0, cx - (x + w))
        dy = max(y - cy, 0, cy - (y + h))
        return math.hypot(dx, dy)

    maxd = CFG.neigh_max_dist_ratio * max(sw, sh)
    neigh = [nb for nb in neigh if _dist_to_rect(nb["centroid"][0], nb["centroid"][1], subj["bbox"]) <= maxd]

    # Fallback: si no hay vecinos por color, busca cajas de texto (MSER)
    used_method = "color"
    if not neigh:
        neigh = find_text_neighbors(bgr, subj["bbox"])
        used_method = "mser" if neigh else "none"

    subj_c = subj["centroid"]
    idx_by_side = assign_orientations(subj_c, neigh)

    owners_idx_to_name: Dict[int, str] = {}
    owners_idx_conf: Dict[int, float] = {}
    owners_detected: List[str] = []

    for i, nb in enumerate(neigh):
        name, conf = ocr_name_near(bgr, nb["bbox"])
        x, y, w, h = nb["bbox"]
        line2_box = (x - int(0.2*w), y + h, int(w*1.4), int(h * 1.0))
        l2_name, l2_conf = ocr_name_near(bgr, line2_box)
        combined = maybe_concat_second_line([name, l2_name]) if name else l2_name
        final = clean_candidate_text(postprocess_name(combined or name or l2_name))
        if final:
            owners_idx_to_name[i] = final
            owners_idx_conf[i] = max(conf, l2_conf)
            owners_detected.append(final)

    ldr = Linderos(norte=[], sur=[], este=[], oeste=[])
    for side, idxs in idx_by_side.items():
        for i in idxs:
            nm = owners_idx_to_name.get(i)
            if nm and nm not in getattr(ldr, side):
                getattr(ldr, side).append(nm)

    notarial = generate_notarial_text({"linderos": ldr.dict()})
    files = {"notarial_text.txt": base64.b64encode(notarial.encode("utf-8")).decode("ascii")}

    debug = None
    if CFG.diag_mode:
        debug = {
            "subject_centroid": subj_c,
            "neighbors": [
                {
                    "centroid": nb["centroid"],
                    "bbox": nb["bbox"],
                    "area": nb.get("area", 0),
                    "name": owners_idx_to_name.get(i),
                    "conf": owners_idx_conf.get(i, 0.0),
                }
                for i, nb in enumerate(neigh)
            ],
            "sides": idx_by_side,
            "method": used_method,
        }

    return ExtractResult(linderos=ldr, owners_detected=list(dict.fromkeys(owners_detected)), notarial_text=notarial, note=None, debug=debug, files=files)

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





