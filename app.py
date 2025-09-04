"""
app.py — autoCata v0.7.0
Pipeline real: OCR con Tesseract + extracción visual de colindantes + redacción notarial (GPT‑4o)

▶ Endpoints
- POST /extract  → procesa 1..5 PDFs de certificaciones catastrales (secuencial)

▶ Requisitos del entorno (Railway / local)
- Python 3.9+
- Paquetes pip:
  fastapi, uvicorn, pydantic, python-multipart, pillow, pdf2image, pytesseract,
  opencv-python-headless, numpy, openai (>=1.40), python-dotenv (opcional)
- Sistema: tesseract-ocr instalado en el host (apt-get install tesseract-ocr)
- Poppler para pdf2image (apt-get install poppler-utils)

▶ Variables de entorno relevantes (todas opcionales salvo AUTH_TOKEN y OPENAI_API_KEY si se usa GPT)
- AUTH_TOKEN                        → token fijo para Authorization: Bearer <token>
- OPENAI_API_KEY                    → API key de OpenAI (GPT‑4o)
- MODEL_NOTARIAL                    → por defecto "gpt-4o-mini"
- PDF_DPI                           → por defecto 300
- FAST_MODE                         → "1" para usar umbrales rápidos
- TEXT_ONLY                         → "1" para saltar CV y sólo extraer texto (respaldo)
- NAME_HINTS                        → lista separada por | con pistas de nombres
- NAME_HINTS_FILE                   → ruta a fichero con pistas de nombres (una por línea)
- SECOND_LINE_FORCE                 → "1" fuerza concatenación 2ª línea del nombre
- SECOND_LINE_MAXCHARS              → por defecto 28
- SECOND_LINE_MAXTOKENS             → por defecto 5
- SECOND_LINE_STRICT                → "1" activa heurística estricta
- NEIGH_MIN_AREA_HARD               → área mínima de contorno vecino (px) (p.ej. 1800)
- SIDE_MAX_DIST_FRAC                → fracción máx. de distancia para asignación a cardinal (0..1)
- ROW_BAND_FRAC                     → fracción de banda superior/inferior para Norte/Sur (fallback)
- AUTO_DPI, FAST_DPI, SLOW_DPI      → perfiles opcionales de DPI
- DIAG_MODE                         → "1" adjunta datos debug (cajas, puntos, etc.)
- REORDER_TO_NOMBRE_APELLIDOS       → "1" reordena a Nombre Apellidos si detecta "APELLIDOS, NOMBRE"
- REORDER_MIN_CONF                  → confianza mínima OCR 0..100 para reordenar

▶ Cumplimiento de requisitos marcados por el usuario (2025‑09‑03)
1) Redacción notarial en JSON (campo notarial_text) y además como archivo .txt descargable
2) Si el PDF no contiene OCR/metadata legibles → error 400 explicando el motivo
3) Todo gestionado desde /extract; límite 5 PDFs por subida; procesamiento secuencial
4) Uso exclusivo de GPT‑4o para redacción notarial (si hay API key); plantilla local si no
5) Integración con tokens fijos (Bearer AUTH_TOKEN) para WordPress plugin

"""
from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# OCR y CV
import pytesseract
import cv2
from pdf2image import convert_from_bytes

# OpenAI (solo para la redacción notarial)
try:
    from openai import OpenAI
except Exception:  # paquete no instalado
    OpenAI = None  # type: ignore

__version__ = "0.7.0"

# ----------------------------------------
# Configuración y utilidades
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

    neigh_min_area_hard: int = int(os.getenv("NEIGH_MIN_AREA_HARD", "1800"))
    side_max_dist_frac: float = float(os.getenv("SIDE_MAX_DIST_FRAC", "0.65"))
    row_band_frac: float = float(os.getenv("ROW_BAND_FRAC", "0.25"))

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
# Modelos de I/O
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
    files: Optional[Dict[str, str]] = None  # {"notarial_text.txt": base64}


class MultiResult(BaseModel):
    results: List[ExtractResult]
    version: str


# ----------------------------------------
# Seguridad: token simple tipo Bearer (con esquema para Swagger)
# ----------------------------------------
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

bearer_scheme = HTTPBearer(auto_error=False)

def require_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)):
    if CFG.auth_token is None:
        return  # sin auth en entorno → no aplicar (modo dev)
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Falta o es inválido el encabezado Authorization (Bearer)")
    token = credentials.credentials
    if token != CFG.auth_token:
        raise HTTPException(status_code=403, detail="Token no autorizado")


# ----------------------------------------
# OCR utilidades
# ----------------------------------------

def run_ocr(img_bgr: np.ndarray, psm: int = 6, lang: str = "spa") -> Tuple[str, float]:
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


# ----------------------------------------
# Conversión PDF → imágenes
# ----------------------------------------

def pdf_to_images(pdf_bytes: bytes, dpi: Optional[int] = None) -> List[np.ndarray]:
    use_dpi = dpi or CFG.pdf_dpi
    images = convert_from_bytes(pdf_bytes, dpi=use_dpi, fmt="png")
    out = []
    for im in images:
        arr = np.array(im)  # RGBA/RGB
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        out.append(arr)
    return out


# ----------------------------------------
# Detección de parcelas por color (verde sujeto / rosa vecinos)
# ----------------------------------------

def detect_masks(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Umbrales heurísticos; pueden ajustarse según documentos
    # Verde (parcela objeto de estudio)
    lower_green = np.array([35, 25, 40])
    upper_green = np.array([85, 255, 255])
    # Rosa (vecinos) – tonos magenta/rosados
    lower_pink1 = np.array([140, 30, 60])
    upper_pink1 = np.array([179, 255, 255])
    lower_pink2 = np.array([0, 30, 60])
    upper_pink2 = np.array([10, 255, 255])

    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_pink = cv2.inRange(hsv, lower_pink1, upper_pink1) | cv2.inRange(hsv, lower_pink2, upper_pink2)

    # Limpieza
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, k, iterations=2)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, k, iterations=2)

    mask_pink = cv2.morphologyEx(mask_pink, cv2.MORPH_OPEN, k, iterations=2)
    mask_pink = cv2.morphologyEx(mask_pink, cv2.MORPH_CLOSE, k, iterations=2)

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
        out.append({
            "contour": c,
            "centroid": (cx, cy),
            "bbox": (x, y, w, h),
            "area": int(area),
        })
    # ordenar por área desc
    out.sort(key=lambda d: -d["area"])
    return out


def choose_subject(subjects: List[Dict]) -> Optional[Dict]:
    return subjects[0] if subjects else None


# ----------------------------------------
# Asignación de orientaciones
# ----------------------------------------

def angle_between(p0: Tuple[int, int], p1: Tuple[int, int]) -> float:
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    ang = math.degrees(math.atan2(-dy, dx))  # 0° hacia la derecha, 90° hacia arriba
    ang = (ang + 360.0) % 360.0
    return ang


def angle_to_cardinals(ang: float) -> List[str]:
    """Devuelve una o dos orientaciones cardinales en español.
    Reglas: bandas de 45° → combina adyacentes (NE, SE, SO, NO)
    337.5..22.5 → Este; 22.5..67.5 → NE; 67.5..112.5 → Norte; etc.
    """
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
        if lo > hi and (ang >= lo or ang < hi):  # banda 337.5..360 o 0..22.5
            return labels
    return ["este"]


def assign_orientations(subject_c: Tuple[int, int], neighbors: List[Dict]) -> Dict[str, List[int]]:
    idx_by_side: Dict[str, List[int]] = {"norte": [], "este": [], "sur": [], "oeste": []}
    for i, nb in enumerate(neighbors):
        ang = angle_between(subject_c, nb["centroid"])  # 0°→E, 90°→N
        sides = angle_to_cardinals(ang)
        for s in sides:
            idx_by_side[s].append(i)
    return idx_by_side


# ----------------------------------------
# OCR de nombres en entorno del contorno
# ----------------------------------------

def ocr_name_near(bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[str, float]:
    x, y, w, h = bbox
    pad = int(max(8, 0.07 * max(w, h)))
    H, W = bgr.shape[:2]
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)
    crop = bgr[y0:y1, x0:x1]

    # prepro para realzar texto negro
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 60, 60)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 31, 11)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k, iterations=1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k, iterations=1)
    inv = 255 - thr
    inv_bgr = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)

    text, conf = run_ocr(inv_bgr, psm=6, lang="spa")
    text = postprocess_name(text)
    return text, conf


def postprocess_name(text: str) -> str:
    s = text
    s = re.sub(r"\s+", " ", s).strip()
    # Correcciones simples
    s = s.replace("  ", " ")
    # Normaliza comas en formato "APELLIDOS, NOMBRE"
    if CFG.reorder_to_nombre_apellidos and "," in s:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) == 2 and len(parts[0]) >= 3 and len(parts[1]) >= 2:
            s = f"{parts[1]} {parts[0]}"
    # Quita coletillas típicas
    s = re.sub(r"\b(TITULAR EN INVESTIGACION|TITULAR EN INVESTIGACIÓN)\b", "Titular en investigación", s, flags=re.I)
    return s


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
            return top  # no parece apellido claro
        return f"{top} {below}"
    return top


# ----------------------------------------
# Redacción notarial con GPT‑4o
# ----------------------------------------

def generate_notarial_text(extracted: Dict) -> str:
    """Genera un texto notarial en español usando OpenAI GPT si hay API key; si no, plantilla local."""
    owners_by_side = extracted.get("linderos", {})
    def join_side(side: str) -> str:
        vals = owners_by_side.get(side, []) or []
        vals = list(dict.fromkeys(vals))  # únicos preservando orden
        if not vals:
            return ""
        if len(vals) == 1:
            return f"{side.capitalize()}, {vals[0]}"
        # Si hay varios en el mismo lado, únelos con 'y'
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

    # Fallback local
    if sides_text:
        return f"Linda: {sides_text}."
    return "No se han podido determinar linderos suficientes para una redacción notarial fiable."


# ----------------------------------------
# Proceso principal por PDF
# ----------------------------------------

def process_pdf(pdf_bytes: bytes) -> ExtractResult:
    # 1) Convertir a imágenes
    dpi = CFG.fast_dpi if CFG.fast_mode else CFG.pdf_dpi
    pages = pdf_to_images(pdf_bytes, dpi=dpi)
    if not pages:
        raise HTTPException(status_code=400, detail="No se pudo renderizar el PDF a imágenes")

    # 2) Comprobación de legibilidad OCR mínima (política de rechazo 400)
    sample = pages[0]
    sample_text, conf = run_ocr(sample, psm=6, lang="spa")
    if len(sample_text) < 20:  # umbral bajo pero evita binarios/escaneos ilegibles
        raise HTTPException(status_code=400, detail="El PDF no contiene texto OCR legible o metadatos reconocibles para su análisis.")

    # Si se pide sólo texto (bypass CV)
    if CFG.text_only:
        ldr = Linderos(norte=[], sur=[], este=[], oeste=[])
        owners = []
        notarial = generate_notarial_text({"linderos": ldr.dict()})
        files = {"notarial_text.txt": base64.b64encode(notarial.encode("utf-8")).decode("ascii")}
        return ExtractResult(
            linderos=ldr,
            owners_detected=owners,
            notarial_text=notarial,
            note="TEXT_ONLY activo",
            debug={"ocr_conf": conf},
            files=files,
        )

    # 3) Extracción visual (primer página suficiente en CdG típica)
    bgr = pages[0]
    m_green, m_pink = detect_masks(bgr)
    subs = contours_and_centroids(m_green)
    neigh = contours_and_centroids(m_pink)

    subj = choose_subject(subs)
    if not subj:
        # Fallback por filas (arriba es Norte, abajo Sur). Sirve para casos sin color marcados
        H, W = bgr.shape[:2]
        band = int(H * CFG.row_band_frac)
        top_band = (0, 0, W, band)
        bot_band = (0, H - band, W, band)
        # OCR bandas
        top_txt, _ = run_ocr(bgr[0:band, :, :], psm=6)
        bot_txt, _ = run_ocr(bgr[H - band:H, :, :], psm=6)
        ldr = Linderos(
            norte=[top_txt[:60]] if top_txt else [],
            sur=[bot_txt[:60]] if bot_txt else [],
            este=[],
            oeste=[],
        )
        notarial = generate_notarial_text({"linderos": ldr.dict()})
        files = {"notarial_text.txt": base64.b64encode(notarial.encode("utf-8")).decode("ascii")}
        return ExtractResult(
            linderos=ldr,
            owners_detected=[t for t in [top_txt, bot_txt] if t],
            notarial_text=notarial,
            note="Fallo detección de parcela principal; se aplicó fallback por bandas",
            debug={"bands": {"row_band_frac": CFG.row_band_frac}},
            files=files,
        )

    subj_c = subj["centroid"]
    idx_by_side = assign_orientations(subj_c, neigh)

    # 4) OCR de nombres alrededor de cada vecino
    owners_idx_to_name: Dict[int, str] = {}
    owners_idx_conf: Dict[int, float] = {}
    owners_detected: List[str] = []

    for i, nb in enumerate(neigh):
        name, conf = ocr_name_near(bgr, nb["bbox"])
        # Heurística de segunda línea: intentar leer justo debajo de la caja
        x, y, w, h = nb["bbox"]
        line2_box = (x, y + h, w, int(h * 0.7))
        l2_name, l2_conf = ocr_name_near(bgr, line2_box)
        combined = maybe_concat_second_line([name, l2_name]) if name else l2_name
        final = postprocess_name(combined or name or l2_name)

        if final:
            owners_idx_to_name[i] = final
            owners_idx_conf[i] = max(conf, l2_conf)
            owners_detected.append(final)

    # 5) Construcción de linderos (permite múltiples por lado)
    ldr = Linderos(norte=[], sur=[], este=[], oeste=[])
    for side, idxs in idx_by_side.items():
        for i in idxs:
            nm = owners_idx_to_name.get(i)
            if nm and nm not in getattr(ldr, side):
                getattr(ldr, side).append(nm)

    # 6) Redacción notarial
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
                    "area": nb["area"],
                    "name": owners_idx_to_name.get(i),
                    "conf": owners_idx_conf.get(i, 0.0),
                }
                for i, nb in enumerate(neigh)
            ],
            "sides": idx_by_side,
        }

    return ExtractResult(
        linderos=ldr,
        owners_detected=list(dict.fromkeys(owners_detected)),
        notarial_text=notarial,
        note=None,
        debug=debug,
        files=files,
    )


# ----------------------------------------
# Endpoint principal /extract
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
            res = process_pdf(pdf_bytes)
            results.append(res)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Error procesando %s", f.filename)
            raise HTTPException(status_code=500, detail=f"Error interno procesando {f.filename}: {e}")

    return MultiResult(results=results, version=__version__)


# ----------------------------------------
# Salud
# ----------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


# ----------------------------------------
# Ejecución local (uvicorn)
# ----------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)

