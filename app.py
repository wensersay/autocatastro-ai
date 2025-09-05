"""
app.py — autoCata v0.9.3 (row+color fusion, anti-dirección)

Correcciones sobre 0.9.1 a raíz del caso «74 Marouzas Os Luis 14»:
- **Fuente de verdad = tabla de “Apellidos y nombre / Razón social”** (row-based).
  El OCR “cerca del vecino” por color queda como **fallback**. 
- **Filtro anti-dirección**: descartamos candidatos con dígitos o vocabulario
  de vía pública (CALLE, RÚA, AVDA, CAMIÑO, KM, Nº, etc.).
- **Fusión**: se combinan resultados (row primero, color después), sin duplicar.
- Se mantiene **asignación smart** por cobertura angular (cardinales > diagonal).
- Redacción **agrupada** por titular («Este y Sur, Fulano») y vaciado de diagonales.

Variables recomendadas (Railway → Variables):
- FAST_MODE=1, PDF_DPI=320, DPI_FAST=220, EDGE_BUDGET_MS=25000
- NEIGH_MIN_AREA_HARD=300, NEIGH_MAX_DIST_RATIO=2.2
- ROW_OWNER_LINES=2, SECOND_LINE_FORCE=1, SECOND_LINE_MAXCHARS=40, SECOND_LINE_MAXTOKENS=8
- PROPERCASE=1
- SECTOR_ASSIGN_MODE=smart
- DIAG_KEEP_DOMINANCE=0.55, CARD_PAIR_MIN_EACH=0.20, CARD_PAIR_MIN_COMBINED=0.50, CARD_SINGLE_MIN=0.30
- DIAG_TO_CARDINALS=0
- OWNER_ALLOW_DIGITS=0
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

__version__ = "0.9.3"

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
    second_line_maxchars: int = int(os.getenv("SECOND_LINE_MAXCHARS", "64"))
    second_line_maxtokens: int = int(os.getenv("SECOND_LINE_MAXTOKENS", "14"))
    second_line_scan_extra_pct: float = float(os.getenv("SECOND_LINE_SCAN_EXTRA_PCT", "0.03"))  # mini‑tweak: escaneo bajo la banda
    second_line_scan_max_pct: float = float(os.getenv("SECOND_LINE_SCAN_MAX_PCT", "0.06"))     # límite superior del escaneo escalonado

    neigh_min_area_hard: int = int(os.getenv("NEIGH_MIN_AREA_HARD", "600"))
    neigh_max_dist_ratio: float = float(os.getenv("NEIGH_MAX_DIST_RATIO", "1.8"))

    row_owner_lines: int = int(os.getenv("ROW_OWNER_LINES", "2"))

    propercase: bool = os.getenv("PROPERCASE", "1") == "1"

    sector_assign_mode: str = os.getenv("SECTOR_ASSIGN_MODE", "smart")
    diag_keep_dominance: float = float(os.getenv("DIAG_KEEP_DOMINANCE", "0.60"))
    card_pair_min_each: float = float(os.getenv("CARD_PAIR_MIN_EACH", "0.24"))
    card_pair_min_combined: float = float(os.getenv("CARD_PAIR_MIN_COMBINED", "0.50"))
    card_single_min: float = float(os.getenv("CARD_SINGLE_MIN", "0.28"))

    diag_to_cardinals: bool = os.getenv("DIAG_TO_CARDINALS", "0") == "1"

    # Multilínea robusto en columna de Nombres: alineación y separación vertical
    l2_align_tol_pct: float = float(os.getenv("L2_ALIGN_TOL_PCT", "0.12"))
    l2_max_gap_factor: float = float(os.getenv("L2_MAX_GAP_FACTOR", "1.8"))
    l2_max_lines: int = int(os.getenv("L2_MAX_LINES", "3"))

    owner_allow_digits: bool = os.getenv("OWNER_ALLOW_DIGITS", "1") == "1"

    # Snaps específicos para asignación basada en filas (row)
    row_angle_snap_deg: float = float(os.getenv("ROW_ANGLE_SNAP_DEG", "30"))
    row_card_dom_factor: float = float(os.getenv("ROW_CARD_DOM_FACTOR", "1.07"))

    angle_snap_deg: float = float(os.getenv("ANGLE_SNAP_DEG", "24"))

    # Cobertura mínima en cardinal para hacer snap desde diagonal por ángulo
    card_snap_min_cov: float = float(os.getenv("CARD_SNAP_MIN_COV", "0.12"))

    anti_header_kws: List[str] = field(default_factory=lambda: [
        s.strip() for s in os.getenv("ANTI_HEADER_KWS", "TITULARIDAD|PRINCIPAL|TITULAR|SECUNDARIA|S/S|SS").split("|")
        if s.strip()
    ])

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

ANTI_HEADER = set(strip_accents(s).upper() for s in CFG.anti_header_kws)

LOWER_CONNECTORS = {"de","del","la","las","los","y","da","do","das","dos"}
ACCENT_MAP = {
    "JOSE": "José", "RODRIGUEZ": "Rodríguez", "ALVAREZ": "Álvarez", "LOPEZ": "López",
    "FERNANDEZ": "Fernández", "VAZQUEZ": "Vázquez", "MARTIN": "Martín", "MARTINEZ": "Martínez",
    "PEREZ": "Pérez", "GOMEZ": "Gómez", "GARCIA": "García", "NUNEZ": "Núñez",
    "SANCHEZ": "Sánchez", "RAMON": "Ramón", "SAVINAO": "Saviñao"
}
GIVEN_NAMES = {
    "JOSE","JOSÉ","LUIS","MARIA","MARÍA","ANTONIO","MANUEL","ANA","JUAN","CARLOS","PABLO",
    "ROGELIO","FRANCISCO","MARTA","ELENA","LAURA","JULIO","RAMON","RAMÓN","RAFAEL","SERGIO"
}
# Permitir ampliar vía entorno: GIVEN_NAMES_EXTRA="PABLO|LUISA|..."
_gn_extra = os.getenv("GIVEN_NAMES_EXTRA", "").strip()
if _gn_extra:
    for _tok in _gn_extra.split("|"):
        _t = _tok.strip()
        if _t:
            GIVEN_NAMES.add(strip_accents(_t).upper())
UPPER_NAME_RE = re.compile(r"^[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\.'\-]+$", re.UNICODE)
BAD_TOKENS = {"POLÍGONO","POLIGONO","PARCELA","APELLIDOS","NOMBRE","RAZON","RAZÓN","SOCIAL","NIF","DOMICILIO","LOCALIZACIÓN","LOCALIZACION","REFERENCIA","CATASTRAL","TITULARIDAD","PRINCIPAL"}
GEO_TOKENS = {"LUGO","BARCELONA","MADRID","VALENCIA","SEVILLA","CORUÑA","A CORUÑA","MONFORTE","LEM","LEMOS","HOSPITALET","L'HOSPITALET","SAVIAO","SAVIÑAO","GALICIA","[LUGO]","[BARCELONA]"}
NAME_CONNECTORS = {"DE","DEL","LA","LOS","LAS","DA","DO","DAS","DOS","Y"}
ADDRESS_KWS = {
    "CALLE","CALLE.","RUA","RÚA","AVDA","AVENIDA","CAMINO","CAMIÑO","LUGAR","PLAZA","PRAZA",
    "KM","KILOMETRO","KILÓMETRO","Nº","NO.","NUM","NUM.","CODIGO","CÓDIGO","POSTAL","CP","PARROQUIA",
    "MUNICIPIO","CONCELLO","PROVINCIA","PAIS","PAÍS","ESPAÑA","ESPAÑA","PORTAL","ESCALERA","PISO"
}


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
    """Si el/los *últimos* tokens son nombre(s) de pila (1–2), muévelos delante.
    Ej.: "Pérez Taboada Julio" → "Julio Pérez Taboada";
         "Rodríguez Álvarez José Ramón" → "José Ramón Rodríguez Álvarez".
    """
    toks = s.split()
    if len(toks) < 2:
        return s

    tail: list[str] = []
    i = len(toks) - 1
    # Tomamos hasta 2 nombres al final
    for _ in range(2):
        if i < 0:
            break
        up = strip_accents(toks[i]).upper()
        if up in GIVEN_NAMES:
            tail.append(toks[i])
            i -= 1
            continue
        break

    if tail:
        tail = list(reversed(tail))
        rest = toks[: i + 1]
        return " ".join(tail + rest)
    return s

# --- Normalizador de orden en nombres compuestos comunes (pares) ---
_COMPOSITE_SWAP_PAIRS = {
    ("PABLO","JUAN"),       # → Juan Pablo
    ("LUISA","MARIA"),     # → María Luisa
    ("LUIS","JOSE"),       # → José Luis
    ("ANTONIO","JOSE"),    # → José Antonio
    ("MANUEL","JOSE"),     # → José Manuel
}

def normalize_given_pair_order(s: str) -> str:
    toks = s.split()
    if len(toks) >= 2:
        a = strip_accents(toks[0]).upper()
        b = strip_accents(toks[1]).upper()
        if (a, b) in _COMPOSITE_SWAP_PAIRS:
            toks[0], toks[1] = toks[1], toks[0]
            return " ".join(toks)
    return s

HONORIFICS = {
    "D","D.","Dª","D.ª","DÑA","DOÑA","DON","SR","SR.","SRA","SRA.","SRES.",
    "EXCMO.","EXCMA.","ILMO.","ILMA.",
}

def strip_honorifics_and_initials(s: str) -> str:
    """Elimina tratamientos (D., Don, Doña, Sr., …) y **iniciales sueltas** al inicio,
    p.ej. "A"/"A." que a veces aparece por OCR. Conserva conectores como "y".
    """
    toks = s.split()
    while toks:
        if not toks:
            break
        t0 = toks[0]
        up0 = strip_accents(t0).upper()
        # Inicial suelta de 1-2 chars (no "y") o tratamiento
        if (len(t0) <= 2 and up0 not in {"Y"}) or (up0 in HONORIFICS):
            # Quitar también variantes con punto (p. ej. "A.")
            if up0 in HONORIFICS or re.match(r"^[A-Z]\.?$", t0):
                toks = toks[1:]
                continue
        break
    return " ".join(toks)

    tail: list[str] = []
    i = len(toks) - 1
    # Tomamos hasta 2 nombres al final
    for _ in range(2):
        if i < 0:
            break
        up = strip_accents(toks[i]).upper()
        if up in GIVEN_NAMES:
            tail.append(toks[i])
            i -= 1
            continue
        break

    if tail:
        tail = list(reversed(tail))
        rest = toks[: i + 1]
        return " ".join(tail + rest)
    return s

HONORIFICS = {
    "D","D.","Dª","D.ª","DÑA","DOÑA","DON","SR","SR.","SRA","SRA.","SRES.",
    "EXCMO.","EXCMA.","ILMO.","ILMA.",
}

def strip_honorifics_and_initials(s: str) -> str:
    """Elimina tratamientos (D., Don, Doña, Sr., …) y **iniciales sueltas** al inicio,
    p.ej. "A"/"A." que a veces aparece por OCR. Conserva conectores como "y".
    """
    toks = s.split()
    while toks:
        if not toks:
            break
        t0 = toks[0]
        up0 = strip_accents(t0).upper()
        # Inicial suelta de 1-2 chars (no "y") o tratamiento
        if (len(t0) <= 2 and up0 not in {"Y"}) or (up0 in HONORIFICS):
            # Quitar también variantes con punto (p. ej. "A.")
            if up0 in HONORIFICS or re.match(r"^[A-Z]\.?$", t0):
                toks = toks[1:]
                continue
        break
    return " ".join(toks)
    toks = [t for t in re.split(r"\s+", s.strip()) if t]
    if len(toks) < 2:
        return s

    norm = [strip_accents(t).upper() for t in toks]
    GIVEN = {
        "JOSE","JOSÉ","JUAN","PABLO","LUIS","LUISA","MARIA","MARÍA",
        "SERGIO","ELENA","ANTONIO","MANUEL","CARLOS","ANA","LAURA","MARTA"
    }

    tail = 0
    i = len(toks) - 1
    while i >= 0 and tail < 2 and norm[i] in GIVEN:
        tail += 1
        i -= 1

    if tail == 0:
        return s

    names_tail = toks[-tail:]
    surnames   = toks[:-tail]

    out = names_tail + surnames
    pretty = propercase_spanish(" ".join(out))
    return pretty


def postprocess_name(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) == 2 and len(parts[0]) >= 3 and len(parts[1]) >= 2:
            s = f"{parts[1]} {parts[0]}"
    # Normaliza espacios sin usar regex con escapes
    s = " ".join(s.split())

    s = strip_honorifics_and_initials(s)
    s = propercase_spanish(s)
    s = reorder_surname_first(s)
    s = normalize_given_pair_order(s)
    return s


def looks_like_address(s: str) -> bool:
    up = strip_accents(s).upper()
    if any(kw in up for kw in ADDRESS_KWS):
        return True
    if re.search(r"\b\d{1,4}\b", up):  # número suelto típico de portal/km
        return True
    return False


def drop_anti_header_tokens(s: str) -> str:
    if not s:
        return ""
    toks = s.strip().split()
    kept = []
    for t in toks:
        raw = t.strip(".,;:()[]{}")
        norm = strip_accents(raw).upper()
        if norm in ANTI_HEADER:
            continue
        kept.append(t)
    out = " ".join([t for t in kept if t]).strip()
    return " ".join(out.split())


def clean_candidate_text(s: str) -> str:
    if not s:
        return ""
    # 1) Quitar tokens de cabecera de tabla
    s = drop_anti_header_tokens(s)

    # 2) Sanitizar sin regex
    out_chars = []
    for ch in s:
        cat = unicodedata.category(ch)
        if cat.startswith('L') or cat.startswith('N') or ch in " .'-":
            out_chars.append(ch)
        else:
            out_chars.append(' ')
    s = "".join(out_chars)
    s = " ".join(s.split()).strip(" .-'")
    up = strip_accents(s).upper()

    # 3) Defensa frente a símbolos / rótulos
    SYMBOLS_BLOCK = ["=", "*", "_", "/", "|", "[", "]", "{", "}", "<", ">"]
    for ch in SYMBOLS_BLOCK:
        if ch in up:
            return ""
    for ch in SYMBOLS_BLOCK:
        if ch in up:
            return ""
    STOP = {
    # rótulos generales de plano/tabla
    "DATOS","ESCALA","LEYENDA","MAPA","COORDENADAS","COORD","REFERENCIA","CATASTRAL",
    # parcela/polígono y variantes abreviadas
    "PARCELA","PARCELAS","PARCELARIO","PARC","POLIGONO","POLÍGONO","HOJA",
    # listados y cabeceras frecuentes en catastro
    "RELACION","RELACIÓN","LISTADO","LISTA","RELACIÓN DE","RELACION DE",
    # otros rótulos administrativos
    "Nº","NO","NUM","NUM.","CODIGO","CÓDIGO","POSTAL","CP"
}
    if any(kw in up for kw in STOP):
        return ""

    # 4) Direcciones / números
    if looks_like_address(up):
        return ""
    if not CFG.owner_allow_digits and any(c.isdigit() for c in up):
        if not any(tok in up for tok in ("S.L","S L","S.A","S A","SCOOP","COOP"," CB","CB ")):
            return ""

    # 5) Requiere al menos 2 tokens "de nombre"
    tokens = s.split()
    good = [t for t in tokens if (len(t) >= 2 and (t[0].isupper() or t.isupper()))]
    return " ".join(tokens[:7]) if len(good) >= 2 else ""

# --- Normalización y similitud de titulares (dedupe row+color) ---
TOKEN_STOP = set(t.upper() for t in NAME_CONNECTORS)

def _norm_tokens_for_match(s: str) -> List[str]:
    s = strip_accents((s or "")).upper()
    s = re.sub(r"[^A-Z0-9\s]", " ", s)
    toks = [t for t in s.split() if t and t not in TOKEN_STOP]
    # quitar tokens de 1 letra y números aislados
    toks = [t for t in toks if (len(t) > 1 and not t.isdigit())]
    return toks

def _jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb) or 1
    return inter / union

def same_person(a: str, b: str) -> bool:
    ta = _norm_tokens_for_match(a)
    tb = _norm_tokens_for_match(b)
    if not ta or not tb:
        return False
    # contención fuerte o jaccard alto
    sa, sb = set(ta), set(tb)
    if len(sa) >= 2 and (sa <= sb or sb <= sa):
        return True
    return _jaccard(ta, tb) >= 0.7

# ----------------------------------------
# OCR básicos
# ----------------------------------------

def ocr_image_to_data_lines(img_bgr: np.ndarray) -> List[Tuple[str, Tuple[int,int,int,int]]]:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    data = pytesseract.image_to_data(pil, config="--oem 1 --psm 6 -l spa+eng", output_type=pytesseract.Output.DICT)
    n = len(data.get("text", []))
    lines: Dict[Tuple[int,int], Dict] = {}
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        conf = data.get("conf", [-1]*n)[i]
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


def extract_name_multiline_from_roi(roi_bgr: np.ndarray) -> str:
    """Une L1 + (L2, L3) dentro de la celda de nombres, de forma robusta.
    No limpiamos cada línea por separado para no descartar segundas líneas de 1 token
    (p.ej., "Pablo"). Limpiamos sólo al final.
    Criterios L2/L3:
      - Alineación izquierda similar (tolerancia % ancho ROI) **o** IoU horizontal ≥ 0.40 con L1.
      - Salto vertical acotado por altura media de línea.
      - Aceptamos L2/L3 de 1–3 tokens si parecen nombres (GIVEN_NAMES o patrón Capitalizada).
    """
    lines = ocr_image_to_data_lines(roi_bgr)
    if not lines:
        return ""

    W = roi_bgr.shape[1]
    tol_x = int(max(2, CFG.l2_align_tol_pct * W))

    # Datos de L1
    text0, (x0, y0, w0, h0) = lines[0]
    base_left = x0
    base_right = x0 + w0

    # Altura media de línea
    hs = [bbox[3] for (_t, bbox) in lines]
    med_h = float(np.median(hs)) if hs else max(1.0, h0)

    def horiz_iou(ax0, ax1, bx0, bx1) -> float:
        inter = max(0, min(ax1, bx1) - max(ax0, bx0))
        if inter <= 0:
            return 0.0
        union = (ax1 - ax0) + (bx1 - bx0) - inter
        return inter / max(1.0, union)

    # Empezamos con L1 sin limpiar todavía; limpieza sólo al final
    joined_raw: list[str] = [postprocess_name(text0)]
    last_bottom = y0 + h0
    used = 1

    for i in range(1, len(lines)):
        if used >= max(1, CFG.l2_max_lines):
            break
        t, (lx, ly, lw, lh) = lines[i]

        # Filtro vertical: no subir y controlar salto excesivo
        if ly < y0:
            continue
        gap = ly - last_bottom
        if gap > CFG.l2_max_gap_factor * med_h:
            continue

        # Alineación u overlap horizontal respecto a L1
        aligned = abs(lx - base_left) <= tol_x
        iou_ok = horiz_iou(lx, lx + lw, base_left, base_right) >= 0.40
        if not (aligned or iou_ok):
            continue

        # Texto crudo de la línea, sin limpiar por completo
        t_raw = drop_anti_header_tokens(postprocess_name(t))
        toks = [tok for tok in t_raw.split() if tok]
        if not toks:
            continue

        accept = False
        # Caso típico: una o dos palabras (p.ej., "Pablo", "María Luisa")
        if 1 <= len(toks) <= 3:
            if all(not any(ch.isdigit() for ch in tok) for tok in toks):
                namey = sum(
                    1 for tok in toks
                    if strip_accents(tok).upper() in GIVEN_NAMES or re.match(r"^[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]{1,}$", tok)
                )
                if namey >= 1:
                    accept = True
        else:
            # Si es más larga, comprobar con limpieza clásica
            tcheck = clean_candidate_text(t_raw)
            if tcheck:
                t_raw = tcheck
                accept = True

        if not accept:
            continue

        joined_raw.append(" ".join(toks))
        last_bottom = ly + lh
        used += 1

    full = clean_candidate_text(postprocess_name(" ".join([p for p in joined_raw if p])))
    return full


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
# PDF → imagen + máscaras y contornos
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
# Cobertura angular y decisión de sectores (smart)
# ----------------------------------------

SIDES8 = ["este","noreste","norte","noroeste","oeste","suroeste","sur","sureste"]
PAIRS = {
    "noreste": ("norte", "este"),
    "sureste": ("sur", "este"),
    "suroeste": ("sur", "oeste"),
    "noroeste": ("norte", "oeste"),
}

CARDINAL_CENTERS = {
    "este": 0.0, "noreste": 45.0, "norte": 90.0, "noroeste": 135.0,
    "oeste": 180.0, "suroeste": 225.0, "sur": 270.0, "sureste": 315.0,
}

def _ang_dist(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)

def snap_diagonal_to_cardinal_by_angle(diag: str, angle: float, cov: Dict[str, float]) -> Optional[str]:
    """Si un vecino cae angularmente muy cerca de un cardinal, preferir el cardinal.
    Requiere proximidad angular (<= ANGLE_SNAP_DEG) y una cobertura mínima en el cardinal.
    """
    if diag not in PAIRS:
        return None
    a, b = PAIRS[diag]
    da = _ang_dist(angle, CARDINAL_CENTERS[a])
    db = _ang_dist(angle, CARDINAL_CENTERS[b])
    thr = CFG.angle_snap_deg
    # cobertura mínima configurable
    min_cov = CFG.card_snap_min_cov
    cand = None
    if da <= db and da <= thr and cov.get(a, 0.0) >= min_cov:
        cand = a
    elif db < da and db <= thr and cov.get(b, 0.0) >= min_cov:
        cand = b
    return cand

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
    """Decisión de lados con preferencia a cardinal cuando domina.
    - Si ambos cardinales de una diagonal están bien presentes → pareja.
    - Si una diagonal es débil o un cardinal domina claramente → ese cardinal.
    - Si un cardinal supera el umbral y es el mejor → ese cardinal.
    - Último recurso: el mejor lado bruto.
    """
    # 1) Pareja de cardinales fuerte
    for diag, (a, b) in PAIRS.items():
        if (
            cov[a] >= CFG.card_pair_min_each
            and cov[b] >= CFG.card_pair_min_each
            and (cov[a] + cov[b]) >= CFG.card_pair_min_combined
        ):
            return [a, b]

    # 2) Mejor lado bruto
    best_side = max(SIDES8, key=lambda s: cov.get(s, 0.0))
    best_val  = cov.get(best_side, 0.0)

    # 3) Snap-to-cardinal si la diagonal no domina o un cardinal domina claramente
    CARD_DOM_FACTOR = 1.20    # ~20% mayor que el otro
    DIAG_WEAK_MAX   = 0.42    # diagonal débil

    if best_side in PAIRS:
        a, b = PAIRS[best_side]
        a_v, b_v = cov.get(a, 0.0), cov.get(b, 0.0)
        if (
            best_val < DIAG_WEAK_MAX
            or (a_v >= CFG.card_single_min and a_v >= b_v * CARD_DOM_FACTOR)
            or (b_v >= CFG.card_single_min and b_v >= a_v * CARD_DOM_FACTOR)
        ):
            if a_v >= b_v and a_v >= CFG.card_single_min:
                return [a]
            if b_v >  a_v and b_v >= CFG.card_single_min:
                return [b]

    # 4) Cardinal fuerte y además mejor
    for c in ("este","sur","oeste","norte"):
        if cov.get(c, 0.0) >= CFG.card_single_min and c == best_side:
            return [c]

    # 5) Último recurso
    return [best_side]

# ----------------------------------------
# OCR cerca del vecino (fallback)
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
        if sum(ch.isdigit() for ch in U) > 0:
            continue
        if not UPPER_NAME_RE.match(U):
            continue
        name = _clean_owner_line(U)
        if len(name) >= 6:
            return name
    return ""


def _find_header_and_owner_band(bgr: np.ndarray, row_y: int, x_text0: int, x_text1: int, lines: int = 2) -> Tuple[int,int,int,int]:
    """Devuelve (x0, x1, y0, y1) para la banda del NOMBRE del titular.
    Se busca 'APELLIDOS' y se coloca y0 justo debajo. Alto ampliado para capturar L2 (mini‑tweak ROI).
    """
    h, w = bgr.shape[:2]

    # Altura base de la banda y extra si forzamos 2ª línea (mini‑tweak ROI)
    lh = int(h * (0.075 if lines == 1 else 0.08))
    extra_lh = int(h * (0.020 if lines == 1 else 0.028)) if CFG.second_line_force else 0

    pad_y = int(h * 0.06)
    y0s = max(0, row_y - pad_y)
    y1s = min(h, row_y + pad_y)

    band = bgr[y0s:y1s, x_text0:x_text1]
    if band.size == 0:
        # Fallback conservador
        y0 = max(0, row_y - int(h * 0.012))
        y1 = min(h, y0 + lh + extra_lh)
        x0 = x_text0
        x1 = int(x_text0 + 0.58 * (x_text1 - x_text0))
        return x0, x1, y0, y1

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
            if not t:
                continue
            T = t.upper()
            if "APELLIDOS" in T:
                header_bottom = max(header_bottom or 0, ty + hh)
            if T == "NIF":
                x_nif = lx
                header_bottom = max(header_bottom or 0, ty + hh)

        if header_bottom is not None:
            y0 = y0s + header_bottom + 6
            y1 = min(h, y0 + lh + extra_lh)
            if x_nif is not None:
                x0 = x_text0
                x1 = min(x_text1, x_text0 + x_nif - 8)
            else:
                x0 = x_text0
                x1 = int(x_text0 + 0.58 * (x_text1 - x_text0))
            if x1 - x0 > (x_text1 - x_text0) * 0.22:
                return x0, x1, y0, y1

    # Fallback si no se detecta cabecera 'APELLIDOS'/'NIF'
    y0 = max(0, row_y - int(h * 0.012))
    y1 = min(h, y0 + lh + extra_lh)
    x0 = x_text0
    x1 = int(x_text0 + 0.58 * (x_text1 - x_text0))
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
        if owner:
            # Intentar completar con nombre multilínea dentro del ROI (captura L2/L3)
            # Expandimos la ROI hacia abajo para intentar capturar L2/L3
            y_extra = int(max((y1 - y0) * 0.8, roi.shape[0] * 0.5)) if roi.size else 0
            y1e = min(bgr.shape[0], y1 + max(8, int(max(CFG.second_line_scan_extra_pct * bgr.shape[0], y_extra))))
            roi_ext = bgr[y0:y1e, x0:x1]
            owner_ml = extract_name_multiline_from_roi(roi_ext)
            # Preferimos la versión multilínea si añade tokens o contiene al nombre base
            if owner_ml:
                if len(owner_ml.split()) > len(owner.split()) or owner_ml.startswith(owner.split()[0]):
                    owner = owner_ml
        else:
            rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            lines_data = ocr_image_to_data_lines(rgb)
            if lines_data:
                # Candidatos: L1, L1+L2, L1+L2+L3 (si son cortas)
                segs = [(lines_data[j][0] or "").strip() for j in range(min(3, len(lines_data)))]
                variants = []
                if segs and segs[0]:
                    variants.append(segs[0])
                    if len(segs) >= 2 and len(segs[1]) <= CFG.second_line_maxchars and len(segs[1].split()) <= CFG.second_line_maxtokens:
                        variants.append(segs[0] + " " + segs[1])
                    if len(segs) >= 3 and len(segs[2]) <= CFG.second_line_maxchars and len(segs[2].split()) <= CFG.second_line_maxtokens:
                        variants.append(segs[0] + " " + segs[1] + " " + segs[2])

                    # NUEVO: si L2/L3 son 1–2 tokens y parecen nombres propios, añadirlos al final de L1
                    extra_given: list[str] = []
                    # Si no se detecta en L2/L3, probar una micro‑banda por debajo del ROI (por si L2 quedó fuera)
                    # Escaneo escalonado por debajo del ROI para capturar L2 si quedó fuera
                    if not extra_given and CFG.second_line_scan_extra_pct > 0:
                        spans = []
                        base = max(0.0, float(CFG.second_line_scan_extra_pct))
                        top  = max(base, float(getattr(CFG, "second_line_scan_max_pct", 0.06)))
                        mid  = (base + top) / 2.0
                        for pct in [base, mid, top]:
                            if pct not in spans:
                                spans.append(pct)
                        for pct in spans:
                            y0b = min(h - 1, y1 + 1)
                            y1b = min(h, y0b + max(int(h * pct), max(24, int((y1 - y0) * 0.5))))
                            if y1b <= y0b:
                                continue
                            roi_below = bgr[y0b:y1b, x0:x1]
                            if roi_below.size == 0:
                                continue
                            gg = cv2.cvtColor(roi_below, cv2.COLOR_BGR2GRAY)
                            gg = cv2.resize(gg, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
                            gg = _enhance_gray(gg)
                            # Usar líneas para capturar 'Pablo' incluso si aparece solo en una línea
                            # Usar extractor multilinea robusto directamente sobre ROI a color
                            name_below_full = extract_name_multiline_from_roi(roi_below)
                            if name_below_full:
                                toks = [t for t in name_below_full.split() if t]
                                extra_taken: list[str] = []
                                for t in toks:
                                    up = strip_accents(t).upper()
                                    if up in NAME_CONNECTORS:
                                        continue
                                    # Aceptar 1–2 nombres de pila probables
                                    if up in GIVEN_NAMES or re.match(r"^[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]{1,}$", t):
                                        extra_taken.append(t)
                                        if len(extra_taken) >= 2:
                                            break
                                    else:
                                        break
                                if extra_taken:
                                    extra_given.extend(extra_taken)
                                    break
                    for j in (1, 2):
                        if j < len(segs) and segs[j]:
                            toksj = [t for t in segs[j].split() if t]
                            if 1 <= len(toksj) <= 2:
                                buf: list[str] = []
                                all_given = True
                                for t in toksj:
                                    up = strip_accents(t).upper()
                                    if up in GIVEN_NAMES:
                                        buf.append(t)
                                    else:
                                        all_given = False
                                        break
                                if all_given:
                                    extra_given.extend(buf)
                    if extra_given:
                        # Variante A: insertar en L1 tal cual (tras el bloque de nombres de pila inicial)
                        base_tokens = [t for t in segs[0].split() if t]
                        i = 0
                        while i < len(base_tokens) and strip_accents(base_tokens[i]).upper() in GIVEN_NAMES:
                            i += 1
                        prefix = base_tokens[:i]
                        rest = base_tokens[i:]
                        # Evitar duplicar nombres ya presentes en el prefijo
                        pref_set = {strip_accents(t).upper() for t in prefix}
                        extra_clean = [t for t in extra_given if strip_accents(t).upper() not in pref_set]
                        merged = prefix + extra_clean + rest
                        variants.insert(0, " ".join(merged))

                        # Variante B: reordenar L1 primero (apellidos→final) y luego insertar tras los nombres de pila
                        l1_reord = reorder_surname_first(segs[0])
                        base_tokens2 = [t for t in l1_reord.split() if t]
                        i2 = 0
                        while i2 < len(base_tokens2) and strip_accents(base_tokens2[i2]).upper() in GIVEN_NAMES:
                            i2 += 1
                        prefix2 = base_tokens2[:i2]
                        rest2 = base_tokens2[i2:]
                        pref_set2 = {strip_accents(t).upper() for t in prefix2}
                        extra_clean2 = [t for t in extra_given if strip_accents(t).upper() not in pref_set2]
                        merged2 = prefix2 + extra_clean2 + rest2
                        variants.insert(0, " ".join(merged2))

                # Puntuar por nº tokens útiles (máx) y longitud
                def _score_name(s: str) -> Tuple[int,int]:
                    t = clean_candidate_text(postprocess_name(s))
                    return (len(t.split()), len(t))
                best_txt, best_sc = "", (-1,-1)
                for v in variants:
                    cand = _pick_owner_from_text(v)
                    if not cand:
                        cand = _pick_owner_from_text(reorder_surname_first(v))
                    t = clean_candidate_text(postprocess_name(cand)) if cand else ""
                    sc = (len(t.split()), len(t))
                    if sc > best_sc:
                        best_sc, best_txt = sc, t
                owner = best_txt or owner
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
    def _centroids(mask: np.ndarray, min_area: int):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cnts:
            a = cv2.contourArea(c)
            if a < min_area: continue
            M = cv2.moments(c)
            if M["m00"] == 0: continue
            cx = int(M["m10"] / M["m00"]); cy = int(M["m01"] / M["m00"])
            out.append((cx, cy, int(a)))
        out.sort(key=lambda x: -x[2])
        return out
    mg, mp = detect_masks(crop)
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
    def side_of(main_xy: Tuple[int,int], pt_xy: Tuple[int,int]) -> str:
        cx, cy = main_xy; x, y = pt_xy
        sx, sy = x - cx, y - cy
        ang = math.degrees(math.atan2(-(sy), sx))
        if -22.5 <= ang <= 22.5: return "este"
        if 22.5 < ang <= 67.5:  return "noreste"
        if 67.5 < ang <= 112.5: return "norte"
        if 112.5 < ang <= 157.5:return "noroeste"
        if -67.5 <= ang < -22.5:return "sureste"
        if -112.5 <= ang < -67.5:return "sur"
        if -157.5 <= ang < -112.5:return "suroeste"
        return "oeste"
    for (mcx, mcy, _a) in mains_abs[:8]:
        best = None; best_d = 1e18
        for (nx, ny, _na) in neighs_abs:
            d = (nx-mcx)**2 + (ny-mcy)**2
            if d < best_d:
                best_d = d; best = (nx, ny)
        side = ""
        snapped_to = None
        row_angle = None
        if best is not None and best_d < (w*0.28)**2:
            side = side_of((mcx, mcy), best)
            # snap de diagonal a cardinal según geometría row (sin cobertura)
            dx, dy = best[0]-mcx, best[1]-mcy
            row_angle = (math.degrees(math.atan2(-(dy), dx)) + 360.0) % 360.0
            if side in ("noreste","noroeste","suroeste","sureste"):
                # criterio 1: dominancia horizontal/vertical
                ax, ay = abs(dx), abs(dy)
                f = CFG.row_card_dom_factor
                if ax >= ay * f:
                    snapped_to = "este" if dx > 0 else "oeste"
                elif ay >= ax * f:
                    snapped_to = "sur" if dy > 0 else "norte"
                # criterio 2: cercanía angular a cardinal
                if not snapped_to:
                    centers = {"este":0.0, "norte":90.0, "oeste":180.0, "sur":270.0}
                    def _angdist(a,b):
                        d = abs((a-b) % 360.0); return min(d, 360.0-d)
                    cand = min(centers.items(), key=lambda kv: _angdist(row_angle, kv[1]))[0]
                    if _angdist(row_angle, centers[cand]) <= CFG.row_angle_snap_deg:
                        snapped_to = cand
                if snapped_to:
                    side = snapped_to
        owner, _roi, attempt_id = _extract_owner_from_row(bgr, row_y=mcy, lines=max(1, CFG.row_owner_lines))
        if side and owner:
            if owner not in ldr[side]:
                ldr[side].append(owner); used_rows += 1
        rows_dbg.append({"row_y": mcy, "main_center": [mcx, mcy], "neigh_center": list(best) if best else None, "side": side, "owner": owner, "roi_attempt": attempt_id, "row_angle": row_angle, "snapped_to": snapped_to})
    return ldr, {"rows": rows_dbg, "used_rows": used_rows}

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
    owner_sides: Dict[str, List[str]] = {}
    for side in SIDE_ORDER:
        for nm in owners_by_side.get(side, []) or []:
            owner_sides.setdefault(nm, []).append(side)
    sides_left: Dict[str, List[str]] = {s: [] for s in SIDE_ORDER}
    grouped: List[Tuple[List[str], str]] = []
    for owner, sides in owner_sides.items():
        if len(sides) >= 2:
            grouped.append((sorted(sides, key=lambda x: SIDE_ORDER.index(x)), owner))
        else:
            s = sides[0]
            sides_left.setdefault(s, []).append(owner)
    for side in SIDE_ORDER:
        for nm in owners_by_side.get(side, []) or []:
            if nm not in owner_sides or len(owner_sides[nm]) == 1:
                if nm not in sides_left[side]:
                    sides_left[side].append(nm)
    parts: List[str] = []
    grouped.sort(key=lambda t: SIDE_ORDER.index(t[0][0]))
    for sides, owner in grouped:
        parts.append(f"{join_sides_spanish(sides)}, {owner}")
    for side in SIDE_ORDER:
        arr = list(dict.fromkeys(sides_left[side]))
        if not arr:
            continue
        if len(arr) == 1:
            parts.append(f"{side.capitalize()}, {arr[0]}")
        else:
            parts.append(f"{side.capitalize()}, " + ", ".join(arr[:-1]) + f" y {arr[-1]}")
    return ("Linda: " + "; ".join(parts) + ".") if parts else "No se han podido determinar linderos suficientes para una redacción notarial fiable."

# ----------------------------------------
# Proceso principal
# ----------------------------------------

def process_pdf(pdf_bytes: bytes) -> ExtractResult:
    bgr = load_map_page_bgr(pdf_bytes)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    txt_gate = pytesseract.image_to_string(Image.fromarray(rgb), config="--oem 1 --psm 6 -l spa+eng") or ""
    if len(txt_gate.strip()) < 12:
        raise HTTPException(status_code=400, detail="El PDF no contiene texto OCR legible o metadatos reconocibles para su análisis.")

    # 1) Row-based (fuente de verdad)
    ldr_row, rows_dbg = detect_rows_and_extract8(bgr)

    # 2) Color-based (con cobertura angular + smart) — fallback/complemento
    owners_detected_union: List[str] = []
    mask_green, mask_pink = detect_masks(bgr)
    subs = contours_and_centroids(mask_green)
    neigh = contours_and_centroids(mask_pink)
    subj = subs[0] if subs else None

    ldr_color: Dict[str, List[str]] = {k: [] for k in SIDE_ORDER}
    debug_rows: List[Dict] = []

    if subj:
        subj_c = subj["centroid"]
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
            angle_val = angle_between(subj_c, nb["centroid"])
            sides = decide_sides_smart(coverage) if CFG.sector_assign_mode == "smart" else [angle_to_8(angle_val)]
            # snap diagonal → cardinal por ángulo si procede
            if len(sides) == 1 and sides[0] in PAIRS:
                snapped = snap_diagonal_to_cardinal_by_angle(sides[0], angle_val, coverage)
                if snapped:
                    sides = [snapped]
            name = ocr_name_near_with_linejoin(bgr, nb["bbox"], subj_c)
            if not name:
                continue
            for side in sides:
                if name not in ldr_color[side]:
                    ldr_color[side].append(name)
            if name not in owners_detected_union:
                owners_detected_union.append(name)
            debug_rows.append({"neighbor_bbox": list(nb["bbox"]), "centroid": list(nb["centroid"]), "angle": angle_val, "coverage": coverage, "assigned": sides, "owner": name})

    # 3) Fusión (prioridad a row-based)
    # 3.a) Canonizar lista de titulares encontrados por tabla (row)
    row_canon: List[str] = []
    for side in SIDE_ORDER:
        for nm in (ldr_row.get(side, []) or []):
            nm_pp = clean_candidate_text(postprocess_name(nm))
            if nm_pp and all(not same_person(nm_pp, ex) for ex in row_canon):
                row_canon.append(nm_pp)

    # 3.b) Inicializar salida con los row tal cual por lado
    ldr_out: Dict[str, List[str]] = {k: [] for k in SIDE_ORDER}
    for side in SIDE_ORDER:
        seen_side: List[str] = []
        for nm in (ldr_row.get(side, []) or []):
            nm_pp = clean_candidate_text(postprocess_name(nm))
            if nm_pp and all(not same_person(nm_pp, ex) for ex in seen_side):
                ldr_out[side].append(nm_pp)
                seen_side.append(nm_pp)

    # 3.c) Añadir color mapeando a canónicos de row si son la misma persona (respeta el lado del row)
    for side in SIDE_ORDER:
        for nm in (ldr_color.get(side, []) or []):
            nm_pp = clean_candidate_text(postprocess_name(nm))
            if not nm_pp:
                continue
            # mapear a canónico de row si hace match
            mapped = None
            for r in row_canon:
                if same_person(nm_pp, r):
                    mapped = r
                    break
            candidate = mapped or nm_pp
            # si el candidato ya aparece por row en algún lado, **no** lo dupliques en otro lado: respeta el/los lados del row
            row_sides_for_candidate = [s for s in SIDE_ORDER if any(same_person(candidate, ex) for ex in ldr_out[s])]
            if row_sides_for_candidate:
                continue
            # si no aparece aún por row, añadir en el lado propuesto por color evitando duplicados por similitud
            if any(same_person(candidate, ex) for ex in ldr_out[side]):
                continue
            ldr_out[side].append(candidate)

    # 3.d) Limpiar diagonales por titular: solo si el **mismo titular** aparece en **ambos** cardinales
    for diag, (a, b) in PAIRS.items():
        if ldr_out[diag]:
            keep = []
            for nm in ldr_out[diag]:
                in_a = any(same_person(nm, ex) for ex in ldr_out[a])
                in_b = any(same_person(nm, ex) for ex in ldr_out[b])
                # eliminar de la diagonal únicamente si está en A **y** en B
                if in_a and in_b:
                    continue
                keep.append(nm)
            ldr_out[diag] = keep

    # 5) Notarial agrupado
    notarial = generate_notarial_text_grouped(ldr_out)
    files = {"notarial_text.txt": base64.b64encode(notarial.encode("utf-8")).decode("ascii")}

    debug = {"rows": rows_dbg.get("rows", []) + debug_rows, "subject_centroid": list(subs[0]["centroid"]) if subs else None}

    owners_detected = []
    for side in SIDE_ORDER:
        for nm in ldr_out[side]:
            if nm not in owners_detected:
                owners_detected.append(nm)

    return ExtractResult(
        linderos=Linderos(**ldr_out),
        owners_detected=owners_detected[:24],
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





