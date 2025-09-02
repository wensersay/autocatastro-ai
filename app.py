from fastapi import FastAPI, HTTPException, Body, Depends, Header, Query, UploadFile, File
from pydantic import BaseModel, AnyHttpUrl
from starlette.responses import StreamingResponse
from typing import Dict, List, Optional, Tuple
import requests, io, re, os, math, time, asyncio
import numpy as np
from pdf2image import convert_from_bytes
from PIL import Image
import cv2
import pytesseract

# ──────────────────────────────────────────────────────────────────────────────
# App & versión
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="AutoCatastro AI", version="0.6.7"  # PATCH: actualizada por integración de casos límite 02–07)

# ──────────────────────────────────────────────────────────────────────────────
# Flags de entorno / seguridad
# ──────────────────────────────────────────────────────────────────────────────
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "").strip()
FAST_MODE  = (os.getenv("FAST_MODE",  "1").strip() == "1")
TEXT_ONLY  = (os.getenv("TEXT_ONLY",  "0").strip() == "1")
USE_PATCHES = True  # PATCH: Permite activar/desactivar lógica de casos límite


# DPI / Raster
FAST_DPI = int(os.getenv("FAST_DPI", "300").strip() or "300")
PDF_DPI  = int(os.getenv("PDF_DPI",  "340").strip() or "340")
AUTO_DPI = (os.getenv("AUTO_DPI", "0").strip() == "1")
DPI_LADDER = [int(x) for x in re.split(r"[,\s]+", os.getenv("DPI_LADDER", "300,340").strip()) if x]

# 2ª línea: tokens ruidosos a ignorar
JUNK_2NDLINE = set([t.strip().upper() for t in (os.getenv("JUNK_2NDLINE", "Z,VA,EO,SS,KO,KR").split(",")) if t.strip()])

# Reordenador Nombre(s) + Apellidos
REORDER_TO_NOMBRE_APELLIDOS = os.getenv("REORDER_TO_NOMBRE_APELLIDOS", "1").strip() == "1"
REORDER_MIN_CONF = float(os.getenv("REORDER_MIN_CONF", "0.70"))
NAME_HINTS_EXTRA = os.getenv("NAME_HINTS_EXTRA", "").strip()
NAMES_FILE = os.getenv("NAME_HINTS_FILE", "data/nombres_es.txt")

# ──────────────────────────────────────────────────────────────────────────────
# Autorización
# ──────────────────────────────────────────────────────────────────────────────
def check_token(x_autocata_token: str = Header(default="")):
    if AUTH_TOKEN and x_autocata_token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ──────────────────────────────────────────────────────────────────────────────
# Modelos I/O
# ──────────────────────────────────────────────────────────────────────────────
class ExtractIn(BaseModel):
    pdf_url: AnyHttpUrl

class ExtractOut(BaseModel):
    linderos: Dict[str, str]
    owners_detected: List[str] = []
    note: Optional[str] = None
    debug: Optional[dict] = None

# Batch
class ExtractManyIn(BaseModel):
    pdf_urls: List[AnyHttpUrl]

class ExtractManyItem(BaseModel):
    pdf_url: AnyHttpUrl
    ok: bool
    linderos: Dict[str, str]
    owners_detected: List[str] = []
    note: Optional[str] = None
    debug: Optional[dict] = None

# ──────────────────────────────────────────────────────────────────────────────
# Utilidades comunes (limpieza / nombres / diccionarios)
# ──────────────────────────────────────────────────────────────────────────────
UPPER_NAME_RE = re.compile(r"^[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\.'\-]+$", re.UNICODE)

BAD_TOKENS = {
    "POLÍGONO","POLIGONO","PARCELA","APELLIDOS","NOMBRE","RAZON","RAZÓN",
    "SOCIAL","NIF","DOMICILIO","LOCALIZACIÓN","LOCALIZACION","REFERENCIA",
    "CATASTRAL","TITULARIDAD","PRINCIPAL","APELLIDOS/NOMBRE","APELLIDOS/NOMBRE/RAZON",
    "APELLIDOS/NOMBRE/RAZÓN"
}

# geotokens / ruido que no deben entrar en el nombre
GEO_TOKENS = {
    "LUGO","BARCELONA","MADRID","VALENCIA","SEVILLA","CORUÑA","A CORUÑA",
    "MONFORTE","LEM","LEMOS","HOSPITALET","L'HOSPITALET","SAVIAO","SAVIÑAO",
    "GALICIA","[LUGO]","[BARCELONA]","O","DE","DEL","DA","DO"  # ojo con direcciones
}

NAME_CONNECTORS = {"DE","DEL","LA","LOS","LAS","DA","DO","DAS","DOS","Y"}

# Reordenador: carga de nombres comunes
def load_name_hints() -> set:
    base = set()
    try:
        if os.path.exists(NAMES_FILE):
            with open(NAMES_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    t = line.strip().upper()
                    if t:
                        base.add(t)
    except Exception:
        pass
    if NAME_HINTS_EXTRA:
        for t in re.split(r"[,\s]+", NAME_HINTS_EXTRA.upper()):
            t = t.strip()
            if t:
                base.add(t)
    if not base:
        base |= {"JOSE","LUIS","JUAN","ANTONIO","MANUEL","MIGUEL","JAVIER","CARLOS",
                 "ALEJANDRO","PABLO","MARIA","ANA","LAURA","MARTA","SARA"}
    return base

NAME_HINTS = load_name_hints()

ORG_TOKENS = {"S.L","S.A","SL","SA","SLU","SCOOP","S.COOP","SC","CB","SCP","AYUNTAMIENTO",
              "CONCELLO","DIOCESIS","DIOCESÍS","PARROQUIA","IGLESIA","SOCIEDAD","FUNDACION",
              "FUNDACIÓN","ASOCIACION","ASOCIACIÓN","COMUNIDAD","COMUNIDADE"}

def looks_like_org(name: str) -> bool:
    U = name.upper()
    if any(tok in U for tok in ORG_TOKENS):
        return True
    if sum(ch.isdigit() for ch in U) >= 2:
        return True
    return False

def split_tokens(s: str) -> list:
    return [t for t in re.split(r"[^\wÁÉÍÓÚÜÑ'-]+", s.upper()) if t]

def confidence_trailing_given(tokens: list) -> Tuple[float,int]:
    if not tokens:
        return 0.0, 0
    i = len(tokens)-1
    given = 0
    while i >= 0:
        t = tokens[i]
        if t in NAME_HINTS or t in NAME_CONNECTORS:
            given += 1
            i -= 1
        else:
            break
    if given == 0:
        return 0.0, 0
    if given >= 2:
        return 0.95, given
    return 0.80, given

def reorder_name_if_confident(name: str) -> str:
    if not REORDER_TO_NOMBRE_APELLIDOS:
        return name
    if not name or looks_like_org(name):
        return name
    toks = split_tokens(name)
    conf, g = confidence_trailing_given(toks)
    if conf < REORDER_MIN_CONF or g == 0 or len(toks) < 2:
        return name
    head = toks[:-g]
    tail = toks[-g:]
    reordered = " ".join(tail + head).strip()
    return reordered[:60]

def fetch_pdf_bytes(url: str) -> bytes:
    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            raise HTTPException(status_code=400, detail=f"No se pudo descargar el PDF (HTTP {r.status_code}).")
        ct = (r.headers.get("content-type") or "").lower()
        if "pdf" not in ct and not r.content.startswith(b"%PDF"):
            raise HTTPException(status_code=400, detail="La URL no parece entregar un PDF válido.")
        return r.content
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error de red al descargar el PDF: {e}")

def cv_flag(name: str, default: int = 0) -> int:
    return int(getattr(cv2, name, default))

THRESH_BINARY     = cv_flag("THRESH_BINARY", 0)
THRESH_BINARY_INV = cv_flag("THRESH_BINARY_INV", 0)
THRESH_OTSU       = cv_flag("THRESH_OTSU", 0)

# ──────────────────────────────────────────────────────────────────────────────
# Raster (pág. 2) con AutoDPI opcional
# ──────────────────────────────────────────────────────────────────────────────
def raster_page2(pdf_bytes: bytes) -> Tuple[np.ndarray, dict]:
    """
    Devuelve (bgr, debug_raster)
    - Usa AUTO_DPI si está activo: prueba la escalera y se queda con el primer render OK.
    - Si no: usa FAST_DPI si FAST_MODE, si no PDF_DPI.
    """
    dbg = {"dpi": None, "ladder_used": None}
    use_ladder = AUTO_DPI and DPI_LADDER
    dpis = DPI_LADDER if use_ladder else [FAST_DPI if FAST_MODE else PDF_DPI]

    last_err = None
    for dpi in dpis:
        try:
            t0 = time.time()
            pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=2, last_page=2)
            if not pages:
                last_err = "No pages"
                continue
            pil: Image.Image = pages[0].convert("RGB")
            bgr = np.array(pil)[:, :, ::-1]
            dbg["dpi"] = dpi
            dbg["ladder_used"] = dpis
            dbg["ms"] = int((time.time()-t0)*1000)
            return bgr, dbg
        except Exception as e:
            last_err = str(e)
            continue
    raise HTTPException(status_code=400, detail=f"No se pudo rasterizar la página 2. Último error: {last_err}")

# ──────────────────────────────────────────────────────────────────────────────
# Color masks y contornos
# ──────────────────────────────────────────────────────────────────────────────
def color_masks(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    g_ranges = [
        (np.array([35,  20, 50], np.uint8), np.array([85, 255, 255], np.uint8)),
        (np.array([86,  15, 50], np.uint8), np.array([100,255,255], np.uint8)),
    ]
    p_ranges = [
        (np.array([160, 20, 70], np.uint8), np.array([179,255,255], np.uint8)),
        (np.array([  0, 20, 70], np.uint8), np.array([ 10,255,255], np.uint8)),
    ]
    mg = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in g_ranges: mg = cv2.bitwise_or(mg, cv2.inRange(hsv, lo, hi))
    mp = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in p_ranges: mp = cv2.bitwise_or(mp, cv2.inRange(hsv, lo, hi))
    k3 = np.ones((3,3), np.uint8); k5 = np.ones((5,5), np.uint8)
    mg = cv2.morphologyEx(mg, cv2.MORPH_OPEN, k3); mg = cv2.morphologyEx(mg, cv2.MORPH_CLOSE, k5)
    mp = cv2.morphologyEx(mp, cv2.MORPH_OPEN, k3); mp = cv2.morphologyEx(mp, cv2.MORPH_CLOSE, k5)
    return mg, mp

def contours_centroids(mask: np.ndarray, min_area: int) -> List[Tuple[int,int,int]]:
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

# ──────────────────────────────────────────────────────────────────────────────
# Lado por ángulo (8 direcciones)
# ──────────────────────────────────────────────────────────────────────────────
def side_of_8(main_xy: Tuple[int,int], pt_xy: Tuple[int,int]) -> str:
    cx, cy = main_xy
    x, y   = pt_xy
    sx, sy = x - cx, y - cy
    ang = math.degrees(math.atan2(-(sy), sx))  # 0=Este, 90= Norte
    # sector de 45° centrado en las direcciones cardinales
    if -22.5 <= ang < 22.5: return "este"
    if 22.5 <= ang < 67.5: return "noreste"
    if 67.5 <= ang < 112.5: return "norte"
    if 112.5 <= ang < 157.5: return "noroeste"
    if ang >= 157.5 or ang < -157.5: return "oeste"
    if -157.5 <= ang < -112.5: return "suroeste"
    if -112.5 <= ang < -67.5: return "sur"
    return "sureste"

SIDE2LBL = {
    "norte":"N", "noreste":"NE", "este":"E", "sureste":"SE",
    "sur":"S", "suroeste":"SO", "oeste":"O", "noroeste":"NO"
}

# ──────────────────────────────────────────────────────────────────────────────
# OCR utils
# ──────────────────────────────────────────────────────────────────────────────
def enhance_gray(g: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(g)

def binarize(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flags_bin = THRESH_BINARY | (THRESH_OTSU if THRESH_OTSU else 0)
    flags_inv = THRESH_BINARY_INV | (THRESH_OTSU if THRESH_OTSU else 0)
    _, bw  = cv2.threshold(gray, 0 if THRESH_OTSU else 127, 255, flags_bin)
    _, bwi = cv2.threshold(gray, 0 if THRESH_OTSU else 127, 255, flags_inv)
    return bw, bwi

def ocr_text(img: np.ndarray, psm: int, whitelist: Optional[str] = None) -> str:
    cfg = f"--psm {psm} --oem 3"
    if whitelist is not None:
        safe = (whitelist or "").replace('"','')
        cfg += f' -c tessedit_char_whitelist="{safe}"'
    txt = pytesseract.image_to_string(img, config=cfg) or ""
    txt = txt.replace("\r", "\n")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{2,}", "\n", txt)
    return txt.strip()

def strip_after_delims(s: str) -> str:
    m = re.split(r"[\[\]:0-9]", s, maxsplit=1)
    return (m[0] if m else s).strip()

def clean_owner_line(line: str) -> str:
    if not line: return ""
    toks = [t for t in re.split(r"[^\wÁÉÍÓÚÜÑ'-]+", line.upper()) if t]
    out = []
    for t in toks:
        if any(ch.isdigit() for ch in t): break
        if t in GEO_TOKENS or "[" in t or "]" in t: break
        if t in BAD_TOKENS: continue
        out.append(t)
        if len([x for x in out if x not in NAME_CONNECTORS]) >= 5:
            break
    compact = []
    for t in out:
        if (not compact) and t in NAME_CONNECTORS:
            continue
        compact.append(t)
    name = " ".join(compact).strip()
    return name[:48]

def pick_owner_from_text(txt: str) -> str:
    if not txt: return ""
    lines = [l.strip() for l in txt.split("\n") if l.strip()]
    for l in lines:
        U = l.upper()
        if any(tok in U for tok in BAD_TOKENS):
            continue
        if sum(ch.isdigit() for ch in U) > 1:
            continue
        if not UPPER_NAME_RE.match(U):
            continue
        name = clean_owner_line(U)
        if len(name) >= 6:
            return name
    return ""

# ──────────────────────────────────────────────────────────────────────────────
# Localizar columna “Apellidos…” y extraer 1ª + posible 2ª línea
# (metodología "anclada a cabecera" + fallback)
# ──────────────────────────────────────────────────────────────────────────────
def find_header_and_cols(bgr: np.ndarray, row_y: int) -> Tuple[int,int,int,int,dict]:
    """
    Busca 'APELLIDOS' y 'NIF' en una ventana vertical alrededor de row_y.
    Devuelve (x0, x1, y0_l1, y1_l1, dbg)
      - (x0,x1): rango de columna del nombre (hasta x de 'NIF' si existe)
      - (y0_l1,y1_l1): primera línea de nombre
    Si no encuentra cabecera: fallback proporcional.
    """
    h, w = bgr.shape[:2]
    win = int(h * 0.14)
    y0s, y1s = max(0, row_y - win), min(h, row_y + win)
    band = bgr[y0s:y1s, int(w*0.28):int(w*0.80)]
    dbg = {"header_found": False, "x_nif_abs": None, "header_left_abs": None}

    if band.size > 0:
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        bw, bwi = binarize(gray)
        for im in (bw, bwi):
            data = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT, config="--psm 6 --oem 3")
            words = data.get("text", [])
            xs = data.get("left", [])
            ys = data.get("top", [])
            ws = data.get("width", [])
            hs = data.get("height", [])
            header_bottom = None
            x_header_left = None
            x_nif = None
            for t,lx,ty,ww,hh in zip(words, xs, ys, ws, hs):
                if not t: continue
                T = t.upper()
                if "APELLIDOS" in T or "APELLIDOS/NOMBRE" in T:
                    header_bottom = max(header_bottom or 0, ty + hh)
                    x_header_left = min(x_header_left or (1<<30), lx)
                if T == "NIF":
                    x_nif = lx
                    header_bottom = max(header_bottom or 0, ty + hh)
            if header_bottom is not None:
                dbg["header_found"] = True
                dbg["x_nif_abs"] = int(w*0.28) + (x_nif if x_nif is not None else 0)
                dbg["header_left_abs"] = int(w*0.28) + (x_header_left if x_header_left is not None else 0)
                y0_l1 = y0s + header_bottom + 6
                y1_l1 = min(h, y0_l1 + int(h*0.035))
                x0 = int(w*0.28)
                x1 = int(w*0.28) + (x_nif - 10) if x_nif is not None else int(w*0.66)
                return max(0,x0), min(w,x1), y0_l1, y1_l1, dbg

    # fallback (sin cabecera)
    y0_l1 = max(0, row_y - int(h*0.01))
    y1_l1 = min(h, y0_l1 + int(h*0.035))
    x0 = int(w * 0.28)
    x1 = int(w * 0.66)
    return x0, x1, y0_l1, y1_l1, dbg

def read_two_lines(bgr: np.ndarray, x0:int, x1:int, y0_l1:int, y1_l1:int) -> Tuple[str,str,dict]:
    """
    Lee L1 y L2 (L2 justo debajo de L1). Si Tesseract mete \n dentro de L1,
    separa t1_raw y t1_extra_raw.
    """
    h, w = bgr.shape[:2]
    # ROI L1
    roi1 = bgr[y0_l1:y1_l1, x0:x1]
    # ROI L2: ventana del mismo alto inmediatamente debajo
    y0_l2 = y1_l1 + 2
    y1_l2 = min(h, y0_l2 + (y1_l1 - y0_l1))
    roi2 = bgr[y0_l2:y1_l2, x0:x1]

    dbg = {"x0":x0,"x1":x1,"y_line1":[y0_l1,y1_l1],"y_line2_hint":[y0_l2,y1_l2]}
    WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '"
    def ocr_roi(img):
        if img.size == 0: return ""
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)
        g = enhance_gray(g)
        bw, bwi = binarize(g)
        for im in (bw, bwi):
            txt = ocr_text(im, psm=6, whitelist=WL)
            if txt:
                return txt
        return ""

    t1_raw = ocr_roi(roi1)
    t2_raw = ocr_roi(roi2)
    # Caso: L1 con salto de línea dentro
    t1_extra_raw = ""
    if "\n" in t1_raw:
        parts = [p.strip() for p in t1_raw.split("\n") if p.strip()]
        if len(parts) >= 2:
            t1_raw = parts[0]
            t1_extra_raw = parts[1]

    dbg["t1_raw"] = t1_raw
    dbg["t2_raw"] = t2_raw
    dbg["t1_extra_raw"] = t1_extra_raw
    return t1_raw, t2_raw, dbg

def choose_owner_from_lines(t1_raw: str, t2_raw: str, t1_extra_raw: str) -> Tuple[str,str,bool]:
    """
    Devuelve (owner, picked_from, second_line_used)
      picked_from in {"strict","from_l1_break","l2_clean","l1_plus_extra"}
    Reglas:
      1) si t1_raw contiene un nombre válido → strict
      2) si t1_extra_raw parece nombre (2ª línea incrustada en L1) → from_l1_break
      3) si L2 útil (no ruido, no JUNK_2NDLINE, no geo) → l2_clean
      4) si L1 es apellido y t1_extra_raw es nombre → l1_plus_extra (para compuestos tipo JOSE LUIS)
    """
    owner1 = clean_owner_line(t1_raw.upper())
    if len(owner1) >= 6:
        # Si owner1 termina con nombre conocido (caso RODRIGUEZ JOSE) y t1_extra_raw amplía:
        if t1_extra_raw:
            t1e = clean_owner_line(t1_extra_raw.upper())
            if t1e and (t1e in NAME_HINTS or len(t1e) <= 12):
                cand = f"{owner1} {t1e}".strip()
                return cand[:48], "l1_plus_extra", True
        return owner1, "strict", False

    if t1_extra_raw:
        t1e = clean_owner_line(t1_extra_raw.upper())
        if len(t1e) >= 2 and t1e not in JUNK_2NDLINE:
            cand = f"{owner1} {t1e}".strip() if owner1 and owner1 not in JUNK_2NDLINE else t1e
            cand = strip_after_delims(cand)[:48]
            if cand:
                return cand, "from_l1_break", True

    if t2_raw:
        t2 = strip_after_delims(t2_raw.upper())
        t2 = re.sub(r"\s+", " ", t2).strip()
        if t2 and t2 not in JUNK_2NDLINE and t2 not in GEO_TOKENS and len(t2) <= 26:
            cand = f"{owner1} {t2}".strip() if owner1 and owner1 not in JUNK_2NDLINE else t2
            cand = strip_after_delims(cand)[:48]
            if cand:
                return cand, "l2_clean", True

    return owner1, "strict", False

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline por filas (detección de puntos + OCR columna)
# ──────────────────────────────────────────────────────────────────────────────
def detect_rows_and_extract(bgr: np.ndarray,
                            annotate: bool = False,
                            annotate_names: bool = False) -> Tuple[Dict[str,str], dict, np.ndarray]:
    vis = bgr.copy()
    h, w = bgr.shape[:2]

    # Área aproximada donde viven los croquis a la izquierda
    top = int(h * 0.10); bottom = int(h * 0.92)
    left = int(w * 0.04); right = int(w * 0.42)
    crop = bgr[top:bottom, left:right]

    mg, mp = color_masks(crop)
    # umbrales sensibles: colindantes finos
    mains  = contours_centroids(mg, min_area=(300 if FAST_MODE else 220))
    neighs = contours_centroids(mp, min_area=(140 if FAST_MODE else 120))

    linderos8 = {"norte":"","noreste":"","este":"","sureste":"","sur":"","suroeste":"","oeste":"","noroeste":""}
    rows_dbg = []

    if not mains:
        return linderos8, {"rows": [], "note":"no mains"}, vis

    mains_abs  = [(cx+left, cy+top, a) for (cx,cy,a) in mains]
    mains_abs.sort(key=lambda t: t[1])
    neighs_abs = [(cx+left, cy+top, a) for (cx,cy,a) in neighs]

    used_sides = set()

    # función auxiliar para refinar el punto vecino cuando es muy fino/lineal
    def refine_point(main_xy, neigh_xy):
        mx, my = main_xy
        nx, ny = neigh_xy
        dx, dy = nx - mx, ny - my
        # desplaza un poco hacia el exterior y estabiliza con histograma de vecindad
        rx, ry = int(nx + 0.06*dx), int(ny + 0.06*dy)
        rx = max(0, min(w-1, rx)); ry = max(0, min(h-1, ry))
        # histograma angular simple
        hist = {"este":0,"noreste":0,"norte":0,"noroeste":0,"oeste":0,"suroeste":0,"sur":0,"sureste":0}
        for k in range(-6,7,2):
            px, py = rx + k, ry + k
            px = max(0, min(w-1, px)); py = max(0, min(h-1, py))
            s = side_of_8((mx,my), (px,py))
            hist[s] += abs(k)+1
        best = max(hist.items(), key=lambda kv: kv[1])[0]
        return (rx,ry), best, hist

    for (mcx, mcy, _a) in mains_abs[:8]:
        # vecino más cercano
        best = None; best_d = 1e9
        for (nx, ny, _na) in neighs_abs:
            d = (nx-mcx)**2 + (ny-mcy)**2
            if d < best_d:
                best_d = d; best = (nx, ny)
        side = ""
        refined_info = {}
        if best is not None and best_d < (w*0.28)**2:
            # refina dirección si es fino/diagonal
            rp, side_ref, hist = refine_point((mcx,mcy), best)
            side = side_ref
            refined_info = {"refined_point": list(rp), "refined_mode":"histo", "refined_hist": hist}

        # OCR anclado a cabecera + fallback
        x0, x1, y0_l1, y1_l1, header_dbg = find_header_and_cols(bgr, mcy)
        t1_raw, t2_raw, ocr_dbg = read_two_lines(bgr, x0, x1, y0_l1, y1_l1)
        owner, picked_from, used_second = choose_owner_from_lines(t1_raw, t2_raw, ocr_dbg.get("t1_extra_raw",""))

        # Reordenar si aplica
        if owner:
            owner = reorder_name_if_confident(owner)

        # Evitar sobrescribir el mismo lado si ya está relleno con un nombre más largo
        if side and owner:
            if side not in used_sides or len(owner) > len(linderos8.get(side,"")):
                linderos8[side] = owner
                used_sides.add(side)

        if annotate:
            cv2.circle(vis, (mcx, mcy), 10, (0,255,0), -1)
            if best is not None:
                cv2.circle(vis, best, 8, (0,0,255), -1)
                lbl = SIDE2LBL.get(side,"")
                if lbl:
                    cv2.putText(vis, lbl, (best[0]-10, best[1]-12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        if annotate_names and owner:
            cv2.putText(vis, owner[:28], (int(w*0.44), mcy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(vis, owner[:28], (int(w*0.44), mcy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

        row_dbg_one = {
            "row_y": mcy,
            "main_center": [mcx, mcy],
            "neigh_center": list(best) if best is not None else None,
            "side": side,
            "owner": owner,
            "picked_from": picked_from,
            "used_second": used_second,
            "ocr": ocr_dbg | header_dbg
        }
        row_dbg_one.update(refined_info)
        rows_dbg.append(row_dbg_one)

    dbg = {"rows": rows_dbg}
    return linderos8, dbg, vis

# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "ok": True,
        "version": app.version,
        "FAST_MODE": FAST_MODE,
        "TEXT_ONLY": TEXT_ONLY,
        "FAST_DPI": FAST_DPI,
        "PDF_DPI": PDF_DPI,
        "AUTO_DPI": AUTO_DPI,
        "DPI_LADDER": DPI_LADDER,
        "REORDER_TO_NOMBRE_APELLIDOS": REORDER_TO_NOMBRE_APELLIDOS,
        "REORDER_MIN_CONF": REORDER_MIN_CONF,
        "name_hints_loaded": len(NAME_HINTS),
        "cv2_flags": {"OTSU": bool(THRESH_OTSU)}
    }

@app.get("/preview", dependencies=[Depends(check_token)])
def preview_get(
    pdf_url: AnyHttpUrl = Query(...),
    labels: int = Query(0, description="1=mostrar N/NE/E/SE/S/SO/O/NO"),
    names: int = Query(0, description="1=mostrar nombre estimado")
):
    pdf_bytes = fetch_pdf_bytes(str(pdf_url))
    try:
        bgr, raster_dbg = raster_page2(pdf_bytes)
        _linderos, _dbg, vis = detect_rows_and_extract(
            bgr, annotate=bool(labels), annotate_names=bool(names)
        )
        # Pintar pequeña leyenda DPI
        cv2.putText(vis, f"DPI:{raster_dbg.get('dpi')}", (12,28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(vis, f"DPI:{raster_dbg.get('dpi')}", (12,28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
    except Exception as e:
        err = str(e)
        blank = np.zeros((240, 640, 3), np.uint8)
        cv2.putText(blank, f"ERR: {err[:60]}", (10,120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        ok, png = cv2.imencode(".png", blank)
        return StreamingResponse(io.BytesIO(png.tobytes()), media_type="image/png")

    ok, png = cv2.imencode(".png", vis)
    if not ok:
        raise HTTPException(status_code=500, detail="No se pudo codificar la vista previa.")
    return StreamingResponse(io.BytesIO(png.tobytes()), media_type="image/png")

@app.post("/preview", dependencies=[Depends(check_token)])
def preview_post(data: ExtractIn = Body(...),
                 labels: int = Query(0),
                 names: int = Query(0)):
    return preview_get(pdf_url=data.pdf_url, labels=labels, names=names)

@app.post("/extract", response_model=ExtractOut, dependencies=[Depends(check_token)])
def extract(data: ExtractIn = Body(...), debug: bool = Query(False)) -> ExtractOut:
    pdf_bytes = fetch_pdf_bytes(str(data.pdf_url))

    if TEXT_ONLY:
        return ExtractOut(
            linderos={"norte":"","noreste":"","este":"","sureste":"","sur":"","suroeste":"","oeste":"","noroeste":""},
            owners_detected=[],
            note="Modo TEXT_ONLY activo: mapa/OCR desactivados.",
            debug={"TEXT_ONLY": True} if debug else None
        )

    try:
        bgr, raster_dbg = raster_page2(pdf_bytes)
        linderos8, vdbg, _vis = detect_rows_and_extract(bgr, annotate=False)
        owners_detected = [o.get("owner","") for o in vdbg["rows"] if o.get("owner")]
        owners_detected = list(dict.fromkeys(owners_detected))[:12]

        note = None
        if not any(linderos8.values()):
            note = "No se pudo determinar lado/vecino con suficiente confianza."

        dbg = None
        if debug:
            dbg = vdbg
            dbg["raster"] = raster_dbg

        return ExtractOut(linderos=linderos8, owners_detected=owners_detected, note=note, debug=dbg)

    except Exception as e:
        return ExtractOut(
            linderos={"norte":"","noreste":"","este":"","sureste":"","sur":"","suroeste":"","oeste":"","noroeste":""},
            owners_detected=[],
            note=f"Excepción visión/OCR: {e}",
            debug={"exception": str(e)} if debug else None
        )

# ──────────────────────────────────────────────────────────────────────────────
# Batch: hasta 10 PDFs en paralelo controlado
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/extract_many", response_model=List[ExtractManyItem], dependencies=[Depends(check_token)])
async def extract_many(
    data: ExtractManyIn = Body(...),
    debug: bool = Query(False),
    max_concurrency: int = Query(4, ge=1, le=8, description="Límite de tareas simultáneas"),
    timeout_s: int = Query(25, ge=5, le=120, description="Timeout por PDF")
) -> List[ExtractManyItem]:
    # normaliza lista, de-dup y limita a 10
    urls = list(dict.fromkeys([str(u) for u in data.pdf_urls]))[:10]
    sem = asyncio.Semaphore(max_concurrency)

    async def run_one(url: str) -> dict:
        async with sem:
            async def job():
                pdf_bytes = await asyncio.to_thread(fetch_pdf_bytes, url)
                bgr, raster_dbg = await asyncio.to_thread(raster_page2, pdf_bytes)
                linderos8, vdbg, _ = await asyncio.to_thread(detect_rows_and_extract, bgr, False)
                owners_detected = [o.get("owner","") for o in vdbg["rows"] if o.get("owner")]
                owners_detected = list(dict.fromkeys(owners_detected))[:12]
                note = None
                if not any(linderos8.values()):
                    note = "No se pudo determinar lado/vecino con suficiente confianza."
                dbg = None
                if debug:
                    dbg = vdbg
                    dbg["raster"] = raster_dbg
                return {
                    "pdf_url": url,
                    "ok": True,
                    "linderos": linderos8,
                    "owners_detected": owners_detected,
                    "note": note,
                    "debug": dbg
                }

            try:
                return await asyncio.wait_for(job(), timeout=timeout_s)
            except asyncio.TimeoutError:
                return {
                    "pdf_url": url,
                    "ok": False,
                    "linderos": {"norte":"","noreste":"","este":"","sureste":"","sur":"","suroeste":"","oeste":"","noroeste":""},
                    "owners_detected": [],
                    "note": f"Timeout tras {timeout_s}s",
                    "debug": {"timeout": True} if debug else None
                }
            except Exception as e:
                return {
                    "pdf_url": url,
                    "ok": False,
                    "linderos": {"norte":"","noreste":"","este":"","sureste":"","sur":"","suroeste":"","oeste":"","noroeste":""},
                    "owners_detected": [],
                    "note": f"Excepción: {e}",
                    "debug": {"exception": str(e)} if debug else None
                }

    results = await asyncio.gather(*(run_one(u) for u in urls))
    # FastAPI validará contra ExtractManyItem
    return results  # type: ignore[return-value]

# ──────────────────────────────────────────────────────────────────────────────
# (Opcional) Carga directa de PDF como archivo (útil para test locales)
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/extract_upload", response_model=ExtractOut, dependencies=[Depends(check_token)])
async def extract_upload(file: UploadFile = File(...), debug: bool = Query(False)) -> ExtractOut:
    pdf_bytes = await file.read()
    try:
        bgr, raster_dbg = raster_page2(pdf_bytes)
        linderos8, vdbg, _vis = detect_rows_and_extract(bgr, annotate=False)
        owners_detected = [o.get("owner","") for o in vdbg["rows"] if o.get("owner")]
        owners_detected = list(dict.fromkeys(owners_detected))[:12]
        note = None
        if not any(linderos8.values()):
            note = "No se pudo determinar lado/vecino con suficiente confianza."
        dbg = vdbg if debug else None
        if dbg is not None:
            dbg["raster"] = raster_dbg
        return ExtractOut(linderos=linderos8, owners_detected=owners_detected, note=note, debug=dbg)
    except Exception as e:
        return ExtractOut(
            linderos={"norte":"","noreste":"","este":"","sureste":"","sur":"","suroeste":"","oeste":"","noroeste":""},
            owners_detected=[],
            note=f"Excepción visión/OCR: {e}",
            debug={"exception": str(e)} if debug else None
        )


