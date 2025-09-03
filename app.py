
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import os
import pytesseract
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import tempfile
import shutil
import json
import openai

# Seguridad
from fastapi.security import HTTPBearer
from fastapi import Depends
import base64
import hashlib
import time

app = FastAPI(
    title="AutoCatastro AI",
    version="0.6.7",
    description="Extracción automática de datos catastrales y redacción notarial"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# === AUTH ===
AUTH_SECRET = os.getenv("AUTH_TOKEN", "AUTH_TOKEN")

def verify_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")

    try:
        token = authorization.split("Bearer ")[1]
        encoded_payload, signature = token.split(".")
        payload_json = base64.b64decode(encoded_payload).decode("utf-8")
        expected_signature = hashlib.sha256((encoded_payload + AUTH_SECRET).encode()).hexdigest()

        if not hmac_compare(signature, expected_signature):
            raise HTTPException(status_code=401, detail="Invalid token signature")

        payload = json.loads(payload_json)
        if payload["exp"] < time.time():
            raise HTTPException(status_code=401, detail="Token expired")

        return payload

    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")

def hmac_compare(sig1, sig2):
    return hashlib.sha256(sig1.encode()).digest() == hashlib.sha256(sig2.encode()).digest()

# === MODELO DE SALIDA ===
class ExtractOut(BaseModel):
    linderos: Dict[str, str]
    owners_detected: list
    note: Optional[str] = None
    debug: Optional[dict] = {}

# === ENDPOINT PRINCIPAL ===
@app.post("/extract", response_model=ExtractOut)
async def extract_data(file: UploadFile = File(...), token=Depends(verify_token)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF")

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, "input.pdf")
        with open(pdf_path, "wb") as f:
            f.write(await file.read())

        images = pdf_to_images(pdf_path)
        if not images:
            raise HTTPException(status_code=400, detail="No se pudo convertir el PDF a imágenes")

        text_total = ""
        all_boxes = []

        for image in images:
            ocr_text = pytesseract.image_to_string(image, lang="spa")
            text_total += ocr_text

        linderos = detectar_linderos(text_total)
        owners = detectar_titulares(text_total)

        redaccion = generar_redaccion_notarial(linderos, owners)

        # Guardar también como archivo txt
        redaccion_path = os.path.join(tmpdir, "redaccion.txt")
        with open(redaccion_path, "w") as f:
            f.write(redaccion)

        return {
            "linderos": linderos,
            "owners_detected": owners,
            "note": redaccion,
            "debug": {}
        }

# === UTILIDADES ===

def pdf_to_images(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        images = []
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            img_data = pix.tobytes("png")
            img_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img_file.write(img_data)
            img_file.seek(0)
            images.append(Image.open(img_file.name))
        return images
    except Exception as e:
        return []

def detectar_linderos(texto):
    puntos = ["norte", "sur", "este", "oeste"]
    resultado = {}
    for punto in puntos:
        resultado[punto] = f"Ejemplo {punto.upper()}"
    return resultado

def detectar_titulares(texto):
    return ["Ejemplo Titular 1", "Ejemplo Titular 2"]

def generar_redaccion_notarial(linderos, titulares):
    return "Mock de redacción notarial"

@app.get("/health")
async def health():
    return {"status": "ok"}
