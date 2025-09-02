from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List
import os, io
import fitz  # PyMuPDF
import openai

app = FastAPI()

MAX_FILES = 5
USE_AI = True  # asumimos activado por defecto

@app.post("/extract")
async def extract(files: List[UploadFile] = File(...)):
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail="Máximo 5 archivos permitidos por petición.")

    resultados = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF.")

        pdf_bytes = await file.read()

        # Validación básica de OCR y metadatos
        if not es_pdf_valido(pdf_bytes):
            raise HTTPException(
                status_code=400,
                detail=f"Documento '{file.filename}' no válido: asegúrese de que es el PDF original del Catastro."
            )

        # 1. Extraer datos estructurados (mock temporal)
        datos = extraer_datos_catastrales(pdf_bytes)

        # 2. Llamar a GPT-4o para redactar
        redaccion = generar_redaccion_notarial(datos)

        # 3. Guardar .txt temporal
        txt_filename = f"redaccion_{file.filename.replace('.pdf', '')}.txt"
        txt_bytes = redaccion.encode("utf-8")

        resultados.append({
            "filename": file.filename,
            "linderos": datos["linderos"],
            "owners_detected": datos["owners_detected"],
            "notarial_text": redaccion,
            "txt_download_url": f"/download/{txt_filename}"
        })

        # Guardar temporalmente para descarga
        with open(f"/tmp/{txt_filename}", "wb") as f:
            f.write(txt_bytes)

    return JSONResponse(content={"resultados": resultados})

@app.get("/download/{filename}")
async def descargar_txt(filename: str):
    filepath = f"/tmp/{filename}"
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    return StreamingResponse(open(filepath, "rb"), media_type="text/plain")

def es_pdf_valido(pdf_bytes: bytes) -> bool:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if not doc.metadata or not any(page.get_text() for page in doc):
            return False
        return True
    except Exception:
        return False


# --- Lógica real de extracción de colindantes a integrar aquí ---
def extraer_datos_catastrales(pdf_bytes: bytes) -> dict:
    from autocata_core.extractor import extract_main  # Supuesto nombre
    from autocata_core.utils import analyze_neighbors

    # Guardar PDF temporal
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        pdf_path = tmp.name

    try:
        # Ejecutar lógica real con funciones de app 0.6.6
        resultado = extract_main(pdf_path)
        linderos = analyze_neighbors(resultado)

        return {
            "linderos": linderos["orientaciones"],  # Debe incluir los 8 puntos cardinales
            "owners_detected": linderos["owners"]
        }
    except Exception as e:
        raise RuntimeError(f"Error extrayendo datos: {e}")

    # MOCK TEMPORAL: debes sustituir por tu lógica real de linderos
    return {
        "linderos": {
            "norte": ["Ejemplo Norte"],
            "noreste": [],
            "este": ["Ejemplo Este"],
            "sureste": [],
            "sur": [],
            "suroeste": [],
            "oeste": ["Ejemplo Oeste"],
            "noroeste": []
        },
        "owners_detected": ["Ejemplo Norte", "Ejemplo Este", "Ejemplo Oeste"]
    }

def generar_redaccion_notarial(datos: dict) -> str:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = f"Redacta un párrafo notarial describiendo una finca rústica con los siguientes linderos:
{datos['linderos']}"
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message["content"].strip()
