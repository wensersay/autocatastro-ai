from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional
import fitz  # PyMuPDF
import io
import os
import uuid
import openai

app = FastAPI()

# Simulación de lógica de extracción de linderos (placeholder)
def extraer_linderos(pdf_bytes: bytes) -> dict:
    # Aquí se usaría tu lógica de la versión 0.6.6 real
    return {
        "norte": ["Dosinda Vázquez Pombo"],
        "sur": ["Rogelio Mosquera López"],
        "este": ["Rogelio Mosquera López", "José Luis Rodríguez Álvarez"],
        "oeste": ["José Varela Fernández"],
        "noroeste": [],
        "noreste": [],
        "suroeste": [],
        "sureste": []
    }

# Generar redacción notarial con IA
def redactar_texto_notarial(linderos: dict) -> str:
    texto = f"""Linda al norte con {', '.join(linderos['norte']) or '---'},
al sur con {', '.join(linderos['sur']) or '---'},
al este con {', '.join(linderos['este']) or '---'},
y al oeste con {', '.join(linderos['oeste']) or '---'}."""
    prompt = f"Redacta un párrafo notarial formal describiendo una finca rústica con estos linderos: {texto}"

    openai.api_key = os.getenv("OPENAI_API_KEY")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR al generar redacción automática: {e}]"

@app.post("/extract")
async def extract(
    files: List[UploadFile] = File(...),
    authorization: Optional[str] = Header(None)
):
    auth_token = os.getenv("AUTH_TOKEN")
    if authorization != f"Bearer {auth_token}":
        raise HTTPException(status_code=401, detail="Token de autorización inválido")

    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Solo se permiten hasta 5 archivos PDF por petición")

    resultados = []

    for file in files:
        content = await file.read()

        try:
            pdf = fitz.open(stream=content, filetype="pdf")
        except:
            raise HTTPException(status_code=400, detail="Documento PDF no válido")

        if not any(page.get_text().strip() for page in pdf):
            raise HTTPException(status_code=400, detail="Documento no contiene texto OCR válido")

        linderos = extraer_linderos(content)
        redaccion = redactar_texto_notarial(linderos)

        # Guardar redacción como .txt descargable
        filename = f"{uuid.uuid4().hex}.txt"
        filepath = f"/tmp/{filename}"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(redaccion)

        resultados.append({
            "filename": file.filename,
            "linderos": linderos,
            "notarial_text": redaccion,
            "txt_download_link": f"/download/{filename}"
        })

    return JSONResponse(content={"resultados": resultados})

@app.get("/download/{filename}")
async def download_txt(filename: str):
    filepath = f"/tmp/{filename}"
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    return FileResponse(filepath, media_type="text/plain", filename=filename)
