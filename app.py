from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import FileResponse
from jose import JWTError, jwt
from typing import Optional
import os
from pathlib import Path

# Configuración
AUTH_TOKEN = os.environ.get("AUTH_TOKEN")
ALGORITHM = "HS256"
TEMP_DIR = "/tmp/autocata_texts"
os.makedirs(TEMP_DIR, exist_ok=True)

app = FastAPI(title="autoCata", version="0.7.0", description="Microservicio para certificaciones catastrales")

# Función para verificar el token JWT
def verify_token(token: str):
    try:
        payload = jwt.decode(token, AUTH_TOKEN, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Endpoint principal /extract
@app.post("/extract")
async def extract_from_pdf(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None)
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid")

    token = authorization.replace("Bearer ", "").strip()
    verify_token(token)

    # Simulación de lógica real (mock por ahora)
    linderos = {
        "norte": "Ejemplo NORTE",
        "sur": "Ejemplo SUR",
        "este": "Ejemplo ESTE",
        "oeste": "Ejemplo OESTE"
    }
    owners_detected = ["Ejemplo Titular 1", "Ejemplo Titular 2"]
    notarial_text = "Linda al norte con Ejemplo NORTE, al sur con Ejemplo SUR, al este con Ejemplo ESTE y al oeste con Ejemplo OESTE."

    # Guardar redacción notarial como archivo .txt
    filename_base = Path(file.filename).stem.replace(" ", "_")
    txt_path = f"{TEMP_DIR}/{filename_base}.txt"
    with open(txt_path, "w") as f:
        f.write(notarial_text)

    return {
        "linderos": linderos,
        "owners_detected": owners_detected,
        "notarial_text": notarial_text,
        "download_url": f"/download/{filename_base}.txt"
    }

# Endpoint de descarga protegida por token
@app.get("/download/{filename}")
async def download_file(
    filename: str,
    authorization: Optional[str] = Header(None)
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid")

    token = authorization.replace("Bearer ", "").strip()
    verify_token(token)

    file_path = os.path.join(TEMP_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Archivo no encontrado")

    return FileResponse(file_path, media_type="text/plain", filename=filename)