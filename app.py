
from fastapi import FastAPI, HTTPException, UploadFile, File, Header
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional
from jose import JWTError, jwt
import os
import shutil

app = FastAPI()

AUTH_TOKEN = os.environ.get("AUTH_TOKEN", "secret")
ALGORITHM = "HS256"

def verify_token(token: str):
    try:
        payload = jwt.decode(token, AUTH_TOKEN, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/extract")
async def extract_from_pdf(file: UploadFile = File(...), authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid")
    token = authorization.replace("Bearer ", "").strip()
    verify_token(token)

    filename = file.filename.replace(".pdf", ".txt")
    file_path = f"/mnt/data/{filename}"
    notarial_text = "Linda al norte con Ejemplo NORTE, al sur con Ejemplo SUR, al este con Ejemplo ESTE y al oeste con Ejemplo OESTE."
    with open(file_path, "w") as f:
        f.write(notarial_text)

    return {
        "linderos": {
            "norte": "Ejemplo NORTE",
            "sur": "Ejemplo SUR",
            "este": "Ejemplo ESTE",
            "oeste": "Ejemplo OESTE"
        },
        "owners_detected": ["Ejemplo Titular 1", "Ejemplo Titular 2"],
        "notarial_text": notarial_text,
        "download_url": f"/download/{filename}"
    }

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = f"/mnt/data/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type='text/plain', filename=filename)
