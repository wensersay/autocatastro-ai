
from fastapi import FastAPI, HTTPException, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from jose import jwt, JWTError
import os

app = FastAPI(title="AutoCatastro AI", version="0.6.7")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AUTH_TOKEN = os.environ.get("AUTH_TOKEN")
ALGORITHM = "HS256"

def verify_token(token: str):
    try:
        payload = jwt.decode(token, AUTH_TOKEN, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/extract")
async def extract_from_pdf(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None)
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid")
    token = authorization.replace("Bearer ", "").strip()
    verify_token(token)

    # Aquí se insertaría la lógica real de OCR, colindantes y redacción notarial
    # Temporalmente se devuelve un mock
    return {
        "linderos": {
            "norte": "Ejemplo NORTE",
            "sur": "Ejemplo SUR",
            "este": "Ejemplo ESTE",
            "oeste": "Ejemplo OESTE"
        },
        "owners_detected": ["Ejemplo Titular 1", "Ejemplo Titular 2"],
        "note": "Mock de redacción notarial",
        "debug": {}
    }
