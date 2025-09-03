
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pytesseract, cv2, numpy as np, fitz, os
from PIL import Image
import io

app = FastAPI(
    title="AutoCatastro AI",
    version="0.6.7",
)

# Seguridad Swagger (Bearer token)
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description="Extracción automática de datos catastrales y redacción notarial",
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    openapi_schema["security"] = [{"BearerAuth": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Middleware CORS (por si se accede desde frontend externo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependencia para verificar AUTH_TOKEN
def verify_token(authorization: str = Header(...)):
    expected = os.getenv("AUTH_TOKEN")
    if not expected or authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Unauthorized")

class ExtractOut(BaseModel):
    linderos: Dict[str, str]
    owners_detected: List[str]
    note: Optional[str] = None
    debug: Optional[Dict[str, Any]] = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/extract", response_model=ExtractOut, dependencies=[Depends(verify_token)])
async def extract(file: UploadFile = File(...)):
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Archivo vacío o corrupto.")
    # Simulación lógica de extracción
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
