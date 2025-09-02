# autocatastro-ai

Microservicio FastAPI para la **extracción automática de datos de certificaciones catastrales** y **redacción notarial** utilizando GPT‑4o de OpenAI.

---

## 🚀 Endpoints disponibles

### 📥 POST `/extract`
Permite subir hasta **5 PDFs catastrales originales** por petición. Devuelve un JSON con:

- ✅ Lista de titulares colindantes por punto cardinal (norte, sur, este, etc.)
- 📄 Párrafo notarial redactado con inteligencia artificial
- 🔗 Enlace para descargar la redacción en `.txt`

### 📤 GET `/download/{filename}`
Permite descargar el archivo `.txt` generado para copiar y pegar fácilmente en la escritura.

---

## ⚙️ Requisitos para ejecutar

- Python 3.8 o superior
- Clave válida de OpenAI en la variable `OPENAI_API_KEY`
- Entorno como Railway, Render o servidor propio

---

## 🧪 Variables de entorno necesarias

Configura estas variables en Railway o tu `.env`:

```env
PORT=8080
OPENAI_API_KEY=sk-...             # Tu clave de OpenAI
AUTH_TOKEN=c05f098127...          # Token de autenticación opcional

# Variables opcionales de comportamiento:
AUTO_DPI=1
DIAG_MODE=8
DIAG_SNAP_DEG=12
FAST_DPI=300
FAST_MODE=1
NAME_HINTS=1
SECOND_LINE_FORCE=1
PDF_DPI=300
TEXT_ONLY=0
```

---

## 📦 Ejemplo de uso (con curl)

```bash
curl -X POST https://TU-APP.railway.app/extract \
  -F "files=@cert1.pdf" \
  -F "files=@cert2.pdf"
```

La respuesta incluirá:
- `linderos` (con los 8 puntos cardinales)
- `owners_detected` (titulares detectados)
- `notarial_text` (párrafo legal listo)
- `txt_download_url` (enlace de descarga)

---

## 🛑 Validaciones de calidad

- ❌ Se rechazan PDFs sin OCR o sin metadatos válidos (escaneos o fotocopias).
- ✅ Solo se aceptan PDFs **originales descargados del Catastro**.
- 📤 Procesamiento uno a uno para evitar errores y timeouts.

---

## 🧾 Redacción notarial generada

El texto generado está pensado para ser copiado y pegado directamente en la escritura, o descargado en formato `.txt`.

---

## 📌 Nota

Este microservicio está diseñado para ser usado dentro de un **plugin WordPress** conectado por suscripción, pensado para notarios en España.
