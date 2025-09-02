from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    datos = {
        "norte": "Dosinda Vázquez Pombo",
        "sur": "Rogelio Mosquera López",
        "este": "José Luis Rodríguez Álvarez",
        "oeste": "José Varela Fernández"
    }

    prompt = f"""Redacta un párrafo notarial describiendo una finca rústica con los siguientes linderos:
Norte: {datos['norte']}
Sur: {datos['sur']}
Este: {datos['este']}
Oeste: {datos['oeste']}"""

    return {"prompt": prompt}
