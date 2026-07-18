"""
app.py — Servidor FastAPI local para el clasificador Fonoayuda 147
Colocar en la raíz del proyecto (mismo nivel que main.py)
Ejecutar con: python -m uvicorn app:app --reload
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
import uvicorn

from src.predictor import Predictor

app = FastAPI(title="Clasificador MINNA - Fonoayuda 147")

# Cargar modelos al iniciar
print("⏳ Cargando modelos...")
predictor = Predictor()
models_loaded = predictor.load_models()

if models_loaded:
    print("✅ Modelos listos. Abrí http://localhost:8000 en tu navegador.")
else:
    print("❌ No se pudieron cargar los modelos.")


def get_html():
    """Lee el HTML desde templates/index.html"""
    html_path = Path(__file__).parent / "templates" / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=get_html())


@app.post("/clasificar")
async def clasificar(request: Request):
    body = await request.json()
    texto = body.get("texto", "").strip()

    if not texto:
        return JSONResponse({"error": "El texto no puede estar vacío."}, status_code=400)

    if not models_loaded:
        return JSONResponse({"error": "Los modelos no están cargados."}, status_code=503)

    resultado = predictor.predict_single_text(texto)

    if resultado is None:
        return JSONResponse({"error": "No se pudo realizar la predicción."}, status_code=500)

    response = {
        "texto": resultado["texto"],
        "texto_limpio": resultado["texto_limpio"],
        "labels": resultado["labels"],
        "sbm": None,
        "beto": None,
    }

    if resultado["sbm"] is not None:
        response["sbm"] = {
            "pred": [int(p) for p in resultado["sbm"]["pred"]],
            "proba": [round(float(p), 4) for p in resultado["sbm"]["proba"]],
        }

    if resultado["beto"] is not None:
        response["beto"] = {
            "pred": [int(p) for p in resultado["beto"]["pred"]],
            "proba": [round(float(p), 4) for p in resultado["beto"]["proba"]],
        }

    return JSONResponse(response)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)