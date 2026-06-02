from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pysentimiento import create_analyzer
import pandas as pd
import io

app = FastAPI(
    title="SentiMX API",
    description="API de análisis de sentimientos en español",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo al iniciar
print("Cargando modelo...")
analyzer = create_analyzer(task="sentiment", lang="es")
print("Modelo listo")

MAPA = {"POS": "positivo", "NEG": "negativo", "NEU": "neutro"}

# --- Modelos Pydantic ---
class TextoRequest(BaseModel):
    texto: str

class TextoResponse(BaseModel):
    texto: str
    sentimiento: str
    confianza: float


# --- Endpoints ---
@app.get("/")
def root():
    return {"mensaje": "SentiMX API funcionando", "version": "1.0.0"}


@app.post("/analizar", response_model=TextoResponse)
def analizar_texto(request: TextoRequest):
    if not request.texto.strip():
        raise HTTPException(status_code=400, detail="El texto no puede estar vacío")
    
    resultado = analyzer.predict(request.texto[:512])
    sentimiento = MAPA.get(resultado.output, "neutro")
    confianza = round(max(resultado.probas.values()), 4)

    return {
        "texto": request.texto,
        "sentimiento": sentimiento,
        "confianza": confianza
    }


@app.post("/analizar-csv")
async def analizar_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos .csv")
    
    contenido = await file.read()
    df = pd.read_csv(io.StringIO(contenido.decode("utf-8")))

    if "texto" not in df.columns:
        raise HTTPException(status_code=400, detail="El CSV debe tener una columna llamada 'texto'")

    resultados = []
    for _, row in df.iterrows():
        texto = str(row["texto"])[:512]
        resultado = analyzer.predict(texto)
        sentimiento = MAPA.get(resultado.output, "neutro")
        resultados.append({
            "texto": texto,
            "sentimiento": sentimiento,
            "confianza": round(max(resultado.probas.values()), 4)
        })

    positivos  = sum(1 for r in resultados if r["sentimiento"] == "positivo")
    negativos  = sum(1 for r in resultados if r["sentimiento"] == "negativo")
    neutros    = sum(1 for r in resultados if r["sentimiento"] == "neutro")
    total      = len(resultados)

    return {
        "total": total,
        "resumen": {
            "positivos": positivos,
            "negativos": negativos,
            "neutros": neutros,
            "pct_positivo": round(positivos / total * 100, 1),
            "pct_negativo": round(negativos / total * 100, 1),
            "pct_neutro":   round(neutros   / total * 100, 1),
        },
        "resultados": resultados
    }