# SentiMX API

REST API para análisis de sentimientos en español, construida con FastAPI y pysentimiento (BETO).

## Demo

- **API:** https://chesebread-sentimx-api.hf.space
- **Docs (Swagger):** https://chesebread-sentimx-api.hf.space/docs
- **Frontend:** https://sentimx-frontend-ivory.vercel.app

## Endpoints

| Método | Endpoint | Descripción |
|---|---|---|
| GET | `/` | Status de la API |
| POST | `/analizar` | Analiza un texto individual |
| POST | `/analizar-csv` | Analiza múltiples reviews desde un CSV |

## Stack

- FastAPI
- pysentimiento (BETO fine-tuned)
- Pandas
- Docker
- Hugging Face Spaces

## Instalación local

```bash
git clone https://github.com/SaulVaz/sentimx-api.git
cd sentimx-api
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Abre http://localhost:8000/docs

## Modelo

Se compararon dos enfoques en la fase de investigación:

| Modelo | Accuracy | Macro F1 |
|---|---|---|
| TF-IDF + SVM (Scikit-Learn) | 66.7% | 0.63 |
| pysentimiento (BETO) | 90.5% | 0.88 |

pysentimiento fue seleccionado para producción por su rendimiento superior en todas las clases, especialmente en la detección de sentimiento neutro.
