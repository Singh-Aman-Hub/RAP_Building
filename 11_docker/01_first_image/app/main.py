"""
GOAL: A tiny FastAPI app (health check + toy sentiment model) that we will put inside a Docker image.

RUN (from 11_docker/01_first_image):   ../../.venv/bin/python -m uvicorn app.main:app --port 8000
(Use the cd form: folder names starting with digits cannot be imported as Python packages from the repo root.)
EXPECTED OUTPUT: "Uvicorn running on http://127.0.0.1:8000"; then in another terminal
    curl localhost:8000/health                      -> {"status":"ok"}
    curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"text":"I love it"}'
                                                    -> {"label":"positive","score":1}
NEEDS: fastapi, uvicorn (see requirements.txt next to this folder). No internet needed.

Why a "toy" model? Docker does not care what the app does. A keyword counter keeps the image tiny and
the build fast, so you can focus on Docker. To containerise the REAL API from module 09, see ../README.md
("Swapping in the module-09 app").
"""
from fastapi import FastAPI            # FastAPI = the web framework (3rd-party). Builds the routes.
from pydantic import BaseModel         # BaseModel = declares + validates the shape of the request body.

app = FastAPI(title="Toy sentiment API")   # `app` is what uvicorn is told to serve ("app.main:app")

POSITIVE = {"good", "great", "love", "excellent", "happy"}
NEGATIVE = {"bad", "awful", "hate", "terrible", "sad"}


class PredictRequest(BaseModel):
    text: str                          # type hint = validation rule: body must be {"text": "<string>"}


def predict_label(text: str) -> tuple[str, int]:
    """Count positive minus negative words. Returns (label, score)."""
    words = text.lower().split()                                  # "I LOVE it" -> ["i", "love", "it"]
    score = sum(w in POSITIVE for w in words) - sum(w in NEGATIVE for w in words)  # True counts as 1
    label = "positive" if score > 0 else "negative" if score < 0 else "neutral"
    return label, score


@app.get("/health")                    # decorator: "GET /health runs the function below"
def health():
    return {"status": "ok"}            # dict -> JSON automatically


@app.post("/predict")
def predict(req: PredictRequest):      # FastAPI parses the JSON body into `req` or answers 422
    label, score = predict_label(req.text)
    return {"label": label, "score": score}


# TRY IT: add a word to POSITIVE and re-run the curl.
# BREAK IT: send {"txt": "hi"} -> 422 Unprocessable Entity. Read the JSON error: it names the missing field.
