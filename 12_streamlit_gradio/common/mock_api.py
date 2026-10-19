"""
GOAL: A stand-in for the module-09 API so the UIs work standalone. Same contract: GET /health, POST /predict.

RUN: cd 12_streamlit_gradio/common && ../../.venv/bin/python -m uvicorn mock_api:app --port 8000
EXPECTED OUTPUT: curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"text":"I love it"}'
                 -> {"label":"positive","score":0.548}
NEEDS: fastapi, uvicorn, scikit-learn.

If your module-09 API is running on port 8000 instead, stop this one: the UIs do not care which one answers.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model import predict as model_predict   # works because we run from inside common/

app = FastAPI(title="Mock prediction API")


class PredictRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        label, score = model_predict(req.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))   # clean 4xx instead of a 500 crash
    return {"label": label, "score": score}
