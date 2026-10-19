"""
Worked solution: POST /predict_batch with ONE predict_proba call, plus timing vs single calls.
RUN: cd 09_api_development/solutions && python sol_fastapi_batch.py
"""
import sys
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "06_model_api_flask_vs_fastapi"))
from model_loader import load_model  # noqa: E402

app = FastAPI()
MODEL, _ = load_model()


class BatchRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=32)   # list length limits protect the server


@app.post("/predict_batch")
def predict_batch(req: BatchRequest):
    probs = MODEL.predict_proba(req.texts)                  # ONE call for all texts (vectorized)
    out = []
    for text, row in zip(req.texts, probs):
        row = list(row)
        best = max(range(len(row)), key=row.__getitem__)
        out.append({"text": text, "label": str(MODEL.classes_[best]), "confidence": round(float(row[best]), 4)})
    return {"results": out}


if __name__ == "__main__":
    client = TestClient(app)
    texts = ["great film", "awful food"] * 16
    t0 = time.perf_counter(); r = client.post("/predict_batch", json={"texts": texts}); one = time.perf_counter() - t0
    print(r.status_code, r.json()["results"][:2], f"| one batch call: {one*1000:.1f} ms")
    print("too many texts ->", client.post("/predict_batch", json={"texts": ["x"] * 33}).status_code)
