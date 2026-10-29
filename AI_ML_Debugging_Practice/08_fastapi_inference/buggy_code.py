"""
Exercise 08: serving an embedding model with FastAPI.

A small internal service that other teams call to embed text before writing it to their vector DB.

API contract (agreed with the client teams; do not change it):
    POST /embed     request : {"texts": ["first text", "second text", ...]}      (1 or more strings)
                    response: {"model": "<name>", "dim": 384, "embeddings": [[float, ...], ...]}
                    errors  : an empty "texts" list is a client error -> HTTP 422
    GET  /health    response: {"status": "ok", "model_loaded": true}

The __main__ block below plays the role of a client team: it calls the API through FastAPI's TestClient,
so you don't need to start a server.

Run:   python buggy_code.py
       (optional, real server: uvicorn buggy_code:app --reload, then open http://127.0.0.1:8000/docs)
Needs: pip install fastapi httpx sentence-transformers numpy
       (first run downloads sentence-transformers/all-MiniLM-L6-v2, ~90 MB)
"""

import time

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

app = FastAPI(title="Embedding service")
model = None          # the loaded SentenceTransformer, shared by all requests


class EmbedRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/embed")
def embed(request: EmbedRequest):
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    vectors = model.encode(request.texts, normalize_embeddings=True)
    return {"model": MODEL_NAME, "dim": vectors.shape[1], "embeddings": vectors}


# ----------------------------------------------------------------------------------------------------------
# Client-team smoke test. You may add checks here, but the requests themselves follow the contract.
# ----------------------------------------------------------------------------------------------------------
def call(client, method, path, **kwargs):
    start = time.perf_counter()
    response = getattr(client, method)(path, **kwargs)
    ms = (time.perf_counter() - start) * 1000
    body = response.text if len(response.text) < 200 else response.text[:200] + "..."
    print(f"{method.upper():<4} {path:<8} -> {response.status_code}  {ms:7.1f} ms   {body}")
    return response


if __name__ == "__main__":
    with TestClient(app, raise_server_exceptions=False) as client:
        call(client, "get", "/health")
        payload = {"texts": ["How do I reset my password?", "Refunds take 7 working days."]}
        call(client, "post", "/embed", json=payload)
        call(client, "post", "/embed", json=payload)          # second identical call: should be fast
        call(client, "post", "/embed", json={"texts": []})    # contract: 422
        call(client, "get", "/health")
