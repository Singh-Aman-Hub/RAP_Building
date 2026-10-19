"""
fastapi_app.py -- the SAME model API written with FASTAPI (ASGI, async-capable).

RUN (dev):          uvicorn fastapi_app:app --port 8000 --reload      docs: http://127.0.0.1:8000/docs
RUN (production):   uvicorn fastapi_app:app --workers 2 --port 8000
Compare with flask_app.py: same endpoints, same JSON, different framework.
"""

import asyncio
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import get_settings
from model_loader import generate_text, load_model, predict_one, timed


@asynccontextmanager                      # turns this generator into a "startup ... shutdown" context manager
async def lifespan(app: FastAPI):
    # Startup: runs once per worker process before the first request.
    app.state.model, app.state.source = load_model()      # app.state = a place to keep shared objects
    print(f"[pid {os.getpid()}] fastapi_app loaded model from {app.state.source}", flush=True)
    yield                                                   # the app serves requests while paused here
    # Shutdown code would go here (close DB connections, flush logs ...)


app = FastAPI(title="Sentiment API", version="1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, description="Text to classify")   # description shows up in /docs


class PredictResponse(BaseModel):          # response_model: documents AND filters what we return
    label: str
    confidence: float
    model_version: str
    latency_ms: float


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=20, ge=1, le=100)


@app.get("/health")
def health():
    s = get_settings()
    return {"status": "ok", "model_version": s.model_version, "model_source": app.state.source}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Plain `def` (not `async def`): FastAPI runs it in a thread pool so CPU-heavy model code
    # does not freeze the event loop.
    s = get_settings()
    if len(req.text) > s.max_text_length:
        raise HTTPException(status_code=422, detail=f"'text' longer than {s.max_text_length} chars")
    (label, confidence), ms = timed(predict_one, app.state.model, req.text)
    return PredictResponse(label=label, confidence=confidence, model_version=s.model_version, latency_ms=ms)


@app.post("/generate")
def generate(req: GenerateRequest):
    if not get_settings().enable_generate:
        raise HTTPException(status_code=503, detail="generation disabled; start with ENABLE_GENERATE=1")
    return {"text": generate_text(req.prompt, req.max_new_tokens)}


# ----- helper endpoints used by step 07 to explore workers and concurrency -----
@app.get("/whoami")
def whoami():
    return {"pid": os.getpid()}


@app.get("/slow")                          # async def + await asyncio.sleep: the loop serves OTHER requests meanwhile
async def slow(seconds: float = 1):
    await asyncio.sleep(seconds)
    return {"pid": os.getpid(), "slept": True}


@app.get("/slow-sync")                     # plain def + time.sleep: runs in a thread, so still concurrent (up to ~40 threads)
def slow_sync(seconds: float = 1):
    time.sleep(seconds)
    return {"pid": os.getpid(), "slept": True}


@app.get("/slow-blocking")                 # ANTI-PATTERN: async def but a BLOCKING call -> freezes the whole event loop
async def slow_blocking(seconds: float = 1):
    time.sleep(seconds)
    return {"pid": os.getpid(), "slept": True}
