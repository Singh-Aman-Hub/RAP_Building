# Capstone walkthrough: the files, one block at a time

## Goal
Read every file of the capstone (`api/`, `ui/`, `compose.yaml`, `run_local.sh`) in the order a request travels, and know what each block does, exactly, and what breaks without it. Overview and architecture: [README.md](README.md). Vocabulary: [TERMS.md](TERMS.md).

The picture to keep in your head:
```
Browser -> ui/app.py (Streamlit :8501) -- POST /predict --> api/fastapi_app.py (:8000)
                                                               |- config.py        (settings from env vars)
                                                               |- model_loader.py  (load model, predict_one)
                                                               '- artifacts/model.joblib (loaded once at startup)
compose.yaml starts both containers; run_local.sh starts both without Docker.
```

## Run it
```bash
cd 13_capstone_project
./run_local.sh                 # API on :8000 (docs at /docs), UI on :8501; Ctrl+C stops both
# or, with Docker Desktop running:
docker compose up --build
```
Real output of the API checked in-process with FastAPI `TestClient` (no server needed), model `artifacts/model.joblib`:
```
Settings(model_path='artifacts/model.joblib', model_version='0.1.0', max_text_length=500, enable_generate=False)
[pid 27638] fastapi_app loaded model from file:model.joblib
GET  /health                       -> {'status': 'ok', 'model_version': '0.1.0', 'model_source': 'file:model.joblib'}
POST /predict {"text":"I love it"} -> {'label': 'positive', 'confidence': 0.502, 'model_version': '0.1.0', 'latency_ms': 63.55}
POST /predict {"text":""}          -> 422
POST /predict {"text":"x"*600}     -> 422 {'detail': "'text' longer than 500 chars"}
POST /generate                     -> 503 (generation disabled)
```
Docker build and compose were read, not run (Docker may not be installed). The Streamlit UI was read, not launched.

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Settings vs config vs environment variables | 12-factor config, `.env`, `os.environ` | Related | Environment variables are the raw strings; `Settings` is the typed object built from them |
| `dataclass` vs Pydantic `BaseModel` | data holder | No | A dataclass only stores fields; Pydantic also validates and converts input data |
| Startup (`lifespan`) vs per-request loading | load once, warm model | No | `lifespan` runs once per worker before requests; loading in the route would reload the model on every call |
| `def` route vs `async def` route | sync vs async endpoint | No | `def` runs in a thread pool; `async def` runs on the event loop and must never block |
| `EXPOSE` vs `ports:` | document vs publish | No | `EXPOSE` documents; compose `ports: ["8000:8000"]` publishes |
| Artifact vs `model.joblib` | saved model | Related | The artifact folder holds the model file plus `metadata.json` (version, labels, validation accuracy 0.8) |

## Code, step by step

### 1. config.py docstring  (`api/config.py` lines 1–6)
```python
"""
config.py -- ALL settings come from environment variables (12-factor style).

Why: the same code/Docker image must run on your laptop, in CI and in production with
different settings, WITHOUT editing code.  See step 08 for experiments.
"""
```
- **In simple words:** A note saying all settings come from environment variables.
- **Exactly:** 12-factor style: the same code and the same Docker image run on laptop, CI and production; only the environment differs.
- **Why it is here:** Without it you edit code to change a path, and rebuild images per environment.

### 2. Imports and `.env` loading  (`api/config.py` lines 8–15)
```python
import os
from dataclasses import dataclass  # a decorator that writes __init__/__repr__ for a plain data holder

from dotenv import load_dotenv     # third-party: python-dotenv. Reads KEY=value lines from a .env file

# Loads .env (if one exists in this folder or a parent) into os.environ.
# Real environment variables WIN over .env by default (override=False).
load_dotenv()
```
- **In simple words:** Load tools, then read a `.env` file if one exists.
- **Exactly:** `os` for env access; `dataclass` writes `__init__` and `__repr__`; `load_dotenv()` (python-dotenv) copies `KEY=value` lines from a `.env` file into `os.environ`; real environment variables win (`override=False`).
- **Why it is here:** Lets a developer keep local settings in a file that is not committed.

### 3. The Settings container  (`api/config.py` lines 18–23)
```python
@dataclass(frozen=True)            # frozen=True -> attributes cannot be changed after creation
class Settings:
    model_path: str | None         # path to a joblib model file; None -> use the built-in fallback model
    model_version: str             # reported in every response so you know WHICH model answered
    max_text_length: int           # reject giant inputs (protects latency and memory)
    enable_generate: bool          # the /generate endpoint downloads ~330 MB, so it is opt-in
```
- **In simple words:** A small box that holds the four settings, and cannot be changed after it is made.
- **Exactly:** `@dataclass(frozen=True)` makes attribute assignment raise `FrozenInstanceError`. Fields: `model_path: str | None`, `model_version: str`, `max_text_length: int`, `enable_generate: bool`.
- **Why it is here:** Typed, immutable settings cannot be changed by accident in the middle of a request.

### 4. Read the environment now  (`api/config.py` lines 26–33)
```python
def get_settings() -> Settings:
    """Read the environment NOW (not at import time) so tests can change it with monkeypatch."""
    return Settings(
        model_path=os.getenv("MODEL_PATH") or None,          # "" (empty) also becomes None
        model_version=os.getenv("MODEL_VERSION", "0.1.0"),   # getenv(name, default)
        max_text_length=int(os.getenv("MAX_TEXT_LENGTH", "500")),  # env vars are ALWAYS strings -> convert
        enable_generate=os.getenv("ENABLE_GENERATE", "0") == "1",
    )
```
- **In simple words:** Build the settings from the environment, with defaults.
- **Exactly:** Called at request time, not import time, so tests can use `monkeypatch`. `os.getenv("MODEL_PATH") or None` turns an empty string into `None`. Env vars are always strings, hence `int(os.getenv("MAX_TEXT_LENGTH", "500"))` and `== "1"` for the boolean. Real result: `Settings(model_path='artifacts/model.joblib', model_version='0.1.0', max_text_length=500, enable_generate=False)`.
- **Why it is here:** Without the `int()` conversion, comparing `len(text) > "500"` raises `TypeError`.

### 5. model_loader.py docstring and the model contract  (`api/model_loader.py` lines 1–14)
```python
"""
model_loader.py -- everything about getting a model and asking it questions.

Both flask_app.py and fastapi_app.py import THIS file, so the two APIs behave identically
and the only difference between them is the web framework (that is the point of this step).

THE MODEL CONTRACT (anything with these works):
    model.predict_proba(list_of_texts) -> list of probability lists   (one row per text)
    model.classes_                     -> list of label names, same order as the probability columns
scikit-learn Pipelines (like the one from module 03) satisfy this. So does KeywordSentimentModel below.

SWAPPING IN THE REAL MODEL:
    MODEL_PATH=/path/to/model.joblib python flask_app.py
"""
```
- **In simple words:** Explains what a "model" means to this code.
- **Exactly:** Contract: `predict_proba(list_of_texts)` returns one probability list per text; `classes_` names the columns in the same order. A scikit-learn Pipeline satisfies it, and so does the fallback keyword model.
- **Why it is here:** The API does not care which model it holds, so you can swap it by changing `MODEL_PATH`.

### 6. Imports  (`api/model_loader.py` lines 16–19)
```python
import os
import time

from config import get_settings
```
- **In simple words:** Load the standard modules and our settings function.
- **Exactly:** `os`, `time`, and `get_settings` from `config.py` (imports work because uvicorn runs from inside `api/`).
- **Why it is here:** `time` is used for latency, `os` for the file name.

### 7. Fallback model: class and word lists  (`api/model_loader.py` lines 22–27)
```python
class KeywordSentimentModel:
    """Dependency-free fallback so the API runs even before you have trained anything."""

    classes_ = ["negative", "positive"]
    POSITIVE = {"good", "great", "love", "excellent", "amazing", "nice", "happy", "best"}
    NEGATIVE = {"bad", "terrible", "hate", "awful", "poor", "worst", "sad", "boring"}
```
- **In simple words:** A tiny built-in model so the API works before you train anything.
- **Exactly:** `classes_ = ["negative", "positive"]` fixes the column order. Two `set`s hold sentiment words.
- **Why it is here:** The service can start (and tests can run) without a model file.

### 8. Fallback model: predict_proba  (`api/model_loader.py` lines 29–37)
```python
    def predict_proba(self, texts):
        rows = []
        for text in texts:
            words = text.lower().split()
            pos = sum(w.strip(".,!?") in self.POSITIVE for w in words)  # True counts as 1
            neg = sum(w.strip(".,!?") in self.NEGATIVE for w in words)
            p_pos = (pos + 1) / (pos + neg + 2)          # add-one smoothing: never exactly 0 or 1
            rows.append([1 - p_pos, p_pos])
        return rows
```
- **In simple words:** Count happy and sad words and turn them into two probabilities.
- **Exactly:** Strips `.,!?` from each word. `p_pos = (pos + 1) / (pos + neg + 2)` is add-one (Laplace) smoothing, so it never gives exactly 0 or 1. Returns `[[1 - p_pos, p_pos], ...]`. Real values: "I love it" gives `[0.333, 0.667]`, "awful and bad" gives `[0.75, 0.25]` (ordered `negative, positive`).
- **Why it is here:** Mimics scikit-learn's output so the rest of the code is identical.

### 9. load_model: real file or fallback  (`api/model_loader.py` lines 40–46)
```python
def load_model():
    """Return (model, source_description). Real file if MODEL_PATH is set, else the fallback."""
    path = get_settings().model_path
    if path:
        import joblib                                    # lazy import: only needed for real model files
        return joblib.load(path), f"file:{os.path.basename(path)}"   # raises FileNotFoundError if wrong path
    return KeywordSentimentModel(), "fallback-keyword"
```
- **In simple words:** If a model path is set, load that file, otherwise use the fallback.
- **Exactly:** `get_settings().model_path` decides. `joblib.load(path)` unpickles the scikit-learn Pipeline (a lazy import, joblib is only needed here). Returns `(model, "file:model.joblib")` or `(KeywordSentimentModel(), "fallback-keyword")`. A wrong path raises `FileNotFoundError`.
- **Why it is here:** The source string is reported by `/health`, so you can tell which model is running.

### 10. predict_one: one text in, label and confidence out  (`api/model_loader.py` lines 49–53)
```python
def predict_one(model, text: str) -> tuple[str, float]:
    """Run the model on ONE text and return (label, confidence)."""
    probs = list(model.predict_proba([text])[0])         # list() so numpy arrays and lists behave the same
    best = max(range(len(probs)), key=probs.__getitem__) # index of the largest probability
    return str(model.classes_[best]), round(float(probs[best]), 4)
```
- **In simple words:** Ask the model about one sentence and pick the most likely class.
- **Exactly:** `predict_proba([text])[0]` is one row; `list()` makes numpy arrays and lists behave the same. `max(range(len(probs)), key=probs.__getitem__)` is argmax. Returns `(str(label), round(prob, 4))`, e.g. `('positive', 0.6667)` with the fallback.
- **Why it is here:** One small function is the single place where model output becomes API output.

### 11. Optional text generation, loaded lazily  (`api/model_loader.py` lines 56–67)
```python
# ------------------------- optional text generation (lazy) -------------------------
_GENERATOR = None                                        # cache: load the big model once, on first use


def generate_text(prompt: str, max_new_tokens: int = 20) -> str:
    """Generate text with distilgpt2 (downloads ~330 MB the first time; see module 04 steps 06 load model, 07 forward pass, 08 generation loop)."""
    global _GENERATOR
    if _GENERATOR is None:                               # LAZY LOADING: pay the cost only if someone calls /generate
        from transformers import pipeline                # heavy import, so keep it inside the function
        _GENERATOR = pipeline("text-generation", model="distilgpt2")
    out = _GENERATOR(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    return out[0]["generated_text"]
```
- **In simple words:** An extra feature that downloads a large model only if someone asks for it.
- **Exactly:** `_GENERATOR` is a module-level cache. The first call imports `transformers` and builds `pipeline("text-generation", model="distilgpt2")` (about 330 MB download); later calls reuse it. `do_sample=False` is greedy decoding, so output is deterministic.
- **Why it is here:** Without lazy loading every start would import transformers and take seconds and memory.

### 12. timed: measure latency  (`api/model_loader.py` lines 70–74)
```python
def timed(fn, *args):
    """Call fn(*args) and return (result, elapsed_ms). Handy for latency fields."""
    start = time.perf_counter()
    result = fn(*args)
    return result, round((time.perf_counter() - start) * 1000, 2)
```
- **In simple words:** A stopwatch around any function call.
- **Exactly:** `time.perf_counter()` is a monotonic high-resolution clock. Returns `(result, elapsed_ms)` rounded to 2 digits. Real value: `0.0` ms for the keyword model, `63.55` ms for the first call of the joblib pipeline.
- **Why it is here:** Latency is reported in every response, so you can see it without extra tools.

### 13. fastapi_app.py docstring  (`api/fastapi_app.py` lines 1–7)
```python
"""
fastapi_app.py -- the SAME model API written with FASTAPI (ASGI, async-capable).

RUN (dev):          uvicorn fastapi_app:app --port 8000 --reload      docs: http://127.0.0.1:8000/docs
RUN (production):   uvicorn fastapi_app:app --workers 2 --port 8000
Compare with flask_app.py: same endpoints, same JSON, different framework.
"""
```
- **In simple words:** What this file is and how to run it.
- **Exactly:** ASGI app served by uvicorn. Dev: `--reload`; production: `--workers 2`. Docs at `/docs` (Swagger UI).
- **Why it is here:** Documented run lines avoid guessing.

### 14. Imports  (`api/fastapi_app.py` lines 9–18)
```python
import asyncio
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import get_settings
from model_loader import generate_text, load_model, predict_one, timed
```
- **In simple words:** Load tools for async, process id, time, FastAPI, pydantic and our own modules.
- **Exactly:** `asynccontextmanager` builds the startup/shutdown hook; `Field` adds validation limits; local imports `config` and `model_loader`.
- **Why it is here:** Everything the file needs, in one place.

### 15. Startup hook: load the model once  (`api/fastapi_app.py` lines 21–27)
```python
@asynccontextmanager                      # turns this generator into a "startup ... shutdown" context manager
async def lifespan(app: FastAPI):
    # Startup: runs once per worker process before the first request.
    app.state.model, app.state.source = load_model()      # app.state = a place to keep shared objects
    print(f"[pid {os.getpid()}] fastapi_app loaded model from {app.state.source}", flush=True)
    yield                                                   # the app serves requests while paused here
    # Shutdown code would go here (close DB connections, flush logs ...)
```
- **In simple words:** When the server starts, load the model and keep it ready.
- **Exactly:** `lifespan` runs once per worker process: code before `yield` is startup, after `yield` shutdown. `app.state.model, app.state.source = load_model()` stores the shared objects. The print shows the pid, so with `--workers 2` you see two loads.
- **Why it is here:** Loading per request would add hundreds of ms to every call.

### 16. Create the app  (`api/fastapi_app.py` lines 30–30)
```python
app = FastAPI(title="Sentiment API", version="1.0", lifespan=lifespan)
```
- **In simple words:** Build the application and attach the startup hook.
- **Exactly:** `FastAPI(title=..., version="1.0", lifespan=lifespan)` shows in `/docs`.
- **Why it is here:** Without `lifespan=` the model would never be loaded and `/predict` would fail with `AttributeError`.

### 17. Request schema  (`api/fastapi_app.py` lines 33–34)
```python
class PredictRequest(BaseModel):
    text: str = Field(min_length=1, description="Text to classify")   # description shows up in /docs
```
- **In simple words:** The shape of the incoming JSON: a non-empty text.
- **Exactly:** `Field(min_length=1)`: an empty string gives HTTP 422 (real run confirmed).
- **Why it is here:** Rejects garbage before it touches the model.

### 18. Response schema  (`api/fastapi_app.py` lines 37–41)
```python
class PredictResponse(BaseModel):          # response_model: documents AND filters what we return
    label: str
    confidence: float
    model_version: str
    latency_ms: float
```
- **In simple words:** The shape of the answer.
- **Exactly:** Four fields: `label`, `confidence`, `model_version`, `latency_ms`. Used as `response_model` so FastAPI validates and filters the output and documents it in `/docs`.
- **Why it is here:** The UI can rely on these keys.

### 19. Generate request schema  (`api/fastapi_app.py` lines 44–46)
```python
class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=20, ge=1, le=100)
```
- **In simple words:** Rules for the optional generation endpoint.
- **Exactly:** `prompt` non-empty; `max_new_tokens` defaults to 20 and must be between 1 and 100 (`ge`, `le`).
- **Why it is here:** Caps cost and latency of generation.

### 20. Health endpoint  (`api/fastapi_app.py` lines 49–52)
```python
@app.get("/health")
def health():
    s = get_settings()
    return {"status": "ok", "model_version": s.model_version, "model_source": app.state.source}
```
- **In simple words:** Tell callers the service is alive and which model it uses.
- **Exactly:** Returns `{'status': 'ok', 'model_version': '0.1.0', 'model_source': 'file:model.joblib'}`. Used by the compose healthcheck.
- **Why it is here:** Without it the UI would start before the API is ready, or never know it is broken.

### 21. Predict endpoint  (`api/fastapi_app.py` lines 55–63)
```python
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Plain `def` (not `async def`): FastAPI runs it in a thread pool so CPU-heavy model code
    # does not freeze the event loop.
    s = get_settings()
    if len(req.text) > s.max_text_length:
        raise HTTPException(status_code=422, detail=f"'text' longer than {s.max_text_length} chars")
    (label, confidence), ms = timed(predict_one, app.state.model, req.text)
    return PredictResponse(label=label, confidence=confidence, model_version=s.model_version, latency_ms=ms)
```
- **In simple words:** Check the length, run the model, time it, answer.
- **Exactly:** `def` (not `async def`) so FastAPI runs it in a thread pool and CPU-heavy model code does not freeze the event loop. Over-long text raises `HTTPException(422)`. `timed(predict_one, model, text)` returns `((label, confidence), ms)`. Real response: `{'label': 'positive', 'confidence': 0.502, 'model_version': '0.1.0', 'latency_ms': 63.55}`.
- **Why it is here:** This is the route the UI calls; the length guard protects memory and latency.

### 22. Generate endpoint  (`api/fastapi_app.py` lines 66–70)
```python
@app.post("/generate")
def generate(req: GenerateRequest):
    if not get_settings().enable_generate:
        raise HTTPException(status_code=503, detail="generation disabled; start with ENABLE_GENERATE=1")
    return {"text": generate_text(req.prompt, req.max_new_tokens)}
```
- **In simple words:** Text generation, switched off unless enabled.
- **Exactly:** If `ENABLE_GENERATE` is not `1`, returns 503 "generation disabled" (real run: 503). Otherwise calls `generate_text`.
- **Why it is here:** Prevents an accidental 330 MB download in production.

### 23. Whoami helper  (`api/fastapi_app.py` lines 73–76)
```python
# ----- helper endpoints used by step 07 to explore workers and concurrency -----
@app.get("/whoami")
def whoami():
    return {"pid": os.getpid()}
```
- **In simple words:** Reports which process answered.
- **Exactly:** `os.getpid()`. With several workers the pid changes between calls.
- **Why it is here:** Lets you see workers working (step 07 experiments).

### 24. Async sleep helper  (`api/fastapi_app.py` lines 79–82)
```python
@app.get("/slow")                          # async def + await asyncio.sleep: the loop serves OTHER requests meanwhile
async def slow(seconds: float = 1):
    await asyncio.sleep(seconds)
    return {"pid": os.getpid(), "slept": True}
```
- **In simple words:** A slow endpoint that lets other requests through while it waits.
- **Exactly:** `async def` plus `await asyncio.sleep` yields control to the event loop, so concurrent requests are served.
- **Why it is here:** Shows the correct way to wait inside async code.

### 25. Sync sleep helper  (`api/fastapi_app.py` lines 85–88)
```python
@app.get("/slow-sync")                     # plain def + time.sleep: runs in a thread, so still concurrent (up to ~40 threads)
def slow_sync(seconds: float = 1):
    time.sleep(seconds)
    return {"pid": os.getpid(), "slept": True}
```
- **In simple words:** A slow endpoint written as a normal function.
- **Exactly:** `def` + `time.sleep` runs in a worker thread (about 40 by default), so it is still concurrent.
- **Why it is here:** Shows that plain `def` is safe for blocking code.

### 26. Blocking in async: the anti-pattern  (`api/fastapi_app.py` lines 91–94)
```python
@app.get("/slow-blocking")                 # ANTI-PATTERN: async def but a BLOCKING call -> freezes the whole event loop
async def slow_blocking(seconds: float = 1):
    time.sleep(seconds)
    return {"pid": os.getpid(), "slept": True}
```
- **In simple words:** A slow endpoint that blocks everyone.
- **Exactly:** `async def` with `time.sleep` blocks the single event-loop thread, so every other request on that worker waits.
- **Why it is here:** The classic FastAPI performance bug; it is here on purpose to measure it.

### 27. API dependencies  (`api/requirements.txt` lines 1–7)
```text
fastapi
uvicorn[standard]
pydantic
joblib
scikit-learn
numpy
python-dotenv
```
- **In simple words:** The libraries the API image installs.
- **Exactly:** `fastapi`, `uvicorn[standard]`, `pydantic`, `joblib`, `scikit-learn`, `numpy`, `python-dotenv`. Not pinned (unlike step 02 of module 11): good for a demo, risky for reproducibility; the model file was trained with a specific scikit-learn version, so a mismatch can warn or break `joblib.load`.
- **Why it is here:** Pin versions (`==`) or use a lock file for production.

### 28. API image: base and folder  (`api/Dockerfile` lines 1–2)
```dockerfile
FROM python:3.12-slim
WORKDIR /app
```
- **In simple words:** Python base image and a folder `/app`.
- **Exactly:** `python:3.12-slim`, `WORKDIR /app`.
- **Why it is here:** Same as module 11 step 01.

### 29. API image: dependencies first  (`api/Dockerfile` lines 3–5)
```dockerfile
# dependencies first: this layer is cached until requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```
- **In simple words:** Install the libraries before copying code.
- **Exactly:** Layer cache rule from module 11 step 02: this layer is rebuilt only when `requirements.txt` changes.
- **Why it is here:** Rebuilds take seconds, not minutes.

### 30. API image: copy code and artifacts, set model path  (`api/Dockerfile` lines 6–7)
```dockerfile
COPY . .
ENV MODEL_PATH=/app/artifacts/model.joblib
```
- **In simple words:** Copy everything (code and `artifacts/`), and tell the app where the model file is.
- **Exactly:** `COPY . .` includes `artifacts/model.joblib` and `metadata.json`. `ENV MODEL_PATH=/app/artifacts/model.joblib` is the default; compose sets the same value.
- **Why it is here:** Without the model file in the image, `load_model()` raises `FileNotFoundError` at startup and the container exits.

### 31. API image: run as non-root  (`api/Dockerfile` lines 8–9)
```dockerfile
RUN useradd --create-home appuser
USER appuser
```
- **In simple words:** Create an ordinary user and switch to it.
- **Exactly:** `useradd --create-home appuser`, then `USER appuser` (module 11 step 06).
- **Why it is here:** Limits the damage of a compromised process.

### 32. API image: port and start command  (`api/Dockerfile` lines 10–12)
```dockerfile
EXPOSE 8000
# 0.0.0.0 so the port mapping works from outside the container
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
```
- **In simple words:** Document port 8000 and start uvicorn on all interfaces.
- **Exactly:** `EXPOSE 8000` is documentation. `--host 0.0.0.0` lets the published port reach the app. `fastapi_app:app` = file `fastapi_app.py`, variable `app`, resolved relative to `/app`.
- **Why it is here:** With the default host `127.0.0.1` the container would be up and unreachable (module 11 step 05).

### 33. UI docstring  (`ui/app.py` lines 1–12)
```python
"""
GOAL: A Streamlit UI that does NOT hold the model itself; it calls an HTTP API (module 09, or common/mock_api.py)
      and shows loading, success and several kinds of error states.

RUN:
  terminal 1 (API):  cd 12_streamlit_gradio/common && ../../.venv/bin/python -m uvicorn mock_api:app --port 8000
                     (or start your module-09 API on port 8000 instead)
  terminal 2 (UI):   cd 12_streamlit_gradio/03_streamlit_api_client && ../../.venv/bin/streamlit run app.py
EXPECTED OUTPUT: type a sentence, press Predict -> spinner -> label + confidence. Stop the API and press again ->
                 a red "Cannot reach the API" box (the UI itself stays alive).
NEEDS: streamlit, requests.
"""
```
- **In simple words:** Explains that the UI holds no model and calls an API.
- **Exactly:** This file is the same as `12_streamlit_gradio/03_streamlit_api_client/app.py` (checked with `diff`). The run lines in the docstring are for module 12; here use `run_local.sh` or compose.
- **Why it is here:** Real systems separate the UI from the model service so each can be scaled and deployed separately.

### 34. Imports  (`ui/app.py` lines 13–16)
```python
import os

import requests          # 3rd-party HTTP client (httpx is the modern alternative with almost the same API)
import streamlit as st
```
- **In simple words:** Tools for environment, HTTP and Streamlit.
- **Exactly:** `requests` is the HTTP client.
- **Why it is here:** The UI is a thin client.

### 35. Where the API lives  (`ui/app.py` lines 18–19)
```python
# Where is the API? Environment variable first (Docker/Compose sets API_URL), else localhost.
DEFAULT_URL = os.getenv("API_URL", "http://localhost:8000")
```
- **In simple words:** Read the API address from the environment.
- **Exactly:** Compose sets `API_URL=http://api:8000` (service name plus container port); otherwise `http://localhost:8000`.
- **Why it is here:** Inside the ui container `localhost` would be the ui container itself.

### 36. Title, sidebar address and health button  (`ui/app.py` lines 21–29)
```python
st.title("Sentiment UI -> API")
api_url = st.sidebar.text_input("API base URL", DEFAULT_URL)      # sidebar = left panel

if st.sidebar.button("Check /health"):
    try:
        r = requests.get(f"{api_url}/health", timeout=3)          # timeout: NEVER call a network without one
        st.sidebar.success(f"API says: {r.json()}")
    except requests.RequestException as exc:                      # base class of all requests errors
        st.sidebar.error(f"API down: {type(exc).__name__}")
```
- **In simple words:** Heading, an editable address and a "Check /health" button.
- **Exactly:** `requests.get(..., timeout=3)`; `RequestException` is the base class of all requests errors. The sidebar shows `{'status': 'ok', ...}` on success.
- **Why it is here:** A quick check without sending a prediction.

### 37. Text box and validation  (`ui/app.py` lines 31–36)
```python
text = st.text_area("Text to classify", "The service was excellent")   # multi-line input

if st.button("Predict"):
    if not text.strip():
        st.warning("Please type something first.")                # validate BEFORE the network call
    else:
```
- **In simple words:** A text area, and a warning if it is empty.
- **Exactly:** `st.text_area` returns `str`; blank input shows `st.warning` and no network call is made.
- **Why it is here:** Avoids a needless 422.

### 38. Call /predict and show it  (`ui/app.py` lines 37–44)
```python
        try:
            with st.spinner("Calling the API..."):                # shows a spinner while the block runs
                resp = requests.post(f"{api_url}/predict", json={"text": text}, timeout=10)
            resp.raise_for_status()                               # 4xx/5xx -> HTTPError
            data = resp.json()
            st.metric("Label", data["label"], f"score {data.get('confidence', data.get('score'))}")  # 09 API: 'confidence'; mock API: 'score'
            with st.expander("Raw response"):                     # collapsible section
                st.json(data)
```
- **In simple words:** Send the text and show label and confidence.
- **Exactly:** `requests.post(f"{api_url}/predict", json={"text": text}, timeout=10)`; `raise_for_status()`; `data.get('confidence', data.get('score'))` reads this API's `confidence` field. The real API returns `label`, `confidence`, `model_version`, `latency_ms`.
- **Why it is here:** Spinner, metric and raw JSON show state clearly.

### 39. Handle each failure  (`ui/app.py` lines 45–52)
```python
        except requests.ConnectionError:
            st.error(f"Cannot reach the API at {api_url}. Is it running?")
        except requests.Timeout:
            st.error("The API took too long to answer (timeout).")
        except requests.HTTPError as exc:
            st.error(f"API returned {exc.response.status_code}: {exc.response.text}")
        except (KeyError, ValueError):                            # missing key / invalid JSON
            st.error("The API answered, but not in the expected format.")
```
- **In simple words:** A friendly message for each kind of failure.
- **Exactly:** `ConnectionError` (API down), `Timeout`, `HTTPError` (shows status code), `KeyError`/`ValueError` (wrong JSON shape).
- **Why it is here:** The UI stays up and says what is wrong.

### 40. Exercises as comments  (`ui/app.py` lines 54–56)
```python
# TRY IT: change the URL in the sidebar to a wrong port and press Predict.
# TRY IT: add a time.sleep(3) inside mock_api.predict and watch the spinner.
# BREAK IT: change data["label"] to data["labell"] -> KeyError caught by our friendly message; remove the except and see the raw traceback.
```
- **In simple words:** Ideas to try.
- **Exactly:** Wrong port in the sidebar; mis-typed key.
- **Why it is here:** Practice reading failure modes.

### 41. UI dependencies  (`ui/requirements.txt` lines 1–2)
```text
streamlit
requests
```
- **In simple words:** Just two libraries.
- **Exactly:** `streamlit`, `requests`. No scikit-learn: the UI image stays small because the model lives in the API.
- **Why it is here:** Smaller images, faster builds.

### 42. UI image: base, install, copy  (`ui/Dockerfile` lines 1–5)
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
```
- **In simple words:** Same recipe as the API: libraries first, then the single file.
- **Exactly:** `COPY app.py .` copies one file to `/app/app.py`.
- **Why it is here:** Cache order from module 11.

### 43. UI image: non-root user  (`ui/Dockerfile` lines 6–7)
```dockerfile
RUN useradd --create-home appuser
USER appuser
```
- **In simple words:** Run as an ordinary user.
- **Exactly:** `useradd` and `USER appuser`.
- **Why it is here:** Least privilege.

### 44. UI image: port and start command  (`ui/Dockerfile` lines 8–9)
```dockerfile
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
```
- **In simple words:** Document port 8501 and start Streamlit on all interfaces, without opening a browser.
- **Exactly:** `--server.address=0.0.0.0`, `--server.port=8501`, `--server.headless=true`.
- **Why it is here:** Without the address flag the UI is unreachable from the host.

### 45. compose: api service build, port, env  (`compose.yaml` lines 1–6)
```yaml
services:
  api:
    build: ./api
    ports: ["8000:8000"]
    environment:
      MODEL_PATH: /app/artifacts/model.joblib
```
- **In simple words:** Tell Compose how to build and publish the API.
- **Exactly:** `build: ./api` uses `api/Dockerfile`. `ports: ["8000:8000"]` is HOST:CONTAINER. `MODEL_PATH` points to the artifact inside the image.
- **Why it is here:** Without `ports` the API is reachable only from other containers.

### 46. compose: api healthcheck  (`compose.yaml` lines 7–12)
```yaml
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request as u; u.urlopen('http://localhost:8000/health')"]
      interval: 10s
      timeout: 3s
      retries: 3
      start_period: 5s
```
- **In simple words:** A probe that checks the API answers on /health.
- **Exactly:** Runs `python -c "... urlopen('http://localhost:8000/health')"` (curl is not in `python:slim`); every 10 s, 3 s timeout, 3 retries, 5 s `start_period`.
- **Why it is here:** The UI waits on this signal.

### 47. compose: ui service  (`compose.yaml` lines 13–17)
```yaml
  ui:
    build: ./ui
    ports: ["8501:8501"]
    environment:
      API_URL: http://api:8000      # service name = DNS name on the compose network
```
- **In simple words:** Build the UI and tell it where the API is.
- **Exactly:** `API_URL: http://api:8000` uses the service name (Docker DNS) and the container port 8000. `ports: ["8501:8501"]` publishes Streamlit.
- **Why it is here:** `localhost` would not work between containers.

### 48. compose: wait for a healthy API  (`compose.yaml` lines 18–20)
```yaml
    depends_on:
      api:
        condition: service_healthy
```
- **In simple words:** Start the UI only after the API is healthy.
- **Exactly:** `depends_on.api.condition: service_healthy`.
- **Why it is here:** Without it the UI may show "cannot reach the API" during the first seconds.

### 49. run_local.sh: shebang, safe mode, location, Python  (`run_local.sh` lines 1–5)
```bash
#!/usr/bin/env bash
# Run the capstone WITHOUT Docker (two terminals' worth in one script). Ctrl+C stops both.
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
```
- **In simple words:** Prepare the script: stop on errors, go to its own folder, choose the repo's Python.
- **Exactly:** `#!/usr/bin/env bash` picks bash. `set -e` exits on the first failing command. `cd "$(dirname "$0")"` goes to the script's folder so paths work from anywhere. `PY=../.venv/bin/python` is the repo virtual environment.
- **Why it is here:** Without `set -e` the script would carry on after a failure; without the `cd` relative paths break.

### 50. run_local.sh: start the API in the background  (`run_local.sh` lines 6–8)
```bash
(cd api && MODEL_PATH=artifacts/model.joblib $PY -m uvicorn fastapi_app:app --port 8000) &
API=$!
trap "kill $API" EXIT
```
- **In simple words:** Start the API and make sure it stops when the script ends.
- **Exactly:** `( cd api && MODEL_PATH=... python -m uvicorn fastapi_app:app --port 8000 ) &` runs in a subshell in the background; `$!` is its process id; `trap "kill $API" EXIT` kills it when the script exits (including Ctrl+C).
- **Why it is here:** Otherwise the API keeps running after you stop the UI and blocks port 8000.

### 51. run_local.sh: wait, then start the UI  (`run_local.sh` lines 9–10)
```bash
sleep 3
API_URL=http://localhost:8000 $PY -m streamlit run ui/app.py --server.port 8501
```
- **In simple words:** Give the API a moment, then start the UI in the foreground.
- **Exactly:** `sleep 3` is a crude wait for startup (compose uses a healthcheck instead). `API_URL=http://localhost:8000` points the UI at the API; `streamlit run ui/app.py --server.port 8501` blocks until Ctrl+C.
- **Why it is here:** A fixed sleep can be too short on a slow machine; that is why compose uses `service_healthy`.


