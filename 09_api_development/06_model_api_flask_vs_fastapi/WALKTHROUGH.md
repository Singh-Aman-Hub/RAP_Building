# Step 06 — The same model API in Flask and FastAPI

## The story so far
Step 05 made the door safe. Now we put the real thing behind it: a trained model that answers `/predict`, in both frameworks so you see what really differs.

## The pain point
Loading a model takes hundreds of milliseconds to seconds. Doing it inside the request handler makes every user pay that cost, and hard-coded paths make the service fragile.

## The spark
Load the model once when the process starts (module level or FastAPI `lifespan`), keep it in memory, and have each request only call `predict_proba`. Read the model path from an environment variable.

## Goal
Serve a classifier at `POST /predict` with `GET /health`, in both frameworks, sharing one `model_loader.py` and one `config.py`, so the *only* difference is the framework.

## Files
| File | Role |
|---|---|
| `config.py` | Settings from environment variables (`MODEL_PATH`, `MODEL_VERSION`, `MAX_TEXT_LENGTH`, `ENABLE_GENERATE`) |
| `model_loader.py` | `KeywordSentimentModel` fallback, `load_model()`, `predict_one()`, lazy `generate_text()` |
| `flask_app.py` | Flask version (WSGI, manual validation) |
| `fastapi_app.py` | FastAPI version (ASGI, Pydantic + lifespan) |
| `main.py` | In-process comparison of both |

## Run it
```bash
cd 09_api_development/06_model_api_flask_vs_fastapi
python main.py                                          # fallback model, both apps compared
MODEL_PATH=../../03_ml_foundations/08_sklearn_text_classifier/artifacts/model.joblib python main.py   # REAL model from module 03
MODEL_PATH=../../10_ml_system_architecture/02_training_vs_inference_pipelines/artifacts/model.joblib python main.py   # module 10's model
uvicorn fastapi_app:app --port 8000                     # then http://127.0.0.1:8000/docs
python flask_app.py                                     # http://127.0.0.1:5001
ENABLE_GENERATE=1 python main.py                        # + /generate with distilgpt2 (~330 MB first run)
```
Real output (fallback model):
```
health  flask  = {'model_source': 'fallback-keyword', 'model_version': '0.1.0', 'status': 'ok'}
health  fastapi= {'status': 'ok', 'model_version': '0.1.0', 'model_source': 'fallback-keyword'}

'I love this great film'         flask=positive 0.75   fastapi=positive 0.75
'This was terrible and boring'   flask=negative 0.75   fastapi=negative 0.75
'It was a film'                  flask=negative 0.5   fastapi=negative 0.5

bad requests (Flask status / FastAPI status):
  missing text  422 / 422
  empty text    422 / 422
  number        422 / 422
  wrong method   405 / 405
  unknown route  404 / 404
  generate probe  503 / 503
```
With the real module-03 model (`MODEL_PATH` set): `'I love this great film' -> positive 0.6207`, `'This was terrible and boring' -> negative 0.6873` — identical in both apps.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Flask vs FastAPI | Both are Python web frameworks | No | Flask = WSGI, sync, manual validation; FastAPI = ASGI, async-capable, Pydantic validation and automatic docs |
| WSGI vs ASGI | Python web server interfaces | No | WSGI handles one request per call, synchronous; ASGI adds async and websockets/streaming |
| Pydantic model vs dataclass | Typed data holders | No | Pydantic validates and coerces at runtime; a dataclass (like `Settings`) only stores values |
| lifespan vs `before_first_request` / import-time load | Startup hooks | Similar goal | Lifespan is the modern FastAPI hook with explicit shutdown; Flask loads at import time (`before_first_request` was removed in Flask 2.3) |
| 422 vs 400 | Client error codes | No | 400 = malformed request (not JSON); 422 = well-formed but semantically invalid |
| `error` vs `detail` body | Error JSON shape | No | Flask here returns `{"error": ...}`; FastAPI returns `{"detail": ...}` by default |
| 503 vs 500 | Server error codes | No | 500 = unexpected bug; 503 = intentionally unavailable (feature off) |
| Lazy loading vs singleton | Load on first use / one instance | Overlapping | Lazy = when to load; singleton = how many instances. `_GENERATOR` is both |

## Code, step by step

### 1. Module docstring, imports, load .env  (`config.py` lines 1–16)
```python
"""
config.py -- ALL settings come from environment variables (12-factor style).

Why: the same code/Docker image must run on your laptop, in CI and in production with
different settings, WITHOUT editing code.  See step 08 for experiments.
"""

import os
from dataclasses import dataclass  # a decorator that writes __init__/__repr__ for a plain data holder

from dotenv import load_dotenv     # third-party: python-dotenv. Reads KEY=value lines from a .env file

# Loads .env (if one exists in this folder or a parent) into os.environ.
# Real environment variables WIN over .env by default (override=False).
load_dotenv()

```
- **In simple words:** The file says why it exists, then pulls in the tools it needs and reads an optional `.env` file.
- **Exactly:** `os` gives `os.getenv`; `dataclass` generates `__init__`/`__repr__`; `load_dotenv()` (python-dotenv) copies `KEY=value` lines from a `.env` file found in this folder or a parent into `os.environ`, without overriding variables already set in the shell (`override=False`).
- **Why it is here:** Without it every setting would be hard-coded, so laptop, CI and production would need different code.

### 2. The Settings container (frozen dataclass)  (`config.py` lines 18–23)
```python
@dataclass(frozen=True)            # frozen=True -> attributes cannot be changed after creation
class Settings:
    model_path: str | None         # path to a joblib model file; None -> use the built-in fallback model
    model_version: str             # reported in every response so you know WHICH model answered
    max_text_length: int           # reject giant inputs (protects latency and memory)
    enable_generate: bool          # the /generate endpoint downloads ~330 MB, so it is opt-in
```
- **In simple words:** A small labelled box that holds the four settings. Once built, nobody can change what is inside.
- **Exactly:** `@dataclass(frozen=True)` writes `__init__(model_path, model_version, max_text_length, enable_generate)` and makes assignment raise `FrozenInstanceError`. Types: `str | None`, `str`, `int`, `bool`.
- **Why it is here:** One typed object is easier to pass around and test than four loose `os.getenv` calls scattered through the code; frozen prevents a request handler silently changing config.

### 3. get_settings reads the environment right now  (`config.py` lines 26–33)
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
- **In simple words:** Each call looks at the environment again and hands back a fresh Settings box.
- **Exactly:** `os.getenv("MODEL_PATH") or None` turns an empty string into `None`. `os.getenv(name, default)` returns a string, so `int(...)` converts `MAX_TEXT_LENGTH` (default `"500"` -> 500). `enable_generate` is `True` only when the variable is exactly `"1"`. In the real run: `model_version="0.1.0"`, `max_text_length=500`, `enable_generate=False`.
- **Why it is here:** Reading at call time (not import time) lets tests change env vars with `monkeypatch`; reading at import would freeze the first values forever.

### 4. Docstring (the model contract) and imports  (`model_loader.py` lines 1–19)
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

import os
import time

from config import get_settings
```
- **In simple words:** Explains the one rule any model must follow, then imports `os`, `time` and our settings.
- **Exactly:** Contract: `predict_proba(list_of_texts) -> list of probability rows` and `classes_` listing label names in column order. scikit-learn Pipelines satisfy it, and so does the fallback below.
- **Why it is here:** Both web apps import this file, so the only difference between them is the web framework.

### 5. Fallback model class with word lists  (`model_loader.py` lines 22–27)
```python
class KeywordSentimentModel:
    """Dependency-free fallback so the API runs even before you have trained anything."""

    classes_ = ["negative", "positive"]
    POSITIVE = {"good", "great", "love", "excellent", "amazing", "nice", "happy", "best"}
    NEGATIVE = {"bad", "terrible", "hate", "awful", "poor", "worst", "sad", "boring"}
```
- **In simple words:** A tiny stand-in "model" that knows a few happy and sad words, so the API runs before you train anything.
- **Exactly:** `classes_ = ["negative", "positive"]` fixes column order: column 0 = negative, column 1 = positive. `POSITIVE` and `NEGATIVE` are class-level sets (O(1) membership tests).
- **Why it is here:** Without a fallback the API would crash on a fresh checkout with no `model.joblib`.

### 6. predict_proba: count words, smooth, return rows  (`model_loader.py` lines 29–37)
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
- **In simple words:** Count the good words and bad words in each sentence and turn the counts into two probabilities that add up to 1.
- **Exactly:** `text.lower().split()` splits on whitespace; `w.strip(".,!?")` removes punctuation; `sum(bool ...)` counts True as 1. `p_pos = (pos+1)/(pos+neg+2)` is add-one (Laplace) smoothing. Real run: "I love this great film" -> pos=2, neg=0 -> p_pos=3/4=0.75, row `[0.25, 0.75]`; "It was a film" -> 1/2 = 0.5. Returns a list of `[p_neg, p_pos]` lists, same shape contract as scikit-learn.
- **Why it is here:** Smoothing means the model never says exactly 0% or 100%, and avoids a divide-by-zero when no keyword is found.

### 7. load_model: real file or fallback  (`model_loader.py` lines 40–46)
```python
def load_model():
    """Return (model, source_description). Real file if MODEL_PATH is set, else the fallback."""
    path = get_settings().model_path
    if path:
        import joblib                                    # lazy import: only needed for real model files
        return joblib.load(path), f"file:{os.path.basename(path)}"   # raises FileNotFoundError if wrong path
    return KeywordSentimentModel(), "fallback-keyword"
```
- **In simple words:** Pick the real saved model if a path was given, otherwise use the built-in fallback.
- **Exactly:** Returns a tuple `(model, source_description)`. `import joblib` is lazy so the dependency is only needed for real files. `joblib.load(path)` unpickles the object and raises `FileNotFoundError` for a wrong path. Real run: source `"fallback-keyword"`.
- **Why it is here:** This is the single place that decides which model serves; swap models by env var, not by editing code. The source string is shown in `/health` so you can tell which model is live.

### 8. predict_one: argmax over probabilities  (`model_loader.py` lines 49–53)
```python
def predict_one(model, text: str) -> tuple[str, float]:
    """Run the model on ONE text and return (label, confidence)."""
    probs = list(model.predict_proba([text])[0])         # list() so numpy arrays and lists behave the same
    best = max(range(len(probs)), key=probs.__getitem__) # index of the largest probability
    return str(model.classes_[best]), round(float(probs[best]), 4)
```
- **In simple words:** Ask the model about one sentence and return the winning label and how sure it is.
- **Exactly:** `model.predict_proba([text])[0]` takes row 0 of the result (one text in, one row out). `list()` makes numpy arrays and plain lists behave the same. `max(range(n), key=probs.__getitem__)` is argmax: the index with the largest value. Returns `(str(label), round(float, 4))`, e.g. `("positive", 0.75)`.
- **Why it is here:** Both apps need the same label/confidence logic; putting it here guarantees identical answers (the test in `main.py`).

### 9. Cache variable for the lazy generator  (`model_loader.py` lines 56–57)
```python
# ------------------------- optional text generation (lazy) -------------------------
_GENERATOR = None                                        # cache: load the big model once, on first use
```
- **In simple words:** A shelf that starts empty; the big text-generation model is put on it only when first needed.
- **Exactly:** Module-level `_GENERATOR = None`; the leading underscore means "private to this module".
- **Why it is here:** Loading distilgpt2 (~330 MB) at import would slow every start-up and need a download even for people who never call `/generate`.

### 10. generate_text: lazy-load distilgpt2, then generate  (`model_loader.py` lines 60–67)
```python
def generate_text(prompt: str, max_new_tokens: int = 20) -> str:
    """Generate text with distilgpt2 (downloads ~330 MB the first time; see 04_llm_fundamentals step 06 and 08, module 05)."""
    global _GENERATOR
    if _GENERATOR is None:                               # LAZY LOADING: pay the cost only if someone calls /generate
        from transformers import pipeline                # heavy import, so keep it inside the function
        _GENERATOR = pipeline("text-generation", model="distilgpt2")
    out = _GENERATOR(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    return out[0]["generated_text"]
```
- **In simple words:** The first call loads the language model; every later call reuses it.
- **Exactly:** `global _GENERATOR` lets the function assign the module variable. `pipeline("text-generation", model="distilgpt2")` builds tokenizer + model. `do_sample=False` = greedy decoding (deterministic). Returns `out[0]["generated_text"]`, which includes the prompt followed by the new text. `max_new_tokens` caps how many new tokens are produced.
- **Why it is here:** Lazy loading plus caching means you pay the download/memory cost once, and only if the feature is used.

### 11. timed: measure how long a call takes  (`model_loader.py` lines 70–74)
```python
def timed(fn, *args):
    """Call fn(*args) and return (result, elapsed_ms). Handy for latency fields."""
    start = time.perf_counter()
    result = fn(*args)
    return result, round((time.perf_counter() - start) * 1000, 2)
```
- **In simple words:** Start a stopwatch, run the function, stop the stopwatch, return both the answer and the time.
- **Exactly:** `time.perf_counter()` is a monotonic high-resolution clock. `fn(*args)` calls any function with any positional args. Returns `(result, ms)` rounded to 2 decimals; typical real value for the fallback: `0.02` to `0.03` ms.
- **Why it is here:** Latency is a field in every response; measuring in one helper keeps both apps consistent.

### 12. Flask app: docstring and imports  (`flask_app.py` lines 1–16)
```python
"""
flask_app.py -- the model API written with FLASK (WSGI, synchronous).

RUN (dev server):   python flask_app.py                      -> http://127.0.0.1:5001
RUN (production):   gunicorn flask_app:app --workers 2 --bind 127.0.0.1:5001
                    ("flask_app:app" = file flask_app.py, variable `app`)
Compare with fastapi_app.py: same endpoints, same JSON, different framework.
"""

import os
import time

from flask import Flask, jsonify, request

from config import get_settings
from model_loader import generate_text, load_model, predict_one, timed
```
- **In simple words:** Names the file, says how to run it, then imports Flask pieces plus our own helpers.
- **Exactly:** `Flask` = the app class, `jsonify` = dict -> JSON response with `Content-Type: application/json`, `request` = the current request (a proxy object).
- **Why it is here:** These are the only framework-specific imports; the model logic comes from `model_loader`.

### 13. Create the app and load the model once  (`flask_app.py` lines 18–23)
```python
app = Flask(__name__)

# Loaded ONCE per process at import time. Under Gunicorn with 2 workers this line runs TWICE
# (once per worker process) => the model sits in memory twice. See step 07.
MODEL, MODEL_SOURCE = load_model()
print(f"[pid {os.getpid()}] flask_app loaded model from {MODEL_SOURCE}", flush=True)
```
- **In simple words:** Make the web app, then load the model one time when the file starts.
- **Exactly:** `Flask(__name__)` tells Flask where to find resources. `load_model()` returns `(model, source)`; real run printed `[pid 20772] flask_app loaded model from fallback-keyword`. Runs at import time, once per process. `flush=True` pushes the line out immediately.
- **Why it is here:** Loading per request would cost hundreds of ms each time. Under Gunicorn with 2 workers it runs twice, which is why memory doubles.

### 14. GET /health route  (`flask_app.py` lines 26–29)
```python
@app.get("/health")
def health():
    s = get_settings()
    return jsonify(status="ok", model_version=s.model_version, model_source=MODEL_SOURCE)
```
- **In simple words:** A "are you alive and which model are you?" endpoint.
- **Exactly:** `@app.get("/health")` registers a GET route. Real output: `{"model_source":"fallback-keyword","model_version":"0.1.0","status":"ok"}`. Flask sorts JSON keys alphabetically (FastAPI keeps insertion order).
- **Why it is here:** Load balancers and humans use health checks to know the service is up and which version answered.

### 15. POST /predict with manual validation  (`flask_app.py` lines 32–44)
```python
@app.post("/predict")
def predict():
    s = get_settings()
    data = request.get_json(silent=True)                       # None if body is missing / not JSON
    if not isinstance(data, dict):
        return jsonify(error="body must be a JSON object"), 400
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify(error="'text' must be a non-empty string"), 422
    if len(text) > s.max_text_length:
        return jsonify(error=f"'text' longer than {s.max_text_length} chars"), 422
    (label, confidence), ms = timed(predict_one, MODEL, text)
    return jsonify(label=label, confidence=confidence, model_version=s.model_version, latency_ms=ms)
```
- **In simple words:** Check the request by hand step by step, then run the model and return the answer.
- **Exactly:** `request.get_json(silent=True)` returns `None` instead of raising on bad/missing JSON. Rules: not a dict -> 400; `text` not a non-empty string -> 422; too long -> 422. Then `timed(predict_one, MODEL, text)` returns `((label, confidence), ms)`. Real response: `{"confidence":0.6667,"label":"positive","latency_ms":0.03,"model_version":"0.1.0"}`. `{"text": 5}` gives `{"error":"'text' must be a non-empty string"}` with 422.
- **Why it is here:** Never trust client input. Without these checks a number or empty string would crash the model with a 500.

### 16. POST /generate, opt-in and validated  (`flask_app.py` lines 47–55)
```python
@app.post("/generate")
def generate():
    if not get_settings().enable_generate:
        return jsonify(error="generation disabled; start with ENABLE_GENERATE=1"), 503
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        return jsonify(error="'prompt' must be a non-empty string"), 422
    return jsonify(text=generate_text(prompt, int(data.get("max_new_tokens", 20))))
```
- **In simple words:** A text-generation endpoint that is switched off unless you asked for it.
- **Exactly:** Disabled -> 503 Service Unavailable (real probe: 503). Empty/non-string prompt -> 422. `data.get("max_new_tokens", 20)` is cast with `int()`. Returns `{"text": ...}`.
- **Why it is here:** The model download is 330 MB, so the feature is opt-in; 503 tells clients "temporarily not available" rather than "you sent bad data".

### 17. Helper routes /whoami and /slow  (`flask_app.py` lines 58–67)
```python
# ----- helper endpoints used by step 07 to explore workers and concurrency -----
@app.get("/whoami")
def whoami():
    return jsonify(pid=os.getpid())                             # which worker process answered?


@app.get("/slow")
def slow():
    time.sleep(float(request.args.get("seconds", 1)))           # pretend to wait on a DB / model
    return jsonify(pid=os.getpid(), slept=True)
```
- **In simple words:** Two tiny test endpoints: one says which process answered, one pretends to wait.
- **Exactly:** `os.getpid()` is the OS process id (e.g. 20772). `request.args.get("seconds", 1)` reads `?seconds=` from the query string (always a string, so `float()`). `time.sleep` blocks that worker thread. `/slow?seconds=0.5` took 0.50 s in the real run.
- **Why it is here:** Step 07 uses them to show how workers and threads share (or fail to share) load.

### 18. JSON versions of 404 and 405  (`flask_app.py` lines 70–78)
```python
# ----- JSON error pages instead of Flask's default HTML -----
@app.errorhandler(404)
def not_found(_e):
    return jsonify(error="route not found"), 404


@app.errorhandler(405)
def bad_method(_e):
    return jsonify(error="method not allowed"), 405
```
- **In simple words:** Replace Flask's default web-page errors with JSON.
- **Exactly:** `@app.errorhandler(404)` runs when no route matches (real: `{"error":"route not found"}`); 405 when the route exists but not for that method. The second return value is the status code.
- **Why it is here:** API clients expect JSON; HTML error pages break their parsers.

### 19. Catch-all handler for unexpected errors  (`flask_app.py` lines 81–84)
```python
@app.errorhandler(Exception)                                    # last resort: never leak a stack trace to clients
def unexpected(e):
    app.logger.exception("unhandled error")                     # the traceback goes to OUR logs
    return jsonify(error="internal server error"), 500
```
- **In simple words:** If anything unexpected blows up, log it for us and tell the client only "internal server error".
- **Exactly:** Registered for `Exception`, so it catches any unhandled exception. `app.logger.exception` logs the message plus full traceback. Returns status 500.
- **Why it is here:** Leaking a stack trace exposes internals (a security issue) and is useless to clients.

### 20. Run the dev server when executed directly  (`flask_app.py` lines 87–88)
```python
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001)
```
- **In simple words:** Start the built-in development server if you run `python flask_app.py`.
- **Exactly:** `app.run(host="127.0.0.1", port=5001)`: only local connections, port 5001. This is the Werkzeug dev server, one request at a time by default; production uses Gunicorn.
- **Why it is here:** The guard stops the server starting when `main.py` merely imports `app`.

### 21. FastAPI app: docstring and imports  (`fastapi_app.py` lines 1–18)
```python
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
```
- **In simple words:** Same job as Flask's top of file, but with FastAPI and Pydantic tools.
- **Exactly:** `asyncio` for async sleep, `asynccontextmanager` for the lifespan, `FastAPI`/`HTTPException` for the app and errors, `BaseModel`/`Field` for typed request bodies.
- **Why it is here:** FastAPI gets validation and docs from these type declarations, instead of hand-written checks.

### 22. Lifespan: load the model at startup  (`fastapi_app.py` lines 21–27)
```python
@asynccontextmanager                      # turns this generator into a "startup ... shutdown" context manager
async def lifespan(app: FastAPI):
    # Startup: runs once per worker process before the first request.
    app.state.model, app.state.source = load_model()      # app.state = a place to keep shared objects
    print(f"[pid {os.getpid()}] fastapi_app loaded model from {app.state.source}", flush=True)
    yield                                                   # the app serves requests while paused here
    # Shutdown code would go here (close DB connections, flush logs ...)
```
- **In simple words:** Do the one-time setup before serving, and leave a spot for clean-up at shutdown.
- **Exactly:** `@asynccontextmanager` turns the generator into a context manager. Code before `yield` runs at startup, once per worker process; the app serves while paused at `yield`. `app.state` is a shared object bag. Real run: `[pid 20388] fastapi_app loaded model from fallback-keyword`. Under `TestClient`, this only runs inside `with TestClient(app)`.
- **Why it is here:** Without lifespan the model would be missing on first request (`app.state.model` AttributeError).

### 23. Create the FastAPI app  (`fastapi_app.py` line 30)
```python
app = FastAPI(title="Sentiment API", version="1.0", lifespan=lifespan)
```
- **In simple words:** Build the app, give it a title for the docs page, and attach the startup/shutdown logic.
- **Exactly:** `FastAPI(title=..., version=..., lifespan=lifespan)`; the title and version appear in `/docs` and `/openapi.json`.
- **Why it is here:** This `app` object is what uvicorn imports as `fastapi_app:app`.

### 24. Pydantic models for input and output  (`fastapi_app.py` lines 33–46)
```python
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
```
- **In simple words:** Declare what a request and a response look like; FastAPI checks them automatically.
- **Exactly:** `Field(min_length=1)` rejects empty strings. `ge=1, le=100` bounds `max_new_tokens` (1..100). `PredictResponse` as `response_model` filters and documents the output. A bad body `{"text":5}` returns 422 with `{"detail":[{"type":"string_type","loc":["body","text"],"msg":"Input should be a valid string","input":5}]}`.
- **Why it is here:** This replaces ~10 lines of manual checks in Flask and powers `/docs`.

### 25. GET /health  (`fastapi_app.py` lines 49–52)
```python
@app.get("/health")
def health():
    s = get_settings()
    return {"status": "ok", "model_version": s.model_version, "model_source": app.state.source}
```
- **In simple words:** Same health check as the Flask one.
- **Exactly:** Returns a plain dict; FastAPI serialises it as JSON. Real output: `{"status":"ok","model_version":"0.1.0","model_source":"fallback-keyword"}` (insertion order kept).
- **Why it is here:** Same contract as Flask so clients cannot tell the two apps apart.

### 26. POST /predict with typed request  (`fastapi_app.py` lines 55–63)
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
- **In simple words:** Take a validated request, check the length, run the model, return a typed response.
- **Exactly:** `req: PredictRequest` is parsed and validated before the function runs. Plain `def` means FastAPI runs it in a thread pool, so slow model code does not block the event loop. Too long -> `HTTPException(422)`, rendered as `{"detail": "..."}` (note Flask used `{"error": ...}`). Real: `{"label":"positive","confidence":0.6667,"model_version":"0.1.0","latency_ms":0.02}`.
- **Why it is here:** The length limit still needs manual code because it depends on a runtime setting, not a fixed `Field` value.

### 27. POST /generate, opt-in  (`fastapi_app.py` lines 66–70)
```python
@app.post("/generate")
def generate(req: GenerateRequest):
    if not get_settings().enable_generate:
        raise HTTPException(status_code=503, detail="generation disabled; start with ENABLE_GENERATE=1")
    return {"text": generate_text(req.prompt, req.max_new_tokens)}
```
- **In simple words:** Same opt-in text generation as Flask.
- **Exactly:** Disabled -> `HTTPException(503)`; real probe printed 503 for both apps. `GenerateRequest` already validated prompt and token bounds.
- **Why it is here:** Keeps the optional heavy feature safe by default.

### 28. GET /whoami  (`fastapi_app.py` lines 73–76)
```python
# ----- helper endpoints used by step 07 to explore workers and concurrency -----
@app.get("/whoami")
def whoami():
    return {"pid": os.getpid()}
```
- **In simple words:** Returns the process id.
- **Exactly:** `{"pid": 20388}` in the real run. With `--workers 2` different calls can return different pids.
- **Why it is here:** Proof that multiple worker processes exist.

### 29. GET /slow, async and non-blocking  (`fastapi_app.py` lines 79–82)
```python
@app.get("/slow")                          # async def + await asyncio.sleep: the loop serves OTHER requests meanwhile
async def slow(seconds: float = 1):
    await asyncio.sleep(seconds)
    return {"pid": os.getpid(), "slept": True}
```
- **In simple words:** Wait 1 second without freezing the server for other users.
- **Exactly:** `async def` plus `await asyncio.sleep(seconds)` hands control back to the event loop. `seconds: float = 1` becomes the query parameter `?seconds=` with automatic float conversion. Real: `/slow?seconds=0.5` took 0.50 s and returned `{"pid":20773,"slept":true}`.
- **Why it is here:** Shows the correct way to wait in async code.

### 30. GET /slow-sync, blocking but in a thread  (`fastapi_app.py` lines 85–88)
```python
@app.get("/slow-sync")                     # plain def + time.sleep: runs in a thread, so still concurrent (up to ~40 threads)
def slow_sync(seconds: float = 1):
    time.sleep(seconds)
    return {"pid": os.getpid(), "slept": True}
```
- **In simple words:** Same wait, but written as an ordinary function.
- **Exactly:** Plain `def` runs in Starlette's thread pool (default about 40 threads) so many such requests can overlap.
- **Why it is here:** Contrast: it is fine for blocking work as long as the function is not `async def`.

### 31. GET /slow-blocking, the anti-pattern  (`fastapi_app.py` lines 91–94)
```python
@app.get("/slow-blocking")                 # ANTI-PATTERN: async def but a BLOCKING call -> freezes the whole event loop
async def slow_blocking(seconds: float = 1):
    time.sleep(seconds)
    return {"pid": os.getpid(), "slept": True}
```
- **In simple words:** An async function that secretly blocks the whole server.
- **Exactly:** `async def` runs directly on the event loop; `time.sleep` blocks the loop thread, so no other request is served for those seconds.
- **Why it is here:** Included deliberately so step 07 can measure the damage.

### 32. main.py docstring and imports  (`main.py` lines 1–21)
```python
"""
GOAL:   Prove the Flask app and the FastAPI app give the SAME answers for the SAME requests,
        and see the few places where they differ (error body shape, 405/404 handling).

RUN:    cd 09_api_development/06_model_api_flask_vs_fastapi && python main.py
        With a real model:  MODEL_PATH=/path/to/model.joblib python main.py
        With generation:    ENABLE_GENERATE=1 python main.py     (first run downloads distilgpt2 ~330 MB)

EXPECTED OUTPUT (shape):
        POST /predict great film   flask=positive 0.75   fastapi=positive 0.75
        ...
        bad request statuses ...

NEEDS:  flask, fastapi, scikit-learn+joblib only if MODEL_PATH is set.
"""

from fastapi.testclient import TestClient

from config import get_settings
from fastapi_app import app as fastapi_app
from flask_app import app as flask_app
```
- **In simple words:** The file says what it proves and pulls in both apps plus the test client.
- **Exactly:** `fastapi.testclient.TestClient` calls the app in-process (no network, no port). `flask_app.test_client()` does the same for Flask. Importing `flask_app` and `fastapi_app` loads the model once each (the two `loaded model` lines).
- **Why it is here:** No server needed, so the comparison is fast and deterministic.

### 33. Create clients and call /health and /slow  (`main.py` lines 23–30)
```python
if __name__ == "__main__":
    flask_client = flask_app.test_client()
    # `with TestClient(app)` is needed so FastAPI's lifespan (model loading) runs.
    with TestClient(fastapi_app) as fast_client:
        print("health  flask  =", flask_client.get("/health").get_json())
        print("health  fastapi=", fast_client.get("/health").json())
        print("async: ", fast_client.get("/slow").json())
        print("sync: ", fast_client.get("/slow-sync").json())
```
- **In simple words:** Build one pretend browser for each app and ask each one the simple questions.
- **Exactly:** `with TestClient(...)` runs the lifespan (model loading). Real output: `health  flask  = {'model_source': 'fallback-keyword', 'model_version': '0.1.0', 'status': 'ok'}`, same data from FastAPI, `async:  {'pid': 20388, 'slept': True}`. `.get_json()` is Flask's call, `.json()` is the httpx response method.
- **Why it is here:** Without the `with`, `app.state.model` would not exist and every FastAPI call would fail.

### 34. Compare predictions on three sentences  (`main.py` lines 32–36)
```python
        print()
        for text in ["I love this great film", "This was terrible and boring", "It was a film"]:
            f = flask_client.post("/predict", json={"text": text}).get_json()
            g = fast_client.post("/predict", json={"text": text}).json()
            print(f"{text!r:<32} flask={f['label']} {f['confidence']}   fastapi={g['label']} {g['confidence']}")
```
- **In simple words:** Send the same sentences to both apps and print both answers side by side.
- **Exactly:** Real output: `'I love this great film' flask=positive 0.75 fastapi=positive 0.75`; `'This was terrible and boring'` -> negative 0.75; `'It was a film'` -> negative 0.5 (tie breaks to the first max, index 0 = negative). `{text!r:<32}` pads the repr to 32 chars.
- **Why it is here:** This is the proof that two frameworks give identical answers.

### 35. Compare status codes for bad inputs  (`main.py` lines 37–43)
```python
        print()
        print("bad requests (Flask status / FastAPI status):")
        bad = [("missing text", {}), ("empty text", {"text": ""}), ("number", {"text": 5})]
        for name, body in bad:
            fs = flask_client.post("/predict", json=body).status_code
            gs = fast_client.post("/predict", json=body).status_code
            print(f"  {name:<14}{fs} / {gs}")
```
- **In simple words:** Send broken requests to both and compare the status numbers.
- **Exactly:** Real output: `missing text 422 / 422`, `empty text 422 / 422`, `number 422 / 422`. Note Flask's `{}` body is a dict with no text -> 422, but a non-JSON body would be 400.
- **Why it is here:** Status codes are the contract; bodies differ (`error` vs `detail`) but codes agree.

### 36. Wrong method, unknown route, generate probe  (`main.py` lines 44–47)
```python
        print("  wrong method  ", flask_client.get("/predict").status_code, "/", fast_client.get("/predict").status_code)
        print("  unknown route ", flask_client.get("/nope").status_code, "/", fast_client.get("/nope").status_code)
        print("  generate probe ", flask_client.post("/generate", json={"prompt": "Hi"}).status_code, "/",
              fast_client.post("/generate", json={"prompt": "Hi"}).status_code)
```
- **In simple words:** Poke at three more edge cases.
- **Exactly:** Real output: `wrong method 405 / 405`, `unknown route 404 / 404`, `generate probe 503 / 503`.
- **Why it is here:** Proves both frameworks follow HTTP semantics the same way.

### 37. Optional generation and TRY IT notes  (`main.py` lines 48–51)
```python
        if get_settings().enable_generate:
            print("generate:", fast_client.post("/generate", json={"prompt": "The cat", "max_new_tokens": 8}).json())
    # TRY IT: run again with MODEL_VERSION=2.0.0 and watch model_version change in /health.
    # TRY IT: MAX_TEXT_LENGTH=5 python main.py  -> every real sentence now fails with 422.
```
- **In simple words:** Only if you switched generation on, run a real generation call; then suggestions to experiment.
- **Exactly:** `get_settings().enable_generate` is False by default, so this block did not run. With `ENABLE_GENERATE=1` it would download distilgpt2.
- **Why it is here:** Keeps the default run fast and offline.

### 38. Docstring, imports and default model name  (`gpt2_loader.py` lines 1–56)
```python
"""
gpt2_loader.py -- load GPT-2 ONCE and generate text from it; built to sit behind an API.

It is the LLM twin of model_loader.py:
    model_loader.py   -> a small scikit-learn classifier  (predict_one)
    gpt2_loader.py    -> a real language model, GPT-2      (generate)
Same ideas: load at start-up (not per request), keep the model in memory, wrap it in one
tidy function the web framework calls, validate input, report timing.

RUN THE SELF-TEST (first run downloads GPT-2, about 500 MB, into ~/.cache/huggingface):
    python gpt2_loader.py                      (from this folder)
    python 09_api_development/06_model_api_flask_vs_fastapi/gpt2_loader.py     (from the repo root)

USE IT FROM FASTAPI (sketch -- the same two lines work in Flask):
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    from gpt2_loader import get_generator

    @asynccontextmanager
    async def lifespan(app):
        get_generator()                  # load GPT-2 at start-up so the first request is not slow
        yield

    app = FastAPI(lifespan=lifespan)

    class GenerateIn(BaseModel):
        prompt: str = Field(min_length=1)
        max_new_tokens: int = Field(default=30, ge=1, le=200)
        temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    @app.post("/generate")
    def generate(body: GenerateIn):      # plain `def`: FastAPI runs it in a thread pool
        try:
            return get_generator().generate(**body.model_dump())
        except ValueError as err:        # e.g. prompt too long for GPT-2's 1024-token window
            raise HTTPException(status_code=422, detail=str(err))

SETTINGS (environment variables, read when the generator is created):
    GPT2_MODEL_NAME   default "gpt2"   (also works: "distilgpt2" smaller/faster, "gpt2-medium" bigger)
    GPT2_DEVICE       default "cpu"    ("cuda" if you have an NVIDIA GPU, "mps" on Apple silicon)

CONCEPTS USED (details: 04_llm_fundamentals, 05_llm_parameters, 06_llm_training_and_lora):
    tokenizer, logits, sampling (temperature / top-k / top-p / repetition penalty),
    context window, KV cache (generate() uses it automatically), inference mode, lazy singleton.
"""

import os
import threading
import time

import torch                                     # 3rd-party: the tensor / neural-network library GPT-2 runs on
from transformers import AutoModelForCausalLM, AutoTokenizer   # 3rd-party: Hugging Face model + tokenizer loaders

DEFAULT_MODEL_NAME = "gpt2"                      # GPT-2 small: 124 million parameters, 1024-token context window

```
- **In simple words:** Long usage notes, then imports of `torch` and Hugging Face loaders.
- **Exactly:** The docstring shows a FastAPI usage sketch, env settings `GPT2_MODEL_NAME`/`GPT2_DEVICE`, and the concepts used. `DEFAULT_MODEL_NAME="gpt2"` = 124M parameters, 1024-token window. This file is read, not run (needs a ~500 MB download).
- **Why it is here:** It is the LLM twin of `model_loader.py`: same pattern of load once, wrap in one function.

### 39. GPT2Generator.__init__: load tokenizer and model once  (`gpt2_loader.py` lines 58–83)
```python
class GPT2Generator:
    """Owns the tokenizer + model. Create ONE per process and reuse it for every request."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str = "cpu"):
        self.model_name = model_name
        self.device = device

        # Tokenizer: text <-> token ids. It is tiny and fast to load.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # GPT-2 was trained without a padding token. generate() wants one defined, so we reuse
        # the end-of-text token. (With a single prompt per call nothing is actually padded.)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # The model: ~124M weights. from_pretrained downloads once, then reads from the local cache.
        # "CausalLM" = the head that predicts the NEXT token, which is what text generation needs.
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)                    # move the weights to cpu / cuda / mps
        self.model.eval()                        # inference mode: switches off dropout (randomness used for training)

        # GPT-2's maximum number of tokens (prompt + generated). 1024 for every GPT-2 size.
        self.max_context = self.model.config.n_positions

        # model.generate() on one shared model object is not designed for simultaneous calls from
        # several threads (FastAPI runs plain `def` endpoints in a thread pool), so serialise them.
        self._lock = threading.Lock()
```
- **In simple words:** Load the text-to-numbers converter and the model weights, and prepare a lock.
- **Exactly:** `AutoTokenizer.from_pretrained` loads the vocabulary; `pad_token = eos_token` because GPT-2 has no padding token; `AutoModelForCausalLM` is the next-token-prediction head; `.to(device)` moves weights; `.eval()` disables dropout; `n_positions` = 1024; `threading.Lock()` serialises generation.
- **Why it is here:** Loading per request would take seconds; the lock protects a shared model from concurrent thread calls.

### 40. info and count_tokens helpers  (`gpt2_loader.py` lines 85–99)
```python
    # ------------------------------------------------------------------ info
    def info(self) -> dict:
        """Cheap facts about the loaded model; handy for a /health or /model-info endpoint."""
        n_params = sum(p.numel() for p in self.model.parameters())   # numel() = number of values in a tensor
        return {
            "model_name": self.model_name,
            "device": self.device,
            "parameters": n_params,
            "approx_weights_mb": round(n_params * 4 / 1e6),          # float32 = 4 bytes per parameter
            "max_context_tokens": self.max_context,
        }

    def count_tokens(self, text: str) -> int:
        """How many GPT-2 tokens does `text` use? (Prompt cost and the context limit are counted in tokens.)"""
        return len(self.tokenizer.encode(text))
```
- **In simple words:** Two cheap look-up methods: model facts and token counting.
- **Exactly:** `numel()` = number of values in each parameter tensor; weights MB approx `params*4/1e6` (float32 is 4 bytes). `tokenizer.encode(text)` returns a list of token ids; `len` is the token count.
- **Why it is here:** Token counts, not characters, decide cost and context limit.

### 41. generate: validate input and check the context window  (`gpt2_loader.py` lines 101–129)
```python
    # -------------------------------------------------------------- generate
    def generate(self, prompt: str, max_new_tokens: int = 30, temperature: float = 0.0,
                 top_k: int = 50, top_p: float = 1.0, repetition_penalty: float = 1.0,
                 seed: int | None = None) -> dict:
        """Continue `prompt` and return a dict with the NEW text only (not the prompt repeated).

        temperature         0 -> greedy (always the most likely token, deterministic). >0 -> random sampling.
        top_k, top_p        keep only the k most likely tokens / the smallest set with total probability p.
        repetition_penalty  >1.0 discourages repeating tokens already seen (1.0 = off).
        seed                makes sampling repeatable (ignored when temperature == 0).
        Raises ValueError for bad input so the API layer can answer 422 instead of crashing with a 500.
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")

        start = time.perf_counter()

        # 1) text -> token ids.  return_tensors="pt" gives PyTorch tensors; shape (1, n_tokens).
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        # 2) Context-window check: prompt + new tokens must fit in 1024, otherwise GPT-2 errors or degrades.
        if prompt_tokens + max_new_tokens > self.max_context:
            raise ValueError(
                f"prompt uses {prompt_tokens} tokens + max_new_tokens {max_new_tokens} "
                f"> context window {self.max_context}"
            )
```
- **In simple words:** Reject bad prompts early and make sure the prompt plus the new text fits.
- **Exactly:** `inputs = tokenizer(prompt, return_tensors="pt")` gives tensors of shape `(1, n_tokens)`; `prompt_tokens = input_ids.shape[1]`. If `prompt_tokens + max_new_tokens > 1024` raise `ValueError`, which an API maps to 422.
- **Why it is here:** Without the check GPT-2 errors or degrades; a ValueError lets the API answer 422 rather than 500.

### 42. generate: build settings, run, slice, return  (`gpt2_loader.py` lines 131–162)
```python
        # 3) Build the generation settings. Greedy and sampling need different arguments:
        #    passing temperature/top_k/top_p while do_sample=False only triggers warnings.
        do_sample = temperature > 0
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_k=top_k, top_p=top_p)

        # 4) Generate. The lock keeps threads from using the model at the same time.
        with self._lock:
            if do_sample and seed is not None:
                torch.manual_seed(seed)          # global RNG, so it must happen inside the lock too
            with torch.inference_mode():         # no gradient bookkeeping: faster and uses less memory
                output_ids = self.model.generate(**inputs, **gen_kwargs)

        # 5) output_ids = prompt tokens + new tokens. Slice off the prompt, then ids -> text.
        new_ids = output_ids[0][prompt_tokens:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)

        elapsed = time.perf_counter() - start
        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "new_tokens": int(new_ids.shape[0]),
            "latency_ms": round(elapsed * 1000, 1),
            "tokens_per_second": round(new_ids.shape[0] / elapsed, 1) if elapsed > 0 else None,
            "model_name": self.model_name,
        }
```
- **In simple words:** Pick greedy or random sampling, run the model under a lock, remove the prompt from the output, report timing.
- **Exactly:** `do_sample = temperature > 0`; only then are `temperature/top_k/top_p` passed. `torch.manual_seed(seed)` makes sampling repeatable. `torch.inference_mode()` disables gradient tracking. `output_ids` has shape `(1, prompt_tokens + new)`; `output_ids[0][prompt_tokens:]` keeps only new tokens; `decode(..., skip_special_tokens=True)` gives text. Returns text, token counts, latency and tokens/second.
- **Why it is here:** Greedy plus warnings-free kwargs, a lock and inference mode make the endpoint correct, thread-safe and fast.

### 43. Thread-safe singleton get_generator  (`gpt2_loader.py` lines 165–186)
```python
# ---------------------------------------------------------------------------
# Singleton: one generator per process, created the first time it is asked for.
# ---------------------------------------------------------------------------
_GENERATOR: GPT2Generator | None = None
_CREATE_LOCK = threading.Lock()                  # two simultaneous first requests must not load the model twice


def get_generator() -> GPT2Generator:
    """Return the process-wide GPT2Generator, creating (and loading GPT-2) on first call.

    Call it once at start-up (lifespan / before_first_request equivalent) to avoid a slow first request.
    Remember: with Gunicorn/Uvicorn --workers N, EACH worker process loads its own copy (~500 MB for GPT-2).
    """
    global _GENERATOR
    if _GENERATOR is None:
        with _CREATE_LOCK:
            if _GENERATOR is None:               # check again: another thread may have finished while we waited
                _GENERATOR = GPT2Generator(
                    model_name=os.getenv("GPT2_MODEL_NAME", DEFAULT_MODEL_NAME),
                    device=os.getenv("GPT2_DEVICE", "cpu"),
                )
    return _GENERATOR
```
- **In simple words:** Give everyone the same single generator, creating it the first time only.
- **Exactly:** Double-checked locking: test `_GENERATOR is None`, take `_CREATE_LOCK`, test again. Env vars choose model name and device (default `gpt2`, `cpu`). Each worker process holds its own copy (~500 MB).
- **Why it is here:** Two simultaneous first requests must not load the model twice.

### 44. Self-test when run directly  (`gpt2_loader.py` lines 189–214)
```python
# ---------------------------------------------------------------------------
# Self-test: python gpt2_loader.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.perf_counter()
    gen = get_generator()
    print(f"Loaded in {time.perf_counter() - t0:.1f}s ->", gen.info())

    prompt = "The best way to learn machine learning is"
    print("\nPrompt tokens:", gen.count_tokens(prompt))

    # TRY IT: change one setting at a time and compare the outputs.
    print("\n[greedy, temperature=0]  (run it twice: identical output)")
    print(gen.generate(prompt, max_new_tokens=25))

    print("\n[sampling, temperature=0.8, seed=42]  (same seed -> same output)")
    print(gen.generate(prompt, max_new_tokens=25, temperature=0.8, seed=42))

    print("\n[sampling, temperature=1.5, seed=42]  (hotter -> wilder text)")
    print(gen.generate(prompt, max_new_tokens=25, temperature=1.5, seed=42))

    # BREAK IT: this 8-token prompt + 1020 new tokens exceeds the 1024-token window -> ValueError (an API would return 422).
    try:
        gen.generate(prompt, max_new_tokens=1020)
    except ValueError as err:
        print("\nExpected error:", err)
```
- **In simple words:** A small demo that loads, counts tokens, generates greedy and sampled text, then provokes the error.
- **Exactly:** Greedy is deterministic; same `seed` gives same sampled text; 8 prompt tokens + 1020 > 1024 raises `ValueError`, printed as `Expected error:`.
- **Why it is here:** Gives a quick way to check the module without a web server.


## Syntax explained
| Syntax | Meaning | Example |
|---|---|---|
| `from model_loader import ...` | Import names from a sibling file | — |
| `@asynccontextmanager` + `yield` | Code before `yield` = startup, after = shutdown | lifespan in `fastapi_app.py` |
| `app.state.model = ...` | Attach shared objects to the app | — |
| `response_model=PredictResponse` | Declares/filters the response shape | — |
| `max(range(n), key=probs.__getitem__)` | "index of the largest item": `key` says what to compare | `max(range(3), key=[5,9,1].__getitem__)` → 1 |
| `@dataclass(frozen=True)` | Auto-generates `__init__`; frozen = immutable | `config.py` |
| `str \| None` | "a string or None" type hint (Python 3.10+) | — |
| `def` vs `async def` | `def` routes run in a thread pool; `async def` runs on the event loop | step 07 |
| `os.getpid()` | Process id — proves which worker answered | — |

## Libraries & functions
| Name | Kind | Does |
|---|---|---|
| `joblib.load(path)` | 3rd-party | Load a saved sklearn pipeline (module 03/10) |
| `model.predict_proba(list_of_texts)` | sklearn | Probability per class per text |
| `model.classes_` | sklearn | Label names in column order |
| `transformers.pipeline("text-generation", model="distilgpt2")` | 3rd-party | One-liner LLM generation (04_llm_fundamentals step 06 load model, step 08 generation loop) |
| `time.perf_counter()` | stdlib | High-resolution timer for latency |
| `dotenv.load_dotenv()` | 3rd-party | `.env` → `os.environ` |

## Execution trace
`POST /predict {"text": "I love this great film"}`:
1. **Input:** JSON body arrives (Flask: `request.get_json`; FastAPI: parsed into `PredictRequest`).
2. **Processing:** validation → length check against `MAX_TEXT_LENGTH` → `predict_one(model, text)` → `predict_proba` → pick the argmax → `(label, confidence)`; time measured.
3. **Output:** `{"label": "positive", "confidence": 0.75, "model_version": "0.1.0", "latency_ms": 0.02}`.
The model object was created **once at startup**, not per request — loading per request would add hundreds of ms.

## Flask vs FastAPI — what actually differs
| | Flask | FastAPI |
|---|---|---|
| Interface | WSGI | ASGI |
| Input validation | by hand | Pydantic from type hints |
| Bad-input status | whatever you choose (we used 400/422) | 422 automatically |
| Error body | your own `{"error": ...}` | `{"detail": ...}` |
| Docs | none built in | `/docs`, `/openapi.json` |
| Model loading | module level (runs per worker) | `lifespan` startup (runs per worker) |
| Production server | Gunicorn | Uvicorn (or Gunicorn + Uvicorn workers) |

## Experiments
1. `MODEL_VERSION=2.0.0 python main.py` → every response reports 2.0.0.
2. `MAX_TEXT_LENGTH=5 python main.py` → real sentences get 422.
3. Point `MODEL_PATH` at a wrong file → `FileNotFoundError` at **import/startup** — fail fast is good: you find out at deploy, not at the first user request.

## Debugging practice
`MODEL_PATH=/nope/model.joblib python flask_app.py` → `FileNotFoundError: [Errno 2] No such file or directory: '/nope/model.joblib'` and a traceback through `load_model` → `joblib.load`. The first frame in *your* code tells you where; the last line says why. Fix the path (check with `ls`, remember relative paths depend on the **current working directory**).

## Rebuild checklist
- [ ] `/health` and `/predict` in Flask from memory.
- [ ] Same in FastAPI with a Pydantic model.
- [ ] Explain why the model is loaded at startup and what "per worker" implies.

## The new problem (cliffhanger)
It works for one caller. What happens with ten at once, and how many copies of the model fit in memory? Next: [07_servers_workers_concurrency](../07_servers_workers_concurrency/).

## Say it aloud
> **Say it aloud (interview box):** I load the model once at startup into the process, validate input, run `predict_proba`, and return label, confidence and model version. Flask and FastAPI expose the same contract, and `MODEL_PATH` lets the same code serve a different model. Each worker process holds its own copy of the model, so memory grows with the worker count.
