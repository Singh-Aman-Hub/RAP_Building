# Step 06 — The same model API in Flask and FastAPI

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
| `transformers.pipeline("text-generation", model="distilgpt2")` | 3rd-party | One-liner LLM generation (module 04) |
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

## Say it aloud
"I load the model once at startup into the process, validate input, run `predict_proba`, and return label, confidence and the model version. Flask and FastAPI expose the same contract; FastAPI adds Pydantic validation and automatic docs. Configuration comes from environment variables so the same code serves a different model by changing `MODEL_PATH`. Trade-off: each worker process holds its own copy of the model, so memory grows with the worker count."
