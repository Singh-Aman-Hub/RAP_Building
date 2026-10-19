# 13 — Capstone: the whole stack, explained end to end

**Why it matters:** the best interview story is "I built it, containerised it and can explain every arrow." This module wires earlier work into one running system.

## Where this fits
**Builds on:** [03 sklearn model + joblib](../03_ml_foundations/08_sklearn_text_classifier/) (the artifact in `api/artifacts/`), [09 FastAPI app](../09_api_development/06_model_api_flask_vs_fastapi/) (copied to `api/`), [12 Streamlit client](../12_streamlit_gradio/03_streamlit_api_client/) (copied to `ui/`), [11 Dockerfile/Compose patterns](../11_docker/04_compose_multi_container/), [10 architecture](../10_ml_system_architecture/) (what you must be able to draw).
Files are *copied* so each Docker build context is self-contained; the originals remain the teaching versions.

## Architecture
```
Browser ──► ui (Streamlit :8501) ──HTTP POST /predict──► api (FastAPI+Uvicorn :8000) ──► model.joblib (loaded once at startup)
                  API_URL=http://api:8000                  Pydantic validation (422) · /health · /docs
                  (Compose service-name DNS)               MODEL_PATH env var chooses the artifact
```
Request path: user text → Streamlit callback → `httpx.post` → Pydantic validates → `predict_one` (preprocess → model → postprocess) → JSON `{label, confidence, model_version, latency_ms}` → UI metric.

## Run it
Without Docker (uses the repo venv):
```bash
./run_local.sh            # API on :8000 (docs at /docs), UI on :8501
```
With Docker (needs the daemon running):
```bash
docker compose up --build
curl localhost:8000/health
docker compose logs -f api
docker compose down
```
Verified: the API (health, predict, 422 on empty text) via FastAPI TestClient. **Not verified:** the Docker build/compose run (daemon was not running when this was written) — if it fails, that is a debugging drill: read `docker compose logs`, then see [11_docker step 05](../11_docker/05_healthchecks_logs_debugging/).

## Tasks (do these, in order)
1. Run locally; send a request from `/docs`, from `curl`, and from the UI. Say which component each request touched.
2. Retrain with your own examples (`03.../train.py`), copy the new artifact into `api/artifacts/`, bump `MODEL_VERSION`, confirm `/health` shows it. (Module 10: versioning.)
3. Add request logging (one JSON line per prediction) — see [10 step 07](../10_ml_system_architecture/07_logging_and_monitoring/).
4. Enable `/generate` with `ENABLE_GENERATE=1` (distilgpt2, needs `transformers`) and add temperature/top-p fields (module 05).
5. Stretch: expose the module-08 agent as `/agent` with a max-step limit and tool allow-list.
6. Break it three ways (wrong `API_URL`, bad `MODEL_PATH`, bind to 127.0.0.1) and diagnose each from logs only.

## Explain-aloud script (5 minutes)
1. What the system does. 2. Draw the diagram above. 3. Walk one request. 4. Where validation, config and the model load happen. 5. What breaks when requests grow (workers each load the model; batching; replicas — modules 09/10). 6. One trade-off: the toy model is fast but weak; swapping in an LLM raises memory and latency (modules 04–06).

## Interview questions
- *Why does the UI call the API instead of loading the model?* Separation of concerns: one model copy, scalable independently, other clients can reuse it.
- *Why `depends_on` with a healthcheck?* "Started" is not "ready".
- *Why is the model loaded at startup, not per request?* Loading costs seconds; per-request loading destroys latency.
- *Why a non-root user in the Dockerfile?* Limits damage if the app is compromised.
