# 13 — Capstone: the whole stack, explained end to end

**Why it matters:** the best interview story is "I built it, containerised it and can explain every arrow." This module wires earlier work into one running system.

## The story of this module
**Chapter 25, The Grand Synthesis.** Pain: you now know 24 chapters of pieces (arrays, training, tokens, models, prompts, an API, a blueprint, a container, a UI) and an interviewer asks "walk me through something you built". Pieces do not answer that, a running system does. Spark: wire them into one small stack you can start with one command and explain arrow by arrow. What you build: a Streamlit UI that calls a FastAPI service, which loads a trained model once, validates input, reports its version and latency, and ships as two containers under Compose. There is no further cliffhanger. The next step is practice: the [mock interviews](../mock_interviews/) make you say all of this aloud.

## Where this fits
**Builds on:** [03 sklearn model + joblib](../03_ml_foundations/08_sklearn_text_classifier/) (the artifact in `api/artifacts/`), [09 FastAPI app](../09_api_development/06_model_api_flask_vs_fastapi/) (copied to `api/`), [12 Streamlit client](../12_streamlit_gradio/03_streamlit_api_client/) (copied to `ui/`), [11 Dockerfile/Compose patterns](../11_docker/04_compose_multi_container/), [10 architecture](../10_ml_system_architecture/) (what you must be able to draw).
Files are *copied* so each Docker build context is self-contained; the originals remain the teaching versions.

## Chapter-by-chapter map (Ch 1-24 -> where it shows up here)
Only files that exist in this folder are listed. "Not built here" means the chapter is background you must be able to discuss; the swap point is named.

| Ch | Chapter | Where it shows up in the capstone |
|---|---|---|
| 1 | Language of Space (NumPy) | `api/model_loader.py`: `predict_proba` returns rows of probabilities, `max(range(...))` picks the best column |
| 2 | Art of Learning (gradient descent) | `api/artifacts/model.joblib`: the logistic-regression weights were learned this way (trained in [03](../03_ml_foundations/08_sklearn_text_classifier/)) |
| 3 | Modern Engine (autograd) | Not built here (sklearn trains the model); same idea, different library |
| 4 | Overfitting Dilemma | `api/artifacts/metadata.json`: `n_train`, `n_val`, `val_accuracy` (held-out score, not training score) |
| 5 | Decisions and Confidence | `predict_one` returns `(label, confidence)`; the `confidence` field in the JSON response |
| 6 | Text Baseline (TF-IDF) | `model.joblib` is a TF-IDF + classifier pipeline; its limits are why Ch 7-13 exist |
| 7 | Bridge of Babylon (tokenization) | `generate_text` in `api/model_loader.py`: the `transformers` pipeline tokenizes the prompt (opt-in `/generate`) |
| 8 | Giving Words Geometry (embeddings) | Inside the distilgpt2 behind `/generate`; the TF-IDF model has no embeddings (that is the swap point) |
| 9 | Great Council of Tokens (attention) | Inside distilgpt2 behind `/generate`; not re-implemented here |
| 10 | Assembling the Beast (transformer, load model) | `load_model()` / `generate_text()` load a pretrained model once and cache it (`_GENERATOR`) |
| 11 | Oracle Speaks (forward pass, logits) | `predict_proba` is the sklearn analogue; for `/generate`, logits happen inside the pipeline call |
| 12 | Endless Loop (generation, KV cache) | `/generate` in `api/fastapi_app.py` with `max_new_tokens` (greedy, `do_sample=False`) |
| 13 | Life Cycle (pretrain / fine-tune / inference) | The capstone is inference only; `metadata.json` records the training run, `MODEL_VERSION` records which model answers |
| 14 | Rolling the Dice (sampling) | Task 4: add temperature/top-p fields to `/generate` (greedy today) |
| 15 | Squeezing into VRAM (quantization) | Not built here; swap point is `load_model()` / `generate_text()` |
| 16 | Surgical Fine-Tune (LoRA) | Not built here; a tuned model would be a new artifact in `api/artifacts/` plus a new `MODEL_VERSION` |
| 17 | Production Speed (batching, vLLM) | Not built here; talking point for "what breaks when requests grow" (see the explain-aloud script) |
| 18 | Art of Instruction (prompts, JSON) | Not built here; the Pydantic `GenerateRequest` is the input contract a prompt template would sit behind |
| 19 | Memory Externalizer (RAG) | Not built here |
| 20 | Hands for the Brain (agents) | Task 5 (stretch): expose the module-08 agent as `/agent` |
| 21 | Castle Gate (FastAPI) | `api/fastapi_app.py`: lifespan loads the model once, Pydantic models (422), `/health`, `/docs`. Streaming: [09 step 10, SSE](../09_api_development/10_streaming_sse/) is the pattern to add for token-by-token `/generate` (not in this app yet) |
| 22 | Industrial Blueprint | `api/config.py` (env-var settings), `model_version` and `latency_ms` in every response, `metadata.json` (artifact versioning) |
| 23 | Shipping Container (Docker, Compose) | `api/Dockerfile` (slim, cached deps layer, non-root `appuser`, 0.0.0.0), `ui/Dockerfile`, `compose.yaml` (health-gated `depends_on`, service-name DNS) |
| 24 | Face of AI (Streamlit) | `ui/app.py`: thin client with `API_URL` env var, calls `POST /predict` |

## Architecture
```
Browser ──► ui (Streamlit :8501) ──HTTP POST /predict──► api (FastAPI+Uvicorn :8000) ──► model.joblib (loaded once at startup)
                  API_URL=http://api:8000                  Pydantic validation (422) · /health · /docs
                  (Compose service-name DNS)               MODEL_PATH env var chooses the artifact
```
Request path: user text → Streamlit callback → `requests.post` → Pydantic validates → `predict_one` (preprocess → model → postprocess) → JSON `{label, confidence, model_version, latency_ms}` → UI metric.

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

> **Say it aloud (interview box):** I built a sentiment service where a Streamlit UI calls a FastAPI API that loads a trained scikit-learn model once at startup, validates input with Pydantic and returns the label, confidence, model version and latency. Configuration comes from environment variables, and both services ship as non-root containers that Compose starts in health-checked order, with the UI reaching the API by service name. The trade-off is that the toy model is fast but weak, and swapping in an LLM adds memory, latency and the need for batching and streaming.

## Explain-aloud script (5 minutes)
1. What the system does. 2. Draw the diagram above. 3. Walk one request. 4. Where validation, config and the model load happen. 5. What breaks when requests grow (workers each load the model; batching; replicas — modules 09/10). 6. One trade-off: the toy model is fast but weak; swapping in an LLM raises memory and latency (modules 04–06).

## Interview questions
- *Why does the UI call the API instead of loading the model?* Separation of concerns: one model copy, scalable independently, other clients can reuse it.
- *Why `depends_on` with a healthcheck?* "Started" is not "ready".
- *Why is the model loaded at startup, not per request?* Loading costs seconds; per-request loading destroys latency.
- *Why a non-root user in the Dockerfile?* Limits damage if the app is compromised.
