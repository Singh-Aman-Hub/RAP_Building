# Module 09 — Exercises

Rules: change **one** thing at a time, predict the result first, then run. Worked answers are in [solutions/](solutions/).

## Step 01 — HTTP basics
- **Modify:** add a `GET /time` route to the stdlib server returning the current time as JSON.
- **Break & read:** remove `.encode("utf-8")` in `_send_json`; read the `TypeError`.
- **Rebuild from memory:** write a handler that answers `GET /ping` with `{"pong": true}` and 404 otherwise.
- **Explain aloud:** the four parts of an HTTP request and of a response.

## Step 02 — WSGI/ASGI
- **Modify:** make `wsgi_app` return 404 for any path except `/hi`.
- **Break & read:** delete the `await` before `app(...)` in `call_asgi`.
- **Rebuild:** write the smallest ASGI app returning `"pong"` and run it with uvicorn; `curl` it.
- **Explain aloud:** who calls `start_response`/`send`, and why ASGI can do WebSockets.

## Step 03 — Flask
- **Modify:** add `DELETE /items/<id>` returning 204 (and 404 if missing).
- **Break & read:** remove `global _next_id`; read the `UnboundLocalError`.
- **Rebuild:** `/health` + `POST /echo` returning the posted JSON, from a blank file.
- **Explain aloud:** what `@app.get(...)` does; difference between `request.args` and `request.get_json()`.

## Step 04 — FastAPI
- **Modify:** add an optional `price: float = 0` to `ItemIn`; send with and without it.
- **Break & read:** call `/add?a=x&b=3`; read `loc` and `msg` in the 422 body.
- **Rebuild:** `/health` and a typed `/square?x=3`, plus open `/docs`.
- **Explain aloud:** how FastAPI decides whether a parameter is path, query or body.

## Step 05 — Validation
- **Modify:** add `label: str = Field(pattern="^(a|b)$")`; test bad values.
- **Break & read:** `min_length="1"` (import-time error vs request-time error).
- **Rebuild:** a Pydantic model for `{"text": str 1..200, "top_k": int 1..5}` from memory.
- **Explain aloud:** 400 vs 422; 4xx vs 5xx.

## Step 06 — Model API
- **Modify:** add a `GET /labels` endpoint returning `model.classes_` in both apps.
- **Modify (harder):** add `"top_k"` support: return all labels with probabilities when `top_k > 1`.
- **Break & read:** `MODEL_PATH=/nope/model.joblib python main.py`; read the `FileNotFoundError` traceback.
- **Swap:** run with the module-03 artifact, then with module-10's artifact; compare confidences for the same sentence.
- **Rebuild:** `/health` + `/predict` in FastAPI, from scratch, using a stub model class.
- **Explain aloud:** why the model is loaded once, and what changes with 4 workers.

## Step 07 — Servers
- **Modify:** change `n=4` to `n=8` in `timed_parallel`; write predictions *before* running.
- **Break & read:** start a second server on a busy port (`Address already in use`); find the owner with `lsof -i :8123`.
- **Rebuild:** type the Gunicorn and Uvicorn commands for the step-06 apps without looking.
- **Explain aloud:** why `async def` + `time.sleep` takes 4 s and `def` takes 1 s.

## Step 08 — Config
- **Modify:** add a required `API_KEY` setting that raises a clear error if missing.
- **Break & read:** `MAX_TEXT_LENGTH=abc`; improve the error message.
- **Rebuild:** a `Settings` dataclass reading three env vars with defaults.
- **Explain aloud:** precedence of env var vs `.env`; why secrets are not committed.

## Step 09 — Testing
- **Modify:** add a test that `/health` reports the `MODEL_VERSION` env var (use `monkeypatch`).
- **Break & read:** run `pytest` from a different folder after deleting `conftest.py`.
- **Rebuild:** a parametrized test for three bad inputs.
- **Explain aloud:** unit vs smoke vs load tests.

## Capstone-style challenge
Add `POST /predict_batch` taking `{"texts": [...]}` (max 32) and returning a list of results using **one** `predict_proba` call. Compare the time of 32 single calls vs one batch call (connects to [10 step 06](../10_ml_system_architecture/06_batch_vs_realtime/)).

> Worked solutions are provided in `solutions/` for the larger coding tasks (DELETE route, /labels, /predict_batch). The rest are short enough to verify by running your own change.
