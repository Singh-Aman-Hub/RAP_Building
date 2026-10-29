# 08: FastAPI Inference Service

**Level:** Intermediate · **Intentional bugs:** 4 · **Dependencies:** `fastapi`, `httpx` (for `TestClient`), `sentence-transformers`

## Scenario
Your team owns a tiny embedding service that other teams call before writing to their vector DB. The API contract is in the module docstring and **is not up for negotiation**: client teams already depend on it. After a refactor, the client teams report:
- *"Every call returns 422, but we're sending exactly what the contract says."*
- *"When it does work, it's slow, way slower than embedding locally."*
- *"Sending an empty list crashes your server with a 500 error."*
- The `/health` endpoint says the model isn't loaded, even while requests are being served.

The interviewer gives you `buggy_code.py`. Its `__main__` block is the client team's smoke test, run through FastAPI's `TestClient`.

## Expected behaviour
Running `python buggy_code.py` prints five calls, with output like this (timings vary by machine):
```
GET  /health  -> 200   ...   {"status":"ok","model_loaded":true}
POST /embed   -> 200   ...   {"model":"sentence-transformers/all-MiniLM-L6-v2","dim":384,"embeddings":[[...
POST /embed   -> 200   ...   (same body)
POST /embed   -> 422   ...   {"detail":[...]}            <- empty list rejected by validation, not a crash
GET  /health  -> 200   ...   {"status":"ok","model_loaded":true}
```
- `model_loaded` is `true` **before** the first `/embed` call.
- `/embed` responds with JSON matching the contract: `dim` is 384, `embeddings` is a list of 2 lists of 384 floats.
- The model is loaded **once per process**, not once per request. Repeated calls take a small fraction of the first model load. On a laptop CPU, a warm call is tens of milliseconds, not hundreds.
- An empty `texts` list returns **422** with a validation message. It must never reach the model.

## Observed behaviour
```
GET  /health  -> 200      1.9 ms   {"status":"ok","model_loaded":false}
POST /embed   -> 422      1.8 ms   {"detail":[{"type":"missing","loc":["body","text"],"msg":"Field required","input":{"texts":[...]}}]}
POST /embed   -> 422      1.1 ms   ...
POST /embed   -> 422      1.2 ms   ...
GET  /health  -> 200      1.1 ms   {"status":"ok","model_loaded":false}
```
Once the first problem is fixed, the next call returns `500 Internal Server Error`. Because `raise_server_exceptions=False`, you'll need to find the real exception yourself.

## Your task
Fix all 4 bugs. The fix must follow the contract, not change it.

## Constraints
- Don't change the requests in `__main__`. They are the client team's code.
- Keep using Pydantic for request validation. Don't validate by hand with `if` statements inside the endpoint.

## Interview follow-ups
- Where do you load a model in a FastAPI app? Compare module level, a startup/lifespan hook, and a dependency.
- Why can't FastAPI serialise a numpy array, and where else (Qdrant payloads, `json.dumps`, logging) does the same issue appear?
- This endpoint is `def`, not `async def`. If it were `async def` and called `model.encode` directly, what would happen under concurrent load?
- With `gunicorn -w 4`, how many copies of the model are in memory?
