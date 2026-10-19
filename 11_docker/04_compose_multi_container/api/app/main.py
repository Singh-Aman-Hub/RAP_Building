"""
GOAL: Toy sentiment API that counts predictions in REDIS (a second container) - the classic multi-container setup.

RUN locally without redis (falls back to an in-memory counter):
    cd 11_docker/04_compose_multi_container/api && ../../../.venv/bin/python -m uvicorn app.main:app --port 8000
RUN for real:  docker compose up --build   (from 04_compose_multi_container)
EXPECTED OUTPUT: curl localhost:8000/stats -> {"predictions":N,"counter_backend":"redis"|"memory"}
NEEDS: fastapi, uvicorn; `redis` package only when REDIS_HOST is set (listed in api/requirements.txt).
"""
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Toy API + Redis")
POSITIVE = {"good", "great", "love", "excellent", "happy"}
NEGATIVE = {"bad", "awful", "hate", "terrible", "sad"}

# In Compose, the redis service is reachable at the HOSTNAME "redis" (= the service name). That is
# Docker's built-in DNS for user-defined networks. We pass it in via an env var so the code stays portable.
REDIS_HOST = os.getenv("REDIS_HOST")
_memory_counter = 0
_redis = None
if REDIS_HOST:
    import redis                                   # 3rd-party client library, imported only when needed
    _redis = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)  # decode_responses: bytes -> str


class PredictRequest(BaseModel):
    text: str


def incr() -> int:
    global _memory_counter                          # `global`: we assign to a module-level variable
    if _redis is not None:
        return _redis.incr("predictions")           # atomic +1 stored inside redis
    _memory_counter += 1
    return _memory_counter


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness = 'can I do real work?'. Differs from /health ('is the process alive?')."""
    if _redis is not None:
        try:
            _redis.ping()
        except Exception as exc:                    # broad except is OK here: any failure = not ready
            raise HTTPException(status_code=503, detail=f"redis unavailable: {exc}")
    return {"status": "ready"}


@app.post("/predict")
def predict(req: PredictRequest):
    words = req.text.lower().split()
    score = sum(w in POSITIVE for w in words) - sum(w in NEGATIVE for w in words)
    label = "positive" if score > 0 else "negative" if score < 0 else "neutral"
    incr()
    return {"label": label, "score": score}


@app.get("/stats")
def stats():
    n = int(_redis.get("predictions") or 0) if _redis is not None else _memory_counter
    return {"predictions": n, "counter_backend": "redis" if _redis is not None else "memory"}

# TRY IT: `docker compose stop redis` then call /ready -> 503. `docker compose start redis` -> 200 again.
# BREAK IT: set REDIS_HOST=wrongname -> /predict raises redis ConnectionError (read the traceback in `docker compose logs api`).
