"""
GOAL: Same toy API, but configured through ENVIRONMENT VARIABLES and keeping a log file in a DATA folder,
      so we can practise `-e`, `--env-file`, ports, volumes and bind mounts.

RUN (from 11_docker/03_ports_env_volumes):
    MODEL_NAME=my-model-v2 DATA_DIR=./data ../../.venv/bin/python -m uvicorn app.main:app --port 8000
EXPECTED OUTPUT: curl localhost:8000/info -> {"model_name":"my-model-v2","data_dir":"./data","visits":N}
NEEDS: fastapi, uvicorn.
"""
import logging
import os                              # os.getenv reads environment variables (built-in module)
from pathlib import Path               # Path = object-oriented file paths (built-in)

from fastapi import FastAPI
from pydantic import BaseModel

# os.getenv("NAME", default): value of env var NAME, or `default` if it is not set.
MODEL_NAME = os.getenv("MODEL_NAME", "keyword-sentiment-v1")
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)          # make the folder if missing (no error if present)
VISITS_FILE = DATA_DIR / "visits.txt"                # `/` joins paths with Path objects

# Logging to stdout is what Docker expects: `docker logs` shows whatever the process prints.
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("api")

app = FastAPI(title="Toy API with env + volume")
POSITIVE = {"good", "great", "love", "excellent", "happy"}
NEGATIVE = {"bad", "awful", "hate", "terrible", "sad"}


class PredictRequest(BaseModel):
    text: str


def count_visit() -> int:
    """Persist a counter in a file: survives restarts only if DATA_DIR is a volume/bind mount."""
    n = int(VISITS_FILE.read_text()) if VISITS_FILE.exists() else 0
    VISITS_FILE.write_text(str(n + 1))
    return n + 1


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    visits = int(VISITS_FILE.read_text()) if VISITS_FILE.exists() else 0
    return {"model_name": MODEL_NAME, "data_dir": str(DATA_DIR), "visits": visits}


@app.post("/predict")
def predict(req: PredictRequest):
    words = req.text.lower().split()
    score = sum(w in POSITIVE for w in words) - sum(w in NEGATIVE for w in words)
    label = "positive" if score > 0 else "negative" if score < 0 else "neutral"
    n = count_visit()
    log.info("predict #%d model=%s label=%s", n, MODEL_NAME, label)   # visible with `docker logs`
    return {"label": label, "score": score, "model_name": MODEL_NAME}

# TRY IT: change MODEL_NAME and watch /info change without touching code (that is the point of env vars).
# BREAK IT: set DATA_DIR=/nonexistent/deep (as non-root in a container) -> PermissionError traceback at startup.
