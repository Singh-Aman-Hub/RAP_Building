# Step 03 - Ports, environment variables, volumes and bind mounts

## The story so far
[Step 02](../02_layers_and_cache/WALKTHROUGH.md) made builds fast, but the image still has fixed behaviour, and a container's own files vanish when it is removed.

## The pain point
A hard-coded model name means one image per environment. A hard-coded port clashes with whatever else runs on the host. A visit counter written inside the container resets to zero every time the container is replaced.

## The spark
Keep the image identical and push differences outside it: `-p` chooses the host port, `-e` / `--env-file` sets configuration at start, and a **volume** (or a bind mount while developing) keeps data outside the disposable container.

## Goal
Configure a container at run time (no rebuild) and persist data beyond the container's life.

## Run it
```bash
cd 11_docker/03_ports_env_volumes
../../.venv/bin/python check_locally.py   # no Docker needed
./commands.sh                              # line by line, Docker Desktop running
```
Real output of `check_locally.py` (temp dir path varies):
```
{'label': 'positive', 'score': 1, 'model_name': 'test-model'}
{'model_name': 'test-model', 'data_dir': '/var/folders/.../tmpXXXX', 'visits': 1}
OK
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Volume vs bind mount | "named volume", "host mount", `-v` | No | Volume: Docker manages the storage location; bind mount: you name an exact host path |
| `ENV` vs `ARG` | build variable, runtime variable | No | `ENV` persists into the running container; `ARG` exists only during build |
| `-e` vs `--env-file` | env var flags | Same effect | `-e KEY=val` sets one variable; `--env-file f` loads many lines from a file |
| `VOLUME` vs `-v` | "mount a volume" | No | `VOLUME` in the Dockerfile only declares a mount point; `-v` at run time chooses what to mount there |
| `-p 8000:8000` vs `EXPOSE` | publish vs expose | No | `-p` creates a host-to-container port mapping; `EXPOSE` is documentation |

## Code, step by step

### 1. Base, workdir, install, code  (`Dockerfile` lines 1–5)
```dockerfile
FROM python:3.12-slim
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
```
- **In simple words:** Same recipe as steps 01 and 02: Python base, folder, libraries, code.
- **Exactly:** `FROM`, `WORKDIR /code`, `COPY requirements.txt .`, `RUN pip install --no-cache-dir -r requirements.txt`, `COPY app ./app`. Order follows the cache rule from step 02.
- **Why it is here:** A cached install layer keeps rebuilds fast while you experiment with env vars.

### 2. Default environment variables  (`Dockerfile` lines 7–10)
```dockerfile
# ENV sets a DEFAULT inside the image. `docker run -e` or --env-file overrides it at run time.
ENV MODEL_NAME=keyword-sentiment-v1 \
    DATA_DIR=/data \
    LOG_LEVEL=INFO
```
- **In simple words:** Give the app default settings that can be changed without rebuilding.
- **Exactly:** `ENV A=1 B=2` sets three variables (`MODEL_NAME`, `DATA_DIR=/data`, `LOG_LEVEL=INFO`) in the image config; the backslash continues the line. `docker run -e MODEL_NAME=x` or `--env-file` overrides them at start time.
- **Why it is here:** Same image can run as dev, staging or prod; config is outside the code (twelve-factor style).

### 3. Declare the data folder as a volume  (`Dockerfile` lines 12–13)
```dockerfile
# VOLUME documents that /data holds state worth persisting (Docker creates an anonymous volume if you forget).
VOLUME /data
```
- **In simple words:** Mark /data as the place where data must outlive the container.
- **Exactly:** `VOLUME /data` declares a mount point. If you run without `-v`, Docker creates an anonymous volume there. The container's own writable layer is deleted with the container; a volume is not.
- **Why it is here:** Without a volume, the visits counter resets every time the container is removed.

### 4. Document the port and start the server  (`Dockerfile` lines 14–15)
```dockerfile
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
- **In simple words:** Note port 8000 and start uvicorn on all interfaces.
- **Exactly:** `EXPOSE 8000` is documentation; `CMD` exec form runs uvicorn with `--host 0.0.0.0 --port 8000`. Publishing is done by `docker run -p 8080:8000` (host:container).
- **Why it is here:** The host port can change (`-p 9000:8000`) without touching the app.

### 5. Env file  (`sample.env` lines 1–3)
```text
# One KEY=value per line, no quotes, no `export`. Used with: docker run --env-file sample.env ...
MODEL_NAME=model-from-env-file
LOG_LEVEL=DEBUG
```
- **In simple words:** A text file of settings that Docker loads for you.
- **Exactly:** One `KEY=value` per line, no quotes, no `export`. Used with `docker run --env-file sample.env`. Here `MODEL_NAME=model-from-env-file` and `LOG_LEVEL=DEBUG`.
- **Why it is here:** Keeps long config lists out of the command line.

### 6. Module docstring  (`app/main.py` lines 1–9)
```python
"""
GOAL: Same toy API, but configured through ENVIRONMENT VARIABLES and keeping a log file in a DATA folder,
      so we can practise `-e`, `--env-file`, ports, volumes and bind mounts.

RUN (from 11_docker/03_ports_env_volumes):
    MODEL_NAME=my-model-v2 DATA_DIR=./data ../../.venv/bin/python -m uvicorn app.main:app --port 8000
EXPECTED OUTPUT: curl localhost:8000/info -> {"model_name":"my-model-v2","data_dir":"./data","visits":N}
NEEDS: fastapi, uvicorn.
"""
```
- **In simple words:** A note on what the file is and how to run it.
- **Exactly:** Shows the local run line `MODEL_NAME=my-model-v2 DATA_DIR=./data ... uvicorn app.main:app` and the expected `/info` JSON.
- **Why it is here:** Documentation next to the code.

### 7. Imports  (`app/main.py` lines 10–15)
```python
import logging
import os                              # os.getenv reads environment variables (built-in module)
from pathlib import Path               # Path = object-oriented file paths (built-in)

from fastapi import FastAPI
from pydantic import BaseModel
```
- **In simple words:** Load the tools: logging, environment access, file paths, FastAPI, pydantic.
- **Exactly:** `logging`, `os`, `pathlib.Path` are standard library; `FastAPI` and `BaseModel` are third-party.
- **Why it is here:** `os.getenv` is how every config value reaches the app.

### 8. Read settings and prepare the data folder  (`app/main.py` lines 17–21)
```python
# os.getenv("NAME", default): value of env var NAME, or `default` if it is not set.
MODEL_NAME = os.getenv("MODEL_NAME", "keyword-sentiment-v1")
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)          # make the folder if missing (no error if present)
VISITS_FILE = DATA_DIR / "visits.txt"                # `/` joins paths with Path objects
```
- **In simple words:** Read the settings from the environment, with defaults, and make sure the data folder exists.
- **Exactly:** `os.getenv("MODEL_NAME", "keyword-sentiment-v1")` returns the env value (a `str`) or the default. `Path(...)` wraps `DATA_DIR`. `mkdir(parents=True, exist_ok=True)` creates it, no error if present. `DATA_DIR / "visits.txt"` joins paths. Default `./data` locally; `/data` in the image because of `ENV`.
- **Why it is here:** Without `exist_ok=True` the second start would crash with `FileExistsError`; without the env lookup the config is hard-coded.

### 9. Logging to stdout  (`app/main.py` lines 23–25)
```python
# Logging to stdout is what Docker expects: `docker logs` shows whatever the process prints.
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("api")
```
- **In simple words:** Print log lines to the screen so Docker can collect them.
- **Exactly:** `logging.basicConfig(level=..., format=...)` sets the root logger to write to stderr/stdout. The level comes from `LOG_LEVEL` (`INFO` or `DEBUG`). `getLogger("api")` is a named logger. `docker logs <container>` shows everything the main process prints.
- **Why it is here:** Logging to a file inside a container hides it from `docker logs` and grows the writable layer.

### 10. App object and keyword sets  (`app/main.py` lines 27–29)
```python
app = FastAPI(title="Toy API with env + volume")
POSITIVE = {"good", "great", "love", "excellent", "happy"}
NEGATIVE = {"bad", "awful", "hate", "terrible", "sad"}
```
- **In simple words:** Create the web app and the two word lists.
- **Exactly:** Same as step 01: `FastAPI(title=...)` and two sets.
- **Why it is here:** Same toy model so only the config changes.

### 11. Request schema  (`app/main.py` lines 32–33)
```python
class PredictRequest(BaseModel):
    text: str
```
- **In simple words:** A form saying the body has a text string.
- **Exactly:** Pydantic `PredictRequest` with `text: str`; invalid bodies get a 422.
- **Why it is here:** Input validation for free.

### 12. Persist a visit counter in a file  (`app/main.py` lines 36–40)
```python
def count_visit() -> int:
    """Persist a counter in a file: survives restarts only if DATA_DIR is a volume/bind mount."""
    n = int(VISITS_FILE.read_text()) if VISITS_FILE.exists() else 0
    VISITS_FILE.write_text(str(n + 1))
    return n + 1
```
- **In simple words:** Read the counter from a file, add one, save it, return it.
- **Exactly:** `read_text()` returns a `str`, `int(...)` parses it; the file is missing on the first call so `n = 0`. `write_text(str(n + 1))` overwrites. Survives restarts only if `DATA_DIR` is a volume or bind mount.
- **Why it is here:** This is the experiment: it makes the difference between a throwaway container and persisted data visible.

### 13. Health endpoint  (`app/main.py` lines 43–45)
```python
@app.get("/health")
def health():
    return {"status": "ok"}
```
- **In simple words:** The "alive" door.
- **Exactly:** GET `/health` returns `{"status":"ok"}`.
- **Why it is here:** For `curl` checks and later healthchecks.

### 14. Info endpoint  (`app/main.py` lines 48–51)
```python
@app.get("/info")
def info():
    visits = int(VISITS_FILE.read_text()) if VISITS_FILE.exists() else 0
    return {"model_name": MODEL_NAME, "data_dir": str(DATA_DIR), "visits": visits}
```
- **In simple words:** Show which model name and data folder are in use, and the visit count.
- **Exactly:** Returns `{"model_name":..., "data_dir":..., "visits":N}`. Changing `-e MODEL_NAME=...` changes this output without a rebuild.
- **Why it is here:** Proves the env var reached the app.

### 15. Predict endpoint with logging and counter  (`app/main.py` lines 54–61)
```python
@app.post("/predict")
def predict(req: PredictRequest):
    words = req.text.lower().split()
    score = sum(w in POSITIVE for w in words) - sum(w in NEGATIVE for w in words)
    label = "positive" if score > 0 else "negative" if score < 0 else "neutral"
    n = count_visit()
    log.info("predict #%d model=%s label=%s", n, MODEL_NAME, label)   # visible with `docker logs`
    return {"label": label, "score": score, "model_name": MODEL_NAME}
```
- **In simple words:** Score the text, count the visit, log a line, answer.
- **Exactly:** Same score formula as step 01, then `count_visit()` increments the file counter and `log.info("predict #%d model=%s label=%s", ...)` writes a log line (lazy `%` formatting). Response adds `model_name`.
- **Why it is here:** The log line is what you will find with `docker logs`; the counter shows volumes at work.

### 16. Exercises as comments  (`app/main.py` lines 63–64)
```python
# TRY IT: change MODEL_NAME and watch /info change without touching code (that is the point of env vars).
# BREAK IT: set DATA_DIR=/nonexistent/deep (as non-root in a container) -> PermissionError traceback at startup.
```
- **In simple words:** Two ideas to try.
- **Exactly:** TRY IT changes `MODEL_NAME`; BREAK IT uses an unwritable `DATA_DIR` to produce a `PermissionError` at startup.
- **Why it is here:** Learn config and permissions by breaking them.


## Syntax explained
| Syntax | Meaning | Example |
|---|---|---|
| `-p 9000:8000` | host 9000 -> container 8000 | `curl localhost:9000` |
| `-e KEY=value` | set one env var | `-e MODEL_NAME=x` |
| `--env-file f` | many env vars from file | `--env-file sample.env` |
| `-v name:/path` | named volume (Docker-managed) | `-v api-data:/data` |
| `-v /abs/host:/path` | bind mount (host folder) | `-v "$(pwd)/host-data:/data"` |
| `ENV A=1 B=2` | default env in the image | overridable at run time |
| `VOLUME /data` | marks a mount point | |
| `$(pwd)` | shell substitution: current directory | |
| Python `os.getenv("X", "default")` | read env var or default | |
| Python `Path("a") / "b"` | join paths | `Path("/data") / "visits.txt"` |

## Concepts: volume vs bind mount
| | Named volume | Bind mount |
|---|---|---|
| Location | Managed by Docker | Any host path you choose |
| Best for | Databases, app state, production | Dev: live code edits, sharing files with host |
| Portable | Yes (same on any host) | No (depends on host paths) |
| Container's own writable layer | Deleted with the container | n/a |

## Execution trace
1. `docker run -e MODEL_NAME=cli-model ...`: Docker injects the variable into the process environment; `os.getenv` in main.py reads it at import time; `/info` returns `cli-model`.
2. POST `/predict` -> `count_visit()` writes `/data/visits.txt`.
3. With `-v api-data:/data` the file lives in the volume, so a brand-new container sees `visits: 1`. Without `-v`, it would reset to 0.
(Typical output; daemon was off when written.)

## Experiments
1. Run without `-v`, predict, `docker rm -f`, run again: visits resets.
2. Pass both `-e MODEL_NAME=a` and `--env-file sample.env`: which wins? (Explicit `-e` overrides the file.)
3. Bind-mount `app/` over `/code/app` and edit main.py: restart picks up the change without rebuild (dev workflow).

## Debugging practice
`docker run -v host-data:/data` vs `-v ./host-data:/data`: a path *without* a leading `/` or `./` is treated as a **named volume**, not a folder. If you expect files on your disk and see none, check which one you made (`docker volume ls`, `docker inspect <c> --format '{{json .Mounts}}'`).

## Rebuild checklist
- [ ] Run a container with a different host port, an env var and a named volume from memory.
- [ ] Say when you would pick a bind mount over a volume.

## The new problem (cliffhanger)
A real system is not one container. We need an API, a UI and a cache, and typing three long `docker run` commands in the right order, with the right names, is fragile. Next: [step 04, Compose](../04_compose_multi_container/WALKTHROUGH.md).

## Say it aloud
> **Say it aloud (interview box):** I keep configuration in environment variables so one image works in dev and prod, injected with -e or an env file. Containers are ephemeral, so state goes in a named volume, or in a bind mount while I develop. The trade-off is that env vars are visible to anyone who can inspect the container, so real secrets need a proper secrets mechanism.
