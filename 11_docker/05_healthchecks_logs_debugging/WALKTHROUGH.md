# Step 05 - Logs, health checks and debugging broken containers

## The story so far
[Step 04](../04_compose_multi_container/WALKTHROUGH.md) starts a healthy stack when everything is right. Real life delivers the other case.

## The pain point
`curl` hangs, the container shows `Exited (1)`, or it is running but nothing answers. Guessing and rebuilding blindly wastes hours. The cause is almost always one of five small things.

## The spark
Debug a container like any program, in a fixed order: is it running (`ps -a`), what did it say (`logs`), is it listening where and how I think (`docker port`, bind address), is the configuration what the app expects (`exec`, `inspect`)? Health checks separate *alive* from *ready*.

## Goal
Diagnose five deliberately broken setups using only Docker's own tools, no AI.

## Run it
```bash
cd 11_docker/05_healthchecks_logs_debugging
cat DIAGNOSIS.md               # drills; try each BEFORE opening the <details> answer
./run_all_drills.sh            # optional: see all symptoms in one go (needs Docker Desktop)
```
The broken images are expected to misbehave; nothing to run locally.

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| `docker logs` vs `docker exec` vs `docker inspect` | "look at output", "get a shell", "read config" | No | logs = what the main process printed; exec = run a command inside a live container; inspect = Docker's JSON record (state, exit code, ports) |
| Exit code 0 vs 1 vs 137 | "finished", "crashed", "killed/OOM" | No | 0 = ended normally, 1 = app error, 125-127 = Docker/command problem, 137 = SIGKILL (often out of memory) |
| `docker ps` vs `docker ps -a` | running vs all containers | No | Without `-a` stopped (crashed) containers are hidden |
| Healthy vs running | "up", "ok" | No | Running = process alive; healthy = the healthcheck command succeeds |
| `0.0.0.0` vs `127.0.0.1` | all interfaces vs loopback | No | `0.0.0.0` accepts connections from outside the container; `127.0.0.1` only from inside it |

## Code, step by step

Five broken folders, one bug each. They share the same ordinary lines (`FROM`, `WORKDIR`, `COPY requirements.txt`, `RUN pip install`, `COPY app`, `EXPOSE 8000`), explained once in block 1; blocks 2 to 7 show only the buggy lines of each Dockerfile. Nothing here is run (Docker is not required): read the files and predict the symptom before reading the answer.

### 1. Bug 1 (bad CMD path): normal build lines  (`broken_bad_cmd_path/Dockerfile` lines 1–6)
```dockerfile
FROM python:3.12-slim
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
EXPOSE 8000
```
- **In simple words:** Everything up to EXPOSE is a normal, working recipe.
- **Exactly:** Base slim, workdir, install requirements, copy `app` to `/code/app`, `EXPOSE 8000`. The image builds fine.
- **Why it is here:** A build that succeeds proves nothing about runtime; the bug appears only at start.

### 2. Bug 1: wrong module path in CMD  (`broken_bad_cmd_path/Dockerfile` lines 7–8)
```dockerfile
# BUG: the module is app.main, not src.main.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
- **In simple words:** The start command points at a folder that does not exist.
- **Exactly:** `uvicorn src.main:app` tries to import module `src.main`, but the code is at `/code/app/main.py`. Symptom: container exits at once with status 1; `docker logs` shows `ModuleNotFoundError: No module named 'src'`. Fix: `app.main:app`.
- **Why it is here:** Teaches to read logs first, and to match `CMD` to the real path.

### 3. Bug 2 (exits immediately): CMD that finishes  (`broken_exits_immediately/Dockerfile` lines 7–9)
```dockerfile
# BUG: this runs a script that finishes instantly, so PID 1 exits and the container stops (exit code 0).
# A container lives exactly as long as its main process.
CMD ["python", "-c", "print('starting up...')"]
```
- **In simple words:** The start command prints a line and ends, so the container stops.
- **Exactly:** `python -c "print('starting up...')"` runs and returns exit code 0. A container lives exactly as long as its main process (PID 1). Symptom: `docker ps` shows nothing; `docker ps -a` shows `Exited (0)`. Fix: run a long-lived server.
- **Why it is here:** Exit code 0 means "no crash"; so the lesson is that you must look at what the command does, not only at the code.

### 4. Bug 3 (localhost bind): missing --host  (`broken_localhost_bind/Dockerfile` lines 7–8)
```dockerfile
# BUG: uvicorn's default host is 127.0.0.1 = "inside this container only". -p 8000:8000 cannot reach it.
CMD ["uvicorn", "app.main:app", "--port", "8000"]
```
- **In simple words:** The server listens only on the container's own loopback.
- **Exactly:** Uvicorn's default host is `127.0.0.1`. `-p 8000:8000` forwards traffic to the container's external interface, where nothing listens. Symptom: container is "Up", logs look healthy ("Uvicorn running on http://127.0.0.1:8000") but `curl` gets "connection reset". Fix: add `--host 0.0.0.0`.
- **Why it is here:** The most common real-world "works in container, unreachable from outside" bug.

### 5. Bug 4 (missing dependency): requirements never installed  (`broken_missing_dependency/Dockerfile` lines 3–4)
```dockerfile
# BUG: requirements.txt is never installed, so `import fastapi` fails at start-up.
COPY app ./app
```
- **In simple words:** The shopping list is never read.
- **Exactly:** There is no `COPY requirements.txt` and no `RUN pip install`, so `import fastapi` fails. Symptom: exit 1, logs show `ModuleNotFoundError: No module named 'fastapi'`. Fix: add the two instructions before `COPY app`.
- **Why it is here:** The Python line in the traceback tells you it is an environment problem, not a code bug.

### 6. Bug 4: start command  (`broken_missing_dependency/Dockerfile` lines 5–6)
```dockerfile
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
- **In simple words:** Uvicorn is started via `python -m`.
- **Exactly:** `python -m uvicorn app.main:app --host 0.0.0.0 --port 8000` is equivalent to the `uvicorn` command but fails on `-m` import of uvicorn itself.
- **Why it is here:** Shows that the symptom may appear in the start step although the cause is in the build.

### 7. Bug 5 (wrong port): app listens on 8080  (`broken_wrong_port/Dockerfile` lines 6–8)
```dockerfile
EXPOSE 8000
# SYMPTOM TARGET: the app listens on 8080 inside the container, but we will map 8000:8000.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```
- **In simple words:** The server uses a different port from the one we publish.
- **Exactly:** `EXPOSE 8000` is only a note; uvicorn listens on `--port 8080`. With `-p 8000:8000`, Docker forwards to container port 8000 where nothing listens. `docker port <name>` shows the mapping; logs show the real port. Fix: make `-p` and `--port` agree.
- **Why it is here:** Teaches that `EXPOSE` does not make anything listen.

### 8. The app file, same in all five folders: docstring  (`broken_bad_cmd_path/app/main.py` lines 1–15)
```python
"""
GOAL: A tiny FastAPI app (health check + toy sentiment model) that we will put inside a Docker image.

RUN (from 11_docker/01_first_image):   ../../.venv/bin/python -m uvicorn app.main:app --port 8000
(Use the cd form: folder names starting with digits cannot be imported as Python packages from the repo root.)
EXPECTED OUTPUT: "Uvicorn running on http://127.0.0.1:8000"; then in another terminal
    curl localhost:8000/health                      -> {"status":"ok"}
    curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"text":"I love it"}'
                                                    -> {"label":"positive","score":1}
NEEDS: fastapi, uvicorn (see requirements.txt next to this folder). No internet needed.

Why a "toy" model? Docker does not care what the app does. A keyword counter keeps the image tiny and
the build fast, so you can focus on Docker. To containerise the REAL API from module 09, see ../README.md
("Swapping in the module-09 app").
"""
```
- **In simple words:** Each broken folder holds the same small app as step 01.
- **Exactly:** `diff` shows `app/main.py` is byte-identical in the five `broken_*` folders and in step 01. The bug is always in the Dockerfile, not in the Python.
- **Why it is here:** Keeps the debugging focus on Docker.

### 9. The app file: imports, app, keyword sets  (`broken_bad_cmd_path/app/main.py` lines 16–22)
```python
from fastapi import FastAPI            # FastAPI = the web framework (3rd-party). Builds the routes.
from pydantic import BaseModel         # BaseModel = declares + validates the shape of the request body.

app = FastAPI(title="Toy sentiment API")   # `app` is what uvicorn is told to serve ("app.main:app")

POSITIVE = {"good", "great", "love", "excellent", "happy"}
NEGATIVE = {"bad", "awful", "hate", "terrible", "sad"}
```
- **In simple words:** Imports FastAPI and pydantic, creates `app`, defines two word sets.
- **Exactly:** `app` must be importable as `app.main:app`.
- **Why it is here:** Defines the module path that `CMD` must match.

### 10. The app file: request schema and scoring  (`broken_bad_cmd_path/app/main.py` lines 25–34)
```python
class PredictRequest(BaseModel):
    text: str                          # type hint = validation rule: body must be {"text": "<string>"}


def predict_label(text: str) -> tuple[str, int]:
    """Count positive minus negative words. Returns (label, score)."""
    words = text.lower().split()                                  # "I LOVE it" -> ["i", "love", "it"]
    score = sum(w in POSITIVE for w in words) - sum(w in NEGATIVE for w in words)  # True counts as 1
    label = "positive" if score > 0 else "negative" if score < 0 else "neutral"
    return label, score
```
- **In simple words:** Request form and the counting function.
- **Exactly:** `PredictRequest(text: str)`; `predict_label` returns `(label, score)`.
- **Why it is here:** The working code under inspection.

### 11. The app file: routes  (`broken_bad_cmd_path/app/main.py` lines 37–49)
```python
@app.get("/health")                    # decorator: "GET /health runs the function below"
def health():
    return {"status": "ok"}            # dict -> JSON automatically


@app.post("/predict")
def predict(req: PredictRequest):      # FastAPI parses the JSON body into `req` or answers 422
    label, score = predict_label(req.text)
    return {"label": label, "score": score}


# TRY IT: add a word to POSITIVE and re-run the curl.
# BREAK IT: send {"txt": "hi"} -> 422 Unprocessable Entity. Read the JSON error: it names the missing field.
```
- **In simple words:** `/health` and `/predict`, plus the exercise comments.
- **Exactly:** GET `/health` returns `{"status":"ok"}`; POST `/predict` returns `{"label":..., "score":...}`.
- **Why it is here:** `/health` is what you curl after a fix to verify.


## Syntax explained
| Syntax | Meaning | Example |
|---|---|---|
| `docker ps -a` | include stopped containers | STATUS `Exited (1) 3 s ago` |
| `--format '{{.State.ExitCode}}'` | Go template to pick fields | `docker inspect c --format '{{.State.ExitCode}}'` |
| `--entrypoint sh` | override the entrypoint to get a shell | `docker run --rm -it --entrypoint sh img` |
| Exit codes | 0 finished, 1 app error, 125-127 docker/command problems, 137 SIGKILL/OOM, 139 segfault | |
| `HEALTHCHECK` | Dockerfile instruction running a probe command | see debug_cheatsheet.md |

## Execution trace (diagnosis loop)
Input: "curl doesn't work". Processing: `ps -a` (alive?) -> `logs` (what happened?) -> compare listening port/host with `-p` mapping (`docker port`) -> `exec` and look inside. Output: root cause sentence + fix + verification.

## Experiments
1. Add the HEALTHCHECK from the cheat-sheet to `broken_wrong_port` - status shows `unhealthy` after the retries although the container is "Up" (health measures the app, not the process).
2. Make `/health` return 500 deliberately and watch `docker ps` status change.
3. Fix each drill and write the one-line verification (`curl` output / `ps` status).

## Debugging practice
This whole step is the debugging practice. Format to write down for each drill: symptom - command that revealed it - root cause - fix - verification.

## Rebuild checklist
- [ ] List the 6 debug commands from memory with what each tells you.
- [ ] Explain exit code 0 vs 1 vs 137.

## The new problem (cliffhanger)
Now we can diagnose problems, but our image is still large, carries build tools, and runs as root, so a bug in the app is a bug with root rights. Next: [step 06, multi-stage builds and security](../06_multistage_and_security/WALKTHROUGH.md).

## Say it aloud
> **Say it aloud (interview box):** I debug containers like any program: is it running, what do the logs say, is it listening where I think, is the port mapped how I think, and is the config what the app expects. Typical culprits are binding to 127.0.0.1, a wrong port mapping, missing dependencies in the image and a CMD that exits. The trade-off is that a health check tells you the app is unwell, but plain Docker will not fix it, which needs an orchestrator.
