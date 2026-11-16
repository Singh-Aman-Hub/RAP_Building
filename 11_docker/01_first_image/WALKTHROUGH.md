# Step 01 - Your first image and container

## The story so far
Chapter 22 ([module 10](../../10_ml_system_architecture/)) ended with a blueprint for a serving system and one honest problem: it works on my machine only. This step is the first answer. We take the toy FastAPI service from [module 09](../../09_api_development/) and make it run anywhere.

## The pain point
A teammate clones your repo. Their Python is a different version, `fastapi` is not installed, port 8000 is busy, and your `.venv` is not in git. The same code behaves differently on every laptop and server. Writing a longer README does not fix it.

## The spark
Do not ship instructions, ship the machine. A **Dockerfile** is a recipe that starts from a known base (Python 3.12 slim), installs pinned requirements and copies your code. Building it gives an **image**; running the image gives a **container**, the same everywhere.

## Goal
Package a small FastAPI app into an image, run it as a container, and reach it from your laptop.

## Run it
```bash
cd 11_docker/01_first_image
../../.venv/bin/python check_locally.py     # 1. prove the app works WITHOUT Docker (real output below)
cat commands.sh                              # 2. then run its lines one by one (Docker Desktop must be running)
```
Real output of `check_locally.py`:
```
{'status': 'ok'}
{'label': 'positive', 'score': 1}
OK
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Image vs container | "image = class, container = running instance", snapshot vs process | No | Image is the read-only recipe result; container is a live process made from it |
| `CMD` vs `RUN` vs `ENTRYPOINT` | build command vs start command vs fixed start program | No | `RUN` happens at build and saves a layer; `CMD`/`ENTRYPOINT` happen at container start |
| `EXPOSE` vs `-p` (publish) | "open the port" (wrong), port mapping | No | `EXPOSE` is documentation; `-p host:container` actually publishes the port |
| `COPY` vs `ADD` | copy files | Almost | `ADD` also unpacks tar files and fetches URLs; prefer `COPY` |
| Build context vs `WORKDIR` | "the folder", "the path" | No | Context = host files sent to the builder; `WORKDIR` = directory inside the image |


## Code, step by step

### 1. Pick the base image  (`Dockerfile` lines 1–3)
```dockerfile
# ---- Step 01: the smallest Dockerfile that works ----
# FROM: the base image we build on. python:3.12-slim = Debian + Python 3.12, without extra build tools.
FROM python:3.12-slim
```
- **In simple words:** Start from a ready-made little computer that already has Python 3.12.
- **Exactly:** `FROM python:3.12-slim` pulls a Debian-based image with Python 3.12 and no compilers or extras. It becomes the bottom layer; all later instructions stack on it. Line 1 and 2 are comments.
- **Why it is here:** Without a base you have no OS and no Python. `-slim` keeps it around 50 MB instead of 1 GB for the full image.

### 2. Set the working directory  (`Dockerfile` lines 5–6)
```dockerfile
# WORKDIR: create /code and make it the current directory for all later instructions (and at runtime).
WORKDIR /code
```
- **In simple words:** Make a folder called /code inside the image and "cd" into it.
- **Exactly:** `WORKDIR /code` creates the directory if missing and sets it as the current directory for all following `COPY`, `RUN`, `CMD` and for the running container.
- **Why it is here:** Without it files land in `/` and relative paths like `./app` become confusing.

### 3. Copy only the requirements file  (`Dockerfile` lines 8–9)
```dockerfile
# COPY <from-host> <to-image>: copy files from the build context (this folder) into the image.
COPY requirements.txt .
```
- **In simple words:** Put the shopping list (requirements.txt) into the image first, before the code.
- **Exactly:** `COPY requirements.txt .` copies from the build context (this folder) to `/code/requirements.txt`. It becomes its own layer whose cache key is the file content.
- **Why it is here:** Copying requirements alone lets Docker reuse the expensive install layer when only code changes (step 02).

### 4. Install the dependencies  (`Dockerfile` lines 11–13)
```dockerfile
# RUN: execute a shell command at BUILD time. The result is saved as a new image layer.
# --no-cache-dir: do not keep pip's download cache in the image (smaller image).
RUN pip install --no-cache-dir -r requirements.txt
```
- **In simple words:** Run pip while building so the libraries are baked into the image.
- **Exactly:** `RUN` executes in a shell at build time and saves the result as a layer. `--no-cache-dir` stops pip storing its download cache inside the image. Reads `requirements.txt` with `-r`.
- **Why it is here:** Without it the container starts and fails with `ModuleNotFoundError: fastapi`.

### 5. Copy the application code  (`Dockerfile` lines 15–16)
```dockerfile
# Copy the application code (after the install -- see step 02 for why the order matters).
COPY app ./app
```
- **In simple words:** Now copy the code itself.
- **Exactly:** `COPY app ./app` copies host folder `app/` to `/code/app`. Happens after the install, so code edits do not trigger a reinstall.
- **Why it is here:** Without it there is nothing to run. Wrong order here makes every build slow.

### 6. Document the port  (`Dockerfile` lines 18–19)
```dockerfile
# EXPOSE: documentation only! It does NOT publish the port. Publishing happens with `docker run -p`.
EXPOSE 8000
```
- **In simple words:** A note saying "this app listens on 8000".
- **Exactly:** `EXPOSE 8000` writes metadata only (`docker inspect` shows it). It does not open or publish anything.
- **Why it is here:** Tools like `docker run -P` and humans read it. The real publishing is `docker run -p 8000:8000`.

### 7. Default start command  (`Dockerfile` lines 21–23)
```dockerfile
# CMD: the default command when a container starts. Exec form (JSON list) = no shell, signals work.
# --host 0.0.0.0 is ESSENTIAL: 127.0.0.1 inside a container is unreachable from outside it.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
- **In simple words:** What to run when the container starts: the web server.
- **Exactly:** `CMD` in exec form (JSON list) runs `uvicorn app.main:app --host 0.0.0.0 --port 8000` directly, with no shell, so the process is PID 1 and receives `SIGTERM` for clean shutdown. `app.main:app` = module `app/main.py`, variable `app`. Host `0.0.0.0` = listen on all interfaces.
- **Why it is here:** With `127.0.0.1` the app listens only inside the container's own loopback; published ports would show "connection reset" (step 05, broken_localhost_bind).

### 8. Module docstring: what the file is for  (`app/main.py` lines 1–15)
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
- **In simple words:** A comment-like note at the top that says what the file does and how to run it. Python ignores it when running.
- **Exactly:** A triple-quoted string as the first statement becomes the module `__doc__`. It holds GOAL, RUN, EXPECTED OUTPUT and NEEDS. Nothing here executes.
- **Why it is here:** When you open the file in six months, you know in ten seconds what it is and how to start it.

### 9. Import FastAPI and BaseModel  (`app/main.py` lines 16–17)
```python
from fastapi import FastAPI            # FastAPI = the web framework (3rd-party). Builds the routes.
from pydantic import BaseModel         # BaseModel = declares + validates the shape of the request body.
```
- **In simple words:** Bring in the two tools we need: one to build the web app, one to describe what a request body must look like.
- **Exactly:** `FastAPI` is the ASGI web-framework class; `BaseModel` is pydantic's base class that validates and parses data from type hints. Both are third-party and come from `requirements.txt` (installed by the Dockerfile).
- **Why it is here:** Without the install step in the Dockerfile this line raises `ModuleNotFoundError` and the container exits at once (see step 05, broken_missing_dependency).

### 10. Create the app object  (`app/main.py` lines 19–19)
```python
app = FastAPI(title="Toy sentiment API")   # `app` is what uvicorn is told to serve ("app.main:app")
```
- **In simple words:** Make the web application. Uvicorn will be told to run this object.
- **Exactly:** `app` is a `FastAPI` instance with `title="Toy sentiment API"` (shown on the auto docs page `/docs`). The name matters: the Dockerfile `CMD` says `app.main:app` = module `app/main.py`, variable `app`.
- **Why it is here:** If the variable were called anything else, uvicorn would fail with "Attribute 'app' not found in module".

### 11. The two keyword sets  (`app/main.py` lines 21–22)
```python
POSITIVE = {"good", "great", "love", "excellent", "happy"}
NEGATIVE = {"bad", "awful", "hate", "terrible", "sad"}
```
- **In simple words:** Two small word lists: happy words and sad words. This is our whole "model".
- **Exactly:** Two `set[str]` constants. Set membership `w in POSITIVE` is O(1) on average.
- **Why it is here:** A keyword model keeps the image tiny and the build fast, so you can study Docker rather than machine learning.

### 12. Request schema  (`app/main.py` lines 25–26)
```python
class PredictRequest(BaseModel):
    text: str                          # type hint = validation rule: body must be {"text": "<string>"}
```
- **In simple words:** A form that says "the request must contain a text field that is a string".
- **Exactly:** Pydantic model with one field `text: str`. FastAPI reads the JSON body, builds `PredictRequest(text=...)`, and answers HTTP 422 with a JSON error if the field is missing or not a string (BREAK IT at the bottom of the file).
- **Why it is here:** Gives free input validation; bad requests never reach your logic.

### 13. Scoring function  (`app/main.py` lines 29–34)
```python
def predict_label(text: str) -> tuple[str, int]:
    """Count positive minus negative words. Returns (label, score)."""
    words = text.lower().split()                                  # "I LOVE it" -> ["i", "love", "it"]
    score = sum(w in POSITIVE for w in words) - sum(w in NEGATIVE for w in words)  # True counts as 1
    label = "positive" if score > 0 else "negative" if score < 0 else "neutral"
    return label, score
```
- **In simple words:** Count happy words minus sad words and turn the result into a label.
- **Exactly:** `text.lower().split()` splits on whitespace; `sum(w in POSITIVE for w in words)` sums booleans (True is 1). Example: "I love it" gives words `['i','love','it']`, score `1 - 0 = 1`, label `positive`. Returns `tuple[str, int]`.
- **Why it is here:** Keeping logic in a plain function (not inside the route) lets `check_locally.py` and tests call it without a server.

### 14. Health endpoint  (`app/main.py` lines 37–39)
```python
@app.get("/health")                    # decorator: "GET /health runs the function below"
def health():
    return {"status": "ok"}            # dict -> JSON automatically
```
- **In simple words:** A "are you alive?" door. It always answers ok.
- **Exactly:** `@app.get("/health")` registers a GET route; the returned dict becomes JSON `{"status":"ok"}` with HTTP 200.
- **Why it is here:** Healthchecks, compose and load balancers poll this URL to know the container is ready (steps 04 and 05).

### 15. Predict endpoint  (`app/main.py` lines 42–45)
```python
@app.post("/predict")
def predict(req: PredictRequest):      # FastAPI parses the JSON body into `req` or answers 422
    label, score = predict_label(req.text)
    return {"label": label, "score": score}
```
- **In simple words:** The real work door: send text, get a label back.
- **Exactly:** `@app.post("/predict")`; the parameter `req: PredictRequest` tells FastAPI to parse and validate the JSON body. Returns `{"label": ..., "score": ...}`, e.g. `{"label":"positive","score":1}` for "I love it".
- **Why it is here:** This is the one endpoint a client actually calls.

### 16. Exercises as comments  (`app/main.py` lines 48–49)
```python
# TRY IT: add a word to POSITIVE and re-run the curl.
# BREAK IT: send {"txt": "hi"} -> 422 Unprocessable Entity. Read the JSON error: it names the missing field.
```
- **In simple words:** Two suggestions for you to try.
- **Exactly:** Comments only: TRY IT (add a word) and BREAK IT (send `{"txt":"hi"}` and read the 422 JSON error).
- **Why it is here:** Learning by changing one thing at a time.


## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `FROM image:tag` | Start from a base image | `FROM python:3.12-slim` |
| `WORKDIR /code` | `cd` + `mkdir -p` for later steps | `WORKDIR /code` |
| `COPY src dst` | Host (build context) -> image | `COPY requirements.txt .` |
| `RUN cmd` | Run at **build** time, creates a layer | `RUN pip install -r requirements.txt` |
| `EXPOSE 8000` | Documentation only | does not publish anything |
| `CMD [...]` | Default **run-time** command, exec form | `CMD ["uvicorn", "app.main:app"]` |
| `-t name:tag` | Name the built image | `docker build -t toy-api:v1 .` |
| `-p HOST:CONTAINER` | Publish a port | `-p 8000:8000` |
| `-d` | Detached (background) | `docker run -d ...` |
| the trailing `.` in `docker build` | The **build context** (folder sent to Docker) | `docker build -t x .` |
| Python: `@app.get("/health")` | Decorator: registers the function as a route | see main.py |
| Python: `sum(w in POSITIVE for w in words)` | Generator expression; `True` counts as 1 | `sum([True, False, True])` = 2 |

## Libraries & commands
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `docker build` | Docker CLI | Builds an image from a Dockerfile | `-t`, `-f` (file), context path | image ID |
| `docker run` | Docker CLI | Creates + starts a container | `-d -p -e -v --name --rm` | container ID |
| `docker ps [-a]` | Docker CLI | Lists (all) containers | `-a` includes exited | table |
| `docker logs` | Docker CLI | Shows container stdout/stderr | `-f` follow, `--tail N` | text |
| `FastAPI` | 3rd-party | Web framework (routes, validation, docs) | `title=` | app object |
| `BaseModel` (pydantic) | 3rd-party | Validates request body shape | type-hinted fields | model instance |
| `TestClient` | 3rd-party (starlette) | Calls the app in-process for tests | app | `Response` |

## Execution trace (input -> processing -> output)
1. **Input**: `docker build -t toy-api:v1 .` - the CLI zips the folder (minus `.dockerignore`) = build context.
2. **Processing**: Docker executes each Dockerfile line; `FROM` pulls python:3.12-slim, `RUN pip install` makes a layer with fastapi+uvicorn, `COPY app` adds your code.
3. **Output**: an image `toy-api:v1` (typically ~150-200 MB; Python slim ~120 MB + packages).
4. `docker run -d -p 8000:8000 toy-api:v1`: Docker makes a container (isolated process), runs `uvicorn ... --host 0.0.0.0 --port 8000`, and forwards host port 8000 to container port 8000.
5. `curl localhost:8000/health` -> `{"status":"ok"}`. (Typical output - the daemon was not running when this was written, so Docker commands were not captured; the local check above was.)

## Experiments (one change at a time)
1. Remove `--host 0.0.0.0` -> rebuild -> curl fails (see `05.../broken_localhost_bind`).
2. Run with `-p 9000:8000` -> reachable at 9000, not 8000.
3. Run two containers from the same image with different host ports: one image, many containers.
4. Edit `POSITIVE` in main.py, `docker run` again WITHOUT rebuilding -> unchanged behaviour. Why? (Image is a snapshot.)

## Debugging practice
`docker run -d -p 8000:8000 toy-api:v1` when port 8000 is already used -> `Bind for 0.0.0.0:8000 failed: port is already allocated`. Read: *host* port taken. Find with `docker ps` or `lsof -i :8000`; choose another host port.

## Rebuild checklist
- [ ] Write the 7-line Dockerfile from memory (FROM, WORKDIR, COPY req, RUN pip, COPY app, EXPOSE, CMD).
- [ ] Write build + run commands from memory.

## The new problem (cliffhanger)
The image works, but every time you change one line of code, `docker build` seems to reinstall every package. A 1-second edit costs a 30-second build. Why, and how do we make rebuilds fast? Next: [step 02, layers and cache](../02_layers_and_cache/WALKTHROUGH.md).

## Say it aloud
> **Say it aloud (interview box):** An image is a read-only template built from a Dockerfile layer by layer, and a container is a running instance of it with its own filesystem and network. I used a slim Python base, installed pinned requirements, copied my code and started uvicorn on 0.0.0.0 so the published port is reachable. The trade-off is that every code change needs a rebuild, which is why layer ordering matters.
