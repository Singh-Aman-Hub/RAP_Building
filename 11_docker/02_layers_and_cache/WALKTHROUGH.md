# Step 02 - Layers, the build cache and base-image choice

## The story so far
[Step 01](../01_first_image/WALKTHROUGH.md) gave us a working image, but a one-line code edit made the rebuild feel as slow as the first build.

## The pain point
`Dockerfile.bad_order` runs `COPY . .` before `pip install`. Any edit to any file changes that layer, and Docker then rebuilds every layer after it, including the slow dependency install. You wait, every time.

## The spark
Each instruction is a cached layer, and a layer is reused only if everything before it is unchanged. So order instructions from least to most frequently changing: copy `requirements.txt`, install, and only then copy the source. Pick `slim`, not `alpine`, as the base so ML wheels install.

## Goal
Understand that each Dockerfile instruction is a cached layer, order them to make rebuilds fast, and choose a base image.

## Run it
```bash
cd 11_docker/02_layers_and_cache
cat Dockerfile.bad_order Dockerfile.good_order
./commands.sh      # run line by line; Docker Desktop must be running
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Layer vs image | "image = stack of layers" | No | A layer is one filesystem diff made by one instruction; an image is the ordered stack plus metadata |
| Build cache vs layer cache | "Docker cache", "cache hit/miss" | Yes | Same thing: Docker reuses a layer if the instruction text and its input files are unchanged |
| `slim` vs `alpine` vs full | Debian-slim, musl, "fat" image | No | slim = Debian minus extras (glibc); alpine = musl libc, tiny but many ML wheels missing; full = compilers included, ~1 GB |
| `.dockerignore` vs `.gitignore` | ignore files | No | `.dockerignore` filters the build context sent to Docker; `.gitignore` filters what git tracks |
| Build context vs image content | "what Docker can see" | No | Context is everything sent to the builder; only files you `COPY` end up in the image |

## Code, step by step

### 1. Bad order: base image and comment  (`Dockerfile.bad_order` lines 1–3)
```dockerfile
# BAD layer order: any code edit invalidates the COPY layer, so pip install re-runs EVERY build.
# (Dockerfile comments must be on their own line: a '#' after an instruction is passed as an argument!)
FROM python:3.12-slim
```
- **In simple words:** Same starting point as before, plus a note that this version is deliberately slow.
- **Exactly:** `FROM python:3.12-slim` is layer 0. The comment states the rule: a changed `COPY` input invalidates that layer and every layer after it. A `#` after an instruction on the same line would be passed as an argument, so comments sit on their own lines.
- **Why it is here:** Shows the baseline to compare with `Dockerfile.good_order`.

### 2. Bad order: copy everything at once  (`Dockerfile.bad_order` lines 4–6)
```dockerfile
WORKDIR /code
# copies code AND requirements together
COPY . .
```
- **In simple words:** Put the whole project folder (code and shopping list) into the image in one go.
- **Exactly:** `COPY . .` copies the entire build context (minus `.dockerignore` entries) to `/code`. Docker hashes every copied file; one edited `.py` file changes the hash.
- **Why it is here:** This is the mistake: any code edit invalidates this layer.

### 3. Bad order: install after the copy  (`Dockerfile.bad_order` lines 7–9)
```dockerfile
# re-runs whenever ANY file changed
RUN pip install --no-cache-dir -r requirements.txt
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
- **In simple words:** Install libraries after the copy, so every code edit also reinstalls them.
- **Exactly:** Because the `COPY . .` layer changed, the `RUN pip install` layer below it is a cache miss and re-downloads packages every build. Then `CMD` starts uvicorn as in step 01.
- **Why it is here:** Each tiny code edit costs the full install time (seconds here, minutes with torch).

### 4. Good order: base, workdir, requirements first  (`Dockerfile.good_order` lines 1–4)
```dockerfile
# GOOD layer order: least-frequently-changing things first, most-frequently-changing last.
FROM python:3.12-slim
WORKDIR /code
# layer A: changes only when dependencies change
```
- **In simple words:** Copy only the shopping list first.
- **Exactly:** `COPY requirements.txt .` is "layer A". Its cache key is the content of that one file, so it changes only when you change dependencies.
- **Why it is here:** Isolates the slow-changing input from the fast-changing one.

### 5. Good order: cached install  (`Dockerfile.good_order` lines 5–7)
```dockerfile
COPY requirements.txt .
# layer B: CACHED as long as layer A is unchanged
RUN pip install --no-cache-dir -r requirements.txt
```
- **In simple words:** Install libraries now. If the shopping list did not change, Docker skips this.
- **Exactly:** Layer B is reused ("CACHED" in the build log) while layer A is unchanged. This is the layer worth caching: it is the slowest.
- **Why it is here:** Saves the whole install step on 99 percent of builds.

### 6. Good order: code last, then start command  (`Dockerfile.good_order` lines 8–10)
```dockerfile
# layer C: changes on every code edit (cheap to rebuild)
COPY app ./app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
- **In simple words:** Copy the code last, because it changes most often.
- **Exactly:** Layer C `COPY app ./app` is rebuilt on every code edit (milliseconds). `CMD` is metadata, no filesystem layer.
- **Why it is here:** Rule of thumb: least-frequently-changing first, most-frequently-changing last.

### 7. Alpine: tiny base, but musl  (`Dockerfile.alpine` lines 1–3)
```dockerfile
# Alpine = tiny (~5 MB) BUT uses musl libc instead of glibc. Many Python wheels (numpy, torch, pandas)
# have no musl wheel, so pip compiles from source: slow builds, sometimes impossible. For ML, prefer -slim.
FROM python:3.12-alpine
```
- **In simple words:** A much smaller base image with a warning attached.
- **Exactly:** Alpine uses musl libc, not glibc. PyPI binary wheels for numpy, pandas or torch are built for glibc ("manylinux"), so pip may compile from source or fail.
- **Why it is here:** Explains why ML images use `-slim` and not `alpine`.

### 8. Alpine: same recipe as the good order  (`Dockerfile.alpine` lines 4–8)
```dockerfile
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
- **In simple words:** The rest is the good-order recipe on a different base.
- **Exactly:** `WORKDIR`, `COPY requirements.txt`, `RUN pip install`, `COPY app`, `CMD` as in `Dockerfile.good_order`. Only the base differs.
- **Why it is here:** Changing one thing (the base) is what makes the comparison fair.

### 9. Pinned dependencies  (`requirements.txt` lines 1–2)
```text
fastapi==0.146.0
uvicorn[standard]==0.54.0
```
- **In simple words:** The shopping list, with exact versions.
- **Exactly:** `fastapi==0.146.0` pins an exact version; `uvicorn[standard]==0.54.0` also installs the optional "standard" extras (uvloop, httptools). Editing this file is what invalidates layer A.
- **Why it is here:** Pins make builds reproducible: the same file gives the same image next month.

### 10. Keep junk out of the build context  (`.dockerignore` lines 1–5)
```text
.venv
__pycache__
*.pyc
.git
.env
```
- **In simple words:** A "do not send" list for the builder.
- **Exactly:** Patterns relative to the context root: `.venv`, `__pycache__`, `*.pyc`, `.git`, `.env`. Matching paths are not sent to the Docker daemon and cannot be copied.
- **Why it is here:** Without it `COPY . .` would copy your huge `.venv` (invalidating the cache on every change) and your `.env` secrets into the image.

### 11. Module docstring: what the file is for  (`app/main.py` lines 1–15)
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

### 12. Import FastAPI and BaseModel  (`app/main.py` lines 16–17)
```python
from fastapi import FastAPI            # FastAPI = the web framework (3rd-party). Builds the routes.
from pydantic import BaseModel         # BaseModel = declares + validates the shape of the request body.
```
- **In simple words:** Bring in the two tools we need: one to build the web app, one to describe what a request body must look like.
- **Exactly:** `FastAPI` is the ASGI web-framework class; `BaseModel` is pydantic's base class that validates and parses data from type hints. Both are third-party and come from `requirements.txt` (installed by the Dockerfile).
- **Why it is here:** Without the install step in the Dockerfile this line raises `ModuleNotFoundError` and the container exits at once (see step 05, broken_missing_dependency).

### 13. Create the app object  (`app/main.py` lines 19–19)
```python
app = FastAPI(title="Toy sentiment API")   # `app` is what uvicorn is told to serve ("app.main:app")
```
- **In simple words:** Make the web application. Uvicorn will be told to run this object.
- **Exactly:** `app` is a `FastAPI` instance with `title="Toy sentiment API"` (shown on the auto docs page `/docs`). The name matters: the Dockerfile `CMD` says `app.main:app` = module `app/main.py`, variable `app`.
- **Why it is here:** If the variable were called anything else, uvicorn would fail with "Attribute 'app' not found in module".

### 14. The two keyword sets  (`app/main.py` lines 21–22)
```python
POSITIVE = {"good", "great", "love", "excellent", "happy"}
NEGATIVE = {"bad", "awful", "hate", "terrible", "sad"}
```
- **In simple words:** Two small word lists: happy words and sad words. This is our whole "model".
- **Exactly:** Two `set[str]` constants. Set membership `w in POSITIVE` is O(1) on average.
- **Why it is here:** A keyword model keeps the image tiny and the build fast, so you can study Docker rather than machine learning.

### 15. Request schema  (`app/main.py` lines 25–26)
```python
class PredictRequest(BaseModel):
    text: str                          # type hint = validation rule: body must be {"text": "<string>"}
```
- **In simple words:** A form that says "the request must contain a text field that is a string".
- **Exactly:** Pydantic model with one field `text: str`. FastAPI reads the JSON body, builds `PredictRequest(text=...)`, and answers HTTP 422 with a JSON error if the field is missing or not a string (BREAK IT at the bottom of the file).
- **Why it is here:** Gives free input validation; bad requests never reach your logic.

### 16. Scoring function  (`app/main.py` lines 29–34)
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

### 17. Health endpoint  (`app/main.py` lines 37–39)
```python
@app.get("/health")                    # decorator: "GET /health runs the function below"
def health():
    return {"status": "ok"}            # dict -> JSON automatically
```
- **In simple words:** A "are you alive?" door. It always answers ok.
- **Exactly:** `@app.get("/health")` registers a GET route; the returned dict becomes JSON `{"status":"ok"}` with HTTP 200.
- **Why it is here:** Healthchecks, compose and load balancers poll this URL to know the container is ready (steps 04 and 05).

### 18. Predict endpoint  (`app/main.py` lines 42–45)
```python
@app.post("/predict")
def predict(req: PredictRequest):      # FastAPI parses the JSON body into `req` or answers 422
    label, score = predict_label(req.text)
    return {"label": label, "score": score}
```
- **In simple words:** The real work door: send text, get a label back.
- **Exactly:** `@app.post("/predict")`; the parameter `req: PredictRequest` tells FastAPI to parse and validate the JSON body. Returns `{"label": ..., "score": ...}`, e.g. `{"label":"positive","score":1}` for "I love it".
- **Why it is here:** This is the one endpoint a client actually calls.

### 19. Exercises as comments  (`app/main.py` lines 48–49)
```python
# TRY IT: add a word to POSITIVE and re-run the curl.
# BREAK IT: send {"txt": "hi"} -> 422 Unprocessable Entity. Read the JSON error: it names the missing field.
```
- **In simple words:** Two suggestions for you to try.
- **Exactly:** Comments only: TRY IT (add a word) and BREAK IT (send `{"txt":"hi"}` and read the 422 JSON error).
- **Why it is here:** Learning by changing one thing at a time.


## Syntax explained
| Syntax | Meaning | Example |
|---|---|---|
| `docker build -f FILE` | Use a Dockerfile with a non-default name | `-f Dockerfile.good_order` |
| `CACHED` in build output | Layer reused because instruction + inputs are unchanged | `=> CACHED [3/5] RUN pip install ...` |
| `COPY . .` | Copy whole build context (first `.`) into WORKDIR (second `.`) | cache-busting when anything changes |
| Comment placement | `#` only starts a comment at the *beginning* of a line | `COPY a b # x` would pass `#`, `x` as files! |

## Concepts
| Name | What it does |
|---|---|
| Layer | Filesystem diff produced by one instruction; stacked read-only |
| Cache rule | Docker reuses a layer if the instruction text AND (for COPY) file contents are unchanged AND all previous layers were reused. First change invalidates everything below it |
| `docker history` | Lists layers and sizes |
| `python:3.12` vs `-slim` vs `-alpine` | ~1 GB (tools, gcc) / ~120 MB (Python only) / ~50 MB (musl libc, wheels often missing) |

## Execution trace
- First build of both: every step runs (pip install ~10-30 s).
- Edit one comment in `app/main.py`, rebuild:
  - `bad_order`: `COPY . .` changed -> pip install runs again.
  - `good_order`: `COPY requirements.txt` unchanged -> pip layer `CACHED`; only `COPY app` re-runs (<1 s).
- Typical numbers (not captured, daemon off): bad ~20 s, good ~1 s on rebuild.

## Experiments
1. Add a package to requirements.txt -> good_order re-runs pip (correct: dependencies changed).
2. Move `COPY app` above `RUN pip install` in good_order -> you recreated the bad order.
3. Build `Dockerfile.alpine`, add `numpy` to requirements: observe the slow compile or failure. Why do ML projects avoid alpine?

## Debugging practice
`COPY requirements.txt .` fails with `"/requirements.txt": not found` -> the file is not in the *build context* (wrong folder, or listed in `.dockerignore`). Check the context path (last argument of `docker build`).

## Rebuild checklist
- [ ] Explain why dependencies are copied and installed before application code.
- [ ] Write a `.dockerignore` that excludes `.venv`, `.git`, `__pycache__`, `.env`.

## The new problem (cliffhanger)
The image builds fast now, but it has everything baked in: port, model name, and any data it writes disappears when the container is removed. How do we configure a container without rebuilding, and keep data? Next: [step 03, ports, env and volumes](../03_ports_env_volumes/WALKTHROUGH.md).

## Say it aloud
> **Say it aloud (interview box):** Docker caches each layer, and once an instruction's inputs change that layer and every later one rebuilds. So I copy requirements first, install, then copy source, because code changes constantly and dependencies rarely. The trade-off for base images is that slim is compatible and small, while alpine is smaller but its musl libc breaks many scientific wheels.
