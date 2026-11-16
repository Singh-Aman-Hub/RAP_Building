# Step 06 - Multi-stage builds, small images and security basics

## The story so far
[Step 05](../05_healthchecks_logs_debugging/WALKTHROUGH.md) taught us to diagnose containers. This step makes the image worth shipping.

## The pain point
`Dockerfile.single` installs and runs in one stage: compilers and pip caches stay in the final image, it is large, it has more known vulnerabilities, and the process runs as root. If the app is compromised, the attacker is root in the container.

## The spark
Build in one stage and copy only the results into a clean slim runtime stage (**multi-stage**). Create and switch to a **non-root** user, pin versions, keep secrets out of layers (see `SECURITY_CHECKLIST.md`) and add a `HEALTHCHECK`.

## Goal
Produce a small, non-root, health-checked image and know the security habits interviewers ask about.

## Run it
```bash
cd 11_docker/06_multistage_and_security
cat Dockerfile Dockerfile.single SECURITY_CHECKLIST.md
./commands.sh                     # needs Docker Desktop
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Multi-stage build vs builder pattern | "build stage + runtime stage", "builder image" | Same idea | Multi-stage does it inside one Dockerfile with `COPY --from`; the old builder pattern used two Dockerfiles |
| `CMD` vs `ENTRYPOINT` | default command vs fixed executable | No | `ENTRYPOINT` is the program; `CMD` is its default args and is replaced by `docker run image args` |
| Root vs non-root user | `USER`, "unprivileged", least privilege | No | Root in a container can do more damage if the app is exploited; `USER appuser` limits it |
| `HEALTHCHECK` vs compose `healthcheck:` | probe definition | Same mechanism | Dockerfile bakes it into the image; compose defines it per service and overrides |
| Image size vs attack surface | "smaller image", "fewer packages" | Related, not the same | Size is megabytes; attack surface is the number of installed programs that could have vulnerabilities |

## Code, step by step

### 1. Stage 1: the builder  (`Dockerfile` lines 1–7)
```dockerfile
# ======================= Stage 1: builder =======================
# Has compilers / build tools if wheels need compiling. Everything here is THROWN AWAY except what we COPY out.
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.txt .
# --prefix=/install puts installed packages in one folder that we can copy wholesale into the final image.
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
```
- **In simple words:** A temporary workshop where libraries are installed. We only keep the finished libraries.
- **Exactly:** `FROM python:3.12-slim AS builder` names the stage. `pip install --prefix=/install` puts packages into `/install/lib/python3.12/site-packages` and `/install/bin` so they can be copied as one folder.
- **Why it is here:** Build tools and pip caches stay in this stage and are not in the final image (matters for torch or packages that compile).

### 2. Stage 2: a clean runtime base  (`Dockerfile` lines 9–10)
```dockerfile
# ======================= Stage 2: runtime =======================
FROM python:3.12-slim AS runtime
```
- **In simple words:** Start again from a fresh, clean image.
- **Exactly:** A second `FROM` starts a new image; nothing from the builder exists unless copied. `AS runtime` names it.
- **Why it is here:** The final image is whatever the last stage contains.

### 3. Create a non-root user  (`Dockerfile` lines 12–13)
```dockerfile
# Security: create an unprivileged user. If the app is hacked, the attacker is not root in the container.
RUN useradd --create-home --uid 10001 appuser
```
- **In simple words:** Make an ordinary user with no admin rights.
- **Exactly:** `useradd --create-home --uid 10001 appuser` adds a user with fixed numeric UID 10001, above the reserved range, so volume ownership is predictable.
- **Why it is here:** If the app is compromised, the attacker is not root inside the container.

### 4. Copy the libraries from the builder  (`Dockerfile` lines 15–17)
```dockerfile
WORKDIR /code
# Copy ONLY the installed packages from the builder stage (no pip cache, no build tools).
COPY --from=builder /install /usr/local
```
- **In simple words:** Take only the installed libraries from the workshop.
- **Exactly:** `COPY --from=builder /install /usr/local` merges `/install` into `/usr/local`, where Python finds `site-packages` and `bin`.
- **Why it is here:** This is the multi-stage trick: the final image is smaller and has fewer tools for an attacker.

### 5. Copy the code with correct owner  (`Dockerfile` lines 18–19)
```dockerfile
# --chown so the unprivileged user owns its code.
COPY --chown=appuser:appuser app ./app
```
- **In simple words:** Copy the app and make appuser its owner.
- **Exactly:** `COPY --chown=appuser:appuser app ./app` sets owner and group at copy time (cheaper than a later `chown` which duplicates the layer).
- **Why it is here:** The unprivileged user can read the code it runs.

### 6. Python hygiene environment variables  (`Dockerfile` lines 21–23)
```dockerfile
# Python hygiene: no .pyc files; unbuffered stdout so `docker logs` shows lines immediately.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
```
- **In simple words:** Two small switches for containers.
- **Exactly:** `PYTHONDONTWRITEBYTECODE=1` stops `.pyc` files; `PYTHONUNBUFFERED=1` sends stdout straight out, no buffering.
- **Why it is here:** Without unbuffered output, `docker logs` can lag or miss lines when the process crashes.

### 7. Drop privileges and document the port  (`Dockerfile` lines 25–26)
```dockerfile
USER appuser
EXPOSE 8000
```
- **In simple words:** From now on run as appuser.
- **Exactly:** `USER appuser` applies to later instructions and to the running container. `EXPOSE 8000` is documentation.
- **Why it is here:** Must come after steps that need root (installing, chown).

### 8. Built-in healthcheck  (`Dockerfile` lines 28–30)
```dockerfile
# Docker will mark the container "unhealthy" if /health stops answering.
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')" || exit 1
```
- **In simple words:** Tell Docker how to check that the app still answers.
- **Exactly:** Every 10 s, with 3 s timeout, 5 s start period and 3 retries, run the `urllib` one-liner; failure becomes exit 1 and state `unhealthy` after 3 failures. Visible in `docker ps` as `(healthy)`.
- **Why it is here:** Orchestrators and compose use it to restart or route around bad containers.

### 9. ENTRYPOINT plus CMD  (`Dockerfile` lines 32–34)
```dockerfile
# ENTRYPOINT = the fixed program; CMD = its default arguments (easy to override at `docker run`).
ENTRYPOINT ["uvicorn"]
CMD ["app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
- **In simple words:** A fixed program (uvicorn) with default arguments that are easy to change.
- **Exactly:** Exec-form `ENTRYPOINT ["uvicorn"]` + `CMD ["app.main:app", "--host", "0.0.0.0", "--port", "8000"]` = `uvicorn app.main:app --host 0.0.0.0 --port 8000`. `docker run img app.main:app --port 9000` replaces only the `CMD` part.
- **Why it is here:** Operators can change arguments without retyping the program name.

### 10. Baseline: full image, one stage  (`Dockerfile.single` lines 1–2)
```dockerfile
# Baseline: everything in one stage, runs as root. Compare its size and security with Dockerfile.
FROM python:3.12
```
- **In simple words:** The naive version for comparison.
- **Exactly:** `FROM python:3.12` is the full Debian image (compilers, git; around 1 GB). A comment says "runs as root".
- **Why it is here:** Measure the size and `docker run img whoami` against the multi-stage image.

### 11. Baseline: same steps, no stages, no user  (`Dockerfile.single` lines 3–7)
```dockerfile
WORKDIR /code
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app ./app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
- **In simple words:** Install and run everything in one image as root.
- **Exactly:** `pip install` without `--no-cache-dir` leaves the cache in the layer; no `USER`, no healthcheck.
- **Why it is here:** Shows what the improvements in `Dockerfile` buy: size and security.

### 12. Module docstring: what the file is for  (`app/main.py` lines 1–15)
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

### 13. Import FastAPI and BaseModel  (`app/main.py` lines 16–17)
```python
from fastapi import FastAPI            # FastAPI = the web framework (3rd-party). Builds the routes.
from pydantic import BaseModel         # BaseModel = declares + validates the shape of the request body.
```
- **In simple words:** Bring in the two tools we need: one to build the web app, one to describe what a request body must look like.
- **Exactly:** `FastAPI` is the ASGI web-framework class; `BaseModel` is pydantic's base class that validates and parses data from type hints. Both are third-party and come from `requirements.txt` (installed by the Dockerfile).
- **Why it is here:** Without the install step in the Dockerfile this line raises `ModuleNotFoundError` and the container exits at once (see step 05, broken_missing_dependency).

### 14. Create the app object  (`app/main.py` lines 19–19)
```python
app = FastAPI(title="Toy sentiment API")   # `app` is what uvicorn is told to serve ("app.main:app")
```
- **In simple words:** Make the web application. Uvicorn will be told to run this object.
- **Exactly:** `app` is a `FastAPI` instance with `title="Toy sentiment API"` (shown on the auto docs page `/docs`). The name matters: the Dockerfile `CMD` says `app.main:app` = module `app/main.py`, variable `app`.
- **Why it is here:** If the variable were called anything else, uvicorn would fail with "Attribute 'app' not found in module".

### 15. The two keyword sets  (`app/main.py` lines 21–22)
```python
POSITIVE = {"good", "great", "love", "excellent", "happy"}
NEGATIVE = {"bad", "awful", "hate", "terrible", "sad"}
```
- **In simple words:** Two small word lists: happy words and sad words. This is our whole "model".
- **Exactly:** Two `set[str]` constants. Set membership `w in POSITIVE` is O(1) on average.
- **Why it is here:** A keyword model keeps the image tiny and the build fast, so you can study Docker rather than machine learning.

### 16. Request schema  (`app/main.py` lines 25–26)
```python
class PredictRequest(BaseModel):
    text: str                          # type hint = validation rule: body must be {"text": "<string>"}
```
- **In simple words:** A form that says "the request must contain a text field that is a string".
- **Exactly:** Pydantic model with one field `text: str`. FastAPI reads the JSON body, builds `PredictRequest(text=...)`, and answers HTTP 422 with a JSON error if the field is missing or not a string (BREAK IT at the bottom of the file).
- **Why it is here:** Gives free input validation; bad requests never reach your logic.

### 17. Scoring function  (`app/main.py` lines 29–34)
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

### 18. Health endpoint  (`app/main.py` lines 37–39)
```python
@app.get("/health")                    # decorator: "GET /health runs the function below"
def health():
    return {"status": "ok"}            # dict -> JSON automatically
```
- **In simple words:** A "are you alive?" door. It always answers ok.
- **Exactly:** `@app.get("/health")` registers a GET route; the returned dict becomes JSON `{"status":"ok"}` with HTTP 200.
- **Why it is here:** Healthchecks, compose and load balancers poll this URL to know the container is ready (steps 04 and 05).

### 19. Predict endpoint  (`app/main.py` lines 42–45)
```python
@app.post("/predict")
def predict(req: PredictRequest):      # FastAPI parses the JSON body into `req` or answers 422
    label, score = predict_label(req.text)
    return {"label": label, "score": score}
```
- **In simple words:** The real work door: send text, get a label back.
- **Exactly:** `@app.post("/predict")`; the parameter `req: PredictRequest` tells FastAPI to parse and validate the JSON body. Returns `{"label": ..., "score": ...}`, e.g. `{"label":"positive","score":1}` for "I love it".
- **Why it is here:** This is the one endpoint a client actually calls.

### 20. Exercises as comments  (`app/main.py` lines 48–49)
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
| `FROM x AS builder` | Name a stage | second `FROM` starts a fresh image |
| `COPY --from=builder /install /usr/local` | Copy files out of an earlier stage | |
| `pip install --prefix=/install` | Install to a custom folder | |
| `useradd --create-home --uid 10001 appuser` | Make an unprivileged user | |
| `USER appuser` | Everything after (and the running process) uses it | |
| `COPY --chown=u:g` | Set ownership while copying | |
| `ENTRYPOINT [...]` + `CMD [...]` | fixed program + default args | `docker run img --version` |
| `ENV PYTHONUNBUFFERED=1` | flush prints immediately so `docker logs` is live | |

## Execution trace
1. Stage `builder`: installs deps into `/install`. Stage `runtime`: starts from a clean slim image, copies only `/install` and the code. The final image contains no builder layers.
2. `docker images sec-demo`: `single` (based on full `python:3.12`, ~1 GB) vs `multi` (slim, ~150-200 MB). (Typical values, not captured.)
3. `docker exec sec whoami` -> `appuser`. After ~10 s `docker ps` shows `(healthy)`.
Note: with only pure-Python wheels the multi-stage gain is mostly "slim vs full + no pip cache"; the pattern pays off hugely when builder needs gcc/compilers.

## Experiments
1. Add `numpy` to requirements; compare sizes again.
2. Remove `USER appuser`; run `docker exec sec whoami` -> `root`.
3. Run `docker run --rm sec-demo:multi app.main:app --port 9000` - what replaced what? (CMD args replaced; ENTRYPOINT intact.)

## Debugging practice
`PermissionError: [Errno 13] Permission denied: '/data'` after adding `USER` -> the non-root user cannot write to a root-owned folder. Fix with `RUN mkdir /data && chown appuser /data` before `USER`.

## Rebuild checklist
- [ ] Sketch a 2-stage Dockerfile from memory.
- [ ] List 5 hardening practices.

## The new problem (cliffhanger)
The service is now portable, small and safer, but its only users are programs and people who can type `curl`. A product manager cannot test it. We need a face for the model. Next: [module 12, Streamlit and Gradio](../../12_streamlit_gradio/01_streamlit_basics/WALKTHROUGH.md).

## Say it aloud
> **Say it aloud (interview box):** A multi-stage build installs or compiles in one stage and copies only the results into a clean runtime stage, so the final image is smaller and has no build tools. I also run as a non-root user, pin versions, keep secrets out of layers and add a health check. The trade-off is a more complex Dockerfile and a slower first build, in exchange for a smaller and safer image.
