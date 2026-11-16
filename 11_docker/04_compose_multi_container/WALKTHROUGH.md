# Step 04 - Docker Compose: API + UI + Redis

## The story so far
[Step 03](../03_ports_env_volumes/WALKTHROUGH.md) configured one container. Our system needs three: an API, a Streamlit UI and Redis as a cache.

## The pain point
By hand you start Redis, wait, start the API with the right Redis address, wait, start the UI with the API address. Inside a container `localhost` means that container itself, so the UI cannot reach the API through it. One typo and the stack is half-alive.

## The spark
Describe the whole stack once in a `compose.yaml`. Compose creates a shared network where each service name is a DNS name (`http://api:8000`, host `redis`), and `depends_on` with a health condition starts services only when the one they need is ready.

## Goal
Run three cooperating containers with one command and understand how they find each other.

## Run it
```bash
cd 11_docker/04_compose_multi_container
docker compose config > /dev/null && echo valid     # works without the daemon (verified when written)
./commands.sh                                        # needs Docker Desktop
```
Local (no Docker) check of the API in memory mode, real output:
```
{'label': 'negative', 'score': -1} {'predictions': 1, 'counter_backend': 'memory'} {'status': 'ready'}
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Compose service vs container | "service", "container", "replica" | No | A service is the definition in YAML; a container is one running instance of it (a service can have several) |
| `docker compose` vs `docker-compose` | Compose v2 vs v1 | Mostly | `docker compose` (plugin, with a space) is current; `docker-compose` (hyphen, Python) is the retired v1 |
| Liveness vs readiness | `/health` vs `/ready`, "alive" vs "ready" | No | Liveness: process is up; readiness: its dependencies work and it can serve |
| `depends_on` (plain) vs `condition: service_healthy` | start order, wait for ready | No | Plain `depends_on` waits for start only; `service_healthy` waits for the healthcheck to pass |
| Named volume vs bind mount | `redis-data:/data` vs `./data:/data` | No | Named volume is Docker-managed; bind mount maps a host path |

## Code, step by step

### 1. Top-level layout  (`compose.yaml` lines 1–3)
```yaml
# Docker Compose file: describes a WHOLE application (several containers) in one YAML file.
# Top-level keys: services (containers), volumes (named volumes), networks (optional; Compose makes a default one).
services:
```
- **In simple words:** A header comment, then the list of containers that make up the app.
- **Exactly:** YAML top-level keys: `services` (containers), `volumes` (named volumes), optionally `networks`. Compose creates one default network for all services.
- **Why it is here:** One file replaces three long `docker run` commands.

### 2. redis service and its volume  (`compose.yaml` lines 4–7)
```yaml
  redis:                                  # service name = DNS hostname on the compose network
    image: ${REDIS_IMAGE}                 # use a ready-made image instead of building one
    volumes:
      - redis-data:/data                  # named volume: counter survives `docker compose down`
```
- **In simple words:** Use a ready-made Redis image and give it a disk that survives.
- **Exactly:** `image: ${REDIS_IMAGE}` is filled from `.env` (`redis:7-alpine`). `redis-data:/data` mounts named volume `redis-data` at `/data` (where Redis writes). The service name `redis` is its DNS hostname on the Compose network.
- **Why it is here:** Without the volume, `docker compose down` and `up` resets the counter.

### 3. redis healthcheck  (`compose.yaml` lines 8–12)
```yaml
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]  # healthy when redis answers PONG
      interval: 5s
      timeout: 3s
      retries: 5
```
- **In simple words:** A little test that asks Redis "are you OK?" every few seconds.
- **Exactly:** `test: ["CMD", "redis-cli", "ping"]` runs inside the container; exit code 0 (answer PONG) means healthy. `interval: 5s` between probes, `timeout: 3s` per probe, `retries: 5` failures before `unhealthy`.
- **Why it is here:** Gives `depends_on: service_healthy` something to wait for.

### 4. api service: build and env  (`compose.yaml` lines 14–17)
```yaml
  api:
    build: ./api                          # build from ./api/Dockerfile
    environment:
      REDIS_HOST: redis                   # <- the service name above
```
- **In simple words:** Build the API from our own Dockerfile and tell it where Redis lives.
- **Exactly:** `build: ./api` uses `./api/Dockerfile`; build context is `./api`. `REDIS_HOST: redis` is the service name above; the app reads it with `os.getenv`.
- **Why it is here:** Hard-coding an IP would break because container IPs change; the name resolves through Docker DNS.

### 5. api ports and start order  (`compose.yaml` lines 18–22)
```yaml
    ports:
      - "${API_PORT}:8000"                # HOST:CONTAINER, like docker run -p
    depends_on:
      redis:
        condition: service_healthy        # wait for redis HEALTHY, not just started
```
- **In simple words:** Open the API to your laptop and make it wait for Redis.
- **Exactly:** `"${API_PORT}:8000"` is HOST:CONTAINER (8000:8000 from `.env`). `depends_on.redis.condition: service_healthy` delays the API container start until redis is healthy.
- **Why it is here:** Without it the API may start first and the first `/predict` fails with a connection error.

### 6. api healthcheck  (`compose.yaml` lines 23–29)
```yaml
    healthcheck:
      # python is already in the image; curl is NOT in python:slim, so we use urllib instead.
      test: ["CMD", "python", "-c", "import urllib.request as u; u.urlopen('http://localhost:8000/health')"]
      interval: 10s
      timeout: 3s
      retries: 3
      start_period: 5s                    # grace period while the app boots
```
- **In simple words:** A probe that checks whether the API answers.
- **Exactly:** Runs `python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"`; a non-2xx or connection error raises, exit code 1. `curl` is not in `python:slim`, so `urllib` is used. `start_period: 5s` gives a grace period during boot, failures do not count.
- **Why it is here:** The UI waits for this; without a healthcheck compose cannot know the API is ready.

### 7. ui service  (`compose.yaml` lines 31–36)
```yaml
  ui:
    build: ./ui
    environment:
      API_URL: http://api:8000            # container-to-container: service name, CONTAINER port
    ports:
      - "${UI_PORT}:8501"
```
- **In simple words:** Build the web page container and point it at the API by name.
- **Exactly:** `API_URL: http://api:8000` uses the service name and the CONTAINER port (not the host-published one). `"${UI_PORT}:8501"` publishes Streamlit's port.
- **Why it is here:** Inside the ui container, `localhost` is the ui container itself, so `http://localhost:8000` would fail.

### 8. ui start order  (`compose.yaml` lines 37–39)
```yaml
    depends_on:
      api:
        condition: service_healthy
```
- **In simple words:** Start the UI only after the API is healthy.
- **Exactly:** `depends_on: api: condition: service_healthy`.
- **Why it is here:** Avoids a UI that shows errors during the first seconds.

### 9. Declare the named volume  (`compose.yaml` lines 41–42)
```yaml
volumes:
  redis-data:
```
- **In simple words:** Tell Compose that `redis-data` exists.
- **Exactly:** Top-level `volumes:` entry with empty body = default local driver.
- **Why it is here:** Referencing an undeclared volume is a compose error.

### 10. Compose variable file  (`.env` lines 1–4)
```text
# Compose automatically reads ./.env to fill ${VARIABLES} inside compose.yaml.
API_PORT=8000
UI_PORT=8501
REDIS_IMAGE=redis:7-alpine
```
- **In simple words:** Settings Compose fills into `${...}` placeholders.
- **Exactly:** `API_PORT=8000`, `UI_PORT=8501`, `REDIS_IMAGE=redis:7-alpine`. Compose reads `./.env` automatically (it is not the same as an `env_file:` for the container).
- **Why it is here:** Change a port or an image version in one place.

### 11. API image recipe  (`api/Dockerfile` lines 1–7)
```dockerfile
FROM python:3.12-slim
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
- **In simple words:** The same recipe as step 01.
- **Exactly:** Base slim, `WORKDIR /code`, requirements before code, `EXPOSE 8000`, uvicorn on `0.0.0.0:8000`.
- **Why it is here:** Good cache order and reachable bind address.

### 12. UI image recipe up to EXPOSE  (`ui/Dockerfile` lines 1–6)
```dockerfile
FROM python:3.12-slim
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
EXPOSE 8501
```
- **In simple words:** Same as the API but copying a single file `app.py`.
- **Exactly:** `COPY app.py .` copies the file into `/code/`; `EXPOSE 8501` is Streamlit's default port.
- **Why it is here:** Same cache logic.

### 13. UI start command  (`ui/Dockerfile` lines 7–8)
```dockerfile
# --server.address 0.0.0.0 for the same reason as uvicorn's --host: be reachable from outside the container.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
- **In simple words:** Start Streamlit listening on all interfaces.
- **Exactly:** `streamlit run app.py --server.port=8501 --server.address=0.0.0.0`. Streamlit defaults to localhost-only, like uvicorn.
- **Why it is here:** Without the address flag the UI is not reachable via `-p`.

### 14. UI docstring  (`ui/app.py` lines 1–7)
```python
"""
GOAL: A minimal Streamlit UI that talks to the API container. Shows how one container calls another.

The API address is NOT localhost: inside the ui container, "localhost" means the ui container itself.
We use the Compose service name instead:  http://api:8000   (given via the API_URL env var).
RUN locally against a locally running API:  API_URL=http://localhost:8000 streamlit run ui/app.py
"""
```
- **In simple words:** Explains why the API address is not localhost.
- **Exactly:** Inside the ui container `localhost` = itself. The Compose service name works instead.
- **Why it is here:** This is the most common beginner multi-container mistake.

### 15. Imports and API URL  (`ui/app.py` lines 8–13)
```python
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
```
- **In simple words:** Load libraries and read where the API is.
- **Exactly:** `os.getenv("API_URL", "http://localhost:8000")` gives the Compose value in a container and a local default otherwise.
- **Why it is here:** Same file runs locally and in Compose.

### 16. Title and text box  (`ui/app.py` lines 15–16)
```python
st.title("Sentiment UI")
text = st.text_input("Type a sentence", "I love containers")
```
- **In simple words:** Draw a title and an input box with a default sentence.
- **Exactly:** `st.text_input(label, value)` returns the current `str` on every rerun.
- **Why it is here:** The user input.

### 17. Call the API and show the result  (`ui/app.py` lines 17–22)
```python
if st.button("Predict"):
    try:
        r = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=5)
        r.raise_for_status()                      # turns 4xx/5xx into an exception
        st.json(r.json())
        st.caption(f"API stats: {requests.get(f'{API_URL}/stats', timeout=5).json()}")
```
- **In simple words:** When the button is clicked, send the text to the API and show the answer.
- **Exactly:** `requests.post(url, json=..., timeout=5)` sends JSON. `raise_for_status()` turns 4xx/5xx into an exception. `st.json` prints the response; the second call `GET /stats` shows the counter and backend (`redis` or `memory`).
- **Why it is here:** `timeout=5` stops the UI from hanging forever when the API is down.

### 18. Show a friendly error  (`ui/app.py` lines 23–24)
```python
    except requests.RequestException as exc:
        st.error(f"Could not reach the API at {API_URL}: {exc}")
```
- **In simple words:** If the API cannot be reached, say so.
- **Exactly:** `requests.RequestException` is the base class of connection and HTTP errors; `st.error` shows a red box.
- **Why it is here:** A beginner sees the cause (wrong URL) instead of a stack trace.

### 19. API docstring  (`api/app/main.py` lines 1–9)
```python
"""
GOAL: Toy sentiment API that counts predictions in REDIS (a second container) - the classic multi-container setup.

RUN locally without redis (falls back to an in-memory counter):
    cd 11_docker/04_compose_multi_container/api && ../../../.venv/bin/python -m uvicorn app.main:app --port 8000
RUN for real:  docker compose up --build   (from 04_compose_multi_container)
EXPECTED OUTPUT: curl localhost:8000/stats -> {"predictions":N,"counter_backend":"redis"|"memory"}
NEEDS: fastapi, uvicorn; `redis` package only when REDIS_HOST is set (listed in api/requirements.txt).
"""
```
- **In simple words:** What the file does and how to run it.
- **Exactly:** Counts predictions in Redis; falls back to an in-memory counter when `REDIS_HOST` is not set.
- **Why it is here:** Lets you test without Docker.

### 20. Imports  (`api/app/main.py` lines 10–13)
```python
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
```
- **In simple words:** Load the libraries.
- **Exactly:** `os`, `FastAPI`, `HTTPException` (to return 503), `BaseModel`.
- **Why it is here:** `HTTPException` is how a route returns a status code other than 200.

### 21. App and keyword sets  (`api/app/main.py` lines 15–17)
```python
app = FastAPI(title="Toy API + Redis")
POSITIVE = {"good", "great", "love", "excellent", "happy"}
NEGATIVE = {"bad", "awful", "hate", "terrible", "sad"}
```
- **In simple words:** Create the app and the word lists.
- **Exactly:** Same toy model as step 01.
- **Why it is here:** Keep the focus on Docker.

### 22. Connect to Redis only if configured  (`api/app/main.py` lines 19–26)
```python
# In Compose, the redis service is reachable at the HOSTNAME "redis" (= the service name). That is
# Docker's built-in DNS for user-defined networks. We pass it in via an env var so the code stays portable.
REDIS_HOST = os.getenv("REDIS_HOST")
_memory_counter = 0
_redis = None
if REDIS_HOST:
    import redis                                   # 3rd-party client library, imported only when needed
    _redis = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)  # decode_responses: bytes -> str
```
- **In simple words:** If a Redis hostname was provided, create a client; otherwise stay in memory.
- **Exactly:** `REDIS_HOST` is `None` or `str`. `import redis` runs only inside the `if`, so the package is optional locally. `redis.Redis(host=..., port=6379, decode_responses=True)` creates a lazy client; the connection is opened on the first command, and `decode_responses` returns `str` instead of `bytes`.
- **Why it is here:** The service name `redis` resolves via Docker's built-in DNS on the Compose network.

### 23. Request schema  (`api/app/main.py` lines 29–30)
```python
class PredictRequest(BaseModel):
    text: str
```
- **In simple words:** A form with one text field.
- **Exactly:** Pydantic `PredictRequest`.
- **Why it is here:** Validation.

### 24. Increment the counter  (`api/app/main.py` lines 33–38)
```python
def incr() -> int:
    global _memory_counter                          # `global`: we assign to a module-level variable
    if _redis is not None:
        return _redis.incr("predictions")           # atomic +1 stored inside redis
    _memory_counter += 1
    return _memory_counter
```
- **In simple words:** Add one to the counter, in Redis if available, else in a variable.
- **Exactly:** `redis.incr("predictions")` is an atomic server-side +1 that returns the new int. `global _memory_counter` is needed to assign the module-level variable.
- **Why it is here:** Atomic increments stay correct with several API workers.

### 25. Liveness endpoint  (`api/app/main.py` lines 41–43)
```python
@app.get("/health")
def health():
    return {"status": "ok"}
```
- **In simple words:** "Is the process up?".
- **Exactly:** GET `/health` always returns `{"status":"ok"}`; used by the compose healthcheck.
- **Why it is here:** A cheap probe that does not depend on Redis.

### 26. Readiness endpoint  (`api/app/main.py` lines 46–54)
```python
@app.get("/ready")
def ready():
    """Readiness = 'can I do real work?'. Differs from /health ('is the process alive?')."""
    if _redis is not None:
        try:
            _redis.ping()
        except Exception as exc:                    # broad except is OK here: any failure = not ready
            raise HTTPException(status_code=503, detail=f"redis unavailable: {exc}")
    return {"status": "ready"}
```
- **In simple words:** "Can I really do work?" It asks Redis.
- **Exactly:** `_redis.ping()` raises if Redis is unreachable; the route then raises `HTTPException(503, ...)`. Without Redis (local mode) it returns `{"status":"ready"}`.
- **Why it is here:** Liveness vs readiness: a load balancer should stop sending traffic when readiness fails, but not necessarily restart the container.

### 27. Predict endpoint  (`api/app/main.py` lines 57–63)
```python
@app.post("/predict")
def predict(req: PredictRequest):
    words = req.text.lower().split()
    score = sum(w in POSITIVE for w in words) - sum(w in NEGATIVE for w in words)
    label = "positive" if score > 0 else "negative" if score < 0 else "neutral"
    incr()
    return {"label": label, "score": score}
```
- **In simple words:** Score, count, answer.
- **Exactly:** Same scoring as step 01, plus `incr()` after computing the label.
- **Why it is here:** The counter is the shared state that shows both containers work together.

### 28. Stats endpoint  (`api/app/main.py` lines 66–69)
```python
@app.get("/stats")
def stats():
    n = int(_redis.get("predictions") or 0) if _redis is not None else _memory_counter
    return {"predictions": n, "counter_backend": "redis" if _redis is not None else "memory"}
```
- **In simple words:** Show how many predictions have been made and where the count is stored.
- **Exactly:** `_redis.get("predictions")` returns `None` before the first predict, so `or 0`. Response `{"predictions":N,"counter_backend":"redis"|"memory"}`.
- **Why it is here:** Proves Redis is really in use.

### 29. Exercises as comments  (`api/app/main.py` lines 71–72)
```python
# TRY IT: `docker compose stop redis` then call /ready -> 503. `docker compose start redis` -> 200 again.
# BREAK IT: set REDIS_HOST=wrongname -> /predict raises redis ConnectionError (read the traceback in `docker compose logs api`).
```
- **In simple words:** Stop Redis and watch readiness fail.
- **Exactly:** TRY IT: `docker compose stop redis` makes `/ready` return 503. BREAK IT: `REDIS_HOST=wrongname` makes `/predict` raise a redis `ConnectionError`.
- **Why it is here:** See how failure looks in logs.


## Syntax explained (YAML + Compose)
| Syntax | Meaning | Example |
|---|---|---|
| YAML indentation | Structure (2 spaces, no tabs) | `services:` then `  api:` |
| `- item` | list element | `- "8000:8000"` |
| `key: value` | mapping | `REDIS_HOST: redis` |
| `${VAR}` | filled from `.env` / shell | `${API_PORT}` |
| `build: ./api` vs `image: redis:7-alpine` | build locally vs pull | |
| `depends_on.condition: service_healthy` | start order based on health | |
| `volumes:` (top level) | declares named volumes | `redis-data:` |

## Commands
| Command | Does | Notes |
|---|---|---|
| `docker compose up --build -d` | build + start all | `-d` detached |
| `docker compose ps` | status incl. health | |
| `docker compose logs -f api` | follow one service | |
| `docker compose exec api sh` | shell in running service | |
| `docker compose down [-v]` | stop+remove (+volumes) | `-v` deletes data |

## Execution trace
1. Compose creates a project network (bridge, user-defined). Every service joins it and gets a DNS name = service name.
2. `redis` starts; its healthcheck (`redis-cli ping`) turns healthy.
3. `api` starts (depends_on healthy), reads `REDIS_HOST=redis`; `redis.Redis(host="redis")` resolves via Docker DNS.
4. `ui` starts once api is healthy; browser (host) -> `localhost:8501` -> ui container -> `http://api:8000` (container port, not the host-mapped one) -> redis.
5. `down` removes containers; `redis-data` volume keeps the counter.
Important distinction: the *browser* uses published host ports; *containers* talk over the network by service name and container port.

## Experiments
1. Change `API_URL` in the ui service to `http://localhost:8000`: the UI fails ("connection refused"). Why? (localhost = the ui container.)
2. `docker compose stop redis`; call `/ready` -> 503; `/health` still 200.
3. Remove the volume mapping; `down` then `up`: the counter resets.
4. Delete the `ports:` of `api`: UI still works, `curl localhost:8000` does not (internal-only service).

## Debugging practice
`dependency failed to start: container ...-redis-1 is unhealthy` -> run `docker compose logs redis` and `docker inspect --format '{{json .State.Health}}' <container>` to see the last health probe outputs.

## Rebuild checklist
- [ ] Draw the network: browser, ui, api, redis, with the hostnames and ports used on each arrow.
- [ ] Write a 2-service compose file from memory.

## The new problem (cliffhanger)
One `docker compose up` works, until it does not: a container is unhealthy, exits at once, or is up but unreachable. Now you need a method, not luck, to find out why. Next: [step 05, health checks, logs and debugging](../05_healthchecks_logs_debugging/WALKTHROUGH.md).

## Say it aloud
> **Say it aloud (interview box):** Compose describes a multi-container app declaratively, and its services share a network and reach each other by service name, so the API connects to host redis while only published ports reach my laptop. depends_on with a health condition orders start-up, because started does not mean ready. The trade-off is that Compose suits dev and single-host deploys but has no multi-node scaling or self-healing like Kubernetes.
