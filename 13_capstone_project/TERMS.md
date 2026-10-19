# Terms — Capstone: the whole stack

Folder: `13_capstone_project` · Files covered: `api/fastapi_app.py`, `api/config.py`, `api/model_loader.py`, `api/Dockerfile`, `api/requirements.txt`, `api/artifacts/`, `ui/app.py`, `ui/Dockerfile`, `ui/requirements.txt`, `compose.yaml`, `run_local.sh`, `README.md`
Back to module: [13_capstone_project README](README.md)

**Terms in this folder:** [Capstone architecture (UI → API → model) (concept)](#capstone-architecture-ui--api--model-concept) · [FastAPI lifespan (third-party: fastapi)](#fastapi-lifespan-third-party-fastapi) · [asynccontextmanager (standard library: contextlib)](#asynccontextmanager-standard-library-contextlib) · [app.state (third-party: fastapi)](#appstate-third-party-fastapi) · [Pydantic Field validation (third-party: pydantic)](#pydantic-field-validation-third-party-pydantic) · [response_model (third-party: fastapi)](#response_model-third-party-fastapi) · [Sync def endpoint runs in a thread pool (concept)](#sync-def-endpoint-runs-in-a-thread-pool-concept) · [HTTPException 422 for oversized input (third-party: fastapi)](#httpexception-422-for-oversized-input-third-party-fastapi) · [Health endpoint (concept)](#health-endpoint-concept) · [Model version in responses (concept)](#model-version-in-responses-concept) · [time.perf_counter and timed helper (standard library)](#timeperf_counter-and-timed-helper-standard-library) · [*args (syntax)](#args-syntax) · [Environment-based configuration (12-factor) (concept)](#environment-based-configuration-12-factor-concept) · [@dataclass(frozen=True) (standard library: dataclasses)](#dataclassfrozentrue-standard-library-dataclasses) · [python-dotenv and load_dotenv (third-party: python-dotenv)](#python-dotenv-and-load_dotenv-third-party-python-dotenv) · [os.getenv and type conversion (Python built-in module function)](#osgetenv-and-type-conversion-python-built-in-module-function) · [Type hint str | None (syntax)](#type-hint-str--none-syntax) · [Reading settings at call time (concept)](#reading-settings-at-call-time-concept) · [Model contract (predict_proba, classes_) (concept)](#model-contract-predict_proba-classes_-concept) · [KeywordSentimentModel (fallback model) (concept)](#keywordsentimentmodel-fallback-model-concept) · [joblib.load (third-party: joblib)](#joblibload-third-party-joblib) · [Model artifact and metadata.json (concept)](#model-artifact-and-metadatajson-concept) · [Lazy import and lazy loading (concept)](#lazy-import-and-lazy-loading-concept) · [transformers pipeline (text-generation) (third-party: transformers)](#transformers-pipeline-text-generation-third-party-transformers) · [Feature flag (ENABLE_GENERATE) (concept)](#feature-flag-enable_generate-concept) · [async def vs def, and blocking calls (concept)](#async-def-vs-def-and-blocking-calls-concept) · [os.getpid (Python built-in module function)](#osgetpid-python-built-in-module-function) · [Uvicorn (third-party: uvicorn)](#uvicorn-third-party-uvicorn) · [fastapi_app:app (uvicorn import string)](#fastapi_appapp-uvicorn-import-string) · [API Dockerfile (concept)](#api-dockerfile-concept) · [ENV MODEL_PATH in the image (Dockerfile instruction)](#env-model_path-in-the-image-dockerfile-instruction) · [Non-root user in the image (security concept)](#non-root-user-in-the-image-security-concept) · [Layer caching with requirements first (concept)](#layer-caching-with-requirements-first-concept) · [requirements.txt for api and ui (file)](#requirementstxt-for-api-and-ui-file) · [Streamlit UI container (concept)](#streamlit-ui-container-concept) · [API_URL environment variable (concept)](#api_url-environment-variable-concept) · [Docker Compose services (concept)](#docker-compose-services-concept) · [ports HOST:CONTAINER (compose.yaml key)](#ports-hostcontainer-composeyaml-key) · [healthcheck and depends_on service_healthy (compose.yaml keys)](#healthcheck-and-depends_on-service_healthy-composeyaml-keys) · [urllib.request.urlopen in a probe (standard library)](#urllibrequesturlopen-in-a-probe-standard-library) · [run_local.sh (shell script)](#run_localsh-shell-script) · [set -e (shell syntax)](#set--e-shell-syntax) · [Background job and trap (shell syntax)](#background-job-and-trap-shell-syntax) · [python -m uvicorn / python -m streamlit (command)](#python--m-uvicorn--python--m-streamlit-command) · [FastAPI TestClient verification (concept)](#fastapi-testclient-verification-concept) · [Explain-aloud walkthrough (concept)](#explain-aloud-walkthrough-concept)

---

## Capstone architecture (UI → API → model) `(concept)`
**1. Definition:** A two-service design in which a Streamlit UI sends HTTP requests to a FastAPI service, which loads a trained model once at startup and returns predictions as JSON.
**2. In plain words:** The UI only draws the page and the API owns the model. That separation means one model copy, independent scaling and any other client (curl, tests, another app) can reuse the same API.
**3. Example to quote:** > "A request goes browser → Streamlit → POST `/predict` → Pydantic validation → model → JSON → back to the UI."
**Seen in:** `README.md`

## FastAPI lifespan `(third-party: fastapi)`
**1. Definition:** An async context manager passed as `lifespan=` to `FastAPI` whose code before `yield` runs once at start-up and code after it at shutdown.
**2. In plain words:** It's the right place to load the model: once per worker process, before the first request. Per-request loading would add seconds of latency to every call.
**3. Example to quote:** > "My `lifespan` function loads `model.joblib` into `app.state` before the first request is served."
**Seen in:** `api/fastapi_app.py`

## asynccontextmanager `(standard library: contextlib)`
**1. Definition:** A decorator that turns an async generator with one `yield` into an async context manager: set-up before the `yield`, tear-down after it.
**2. In plain words:** It's how you write "start-up ... shut-down" logic in one function. The app serves requests while paused at the `yield`.
**3. Example to quote:** > "I decorate `lifespan` with `@asynccontextmanager` so FastAPI can run my start-up and shutdown code."
**Seen in:** `api/fastapi_app.py`

## app.state `(third-party: fastapi)`
**1. Definition:** An object attached to the application where you can store shared values, such as the loaded model, accessible from request handlers.
**2. In plain words:** It's a handy shelf for things created at start-up. Each worker process has its own copy.
**3. Example to quote:** > "`app.state.model, app.state.source = load_model()` stores the model and where it came from."
**Seen in:** `api/fastapi_app.py`

## Pydantic Field validation `(third-party: pydantic)`
**1. Definition:** `Field(min_length=1, ge=1, le=100, default=...)` attaches constraints and metadata to a model field.
**2. In plain words:** The request schema itself rejects an empty `text` or a `max_new_tokens` outside 1–100 with a 422, before your function runs, and the constraints show up in `/docs`.
**3. Example to quote:** > "Posting `{"text": ""}` returns 422 because `text` has `min_length=1`."
```python
text: str = Field(min_length=1, description="Text to classify")
```
**Seen in:** `api/fastapi_app.py`

## response_model `(third-party: fastapi)`
**1. Definition:** A `response_model=` argument that declares the response schema, which FastAPI uses to validate, filter and document what the endpoint returns.
**2. In plain words:** It guarantees the response has exactly the declared fields (`label`, `confidence`, `model_version`, `latency_ms`) and shows them in the docs.
**3. Example to quote:** > "`@app.post("/predict", response_model=PredictResponse)` documents and enforces the reply format."
**Seen in:** `api/fastapi_app.py`

## Sync def endpoint runs in a thread pool `(concept)`
**1. Definition:** FastAPI runs ordinary `def` endpoints in a worker thread pool so blocking code doesn't freeze the event loop, while `async def` endpoints run on the loop itself.
**2. In plain words:** Model inference is CPU-bound and blocking, so a plain `def` is the safe choice. Using `async def` with a blocking call would stall every other request.
**3. Example to quote:** > "I wrote `predict` as a plain `def` so CPU-heavy model work runs in a thread, not on the event loop."
**Seen in:** `api/fastapi_app.py`

## HTTPException 422 for oversized input `(third-party: fastapi)`
**1. Definition:** Raising `HTTPException(status_code=422, detail=...)` rejects a request with a client-error status and a message.
**2. In plain words:** A maximum text length protects latency and memory. Rejecting giant inputs early is a cheap safety measure.
**3. Example to quote:** > "Text longer than `MAX_TEXT_LENGTH` returns 422 with a message naming the limit."
**Seen in:** `api/fastapi_app.py`

## Health endpoint `(concept)`
**1. Definition:** A lightweight `GET /health` endpoint returning status plus metadata such as the model version and source, used by humans, Docker health checks and load balancers.
**2. In plain words:** It answers "is the service up and which model is it running?" without doing real work.
**3. Example to quote:** > "`curl localhost:8000/health` returns the status, the model version and `file:model.joblib` as the source."
**Seen in:** `api/fastapi_app.py`, `compose.yaml`

## Model version in responses `(concept)`
**1. Definition:** Including an identifier of the model that produced an answer in each response and in `/health`.
**2. In plain words:** When behaviour changes you need to know which model answered. After retraining you bump `MODEL_VERSION` and confirm `/health` shows it.
**3. Example to quote:** > "Every prediction carries `model_version: 0.1.0`, so I can tell which artifact served it."
**Seen in:** `api/fastapi_app.py`, `api/config.py`

## time.perf_counter and timed helper `(standard library)`
**1. Definition:** `time.perf_counter()` returns a high-resolution timer value in seconds; the `timed(fn, *args)` helper uses it to return the result and elapsed milliseconds.
**2. In plain words:** It measures how long a prediction took, which is reported as `latency_ms`. Latency is one of the first numbers asked about for any service.
**3. Example to quote:** > "`(label, confidence), ms = timed(predict_one, model, text)` gives me the prediction and its latency in milliseconds."
**Seen in:** `api/model_loader.py`

## *args `(syntax)`
**1. Definition:** A parameter such as `*args` that collects extra positional arguments into a tuple, and `fn(*args)` expands them again in a call.
**2. In plain words:** It lets `timed` wrap any function without knowing its arguments in advance.
**3. Example to quote:** > "`timed(fn, *args)` calls `fn(*args)`, so it works for any function I pass."
**Seen in:** `api/model_loader.py`

## Environment-based configuration (12-factor) `(concept)`
**1. Definition:** Reading all settings from environment variables so the same code and image run in any environment with different values.
**2. In plain words:** Model path, version, maximum text length and feature flags come from the environment. You change behaviour per environment without editing or rebuilding code.
**3. Example to quote:** > "I set `MODEL_PATH` in Compose and the same image loads a different model elsewhere."
**Seen in:** `api/config.py`, `compose.yaml`

## @dataclass(frozen=True) `(standard library: dataclasses)`
**1. Definition:** The `dataclass` decorator auto-generates `__init__` and `__repr__` for a class of annotated fields; `frozen=True` makes instances immutable.
**2. In plain words:** `Settings` is a tidy bundle of values that can't be changed accidentally after creation. It's less code than writing `__init__` by hand.
**3. Example to quote:** > "My `Settings` dataclass is frozen, so no part of the app can quietly change the configuration."
**Seen in:** `api/config.py`

## python-dotenv and load_dotenv `(third-party: python-dotenv)`
**1. Definition:** `load_dotenv()` reads `KEY=value` lines from a `.env` file into `os.environ`; by default real environment variables win over the file.
**2. In plain words:** It's a convenient way to keep local settings in a file. In containers you normally pass real environment variables instead.
**3. Example to quote:** > "I call `load_dotenv()` at the top of `config.py`, so a local `.env` file fills in settings when I develop."
**Seen in:** `api/config.py`

## os.getenv and type conversion `(Python built-in module function)`
**1. Definition:** `os.getenv(name, default)` returns the variable as a string or the default; numbers and booleans must be converted explicitly.
**2. In plain words:** Environment variables are always strings, so `int(os.getenv("MAX_TEXT_LENGTH", "500"))` converts it and `== "1"` turns a flag into a boolean.
**3. Example to quote:** > "`enable_generate=os.getenv("ENABLE_GENERATE", "0") == "1"` makes the flag a real boolean."
**Seen in:** `api/config.py`

## Type hint `str | None` `(syntax)`
**1. Definition:** A union type annotation meaning the value is either a string or `None`.
**2. In plain words:** `model_path: str | None` says "a path, or nothing, in which case use the fallback model".
**3. Example to quote:** > "`model_path` is `None` when `MODEL_PATH` is empty, so the loader uses the keyword fallback."
**Seen in:** `api/config.py`

## Reading settings at call time `(concept)`
**1. Definition:** Reading the environment inside a function, `get_settings()`, rather than once at import time.
**2. In plain words:** Tests can then change environment variables with `monkeypatch` and see the effect immediately.
**3. Example to quote:** > "I call `get_settings()` inside each request so a test can change `MAX_TEXT_LENGTH` on the fly."
**Seen in:** `api/config.py`

## Model contract (predict_proba, classes_) `(concept)`
**1. Definition:** An agreed interface where any model object offers `predict_proba(list_of_texts)` returning one probability row per text, and `classes_` listing the labels in column order.
**2. In plain words:** Any object that follows these two rules can be swapped in: a scikit-learn pipeline, or the keyword fallback class. The API code never needs to know which.
**3. Example to quote:** > "Both my scikit-learn pipeline and the dependency-free fallback satisfy the contract, so I can swap them with `MODEL_PATH`."
**Seen in:** `api/model_loader.py`

## KeywordSentimentModel (fallback model) `(concept)`
**1. Definition:** A dependency-free class that counts positive and negative keywords and applies add-one smoothing to produce probabilities.
**2. In plain words:** It lets the API start and be tested before any model is trained. Smoothing, `(pos + 1) / (pos + neg + 2)`, avoids probabilities of exactly 0 or 1.
**3. Example to quote:** > "With no `MODEL_PATH` set, the API still answers using keyword counting, reporting `fallback-keyword` as the source."
**Seen in:** `api/model_loader.py`

## joblib.load `(third-party: joblib)`
**1. Definition:** Loads a Python object, such as a trained scikit-learn model, that was saved to disk with `joblib.dump`.
**2. In plain words:** It restores the trained pipeline from `model.joblib`. Only load files you trust, because loading pickled data can run arbitrary code. A wrong path raises `FileNotFoundError`.
**3. Example to quote:** > "At start-up `joblib.load(path)` restores my trained classifier from `artifacts/model.joblib`."
**Seen in:** `api/model_loader.py`

## Model artifact and metadata.json `(concept)`
**1. Definition:** The serialised trained model file (`model.joblib`) together with a metadata file describing it, stored in `api/artifacts/`.
**2. In plain words:** Training produces an artifact that serving consumes. The metadata records what the model is, which supports versioning and retraining.
**3. Example to quote:** > "I retrained, copied the new `model.joblib` into `api/artifacts/` and bumped `MODEL_VERSION`."
**Seen in:** `api/artifacts/`, `README.md`

## Lazy import and lazy loading `(concept)`
**1. Definition:** Importing heavy libraries or loading big models only inside the function that needs them, the first time it's called.
**2. In plain words:** `joblib` is imported only when a model file is used, and `transformers` only when `/generate` is called. The service starts fast and doesn't download 330 MB unless you ask.
**3. Example to quote:** > "`/generate` is opt-in; distilgpt2 loads only the first time someone calls it with `ENABLE_GENERATE=1`."
**Seen in:** `api/model_loader.py`

## transformers pipeline (text-generation) `(third-party: transformers)`
**1. Definition:** `pipeline("text-generation", model="distilgpt2")` wraps a tokenizer and a language model into a callable that generates text from a prompt.
**2. In plain words:** It's the quickest way to run a small language model. Passing `max_new_tokens` bounds the length and `do_sample=False` makes the output deterministic (greedy).
**3. Example to quote:** > "The optional `/generate` endpoint uses `pipeline("text-generation", model="distilgpt2")` with greedy decoding."
**Seen in:** `api/model_loader.py`

## Feature flag (ENABLE_GENERATE) `(concept)`
**1. Definition:** A configuration switch that turns an optional feature on or off without changing code.
**2. In plain words:** The `/generate` endpoint is disabled by default and returns 503 until you set `ENABLE_GENERATE=1`, so nobody pays the download and memory cost accidentally.
**3. Example to quote:** > "Without the flag, `/generate` replies 503 'generation disabled'."
**Seen in:** `api/fastapi_app.py`, `api/config.py`

## async def vs def, and blocking calls `(concept)`
**1. Definition:** `async def` handlers run on the event loop and should use `await` for waiting; a blocking call such as `time.sleep` inside one freezes the whole loop.
**2. In plain words:** The helper endpoints `/slow`, `/slow-sync` and `/slow-blocking` demonstrate that: `await asyncio.sleep` lets other requests proceed, a plain `def` runs in a thread, but blocking inside `async def` stalls everything. They exist to explore workers and concurrency (module 09).
**3. Example to quote:** > "`/slow-blocking` is an anti-pattern: `async def` with `time.sleep` blocks the event loop for every user."
**Seen in:** `api/fastapi_app.py`

## os.getpid `(Python built-in module function)`
**1. Definition:** Returns the current process's numeric ID.
**2. In plain words:** `/whoami` returns the PID, so with several workers you can see which process answered each request.
**3. Example to quote:** > "Calling `/whoami` repeatedly with two workers shows two different PIDs."
**Seen in:** `api/fastapi_app.py`

## Uvicorn `(third-party: uvicorn)`
**1. Definition:** An ASGI server that runs async Python web apps such as FastAPI.
**2. In plain words:** FastAPI defines the routes and uvicorn listens on the port. In the capstone `run_local.sh` starts it, and the Dockerfile `CMD` starts it inside the container.
**3. Example to quote:** > "The framework is FastAPI and the server is uvicorn: `uvicorn fastapi_app:app --port 8000`."
**Seen in:** `api/Dockerfile`, `run_local.sh`

## fastapi_app:app `(uvicorn import string)`
**1. Definition:** A `module:variable` string telling uvicorn to import the `app` object from `fastapi_app.py`.
**2. In plain words:** If the module name or variable is wrong, uvicorn fails with an import error at start-up.
**3. Example to quote:** > "`uvicorn fastapi_app:app` serves the `app` variable inside `fastapi_app.py`."
**Seen in:** `api/Dockerfile`, `run_local.sh`

## API Dockerfile `(concept)`
**1. Definition:** A Dockerfile that starts from `python:3.12-slim`, installs pinned dependencies in a cached layer, copies the code, sets `MODEL_PATH`, creates a non-root user and starts uvicorn on `0.0.0.0:8000`.
**2. In plain words:** Each choice maps to something from module 11: layer order for fast rebuilds, `ENV` for configuration, `USER` for safety, and `0.0.0.0` so the published port is reachable.
**3. Example to quote:** > "My API image copies `requirements.txt` first, so a code edit doesn't reinstall the dependencies."
**Seen in:** `api/Dockerfile`

## ENV MODEL_PATH in the image `(Dockerfile instruction)`
**1. Definition:** `ENV MODEL_PATH=/app/artifacts/model.joblib` sets a default environment variable inside the image, overridable at run time.
**2. In plain words:** The baked-in model path makes the container work out of the box, while `-e` or Compose `environment:` can point it elsewhere.
**3. Example to quote:** > "The image defaults to the bundled model, and Compose also sets `MODEL_PATH` explicitly."
**Seen in:** `api/Dockerfile`, `compose.yaml`

## Non-root user in the image `(security concept)`
**1. Definition:** `RUN useradd --create-home appuser` followed by `USER appuser` makes the container process run without root privileges.
**2. In plain words:** If the service were compromised, the attacker would have limited rights inside the container.
**3. Example to quote:** > "Both Dockerfiles end by switching to `appuser`, so nothing runs as root."
**Seen in:** `api/Dockerfile`, `ui/Dockerfile`

## Layer caching with requirements first `(concept)`
**1. Definition:** Copying `requirements.txt` and running `pip install` before copying the rest of the code so the dependency layer stays cached.
**2. In plain words:** Dependencies change rarely and code changes often, so this ordering makes rebuilds after a code edit nearly instant.
**3. Example to quote:** > "I `COPY requirements.txt .` and install first, then `COPY . .`."
**Seen in:** `api/Dockerfile`, `ui/Dockerfile`

## requirements.txt for api and ui `(file)`
**1. Definition:** Lists of the Python packages each image installs; the API needs `fastapi`, `uvicorn`, `pydantic`, `joblib`, `scikit-learn` and `numpy`, the UI lists `streamlit` and `requests`.
**2. In plain words:** An image contains only what its requirements list. A package that the code imports but the list omits fails at start-up inside the container, even though it works on your laptop.
**3. Example to quote:** > "If the code imports a library that's missing from `requirements.txt`, the container exits with a ModuleNotFoundError."
**Seen in:** `api/requirements.txt`, `ui/requirements.txt`

## Streamlit UI container `(concept)`
**1. Definition:** The UI image runs `streamlit run app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true`.
**2. In plain words:** It binds to all interfaces so the published port works, and headless stops Streamlit trying to open a browser inside the container.
**3. Example to quote:** > "The UI's `CMD` uses `--server.address=0.0.0.0` for the same reason the API uses `--host 0.0.0.0`."
**Seen in:** `ui/Dockerfile`

## API_URL environment variable `(concept)`
**1. Definition:** The variable the UI reads to find the API: `http://localhost:8000` by default, `http://api:8000` in Compose.
**2. In plain words:** Inside the UI container `localhost` means that container itself, so Compose uses the service name `api` instead.
**3. Example to quote:** > "In Compose I set `API_URL: http://api:8000`; with `localhost` the UI would get connection refused."
**Seen in:** `compose.yaml`, `ui/app.py`

## Docker Compose services `(concept)`
**1. Definition:** `compose.yaml` here defines two services, `api` (built from `./api`, port 8000) and `ui` (built from `./ui`, port 8501), on a shared network.
**2. In plain words:** One `docker compose up --build` builds both images and starts them. Service names double as hostnames, so the UI reaches the API at `http://api:8000`.
**3. Example to quote:** > "`docker compose up --build` starts my API and UI together; `docker compose down` stops them."
**Seen in:** `compose.yaml`

## ports HOST:CONTAINER `(compose.yaml key)`
**1. Definition:** `ports: ["8000:8000"]` publishes a container port on the same host port, with the host number first.
**2. In plain words:** It's what makes the services reachable from your browser; container-to-container traffic doesn't need it.
**3. Example to quote:** > "I publish 8000 for the API and 8501 for the UI so I can reach both from my laptop."
**Seen in:** `compose.yaml`

## healthcheck and depends_on service_healthy `(compose.yaml keys)`
**1. Definition:** A `healthcheck` runs a probe inside the container (here a Python one-liner using `urllib` against `/health`), and `depends_on` with `condition: service_healthy` delays the UI until the API reports healthy.
**2. In plain words:** "Started" doesn't mean "ready". The UI waits until the model is loaded and `/health` answers, so it never talks to an API that isn't ready.
**3. Example to quote:** > "The UI starts only after the API is healthy, because of `depends_on: api: condition: service_healthy`."
**Seen in:** `compose.yaml`

## urllib.request.urlopen in a probe `(standard library)`
**1. Definition:** Opens a URL and raises an exception on failure; as a health probe its non-zero exit signals unhealthy.
**2. In plain words:** `python:slim` doesn't include curl, so the probe uses Python itself.
**3. Example to quote:** > "My health check is `python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"`."
**Seen in:** `compose.yaml`

## run_local.sh `(shell script)`
**1. Definition:** A Bash script that starts the API in the background, waits a few seconds, then runs the Streamlit UI against it, stopping the API when the script exits.
**2. In plain words:** It lets you run the whole capstone without Docker using the repo's virtual environment. It uses `set -e`, a background job, `trap` and `kill` to clean up.
**3. Example to quote:** > "`./run_local.sh` gives me the API on port 8000 and the UI on 8501, and Ctrl+C stops both."
**Seen in:** `run_local.sh`

## set -e `(shell syntax)`
**1. Definition:** A Bash option that makes the script exit immediately if any command fails.
**2. In plain words:** It stops a script from continuing in a broken state after an error.
**3. Example to quote:** > "With `set -e`, if the API fails to start the script stops instead of launching a UI with nothing behind it."
**Seen in:** `run_local.sh`

## Background job and trap `(shell syntax)`
**1. Definition:** A trailing `&` runs a command in the background; `$!` is its process ID; `trap "kill $API" EXIT` runs the given command when the script exits.
**2. In plain words:** The API runs in the background while the UI runs in the foreground, and the trap guarantees the API is stopped afterwards.
**3. Example to quote:** > "I save the background API's PID in `API=$!` and use `trap` to kill it on exit."
**Seen in:** `run_local.sh`

## python -m uvicorn / python -m streamlit `(command)`
**1. Definition:** Runs a package's command-line entry point through the chosen Python interpreter with `-m`.
**2. In plain words:** Using `../.venv/bin/python -m ...` ensures the repo's virtual environment is used, regardless of what `python` points to on your machine.
**3. Example to quote:** > "I use `$PY -m uvicorn` where `PY=../.venv/bin/python`, so the right packages are always found."
**Seen in:** `run_local.sh`

## FastAPI TestClient verification `(concept)`
**1. Definition:** Using `fastapi.testclient.TestClient` inside a `with` block to run the app (including its lifespan) in-process and check `/health`, `/predict` and the 422 case.
**2. In plain words:** It proves the API works without starting a server or Docker. The capstone's API was verified this way; the Docker and Compose runs were not.
**3. Example to quote:** > "`/health` returned the version, `/predict` returned a label and confidence, and an empty text returned 422."
**Seen in:** `README.md`

## Explain-aloud walkthrough `(concept)`
**1. Definition:** A five-minute spoken script covering what the system does, its diagram, one request's path, where validation, config and model loading happen, scaling concerns and one trade-off.
**2. In plain words:** It turns the project into an interview story. The trade-off to name is that the toy model is fast but weak, and swapping in an LLM raises memory and latency.
**3. Example to quote:** > "When requests grow, each worker loads its own model copy, so I'd batch, add replicas or move the model into a dedicated service."
**Seen in:** `README.md`
