# Terms — Step 06: The same model API in Flask and FastAPI

Folder: `09_api_development/06_model_api_flask_vs_fastapi` · Files covered: `main.py`, `config.py`, `model_loader.py`, `flask_app.py`, `fastapi_app.py`, `.env.example`, `WALKTHROUGH.md`
Back to module: [09 API development](../README.md)

**Terms in this folder:** [Model-serving API](#model-serving-api-concept) · [Health endpoint (/health)](#health-endpoint-health-concept) · [Model contract (predict_proba, classes_)](#model-contract-predict_proba-classes_-concept) · [predict_proba](#predict_proba-third-party-scikit-learn) · [classes_](#classes_-third-party-scikit-learn) · [KeywordSentimentModel (fallback model)](#keywordsentimentmodel-fallback-model-concept) · [Add-one (Laplace) smoothing](#add-one-laplace-smoothing-concept) · [Environment variables for config](#environment-variables-for-config-concept) · [os.getenv](#osgetenv-standard-library) · [python-dotenv / load_dotenv](#python-dotenv-load_dotenv-third-party-python-dotenv) · [@dataclass(frozen=True)](#dataclassfrozentrue-standard-library) · [Settings read at call time (get_settings)](#settings-read-at-call-time-get_settings-concept) · [Optional type hint (str | None)](#optional-type-hint-str-none-syntax) · [`or None` idiom](#or-none-idiom-syntax) · [joblib.load](#joblibload-third-party-joblib) · [Lazy import](#lazy-import-concept) · [Lazy loading with a cached global (_GENERATOR)](#lazy-loading-with-a-cached-global-_generator-concept) · [transformers.pipeline](#transformerspipeline-third-party-transformers) · [max(range(n), key=probs.__getitem__)](#maxrangen-keyprobs__getitem__-syntax) · [Argmax](#argmax-concept) · [Sum of booleans](#sum-of-booleans-syntax) · [time.perf_counter](#timeperf_counter-standard-library) · [*args and function-as-argument](#args-and-function-as-argument-syntax) · [Load model once at startup](#load-model-once-at-startup-concept) · [Module-level code runs on import](#module-level-code-runs-on-import-concept) · [os.getpid](#osgetpid-standard-library) · [print(..., flush=True)](#print-flushtrue-python-built-in) · [Flask error handlers (404, 405, Exception)](#flask-error-handlers-404-405-exception-third-party-flask) · [503 Service Unavailable](#503-service-unavailable-concept) · [FastAPI lifespan](#fastapi-lifespan-third-party-fastapi) · [@asynccontextmanager and yield](#asynccontextmanager-and-yield-standard-library) · [app.state](#appstate-third-party-fastapi) · [response_model](#response_model-third-party-fastapi) · [Field(min_length, ge, le, description)](#fieldmin_length-ge-le-description-third-party-pydantic) · [def vs async def routes](#def-vs-async-def-routes-concept) · [asyncio.sleep vs time.sleep](#asynciosleep-vs-timesleep-standard-library) · [Gunicorn command (flask_app:app --workers)](#gunicorn-command-flask_appapp---workers-command) · [TestClient as a context manager](#testclient-as-a-context-manager-third-party-fastapi) · [repr format !r](#repr-format-r-syntax)

---

## Model-serving API  `(concept)`
**1. Definition:** A web service that loads a trained model into memory once and exposes it over HTTP endpoints (here `/health` and `/predict`) so other programs can request predictions.
**2. In plain words:** Instead of everyone running the model themselves, one service holds the model and answers requests. A health endpoint shows it is alive, and a predict endpoint takes input and returns the model's answer plus metadata like the model version. This is the standard way a model is shipped.
**3. Example to quote:**
> "I serve a sentiment classifier at `POST /predict`. It returns the label, confidence, model version and latency, and `GET /health` lets a monitor check it is up."

**Seen in:** whole step

## Health endpoint (/health)  `(concept)`
**1. Definition:** A lightweight endpoint that returns a small status payload showing the service is running and which model it loaded.
**2. In plain words:** Load balancers, Docker health checks and humans call it to see if the app is alive without running a prediction. Here it also reports the model version and source, so you can verify a deployment loaded the right model.
**3. Example to quote:**
> "`GET /health` returns `{'status': 'ok', 'model_version': '0.1.0', 'model_source': 'fallback-keyword'}`, so I know which model is live."

**Seen in:** `health` in both apps

## Model contract (predict_proba, classes_)  `(concept)`
**1. Definition:** The minimal interface the loader expects of a model: `predict_proba(list_of_texts)` returns one probability row per text, and `classes_` lists the label names in the same column order.
**2. In plain words:** Anything with those two members can be served, whether a scikit-learn Pipeline or the tiny fallback class. This keeps the API code independent of the model type, so you can swap models without touching endpoints.
**3. Example to quote:**
> "My API only needs `predict_proba` and `classes_`, so the fallback keyword model and the scikit-learn pipeline from module 03 are interchangeable."

**Seen in:** `model_loader.py` docstring

## predict_proba  `(third-party: scikit-learn)`
**1. Definition:** A scikit-learn classifier method returning an array of class probabilities for each input row, columns ordered as in `classes_`.
**2. In plain words:** Instead of just a label it tells how confident the model is in each class. The API uses the largest probability as the "confidence" it reports.
**3. Example to quote:**
> "`model.predict_proba(['great film'])` might return `[[0.25, 0.75]]`, so with `classes_ = ['negative', 'positive']` the answer is positive at 0.75."

**Seen in:** `predict_one`, `KeywordSentimentModel`

## classes_  `(third-party: scikit-learn)`
**1. Definition:** An attribute of a fitted classifier listing the class labels in the order of the `predict_proba` columns.
**2. In plain words:** It maps column positions back to names. Without it you would not know whether column 1 means positive or negative.
**3. Example to quote:**
> "I take the index of the largest probability and return `model.classes_[best]` to get the label name."

**Seen in:** `predict_one`

## KeywordSentimentModel (fallback model)  `(concept)`
**1. Definition:** A dependency-free class in `model_loader.py` that scores text by counting positive and negative keywords, implementing the same contract as a real model.
**2. In plain words:** It lets the API run before you have trained anything, and keeps the tests independent of large files. It uses add-one smoothing so probabilities are never exactly 0 or 1.
**3. Example to quote:**
> "With no `MODEL_PATH` set, the API uses a keyword model: 'I love this great film' has two positive words and no negative, so p_pos = 3/4 = 0.75."

**Seen in:** `model_loader.py`

## Add-one (Laplace) smoothing  `(concept)`
**1. Definition:** Adding 1 to each count (and 2 to the total for two classes) so no probability is exactly 0 or 1.
**2. In plain words:** With zero keywords, the model answers 0.5 ("don't know") rather than dividing by zero or claiming certainty. The formula `(pos + 1) / (pos + neg + 2)` gives that.
**3. Example to quote:**
> "For 'It was a film' there are no keywords, so `(0+1)/(0+0+2)` gives 0.5, an honest 'no idea'."

**Seen in:** `KeywordSentimentModel.predict_proba`

## Environment variables for config  `(concept)`
**1. Definition:** Settings read from the process environment (`os.environ`) rather than hard-coded, so the same code can run with different configuration in different places.
**2. In plain words:** Changing `MODEL_PATH` or `MODEL_VERSION` changes behaviour without editing code. All environment variables are strings, so numbers and booleans must be converted. This is how Docker and cloud platforms configure apps.
**3. Example to quote:**
> "`MODEL_PATH=.../model.joblib python main.py` serves the real model; with nothing set it uses the fallback. Same code, different config."

**Seen in:** `config.py`, `.env.example`

## os.getenv  `(standard library)`
**1. Definition:** `os.getenv(name, default=None)` returns the value of an environment variable as a string, or the default if it is not set.
**2. In plain words:** It never raises for a missing variable, which is handy for optional settings. Because the result is always a string you must convert it, e.g. `int(os.getenv("MAX_TEXT_LENGTH", "500"))`.
**3. Example to quote:**
> "`os.getenv('MODEL_VERSION', '0.1.0')` gives the version from the environment or falls back to 0.1.0."

```python
import os
print(os.getenv("NOPE", "default"))   # default
```

**Seen in:** `get_settings`

## python-dotenv / load_dotenv  `(third-party: python-dotenv)`
**1. Definition:** `dotenv.load_dotenv()` reads `KEY=value` lines from a `.env` file and puts them into `os.environ`; by default real environment variables take precedence.
**2. In plain words:** It gives you convenient local configuration without exporting variables by hand. `.env` files can hold secrets, so they must never be committed; `.env.example` shows the format without real values.
**3. Example to quote:**
> "I keep `MODEL_VERSION` and `MAX_TEXT_LENGTH` in a local `.env`, loaded by `load_dotenv()`, and commit only `.env.example`."

**Seen in:** `config.py`, `.env.example`

## @dataclass(frozen=True)  `(standard library)`
**1. Definition:** The `dataclasses.dataclass` decorator generates `__init__`, `__repr__` and comparison methods from annotated fields; `frozen=True` makes instances immutable.
**2. In plain words:** It is a short way to write a plain data holder like `Settings`. Freezing prevents code from changing the settings by accident after they are read.
**3. Example to quote:**
> "`Settings` is a frozen dataclass with `model_path`, `model_version`, `max_text_length` and `enable_generate`; trying to assign to it raises an error."

```python
from dataclasses import dataclass
@dataclass(frozen=True)
class P:
    x: int
print(P(1))   # P(x=1)
```

**Seen in:** `config.py`

## Settings read at call time (get_settings)  `(concept)`
**1. Definition:** A function that reads the environment each time it is called, instead of once at import, returning a fresh `Settings` object.
**2. In plain words:** Reading at call time lets tests change environment variables (for example with `monkeypatch`) and see the effect immediately. Reading at import would freeze the values at program start.
**3. Example to quote:**
> "`get_settings()` re-reads the environment on each call, so a test can set `MAX_TEXT_LENGTH=5` and the next request sees it."

**Seen in:** `config.py`

## Optional type hint (str | None)  `(syntax)`
**1. Definition:** The annotation `str | None` (Python 3.10+) means a value that is either a string or `None`.
**2. In plain words:** `model_path: str | None` documents that "no path" is allowed and means "use the fallback model".
**3. Example to quote:**
> "`model_path: str | None` says the setting may be absent, in which case the API falls back to the keyword model."

**Seen in:** `config.py`

## `or None` idiom  `(syntax)`
**1. Definition:** `os.getenv("X") or None` converts any falsy value, such as an empty string, to `None`.
**2. In plain words:** An empty `MODEL_PATH=""` should mean "not set", not a path of nothing. The `or None` makes the later `if path:` check simple.
**3. Example to quote:**
> "`os.getenv('MODEL_PATH') or None` turns an empty variable into `None`, so I use the fallback model."

**Seen in:** `get_settings`

## joblib.load  `(third-party: joblib)`
**1. Definition:** Loads a Python object (for example a scikit-learn pipeline) previously saved with `joblib.dump`.
**2. In plain words:** It is how the API gets a trained model from a file. It raises `FileNotFoundError` if the path is wrong, so a bad `MODEL_PATH` fails at startup. Like pickle, only load files you trust.
**3. Example to quote:**
> "If `MODEL_PATH` is set I call `joblib.load(path)` to get the trained pipeline from module 03."

**Seen in:** `load_model`

## Lazy import  `(concept)`
**1. Definition:** Importing a module inside a function, only when it is actually needed, rather than at the top of the file.
**2. In plain words:** Importing `transformers` or `joblib` costs time and memory. Importing them inside the function that needs them means users who never call `/generate` never pay that cost.
**3. Example to quote:**
> "I import `transformers` inside `generate_text`, so the API starts quickly and only loads it if someone uses `/generate`."

**Seen in:** `load_model`, `generate_text`

## Lazy loading with a cached global (_GENERATOR)  `(concept)`
**1. Definition:** A module-level variable starts as `None`; the first call creates the expensive object and stores it, and later calls reuse it.
**2. In plain words:** Loading distilgpt2 takes seconds and about 330 MB. Doing it on first use and caching it means startup stays fast and the cost is paid only once.
**3. Example to quote:**
> "`_GENERATOR` is `None` until the first `/generate` call; then I build the pipeline once and reuse it."

**Seen in:** `generate_text`

## transformers.pipeline  `(third-party: transformers)`
**1. Definition:** A Hugging Face helper `pipeline(task, model=...)` that bundles a tokenizer and model for a task such as `"text-generation"`.
**2. In plain words:** One line gives a ready-to-use generator. `do_sample=False` makes it greedy (deterministic) and `max_new_tokens` limits the output length. Details are in modules 04 and 05.
**3. Example to quote:**
> "`pipeline('text-generation', model='distilgpt2')` loads the tokenizer and model together, and calling it with a prompt returns generated text."

**Seen in:** `generate_text`

## max(range(n), key=probs.__getitem__)  `(syntax)`
**1. Definition:** Finds the index of the largest item by asking `max` to compare indexes using the value at each index (`key` is the function giving the comparison value).
**2. In plain words:** `max(probs)` returns the biggest value, but we also need *where* it is, to look up the label. Using `key=probs.__getitem__` ranks indexes by their probability, which is a dependency-free "argmax".
**3. Example to quote:**
> "With `probs = [0.25, 0.75]`, `max(range(2), key=probs.__getitem__)` returns 1, the index of the positive class."

```python
probs = [5, 9, 1]
print(max(range(3), key=probs.__getitem__))   # 1
```

**Seen in:** `predict_one`

## Argmax  `(concept)`
**1. Definition:** The index (or input) at which a sequence of values is largest.
**2. In plain words:** For a classifier, argmax over class probabilities picks the predicted class. Language models use the same idea when choosing the single most likely next token (greedy decoding).
**3. Example to quote:**
> "The predicted label is the argmax of the probabilities; confidence is the probability at that index."

**Seen in:** `predict_one`

## Sum of booleans  `(syntax)`
**1. Definition:** In Python `True` counts as 1 and `False` as 0, so `sum(cond for ...)` counts how many items satisfy a condition.
**2. In plain words:** `sum(w in POSITIVE for w in words)` counts the positive words in one line.
**3. Example to quote:**
> "I count positive words with `sum(w.strip('.,!?') in self.POSITIVE for w in words)`, with each match adding 1."

```python
print(sum(x > 1 for x in [1, 2, 3]))   # 2
```

**Seen in:** `KeywordSentimentModel.predict_proba`

## time.perf_counter  `(standard library)`
**1. Definition:** Returns a high-resolution timer value in seconds, suitable for measuring elapsed time between two calls.
**2. In plain words:** Subtract two readings to get how long something took; multiply by 1000 for milliseconds. It is better than `time.time()` for timing because it is monotonic.
**3. Example to quote:**
> "`timed(fn, *args)` measures how long `predict_one` takes and the API returns it as `latency_ms`."

```python
import time
t = time.perf_counter(); time.sleep(0.01); print(time.perf_counter() - t > 0)
```

**Seen in:** `timed`

## *args and function-as-argument  `(syntax)`
**1. Definition:** `def timed(fn, *args)` takes a function and any number of positional arguments, and calls `fn(*args)` by unpacking them.
**2. In plain words:** The helper works with any function, because functions are ordinary values in Python. It returns both the result and the elapsed milliseconds.
**3. Example to quote:**
> "`timed(predict_one, MODEL, text)` calls `predict_one(MODEL, text)` and gives me `(result, ms)`."

**Seen in:** `timed`

## Load model once at startup  `(concept)`
**1. Definition:** Creating the model object when the process starts (module level in Flask, lifespan in FastAPI) and reusing it for every request.
**2. In plain words:** Loading a model can take seconds, so doing it per request would ruin latency. The trade-off is that each worker process holds its own copy, so memory grows with the worker count.
**3. Example to quote:**
> "Under Gunicorn with 2 workers the load line runs twice, once per process, so the model sits in memory twice."

**Seen in:** `flask_app.py` comment, `WALKTHROUGH.md`

## Module-level code runs on import  `(concept)`
**1. Definition:** Statements at the top level of a Python file execute when the file is imported, once per process.
**2. In plain words:** Flask's `MODEL, MODEL_SOURCE = load_model()` runs as soon as the module is imported by the server worker. If the model path is wrong, the process fails immediately, which is a good "fail fast" behaviour.
**3. Example to quote:**
> "A wrong `MODEL_PATH` raises `FileNotFoundError` at import, so deployment fails at startup, not at the first user request."

**Seen in:** `flask_app.py`

## os.getpid  `(standard library)`
**1. Definition:** Returns the current process's id number.
**2. In plain words:** Printing the pid shows which worker process handled a request, which proves multiple workers exist and each loaded the model.
**3. Example to quote:**
> "`/whoami` returns the pid so I can see requests being served by different worker processes."

**Seen in:** `/whoami`, startup print

## print(..., flush=True)  `(Python built-in)`
**1. Definition:** `flush=True` forces the text to be written out immediately instead of waiting in the output buffer.
**2. In plain words:** Under servers and Docker, buffered output can appear late or not before a crash. Flushing makes startup messages show up right away in the logs.
**3. Example to quote:**
> "I print the pid and model source with `flush=True` so the line appears in the server log at startup."

**Seen in:** `flask_app.py`, `fastapi_app.py`

## Flask error handlers (404, 405, Exception)  `(third-party: flask)`
**1. Definition:** Functions registered with `@app.errorhandler(...)` that build the response for HTTP errors or unhandled exceptions.
**2. In plain words:** They replace default HTML error pages with JSON. The catch-all `Exception` handler logs the traceback on the server and returns a generic 500, so clients never see internals.
**3. Example to quote:**
> "The `Exception` handler calls `app.logger.exception(...)` so the traceback goes to my logs, while the client gets only `internal server error`."

**Seen in:** `flask_app.py`

## 503 Service Unavailable  `(concept)`
**1. Definition:** HTTP status meaning the server cannot handle the request right now, for example because a feature is disabled or it is overloaded.
**2. In plain words:** `/generate` is opt-in because it downloads about 330 MB; when it is off the API returns 503 with a hint, rather than failing in a confusing way.
**3. Example to quote:**
> "`/generate` returns 503 'generation disabled; start with ENABLE_GENERATE=1' until I turn the feature on."

**Seen in:** `/generate` handlers

## FastAPI lifespan  `(third-party: fastapi)`
**1. Definition:** An async context manager passed as `FastAPI(lifespan=...)`; code before `yield` runs once at startup and code after it at shutdown.
**2. In plain words:** It is the proper place to load the model or open connections. Each worker process runs it once. The `with TestClient(app)` form is needed in tests so it actually runs.
**3. Example to quote:**
> "In `lifespan` I set `app.state.model, app.state.source = load_model()`, then `yield` while the app serves requests."

**Seen in:** `fastapi_app.py`

## @asynccontextmanager and yield  `(standard library)`
**1. Definition:** `contextlib.asynccontextmanager` turns an async generator with one `yield` into an async context manager: setup before `yield`, teardown after.
**2. In plain words:** It is a compact way to write "do this at start, that at the end". The app serves requests while paused at `yield`.
**3. Example to quote:**
> "My lifespan function loads the model, then `yield`s; anything after the `yield` would run on shutdown."

**Seen in:** `fastapi_app.py`

## app.state  `(third-party: fastapi)`
**1. Definition:** A simple namespace on the FastAPI app for storing objects shared across requests, such as the loaded model.
**2. In plain words:** It avoids global variables: handlers read `app.state.model`. Each worker process has its own `app.state`.
**3. Example to quote:**
> "I store the model on `app.state.model` at startup and pass it to `predict_one` in the endpoint."

**Seen in:** `fastapi_app.py`

## response_model  `(third-party: fastapi)`
**1. Definition:** A route option naming a Pydantic model that describes and filters the response; FastAPI validates and serialises the return value to that shape and documents it.
**2. In plain words:** It guarantees the response has exactly the declared fields, and `/docs` shows them. It also prevents accidentally leaking extra data.
**3. Example to quote:**
> "`response_model=PredictResponse` documents that I return `label`, `confidence`, `model_version` and `latency_ms`."

**Seen in:** `@app.post("/predict", response_model=PredictResponse)`

## Field(min_length, ge, le, description)  `(third-party: pydantic)`
**1. Definition:** Pydantic's `Field` attaches constraints (`min_length`, `ge`, `le`, `default`) and a documentation `description` to a model field.
**2. In plain words:** `max_new_tokens` is limited to 1 to 100 so a client cannot request a huge generation. The description also appears in `/docs`.
**3. Example to quote:**
> "`max_new_tokens: int = Field(default=20, ge=1, le=100)` stops a client from asking for 10,000 tokens."

**Seen in:** `GenerateRequest`, `PredictRequest`

## def vs async def routes  `(concept)`
**1. Definition:** In FastAPI, a plain `def` route runs in a worker thread pool, while an `async def` route runs directly on the event loop.
**2. In plain words:** CPU-heavy model code should be in plain `def` so it does not block the event loop. An `async def` handler that makes a blocking call (such as `time.sleep`) freezes all other requests on that loop, as `/slow-blocking` demonstrates.
**3. Example to quote:**
> "My `/predict` is a plain `def`, so FastAPI runs the model in a thread and the event loop stays free."

**Seen in:** `predict`, `slow`, `slow_sync`, `slow_blocking`

## asyncio.sleep vs time.sleep  `(standard library)`
**1. Definition:** `await asyncio.sleep(n)` pauses only the current coroutine, letting the event loop run others; `time.sleep(n)` blocks the entire thread.
**2. In plain words:** Inside `async def`, only the awaitable form is safe. That is why `/slow` stays responsive while `/slow-blocking`, which uses `time.sleep` in an async function, freezes the whole server worker.
**3. Example to quote:**
> "`/slow-blocking` uses `time.sleep` inside `async def`: while it sleeps, no other request on that worker gets served."

**Seen in:** `fastapi_app.py` slow endpoints

## Gunicorn command (flask_app:app --workers)  `(command)`
**1. Definition:** `gunicorn flask_app:app --workers 2 --bind 127.0.0.1:5001` runs the WSGI app `app` from `flask_app.py` with 2 worker processes.
**2. In plain words:** Gunicorn is the production server for Flask. Each worker is a separate process with its own memory, so each loads its own model copy.
**3. Example to quote:**
> "In production I run `gunicorn flask_app:app --workers 2`; I'd see my startup print twice, once per pid."

```bash
gunicorn flask_app:app --workers 2 --bind 127.0.0.1:5001
```

**Seen in:** `flask_app.py` docstring

## TestClient as a context manager  `(third-party: fastapi)`
**1. Definition:** Using `with TestClient(app) as client:` triggers the app's startup and shutdown (lifespan) events; without `with`, they do not run.
**2. In plain words:** If you forget the `with`, the model is never loaded and requests fail because `app.state.model` is missing. This is a common testing gotcha.
**3. Example to quote:**
> "I use `with TestClient(fastapi_app) as fast_client` so the lifespan loads the model before my requests."

**Seen in:** `main.py`

## repr format !r  `(syntax)`
**1. Definition:** In an f-string, `{value!r}` inserts the value's `repr()`, which includes quotes for strings.
**2. In plain words:** It makes the sentence in the table visibly quoted, which is helpful when text could contain spaces or be empty.
**3. Example to quote:**
> "`f\"{text!r:<32}\"` prints `'I love this great film'` with quotes, padded to 32 characters."

```python
print(f"{'hi'!r}")   # 'hi'
```

**Seen in:** `main.py`
