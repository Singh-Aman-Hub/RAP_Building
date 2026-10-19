# Module 09 — API Development: Flask + Gunicorn and FastAPI + Uvicorn

*(Syllabus topic 6.)*

## 1. Why this matters
A trained model is useless until another program can call it. In the interview you will be asked to build a health-check and a POST endpoint, validate input, return correct status codes, and — the favourite follow-up — **explain what the framework does vs what the server does**. This module gives you both frameworks, side by side, serving a real model.

## 2. Where this fits
**Builds on**
- [00_python_foundations](../00_python_foundations/) — functions, dicts, exceptions, decorators, type hints (every route is a function with a decorator; Pydantic *is* type hints).
- [01_debugging_and_environment](../01_debugging_and_environment/) — reading tracebacks, venvs, `ModuleNotFoundError`, "address already in use". You will use all of them here.
- [03_ml_foundations](../03_ml_foundations/) — the saved scikit-learn pipeline (`08_sklearn_text_classifier/artifacts/model.joblib`) that `/predict` serves through the `MODEL_PATH` environment variable.
- [04_llm_fundamentals](../04_llm_fundamentals/) / [05_llm_parameters](../05_llm_parameters/) — the optional `/generate` endpoint wraps a Hugging Face `pipeline` with `max_new_tokens`.
- [08_agent_harness](../08_agent_harness/) — HTTP/JSON thinking: an LLM API call is just an HTTP POST with a JSON body, exactly like the requests you send here, and a tool can be "call this API".

**Leads to**
- [10_ml_system_architecture](../10_ml_system_architecture/) — this API is the "serving" box in the architecture; module 10 adds registry, logging, monitoring, load tests.
- [11_docker](../11_docker/) — you will containerise *exactly* `06_model_api_flask_vs_fastapi` (same `uvicorn fastapi_app:app` command, `MODEL_PATH`/ports from this module's config).
- [12_streamlit_gradio](../12_streamlit_gradio/) — the UI calls `POST /predict` with `httpx`/`requests`.
- [13_capstone_project](../13_capstone_project/) — everything wired together.

## 3. Concept map
```
                 HTTP (step 01): method + path + headers + body  ->  status + headers + body
                                  |
        REST conventions: nouns in URLs, verbs = methods, status codes tell the outcome (step 05)
                                  |
   ┌──────────────────────────────┴───────────────────────────────┐
   │ FRAMEWORK (your code: routes, validation)                    │  SERVER (sockets, processes)
   │  Flask (03)   ── speaks WSGI ──┐                             │  Gunicorn  ─ workers = processes
   │  FastAPI (04) ── speaks ASGI ──┤  contract (02)              │  Uvicorn   ─ event loop
   └───────────────┬────────────────┴─────────────────────────────┘        │ (07)
                   │                                                        │
        validation (05): manual vs Pydantic                  concurrency: sync vs async vs processes
                   │                                                        │
        the model API (06): load model ONCE per worker  ────────  memory x workers, load testing (10/08)
                   │
        config via env vars (08)  ──►  Docker ENV / -e (11)
                   │
        tests: pytest + curl + /docs (09)
```

## 4. Study path
| # | Folder | You learn | Run | Time |
|---|---|---|---|---|
| 01 | [01_http_basics](01_http_basics/) | request/response anatomy, status codes, stdlib server | `python main.py` | 45 min |
| 02 | [02_wsgi_asgi_by_hand](02_wsgi_asgi_by_hand/) | WSGI vs ASGI contracts, sync vs async | `python main.py` | 60 min |
| 03 | [03_flask_basics](03_flask_basics/) | routes, args, JSON, 201/404 | `python main.py` | 90 min |
| 04 | [04_fastapi_basics](04_fastapi_basics/) | type-hint routing, Pydantic body, `/docs` | `python main.py` | 90 min |
| 05 | [05_validation_and_errors](05_validation_and_errors/) | Pydantic, 400 vs 422, error handlers | `python main.py` | 60 min |
| 06 | [06_model_api_flask_vs_fastapi](06_model_api_flask_vs_fastapi/) | **the model API** in both frameworks, config, lazy loading | `python main.py` | 120 min |
| 07 | [07_servers_workers_concurrency](07_servers_workers_concurrency/) | Gunicorn/Uvicorn, workers, threads, async | `python main.py` | 90 min |
| 08 | [08_config_and_env](08_config_and_env/) | env vars, `.env`, precedence | `python main.py` | 45 min |
| 09 | [09_testing_and_docs](09_testing_and_docs/) | pytest, httpx smoke test, curl, OpenAPI | `pytest -v` / `python main.py` | 75 min |

Total: **~11–12 hours** + [exercises](exercises.md). Always run from the folder shown (or pass the full path).

## 5. The syllabus topics in plain English

**HTTP and REST basics.** HTTP is request/response. A *request* = method (verb) + path + headers + optional body; a *response* = status code + headers + body. REST is a naming convention: URLs are nouns (`/items/7`), methods are verbs (GET read, POST create, PUT/PATCH update, DELETE remove). *Limitation:* REST doesn't fit everything (streaming tokens from an LLM uses Server-Sent Events/WebSockets).

**GET vs POST.** GET reads, has no body, can be cached/bookmarked, data in the URL (don't put secrets there). POST sends a body, used when the request changes something or the input is large — which is why `/predict` is a POST even though it doesn't change state (the text can be long and shouldn't sit in URLs/logs).

**Routes and methods.** A route = (method, path) → function. Same path with different methods are different routes; wrong method → 405.

**Request and response handling.** Read input from path, query string, headers or JSON body; return JSON with a status code. Query values are strings until converted.

**Input validation and Pydantic.** Check type, range and size before the model sees it. Pydantic declares those rules as a class; FastAPI runs them automatically (422). *Trade-off:* strictness vs convenience (Pydantic converts `"5"` → `5` for int fields in lax mode).

**Error handling and status codes.** 4xx → client must change something; 5xx → server's fault. Return JSON errors with a helpful message; never leak stack traces; log them server-side.

**Flask fundamentals.** Micro-framework on WSGI. Tiny, flexible, you assemble validation/docs yourself. *Limitation:* sync-first, manual validation.

**FastAPI fundamentals.** ASGI framework using type hints; automatic validation, OpenAPI docs, async support. *Limitation:* async pitfalls (blocking calls), more concepts.

**WSGI vs ASGI.** Two contracts between framework and server. WSGI = sync callable; ASGI = async callable with scope/receive/send, supports WebSockets. You wrote both by hand in step 02.

**Gunicorn and Uvicorn.** Servers that actually listen on the port. Gunicorn: pre-fork process manager for WSGI (can also manage Uvicorn workers). Uvicorn: fast ASGI server. The Flask/FastAPI built-in dev servers are not for production.

**Workers and concurrency.** Worker = OS process with its own memory. More workers → more parallel requests *and* more RAM (one model copy each). Threads share memory but the GIL limits CPU-bound Python. Async overlaps waiting. Pick by workload: I/O-bound → async/threads; CPU-bound model → processes/replicas/batching.

**Environment variables.** Settings outside the code; strings only; secrets never in git; `.env` for local convenience.

**API testing and documentation.** In-process test clients (fast), smoke tests on a real server, curl for ad-hoc probing, `/docs` and `/openapi.json` for FastAPI.

## 6. Common mistakes and gotchas
- Port 5000 busy on macOS (AirPlay) → we use 5001.
- `async def` route calling a blocking function (`time.sleep`, `requests.get`, a heavy model call) → freezes every request on that worker.
- Loading the model **inside** the request handler → hundreds of ms per call; load once at startup.
- `debug=True` / `--reload` in production.
- Forgetting `Content-Type: application/json` with curl → Flask's `get_json` returns `None`/415.
- Storing state in a global dict and running several workers → each worker has different state.
- `uvicorn main:app` run from the wrong directory → `ModuleNotFoundError`.
- Relative `MODEL_PATH` depends on the **current working directory**, not on the script location.
- Returning 200 with `{"error": ...}` — clients and monitors rely on status codes.

## 7. Self-check
- [ ] I can write a health-check endpoint in Flask and FastAPI without notes.
- [ ] I can write a POST endpoint that validates input and returns 201/422/400 correctly.
- [ ] I can explain GET vs POST and give two reasons `/predict` is POST.
- [ ] I can explain WSGI vs ASGI and who implements each side.
- [ ] I can start the same app with Gunicorn and Uvicorn and say what `app:app` means.
- [ ] I can explain why more workers = more memory for a model service.
- [ ] I can predict timings for sync / threaded / async handlers (step 07).
- [ ] I can load config from environment variables and explain precedence with `.env`.
- [ ] I can run pytest, a curl POST, and open `/docs`.
- [ ] I can swap the fallback model for a real joblib model via `MODEL_PATH`.
- [ ] I can explain the role of the framework and the server separately.

## 8. Interview questions
1. **What is the difference between Flask and FastAPI?** Flask: WSGI micro-framework, sync, manual validation. FastAPI: ASGI, async-capable, automatic Pydantic validation and OpenAPI docs. Choose FastAPI for new typed APIs/async I/O; Flask for small/simple or existing code. Both can serve ML models equally well — inference speed is the model's, not the framework's.
2. **What does Gunicorn do that `flask run` doesn't?** Manages multiple worker processes, restarts crashed workers, handles concurrency and graceful reloads; the dev server is single-purpose for development and not hardened.
3. **WSGI vs ASGI?** Sync callable `(environ, start_response)` vs async callable `(scope, receive, send)`; ASGI adds WebSockets/streaming and non-blocking I/O.
4. **GET vs POST?** GET: read, idempotent, params in URL, cacheable. POST: sends a body, non-idempotent by default; use for large/sensitive inputs and state changes.
5. **Which status code for: invalid JSON? missing field? unknown ID? model crashed? overloaded?** 400, 422, 404, 500, 503/429.
6. **Why validate input at the API?** Protect the model from bad/huge inputs, give clear errors, security, stable contract. Limitation: validation adds a few ms and must be kept in sync with the model's real constraints.
7. **Where do you load the model and why?** Once per worker at startup (module level or FastAPI lifespan), never per request — loading is slow; per-request would dominate latency.
8. **How many workers should I run?** Depends on CPU cores, RAM per worker (model size), and request time; start with cores for CPU-bound work, measure with a load test, watch memory.
9. **What happens if I put `time.sleep` in an `async def` route?** It blocks the event loop; all requests on that worker wait. Use `await asyncio.sleep` or a plain `def` route (thread pool).
10. **How do you pass config/secrets to the service?** Environment variables (and a secrets manager in production); `.env` locally, never committed.
11. **How do you test an API?** Unit tests with the framework's test client, smoke tests on the running service, curl/Swagger for manual checks, load tests for performance.
12. **How would you serve an LLM that takes 2 s per request?** More replicas and/or batching, async for waiting on an external API, timeouts/queue limits, streaming responses; discuss memory per worker (see module 10 step 08).
13. **Why can't I keep state in a Python global in a multi-worker API?** Each worker is a separate process with its own copy; use a database/Redis.
