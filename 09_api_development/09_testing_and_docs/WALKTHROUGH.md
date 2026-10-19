# Step 09 — Testing and documenting APIs

## Goal
Three levels of API checking: (1) **unit tests** in-process with pytest, (2) a **smoke test** against a real server with httpx, (3) manual probing with **curl** and the generated docs.

## Run it
```bash
cd 09_api_development/09_testing_and_docs
pytest -v          # 14 in-process tests, ~0.3 s
python main.py     # smoke test against a real uvicorn
bash curl_cheatsheet.sh   # needs a server running (see file header)
```
Real output of `pytest -q`: `14 passed in 0.26s`. Real smoke output:
```
GET /health                 -> 200 {"status":"ok","model_version":"0.1.0","model_source":"fallback-keyword"}
POST /predict               -> 200 {"label":"positive","confidence":0.75,...}
POST /predict (empty)       -> 422 {"detail":[{"type":"string_too_short","loc":["body","text"],...
GET /predict (wrong method) -> 405 {"detail":"Method Not Allowed"}
GET /nope                   -> 404 {"detail":"Not Found"}
GET /openapi.json           -> 200 ...
GET /docs                   -> 200 ...
All smoke checks passed.
```
Run the same tests against the **real model**: `MODEL_PATH=<path to model.joblib> pytest -q` (verified: 14 passed with module 03's artifact).

## Syntax explained
| Syntax | Meaning | Example |
|---|---|---|
| `def test_x():` | pytest runs every function starting with `test_` | — |
| `assert cond` | Fail the test if `cond` is False; pytest prints both sides | `assert 1+1 == 2` |
| `@pytest.fixture` | Reusable setup; tests receive it by naming it as a parameter | — |
| `yield c` in a fixture | Give the object to the test, then run cleanup after | — |
| `@pytest.mark.parametrize("body", [...])` | Run one test body over many inputs | — |
| `monkeypatch.setenv(k, v)` | Temporarily change env; auto-reverted | — |
| `conftest.py` | Auto-loaded pytest helper file | — |
| `try: ... finally: server.terminate()` | Always stop the server | — |

## Libraries & functions
| Name | Kind | Does |
|---|---|---|
| `pytest` | 3rd-party | Test runner; `-v` verbose, `-x` stop at first failure, `-k name` filter |
| `app.test_client()` (Flask) | flask | Call app in-process |
| `TestClient(app)` (FastAPI) | fastapi | Same; use `with` to run lifespan |
| `httpx.Client(base_url=, timeout=)` | 3rd-party | Real HTTP client with connection reuse |
| `curl -i / -X / -H / -d / -w` | CLI | show headers / method / header / body / format output |

## Testing pyramid for an ML API
| Level | What it catches | Tool | Speed |
|---|---|---|---|
| Unit | wrong status, validation, config | pytest + test client | ms |
| Smoke | "server starts, endpoints answer" after deploy | httpx/curl | seconds |
| Model quality | accuracy regression | fixed eval set (the prompt-testing idea from module 07) | seconds–minutes |
| Load | latency/throughput under traffic | load_test (module 10 step 08) | minutes |

## Auto-generated documentation
FastAPI builds an **OpenAPI** schema from your code: `/openapi.json` (machine-readable), `/docs` (Swagger UI: click *Try it out*), `/redoc`. For Flask, docs are manual (or add extensions such as flask-smorest). Good docs = examples + error responses; `Field(description=...)` and `response_model` feed `/docs`.

## Execution trace
`test_predict_rejects_bad_input[body2]` (`{"text": 5}`): fixture builds `TestClient` (lifespan loads model) → test posts → Pydantic rejects int → 422 → `assert ... == 422` passes → fixture exits, lifespan shutdown runs.

## Experiments
1. Change an expected value to make a test fail; read the pytest diff output.
2. Add a test for `/health` returning `model_version` equal to the env var (use `monkeypatch`).
3. Run `pytest -k flask -v` then `pytest -k fastapi -v`.

## Debugging practice
`ModuleNotFoundError: No module named 'flask_app'` when running pytest from another folder: the fixture `conftest.py` adds the step-06 folder to `sys.path`. Remove it, re-run, read the error, and restore it — you just learned why test imports break.

## Rebuild checklist
- [ ] One pytest for `/health`, one parametrized for bad input.
- [ ] A curl command that POSTs JSON and prints only the status code.
- [ ] Find `/docs` and call `/predict` from the browser.

## Say it aloud
"I test at three levels: fast in-process unit tests with the framework's test client, a smoke test against a really running server, and manual curl or the `/docs` page. Tests assert status codes and response shape for good and bad input. Trade-off: in-process tests skip the real server, networking and worker behaviour, so I still need a smoke and load test."
