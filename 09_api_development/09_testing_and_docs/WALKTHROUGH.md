# Step 09 — Testing and documenting APIs

## The story so far
Step 08 let us change behaviour through configuration. Every such change can silently break an endpoint, and a person with curl does not scale.

## The pain point
Checking the API by hand after each change is slow and you forget cases (bad input, missing field, unknown id). Bugs reach users before you do.

## The spark
Write tests that call the API like a client: fast in-process tests with the framework's test client, one smoke test against a real server, curl and `/docs` for quick looks.

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

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Unit test vs smoke test | Tests at different levels | No | Unit tests run in-process on one piece and are fast; a smoke test hits a deployed/real server for a few critical paths |
| Integration test vs smoke test | Multi-component tests | Overlapping | Integration tests check components working together in depth; smoke tests are shallow and broad |
| fixture vs setup/teardown | Reusable test set-up | Similar | pytest fixtures are injected by parameter name and can yield for cleanup |
| parametrize vs loop in a test | Many inputs | No | parametrize reports each case separately; a loop stops at the first failure |
| `monkeypatch` vs mock | Test doubles | Overlapping | monkeypatch temporarily sets env vars/attributes and undoes them; mock objects fake behaviour/calls |
| Flask `test_client` vs FastAPI `TestClient` vs httpx `Client` | HTTP test clients | No | The first two call the app in-process; httpx `Client` talks to a real server over the network |
| OpenAPI vs Swagger UI vs `/docs` | API documentation | Related | OpenAPI = the spec (`/openapi.json`); Swagger UI = the web page that renders it, served at `/docs` |
| 400 vs 405 vs 404 vs 422 | Client error codes | No | 400 bad request; 404 no route; 405 route exists, wrong method; 422 valid JSON, invalid values |

## Code, step by step

### 1. Docstring: smoke test goal and how to run  (`main.py` lines 1–18)
```python
"""
GOAL:   Test a REAL running server over the network (not in-process): start uvicorn, call it with
        httpx, check status codes + JSON, read the auto docs, then shut the server down.
        This is a "smoke test" -- the quick check you run after deploying.

RUN:    cd 09_api_development/09_testing_and_docs
        python main.py            # smoke test with a real server
        pytest -v                 # the in-process unit tests (fast; no ports)
        bash curl_cheatsheet.sh   # same checks with curl (needs a server running)

EXPECTED OUTPUT (shape):
        GET /health -> 200 {...}
        POST /predict -> 200 {...}
        ...
        All smoke checks passed.

NEEDS:  uvicorn, httpx, pytest
"""
```
- **In simple words:** Describes a "smoke test": start the real server, poke it, stop it.
- **Exactly:** Three levels are named: `python main.py` (real server), `pytest -v` (in-process), `bash curl_cheatsheet.sh` (curl). Real run ends with `All smoke checks passed.`
- **Why it is here:** A smoke test is the quick check you run right after deploying.

### 2. Imports and constants  (`main.py` lines 20–29)
```python
import subprocess
import sys
import time
from pathlib import Path

import httpx

APP_DIR = Path(__file__).resolve().parents[1] / "06_model_api_flask_vs_fastapi"
PORT = 8124
BASE = f"http://127.0.0.1:{PORT}"
```
- **In simple words:** Tools to launch a server and call it, plus where the app is and which port to use.
- **Exactly:** `APP_DIR` is the step 06 folder; `PORT = 8124`; `BASE = "http://127.0.0.1:8124"` built with an f-string.
- **Why it is here:** A fixed unusual port avoids clashing with ports 8000/8123 used elsewhere.

### 3. check helper: print and assert the status  (`main.py` lines 32–34)
```python
def check(label: str, resp: httpx.Response, expected_status: int) -> None:
    print(f"{label:<28}-> {resp.status_code} {resp.text[:90]}")
    assert resp.status_code == expected_status, f"{label}: expected {expected_status}, got {resp.status_code}"
```
- **In simple words:** Print what came back, and stop with an error if the status is not what we expected.
- **Exactly:** `resp.status_code` is an int; `resp.text[:90]` is the first 90 characters of the body; `{label:<28}` pads the label. `assert cond, msg` raises `AssertionError(msg)` when false.
- **Why it is here:** Without the assert the script would print and still look successful when the API is broken.

### 4. Start uvicorn as a background process  (`main.py` lines 37–39)
```python
if __name__ == "__main__":
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "fastapi_app:app", "--port", str(PORT),
                               "--log-level", "warning"], cwd=APP_DIR)
```
- **In simple words:** Launch the real server in a separate program.
- **Exactly:** `subprocess.Popen([python, "-m", "uvicorn", "fastapi_app:app", "--port", "8124", "--log-level", "warning"], cwd=APP_DIR)` returns immediately with a handle.
- **Why it is here:** Testing a REAL server over a socket catches problems in-process tests cannot (ports, start-up, packaging).

### 5. Wait until the server answers  (`main.py` lines 40–46)
```python
    try:                                                   # try/finally guarantees the server is stopped
        for _ in range(100):                               # wait until it answers
            try:
                httpx.get(BASE + "/health", timeout=1)
                break
            except httpx.TransportError:
                time.sleep(0.2)
```
- **In simple words:** Keep knocking on the door until someone opens.
- **Exactly:** Up to 100 attempts, 0.2 s apart; a refused connection raises `httpx.TransportError` and we retry; the first successful `GET /health` triggers `break`. (There is no error if it never comes up: the later checks would then fail.) `try:` here pairs with `finally` below.
- **Why it is here:** Without the wait, the first check races the server boot and fails with connection refused.

### 6. Run the seven smoke checks  (`main.py` lines 48–55)
```python
        with httpx.Client(base_url=BASE, timeout=5) as client:   # a Client reuses the connection and a base URL
            check("GET /health", client.get("/health"), 200)
            check("POST /predict", client.post("/predict", json={"text": "great, I love it"}), 200)
            check("POST /predict (empty)", client.post("/predict", json={"text": ""}), 422)
            check("GET /predict (wrong method)", client.get("/predict"), 405)
            check("GET /nope", client.get("/nope"), 404)
            check("GET /openapi.json", client.get("/openapi.json"), 200)
            check("GET /docs", client.get("/docs"), 200)
```
- **In simple words:** Ask the server seven questions and verify each answer's status code.
- **Exactly:** `httpx.Client(base_url=BASE, timeout=5)` reuses one connection and prefixes the base URL. Real results: `/health` 200, `/predict` 200 `{"label":"positive","confidence":0.75,...}`, empty text 422 (`string_too_short`), wrong method 405 `{"detail":"Method Not Allowed"}`, `/nope` 404 `{"detail":"Not Found"}`, `/openapi.json` 200 (`"openapi":"3.1.0"`), `/docs` 200 (HTML).
- **Why it is here:** Covers the happy path, bad input, wrong method, unknown route and the auto-generated docs in under a second.

### 7. Print success and always stop the server  (`main.py` lines 56–59)
```python
        print("All smoke checks passed.")
    finally:
        server.terminate()
        server.wait(timeout=10)
```
- **In simple words:** Say everything passed, and make sure the server is shut down even if a check failed.
- **Exactly:** `finally` runs on success or exception; `terminate()` sends SIGTERM, `wait(timeout=10)` waits for exit. Prints `All smoke checks passed.`
- **Why it is here:** Otherwise a failed assertion would leave a server running on port 8124.

### 8. TRY IT notes  (`main.py` lines 60–61)
```python
    # TRY IT: change an expected status to 201 and read the AssertionError message.
    # TRY IT: open http://127.0.0.1:8000/docs in a browser after `uvicorn fastapi_app:app` and use "Try it out".
```
- **In simple words:** Two suggestions for experiments.
- **Exactly:** Change an expected status to see an `AssertionError` message; open `/docs` and use "Try it out".
- **Why it is here:** Learning by breaking.

### 9. conftest: make step 06 apps importable  (`conftest.py` lines 1–5)
```python
"""pytest auto-loads conftest.py. We use it to make the step-06 apps importable from the tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "06_model_api_flask_vs_fastapi"))
```
- **In simple words:** A special file pytest reads automatically before the tests.
- **Exactly:** `sys.path.insert(0, ".../06_model_api_flask_vs_fastapi")` lets `from flask_app import app` and `from fastapi_app import app` work from this folder.
- **Why it is here:** Without it every test file would fail with ModuleNotFoundError.

### 10. FastAPI tests: imports  (`test_fastapi_api.py` lines 1–5)
```python
"""pytest tests for the FastAPI API.   RUN: cd 09_api_development/09_testing_and_docs && pytest -v"""
import pytest
from fastapi.testclient import TestClient

from fastapi_app import app
```
- **In simple words:** Bring in pytest, the FastAPI test client and the app.
- **Exactly:** `TestClient` calls the ASGI app in-process.
- **Why it is here:** Fast tests need no real server or port.

### 11. Fixture: a client that runs the lifespan  (`test_fastapi_api.py` lines 8–11)
```python
@pytest.fixture()
def client():
    with TestClient(app) as c:          # `with` runs startup/lifespan (loads the model)
        yield c                         # yield = give the client to the test, then clean up afterwards
```
- **In simple words:** A reusable "set-up" that hands each test a ready-to-use client.
- **Exactly:** `@pytest.fixture()` registers it; `with TestClient(app) as c` runs startup (loads the model); `yield c` gives it to the test and resumes for cleanup afterwards.
- **Why it is here:** Without `with`, `app.state.model` is not set and every request would fail.

### 12. test_health_ok  (`test_fastapi_api.py` lines 14–15)
```python
def test_health_ok(client):
    assert client.get("/health").json()["status"] == "ok"
```
- **In simple words:** Check that /health says ok.
- **Exactly:** pytest passes the `client` fixture because the parameter has that name. `assert ... == "ok"` fails the test otherwise.
- **Why it is here:** The simplest "is it alive" regression test.

### 13. test_predict_negative  (`test_fastapi_api.py` lines 18–21)
```python
def test_predict_negative(client):
    body = client.post("/predict", json={"text": "terrible and boring"}).json()
    assert body["label"] == "negative"
    assert 0.0 <= body["confidence"] <= 1.0
```
- **In simple words:** Check a negative sentence is classified negative with a valid confidence.
- **Exactly:** Input "terrible and boring" -> label `negative`; `0.0 <= confidence <= 1.0` is a chained comparison.
- **Why it is here:** Tests behaviour and the shape of the response, not exact floats.

### 14. Parametrized bad-input test  (`test_fastapi_api.py` lines 24–26)
```python
@pytest.mark.parametrize("body", [{}, {"text": ""}, {"text": 5}])
def test_predict_rejects_bad_input(client, body):
    assert client.post("/predict", json=body).status_code == 422
```
- **In simple words:** One test, run three times with different bad bodies.
- **Exactly:** `@pytest.mark.parametrize("body", [{}, {"text": ""}, {"text": 5}])` creates cases `[body0]`, `[body1]`, `[body2]`; each must return 422. Real pytest output shows all three PASSED.
- **Why it is here:** Adding a case is one list item, not a copied function.

### 15. test_openapi_lists_predict  (`test_fastapi_api.py` lines 29–30)
```python
def test_openapi_lists_predict(client):
    assert "/predict" in client.get("/openapi.json").json()["paths"]
```
- **In simple words:** Check the auto-generated API description mentions /predict.
- **Exactly:** `/openapi.json` is the machine-readable spec; `["paths"]` is a dict keyed by URL path.
- **Why it is here:** Guards the documentation contract that clients and `/docs` depend on.

### 16. test_docs_page_served  (`test_fastapi_api.py` lines 33–35)
```python
def test_docs_page_served(client):
    resp = client.get("/docs")
    assert resp.status_code == 200 and "swagger" in resp.text.lower()
```
- **In simple words:** Check the interactive docs page loads.
- **Exactly:** Status 200 and the HTML contains "swagger" (Swagger UI).
- **Why it is here:** Catches a broken docs setup.

### 17. Flask tests: imports  (`test_flask_api.py` lines 1–4)
```python
"""pytest tests for the Flask API.   RUN: cd 09_api_development/09_testing_and_docs && pytest -v"""
import pytest

from flask_app import app
```
- **In simple words:** Bring in pytest and the Flask app.
- **Exactly:** `from flask_app import app` works because of conftest.
- **Why it is here:** Same in-process idea as FastAPI.

### 18. Flask fixture: a test client  (`test_flask_api.py` lines 7–10)
```python
@pytest.fixture()                      # a fixture = reusable setup; pytest passes its result to tests that ask for it
def client():
    app.config["TESTING"] = True
    return app.test_client()
```
- **In simple words:** A set-up helper returning a pretend browser for the Flask app.
- **Exactly:** `app.config["TESTING"] = True` makes Flask propagate errors; `app.test_client()` calls routes without a network. Uses `return` (no cleanup needed) rather than `yield`.
- **Why it is here:** Reusable set-up keeps each test short.

### 19. test_health_ok  (`test_flask_api.py` lines 13–16)
```python
def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200      # `assert` = the test fails if the condition is False
    assert resp.get_json()["status"] == "ok"
```
- **In simple words:** Check /health returns 200 and status ok.
- **Exactly:** `resp.get_json()` parses the JSON body.
- **Why it is here:** Basic alive check.

### 20. test_predict_positive  (`test_flask_api.py` lines 19–22)
```python
def test_predict_positive(client):
    resp = client.post("/predict", json={"text": "I love this great film"})
    assert resp.status_code == 200
    assert resp.get_json()["label"] == "positive"
```
- **In simple words:** Check a happy sentence is positive.
- **Exactly:** "I love this great film" -> `positive` (0.75 with the fallback model).
- **Why it is here:** Covers the main path.

### 21. Parametrized bad inputs  (`test_flask_api.py` lines 25–27)
```python
@pytest.mark.parametrize("body", [{}, {"text": ""}, {"text": 5}])   # run the same test with 3 inputs
def test_predict_rejects_bad_input(client, body):
    assert client.post("/predict", json=body).status_code == 422
```
- **In simple words:** Three bad bodies, each must give 422.
- **Exactly:** Same three bodies as the FastAPI test; Flask's hand-written validation gives the same status.
- **Why it is here:** Shows the two apps agree.

### 22. test_predict_not_json  (`test_flask_api.py` lines 30–32)
```python
def test_predict_not_json(client):
    resp = client.post("/predict", data="hello", content_type="application/json")
    assert resp.status_code == 400
```
- **In simple words:** Send plain text labelled as JSON and expect 400.
- **Exactly:** `data="hello"` with `content_type="application/json"`; `get_json(silent=True)` returns `None`, so the app answers 400 (not 422).
- **Why it is here:** Distinguishes "unparseable body" (400) from "invalid values" (422).

### 23. test_max_length_comes_from_env  (`test_flask_api.py` lines 35–37)
```python
def test_max_length_comes_from_env(client, monkeypatch):
    monkeypatch.setenv("MAX_TEXT_LENGTH", "5")   # monkeypatch undoes the change after the test
    assert client.post("/predict", json={"text": "far too long"}).status_code == 422
```
- **In simple words:** Check that the length limit really comes from an environment variable.
- **Exactly:** `monkeypatch.setenv("MAX_TEXT_LENGTH","5")` sets it for this test only and is undone afterwards; "far too long" has 12 chars > 5 -> 422.
- **Why it is here:** Works only because `get_settings()` reads the env at call time (step 08).

### 24. Shebang, usage and base URL  (`curl_cheatsheet.sh` lines 1–6)
```bash
#!/usr/bin/env bash
# curl cheat sheet for the model API. Start a server first (in another terminal):
#   cd 09_api_development/06_model_api_flask_vs_fastapi && uvicorn fastapi_app:app --port 8000
# Then:  bash curl_cheatsheet.sh        (use PORT=5001 bash curl_cheatsheet.sh for the Flask dev server)
PORT=${PORT:-8000}
BASE=http://127.0.0.1:$PORT
```
- **In simple words:** Header comment and the address to test.
- **Exactly:** `#!/usr/bin/env bash` picks bash; `PORT=${PORT:-8000}` uses the env var if set, else 8000; `BASE` becomes `http://127.0.0.1:8000`.
- **Why it is here:** One script works for both servers: `PORT=5001` for Flask.

### 25. GET /health with headers  (`curl_cheatsheet.sh` lines 8–9)
```bash
echo "--- GET /health (-i shows status line + headers)"
curl -i -s $BASE/health
```
- **In simple words:** Show the status line and headers as well as the body.
- **Exactly:** `-i` includes headers, `-s` silences the progress meter. Real output: `HTTP/1.1 200 OK`, `server: uvicorn`, `content-type: application/json`, `content-length: 73`, then the JSON.
- **Why it is here:** Headers are half of what an HTTP response is.

### 26. POST /predict with JSON  (`curl_cheatsheet.sh` lines 11–12)
```bash
echo; echo "--- POST /predict (-X method, -H header, -d body)"
curl -s -X POST $BASE/predict -H "Content-Type: application/json" -d '{"text": "I love this"}'
```
- **In simple words:** Send a JSON body to the prediction endpoint.
- **Exactly:** `-X POST` method, `-H` header, `-d` body. Real: `{"label":"positive","confidence":0.6667,"model_version":"0.1.0","latency_ms":0.06}`.
- **Why it is here:** Without the `Content-Type` header the server does not treat the body as JSON.

### 27. Print only the status code  (`curl_cheatsheet.sh` lines 14–15)
```bash
echo; echo "--- invalid body: -w prints just the status code"
curl -s -o /dev/null -w "%{http_code}\n" -X POST $BASE/predict -H "Content-Type: application/json" -d '{"text": ""}'
```
- **In simple words:** Throw away the body and print just the number.
- **Exactly:** `-o /dev/null` discards the body; `-w "%{http_code}\n"` prints the status after the transfer. Real: `422`.
- **Why it is here:** Handy in scripts and CI.

### 28. Wrong method  (`curl_cheatsheet.sh` lines 17–18)
```bash
echo "--- wrong method"
curl -s -o /dev/null -w "%{http_code}\n" -X GET $BASE/predict
```
- **In simple words:** GET on a POST-only route.
- **Exactly:** Real: `405`.
- **Why it is here:** Shows method checking.

### 29. Timing a request  (`curl_cheatsheet.sh` lines 20–21)
```bash
echo "--- timing (-w with time_total)"
curl -s -o /dev/null -w "total: %{time_total}s\n" $BASE/health
```
- **In simple words:** Print how long the request took.
- **Exactly:** `%{time_total}` is total seconds. Real: `total: 0.002484s`.
- **Why it is here:** Quick latency check without code.


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

## The new problem (cliffhanger)
All tests pass, but every endpoint so far returns the answer only when it is complete. A chat model needs several seconds, and the user stares at a blank screen. Next: [10_streaming_sse](../10_streaming_sse/).

## Say it aloud
> **Say it aloud (interview box):** I test an API at three levels: fast in-process tests with the framework's test client, a smoke test against a really running server, and manual curl or the `/docs` page. The tests assert status codes and response shape for both good and bad input. In-process tests skip the real server and its workers, so I still need a smoke test and a load test.
