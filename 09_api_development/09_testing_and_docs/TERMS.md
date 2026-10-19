# Terms — Step 09: Testing and docs

Folder: `09_api_development/09_testing_and_docs` · Files covered: `main.py`, `conftest.py`, `test_flask_api.py`, `test_fastapi_api.py`, `curl_cheatsheet.sh`, `WALKTHROUGH.md`
Back to module: [09 API development](../README.md)

**Terms in this folder:** [Unit test](#unit-test-concept) · [Smoke test](#smoke-test-concept) · [Testing pyramid for an ML API](#testing-pyramid-for-an-ml-api-concept) · [pytest](#pytest-third-party-pytest) · [assert](#assert-python-built-in) · [test_ naming convention](#test_-naming-convention-concept) · [Fixture (@pytest.fixture)](#fixture-pytestfixture-third-party-pytest) · [yield in a fixture](#yield-in-a-fixture-syntax) · [@pytest.mark.parametrize](#pytestmarkparametrize-third-party-pytest) · [monkeypatch.setenv](#monkeypatchsetenv-third-party-pytest) · [conftest.py](#conftestpy-concept) · [sys.path.insert with Path(__file__)](#syspathinsert-with-path__file__-standard-library) · [app.config["TESTING"] = True](#appconfigtesting-true-third-party-flask) · [Flask test_client](#flask-test_client-third-party-flask) · [FastAPI TestClient with lifespan](#fastapi-testclient-with-lifespan-third-party-fastapi) · [client.post(..., data=..., content_type=...)](#clientpost-data-content_type-third-party-flask) · [Range assertion (0.0 <= x <= 1.0)](#range-assertion-00-x-10-syntax) · [OpenAPI schema test](#openapi-schema-test-concept) · [Swagger UI (/docs)](#swagger-ui-docs-concept) · [subprocess.Popen with try/finally](#subprocesspopen-with-tryfinally-standard-library) · [httpx.Client(base_url, timeout)](#httpxclientbase_url-timeout-third-party-httpx) · [httpx.TransportError](#httpxtransporterror-third-party-httpx) · [Helper function with expected status](#helper-function-with-expected-status-concept) · [Assertion message](#assertion-message-syntax) · [String slicing (resp.text[:90])](#string-slicing-resptext90-syntax) · [curl flags (-i, -s, -X, -H, -d, -o, -w)](#curl-flags--i--s--x--h--d--o--w-command) · [curl -w time_total](#curl--w-time_total-command) · [Shell parameter default (${PORT:-8000})](#shell-parameter-default-port-8000-syntax) · [Auto-generated documentation](#auto-generated-documentation-concept) · [ModuleNotFoundError in tests](#modulenotfounderror-in-tests-concept)

---

## Unit test  `(concept)`
**1. Definition:** A small automated check that runs a piece of code with a known input and verifies the output, here by calling the API in-process and checking the status code and JSON.
**2. In plain words:** Unit tests are your safety net: after any change you run them in a fraction of a second and learn if you broke something. They cover good input and bad input, because errors are part of the API's behaviour.
**3. Example to quote:**
> "My tests post `{'text': ''}` and assert the answer is 422, so I know validation still works after any refactor."

**Seen in:** `test_flask_api.py`, `test_fastapi_api.py`

## Smoke test  `(concept)`
**1. Definition:** A quick check against a real running deployment that the main endpoints answer with the expected statuses.
**2. In plain words:** Unit tests skip the real server and network. A smoke test starts the actual server and confirms that it boots and answers; you run it after a deploy as a "does it even work?" check.
**3. Example to quote:**
> "`python main.py` starts uvicorn, calls seven endpoints with httpx, and prints 'All smoke checks passed.'"

**Seen in:** `main.py`

## Testing pyramid for an ML API  `(concept)`
**1. Definition:** A layered strategy: many fast unit tests, fewer smoke tests, a model-quality check on a fixed evaluation set, and occasional load tests.
**2. In plain words:** Each level catches different problems: unit tests catch wrong status codes and validation; smoke tests catch "server doesn't start"; quality checks catch accuracy regressions; load tests catch latency problems.
**3. Example to quote:**
> "I run unit tests on every change, a smoke test after deploy, and a load test before expecting traffic to grow."

**Seen in:** `WALKTHROUGH.md`

## pytest  `(third-party: pytest)`
**1. Definition:** A Python test framework that discovers functions named `test_*` in files named `test_*.py`, runs them and reports failures with detailed diffs.
**2. In plain words:** You write plain functions with `assert`; pytest finds and runs them. Useful flags: `-v` (verbose), `-x` (stop at first failure), `-k name` (run tests matching a name), `-q` (quiet).
**3. Example to quote:**
> "`pytest -v` ran 14 tests in about 0.3 seconds; `pytest -k flask -v` runs only the Flask ones."

```bash
pytest -v
```

**Seen in:** `WALKTHROUGH.md`, test files

## assert  `(Python built-in)`
**1. Definition:** `assert condition` raises `AssertionError` if the condition is false; pytest rewrites it to show both sides of the comparison.
**2. In plain words:** It is how a test states what must be true. When it fails, pytest prints the values on each side, which makes the reason clear.
**3. Example to quote:**
> "`assert resp.status_code == 200` fails the test with a clear diff if the API returns anything else."

```python
assert 1 + 1 == 2
```

**Seen in:** all tests, `check` in `main.py`

## test_ naming convention  `(concept)`
**1. Definition:** pytest collects functions whose names start with `test_` (in files named `test_*.py`).
**2. In plain words:** There is nothing to register: name a function `test_health_ok` and pytest runs it. A function named `check_health` would be silently ignored.
**3. Example to quote:**
> "`def test_health_ok(client)` is picked up automatically because it starts with `test_`."

**Seen in:** test files

## Fixture (@pytest.fixture)  `(third-party: pytest)`
**1. Definition:** A function decorated with `@pytest.fixture` that provides reusable setup; any test that names it as a parameter receives its return value.
**2. In plain words:** Instead of building a client in every test, you define `client` once and tests just ask for it. It reduces repetition and keeps setup in one place.
**3. Example to quote:**
> "`def test_health_ok(client)` gets a ready test client because the `client` fixture builds it."

```python
import pytest
@pytest.fixture()
def client(): return "ready"
```

**Seen in:** both test files

## yield in a fixture  `(syntax)`
**1. Definition:** In a fixture, code before `yield` is setup and code after it is teardown; the yielded value is given to the test.
**2. In plain words:** The FastAPI fixture enters `with TestClient(app) as c:`, yields `c` to the test, and when the test finishes the `with` block exits and runs shutdown (lifespan cleanup).
**3. Example to quote:**
> "My fixture does `with TestClient(app) as c: yield c`, so the model loads before the test and shutdown runs after."

**Seen in:** `test_fastapi_api.py`

## @pytest.mark.parametrize  `(third-party: pytest)`
**1. Definition:** A decorator that runs one test function once per entry in a list of argument values.
**2. In plain words:** Rather than three near-identical tests for `{}`, `{'text': ''}` and `{'text': 5}`, you write one test and list the inputs. Each input is reported as its own test.
**3. Example to quote:**
> "One parametrized test checks that three different bad bodies all return 422."

```python
@pytest.mark.parametrize("n", [1, 2, 3])
def test_pos(n): assert n > 0
```

**Seen in:** `test_predict_rejects_bad_input`

## monkeypatch.setenv  `(third-party: pytest)`
**1. Definition:** A pytest fixture whose `setenv(name, value)` temporarily sets an environment variable and automatically restores the old state after the test.
**2. In plain words:** It lets a test change configuration (like `MAX_TEXT_LENGTH=5`) without affecting other tests. It works here because `get_settings()` reads the environment at call time.
**3. Example to quote:**
> "`monkeypatch.setenv('MAX_TEXT_LENGTH', '5')` makes my next request too long, and it's undone when the test ends."

**Seen in:** `test_max_length_comes_from_env`

## conftest.py  `(concept)`
**1. Definition:** A special pytest file that is loaded automatically, used for shared fixtures and setup for tests in its folder.
**2. In plain words:** Here it puts the step-06 folder on `sys.path` so the tests can `import flask_app`. Remove it and you get `ModuleNotFoundError`, which shows why test imports can break.
**3. Example to quote:**
> "My `conftest.py` adds step 06 to `sys.path`; without it pytest can't find `flask_app`."

**Seen in:** `conftest.py`

## sys.path.insert with Path(__file__)  `(standard library)`
**1. Definition:** `sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "folder"))` adds a sibling folder to Python's import search path.
**2. In plain words:** It lets code in one numbered folder import modules from another without installing packages. It is a pragmatic teaching setup; real projects use packages.
**3. Example to quote:**
> "The conftest inserts `.../06_model_api_flask_vs_fastapi` at the front of `sys.path`, so `from fastapi_app import app` works."

**Seen in:** `conftest.py`

## app.config["TESTING"] = True  `(third-party: flask)`
**1. Definition:** A Flask configuration flag that makes errors propagate to the test client instead of being turned into generic 500 pages.
**2. In plain words:** In test mode, unhandled exceptions show up as real exceptions in the test, which makes bugs easier to see.
**3. Example to quote:**
> "The Flask fixture sets `app.config['TESTING'] = True` before returning `app.test_client()`."

**Seen in:** `test_flask_api.py`

## Flask test_client  `(third-party: flask)`
**1. Definition:** An in-process client for a Flask app that sends requests without a network socket.
**2. In plain words:** Tests are fast and need no server. You use `client.get(...)` and `client.post(..., json=...)` and examine `status_code` and `get_json()`.
**3. Example to quote:**
> "`client.post('/predict', json={'text': 'I love this great film'})` returned label positive without any server running."

**Seen in:** `test_flask_api.py`

## FastAPI TestClient with lifespan  `(third-party: fastapi)`
**1. Definition:** `TestClient(app)` calls a FastAPI app in-process; using it as `with TestClient(app) as c:` triggers startup and shutdown events.
**2. In plain words:** Without the `with`, the lifespan does not run and the model is never loaded, so tests would fail oddly. The fixture does it once per test.
**3. Example to quote:**
> "My fixture uses `with TestClient(app) as c:` so the model loads before each test."

**Seen in:** `test_fastapi_api.py`

## client.post(..., data=..., content_type=...)  `(third-party: flask)`
**1. Definition:** Flask test client arguments to send raw body text with a given content type, used here to simulate invalid JSON.
**2. In plain words:** `json=` always sends valid JSON, so to test a broken body we send `data='hello'` and label it `application/json`. The expected answer from Flask's hand-written handler is 400.
**3. Example to quote:**
> "`client.post('/predict', data='hello', content_type='application/json')` triggers the 'not JSON' branch and returns 400."

**Seen in:** `test_predict_not_json`

## Range assertion (0.0 <= x <= 1.0)  `(syntax)`
**1. Definition:** An assertion using a chained comparison to check a value lies within bounds.
**2. In plain words:** For model outputs you often cannot assert an exact number, but you can assert a valid range, such as a confidence between 0 and 1.
**3. Example to quote:**
> "`assert 0.0 <= body['confidence'] <= 1.0` checks the probability is valid without hard-coding a value."

**Seen in:** `test_predict_negative`

## OpenAPI schema test  `(concept)`
**1. Definition:** A test that fetches `/openapi.json` and checks that expected paths are listed, proving the API documents itself.
**2. In plain words:** It guards against accidentally dropping or renaming an endpoint, since the schema is generated from the code.
**3. Example to quote:**
> "`assert '/predict' in client.get('/openapi.json').json()['paths']` confirms my endpoint is documented."

**Seen in:** `test_openapi_lists_predict`

## Swagger UI (/docs)  `(concept)`
**1. Definition:** The interactive API documentation page FastAPI serves at `/docs`, generated from the OpenAPI schema, where you can send test requests from the browser.
**2. In plain words:** Click an endpoint, press "Try it out", type JSON and execute. It is the quickest manual test and doubles as living documentation. Flask has no such page built in.
**3. Example to quote:**
> "I opened `/docs`, expanded `POST /predict`, used 'Try it out' and saw the same JSON as in curl."

**Seen in:** `main.py`, `WALKTHROUGH.md`

## subprocess.Popen with try/finally  `(standard library)`
**1. Definition:** `Popen` starts a child process; wrapping the test in `try: ... finally:` guarantees `terminate()` and `wait()` run even if an assertion fails.
**2. In plain words:** A failed smoke test must not leave a server running on the port, which would break the next run. The `finally` block is the cleanup guarantee.
**3. Example to quote:**
> "If a check fails, `finally: server.terminate()` still stops uvicorn, so port 8124 is free for the next run."

**Seen in:** `main.py`

## httpx.Client(base_url, timeout)  `(third-party: httpx)`
**1. Definition:** An httpx session that reuses connections and prepends a base URL to relative paths; `timeout` limits how long to wait.
**2. In plain words:** With `base_url=BASE` you write `client.get('/health')` instead of repeating the host. The `with` block closes the connection afterwards. Always set a timeout so a stuck server cannot hang the test.
**3. Example to quote:**
> "`with httpx.Client(base_url=BASE, timeout=5) as client:` lets me call `client.get('/health')` and fails fast after 5 seconds."

**Seen in:** `main.py`

## httpx.TransportError  `(third-party: httpx)`
**1. Definition:** The base exception for network-level failures in httpx, such as connection refused or timeouts.
**2. In plain words:** While the server is still starting, connecting fails. The readiness loop treats this error as "not yet", sleeps and tries again.
**3. Example to quote:**
> "I catch `httpx.TransportError` while waiting for uvicorn to open its port, then retry after 0.2 seconds."

**Seen in:** `main.py`

## Helper function with expected status  `(concept)`
**1. Definition:** A small function (`check`) that prints a response's status and body preview and asserts the status equals the expected value.
**2. In plain words:** It keeps each smoke check to one line and gives a readable error message such as "GET /nope: expected 200, got 404".
**3. Example to quote:**
> "`check('GET /nope', client.get('/nope'), 404)` prints the result and fails with a message if it isn't a 404."

**Seen in:** `check`

## Assertion message  `(syntax)`
**1. Definition:** `assert condition, "message"` includes a custom message in the `AssertionError` when the condition is false.
**2. In plain words:** It tells you which check failed and why, without opening the code, for example "POST /predict: expected 200, got 500".
**3. Example to quote:**
> "My `check` asserts with `f'{label}: expected {expected_status}, got {resp.status_code}'` so a failure explains itself."

**Seen in:** `check`

## String slicing (resp.text[:90])  `(syntax)`
**1. Definition:** `s[:90]` returns the first 90 characters of a string.
**2. In plain words:** It truncates long response bodies so the smoke-test output stays readable.
**3. Example to quote:**
> "`resp.text[:90]` shows just the start of each body so the log lines stay short."

```python
print("abcdef"[:3])   # abc
```

**Seen in:** `check`

## curl flags (-i, -s, -X, -H, -d, -o, -w)  `(command)`
**1. Definition:** `-i` includes response headers, `-s` is silent (no progress meter), `-X` sets the HTTP method, `-H` adds a header, `-d` sends a body, `-o` redirects output to a file (here `/dev/null`), and `-w` prints a formatted summary such as the status code.
**2. In plain words:** These combine to probe an API from the terminal. For example `-s -o /dev/null -w "%{http_code}\n"` prints only the status code, which is perfect for quick checks in scripts.
**3. Example to quote:**
> "`curl -s -X POST $BASE/predict -H 'Content-Type: application/json' -d '{\"text\": \"I love this\"}'` sends a prediction request and prints the JSON."

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/health
```

**Seen in:** `curl_cheatsheet.sh`

## curl -w time_total  `(command)`
**1. Definition:** The `%{time_total}` variable in `curl -w` prints the total time the request took in seconds.
**2. In plain words:** It is a no-install way to measure the latency of an endpoint from the client's point of view.
**3. Example to quote:**
> "`curl -s -o /dev/null -w 'total: %{time_total}s\\n' $BASE/health` shows how long the health call took."

**Seen in:** `curl_cheatsheet.sh`

## Shell parameter default (${PORT:-8000})  `(syntax)`
**1. Definition:** In bash, `${VAR:-default}` expands to `VAR` if it is set and non-empty, otherwise to `default`.
**2. In plain words:** The script uses port 8000 unless you run `PORT=5001 bash curl_cheatsheet.sh`, so one file works for both the Flask and FastAPI servers.
**3. Example to quote:**
> "`PORT=${PORT:-8000}` means FastAPI's port by default, and I override it with `PORT=5001` for Flask."

```bash
PORT=${PORT:-8000}
```

**Seen in:** `curl_cheatsheet.sh`

## Auto-generated documentation  `(concept)`
**1. Definition:** Documentation built automatically from code: FastAPI produces an OpenAPI schema at `/openapi.json` plus `/docs` and `/redoc`; Flask needs extensions or manual docs.
**2. In plain words:** Because it comes from the code, it cannot drift out of date. Adding `Field(description=...)` and `response_model` makes the docs clearer.
**3. Example to quote:**
> "FastAPI documents itself; for Flask I'd have to add an extension or write the docs by hand."

**Seen in:** `WALKTHROUGH.md`

## ModuleNotFoundError in tests  `(concept)`
**1. Definition:** An import error raised when Python cannot find a module on `sys.path`, commonly seen when pytest runs from a different folder than expected.
**2. In plain words:** Removing `conftest.py`'s path insert makes `import flask_app` fail. Fixing import paths, not the test logic, is the cure.
**3. Example to quote:**
> "I got `No module named 'flask_app'` until the conftest added step 06 to `sys.path`."

**Seen in:** `WALKTHROUGH.md` debugging practice
