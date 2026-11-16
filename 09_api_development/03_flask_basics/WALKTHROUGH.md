# Step 03 — Flask fundamentals

## The story so far
Step 02 gave us the contract, but writing raw `environ` and `send` code for every route is painful. We want to write ordinary functions and let a framework do the plumbing.

## The pain point
With raw WSGI you check `PATH_INFO` with long if-chains, build header lists, and encode JSON yourself. One typo in a header and the client gets garbage.

## The spark
Register a function on a route with a decorator, read input from `request`, return a dict and a status code. Flask turns that into a valid WSGI response and gives you a test client for free.

## Goal
Build routes, read query/path/JSON input, return JSON with the right status codes, and test without a server.

## Run it
```bash
cd 09_api_development/03_flask_basics
python main.py            # demo via test client
python main.py --serve    # real dev server on :5001
curl -s http://127.0.0.1:5001/add?a=2\&b=3
curl -s -X POST http://127.0.0.1:5001/items -H 'Content-Type: application/json' -d '{"name":"pen"}'
```
Real output of `python main.py`:
```
GET  /health        -> 200 {'status': 'ok'}
GET  /add?a=2&b=3   -> 200 {'result': 5.0}
GET  /add?a=x       -> 400 {'error': 'query params a and b must be numbers'}
POST /items         -> 201 {'id': 1, 'name': 'pen'}
GET  /items/1       -> 200 {'id': 1, 'name': 'pen'}
GET  /items/99      -> 404 {'error': 'item 99 not found'}
POST /items (empty) -> 400 {'error': "JSON body with 'name' is required"}
DELETE /items/1     -> 405 None
```
Notice the last line: `None` because Flask's default 405 page is **HTML**, not JSON. Step 06 adds JSON error handlers.

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Route | endpoint, URL rule, path | Nearly | Route = URL pattern plus method mapped to a function; Flask also uses "endpoint" for the function's internal name. |
| Query param vs path param | query string / URL variable | No | Query: `?a=2&b=3` after the `?`; path: part of the path itself, `/items/1`. |
| `jsonify` vs returning a dict | JSON response | Mostly | Modern Flask converts a returned dict to JSON too; `jsonify` also handles keyword args and lists explicitly. |
| `request.args` vs `request.form` vs `request.get_json()` | query / form / JSON body | No | Three different places data arrives: URL, form-encoded body, JSON body. |
| Test client vs dev server | `app.test_client()` vs `app.run()` | No | Test client calls the app in-process; the dev server listens on a real port. |
| Dev server vs Gunicorn | Werkzeug server, `flask run` vs production WSGI server | No | Dev server is single-purpose for development; gunicorn manages multiple worker processes. |
| 201 vs 200 | Created vs OK | No | 201 says a new resource was created; 200 is generic success. |

## Code, step by step

### 1. The file header: goal, two run modes, expected output  (`main.py` lines 1–19)

```python
"""
GOAL:   Learn Flask fundamentals: routes, HTTP methods, query params, path params, JSON in/out,
        status codes -- using a tiny in-memory "items" API.

RUN:    cd 09_api_development/03_flask_basics
        python main.py            # runs a demo using Flask's built-in test client (no server needed)
        python main.py --serve    # starts a real dev server on http://127.0.0.1:5001
        curl http://127.0.0.1:5001/health      (in another terminal)
        NOTE: port 5001, not 5000 -- macOS AirPlay Receiver often occupies 5000.

EXPECTED OUTPUT (shape):
        GET  /health        -> 200 {'status': 'ok'}
        GET  /add?a=2&b=3   -> 200 {'result': 5.0}
        POST /items         -> 201 {'id': 1, 'name': 'pen'}
        GET  /items/1       -> 200 {'id': 1, 'name': 'pen'}
        GET  /items/99      -> 404 {'error': 'item 99 not found'}

NEEDS:  flask
"""
```

- **In simple words:** A note on what the file teaches and how to run it (demo mode or real server).
- **Exactly:** Docstring only. Port 5001 is used because macOS AirPlay Receiver often holds 5000.
- **Why it is here:** Tells you the two ways to run it and why the port is unusual.

### 2. Imports and create the Flask app  (`main.py` lines 21–25)

```python
import sys

from flask import Flask, jsonify, request  # Flask = the app class; jsonify = dict -> JSON response; request = current request

app = Flask(__name__)  # __name__ tells Flask where this module lives (used to find templates/static files)
```

- **In simple words:** Load Flask and make the application object that will hold all our routes.
- **Exactly:** `sys` gives `sys.argv` for the `--serve` flag. `Flask(__name__)` builds a WSGI application; `__name__` is `"__main__"` when run directly and tells Flask the root folder for templates/static files. `jsonify` builds a JSON `Response` (adds `Content-Type: application/json`). `request` is a context-local proxy to the current request.
- **Why it is here:** `app` is the one object that collects routes and is what a WSGI server (gunicorn) would be pointed at.

### 3. Fake database and id counter  (`main.py` lines 27–28)

```python
ITEMS: dict[int, dict] = {}   # our fake "database": id -> item. Lives in memory, lost on restart.
_next_id = 1                  # counter used to create ids
```

- **In simple words:** A dictionary pretending to be a database, and a number for the next id.
- **Exactly:** `ITEMS: dict[int, dict] = {}` is empty at start and keyed by integer id. `_next_id = 1`. Both live in the process memory: lost on restart and, with several worker processes, each worker would have its own copy.
- **Why it is here:** Lets us show create and read without installing a real database.

### 4. The /health route  (`main.py` lines 31–33)

```python
@app.get("/health")           # decorator: "when a GET request arrives for /health, call the function below"
def health():
    return jsonify(status="ok")   # Flask turns the return value into an HTTP response (200 by default)
```

- **In simple words:** When someone asks `GET /health`, answer that we are alive.
- **Exactly:** `@app.get("/health")` is shorthand for `@app.route("/health", methods=["GET"])`; it registers the function in the URL map. `jsonify(status="ok")` builds `{"status": "ok"}`; returning a response object means status 200 by default.
- **Why it is here:** Health endpoints are what load balancers and Kubernetes probe to know a service is up.

### 5. The /add route: query params and bad-input handling  (`main.py` lines 36–44)

```python
@app.get("/add")
def add():
    # Query string: /add?a=2&b=3  ->  request.args is a dict-like object {"a": "2", "b": "3"}
    try:
        a = float(request.args["a"])      # all query values arrive as STRINGS -> convert
        b = float(request.args["b"])
    except (KeyError, ValueError):        # missing key OR not a number
        return jsonify(error="query params a and b must be numbers"), 400   # (body, status) tuple
    return jsonify(result=a + b)
```

- **In simple words:** Read `a` and `b` from the URL, add them; if they are missing or not numbers, say 400.
- **Exactly:** `request.args` is an immutable multi-dict of the query string: for `/add?a=2&b=3` it is `{'a': '2', 'b': '3'}` (strings!). `request.args["a"]` raises `KeyError` if missing; `float("x")` raises `ValueError`. We catch both and return the tuple `(response, 400)`. Real runs: `a=2&b=3` gives `{'result': 5.0}` (float, so `5.0`); `a=x` gives `400 {'error': 'query params a and b must be numbers'}`.
- **Why it is here:** Without the conversion you would concatenate strings (`"2"+"3"`). Without the try/except bad input becomes a 500 instead of an informative 400.

### 6. The POST /items route: create an item  (`main.py` lines 47–56)

```python
@app.post("/items")                       # POST = create something new
def create_item():
    global _next_id                       # we modify a module-level variable, so we must declare it global
    data = request.get_json(silent=True)  # parse JSON body; silent=True -> None instead of raising on bad JSON
    if not data or "name" not in data:
        return jsonify(error="JSON body with 'name' is required"), 400
    item = {"id": _next_id, "name": data["name"]}
    ITEMS[_next_id] = item
    _next_id += 1
    return jsonify(item), 201             # 201 Created is the right status for a successful creation
```

- **In simple words:** Take the JSON the client sent, store a new item with the next id, reply 201 Created.
- **Exactly:** `global _next_id` is required because we rebind the module variable. `request.get_json(silent=True)` parses the body; `silent=True` returns `None` on bad JSON or wrong content type instead of raising. `if not data or "name" not in data` rejects `None`, `{}` and missing name with 400 (the empty `{}` case is real: `400 {'error': "JSON body with 'name' is required"}`). Otherwise `item = {'id': 1, 'name': 'pen'}` is stored and returned as `(response, 201)`.
- **Why it is here:** 201 (not 200) tells the client something was created. The validation here is manual, which is the pain step 05 and FastAPI fix.

### 7. The GET /items/<int:item_id> route  (`main.py` lines 59–64)

```python
@app.get("/items/<int:item_id>")          # <int:item_id> = path parameter, converted to int automatically
def get_item(item_id: int):
    item = ITEMS.get(item_id)             # dict.get returns None instead of raising KeyError
    if item is None:
        return jsonify(error=f"item {item_id} not found"), 404
    return jsonify(item)
```

- **In simple words:** Look up one item by the id in the URL; 404 if it is not there.
- **Exactly:** `<int:item_id>` is a path parameter with the `int` converter: `/items/1` gives `item_id == 1` (an `int`), and `/items/abc` would not match any route (404). `ITEMS.get(item_id)` returns `None` instead of raising `KeyError`. Real: `/items/1` gives `200 {'id': 1, 'name': 'pen'}`, `/items/99` gives `404 {'error': 'item 99 not found'}`.
- **Why it is here:** Distinguishes "the route exists but the thing does not" (404 with message) from other errors.

### 8. Demo helper: create a test client and a print function  (`main.py` lines 67–71)

```python
def demo():
    """Call the app in-process with Flask's test client: fast, no ports, perfect for tests."""
    client = app.test_client()
    def show(label, resp):                # small nested helper function
        print(f"{label:<20}-> {resp.status_code} {resp.get_json()}")
```

- **In simple words:** Run the app inside this process, with no network, and a small function that prints results neatly.
- **Exactly:** `app.test_client()` returns an object that builds a WSGI `environ`, calls the app directly and returns the response. `show` is a nested function: `f"{label:<20}"` pads the label to 20 characters left-aligned; `resp.get_json()` parses the body.
- **Why it is here:** Tests and demos run in milliseconds with no port conflicts.

### 9. Call every route, including bad ones  (`main.py` lines 72–80)

```python
    show("GET  /health", client.get("/health"))
    show("GET  /add?a=2&b=3", client.get("/add?a=2&b=3"))
    show("GET  /add?a=x", client.get("/add?a=x"))                          # BREAK IT: bad input -> 400
    show("POST /items", client.post("/items", json={"name": "pen"}))       # json= sets body AND Content-Type
    show("GET  /items/1", client.get("/items/1"))
    show("GET  /items/99", client.get("/items/99"))
    show("POST /items (empty)", client.post("/items", json={}))
    show("DELETE /items/1", client.delete("/items/1"))                     # no such route/method -> 405
    # TRY IT: add a DELETE route that removes an item and returns 204 with no body.
```

- **In simple words:** Eight requests: good ones and deliberately bad ones to see the status codes.
- **Exactly:** Real output: `/health` 200; `/add?a=2&b=3` 200 `{'result': 5.0}`; `/add?a=x` 400; `POST /items` 201 `{'id': 1, 'name': 'pen'}`; `GET /items/1` 200; `/items/99` 404; `POST /items` with `{}` 400; `DELETE /items/1` **405 None**. The 405 appears because a route exists for that path with other methods (GET), but none allows DELETE; Flask's default 405 page is HTML, so `get_json()` returns `None`. `json={...}` serialises the body and sets `Content-Type: application/json`.
- **Why it is here:** Shows that error paths matter as much as the happy path, and exposes the HTML-vs-JSON default-error issue (fixed in step 05 with `errorhandler`).

### 10. Choose between demo mode and a real server  (`main.py` lines 83–87)

```python
if __name__ == "__main__":
    if "--serve" in sys.argv:
        app.run(host="127.0.0.1", port=5001, debug=False)   # debug=True auto-reloads but NEVER use it in production
    else:
        demo()
```

- **In simple words:** No flag: run the demo. With `--serve`: start an actual server you can curl.
- **Exactly:** `sys.argv` is the list of command-line words; `"--serve" in sys.argv` is a crude flag check. `app.run(host="127.0.0.1", port=5001, debug=False)` starts Werkzeug's development WSGI server. `debug=True` adds the auto-reloader and an interactive debugger that allows remote code execution, so never use it in production.
- **Why it is here:** One file serves both the quick demo and the real server. In production you would instead run `gunicorn main:app`.


## Syntax explained
| Syntax | Meaning | Example |
|---|---|---|
| `@app.get("/health")` | **Decorator**: registers the function below as the handler for that route+method | `@app.post("/x")` |
| `<int:item_id>` | **Path parameter** with automatic conversion | `/items/7` → `item_id=7` |
| `return jsonify(...), 201` | A **tuple** `(body, status)` sets the status code | `return "x", 404` |
| `global _next_id` | Needed to *assign* to a module-level variable inside a function | — |
| `try / except (KeyError, ValueError)` | Catch several error types at once | — |
| `dict.get(key)` | Returns `None` instead of raising `KeyError` | `{}.get("a")` → `None` |
| `dict[int, dict]` | Type hint: dict with int keys and dict values | — |

## Libraries & functions
| Name | Kind | What it does | Important args | Returns |
|---|---|---|---|---|
| `Flask(__name__)` | flask | Creates the app | import name | app |
| `request.args` | flask | Query-string params (strings!) | `["a"]`, `.get("a", default)` | dict-like |
| `request.get_json(silent=True)` | flask | Parse JSON body | `silent=True` → `None` on failure | dict/list/None |
| `jsonify(**kw)` | flask | Make a JSON response with correct `Content-Type` | kwargs or one object | Response |
| `app.test_client()` | flask | In-process fake client (no sockets) | — | client with `.get/.post` |
| `client.post(url, json=...)` | flask | Send JSON body and set header | `json=` | response (`.status_code`, `.get_json()`) |
| `app.run(host, port, debug)` | flask | Built-in **dev** server | `debug` never in prod | — |

## Execution trace
`POST /items {"name":"pen"}`: Flask matches `(POST, /items)` → `create_item()` → `request.get_json(silent=True)` = `{'name':'pen'}` → validation passes → `item = {'id': 1, 'name': 'pen'}` stored in `ITEMS[1]` → returns `(jsonify(item), 201)` → client sees `201 {'id': 1, 'name': 'pen'}`.

## Experiments
1. Add `@app.delete("/items/<int:item_id>")` returning `("", 204)`.
2. Change `<int:item_id>` to `<item_id>` and request `/items/abc` → now it reaches your code as a str and the `.get` returns 404 instead of Flask rejecting it.
3. Restart the server → items are gone (memory only). Why can't that work with 2 Gunicorn workers? (each has its own `ITEMS`!)

## Debugging practice
Delete `global _next_id` and POST: `UnboundLocalError: cannot access local variable '_next_id'` — Python sees the assignment `_next_id += 1` and treats the name as local. Traceback points at the exact line.

## Rebuild checklist
- [ ] `/health` GET route.
- [ ] POST route that validates JSON and returns 201.
- [ ] Path parameter route returning 404 for missing.
- [ ] A test using `test_client()`.

## The new problem (cliffhanger)
Flask worked, but every check (is `a` a number? is `name` present?) was a hand-written try/except, and nothing documents the API. Next: [04_fastapi_basics](../04_fastapi_basics/).

## Say it aloud
> **Say it aloud (interview box):** Flask is a small WSGI micro-framework where I register functions on routes with decorators, read input from `request`, and return JSON plus a status code. I can test it in-process with `test_client()` without starting a server. Its limit is that validation and docs are all hand-written and the built-in server is for development only.
