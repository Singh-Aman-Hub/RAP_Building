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
