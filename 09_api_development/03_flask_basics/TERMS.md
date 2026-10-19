# Terms — Step 03: Flask basics

Folder: `09_api_development/03_flask_basics` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [09 API development](../README.md)

**Terms in this folder:** [Flask](#flask-third-party-flask) · [Flask(__name__)](#flask__name__-third-party-flask) · [Route decorator (@app.get / @app.post)](#route-decorator-appget-apppost-syntax) · [jsonify](#jsonify-third-party-flask) · [request (flask)](#request-flask-third-party-flask) · [Query string and request.args](#query-string-and-requestargs-concept) · [float()](#float-python-built-in) · [try / except with several exceptions](#try-except-with-several-exceptions-syntax) · [KeyError](#keyerror-python-built-in) · [Status code as tuple](#status-code-as-tuple-syntax) · [201 Created / 400 / 404 / 405](#201-created-400-404-405-concept) · [request.get_json(silent=True)](#requestget_jsonsilenttrue-third-party-flask) · [global statement](#global-statement-syntax) · [Global in-memory state (ITEMS dict)](#global-in-memory-state-items-dict-concept) · [Path parameter (<int:item_id>)](#path-parameter-intitem_id-syntax) · [dict.get](#dictget-python-built-in) · [Type hint dict[int, dict]](#type-hint-dictint-dict-syntax) · [app.test_client()](#apptest_client-third-party-flask) · [client.post(url, json=...)](#clientposturl-json-third-party-flask) · [response.status_code / get_json()](#responsestatus_code-get_json-third-party-flask) · [Nested helper function](#nested-helper-function-syntax) · [f-string format spec ({label:<20})](#f-string-format-spec-label20-syntax) · [app.run(host, port, debug)](#apprunhost-port-debug-third-party-flask) · [sys.argv](#sysargv-standard-library) · [Development server vs production server](#development-server-vs-production-server-concept) · [UnboundLocalError](#unboundlocalerror-python-built-in)

---

## Flask  `(third-party: flask)`
**1. Definition:** A lightweight Python micro-framework for building web applications and APIs, implementing the WSGI interface.
**2. In plain words:** Flask gives you the basics (routing, request/response objects, JSON helpers) and leaves everything else, such as validation and docs, to you. That makes it quick to learn and flexible, but you write more by hand than in FastAPI.
**3. Example to quote:**
> "I built the items API in Flask: a few decorated functions, no config. Flask has no built-in validation or auto docs, so I write those checks myself."

**Seen in:** `from flask import Flask, jsonify, request`

## Flask(__name__)  `(third-party: flask)`
**1. Definition:** Creates the Flask application object; `__name__` is the current module's name, used by Flask to locate resources like templates and static files.
**2. In plain words:** `app` is the central object: routes are registered on it and servers run it. Passing `__name__` is the standard idiom; when the file is run directly, `__name__` is `"__main__"`.
**3. Example to quote:**
> "`app = Flask(__name__)` creates my application, and every route decorator hangs off `app`."

**Seen in:** top of `main.py`

## Route decorator (@app.get / @app.post)  `(syntax)`
**1. Definition:** A decorator that registers the function beneath it as the handler for a given URL path and HTTP method.
**2. In plain words:** A decorator is a function that wraps another function. `@app.get("/health")` says "when a GET for `/health` arrives, call this function". The same path can have different handlers for different methods.
**3. Example to quote:**
> "`@app.post('/items')` over `create_item` means POST requests to `/items` run that function; GET requests to the same path would need their own route."

```python
@app.get("/health")
def health():
    return {"status": "ok"}
```

**Seen in:** all route definitions

## jsonify  `(third-party: flask)`
**1. Definition:** `flask.jsonify(...)` converts its arguments (keyword arguments or a single object) to JSON and returns a `Response` with `Content-Type: application/json`.
**2. In plain words:** It saves you from calling `json.dumps` and setting headers by hand. Returning `jsonify(status="ok")` produces `{"status": "ok"}` with the right content type.
**3. Example to quote:**
> "`return jsonify(result=a + b)` sends `{'result': 5.0}` as JSON with the correct header."

**Seen in:** every handler

## request (flask)  `(third-party: flask)`
**1. Definition:** A proxy object for the HTTP request currently being handled, exposing `args`, `form`, `headers`, `get_json()` and more.
**2. In plain words:** Inside a handler, `request` is "the thing the client sent". It looks global but always refers to the current request, even with many requests in flight.
**3. Example to quote:**
> "I read the query string from `request.args` and the JSON body with `request.get_json()`."

**Seen in:** `add`, `create_item`

## Query string and request.args  `(concept)`
**1. Definition:** The part of a URL after `?`, such as `?a=2&b=3`; Flask exposes it as `request.args`, a dict-like object whose values are strings.
**2. In plain words:** Query parameters are optional inputs in the URL, typically used with GET. Because everything arrives as text, you must convert numbers yourself (`float(...)`) and handle conversion errors.
**3. Example to quote:**
> "`/add?a=2&b=3` gives `request.args` of `{'a': '2', 'b': '3'}`, so I call `float()` on each before adding."

**Seen in:** `add`

## float()  `(Python built-in)`
**1. Definition:** `float(x)` converts a number or numeric string to a floating-point number, raising `ValueError` if the string is not numeric.
**2. In plain words:** Query values are strings, so `"2" + "3"` would give `"23"`. `float("2") + float("3")` gives `5.0`. A non-numeric value raises `ValueError`, which the code turns into a 400.
**3. Example to quote:**
> "`float(request.args['a'])` turns '2' into 2.0; `a=x` raises `ValueError` and I reply 400."

```python
print(float("2") + float("3"))   # 5.0
```

**Seen in:** `add`

## try / except with several exceptions  `(syntax)`
**1. Definition:** `except (KeyError, ValueError):` catches any of the listed exception types in one clause.
**2. In plain words:** Missing parameter raises `KeyError`; bad number raises `ValueError`. Both mean "client sent bad input", so one handler returns a 400 for both.
**3. Example to quote:**
> "`except (KeyError, ValueError)` covers both a missing `a` or `b` and a non-numeric value, and returns a 400."

**Seen in:** `add`

## KeyError  `(Python built-in)`
**1. Definition:** The exception raised when a dictionary key is not found, e.g. `request.args["a"]` when `a` is absent.
**2. In plain words:** It means "I looked for a key that is not there". Use `dict.get` to avoid it, or catch it deliberately when a missing key is a client error.
**3. Example to quote:**
> "Calling `/add` with no query string raises `KeyError: 'a'`, which I catch and turn into `400`."

**Seen in:** `add`

## Status code as tuple  `(syntax)`
**1. Definition:** In Flask a handler can return a tuple `(body, status)` to set the HTTP status code, e.g. `return jsonify(error="..."), 400`.
**2. In plain words:** Without the second element the status is 200. A tuple is the quickest way to return an error status or 201 for a creation.
**3. Example to quote:**
> "`return jsonify(item), 201` tells the client the item was created; `return jsonify(error=...), 404` says it was not found."

**Seen in:** `create_item`, `get_item`

## 201 Created / 400 / 404 / 405  `(concept)`
**1. Definition:** Standard HTTP status codes: 201 means a resource was created, 400 means a malformed request, 404 means not found, 405 means the method is not allowed for that path.
**2. In plain words:** Each code tells the client what to do next. POST that creates something should say 201, not just 200. A 405 happens when the path exists but not for that verb, as with `DELETE /items/1` here.
**3. Example to quote:**
> "My `POST /items` returns 201, a missing item returns 404, and `DELETE /items/1` returns 405 because I defined no DELETE route."

**Seen in:** `demo`, `WALKTHROUGH.md` output

## request.get_json(silent=True)  `(third-party: flask)`
**1. Definition:** Parses the request body as JSON and returns the result; with `silent=True` it returns `None` instead of raising an error when parsing fails.
**2. In plain words:** Clients often send empty or broken bodies. With `silent=True` you can check `if not data` and return a clean 400 yourself. Without it, Flask raises its own error, which you would otherwise have to handle.
**3. Example to quote:**
> "`data = request.get_json(silent=True)` returns `None` for bad JSON, so I check `if not data or 'name' not in data` and return 400."

**Seen in:** `create_item`

## global statement  `(syntax)`
**1. Definition:** `global name` inside a function declares that assignments to `name` refer to the module-level variable instead of creating a local one.
**2. In plain words:** Without it, `_next_id += 1` makes Python treat `_next_id` as local and raises `UnboundLocalError`. Global mutable state like this is fine for a demo but unsafe in real servers.
**3. Example to quote:**
> "I declare `global _next_id` so incrementing the counter changes the module-level one. Without it I get `UnboundLocalError`."

**Seen in:** `create_item`

## Global in-memory state (ITEMS dict)  `(concept)`
**1. Definition:** Data kept in a Python variable inside the running process; it is lost on restart and not shared between processes.
**2. In plain words:** `ITEMS` acts as a fake database. With two Gunicorn workers each worker would have its own copy, so an item created in one would be missing in the other. Real APIs store state in a database or shared cache.
**3. Example to quote:**
> "My `ITEMS` dict resets on restart and wouldn't be shared across Gunicorn workers, so a real service needs a database."

**Seen in:** `ITEMS`, `WALKTHROUGH.md` experiments

## Path parameter (<int:item_id>)  `(syntax)`
**1. Definition:** A variable part of a URL path declared with angle brackets in the route, with an optional converter such as `int`, passed to the handler as an argument.
**2. In plain words:** `/items/<int:item_id>` matches `/items/7` and calls `get_item(7)`, already converted to an integer. A path like `/items/abc` does not match the `int` converter, so Flask returns 404 before your code runs.
**3. Example to quote:**
> "`/items/<int:item_id>` gives my function `item_id=7` as a real integer for the URL `/items/7`."

**Seen in:** `get_item`

## dict.get  `(Python built-in)`
**1. Definition:** `d.get(key)` returns the value for `key` or `None` if absent, without raising `KeyError`.
**2. In plain words:** A lookup that is allowed to fail quietly. `ITEMS.get(item_id)` returns `None` for an unknown id, which the code turns into a 404.
**3. Example to quote:**
> "`ITEMS.get(99)` returns `None`, so I respond `404 {'error': 'item 99 not found'}`."

```python
print({}.get("a"))   # None
```

**Seen in:** `get_item`

## Type hint dict[int, dict]  `(syntax)`
**1. Definition:** An annotation such as `dict[int, dict]` stating that a variable is a dictionary with `int` keys and `dict` values; it is not enforced at runtime.
**2. In plain words:** It documents the shape of `ITEMS` (id to item). Tools and readers benefit, and later FastAPI and Pydantic turn such hints into real validation.
**3. Example to quote:**
> "`ITEMS: dict[int, dict] = {}` tells the reader this maps item ids to item dicts."

**Seen in:** `ITEMS`

## app.test_client()  `(third-party: flask)`
**1. Definition:** Returns a test client that sends requests to the Flask app in-process, without a network socket.
**2. In plain words:** It is fast and needs no running server, which makes it perfect for automated tests. You call `client.get("/health")` and inspect `status_code` and `get_json()`.
**3. Example to quote:**
> "My `demo()` uses `app.test_client()`, so I can call `/items` and check the 201 without opening any port."

**Seen in:** `demo`

## client.post(url, json=...)  `(third-party: flask)`
**1. Definition:** Sends a POST through the test client; the `json=` argument serialises the object into the body and sets `Content-Type: application/json`.
**2. In plain words:** You do not encode or set headers yourself. Compare with the raw `urllib` code in step 01, where you did both by hand.
**3. Example to quote:**
> "`client.post('/items', json={'name': 'pen'})` sends the JSON body and the right header, and I get back 201."

**Seen in:** `demo`

## response.status_code / get_json()  `(third-party: flask)`
**1. Definition:** Attributes and methods of a test response: `status_code` is the integer HTTP status and `get_json()` parses the body as JSON (returning `None` if the body is not JSON).
**2. In plain words:** They let a test assert both the status and the content. The 405 case prints `None` because Flask's default 405 page is HTML.
**3. Example to quote:**
> "I print `resp.status_code` and `resp.get_json()`; for the DELETE it shows `405 None` because Flask answered with HTML."

**Seen in:** `demo`, `show`

## Nested helper function  `(syntax)`
**1. Definition:** A function defined inside another function, visible only within it, used here as `show(label, resp)`.
**2. In plain words:** It avoids repeating the same print formatting for every call in `demo`, without adding to the module's namespace.
**3. Example to quote:**
> "`show` is defined inside `demo` just to format output, so I don't repeat the `print` line seven times."

**Seen in:** `demo`

## f-string format spec ({label:<20})  `(syntax)`
**1. Definition:** Inside an f-string, `{value:<20}` left-aligns the value in a field 20 characters wide.
**2. In plain words:** It makes the printed columns line up so the results are easy to scan.
**3. Example to quote:**
> "`f\"{label:<20}-> {resp.status_code}\"` pads the label to 20 characters so the arrows line up."

```python
print(f"{'a':<5}|")   # a    |
```

**Seen in:** `show`

## app.run(host, port, debug)  `(third-party: flask)`
**1. Definition:** Starts Flask's built-in development server on the given host and port; `debug=True` enables auto-reload and an interactive debugger.
**2. In plain words:** This is only for local development. Never expose `debug=True` publicly, because the debugger allows running code. In production use a WSGI server like Gunicorn. Port 5001 is used here because macOS AirPlay often occupies 5000.
**3. Example to quote:**
> "`python main.py --serve` runs `app.run(host='127.0.0.1', port=5001, debug=False)`. For production I would use Gunicorn instead."

**Seen in:** bottom of `main.py`

## sys.argv  `(standard library)`
**1. Definition:** A list of command-line arguments passed to the script; `sys.argv[0]` is the script name.
**2. In plain words:** The script checks `"--serve" in sys.argv` to choose between running a demo and starting a server. It is the simplest way to add a flag without a library like `argparse`.
**3. Example to quote:**
> "`python main.py --serve` puts `'--serve'` in `sys.argv`, so the script starts the real dev server."

**Seen in:** `if __name__ == "__main__":`

## Development server vs production server  `(concept)`
**1. Definition:** Flask's built-in server is single-purpose and not hardened or efficient; production uses a dedicated WSGI server such as Gunicorn.
**2. In plain words:** The dev server is good for trying things out. It lacks the process management, robustness and performance of Gunicorn, so a real deployment never uses it.
**3. Example to quote:**
> "I use `app.run` while developing, but in the Docker image the command is `gunicorn`, which is built for production traffic."

**Seen in:** `WALKTHROUGH.md` "Say it aloud"

## UnboundLocalError  `(Python built-in)`
**1. Definition:** An exception raised when a function reads a local variable before it has been assigned, often because an assignment makes the name local.
**2. In plain words:** If a function assigns to a name (like `_next_id += 1`) Python treats it as local for the whole function, so reading it first fails. Declare it `global` or avoid the global state.
**3. Example to quote:**
> "Removing `global _next_id` gives 'cannot access local variable `_next_id`', pointing at the exact line in the traceback."

**Seen in:** `WALKTHROUGH.md` debugging practice
