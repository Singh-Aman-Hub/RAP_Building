# Terms — Step 04: FastAPI basics

Folder: `09_api_development/04_fastapi_basics` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [09 API development](../README.md)

**Terms in this folder:** [FastAPI](#fastapi-third-party-fastapi) · [FastAPI(title=..., version=...)](#fastapititle-version-third-party-fastapi) · [Type hints drive behaviour](#type-hints-drive-behaviour-concept) · [Query parameter](#query-parameter-concept) · [Path parameter ({item_id})](#path-parameter-item_id-syntax) · [Request body and Pydantic model parameter](#request-body-and-pydantic-model-parameter-concept) · [BaseModel](#basemodel-third-party-pydantic) · [Required field](#required-field-concept) · [@app.get / @app.post decorators](#appget-apppost-decorators-syntax) · [status_code (route option)](#status_code-route-option-third-party-fastapi) · [Returning a dict](#returning-a-dict-concept) · [HTTPException](#httpexception-third-party-fastapi) · [422 Unprocessable Content](#422-unprocessable-content-concept) · [TestClient](#testclient-third-party-fastapi) · [client.get(url, params=...) / client.post(url, json=...)](#clientgeturl-params-clientposturl-json-third-party-fastapi) · [Uvicorn command (--reload, --port)](#uvicorn-command---reload---port-command) · [Automatic docs (/docs, /redoc, /openapi.json)](#automatic-docs-docs-redoc-openapijson-concept) · [OpenAPI](#openapi-concept) · [Global mutable state (ITEMS, _next_id)](#global-mutable-state-items-_next_id-concept) · [global statement](#global-statement-syntax) · [f-string](#f-string-syntax) · [Nested function (show)](#nested-function-show-syntax) · [dict indexing (body["detail"][0]["msg"])](#dict-indexing-bodydetail0msg-syntax) · [sorted()](#sorted-python-built-in)

---

## FastAPI  `(third-party: fastapi)`
**1. Definition:** A modern Python web framework for building APIs, built on Starlette (ASGI) and Pydantic, which uses type hints to parse and validate requests and to generate OpenAPI documentation.
**2. In plain words:** You write ordinary functions with typed parameters, and FastAPI figures out where each value comes from (URL, query, body), checks it, and documents the endpoint automatically. Compared with Flask, you write far less validation code. It needs an ASGI server such as Uvicorn to run.
**3. Example to quote:**
> "I rebuilt the Flask items API in FastAPI. The type hints gave me parsing, 422 validation errors and a `/docs` page without extra code."

**Seen in:** `from fastapi import FastAPI`

## FastAPI(title=..., version=...)  `(third-party: fastapi)`
**1. Definition:** Creates the application object; `title` and `version` are metadata that appear in the generated documentation.
**2. In plain words:** `app` is what the server runs and where routes are registered. The metadata only affects the docs page, not behaviour.
**3. Example to quote:**
> "`app = FastAPI(title='Items API', version='0.1')` makes the docs page show that name and version."

**Seen in:** top of `main.py`

## Type hints drive behaviour  `(concept)`
**1. Definition:** FastAPI inspects the annotations of a route function's parameters to decide each one's source (path, query or body) and type, and to validate and convert incoming data accordingly.
**2. In plain words:** `def add(a: float, b: float)` tells FastAPI "these are query parameters and they must be numbers". If someone sends `a=x`, the function never runs and the caller gets a 422. In plain Python, hints are only documentation; FastAPI makes them do real work.
**3. Example to quote:**
> "`a: float` means FastAPI tries `float('x')`, fails, and returns 422 before my code runs. In Flask I had to write the `try/except` myself."

**Seen in:** `add`

## Query parameter  `(concept)`
**1. Definition:** An optional piece of data in the URL after `?`, such as `?a=2&b=3`; in FastAPI any function parameter not in the path and not a Pydantic model is a query parameter.
**2. In plain words:** It is the usual way to pass small inputs with GET. FastAPI converts it to the hinted type automatically, and giving the parameter a default makes it optional.
**3. Example to quote:**
> "`GET /add?a=2&b=3` returns `{'result': 5.0}` because both query values were converted to float for me."

**Seen in:** `add`

## Path parameter ({item_id})  `(syntax)`
**1. Definition:** A variable part of the URL declared in curly braces in the route, whose value is passed to the function parameter of the same name and converted to its hinted type.
**2. In plain words:** `/items/{item_id}` with `item_id: int` means `/items/7` calls `get_item(7)`. `/items/abc` is rejected with 422 because `abc` is not an integer.
**3. Example to quote:**
> "`GET /items/abc` returns 422 'Input should be a valid integer' because `item_id: int` could not parse it."

**Seen in:** `get_item`

## Request body and Pydantic model parameter  `(concept)`
**1. Definition:** When a route parameter is annotated with a Pydantic model class, FastAPI reads the JSON request body, validates it against the model and passes you the resulting object.
**2. In plain words:** `create_item(item: ItemIn)` means "the body must look like `ItemIn`". You then use `item.name` as a normal attribute. A missing or wrong-typed field produces a 422 automatically.
**3. Example to quote:**
> "`POST /items` with `{}` returns 422 'Field required' because `ItemIn.name` has no default."

**Seen in:** `create_item`

## BaseModel  `(third-party: pydantic)`
**1. Definition:** Pydantic's base class for data models; fields declared as annotated class attributes are validated and converted when an instance is created.
**2. In plain words:** It is a typed, self-checking description of data shape. `class ItemIn(BaseModel): name: str` says "an ItemIn has a required string `name`". Pydantic raises a `ValidationError` for invalid data; FastAPI turns that into a 422 response.
**3. Example to quote:**
> "`ItemIn` is my request schema. If a client sends `{'name': 5}`, Pydantic rejects it because 5 is not a string."

```python
from pydantic import BaseModel
class ItemIn(BaseModel):
    name: str
print(ItemIn(name="pen"))   # name='pen'
```

**Seen in:** `class ItemIn(BaseModel)`

## Required field  `(concept)`
**1. Definition:** A model field with no default value must be supplied by the caller; omitting it causes a validation error.
**2. In plain words:** Writing `name: str` without `= something` is how you say "mandatory". Giving a default (like `price: float = 0`) makes it optional.
**3. Example to quote:**
> "`name: str` is required, so an empty JSON body gets 422 'Field required'. Adding `= ''` would make it optional."

**Seen in:** `ItemIn`

## @app.get / @app.post decorators  `(syntax)`
**1. Definition:** Decorators that register the function below as the handler for a path and HTTP method, optionally with settings such as `status_code`.
**2. In plain words:** They work like Flask's, with extra options. `@app.post("/items", status_code=201)` sets the success status for you, so the function just returns the data.
**3. Example to quote:**
> "`@app.post('/items', status_code=201)` tells FastAPI to answer 201 Created when my function returns normally."

**Seen in:** all routes

## status_code (route option)  `(third-party: fastapi)`
**1. Definition:** An argument of the route decorator specifying the HTTP status returned on success.
**2. In plain words:** The default is 200. For creation use 201 so clients can tell the difference. Error statuses are raised with `HTTPException` instead.
**3. Example to quote:**
> "My create endpoint declares `status_code=201`, so a successful POST returns 201 and the new item."

**Seen in:** `@app.post("/items", status_code=201)`

## Returning a dict  `(concept)`
**1. Definition:** FastAPI converts a returned dict, list or Pydantic model to a JSON response automatically.
**2. In plain words:** There is no `jsonify` step as in Flask; return the plain data and FastAPI serialises it with the right `Content-Type`.
**3. Example to quote:**
> "`return {'status': 'ok'}` is enough; FastAPI sends it as JSON."

**Seen in:** `health`, `add`

## HTTPException  `(third-party: fastapi)`
**1. Definition:** An exception that, when raised in a route, stops processing and returns an HTTP error response with the given `status_code` and a JSON body `{"detail": ...}`.
**2. In plain words:** It is how you return errors such as 404 from deep inside your function. Because it is an exception it ends the handler immediately, so no `return` is needed.
**3. Example to quote:**
> "`raise HTTPException(status_code=404, detail='item 99 not found')` gives the client `404 {'detail': 'item 99 not found'}`."

**Seen in:** `get_item`

## 422 Unprocessable Content  `(concept)`
**1. Definition:** The HTTP status FastAPI returns when a request is well-formed but its data fails validation, with a JSON list describing each failing field.
**2. In plain words:** The JSON parsed fine, but the values are wrong (a string where a number was expected, a missing field). The response includes `loc` (where), `msg` (why) and `type`, so a client can fix the exact field.
**3. Example to quote:**
> "`/add?a=x&b=3` returns 422 with 'Input should be a valid number', and my `add` function never runs."

**Seen in:** `demo`, `WALKTHROUGH.md` output

## TestClient  `(third-party: fastapi)`
**1. Definition:** `fastapi.testclient.TestClient(app)` is an in-process HTTP client (built on httpx) that calls a FastAPI app without starting a server.
**2. In plain words:** It is FastAPI's equivalent of Flask's `test_client()`. Tests are fast and need no open ports. It requires `httpx` to be installed.
**3. Example to quote:**
> "`client = TestClient(app)` lets my demo call `/health` and check the response without launching Uvicorn."

**Seen in:** `demo`

## client.get(url, params=...) / client.post(url, json=...)  `(third-party: fastapi)`
**1. Definition:** Test-client methods that send requests; `params=` builds the query string and `json=` sends a JSON body with the right header.
**2. In plain words:** You pass Python dicts and the client encodes them. `params={'a': 2, 'b': 3}` becomes `?a=2&b=3`.
**3. Example to quote:**
> "`client.get('/add', params={'a': 'x', 'b': 3})` deliberately sends a non-number to trigger the 422."

**Seen in:** `demo`

## Uvicorn command (--reload, --port)  `(command)`
**1. Definition:** `uvicorn main:app --port 8000` runs the ASGI app named `app` in `main.py`; `--reload` restarts it when files change and `--workers N` starts several processes.
**2. In plain words:** The `module:variable` string tells Uvicorn what to import. `--reload` is only for development because it watches files and adds overhead.
**3. Example to quote:**
> "In development I run `uvicorn main:app --port 8000 --reload`; in production I'd drop `--reload` and set `--workers`."

```bash
uvicorn main:app --port 8000 --reload
```

**Seen in:** docstring, `WALKTHROUGH.md`

## Automatic docs (/docs, /redoc, /openapi.json)  `(concept)`
**1. Definition:** FastAPI generates an OpenAPI schema from route signatures and models, served at `/openapi.json`, with interactive Swagger UI at `/docs` and ReDoc at `/redoc`.
**2. In plain words:** The documentation is built from your code, so it cannot drift out of date. You can click "Try it out" in `/docs` to call endpoints from the browser, which is a quick way to test.
**3. Example to quote:**
> "I opened `/docs`, expanded `POST /items`, and called it from the browser; the same data is machine-readable at `/openapi.json`."

**Seen in:** `demo`, `WALKTHROUGH.md`

## OpenAPI  `(concept)`
**1. Definition:** A standard, language-neutral JSON/YAML format for describing HTTP APIs: paths, parameters, request and response schemas.
**2. In plain words:** It is the "blueprint" other tools use to generate documentation, client code and tests. FastAPI writes it for you from your code.
**3. Example to quote:**
> "`client.get('/openapi.json')` returned paths `/add`, `/health`, `/items` and `/items/{item_id}`, all generated from my functions."

**Seen in:** `demo`

## Global mutable state (ITEMS, _next_id)  `(concept)`
**1. Definition:** Module-level variables modified by request handlers, here an in-memory dict and counter acting as a fake database.
**2. In plain words:** Works for a demo, but data vanishes on restart and each worker process has its own copy. A real service uses a database.
**3. Example to quote:**
> "My items live in a dict, so they reset on restart and wouldn't be shared between two Uvicorn workers."

**Seen in:** `ITEMS`, `_next_id`

## global statement  `(syntax)`
**1. Definition:** `global name` lets a function assign to a module-level variable instead of creating a local one.
**2. In plain words:** `create_item` increments `_next_id`; without `global` Python would raise `UnboundLocalError`.
**3. Example to quote:**
> "I declare `global _next_id` before `_next_id += 1` so the counter lives at module level."

**Seen in:** `create_item`

## f-string  `(syntax)`
**1. Definition:** A string prefixed with `f` that evaluates expressions in `{}` and inserts them into the text.
**2. In plain words:** Used to build the error message and the aligned output lines. Format specs such as `{label:<20}` pad the text.
**3. Example to quote:**
> "`detail=f'item {item_id} not found'` becomes 'item 99 not found' for id 99."

```python
item_id = 99
print(f"item {item_id} not found")
```

**Seen in:** `get_item`, `demo`

## Nested function (show)  `(syntax)`
**1. Definition:** A function defined inside another function to hold a small helper used only there.
**2. In plain words:** `show(label, resp)` formats one result line so `demo` doesn't repeat the printing code.
**3. Example to quote:**
> "I defined `show` inside `demo` so each call is one short line and the formatting lives in one place."

**Seen in:** `demo`

## dict indexing (body["detail"][0]["msg"])  `(syntax)`
**1. Definition:** Chained `[...]` lookups that go step by step into nested dicts and lists.
**2. In plain words:** A 422 response's JSON has a `detail` list; the first entry is a dict with `msg`. Chaining `[...]` pulls that text out for compact printing.
**3. Example to quote:**
> "`body['detail'][0]['msg']` extracts the first validation error's message, like 'Field required'."

**Seen in:** `demo`

## sorted()  `(Python built-in)`
**1. Definition:** `sorted(iterable)` returns a new sorted list from the items of any iterable (for a dict, its keys).
**2. In plain words:** It makes the printed paths appear in a stable order. It does not change the original data.
**3. Example to quote:**
> "`sorted(client.get('/openapi.json').json()['paths'])` lists the API's paths alphabetically."

```python
print(sorted({"b": 1, "a": 2}))   # ['a', 'b']
```

**Seen in:** `demo`
