# Step 04 — FastAPI fundamentals

## The story so far
Flask made routes easy but left validation and documentation to us. Step 03's items API is the same API we rebuild here, so the only difference is what the framework gives for free.

## The pain point
In Flask, bad input means writing `try/except`, choosing a status code, and keeping separate docs in sync with the code. Forgetting one check lets junk reach the model.

## The spark
Read the type hints. A parameter typed `float` must be a float, a Pydantic class is a JSON body, and FastAPI returns 422 and generates `/docs` from the same declarations.

## Goal
Rebuild step 03's API with FastAPI and notice what the type hints give you for free: parsing, validation (422), docs.

## Run it
```bash
cd 09_api_development/04_fastapi_basics
python main.py                           # TestClient demo
uvicorn main:app --port 8000 --reload    # real server; open http://127.0.0.1:8000/docs
```
Real output of `python main.py`:
```
GET  /health        -> 200 {'status': 'ok'}
GET  /add?a=2&b=3   -> 200 {'result': 5.0}
GET  /add?a=x&b=3   -> 422 Input should be a valid number, unable to parse string as a number
POST /items         -> 201 {'id': 1, 'name': 'pen'}
GET  /items/1       -> 200 {'id': 1, 'name': 'pen'}
GET  /items/99      -> 404 {'detail': 'item 99 not found'}
GET  /items/abc     -> 422 Input should be a valid integer, unable to parse string as an integer
POST /items (empty) -> 422 Field required
docs are generated from the code at /docs and the schema at /openapi.json:
  paths -> ['/add', '/health', '/items', '/items/{item_id}']
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Pydantic model | schema, `BaseModel`, DTO, data class with validation | Related | A class whose fields have types, validated and converted when you create an instance. |
| 422 vs 400 | Unprocessable Entity vs Bad Request | No | FastAPI uses 422 for well-formed JSON that fails validation; 400 is for malformed requests (and is what Flask code often uses for both). |
| Query param vs path param vs body | URL query, URL segment, JSON payload | No | FastAPI infers which from the signature: path name in the route, scalar otherwise = query, Pydantic model = body. |
| `HTTPException` vs `abort` (Flask) | raise an error status | Similar | Both stop the handler and return an error; FastAPI's carries `detail`, Flask's `abort` returns an HTML error page by default. |
| OpenAPI vs Swagger UI vs `/docs` | API spec, docs page | No | OpenAPI is the JSON spec (`/openapi.json`); Swagger UI is the interactive page rendered from it at `/docs`; ReDoc at `/redoc` is another. |
| Uvicorn vs FastAPI | ASGI server vs framework | No | FastAPI defines the app; uvicorn runs it and speaks HTTP. |
| `TestClient` | httpx client, Flask `test_client` | Similar | Both call the app in-process with no network. |

## Code, step by step

### 1. The file header  (`main.py` lines 1–18)

```python
"""
GOAL:   Learn FastAPI fundamentals: the same tiny "items" API as step 03, but with type hints
        doing the parsing/validation work and automatic interactive docs.

RUN:    cd 09_api_development/04_fastapi_basics
        python main.py                     # demo with FastAPI's TestClient (no server needed)
        uvicorn main:app --port 8000       # real server; then open http://127.0.0.1:8000/docs
        uvicorn main:app --port 8000 --reload   # dev mode: restarts when you save the file

EXPECTED OUTPUT (shape):
        GET  /health        -> 200 {'status': 'ok'}
        GET  /add?a=2&b=3   -> 200 {'result': 5.0}
        GET  /add?a=x&b=3   -> 422 (validation error list)
        POST /items         -> 201 {'id': 1, 'name': 'pen'}
        GET  /items/99      -> 404 {'detail': 'item 99 not found'}

NEEDS:  fastapi, uvicorn, httpx (TestClient uses httpx)
"""
```

- **In simple words:** What the file teaches: the same items API as step 03, done with type hints.
- **Exactly:** Docstring only. It notes that `uvicorn main:app` serves it and `/docs` shows generated docs.
- **Why it is here:** Orientation.

### 2. Imports and create the FastAPI app  (`main.py` lines 20–24)

```python
from fastapi import FastAPI, HTTPException   # FastAPI = app class; HTTPException = raise to return an error status
from fastapi.testclient import TestClient    # lets us call the app in-process, like Flask's test_client
from pydantic import BaseModel               # base class for data shapes with automatic validation

app = FastAPI(title="Items API", version="0.1")   # title/version appear in the auto-generated docs
```

- **In simple words:** Load FastAPI, a test client and Pydantic, then make the app.
- **Exactly:** `FastAPI` is the ASGI application class. `HTTPException` is raised to return an error status. `TestClient` (built on httpx) calls the app in-process. `BaseModel` is Pydantic's base class. `FastAPI(title=..., version=...)` stores metadata that goes into `/openapi.json` and `/docs`.
- **Why it is here:** `app` is what uvicorn is pointed at (`main:app`).

### 3. Fake database and id counter  (`main.py` lines 26–27)

```python
ITEMS: dict[int, dict] = {}
_next_id = 1
```

- **In simple words:** A dict as a pretend database and the next id.
- **Exactly:** Same as step 03: in-memory, lost on restart, not shared across worker processes.
- **Why it is here:** Keeps the focus on the framework, not storage.

### 4. Declare the request body shape with Pydantic  (`main.py` lines 30–31)

```python
class ItemIn(BaseModel):          # a Pydantic model = the SHAPE of the JSON we accept
    name: str                     # required string; a missing or non-string value -> automatic 422
```

- **In simple words:** A form template: the body must have a `name` that is text.
- **Exactly:** `ItemIn(BaseModel)` with `name: str` means the JSON must be an object with a required string `name`. Missing gives `422 Field required` (real). In Pydantic v2's default (lax) mode some conversions happen (`"5"` to int for an `int` field), but a number is NOT turned into a string, so `{"name": 5}` gives a 422.
- **Why it is here:** The model is both validation and documentation: it appears in the OpenAPI schema.

### 5. The /health route  (`main.py` lines 34–36)

```python
@app.get("/health")
def health():
    return {"status": "ok"}       # return a plain dict; FastAPI converts it to JSON
```

- **In simple words:** Same health check as Flask, but we just return a dict.
- **Exactly:** `@app.get("/health")` registers the path operation. A returned dict is converted to JSON with `jsonable_encoder`; status is 200.
- **Why it is here:** No `jsonify` call needed: less boilerplate.

### 6. The /add route: parameters from type hints  (`main.py` lines 39–41)

```python
@app.get("/add",status_code=200)
def add(a: float, b: float):      # type hints on parameters = query params, parsed AND validated for us
    return {"result": a + b}
```

- **In simple words:** Write the function the way you would in plain Python, and FastAPI figures out the URL parts and checks them.
- **Exactly:** Parameters not in the path and not Pydantic models are query parameters. `a: float` means: take `a` from `?a=`, parse to float, reject with 422 if it cannot. `status_code=200` is the default, stated explicitly. Real: `a=2&b=3` returns `{'result': 5.0}`; `a=x` returns `422 Input should be a valid number, unable to parse string as a number` (the `msg` of the first error). Your function never runs for bad input.
- **Why it is here:** This replaces Flask's 8 lines of manual `try/except float(...)` in step 03.

### 7. The POST /items route with a Pydantic body  (`main.py` lines 44–50)

```python
@app.post("/items", status_code=201)   # status_code sets the success status
def create_item(item: ItemIn):         # a Pydantic-typed parameter = the JSON request body
    global _next_id
    new = {"id": _next_id, "name": item.name}
    ITEMS[_next_id] = new
    _next_id += 1
    return new
```

- **In simple words:** Accept a JSON body shaped like `ItemIn`, create the item, return 201.
- **Exactly:** A parameter typed with a Pydantic model (`item: ItemIn`) is read from the JSON body. FastAPI parses and validates it, then calls the function with an `ItemIn` object, so `item.name` is a `str`. `status_code=201` sets the success status. Real: `{'id': 1, 'name': 'pen'}` with 201; body `{}` gives `422 Field required`. The returned dict is serialised to JSON.
- **Why it is here:** Replaces the manual `get_json(silent=True)` and `"name" not in data` checks.

### 8. The GET /items/{item_id} route and 404  (`main.py` lines 53–57)

```python
@app.get("/items/{item_id}")      # {item_id} = path parameter; the `int` hint below converts + validates it
def get_item(item_id: int):
    if item_id not in ITEMS:
        raise HTTPException(status_code=404, detail=f"item {item_id} not found")   # becomes {"detail": ...}
    return ITEMS[item_id]
```

- **In simple words:** Look up an item by its id in the URL and complain properly if missing.
- **Exactly:** `{item_id}` in the path is a path parameter; the `int` hint converts `"1"` to `1` and rejects `"abc"` with `422 Input should be a valid integer, unable to parse string as an integer` (real). `raise HTTPException(status_code=404, detail=...)` stops the function and FastAPI sends `404 {'detail': 'item 99 not found'}`. Note the key is `detail`, not Flask's custom `error`.
- **Why it is here:** Raising an exception is how you leave a path operation early with an error status.

### 9. Demo helper: test client and a print function  (`main.py` lines 60–66)

```python
def demo():
    client = TestClient(app)
    def show(label, resp):
        body = resp.json()
        if resp.status_code == 422:                      # keep the validation output short
            body = body["detail"][0]["msg"]
        print(f"{label:<20}-> {resp.status_code} {body}")
```

- **In simple words:** Run the API in-process and print each result on one line.
- **Exactly:** `TestClient(app)` wraps the ASGI app and sends requests using httpx without a network. `show` prints status and body; for 422 responses it shortens the body to `body["detail"][0]["msg"]`, because FastAPI's 422 body is `{"detail": [ {type, loc, msg, input}, ...]}`.
- **Why it is here:** Keeps the output readable.

### 10. Call each route and list the generated docs paths  (`main.py` lines 67–76)

```python
    show("GET  /health", client.get("/health"))
    show("GET  /add?a=2&b=3", client.get("/add", params={"a": 2, "b": 3}))
    show("GET  /add?a=x&b=3", client.get("/add", params={"a": "x", "b": 3}))   # BREAK IT: not a float
    show("POST /items", client.post("/items", json={"name": "pen"}))
    show("GET  /items/1", client.get("/items/1"))
    show("GET  /items/99", client.get("/items/99"))
    show("GET  /items/abc", client.get("/items/abc"))    # path param not an int -> 422
    show("POST /items (empty)", client.post("/items", json={}))
    print("docs are generated from the code at /docs and the schema at /openapi.json:")
    print("  paths ->", sorted(client.get("/openapi.json").json()["paths"]))
```

- **In simple words:** Send eight requests, then ask the app for its own self-description.
- **Exactly:** `params={"a": 2, "b": 3}` builds the query string. Real output: `/health` 200; `/add` 200 `{'result': 5.0}`; `a=x` 422; `POST /items` 201 `{'id': 1, 'name': 'pen'}`; `/items/1` 200; `/items/99` 404 `{'detail': 'item 99 not found'}`; `/items/abc` 422; empty body 422 `Field required`. `/openapi.json` is generated from the code: `sorted(...["paths"])` gives `['/add', '/health', '/items', '/items/{item_id}']`.
- **Why it is here:** Proves docs come from the same code that handles requests, so they cannot drift.

### 11. Entry point: run the demo  (`main.py` lines 77–81)

```python
    # TRY IT: change `a: float` to `a: int` and call /add?a=2.5.


if __name__ == "__main__":
    demo()
```

- **In simple words:** When run directly, execute the demo. The TRY IT line suggests an experiment.
- **Exactly:** `if __name__ == "__main__": demo()`. The real server is started separately with `uvicorn main:app --port 8000`, so the demo does not need a port.
- **Why it is here:** The guard keeps imports (by uvicorn) from triggering the demo.


## Syntax explained
| Syntax | Meaning | Example |
|---|---|---|
| `def add(a: float, b: float)` | **Type hints drive behaviour**: names not in the path become *query params*, converted to float | `/add?a=1&b=2` |
| `item: ItemIn` | A Pydantic model parameter = JSON **request body** | — |
| `class ItemIn(BaseModel): name: str` | Declares the data shape; `name: str` = required string | — |
| `@app.post("/items", status_code=201)` | Decorator with options | — |
| `raise HTTPException(404, detail=...)` | Stop and return an error response | — |
| `/items/{item_id}` | Path parameter (curly braces, typed by the function argument) | — |

## Libraries & functions
| Name | Kind | What it does | Important args |
|---|---|---|---|
| `FastAPI(title=, version=)` | fastapi | The app object (ASGI) | metadata shown in docs |
| `BaseModel` | pydantic | Validates + converts input into typed objects | fields as annotations |
| `HTTPException(status_code, detail)` | fastapi | Error → JSON `{"detail": ...}` | — |
| `TestClient(app)` | fastapi (uses httpx) | In-process client | `.get(url, params=)`, `.post(url, json=)` |
| `uvicorn main:app` | uvicorn CLI | ASGI server | `--reload` dev only, `--port`, `--workers` |
| `/docs`, `/redoc`, `/openapi.json` | fastapi | Auto docs generated from your code | — |

## Execution trace
`GET /add?a=x&b=3`: uvicorn gives FastAPI the request → FastAPI sees `a: float` → tries `float("x")` → fails → it **never calls `add`**; it returns `422` with a JSON list saying *which* field failed (`loc`), *why* (`msg`). Compare Flask: your own `try/except` was required.

## Experiments
1. Change `a: float` → `a: int`; call `/add?a=2.5` → 422.
2. Give `a` a default: `a: float = 0` → it becomes optional.
3. Add `class ItemIn: name: str; price: float = 0` and POST with and without price.

## Debugging practice
POST `{"name": 5}` — Pydantic v2 rejects an int for `str` (`Input should be a valid string`). Read `err["loc"] = ["body","name"]` to find the field. If a *route function* itself crashes you instead get a 500 and the traceback in the **server** terminal — learn to look there.

## Rebuild checklist
- [ ] Route with query params typed.
- [ ] POST with Pydantic body + `status_code=201`.
- [ ] Raise a 404 with `HTTPException`.
- [ ] Open `/docs` and call an endpoint from the browser.

## The new problem (cliffhanger)
Type hints give us 422 automatically, but which errors deserve 400, 404, 422 or 500, and how do we make messages useful? Next: [05_validation_and_errors](../05_validation_and_errors/).

## Say it aloud
> **Say it aloud (interview box):** FastAPI is an ASGI framework built on Starlette and Pydantic where type hints decide where each parameter comes from and how it is validated. Bad input returns 422 automatically and the OpenAPI docs appear at `/docs`, served by an ASGI server like Uvicorn. Its limit is more concepts (Pydantic, `def` versus `async def`), and it does not make my model code any faster.
