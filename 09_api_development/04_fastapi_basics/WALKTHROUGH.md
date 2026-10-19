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
