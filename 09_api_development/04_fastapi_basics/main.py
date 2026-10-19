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

from fastapi import FastAPI, HTTPException   # FastAPI = app class; HTTPException = raise to return an error status
from fastapi.testclient import TestClient    # lets us call the app in-process, like Flask's test_client
from pydantic import BaseModel               # base class for data shapes with automatic validation

app = FastAPI(title="Items API", version="0.1")   # title/version appear in the auto-generated docs

ITEMS: dict[int, dict] = {}
_next_id = 1


class ItemIn(BaseModel):          # a Pydantic model = the SHAPE of the JSON we accept
    name: str                     # required string; a missing or non-string value -> automatic 422


@app.get("/health")
def health():
    return {"status": "ok"}       # return a plain dict; FastAPI converts it to JSON


@app.get("/add")
def add(a: float, b: float):      # type hints on parameters = query params, parsed AND validated for us
    return {"result": a + b}


@app.post("/items", status_code=201)   # status_code sets the success status
def create_item(item: ItemIn):         # a Pydantic-typed parameter = the JSON request body
    global _next_id
    new = {"id": _next_id, "name": item.name}
    ITEMS[_next_id] = new
    _next_id += 1
    return new


@app.get("/items/{item_id}")      # {item_id} = path parameter; the `int` hint below converts + validates it
def get_item(item_id: int):
    if item_id not in ITEMS:
        raise HTTPException(status_code=404, detail=f"item {item_id} not found")   # becomes {"detail": ...}
    return ITEMS[item_id]


def demo():
    client = TestClient(app)
    def show(label, resp):
        body = resp.json()
        if resp.status_code == 422:                      # keep the validation output short
            body = body["detail"][0]["msg"]
        print(f"{label:<20}-> {resp.status_code} {body}")
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
    # TRY IT: change `a: float` to `a: int` and call /add?a=2.5.


if __name__ == "__main__":
    demo()
