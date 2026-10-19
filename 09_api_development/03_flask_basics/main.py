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

import sys

from flask import Flask, jsonify, request  # Flask = the app class; jsonify = dict -> JSON response; request = current request

app = Flask(__name__)  # __name__ tells Flask where this module lives (used to find templates/static files)

ITEMS: dict[int, dict] = {}   # our fake "database": id -> item. Lives in memory, lost on restart.
_next_id = 1                  # counter used to create ids


@app.get("/health")           # decorator: "when a GET request arrives for /health, call the function below"
def health():
    return jsonify(status="ok")   # Flask turns the return value into an HTTP response (200 by default)


@app.get("/add")
def add():
    # Query string: /add?a=2&b=3  ->  request.args is a dict-like object {"a": "2", "b": "3"}
    try:
        a = float(request.args["a"])      # all query values arrive as STRINGS -> convert
        b = float(request.args["b"])
    except (KeyError, ValueError):        # missing key OR not a number
        return jsonify(error="query params a and b must be numbers"), 400   # (body, status) tuple
    return jsonify(result=a + b)


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


@app.get("/items/<int:item_id>")          # <int:item_id> = path parameter, converted to int automatically
def get_item(item_id: int):
    item = ITEMS.get(item_id)             # dict.get returns None instead of raising KeyError
    if item is None:
        return jsonify(error=f"item {item_id} not found"), 404
    return jsonify(item)


def demo():
    """Call the app in-process with Flask's test client: fast, no ports, perfect for tests."""
    client = app.test_client()
    def show(label, resp):                # small nested helper function
        print(f"{label:<20}-> {resp.status_code} {resp.get_json()}")
    show("GET  /health", client.get("/health"))
    show("GET  /add?a=2&b=3", client.get("/add?a=2&b=3"))
    show("GET  /add?a=x", client.get("/add?a=x"))                          # BREAK IT: bad input -> 400
    show("POST /items", client.post("/items", json={"name": "pen"}))       # json= sets body AND Content-Type
    show("GET  /items/1", client.get("/items/1"))
    show("GET  /items/99", client.get("/items/99"))
    show("POST /items (empty)", client.post("/items", json={}))
    show("DELETE /items/1", client.delete("/items/1"))                     # no such route/method -> 405
    # TRY IT: add a DELETE route that removes an item and returns 204 with no body.


if __name__ == "__main__":
    if "--serve" in sys.argv:
        app.run(host="127.0.0.1", port=5001, debug=False)   # debug=True auto-reloads but NEVER use it in production
    else:
        demo()
