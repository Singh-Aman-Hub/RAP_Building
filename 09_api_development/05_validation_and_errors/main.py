"""
GOAL:   Compare input validation + error handling by hand (Flask) vs with Pydantic (FastAPI),
        and learn which HTTP status code means what.

RUN:    cd 09_api_development/05_validation_and_errors && python main.py

EXPECTED OUTPUT (shape): first a bare Pydantic demo (ValidationError with a readable list),
        then a table of the same 5 bad requests sent to Flask and to FastAPI with the status each returns.

NEEDS:  flask, fastapi, pydantic
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from flask import Flask, jsonify, request
from pydantic import BaseModel, Field, ValidationError  # Field adds constraints (min_length, ge, le ...)


# ----------------------------- 1. Pydantic on its own ------------------------------
class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=200)   # must be a str, 1..200 characters
    top_k: int = Field(default=1, ge=1, le=5)         # optional int, default 1, allowed 1..5 (ge = >=, le = <=)


def pydantic_demo():
    ok = PredictRequest(text="great movie")           # valid -> an object with attributes
    print("valid  ->", ok, "| ok.text =", ok.text)
    print("dict   ->", ok.model_dump())               # back to a plain dict (Pydantic v2 name)
    try:
        PredictRequest(text="", top_k=99)             # two mistakes at once
    except ValidationError as err:                    # Pydantic collects ALL problems, not just the first
        for e in err.errors():                        # list of dicts: loc (where), msg (why), type
            print("invalid->", e["loc"], e["msg"])


# ----------------------- 2. Flask: validate manually -------------------------------
flask_app = Flask(__name__)


@flask_app.post("/predict")
def flask_predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="body must be a JSON object"), 400          # malformed request
    text = data.get("text")
    if not isinstance(text, str) or not (1 <= len(text) <= 200):
        return jsonify(error="'text' must be a string of 1..200 chars"), 422   # well-formed but invalid
    top_k = data.get("top_k", 1)
    if not isinstance(top_k, int) or not (1 <= top_k <= 5):
        return jsonify(error="'top_k' must be an int 1..5"), 422
    return jsonify(label="positive", top_k=top_k)


@flask_app.errorhandler(404)       # register a function to turn "no such route" into JSON (default is HTML)
def not_found(_err):
    return jsonify(error="route not found"), 404


# ----------------------- 3. FastAPI: the model validates ---------------------------
fastapi_app = FastAPI()


@fastapi_app.post("/predict")
def fastapi_predict(req: PredictRequest):             # same Pydantic model as above, reused as-is
    return {"label": "positive", "top_k": req.top_k}


# ----------------------------- 4. Compare ------------------------------------------
CASES = [
    ("valid",              {"json": {"text": "nice"}}),
    ("empty text",         {"json": {"text": ""}}),
    ("wrong type",         {"json": {"text": 123}}),
    ("top_k out of range", {"json": {"text": "nice", "top_k": 99}}),
    ("not JSON",           {"content": b"hello", "headers": {"Content-Type": "application/json"}}),
]


def compare():
    flask_client = flask_app.test_client()
    fast_client = TestClient(fastapi_app)
    print(f"{'case':<20}{'Flask':>7}{'FastAPI':>9}")
    for name, kwargs in CASES:
        f = flask_client.post("/predict", **({"data": kwargs["content"], "headers": kwargs["headers"]}
                                              if "content" in kwargs else kwargs))   # test clients name the args differently
        g = fast_client.post("/predict", **kwargs)
        print(f"{name:<20}{f.status_code:>7}{g.status_code:>9}")
    print("\nFastAPI 422 body (note 'loc' says WHERE the problem is):")
    print(fast_client.post("/predict", json={"text": ""}).json()["detail"][0])


if __name__ == "__main__":
    pydantic_demo()
    print()
    compare()
    # TRY IT: add `label: str = Field(pattern="^(a|b)$")` to PredictRequest and send a bad label.
    # BREAK IT: change `min_length=1` to `min_length="1"` and read the Pydantic error at import time.
