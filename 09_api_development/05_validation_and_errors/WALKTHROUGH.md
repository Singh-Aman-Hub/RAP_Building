# Step 05 — Validation, Pydantic and HTTP status codes

## The story so far
Step 04 showed FastAPI rejecting bad input on its own. Before a real model sits behind this door, we must decide exactly what is allowed in and what the caller is told when it is not.

## The pain point
A model that receives an empty string, a 10 MB text or a number where text belongs will crash with a 500, waste compute, or answer nonsense. A bare 500 also tells the client nothing about what to fix.

## The spark
Validate at the door (type, range, size) with Pydantic, and pick the status code by who is at fault: 4xx means the client must change something, 5xx means we broke.

## Goal
Compare hand-written validation (Flask) with Pydantic (FastAPI) and choose the right status code.

## Run it
```bash
cd 09_api_development/05_validation_and_errors && python main.py
```
Real output:
```
valid  -> text='great movie' top_k=1 | ok.text = great movie
dict   -> {'text': 'great movie', 'top_k': 1}
invalid-> ('text',) String should have at least 1 character
invalid-> ('top_k',) Input should be less than or equal to 5

case                  Flask  FastAPI
valid                   200      200
empty text              422      422
wrong type              422      422
top_k out of range      422      422
not JSON                400      422

FastAPI 422 body (note 'loc' says WHERE the problem is):
{'type': 'string_too_short', 'loc': ['body', 'text'], 'msg': 'String should have at least 1 character', 'input': '', 'ctx': {'min_length': 1}}
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Validation | input checking, schema validation, parsing | Overlapping | Validation checks data against rules; Pydantic also converts (parses) it, e.g. `"3"` to `3`. |
| 400 vs 422 | Bad Request vs Unprocessable Entity | No | 400: request could not be understood (bad JSON); 422: understood but fails the rules. Many Flask APIs use 400 for both. |
| Pydantic `BaseModel` vs `dataclass` vs `TypedDict` | data classes | No | Only Pydantic validates and converts at runtime; the others are hints or plain containers. |
| `model_dump()` vs `.dict()` | serialise model to dict | Same job | `model_dump` is Pydantic v2; `.dict()` is the deprecated v1 name. |
| `ValidationError` (Pydantic) vs `RequestValidationError` (FastAPI) | validation exceptions | No | Pydantic raises the first; FastAPI wraps request failures in the second and turns it into the 422 response. |
| `errorhandler` (Flask) vs `exception_handler` (FastAPI) | global error handlers | Same idea | Both map an exception or status to a custom response. |
| `Field(ge=..)` vs `Field(gt=..)` | greater or equal vs greater than | No | `ge` includes the bound; `gt` excludes it (same for `le`/`lt`). |

## Code, step by step

### 1. The file header  (`main.py` lines 1–11)

```python
"""
GOAL:   Compare input validation + error handling by hand (Flask) vs with Pydantic (FastAPI),
        and learn which HTTP status code means what.

RUN:    cd 09_api_development/05_validation_and_errors && python main.py

EXPECTED OUTPUT (shape): first a bare Pydantic demo (ValidationError with a readable list),
        then a table of the same 5 bad requests sent to Flask and to FastAPI with the status each returns.

NEEDS:  flask, fastapi, pydantic
"""
```

- **In simple words:** What the file compares: manual validation (Flask) against Pydantic (FastAPI).
- **Exactly:** Docstring only. It expects a Pydantic demo and then a five-row status-code table.
- **Why it is here:** Orientation.

### 2. Imports  (`main.py` lines 13–16)

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient
from flask import Flask, jsonify, request
from pydantic import BaseModel, Field, ValidationError  # Field adds constraints (min_length, ge, le ...)
```

- **In simple words:** Load both frameworks and Pydantic's pieces.
- **Exactly:** `FastAPI`/`TestClient` and `Flask`/`jsonify`/`request` as in steps 03 and 04. From Pydantic: `BaseModel` (base for validated models), `Field` (adds constraints like `min_length`, `gt`, `ge`, `le`), `ValidationError` (the exception raised when data is invalid).
- **Why it is here:** Both apps run in one file so the same bad requests can be sent to each.

### 3. Define the Pydantic model with constraints  (`main.py` lines 19–22)

```python
# ----------------------------- 1. Pydantic on its own ------------------------------
class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=200)   # must be a str, 1..200 characters
    top_k: int = Field(default=1, gt=6, le=6)         # optional int, default 1, allowed 1..5 (ge = >=, le = <=)
```

- **In simple words:** A form with rules: text must be 1 to 200 characters; top_k is an optional whole number.
- **Exactly:** `text: str = Field(min_length=1, max_length=200)` is required (no default). `top_k: int = Field(default=1, gt=6, le=6)` is optional with default 1. **Note: as written, `gt=6, le=6` means "greater than 6 and at most 6", which no integer satisfies, so any explicit `top_k` is rejected.** The comment and the rest of the lesson intend 1..5, i.e. `ge=1, le=5`. Defaults are not validated, so omitting `top_k` still works (the valid case gives `top_k=1`). Because of this, real output says `Input should be greater than 6` where older text in this file says `less than or equal to 5`.
- **Why it is here:** One class defines type, range and default for the incoming JSON, and FastAPI also derives the docs from it. The constraint values are the bug to spot: always test the boundary values of a constraint.

### 4. Create a valid object and convert it back to a dict  (`main.py` lines 25–28)

```python
def pydantic_demo():
    ok = PredictRequest(text="great movie")           # valid -> an object with attributes
    print("valid  ->", ok, "| ok.text =", ok.text)
    print("dict   ->", ok.model_dump())               # back to a plain dict (Pydantic v2 name)
```

- **In simple words:** Show the happy path: good data becomes an object with attributes.
- **Exactly:** `PredictRequest(text="great movie")` validates and returns an instance: `text='great movie' top_k=1`. `ok.text` is `'great movie'`. `ok.model_dump()` returns `{'text': 'great movie', 'top_k': 1}` (the Pydantic v2 name; v1 called it `.dict()`).
- **Why it is here:** Shows the model is a real typed object, not a dict, and that defaults get filled in.

### 5. Trigger two errors at once and print each  (`main.py` lines 29–33)

```python
    try:
        PredictRequest(text="", top_k=5)             # two mistakes at once
    except ValidationError as err:                    # Pydantic collects ALL problems, not just the first
        for e in err.errors():                        # list of dicts: loc (where), msg (why), type
            print("invalid->", e["loc"], e["msg"])
```

- **In simple words:** Break two rules in one go and see that Pydantic reports both.
- **Exactly:** `PredictRequest(text="", top_k=5)` raises `ValidationError`. `err.errors()` is a list of dicts with `loc` (tuple path), `msg`, `type`, `input`. Real output: `('text',) String should have at least 1 character` and `('top_k',) Input should be greater than 6` (the second because of the constraint bug above; with `ge=1, le=5` this value would pass and only `text` would fail).
- **Why it is here:** Collecting all errors lets a client fix everything in one round trip instead of one error at a time.

### 6. Create the Flask app (manual validation)  (`main.py` lines 36–38)

```python
# ----------------------- 2. Flask: validate manually -------------------------------
flask_app = Flask(__name__)

```

- **In simple words:** The Flask half of the comparison.
- **Exactly:** `Flask(__name__)` as in step 03.
- **Why it is here:** Flask has no built-in validation, so we must write it.

### 7. Flask: parse JSON, reject non-objects with 400  (`main.py` lines 40–44)

```python
@flask_app.post("/predict")
def flask_predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="body must be a JSON object"), 400          # malformed request
```

- **In simple words:** First check the body is a JSON object at all.
- **Exactly:** `request.get_json(silent=True)` returns `None` for invalid JSON; `isinstance(data, dict)` also rejects lists/strings. Status 400 means malformed. Real: the case `not JSON` gives Flask 400.
- **Why it is here:** Everything after assumes `data` is a dict.

### 8. Flask: validate fields by hand  (`main.py` lines 45–51)

```python
    text = data.get("text")
    if not isinstance(text, str) or not (1 <= len(text) <= 200):
        return jsonify(error="'text' must be a string of 1..200 chars"), 422   # well-formed but invalid
    top_k = data.get("top_k", 1)
    if not isinstance(top_k, int) or not (1 <= top_k <= 5):
        return jsonify(error="'top_k' must be an int 1..5"), 422
    return jsonify(label="positive", top_k=top_k)
```

- **In simple words:** Check text and top_k with `if` statements and return 422 with a message if they are wrong.
- **Exactly:** `data.get("text")` is `None` if missing. `isinstance(text, str)` and `1 <= len(text) <= 200` is the chained comparison for "between". `data.get("top_k", 1)` uses 1 as default. Here the range is correctly 1..5. Real: empty text, wrong type (`123`) and `top_k=99` each give 422; valid gives `200 {'label': 'positive', 'top_k': 1}`. Note Flask's ranges are separate code from the Pydantic model's, so the two apps disagree (the model rejects every explicit `top_k`).
- **Why it is here:** This is the boilerplate Pydantic removes; it also shows duplicated rules can drift apart.

### 9. Flask: JSON 404 handler  (`main.py` lines 54–56)

```python
@flask_app.errorhandler(404)       # register a function to turn "no such route" into JSON (default is HTML)
def not_found(_err):
    return jsonify(error="route not found"), 404
```

- **In simple words:** Replace Flask's default HTML 404 page with a JSON one.
- **Exactly:** `@flask_app.errorhandler(404)` registers a function called when a 404 is raised; the `_err` argument is the exception (unused). It returns `(jsonify(...), 404)`.
- **Why it is here:** API clients expect JSON everywhere; HTML error pages break `resp.json()` (we saw `None` for 405 in step 03).

### 10. FastAPI: same endpoint, validation by the model  (`main.py` lines 59–65)

```python
# ----------------------- 3. FastAPI: the model validates ---------------------------
fastapi_app = FastAPI()


@fastapi_app.post("/predict")
def fastapi_predict(req: PredictRequest):             # same Pydantic model as above, reused as-is
    return {"label": "positive", "top_k": req.top_k}
```

- **In simple words:** The FastAPI version is three lines because the model does the checking.
- **Exactly:** `req: PredictRequest` makes the body validated before the function runs; any failure becomes `422` with a `detail` list. If it passes, the function returns `{'label': 'positive', 'top_k': req.top_k}`. Malformed JSON also gives 422 here (Flask gave 400).
- **Why it is here:** Same behaviour with a fraction of the code, reusing the model from block 3.

### 11. The five test cases  (`main.py` lines 68–75)

```python
# ----------------------------- 4. Compare ------------------------------------------
CASES = [
    ("valid",              {"json": {"text": "nice"}}),
    ("empty text",         {"json": {"text": ""}}),
    ("wrong type",         {"json": {"text": 123}}),
    ("top_k out of range", {"json": {"text": "nice", "top_k": 99}}),
    ("not JSON",           {"content": b"hello", "headers": {"Content-Type": "application/json"}}),
]
```

- **In simple words:** A list of five requests, from good to broken.
- **Exactly:** `CASES` is a list of `(name, kwargs)` tuples. `kwargs` are the arguments for the test client: `json=` for JSON bodies, `content=` + `headers=` for raw bytes. The last case sends `b"hello"` labelled as JSON.
- **Why it is here:** Same inputs for both frameworks make the comparison fair.

### 12. Compare: set up clients and print the header row  (`main.py` lines 78–81)

```python
def compare():
    flask_client = flask_app.test_client()
    fast_client = TestClient(fastapi_app)
    print(f"{'case':<20}{'Flask':>7}{'FastAPI':>9}")
```

- **In simple words:** Create one client per framework and print column titles.
- **Exactly:** `flask_app.test_client()` and `TestClient(fastapi_app)`. The f-string `{'case':<20}{'Flask':>7}{'FastAPI':>9}` left-aligns in 20 characters and right-aligns in 7 and 9.
- **Why it is here:** Fixed-width columns line up the status codes.

### 13. Send each case to both apps and print the codes  (`main.py` lines 82–86)

```python
    for name, kwargs in CASES:
        f = flask_client.post("/predict", **({"data": kwargs["content"], "headers": kwargs["headers"]}
                                              if "content" in kwargs else kwargs))   # test clients name the args differently
        g = fast_client.post("/predict", **kwargs)
        print(f"{name:<20}{f.status_code:>7}{g.status_code:>9}")
```

- **In simple words:** For every case, send to Flask, send to FastAPI, print both status codes.
- **Exactly:** The Flask test client wants `data=` for raw bytes while httpx wants `content=`, so the conditional expression renames the argument for Flask; `**` unpacks a dict into keyword arguments. Real table: valid 200/200, empty text 422/422, wrong type 422/422, top_k out of range 422/422, not JSON **400/422**.
- **Why it is here:** The one differing row, not JSON, is the interview point: Flask code chooses its own status; FastAPI uses 422 for every validation failure.

### 14. Print one FastAPI 422 body  (`main.py` lines 87–88)

```python
    print("\nFastAPI 422 body (note 'loc' says WHERE the problem is):")
    print(fast_client.post("/predict", json={"text": ""}).json()["detail"][0])
```

- **In simple words:** Show what the client sees for a validation error.
- **Exactly:** Real: `{'type': 'string_too_short', 'loc': ['body', 'text'], 'msg': 'String should have at least 1 character', 'input': '', 'ctx': {'min_length': 1}}`. `loc` says where (`body`, then field name), `type` is a machine-readable code, `input` is the offending value.
- **Why it is here:** This structure lets a front-end highlight the exact field.

### 15. Entry point and experiments  (`main.py` lines 91–96)

```python
if __name__ == "__main__":
    pydantic_demo()
    print()
    compare()
    # TRY IT: add `label: str = Field(pattern="^(a|b)$")` to PredictRequest and send a bad label.
    # BREAK IT: change `min_length=1` to `min_length="1"` and read the Pydantic error at import time.
```

- **In simple words:** Run the Pydantic demo, a blank line, then the comparison. The comments are ideas to try.
- **Exactly:** `if __name__ == "__main__":` runs `pydantic_demo()`, `print()` and `compare()`. The BREAK IT comment: `min_length="1"` fails when the class is defined (import time), not per request.
- **Why it is here:** Keeps imports side-effect free.


## Status codes cheat sheet
| Code | Name | Use when |
|---|---|---|
| 200 | OK | success with a body |
| 201 | Created | POST created something |
| 204 | No Content | success, nothing to return (DELETE) |
| 400 | Bad Request | malformed request (not JSON, missing header) |
| 401 / 403 | Unauthorized / Forbidden | not logged in / not allowed |
| 404 | Not Found | no such route or resource |
| 405 | Method Not Allowed | route exists, wrong verb |
| 422 | Unprocessable Content | JSON is well-formed but values are invalid |
| 429 | Too Many Requests | rate limit / load shedding (step 10/08) |
| 500 | Internal Server Error | **your** bug — never blame the client |
| 503 | Service Unavailable | overloaded or dependency down |
| 504 | Gateway Timeout | upstream too slow |
Rule: **4xx = the client should change something; 5xx = the server must be fixed.**

## Syntax explained
| Syntax | Meaning | Example |
|---|---|---|
| `Field(min_length=1, max_length=200)` | Constraints on a field | `Field(ge=1, le=5)` |
| `ge`, `le` | greater-or-equal / less-or-equal | — |
| `try / except ValidationError as err` | Pydantic raises one error holding *all* problems | — |
| `err.errors()` | list of dicts: `loc`, `msg`, `type` | — |
| `isinstance(x, str)` | Type check (built-in) | `isinstance(5, int)` → True |
| `"1 <= len(text) <= 200"` | Chained comparison | — |
| `**kwargs` unpacking | Pass a dict as keyword arguments | `f(**{"json": {...}})` |
| `@app.errorhandler(404)` | Flask: customise an error response | — |

## Libraries & functions
| Name | Kind | What it does |
|---|---|---|
| `pydantic.BaseModel` | 3rd-party | Declarative validation, converts types when safe (`"5"` → `5` in lax mode for ints) |
| `model.model_dump()` | pydantic v2 | object → dict (v1 called it `.dict()`) |
| `ValidationError` | pydantic | Raised on invalid data |
| `request.get_json(silent=True)` | flask | `None` instead of raising 400 on bad JSON |

## Execution trace
`{"text": "", "top_k": 99}` into `PredictRequest(...)`: field `text` fails `min_length=1`; field `top_k` fails `le=5`; Pydantic collects both → raises `ValidationError` with two entries → FastAPI converts it to a 422 response without running your function.

## Experiments
1. Add `label: str = Field(pattern="^(a|b)$")` and send a bad label.
2. Send `{"text": "ok", "top_k": "3"}` (string 3) — Pydantic converts to int 3; Flask's manual `isinstance` rejects it. Which behaviour do you want?
3. Add a custom Flask 405 handler returning JSON.

## Debugging practice
Change `min_length=1` to `min_length="1"`: Pydantic fails at **import time** (a bad model definition), not at request time — a different traceback from a bad request. Learn to tell "my model class is wrong" from "the client's data is wrong".

## Rebuild checklist
- [ ] Write a Pydantic model with 2 constrained fields.
- [ ] List which codes you'd return for: bad JSON, missing field, unknown route, wrong verb, crash.

## The new problem (cliffhanger)
The API is safe on toy items. Now it must serve a real model, which is big, slow to load, and must not be loaded on every request. Next: [06_model_api_flask_vs_fastapi](../06_model_api_flask_vs_fastapi/).

## Say it aloud
> **Say it aloud (interview box):** I never trust input, so I validate type, range and size before it reaches the model, and Pydantic does that declaratively. I return a 4xx when the client is wrong and a 5xx when I am, with a clear error message. The cost is that strict validation rejects some borderline inputs, so the messages must tell the caller how to fix them.
