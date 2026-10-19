# Terms — Step 05: Validation and errors

Folder: `09_api_development/05_validation_and_errors` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [09 API development](../README.md)

**Terms in this folder:** [Input validation](#input-validation-concept) · [Pydantic BaseModel](#pydantic-basemodel-third-party-pydantic) · [Field(min_length, max_length, ge, le, default)](#fieldmin_length-max_length-ge-le-default-third-party-pydantic) · [ValidationError](#validationerror-third-party-pydantic) · [err.errors()](#errerrors-third-party-pydantic) · [model_dump()](#model_dump-third-party-pydantic) · [Lax type conversion in Pydantic](#lax-type-conversion-in-pydantic-concept) · [isinstance()](#isinstance-python-built-in) · [Chained comparison (1 <= len(text) <= 200)](#chained-comparison-1-lentext-200-syntax) · [request.get_json(silent=True)](#requestget_jsonsilenttrue-third-party-flask) · [400 vs 422](#400-vs-422-concept) · [4xx vs 5xx rule](#4xx-vs-5xx-rule-concept) · [HTTP status code cheat sheet](#http-status-code-cheat-sheet-concept) · [errorhandler (Flask)](#errorhandler-flask-third-party-flask) · [jsonify](#jsonify-third-party-flask) · [Reusing one Pydantic model across frameworks](#reusing-one-pydantic-model-across-frameworks-concept) · [FastAPI(), TestClient](#fastapi-testclient-third-party-fastapi) · [Flask test client (data=, headers=)](#flask-test-client-data-headers-third-party-flask) · [Keyword unpacking (**kwargs)](#keyword-unpacking-kwargs-syntax) · [List of tuples as test table](#list-of-tuples-as-test-table-concept) · [f-string alignment ({name:<20}{f.status_code:>7})](#f-string-alignment-name20fstatus_code7-syntax) · [Error raised at import time vs request time](#error-raised-at-import-time-vs-request-time-concept)

---

## Input validation  `(concept)`
**1. Definition:** Checking that incoming data has the expected type, range, size and shape before using it, and rejecting it with a clear error otherwise.
**2. In plain words:** Never trust input from outside your program. A model given an empty string, a 10 MB text or the wrong type may crash or be slow, so you check first and tell the caller what was wrong. This step compares doing it by hand and doing it declaratively with Pydantic.
**3. Example to quote:**
> "Before text reaches my model I check it is a string of 1 to 200 characters. Bad input gets a 422 with the reason, not a 500."

**Seen in:** whole step

## Pydantic BaseModel  `(third-party: pydantic)`
**1. Definition:** A base class for declarative data models; annotated fields are validated and, when safe, converted at instantiation.
**2. In plain words:** You describe the shape once (`text: str`, `top_k: int`) and Pydantic enforces it, so you do not write `isinstance` checks. The same class is used by FastAPI to validate request bodies and to document them.
**3. Example to quote:**
> "`PredictRequest(text='great movie')` returns an object with attributes; `PredictRequest(text='')` raises a `ValidationError`."

```python
from pydantic import BaseModel
class M(BaseModel):
    x: int
print(M(x="5"))   # x=5  (lax mode converts "5" to 5)
```

**Seen in:** `class PredictRequest(BaseModel)`

## Field(min_length, max_length, ge, le, default)  `(third-party: pydantic)`
**1. Definition:** `pydantic.Field(...)` adds constraints and defaults to a model field: `min_length`/`max_length` for strings, `ge`/`le` for numbers (greater/less than or equal), `default` for the fallback value.
**2. In plain words:** It is how you say "1 to 200 characters" or "an integer from 1 to 5, default 1". The errors then say exactly which limit was broken.
**3. Example to quote:**
> "`top_k: int = Field(default=1, ge=1, le=5)` is optional, defaults to 1 and must be between 1 and 5."

**Seen in:** `PredictRequest`

## ValidationError  `(third-party: pydantic)`
**1. Definition:** The exception Pydantic raises when data fails validation; it holds the list of every failing field.
**2. In plain words:** Pydantic collects all the problems instead of stopping at the first. Call `.errors()` to get a list of dicts with `loc`, `msg` and `type`. FastAPI converts this into a 422 response automatically.
**3. Example to quote:**
> "Sending `text=''` and `top_k=99` together raised one `ValidationError` with two entries: text too short and top_k too large."

**Seen in:** `pydantic_demo`

## err.errors()  `(third-party: pydantic)`
**1. Definition:** Returns a list of dictionaries, one per validation problem, with keys such as `loc` (location), `msg` (message) and `type`.
**2. In plain words:** It makes errors machine-readable, so you can show clients exactly which field to fix. `loc` for a body field looks like `['body', 'text']` in FastAPI.
**3. Example to quote:**
> "I loop `for e in err.errors()` and print `e['loc']` and `e['msg']`, such as `('text',) String should have at least 1 character`."

**Seen in:** `pydantic_demo`

## model_dump()  `(third-party: pydantic)`
**1. Definition:** Pydantic v2 method that converts a model instance back into a plain dict (v1 called it `.dict()`).
**2. In plain words:** Useful when you need JSON-friendly data to log, store or return. The old name is deprecated, so use the new one.
**3. Example to quote:**
> "`ok.model_dump()` gives `{'text': 'great movie', 'top_k': 1}`, showing the default was filled in."

**Seen in:** `pydantic_demo`

## Lax type conversion in Pydantic  `(concept)`
**1. Definition:** By default Pydantic converts compatible values to the declared type (for example the string `"3"` to the integer 3) and rejects incompatible ones.
**2. In plain words:** This is friendlier to clients but means "validated" can differ from a hand-written strict check. Decide whether `"3"` should be accepted, and use strict mode if not.
**3. Example to quote:**
> "Pydantic accepts `top_k: '3'` and converts it to 3, whereas my manual Flask `isinstance` check rejects it. I should choose the behaviour on purpose."

**Seen in:** `WALKTHROUGH.md` experiment 2

## isinstance()  `(Python built-in)`
**1. Definition:** `isinstance(obj, cls)` returns `True` if `obj` is an instance of `cls` (or a subclass), otherwise `False`.
**2. In plain words:** It is the hand-written way to check types. In the Flask version each field is checked this way. Pydantic does it declaratively, and `isinstance(True, int)` being `True` is a classic gotcha.
**3. Example to quote:**
> "`isinstance(data, dict)` makes sure the body is a JSON object, not a list or a number."

```python
print(isinstance(5, int))   # True
```

**Seen in:** `flask_predict`

## Chained comparison (1 <= len(text) <= 200)  `(syntax)`
**1. Definition:** Python allows `a <= b <= c`, meaning `a <= b and b <= c`.
**2. In plain words:** It reads like math and is the compact way to check a range. Here it validates text length.
**3. Example to quote:**
> "`1 <= len(text) <= 200` checks the text is non-empty and at most 200 characters in one expression."

```python
print(1 <= 5 <= 10)   # True
```

**Seen in:** `flask_predict`

## request.get_json(silent=True)  `(third-party: flask)`
**1. Definition:** Flask method that parses the request body as JSON, returning `None` instead of raising an error when parsing fails.
**2. In plain words:** Lets the handler decide the error message. If the result is not a dict, the handler returns a 400 for a malformed body.
**3. Example to quote:**
> "With `silent=True`, a body of `hello` gives `None`, so I answer `400 body must be a JSON object`."

**Seen in:** `flask_predict`

## 400 vs 422  `(concept)`
**1. Definition:** 400 Bad Request means the request itself is malformed (for example unparseable JSON); 422 Unprocessable Content means it is syntactically valid but its values break the rules.
**2. In plain words:** "Not JSON at all" is 400; "JSON, but the text is empty" is 422. FastAPI returns 422 even for unparseable JSON bodies, while the hand-written Flask version returns 400, which is why that row of the comparison table differs.
**3. Example to quote:**
> "For the 'not JSON' case Flask returns 400 and FastAPI returns 422. For an empty text both return 422."

**Seen in:** `compare`, `WALKTHROUGH.md` table

## 4xx vs 5xx rule  `(concept)`
**1. Definition:** 4xx status codes mean the client should change the request; 5xx codes mean the server failed and must be fixed.
**2. In plain words:** It decides who is responsible. A crash from bad input handling that returns 500 is your bug; returning 422 for bad input tells the client to retry with corrected data.
**3. Example to quote:**
> "If my API returns 500 for an empty string, that's my validation gap. It should be a 422."

**Seen in:** `WALKTHROUGH.md` cheat sheet

## HTTP status code cheat sheet  `(concept)`
**1. Definition:** The common codes: 200 OK, 201 Created, 204 No Content, 400 Bad Request, 401/403 auth errors, 404 Not Found, 405 Method Not Allowed, 422 Unprocessable Content, 429 Too Many Requests, 500 Internal Server Error, 503 Service Unavailable, 504 Gateway Timeout.
**2. In plain words:** Using the right code lets clients, proxies and monitoring react correctly. 429 and 503 matter for ML services under load, because they tell clients to back off and retry later.
**3. Example to quote:**
> "If my model queue is full I return 429 or 503 so the client retries later, instead of timing out."

**Seen in:** `WALKTHROUGH.md`

## errorhandler (Flask)  `(third-party: flask)`
**1. Definition:** `@app.errorhandler(code)` registers a function that builds the response for a given HTTP error, replacing Flask's default HTML page.
**2. In plain words:** By default Flask returns an HTML page for 404 or 405. An API should return JSON, so you register a handler that does so.
**3. Example to quote:**
> "`@flask_app.errorhandler(404)` returns `{'error': 'route not found'}` as JSON instead of Flask's HTML error page."

**Seen in:** `not_found`

## jsonify  `(third-party: flask)`
**1. Definition:** Flask helper that turns data into a JSON response with `Content-Type: application/json`.
**2. In plain words:** Combined with a tuple, `jsonify(error='...'), 422` returns both a JSON body and a status code.
**3. Example to quote:**
> "`return jsonify(error=\"'top_k' must be an int 1..5\"), 422` sends a JSON error with the 422 status."

**Seen in:** `flask_predict`

## Reusing one Pydantic model across frameworks  `(concept)`
**1. Definition:** A Pydantic model is independent of any web framework, so the same class can be used by FastAPI for request validation and by plain code or Flask handlers.
**2. In plain words:** It shows the model is just data validation. You could use `PredictRequest(**data)` inside Flask too and catch `ValidationError` yourself, getting the same rules in both.
**3. Example to quote:**
> "FastAPI's `fastapi_predict(req: PredictRequest)` uses the same `PredictRequest` class that I called directly in `pydantic_demo`."

**Seen in:** `fastapi_predict`

## FastAPI(), TestClient  `(third-party: fastapi)`
**1. Definition:** `FastAPI()` creates an ASGI app; `TestClient(app)` calls it in-process without a server.
**2. In plain words:** Here a second app is created purely to compare its responses with Flask's under identical inputs.
**3. Example to quote:**
> "I send five identical bad requests to a Flask test client and a FastAPI `TestClient` and print both status codes side by side."

**Seen in:** `compare`

## Flask test client (data=, headers=)  `(third-party: flask)`
**1. Definition:** Flask's `test_client().post(...)` accepts raw `data` bytes and `headers`, while FastAPI's `TestClient` uses `content=` for raw bytes.
**2. In plain words:** The two test clients name arguments differently, so the code maps `content` to `data` for Flask. A good reminder that frameworks differ in small details.
**3. Example to quote:**
> "To send the not-JSON case I pass `data=b'hello'` and a JSON `Content-Type` header to Flask, and `content=b'hello'` to FastAPI."

**Seen in:** `compare`

## Keyword unpacking (**kwargs)  `(syntax)`
**1. Definition:** Writing `**d` in a call passes each key-value pair of the dict `d` as a keyword argument.
**2. In plain words:** Each case is stored as a dict of arguments such as `{'json': {...}}`; `post('/predict', **kwargs)` spreads it into `post(..., json={...})`. It keeps the table of cases tidy.
**3. Example to quote:**
> "`fast_client.post('/predict', **kwargs)` expands the case dict into the keyword arguments the client expects."

```python
def f(a, b): return a + b
print(f(**{"a": 1, "b": 2}))   # 3
```

**Seen in:** `compare`

## List of tuples as test table  `(concept)`
**1. Definition:** A list such as `CASES = [(name, args), ...]` holding named test inputs that a loop runs through.
**2. In plain words:** Add a row and the comparison table grows, with no copy-pasted code. This pattern is also the base of parametrised tests in pytest.
**3. Example to quote:**
> "`CASES` has five named bad requests; one loop sends each to both apps and prints the status codes."

**Seen in:** `CASES`

## f-string alignment ({name:<20}{f.status_code:>7})  `(syntax)`
**1. Definition:** Format specs inside f-strings: `<20` left-aligns in 20 characters and `>7` right-aligns in 7.
**2. In plain words:** It lines the output into neat columns, so the Flask and FastAPI status codes can be compared at a glance.
**3. Example to quote:**
> "`f\"{name:<20}{f.status_code:>7}{g.status_code:>9}\"` prints one row of the comparison table."

```python
print(f"{'ab':<5}{200:>5}")   # ab      200
```

**Seen in:** `compare`

## Error raised at import time vs request time  `(concept)`
**1. Definition:** A mistake in a model definition (for example `min_length="1"`) fails when the module is imported, whereas bad client data fails when a request is processed.
**2. In plain words:** The traceback and the fix differ: one means your code is wrong, the other means the caller's data is wrong. Knowing which you have saves debugging time.
**3. Example to quote:**
> "Changing `min_length=1` to `'1'` crashes as soon as I start the app, with a Pydantic definition error. A bad request only produces a 422."

**Seen in:** `WALKTHROUGH.md` debugging practice
