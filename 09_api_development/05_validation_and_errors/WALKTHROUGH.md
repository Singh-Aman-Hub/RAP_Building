# Step 05 — Validation, Pydantic and HTTP status codes

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

## Say it aloud
"Never trust input: validate type, range and size before it reaches the model. Pydantic does that declaratively and FastAPI returns a 422 listing every bad field; in Flask I validate by hand or add Pydantic myself. I return 4xx when the client is wrong and 5xx when I am. Trade-off: strict validation rejects some borderline inputs, so error messages must be clear."
