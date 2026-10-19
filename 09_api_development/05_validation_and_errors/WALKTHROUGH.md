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
