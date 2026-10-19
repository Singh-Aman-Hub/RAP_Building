# Walkthrough 05 — Type errors and input validation

## Goal
Stop bad data at the door with clear messages, instead of letting it explode three functions later.

## Run it
```bash
cd 01_debugging_and_environment/05_types_and_validation
python main.py
python broken_types.py      # EXPECTED crash (fix the 3 bugs one at a time)
```

## Syntax explained
| Syntax | Meaning |
|---|---|
| `isinstance(x, int)` | Is `x` an int? (also True for `bool` — careful) |
| `except (TypeError, ValueError) as err:` | Handle either type |
| `class M(BaseModel): field: type = Field(...)` | Pydantic model: schema declared with type hints |
| `Field(gt=0, le=512, min_length=1)` | Constraints (greater than, <=, length) |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does | Returns |
|---|---|---|---|
| `isinstance` | built-in | Type check | bool |
| `pydantic.BaseModel` | 3rd-party | Validate/convert data into typed objects | model instance (raises `ValidationError`) |
| `ValidationError.errors()` | 3rd-party | All problems: `loc`, `msg`, `type` | list of dicts |

## Execution trace (real output)
```
ok: 20
rejected -> TypeError: max_tokens must be int, got str: '20'
rejected -> ValueError: max_tokens must be > 0, got -1
rejected -> TypeError: max_tokens must be int, got NoneType: None

valid: prompt='Hello' max_tokens=30 temperature=1.0 | type of max_tokens: int
3 problems found:
  - ('prompt',) String should have at least 1 character
  - ('max_tokens',) Input should be greater than 0
  - ('temperature',) Input should be less than or equal to 2
```
Note Pydantic *coerced* `"30"` to `30` (lax mode) but rejected impossible values, and reported **all** problems at once.
`broken_types.py` first fails with `TypeError: sequence item 0: expected str instance, int found` (token IDs passed where text expected). Bug 2: `clean()` has no `return` -> returns `None` -> `AttributeError: 'NoneType' object has no attribute 'split'`. Bug 3: `int + "1"` -> `TypeError`.

## Experiments
1. Pass `max_tokens=True` to `set_max_tokens` — it passes `isinstance(..., int)` (bool is a subclass of int).
2. Add `stop: list[str] = []` to the Pydantic model and pass `stop="x"`; read the error.
3. Use `model_config = {"extra": "forbid"}` and pass an unknown field.

## Debugging practice
For `TypeError`/`AttributeError`: print `type(x)` and `repr(x)` of the offending variable right above the failing line. `'NoneType'` almost always means a function forgot to `return` or a lookup found nothing.

## Rebuild checklist
- [ ] Write `isinstance` validation raising a helpful error
- [ ] Define a Pydantic model with 3 constrained fields
- [ ] Explain coerce vs reject

## Say it aloud
"I validate input at the boundary. Hand-written checks work for small cases; Pydantic declares the schema with type hints, converts compatible values and reports every violation. FastAPI uses it to return HTTP 422 automatically."
Trade-off: validation costs a little CPU and strict schemas can reject slightly different but valid data from clients.
