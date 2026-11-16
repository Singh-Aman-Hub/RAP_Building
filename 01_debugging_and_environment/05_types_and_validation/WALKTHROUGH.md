# Walkthrough 05 — Type errors and input validation

## Goal
Stop bad data at the door with clear messages, instead of letting it explode three functions later.

## Run it
```bash
cd 01_debugging_and_environment/05_types_and_validation
python main.py
python broken_types.py      # EXPECTED crash (fix the 3 bugs one at a time)
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| `isinstance` vs `type(x) == T` | type check | No | `isinstance` also accepts subclasses; `type(x) == T` does not. |
| validation vs parsing/coercion | check vs convert | No | Pydantic does both by default: it converts `"30"` to `30` if it can. |
| `ValidationError` vs `ValueError` | Pydantic error vs built-in | Related | Pydantic v2's `ValidationError` subclasses `ValueError`. |
| type hint vs validation | annotation vs enforcement | No | Hints alone are not enforced; Pydantic enforces them. |
| `TypeError` vs `AttributeError` on None | operation vs lookup on wrong object | No | Calling a method on `None` is `AttributeError`; adding `None` to a number is `TypeError`. |

## Code, step by step

### 1. Header docstring  (`main.py` lines 1–10)

```python
"""
GOAL:  Catch wrong-type / wrong-value input early using isinstance checks and Pydantic.

RUN:   cd 01_debugging_and_environment/05_types_and_validation && python main.py
       Crashing version (EXPECTED):  python broken_types.py

EXPECTED OUTPUT: friendly validation errors instead of crashes deep inside the code.

NEEDS: pydantic (installed in .venv), no internet.
"""
```

- **In simple words:** Label for the file: catch wrong input early.
- **Exactly:** Docstring; nothing runs.
- **Why it is here:** Names the lesson (fail at the door, not deep inside).

### 2. Import Pydantic pieces  (`main.py` lines 12–12)

```python
from pydantic import BaseModel, Field, ValidationError  # pydantic: validates data using type hints
```

- **In simple words:** Bring in the validation toolkit.
- **Exactly:** `BaseModel` (a class whose fields are checked using type hints), `Field` (adds rules such as min length), `ValidationError` (the exception raised on bad data). Pydantic v2 here.
- **Why it is here:** Pydantic is what FastAPI uses for request validation.

### 3. Hand-written validation function  (`main.py` lines 15–21)

```python
# ---- hand-written validation ---------------------------------------------------------
def set_max_tokens(value):
    if not isinstance(value, int):                 # isinstance(obj, type) -> bool
        raise TypeError(f"max_tokens must be int, got {type(value).__name__}: {value!r}")
    if value <= 0:
        raise ValueError(f"max_tokens must be > 0, got {value}")
    return value
```

- **In simple words:** A bouncer: reject values that are not a positive integer, with a clear message.
- **Exactly:** `isinstance(value, int)` is a type check; `"20"` fails it. `type(value).__name__` names the type (`str`, `NoneType`). A second check rejects `value <= 0`. Raises `TypeError` for wrong type, `ValueError` for bad value, else returns the value.
- **Why it is here:** Without it, a string would travel on and crash later with a confusing error. Note `True` is an `int` in Python, so `isinstance(True, int)` passes.

### 4. Try four candidates  (`main.py` lines 24–28)

```python
for candidate in [20, "20", -1, None]:
    try:
        print("ok:", set_max_tokens(candidate))
    except (TypeError, ValueError) as err:         # catch several types with a tuple
        print(f"rejected -> {type(err).__name__}: {err}")
```

- **In simple words:** Feed four inputs and show which pass.
- **Exactly:** `except (TypeError, ValueError) as err` catches either type. Real output: `ok: 20`; `rejected -> TypeError: max_tokens must be int, got str: '20'`; `rejected -> ValueError: max_tokens must be > 0, got -1`; `rejected -> TypeError: max_tokens must be int, got NoneType: None`.
- **Why it is here:** Shows the message names what was wrong and what was received.

### 5. The same rules as a Pydantic model  (`main.py` lines 31–35)

```python
# ---- Pydantic: the same idea declared with type hints -----------------------------------
class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=1)              # at least one character
    max_tokens: int = Field(default=20, gt=0, le=512)   # gt = greater than, le = less or equal
    temperature: float = Field(default=1.0, ge=0, le=2)
```

- **In simple words:** Declare the rules once as a class; Pydantic checks them automatically.
- **Exactly:** `prompt: str = Field(min_length=1)` (required, non-empty). `max_tokens: int = Field(default=20, gt=0, le=512)` means 0 < x <= 512. `temperature: float = Field(default=1.0, ge=0, le=2)` means 0 <= x <= 2.
- **Why it is here:** Rules sit next to the field names, are reusable and generate error lists for free.

### 6. Valid input with coercion  (`main.py` lines 38–39)

```python
good = GenerationRequest(prompt="Hello", max_tokens="30")   # "30" is coerced to int 30
print("\nvalid:", good, "| type of max_tokens:", type(good.max_tokens).__name__)
```

- **In simple words:** Pydantic converts compatible values: the text `"30"` becomes the integer 30.
- **Exactly:** Output: `valid: prompt='Hello' max_tokens=30 temperature=1.0 | type of max_tokens: int`. In default "lax" mode `"30"` -> `30`; `"abc"` would fail.
- **Why it is here:** Handy for query strings and env vars, but a surprise if you expected strictness.

### 7. Invalid input: collect all errors  (`main.py` lines 41–46)

```python
try:
    GenerationRequest(prompt="", max_tokens=-5, temperature=9)
except ValidationError as err:
    print(f"\n{err.error_count()} problems found:")
    for e in err.errors():                         # list of dicts: where / what / why
        print("  -", e["loc"], e["msg"])
```

- **In simple words:** Break three rules at once and list them all.
- **Exactly:** `ValidationError` holds every failure: `err.error_count()` is `3`; `err.errors()` is a list of dicts with `loc` (field path tuple) and `msg`. Real output: `('prompt',) String should have at least 1 character`, `('max_tokens',) Input should be greater than 0`, `('temperature',) Input should be less than or equal to 2`.
- **Why it is here:** Unlike the hand-written function that stops at the first problem, the caller gets the complete list.

### 8. Comment and TRY/BREAK IT lines  (`main.py` lines 48–51)

```python
# This is exactly what FastAPI does for you in module 09 (it answers HTTP 422 with this list).

# TRY IT: add a field `stop: list[str] = []` and pass stop="x" -> read the error message.
# BREAK IT: see broken_types.py for the 3 most common type bugs.
```

- **In simple words:** Link to FastAPI and two experiments (comments only).
- **Exactly:** FastAPI converts a `ValidationError` into an HTTP 422 response with this same list.
- **Why it is here:** Connects to the API module later.

### 9. broken_types.py: docstring  (`broken_types.py` lines 1–1)

```python
"""EXPECTED TO CRASH. Three typical type bugs; the script stops at the first. Fix one, re-run, repeat."""
```

- **In simple words:** Announces three typical type bugs; the script stops at the first.
- **Exactly:** One-line docstring.
- **Why it is here:** Sets the fix-one-re-run drill.

### 10. broken_types: three small helper functions  (`broken_types.py` lines 4–13)

```python
def tokens_to_text(tokens):
    return " ".join(tokens)


def first_word(text):
    return text.split()[0]


def total_length(words):
    return sum(len(w) for w in words)
```

- **In simple words:** Join tokens, take the first word, add up word lengths.
- **Exactly:** `" ".join(tokens)` requires every item to be `str`. `text.split()[0]` requires `text` to be a string. `sum(len(w) for w in words)` returns an `int`.
- **Why it is here:** Each helper works with the right input type and fails with the wrong one.

### 11. broken_types bug 1: list of ints joined  (`broken_types.py` lines 16–18)

```python
# Bug 1: a list of ints passed where list of str is expected
print(tokens_to_text(["hello", "world"]))
print(tokens_to_text([101, 7592, 2088]))
```

- **In simple words:** Two calls; the first fine, the second passes numbers.
- **Exactly:** First prints `hello world`. Second raises `TypeError: sequence item 0: expected str instance, int found` inside `join` (line 5), called from line 18 (token ids from a tokenizer instead of text). Fix: `" ".join(map(str, tokens))` or decode the ids.
- **Why it is here:** The script stops here, so bugs 2 and 3 are not reached until you fix this.

### 12. broken_types bug 2: missing return  (`broken_types.py` lines 20–24)

```python
# Bug 2 (reached after fixing 1): a function that returns None implicitly
def clean(text):
    text.strip().lower()          # forgot `return`

print(first_word(clean("  Hello World ")))
```

- **In simple words:** A cleaning function that computes but forgets to return.
- **Exactly:** `clean` ends without `return`, so it returns `None`. `first_word(None)` then calls `None.split()` -> `AttributeError: 'NoneType' object has no attribute 'split'` (raised at line 9 in `first_word`, root cause at line 22).
- **Why it is here:** Classic: the crash is in the caller, the bug is in the callee that forgot `return`.

### 13. broken_types bug 3: int plus str  (`broken_types.py` lines 26–27)

```python
# Bug 3: string where number expected (e.g. value read from input() or an env var)
print(total_length(["a", "bb"]) + "1")
```

- **In simple words:** Add the number 3 to the text "1".
- **Exactly:** `total_length(["a","bb"])` is `3` (int). `3 + "1"` raises `TypeError: unsupported operand type(s) for +: 'int' and 'str'`. Fix: `int("1")` or `str(3) + "1"` depending on intent.
- **Why it is here:** Values from `input()`, env vars and files are strings; convert explicitly.

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
