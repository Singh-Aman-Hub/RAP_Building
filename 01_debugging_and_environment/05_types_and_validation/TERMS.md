# Terms — Type Errors and Input Validation

Folder: `01_debugging_and_environment/05_types_and_validation` · Files covered: `main.py`, `broken_types.py`, `WALKTHROUGH.md`
Back to module: [01 Debugging and environment README](../README.md)

**Terms in this folder:** [Pydantic](#pydantic) · [BaseModel](#basemodel) · [Field constraints](#field-constraints) · [ValidationError](#validationerror) · [Input validation](#input-validation) · [isinstance()](#isinstance) · [raise TypeError / ValueError](#raise-typeerror--valueerror) · [except with a tuple of types](#except-with-a-tuple-of-types) · [repr in error messages](#repr-in-error-messages) · [Type coercion (lax mode)](#type-coercion-lax-mode) · [error_count() and errors()](#error_count-and-errors) · [Pydantic default values](#pydantic-default-values) · [bool is a subclass of int](#bool-is-a-subclass-of-int) · [extra="forbid"](#extraforbid) · [HTTP 422](#http-422) · [str.join with non-strings](#strjoin-with-non-strings) · [Implicit None return](#implicit-none-return) · [AttributeError on NoneType](#attributeerror-on-nonetype) · [str + int TypeError](#str--int-typeerror)

---

## Pydantic  `(third-party: pydantic)`
**1. Definition:** Pydantic is a third-party library that validates and converts data according to Python type hints, producing typed objects or a detailed `ValidationError`.
**2. In plain words:** You describe the shape of the data you expect (a prompt string, a number between 0 and 512, ...) and Pydantic checks incoming data against it. It's the engine behind FastAPI's request validation.
**3. Example to quote:**
> "I use Pydantic to declare my request schema, so bad input is rejected with a clear message instead of crashing deep inside my code."
```python
from pydantic import BaseModel
```
**Seen in:** `main.py`, import

## BaseModel  `(third-party: pydantic)`
**1. Definition:** `pydantic.BaseModel` is the base class you subclass to define a schema; each annotated attribute becomes a validated field.
**2. In plain words:** `class GenerationRequest(BaseModel)` with `prompt: str` means "an object with a `prompt` that must be text". Creating an instance validates the data on the spot.
**3. Example to quote:**
> "`GenerationRequest(prompt='Hello', max_tokens='30')` produced a typed object where `max_tokens` is the int 30."
```python
class Req(BaseModel):
    prompt: str
```
**Seen in:** `main.py`, `GenerationRequest`

## Field constraints  `(third-party: pydantic)`
**1. Definition:** `Field(...)` attaches constraints and defaults to a model field, such as `min_length`, `gt` (greater than), `ge` (greater or equal), `lt`, `le` (less or equal).
**2. In plain words:** `Field(default=20, gt=0, le=512)` says "optional, defaults to 20, must be above 0 and at most 512". `Field(default=1.0, ge=0, le=2)` encodes the allowed temperature range.
**3. Example to quote:**
> "I wrote `max_tokens: int = Field(default=20, gt=0, le=512)` so values outside 1 to 512 are rejected automatically."
```python
temperature: float = Field(default=1.0, ge=0, le=2)
```
**Seen in:** `main.py`, `GenerationRequest`

## ValidationError  `(third-party: pydantic)`
**1. Definition:** `pydantic.ValidationError` is raised when data fails a model's validation; it holds a list of every problem found.
**2. In plain words:** Instead of stopping at the first mistake, Pydantic collects them all. One call with an empty prompt, `-5` tokens and temperature 9 reports three problems at once.
**3. Example to quote:**
> "Pydantic reported 3 problems at once: empty prompt, max_tokens not greater than 0, and temperature above 2."
```python
try:
    GenerationRequest(prompt="")
except ValidationError as err:
    print(err.error_count())
```
**Seen in:** `main.py`

## Input validation  `(concept)`
**1. Definition:** Input validation is checking data for correct type, range and shape at the boundary where it enters your program, before using it.
**2. In plain words:** Stopping bad data at the door gives a clear message ("max_tokens must be int, got str: '20'") instead of a confusing crash three functions later. This idea reappears in tool arguments (module 08) and FastAPI (module 09).
**3. Example to quote:**
> "I validate at the boundary, so a wrong value is rejected with a clear message instead of exploding later."
```python
if value <= 0:
    raise ValueError("must be > 0")
```
**Seen in:** `main.py`, `set_max_tokens`; `WALKTHROUGH.md` goal

## isinstance()  `(Python built-in)`
**1. Definition:** `isinstance(obj, cls)` returns `True` if `obj` is an instance of `cls` (or of a subclass).
**2. In plain words:** It's the standard type check, as in `isinstance(value, int)`. Caution: `bool` is a subclass of `int`, so `True` passes an int check.
**3. Example to quote:**
> "`isinstance('20', int)` is `False`, so my function rejected the string with a TypeError."
```python
print(isinstance(5, int), isinstance("5", int))   # True False
```
**Seen in:** `main.py`, `set_max_tokens`

## raise TypeError / ValueError  `(syntax)`
**1. Definition:** `raise` with `TypeError` signals a wrong type; with `ValueError` it signals a right type but unacceptable value.
**2. In plain words:** Choosing the right class helps callers handle each case. The message should include what was expected and what was received.
**3. Example to quote:**
> "I raise `TypeError` for `'20'` and `ValueError` for `-1`, each with a message showing the bad value."
```python
raise ValueError(f"max_tokens must be > 0, got {value}")
```
**Seen in:** `main.py`, `set_max_tokens`

## except with a tuple of types  `(syntax)`
**1. Definition:** `except (TypeError, ValueError) as err:` catches either listed exception type with one handler.
**2. In plain words:** When both error kinds should be handled the same way, group them in a tuple instead of writing two blocks.
**3. Example to quote:**
> "I caught `(TypeError, ValueError)` together to print 'rejected' for any of the bad candidates."
```python
try:
    int("x")
except (TypeError, ValueError) as err:
    print(err)
```
**Seen in:** `main.py`, candidate loop

## repr in error messages  `(concept)`
**1. Definition:** Using `{value!r}` in an error message shows the value's `repr`, including quotes for strings and `None` as-is.
**2. In plain words:** `got str: '20'` immediately shows it was a string, while `got NoneType: None` shows a missing value. It removes guessing.
**3. Example to quote:**
> "My message says `got str: '20'`, and the quotes tell me it's text, not a number."
```python
print(f"got {type('20').__name__}: {'20'!r}")
```
**Seen in:** `main.py`, `set_max_tokens`

## Type coercion (lax mode)  `(concept)`
**1. Definition:** Coercion is automatic conversion of compatible input to the declared type; Pydantic's default (lax) mode converts `"30"` to the int `30`.
**2. In plain words:** Useful for data from forms or JSON, but it can surprise you. Impossible values like `"abc"` for an int are still rejected. Strict schemas reject even convertible values.
**3. Example to quote:**
> "`max_tokens='30'` became the int 30, but `'abc'` would have been rejected."
```python
good = GenerationRequest(prompt="Hello", max_tokens="30")
print(type(good.max_tokens).__name__)   # int
```
**Seen in:** `main.py`, `good`; `WALKTHROUGH.md`

## error_count() and errors()  `(third-party: pydantic)`
**1. Definition:** On a `ValidationError`, `error_count()` returns the number of problems and `errors()` returns a list of dicts with `loc` (field location), `msg` and `type`.
**2. In plain words:** You can loop over `errors()` and print exactly which field failed and why, which is how an API builds a helpful 422 response.
**3. Example to quote:**
> "I looped over `err.errors()` and printed `('prompt',) String should have at least 1 character`."
```python
for e in err.errors():
    print(e["loc"], e["msg"])
```
**Seen in:** `main.py`

## Pydantic default values  `(syntax)`
**1. Definition:** A field assigned a value (or `Field(default=...)`) is optional and takes that value when omitted.
**2. In plain words:** `max_tokens` defaults to 20, so a request containing only a prompt is still valid. Required fields have no default.
**3. Example to quote:**
> "`temperature` defaulted to 1.0 when I didn't pass it, as the printed model showed."
```python
print(GenerationRequest(prompt="Hi").max_tokens)   # 20
```
**Seen in:** `main.py`, `GenerationRequest`

## bool is a subclass of int  `(concept)`
**1. Definition:** In Python `bool` inherits from `int`, so `True == 1` and `isinstance(True, int)` is `True`.
**2. In plain words:** A strict "must be int" check using `isinstance` therefore lets `True` through. Be aware when validating numbers.
**3. Example to quote:**
> "Passing `max_tokens=True` passed my `isinstance` check because bool is a subclass of int."
```python
print(isinstance(True, int))   # True
```
**Seen in:** `WALKTHROUGH.md`, experiment 1

## extra="forbid"  `(third-party: pydantic)`
**1. Definition:** Setting `model_config = {"extra": "forbid"}` on a Pydantic model makes it reject fields that are not declared.
**2. In plain words:** By default unknown fields are ignored, which can hide typos like `max_token`. Forbidding extras turns those into errors.
**3. Example to quote:**
> "With `extra: forbid`, a request containing `max_token` (typo) is rejected instead of silently ignored."
```python
class Req(BaseModel):
    model_config = {"extra": "forbid"}
```
**Seen in:** `WALKTHROUGH.md`, experiment 3

## HTTP 422  `(concept)`
**1. Definition:** HTTP 422 "Unprocessable Entity" is the status code FastAPI returns when a request body fails validation.
**2. In plain words:** The body contains the same list of field errors Pydantic produces, so clients know what to fix. You will see this in module 09.
**3. Example to quote:**
> "FastAPI answers invalid requests with 422 and the list of field errors from Pydantic."
```text
POST /predict {"text": ""} -> 422
```
**Seen in:** `main.py`, comment; `WALKTHROUGH.md`, say it aloud

## str.join with non-strings  `(Python built-in)`
**1. Definition:** `" ".join(items)` requires every item to be a `str`; otherwise it raises `TypeError: sequence item 0: expected str instance, int found`.
**2. In plain words:** In `broken_types.py`, token IDs (ints) were passed where words (strings) were expected. The message names the position and expected type.
**3. Example to quote:**
> "`' '.join([101, 7592])` raised a TypeError because token IDs are ints, not text."
```python
print(" ".join(["hello", "world"]))
```
**Seen in:** `broken_types.py`, bug 1

## Implicit None return  `(concept)`
**1. Definition:** A function without a `return` statement returns `None`.
**2. In plain words:** `clean()` computed `text.strip().lower()` but forgot to return it, so the caller got `None`. A later `.split()` on `None` crashes. Always return the result.
**3. Example to quote:**
> "My `clean` function forgot `return`, so `first_word(clean(...))` received `None`."
```python
def clean(text):
    return text.strip().lower()
```
**Seen in:** `broken_types.py`, bug 2

## AttributeError on NoneType  `(Python built-in)`
**1. Definition:** `'NoneType' object has no attribute 'split'` means a method was called on `None`.
**2. In plain words:** `NoneType` almost always means a function forgot to return or a lookup found nothing. Print `type(x)` and `repr(x)` just above the failing line.
**3. Example to quote:**
> "The error `'NoneType' object has no attribute 'split'` made me check whether the previous function returned a value."
```python
print(type(None).__name__)   # NoneType
```
**Seen in:** `broken_types.py`, bug 2; `WALKTHROUGH.md`

## str + int TypeError  `(concept)`
**1. Definition:** Adding a string and a number with `+` raises `TypeError`, because Python won't convert implicitly.
**2. In plain words:** `total_length(...) + "1"` adds an int to a string. Values read from `input()` or environment variables are strings, so convert them with `int()` first.
**3. Example to quote:**
> "`3 + '1'` fails; I convert with `int('1')` because env var values are always strings."
```python
print(3 + int("1"))   # 4
```
**Seen in:** `broken_types.py`, bug 3
