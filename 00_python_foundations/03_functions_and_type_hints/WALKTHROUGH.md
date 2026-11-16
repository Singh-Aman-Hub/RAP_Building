# Walkthrough 03 — Functions and type hints

## Goal
Write reusable functions with defaults, keyword arguments, return values and type hints — the exact shape of `model.generate(...)`.

## Run it
```bash
cd 00_python_foundations/03_functions_and_type_hints
python main.py
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| parameter vs argument | formal/actual parameter | No | Parameter is the name in `def`; argument is the value passed in the call. |
| type hint | type annotation, type annotation syntax | Same | Not enforced at runtime; tools like mypy and Pydantic read them. |
| `Optional[int]` | `int | None`, `Union[int, None]` | Same meaning | `int | None` is the modern spelling (3.10+). |
| `*args` / `**kwargs` | varargs, rest parameters / keyword varargs | Rough | `*args` -> tuple of positionals; `**kwargs` -> dict of keywords. |
| lambda | anonymous function, inline function | Same | Single expression only, no statements, no name. |
| docstring | documentation string, `__doc__` | Same | String literal as first statement of a function/module/class. |

## Code, step by step

### 1. File header (docstring)  (`main.py` lines 1–11)

```python
"""
GOAL:  Write functions with parameters, defaults, return values and type hints.

RUN:   cd 00_python_foundations/03_functions_and_type_hints && python main.py

EXPECTED OUTPUT (first lines):
    add(2, 3) = 5
    ...

NEEDS: nothing (standard library only), no internet.
"""
```

- **In simple words:** The file's label: goal, run command, expected output.
- **Exactly:** Module docstring; no effect at runtime.
- **Why it is here:** Orients the reader.

### 2. Import Optional and define add  (`main.py` lines 13–21)

```python
from typing import Optional  # Optional[int] means "an int or None" (type-hint helper)


# def name(parameters): body.  `return` hands a value back to the caller.
def add(a, b):
    return a + b


print("add(2, 3) =", add(2, 3))
```

- **In simple words:** A function is a named recipe: give it inputs, it hands back a result. `Optional` is a hint word imported for later.
- **Exactly:** `from typing import Optional` loads a typing helper (`Optional[int]` = `int | None`). `def add(a, b): return a + b` takes two parameters; `add(2, 3)` returns `5`, printed as `add(2, 3) = 5`. `print` with a comma-separated list joins with a space.
- **Why it is here:** Functions remove copy-paste and give a piece of logic a name you can test.

### 3. Default values and keyword arguments  (`main.py` lines 23–32)

```python
# ---- default values and keyword arguments --------------------------------
# Parameters with defaults are optional. This is exactly how generate(max_new_tokens=20) works.
def generate_text(prompt, max_new_tokens=5, temperature=1.0):
    return f"[{prompt}] + {max_new_tokens} tokens @ temp {temperature}"


print(generate_text("Hi"))                                 # uses both defaults
print(generate_text("Hi", 20))                             # positional: fills max_new_tokens
print(generate_text("Hi", temperature=0.2))                # keyword: skip a parameter by name
print(generate_text(prompt="Hi", temperature=0, max_new_tokens=1))  # keywords can be in any order
```

- **In simple words:** Parameters with a default are optional. You can pass values by position or by name.
- **Exactly:** `generate_text(prompt, max_new_tokens=5, temperature=1.0)`. Calls and real output: `generate_text("Hi")` -> `[Hi] + 5 tokens @ temp 1.0`; `generate_text("Hi", 20)` fills `max_new_tokens` positionally -> 20 tokens; `temperature=0.2` skips a parameter by name; all-keyword call works in any order -> `[Hi] + 1 tokens @ temp 0`.
- **Why it is here:** Real APIs (`model.generate(max_new_tokens=20)`) rely on this: you override only what you need.

### 4. Type hints on average and find_index  (`main.py` lines 34–46)

```python
# ---- type hints ----------------------------------------------------------
# `: type` after a parameter and `-> type` after the parentheses are HINTS for humans and tools.
# Python does NOT enforce them at runtime (see the demo below), but editors and Pydantic/FastAPI use them.
def average(numbers: list[float]) -> float:
    return sum(numbers) / len(numbers)          # sum() and len() are built-ins


def find_index(items: list[str], target: str) -> Optional[int]:
    """Return the index of target, or None if absent.  (A docstring: triple-quoted text that documents.)"""
    for i, item in enumerate(items):
        if item == target:
            return i
    return None                                  # explicit "not found"
```

- **In simple words:** Hints are labels saying what types are expected. They help humans and editors, but Python does not enforce them.
- **Exactly:** `numbers: list[float]` and `-> float` annotate input and return. `sum(numbers) / len(numbers)` divides total by count. `find_index` has a docstring (the triple-quoted line, stored in `__doc__`), loops with `enumerate` and returns the index of the first match, else `None`. `Optional[int]` means `int` or `None`.
- **Why it is here:** Hints power editors, `mypy`, Pydantic and FastAPI. `return None` makes the not-found case explicit.

### 5. Call both functions  (`main.py` lines 49–50)

```python
print("average:", average([1.0, 2.0, 6.0]))
print("find_index:", find_index(["a", "b"], "b"), find_index(["a", "b"], "z"))
```

- **In simple words:** Check the two functions with sample data.
- **Exactly:** `average([1.0, 2.0, 6.0])` = `9.0/3` = `3.0`. `find_index(["a","b"], "b")` -> `1`; with `"z"` -> `None`. Output: `average: 3.0` and `find_index: 1 None`.
- **Why it is here:** Confirms the hinted behaviour before we break it.

### 6. Prove hints are not enforced  (`main.py` lines 52–57)

```python
# Hints are not enforced:
print("average of strings 'works' until it crashes:", end=" ")
try:
    print(average(["1", "2"]))                   # wrong type, no hint error, a TypeError inside sum
except TypeError as err:
    print("TypeError ->", err)
```

- **In simple words:** Pass strings where numbers are expected and see that Python only complains when the code actually runs into trouble.
- **Exactly:** `print(..., end=" ")` keeps the cursor on the same line. `average(["1","2"])` has no hint check; `sum` tries `0 + "1"` and raises `TypeError: unsupported operand type(s) for +: 'int' and 'str'`, caught by `except TypeError as err` and printed.
- **Why it is here:** Teaches that type hints are documentation plus tool support, not runtime validation; use Pydantic when you need real checks.

### 7. Return several values  (`main.py` lines 59–65)

```python
# ---- returning several values (a tuple) and unpacking --------------------
def min_max(values: list[int]) -> tuple[int, int]:
    return min(values), max(values)              # returns ONE tuple (low, high)


low, high = min_max([4, 9, 1])                   # tuple unpacking
print("low, high =", low, high)
```

- **In simple words:** A function can seem to return two things; really it returns one tuple that you unpack.
- **Exactly:** `min_max([4, 9, 1])` returns `(1, 9)`. `low, high = ...` unpacks it into `low=1`, `high=9`. Output `low, high = 1 9`. Hint `tuple[int, int]` says a pair of ints.
- **Why it is here:** Common pattern: return `(value, error)` or `(loss, accuracy)` without a class.

### 8. *args and **kwargs  (`main.py` lines 67–73)

```python
# ---- *args and **kwargs --------------------------------------------------
# *args collects extra positional arguments into a tuple; **kwargs collects extra keywords into a dict.
def show(*args, **kwargs):
    print("args:", args, "| kwargs:", kwargs)


show(1, 2, name="gpt2", layers=12)
```

- **In simple words:** `*args` catches extra positional arguments, `**kwargs` catches extra named ones, like a catch-all tray.
- **Exactly:** `show(1, 2, name="gpt2", layers=12)` gives `args == (1, 2)` (a tuple) and `kwargs == {'name': 'gpt2', 'layers': 12}` (a dict). Real output: `args: (1, 2) | kwargs: {'name': 'gpt2', 'layers': 12}`.
- **Why it is here:** Needed for wrappers and decorators (step 07) and for forwarding options to library calls.

### 9. The mutable default trap  (`main.py` lines 75–81)

```python
# ---- scope and the mutable-default trap (classic interview question) -----
def bad_append(x, bucket=[]):                    # the [] is created ONCE, shared by every call
    bucket.append(x)
    return bucket


print(bad_append(1), bad_append(2))              # surprise: [1, 2] [1, 2]  (same list!)
```

- **In simple words:** A default list is created only once, when `def` runs, and then reused by every call.
- **Exactly:** `bad_append(1)` and `bad_append(2)` share the same list object, and `print` evaluates both before printing, so both show `[1, 2]`. Real output: `[1, 2] [1, 2]`.
- **Why it is here:** Classic bug and classic interview question; data leaks between calls.

### 10. The fix: None as default  (`main.py` lines 84–91)

```python
def good_append(x, bucket=None):
    if bucket is None:                           # create a fresh list per call
        bucket = []
    bucket.append(x)
    return bucket


print(good_append(1), good_append(2))            # [1] [2]
```

- **In simple words:** Use `None` as a placeholder and build a fresh list inside the function.
- **Exactly:** `if bucket is None: bucket = []` runs per call, so each call gets its own list. Output `[1] [2]`. Use `is None`, not `== None`.
- **Why it is here:** This is the standard safe pattern for any mutable default (list, dict, set).

### 11. lambda as a sort key  (`main.py` lines 93–95)

```python
# ---- lambda: a tiny anonymous function, handy for sorting keys -----------
words = ["banana", "fig", "apple"]
print(sorted(words, key=lambda w: len(w)))       # sort by length
```

- **In simple words:** A lambda is a one-line unnamed function, handy for "sort by this".
- **Exactly:** `sorted(words, key=lambda w: len(w))` computes the length of each word and sorts by it: `['fig', 'apple', 'banana']` (3, 5, 6). `sorted` returns a new list.
- **Why it is here:** Sorting and filtering by a rule is everyday data work; a lambda avoids a throwaway `def`.

### 12. TRY IT and BREAK IT comments  (`main.py` lines 97–98)

```python
# TRY IT: give generate_text a new parameter top_k=50 and call it with top_k=1.
# BREAK IT: call add(1) (missing argument) and read the TypeError message.
```

- **In simple words:** Experiments for you (comments only).
- **Exactly:** `add(1)` raises `TypeError: add() missing 1 required positional argument: 'b'`.
- **Why it is here:** Practice reading a common error message.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `def f(a, b=2):` | Define function; `b` has a default so it is optional | `def gen(p, max_new_tokens=5)` |
| `return x` | Send a value back (no return -> `None`) | `return a + b` |
| `f(x, temperature=0.2)` | Keyword argument: set a parameter by name | see main.py |
| `def f(x: int) -> str:` | Type hints (documentation, not enforcement) | `def average(n: list[float]) -> float` |
| `Optional[int]` | `int` or `None` | `find_index` |
| `"""docstring"""` | Text documenting the function | first line of `find_index` |
| `return a, b` / `lo, hi = f()` | Return a tuple / unpack it | `min_max` |
| `*args`, `**kwargs` | Collect extra positional args into a tuple / keywords into a dict | `show(1, 2, name="gpt2")` |
| `lambda w: len(w)` | Tiny nameless function | `sorted(words, key=lambda w: len(w))` |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does | Returns |
|---|---|---|---|
| `sum(iterable)`, `len(x)`, `min`, `max`, `sorted(x, key=)` | built-in | Basic aggregation/sorting | number / new list |
| `typing.Optional` | standard library | Type-hint helper only | — |

## Execution trace (real output)
`generate_text("Hi", temperature=0.2)` -> `max_new_tokens` falls back to its default 5:
```
[Hi] + 5 tokens @ temp 0.2
```
Type hints are not enforced: `average(["1", "2"])` fails inside `sum` with
`TypeError: unsupported operand type(s) for +: 'int' and 'str'` (no hint error).
Mutable-default trap:
```
[1, 2] [1, 2]     <- bad_append: the same default list is reused
[1] [2]           <- good_append
```

## Experiments
1. Add `top_k=50` to `generate_text` and call it with `top_k=1`: only the call with the keyword changes.
2. Change the lambda key to `lambda w: w[-1]`: sorts by last letter -> `['banana', 'apple', 'fig']`. Predict first, then run.
3. Call `show()` with no args: prints `args: () | kwargs: {}`.

## Debugging practice
Call `add(1)`: `TypeError: add() missing 1 required positional argument: 'b'`. The message names the function and the missing parameter — no need to read further.

## Rebuild checklist
- [ ] Function with 2 defaulted parameters called positionally and by keyword
- [ ] Function returning two values, unpacked by the caller
- [ ] Explain the mutable default argument bug and fix it with `None`

## Say it aloud
"A function groups reusable steps. Defaults make parameters optional and keyword arguments let me change one setting without touching the rest, like `generate(max_new_tokens=20)`. Type hints document what goes in and out; Python doesn't enforce them, but editors, FastAPI and Pydantic read them. Never use a mutable default like `[]`; use `None` and create the list inside."
Trade-off: hints add clarity but give no runtime safety on their own.
