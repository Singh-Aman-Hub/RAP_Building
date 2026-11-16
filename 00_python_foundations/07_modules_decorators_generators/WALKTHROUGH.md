# Walkthrough 07 — Imports, decorators, generators, `__main__`

## Goal
Decode `import`, `@app.get(...)`, `yield` and `if __name__ == "__main__":` — they appear in nearly every FastAPI / PyTorch file.

## Run it
```bash
cd 00_python_foundations/07_modules_decorators_generators
python main.py
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| decorator | wrapper, higher-order function, `@` syntax | Same idea | `@f def g` equals `g = f(g)`. |
| generator vs iterator vs iterable | lazy sequences | No | Iterable can be looped; iterator has `next`; generator is an iterator written with `yield`. |
| module vs package vs library | import unit, folder of modules, distribution | No | Module is one `.py`; package is a folder with modules; library is what you `pip install`. |
| `import x` vs `from x import y` | whole module vs names | No | Second form puts the name directly in your namespace. |
| `Counter` vs `defaultdict` vs `dict` | counting dict, auto-default dict, plain | No | All dict-like; they differ in how missing keys behave. |
| `yield` vs `return` | pause vs finish | No | `return` ends the function; `yield` suspends it for later. |

## Code, step by step

### 1. File header (docstring)  (`main.py` lines 1–11)

```python
"""
GOAL:  Understand imports, `if __name__ == "__main__"`, decorators and generators (all appear in FastAPI/PyTorch code).

RUN:   cd 00_python_foundations/07_modules_decorators_generators && python main.py

EXPECTED OUTPUT (first lines):
    sqrt(16) = 4.0
    ...

NEEDS: nothing (standard library only), no internet.
"""
```

- **In simple words:** The file's label.
- **Exactly:** Module docstring; no runtime effect.
- **Why it is here:** Orients the reader.

### 2. Imports and two handy collections  (`main.py` lines 13–24)

```python
# ---- imports ----------------------------------------------------------------
import math                            # import a whole module; use math.sqrt(...)
from collections import Counter, defaultdict  # import specific names
import time as t                       # alias with `as`
# `import X` finds X in: the current folder, installed packages (site-packages), the standard library.
# If two places have the same name, the first found wins -> naming your file random.py breaks `import random`!

print("sqrt(16) =", math.sqrt(16))
print(Counter("banana"))                # Counter: dict subclass for counting (also useful for LeetCode)
groups = defaultdict(list)              # dict that creates a missing value automatically
groups["fruit"].append("apple")
print(dict(groups))
```

- **In simple words:** `import` lends you code written by others: the whole module, selected names, or a nickname.
- **Exactly:** `import math` -> `math.sqrt(16)` = `4.0`. `from collections import Counter, defaultdict` pulls two names. `import time as t` aliases. `Counter("banana")` prints `Counter({'a': 3, 'n': 2, 'b': 1})`. `defaultdict(list)` creates `[]` for a missing key, so `groups["fruit"].append("apple")` works; `dict(groups)` -> `{'fruit': ['apple']}`.
- **Why it is here:** Imports are how every project reuses libraries; the search-order comment explains why a file named `random.py` shadows the real module.

### 3. Write a timing decorator  (`main.py` lines 26–36)

```python
# ---- decorators ----------------------------------------------------------------
# A decorator is a function that takes a function and returns an improved function.
# `@something` above a def is shorthand for: f = something(f)
def timed(func):
    def wrapper(*args, **kwargs):       # accepts anything, passes it through
        start = t.perf_counter()
        result = func(*args, **kwargs)  # call the original function
        elapsed = t.perf_counter() - start
        print(f"{func.__name__} took {elapsed * 1000:.2f} ms")
        return result
    return wrapper
```

- **In simple words:** A decorator is a wrapper: it takes a function and returns a souped-up version that also times it.
- **Exactly:** `timed(func)` defines inner `wrapper(*args, **kwargs)`, records `t.perf_counter()` before and after calling `func(*args, **kwargs)`, prints `{func.__name__} took X ms`, and returns the original result. `timed` returns `wrapper` (the function itself, not a call).
- **Why it is here:** Adds behaviour (timing, logging, caching) without editing the function body.

### 4. Apply the decorator with @timed  (`main.py` lines 39–44)

```python
@timed                                  # same as: slow_sum = timed(slow_sum)
def slow_sum(n):
    return sum(range(n))


print(slow_sum(1_000_000))
```

- **In simple words:** `@timed` above a def is shorthand for `slow_sum = timed(slow_sum)`.
- **Exactly:** `slow_sum(1_000_000)` now runs `wrapper`: prints e.g. `slow_sum took 65.26 ms` (time varies by machine), then `499999500000` (sum 0..999999).
- **Why it is here:** This is the exact mechanism behind `@app.get`, `@torch.no_grad()` and `@st.cache_resource`.

### 5. Comment: where decorators appear  (`main.py` lines 46–47)

```python
# You will meet decorators as: @app.get("/health") in FastAPI, @st.cache_resource in Streamlit,
# @torch.no_grad() in PyTorch, @dataclass (module 05 of this chapter).
```

- **In simple words:** A reminder of real libraries that use the same idea.
- **Exactly:** Comments only; no runtime effect.
- **Why it is here:** Connects the toy example to real frameworks.

### 6. Generator that yields batches  (`main.py` lines 49–57)

```python
# ---- generators: functions that `yield` one value at a time -------------------
def batches(items, size):
    """Yield successive chunks. Memory-friendly: only one chunk exists at a time."""
    for i in range(0, len(items), size):    # range(start, stop, step)
        yield items[i:i + size]             # pause here, hand back a chunk, resume next time


for batch in batches(list(range(7)), 3):    # this is how data loaders feed a model
    print("batch:", batch)
```

- **In simple words:** A generator is a function that can pause and hand back one chunk at a time instead of returning everything at once.
- **Exactly:** `range(0, len(items), size)` steps by `size`: 0, 3, 6 for 7 items. `yield items[i:i+size]` returns a slice and pauses. Output: `batch: [0, 1, 2]`, `batch: [3, 4, 5]`, `batch: [6]`.
- **Why it is here:** Only one chunk lives in memory; this is how data loaders feed models.

### 7. Pull values manually with next()  (`main.py` lines 59–61)

```python
gen = batches([1, 2, 3, 4], 2)
print(next(gen), next(gen))                 # next() pulls one value manually
# next(gen) again would raise StopIteration: the generator is exhausted.
```

- **In simple words:** `next(gen)` asks the generator for its next value.
- **Exactly:** `gen = batches([1,2,3,4], 2)` creates the generator but runs nothing yet. Two `next` calls give `[1, 2]` and `[3, 4]`; a third raises `StopIteration`.
- **Why it is here:** Shows laziness and what a `for` loop does behind the scenes.

### 8. The main guard  (`main.py` lines 63–72)

```python
# ---- the main guard --------------------------------------------------------------
# When you run `python main.py`, __name__ == "__main__".
# When another file does `import main`, __name__ == "main" so this block is skipped.
# That lets a file be both a reusable module AND a runnable script.
def main():
    print("running as a script, __name__ =", __name__)


if __name__ == "__main__":
    main()
```

- **In simple words:** Code under `if __name__ == "__main__":` runs only when you run the file directly, not when another file imports it.
- **Exactly:** Running `python main.py` sets `__name__` to `"__main__"`, so `main()` prints `running as a script, __name__ = __main__`. If imported, `__name__` would be `"main"` and the block is skipped.
- **Why it is here:** Lets a file be both a reusable module and a script; without it importing would trigger side effects.

### 9. TRY IT and BREAK IT comments  (`main.py` lines 74–75)

```python
# TRY IT: add @timed to `batches`... what does it print and why is the time ~0? (Hint: generators are lazy.)
# BREAK IT: create a file named math.py next to this script, then run again -> weird AttributeError. Delete it after!
```

- **In simple words:** Experiments for you (comments only).
- **Exactly:** A local `math.py` shadows the standard `math`, causing an `AttributeError` on `math.sqrt`.
- **Why it is here:** Practice the import-shadowing bug (module 01 step 03).

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `import math` | Load module; use `math.sqrt` | |
| `from collections import Counter` | Import one name directly | |
| `import time as t` | Alias | `t.perf_counter()` |
| `@decorator` above `def` | Same as `f = decorator(f)` | `@timed` |
| inner `def wrapper(*args, **kwargs)` | Wrapper that forwards any arguments | `timed` |
| `yield value` | Produce a value, pause, resume later (generator) | `batches` |
| `next(gen)` | Pull next value from a generator | |
| `if __name__ == "__main__":` | Run only when executed directly, not when imported | `main()` |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does | Returns |
|---|---|---|---|
| `math.sqrt` | standard lib | Square root | float |
| `collections.Counter` | standard lib | Count hashable items (dict subclass) | `Counter` |
| `collections.defaultdict(list)` | standard lib | Dict that auto-creates missing values | dict-like |
| `time.perf_counter()` | standard lib | High-resolution timer | float seconds |

## Execution trace (real output)
`@timed` replaces `slow_sum` with `wrapper`; calling `slow_sum(1_000_000)` runs the timer, calls the original, prints elapsed time, returns the result:
```
slow_sum took 13.45 ms
499999500000
```
Generator: `batches(list(range(7)), 3)` yields `[0,1,2]`, `[3,4,5]`, `[6]` one at a time:
```
batch: [0, 1, 2]
batch: [3, 4, 5]
batch: [6]
```
Running directly prints `running as a script, __name__ = __main__`.

## Experiments
1. Add `@timed` to `batches`. It reports ~0 ms because the body does not run until you iterate (lazy).
2. Change batch size to 1 and 10.
3. Import this file from another script (`import main` inside the folder): the `main()` call does **not** run, thanks to the guard. ([module 01 step 06](../../01_debugging_and_environment/06_http_and_network/broken_http.py) relies on exactly this.)

## Debugging practice
Create `math.py` next to the script and rerun: `AttributeError: module 'math' has no attribute 'sqrt'` — your file shadows the standard library because the script folder is first on `sys.path`. Delete it (and `__pycache__`). See [module 01 step 03](../../01_debugging_and_environment/03_import_and_environment/).

## Rebuild checklist
- [ ] Decorator that prints "calling <name>" before the function
- [ ] Generator yielding chunks of a list
- [ ] File with a `main()` guarded by `__name__`

## Say it aloud
"A decorator wraps a function to add behaviour, like FastAPI's `@app.get` registering a route. A generator uses `yield` to produce values lazily, which saves memory when feeding batches to a model. The `__main__` guard lets a file be imported without running its script code."
Trade-off: decorators can hide what really runs (stack traces go through `wrapper`); generators can be consumed only once.
