# Walkthrough 07 — Imports, decorators, generators, `__main__`

## Goal
Decode `import`, `@app.get(...)`, `yield` and `if __name__ == "__main__":` — they appear in nearly every FastAPI / PyTorch file.

## Run it
```bash
cd 00_python_foundations/07_modules_decorators_generators
python main.py
```

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
