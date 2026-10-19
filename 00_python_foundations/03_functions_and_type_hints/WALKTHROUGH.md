# Walkthrough 03 — Functions and type hints

## Goal
Write reusable functions with defaults, keyword arguments, return values and type hints — the exact shape of `model.generate(...)`.

## Run it
```bash
cd 00_python_foundations/03_functions_and_type_hints
python main.py
```

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
