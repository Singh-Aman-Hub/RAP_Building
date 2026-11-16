# Walkthrough 02 — Syntax vs runtime errors

## Goal
Tell apart errors found **before** the program runs (SyntaxError) from errors found **while** it runs, and know the 10 most common types.

## Run it
```bash
cd 01_debugging_and_environment/02_syntax_and_runtime_errors
python main.py
python broken_syntax.py     # EXPECTED: SyntaxError, nothing printed first
python broken_runtime.py    # EXPECTED: crashes after printing two averages
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| SyntaxError vs runtime error | compile-time vs run-time error | No | Syntax errors stop the file before any line runs; runtime errors happen mid-run. |
| `TypeError` vs `ValueError` | wrong type vs wrong value | No | `"abc" + 1` is TypeError; `int("abc")` is ValueError. |
| `KeyError` vs `AttributeError` | dict lookup vs dot lookup | No | `d["x"]` vs `obj.x`. |
| `NameError` vs `AttributeError` | bare name vs attribute | No | NameError: the name does not exist at all; AttributeError: the object lacks the attribute. |
| `Exception` vs `BaseException` | catch-all vs top-level | No | `except Exception` skips `KeyboardInterrupt` and `SystemExit`. |

## Code, step by step

### 1. File header and comment  (`main.py` lines 1–12)

```python
"""
GOAL:  Meet the most common built-in exception types and learn what each one usually means.

RUN:   cd 01_debugging_and_environment/02_syntax_and_runtime_errors && python main.py
       Crashing versions (EXPECTED to fail):  python broken_syntax.py   |   python broken_runtime.py

EXPECTED OUTPUT: a table  "ErrorType: message  -> usual cause".

NEEDS: nothing (standard library only), no internet.
"""

# Each case is (label, function that fails, usual cause). We run each and print the real message.
```

- **In simple words:** The label plus a note that we run each failing function and print the real message.
- **Exactly:** Docstring and comment; nothing runs.
- **Why it is here:** Explains the table-style output.

### 2. Ten tiny functions that each fail on purpose  (`main.py` lines 13–24)

```python
def index_error():    return [1, 2, 3][5]
def key_error():      return {"a": 1}["b"]
def type_error():     return "age: " + 30
def value_error():    return int("abc")
def attr_error():     return None.upper()          # classic: a function returned None
def name_error():     return undefined_variable    # noqa: F821  (typo or never defined)
def zero_error():     return 1 / 0
def file_error():     return open("no_such_file.txt")
def import_error():   import not_a_real_module     # noqa: F401
def recursion_error():
    def f(): return f()
    return f()
```

- **In simple words:** Each one-liner makes exactly one kind of mistake.
- **Exactly:** `[1,2,3][5]` IndexError; `{"a":1}["b"]` KeyError; `"age: " + 30` TypeError; `int("abc")` ValueError; `None.upper()` AttributeError; `undefined_variable` NameError; `1 / 0` ZeroDivisionError; `open("no_such_file.txt")` FileNotFoundError; `import not_a_real_module` ModuleNotFoundError; `f()` that calls itself forever RecursionError.
- **Why it is here:** One side-by-side catalogue of the exceptions you meet daily.

### 3. A table of (label, function, usual cause)  (`main.py` lines 27–38)

```python
CASES = [
    ("IndexError", index_error, "index beyond list length (off-by-one?)"),
    ("KeyError", key_error, "dict key missing -> use .get() or check `in`"),
    ("TypeError", type_error, "wrong type for an operation / wrong number of args"),
    ("ValueError", value_error, "right type, impossible value"),
    ("AttributeError", attr_error, "object has no such attribute; often the object is None"),
    ("NameError", name_error, "typo, or used before assignment"),
    ("ZeroDivisionError", zero_error, "denominator is 0 (empty list average?)"),
    ("FileNotFoundError", file_error, "wrong path or wrong working directory"),
    ("ModuleNotFoundError", import_error, "package not installed in THIS interpreter"),
    ("RecursionError", recursion_error, "function calls itself without a base case"),
]
```

- **In simple words:** A list pairing each failing function with a short hint for the typical cause.
- **Exactly:** `CASES` is a `list` of 10 tuples `(str, function, str)`. Functions are stored without calling them (no parentheses).
- **Why it is here:** Data-driven loop below can stay short; adding a case is one line.

### 4. Run each case and print the real message  (`main.py` lines 40–45)

```python
for label, func, cause in CASES:
    try:
        func()
    except Exception as err:               # Exception = base class of almost all runtime errors
        print(f"{type(err).__name__}: {err}")
        print(f"    -> usual cause: {cause}\n")
```

- **In simple words:** Call each function inside try/except, catch whatever happens and print the error name, message and hint.
- **Exactly:** `except Exception as err` catches almost all runtime errors. `type(err).__name__` is the class name, `err` prints the message. Real lines: `IndexError: list index out of range`, `KeyError: 'b'`, `TypeError: can only concatenate str (not "int") to str`, `AttributeError: 'NoneType' object has no attribute 'upper'`, `RecursionError: maximum recursion depth exceeded`.
- **Why it is here:** Shows that every exception has the same shape (type + message) and the message usually names the culprit.

### 5. SyntaxError via compile()  (`main.py` lines 47–54)

```python
# ---- SyntaxError is different: it happens BEFORE the program runs (at compile time) ------
# compile() turns source text into code and raises SyntaxError if the text is not valid Python.
bad_source = "if True print('hi')"
try:
    compile(bad_source, "<example>", "exec")
except SyntaxError as err:
    print("SyntaxError:", err.msg, "| line", err.lineno)
    print("    -> a SyntaxError means NOTHING in the file ran. Look at the line shown AND the line before it.")
```

- **In simple words:** A SyntaxError is found before running: the source text is not valid Python at all.
- **Exactly:** `compile("if True print('hi')", "<example>", "exec")` tries to turn text into code and raises `SyntaxError`. Real output: `SyntaxError: invalid syntax | line 1`. `err.msg` is the message, `err.lineno` the line.
- **Why it is here:** Because the whole file is compiled first, a syntax error means nothing in the file ran.

### 6. TRY IT and BREAK IT comments  (`main.py` lines 56–57)

```python
# TRY IT: add your own case (e.g. [].pop()) to CASES and predict the message first.
# BREAK IT: run broken_syntax.py and notice that nothing is printed before the error.
```

- **In simple words:** Experiments for you (comments only).
- **Exactly:** `[].pop()` raises `IndexError: pop from empty list`.
- **Why it is here:** Practice predicting an error message.

### 7. broken_syntax.py: docstring  (`broken_syntax.py` lines 1–1)

```python
"""EXPECTED TO CRASH with SyntaxError before printing anything. Find ALL problems (there are 3)."""
```

- **In simple words:** States that this file fails with SyntaxError before printing anything and hides 3 problems.
- **Exactly:** One-line docstring.
- **Why it is here:** Sets the challenge.

### 8. broken_syntax: a print that never runs  (`broken_syntax.py` lines 2–2)

```python
print("this line never runs, because the whole file is compiled first")
```

- **In simple words:** A normal print line at the top.
- **Exactly:** Python reads the whole file and compiles it before running any line, so this `print` never executes.
- **Why it is here:** Proves the key fact: syntax errors stop everything.

### 9. broken_syntax: missing colon after def  (`broken_syntax.py` lines 4–5)

```python
def greet(name)
    print("Hello, " + name)
```

- **In simple words:** A function header without its colon.
- **Exactly:** `def greet(name)` needs `:` at the end. Real error: `SyntaxError: expected ':'`, reported at line 4 with a caret after `)`. This is the first error and the only one Python shows.
- **Why it is here:** Python stops at the first syntax error; fix it and re-run to see the next.

### 10. broken_syntax: unclosed bracket  (`broken_syntax.py` lines 7–8)

```python
values = [1, 2, 3
print(values)
```

- **In simple words:** A list literal missing `]`.
- **Exactly:** `values = [1, 2, 3` never closes. Python 3.12 says `'[' was never closed` at line 7 (older versions pointed at line 8, the next line). This is why the brief says to look at the line before the reported one.
- **Why it is here:** Unbalanced brackets often get blamed on the following line.

### 11. broken_syntax: = instead of ==  (`broken_syntax.py` lines 10–11)

```python
if values[0] = 1:
    print("one")
```

- **In simple words:** Assignment used where a comparison was meant.
- **Exactly:** `if values[0] = 1:` raises `SyntaxError: cannot assign to subscript here. Maybe you meant '==' instead of '='?`. `=` stores, `==` compares.
- **Why it is here:** A classic typo that Python 3.10+ helpfully explains.

### 12. broken_runtime.py: docstring  (`broken_runtime.py` lines 1–1)

```python
"""EXPECTED TO CRASH at runtime (after printing two lines). Use the traceback to find the root cause."""
```

- **In simple words:** Says the file crashes at runtime, after printing two lines.
- **Exactly:** One-line docstring.
- **Why it is here:** Contrasts with syntax errors: here some code runs first.

### 13. broken_runtime: average divides by length  (`broken_runtime.py` lines 4–5)

```python
def average(xs):
    return sum(xs) / len(xs)
```

- **In simple words:** Sum divided by count.
- **Exactly:** `sum(xs) / len(xs)`; for `[2, 4]` is `6/2 = 3.0`. For `[]` `sum` is 0 and `len` is 0, so `0/0` raises `ZeroDivisionError: division by zero`.
- **Why it is here:** Averaging an empty list is the textbook ZeroDivisionError.

### 14. broken_runtime: loop over batches, last one empty  (`broken_runtime.py` lines 8–11)

```python
print("average of [2, 4]:", average([2, 4]))
batches = [[1, 2], [3], []]          # note the last batch is empty
for i, b in enumerate(batches):
    print(f"batch {i} average:", average(b))
```

- **In simple words:** Print the average of two normal batches, then hit an empty one.
- **Exactly:** Output: `average of [2, 4]: 3.0`, `batch 0 average: 1.5`, `batch 1 average: 3.0`, then the traceback: `line 11, in <module>` (call site), `line 5, in average` (raiser), `ZeroDivisionError: division by zero`. The root cause is the empty batch `[]` created on line 9, not the division itself. Fix: guard with `if not xs: return 0.0` or similar.
- **Why it is here:** Shows that the raising line and the root cause differ: the data that reached the function was wrong.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `except Exception as err:` | Catch almost any runtime error | `type(err).__name__` -> `'KeyError'` |
| list of tuples `[(a, b, c), ...]` + `for x, y, z in ...` | Table-driven loop | `CASES` |
| `compile(src, name, "exec")` | Check Python source without running it | raises `SyntaxError` |

## Libraries & functions
All built-ins: `compile`, `open`, `int`, `type`. `err.msg` / `err.lineno` are attributes of `SyntaxError`.

## Execution trace (real output)
```
IndexError: list index out of range
KeyError: 'b'
TypeError: can only concatenate str (not "int") to str
ValueError: invalid literal for int() with base 10: 'abc'
AttributeError: 'NoneType' object has no attribute 'upper'
NameError: name 'undefined_variable' is not defined
ZeroDivisionError: division by zero
FileNotFoundError: [Errno 2] No such file or directory: 'no_such_file.txt'
ModuleNotFoundError: No module named 'not_a_real_module'
RecursionError: maximum recursion depth exceeded
SyntaxError: invalid syntax | line 1
```
`broken_syntax.py` output (nothing before it printed):
```
  File ".../broken_syntax.py", line 4
    def greet(name)
                   ^
SyntaxError: expected ':'
```
Problems to find: missing `:` on `def`, missing `]`, and `=` instead of `==` in `if`. Python reports one at a time: fix, re-run, repeat.
`broken_runtime.py` ends with `ZeroDivisionError: division by zero` at `average(b)` for the empty batch `[]` (batches 0 and 1 printed `1.5` and `3.0`). Root cause: empty input is not handled.

## Experiments
1. Fix the colon in `broken_syntax.py` only; read the next error.
2. In `broken_runtime.py`, guard with `if not xs: return 0.0`; decide whether `0.0` is honest or whether raising `ValueError` is better.
3. Add `[].pop()` to `CASES`; predict the type (`IndexError: pop from empty list`).

## Debugging practice
For SyntaxError look at the reported line **and the line before** (an unclosed bracket makes the next line "wrong"). Editors highlight these; `python -m py_compile file.py` checks a file without running it.

## Rebuild checklist
- [ ] Name the 10 errors in the table and one cause each
- [ ] Explain why SyntaxError prevents any output
- [ ] Fix `broken_runtime.py` and add an assert/test for the empty case

## Say it aloud
"Syntax errors are caught when Python compiles the file, so nothing runs. Runtime errors happen mid-execution; the exception name tells the category — KeyError for a missing dict key, TypeError for a wrong type, AttributeError often for `None`. I fix the cause of bad data, not just the symptom."
Trade-off: catching `Exception` broadly keeps a server alive but can hide bugs — log them with the traceback.
