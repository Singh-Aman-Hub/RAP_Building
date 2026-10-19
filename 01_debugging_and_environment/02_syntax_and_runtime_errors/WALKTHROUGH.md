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
