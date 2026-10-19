# Terms — Syntax and Runtime Errors

Folder: `01_debugging_and_environment/02_syntax_and_runtime_errors` · Files covered: `main.py`, `broken_syntax.py`, `broken_runtime.py`, `WALKTHROUGH.md`
Back to module: [01 Debugging and environment README](../README.md)

**Terms in this folder:** [Syntax error vs runtime error](#syntax-error-vs-runtime-error) · [SyntaxError](#syntaxerror) · [compile()](#compile) · [py_compile](#py_compile) · [IndexError](#indexerror) · [KeyError](#keyerror) · [TypeError](#typeerror) · [ValueError](#valueerror) · [AttributeError](#attributeerror) · [NameError](#nameerror) · [ZeroDivisionError](#zerodivisionerror) · [FileNotFoundError](#filenotfounderror) · [ModuleNotFoundError](#modulenotfounderror) · [RecursionError](#recursionerror) · [except Exception as err](#except-exception-as-err) · [type(err).__name__](#typeerr__name__) · [Table-driven loop](#table-driven-loop) · [Edge case: empty input](#edge-case-empty-input) · [noqa comment](#noqa-comment)

---

## Syntax error vs runtime error  `(concept)`
**1. Definition:** A syntax error is detected when Python compiles the file, before any code runs; a runtime error (exception) occurs while the program is executing valid code.
**2. In plain words:** With a syntax error, nothing in the file runs, not even the first `print`. With a runtime error, earlier lines run and print before the crash. That difference is a quick diagnostic clue.
**3. Example to quote:**
> "`broken_syntax.py` printed nothing at all, which told me it was a syntax error; `broken_runtime.py` printed two averages before crashing, so it was a runtime error."
```python
# SyntaxError -> nothing runs;  ZeroDivisionError -> earlier lines ran
```
**Seen in:** `main.py` comments; `broken_syntax.py`, `broken_runtime.py`

## SyntaxError  `(Python built-in)`
**1. Definition:** `SyntaxError` is raised when the source text is not valid Python grammar, such as a missing colon, unclosed bracket, or `=` where `==` is needed.
**2. In plain words:** Python reports one problem at a time, with a caret pointing near it. Look at the reported line and the line before it, since an unclosed bracket makes the next line look wrong. Fix, rerun, repeat.
**3. Example to quote:**
> "`broken_syntax.py` had three problems: a missing `:` after `def greet(name)`, a missing `]`, and `=` instead of `==` in the `if`."
```python
# def greet(name)   <- missing colon -> SyntaxError: expected ':'
```
**Seen in:** `broken_syntax.py`; `main.py`, `bad_source`

## compile()  `(Python built-in)`
**1. Definition:** `compile(source, filename, mode)` turns source text into a code object without running it, raising `SyntaxError` if the text is invalid.
**2. In plain words:** It lets the demo test a broken snippet safely. `compile("if True print('hi')", "<example>", "exec")` fails with `invalid syntax`. The `err.msg` and `err.lineno` fields describe the problem.
**3. Example to quote:**
> "I used `compile()` on `if True print('hi')` to show a `SyntaxError` on line 1 without running anything."
```python
try:
    compile("if True print('hi')", "<example>", "exec")
except SyntaxError as err:
    print(err.msg, err.lineno)
```
**Seen in:** `main.py`, SyntaxError section

## py_compile  `(command)`
**1. Definition:** `python -m py_compile file.py` byte-compiles a file to check its syntax without executing it.
**2. In plain words:** A quick way to check "is this file even valid Python?" before running it. It reports the first syntax problem it finds.
**3. Example to quote:**
> "Before running a long script I run `python -m py_compile script.py` to catch typos."
```bash
python -m py_compile main.py
```
**Seen in:** `WALKTHROUGH.md`, debugging practice

## IndexError  `(Python built-in)`
**1. Definition:** `IndexError` is raised when a sequence index is out of range.
**2. In plain words:** `[1, 2, 3][5]` fails because the list only has positions 0 to 2. It's often an off-by-one mistake or an empty list. `[].pop()` raises `IndexError: pop from empty list`.
**3. Example to quote:**
> "`[1, 2, 3][5]` raised `IndexError: list index out of range`; I checked `len()` and my loop bound."
```python
# [1, 2, 3][5]
```
**Seen in:** `main.py`, `index_error`

## KeyError  `(Python built-in)`
**1. Definition:** `KeyError` is raised when a dict key is missing.
**2. In plain words:** `{"a": 1}["b"]` fails with `KeyError: 'b'`. Use `.get()` or `if key in d` when the key may be absent.
**3. Example to quote:**
> "The table shows `KeyError: 'b'`, and the usual fix is `.get()` or an `in` check."
```python
{"a": 1}.get("b")   # None
```
**Seen in:** `main.py`, `key_error`

## TypeError  `(Python built-in)`
**1. Definition:** `TypeError` is raised when an operation is applied to a value of the wrong type or called with the wrong number of arguments.
**2. In plain words:** `"age: " + 30` can't mix str and int. The message says exactly what clashed: `can only concatenate str (not "int") to str`. Convert with `str()` or use an f-string.
**3. Example to quote:**
> "`'age: ' + 30` raised a `TypeError`, and `f'age: {30}'` fixes it."
```python
print(f"age: {30}")
```
**Seen in:** `main.py`, `type_error`

## ValueError  `(Python built-in)`
**1. Definition:** `ValueError` is raised when an argument has the right type but an unacceptable value.
**2. In plain words:** `int("abc")` receives a string (right type) that can't be turned into a number. It is the standard exception for invalid input values.
**3. Example to quote:**
> "`int('abc')` raised `ValueError: invalid literal for int() with base 10: 'abc'`."
```python
# int("abc")
```
**Seen in:** `main.py`, `value_error`

## AttributeError  `(Python built-in)`
**1. Definition:** `AttributeError` is raised when an object has no attribute or method with the requested name.
**2. In plain words:** The classic cause is calling a method on `None` after a function silently returned nothing: `None.upper()` gives `'NoneType' object has no attribute 'upper'`. Trace back where the `None` came from.
**3. Example to quote:**
> "`'NoneType' object has no attribute 'upper'` told me a function returned `None` where I expected a string."
```python
# None.upper()
```
**Seen in:** `main.py`, `attr_error`

## NameError  `(Python built-in)`
**1. Definition:** `NameError` is raised when a name is used that has not been defined in scope.
**2. In plain words:** Typically a typo or a variable used before assignment. The message names it: `name 'undefined_variable' is not defined`.
**3. Example to quote:**
> "`NameError: name 'undefined_variable' is not defined` meant a typo in a variable name."
```python
# print(undefined_variable)
```
**Seen in:** `main.py`, `name_error`

## ZeroDivisionError  `(Python built-in)`
**1. Definition:** `ZeroDivisionError` is raised when dividing by zero.
**2. In plain words:** In `broken_runtime.py`, the empty batch `[]` makes `len(xs)` equal 0, so the average crashes. Root cause: empty input wasn't handled. You can guard with `if not xs`, but decide whether returning `0.0` is honest or whether raising `ValueError` is better.
**3. Example to quote:**
> "The average of `[]` raised `ZeroDivisionError` because `len([])` is 0; I added a check for empty input."
```python
def average(xs):
    if not xs:
        raise ValueError("empty list")
    return sum(xs) / len(xs)
```
**Seen in:** `broken_runtime.py`; `main.py`, `zero_error`

## FileNotFoundError  `(Python built-in)`
**1. Definition:** `FileNotFoundError` is raised when opening a path that does not exist.
**2. In plain words:** Usually the path is wrong or you're running from a different working directory. The message includes the path Python tried.
**3. Example to quote:**
> "`open('no_such_file.txt')` raised `FileNotFoundError: [Errno 2] No such file or directory`."
```python
# open("no_such_file.txt")
```
**Seen in:** `main.py`, `file_error`

## ModuleNotFoundError  `(Python built-in)`
**1. Definition:** `ModuleNotFoundError` (a subclass of `ImportError`) is raised when `import X` cannot find module `X`.
**2. In plain words:** Most often the package isn't installed in the interpreter you're running, or there's a typo. Check which Python is active before reinstalling.
**3. Example to quote:**
> "`No module named 'not_a_real_module'` meant it isn't installed in this interpreter."
```python
# import not_a_real_module
```
**Seen in:** `main.py`, `import_error`

## RecursionError  `(Python built-in)`
**1. Definition:** `RecursionError` is raised when the maximum recursion depth is exceeded.
**2. In plain words:** A function that calls itself with no stopping condition (base case) keeps nesting until Python gives up. Add a base case that ends the recursion.
**3. Example to quote:**
> "`def f(): return f()` hit `RecursionError: maximum recursion depth exceeded` because it has no base case."
```python
def fact(n):
    return 1 if n <= 1 else n * fact(n - 1)
```
**Seen in:** `main.py`, `recursion_error`

## except Exception as err  `(syntax)`
**1. Definition:** `except Exception as err:` catches almost every runtime exception, since `Exception` is the base class of most error types.
**2. In plain words:** The demo uses it only to loop over many failing functions and print each message. In real code, broad catching can hide bugs, so log the traceback if you use it.
**3. Example to quote:**
> "I caught everything with `except Exception as err` just to print each error type and message in one loop."
```python
try:
    int("x")
except Exception as err:
    print(err)
```
**Seen in:** `main.py`, loop over `CASES`

## type(err).__name__  `(Python built-in)`
**1. Definition:** `type(err).__name__` returns the exception's class name as a string.
**2. In plain words:** It lets you print `KeyError` or `ValueError` next to the message, giving the same first line a traceback has.
**3. Example to quote:**
> "`print(f'{type(err).__name__}: {err}')` prints lines like `KeyError: 'b'`."
```python
print(type(KeyError("b")).__name__)   # KeyError
```
**Seen in:** `main.py`, loop over `CASES`

## Table-driven loop  `(concept)`
**1. Definition:** A table-driven loop stores cases as data (a list of tuples) and processes each with one loop, instead of writing repeated code.
**2. In plain words:** `CASES` holds a label, a failing function and a usual cause. One `for label, func, cause in CASES:` runs all ten. Adding an eleventh case is one line.
**3. Example to quote:**
> "I added `[].pop()` as one more tuple in `CASES` and the loop picked it up automatically."
```python
for label, func, cause in CASES:
    ...
```
**Seen in:** `main.py`, `CASES`

## Edge case: empty input  `(concept)`
**1. Definition:** An edge case is an unusual but valid input, such as an empty list, that a function must still handle correctly.
**2. In plain words:** The list `batches` ends with `[]`, and `average([])` crashes. Always ask "what if the input is empty?" Then either return a sensible value or raise a clear error.
**3. Example to quote:**
> "My average function crashed on the empty batch, so I added a guard and a test for the empty case."
```python
assert average([2, 4]) == 3
```
**Seen in:** `broken_runtime.py`, `batches`

## noqa comment  `(syntax)`
**1. Definition:** A trailing `# noqa: CODE` comment tells linters to ignore a specific warning on that line.
**2. In plain words:** The demo deliberately uses an undefined name and an unused import. `# noqa: F821` stops tools from flagging these intentional mistakes. It doesn't change how Python runs.
**3. Example to quote:**
> "I marked the intentional bug with `# noqa: F821` so my linter doesn't complain about the undefined name."
```python
return undefined_variable  # noqa: F821
```
**Seen in:** `main.py`, `name_error`, `import_error`
