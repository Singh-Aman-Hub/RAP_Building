# Terms — Exceptions, Files and JSON

Folder: `00_python_foundations/06_exceptions_files_json` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [00 Python foundations README](../README.md)

**Terms in this folder:** [json (module)](#json-module) · [pathlib.Path](#pathlibpath) · [Exception](#exception) · [try / except](#try--except) · [except ... as err](#except--as-err) · [ZeroDivisionError](#zerodivisionerror) · [else clause (try)](#else-clause-try) · [finally clause](#finally-clause) · [raise](#raise) · [ValueError](#valueerror) · [Chained comparison](#chained-comparison) · [Bare except](#bare-except) · [__file__ and script-relative paths](#__file__-and-script-relative-paths) · [Path / operator](#path--operator) · [Path.mkdir(exist_ok=True)](#pathmkdirexist_oktrue) · [read_text / write_text](#read_text--write_text) · [encoding="utf-8"](#encodingutf-8) · [open() and file modes](#open-and-file-modes) · [with statement](#with-statement) · [Iterating a file line by line](#iterating-a-file-line-by-line) · [json.dumps](#jsondumps) · [json.loads](#jsonloads) · [JSON type mapping](#json-type-mapping) · [JSONDecodeError](#jsondecodeerror) · [FileNotFoundError](#filenotfounderror)

---

## json (module)  `(standard library)`
**1. Definition:** `json` is the standard-library module for converting between Python objects and JSON text (JavaScript Object Notation).
**2. In plain words:** JSON is the text format APIs and LLM tool calls use to exchange data. This module turns a Python dict into text you can store or send, and turns text back into a dict.
**3. Example to quote:**
> "I use the `json` module to save my config as text and load it later, because JSON is what APIs speak."
```python
import json
```
**Seen in:** `main.py`, import and JSON section

## pathlib.Path  `(standard library)`
**1. Definition:** `pathlib.Path` is a class representing a filesystem path with methods like `exists()`, `mkdir()`, `read_text()` and `write_text()`, and the `/` operator for joining.
**2. In plain words:** Instead of gluing path strings together, a `Path` object knows how to build, inspect and read paths, and it works on Mac, Linux and Windows. It's the modern replacement for string paths.
**3. Example to quote:**
> "I use `Path` so `HERE / '_scratch'` builds a correct path on any operating system."
```python
from pathlib import Path
print(Path("a") / "b.txt")
```
**Seen in:** `main.py`, import and paths section

## Exception  `(concept)`
**1. Definition:** An exception is an object raised when an error occurs, which interrupts normal flow until some `except` handler catches it or the program crashes.
**2. In plain words:** Rather than silently continuing with bad data, Python stops and raises an error with a type and message. You decide what to do: recover, log, or let it crash. Catching narrow exceptions keeps real bugs visible.
**3. Example to quote:**
> "When I divided by zero Python raised `ZeroDivisionError`, and my function caught it and returned `None` instead of crashing."
```python
try:
    1 / 0
except ZeroDivisionError:
    print("caught")
```
**Seen in:** `main.py`, `safe_divide`

## try / except  `(syntax)`
**1. Definition:** `try:` wraps code that might raise an exception, and each `except ExceptionType:` block handles that type.
**2. In plain words:** It reads "try this; if that specific error happens, do this instead". Multiple `except` clauses handle different error types differently, as `safe_divide` does for `ZeroDivisionError` and `TypeError`.
**3. Example to quote:**
> "`safe_divide(1, 'x')` hit the `except TypeError` branch and printed 'need numbers'."
```python
try:
    1 / "x"
except TypeError:
    print("need numbers")
```
**Seen in:** `main.py`, `safe_divide`

## except ... as err  `(syntax)`
**1. Definition:** `except SomeError as err:` binds the caught exception object to the name `err`, giving access to its message and attributes.
**2. In plain words:** You can print `err` to see Python's explanation, such as `division by zero`, or read fields like `err.filename`. It's how you log what actually went wrong.
**3. Example to quote:**
> "I wrote `except ZeroDivisionError as err: print(err)` and saw 'division by zero'."
```python
except ValueError as err:
    print(err)
```
**Seen in:** `main.py`, `safe_divide`, `set_temperature` handlers

## ZeroDivisionError  `(Python built-in)`
**1. Definition:** `ZeroDivisionError` is the exception raised when dividing or taking a modulo by zero.
**2. In plain words:** `1 / 0` is undefined, so Python raises this error. Typical real cause: dividing an average by `len(items)` when the list is empty. Check the denominator or catch the error.
**3. Example to quote:**
> "`safe_divide(1, 0)` printed 'caught ZeroDivisionError: division by zero' and returned `None`."
```python
# 1 / 0  -> ZeroDivisionError
```
**Seen in:** `main.py`, `safe_divide(1, 0)`

## else clause (try)  `(syntax)`
**1. Definition:** The `else:` block after `except` clauses runs only if the `try` block finished without raising.
**2. In plain words:** It separates "the risky part" from "what to do on success". In `safe_divide`, the result is returned from `else`, so it only happens when the division worked.
**3. Example to quote:**
> "`safe_divide(6, 3)` skipped every `except`, ran the `else` branch and returned 2.0."
```python
try:
    x = 6 / 3
except ZeroDivisionError:
    x = None
else:
    print("no error")
```
**Seen in:** `main.py`, `safe_divide`

## finally clause  `(syntax)`
**1. Definition:** The `finally:` block always runs after `try`/`except`/`else`, whether or not an exception occurred, even if the function returns early.
**2. In plain words:** It's for cleanup that must happen no matter what: closing files, releasing locks. In the demo it prints "(division attempted)" after every call, including the failing ones.
**3. Example to quote:**
> "My 'division attempted' line printed after all three calls, even when the function returned from an `except`, because it is in `finally`."
```python
try:
    pass
finally:
    print("always")
```
**Seen in:** `main.py`, `safe_divide`

## raise  `(syntax)`
**1. Definition:** `raise ExceptionType("message")` deliberately triggers an exception.
**2. In plain words:** Use it to reject invalid input as early as possible, with a clear message. This is how libraries tell you "temperature must be in [0, 2]".
**3. Example to quote:**
> "`set_temperature(5)` raises `ValueError: temperature must be in [0, 2], got 5` so bad config fails loudly."
```python
raise ValueError("bad value")
```
**Seen in:** `main.py`, `set_temperature`

## ValueError  `(Python built-in)`
**1. Definition:** `ValueError` is raised when a function receives an argument of the right type but an inappropriate value.
**2. In plain words:** The type is fine (a number) but the value isn't acceptable (5 is out of range). `int("abc")` also raises it. It is the standard error to use for input validation.
**3. Example to quote:**
> "I raise `ValueError` when temperature is outside 0 to 2, and my caller catches it with `except ValueError as err`."
```python
int("abc")   # ValueError
```
**Seen in:** `main.py`, `set_temperature`

## Chained comparison  `(syntax)`
**1. Definition:** `a <= x <= b` is a single expression equivalent to `a <= x and x <= b`.
**2. In plain words:** Python lets you write ranges the way you do in maths. `0 <= t <= 2` reads naturally as "t is between 0 and 2". With `not` in front it flags values outside the range.
**3. Example to quote:**
> "`if not 0 <= t <= 2:` checks that temperature is out of bounds in one readable line."
```python
print(0 <= 1 <= 2)   # True
```
**Seen in:** `main.py`, `set_temperature`

## Bare except  `(concept)`
**1. Definition:** A bare `except:` catches every exception, including `KeyboardInterrupt` and `SystemExit`, regardless of type.
**2. In plain words:** It hides real bugs and makes programs hard to stop. The rule of thumb in the file: catch the narrowest exception you can actually handle.
**3. Example to quote:**
> "I never write a bare `except:`; I catch `ValueError` or `FileNotFoundError` so unexpected bugs still show up."
```python
# Avoid:  except:
# Prefer: except ValueError:
```
**Seen in:** `main.py`, comment under `set_temperature`

## __file__ and script-relative paths  `(concept)`
**1. Definition:** `__file__` is a built-in variable holding the path of the running script; `Path(__file__).resolve().parent` gives the absolute path of its folder.
**2. In plain words:** Relative paths like `"data.txt"` depend on where you ran the command from. Building paths from `__file__` makes the script find its own files from anywhere.
**3. Example to quote:**
> "I set `HERE = Path(__file__).resolve().parent` so the script writes `_scratch/` next to itself no matter my current directory."
```python
HERE = Path(__file__).resolve().parent
```
**Seen in:** `main.py`, paths section

## Path / operator  `(syntax)`
**1. Definition:** With `Path` objects, the `/` operator joins path components into a new `Path`.
**2. In plain words:** `HERE / "_scratch"` is clearer and safer than string concatenation. It uses the correct separator for the operating system.
**3. Example to quote:**
> "`note = scratch / 'note.txt'` builds the file path without worrying about slashes."
```python
p = Path("a") / "b" / "c.txt"
```
**Seen in:** `main.py`, `scratch = HERE / "_scratch"`

## Path.mkdir(exist_ok=True)  `(standard library)`
**1. Definition:** `Path.mkdir()` creates a directory; with `exist_ok=True` it does nothing instead of raising `FileExistsError` if the directory already exists.
**2. In plain words:** It makes the script safe to run repeatedly. Without `exist_ok=True`, the second run would crash because the folder already exists.
**3. Example to quote:**
> "`scratch.mkdir(exist_ok=True)` creates the folder on the first run and quietly does nothing afterwards."
```python
Path("out").mkdir(exist_ok=True)
```
**Seen in:** `main.py`, `scratch.mkdir(exist_ok=True)`

## read_text / write_text  `(standard library)`
**1. Definition:** `Path.read_text(encoding=...)` returns a file's contents as a string; `Path.write_text(text, encoding=...)` writes (overwriting) a string to the file.
**2. In plain words:** They are one-line shortcuts for small files, with no need to open and close manually. `.splitlines()` then splits the text into a list of lines.
**3. Example to quote:**
> "I wrote 'line one\nline two' with `write_text` and read it back with `read_text().splitlines()` to get `['line one', 'line two']`."
```python
p.write_text("hi", encoding="utf-8")
print(p.read_text(encoding="utf-8"))
```
**Seen in:** `main.py`, text files section

## encoding="utf-8"  `(concept)`
**1. Definition:** The `encoding` argument tells Python how to convert text to bytes on disk; `utf-8` is the universal standard that handles all languages.
**2. In plain words:** If you leave it out, Python uses the operating system's default, which differs between machines and can garble characters like "é". Always set `encoding="utf-8"` when reading or writing text.
**3. Example to quote:**
> "I always pass `encoding='utf-8'` so a file with 'café' reads the same on Mac and Windows."
```python
open("a.txt", "w", encoding="utf-8")
```
**Seen in:** `main.py`, file I/O calls; BREAK IT comment

## open() and file modes  `(Python built-in)`
**1. Definition:** `open(path, mode, encoding=...)` returns a file object; mode `"r"` reads, `"w"` overwrites, and `"a"` appends.
**2. In plain words:** The mode decides what happens to existing content. `"w"` erases the file first, which is why changing `"a"` to `"w"` in the demo makes the first two lines disappear.
**3. Example to quote:**
> "I opened `note.txt` with mode `'a'` to add 'line three' without deleting the existing lines."
```python
with open("note.txt", "a", encoding="utf-8") as f:
    f.write("more\n")
```
**Seen in:** `main.py`, text files section

## with statement  `(syntax)`
**1. Definition:** `with expression as name:` runs a block using a context manager that sets up a resource and guarantees cleanup (such as closing a file) when the block ends, even after an error.
**2. In plain words:** Forgetting to close a file can lose data or leak resources. `with open(...) as f:` closes it for you automatically, so it's the standard way to work with files.
**3. Example to quote:**
> "I used `with open(note, 'a') as f:` so the file is closed automatically even if `write` fails."
```python
with open("x.txt") as f:
    data = f.read()
```
**Seen in:** `main.py`, text files section

## Iterating a file line by line  `(concept)`
**1. Definition:** A file object is iterable: `for line in f:` yields one line at a time (including the trailing newline).
**2. In plain words:** It reads lazily, so even a huge file doesn't need to fit in memory. Use `line.rstrip()` to remove the trailing newline before printing.
**3. Example to quote:**
> "`for line in f: print('read:', line.rstrip())` printed 'line one', 'line two', 'line three' one by one."
```python
for line in open("note.txt"):
    print(line.rstrip())
```
**Seen in:** `main.py`, text files section

## json.dumps  `(standard library)`
**1. Definition:** `json.dumps(obj, indent=None)` serialises a Python object to a JSON-formatted string; `indent=2` pretty-prints it.
**2. In plain words:** It turns a dict into text you can write to a file or send over HTTP. "dumps" means "dump to string". Only JSON-compatible types (dict, list, str, numbers, bool, None) are allowed.
**3. Example to quote:**
> "`json.dumps(config, indent=2)` produced readable JSON text that I saved as `config.json`."
```python
print(json.dumps({"ok": True}))   # {"ok": true}
```
**Seen in:** `main.py`, JSON section

## json.loads  `(standard library)`
**1. Definition:** `json.loads(text)` parses a JSON string and returns the matching Python object, raising `json.JSONDecodeError` if the text is invalid.
**2. In plain words:** It's the reverse of `dumps`: text in, dict out. Because LLMs often produce slightly broken JSON, always wrap `loads` in a `try/except`. Note "loads" means "load from string"; `json.load` reads from a file object.
**3. Example to quote:**
> "`json.loads(cfg_path.read_text())` gave me back a dict, and `loaded['max_new_tokens']` was an int."
```python
data = json.loads('{"a": 1}')
```
**Seen in:** `main.py`, JSON section

## JSON type mapping  `(concept)`
**1. Definition:** JSON maps Python `True`/`False`/`None` to `true`/`false`/`null`, dicts to objects, lists to arrays, and requires double-quoted strings.
**2. In plain words:** The text looks like a Python dict but isn't identical: `{"ok": True}` becomes `{"ok": true}`. Single quotes are invalid JSON, which is a classic LLM output bug.
**3. Example to quote:**
> "`json.dumps({'ok': True, 'none': None})` printed `{\"ok\": true, \"none\": null}`."
```python
print(json.dumps({"none": None}))   # {"none": null}
```
**Seen in:** `main.py`, JSON section

## JSONDecodeError  `(standard library)`
**1. Definition:** `json.JSONDecodeError` (a subclass of `ValueError`) is raised by `json.loads` when the text is not valid JSON.
**2. In plain words:** Its message includes the line and column of the problem, e.g. "Expecting property name enclosed in double quotes". When a model returns bad JSON, catch this error and retry or report it.
**3. Example to quote:**
> "Parsing `{'single': 'quotes'}` raised `JSONDecodeError`, so I caught it and printed 'bad JSON'."
```python
try:
    json.loads("{'a': 1}")
except json.JSONDecodeError as err:
    print(err)
```
**Seen in:** `main.py`, "Handling bad input gracefully"

## FileNotFoundError  `(Python built-in)`
**1. Definition:** `FileNotFoundError` is the exception raised when you try to open or read a file or directory that does not exist.
**2. In plain words:** Usually the path is wrong or the working directory isn't what you assumed. The error object has a `.filename` attribute that tells you exactly which path Python tried. Check with `Path.exists()` or build the path from `__file__`.
**3. Example to quote:**
> "Reading `missing.txt` raised `FileNotFoundError`, and `err.filename` showed me the exact path that didn't exist."
```python
try:
    Path("nope.txt").read_text()
except FileNotFoundError as err:
    print(err.filename)
```
**Seen in:** `main.py`, "Missing file"; `WALKTHROUGH.md` debugging practice
