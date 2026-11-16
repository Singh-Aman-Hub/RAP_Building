# Walkthrough 06 — Exceptions, files, pathlib, JSON

## Goal
Fail gracefully, read/write files safely, and move data in and out of JSON (the language of APIs and LLM tool calls).

## Run it
```bash
cd 00_python_foundations/06_exceptions_files_json
python main.py          # creates ./_scratch/ with note.txt and config.json
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| exception | error, `Exception` object | Roughly | Anything raised; catch with `except`. Not every bug is a crash. |
| `try/except` vs `if` check | EAFP vs LBYL | No | EAFP: try then handle failure; LBYL: look before you leap with `if`. |
| `raise` vs `return` | signal error vs give value | No | `raise` unwinds the stack until caught. |
| `json.dumps/loads` vs `dump/load` | string vs file object | No | The `s` versions work with strings; the others with open files. |
| `Path` vs `os.path` | pathlib vs string functions | Same job | `Path` is object-oriented and preferred in modern code. |
| `open()` vs `Path.read_text()` | manual vs one-liner | Same result | `open` gives streaming/line control; `read_text` loads everything. |

## Code, step by step

### 1. File header (docstring)  (`main.py` lines 1–12)

```python
"""
GOAL:  Handle errors with try/except and read/write files (text, JSON) with pathlib.

RUN:   cd 00_python_foundations/06_exceptions_files_json && python main.py
       (it writes a small file into a `_scratch/` folder next to this script)

EXPECTED OUTPUT (first lines):
    caught ZeroDivisionError: division by zero
    ...

NEEDS: nothing (standard library only), no internet.
"""
```

- **In simple words:** The file's label, including a note that it writes a `_scratch/` folder.
- **Exactly:** Module docstring; no runtime effect.
- **Why it is here:** Warns the reader about side effects.

### 2. Import json and Path  (`main.py` lines 14–15)

```python
import json                    # standard-library JSON reader/writer
from pathlib import Path       # object-oriented file paths (preferred over raw strings)
```

- **In simple words:** Bring in the JSON reader/writer and a modern way to handle file paths.
- **Exactly:** `json` is the standard-library JSON codec; `Path` is `pathlib.Path`, an object that represents a file path.
- **Why it is here:** Paths as objects avoid string-gluing mistakes and work on every OS.

### 3. safe_divide with try / except / else / finally  (`main.py` lines 17–30)

```python
# ---- try / except / else / finally --------------------------------------
def safe_divide(a, b):
    try:                                    # code that might fail
        result = a / b
    except ZeroDivisionError as err:        # catch ONE specific error type; `as err` names it
        print("caught ZeroDivisionError:", err)
        return None
    except TypeError:
        print("caught TypeError: need numbers")
        return None
    else:                                   # runs only if try had NO error
        return result
    finally:                                # ALWAYS runs (cleanup goes here)
        print("  (division attempted)")
```

- **In simple words:** A safety net: try the risky division, handle known failures politely, always print a cleanup line.
- **Exactly:** `try` runs `a / b`. `except ZeroDivisionError as err` catches only that type and binds the exception to `err`. `except TypeError` catches e.g. `1 / "x"`. `else` runs only if no exception occurred. `finally` always runs, even after `return`.
- **Why it is here:** Real programs hit bad input; targeted handlers keep the program alive without hiding real bugs.

### 4. Call safe_divide three ways  (`main.py` lines 33–35)

```python
print(safe_divide(1, 0))
print(safe_divide(1, "x"))
print(safe_divide(6, 3))
```

- **In simple words:** Try a failing, a type-error and a fine call.
- **Exactly:** `safe_divide(1, 0)` prints `caught ZeroDivisionError: division by zero`, then `  (division attempted)`, then `None`. `safe_divide(1, "x")` prints `caught TypeError: need numbers`, the finally line, `None`. `safe_divide(6, 3)` prints the finally line then `2.0` (finally prints before the return value is printed by the caller).
- **Why it is here:** Shows the order: handler, finally, then the caller's print.

### 5. Raise your own ValueError  (`main.py` lines 37–49)

```python
# ---- raising your own errors --------------------------------------------
def set_temperature(t):
    if not 0 <= t <= 2:                     # chained comparison
        raise ValueError(f"temperature must be in [0, 2], got {t}")
    return t


try:
    set_temperature(5)
except ValueError as err:
    print("validation failed:", err)

# Rule of thumb: catch the narrowest exception you can handle. Never write bare `except:`.
```

- **In simple words:** Reject bad input yourself with a clear message.
- **Exactly:** `not 0 <= t <= 2` is a chained comparison. `raise ValueError(f"...")` stops the function. The `try` catches it and prints `validation failed: temperature must be in [0, 2], got 5`.
- **Why it is here:** Failing early with a precise message is cheaper to debug than a silent wrong value.

### 6. Build a path relative to this file  (`main.py` lines 51–57)

```python
# ---- paths ----------------------------------------------------------------
# __file__ is this script's path. .resolve() makes it absolute; .parent is its folder.
# Building paths from __file__ means the script works no matter which folder you run it from.
HERE = Path(__file__).resolve().parent
scratch = HERE / "_scratch"                 # `/` joins path pieces (works on Mac, Linux, Windows)
scratch.mkdir(exist_ok=True)                # create folder; no error if it already exists
print("scratch dir:", scratch.name, "| exists:", scratch.exists())
```

- **In simple words:** Find the folder the script lives in, and make a `_scratch` folder there.
- **Exactly:** `Path(__file__).resolve().parent` is the absolute folder of the script. `HERE / "_scratch"` joins paths with `/`. `mkdir(exist_ok=True)` does not fail if it exists. Output: `scratch dir: _scratch | exists: True`.
- **Why it is here:** Using `__file__` makes the script work from any current working directory.

### 7. Write and read a text file in one line each  (`main.py` lines 59–62)

```python
# ---- text files ---------------------------------------------------------
note = scratch / "note.txt"
note.write_text("line one\nline two\n", encoding="utf-8")   # one-liner write
print(note.read_text(encoding="utf-8").splitlines())        # -> ['line one', 'line two']
```

- **In simple words:** Quick save and load of a small text file.
- **Exactly:** `write_text(..., encoding="utf-8")` creates/overwrites the file with two lines. `read_text().splitlines()` -> `['line one', 'line two']`.
- **Why it is here:** The simplest way to handle small files; always pass `encoding`.

### 8. Append and read with open()  (`main.py` lines 64–69)

```python
# `with open(...)` guarantees the file is closed even if an error happens inside.
with open(note, "a", encoding="utf-8") as f:    # "a" = append, "w" = overwrite, "r" = read
    f.write("line three\n")
with open(note, encoding="utf-8") as f:
    for line in f:                              # files iterate line by line (memory friendly)
        print("read:", line.rstrip())
```

- **In simple words:** `with open(...)` opens a file and guarantees it closes afterwards.
- **Exactly:** Mode `"a"` appends `line three`. Reading iterates the file line by line (lazy). `line.rstrip()` drops the trailing newline. Prints `read: line one`, `read: line two`, `read: line three`.
- **Why it is here:** Prevents leaked file handles and handles large files without loading everything.

### 9. Write and load JSON  (`main.py` lines 71–77)

```python
# ---- JSON: the lingua franca of APIs and LLM tool calls ------------------------
config = {"model": "distilgpt2", "max_new_tokens": 20, "stop": ["\n"], "sample": True}
cfg_path = scratch / "config.json"
cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")   # dict -> JSON text
loaded = json.loads(cfg_path.read_text(encoding="utf-8"))             # JSON text -> dict
print(loaded["model"], type(loaded["max_new_tokens"]))
print(json.dumps({"ok": True, "none": None}))   # note: True -> true, None -> null in JSON
```

- **In simple words:** Turn a Python dict into JSON text and back.
- **Exactly:** `json.dumps(config, indent=2)` -> text (`True` becomes `true`, `None` becomes `null`). `json.loads` -> dict again; `loaded["max_new_tokens"]` is an `int` (`<class 'int'>`). Prints `distilgpt2 <class 'int'>` and `{"ok": true, "none": null}`.
- **Why it is here:** JSON is the format of API requests and LLM tool calls.

### 10. Catch invalid JSON  (`main.py` lines 79–83)

```python
# Handling bad input gracefully (LLMs often return broken JSON!)
try:
    json.loads("{'single': 'quotes'}")          # invalid JSON
except json.JSONDecodeError as err:
    print("bad JSON:", err)
```

- **In simple words:** Real model output is often not valid JSON, so wrap parsing in try/except.
- **Exactly:** `{'single': 'quotes'}` uses single quotes, which JSON forbids; `json.loads` raises `JSONDecodeError`. Output: `bad JSON: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)`.
- **Why it is here:** Pointing at the exact column tells you what to repair or re-prompt.

### 11. Catch a missing file  (`main.py` lines 85–89)

```python
# Missing file
try:
    (scratch / "missing.txt").read_text()
except FileNotFoundError as err:
    print("FileNotFoundError:", err.filename.split("/")[-1])
```

- **In simple words:** Show that reading a non-existent file raises a specific error you can catch.
- **Exactly:** `read_text()` raises `FileNotFoundError`; `err.filename` holds the full path, we print only the last part: `missing.txt`.
- **Why it is here:** Missing files are the most common path-related bug (see module 01 step 04).

### 12. TRY IT and BREAK IT comments  (`main.py` lines 91–92)

```python
# TRY IT: change "a" to "w" in the append block and see what the first lines become.
# BREAK IT: remove encoding="utf-8" and write "café"... on Windows this can differ; always set it.
```

- **In simple words:** Experiments for you (comments only).
- **Exactly:** Changing `"a"` to `"w"` would overwrite so only `line three` remains.
- **Why it is here:** Shows file modes.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `try: ... except X as e:` | Run code; handle a specific error | `except ZeroDivisionError as err:` |
| `else:` / `finally:` | Runs if no error / always runs | cleanup in `finally` |
| `raise ValueError("msg")` | Raise your own error | `set_temperature` |
| `except (A, B):` | Catch several types | step 05 of module 01 |
| `with open(p) as f:` | Context manager: closes the file automatically | read/append |
| `Path(__file__).resolve().parent / "x"` | Script-relative path | `HERE / "_scratch"` |
| `0 <= t <= 2` | Chained comparison | validation |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `open(path, mode, encoding=)` | built-in | Open a file | `"r"`, `"w"`, `"a"`; `encoding="utf-8"` | file object |
| `pathlib.Path` | standard lib | Path object; `.exists()`, `.mkdir(exist_ok=True)`, `.read_text()`, `.write_text()`, `.parent`, `/` | — | Path / str / None |
| `json.dumps(obj, indent=2)` | standard lib | Python -> JSON text | `indent` | str |
| `json.loads(text)` | standard lib | JSON text -> Python | — | dict/list/... (raises `JSONDecodeError`) |

## Execution trace (real output)
`safe_divide(1, 0)`: `try` raises -> `except ZeroDivisionError` prints -> `finally` prints -> returns `None`:
```
caught ZeroDivisionError: division by zero
  (division attempted)
None
```
Files: write two lines, append a third, read back with a loop: `read: line one / line two / line three`.
JSON: `{"ok": True, "none": None}` becomes `{"ok": true, "none": null}`.
Invalid JSON: `bad JSON: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)`.

## Experiments
1. Change mode `"a"` to `"w"`: the first two lines disappear.
2. Add `"max_new_tokens": "twenty"` to the config, load it and try `+ 1`: observe `TypeError` — JSON preserves types but not meaning.
3. Wrap `set_temperature(5)` in `except Exception` vs `except ValueError` and print `type(err)`.

## Debugging practice
`(scratch / "missing.txt").read_text()` -> `FileNotFoundError: [Errno 2] No such file or directory: '.../missing.txt'`. Check: is the path absolute? what is `os.getcwd()`? (deep dive in [module 01 step 04](../../01_debugging_and_environment/04_paths_and_cwd/)).

## Rebuild checklist
- [ ] Function raising `ValueError` on bad input; caller catching it
- [ ] Write a dict to JSON and read it back
- [ ] Open a file with `with`, loop over lines

## Say it aloud
"Exceptions let a program recover instead of crashing; I catch the narrowest type I can handle, never a bare `except`. `with open(...)` closes files automatically. I build paths from `Path(__file__)` so scripts work from any folder. JSON converts dicts and lists to text for APIs, and `json.loads` raises if a model returns malformed JSON, so I always wrap it."
Trade-off: too-broad `except` hides real bugs; too little handling crashes servers.
