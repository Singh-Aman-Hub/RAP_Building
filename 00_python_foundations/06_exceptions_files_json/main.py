"""
GOAL:  Handle errors with try/except and read/write files (text, JSON) with pathlib.

RUN:   cd 00_python_foundations/06_exceptions_files_json && python main.py
       (it writes a small file into a `_scratch/` folder next to this script)

EXPECTED OUTPUT (first lines):
    caught ZeroDivisionError: division by zero
    ...

NEEDS: nothing (standard library only), no internet.
"""

import json                    # standard-library JSON reader/writer
from pathlib import Path       # object-oriented file paths (preferred over raw strings)

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


print(safe_divide(1, 0))
print(safe_divide(1, "x"))
print(safe_divide(6, 3))

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

# ---- paths ----------------------------------------------------------------
# __file__ is this script's path. .resolve() makes it absolute; .parent is its folder.
# Building paths from __file__ means the script works no matter which folder you run it from.
HERE = Path(__file__).resolve().parent
scratch = HERE / "_scratch"                 # `/` joins path pieces (works on Mac, Linux, Windows)
scratch.mkdir(exist_ok=True)                # create folder; no error if it already exists
print("scratch dir:", scratch.name, "| exists:", scratch.exists())

# ---- text files ---------------------------------------------------------
note = scratch / "note.txt"
note.write_text("line one\nline two\n", encoding="utf-8")   # one-liner write
print(note.read_text(encoding="utf-8").splitlines())        # -> ['line one', 'line two']

# `with open(...)` guarantees the file is closed even if an error happens inside.
with open(note, "a", encoding="utf-8") as f:    # "a" = append, "w" = overwrite, "r" = read
    f.write("line three\n")
with open(note, encoding="utf-8") as f:
    for line in f:                              # files iterate line by line (memory friendly)
        print("read:", line.rstrip())

# ---- JSON: the lingua franca of APIs and LLM tool calls ------------------------
config = {"model": "distilgpt2", "max_new_tokens": 20, "stop": ["\n"], "sample": True}
cfg_path = scratch / "config.json"
cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")   # dict -> JSON text
loaded = json.loads(cfg_path.read_text(encoding="utf-8"))             # JSON text -> dict
print(loaded["model"], type(loaded["max_new_tokens"]))
print(json.dumps({"ok": True, "none": None}))   # note: True -> true, None -> null in JSON

# Handling bad input gracefully (LLMs often return broken JSON!)
try:
    json.loads("{'single': 'quotes'}")          # invalid JSON
except json.JSONDecodeError as err:
    print("bad JSON:", err)

# Missing file
try:
    (scratch / "missing.txt").read_text()
except FileNotFoundError as err:
    print("FileNotFoundError:", err.filename.split("/")[-1])

# TRY IT: change "a" to "w" in the append block and see what the first lines become.
# BREAK IT: remove encoding="utf-8" and write "café"... on Windows this can differ; always set it.
