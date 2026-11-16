# Walkthrough 03 — Import errors, virtual environments, versions

## Goal
Answer "which Python, which packages, which versions" before touching code.

## Run it
```bash
cd 01_debugging_and_environment/03_import_and_environment
python main.py                           # environment report
python broken_import.py 1                # EXPECTED crash (also try 2 and 3)
```
Create/activate a venv (from repo root): `python3 -m venv .venv && source .venv/bin/activate` (or `uv venv && source .venv/bin/activate`), then `pip install -r requirements.txt`.

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| virtual environment | venv, virtualenv, `.venv`, conda env | Roughly | A private folder of packages and a Python launcher; conda envs do the same via a different tool. |
| `ModuleNotFoundError` vs `ImportError` | module missing vs name missing | Subclass | `ModuleNotFoundError` is a subclass of `ImportError`. |
| `sys.path` vs `PYTHONPATH` | search list vs env var that adds to it | No | `PYTHONPATH` entries get inserted into `sys.path`. |
| package vs module vs distribution | folder vs file vs pip name | No | `pip install pillow` installs distribution `pillow`, imported as `PIL`. |
| `pip` vs `python -m pip` | installer vs installer tied to a given Python | Same program, safer | `python -m pip` guarantees the pip of that interpreter. |

## Code, step by step

### 1. Header docstring  (`main.py` lines 1–10)

```python
"""
GOAL:  Answer "WHICH Python am I running, and WHICH packages can it see?" -- the first question for any import error.

RUN:   cd 01_debugging_and_environment/03_import_and_environment && python main.py
       Crashing versions (EXPECTED):  python broken_import.py 1|2|3

EXPECTED OUTPUT: interpreter path, version, package versions, first sys.path entries.

NEEDS: nothing (standard library only), no internet.
"""
```

- **In simple words:** Label for the file: it answers "which Python am I running and what can it see?".
- **Exactly:** Docstring; nothing runs.
- **Why it is here:** Names the first question to ask for any import error.

### 2. Four standard-library imports  (`main.py` lines 12–15)

```python
import sys
import platform
import importlib.metadata as md          # read installed-package versions
import importlib.util                    # check if a module can be found without importing it
```

- **In simple words:** Load the tools for asking about the interpreter, the OS, installed packages and module lookup.
- **Exactly:** `sys` (interpreter state: path, prefix, executable), `platform` (version info), `importlib.metadata as md` (reads installed package versions), `importlib.util` (`find_spec` locates a module without importing it).
- **Why it is here:** All four are standard library, so this diagnostic script works even in a broken environment.

### 3. Which interpreter and is it a venv?  (`main.py` lines 17–20)

```python
print("Python executable :", sys.executable)        # THE interpreter running this script
print("Python version    :", platform.python_version())
print("In a virtualenv?  :", sys.prefix != sys.base_prefix)  # True inside venv/.venv
print("sys.prefix        :", sys.prefix)
```

- **In simple words:** Print the exact Python program running this script, its version and whether a virtual environment is active.
- **Exactly:** `sys.executable` here is `.../RAP_interview/.venv/bin/python`; `platform.python_version()` is `3.12.13`. `sys.prefix != sys.base_prefix` is `True` inside a venv (prefix points to `.venv`, base prefix to the system Python). Real output: `In a virtualenv? : True`.
- **Why it is here:** Most `ModuleNotFoundError`s are "you installed it in a different Python than the one running"; this line proves which one runs.

### 4. Show where import searches  (`main.py` lines 22–24)

```python
print("\nsys.path (where `import` searches, in order):")
for p in sys.path[:4]:
    print("   ", p)                         # first entry is this script's folder -> can shadow packages!
```

- **In simple words:** Print the first four folders Python will look in, in order.
- **Exactly:** `sys.path[:4]` is a list of strings. Entry 0 is the script's own folder (here `.../03_import_and_environment`), then the zip and `lib/python3.12` of the standard library, then `lib-dynload`. First match wins.
- **Why it is here:** Because the script folder comes first, a local file named `json.py` or `random.py` shadows the real module.

### 5. Check installed package versions  (`main.py` lines 26–31)

```python
print("\nInstalled versions:")
for pkg in ["numpy", "torch", "transformers", "fastapi", "pydantic", "not-installed-pkg"]:
    try:
        print(f"   {pkg:18s} {md.version(pkg)}")   # md.version raises if the package is missing
    except md.PackageNotFoundError:
        print(f"   {pkg:18s} NOT INSTALLED in this interpreter")
```

- **In simple words:** For each package name, print its version, or say it is not installed.
- **Exactly:** `md.version(pkg)` returns a string; on this machine numpy 2.5.4, torch 2.14.1, transformers 5.20.1, fastapi 0.146.0, pydantic 2.14.1. For `not-installed-pkg` it raises `PackageNotFoundError`, caught and printed as `NOT INSTALLED in this interpreter`. `{pkg:18s}` pads the name to 18 characters for neat columns.
- **Why it is here:** Version mismatches (e.g. an API that changed between releases) are a common root cause; this shows the facts.

### 6. Can a module be found without importing it?  (`main.py` lines 33–36)

```python
# find_spec returns None if Python cannot locate a module. Safer than try/except around import for checks.
for mod in ["json", "numpy", "numpyy"]:
    spec = importlib.util.find_spec(mod)
    print(f"can import {mod!r}? {spec is not None}", "->", getattr(spec, "origin", None))
```

- **In simple words:** Ask Python whether it can locate a module, and where.
- **Exactly:** `find_spec(name)` returns a `ModuleSpec` or `None`. Output: `json` -> True with a stdlib path; `numpy` -> True with a `.venv/.../site-packages/numpy/__init__.py` path; `numpyy` -> `False -> None`. `getattr(spec, "origin", None)` avoids an error when `spec` is `None`.
- **Why it is here:** Safer than try/except around a real import because nothing in the module gets executed.

### 7. Compare against a required version  (`main.py` lines 38–43)

```python
# Version-conflict check: compare against a requirement you expect.
def version_tuple(v): return tuple(int(x) for x in v.split(".")[:2] if x.isdigit())

need = (1, 20)
have = version_tuple(md.version("numpy"))
print("\nnumpy >= 1.20 ?", have >= need, f"(have {have})")
```

- **In simple words:** Turn "2.5.4" into `(2, 5)` and compare it with the minimum you need.
- **Exactly:** `version_tuple` splits on `.`, keeps the first two pieces and drops non-numeric ones; tuples compare element by element. `(2, 5) >= (1, 20)` is `True`. Comparing tuples, not strings, matters: `"1.9" > "1.20"` as strings is `True` (wrong).
- **Why it is here:** Version conflicts need a numeric comparison, not a text one.

### 8. TRY IT and BREAK IT comments  (`main.py` lines 45–46)

```python
# TRY IT: run `python3 main.py` vs `.venv/bin/python main.py` and compare the 'Python executable' lines.
# BREAK IT: add a file named json.py in this folder, run again, then delete it (module shadowing).
```

- **In simple words:** Experiments for you (comments only).
- **Exactly:** Running with system `python3` shows a different `sys.executable` and probably missing packages; a local `json.py` shadows the stdlib one.
- **Why it is here:** Makes the environment mismatch visible.

### 9. broken_import.py: docstring with three cases  (`broken_import.py` lines 1–7)

```python
"""
EXPECTED TO CRASH.  Choose a case:   python broken_import.py 1    (or 2, 3)
 1: package name typo / not installed         -> ModuleNotFoundError
 2: module exists but the NAME is wrong       -> ImportError (cannot import name ...)
 3: wrong attribute on a real module          -> AttributeError
For each: read the last line, then decide which check from main.py would have found the cause.
"""
```

- **In simple words:** Explains that you choose a case number on the command line and each fails differently.
- **Exactly:** Docstring: case 1 `ModuleNotFoundError`, case 2 `ImportError`, case 3 `AttributeError`.
- **Why it is here:** Compares three look-alike errors side by side.

### 10. broken_import: read the case number  (`broken_import.py` lines 8–10)

```python
import sys

case = sys.argv[1] if len(sys.argv) > 1 else "1"
```

- **In simple words:** Take the first command-line argument, default to "1".
- **Exactly:** `sys.argv` is the list of command-line words; `sys.argv[1]` is the first argument after the script name. `case` is a string.
- **Why it is here:** Lets one file demonstrate three failures.

### 11. broken_import case 1: package typo  (`broken_import.py` lines 12–13)

```python
if case == "1":
    import numpyy                      # typo of numpy
```

- **In simple words:** Import a package whose name has an extra `y`.
- **Exactly:** `import numpyy` fails: real last line `ModuleNotFoundError: No module named 'numpyy'`, raised at line 13. No such module on `sys.path`; check with `find_spec` and `md.version`.
- **Why it is here:** Typo or not installed in THIS interpreter.

### 12. broken_import case 2: right module, wrong name  (`broken_import.py` lines 14–15)

```python
elif case == "2":
    from json import load_string       # json has loads, not load_string
```

- **In simple words:** Import a name that the module does not contain.
- **Exactly:** `from json import load_string` raises `ImportError: cannot import name 'load_string' from 'json' (.../json/__init__.py)`. The module was found; the name inside it was not (real names: `load`, `loads`).
- **Why it is here:** ImportError (name missing) differs from ModuleNotFoundError (module missing).

### 13. broken_import case 3: wrong attribute  (`broken_import.py` lines 16–18)

```python
elif case == "3":
    import math
    print(math.squareroot(9))          # real function is math.sqrt
```

- **In simple words:** Import works, but the function you call does not exist.
- **Exactly:** `import math` succeeds; `math.squareroot(9)` raises `AttributeError: module 'math' has no attribute 'squareroot'` at line 18. Correct name: `math.sqrt`.
- **Why it is here:** AttributeError appears at use time, not import time.

### 14. broken_import: fallback message  (`broken_import.py` lines 19–20)

```python
else:
    print("choose 1, 2 or 3")
```

- **In simple words:** Any other argument prints a hint instead of crashing.
- **Exactly:** `python broken_import.py 9` prints `choose 1, 2 or 3`.
- **Why it is here:** Keeps the demo friendly.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `import pkg` / `from pkg import name` | Load a module / one name from it |
| `sys.argv` | List of command-line arguments (`argv[0]` is the script) |
| `try: import x / except ImportError:` | Optional dependency pattern |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does | Returns |
|---|---|---|---|
| `sys.executable` | standard lib | Path of the running interpreter | str |
| `sys.prefix` vs `sys.base_prefix` | standard lib | Differ when inside a venv | str |
| `sys.path` | standard lib | Folders searched by `import` (script folder first) | list |
| `importlib.metadata.version(name)` | standard lib | Installed package version | str (`PackageNotFoundError` if absent) |
| `importlib.util.find_spec(name)` | standard lib | Can this module be found? | spec or `None` |

## Execution trace (real output)
```
Python executable : .../RAP_interview/.venv/bin/python
Python version    : 3.12.13
In a virtualenv?  : True
   numpy              2.5.4
   torch              2.14.1
   not-installed-pkg  NOT INSTALLED in this interpreter
can import 'numpyy'? False -> None
```
Your numbers may differ slightly. The three broken cases end with:
```
ModuleNotFoundError: No module named 'numpyy'
ImportError: cannot import name 'load_string' from 'json' (...json/__init__.py)
AttributeError: module 'math' has no attribute 'squareroot'
```
Decision tree for `ModuleNotFoundError`: (1) typo? (2) is it installed — `python -m pip show NAME`; (3) installed for a **different** interpreter — compare `sys.executable` with `which python` / `which pip`; (4) is a local file named like the package shadowing it (first entry of `sys.path`)? Always use `python -m pip install ...` so pip and python match.

Version conflicts: `pip install` may print "requires X>=a but you have b"; use `pip list`, `pip show pkg`, `pip check`. Fix by pinning compatible versions in `requirements.txt` (e.g. `numpy<2`) in a fresh venv rather than patching in place.

## Experiments
1. Run with the system Python (`/usr/bin/python3 main.py`) and compare `numpy` lines.
2. Create `json.py` in the folder and run `main.py` — observe the failure, then delete it and `__pycache__`.
3. `python -m pip check` — reports broken dependency pairs, if any.

## Debugging practice
For each broken case: read the last line, name the category (missing package / wrong name / wrong attribute), and name the first check (`pip show`, `dir(json)`, `help(math)`).

## Rebuild checklist
- [ ] Create a venv, activate, install one package, verify with `pip show`
- [ ] Print the interpreter path and package version from inside Python
- [ ] Explain shadowing

## Say it aloud
"For import errors I first check which interpreter is running and whether the package is installed there. Using a virtual environment isolates each project's versions. I use `python -m pip` so the installer matches the interpreter, and I watch for local files shadowing library names."
Trade-off: venvs avoid conflicts but cost disk space and must be recreated on every machine — Docker (module 11) packages that reproducibly.
