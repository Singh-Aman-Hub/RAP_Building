# Terms — Imports, Environments and Versions

Folder: `01_debugging_and_environment/03_import_and_environment` · Files covered: `main.py`, `broken_import.py`, `WALKTHROUGH.md`
Back to module: [01 Debugging and environment README](../README.md)

**Terms in this folder:** [Interpreter and sys.executable](#interpreter-and-sysexecutable) · [platform.python_version()](#platformpython_version) · [Virtual environment (venv)](#virtual-environment-venv) · [sys.prefix vs sys.base_prefix](#sysprefix-vs-sysbase_prefix) · [sys.path](#syspath) · [Module shadowing](#module-shadowing) · [importlib.metadata.version](#importlibmetadataversion) · [PackageNotFoundError](#packagenotfounderror) · [importlib.util.find_spec](#importlibutilfind_spec) · [F-string width specifier (:18s)](#f-string-width-specifier-18s) · [getattr with default](#getattr-with-default) · [Version tuple comparison](#version-tuple-comparison) · [Version conflict](#version-conflict) · [ModuleNotFoundError](#modulenotfounderror) · [ImportError (cannot import name)](#importerror-cannot-import-name) · [AttributeError on a module](#attributeerror-on-a-module) · [sys.argv](#sysargv) · [python -m pip](#python--m-pip) · [pip show / list / check](#pip-show--list--check) · [requirements.txt](#requirementstxt) · [Optional dependency pattern](#optional-dependency-pattern)

---

## Interpreter and sys.executable  `(concept)`
**1. Definition:** The interpreter is the Python program that runs your code; `sys.executable` is the absolute path of the interpreter currently running the script.
**2. In plain words:** A computer can have several Pythons (system, Homebrew, venv). A package installed for one is invisible to another. So the first question for any import problem is: which Python is actually running? `sys.executable` answers it.
**3. Example to quote:**
> "My script printed `.../.venv/bin/python`, so I knew I was in the project venv and not the system Python."
```python
import sys
print(sys.executable)
```
**Seen in:** `main.py`, first `print`

## platform.python_version()  `(standard library)`
**1. Definition:** `platform.python_version()` returns the running Python version as a string such as `'3.12.13'`.
**2. In plain words:** Libraries support specific Python versions, so you often need to know yours. It's a quick sanity check before blaming a package.
**3. Example to quote:**
> "I printed `platform.python_version()` and saw 3.12.13, which all my libraries support."
```python
import platform
print(platform.python_version())
```
**Seen in:** `main.py`

## Virtual environment (venv)  `(concept)`
**1. Definition:** A virtual environment is an isolated Python installation with its own set of installed packages, created with `python -m venv .venv` (or `uv venv`).
**2. In plain words:** Each project gets its own box of packages, so project A can use numpy 1.x while project B uses numpy 2.x. You activate it with `source .venv/bin/activate`. The downside is disk space and having to recreate it on each machine; Docker later packages this reproducibly.
**3. Example to quote:**
> "I create a venv per project so version conflicts in one project don't break another."
```bash
python3 -m venv .venv && source .venv/bin/activate
```
**Seen in:** `main.py`, "In a virtualenv?"; `WALKTHROUGH.md`, run it

## sys.prefix vs sys.base_prefix  `(standard library)`
**1. Definition:** `sys.prefix` is the root of the active environment; `sys.base_prefix` is the root of the base Python it was created from; they differ inside a virtual environment.
**2. In plain words:** Comparing them is the standard way to test "am I inside a venv?". `sys.prefix != sys.base_prefix` is `True` inside a venv.
**3. Example to quote:**
> "`sys.prefix != sys.base_prefix` printed `True`, confirming my venv was active."
```python
print(sys.prefix != sys.base_prefix)
```
**Seen in:** `main.py`, "In a virtualenv?" line

## sys.path  `(standard library)`
**1. Definition:** `sys.path` is the list of directories Python searches, in order, when you `import` a module.
**2. In plain words:** The first entry is the script's own folder, then installed packages and the standard library. If two places hold a module with the same name, the first wins, which is how shadowing happens.
**3. Example to quote:**
> "I printed `sys.path[:4]` and saw my script's folder first, which explains why a local `json.py` would override the real one."
```python
print(sys.path[:2])
```
**Seen in:** `main.py`, `sys.path` loop

## Module shadowing  `(concept)`
**1. Definition:** Shadowing happens when a local file or folder has the same name as a library, so `import name` loads the local one instead.
**2. In plain words:** Naming your file `json.py`, `random.py` or `numpy.py` breaks imports in confusing ways. Rename it and delete the leftover `__pycache__`.
**3. Example to quote:**
> "After I created `json.py` in the folder, `import json` loaded my file, so I deleted it and its `__pycache__`."
```python
import json
print(json.__file__)   # shows which file was loaded
```
**Seen in:** `main.py` BREAK IT and comment on `sys.path`; `WALKTHROUGH.md` experiment 2

## importlib.metadata.version  `(standard library)`
**1. Definition:** `importlib.metadata.version("pkg")` returns the installed version string of a distribution, raising `PackageNotFoundError` if it isn't installed.
**2. In plain words:** It answers "which version of numpy do I have?" from inside Python. It is imported as `md` in the script.
**3. Example to quote:**
> "`md.version('numpy')` returned 2.5.4, so I knew which numpy my interpreter had."
```python
import importlib.metadata as md
print(md.version("pydantic"))
```
**Seen in:** `main.py`, "Installed versions" loop

## PackageNotFoundError  `(standard library)`
**1. Definition:** `importlib.metadata.PackageNotFoundError` is raised when a requested distribution is not installed in the current environment.
**2. In plain words:** It is the safe way to detect a missing package. The script catches it and prints "NOT INSTALLED in this interpreter".
**3. Example to quote:**
> "`not-installed-pkg` triggered `PackageNotFoundError`, which my loop caught and reported."
```python
try:
    md.version("nope")
except md.PackageNotFoundError:
    print("missing")
```
**Seen in:** `main.py`

## importlib.util.find_spec  `(standard library)`
**1. Definition:** `importlib.util.find_spec(name)` returns a module spec if Python can locate the module, or `None` if not, without importing it.
**2. In plain words:** It lets you check "can I import this?" without actually importing or crashing. `spec.origin` shows the file it would load.
**3. Example to quote:**
> "`find_spec('numpyy')` returned `None`, so I knew that typo module doesn't exist."
```python
import importlib.util
print(importlib.util.find_spec("json") is not None)   # True
```
**Seen in:** `main.py`

## F-string width specifier (:18s)  `(syntax)`
**1. Definition:** In an f-string, `{value:18s}` pads a string to a minimum width of 18 characters.
**2. In plain words:** It aligns columns in printed output so package names line up. It only affects display.
**3. Example to quote:**
> "I wrote `{pkg:18s}` so the package names and versions form neat columns."
```python
print(f"{'numpy':18s}2.5.4")
```
**Seen in:** `main.py`, installed versions

## getattr with default  `(Python built-in)`
**1. Definition:** `getattr(obj, "name", default)` returns the attribute if it exists, otherwise `default`.
**2. In plain words:** `getattr(spec, "origin", None)` is safe even when `spec` is `None`, because it falls back to `None` instead of raising `AttributeError`.
**3. Example to quote:**
> "I used `getattr(spec, 'origin', None)` so a missing module prints `None` instead of crashing."
```python
print(getattr(None, "origin", None))   # None
```
**Seen in:** `main.py`, `find_spec` loop

## Version tuple comparison  `(concept)`
**1. Definition:** Converting a version string like `"2.5.4"` to a tuple of ints (such as `(2, 5)`) lets you compare versions numerically.
**2. In plain words:** Comparing the raw strings is wrong: `"10" < "9"` as text. Tuples of ints compare element by element, so `(2, 5) >= (1, 20)` is `True`.
**3. Example to quote:**
> "I turned the numpy version into a tuple and checked `have >= (1, 20)`, which was `True`."
```python
print((2, 5) >= (1, 20))   # True
```
**Seen in:** `main.py`, `version_tuple`

## Version conflict  `(concept)`
**1. Definition:** A version conflict occurs when two packages require incompatible versions of a shared dependency, or an installed version doesn't satisfy a requirement.
**2. In plain words:** pip may print "requires X>=a but you have b". Inspect with `pip list`, `pip show` and `pip check`, then fix by pinning compatible versions in `requirements.txt` in a fresh venv rather than patching in place.
**3. Example to quote:**
> "I ran `python -m pip check` to list broken dependency pairs, then pinned `numpy<2` in a fresh venv."
```bash
python -m pip check
```
**Seen in:** `WALKTHROUGH.md`, version conflicts

## ModuleNotFoundError  `(Python built-in)`
**1. Definition:** `ModuleNotFoundError` is raised when `import X` cannot find module `X`.
**2. In plain words:** Decision tree: typo? installed (`python -m pip show NAME`)? installed in a different interpreter? shadowed by a local file? `import numpyy` fails with `No module named 'numpyy'`.
**3. Example to quote:**
> "`python broken_import.py 1` raised `ModuleNotFoundError: No module named 'numpyy'`, a typo for numpy."
```python
# import numpyy
```
**Seen in:** `broken_import.py`, case 1

## ImportError (cannot import name)  `(Python built-in)`
**1. Definition:** `ImportError` is raised when a module exists but the requested name cannot be imported from it.
**2. In plain words:** `from json import load_string` fails because `json` has `loads`, not `load_string`. Check the spelling with `dir(json)` or the documentation.
**3. Example to quote:**
> "`cannot import name 'load_string' from 'json'` told me the module is fine, but the name is wrong."
```python
print([n for n in dir(json) if n.startswith("load")])
```
**Seen in:** `broken_import.py`, case 2

## AttributeError on a module  `(Python built-in)`
**1. Definition:** Accessing a name that doesn't exist on an imported module raises `AttributeError: module 'x' has no attribute 'y'`.
**2. In plain words:** `math.squareroot(9)` fails because the real function is `math.sqrt`. Use `dir(module)` or `help(module)` to find the right name.
**3. Example to quote:**
> "`module 'math' has no attribute 'squareroot'` pointed at a wrong function name; the right one is `math.sqrt`."
```python
import math
print(math.sqrt(9))
```
**Seen in:** `broken_import.py`, case 3

## sys.argv  `(standard library)`
**1. Definition:** `sys.argv` is the list of command-line arguments passed to the script, with `argv[0]` being the script name.
**2. In plain words:** `python broken_import.py 2` makes `sys.argv[1]` equal `"2"`, which is how the script chooses a case. Values are always strings.
**3. Example to quote:**
> "`case = sys.argv[1] if len(sys.argv) > 1 else '1'` picks the demo from the command line with a default."
```python
import sys
print(sys.argv)
```
**Seen in:** `broken_import.py`

## python -m pip  `(command)`
**1. Definition:** `python -m pip <command>` runs pip as a module of that specific Python interpreter, so it installs into the same environment that Python uses.
**2. In plain words:** A bare `pip` might belong to a different Python. Using `python -m pip install ...` guarantees installer and interpreter match.
**3. Example to quote:**
> "I always use `python -m pip install` so the package lands in the interpreter I'm running."
```bash
python -m pip install numpy
```
**Seen in:** `WALKTHROUGH.md`

## pip show / list / check  `(command)`
**1. Definition:** `pip show pkg` prints details of one installed package, `pip list` lists all installed packages and versions, and `pip check` reports broken dependencies.
**2. In plain words:** These are the three quick inspection commands when a library misbehaves: is it installed, which version, and are its requirements satisfied?
**3. Example to quote:**
> "I ran `python -m pip show numpy` to confirm the version and where it's installed."
```bash
python -m pip list
```
**Seen in:** `WALKTHROUGH.md`

## requirements.txt  `(concept)`
**1. Definition:** `requirements.txt` is a text file listing the packages (optionally with version pins) a project needs, installed with `pip install -r requirements.txt`.
**2. In plain words:** It makes setup repeatable on another machine or in Docker. Pinning versions like `numpy<2` prevents surprise upgrades.
**3. Example to quote:**
> "I keep a `requirements.txt` with pinned versions so a teammate or Docker can rebuild the same environment."
```text
numpy<2
```
**Seen in:** `WALKTHROUGH.md`

## Optional dependency pattern  `(concept)`
**1. Definition:** Wrapping `import x` in `try/except ImportError` lets code work with or without an optional library.
**2. In plain words:** If a feature needs an extra package, import it inside a `try`; if missing, fall back or print a helpful message instead of crashing.
**3. Example to quote:**
> "I wrapped the `anthropic` import in `try/except ImportError` so the mock backend still works when it's not installed."
```python
try:
    import numpy
except ImportError:
    numpy = None
```
**Seen in:** `WALKTHROUGH.md`, syntax table
