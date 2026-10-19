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
