# Walkthrough 04 — File paths and the working directory

## Goal
Understand why `open("data.txt")` works in one terminal folder and fails in another, and make paths robust.

## Run it
```bash
cd 01_debugging_and_environment/04_paths_and_cwd
python main.py
python broken_paths.py                           # works here
cd ../..                                         # repo root
python 01_debugging_and_environment/04_paths_and_cwd/broken_paths.py   # EXPECTED crash
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| cwd | working directory, current directory, `pwd` | Same | Where relative paths start; set by the shell, not by the script's location. |
| relative vs absolute path | `data.txt` vs `/Users/.../data.txt` | No | Relative paths depend on cwd; absolute ones do not. |
| `__file__` vs `os.getcwd()` | script location vs run location | No | One is where the code lives, the other where you stand. |
| `Path.resolve()` vs `os.path.abspath` | make path absolute | Almost | `resolve` also follows symlinks. |
| `Path` vs `os.path` | pathlib vs older string functions | Same job | Prefer `Path`. |

## Code, step by step

### 1. Header docstring  (`main.py` lines 1–11)

```python
"""
GOAL:  Understand the working directory (cwd) and why relative paths break depending on WHERE you run.

RUN:   cd 01_debugging_and_environment/04_paths_and_cwd && python main.py
       then from repo root:  python 01_debugging_and_environment/04_paths_and_cwd/main.py
       Crashing version (EXPECTED when run from the wrong folder):  python broken_paths.py

EXPECTED OUTPUT: cwd, script folder, and the same file read two ways.

NEEDS: nothing (standard library only), no internet.
"""
```

- **In simple words:** Label for the file: relative paths depend on where you run the command.
- **Exactly:** Docstring; nothing runs.
- **Why it is here:** Names the bug class (wrong working directory).

### 2. Import os and Path  (`main.py` lines 13–14)

```python
import os
from pathlib import Path
```

- **In simple words:** Two tools: `os` for the current directory, `Path` for building paths.
- **Exactly:** `os.getcwd()` returns the current working directory (cwd) string; `pathlib.Path` represents a filesystem path.
- **Why it is here:** Together they let you compare "where I am" with "where the script is".

### 3. Print cwd, script file and script folder  (`main.py` lines 16–18)

```python
print("cwd (where you ran the command):", os.getcwd())
print("script file                      :", Path(__file__).resolve())
print("script folder                    :", Path(__file__).resolve().parent)
```

- **In simple words:** Show three different locations that are easy to confuse.
- **Exactly:** `os.getcwd()` = the folder you ran the command in (here `.../04_paths_and_cwd`, but `.../RAP_interview` when run from the repo root). `Path(__file__).resolve()` is the absolute path of `main.py`. `.parent` is its folder. cwd changes with how you run; the script folder never does.
- **Why it is here:** Most "file not found" bugs are cwd != script folder.

### 4. A relative path (fragile)  (`main.py` lines 20–22)

```python
# Fragile: relative to the CWD, so it only works if cwd == this folder.
relative = Path("data.txt")
print("\nrelative path exists from here?  :", relative.exists())
```

- **In simple words:** Look for `data.txt` relative to wherever you are standing.
- **Exactly:** `Path("data.txt").exists()` is `True` when cwd is this folder and `False` when run from the repo root (real run from root: `relative path exists from here? : False`).
- **Why it is here:** Shows the fragile version so you can recognise it in other people's code.

### 5. An anchored path (robust)  (`main.py` lines 24–27)

```python
# Robust: anchored to the script's own location -> works from anywhere.
anchored = Path(__file__).resolve().parent / "data.txt"
print("anchored path exists?            :", anchored.exists())
print("content:", anchored.read_text().strip())
```

- **In simple words:** Build the path from the script's own location so it works from anywhere.
- **Exactly:** `Path(__file__).resolve().parent / "data.txt"`: `/` joins path parts. `.exists()` is `True` even from the repo root. `read_text().strip()` returns `hello from data.txt`.
- **Why it is here:** This is the fix: never depend on cwd for files that ship with the code.

### 6. Debug helper: print the absolute path  (`main.py` lines 29–30)

```python
# Debug helper: print absolute path before opening. Often the path is not what you think.
print("\nabsolute form of relative path   :", relative.resolve())
```

- **In simple words:** Display what a relative path really points to.
- **Exactly:** `relative.resolve()` turns it into an absolute path using the cwd: `.../04_paths_and_cwd/data.txt` when run from this folder; from the repo root it would print `.../RAP_interview/data.txt` (a file that does not exist).
- **Why it is here:** Often the path is not what you think; printing it solves the case in seconds.

### 7. TRY IT and BREAK IT comments  (`main.py` lines 32–33)

```python
# TRY IT: run from repo root and compare the 'relative path exists' line.
# BREAK IT: change "data.txt" to "Data.txt" -> macOS may still find it (case-insensitive) but Linux/Docker will NOT.
```

- **In simple words:** Experiments for you (comments only).
- **Exactly:** `Data.txt` still works on case-insensitive macOS but fails on Linux and Docker.
- **Why it is here:** A bug that only appears in production/CI.

### 8. broken_paths.py: docstring  (`broken_paths.py` lines 1–6)

```python
"""
EXPECTED TO CRASH when run from any folder except this one.
  works : cd 01_debugging_and_environment/04_paths_and_cwd && python broken_paths.py
  fails : cd <repo root> && python 01_debugging_and_environment/04_paths_and_cwd/broken_paths.py
Fix it by anchoring the path to __file__ (see main.py), then verify from BOTH folders.
"""
```

- **In simple words:** Says: works from this folder, fails from anywhere else; you must anchor the path.
- **Exactly:** Docstring only.
- **Why it is here:** Sets the challenge.

### 9. broken_paths: open a relative path  (`broken_paths.py` lines 7–8)

```python
with open("data.txt") as f:
    print(f.read())
```

- **In simple words:** Open `data.txt` and print it.
- **Exactly:** `open("data.txt")` resolves relative to cwd. From this folder it prints `hello from data.txt`. From the repo root: `FileNotFoundError: [Errno 2] No such file or directory: 'data.txt'` at line 7. The file exists, but not in the folder the command was run from. Fix: `Path(__file__).resolve().parent / "data.txt"`.
- **Why it is here:** The whole lesson in two lines: same code, different cwd, different result.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `Path("data.txt")` | A **relative** path: resolved from the current working directory |
| `Path(__file__).resolve().parent / "data.txt"` | A path anchored to the script location |
| `os.getcwd()` | The directory your shell was in when you ran the command |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does | Returns |
|---|---|---|---|
| `os.getcwd()` | standard lib | Current working directory | str |
| `Path.exists()` / `.resolve()` / `.read_text()` | standard lib | Exists? / absolute path / read content | bool / Path / str |
| `__file__` | built-in variable | Path of the running script | str |

## Execution trace (real output)
Run in its own folder:
```
relative path exists from here?  : True
anchored path exists?            : True
content: hello from data.txt
```
Run from the repo root: relative check becomes `False`, anchored stays `True`, and `broken_paths.py` ends:
```
FileNotFoundError: [Errno 2] No such file or directory: 'data.txt'
```
Fix (verify from BOTH folders): `open(Path(__file__).resolve().parent / "data.txt")`.

## Experiments
1. Change `data.txt` to `Data.txt` — macOS may still succeed (case-insensitive) while Linux/Docker fails: a classic "works on my machine".
2. `print(Path("data.txt").resolve())` shows where Python is really looking.
3. Run `python -c "import os; print(os.getcwd())"` from several folders.

## Debugging practice
Whenever you get `FileNotFoundError`: print the **absolute** path, `os.getcwd()`, and `os.listdir(".")`. Compare with where the file really is. In Docker the cwd is `WORKDIR` and only copied files exist.

## Rebuild checklist
- [ ] Explain cwd vs script folder
- [ ] Convert a relative open() to an anchored path
- [ ] Verify the fix from two directories

## Say it aloud
"Relative paths depend on where the process starts, not where the file lives. I anchor paths with `Path(__file__).resolve().parent` and print the absolute path when debugging. In containers the working directory and file layout differ, so this matters even more."
Trade-off: anchoring to `__file__` is robust for scripts but not for packaged apps; there, use package resources or a configured data directory (env var).
