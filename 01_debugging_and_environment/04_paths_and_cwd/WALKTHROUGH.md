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
