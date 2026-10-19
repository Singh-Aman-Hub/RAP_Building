# Walkthrough 06 — Exceptions, files, pathlib, JSON

## Goal
Fail gracefully, read/write files safely, and move data in and out of JSON (the language of APIs and LLM tool calls).

## Run it
```bash
cd 00_python_foundations/06_exceptions_files_json
python main.py          # creates ./_scratch/ with note.txt and config.json
```

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
