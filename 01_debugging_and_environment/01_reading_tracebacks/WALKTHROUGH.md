# Walkthrough 01 — Reading a traceback

## Goal
Read any Python traceback in 30 seconds: **bottom line first, then the frame above it, then climb.**

## Run it
```bash
cd 01_debugging_and_environment/01_reading_tracebacks
python main.py                 # prints a traceback on purpose and dissects it
python broken_traceback.py     # EXPECTED to crash: your drill
```

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `except KeyError as err:` | Catch and name the exception object | `err.args` -> `('max_tokens',)` |
| `err.__traceback__` | The traceback attached to the exception | passed to `extract_tb` |
| `d["k"]` vs `d.get("k", default)` | Raises `KeyError` vs returns default | |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does | Returns |
|---|---|---|---|
| `traceback.format_exc()` | standard lib | Current exception as the same text Python prints | str |
| `traceback.extract_tb(tb)` | standard lib | Structured frames (`filename`, `lineno`, `name`, `line`) | list of `FrameSummary` |

## Execution trace (real output)
```
Traceback (most recent call last):
  File ".../main.py", line 29, in <module>          <- outermost call
    run()
  File ".../main.py", line 25, in run               <- middle: the CALLER
    return get_limit(config)
  File ".../main.py", line 20, in get_limit         <- innermost: where it was RAISED
    return config["max_tokens"]
KeyError: 'max_tokens'                              <- READ THIS FIRST
```
Reading recipe: (1) last line = type + message; (2) the frame right above = the line that blew up; (3) walk upward to see who passed the bad value; (4) decide whether the bug is the line that raised, or the caller that supplied bad data. Here the data (`config` lacks the key) is the real issue.

For `broken_traceback.py` the real output ends:
```
  File ".../broken_traceback.py", line 21, in report
    lines.append(f"{name}: {scores[name]}")
KeyError: 'carol'
```
Root cause: `"carol"` was never in `scores` (bad input), not a bug in `report` itself.

## Experiments
1. Use `config.get("max_tokens", 100)`; no exception, so only the `try` body runs.
2. Add a fourth nested function call; the traceback gains one frame.
3. Catch `KeyError` in `get_limit` and `raise ValueError("config missing max_tokens") from err` — see two chained tracebacks ("The above exception was the direct cause...").

## Debugging practice
Do the drill in `broken_traceback.py`: answer (a) type, (b) raising line, (c) calling line, (d) root cause, in writing, before any fix. Then fix by validating the names and verify by re-running.

## Rebuild checklist
- [ ] Explain the order of frames (oldest on top)
- [ ] Locate the raising line and the calling line in a new traceback
- [ ] Explain `KeyError` vs `.get`

## Say it aloud
"I read a traceback from the bottom: the last line gives the exception type and message, the frame above shows the exact line that raised, and frames above that show the call chain. Then I ask whether the bug is in that line or in the data passed in."
Trade-off: the raising line is not always the bug — often it's a bad value created much earlier.
