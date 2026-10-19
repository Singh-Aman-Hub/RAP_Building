# Walkthrough 08 — Logging vs print debugging

## Goal
Use `logging` for anything that runs longer than a quick experiment: levels, timestamps, tracebacks, files.

## Run it
```bash
cd 01_debugging_and_environment/08_logging_vs_print
python main.py
LOG_LEVEL=DEBUG python main.py       # show debug lines
LOG_LEVEL=WARNING python main.py     # hide info
cat _scratch/app.log
```

## Syntax explained
| Syntax | Meaning |
|---|---|
| `logging.getLogger("name")` | A named logger (name appears in the output) |
| `log.info("x=%s", x)` | Lazy formatting: string built only if the level is enabled |
| `%(asctime)s %(levelname)s %(lineno)d %(message)s` | Format fields |
| `os.environ.get("LOG_LEVEL", "INFO")` | Level chosen without editing code |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does |
|---|---|---|
| `logging.basicConfig(level, format, handlers)` | standard lib | One-time global setup |
| `logging.StreamHandler()` / `FileHandler(path)` | standard lib | Send logs to terminal / file |
| `log.debug/info/warning/error/exception/critical` | standard lib | Emit at a level; `exception` adds the traceback |

## Execution trace (real output)
```
2026-10-19 11:04:26,082 INFO     inference:38  service starting
2026-10-19 11:04:26,082 WARNING  inference:32  empty input, returning default
2026-10-19 11:04:26,082 ERROR    inference:46  prediction failed
Traceback (most recent call last):
  ...
ZeroDivisionError: division by zero
```
DEBUG lines are absent at the default INFO level; `print()` lines have no time, level or source.

## When to use what
| Situation | Tool |
|---|---|
| Quick look at a value in a notebook/script | `print(type(x), x)` (use `repr`) |
| Anything in a server/container/agent loop | `logging` |
| Step through line by line | `breakpoint()` (pdb): commands `n` next, `s` step in, `p x` print, `c` continue, `q` quit |
| "Is my assumption true?" | `assert cond, "message"` |

## Experiments
1. `LOG_LEVEL=DEBUG python main.py` and find the new line.
2. Change the format to include `%(name)s` only.
3. Insert `breakpoint()` before `return "positive"...` and inspect `text` with `p text`.

## Debugging practice
Take `01_reading_tracebacks/main.py`, replace the `print(text)` with `logging.exception("run failed")` and confirm you get the same traceback with a timestamp.

## Rebuild checklist
- [ ] Configure logging with level and format
- [ ] Choose the right level for 5 example messages
- [ ] Log an exception with traceback

## Say it aloud
"print is fine for a quick look, but logging gives levels, timestamps, source locations and file/stream routing, and can be turned up or down without code changes. In a server I log requests at INFO, anomalies at WARNING and failures with `logger.exception` so the traceback is preserved."
Trade-off: too much logging costs I/O and can leak sensitive data (prompts, keys); log IDs and sizes, not secrets.
