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

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| logging vs print | logs vs stdout debugging | No | Logging adds levels, time, source, files and handlers; print is raw text. |
| log level | severity (DEBUG, INFO, WARNING, ERROR, CRITICAL) | Same | Messages below the configured level are dropped. |
| logger vs handler vs formatter | who speaks / where it goes / how it looks | No | Logger creates records, handler delivers them, formatter styles them. |
| `log.error` vs `log.exception` | error line vs error + traceback | Nearly | `exception` = `error` plus traceback, used in `except`. |
| `stdout` vs `stderr` | normal vs diagnostic stream | No | `StreamHandler()` writes to stderr by default; `print` writes to stdout. |

## Code, step by step

### 1. Header docstring  (`main.py` lines 1–10)

```python
"""
GOAL:  Replace scattered print() debugging with the logging module (levels, format, file output, exceptions).

RUN:   cd 01_debugging_and_environment/08_logging_vs_print && python main.py
       quieter/louder:  LOG_LEVEL=DEBUG python main.py     LOG_LEVEL=WARNING python main.py

EXPECTED OUTPUT: timestamped log lines at different levels, and a log file in _scratch/.

NEEDS: nothing (standard library only).
"""
```

- **In simple words:** Label for the file: replace `print` debugging with the `logging` module.
- **Exactly:** Docstring; nothing runs. Shows `LOG_LEVEL=DEBUG python main.py` for more detail.
- **Why it is here:** Names the lesson.

### 2. Imports and make the _scratch folder  (`main.py` lines 12–17)

```python
import logging
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
(HERE / "_scratch").mkdir(exist_ok=True)
```

- **In simple words:** Get the tools and make sure the folder for the log file exists.
- **Exactly:** `logging` (standard library), `os` (read `LOG_LEVEL`), `Path`. `HERE` is the folder of this script; `(HERE / "_scratch").mkdir(exist_ok=True)` creates it, no error if present.
- **Why it is here:** `FileHandler` fails if the folder does not exist.

### 3. Configure logging once with basicConfig  (`main.py` lines 19–25)

```python
# basicConfig sets up the root logger ONCE for the whole program.
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),           # messages below this level are dropped
    format="%(asctime)s %(levelname)-8s %(name)s:%(lineno)d  %(message)s",   # what each line looks like
    handlers=[logging.StreamHandler(),                    # to the terminal
              logging.FileHandler(HERE / "_scratch" / "app.log", mode="w")],  # and to a file
)
```

- **In simple words:** Decide the volume level, line format and the two destinations (screen and file) for every message in the program.
- **Exactly:** `level=os.environ.get("LOG_LEVEL", "INFO")` drops anything below INFO by default. `format` uses fields: `%(asctime)s` time, `%(levelname)-8s` level padded to 8, `%(name)s` logger name, `%(lineno)d` line number, `%(message)s` the text. Two handlers: `StreamHandler()` (stderr/terminal) and `FileHandler(..., mode="w")` (overwrite `_scratch/app.log` each run). Configures the ROOT logger once; later calls are ignored.
- **Why it is here:** Gives every line time, level and source for free, and sends it to a file too; this is what you read when a server misbehaves.

### 4. Create a named logger  (`main.py` lines 26–26)

```python
log = logging.getLogger("inference")   # a named logger: shows which part of the app spoke
```

- **In simple words:** Get a labelled speaker, so each line says which part of the app talked.
- **Exactly:** `logging.getLogger("inference")` returns the logger named `inference`; it passes records up to the root logger's handlers. Appears as `inference:38` in each line.
- **Why it is here:** In a big app, names let you filter by component.

### 5. predict() with debug and warning logs  (`main.py` lines 29–34)

```python
def predict(text):
    log.debug("predict() called with %r", text)           # %-style args: formatting only if the level is enabled
    if not text:
        log.warning("empty input, returning default")
        return "unknown"
    return "positive" if "good" in text else "negative"
```

- **In simple words:** A toy classifier that logs what it receives and complains about empty input.
- **Exactly:** `log.debug("... %r", text)` uses lazy %-style arguments: the string is only built if DEBUG is enabled. Default INFO hides it. `log.warning(...)` fires for empty text and returns `"unknown"`. Otherwise returns `"positive"` if `"good"` is in the text else `"negative"`.
- **Why it is here:** Debug lines can stay in the code permanently without noise; warnings flag odd-but-survivable cases.

### 6. Call it and compare with print  (`main.py` lines 37–40)

```python
# Levels from least to most severe: DEBUG < INFO < WARNING < ERROR < CRITICAL
log.info("service starting")
print("print() output has no time, level or source ->", predict("good movie"))
predict("")
```

- **In simple words:** Log a start message, call predict twice, and show a plain print for contrast.
- **Exactly:** Real output: `... INFO inference:38 service starting`, then `... WARNING inference:32 empty input, returning default` (from `predict("")`). The `print` line has no time, level or source: `print() output has no time, level or source -> positive`. (It appears last in a piped run because stdout is buffered while stderr is not.)
- **Why it is here:** Shows concretely what `print` lacks.

### 7. log.exception inside except  (`main.py` lines 42–46)

```python
# logger.exception() logs an ERROR *with the full traceback* -- use it inside `except`.
try:
    1 / 0
except ZeroDivisionError:
    log.exception("prediction failed")
```

- **In simple words:** Record an error together with the full traceback.
- **Exactly:** `1 / 0` raises `ZeroDivisionError`. `log.exception("prediction failed")` logs at ERROR level and appends the traceback: `... ERROR inference:46 prediction failed` followed by `Traceback ... line 44 ... ZeroDivisionError: division by zero`. Only valid inside an `except` block.
- **Why it is here:** The traceback is exactly what you need later, and you did not have to crash to get it.

### 8. Final info line with the log path  (`main.py` lines 48–48)

```python
log.info("done; see %s", HERE / "_scratch" / "app.log")
```

- **In simple words:** Tell the user where the log file is.
- **Exactly:** `%s` is filled lazily with the path of `_scratch/app.log`.
- **Why it is here:** A pointer to the evidence saves hunting.

### 9. Comments: when to use print vs logging  (`main.py` lines 50–53)

```python
# WHEN print() IS FINE: quick throwaway checks while exploring (print(type(x), x.shape)).
# WHEN LOGGING WINS: anything that runs in a server/container, needs levels, timestamps, files, or must stay in code.
# TRY IT: run with LOG_LEVEL=DEBUG and see the extra line from predict().
# BREAK IT: call logging.basicConfig(level="INFO") a SECOND time later in the file: it is silently ignored (it only configures once).
```

- **In simple words:** Advice plus two experiments (comments only).
- **Exactly:** `LOG_LEVEL=DEBUG` adds `predict() called with 'good movie'`. A second `basicConfig` call is silently ignored (it only configures once; `force=True` overrides).
- **Why it is here:** Gives the rule of thumb for interviews.

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
