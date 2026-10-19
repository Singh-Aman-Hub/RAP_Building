# Terms — Logging vs Print Debugging

Folder: `01_debugging_and_environment/08_logging_vs_print` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [01 Debugging and environment README](../README.md)

**Terms in this folder:** [logging module](#logging-module) · [logging.basicConfig](#loggingbasicconfig) · [Log level](#log-level) · [Log format string](#log-format-string) · [Handler (StreamHandler and FileHandler)](#handler-streamhandler-and-filehandler) · [Named logger (getLogger)](#named-logger-getlogger) · [Lazy %-style formatting](#lazy--style-formatting) · [log.debug / info / warning](#logdebug--info--warning) · [log.exception](#logexception) · [LOG_LEVEL environment variable](#log_level-environment-variable) · [Path.mkdir for a log folder](#pathmkdir-for-a-log-folder) · [print() vs logging](#print-vs-logging) · [breakpoint() and pdb](#breakpoint-and-pdb) · [assert](#assert) · [One-time configuration](#one-time-configuration) · [Sensitive data in logs](#sensitive-data-in-logs)

---

## logging module  `(standard library)`
**1. Definition:** `logging` is the standard-library module for emitting structured diagnostic messages with levels, timestamps, source information and configurable destinations.
**2. In plain words:** Think of it as `print` with superpowers: every message has a severity, a time and a source, and you can send messages to the terminal and a file at once. You can turn the volume up or down without editing the code. Servers and containers rely on it.
**3. Example to quote:**
> "I use `logging` in anything that runs as a service, so I get timestamps, levels and files without extra code."
```python
import logging
```
**Seen in:** `main.py`, import

## logging.basicConfig  `(standard library)`
**1. Definition:** `logging.basicConfig(level=..., format=..., handlers=[...])` configures the root logger for the whole program.
**2. In plain words:** One call decides the minimum level, the line format and where logs go. It must happen early. In `main.py` it sends logs to both the terminal and `_scratch/app.log`.
**3. Example to quote:**
> "I call `basicConfig` once at startup with a level, a format and two handlers."
```python
logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")
```
**Seen in:** `main.py`

## Log level  `(concept)`
**1. Definition:** A log level ranks the severity of a message: DEBUG < INFO < WARNING < ERROR < CRITICAL; messages below the configured level are dropped.
**2. In plain words:** DEBUG is for developer detail, INFO for normal events, WARNING for odd-but-handled situations, ERROR for failures, CRITICAL for the worst. At the default INFO level, DEBUG lines are hidden.
**3. Example to quote:**
> "At the default INFO level my `predict()` debug line didn't show; with `LOG_LEVEL=DEBUG` it did."
```python
log.warning("empty input, returning default")
```
**Seen in:** `main.py`; `WALKTHROUGH.md`

## Log format string  `(syntax)`
**1. Definition:** The `format` string uses fields like `%(asctime)s`, `%(levelname)s`, `%(name)s`, `%(lineno)d` and `%(message)s` to define each log line.
**2. In plain words:** It controls which details appear: time, severity, which logger spoke, the line number, and the message. `%(levelname)-8s` pads the level to 8 characters so columns align.
**3. Example to quote:**
> "My format shows time, level, logger name and line number, so each line tells me when and where it happened."
```python
fmt = "%(asctime)s %(levelname)-8s %(name)s:%(lineno)d  %(message)s"
```
**Seen in:** `main.py`, `basicConfig`

## Handler (StreamHandler and FileHandler)  `(standard library)`
**1. Definition:** A handler decides where log records go: `StreamHandler()` writes to the terminal (stderr) and `FileHandler(path)` writes to a file.
**2. In plain words:** Using both means you see logs live and also keep them on disk. `mode="w"` overwrites the file each run.
**3. Example to quote:**
> "I attached a `StreamHandler` and a `FileHandler`, so logs appear on screen and in `_scratch/app.log`."
```python
handlers=[logging.StreamHandler(), logging.FileHandler("app.log", mode="w")]
```
**Seen in:** `main.py`

## Named logger (getLogger)  `(standard library)`
**1. Definition:** `logging.getLogger("name")` returns a logger identified by a name that appears in output via `%(name)s`.
**2. In plain words:** Naming loggers shows which part of the app spoke, e.g. `inference`. Different modules usually have their own loggers.
**3. Example to quote:**
> "`log = logging.getLogger('inference')` tags every message from my prediction code."
```python
log = logging.getLogger("inference")
```
**Seen in:** `main.py`

## Lazy %-style formatting  `(syntax)`
**1. Definition:** Calling `log.info("x=%s", x)` passes the value as an argument so the message string is built only if that level is enabled.
**2. In plain words:** It avoids wasted work formatting debug messages that nobody will see. Use `%s`/`%r` placeholders rather than f-strings in log calls.
**3. Example to quote:**
> "`log.debug('predict() called with %r', text)` does no formatting work when DEBUG is off."
```python
log.info("done; see %s", path)
```
**Seen in:** `main.py`

## log.debug / info / warning  `(standard library)`
**1. Definition:** Logger methods `debug`, `info`, `warning`, `error` and `critical` emit a message at the corresponding level.
**2. In plain words:** Pick the level that matches the situation: INFO for "service starting", WARNING for "empty input, returning default".
**3. Example to quote:**
> "I log 'service starting' at INFO and 'empty input' at WARNING, so I can filter by severity."
```python
log.info("service starting")
```
**Seen in:** `main.py`

## log.exception  `(standard library)`
**1. Definition:** `log.exception(msg)` logs an ERROR message and automatically appends the current exception's traceback; it should be called inside an `except` block.
**2. In plain words:** It preserves the full traceback in your logs with a timestamp, so you can investigate failures later. It replaces `print(traceback.format_exc())`.
**3. Example to quote:**
> "In my `except ZeroDivisionError:` I call `log.exception('prediction failed')` and get the full traceback in the log."
```python
try:
    1 / 0
except ZeroDivisionError:
    log.exception("prediction failed")
```
**Seen in:** `main.py`

## LOG_LEVEL environment variable  `(concept)`
**1. Definition:** Reading the log level from `os.environ.get("LOG_LEVEL", "INFO")` lets you change verbosity per run without editing code.
**2. In plain words:** `LOG_LEVEL=DEBUG python main.py` makes the output chattier; `LOG_LEVEL=WARNING` hides info messages. This is the standard way to tune logging in containers.
**3. Example to quote:**
> "I run with `LOG_LEVEL=DEBUG` while debugging and leave it at INFO in production."
```bash
LOG_LEVEL=DEBUG python main.py
```
**Seen in:** `main.py`; `WALKTHROUGH.md`

## Path.mkdir for a log folder  `(standard library)`
**1. Definition:** `(HERE / "_scratch").mkdir(exist_ok=True)` ensures the log directory exists before the file handler tries to open a file in it.
**2. In plain words:** If the folder doesn't exist, `FileHandler` would fail. `exist_ok=True` makes repeated runs safe.
**3. Example to quote:**
> "I create `_scratch` first so the file handler has somewhere to write."
```python
(HERE / "_scratch").mkdir(exist_ok=True)
```
**Seen in:** `main.py`

## print() vs logging  `(concept)`
**1. Definition:** `print` writes plain text to standard output; `logging` writes leveled, timestamped, routable records.
**2. In plain words:** `print` is fine for a quick look at a value (`print(type(x), x)`). For servers, containers and agent loops, use logging so you get levels, timestamps and files. Printing everywhere gives no way to filter or turn it off.
**3. Example to quote:**
> "The `print()` line in my output has no time, level or source; the log lines have all three."
```python
print(type(x), x)   # quick look
```
**Seen in:** `main.py`, comments at the end

## breakpoint() and pdb  `(Python built-in)`
**1. Definition:** `breakpoint()` pauses execution and opens Python's interactive debugger (pdb) at that line.
**2. In plain words:** You can inspect variables and step through code: `n` next line, `s` step into, `p x` print `x`, `c` continue, `q` quit. It's the right tool when you need to look inside a running function.
**3. Example to quote:**
> "I put `breakpoint()` before the return in `predict` and typed `p text` to see the input."
```python
breakpoint()
```
**Seen in:** `WALKTHROUGH.md`, when to use what

## assert  `(syntax)`
**1. Definition:** `assert condition, "message"` raises `AssertionError` with the message if the condition is false.
**2. In plain words:** It tests an assumption ("is my list really non-empty?") while debugging. It's for catching your own mistakes, not for validating user input.
**3. Example to quote:**
> "I add `assert len(items) > 0, 'items is empty'` to confirm my assumption before continuing."
```python
assert 1 + 1 == 2, "math is broken"
```
**Seen in:** `WALKTHROUGH.md`, when to use what

## One-time configuration  `(concept)`
**1. Definition:** `logging.basicConfig` configures logging only once; later calls are silently ignored if the root logger already has handlers.
**2. In plain words:** If you call it a second time with a different level, nothing changes, which is a confusing bug. Configure at the program's entry point only.
**3. Example to quote:**
> "My second `basicConfig(level='INFO')` call did nothing, because the root logger was already configured."
```python
# second basicConfig() call is ignored
```
**Seen in:** `main.py`, BREAK IT

## Sensitive data in logs  `(concept)`
**1. Definition:** Logs can leak secrets or private content (prompts, API keys, personal data) if you write raw values into them.
**2. In plain words:** Excess logging also costs I/O. Prefer logging IDs, sizes and status, not the secret content itself.
**3. Example to quote:**
> "I log the prompt length and a request ID, not the prompt text or the API key."
```python
log.info("prompt_chars=%d", len(prompt))
```
**Seen in:** `WALKTHROUGH.md`, trade-off
