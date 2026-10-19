# Terms — Reading Tracebacks

Folder: `01_debugging_and_environment/01_reading_tracebacks` · Files covered: `main.py`, `broken_traceback.py`, `WALKTHROUGH.md`
Back to module: [01 Debugging and environment README](../README.md)

**Terms in this folder:** [Traceback](#traceback) · [traceback module](#traceback-module) · [Stack frame](#stack-frame) · [Call stack order](#call-stack-order) · [KeyError](#keyerror) · [dict.get() default](#dictget-default) · [except ... as err](#except--as-err) · [err.args](#errargs) · [err.__traceback__](#err__traceback__) · [traceback.format_exc()](#tracebackformat_exc) · [traceback.extract_tb()](#tracebackextract_tb) · [Reading recipe (bottom to top)](#reading-recipe-bottom-to-top) · [Raise site vs call site](#raise-site-vs-call-site) · [Root cause](#root-cause) · [Exception chaining (raise ... from)](#exception-chaining-raise--from) · [NameError](#nameerror) · [max(..., key=...)](#max--key) · [Drill discipline](#drill-discipline)

---

## Traceback  `(concept)`
**1. Definition:** A traceback is the report Python prints when an exception is not handled, listing the chain of function calls (stack frames) that led to the error, followed by the exception type and message.
**2. In plain words:** It's Python telling you "here's the path I took and where it fell over". The last line says what went wrong; the lines above say where. Reading it calmly from the bottom saves a lot of time.
**3. Example to quote:**
> "My traceback ended with `KeyError: 'max_tokens'`, so I knew a dict key was missing before even looking at the line numbers."
```python
d = {}
d["max_tokens"]   # KeyError
```
**Seen in:** `main.py`, printed output; `WALKTHROUGH.md`

## traceback module  `(standard library)`
**1. Definition:** `traceback` is a standard-library module for capturing, formatting and inspecting tracebacks programmatically.
**2. In plain words:** Normally Python prints a traceback only when the program crashes. This module lets you grab the same text inside an `except` block, for logging or teaching, without crashing.
**3. Example to quote:**
> "I imported `traceback` so I could print the traceback of a caught `KeyError` and take it apart."
```python
import traceback
```
**Seen in:** `main.py`, import

## Stack frame  `(concept)`
**1. Definition:** A stack frame is one entry in the call stack, representing a single active function call with its file, line number, function name and source line.
**2. In plain words:** Each time a function calls another, Python adds a frame. A traceback lists them, so you can see `run` called `get_limit`, which raised the error. Each frame has a filename, line, function name and the code on that line.
**3. Example to quote:**
> "The traceback had three frames: module level, `run`, and `get_limit`; the last one is where the exception was raised."
```python
# File "main.py", line 20, in get_limit
```
**Seen in:** `main.py`, frame loop; `WALKTHROUGH.md` execution trace

## Call stack order  `(concept)`
**1. Definition:** In a Python traceback the oldest call is at the top and the most recent (innermost) call is at the bottom, just above the error line.
**2. In plain words:** "Most recent call last" is printed at the top of every traceback. That is why you start reading at the bottom: the innermost frame is where things actually broke.
**3. Example to quote:**
> "I read bottom to top: `get_limit` raised it, `run` called it, and the module level started everything."
```python
# Traceback (most recent call last):
```
**Seen in:** `main.py`, anatomy comments

## KeyError  `(Python built-in)`
**1. Definition:** `KeyError` is raised when a mapping (dict) is accessed with a key that does not exist.
**2. In plain words:** `config["max_tokens"]` fails because the config only has `"model"`. The error message shows the missing key. Either add the key, use `.get()`, or fix the data that was supposed to contain it.
**3. Example to quote:**
> "In `broken_traceback.py`, `scores['carol']` raised `KeyError: 'carol'` because carol was never in the scores dict."
```python
{"a": 1}["b"]   # KeyError: 'b'
```
**Seen in:** `main.py`, `get_limit`; `broken_traceback.py`, `report`

## dict.get() default  `(Python built-in)`
**1. Definition:** `d.get(key, default)` returns the value for `key` if present, otherwise `default`, without raising.
**2. In plain words:** Changing `config["max_tokens"]` to `config.get("max_tokens", 100)` removes the crash, because a sensible default fills in. But be careful: it can hide a genuinely missing setting.
**3. Example to quote:**
> "I replaced the indexing with `config.get('max_tokens', 100)` and the `except` branch no longer ran."
```python
print({}.get("max_tokens", 100))   # 100
```
**Seen in:** `main.py` TRY IT; `WALKTHROUGH.md` syntax table

## except ... as err  `(syntax)`
**1. Definition:** `except KeyError as err:` catches a `KeyError` and binds the exception object to `err`.
**2. In plain words:** Once you have `err`, you can inspect its arguments and traceback. Here it lets the script print `err.args` and walk `err.__traceback__`.
**3. Example to quote:**
> "I wrote `except KeyError as err:` so I could examine `err.args` and the attached traceback."
```python
try:
    {}["x"]
except KeyError as err:
    print(err.args)
```
**Seen in:** `main.py`, try/except block

## err.args  `(Python built-in)`
**1. Definition:** `args` is the tuple of arguments the exception was created with, usually containing the message or offending value.
**2. In plain words:** For a `KeyError`, `args` is `('max_tokens',)`, the missing key itself. It's a clean way to get the bad value programmatically.
**3. Example to quote:**
> "`err.args` printed `('max_tokens',)`, telling me exactly which key was missing."
```python
try:
    {}["k"]
except KeyError as e:
    print(e.args)   # ('k',)
```
**Seen in:** `main.py`, anatomy item 4

## err.__traceback__  `(Python built-in)`
**1. Definition:** `__traceback__` is the attribute of an exception holding the traceback object that records where it was raised.
**2. In plain words:** It's the raw data behind the printed traceback. You pass it to `traceback.extract_tb` to loop over frames in code.
**3. Example to quote:**
> "I passed `err.__traceback__` to `extract_tb` to list every frame with its file, line and function."
```python
traceback.extract_tb(err.__traceback__)
```
**Seen in:** `main.py`, frame loop

## traceback.format_exc()  `(standard library)`
**1. Definition:** `traceback.format_exc()` returns, as a string, the formatted traceback of the exception currently being handled.
**2. In plain words:** It gives you the same text Python would print on a crash, but as a value you can print, log or split. It only works inside an `except` block.
**3. Example to quote:**
> "`traceback.format_exc()` gave me the full traceback as text, and `.splitlines()[-1]` was `KeyError: 'max_tokens'`."
```python
try:
    1 / 0
except ZeroDivisionError:
    print(traceback.format_exc())
```
**Seen in:** `main.py`, `text = traceback.format_exc()`

## traceback.extract_tb()  `(standard library)`
**1. Definition:** `traceback.extract_tb(tb)` converts a traceback object into a list of `FrameSummary` items with `filename`, `lineno`, `name` and `line`.
**2. In plain words:** It lets you inspect each frame as data rather than text. The script prints `file=main.py line=20 function=get_limit code=...` for every frame.
**3. Example to quote:**
> "I looped over `extract_tb(err.__traceback__)` and printed each frame's function name and line number."
```python
for f in traceback.extract_tb(err.__traceback__):
    print(f.name, f.lineno)
```
**Seen in:** `main.py`, frame loop

## Reading recipe (bottom to top)  `(concept)`
**1. Definition:** A systematic way to read a traceback: take the last line (error type and message), then the frame just above it (the raising line), then climb up the call chain.
**2. In plain words:** Step 1: what happened? Step 2: on which line? Step 3: who called that, and with what data? Step 4: is the bug in the raising line or in the caller's data? This turns a scary wall of text into four quick questions.
**3. Example to quote:**
> "I check the last line first, then the raising line, then the callers, and only then decide where to fix."
```python
# last line -> raising frame -> callers -> decide
```
**Seen in:** `WALKTHROUGH.md`, execution trace

## Raise site vs call site  `(concept)`
**1. Definition:** The raise site is the line where the exception was raised; the call site is the line that invoked the function containing it.
**2. In plain words:** The line that blows up is not always where the bug is. In `main.py` the crash is in `get_limit`, but the real problem is that `load_config()` never provided `max_tokens`. Look at both.
**3. Example to quote:**
> "The traceback blamed `get_limit`, but the root issue was the config passed in from `run`."
```python
# raise site: config["max_tokens"]   call site: get_limit(config)
```
**Seen in:** `main.py`, comments on frames

## Root cause  `(concept)`
**1. Definition:** The root cause is the underlying reason a failure occurs, as opposed to the line where the symptom appears.
**2. In plain words:** In `broken_traceback.py` the failing line indexes `scores[name]`, but the root cause is that "carol" is invalid input. Fixing only the symptom (swallowing the error) would hide it. Write down the cause before fixing.
**3. Example to quote:**
> "The root cause was a name that isn't in the scores dict, not a bug in `report`; I fixed it by validating the names."
```python
missing = [n for n in names if n not in scores]
```
**Seen in:** `broken_traceback.py`; `WALKTHROUGH.md`

## Exception chaining (raise ... from)  `(syntax)`
**1. Definition:** `raise NewError(...) from err` raises a new exception while preserving the original as its cause, so both appear in the traceback.
**2. In plain words:** You can translate a low-level `KeyError` into a clearer `ValueError("config missing max_tokens")` without losing the original details. The output says "The above exception was the direct cause of the following exception".
**3. Example to quote:**
> "I wrapped the `KeyError` in a `ValueError` with `from err`, and the traceback showed both."
```python
try:
    {}["k"]
except KeyError as err:
    raise ValueError("missing k") from err
```
**Seen in:** `WALKTHROUGH.md`, experiment 3

## NameError  `(Python built-in)`
**1. Definition:** `NameError` is raised when code uses a name that has not been defined in the current scope.
**2. In plain words:** If you rename a parameter but not its use inside the function, Python can't find the old name. The traceback gives the line number and the missing name, which makes it quick to fix.
**3. Example to quote:**
> "After renaming the parameter, I got `NameError: name 'config' is not defined` pointing at the old name."
```python
# print(undefined_name)  -> NameError
```
**Seen in:** `main.py`, BREAK IT

## max(..., key=...)  `(Python built-in)`
**1. Definition:** `max(iterable, key=func)` returns the item for which `func(item)` is largest; `scores.get` as the key picks the dict key with the highest value.
**2. In plain words:** `max(scores, key=scores.get)` iterates over the dict's keys, but compares them by their scores, so it returns the name with the best score ("alice").
**3. Example to quote:**
> "`max(scores, key=scores.get)` returned 'alice' because her score 0.9 was the highest."
```python
print(max({"a": 1, "b": 5}, key={"a": 1, "b": 5}.get))   # b
```
**Seen in:** `broken_traceback.py`, `best_student`

## Drill discipline  `(concept)`
**1. Definition:** A drill discipline is answering fixed diagnostic questions in writing (error type, raising line, calling line, root cause) before changing any code.
**2. In plain words:** It trains you to read first and fix second, which is exactly what an interviewer wants to see when you debug without AI. Writing the answers also exposes guesses.
**3. Example to quote:**
> "Before fixing `broken_traceback.py`, I wrote: KeyError, line 21, called from line 25, cause: 'carol' not in scores."
```python
# (a) type (b) raising line (c) call line (d) root cause
```
**Seen in:** `broken_traceback.py` docstring; `WALKTHROUGH.md` debugging practice
