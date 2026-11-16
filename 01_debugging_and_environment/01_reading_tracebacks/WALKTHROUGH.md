# Walkthrough 01 — Reading a traceback

## Goal
Read any Python traceback in 30 seconds: **bottom line first, then the frame above it, then climb.**

## Run it
```bash
cd 01_debugging_and_environment/01_reading_tracebacks
python main.py                 # prints a traceback on purpose and dissects it
python broken_traceback.py     # EXPECTED to crash: your drill
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| traceback | stack trace, backtrace, call stack dump | Same | Python prints oldest call first, error last; many languages print newest first. |
| exception | error, raised error | Roughly | An exception is the object; the traceback is the report about where it travelled. |
| frame | stack frame, call frame | Same | One entry in the call chain: file, line, function. |
| `KeyError` vs `IndexError` | missing key vs bad position | No | `KeyError` is for dicts, `IndexError` for lists/tuples. |
| root cause vs symptom | where it crashed vs why | No | The raising line is the symptom; the root cause may be earlier data. |

## Code, step by step

### 1. Header docstring and import traceback  (`main.py` lines 1–12)

```python
"""
GOAL:  Learn the anatomy of a traceback by producing one on purpose and taking it apart.

RUN:   cd 01_debugging_and_environment/01_reading_tracebacks && python main.py
       Then run the crashing version:  python broken_traceback.py   (EXPECTED to crash)

EXPECTED OUTPUT: a printed traceback plus a breakdown of its parts.

NEEDS: nothing (standard library only), no internet.
"""

import traceback  # standard library: lets us capture/print tracebacks as text
```

- **In simple words:** The label for the file, then the one library we need: a tool for turning a crash report into text we can inspect.
- **Exactly:** `import traceback` loads the standard-library module whose `format_exc()` and `extract_tb()` expose what Python prints on a crash.
- **Why it is here:** Without it we could only watch the crash scroll past; with it we can take the traceback apart in code.

### 2. load_config returns a dict without max_tokens  (`main.py` lines 15–16)

```python
def load_config():
    return {"model": "gpt2"}            # note: there is NO "max_tokens" key
```

- **In simple words:** A tiny function that hands back a settings dictionary. It deliberately lacks the key `max_tokens`.
- **Exactly:** Returns `{"model": "gpt2"}`, a `dict[str, str]` with one key.
- **Why it is here:** The missing key is the seed of the bug we are about to trigger.

### 3. get_limit asks for the missing key  (`main.py` lines 19–20)

```python
def get_limit(config):
    return config["max_tokens"]         # <-- the real bug: KeyError happens HERE
```

- **In simple words:** This function trusts the dict to have `max_tokens` and reads it directly.
- **Exactly:** `config["max_tokens"]` uses square-bracket lookup; if the key is absent Python raises `KeyError: 'max_tokens'`. This is the line where the exception is RAISED (line 20).
- **Why it is here:** `[]` fails loudly (good for finding bugs); `.get()` would silently return `None`. The traceback's innermost frame points here.

### 4. run calls the two functions  (`main.py` lines 23–25)

```python
def run():
    config = load_config()
    return get_limit(config)            # <-- call site (middle frame)
```

- **In simple words:** The middle layer: gets the config, then passes it to `get_limit`.
- **Exactly:** `run()` calls `load_config()` (fine) then `get_limit(config)` on line 25. That line is the call site, the middle frame of the traceback.
- **Why it is here:** Gives the traceback three levels so you can practise reading who called whom.

### 5. Run it inside try/except and capture the text  (`main.py` lines 28–33)

```python
try:
    run()                               # <-- outermost frame (module level)
except KeyError as err:
    # traceback.format_exc() returns the same text Python prints when a program crashes.
    text = traceback.format_exc()
    print(text)
```

- **In simple words:** Run the failing code but catch the crash so the script can explain it instead of dying.
- **Exactly:** `run()` at line 29 is the outermost frame (`<module>`). `except KeyError as err` binds the exception object. `traceback.format_exc()` returns the same multi-line string Python would print: three `File ..., line N, in ...` entries (lines 29, 25, 20) and the final `KeyError: 'max_tokens'`. Python 3.11+ adds `^^^^` markers under the failing expression.
- **Why it is here:** Catching it lets us study a real traceback without ending the program.

### 6. Print the anatomy: last line first  (`main.py` lines 35–40)

```python
    # ---- anatomy: read BOTTOM -> TOP ----------------------------------------
    last_line = text.strip().splitlines()[-1]
    print("1) LAST LINE  = error type + message  :", last_line)
    print("2) The frame just above it            : where the exception was RAISED")
    print("3) Frames above that                  : who called whom (oldest call at the top)")
    print("4) err.args                           :", err.args)
```

- **In simple words:** Reading a traceback bottom to top: the last line says WHAT went wrong, the lines above say WHERE and via WHOM.
- **Exactly:** `text.strip().splitlines()[-1]` is the last line: `KeyError: 'max_tokens'`. `err.args` is the tuple of arguments the exception was created with: `('max_tokens',)`. Lines 2-3 are fixed explanatory prints.
- **Why it is here:** Reading bottom-up is the key habit: error type and message first, then the raising frame, then the callers.

### 7. Walk every frame as structured data  (`main.py` lines 42–45)

```python
    # tb_frame / traceback.extract_tb give structured access to each frame
    for frame in traceback.extract_tb(err.__traceback__):
        print(f"   frame: file={frame.filename.split('/')[-1]} line={frame.lineno} "
              f"function={frame.name} code={frame.line!r}")
```

- **In simple words:** Print each stack frame (file, line, function, code) as a tidy list.
- **Exactly:** `err.__traceback__` is the traceback object; `traceback.extract_tb()` returns a list of `FrameSummary` objects with `.filename`, `.lineno`, `.name`, `.line`. Real output: `line=29 function=<module>`, `line=25 function=run`, `line=20 function=get_limit` (oldest call first, raising frame last). `frame.line!r` shows the source text with quotes.
- **Why it is here:** Logging tools and error trackers read exactly this structure.

### 8. TRY IT and BREAK IT comments  (`main.py` lines 47–48)

```python
# TRY IT: change config["max_tokens"] to config.get("max_tokens", 100) and re-run: no error, no except branch.
# BREAK IT: rename get_limit's parameter but not its use inside -> NameError. Read the line number it reports.
```

- **In simple words:** Experiments for you (comments only).
- **Exactly:** `config.get("max_tokens", 100)` returns `100` and removes the error; renaming a parameter but not its use gives `NameError`.
- **Why it is here:** Shows the two standard fixes and a second error type.

### 9. broken_traceback.py: the drill instructions  (`broken_traceback.py` lines 1–5)

```python
"""
EXPECTED TO CRASH.  Do the drill: do NOT fix it until you can answer
  (a) what is the error type?  (b) which line RAISED it?  (c) which line CALLED the function?  (d) root cause?
RUN: python broken_traceback.py
"""
```

- **In simple words:** This file is the exercise: it will crash, and you must answer four questions BEFORE fixing it.
- **Exactly:** Docstring only. The four questions: error type, raising line, calling line, root cause.
- **Why it is here:** Forces you to read the traceback instead of guessing.

### 10. broken: load_scores returns two students  (`broken_traceback.py` lines 8–9)

```python
def load_scores():
    return {"alice": 0.9, "bob": 0.7}
```

- **In simple words:** Returns the scores of two people.
- **Exactly:** `{"alice": 0.9, "bob": 0.7}`: only two keys; there is no "carol".
- **Why it is here:** The data that does not match what the caller expects is the root cause.

### 11. broken: best_student (works fine)  (`broken_traceback.py` lines 12–14)

```python
def best_student(scores):
    best = max(scores, key=scores.get)
    return best, scores[best]
```

- **In simple words:** Finds the key with the highest score and returns it with its score.
- **Exactly:** `max(scores, key=scores.get)` compares keys by their values -> `'alice'`; returns `('alice', 0.9)`. This function never fails.
- **Why it is here:** A working neighbour: not every function in the file is guilty, so you must follow the traceback, not suspicion.

### 12. broken: report builds one line per name  (`broken_traceback.py` lines 17–22)

```python
def report(names):
    scores = load_scores()
    lines = []
    for name in names:
        lines.append(f"{name}: {scores[name]}")   # one of these names is not in scores
    return "\n".join(lines)
```

- **In simple words:** For each requested name, look up its score and make a text line.
- **Exactly:** `scores[name]` on line 21 raises `KeyError: 'carol'` on the third iteration because `carol` is not a key. The first two lines were built but lost because `return` is never reached.
- **Why it is here:** This is the line that RAISED the error (line 21).

### 13. broken: the two calls  (`broken_traceback.py` lines 25–26)

```python
print(best_student(load_scores()))
print(report(["alice", "bob", "carol"]))
```

- **In simple words:** First a good call, then the call that crashes.
- **Exactly:** Line 25 prints `('alice', 0.9)`. Line 26 calls `report(["alice","bob","carol"])` and the traceback shows: `line 26, in <module>` (the CALLER), `line 21, in report` (the RAISER) and `KeyError: 'carol'`. Root cause: the input names list contains a name the data does not have.
- **Why it is here:** Answers (b) raised on 21, (c) called from 26, (d) caller/data mismatch; fix with `.get()` or a membership check.

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
