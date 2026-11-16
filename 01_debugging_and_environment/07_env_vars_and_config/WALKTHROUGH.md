# Walkthrough 07 — Environment variables and configuration

## Goal
Read settings (ports, model names, secrets) from the environment with defaults and clear failures. The same mechanism configures Docker containers ([module 11](../../11_docker/)) and API keys ([module 08](../../08_agent_harness/)).

## Run it
```bash
cd 01_debugging_and_environment/07_env_vars_and_config
python main.py
PORT=9000 MODEL_NAME=gpt2 python main.py     # VAR=value before the command sets it for that run
PORT=abc python main.py                      # clear config error
python broken_env.py                         # EXPECTED crash
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| environment variable | env var, shell variable, `os.environ` entry | Roughly | Env vars are exported to child processes; plain shell variables are not. |
| `.env` file | dotenv file | Same | A text file of `KEY=VALUE` lines, read by python-dotenv, not by Python itself. |
| `os.environ["X"]` vs `os.environ.get("X")` | strict vs forgiving | No | `[]` raises KeyError when missing; `.get` returns `None` or a default. |
| `os.getenv` vs `os.environ.get` | same lookup | Yes | `os.getenv(k, d)` is a shortcut for `os.environ.get(k, d)`. |
| config vs secret | settings vs credentials | No | Config can live in git; secrets must not. |
| `SystemExit` vs `sys.exit` | exit exception vs exit function | Same effect | `sys.exit()` raises `SystemExit`. |

## Code, step by step

### 1. Header docstring  (`main.py` lines 1–11)

```python
"""
GOAL:  Read configuration from environment variables safely (defaults, type conversion, required values).

RUN:   cd 01_debugging_and_environment/07_env_vars_and_config && python main.py
       with overrides:  PORT=9000 MODEL_NAME=gpt2 python main.py
       Crashing version (EXPECTED):  python broken_env.py

EXPECTED OUTPUT: the resolved settings, with a clear error for bad values.

NEEDS: nothing (standard library only).
"""
```

- **In simple words:** Label for the file: read configuration from the environment safely.
- **Exactly:** Docstring; nothing runs.
- **Why it is here:** Shows the override syntax `PORT=9000 python main.py`.

### 2. Import os and inspect os.environ  (`main.py` lines 13–16)

```python
import os

# os.environ behaves like a dict of strings. EVERYTHING in it is a string -- even PORT=8000.
print("HOME is set?", "HOME" in os.environ)
```

- **In simple words:** `os.environ` is the set of settings your shell hands to the program.
- **Exactly:** `os.environ` behaves like a dict of `str` to `str`; everything in it is text, even `PORT=8000`. `"HOME" in os.environ` -> `True`.
- **Why it is here:** Knowing it is a dict of strings explains most config bugs.

### 3. Read a variable with a default  (`main.py` lines 18–20)

```python
# .get(name, default): never raises, returns default if missing
model_name = os.environ.get("MODEL_NAME", "distilgpt2")
print("MODEL_NAME:", model_name)
```

- **In simple words:** Ask for `MODEL_NAME`; if it is not set, use `distilgpt2`.
- **Exactly:** `os.environ.get(name, default)` never raises. Output: `MODEL_NAME: distilgpt2`.
- **Why it is here:** Using `[]` would crash with `KeyError` when the variable is missing.

### 4. get_int: convert and fail helpfully  (`main.py` lines 22–28)

```python
# Convert types yourself and fail with a HELPFUL message
def get_int(name, default):
    raw = os.environ.get(name, str(default))
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"Config error: {name} must be an integer, got {raw!r}")
```

- **In simple words:** Read a number setting and stop with a clear message if it is not a number.
- **Exactly:** `os.environ.get(name, str(default))` gives text; `int(raw)` converts. A `ValueError` is turned into `SystemExit("Config error: PORT must be an integer, got 'abc'")`, which ends the program with that message and no traceback.
- **Why it is here:** Without it, a typo in `PORT` crashes deep in the server with an unrelated error.

### 5. get_required: demand a value  (`main.py` lines 31–35)

```python
def get_required(name):
    value = os.environ.get(name)
    if not value:                          # missing or empty string
        raise SystemExit(f"Config error: required environment variable {name} is not set")
    return value
```

- **In simple words:** For settings with no sensible default (API keys), refuse to start without them.
- **Exactly:** `if not value` catches both missing (`None`) and empty string. Raises `SystemExit` with a clear message. (This helper is defined but not called in this run.)
- **Why it is here:** Failing at startup is better than failing at the first API call.

### 6. Resolve PORT and DEBUG  (`main.py` lines 38–41)

```python
port = get_int("PORT", 8000)
print("PORT:", port, type(port).__name__)
debug = os.environ.get("DEBUG", "false").lower() in {"1", "true", "yes"}   # booleans need parsing too
print("DEBUG:", debug)
```

- **In simple words:** Read the port as an `int` and the debug flag as a `bool`.
- **Exactly:** `get_int("PORT", 8000)` -> `8000` (type `int`). `"false".lower() in {"1", "true", "yes"}` -> `False`. Real output: `PORT: 8000 int`, `DEBUG: False`. Note `bool("false")` would be `True` (non-empty string), so booleans need explicit parsing.
- **Why it is here:** Config values arrive as strings; each type needs its own parsing.

### 7. Check a secret without printing it  (`main.py` lines 43–45)

```python
# Secrets (API keys) belong in env vars or a .env file, NEVER in code or git.
key = os.environ.get("ANTHROPIC_API_KEY")
print("API key configured?", bool(key))   # print True/False, never the key itself
```

- **In simple words:** Learn whether the API key is present, never show it.
- **Exactly:** `os.environ.get("ANTHROPIC_API_KEY")` is `None` if unset; `bool(key)` -> `False`. Prints `API key configured? False`.
- **Why it is here:** Keys in logs or git history are a security incident.

### 8. .env file support  (`main.py` lines 47–54)

```python
# A .env file is just KEY=VALUE lines; python-dotenv loads them into os.environ.
try:
    from dotenv import load_dotenv
    from pathlib import Path
    loaded = load_dotenv(Path(__file__).resolve().parent / ".env")   # False if file missing
    print(".env loaded?", loaded, "(copy .env.example to .env to try)")
except ImportError:
    print("python-dotenv not installed (optional)")
```

- **In simple words:** Optionally load KEY=VALUE lines from a `.env` file next to the script.
- **Exactly:** `load_dotenv(path)` (python-dotenv) copies lines into `os.environ` and returns `True` if the file existed, else `False`. Real output: `.env loaded? False (copy .env.example to .env to try)`. `except ImportError` handles the package being absent.
- **Why it is here:** Convenient local config without exporting variables each time; `.env` stays out of git.

### 9. TRY IT and BREAK IT comments  (`main.py` lines 56–57)

```python
# TRY IT: run `PORT=9000 python main.py` -- note the shell syntax VAR=value command.
# BREAK IT: run `PORT=abc python main.py` and read the message.
```

- **In simple words:** Experiments for you (comments only).
- **Exactly:** `PORT=abc python main.py` prints `Config error: PORT must be an integer, got 'abc'`.
- **Why it is here:** Shows the helpful error in action.

### 10. broken_env.py: docstring  (`broken_env.py` lines 1–1)

```python
"""EXPECTED TO CRASH. Two config bugs. Fix with .get() + defaults and int() with a clear error (see main.py)."""
```

- **In simple words:** Announces two config bugs.
- **Exactly:** One-line docstring.
- **Why it is here:** Sets the exercise.

### 11. broken_env bug 1: required variable missing  (`broken_env.py` lines 2–5)

```python
import os

api_key = os.environ["SERVICE_API_KEY"]      # KeyError if the variable is not set
print("key length:", len(api_key))
```

- **In simple words:** Read a variable with square brackets and use it.
- **Exactly:** `os.environ["SERVICE_API_KEY"]` raises `KeyError: 'SERVICE_API_KEY'` when unset (real traceback ends at line 4, plus a frame inside `<frozen os>`). With `SERVICE_API_KEY=abc` it passes and prints `key length: 3`. Fix: `.get()` plus a clear message.
- **Why it is here:** Typical "works on my machine" bug: the variable exists only in your shell.

### 12. broken_env bug 2: PORT is a string  (`broken_env.py` lines 7–9)

```python
os.environ["PORT"] = "8000"                  # pretend the shell exported this
port = os.environ["PORT"]
print("next port:", port + 1)                # a string, not an int!
```

- **In simple words:** Set PORT then try to add one to it.
- **Exactly:** `os.environ["PORT"] = "8000"` stores a string (it must be one). `port + 1` raises `TypeError: can only concatenate str (not "int") to str` at line 9. Fix: `int(os.environ["PORT"]) + 1`.
- **Why it is here:** Everything in the environment is text; convert explicitly.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `os.environ["X"]` | Required: raises `KeyError` if missing |
| `os.environ.get("X", default)` | Optional with default |
| `raise SystemExit("msg")` | Stop the program with a message (exit code 1) |
| `value.lower() in {"1","true","yes"}` | Parse booleans from strings |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does | Returns |
|---|---|---|---|
| `os.environ` | standard lib | Mapping of env vars (all strings) | dict-like |
| `dotenv.load_dotenv(path)` | 3rd-party (`python-dotenv`) | Loads `KEY=VALUE` lines from `.env` into `os.environ` | bool |

## Execution trace (real output)
```
HOME is set? True
MODEL_NAME: distilgpt2
PORT: 8000 int
DEBUG: False
API key configured? False
.env loaded? False (copy .env.example to .env to try)
```
With `PORT=abc`: `Config error: PORT must be an integer, got 'abc'`.
`broken_env.py` ends with `KeyError: 'SERVICE_API_KEY'`; bug 2 is `"8000" + 1` -> `TypeError` because env values are always strings.

## Experiments
1. Copy `.env.example` to `.env`; run again: `.env loaded? True`, and real environment variables win over `.env` values (`override=False` is the default) - test with `PORT=1 python main.py`.
2. `DEBUG=yes python main.py`.
3. Print `bool("false")` -> `True` (non-empty string!) — why parsing is needed.

## Debugging practice
When "it works locally, not in Docker/CI": print which variables exist (`docker exec C env`, or in Python `sorted(os.environ)`), never the secret values.

## Rebuild checklist
- [ ] Read a string, int and bool setting with defaults
- [ ] Fail early with a human-readable message when a required variable is missing
- [ ] Explain why secrets stay out of source control

## Say it aloud
"Environment variables separate configuration from code, so the same program runs on my laptop, in Docker and in production. They are always strings, so I convert and validate them at startup, and I fail fast with a clear message if a required one is missing."
Trade-off: env vars are easy to misconfigure and have no schema; large configs belong in a validated settings object (Pydantic settings).
