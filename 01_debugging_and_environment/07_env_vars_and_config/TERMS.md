# Terms — Environment Variables and Configuration

Folder: `01_debugging_and_environment/07_env_vars_and_config` · Files covered: `main.py`, `broken_env.py`, `.env.example`, `WALKTHROUGH.md`
Back to module: [01 Debugging and environment README](../README.md)

**Terms in this folder:** [Environment variable](#environment-variable) · [os.environ](#osenviron) · [os.environ.get with default](#osenvironget-with-default) · [os.environ["X"] (required)](#osenvironx-required) · [Everything in the environment is a string](#everything-in-the-environment-is-a-string) · [Type conversion with a helpful error](#type-conversion-with-a-helpful-error) · [raise SystemExit](#raise-systemexit) · [Required vs optional settings](#required-vs-optional-settings) · [Parsing booleans](#parsing-booleans) · [Secrets and API keys](#secrets-and-api-keys) · [.env file](#env-file) · [python-dotenv / load_dotenv](#python-dotenv--load_dotenv) · [override=False precedence](#overridefalse-precedence) · [VAR=value command (shell syntax)](#varvalue-command-shell-syntax) · [ImportError guard for optional libraries](#importerror-guard-for-optional-libraries) · [Fail fast](#fail-fast) · [KeyError from a missing variable](#keyerror-from-a-missing-variable) · [Settings object (Pydantic settings)](#settings-object-pydantic-settings) · [Set membership for parsing](#set-membership-for-parsing)

---

## Environment variable  `(concept)`
**1. Definition:** An environment variable is a named string value provided by the operating system or launcher to a process, commonly used for configuration such as ports, model names and secrets.
**2. In plain words:** Instead of editing code to change a setting, you set `PORT=9000` before running the program. The same code then behaves differently on your laptop, in Docker and in production. They have no schema, so they are easy to misconfigure.
**3. Example to quote:**
> "I read `MODEL_NAME` from the environment so the same script can load `distilgpt2` locally and `gpt2` elsewhere."
```bash
MODEL_NAME=gpt2 python main.py
```
**Seen in:** `main.py`; `WALKTHROUGH.md`

## os.environ  `(standard library)`
**1. Definition:** `os.environ` is a dict-like mapping of the process's environment variables, with string keys and string values.
**2. In plain words:** It behaves like a dictionary: you can check `"HOME" in os.environ`, index it, or use `.get`. Changes you make to it affect only the current process and its children.
**3. Example to quote:**
> "`'HOME' in os.environ` returned `True`, so I knew environment access works."
```python
import os
print("HOME" in os.environ)
```
**Seen in:** `main.py`, first lines

## os.environ.get with default  `(standard library)`
**1. Definition:** `os.environ.get("NAME", default)` returns the variable's value, or `default` if it is not set, without raising.
**2. In plain words:** It's the usual way to read optional settings: `MODEL_NAME` falls back to `"distilgpt2"` when nothing is set.
**3. Example to quote:**
> "`os.environ.get('MODEL_NAME', 'distilgpt2')` printed `distilgpt2` because I hadn't set the variable."
```python
model_name = os.environ.get("MODEL_NAME", "distilgpt2")
```
**Seen in:** `main.py`

## os.environ["X"] (required)  `(standard library)`
**1. Definition:** Indexing `os.environ["X"]` returns the value or raises `KeyError` if the variable is missing.
**2. In plain words:** Use it when a variable is mandatory, but be aware the error is a bare `KeyError`. A friendlier approach is to check and raise a clear message, as `get_required` does.
**3. Example to quote:**
> "`broken_env.py` crashed with `KeyError: 'SERVICE_API_KEY'` because the variable wasn't set."
```python
# os.environ["SERVICE_API_KEY"]  -> KeyError if unset
```
**Seen in:** `broken_env.py`

## Everything in the environment is a string  `(concept)`
**1. Definition:** All environment variable values are strings, even when they look numeric, like `PORT=8000`.
**2. In plain words:** `os.environ["PORT"] + 1` fails with `TypeError` because `"8000"` is text. Convert with `int()`. Likewise `bool("false")` is `True` because the string is non-empty.
**3. Example to quote:**
> "`port + 1` crashed because `port` was the string `'8000'`; I convert with `int(port)` first."
```python
port = int(os.environ.get("PORT", "8000"))
```
**Seen in:** `main.py`, comment; `broken_env.py`, bug 2

## Type conversion with a helpful error  `(concept)`
**1. Definition:** A helper like `get_int(name, default)` reads a variable, converts it with `int()`, and reports a clear message when conversion fails.
**2. In plain words:** `PORT=abc` gives `Config error: PORT must be an integer, got 'abc'` instead of a confusing `ValueError` from deep in the program. The default is passed as `str(default)` so both paths use the same conversion.
**3. Example to quote:**
> "With `PORT=abc` my program stops with 'PORT must be an integer, got 'abc'' rather than a cryptic traceback."
```python
def get_int(name, default):
    return int(os.environ.get(name, str(default)))
```
**Seen in:** `main.py`, `get_int`

## raise SystemExit  `(Python built-in)`
**1. Definition:** `raise SystemExit("message")` stops the program, printing the message to stderr and exiting with a non-zero status.
**2. In plain words:** It's a clean way to abort on a configuration error without a long traceback. A non-zero exit code also tells shells, CI and Docker that the program failed.
**3. Example to quote:**
> "On a bad config I `raise SystemExit('Config error: ...')` so the process exits with code 1 and a readable message."
```python
raise SystemExit("Config error: PORT must be an integer")
```
**Seen in:** `main.py`, `get_int`, `get_required`

## Required vs optional settings  `(concept)`
**1. Definition:** Optional settings have defaults; required settings have none and must be provided or the program should stop.
**2. In plain words:** `get_required("X")` treats both a missing and an empty value as an error (`if not value`). Use required for things like API keys that have no safe default.
**3. Example to quote:**
> "`MODEL_NAME` is optional with a default, but the API key is required, so startup fails if it's missing."
```python
if not value:
    raise SystemExit("X is required")
```
**Seen in:** `main.py`, `get_required`

## Parsing booleans  `(concept)`
**1. Definition:** Boolean settings from the environment must be parsed from text, e.g. `value.lower() in {"1", "true", "yes"}`.
**2. In plain words:** `bool("false")` is `True`, because any non-empty string is truthy. Parsing explicitly avoids that trap.
**3. Example to quote:**
> "`DEBUG=yes` becomes `True` through my parser, while `bool('false')` would wrongly give `True`."
```python
debug = os.environ.get("DEBUG", "false").lower() in {"1", "true", "yes"}
```
**Seen in:** `main.py`, `debug`

## Secrets and API keys  `(concept)`
**1. Definition:** Secrets such as API keys are sensitive credentials that must be supplied via environment variables or a `.env` file, never hard-coded in source or committed to git.
**2. In plain words:** If a key is in your code or repo, anyone with access can use it. Print only whether it's configured (`bool(key)`), never the value.
**3. Example to quote:**
> "I print `API key configured? True/False` and never the key itself, and `.env` is in `.gitignore`."
```python
print(bool(os.environ.get("ANTHROPIC_API_KEY")))
```
**Seen in:** `main.py`, API key section

## .env file  `(concept)`
**1. Definition:** A `.env` file is a plain-text file of `KEY=VALUE` lines (no spaces around `=`) used to set environment variables for local development.
**2. In plain words:** It saves retyping variables each run. Commit a `.env.example` showing the names, but never commit the real `.env`. In this folder, `.env.example` has `MODEL_NAME=gpt2` and `PORT=9001`.
**3. Example to quote:**
> "I copy `.env.example` to `.env` and fill in local values; `.env` stays out of git."
```text
MODEL_NAME=gpt2
PORT=9001
```
**Seen in:** `.env.example`; `main.py`

## python-dotenv / load_dotenv  `(third-party: python-dotenv)`
**1. Definition:** `dotenv.load_dotenv(path)` reads a `.env` file and loads its entries into `os.environ`, returning `True` if the file was found.
**2. In plain words:** It bridges `.env` files and your code. It returns `False` if the file is missing, so the script can say ".env loaded? False".
**3. Example to quote:**
> "`load_dotenv(Path(__file__).resolve().parent / '.env')` returned `False` because I hadn't created the file yet."
```python
from dotenv import load_dotenv
load_dotenv(".env")
```
**Seen in:** `main.py`

## override=False precedence  `(concept)`
**1. Definition:** By default `load_dotenv` does not overwrite variables that are already set in the real environment (`override=False`), so real environment variables win over `.env` values.
**2. In plain words:** If the shell sets `PORT=1`, a `.env` with `PORT=9001` doesn't change it. This lets production settings override local defaults.
**3. Example to quote:**
> "`PORT=1 python main.py` still printed 1 even with a `.env` file, because environment variables win."
```bash
PORT=1 python main.py
```
**Seen in:** `WALKTHROUGH.md`, experiment 1

## VAR=value command (shell syntax)  `(command)`
**1. Definition:** Prefixing a command with `VAR=value` sets that variable only for that one command's process.
**2. In plain words:** `PORT=9000 python main.py` runs the script with `PORT` set, without changing your shell's settings afterward. It's the quick way to test overrides.
**3. Example to quote:**
> "I tested the override with `PORT=9000 MODEL_NAME=gpt2 python main.py` and the output changed accordingly."
```bash
PORT=9000 python main.py
```
**Seen in:** `main.py`, TRY IT; `WALKTHROUGH.md`

## ImportError guard for optional libraries  `(syntax)`
**1. Definition:** A `try: from dotenv import load_dotenv / except ImportError:` block lets the script continue when an optional library is missing.
**2. In plain words:** The script prints "python-dotenv not installed (optional)" instead of crashing. It's good for non-essential features.
**3. Example to quote:**
> "I wrapped the dotenv import in `try/except ImportError`, so the script still works without it."
```python
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
```
**Seen in:** `main.py`

## Fail fast  `(concept)`
**1. Definition:** Fail fast means detecting invalid configuration or input at startup and stopping immediately with a clear message.
**2. In plain words:** It is far easier to debug "PORT must be an integer" at launch than a strange failure an hour later. Validate settings once, at the start.
**3. Example to quote:**
> "My app validates all env settings at startup so a missing key stops it immediately, not mid-request."
```python
port = get_int("PORT", 8000)
```
**Seen in:** `main.py`; `WALKTHROUGH.md`

## KeyError from a missing variable  `(Python built-in)`
**1. Definition:** Indexing `os.environ` with an unset name raises `KeyError: 'NAME'`.
**2. In plain words:** The message shows only the name, so you need to realise the variable isn't set in this process. In Docker, check with `docker exec <container> env`.
**3. Example to quote:**
> "`KeyError: 'SERVICE_API_KEY'` meant the variable wasn't exported in this shell."
```python
print(sorted(os.environ)[:5])
```
**Seen in:** `broken_env.py`

## Settings object (Pydantic settings)  `(concept)`
**1. Definition:** A settings object is a validated, typed class holding all configuration values, often built with Pydantic settings that read from environment variables.
**2. In plain words:** Env vars have no schema. For larger configs, a typed settings class validates everything in one place. This comes up again in module 09.
**3. Example to quote:**
> "For many settings I'd use a validated settings class instead of scattered `os.environ.get` calls."
```python
# class Settings(BaseSettings): port: int = 8000
```
**Seen in:** `WALKTHROUGH.md`, trade-off

## Set membership for parsing  `(syntax)`
**1. Definition:** `x in {"1", "true", "yes"}` tests membership in a set literal, returning a boolean.
**2. In plain words:** It's a compact way to check "is this one of the accepted spellings?". Sets give fast lookups and read clearly.
**3. Example to quote:**
> "`value.lower() in {'1', 'true', 'yes'}` accepts three spellings of true."
```python
print("yes" in {"1", "true", "yes"})   # True
```
**Seen in:** `main.py`, `debug`
