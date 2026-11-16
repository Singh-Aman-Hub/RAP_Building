# Step 08 — Environment variables and configuration

## The story so far
Step 07 showed that workers, ports and model paths change between a laptop and a server. Hard-coding them means editing code, and code is exactly what we want to keep identical.

## The pain point
A password or model path baked into the source ends up in git. A different value for staging needs a different build, and nobody remembers which one is running.

## The spark
Keep settings outside the code in environment variables, read them in one settings object, convert types explicitly, and fail fast when something required is missing.

## Goal
Keep settings (model path, versions, secrets, limits) **outside the code** so the same code runs on a laptop, in Docker (module 11) and in production.

## Run it
```bash
cd 09_api_development/08_config_and_env && python main.py
```
Real output:
```
1. os.environ['DEMO_NAME'] = bob
2. getenv with default     = fallback
   os.environ[...] raised KeyError: 'DOES_NOT_EXIST'
3. type = str | get_settings().max_text_length = 42 ( int )
   bool('false') = True -> that is why config.py compares the string to '1'
4. dotenv_values       = {'MODEL_VERSION': 'from-dotenv', 'MAX_TEXT_LENGTH': '999'}
5. override=False       -> MODEL_VERSION = from-shell
   override=True        -> MODEL_VERSION = from-dotenv
6. model_path = None | version = from-dotenv
```

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Environment variable vs `.env` file | Config / env vars / dotenv | No | An env variable lives in the process environment; a `.env` file is a text file that a library copies into the environment |
| `os.environ[...]` vs `os.getenv(...)` | Required vs optional lookup | No | `[]` raises KeyError if missing; `getenv` returns `None` or a default |
| `load_dotenv` vs `dotenv_values` | Two python-dotenv functions | No | `load_dotenv` mutates `os.environ`; `dotenv_values` just returns a dict |
| `override=False` vs `override=True` | Precedence switch | No | False: existing env wins; True: the `.env` file wins |
| Config vs secrets | Settings / credentials | Overlapping | Secrets (API keys) are config that must never be printed or committed |
| 12-factor config vs config file (YAML/JSON) | Where settings live | No | 12-factor stores config in the environment; config files live on disk beside the code |

## Code, step by step

### 1. Docstring: goal and needs  (`main.py` lines 1–11)
```python
"""
GOAL:   Understand environment variables and configuration: how a program reads settings
        from outside its code, the .env file convention, precedence rules, and the classic
        "everything is a string" bug.

RUN:    cd 09_api_development/08_config_and_env && python main.py

EXPECTED OUTPUT (shape): a numbered list of small experiments (1..6) with their results.

NEEDS:  python-dotenv (third-party). os / tempfile are built-in.
"""
```
- **In simple words:** Says what the script teaches: settings that live outside the code.
- **Exactly:** Expected output is six numbered mini-experiments. Needs python-dotenv; `os`/`tempfile` are built in.
- **Why it is here:** Orients you before the code.

### 2. Imports  (`main.py` lines 13–18)
```python
import os
import sys
import tempfile
from pathlib import Path

from dotenv import dotenv_values, load_dotenv    # dotenv_values: read a .env file into a dict WITHOUT touching os.environ
```
- **In simple words:** Bring in tools for environment variables, temporary folders, paths and `.env` files.
- **Exactly:** `dotenv_values(path)` returns a dict of a `.env` file WITHOUT changing `os.environ`; `load_dotenv(path)` copies those values INTO `os.environ`.
- **Why it is here:** The two dotenv functions are the heart of experiments 4 and 5.

### 3. Import the real Settings from step 06  (`main.py` lines 20–22)
```python
# Reuse the real settings object from step 06 (shows config is not just a toy).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "06_model_api_flask_vs_fastapi"))
from config import get_settings  # noqa: E402  (import after sys.path change is intentional)
```
- **In simple words:** Reuse the real config file from the previous step so the demo is not a toy.
- **Exactly:** `sys.path.insert(0, ...)` puts the step 06 folder first on Python's import search path; `parents[1]` is `09_api_development`. Then `from config import get_settings`. `# noqa: E402` silences the linter about an import not at the top. Note: importing `config` also runs `load_dotenv()` there.
- **Why it is here:** Without the path change `import config` would fail with ModuleNotFoundError.

### 4. Experiment 1: os.environ is a dict of strings  (`main.py` lines 24–27)
```python
if __name__ == "__main__":
    # 1. os.environ behaves like a dict of strings
    os.environ["DEMO_NAME"] = "bob"                               # set (only for THIS process and its children)
    print("1. os.environ['DEMO_NAME'] =", os.environ["DEMO_NAME"])
```
- **In simple words:** Set a variable and read it back, like a notebook of name and value pairs.
- **Exactly:** `os.environ["DEMO_NAME"] = "bob"` affects only this process and children it starts. Real output: `1. os.environ['DEMO_NAME'] = bob`.
- **Why it is here:** Shows environment variables are just key/value strings the OS gives each process.

### 5. Experiment 2: missing variable, getenv vs KeyError  (`main.py` lines 29–34)
```python
    # 2. os.environ[...] raises KeyError if missing; os.getenv returns a default instead
    print("2. getenv with default     =", os.getenv("DOES_NOT_EXIST", "fallback"))
    try:
        os.environ["DOES_NOT_EXIST"]                              # BREAK IT: read the KeyError traceback
    except KeyError as err:
        print("   os.environ[...] raised KeyError:", err)
```
- **In simple words:** Asking for something that is not there either fails loudly or gives a default, depending on how you ask.
- **Exactly:** `os.getenv(name, default)` returns the default (`fallback`); `os.environ[name]` raises `KeyError: 'DOES_NOT_EXIST'`. Real output: `2. getenv with default     = fallback` then `   os.environ[...] raised KeyError: 'DOES_NOT_EXIST'`.
- **Why it is here:** Use `[]` for required settings (fail fast), `getenv` for optional ones.

### 6. Experiment 3: everything is a string  (`main.py` lines 36–42)
```python
    # 3. Everything is a string -> convert explicitly
    os.environ["MAX_TEXT_LENGTH"] = "42"
    print("3. type =", type(os.environ["MAX_TEXT_LENGTH"]).__name__,
          "| get_settings().max_text_length =", get_settings().max_text_length,
          "(", type(get_settings().max_text_length).__name__, ")")
    os.environ["ENABLE_GENERATE"] = "false"                       # classic bug: bool("false") would be True!
    print("   bool('false') =", bool("false"), "-> that is why config.py compares the string to '1'")
```
- **In simple words:** Environment values are always text, so numbers and yes/no must be converted on purpose.
- **Exactly:** `type(os.environ["MAX_TEXT_LENGTH"])` is `str`; `get_settings().max_text_length` is `42` as `int` because config calls `int()`. `bool("false")` is `True` because any non-empty string is truthy. Real output: `3. type = str | get_settings().max_text_length = 42 ( int )` and `bool('false') = True`.
- **Why it is here:** The classic bug: setting `ENABLE_GENERATE=false` and getting it enabled. That is why config compares to `"1"`.

### 7. Experiment 4: read a .env file into a dict  (`main.py` lines 44–48)
```python
    # 4. .env file: KEY=value lines, loaded into os.environ
    with tempfile.TemporaryDirectory() as tmp:                    # a folder that deletes itself afterwards
        env_file = Path(tmp) / ".env"
        env_file.write_text("MODEL_VERSION=from-dotenv\nMAX_TEXT_LENGTH=999\n")
        print("4. dotenv_values       =", dict(dotenv_values(env_file)))
```
- **In simple words:** Make a throw-away `.env` file and peek inside without changing anything.
- **Exactly:** `tempfile.TemporaryDirectory()` is deleted on exit. `write_text` creates two `KEY=value` lines. `dotenv_values(env_file)` returns `{'MODEL_VERSION': 'from-dotenv', 'MAX_TEXT_LENGTH': '999'}` (values are strings).
- **Why it is here:** Lets you inspect config without side effects.

### 8. Experiment 5: precedence, shell beats .env  (`main.py` lines 50–55)
```python
        # 5. Precedence: a real environment variable beats .env unless override=True
        os.environ["MODEL_VERSION"] = "from-shell"
        load_dotenv(env_file)                                     # override=False (default)
        print("5. override=False       -> MODEL_VERSION =", os.environ["MODEL_VERSION"])
        load_dotenv(env_file, override=True)
        print("   override=True        -> MODEL_VERSION =", os.environ["MODEL_VERSION"])
```
- **In simple words:** Decide who wins when the shell and the .env file disagree.
- **Exactly:** With `override=False` (default) `MODEL_VERSION` stays `from-shell`; with `override=True` it becomes `from-dotenv`. Real output confirms both lines.
- **Why it is here:** Production must be able to override a developer `.env` without editing files.

### 9. Experiment 6: settings read at call time  (`main.py` lines 57–61)
```python
    # 6. Settings are read when get_settings() is CALLED, so changing the env changes behaviour
    os.environ["MODEL_PATH"] = ""                                 # empty string -> treated as "not set"
    print("6. model_path =", get_settings().model_path, "| version =", get_settings().model_version)
    # TRY IT: from a shell run  MODEL_VERSION=9 python main.py  and see which lines change (hint: line 5/6 are overridden inside the script).
    # TRY IT: never print secrets (API keys) -- practise printing only the first 4 characters.
```
- **In simple words:** Change the environment and the settings object changes too, because it reads fresh each call.
- **Exactly:** `MODEL_PATH=""` is treated as unset by `os.getenv(...) or None`, so `model_path` is `None`. Real output: `6. model_path = None | version = from-dotenv` (the version stayed from experiment 5's override).
- **Why it is here:** If config were read at import time, tests and demos could not change it.


## Syntax explained
| Syntax | Meaning | Example |
|---|---|---|
| `os.environ["X"]` | Dict-like access, **KeyError if missing** | — |
| `os.getenv("X", "default")` | Safe access with default | — |
| `int(os.getenv("N", "5"))` | Env values are strings → convert | — |
| `x or None` | Empty string → `None` | `"" or None` → `None` |
| `with tempfile.TemporaryDirectory() as tmp:` | Temporary folder, auto-deleted | — |
| `Path(tmp) / ".env"` | `/` joins paths | — |
| `# noqa: E402` | Tell the linter "I know the import is late on purpose" | — |

## Libraries & functions
| Name | Kind | Does |
|---|---|---|
| `os.environ`, `os.getenv` | stdlib | Read process environment |
| `dotenv.load_dotenv(path, override=False)` | 3rd-party | `.env` lines → `os.environ`; real env wins unless `override=True` |
| `dotenv.dotenv_values(path)` | 3rd-party | Read `.env` into a dict **without** touching `os.environ` |
| `dataclasses.dataclass(frozen=True)` | stdlib | Immutable settings object |
| (optional) `pydantic-settings` `BaseSettings` | 3rd-party | Typed settings with validation — same idea, more automatic; `pip install pydantic-settings` |

## Rules worth memorising
1. **Precedence:** command-line env var > `.env` > default in code.
2. **Secrets (API keys) never go in git.** Commit `.env.example`, ignore `.env`. In Docker pass `-e KEY=...` or `--env-file`.
3. **Read config at the edge** (one `config.py`), not scattered `os.getenv` calls.
4. **Fail fast:** if a required variable is missing, crash at startup with a clear message.
5. Don't print secrets; log only a masked prefix.

Where it connects: Docker (module 11) sets `ENV`/`-e` for exactly these names; the capstone passes `MODEL_PATH` into the container; module 08 reads `ANTHROPIC_API_KEY` the same way.

## Execution trace
Experiment 5: shell variable `MODEL_VERSION=from-shell` exists → `load_dotenv(file)` sees the key already set → leaves it → prints `from-shell`. With `override=True` the file value replaces it.

## Experiments
1. `MODEL_VERSION=9 python ../06_model_api_flask_vs_fastapi/main.py` → responses say 9.
2. Create `06_model_api_flask_vs_fastapi/.env` from `.env.example`, change a value, run again, then delete it.
3. Set `MAX_TEXT_LENGTH=abc` → `ValueError: invalid literal for int()` at startup — fail fast; improve the error message yourself.

## Debugging practice
"My `.env` change has no effect" → checklist: (a) is the file in the working directory or a parent? (b) is the shell exporting the same name (it wins)? (c) did you restart the server (env is read at start)? (d) typo in the variable name? (print `os.environ.get(...)`).

## Rebuild checklist
- [ ] Read an int and a bool from the environment safely.
- [ ] Explain precedence of env vs `.env`.

## The new problem (cliffhanger)
Now we can change behaviour without touching code, which is great until a change quietly breaks the API. We need automatic checks. Next: [09_testing_and_docs](../09_testing_and_docs/).

## Say it aloud
> **Say it aloud (interview box):** Configuration lives in environment variables so one build works everywhere and secrets stay out of the repo. I read them in a single settings object, convert types explicitly because everything is a string, and fail fast when something required is missing. The catch is that env vars are invisible and untyped, so I document them in `.env.example` and validate them.
