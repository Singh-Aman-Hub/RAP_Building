# Terms — Step 08: Config and environment variables

Folder: `09_api_development/08_config_and_env` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [09 API development](../README.md)

**Terms in this folder:** [Environment variable](#environment-variable-concept) · [Configuration outside the code (12-factor style)](#configuration-outside-the-code-12-factor-style-concept) · [os.environ](#osenviron-standard-library) · [os.getenv with default](#osgetenv-with-default-standard-library) · [KeyError](#keyerror-python-built-in) · [Everything is a string](#everything-is-a-string-concept) · [int() conversion of env values](#int-conversion-of-env-values-python-built-in) · [type(...).__name__](#type__name__-python-built-in) · [.env file](#env-file-concept) · [load_dotenv](#load_dotenv-third-party-python-dotenv) · [dotenv_values](#dotenv_values-third-party-python-dotenv) · [Precedence of configuration sources](#precedence-of-configuration-sources-concept) · [tempfile.TemporaryDirectory](#tempfiletemporarydirectory-standard-library) · [Path / operator and write_text](#path-operator-and-write_text-standard-library) · [sys.path.insert](#syspathinsert-standard-library) · [# noqa: E402](#noqa-e402-syntax) · [Settings object (dataclass) read at call time](#settings-object-dataclass-read-at-call-time-concept) · [Empty string means not set](#empty-string-means-not-set-concept) · [Secrets handling](#secrets-handling-concept) · [Fail fast](#fail-fast-concept) · [pydantic-settings BaseSettings](#pydantic-settings-basesettings-third-party-pydantic-settings) · [Docker ENV / -e](#docker-env--e-command) · [Debugging "my .env change has no effect"](#debugging-my-env-change-has-no-effect-concept)

---

## Environment variable  `(concept)`
**1. Definition:** A named string value stored in a process's environment, inherited by child processes, and used to pass configuration from outside the program.
**2. In plain words:** Instead of writing a model path into the code, you set `MODEL_PATH` outside and the program reads it. Change the variable and behaviour changes with no code edit. Docker, cloud platforms and CI all configure apps this way.
**3. Example to quote:**
> "`MODEL_VERSION=9 python main.py` changes the reported version without touching any file."

**Seen in:** whole step

## Configuration outside the code (12-factor style)  `(concept)`
**1. Definition:** The practice of keeping settings that differ between environments (paths, limits, secrets) out of the source code so one build runs anywhere.
**2. In plain words:** The same Docker image should run on a laptop, in testing and in production. Only the environment variables differ, which also keeps secrets out of the repository.
**3. Example to quote:**
> "My image is identical everywhere; production just sets a different `MODEL_PATH` and an API key through the environment."

**Seen in:** `WALKTHROUGH.md` goal and rules

## os.environ  `(standard library)`
**1. Definition:** A dict-like mapping of the current process's environment variables, with string keys and string values; assignments affect this process and its children only.
**2. In plain words:** You can read and set variables like a dictionary. Reading a missing key with `[...]` raises `KeyError`. Changes do not leak to your shell.
**3. Example to quote:**
> "`os.environ['DEMO_NAME'] = 'bob'` sets it for this script only; my terminal doesn't see it."

```python
import os
os.environ["X"] = "1"
print(os.environ["X"])   # 1
```

**Seen in:** experiment 1

## os.getenv with default  `(standard library)`
**1. Definition:** `os.getenv(name, default)` returns the variable's value, or the default if it is not set.
**2. In plain words:** It is the safe lookup that never raises. Use `os.environ[...]` instead when the variable is required and you want a loud failure.
**3. Example to quote:**
> "`os.getenv('DOES_NOT_EXIST', 'fallback')` returns 'fallback' instead of crashing."

```python
import os
print(os.getenv("NOPE", "fallback"))   # fallback
```

**Seen in:** experiment 2

## KeyError  `(Python built-in)`
**1. Definition:** The exception raised when a mapping lookup uses a key that does not exist.
**2. In plain words:** `os.environ['DOES_NOT_EXIST']` raises `KeyError: 'DOES_NOT_EXIST'`. That is useful for required settings: the app fails fast at startup with a clear message.
**3. Example to quote:**
> "If a required variable is missing, `os.environ['KEY']` raises `KeyError`, which tells me exactly which one to set."

**Seen in:** experiment 2

## Everything is a string  `(concept)`
**1. Definition:** All environment variable values are strings, so numbers and booleans must be explicitly converted.
**2. In plain words:** `MAX_TEXT_LENGTH=42` arrives as `"42"` and must go through `int(...)`. Booleans are riskier: `bool("false")` is `True` because any non-empty string is truthy, so compare the string, for example `== "1"`.
**3. Example to quote:**
> "`bool('false')` is True! That's why my config checks `os.getenv('ENABLE_GENERATE') == '1'` instead."

```python
print(bool("false"))   # True
```

**Seen in:** experiment 3

## int() conversion of env values  `(Python built-in)`
**1. Definition:** `int(text)` converts a numeric string to an integer and raises `ValueError` for invalid text.
**2. In plain words:** A typo such as `MAX_TEXT_LENGTH=abc` raises `ValueError: invalid literal for int()` at startup, which is good: fail fast with a message you can improve.
**3. Example to quote:**
> "`get_settings().max_text_length` is an int 42 because the code wraps `os.getenv` in `int()`."

**Seen in:** experiment 3

## type(...).__name__  `(Python built-in)`
**1. Definition:** `type(x)` returns the class of `x`; `.__name__` gives that class's name as a string such as `'str'`.
**2. In plain words:** It is a quick way to print the type so you can see that the environment value is a string while the setting is an int.
**3. Example to quote:**
> "`type(os.environ['MAX_TEXT_LENGTH']).__name__` prints `str`, while the setting prints `int`."

```python
print(type(5).__name__)   # int
```

**Seen in:** experiment 3

## .env file  `(concept)`
**1. Definition:** A text file of `KEY=value` lines holding local configuration, loaded into the environment by tools such as python-dotenv.
**2. In plain words:** It is a convenient place for local settings and secrets. Never commit the real `.env`; commit a `.env.example` with placeholder values instead.
**3. Example to quote:**
> "I keep `MODEL_VERSION=from-dotenv` in `.env` for development and list the same names in `.env.example` for teammates."

**Seen in:** experiment 4

## load_dotenv  `(third-party: python-dotenv)`
**1. Definition:** `load_dotenv(path, override=False)` reads a `.env` file into `os.environ`; with `override=False` it does not replace variables already set.
**2. In plain words:** Real environment variables win by default, so production settings cannot be accidentally overridden by a stray `.env`. Use `override=True` only when you want the file to win.
**3. Example to quote:**
> "With the shell already setting `MODEL_VERSION=from-shell`, `load_dotenv(file)` leaves it; with `override=True` it becomes `from-dotenv`."

**Seen in:** experiment 5

## dotenv_values  `(third-party: python-dotenv)`
**1. Definition:** `dotenv_values(path)` parses a `.env` file into a dict without modifying `os.environ`.
**2. In plain words:** It lets you inspect what the file says without side effects, which is handy in tests and debugging.
**3. Example to quote:**
> "`dict(dotenv_values(env_file))` printed `{'MODEL_VERSION': 'from-dotenv', 'MAX_TEXT_LENGTH': '999'}` without changing my environment."

**Seen in:** experiment 4

## Precedence of configuration sources  `(concept)`
**1. Definition:** The order in which sources override each other: command-line or shell environment first, then `.env`, then defaults in code.
**2. In plain words:** Knowing the order explains "why didn't my `.env` change work?": a shell variable of the same name wins. It is also why production can override a developer `.env`.
**3. Example to quote:**
> "My `.env` change had no effect because the shell was exporting the same variable, and the shell wins."

**Seen in:** experiment 5, `WALKTHROUGH.md` rules

## tempfile.TemporaryDirectory  `(standard library)`
**1. Definition:** A context manager that creates a temporary folder and deletes it with its contents when the block ends.
**2. In plain words:** The demo writes a throwaway `.env` there so it does not leave files around or touch your real configuration.
**3. Example to quote:**
> "`with tempfile.TemporaryDirectory() as tmp:` gives me a scratch folder that disappears afterwards."

**Seen in:** experiment 4

## Path / operator and write_text  `(standard library)`
**1. Definition:** `Path(tmp) / ".env"` joins path parts with the `/` operator, and `Path.write_text(s)` writes a string to that file.
**2. In plain words:** It is the readable way to build paths and write small files, with no manual string concatenation or `open` call.
**3. Example to quote:**
> "`env_file.write_text('MODEL_VERSION=from-dotenv\\n...')` creates the sample `.env` in one line."

```python
from pathlib import Path
print(Path("a") / "b.txt")   # a/b.txt
```

**Seen in:** experiment 4

## sys.path.insert  `(standard library)`
**1. Definition:** `sys.path` is the list of folders Python searches for imports; `sys.path.insert(0, folder)` adds a folder at the front so modules inside it become importable.
**2. In plain words:** The demo reuses step 06's `config.py` from a sibling folder. Folder names starting with digits cannot be imported as packages, so adding the folder to the path is the practical workaround.
**3. Example to quote:**
> "I insert step 06's folder into `sys.path`, so `from config import get_settings` finds the real settings module."

**Seen in:** top of `main.py`

## # noqa: E402  `(syntax)`
**1. Definition:** A comment that tells linters such as flake8 to ignore the rule E402 ("module level import not at top of file") for that line.
**2. In plain words:** The import comes after `sys.path.insert` on purpose, so the linter warning would be a false alarm; the comment records that it is intentional.
**3. Example to quote:**
> "`from config import get_settings  # noqa: E402` says the late import is deliberate, because the path must be set first."

**Seen in:** top of `main.py`

## Settings object (dataclass) read at call time  `(concept)`
**1. Definition:** A single frozen `Settings` object produced by `get_settings()`, which reads the environment each time it is called.
**2. In plain words:** One place defines every setting, its type and default. Because it reads at call time, changing `os.environ` in the demo changes the result immediately, which is also how tests override config.
**3. Example to quote:**
> "After setting `MODEL_PATH=''` and re-calling `get_settings()`, `model_path` is `None`, because an empty string means 'not set'."

**Seen in:** experiments 3 and 6

## Empty string means not set  `(concept)`
**1. Definition:** A convention where an empty value is treated the same as an absent variable, implemented with `os.getenv(name) or None`.
**2. In plain words:** An accidentally empty `MODEL_PATH=""` would otherwise try to load a file called "". Converting to `None` makes the fallback model kick in.
**3. Example to quote:**
> "`MODEL_PATH=''` gives `model_path = None`, so the API uses the fallback model instead of crashing."

**Seen in:** experiment 6

## Secrets handling  `(concept)`
**1. Definition:** Keeping credentials such as API keys out of source control and logs, passing them via environment variables or a secrets manager.
**2. In plain words:** A leaked key can be used by anyone. Commit `.env.example` but git-ignore `.env`, in Docker pass `-e KEY=...` or `--env-file`, and when logging show only a masked prefix like the first four characters.
**3. Example to quote:**
> "I never print the full API key; I log just `key[:4] + '...'` so I can tell which key is in use."

**Seen in:** `TRY IT` comment, `WALKTHROUGH.md` rules

## Fail fast  `(concept)`
**1. Definition:** Crashing at startup with a clear error when required configuration is missing or invalid, instead of failing later during a request.
**2. In plain words:** It is better to learn at deploy time that `MODEL_PATH` is wrong than for the first user to get an error. Startup crashes are visible and easy to fix.
**3. Example to quote:**
> "`MAX_TEXT_LENGTH=abc` crashes immediately with `ValueError` when the settings load, not on the first request."

**Seen in:** `WALKTHROUGH.md` rules and experiment 3

## pydantic-settings BaseSettings  `(third-party: pydantic-settings)`
**1. Definition:** An optional Pydantic extension whose `BaseSettings` class reads typed, validated settings automatically from environment variables and `.env` files.
**2. In plain words:** It does what the hand-written `config.py` does (read, convert, validate), with less code. It must be installed separately (`pip install pydantic-settings`).
**3. Example to quote:**
> "In a larger project I'd replace my `Settings` dataclass with `BaseSettings`, which converts `MAX_TEXT_LENGTH` to int automatically."

**Seen in:** `WALKTHROUGH.md` libraries table

## Docker ENV / -e  `(command)`
**1. Definition:** Dockerfile `ENV NAME=value` sets a default environment variable in an image; `docker run -e NAME=value` or `--env-file` sets or overrides it at run time.
**2. In plain words:** It is the Docker counterpart of what this step does by hand. The same names (`MODEL_PATH`, `MODEL_VERSION`) are used inside the container (module 11).
**3. Example to quote:**
> "`docker run -e MODEL_PATH=/models/model.joblib ...` passes the same variable my app reads with `os.getenv`."

```bash
docker run -e MODEL_VERSION=2.0.0 myapi
```

**Seen in:** `WALKTHROUGH.md` "Where it connects"

## Debugging "my .env change has no effect"  `(concept)`
**1. Definition:** A checklist: is the file in the working directory or a parent, does the shell export the same name (it wins), was the server restarted, and is the variable name spelled correctly.
**2. In plain words:** Environment is read at process start, and shell variables override `.env`, so most "nothing changed" reports are one of these four. Printing `os.environ.get(...)` quickly confirms what the process sees.
**3. Example to quote:**
> "My edit did nothing because the server was still running; the environment is read at start, so I restarted it."

**Seen in:** `WALKTHROUGH.md` debugging practice
