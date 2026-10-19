# Step 08 — Environment variables and configuration

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

## Say it aloud
"Configuration lives in environment variables so one build artifact works everywhere; secrets stay out of the repo. I read them in one settings object, convert types explicitly because everything is a string, and fail fast if something required is missing. Trade-off: env vars are untyped and invisible — so I document them in `.env.example` and validate them."
