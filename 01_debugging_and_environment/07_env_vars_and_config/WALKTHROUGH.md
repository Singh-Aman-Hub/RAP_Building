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
