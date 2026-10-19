"""
GOAL:  Read configuration from environment variables safely (defaults, type conversion, required values).

RUN:   cd 01_debugging_and_environment/07_env_vars_and_config && python main.py
       with overrides:  PORT=9000 MODEL_NAME=gpt2 python main.py
       Crashing version (EXPECTED):  python broken_env.py

EXPECTED OUTPUT: the resolved settings, with a clear error for bad values.

NEEDS: nothing (standard library only).
"""

import os

# os.environ behaves like a dict of strings. EVERYTHING in it is a string -- even PORT=8000.
print("HOME is set?", "HOME" in os.environ)

# .get(name, default): never raises, returns default if missing
model_name = os.environ.get("MODEL_NAME", "distilgpt2")
print("MODEL_NAME:", model_name)

# Convert types yourself and fail with a HELPFUL message
def get_int(name, default):
    raw = os.environ.get(name, str(default))
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"Config error: {name} must be an integer, got {raw!r}")


def get_required(name):
    value = os.environ.get(name)
    if not value:                          # missing or empty string
        raise SystemExit(f"Config error: required environment variable {name} is not set")
    return value


port = get_int("PORT", 8000)
print("PORT:", port, type(port).__name__)
debug = os.environ.get("DEBUG", "false").lower() in {"1", "true", "yes"}   # booleans need parsing too
print("DEBUG:", debug)

# Secrets (API keys) belong in env vars or a .env file, NEVER in code or git.
key = os.environ.get("ANTHROPIC_API_KEY")
print("API key configured?", bool(key))   # print True/False, never the key itself

# A .env file is just KEY=VALUE lines; python-dotenv loads them into os.environ.
try:
    from dotenv import load_dotenv
    from pathlib import Path
    loaded = load_dotenv(Path(__file__).resolve().parent / ".env")   # False if file missing
    print(".env loaded?", loaded, "(copy .env.example to .env to try)")
except ImportError:
    print("python-dotenv not installed (optional)")

# TRY IT: run `PORT=9000 python main.py` -- note the shell syntax VAR=value command.
# BREAK IT: run `PORT=abc python main.py` and read the message.
