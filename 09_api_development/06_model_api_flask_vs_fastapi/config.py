"""
config.py -- ALL settings come from environment variables (12-factor style).

Why: the same code/Docker image must run on your laptop, in CI and in production with
different settings, WITHOUT editing code.  See step 08 for experiments.
"""

import os
from dataclasses import dataclass  # a decorator that writes __init__/__repr__ for a plain data holder

from dotenv import load_dotenv     # third-party: python-dotenv. Reads KEY=value lines from a .env file

# Loads .env (if one exists in this folder or a parent) into os.environ.
# Real environment variables WIN over .env by default (override=False).
load_dotenv()


@dataclass(frozen=True)            # frozen=True -> attributes cannot be changed after creation
class Settings:
    model_path: str | None         # path to a joblib model file; None -> use the built-in fallback model
    model_version: str             # reported in every response so you know WHICH model answered
    max_text_length: int           # reject giant inputs (protects latency and memory)
    enable_generate: bool          # the /generate endpoint downloads ~330 MB, so it is opt-in


def get_settings() -> Settings:
    """Read the environment NOW (not at import time) so tests can change it with monkeypatch."""
    return Settings(
        model_path=os.getenv("MODEL_PATH") or None,          # "" (empty) also becomes None
        model_version=os.getenv("MODEL_VERSION", "0.1.0"),   # getenv(name, default)
        max_text_length=int(os.getenv("MAX_TEXT_LENGTH", "500")),  # env vars are ALWAYS strings -> convert
        enable_generate=os.getenv("ENABLE_GENERATE", "0") == "1",
    )
