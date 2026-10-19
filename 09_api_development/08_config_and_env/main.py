"""
GOAL:   Understand environment variables and configuration: how a program reads settings
        from outside its code, the .env file convention, precedence rules, and the classic
        "everything is a string" bug.

RUN:    cd 09_api_development/08_config_and_env && python main.py

EXPECTED OUTPUT (shape): a numbered list of small experiments (1..6) with their results.

NEEDS:  python-dotenv (third-party). os / tempfile are built-in.
"""

import os
import sys
import tempfile
from pathlib import Path

from dotenv import dotenv_values, load_dotenv    # dotenv_values: read a .env file into a dict WITHOUT touching os.environ

# Reuse the real settings object from step 06 (shows config is not just a toy).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "06_model_api_flask_vs_fastapi"))
from config import get_settings  # noqa: E402  (import after sys.path change is intentional)

if __name__ == "__main__":
    # 1. os.environ behaves like a dict of strings
    os.environ["DEMO_NAME"] = "bob"                               # set (only for THIS process and its children)
    print("1. os.environ['DEMO_NAME'] =", os.environ["DEMO_NAME"])

    # 2. os.environ[...] raises KeyError if missing; os.getenv returns a default instead
    print("2. getenv with default     =", os.getenv("DOES_NOT_EXIST", "fallback"))
    try:
        os.environ["DOES_NOT_EXIST"]                              # BREAK IT: read the KeyError traceback
    except KeyError as err:
        print("   os.environ[...] raised KeyError:", err)

    # 3. Everything is a string -> convert explicitly
    os.environ["MAX_TEXT_LENGTH"] = "42"
    print("3. type =", type(os.environ["MAX_TEXT_LENGTH"]).__name__,
          "| get_settings().max_text_length =", get_settings().max_text_length,
          "(", type(get_settings().max_text_length).__name__, ")")
    os.environ["ENABLE_GENERATE"] = "false"                       # classic bug: bool("false") would be True!
    print("   bool('false') =", bool("false"), "-> that is why config.py compares the string to '1'")

    # 4. .env file: KEY=value lines, loaded into os.environ
    with tempfile.TemporaryDirectory() as tmp:                    # a folder that deletes itself afterwards
        env_file = Path(tmp) / ".env"
        env_file.write_text("MODEL_VERSION=from-dotenv\nMAX_TEXT_LENGTH=999\n")
        print("4. dotenv_values       =", dict(dotenv_values(env_file)))

        # 5. Precedence: a real environment variable beats .env unless override=True
        os.environ["MODEL_VERSION"] = "from-shell"
        load_dotenv(env_file)                                     # override=False (default)
        print("5. override=False       -> MODEL_VERSION =", os.environ["MODEL_VERSION"])
        load_dotenv(env_file, override=True)
        print("   override=True        -> MODEL_VERSION =", os.environ["MODEL_VERSION"])

    # 6. Settings are read when get_settings() is CALLED, so changing the env changes behaviour
    os.environ["MODEL_PATH"] = ""                                 # empty string -> treated as "not set"
    print("6. model_path =", get_settings().model_path, "| version =", get_settings().model_version)
    # TRY IT: from a shell run  MODEL_VERSION=9 python main.py  and see which lines change (hint: line 5/6 are overridden inside the script).
    # TRY IT: never print secrets (API keys) -- practise printing only the first 4 characters.
