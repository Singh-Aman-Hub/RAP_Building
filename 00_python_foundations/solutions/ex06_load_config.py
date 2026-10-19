"""Ex 06: load JSON config with defaults, surviving a missing or broken file."""
import json
from pathlib import Path

DEFAULTS = {"model": "distilgpt2", "max_new_tokens": 20}


def load_config(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        print("no config file, using defaults")
        return dict(DEFAULTS)                      # copy so callers can't mutate DEFAULTS
    except json.JSONDecodeError as err:
        print("config is not valid JSON:", err.msg)
        return dict(DEFAULTS)
    return {**DEFAULTS, **data}                    # ** unpacks dicts; later keys override earlier ones


HERE = Path(__file__).resolve().parent
(HERE / "_bad.json").write_text("{'x': 1}")
print(load_config(HERE / "missing.json"))
print(load_config(HERE / "_bad.json"))
(HERE / "_good.json").write_text('{"max_new_tokens": 5}')
print(load_config(HERE / "_good.json"))
