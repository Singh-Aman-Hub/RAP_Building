"""
GOAL:   Keep every training/generation parameter in ONE config file, load it with
        explained, validated code, and see what each parameter means.

RUN:    from this folder:  python main.py
        from repo root:    .venv/bin/python 05_llm_parameters/01_config_file/main.py
EXPECTED OUTPUT (short): the config printed section by section with a one-line meaning for
        every parameter, then a clear error message when a bad value is supplied.
NEEDS:  Python standard library only (json, pathlib). No internet.

Why a config file?  Experiments (and production services in module 09/10) change
parameters constantly.  Hard-coding them in scripts makes runs impossible to compare
or reproduce.  config.json is the single source of truth; later steps in this module
(03, 06, 07) and module 06 load it via config_loader.py (`load_config`).
"""

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))     # make config_loader.py importable from any cwd
from config_loader import MEANING, load_config, generation_kwargs   # our helper module in this folder


if __name__ == "__main__":                       # runs only when executed directly, not when imported
    cfg = load_config()
    print("model:", cfg["model_name"])
    for section, params in MEANING.items():
        print(f"\n[{section}]")
        for name, meaning in params.items():
            print(f"  {name:20} = {cfg[section][name]!s:8}  # {meaning}")

    print("\nkwargs that would be passed to model.generate(**kwargs):")
    print(" ", generation_kwargs(cfg))

    # Demonstrate validation: copy the config, break one value, and catch the error.
    bad = json.loads(json.dumps(cfg))            # cheap deep copy via JSON round-trip
    bad["generation"]["top_p"] = 1.5
    bad_path = Path(__file__).parent / "_bad_config.json"
    bad_path.write_text(json.dumps(bad))
    try:                                          # try/except = handle an expected error gracefully
        load_config(bad_path)
    except ValueError as err:
        print("\ncaught expected error ->", err)
    finally:
        bad_path.unlink()                         # delete the temp file whatever happened

# TRY IT: set "temperature": 0 in config.json and run - read the ValueError you wrote.
# TRY IT: add a new parameter "seed": 42 to the JSON and make generation_kwargs ignore it (hint: HF doesn't know 'seed').
# BREAK IT: remove a comma in config.json -> json.JSONDecodeError tells you line and column.
# BREAK IT: rename "generation" to "gen" -> KeyError: 'generation'. How would you give a friendlier message?
