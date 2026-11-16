# Step 01 - A config file for every parameter

## The story so far
Chapter 14, Rolling the Dice. The end of [the life-cycle step](../../04_llm_fundamentals/09_pretrain_vs_finetune_vs_inference/WALKTHROUGH.md) left you with a model you can run but not control, shrink or specialise. Control comes first, and it starts with admitting how many dials there are.

## The pain point
Before this step, the generation loop in module 04 had settings scattered as magic numbers: `max_new_tokens=20` here, `lr=5e-5` there. Change one, forget another, and nobody can say which setting produced which output. A typo like `"temprature"` silently does nothing.

## The spark
Put every dial in one `config.json`, load it through one function that checks names and ranges, and describe each dial in a glossary. One file, one source of truth, errors at load time instead of silent surprises.

## Goal
Keep all training and generation parameters in `config.json`, load and validate them with explained code, and learn what each one means.

## Run it
```bash
.venv/bin/python 05_llm_parameters/01_config_file/main.py
```
Files: `config.json` (the data), `config_loader.py` (reusable functions, imported by steps 03/07 and module 06), `main.py` (demo).

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Hyperparameter | setting, knob, config value | Yes | A value you choose before training or generation; not learned from data |
| Parameter (model) | weight | No | Learned numbers inside the model (81.9M for distilgpt2); not what this config holds |
| Generation parameters | decoding parameters, sampling parameters, inference parameters | Yes | temperature, top_k, top_p, max_new_tokens etc. |
| Config file | JSON/YAML settings, `GenerationConfig` (HF), `.env` | Related | The file holds values; `GenerationConfig` is Hugging Face's own saved object for them |
| `max_new_tokens` vs `max_length` | output cap vs total cap | No | `max_new_tokens` counts only generated tokens; `max_length` counts prompt plus output |
| Validation (fail fast) | schema check, sanity check | Yes | Reject nonsense values at load time with a clear message |
| JSON vs YAML vs TOML | config formats | Different formats | JSON is strict and built into Python; YAML allows comments; TOML is common for Python tooling |

## Code, step by step

### 1. The config file: one place for every parameter  (`config.json` lines 1–22)

```json
{
  "model_name": "distilgpt2",

  "training": {
    "learning_rate": 5e-5,
    "batch_size": 4,
    "epochs": 3,
    "optimizer": "adamw",
    "weight_decay": 0.01
  },

  "generation": {
    "do_sample": true,
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.95,
    "max_new_tokens": 40,
    "repetition_penalty": 1.1
  },

  "limits": {
    "context_window": 1024
```

- **In simple words:** A plain text file holding the model name and every training and generation setting.
- **Exactly:** JSON with four top-level keys: `model_name` (`distilgpt2`), `training` (`learning_rate 5e-5`, `batch_size 4`, `epochs 3`, `optimizer adamw`, `weight_decay 0.01`), `generation` (`do_sample true`, `temperature 0.8`, `top_k 50`, `top_p 0.95`, `max_new_tokens 40`, `repetition_penalty 1.1`) and `limits` (`context_window 1024`). JSON booleans are lowercase `true`; no comments or trailing commas allowed.
- **Why it is here:** Changing a setting means editing one file rather than hunting through code; runs can be compared and reproduced.

### 2. main.py docstring: why a config file  (`main.py` lines 1–15)

```python
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
```

- **In simple words:** Explains that experiments change settings constantly, so they live in one file.
- **Exactly:** Docstring only. Later steps (03, 06, 07) and module 06 import the same loader.
- **Why it is here:** A single source of truth prevents two scripts using different values by accident.

### 3. Imports and a path trick  (`main.py` lines 17–22)

```python
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))     # make config_loader.py importable from any cwd
from config_loader import MEANING, load_config, generation_kwargs   # our helper module in this folder
```

- **In simple words:** Make sure Python can find our helper file next to this script, wherever you run it from.
- **Exactly:** `sys.path.insert(0, str(Path(__file__).parent))` puts this folder first on the import search path. `from config_loader import MEANING, load_config, generation_kwargs` imports the dict of explanations and two functions.
- **Why it is here:** Without it, running from the repo root gives `ModuleNotFoundError`. The helper is not named `main.py` to avoid name collisions between steps.

### 4. Load the config and print each parameter with its meaning  (`main.py` lines 25–31)

```python
if __name__ == "__main__":                       # runs only when executed directly, not when imported
    cfg = load_config()
    print("model:", cfg["model_name"])
    for section, params in MEANING.items():
        print(f"\n[{section}]")
        for name, meaning in params.items():
            print(f"  {name:20} = {cfg[section][name]!s:8}  # {meaning}")
```

- **In simple words:** Read the file and print a tidy table: name, value, and one-line explanation.
- **Exactly:** `if __name__ == "__main__"` runs only on direct execution. `load_config()` returns a dict. The loops walk `MEANING` (section -> name -> text) and look up `cfg[section][name]`. `{name:20}` pads to 20 characters; `{value!s:8}` converts with `str` and pads to 8. Real line: `learning_rate = 5e-05 # step size of each weight update...`. Python prints `5e-5` as `5e-05`, and `true` as `True`.
- **Why it is here:** The meaning text next to the value turns a config into documentation.

### 5. Show the kwargs for model.generate  (`main.py` lines 33–34)

```python
    print("\nkwargs that would be passed to model.generate(**kwargs):")
    print(" ", generation_kwargs(cfg))
```

- **In simple words:** Print exactly what would be handed to Hugging Face's `generate`.
- **Exactly:** `generation_kwargs(cfg)` returns a copy of `cfg["generation"]`: `{'do_sample': True, 'temperature': 0.8, 'top_k': 50, 'top_p': 0.95, 'max_new_tokens': 40, 'repetition_penalty': 1.1}`. In step 03 this is used as `model.generate(ids, **kwargs)`.
- **Why it is here:** Keeping generation keys separate from training keys means they can be passed straight to the library.

### 6. Demonstrate validation with a deliberately bad copy  (`main.py` lines 36–46)

```python
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
```

- **In simple words:** Make a broken copy of the config (top_p = 1.5), try to load it, and show the friendly error.
- **Exactly:** `json.loads(json.dumps(cfg))` is a quick deep copy. The bad file `_bad_config.json` is written next to the script. `try/except ValueError as err` catches our error: real output `caught expected error -> generation.top_p must be in (0, 1], got 1.5`. `finally: bad_path.unlink()` deletes the temp file whether or not an error happened.
- **Why it is here:** Shows fail-fast checks and clean-up of temporary files with `finally`.

### 7. Experiments and break-it comments  (`main.py` lines 48–51)

```python
# TRY IT: set "temperature": 0 in config.json and run - read the ValueError you wrote.
# TRY IT: add a new parameter "seed": 42 to the JSON and make generation_kwargs ignore it (hint: HF doesn't know 'seed').
# BREAK IT: remove a comma in config.json -> json.JSONDecodeError tells you line and column.
# BREAK IT: rename "generation" to "gen" -> KeyError: 'generation'. How would you give a friendlier message?
```

- **In simple words:** Suggested edits to config.json to see errors.
- **Exactly:** `temperature: 0` triggers our `ValueError`. A missing comma gives `json.JSONDecodeError` with line and column. Renaming `generation` raises `KeyError: 'generation'`.
- **Why it is here:** Reading these three kinds of error is part of working with config files.

### 8. config_loader.py docstring  (`config_loader.py` lines 1–5)

```python
"""
config_loader.py -- load + validate config.json.  Imported by main.py here and by steps 03, 06, 07.
(Kept separate from main.py because every step's script is called main.py: importing "main" from
another folder would pick the wrong file - a classic name-collision bug, see 01_debugging module.)
"""
```

- **In simple words:** Says this helper is shared by several steps and why it is not called main.py.
- **Exactly:** Docstring only. Every step's script is named `main.py`; importing `main` from another folder would pick the wrong file.
- **Why it is here:** Documents a name-collision bug that is easy to hit in multi-folder projects.

### 9. Imports and the config path  (`config_loader.py` lines 7–10)

```python
import json                       # standard library: read/write JSON text <-> Python dict/list
from pathlib import Path          # standard library: object-oriented file paths

CONFIG_PATH = Path(__file__).parent / "config.json"   # __file__ = this script; works from any cwd
```

- **In simple words:** Tools for JSON and file paths, and the location of config.json next to this file.
- **Exactly:** `Path(__file__).parent / "config.json"` is an absolute-ish path computed from this file's location, so it works from any working directory.
- **Why it is here:** A relative path like `"config.json"` would break when run from another directory.

### 10. A dictionary explaining each parameter  (`config_loader.py` lines 12–33)

```python
# Plain-English meaning of each parameter (a dict of dicts). Keep this next to the loader so
# the documentation cannot drift away from the code.
MEANING = {
    "training": {
        "learning_rate": "step size of each weight update. Too big = unstable, too small = slow (step 06)",
        "batch_size": "how many examples are processed before ONE weight update (step 06)",
        "epochs": "how many full passes over the whole dataset (step 06)",
        "optimizer": "the algorithm that turns gradients into updates (adamw, sgd) (step 06)",
        "weight_decay": "small penalty on big weights; fights overfitting (step 06)",
    },
    "generation": {
        "do_sample": "False = greedy (always the top token); True = random sampling (step 02)",
        "temperature": "<1 sharper/safer, >1 flatter/more random (step 02)",
        "top_k": "only sample from the k most likely tokens (step 02)",
        "top_p": "only sample from the smallest set of tokens whose probabilities add to p (step 02)",
        "max_new_tokens": "hard cap on how many tokens to produce (step 04)",
        "repetition_penalty": ">1 makes already-used tokens less likely, reduces loops (step 02)",
    },
    "limits": {
        "context_window": "max tokens (prompt + output) the model can attend to (step 04)",
    },
}
```

- **In simple words:** One sentence for each setting, stored next to the code that loads it.
- **Exactly:** `MEANING` is a dict of dicts: section -> parameter -> explanation. It refers to the steps where each parameter is studied.
- **Why it is here:** Docs next to code cannot drift out of date as easily as a separate file.

### 11. load_config: read the file  (`config_loader.py` lines 36–42)

```python
def load_config(path=CONFIG_PATH):
    """Read the JSON file, check values are sensible, and return a plain dict.

    Raises ValueError with a message that says WHICH parameter is wrong and why.
    """
    with open(path, encoding="utf-8") as f:      # 'with' closes the file automatically, even on errors
        cfg = json.load(f)                       # JSON text -> Python dict
```

- **In simple words:** Open config.json and turn its text into a Python dict.
- **Exactly:** `open(path, encoding="utf-8")` with `with` closes the file even on error. `json.load(f)` parses JSON into a dict whose numbers become `int` or `float` and `true` becomes `True`. Raises `json.JSONDecodeError` on bad syntax.
- **Why it is here:** Always specify the encoding so behaviour does not depend on the operating system.

### 12. load_config: validate values  (`config_loader.py` lines 44–56)

```python
    g, t, lim = cfg["generation"], cfg["training"], cfg["limits"]   # shortcuts to the sub-dicts
    # Validation: catch silly values EARLY instead of getting confusing behaviour later.
    if g["temperature"] <= 0:
        raise ValueError("generation.temperature must be > 0 (use do_sample=false for greedy decoding)")
    if not 0 < g["top_p"] <= 1:
        raise ValueError(f"generation.top_p must be in (0, 1], got {g['top_p']}")
    if g["top_k"] < 0:
        raise ValueError("generation.top_k must be >= 0 (0 disables it)")
    if g["max_new_tokens"] < 1 or g["max_new_tokens"] > lim["context_window"]:
        raise ValueError("generation.max_new_tokens must be between 1 and limits.context_window")
    if t["learning_rate"] <= 0 or t["batch_size"] < 1 or t["epochs"] < 1:
        raise ValueError("training.learning_rate > 0, batch_size >= 1 and epochs >= 1 are required")
    return cfg
```

- **In simple words:** Reject impossible settings early, with a message that names the exact parameter.
- **Exactly:** Shortcuts `g, t, lim` point to the sub-dicts. Checks: `temperature > 0` (division by zero otherwise), `0 < top_p <= 1`, `top_k >= 0` (0 disables), `1 <= max_new_tokens <= context_window`, and positive learning rate, batch size and epochs. Each raises `ValueError`. Returns `cfg`.
- **Why it is here:** A silly value fails at load time with a clear message instead of producing mysterious behaviour 10 minutes into a run.

### 13. generation_kwargs: copy the generation section  (`config_loader.py` lines 59–61)

```python
def generation_kwargs(cfg):
    """Return only the keys Hugging Face `model.generate(**kwargs)` understands."""
    return dict(cfg["generation"])               # dict(...) makes a copy so callers can't mutate the config
```

- **In simple words:** Hand back the generation settings in a form that can be passed to `generate`.
- **Exactly:** `dict(cfg["generation"])` makes a shallow copy so a caller who edits the result does not change the config.
- **Why it is here:** Prevents hidden side effects between callers.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| JSON | text format for dicts/lists/numbers/true/false (no comments, no trailing commas) | `{"lr": 5e-5, "do_sample": true}` |
| `with open(path) as f:` | open a file and auto-close it | `with open("a.txt") as f: f.read()` |
| `cfg["generation"]["top_k"]` | nested dict access | - |
| `Path(__file__).parent / "config.json"` | `/` joins path pieces; `__file__` is this script's location | - |
| `raise ValueError("...")` | stop with an error you choose | - |
| `try / except / finally` | handle an error / always clean up | - |
| `if __name__ == "__main__":` | run this block only when executed directly, not when imported | - |
| `dict(d)` | shallow copy of a dict | - |
| `{**a, **b}` | merge two dicts (b wins) | `{**{"x":1}, **{"x":2}}` -> `{"x": 2}` |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `json.load(f)` | stdlib | JSON file -> Python objects | file object | dict/list |
| `json.dumps(obj)` / `json.loads(s)` | stdlib | Python <-> JSON string | `indent` for pretty printing | str / object |
| `pathlib.Path` | stdlib | file paths as objects | - | `Path` |
| `Path.write_text`, `.unlink()` | stdlib | write a file / delete it | - | - |

## Parameter meanings (the glossary you will use all module)
| Parameter | Phase | One-line meaning | Where you experiment |
|---|---|---|---|
| `learning_rate` | training | step size of each weight update | step 06 |
| `batch_size` | training | examples per weight update | step 06 |
| `epochs` | training | passes over the dataset | step 06 |
| `optimizer`, `weight_decay` | training | update rule, regulariser | step 06 |
| `do_sample` | inference | False = greedy, True = random sampling | steps 02-03 |
| `temperature` | inference | sharpness of the distribution | step 02 |
| `top_k`, `top_p` | inference | restrict the candidate set | step 02 |
| `repetition_penalty` | inference | discourage repeats | step 02/03 |
| `max_new_tokens` | inference | output length cap | step 04 |
| `context_window` | model limit | max prompt+output tokens | step 04 |

## Execution trace
```
model: distilgpt2
[generation]
  temperature          = 0.8      # <1 sharper/safer, >1 flatter/more random (step 02)
  top_k                = 50
  ...
kwargs that would be passed to model.generate(**kwargs):
  {'do_sample': True, 'temperature': 0.8, 'top_k': 50, 'top_p': 0.95, 'max_new_tokens': 40, 'repetition_penalty': 1.1}

caught expected error -> generation.top_p must be in (0, 1], got 1.5
```
Input: a JSON file. Processing: parse -> validate -> return dict. Output: validated settings, ready for `model.generate(**kwargs)`. This is exactly what module 09's API request body will carry (with Pydantic doing the validation).

## Experiments
- Set `"top_k": 0` (disabled) and check the validator still passes.
- Add `"seed": 42` and make `generation_kwargs` drop it (HF `generate` would reject unknown keys).
- Add a `"stop_strings": ["\n\n"]` parameter and a validation rule for it.

## Debugging practice
1. Delete a comma in `config.json`: `json.decoder.JSONDecodeError: Expecting ',' delimiter: line 8 column 3` - the traceback gives the exact line/column of the *syntax* mistake in the JSON.
2. Rename `"generation"` to `"gen"`: `KeyError: 'generation'` - the key your code asked for doesn't exist. Which is better: crash with a KeyError, or catch it and say "config.json is missing the 'generation' section"? (Exercise.)

## Rebuild checklist
- [ ] Write a loader that reads JSON and validates 3 parameters.
- [ ] Explain every parameter in the file without looking.
- [ ] Explain why validation happens at load time.

## The new problem (cliffhanger)
The dials are named, but you do not yet know what they do. The most surprising one is also the first that matters: why would a model ever pick anything except its single best token? Next: [step 02, sampling from scratch](../02_sampling_from_scratch/WALKTHROUGH.md).

> **Say it aloud (interview box):** A config file holds every training and generation parameter in one validated place, so each experiment is reproducible and a typo fails loudly at load time. I load it with a function that checks names and ranges and reuse it in every later step and in the API. The glossary inside it is the list of dials I can explain in an interview.
