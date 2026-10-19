# Terms — A Config File for Every Parameter

Folder: `05_llm_parameters/01_config_file` · Files covered: `main.py`, `config_loader.py`, `config.json`, `WALKTHROUGH.md`
Back to module: [05 LLM parameters README](../README.md)

**Terms in this folder:** [Config file](#config-file) · [JSON](#json) · [json module](#json-module) · [pathlib.Path](#pathlibpath) · [__file__ and Path.parent](#__file__-and-pathparent) · [Learning rate](#learning-rate) · [Batch size](#batch-size) · [Epochs](#epochs) · [Optimizer (adamw, sgd)](#optimizer-adamw-sgd) · [Weight decay](#weight-decay) · [do_sample](#do_sample) · [Temperature](#temperature) · [top_k](#top_k) · [top_p](#top_p) · [max_new_tokens](#max_new_tokens) · [repetition_penalty](#repetition_penalty) · [Context window](#context-window) · [Dict of dicts (MEANING)](#dict-of-dicts-meaning) · [with open(...) as f](#with-open-as-f) · [json.load](#jsonload) · [Nested dictionary access](#nested-dictionary-access) · [Validation (fail early)](#validation-fail-early) · [raise ValueError](#raise-valueerror) · [Chained comparison](#chained-comparison) · [f-string with a dict key](#f-string-with-a-dict-key) · [dict(...) copy](#dict-copy) · [generation_kwargs / **kwargs](#generation_kwargs--kwargs) · [sys.path.insert](#syspathinsert) · [if __name__ == "__main__"](#if-__name__--__main__) · [json.dumps and json.loads deep copy](#jsondumps-and-jsonloads-deep-copy) · [Path.write_text and unlink](#pathwrite_text-and-unlink) · [try / except / finally](#try--except--finally) · [JSONDecodeError](#jsondecodeerror) · [KeyError](#keyerror)

---

## Config file  `(concept)`
**1. Definition:** A file (here `config.json`) that stores a program's settings separately from its code.
**2. In plain words:** Instead of hard-coding numbers like the learning rate inside scripts, you keep them in one place. That makes experiments repeatable and comparable, and lets a service change settings without code edits. Keep it small and documented so it does not hide surprising defaults.
**3. Example to quote:** > "All my training and generation parameters live in `config.json`, so changing temperature from 0.8 to 0.3 needs no code change."

**Seen in:** `config.json`, `main.py` docstring.

## JSON  `(concept)`
**1. Definition:** JavaScript Object Notation, a text format for dictionaries, lists, numbers, strings, `true`/`false` and `null`.
**2. In plain words:** It is the common language between programs and APIs. It looks like a Python dict but has no comments, no trailing commas and uses lowercase `true`/`false`. A missing comma gives a `JSONDecodeError` with the line and column.
**3. Example to quote:** > "`{"learning_rate": 5e-5, "do_sample": true}` is valid JSON; Python reads it as a dict with `True`."

**Seen in:** `config.json`.

## json module  `(standard library)`
**1. Definition:** Standard-library module that converts between JSON text and Python objects (`load`, `loads`, `dump`, `dumps`).
**2. In plain words:** `load` reads a JSON file into a dict; `dumps` turns a dict into a JSON string. No installation needed.
**3. Example to quote:** > "I used `json.load(f)` to turn `config.json` into a Python dict."
```python
import json
json.loads('{"a": 1}')  # {'a': 1}
```
**Seen in:** `config_loader.py`, `main.py`.

## pathlib.Path  `(standard library)`
**1. Definition:** A class representing file-system paths as objects, supporting joining with `/`, reading, writing and deleting.
**2. In plain words:** It is safer and clearer than gluing path strings together, and works on any operating system.
**3. Example to quote:** > "`Path(__file__).parent / 'config.json'` finds the config next to my script no matter where I run it from."
```python
from pathlib import Path
Path("a") / "b.txt"
```
**Seen in:** `config_loader.py`, `main.py`.

## __file__ and Path.parent  `(syntax)`
**1. Definition:** `__file__` holds the path of the current script; `.parent` gives the folder containing a path.
**2. In plain words:** Building paths relative to the script file means it works from any working directory, which avoids "file not found" bugs.
**3. Example to quote:** > "I use `Path(__file__).parent` so the loader works from any cwd."

**Seen in:** `config_loader.py`, `CONFIG_PATH`.

## Learning rate  `(concept)`
**1. Definition:** The step size of each weight update during training.
**2. In plain words:** Too big and training becomes unstable; too small and it is slow. Here it is 5e-5, a typical fine-tuning value. You will sweep it in step 06.
**3. Example to quote:** > "My config sets `learning_rate` to 5e-5, small enough not to wreck pretrained weights."

**Seen in:** `config.json`, `MEANING`.

## Batch size  `(concept)`
**1. Definition:** The number of training examples processed before one weight update.
**2. In plain words:** Bigger batches give smoother gradients and better hardware use but need more memory. The config uses 4.
**3. Example to quote:** > "With `batch_size` 4, the model sees four examples before each update."

**Seen in:** `config.json`.

## Epochs  `(concept)`
**1. Definition:** One epoch is one full pass over the whole training dataset.
**2. In plain words:** More epochs means more learning but also a higher risk of overfitting. The config uses 3.
**3. Example to quote:** > "Three epochs means the model reads the entire dataset three times."

**Seen in:** `config.json`.

## Optimizer (adamw, sgd)  `(concept)`
**1. Definition:** The algorithm that turns gradients into weight updates; AdamW and SGD are common choices.
**2. In plain words:** SGD takes simple steps downhill; AdamW adapts the step size per weight and is the default for transformers.
**3. Example to quote:** > "My config names `adamw` as the optimizer for fine-tuning."

**Seen in:** `config.json`.

## Weight decay  `(concept)`
**1. Definition:** A small penalty on large weights added during optimisation to reduce overfitting.
**2. In plain words:** It nudges weights toward zero so the model does not rely on a few huge values. The config sets 0.01.
**3. Example to quote:** > "`weight_decay` 0.01 is a mild regulariser."

**Seen in:** `config.json`.

## do_sample  `(third-party: transformers)`
**1. Definition:** Generation flag: `False` gives greedy decoding, `True` samples from the probability distribution.
**2. In plain words:** With sampling off, output is deterministic. Temperature, top_k and top_p only take effect when it is on.
**3. Example to quote:** > "`do_sample: true` lets temperature and top-p change the output."

**Seen in:** `config.json`, `MEANING`.

## Temperature  `(concept)`
**1. Definition:** A number that divides the logits before softmax; below 1 makes the distribution sharper, above 1 flatter.
**2. In plain words:** Low temperature gives safe, repeatable text; high gives varied, riskier text. It must be greater than 0 (use `do_sample=false` for greedy).
**3. Example to quote:** > "I set temperature 0.8 for fairly focused but not rigid output."

**Seen in:** `config.json`, validation in `config_loader.py`.

## top_k  `(concept)`
**1. Definition:** Sampling restriction that keeps only the k most likely tokens.
**2. In plain words:** It cuts off the long tail of unlikely tokens. `0` disables it in this config's validator.
**3. Example to quote:** > "With `top_k` 50 the model only chooses among the 50 best candidates."

**Seen in:** `config.json`.

## top_p  `(concept)`
**1. Definition:** Nucleus sampling: keep the smallest set of tokens whose probabilities add up to at least p.
**2. In plain words:** The candidate list adapts: few tokens when the model is confident, more when it is unsure. Valid values are in (0, 1].
**3. Example to quote:** > "`top_p` 0.95 keeps tokens covering 95% of the probability."

**Seen in:** `config.json`, `config_loader.py` validation.

## max_new_tokens  `(concept)`
**1. Definition:** The maximum number of tokens to generate after the prompt.
**2. In plain words:** It caps answer length, cost and latency. The loader checks it is between 1 and the context window.
**3. Example to quote:** > "I cap output at 40 new tokens."

**Seen in:** `config.json`.

## repetition_penalty  `(concept)`
**1. Definition:** A factor above 1 that lowers the probability of tokens already used, discouraging loops.
**2. In plain words:** It helps stop outputs like "the the the". Too high causes odd wording.
**3. Example to quote:** > "I use `repetition_penalty` 1.1 to reduce loops."

**Seen in:** `config.json`.

## Context window  `(concept)`
**1. Definition:** The maximum number of tokens (prompt plus output) a model can attend to.
**2. In plain words:** For distilgpt2 it is 1024. The loader makes sure `max_new_tokens` cannot exceed it.
**3. Example to quote:** > "My config declares `context_window` 1024, and validation rejects larger `max_new_tokens`."

**Seen in:** `config.json`, `config_loader.py`.

## Dict of dicts (MEANING)  `(syntax)`
**1. Definition:** A dictionary whose values are themselves dictionaries, here mapping section names to parameter explanations.
**2. In plain words:** Storing the plain-English meaning next to the loader keeps documentation from drifting away from code. You loop over `MEANING.items()` to print them.
**3. Example to quote:** > "`MEANING['generation']['temperature']` returns the one-line explanation of temperature."

**Seen in:** `config_loader.py`.

## with open(...) as f  `(syntax)`
**1. Definition:** Opens a file and guarantees it is closed when the block ends, even after an error.
**2. In plain words:** It prevents forgotten open files. `encoding="utf-8"` makes text reading consistent across systems.
**3. Example to quote:** > "I read the config with `with open(path, encoding='utf-8') as f` so it always closes."
```python
with open("a.txt") as f:
    data = f.read()
```
**Seen in:** `config_loader.py`, `load_config`.

## json.load  `(standard library)`
**1. Definition:** Reads JSON from a file object and returns the corresponding Python object.
**2. In plain words:** It turns the config file's text into a nested dict you can index.
**3. Example to quote:** > "`cfg = json.load(f)` gave me `cfg['generation']['top_p']`."

**Seen in:** `config_loader.py`.

## Nested dictionary access  `(syntax)`
**1. Definition:** Reading values from dicts inside dicts by chaining keys, such as `cfg["generation"]["top_k"]`.
**2. In plain words:** A missing key raises `KeyError`. The code assigns shortcuts `g, t, lim` to sub-dicts to keep validation lines short.
**3. Example to quote:** > "I use `g, t, lim = cfg['generation'], cfg['training'], cfg['limits']` to keep checks readable."

**Seen in:** `config_loader.py`, `main.py`.

## Validation (fail early)  `(concept)`
**1. Definition:** Checking that input values are sensible as soon as they are loaded, and failing with a clear message if not.
**2. In plain words:** A bad temperature caught at load time gives a clear error; the same mistake found deep in generation gives confusing behaviour. Module 09's Pydantic does the same job for API inputs.
**3. Example to quote:** > "My loader rejects `top_p` 1.5 with 'generation.top_p must be in (0, 1]', which names the exact parameter."

**Seen in:** `config_loader.py`, `load_config`.

## raise ValueError  `(syntax)`
**1. Definition:** Statement that stops execution and signals an error with a message of your choice.
**2. In plain words:** Use `ValueError` when a value is of the right type but unacceptable. Put the parameter name and the bad value in the message.
**3. Example to quote:** > "`raise ValueError('generation.temperature must be > 0')` tells the user exactly what to fix."

**Seen in:** `config_loader.py`.

## Chained comparison  `(syntax)`
**1. Definition:** Python lets you write `0 < x <= 1` to mean `0 < x and x <= 1`.
**2. In plain words:** It reads like maths. `if not 0 < g["top_p"] <= 1:` flags anything outside the range.
**3. Example to quote:** > "`0 < top_p <= 1` is the valid range for nucleus sampling."
```python
0 < 0.95 <= 1  # True
```
**Seen in:** `config_loader.py`.

## f-string with a dict key  `(syntax)`
**1. Definition:** Embedding an expression such as `{g['top_p']}` inside an f-string; use different quote types inside and outside.
**2. In plain words:** It lets the error message include the actual bad value, which makes debugging much faster.
**3. Example to quote:** > "My error says `got 1.5` because the f-string inserts `g['top_p']`."

**Seen in:** `config_loader.py`.

## dict(...) copy  `(Python built-in)`
**1. Definition:** `dict(d)` creates a new dictionary with the same key-value pairs (a shallow copy).
**2. In plain words:** `generation_kwargs` returns a copy so callers cannot accidentally change the original config.
**3. Example to quote:** > "I return `dict(cfg['generation'])` so edits to the result do not modify `cfg`."

**Seen in:** `config_loader.py`, `generation_kwargs`.

## generation_kwargs / **kwargs  `(syntax)`
**1. Definition:** `generation_kwargs(cfg)` returns a dict of keyword arguments; calling `model.generate(**kwargs)` passes each entry as a named argument.
**2. In plain words:** It lets one config drive the whole call. Only keys that `generate` understands should be included, otherwise it raises an error.
**3. Example to quote:** > "`model.generate(ids, **generation_kwargs(cfg))` applies all the settings from my config."

**Seen in:** `config_loader.py`, `main.py`.

## sys.path.insert  `(standard library)`
**1. Definition:** Adds a folder to the front of Python's module search path.
**2. In plain words:** It makes `config_loader.py` importable no matter where the script is run from. Helper modules have distinct names because every step's script is called `main.py`, and importing `main` would pick the wrong file.
**3. Example to quote:** > "I insert the script folder into `sys.path` so `from config_loader import ...` works from any directory."

**Seen in:** `main.py`.

## if __name__ == "__main__"  `(syntax)`
**1. Definition:** A guard that runs the block only when the file is executed directly, not when it is imported.
**2. In plain words:** It lets a file be both a runnable script and an importable module.
**3. Example to quote:** > "The demo prints only when I run `main.py`, not when another file imports from it."

**Seen in:** `main.py`.

## json.dumps and json.loads deep copy  `(standard library)`
**1. Definition:** `json.loads(json.dumps(obj))` serialises and re-parses an object, producing an independent deep copy of JSON-compatible data.
**2. In plain words:** A cheap trick for copying nested dicts so you can break one value without affecting the original.
**3. Example to quote:** > "I deep-copied the config via JSON, set `top_p` to 1.5, and watched validation reject it."

**Seen in:** `main.py`.

## Path.write_text and unlink  `(standard library)`
**1. Definition:** `Path.write_text(s)` writes a string to a file; `Path.unlink()` deletes the file.
**2. In plain words:** The demo creates a temporary bad config, tests it, and deletes it, leaving the folder clean.
**3. Example to quote:** > "I write `_bad_config.json`, load it to trigger the error, then `unlink()` it."

**Seen in:** `main.py`.

## try / except / finally  `(syntax)`
**1. Definition:** `try` runs code, `except ErrorType` handles a specific error, and `finally` always runs afterward.
**2. In plain words:** The demo expects a `ValueError`, prints it, and deletes the temp file in `finally` whatever happened.
**3. Example to quote:** > "I catch `ValueError` to show the message and use `finally` to clean up."

**Seen in:** `main.py`.

## JSONDecodeError  `(concept)`
**1. Definition:** The exception raised when text is not valid JSON; it reports the line and column of the problem.
**2. In plain words:** A missing comma in `config.json` triggers it, and the message points to the exact spot.
**3. Example to quote:** > "Deleting a comma gave `Expecting ',' delimiter: line 8 column 3`."

**Seen in:** `main.py`, BREAK IT comment.

## KeyError  `(concept)`
**1. Definition:** The exception raised when a dictionary key does not exist.
**2. In plain words:** Renaming `generation` to `gen` makes `cfg['generation']` fail. A friendlier message could say the config is missing that section.
**3. Example to quote:** > "`KeyError: 'generation'` told me the key my code asked for was not in the file."

**Seen in:** `main.py`, BREAK IT comment.
