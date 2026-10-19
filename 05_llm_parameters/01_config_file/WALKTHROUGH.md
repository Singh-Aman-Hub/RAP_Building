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
