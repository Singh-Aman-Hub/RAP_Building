# Step 01 - A config file for every parameter

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

## Say it aloud
"I keep parameters in a config file so experiments are reproducible and the code doesn't change when settings change. The loader reads JSON, validates ranges, and returns a dict I can pass straight into `model.generate`. Validating early turns silent misbehaviour into a clear error. Trade-off: a config that's too flexible hides defaults; keep it small and documented."
