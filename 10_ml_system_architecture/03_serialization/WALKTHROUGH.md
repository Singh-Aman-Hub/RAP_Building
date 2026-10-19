# Step 03 — Serialization: pickle, joblib, safetensors (and why pickle is dangerous)

## The story so far
Step 02 made the artifact the only handoff between training and serving, so the file format has to be trustworthy.

## The pain point
Pickle writes the whole Python object, which is convenient. But loading a pickle runs code from the file, and a pickle made with one library version may fail or misbehave with another.

## The spark
Choose the format on purpose: joblib or pickle only for files you made yourself, data-only formats (safetensors, ONNX) for sharing weights, and always record library versions beside the file.

## Goal
Know how a fitted model becomes a file and back, which format to choose, and the security and compatibility traps.

## Run it
```bash
cd 10_ml_system_architecture/03_serialization && python main.py
```
Real output:
```
1. model.pkl       2310 bytes
1. model.joblib    2481 bytes
   round-trip identical predictions: True True
2. safetensors       400 bytes; arrays: ['coef', 'intercept']; identical: True
3. loading an untrusted pickle:
!!! code ran during pickle.load -- an attacker could run ANYTHING here !!!
4. custom.joblib saved. Loading it from a DIFFERENT script fails unless that script can import `shout`: ...
AttributeError: Can't get attribute 'shout' on <module '__main__' ...>      (from the python -c command)
```

## Format comparison
| Format | Stores | Safe to load from strangers? | Typical use |
|---|---|---|---|
| `pickle` | any Python object (code references + data) | **No** — executes code | quick local caching |
| `joblib` | same as pickle, efficient for big numpy arrays | **No** | scikit-learn models |
| `safetensors` | named tensors/arrays only | **Yes** — no code | Hugging Face LLM weights (`model.safetensors`) |
| `torch.save` (`.pt`) | tensors / pickled objects | Mostly no (pickle inside) unless `weights_only=True` | PyTorch checkpoints |
| ONNX | computation graph + weights | Yes (no Python code) | cross-language/runtime serving |
(For tiny models joblib can even be slightly larger than pickle; its advantage shows with big arrays.)

## Syntax explained
| Syntax | Meaning | Example |
|---|---|---|
| `open(path, "wb")` / `"rb"` | Open for **w**riting / **r**eading in **b**inary | — |
| `__reduce__` | Special method telling pickle how to rebuild an object — the attack hook | — |
| `model[-1]` | Pipelines support indexing; `-1` = last step | `[1,2,3][-1]` → 3 |
| `np.allclose(a, b)` | Arrays equal up to tiny float error | — |
| `python -c "code"` | Run a one-liner in a fresh process (proves cross-process loading) | — |

## Libraries & functions
| Name | Kind | Does |
|---|---|---|
| `pickle.dump / load / dumps / loads` | stdlib | file vs bytes variants |
| `joblib.dump / load` | 3rd-party | like pickle, with `compress=` option |
| `safetensors.numpy.save_file(dict, path)` / `load_file(path)` | 3rd-party | named arrays ⇄ file |
| `os.path.getsize(path)` | stdlib | file size in bytes |

## Execution trace
Security demo: `pickle.dumps(Innocent())` records "to rebuild this, call `print(<message>)`". `pickle.loads(bytes)` **obeys** and calls `print` — we used a harmless function, an attacker would use `os.system`. Therefore *loading = executing*.

## Rules to remember
1. Only unpickle files you produced or trust fully (same for `joblib.load`, `torch.load` without `weights_only=True`).
2. A pickled model requires the **same classes importable** and a **compatible library version** at load time (record versions in metadata — step 02/04).
3. Custom functions/classes are pickled *by name*: keep them in an importable module that is deployed with the model, or use only built-in estimator parts.
4. Pickle files are not a stable archive format: upgrade scikit-learn and old files may fail to load. Keep the training code + data fingerprint to re-create the model.

## Experiments
1. Use `get_training_data(n=2000)` and compare file sizes again.
2. `joblib.dump(model, path, compress=3)` — smaller? How much slower to load?
3. Run the printed `python -c` command and read the `AttributeError` traceback.

## Debugging practice
`AttributeError: Can't get attribute 'shout' on <module '__main__'>` means pickle recorded a reference to `__main__.shout` and the loader script has no such function. Fix by moving `shout` to `helpers.py`, importing it in both the training and serving code.

## Rebuild checklist
- [ ] Save and load a model with joblib in 4 lines.
- [ ] Explain in one sentence why pickle is unsafe.
- [ ] Name two safer alternatives.

## The new problem (cliffhanger)
Now we can save a model safely, but `model.joblib` gets overwritten on every training run. Which one is in production, and how do we go back? Next: [04_model_registry_versioning](../04_model_registry_versioning/).

## Say it aloud
> **Say it aloud (interview box):** Pickle and joblib serialize the whole Python object, which is convenient but unsafe, because loading executes code, so I load only files I trust. For sharing weights I use safetensors or ONNX, which hold data only. I also record library versions, because a pickled model may not load under a different version.
