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

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Serialization | pickling (Python), marshalling, saving/exporting a model | Near-synonyms | Turning an in-memory object into bytes; loading is deserialization / unpickling. |
| pickle | `.pkl` | No | Built-in Python serialiser; stores code references and can run code on load. |
| joblib | `joblib.dump/load` | Built on pickle | Pickle plus efficient storage for big numpy arrays; same security risk. |
| safetensors | `.safetensors` | No | Stores only named tensors/arrays; no code, so safe to load from strangers. |
| ONNX | open model exchange format | No | A portable computation graph (not just numbers) runnable in many runtimes. |
| Weights | parameters, coefficients, checkpoint contents | Near-synonyms | The learned numbers; the model code (architecture) is separate. |
| Model artifact vs checkpoint | saved model, snapshot | Loosely | Artifact = the deployable file; checkpoint = a mid-training snapshot to resume from. |

## Code, step by step

### 1. Docstring: goal, run command, dependencies  (`main.py` lines 1–11)

```python
"""
GOAL:   Learn how models are saved and loaded (joblib, pickle, safetensors), compare them,
        and understand WHY loading a pickle from an untrusted source is dangerous.

RUN:    cd 10_ml_system_architecture/03_serialization && python main.py

EXPECTED OUTPUT (shape): sizes of each format, round-trip equality checks, a harmless demonstration
        of code running during unpickling, and a gotcha with custom functions.

NEEDS:  scikit-learn, joblib, safetensors + numpy (safetensors comes with transformers)
"""
```

- **In simple words:** Says this step compares ways to save a model and shows why pickles are risky.
- **Exactly:** Header comment only.
- **Why it is here:** Orientation.

### 2. Imports and borrowing step 01's data  (`main.py` lines 13–27)

```python
import os
import pickle                                   # built-in: serialize almost any Python object
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
from safetensors.numpy import load_file, save_file   # safetensors: store ONLY named arrays (no code)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_pipeline_end_to_end"))
from data import get_test_data, get_training_data  # noqa: E402
```

- **In simple words:** Collect libraries for each saving format, plus the shared data.
- **Exactly:** `pickle` built-in; `joblib`; `numpy` as `np`; `safetensors.numpy.save_file/load_file` for the numpy flavour; sklearn parts; `sys.path.insert` as in step 02 to import data.py.
- **Why it is here:** Each saving format needs its own tool.

### 3. Train a pipeline and remember its predictions  (`main.py` lines 29–33)

```python
tmp = Path(tempfile.mkdtemp())                  # throwaway folder for this demo
X, y = get_training_data()
X_test, _ = get_test_data()
model = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=1000))]).fit(X, y)
reference = model.predict_proba(X_test)         # remember what the ORIGINAL model says
```

- **In simple words:** Make a model, then note what it says so we can check later copies say the same.
- **Exactly:** `tempfile.mkdtemp()` creates a throwaway directory. `.fit(X, y)` returns the fitted Pipeline. `reference = predict_proba(X_test)` has shape (8, 2).
- **Why it is here:** A round trip is only correct if predictions are identical afterwards.

### 4. Save with pickle and with joblib  (`main.py` lines 35–38)

```python
# ---- 1. pickle and joblib: save the whole Python object (structure + learned numbers) ----
with open(tmp / "model.pkl", "wb") as f:        # "wb" = write binary
    pickle.dump(model, f)
joblib.dump(model, tmp / "model.joblib")        # joblib = pickle + efficient storage of big numpy arrays
```

- **In simple words:** Write the same model twice, using two tools.
- **Exactly:** `open(..., 'wb')` = write binary; `pickle.dump(obj, f)`. `joblib.dump(obj, path)`.
- **Why it is here:** To compare the two formats.

### 5. Print each file's size  (`main.py` lines 39–40)

```python
for name in ("model.pkl", "model.joblib"):
    print(f"1. {name:<13} {os.path.getsize(tmp / name):>6} bytes")
```

- **In simple words:** Compare file sizes.
- **Exactly:** `os.path.getsize` returns bytes. Real: `model.pkl 2310 bytes`, `model.joblib 2481 bytes`. joblib is slightly larger here because the model is tiny; its benefit appears with big arrays.
- **Why it is here:** Size matters for deploy time and storage.

### 6. Load both back and check predictions match  (`main.py` lines 41–45)

```python
with open(tmp / "model.pkl", "rb") as f:
    from_pickle = pickle.load(f)
from_joblib = joblib.load(tmp / "model.joblib")
print("   round-trip identical predictions:",
      np.allclose(reference, from_pickle.predict_proba(X_test)), np.allclose(reference, from_joblib.predict_proba(X_test)))
```

- **In simple words:** Reopen the saved files and see if they behave like the original.
- **Exactly:** `pickle.load(f)` / `joblib.load(path)`. `np.allclose(a, b)` is True if all numbers are equal within a tiny tolerance. Real: `True True`.
- **Why it is here:** Confirms the round trip lost nothing.

### 7. Pull out only the numbers for safetensors  (`main.py` lines 47–48)

```python
# ---- 2. safetensors: store only the numbers (weights), never code ----
weights = {"coef": model[-1].coef_, "intercept": model[-1].intercept_}   # model[-1] = last pipeline step (the classifier)
```

- **In simple words:** Take just the classifier's learned numbers, leaving the code behind.
- **Exactly:** `model[-1]` is the last Pipeline step (LogisticRegression). `coef_` has shape (1, 32) (binary problem, 32 words), `intercept_` shape (1,). They go in a dict of name -> array.
- **Why it is here:** safetensors stores arrays only.

### 8. Save, reload and print the safetensors file  (`main.py` lines 49–52)

```python
save_file(weights, str(tmp / "weights.safetensors"))
back = load_file(str(tmp / "weights.safetensors"))
print(f"2. safetensors    {os.path.getsize(tmp / 'weights.safetensors'):>6} bytes; arrays: {list(back)}; "
      f"identical: {np.array_equal(back['coef'], weights['coef'])}")
```

- **In simple words:** Write the numbers to a file and read them back.
- **Exactly:** `save_file(dict, path)` needs a `str` path; `load_file(path)` returns a dict of arrays. Real: `400 bytes; arrays: ['coef', 'intercept']; identical: True`. `np.array_equal` is exact equality.
- **Why it is here:** Shows the format works and is much smaller (no vocabulary, no code).

### 9. Print the trade-off note  (`main.py` lines 53–54)

```python
print("   (it stores arrays only: to use them you must rebuild the model code yourself -- that is the safety trade-off.)")
print("   Hugging Face LLM checkpoints (module 04 step 06, load pretrained model) use this format: model.safetensors.")
```

- **In simple words:** Explain that numbers alone are not a usable model.
- **Exactly:** Plain prints. The vocabulary and the model structure are not in the file; you must rebuild them. Hugging Face LLM checkpoints use this format.
- **Why it is here:** This is the safety trade-off: no code stored, so none can run.

### 10. Define a class that runs code when unpickled  (`main.py` lines 56–59)

```python
# ---- 3. SECURITY: unpickling RUNS CODE chosen by whoever made the file ----
class Innocent:
    def __reduce__(self):                       # tells pickle "to rebuild me, call this function with these args"
        return (print, ("!!! code ran during pickle.load -- an attacker could run ANYTHING here !!!",))
```

- **In simple words:** A booby-trapped object: when 'opened' it runs a command of the attacker's choosing.
- **Exactly:** `__reduce__` returns `(callable, args)`; pickle will call `callable(*args)` when loading. Here the callable is `print`, harmless; an attacker would use e.g. `os.system`.
- **Why it is here:** Demonstrates the vulnerability in a harmless way.

### 11. Load the malicious pickle and see the code run  (`main.py` lines 61–64)

```python
evil_bytes = pickle.dumps(Innocent())           # the 'model file' an attacker would send you
print("3. loading an untrusted pickle:")
pickle.loads(evil_bytes)                        # prints the message: proof that load == execute
print("   Rule: only load pickle/joblib files you created or fully trust. Prefer safetensors/ONNX for sharing.")
```

- **In simple words:** Opening the file prints the message: loading equals executing.
- **Exactly:** `pickle.dumps` returns bytes (instead of writing a file); `pickle.loads(bytes)` triggers `print(...)`. Real output shows the `!!! code ran during pickle.load ... !!!` line.
- **Why it is here:** Why you only load pickle/joblib files you trust.

### 12. A function defined in the script  (`main.py` lines 66–68)

```python
# ---- 4. GOTCHA: custom functions are pickled BY NAME ----
def shout(text):                                # defined in __main__ of THIS script
    return text.upper()
```

- **In simple words:** A custom text cleaner called `shout`.
- **Exactly:** Defined in `__main__`, i.e. in this script's namespace.
- **Why it is here:** It becomes the problem case in the next block.

### 13. Use shout inside a pipeline and save it  (`main.py` lines 70–72)

```python
custom = Pipeline([("tfidf", TfidfVectorizer(preprocessor=shout)), ("clf", LogisticRegression(max_iter=1000))])
custom.fit(X, y)
joblib.dump(custom, tmp / "custom.joblib")
```

- **In simple words:** Plug our own function into the vectoriser and save.
- **Exactly:** `TfidfVectorizer(preprocessor=shout)` calls `shout` on each text. Pickle stores only the function's NAME (`__main__.shout`), not its body.
- **Why it is here:** Shows that custom code is saved by reference.

### 14. Print the failure you would see elsewhere  (`main.py` lines 73–75)

```python
print("4. custom.joblib saved. Loading it from a DIFFERENT script fails unless that script can import `shout`:")
print("   try:  python -c \"import joblib; joblib.load('" + str(tmp / 'custom.joblib') + "')\"   -> AttributeError: Can't get attribute 'shout' on <module '__main__'>")
print("   Fix: keep custom code in an importable module (my_module.shout) and deploy that module with the model, or use only built-in sklearn parts.")
```

- **In simple words:** Tell the reader what breaks in another script.
- **Exactly:** Prints a `python -c` command and the error `AttributeError: Can't get attribute 'shout' on <module '__main__'>`. Fix: put the function in an importable module or use only built-in sklearn parts.
- **Why it is here:** A saved model that cannot load in the server is a deploy-day failure.

### 15. Comments: try it and break it  (`main.py` lines 76–77)

```python
# TRY IT: train with n=2000 samples (get_training_data(n=2000)) and compare the pickle vs joblib sizes again.
# BREAK IT: run the `python -c` command printed above and read the AttributeError traceback.
```

- **In simple words:** Experiments.
- **Exactly:** Comments only.
- **Why it is here:** Pointers to experiments.


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
