# Terms — Step 03: Serialization

Folder: `10_ml_system_architecture/03_serialization` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [10 ML system architecture](../README.md)

**Terms in this folder:** [Serialization](#serialization-concept) · [pickle](#pickle-standard-library) · [joblib](#joblib-third-party-joblib) · [open(path, "wb") / "rb"](#openpath-wb-rb-python-built-in) · [Round trip](#round-trip-concept) · [np.allclose](#npallclose-third-party-numpy) · [safetensors](#safetensors-third-party-safetensors) · [save_file / load_file (safetensors.numpy)](#save_file-load_file-safetensorsnumpy-third-party-safetensors) · [model[-1] and pipeline indexing](#model-1-and-pipeline-indexing-syntax) · [coef_ and intercept_](#coef_-and-intercept_-third-party-scikit-learn) · [Arbitrary code execution during unpickling](#arbitrary-code-execution-during-unpickling-concept) · [__reduce__](#__reduce__-python-built-in) · [pickle.dumps / pickle.loads](#pickledumps-pickleloads-standard-library) · [Pickled by name (custom function gotcha)](#pickled-by-name-custom-function-gotcha-concept) · [TfidfVectorizer(preprocessor=...)](#tfidfvectorizerpreprocessor-third-party-scikit-learn) · [AttributeError: Can't get attribute](#attributeerror-cant-get-attribute-python-built-in) · [Library version compatibility](#library-version-compatibility-concept) · [Format comparison (pickle, joblib, safetensors, torch.save, ONNX)](#format-comparison-pickle-joblib-safetensors-torchsave-onnx-concept) · [os.path.getsize](#ospathgetsize-standard-library) · [tempfile.mkdtemp and Path /](#tempfilemkdtemp-and-path-standard-library) · [python -c "code"](#python--c-code-command) · [Pipeline round trip with the original reference](#pipeline-round-trip-with-the-original-reference-concept) · [Trade-off: safe formats need the model code](#trade-off-safe-formats-need-the-model-code-concept)

---

## Serialization  `(concept)`
**1. Definition:** Converting an in-memory object into a sequence of bytes (a file) so it can be stored or sent, and later restoring it (deserialization).
**2. In plain words:** A trained model lives only in RAM until you serialize it. Saving it lets another process, such as your API, load the same model without retraining. The format you choose decides speed, size and, importantly, safety.
**3. Example to quote:**
> "After training I serialize the fitted pipeline to a file; the API deserializes it at startup and gets identical predictions."

**Seen in:** whole step

## pickle  `(standard library)`
**1. Definition:** Python's built-in module for serializing almost any Python object, including references to classes and functions; `dump`/`load` work with files and `dumps`/`loads` with bytes.
**2. In plain words:** Convenient but dangerous: unpickling can execute code chosen by whoever made the file. Use it only for data you created or fully trust.
**3. Example to quote:**
> "`pickle.dump(model, f)` saved my pipeline in 2,310 bytes, and `pickle.load` restored it with identical predictions."

```python
import pickle
print(pickle.loads(pickle.dumps({"a": 1})))   # {'a': 1}
```

**Seen in:** section 1

## joblib  `(third-party: joblib)`
**1. Definition:** A library whose `dump` and `load` serialize Python objects like pickle, but store large numpy arrays efficiently; the standard way to save scikit-learn models.
**2. In plain words:** Same safety caveat as pickle. For tiny models it can even be slightly larger (2,481 vs 2,310 bytes here); its advantage shows with big arrays, and `compress=` reduces file size.
**3. Example to quote:**
> "I save scikit-learn models with `joblib.dump`; with large arrays it is more efficient than plain pickle."

**Seen in:** section 1

## open(path, "wb") / "rb"  `(Python built-in)`
**1. Definition:** `open` opens a file; mode `"wb"` means write binary and `"rb"` means read binary.
**2. In plain words:** Pickle writes bytes, not text, so you need binary mode. Using a `with` block closes the file automatically.
**3. Example to quote:**
> "`with open(tmp / 'model.pkl', 'wb') as f: pickle.dump(model, f)` writes the model as bytes."

```python
with open("x.bin", "wb") as f:
    f.write(b"hi")
```

**Seen in:** section 1

## Round trip  `(concept)`
**1. Definition:** Saving an object and loading it back, then verifying it behaves identically.
**2. In plain words:** Always test it: predict with the original and the loaded model on the same inputs and compare. It catches silent breakage in the save/load path.
**3. Example to quote:**
> "`np.allclose(reference, from_pickle.predict_proba(X_test))` printed True for pickle and for joblib."

**Seen in:** section 1

## np.allclose  `(third-party: numpy)`
**1. Definition:** `np.allclose(a, b)` returns True if two arrays are equal element by element within a small tolerance.
**2. In plain words:** Floating-point results can differ in tiny ways, so you compare "close enough" instead of exact equality. `np.array_equal` is the exact version, used for the safetensors check.
**3. Example to quote:**
> "I use `np.allclose` to confirm the loaded model's probabilities match the original's."

```python
import numpy as np
print(np.allclose([1.0], [1.0000001]))   # True
```

**Seen in:** sections 1 and 2

## safetensors  `(third-party: safetensors)`
**1. Definition:** A file format that stores only named tensors or arrays with a small header, and cannot contain executable code.
**2. In plain words:** Because loading cannot run code, it is safe to load files from strangers. Hugging Face LLM checkpoints (`model.safetensors`) use it. The trade-off is that you must rebuild the model code yourself and load the arrays into it.
**3. Example to quote:**
> "My classifier's weights saved as 400 bytes with two named arrays, `coef` and `intercept`, and loading gives back identical numbers."

**Seen in:** section 2

## save_file / load_file (safetensors.numpy)  `(third-party: safetensors)`
**1. Definition:** `save_file(dict_of_arrays, path)` writes named numpy arrays to a `.safetensors` file; `load_file(path)` reads them back into a dict.
**2. In plain words:** You hand over a plain dict of arrays. The file knows only names, shapes and numbers, so there is nothing to execute.
**3. Example to quote:**
> "`save_file({'coef': ..., 'intercept': ...}, path)` wrote the weights, and `load_file(path)` returned the same dict."

**Seen in:** section 2

## model[-1] and pipeline indexing  `(syntax)`
**1. Definition:** A scikit-learn Pipeline supports indexing: `pipeline[-1]` returns the last step (the classifier), and `pipeline[0]` the first.
**2. In plain words:** To pull out the learned weights you need the classifier itself, not the whole pipeline. `.coef_` and `.intercept_` are its learned attributes.
**3. Example to quote:**
> "`model[-1].coef_` gives the logistic regression weights from inside the pipeline."

```python
print([1, 2, 3][-1])   # 3
```

**Seen in:** section 2

## coef_ and intercept_  `(third-party: scikit-learn)`
**1. Definition:** Attributes of a fitted linear model holding its learned weights per feature (`coef_`) and its bias term (`intercept_`).
**2. In plain words:** These numbers are the model. Saving only them in safetensors is safe, but you must recreate the surrounding code (the vectorizer and the classifier) to use them.
**3. Example to quote:**
> "My `coef` array has one weight per vocabulary word; those numbers are what safetensors stored."

**Seen in:** section 2

## Arbitrary code execution during unpickling  `(concept)`
**1. Definition:** Loading a pickle can run any code that the file's creator embedded, because pickle rebuilds objects by calling functions.
**2. In plain words:** "Loading is executing". A downloaded `model.pkl` could delete files or steal keys when you call `pickle.load`. Only load files you made or fully trust, and prefer safetensors or ONNX for sharing.
**3. Example to quote:**
> "My harmless demo made `pickle.loads` call `print`; an attacker would have used `os.system` instead."

**Seen in:** section 3

## __reduce__  `(Python built-in)`
**1. Definition:** A special method that tells pickle how to rebuild an object: it returns a callable and a tuple of arguments, which pickle calls on load.
**2. In plain words:** It is the legitimate hook for custom pickling and also the attack hook. In the demo it returns `(print, (message,))`, so unpickling calls `print(message)`.
**3. Example to quote:**
> "`__reduce__` returns `(print, ('!!! code ran ...',))`, so loading the pickle runs `print` with that message."

**Seen in:** `class Innocent`

## pickle.dumps / pickle.loads  `(standard library)`
**1. Definition:** `dumps(obj)` returns the pickle as bytes and `loads(bytes)` rebuilds the object from bytes.
**2. In plain words:** The in-memory counterparts of `dump`/`load`. Used here to simulate "an attacker sent me these bytes".
**3. Example to quote:**
> "`evil_bytes = pickle.dumps(Innocent())` is the file an attacker would send; `pickle.loads(evil_bytes)` triggers the code."

**Seen in:** section 3

## Pickled by name (custom function gotcha)  `(concept)`
**1. Definition:** Pickle stores a reference (module and name) to functions and classes, not their code, so the loading process must be able to import the same name.
**2. In plain words:** If a model uses your function `shout` defined in `__main__`, loading it from another script fails with `AttributeError: Can't get attribute 'shout' on <module '__main__'>`. Keep custom code in an importable module deployed with the model, or use only built-in estimator parts.
**3. Example to quote:**
> "My `custom.joblib` uses `shout` as a preprocessor; a different script can't load it unless it can import `shout`."

**Seen in:** section 4

## TfidfVectorizer(preprocessor=...)  `(third-party: scikit-learn)`
**1. Definition:** The `preprocessor` parameter accepts a function applied to each raw document before tokenizing.
**2. In plain words:** It lets you plug in custom cleaning, but that function becomes part of the pickled model and creates the by-name dependency described above.
**3. Example to quote:**
> "`TfidfVectorizer(preprocessor=shout)` upper-cases text, and now the saved model depends on my `shout` function."

**Seen in:** section 4

## AttributeError: Can't get attribute  `(Python built-in)`
**1. Definition:** An error raised when pickle cannot find a referenced name (class or function) in the module it expects.
**2. In plain words:** It means "the file refers to something that doesn't exist where I'm loading it". Fix by moving that code into an importable module used in both training and serving.
**3. Example to quote:**
> "`Can't get attribute 'shout' on <module '__main__'>` told me the loader script has no `shout`; I moved it into `helpers.py`."

**Seen in:** section 4, `WALKTHROUGH.md`

## Library version compatibility  `(concept)`
**1. Definition:** A pickled scikit-learn model should be loaded with the same or a compatible library version as the one that saved it.
**2. In plain words:** Internals change between versions, so an old pickle may fail or behave differently. Record versions in metadata and keep the training code and data so you can re-create the model.
**3. Example to quote:**
> "After upgrading scikit-learn my old pickle may not load, which is why I store the version in the metadata."

**Seen in:** `WALKTHROUGH.md` rules

## Format comparison (pickle, joblib, safetensors, torch.save, ONNX)  `(concept)`
**1. Definition:** pickle and joblib store whole Python objects (unsafe to load from strangers), safetensors stores arrays only (safe), `torch.save` uses pickle inside unless `weights_only=True`, and ONNX stores a computation graph plus weights (no Python code).
**2. In plain words:** Pick by use: joblib for your own scikit-learn models, safetensors for sharing weights, ONNX for cross-language serving. The more of "the code" a format stores, the more careful you must be loading it.
**3. Example to quote:**
> "For my own sklearn pipeline I use joblib; for LLM weights from the internet I expect safetensors."

**Seen in:** `WALKTHROUGH.md` table

## os.path.getsize  `(standard library)`
**1. Definition:** Returns the size of a file in bytes.
**2. In plain words:** Used to compare formats. Size matters for storage and load time, though for tiny models the difference is trivial.
**3. Example to quote:**
> "`os.path.getsize(tmp / 'model.pkl')` printed 2310 bytes."

**Seen in:** sections 1 and 2

## tempfile.mkdtemp and Path /  `(standard library)`
**1. Definition:** `mkdtemp()` makes a temporary directory; `Path(...) / "name"` joins path parts.
**2. In plain words:** The demo writes to a throwaway folder so your project stays clean.
**3. Example to quote:**
> "`tmp = Path(tempfile.mkdtemp())` gives a scratch folder, and `tmp / 'model.pkl'` is a file path inside it."

**Seen in:** top of `main.py`

## python -c "code"  `(command)`
**1. Definition:** Runs a short piece of Python passed as a string in a fresh interpreter process.
**2. In plain words:** A fresh process proves that loading works (or fails) without any leftover definitions from the training script, which is exactly what happens in a real serving environment.
**3. Example to quote:**
> "`python -c \"import joblib; joblib.load('custom.joblib')\"` fails in a new process because `shout` doesn't exist there."

```bash
python -c "print(1 + 1)"
```

**Seen in:** section 4

## Pipeline round trip with the original reference  `(concept)`
**1. Definition:** Keeping `reference = model.predict_proba(X_test)` from the original model to compare against reloaded copies.
**2. In plain words:** It is a simple regression test for any save/load path: if the numbers match, the format preserved the behaviour.
**3. Example to quote:**
> "I store the original probabilities as `reference` and check every loaded copy against them."

**Seen in:** top of `main.py`

## Trade-off: safe formats need the model code  `(concept)`
**1. Definition:** Formats that cannot execute code (safetensors, ONNX) hold only data or a graph, so loading requires you to have, or rebuild, the model definition.
**2. In plain words:** Safety costs convenience. With pickle the model arrives ready to use; with safetensors you must write the code that puts the arrays into a model.
**3. Example to quote:**
> "Safetensors gave me just `coef` and `intercept`; to predict I'd have to rebuild the vectorizer and classifier around them."

**Seen in:** section 2 print, `WALKTHROUGH.md`
