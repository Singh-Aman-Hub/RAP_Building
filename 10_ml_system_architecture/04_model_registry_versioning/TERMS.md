# Terms — Step 04: Model registry and versioning

Folder: `10_ml_system_architecture/04_model_registry_versioning` · Files covered: `main.py`, `registry.py`, `WALKTHROUGH.md`
Back to module: [10 ML system architecture](../README.md)

**Terms in this folder:** [Model registry](#model-registry-concept) · [Model versioning](#model-versioning-concept) · [Immutable artifact](#immutable-artifact-concept) · [Alias (production, staging)](#alias-production-staging-concept) · [latest vs pinned vs alias](#latest-vs-pinned-vs-alias-concept) · [Promote and rollback](#promote-and-rollback-concept) · [ModelRegistry class](#modelregistry-class-syntax) · [Zero-padded version names (f"v{n:04d}")](#zero-padded-version-names-fvn04d-syntax) · [sorted with a generator expression](#sorted-with-a-generator-expression-syntax) · [Path.iterdir, is_dir, rglob](#pathiterdir-is_dir-rglob-standard-library) · [mkdir(parents=True, exist_ok=True)](#mkdirparentstrue-exist_oktrue-standard-library) · [metadata.json](#metadatajson-concept) · [Data fingerprint (hashlib.sha256)](#data-fingerprint-hashlibsha256-standard-library) · [String slicing [:12]](#string-slicing-12-syntax) · [str.encode() before hashing](#strencode-before-hashing-python-built-in) · [log_loss](#log_loss-third-party-scikit-learn) · [Tie in accuracy on a toy test set](#tie-in-accuracy-on-a-toy-test-set-concept) · [joblib.dump / joblib.load for versions](#joblibdump-joblibload-for-versions-third-party-joblib) · [raise KeyError for unknown version](#raise-keyerror-for-unknown-version-syntax) · [repr formatting with !r in an error message](#repr-formatting-with-r-in-an-error-message-syntax) · [shutil.rmtree](#shutilrmtree-standard-library) · [Training and registering in one function](#training-and-registering-in-one-function-concept) · [Hyperparameters n and C in metadata](#hyperparameters-n-and-c-in-metadata-concept) · [Registry layout on disk](#registry-layout-on-disk-concept) · [Version code and data as well as weights](#version-code-and-data-as-well-as-weights-concept)

---

## Model registry  `(concept)`
**1. Definition:** A store that keeps every trained model as a numbered, documented artifact with metadata, and lets you load by version or alias.
**2. In plain words:** It answers "which model is live, how was it trained, and how do I go back?". Real tools (MLflow, Weights & Biases, SageMaker, Hugging Face Hub) do this with more features; here a folder with `v0001`, `v0002` and an `aliases.json` file does the job.
**3. Example to quote:**
> "My registry folder has `v0001`, `v0002` and `aliases.json`; serving says 'load production' and the label decides which version that is."

**Seen in:** `registry.py`

## Model versioning  `(concept)`
**1. Definition:** Giving each trained model a unique, immutable identifier so you can reproduce, compare and roll back.
**2. In plain words:** Without versions, "the model" keeps changing under you and you cannot say which one produced a bad answer. Each version records its metrics, parameters, data fingerprint and library versions.
**3. Example to quote:**
> "Every prediction logs `model_version`, so when a complaint arrives I know whether v0001 or v0002 answered."

**Seen in:** whole step

## Immutable artifact  `(concept)`
**1. Definition:** A saved version that is never modified after creation; changes produce a new version instead.
**2. In plain words:** If versions could be edited, reproducibility and rollback would break. New training means `v0003`, never an overwrite of `v0002`.
**3. Example to quote:**
> "`save()` always creates a new `vNNNN` folder; I never edit an old one."

**Seen in:** `ModelRegistry.save`

## Alias (production, staging)  `(concept)`
**1. Definition:** A movable label stored in `aliases.json` that points at a specific version, such as `"production": "v0001"`.
**2. In plain words:** Code says "load production"; operations move the label. Promoting or rolling back is a one-line JSON change, with no code redeploy.
**3. Example to quote:**
> "`reg.promote(v2, 'production')` rolls forward and `reg.promote(v1, 'production')` rolls back, without touching the serving code."

**Seen in:** `promote`, `resolve`

## latest vs pinned vs alias  `(concept)`
**1. Definition:** Three ways to reference a model: `latest` (the newest version), a pinned exact version like `v0002`, or an alias like `production`.
**2. In plain words:** `latest` is convenient but changes under you, pinned is fully reproducible, and an alias is flexible but controlled. Production should use pinned or an alias, rarely `latest`.
**3. Example to quote:**
> "In experiments I load `latest`; in production I load the alias `production` so it only changes when I promote."

**Seen in:** `resolve`

## Promote and rollback  `(concept)`
**1. Definition:** Promoting moves an alias to a new version; rolling back moves it back to the previous one.
**2. In plain words:** If v2 misbehaves in production, you point `production` back to v1 within seconds. The old model is still stored, so nothing needs retraining.
**3. Example to quote:**
> "After promoting v2 I saw problems, so I promoted v1 again; `resolve('production')` printed v0001 and traffic was back on the old model."

**Seen in:** `main.py`

## ModelRegistry class  `(syntax)`
**1. Definition:** A class bundling a root folder and the methods that act on it: `versions`, `save`, `promote`, `resolve` and `load`.
**2. In plain words:** Putting state (the root path) and behaviour together keeps the code tidy and testable. `__init__` creates the folder if needed.
**3. Example to quote:**
> "`reg = ModelRegistry(ROOT)` gives me an object whose methods save, promote and load models in that folder."

**Seen in:** `registry.py`

## Zero-padded version names (f"v{n:04d}")  `(syntax)`
**1. Definition:** The format spec `:04d` prints an integer padded with zeros to width 4, so 7 becomes `0007`.
**2. In plain words:** Padding makes version names sort correctly as text: `v0002` comes before `v0010`, whereas `v10` would sort before `v2`.
**3. Example to quote:**
> "`f'v{n:04d}'` gives `v0001`, `v0002`, ..., so `sorted(...)` puts them in the right order."

```python
print(f"v{7:04d}")   # v0007
```

**Seen in:** `save`

## sorted with a generator expression  `(syntax)`
**1. Definition:** `sorted(p.name for p in root.iterdir() if p.is_dir())` produces a sorted list from a lazily evaluated generator expression.
**2. In plain words:** It lists the version folders in order, oldest to newest. A generator expression has the shape of a list comprehension but without building an intermediate list.
**3. Example to quote:**
> "`versions()` returns `['v0001', 'v0002']` from the folders that start with `v`."

**Seen in:** `versions`

## Path.iterdir, is_dir, rglob  `(standard library)`
**1. Definition:** `iterdir()` lists the entries of a folder, `is_dir()` tests for a directory, and `rglob("*")` walks recursively through all files.
**2. In plain words:** They let the registry discover versions without keeping a separate index, and let the demo print every file it created.
**3. Example to quote:**
> "`ROOT.rglob('*')` listed `aliases.json`, and the `metadata.json` and `model.joblib` in each version."

**Seen in:** `versions`, `main.py`

## mkdir(parents=True, exist_ok=True)  `(standard library)`
**1. Definition:** Creates a directory including any missing parents; `exist_ok=True` avoids an error if it already exists.
**2. In plain words:** Makes setup idempotent: calling the registry constructor twice is safe.
**3. Example to quote:**
> "`self.root.mkdir(parents=True, exist_ok=True)` creates the registry folder only if it's missing."

**Seen in:** `ModelRegistry.__init__`

## metadata.json  `(concept)`
**1. Definition:** A JSON file stored with each version recording version name, creation time, metrics, parameters, data fingerprint and library version.
**2. In plain words:** It is the model's identity card. Months later it answers what data and settings produced it and whether it can be loaded safely.
**3. Example to quote:**
> "`load('production')` returned the model and its metadata: metrics, params and data fingerprint `6ebb39635e9d`."

**Seen in:** `save`, `load`

## Data fingerprint (hashlib.sha256)  `(standard library)`
**1. Definition:** A short hash of the training data, computed with `hashlib.sha256(bytes).hexdigest()`, that changes whenever the data changes.
**2. In plain words:** It lets you check "was this model trained on the same data as that one?" without storing the data. The code keeps the first 12 hex characters.
**3. Example to quote:**
> "Two models with the same fingerprint saw identical training text; a different fingerprint means the data changed."

```python
import hashlib
print(hashlib.sha256(b"abc").hexdigest()[:12])
```

**Seen in:** `save`

## String slicing [:12]  `(syntax)`
**1. Definition:** `s[:12]` returns the first 12 characters of a string.
**2. In plain words:** A full SHA-256 is 64 characters; 12 is enough to tell versions apart and is easier to read.
**3. Example to quote:**
> "I keep `hexdigest()[:12]` as a readable fingerprint."

**Seen in:** `save`

## str.encode() before hashing  `(Python built-in)`
**1. Definition:** Hash functions work on bytes, so text must be encoded with `.encode()` first.
**2. In plain words:** `"\n".join(training_texts).encode()` turns all texts into one byte string to hash.
**3. Example to quote:**
> "I join the training texts with newlines and `.encode()` them before calling `sha256`."

**Seen in:** `save`

## log_loss  `(third-party: scikit-learn)`
**1. Definition:** `log_loss(y_true, y_proba, labels=...)` measures how good predicted probabilities are, heavily penalising confident wrong answers; lower is better.
**2. In plain words:** Two models can tie on accuracy yet differ in confidence. Here both reached accuracy 1.00 but v0002's log loss was 0.009 against 0.671 for v0001, showing it was far more confidently correct. Choose metrics that can tell models apart.
**3. Example to quote:**
> "Both versions scored accuracy 1.00 on my tiny test set, but log loss showed v0002 was much better."

**Seen in:** `train_and_register`

## Tie in accuracy on a toy test set  `(concept)`
**1. Definition:** On very small test sets many models reach the same accuracy, so accuracy cannot rank them.
**2. In plain words:** Eight test sentences cannot separate a weak and a strong model. Use probability-aware metrics and bigger, more realistic test sets before promoting a model.
**3. Example to quote:**
> "Accuracy said v1 equals v2, but log loss said v2 is much better, so I'd promote v2 on that evidence."

**Seen in:** `WALKTHROUGH.md`

## joblib.dump / joblib.load for versions  `(third-party: joblib)`
**1. Definition:** Saves each version's fitted pipeline as `model.joblib` and loads it back by path.
**2. In plain words:** The registry stores the pipeline once per version and reads it when you ask for that version.
**3. Example to quote:**
> "`joblib.dump(model, folder / 'model.joblib')` writes the version's artifact, and `load` reads it back."

**Seen in:** `save`, `load`

## raise KeyError for unknown version  `(syntax)`
**1. Definition:** `raise KeyError(message)` signals that a requested key (here a version or alias) was not found.
**2. In plain words:** A wrong version string in a config file is a classic production bug; a clear message listing the valid versions makes it quick to diagnose.
**3. Example to quote:**
> "`reg.load('v0099')` raises `KeyError: \"cannot resolve 'v0099'\"` so I immediately see the typo."

**Seen in:** `promote`, `resolve`

## repr formatting with !r in an error message  `(syntax)`
**1. Definition:** `{version!r}` inside an f-string inserts the quoted `repr` of a value.
**2. In plain words:** Quotes make it obvious whether the problem is an empty string or stray whitespace in the version name.
**3. Example to quote:**
> "`f'unknown version {version!r}; have {self.versions()}'` shows `'v0099'` in quotes and the valid options."

**Seen in:** `promote`

## shutil.rmtree  `(standard library)`
**1. Definition:** Deletes a directory and everything inside it; `ignore_errors=True` suppresses errors such as a missing folder.
**2. In plain words:** The demo clears `registry_demo` first so each run starts clean. It is dangerous with a wrong path, so only point it at folders you created.
**3. Example to quote:**
> "`shutil.rmtree(ROOT, ignore_errors=True)` wipes the demo registry so the run is repeatable."

**Seen in:** `main.py`

## Training and registering in one function  `(concept)`
**1. Definition:** A helper `train_and_register` that trains a pipeline with given hyperparameters, evaluates it and saves it with metrics and parameters.
**2. In plain words:** It shows how training output plugs into the registry: the model, its scores and its settings all land in one version.
**3. Example to quote:**
> "`train_and_register(reg, n=300, C=10.0)` trains a stronger model and saves it as `v0002` with its metrics."

**Seen in:** `main.py`

## Hyperparameters n and C in metadata  `(concept)`
**1. Definition:** Settings chosen before training (here training size `n` and regularization strength `C`) recorded in the version's metadata.
**2. In plain words:** Writing them down lets you reproduce or compare runs later, for example v0001 used n=20, C=0.1 and v0002 used n=300, C=10.0.
**3. Example to quote:**
> "Pinned load of `v0002` still shows `{'n': 300, 'C': 10.0}`, so I know exactly how it was trained."

**Seen in:** `main.py`

## Registry layout on disk  `(concept)`
**1. Definition:** The folder structure the registry produces: `registry/v0001/model.joblib`, `registry/v0001/metadata.json`, and `registry/aliases.json`.
**2. In plain words:** Files and folders are the database here. That makes the registry transparent and easy to inspect, but real tools add search, access control and remote storage.
**3. Example to quote:**
> "The demo ends by printing five files: `aliases.json` plus a model and metadata file for each of two versions."

**Seen in:** `registry.py` docstring, `main.py`

## Version code and data as well as weights  `(concept)`
**1. Definition:** A registry versions model files, but reproducibility also needs the training code and data to be versioned.
**2. In plain words:** A versioned weights file with unknown code or data cannot be reproduced. The data fingerprint and library versions are a partial answer; Git for code and data versioning tools cover the rest.
**3. Example to quote:**
> "The registry stores a data fingerprint and library versions, but I still keep the training script in Git."

**Seen in:** `WALKTHROUGH.md` "Say it aloud"
