# Step 04 — Model registry and versioning

## The story so far
Step 03 saved models safely, but a single file name means each retrain destroys the previous model.

## The pain point
When a new model is worse, you cannot roll back because the old file is gone. Nobody can say which model answered last Tuesday's complaint.

## The spark
Store every model as an immutable numbered version with metadata (metrics, parameters, data fingerprint, library versions), and let an alias like `production` point at one version.

## Goal
Treat every trained model as an immutable, numbered, documented artifact; move a `production` label between versions; roll back instantly.

## Run it
```bash
cd 10_ml_system_architecture/04_model_registry_versioning && python main.py
```
Real output:
```
saved v0001  n=20   C=0.1   accuracy=1.00 log_loss=0.671
saved v0002  n=300  C=10.0  accuracy=1.00 log_loss=0.009

production -> v0001 | latest -> v0002
loaded v0001 metrics {'accuracy': 1.0, 'log_loss': 0.671} data fingerprint 6ebb39635e9d
after promote  -> v0002
after rollback -> v0001
pinned load of v0002 still works: {'n': 300, 'C': 10.0}
unknown version -> "cannot resolve 'v0099'"

Files: ['aliases.json', 'v0001/metadata.json', 'v0001/model.joblib', 'v0002/metadata.json', 'v0002/model.joblib']
```
Note both models tie on accuracy (a toy test set); **log loss** shows v2 is far more confident and correct — pick metrics that can tell models apart.

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Model registry | model store, model catalogue (MLflow Registry, W&B Registry, HF Hub, SageMaker Model Registry) | Near-synonyms | A place that stores immutable numbered model versions plus metadata. |
| Alias | stage, tag, label, pointer (`production`, `staging`) | Near-synonyms | A movable name that points at a version; versions never move. |
| Version | model ID, run ID, revision | Near-synonyms | An immutable snapshot; `v0001` is never overwritten. |
| Promote | deploy to production, transition stage, set alias | Near-synonyms | Moving an alias to a version; here a JSON edit. |
| Rollback | revert, re-promote old version | Same thing here | Moving the alias back to the previous version. |
| Data fingerprint | dataset hash, data version, checksum | Near-synonyms | A short hash identifying the exact training data. |
| `resolve` vs `load` | look up vs open | No | `resolve` turns a reference into a version name; `load` also reads the files. |

## Code, step by step

### 1. Docstring: layout of the registry on disk  (`registry.py` lines 1–14)

```python
"""
registry.py -- a minimal MODEL REGISTRY on the local filesystem. Real tools (MLflow, W&B, SageMaker, HF Hub)
do the same job with more features.

Layout it creates:
    registry/
      v0001/model.joblib      <- the artifact (never modified after saving)
      v0001/metadata.json     <- metrics, params, library versions, creation time, data fingerprint
      v0002/...
      aliases.json            <- {"production": "v0001", "staging": "v0002"}  movable labels

Why aliases? Code says "load production"; ops moves the label. Rolling back = moving the label back,
no redeploy of code needed.
"""
```

- **In simple words:** A map of the folder: version folders plus one aliases file.
- **Exactly:** Documents `v0001/model.joblib`, `v0001/metadata.json`, `aliases.json`.
- **Why it is here:** The layout is the whole design.

### 2. Imports  (`registry.py` lines 16–22)

```python
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
```

- **In simple words:** Tools for hashing, JSON, time, paths, saving models, and version info.
- **Exactly:** `hashlib` for SHA-256; `joblib`; `sklearn` for `__version__`.
- **Why it is here:** Each is used in `save`.

### 3. ModelRegistry.__init__: make the root folder  (`registry.py` lines 25–29)

```python
class ModelRegistry:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.alias_file = self.root / "aliases.json"
```

- **In simple words:** Create the registry's home folder and remember where aliases are kept.
- **Exactly:** `Path(root)` converts str to Path; `mkdir(parents=True, exist_ok=True)` creates all missing parents silently. `alias_file` is `root/aliases.json`.
- **Why it is here:** The registry works on first use without setup.

### 4. versions(): list version folders in order  (`registry.py` lines 31–34)

```python
    # ---------- write side ----------
    def versions(self) -> list[str]:
        """All version names sorted oldest -> newest ('v0001' sorts correctly because of zero padding)."""
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and p.name.startswith("v"))
```

- **In simple words:** Find all folders named v0001, v0002 ... oldest first.
- **Exactly:** `iterdir()` lists entries; keeps directories starting with `v`; `sorted` is alphabetical which equals numeric order because of zero padding (`v0002` < `v0010`).
- **Why it is here:** Without padding `v10` would sort before `v2`.

### 5. save(): create the next version folder and store the model  (`registry.py` lines 36–41)

```python
    def save(self, model, metrics: dict, params: dict, training_texts: list[str]) -> str:
        n = len(self.versions()) + 1
        version = f"v{n:04d}"                               # f-string format: pad number to 4 digits with zeros
        folder = self.root / version
        folder.mkdir()
        joblib.dump(model, folder / "model.joblib")
```

- **In simple words:** Take the next number, make its folder, put the model inside.
- **Exactly:** `n = len(versions()) + 1`; `f"v{n:04d}"` formats as 4 digits, zero padded (`v0001`). `folder.mkdir()` raises if it exists, so versions are never overwritten. `joblib.dump` writes `model.joblib`.
- **Why it is here:** Immutability: you can always go back.

### 6. Compute a data fingerprint  (`registry.py` lines 42–42)

```python
        fingerprint = hashlib.sha256("\n".join(training_texts).encode()).hexdigest()[:12]   # identifies the training data
```

- **In simple words:** A short ID card for the training data.
- **Exactly:** `"\n".join(texts).encode()` -> bytes; `sha256(...).hexdigest()` is 64 hex chars; `[:12]` keeps the first 12. Real: `6ebb39635e9d` for the 20-text model. Same data -> same fingerprint.
- **Why it is here:** Lets you tell whether two models were trained on the same data.

### 7. Write metadata.json and return the version name  (`registry.py` lines 43–51)

```python
        (folder / "metadata.json").write_text(json.dumps({
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "metrics": metrics,
            "params": params,
            "data_fingerprint": fingerprint,
            "sklearn_version": sklearn.__version__,
        }, indent=2))
        return version
```

- **In simple words:** Attach the model's record card and give back its name.
- **Exactly:** JSON dict: version, created_at, metrics, params, data_fingerprint, sklearn_version. `write_text(json.dumps(..., indent=2))`. Returns e.g. `'v0001'`.
- **Why it is here:** Without metadata, a model file is anonymous.

### 8. _aliases(): read aliases.json or return empty  (`registry.py` lines 53–54)

```python
    def _aliases(self) -> dict:
        return json.loads(self.alias_file.read_text()) if self.alias_file.exists() else {}
```

- **In simple words:** Read the label list; if no labels yet, nothing.
- **Exactly:** Conditional expression: `json.loads(...)` if the file exists else `{}`. Returns a dict.
- **Why it is here:** Handles first use, before any promote.

### 9. promote(): point an alias at a version  (`registry.py` lines 56–61)

```python
    def promote(self, version: str, alias: str = "production") -> None:
        if version not in self.versions():
            raise KeyError(f"unknown version {version!r}; have {self.versions()}")
        aliases = self._aliases()
        aliases[alias] = version
        self.alias_file.write_text(json.dumps(aliases, indent=2))
```

- **In simple words:** Stick the label 'production' on a chosen version.
- **Exactly:** Raises `KeyError` for unknown versions (guard). Reads aliases, sets `aliases[alias] = version`, rewrites the file. Only the JSON changes; model files are untouched.
- **Why it is here:** Promotion and rollback are one edit, no code redeploy.

### 10. resolve(): turn a reference into a version  (`registry.py` lines 63–73)

```python
    # ---------- read side ----------
    def resolve(self, ref: str = "latest") -> str:
        """Turn 'latest' / 'production' / 'v0002' into a concrete version name."""
        if ref == "latest":
            return self.versions()[-1]
        aliases = self._aliases()
        if ref in aliases:
            return aliases[ref]
        if ref in self.versions():
            return ref                                      # pinned exact version
        raise KeyError(f"cannot resolve {ref!r}")
```

- **In simple words:** Accept 'latest', a label like 'production', or an exact version.
- **Exactly:** Order: `latest` -> last of `versions()`; alias lookup; exact version; else `KeyError`. Real: after promoting v1, `resolve('production')` = `v0001` while `resolve('latest')` = `v0002`.
- **Why it is here:** Callers need not know numbers; they say 'production'.

### 11. load(): resolve, then read model and metadata  (`registry.py` lines 75–78)

```python
    def load(self, ref: str = "latest"):
        version = self.resolve(ref)
        folder = self.root / version
        return joblib.load(folder / "model.joblib"), json.loads((folder / "metadata.json").read_text())
```

- **In simple words:** Open the right box and return both the model and its record.
- **Exactly:** Returns the tuple `(model, metadata_dict)`.
- **Why it is here:** One call gives everything serving needs.

### 12. Docstring: goal and expected output  (`main.py` lines 1–13)

```python
"""
GOAL:   Practise MODEL VERSIONING: save every trained model as an immutable numbered version with metadata,
        promote one to "production" with an alias, load by latest / alias / pinned version, then ROLL BACK.

RUN:    cd 10_ml_system_architecture/04_model_registry_versioning && python main.py
        (recreates ./registry_demo each run)

EXPECTED OUTPUT (shape):
        saved v0001 accuracy=...   saved v0002 accuracy=...
        production -> v0001 ... after promote -> v0002 ... after rollback -> v0001

NEEDS:  scikit-learn, joblib
"""
```

- **In simple words:** What the demo does: save two versions, promote, roll back.
- **Exactly:** Header comment.
- **Why it is here:** Orientation.

### 13. Imports, registry class and step 01 data  (`main.py` lines 15–27)

```python
import shutil
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss   # log_loss: lower = more confident AND correct
from sklearn.pipeline import Pipeline

from registry import ModelRegistry

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_pipeline_end_to_end"))
from data import get_test_data, get_training_data  # noqa: E402
```

- **In simple words:** Bring in sklearn parts, metrics and our registry.
- **Exactly:** `log_loss` measures confidence-weighted error (lower is better); `shutil` deletes folders; `ModelRegistry` from registry.py.
- **Why it is here:** Accuracy ties on this tiny test; log loss separates the models.

### 14. Registry location  (`main.py` lines 29–29)

```python
ROOT = Path(__file__).resolve().parent / "registry_demo"
```

- **In simple words:** Where the demo registry lives.
- **Exactly:** `registry_demo` folder next to main.py.
- **Why it is here:** Keeps the demo's files in one place.

### 15. train_and_register(): train and score one model  (`main.py` lines 32–36)

```python
def train_and_register(registry: ModelRegistry, n: int, C: float) -> str:
    X, y = get_training_data(n=n)
    X_test, y_test = get_test_data()
    model = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(C=C, max_iter=1000))]).fit(X, y)
    acc = accuracy_score(y_test, model.predict(X_test))
```

- **In simple words:** Train with given data size and regularisation, then mark it.
- **Exactly:** `n` samples, `C` strength. Accuracy via `accuracy_score`.
- **Why it is here:** A helper so we can create two different models.

### 16. Compute log loss, save to registry, print  (`main.py` lines 37–40)

```python
    ll = log_loss(y_test, model.predict_proba(X_test), labels=list(model.classes_))   # accuracy ties here; log loss does not
    version = registry.save(model, metrics={"accuracy": acc, "log_loss": round(ll, 3)}, params={"n": n, "C": C}, training_texts=X)
    print(f"saved {version}  n={n:<4} C={C:<5} accuracy={acc:.2f} log_loss={ll:.3f}")
    return version
```

- **In simple words:** Also measure log loss, store the model with its numbers, announce it.
- **Exactly:** `log_loss(y_true, probs, labels=...)`: mean of -log(probability given to the true class). Real: v0001 (n=20, C=0.1) log_loss=0.671; v0002 (n=300, C=10.0) log_loss=0.009; both accuracy 1.00. `registry.save` returns the version string.
- **Why it is here:** Accuracy alone cannot choose between them; log loss shows v0002 is far more confident and correct.

### 17. Start clean and create two models  (`main.py` lines 43–48)

```python
if __name__ == "__main__":
    shutil.rmtree(ROOT, ignore_errors=True)               # start clean so the demo is repeatable
    reg = ModelRegistry(ROOT)

    v1 = train_and_register(reg, n=20, C=0.1)             # a weak first model
    v2 = train_and_register(reg, n=300, C=10.0)           # a stronger second model
```

- **In simple words:** Delete any old demo registry, then train a weak and a strong model.
- **Exactly:** `shutil.rmtree(..., ignore_errors=True)`. v1 = 20 samples, strong regularisation; v2 = 300 samples, weak regularisation.
- **Why it is here:** Repeatable demo.

### 18. Promote v1 to production and resolve names  (`main.py` lines 50–51)

```python
    reg.promote(v1, "production")                          # v1 is live
    print("\nproduction ->", reg.resolve("production"), "| latest ->", reg.resolve("latest"))
```

- **In simple words:** Label v1 as live; show what 'production' and 'latest' mean.
- **Exactly:** Real: `production -> v0001 | latest -> v0002`.
- **Why it is here:** Shows alias vs latest are different things.

### 19. Load production and print its record  (`main.py` lines 53–54)

```python
    model, meta = reg.load("production")
    print("loaded", meta["version"], "metrics", meta["metrics"], "data fingerprint", meta["data_fingerprint"])
```

- **In simple words:** Fetch the live model.
- **Exactly:** Real: `loaded v0001 metrics {'accuracy': 1.0, 'log_loss': 0.671} data fingerprint 6ebb39635e9d`.
- **Why it is here:** This is what a server does at startup.

### 20. Roll forward, roll back, load a pinned version  (`main.py` lines 56–60)

```python
    reg.promote(v2, "production")                          # roll FORWARD: one JSON edit, no code change
    print("after promote  ->", reg.resolve("production"))
    reg.promote(v1, "production")                          # ROLL BACK when v2 misbehaves in production
    print("after rollback ->", reg.resolve("production"))
    print("pinned load of v0002 still works:", reg.load("v0002")[1]["params"])
```

- **In simple words:** Move the label to v2, then back to v1; v2 is still reachable.
- **Exactly:** Real: `after promote -> v0002`, `after rollback -> v0001`, `pinned load of v0002 still works: {'n': 300, 'C': 10.0}`.
- **Why it is here:** Rollback in one edit is the point of a registry.

### 21. Try loading a version that does not exist  (`main.py` lines 61–64)

```python
    try:
        reg.load("v0099")                                  # BREAK IT: read the KeyError message
    except KeyError as err:
        print("unknown version ->", err)
```

- **In simple words:** See the error for a bad reference.
- **Exactly:** `KeyError` caught. Real: `unknown version -> "cannot resolve 'v0099'"` (KeyError messages are shown quoted).
- **Why it is here:** Fails with a clear message.

### 22. List every file created, then try-it notes  (`main.py` lines 65–67)

```python
    print("\nFiles:", sorted(str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file()))
    # TRY IT: add a promote(..., "staging") alias and load("staging").
    # TRY IT: make train_and_register refuse to promote if accuracy < the current production accuracy.
```

- **In simple words:** Show what is on disk.
- **Exactly:** `rglob("*")` walks recursively. Real: `['aliases.json', 'v0001/metadata.json', 'v0001/model.joblib', 'v0002/metadata.json', 'v0002/model.joblib']`.
- **Why it is here:** Shows the registry is just folders and one JSON file.


## Why version models?
- **Reproducibility:** which data + params + library produced the model answering users? (`data_fingerprint`, `params`, `sklearn_version` in `metadata.json`).
- **Rollback:** a bad model is fixed by moving a label, not by redeploying code.
- **Comparison / A-B tests:** run v1 and v2 side by side, log `model_version` with every prediction (step 05).
- **Audit:** who/when/what.
Ways to reference a model: `latest` (convenient, risky — changes under you), **pinned** `v0002` (reproducible), **alias** `production`/`staging` (flexible, controlled). Production should use pinned or alias, rarely `latest`.
Real registries: MLflow Model Registry, Weights & Biases, SageMaker, Hugging Face Hub (git tags/revisions).

## Syntax explained
| Syntax | Meaning | Example |
|---|---|---|
| `f"v{n:04d}"` | Format an int padded with zeros to width 4 | `f"{7:04d}"` → `0007` |
| `sorted(p.name for p in root.iterdir() if p.is_dir())` | Generator expression: lazily produce, then sort | — |
| `class ModelRegistry:` with `__init__`, methods | A class bundling data (`root`) and behaviour | — |
| `raise KeyError(msg)` | Signal "not found" | — |
| `hashlib.sha256(bytes).hexdigest()[:12]` | Fingerprint a string; slice first 12 chars | — |
| `shutil.rmtree(path, ignore_errors=True)` | Delete a folder tree (careful!) | — |

## Libraries & functions
| Name | Kind | Does |
|---|---|---|
| `hashlib.sha256` | stdlib | Same input → same short hash; detects "did the training data change?" |
| `log_loss(y_true, proba, labels=...)` | sklearn | Penalizes confident wrong answers; lower = better |
| `Path.iterdir()`, `rglob("*")` | stdlib | List folder / walk recursively |
| `json.loads(path.read_text())` | stdlib | Read JSON file into dict |

## Execution trace
`reg.load("production")`: `resolve("production")` → not "latest" → read `aliases.json` → `{"production": "v0001"}` → folder `registry_demo/v0001` → `joblib.load(model.joblib)` + `json.loads(metadata.json)` → returns `(model, metadata)`.

## Experiments
1. Add a `staging` alias and load it.
2. Make `train_and_register` promote only if `log_loss` beats the current production's.
3. Delete `aliases.json` and call `load("production")` → `KeyError` — design a friendlier error.

## Debugging practice
`KeyError: "cannot resolve 'v0099'"` — a *deliberate* lookup failure. Practise reading the message (what was asked) and the list of valid versions printed by `reg.versions()`. In real incidents "wrong version string in config" is a classic cause.

## Rebuild checklist
- [ ] Save model + metadata into `vNNNN/` folders.
- [ ] Implement `resolve` for `latest`, an alias and a pinned version.
- [ ] Explain why artifacts are immutable.

## The new problem (cliffhanger)
We can now name and roll back models, but we still cannot see what each version actually answered for real users. Next: [05_storage_and_prediction_log](../05_storage_and_prediction_log/).

## Say it aloud
> **Say it aloud (interview box):** I store each trained model as an immutable numbered version with metadata such as metrics, parameters, data fingerprint and library versions. Serving loads a pinned version or an alias like `production`, so promotion and rollback are just label changes. The cost is extra process and storage, and I must version the code and data too, not only the weights.
