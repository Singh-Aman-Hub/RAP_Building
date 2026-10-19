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
