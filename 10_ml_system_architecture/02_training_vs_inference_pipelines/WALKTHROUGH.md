# Step 02 — Training pipeline vs inference pipeline

## The story so far
Step 01 ran everything in one script. In real life training happens offline on a schedule and inference runs online all day, as two separate programs.

## The pain point
If serving cleans text differently from training (say it forgets to lowercase), the model quietly gets worse. No error, no warning, only unhappy users weeks later.

## The spark
Treat them as two pipelines joined by exactly one thing, the saved artifact. Keep preprocessing inside that artifact so both sides run identical code, and record metadata and library versions.

## Goal
See that training (offline, heavy, occasional) and inference (online, light, constant) are different programs joined by **one artifact**, and learn the classic bug when they disagree: **training/serving skew**.

## Run it
```bash
cd 10_ml_system_architecture/02_training_vs_inference_pipelines
python train.py          # writes artifacts/model.joblib + metadata.json
python predict.py        # loads and predicts
python main.py           # both, with timings
python broken_skew.py    # EXPECTED to end with a ValueError traceback
```
Real output of `main.py`:
```
TRAINING pipeline  (offline): took 12 ms, accuracy 1.0, classes ['negative', 'positive']
INFERENCE pipeline (online) : artifact loaded in 0.4 ms (once, at startup)
  'What a superb hotel'          -> {'label': 'positive', 'confidence': 0.816}   (0.44 ms)
  'The delivery was dreadful'    -> {'label': 'negative', 'confidence': 0.739}   (0.30 ms)
  'It arrived on Tuesday'        -> {'label': 'uncertain', 'confidence': 0.526}   (0.28 ms)
```
Real output of `broken_skew.py`:
```
A) accuracy with consistent preprocessing: 1.00   with skew: 0.50  (no error raised!)
...
ValueError: X has 36 features, but LogisticRegression is expecting 32 features as input.
```

## Side-by-side
| | Training pipeline | Inference pipeline |
|---|---|---|
| When | scheduled / on demand | every request |
| Input | big labelled dataset | one text (or a batch) |
| Does | `fit`, evaluate, save | load, validate, `predict_proba`, postprocess |
| Cost | minutes–days, GPUs | milliseconds, must be cheap |
| Failure impact | a delayed model | users see errors now |
| Shares | **preprocessing + feature code + the artifact** | |
`artifacts/model.joblib` is the handoff. It can be served by the module-09 API: `MODEL_PATH=.../artifacts/model.joblib uvicorn fastapi_app:app` (the exact command is printed by `main.py`).

## Syntax explained
| Syntax | Meaning | Example |
|---|---|---|
| `ARTIFACT_DIR.mkdir(exist_ok=True)` | Create a folder, no error if it exists | — |
| `json.dumps(obj, indent=2)` / `Path.write_text` | Pretty-print JSON / write text file | — |
| `raise ValueError("...")` | Signal invalid input yourself | — |
| `datetime.now(timezone.utc).isoformat()` | Timezone-aware timestamp string | — |
| `lambda p: sum(...)` | Small anonymous function | `(lambda x: x+1)(2)` |
| `probs.argmax()` | Index of the largest value (numpy) | — |

## Libraries & functions
| Name | Kind | Does |
|---|---|---|
| `sklearn.__version__` | 3rd-party | Record library versions in metadata — a model should be loaded with a compatible version |
| `joblib.dump(model, path)` | 3rd-party | Save the whole fitted pipeline |
| `pipeline.classes_` | sklearn | Label order matching `predict_proba` columns |
| `probs[best]` | numpy | Probability of the winning class (used as "confidence") |

## Execution trace
`predict_text(model, "It arrived on Tuesday")`: validate (non-empty str) → `predict_proba` → `[0.474, 0.526]` → best = positive at 0.526 → confidence < threshold 0.6 → postprocess rule returns `"uncertain"`. Postprocessing is where business rules live (thresholds, abstaining, mapping labels to actions).

## Training/serving skew — the #1 silent ML bug
- **Part A (silent):** training lowercased the text; serving received `SHOUTING` text and did not → every word unseen → accuracy falls from 1.00 to 0.50 **with no error**. You catch it only with monitoring (step 07) or a test that runs the *same* function in both places.
- **Part B (loud):** refitting a vectorizer at inference creates a different vocabulary (36 vs 32 features) → shape mismatch crash.
Prevention: put preprocessing **inside** the saved pipeline, share code (not copies), pin versions, test the loaded artifact on a fixed sample before deploy.

## Experiments
1. `predict_text(..., threshold=0.9)` → many more `"uncertain"`.
2. `train(n_samples=10)` then re-run predictions → lower confidences.
3. Copy the artifact folder, change `metadata.json`'s `sklearn_version`, and write a check in `load_artifact` that warns when it differs from the installed version.

## Debugging practice
`broken_skew.py` Part B: read the traceback from the bottom: `ValueError: X has 36 features, but LogisticRegression is expecting 32` — the numbers tell you the vocabulary changed between fit and predict. Search the code for `.fit(` calls that run at inference time.

## Rebuild checklist
- [ ] Write `train()` and `predict_text()` as two functions sharing only the saved file.
- [ ] Name three things that must be identical in training and serving.

## The new problem (cliffhanger)
The artifact is now the handoff between two programs, so how it is written to disk matters (format, safety, versions). Next: [03_serialization](../03_serialization/).

## Say it aloud
> **Say it aloud (interview box):** Training is an offline job that fits and evaluates a model and saves an artifact, while inference is an online service that loads that artifact once and answers requests quickly. They must share the same preprocessing, otherwise training/serving skew fails silently. The cost is that preprocessing lives inside the artifact, so changing it means retraining and redeploying.
