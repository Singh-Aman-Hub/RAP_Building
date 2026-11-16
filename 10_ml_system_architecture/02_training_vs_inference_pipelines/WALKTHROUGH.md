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

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Training pipeline | offline pipeline, batch training job, model build | Near-synonyms | Heavy, occasional code that reads data and writes an artifact. |
| Inference pipeline | serving pipeline, online pipeline, prediction service | Near-synonyms | Light, per-request code that only loads the artifact and predicts. |
| Training/serving skew | train-serve skew, preprocessing mismatch | Same idea | Training and serving transform inputs differently, so the model quietly degrades. |
| Data drift | covariate shift, input drift | Near-synonyms | Not skew: skew is a code mismatch; drift is the real world changing over time (step 09). |
| Artifact | model file, checkpoint, bundle | Near-synonyms | Here `model.joblib` plus `metadata.json`: the handover between the two pipelines. |
| Postprocessing | decision rule, thresholding | Near-synonyms | Turning probabilities into the final answer, e.g. 'uncertain' below 0.6. |
| `C` in LogisticRegression | inverse regularisation strength, hyperparameter | No relation to class labels | Small C = stronger regularisation (simpler model); a hyperparameter, not learned. |

## Code, step by step

### 1. Docstring: the training pipeline  (`train.py` lines 1–7)

```python
"""
train.py -- the TRAINING pipeline: OFFLINE, runs occasionally, heavy, produces an artifact.

    data -> split -> fit pipeline -> evaluate -> save model.joblib + metadata.json

Run alone:  python train.py            (writes ./artifacts/model.joblib)
"""
```

- **In simple words:** A label saying this is the slow, once-in-a-while job that produces the model file.
- **Exactly:** Describes the flow data -> split -> fit -> evaluate -> save.
- **Why it is here:** Names the role of the file so nobody puts serving code here.

### 2. Imports and borrowing data.py from step 01  (`train.py` lines 9–22)

```python
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_pipeline_end_to_end"))
from data import get_test_data, get_training_data  # noqa: E402
```

- **In simple words:** Fetch tools, and add step 01's folder to Python's search path so we can reuse its data.
- **Exactly:** `sys.path.insert(0, path)` puts a directory first in the module search list; `Path(__file__).resolve().parents[1]` is the folder one level above this step. `# noqa: E402` tells linters the late import is intentional. `datetime`/`timezone` stamp the time; `json` writes metadata; `sklearn` is imported for its `__version__`.
- **Why it is here:** Without the path line, `from data import ...` raises `ModuleNotFoundError`, since data.py is in a sibling folder.

### 3. Where artifacts are saved  (`train.py` lines 24–24)

```python
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
```

- **In simple words:** Pick the output folder, next to this file.
- **Exactly:** `Path(__file__).resolve().parent / "artifacts"` builds an absolute path; `/` joins path parts.
- **Why it is here:** Using the script's own location instead of the current directory makes it work wherever you run it from.

### 4. Start train(): load train and test data  (`train.py` lines 27–30)

```python
def train(n_samples: int = 200, C: float = 1.0) -> dict:
    """Train, evaluate, save. Returns the metadata dict. C = inverse regularization strength."""
    X_train, y_train = get_training_data(n=n_samples)
    X_test, y_test = get_test_data()
```

- **In simple words:** A function that does the whole training job; first get the data.
- **Exactly:** Signature `train(n_samples=200, C=1.0) -> dict`. `get_training_data(n=200)` gives two lists of 200; `get_test_data()` gives two of 8.
- **Why it is here:** Parameters make the function reusable for experiments (versioning step 04).

### 5. Fit the pipeline and measure accuracy  (`train.py` lines 32–34)

```python
    pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(C=C, max_iter=1000))])
    pipeline.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, pipeline.predict(X_test))
```

- **In simple words:** Teach the model, then mark it on the unseen exam.
- **Exactly:** `Pipeline` = TF-IDF then LogisticRegression, fitted on raw text. `accuracy_score(y_test, pipeline.predict(X_test))` = correct/8. Real value: `1.0`.
- **Why it is here:** Fitting the whole pipeline (not just the model) keeps preprocessing inside the artifact.

### 6. Save the model file  (`train.py` lines 36–37)

```python
    ARTIFACT_DIR.mkdir(exist_ok=True)                       # create folder if missing (no error if present)
    joblib.dump(pipeline, ARTIFACT_DIR / "model.joblib")
```

- **In simple words:** Create the folder if needed and write the trained pipeline to disk.
- **Exactly:** `mkdir(exist_ok=True)` does not raise if it exists. `joblib.dump(obj, path)` pickles the whole pipeline to `artifacts/model.joblib`.
- **Why it is here:** This file is the only thing handed to the inference pipeline.

### 7. Write metadata.json next to the model  (`train.py` lines 38–48)

```python
    metadata = {                                            # everything you'd want to know about this artifact later
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train": len(X_train),
        "params": {"C": C},
        "metrics": {"accuracy": accuracy},
        "classes": [str(c) for c in pipeline.classes_],
        "sklearn_version": sklearn.__version__,             # a model must be loaded with a compatible version
        "python_version": sys.version.split()[0],
    }
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata
```

- **In simple words:** Attach a label to the box: when, how big, which settings, how good, which library versions.
- **Exactly:** Dict with created_at (UTC ISO string like `2026-11-05T07:44:18+00:00`), n_train=200, params `{'C': 1.0}`, metrics `{'accuracy': 1.0}`, classes `['negative','positive']`, sklearn_version `1.9.1`, python_version `3.12.13`. `sys.version.split()[0]` takes just the number. `json.dumps(indent=2)` pretty-prints; `write_text` saves; the function returns the dict.
- **Why it is here:** Months later you must know how the model was made, and a model should be loaded under a compatible sklearn version.

### 8. Run train() when executed as a script  (`train.py` lines 51–52)

```python
if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
```

- **In simple words:** Running the file by itself trains and prints the metadata.
- **Exactly:** `__main__` guard; `json.dumps(..., indent=2)` prints the dict.
- **Why it is here:** Lets other files `import train` without triggering training.

### 9. Docstring: the inference pipeline  (`predict.py` lines 1–7)

```python
"""
predict.py -- the INFERENCE pipeline: ONLINE, runs on every request, light, only LOADS the artifact.

    request -> validate -> (preprocess + features + model, all inside the loaded pipeline) -> postprocess -> response

Note what is NOT here: no fit(), no training data. Inference code must never learn anything.
"""
```

- **In simple words:** This file only uses a finished model; it never learns.
- **Exactly:** States the request -> validate -> model -> postprocess -> response flow.
- **Why it is here:** Documents the rule: no `fit()` in inference code.

### 10. Imports and the artifact folder  (`predict.py` lines 9–14)

```python
import json
from pathlib import Path

import joblib

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
```

- **In simple words:** Same artifacts folder as train.py, so both files agree on where the model lives.
- **Exactly:** `json`, `Path`, `joblib`. `ARTIFACT_DIR` defined identically to train.py's.
- **Why it is here:** Two files pointing at different folders would give 'file not found' at serve time.

### 11. load_artifact(): load model and metadata once  (`predict.py` lines 17–21)

```python
def load_artifact(artifact_dir: Path = ARTIFACT_DIR):
    """Load once at startup (slow), reuse for every request (fast)."""
    model = joblib.load(artifact_dir / "model.joblib")
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    return model, metadata
```

- **In simple words:** Open the box a single time at startup.
- **Exactly:** `joblib.load` returns the Pipeline; `json.loads(text)` returns a dict. Returns a tuple `(model, metadata)`. Real run: load took 0.7 ms.
- **Why it is here:** Loading per request would repeat slow disk I/O on every call.

### 12. predict_text(): validate the input  (`predict.py` lines 24–27)

```python
def predict_text(model, text: str, threshold: float = 0.6) -> dict:
    """Validate -> predict -> postprocess. Postprocessing here = turn probabilities into a decision."""
    if not isinstance(text, str) or not text.strip():            # 1. validate input
        raise ValueError("text must be a non-empty string")
```

- **In simple words:** Refuse empty or non-text input before wasting model time.
- **Exactly:** `isinstance(text, str)` checks type; `text.strip()` is empty for whitespace-only input; `raise ValueError(...)`.
- **Why it is here:** Bad input would otherwise produce a meaningless prediction or a cryptic crash.

### 13. Predict, then turn probabilities into a decision  (`predict.py` lines 28–32)

```python
    probs = model.predict_proba([text])[0]                       # 2. model (preprocess+features inside)
    best = int(probs.argmax())
    confidence = float(probs[best])
    label = str(model.classes_[best]) if confidence >= threshold else "uncertain"   # 3. postprocess rule
    return {"label": label, "confidence": round(confidence, 3)}
```

- **In simple words:** Ask the model, take its best guess, but say 'uncertain' if it is not confident enough.
- **Exactly:** `predict_proba([text])[0]` has shape (2,) summing to 1. `argmax()` is the index of the largest; `model.classes_[best]` is its label. If `confidence < threshold` (0.6) the label becomes `'uncertain'`. Real run: 'What a superb hotel' -> positive 0.816; 'It arrived on Tuesday' -> uncertain 0.526.
- **Why it is here:** A 52% guess presented as a firm answer is worse than admitting doubt.

### 14. Run a demo when executed directly  (`predict.py` lines 35–38)

```python
if __name__ == "__main__":
    model, meta = load_artifact()
    print("loaded model trained at", meta["created_at"])
    print(predict_text(model, "What a superb hotel"))
```

- **In simple words:** Load the artifact and make one prediction.
- **Exactly:** `__main__` guard; prints the training time stamp and a prediction dict.
- **Why it is here:** Shows the inference flow in isolation.

### 15. Docstring: goal, run command, expected output  (`main.py` lines 1–14)

```python
"""
GOAL:   See why training and inference are two DIFFERENT pipelines that share ONE artifact,
        and how the artifact produced here plugs straight into the API from module 09.

RUN:    cd 10_ml_system_architecture/02_training_vs_inference_pipelines && python main.py

EXPECTED OUTPUT (shape):
        TRAINING pipeline  (offline): ... accuracy 1.0
        INFERENCE pipeline (online) : loaded in X ms, then per-request ~0.2 ms
        <predictions incl. one 'uncertain'>
        To serve this model with the module-09 API: MODEL_PATH=... uvicorn ...

NEEDS:  scikit-learn, joblib
"""
```

- **In simple words:** What this demo shows.
- **Exactly:** Describes the two pipelines sharing one artifact.
- **Why it is here:** Orientation.

### 16. Imports: time and both pipelines  (`main.py` lines 16–19)

```python
import time

from predict import ARTIFACT_DIR, load_artifact, predict_text
from train import train
```

- **In simple words:** Bring in both halves.
- **Exactly:** `from predict import ...` and `from train import train` import the two modules we just walked.
- **Why it is here:** Keeps the demo short: it only orchestrates.

### 17. Run and time the TRAINING pipeline  (`main.py` lines 21–25)

```python
if __name__ == "__main__":
    t0 = time.perf_counter()
    meta = train()
    print(f"TRAINING pipeline  (offline): took {(time.perf_counter()-t0)*1000:.0f} ms, "
          f"accuracy {meta['metrics']['accuracy']}, classes {meta['classes']}")
```

- **In simple words:** Train the model and see how long the offline job takes.
- **Exactly:** `time.perf_counter()` is a high-resolution clock; elapsed x 1000 = ms. `train()` returns metadata. Real output: `took 21 ms, accuracy 1.0, classes ['negative', 'positive']`.
- **Why it is here:** Shows training is the slower, once-only part.

### 18. Load the artifact and time it  (`main.py` lines 27–29)

```python
    t0 = time.perf_counter()
    model, meta = load_artifact()
    print(f"INFERENCE pipeline (online) : artifact loaded in {(time.perf_counter()-t0)*1000:.1f} ms (once, at startup)")
```

- **In simple words:** Open the saved model as a server would at startup.
- **Exactly:** `load_artifact()` returns `(model, meta)`. Real: 0.7 ms.
- **Why it is here:** Separates 'startup cost' from 'per-request cost'.

### 19. Predict three texts and time each  (`main.py` lines 31–34)

```python
    for text in ["What a superb hotel", "The delivery was dreadful", "It arrived on Tuesday"]:
        t0 = time.perf_counter()
        result = predict_text(model, text)
        print(f"  {text!r:<30} -> {result}   ({(time.perf_counter()-t0)*1000:.2f} ms)")
```

- **In simple words:** Send three requests through the inference pipeline.
- **Exactly:** `{text!r:<30}` = repr padded to 30 chars. Real results: 'What a superb hotel' -> positive 0.816 (0.75 ms); 'The delivery was dreadful' -> negative 0.739; 'It arrived on Tuesday' -> uncertain 0.526.
- **Why it is here:** Per-request latency is what users feel; the third shows the threshold rule.

### 20. Print how to serve it with module 09, plus try-it notes  (`main.py` lines 36–40)

```python
    print("\nTo serve this exact model with the module-09 API:")
    print(f"  cd 09_api_development/06_model_api_flask_vs_fastapi")
    print(f"  MODEL_PATH={ARTIFACT_DIR}/model.joblib uvicorn fastapi_app:app --port 8000")
    # TRY IT: change threshold in predict_text to 0.9 and see more 'uncertain' answers.
    # TRY IT: call train(n_samples=10) then re-run predictions: how does confidence change?
```

- **In simple words:** Tell the reader how to plug this file into the web API.
- **Exactly:** Prints a `MODEL_PATH=... uvicorn fastapi_app:app --port 8000` command containing the absolute path to `artifacts/model.joblib`. Lines 39-40 are comments.
- **Why it is here:** Connects module 10 to module 09: the artifact is the contract between them.

### 21. Docstring: two kinds of skew, one silent, one loud  (`broken_skew.py` lines 1–10)

```python
"""
broken_skew.py -- EXPECTED TO CRASH at the end. Demonstrates TRAINING/SERVING SKEW, the #1 silent ML bug.

RUN: python broken_skew.py

Part A (silent!): training used lowercase text, serving forgot to lowercase -> the model quietly gets
                  worse. No error, no warning. You only notice through monitoring (step 07).
Part B (loud):    refitting the vectorizer at inference time builds a DIFFERENT vocabulary, so the
                  model's weights no longer line up with the columns -> a crash.
"""
```

- **In simple words:** Explains that this file is meant to crash at the end.
- **Exactly:** Part A: silent accuracy drop. Part B: refitting the vectoriser gives a different vocabulary -> `ValueError`.
- **Why it is here:** Tells the reader the traceback is expected.

### 22. Imports, borrowed data, load both datasets  (`broken_skew.py` lines 12–22)

```python
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_pipeline_end_to_end"))
from data import get_test_data, get_training_data  # noqa: E402

X_train, y_train = get_training_data()
X_test, y_test = get_test_data()
```

- **In simple words:** Same setup as train.py, then fetch 200 train and 8 test sentences.
- **Exactly:** `X_train`, `y_train` (200 each), `X_test`, `y_test` (8 each).
- **Why it is here:** Same data as the working demo, so differences come from the bug only.

### 23. Part A: train on lowercase text, vectoriser not lowercasing  (`broken_skew.py` lines 24–27)

```python
# ---- Part A: preprocessing mismatch ----
train_texts = [t.lower() for t in X_train]                  # TRAINING lowercases ...
vec = TfidfVectorizer(lowercase=False)                      # ... and the vectorizer does NOT (we did it by hand)
clf = LogisticRegression(max_iter=1000).fit(vec.fit_transform(train_texts), y_train)
```

- **In simple words:** Cleaning done by hand before training; the vectoriser is told not to do it.
- **Exactly:** `TfidfVectorizer(lowercase=False)` keeps case, so 'Great' and 'great' would be different words; vocabulary here is learned from already-lowercased text. `LogisticRegression(...).fit(...)` is chained: `fit` returns the model.
- **Why it is here:** This reproduces the real-life situation where cleaning lives in training code but not in the server.

### 24. Simulate shouting users: correct vs forgetful serving  (`broken_skew.py` lines 29–31)

```python
live = [t.upper() for t in X_test]                          # real users sometimes SHOUT
right = clf.predict(vec.transform([t.lower() for t in live]))     # serving applies the same cleaning as training
wrong = clf.predict(vec.transform(live))                           # serving FORGETS to lowercase -> every word is unseen
```

- **In simple words:** Two serving paths: one lowercases first (correct), one forgets.
- **Exactly:** `t.upper()` makes 'GREAT' etc. `vec.transform` only knows lowercase words, so uppercase tokens are all unseen -> an all-zero vector -> the model guesses from its intercept.
- **Why it is here:** Shows how a missing line in serving code changes the answers without any error.

### 25. Compute and print both accuracies  (`broken_skew.py` lines 32–33)

```python
acc = lambda pred: sum(p == y for p, y in zip(pred, y_test)) / len(y_test)   # lambda = tiny anonymous function
print(f"A) accuracy with consistent preprocessing: {acc(right):.2f}   with skew: {acc(wrong):.2f}  (no error raised!)")
```

- **In simple words:** Mark both paths.
- **Exactly:** `lambda pred: ...` computes correct/total. Real output: `1.00` with consistent preprocessing and `0.50` with skew, and no error raised.
- **Why it is here:** 50% on a 2-class problem is a coin flip; nothing crashed, so only monitoring would notice.

### 26. Part B: refit the vectoriser at inference time (crash)  (`broken_skew.py` lines 35–37)

```python
# ---- Part B: refit at inference ----
new_vec = TfidfVectorizer().fit(X_test)                     # BUG: fit on the test data -> a different vocabulary
clf.predict(new_vec.transform(X_test))                      # <- ValueError: X has N features, but LogisticRegression is expecting M
```

- **In simple words:** Build a brand-new vocabulary from the test data, then ask the old model to use it.
- **Exactly:** `TfidfVectorizer().fit(X_test)` learns a vocabulary of 36 words (the test set's), not the training 32. `clf.predict` expects 32 columns. Real error: `ValueError: X has 36 features, but LogisticRegression is expecting 32 features as input.`
- **Why it is here:** Loud failure is the lucky kind of bug; read the message as 'columns do not match what the model was trained on'.


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
