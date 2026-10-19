# Terms — Step 02: Training vs inference pipelines

Folder: `10_ml_system_architecture/02_training_vs_inference_pipelines` · Files covered: `main.py`, `train.py`, `predict.py`, `broken_skew.py`, `artifacts/metadata.json`, `WALKTHROUGH.md`
Back to module: [10 ML system architecture](../README.md)

**Terms in this folder:** [Training pipeline](#training-pipeline-concept) · [Inference pipeline](#inference-pipeline-concept) · [Artifact](#artifact-concept) · [Training/serving skew](#trainingserving-skew-concept) · [Silent failure vs loud failure](#silent-failure-vs-loud-failure-concept) · [Pipeline fit/transform at inference bug](#pipeline-fittransform-at-inference-bug-concept) · [ValueError feature-count mismatch](#valueerror-feature-count-mismatch-python-built-in) · [Postprocessing](#postprocessing-concept) · [Confidence threshold](#confidence-threshold-concept) · [Input validation in inference](#input-validation-in-inference-concept) · [raise ValueError](#raise-valueerror-syntax) · [argmax (probs.argmax())](#argmax-probsargmax-third-party-numpy) · [model.classes_ indexing](#modelclasses_-indexing-third-party-scikit-learn) · [Load once at startup](#load-once-at-startup-concept) · [metadata.json](#metadatajson-concept) · [datetime.now(timezone.utc).isoformat()](#datetimenowtimezoneutcisoformat-standard-library) · [sklearn.__version__ and sys.version](#sklearn__version__-and-sysversion-standard-library) · [Path.mkdir(exist_ok=True)](#pathmkdirexist_oktrue-standard-library) · [Path.write_text and read_text](#pathwrite_text-and-read_text-standard-library) · [json.dumps(indent=2) and json.loads](#jsondumpsindent2-and-jsonloads-standard-library) · [Regularization strength C](#regularization-strength-c-concept) · [Train/test split in the pipeline](#traintest-split-in-the-pipeline-concept) · [lambda for accuracy](#lambda-for-accuracy-syntax) · [TfidfVectorizer(lowercase=False)](#tfidfvectorizerlowercasefalse-third-party-scikit-learn) · [sys.path.insert and # noqa: E402](#syspathinsert-and-noqa-e402-standard-library) · [Serving the artifact with the module-09 API](#serving-the-artifact-with-the-module-09-api-concept) · [time.perf_counter timing](#timeperf_counter-timing-standard-library) · [Training is heavy, inference is light](#training-is-heavy-inference-is-light-concept)

---

## Training pipeline  `(concept)`
**1. Definition:** The offline process that loads data, splits it, fits a model, evaluates it and saves an artifact; it runs occasionally and may be heavy.
**2. In plain words:** Training is the "factory": slow, resource-hungry and run on a schedule or on demand. Its only output that matters to serving is the saved artifact plus metadata describing it.
**3. Example to quote:**
> "`python train.py` fits a Pipeline on 200 examples, evaluates on the test set, and writes `model.joblib` and `metadata.json`."

**Seen in:** `train.py`

## Inference pipeline  `(concept)`
**1. Definition:** The online process that loads the saved artifact once and, for each request, validates input, runs the model and post-processes the result.
**2. In plain words:** Inference is the "shop counter": fast, cheap and always on. It must never learn anything; there is no `fit()` and no training data in `predict.py`.
**3. Example to quote:**
> "My inference code loads the artifact at startup in 0.4 ms and then answers each request in a fraction of a millisecond."

**Seen in:** `predict.py`

## Artifact  `(concept)`
**1. Definition:** The saved output of training that serving needs, here `model.joblib` (the fitted pipeline) and `metadata.json`.
**2. In plain words:** It is the contract between the two pipelines: training produces it, inference consumes it. Keeping it a single file means they cannot disagree about vocabulary or weights.
**3. Example to quote:**
> "The artifact folder is the handoff: the API loads `model.joblib` and I can read `metadata.json` to know how it was trained."

**Seen in:** `train.py`, `predict.py`

## Training/serving skew  `(concept)`
**1. Definition:** A mismatch between how data is processed during training and during serving, causing degraded predictions.
**2. In plain words:** It is the number one silent ML bug. If training lowercased text and serving does not, accuracy can drop from 1.00 to 0.50 with no error at all. Prevent it by keeping preprocessing inside the saved pipeline, sharing code, pinning versions and testing the loaded artifact.
**3. Example to quote:**
> "In `broken_skew.py` forgetting to lowercase at serving halved accuracy, and nothing raised an error; only monitoring would catch it."

**Seen in:** `broken_skew.py`, `WALKTHROUGH.md`

## Silent failure vs loud failure  `(concept)`
**1. Definition:** A silent failure produces wrong results without any error; a loud failure raises an exception or crashes.
**2. In plain words:** Loud failures are easier because you notice them. Part A of the skew demo is silent (accuracy falls), Part B is loud (a shape mismatch `ValueError`). ML systems need monitoring and fixed-sample tests to catch the silent kind.
**3. Example to quote:**
> "The wrong-vocabulary crash is loud and easy to fix; the lowercase mismatch is silent, so I add a regression test on a fixed sample."

**Seen in:** `broken_skew.py`

## Pipeline fit/transform at inference bug  `(concept)`
**1. Definition:** Calling `fit` (or `fit_transform`) on a vectorizer at inference time creates a new vocabulary that does not match the trained model's weights.
**2. In plain words:** The model expects 32 columns but the new vectorizer produced 36, so prediction fails with `ValueError: X has 36 features, but LogisticRegression is expecting 32 features`. The numbers in the message tell you the vocabulary changed.
**3. Example to quote:**
> "`new_vec = TfidfVectorizer().fit(X_test)` is the bug; inference must only `transform`, with the vectorizer saved from training."

**Seen in:** `broken_skew.py` Part B

## ValueError feature-count mismatch  `(Python built-in)`
**1. Definition:** `ValueError` is the exception for an argument with the right type but an invalid value; scikit-learn raises it when the number of input features differs from what the model was trained with.
**2. In plain words:** Read the message from the bottom of the traceback; the two numbers (36 vs 32) point directly at a changed feature space. Search for `.fit(` calls that run at inference time.
**3. Example to quote:**
> "The error 'X has 36 features, but LogisticRegression is expecting 32' told me the vocabulary changed between fit and predict."

**Seen in:** `broken_skew.py`

## Postprocessing  `(concept)`
**1. Definition:** Turning a model's raw output into the final response, for example applying a probability threshold or mapping labels to actions.
**2. In plain words:** This is where business rules live. In `predict_text` a confidence below 0.6 becomes the label `"uncertain"`, so the service can abstain rather than guess.
**3. Example to quote:**
> "'It arrived on Tuesday' has confidence 0.526, below my 0.6 threshold, so the response label is 'uncertain'."

**Seen in:** `predict_text`

## Confidence threshold  `(concept)`
**1. Definition:** A cut-off probability below which the system refuses to give a definite answer.
**2. In plain words:** Raising the threshold makes the service more cautious (more "uncertain" answers) and lowering it makes it answer more often but with more errors. Where you set it is a product decision.
**3. Example to quote:**
> "`threshold=0.9` turned many answers into 'uncertain', trading coverage for reliability."

**Seen in:** `predict_text`

## Input validation in inference  `(concept)`
**1. Definition:** Checking an incoming value is the expected type and non-empty before passing it to the model, raising an error otherwise.
**2. In plain words:** The model should never be asked to handle `None` or an empty string. `isinstance(text, str)` and `text.strip()` guard against that and give a clear `ValueError`.
**3. Example to quote:**
> "`predict_text` raises `ValueError('text must be a non-empty string')` before the model sees bad input."

**Seen in:** `predict_text`

## raise ValueError  `(syntax)`
**1. Definition:** `raise ValueError("message")` signals invalid input by throwing an exception that callers can catch.
**2. In plain words:** It is how your own functions reject bad arguments with a clear reason, instead of crashing later with a confusing error.
**3. Example to quote:**
> "If the text is empty I `raise ValueError`, and the API layer turns that into a 422 response."

**Seen in:** `predict_text`

## argmax (probs.argmax())  `(third-party: numpy)`
**1. Definition:** `array.argmax()` returns the index of the largest value.
**2. In plain words:** The predicted class is the one with the highest probability; the argmax gives its position, which indexes into `model.classes_`.
**3. Example to quote:**
> "`best = int(probs.argmax())` gives index 1 for `[0.474, 0.526]`, which is 'positive'."

```python
import numpy as np
print(np.array([0.2, 0.8]).argmax())   # 1
```

**Seen in:** `predict_text`

## model.classes_ indexing  `(third-party: scikit-learn)`
**1. Definition:** `model.classes_[best]` looks up the label name for the predicted column index.
**2. In plain words:** Without it you would only have a number. It turns index 1 into `'positive'`.
**3. Example to quote:**
> "`str(model.classes_[best])` converts the winning index into the label text."

**Seen in:** `predict_text`

## Load once at startup  `(concept)`
**1. Definition:** `load_artifact()` reads the model from disk one time when the service starts and the loaded object is reused for all requests.
**2. In plain words:** Loading is slow relative to prediction, so doing it per request would waste time. The demo shows the load takes 0.4 ms once, then each prediction about 0.3 ms.
**3. Example to quote:**
> "I load the artifact once at startup and reuse it; per-request loading would dominate latency."

**Seen in:** `predict.py`, `main.py`

## metadata.json  `(concept)`
**1. Definition:** A JSON file stored beside the model recording how it was produced: creation time, training size, parameters, metrics, classes and library versions.
**2. In plain words:** Months later it answers "what is this model and can I load it?". Recording `sklearn_version` matters because a model should be loaded with a compatible library version.
**3. Example to quote:**
> "My `metadata.json` says the model was trained on 200 samples with C=1.0, accuracy 1.0, scikit-learn 1.x, on a given date."

**Seen in:** `train.py`, `artifacts/metadata.json`

## datetime.now(timezone.utc).isoformat()  `(standard library)`
**1. Definition:** `datetime.now(timezone.utc)` is the current time as a timezone-aware UTC datetime; `.isoformat(timespec="seconds")` formats it as a string like `2026-10-19T11:03:00+00:00`.
**2. In plain words:** Storing UTC avoids confusion across timezones, and the ISO format sorts correctly and is understood by everything.
**3. Example to quote:**
> "`created_at` is a UTC ISO timestamp, so I know exactly when each model was trained."

```python
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat(timespec="seconds"))
```

**Seen in:** `train`

## sklearn.__version__ and sys.version  `(standard library)`
**1. Definition:** `sklearn.__version__` is the installed scikit-learn version string; `sys.version.split()[0]` is the Python version number.
**2. In plain words:** Recording them lets you detect incompatible environments before loading, because pickled models can break across versions.
**3. Example to quote:**
> "The metadata stores `sklearn_version` so `load_artifact` can warn if the serving environment differs."

**Seen in:** `train`

## Path.mkdir(exist_ok=True)  `(standard library)`
**1. Definition:** Creates a directory; `exist_ok=True` suppresses the error if it already exists.
**2. In plain words:** Makes training re-runnable: the artifacts folder is created the first time and reused later.
**3. Example to quote:**
> "`ARTIFACT_DIR.mkdir(exist_ok=True)` makes sure the folder exists without failing on the second run."

**Seen in:** `train`

## Path.write_text and read_text  `(standard library)`
**1. Definition:** `Path.write_text(s)` writes a string to a file and `Path.read_text()` reads a file into a string.
**2. In plain words:** Convenient for small files such as JSON metadata, with no open/close boilerplate.
**3. Example to quote:**
> "`(ARTIFACT_DIR / 'metadata.json').write_text(json.dumps(metadata, indent=2))` saves readable metadata."

**Seen in:** `train.py`, `predict.py`

## json.dumps(indent=2) and json.loads  `(standard library)`
**1. Definition:** `json.dumps(obj, indent=2)` converts a Python object to pretty-printed JSON text; `json.loads(s)` parses JSON text back to Python objects.
**2. In plain words:** Indentation makes the metadata file human-readable; loading brings it back as a dict when serving starts.
**3. Example to quote:**
> "I write metadata with `indent=2` so I can open it and read it, and load it with `json.loads`."

**Seen in:** `train.py`, `predict.py`

## Regularization strength C  `(concept)`
**1. Definition:** In scikit-learn's `LogisticRegression`, `C` is the inverse of regularization strength: smaller values mean stronger regularization (simpler model).
**2. In plain words:** Regularization discourages overly complex weights to reduce overfitting. `C=1.0` is the default balance; recording it in metadata makes the training run reproducible.
**3. Example to quote:**
> "`train(n_samples=200, C=1.0)` stores `{'C': 1.0}` in the metadata so I know how this model was configured."

**Seen in:** `train`

## Train/test split in the pipeline  `(concept)`
**1. Definition:** Using separate datasets, one to fit the model and a held-out one to evaluate it.
**2. In plain words:** The test texts are handwritten and never trained on, which gives an honest estimate of how the model will behave on new text.
**3. Example to quote:**
> "`train()` gets training data from `get_training_data` and evaluates on `get_test_data`, which the model never sees during fit."

**Seen in:** `train`

## lambda for accuracy  `(syntax)`
**1. Definition:** `lambda pred: sum(...) / len(y_test)` defines a small anonymous function inline.
**2. In plain words:** The skew demo needs one throwaway accuracy function; `acc = lambda pred: ...` avoids a full `def` for a one-liner.
**3. Example to quote:**
> "`acc = lambda pred: sum(p == y for p, y in zip(pred, y_test)) / len(y_test)` computes accuracy for any prediction list."

**Seen in:** `broken_skew.py`

## TfidfVectorizer(lowercase=False)  `(third-party: scikit-learn)`
**1. Definition:** The `lowercase` parameter controls whether the vectorizer lowercases text; `False` disables it.
**2. In plain words:** In the skew demo the vectorizer deliberately does not lowercase, because lowercasing was done by hand in training. That hand-made step is exactly what serving forgets.
**3. Example to quote:**
> "`TfidfVectorizer(lowercase=False)` means cleaning is my responsibility, which is how skew sneaks in."

**Seen in:** `broken_skew.py`

## sys.path.insert and # noqa: E402  `(standard library)`
**1. Definition:** `sys.path.insert(0, folder)` makes another folder importable; `# noqa: E402` tells the linter the late import is intentional.
**2. In plain words:** The scripts reuse `data.py` from step 01 even though folder names starting with digits cannot be imported as packages.
**3. Example to quote:**
> "I add step 01's folder to `sys.path` so `from data import get_training_data` works here too."

**Seen in:** `train.py`, `broken_skew.py`

## Serving the artifact with the module-09 API  `(concept)`
**1. Definition:** Pointing the FastAPI app's `MODEL_PATH` environment variable at this artifact so the API serves exactly this trained model.
**2. In plain words:** It links modules together: training here produces the file, module 09's API loads it. The command is printed by `main.py`.
**3. Example to quote:**
> "`MODEL_PATH=.../artifacts/model.joblib uvicorn fastapi_app:app --port 8000` serves the model trained in this step."

```bash
MODEL_PATH=artifacts/model.joblib uvicorn fastapi_app:app --port 8000
```

**Seen in:** `main.py` output, `WALKTHROUGH.md`

## time.perf_counter timing  `(standard library)`
**1. Definition:** `time.perf_counter()` returns a high-resolution clock value; the difference between two calls is elapsed time in seconds.
**2. In plain words:** `main.py` uses it to time training (12 ms), artifact loading (0.4 ms) and each prediction (about 0.3 ms), which shows why you load once.
**3. Example to quote:**
> "`(time.perf_counter() - t0) * 1000` converts the elapsed seconds into milliseconds for printing."

**Seen in:** `main.py`

## Training is heavy, inference is light  `(concept)`
**1. Definition:** Training costs minutes to days and may need GPUs, while inference should take milliseconds and be cheap per request.
**2. In plain words:** That difference drives architecture: train offline on big machines, serve online on small, always-on ones. A training failure delays a model; an inference failure shows errors to users right now.
**3. Example to quote:**
> "Training 200 examples took 12 ms here, but real training could take hours, while a request must answer in milliseconds."

**Seen in:** `WALKTHROUGH.md` side-by-side table
