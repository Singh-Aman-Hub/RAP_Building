# Step 08 — A complete mini ML pipeline: train → save → load → predict

## The story so far
Chapter 5 gave us the whole toolkit: a loss, a loop, a validation set, probabilities and honest metrics. But all of it assumed numeric features, and the world's richest data is text. Chapter 6 asks: can we use this toolkit on raw sentences today, with the classic tools?

## The pain point
A classifier cannot read "great food". It needs a fixed-length row of numbers per sentence, and a model that gets that row at training time must get the *identical* transformation at serving time, or its answers are meaningless.

## The spark
Count words, then down-weight the boring ones: that is TF-IDF. Give each word a column, put the weighted counts in a row, and feed it to logistic regression. Wrap both in one `Pipeline` so preprocessing travels with the model, then `joblib.dump` the whole thing. It works: about 0.80 validation accuracy on this tiny dataset.

## Goal
Build the first end-to-end ML artifact: text classifier pipeline, saved with joblib, loaded for inference. **This is reused** by [09_api_development](../../09_api_development/) (served over HTTP), [10_ml_system_architecture](../../10_ml_system_architecture/) (versioning, monitoring) and [13_capstone_project](../../13_capstone_project/).

## Run it
```bash
cd 03_ml_foundations/08_sklearn_text_classifier
python train.py                       # creates artifacts/model.joblib + metadata.json
python predict.py "great food" "bad slow service"
pytest -q                             # tests
python flaw_demo.py                   # Chapter 6 epilogue: proves TF-IDF's two flaws
# from repo root: .venv/bin/python 03_ml_foundations/08_sklearn_text_classifier/train.py
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| TF-IDF | tf-idf weighting, term-frequency inverse-document-frequency | Yes | Word count scaled down for words that appear in many documents |
| Bag of words (BoW) | count vectors, `CountVectorizer` | Close | BoW counts words; TF-IDF re-weights those counts |
| n-gram | bigram, trigram, word pair | Yes | A run of n consecutive words used as one feature; `ngram_range=(1, 2)` means unigrams plus bigrams |
| Logistic regression | logit model, MaxEnt classifier | Yes | A linear model whose score goes through sigmoid (or softmax) to give probabilities. Despite the name it classifies |
| Pipeline | preprocessing + model chain | Yes | One object that runs vectoriser then classifier, so training and inference transform text identically |
| Artifact | saved model, checkpoint, serialised model | Roughly | The file(s) written by training and loaded for serving; here `model.joblib` plus `metadata.json` |
| joblib vs pickle | serialisation | Related | `joblib` is pickle optimised for large numpy arrays; both execute code on load, so only load files you trust |
| Cosine similarity | cosine, angle similarity | Yes | `u.v / (norm(u)*norm(v))`; 1 = same direction, 0 = unrelated |
| Stratified split | class-balanced split | Yes | Keeps the class ratio equal in train and validation |

## Code, step by step

### 1. train.py docstring: what it produces  (`train.py` lines 1–10)

```python
"""
GOAL: Train a text classifier (TF-IDF + logistic regression), evaluate it, and SAVE it as a
      versioned artifact that the API (module 09), architecture notes (module 10) and capstone (13) load.

RUN (from this folder):  python train.py
RUN (from repo root):    .venv/bin/python 03_ml_foundations/08_sklearn_text_classifier/train.py
EXPECTED OUTPUT: validation metrics and "saved model -> artifacts/model.joblib".
NEEDS: scikit-learn, joblib
ARTIFACTS: artifacts/model.joblib (the pipeline) and artifacts/metadata.json (version, metrics, labels)
"""
```

- **In simple words:** A header saying this script trains a text classifier and saves it for the API to use later.
- **Exactly:** Docstring only. It names the two output files: `artifacts/model.joblib` (the whole fitted pipeline) and `artifacts/metadata.json` (version, metrics, labels).
- **Why it is here:** Modules 09, 10 and 13 load these artifacts, so the contract of what is saved is part of the documentation.

### 2. Imports  (`train.py` lines 12–21)

```python
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib                                              # saves/loads Python objects (sklearn models) to disk
from sklearn.feature_extraction.text import TfidfVectorizer  # text -> numeric feature vectors
from sklearn.linear_model import LogisticRegression         # the classifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split         # random split into train / validation
from sklearn.pipeline import Pipeline                        # chains preprocessing + model into ONE object
```

- **In simple words:** Standard tools for files and time, plus the scikit-learn parts we need.
- **Exactly:** `json`, `datetime`, `Path` are standard library. `joblib` saves Python objects. `TfidfVectorizer` converts text to sparse numeric vectors; `LogisticRegression` classifies; `classification_report` prints precision/recall/F1; `train_test_split` splits; `Pipeline` chains steps.
- **Why it is here:** Each import is one stage of the classic ML workflow: featurise, model, evaluate, split, save.

### 3. Import the dataset loader  (`train.py` line 23)

```python
from data import load_dataset
```

- **In simple words:** Reuse the data file next door.
- **Exactly:** `from data import load_dataset` works because the script is run from this folder (or via the pytest rootdir), so `data.py` is on `sys.path`. It returns `(texts, labels)` with 180 items each.
- **Why it is here:** Keeps data creation separate from training logic.

### 4. Where to save, and which version  (`train.py` lines 25–26)

```python
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"   # always relative to THIS file, not the cwd
MODEL_VERSION = "v1"
```

- **In simple words:** Decide the artifacts folder and a version label.
- **Exactly:** `Path(__file__).resolve().parent / "artifacts"` is an absolute path next to this file, so it works from any working directory. `MODEL_VERSION = "v1"` is written into the metadata.
- **Why it is here:** Relative paths like `"artifacts"` break when you run the script from another folder. A version string lets serving code report which model answered.

### 5. Build the pipeline: TF-IDF then logistic regression  (`train.py` lines 29–34)

```python
def build_pipeline():
    """Preprocessing and model travel together, so inference does exactly the same transformation as training."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),  # words and word pairs, weighted by rarity
        ("clf", LogisticRegression(max_iter=1000)),                      # outputs a probability per class
    ])
```

- **In simple words:** Chain two steps: turn sentences into numbers, then learn to classify the numbers.
- **Exactly:** `Pipeline` takes `(name, estimator)` pairs. `TfidfVectorizer(lowercase=True, ngram_range=(1, 2))` learns a vocabulary of unigrams and bigrams (495 features on the 135 training sentences) and outputs a sparse matrix `(n_samples, 495)`. `LogisticRegression(max_iter=1000)` learns one weight vector `coef_` of shape `(1, 495)` plus an intercept, and gives `predict_proba` of shape `(n, 2)`. `max_iter=1000` raises the optimiser's iteration cap from the default 100 to avoid a convergence warning.
- **Why it is here:** Preprocessing and model travel together, so inference applies exactly the training transformation. A vectoriser fitted separately and forgotten is a classic production bug (training/serving skew).

### 6. Load the data and split 75/25  (`train.py` lines 37–40)

```python
def main():
    texts, labels = load_dataset()
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels)  # stratify keeps class balance equal
```

- **In simple words:** Read the 180 sentences and hold back a quarter for testing.
- **Exactly:** `train_test_split(texts, labels, test_size=0.25, random_state=42, stratify=labels)` returns four lists: `X_train` 135, `X_val` 45, `y_train` 135, `y_val` 45. `random_state=42` makes the shuffle repeatable. `stratify=labels` keeps 50/50 positive/negative in both parts.
- **Why it is here:** Evaluation on training data would overstate quality (step 05). Without a seed, metrics change on every run.

### 7. Fit on the training data only  (`train.py` lines 42–43)

```python
    model = build_pipeline()
    model.fit(X_train, y_train)                              # learn vocabulary + weights from TRAIN data only
```

- **In simple words:** Learn the vocabulary and the weights from the training sentences.
- **Exactly:** `Pipeline.fit` calls `tfidf.fit_transform(X_train)` (learns vocabulary and IDF weights, yields `(135, 495)`), then `clf.fit(that_matrix, y_train)`. The validation text never influences the vocabulary.
- **Why it is here:** Fitting the vectoriser on validation text too would leak information. Deleting this line gives `NotFittedError` at predict (the BREAK IT).

### 8. Predict on the validation set and report  (`train.py` lines 45–47)

```python
    predictions = model.predict(X_val)                       # evaluate on unseen data
    report = classification_report(y_val, predictions, output_dict=True, zero_division=0)
    print(classification_report(y_val, predictions, zero_division=0))
```

- **In simple words:** Ask the model about sentences it has not seen and print a score card.
- **Exactly:** `model.predict(X_val)` returns 45 label strings (the pipeline applies `tfidf.transform` then `clf.predict`). `classification_report(..., output_dict=True)` returns a dict so we can store numbers; the second call prints the text table. `zero_division=0` avoids warnings if a class is never predicted. Real run: accuracy 0.80, negative precision 0.77 / recall 0.87, positive 0.84 / 0.73, macro F1 0.798.
- **Why it is here:** Numbers on unseen data are the honest estimate of quality; they also go into the metadata for later comparison.

### 9. Save the fitted pipeline to disk  (`train.py` lines 49–50)

```python
    ARTIFACT_DIR.mkdir(exist_ok=True)                        # create artifacts/ if missing
    joblib.dump(model, ARTIFACT_DIR / "model.joblib")        # serialise the whole pipeline
```

- **In simple words:** Write the trained model to a file so that another program can use it without retraining.
- **Exactly:** `mkdir(exist_ok=True)` creates the folder only if it is missing. `joblib.dump(model, path)` pickles the whole fitted `Pipeline`, including the 495-word vocabulary and weights, to `artifacts/model.joblib`.
- **Why it is here:** Serving should never retrain; and a pickle runs code on load, so only load artifacts you created.

### 10. Save metadata next to the model  (`train.py` lines 51–61)

```python
    metadata = {
        "version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "labels": [str(c) for c in model.classes_],
        "n_train": len(X_train),
        "n_val": len(X_val),
        "val_accuracy": report["accuracy"],
        "val_f1_macro": report["macro avg"]["f1-score"],
    }
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print("saved model -> artifacts/model.joblib  | metadata ->", metadata)
```

- **In simple words:** A small JSON "label on the box": version, when it was trained, what classes, how well it did.
- **Exactly:** `datetime.now(timezone.utc).isoformat(timespec="seconds")` gives a timestamp like `2026-11-05T07:48:46+00:00`. `model.classes_` is `['negative', 'positive']` (alphabetical, the column order of `predict_proba`); `str(c)` turns numpy strings into plain strings so JSON can serialise them. Real values: `n_train 135`, `n_val 45`, `val_accuracy 0.8`, `val_f1_macro 0.7984...`. `json.dumps(indent=2)` pretty-prints, `write_text` saves it.
- **Why it is here:** Without metadata you cannot tell which model a file is or whether a retrain made things worse (module 10 versioning).

### 11. Run main() only when executed as a script  (`train.py` lines 64–67)

```python
if __name__ == "__main__":
    main()
    # TRY IT: change ngram_range to (1, 1) and compare val_accuracy. Then set MODEL_VERSION = "v2" (model versioning in module 10).
    # BREAK IT: delete the line `model.fit(...)` and read the NotFittedError when .predict runs.
```

- **In simple words:** Do the training when you run this file, but not when another file imports it.
- **Exactly:** `__name__ == "__main__"` is true only for direct execution. `flaw_demo.py` does `from train import build_pipeline` and relies on this guard so that importing does not retrain. The comments suggest `ngram_range=(1, 1)` and a version bump.
- **Why it is here:** Without the guard, importing `build_pipeline` would train and overwrite the artifacts as a side effect.

### 12. predict.py docstring  (`predict.py` lines 1–8)

```python
"""
GOAL: Load the saved model and predict.  Exposes load_model() and predict_texts() for reuse by the API in module 09.

RUN (from this folder):  python predict.py "The movie was fantastic"      (run train.py first)
RUN (from repo root):    .venv/bin/python 03_ml_foundations/08_sklearn_text_classifier/predict.py "The movie was fantastic"
EXPECTED OUTPUT: {'text': 'The movie was fantastic', 'label': 'positive', 'confidence': 0.681}
NEEDS: scikit-learn, joblib; artifacts/ created by train.py
"""
```

- **In simple words:** Explains that this file loads the saved model and exposes functions the API will reuse.
- **Exactly:** Docstring only. `python predict.py "The movie was fantastic"` should print label `positive`, confidence `0.681`, which the real run reproduced.
- **Why it is here:** Documents the usage and the dependency on `train.py` having run first.

### 13. Imports and artifact location  (`predict.py` lines 10–17)

```python
import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
```

- **In simple words:** Tools for JSON, command-line arguments, caching, paths and loading the model.
- **Exactly:** `sys` gives `sys.argv`; `lru_cache` is a memoising decorator; `ARTIFACT_DIR` is computed exactly as in `train.py` so the two always agree.
- **Why it is here:** If predict and train disagreed on the folder, the model would not be found.

### 14. Load the model once and cache it  (`predict.py` lines 20–25)

```python
@lru_cache(maxsize=1)                                       # decorator: load from disk ONCE, then reuse (important in servers!)
def load_model():
    path = ARTIFACT_DIR / "model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run train.py first.")
    return joblib.load(path)
```

- **In simple words:** Read the model file from disk the first time, then reuse the same object.
- **Exactly:** `@lru_cache(maxsize=1)` stores the return value of the zero-argument function, so every later call returns the same `Pipeline` without disk access. It raises `FileNotFoundError` with a helpful hint if `model.joblib` is missing. `joblib.load` rebuilds the pipeline.
- **Why it is here:** A web server calling `joblib.load` for every request would spend most of its time reading a file. A clear error message saves a confusing stack trace.

### 15. Read the metadata file  (`predict.py` lines 28–29)

```python
def load_metadata():
    return json.loads((ARTIFACT_DIR / "metadata.json").read_text())
```

- **In simple words:** Return the JSON info saved by training as a dict.
- **Exactly:** `read_text()` returns a string; `json.loads` parses it to a dict like `{'version': 'v1', ...}`. The API in module 09 uses it for a `/health` or `/info` route.
- **Why it is here:** Lets a service report which model version it is serving.

### 16. Validate the input type  (`predict.py` lines 32–35)

```python
def predict_texts(texts):
    """Input: list[str].  Output: list of dicts with label and confidence (probability of the chosen label)."""
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise TypeError("texts must be a list of strings")
```

- **In simple words:** Refuse anything that is not a list of strings, with a clear error.
- **Exactly:** `isinstance(texts, list)` and `all(isinstance(t, str) for t in texts)`. Passing the bare string `"not a list"` raises `TypeError("texts must be a list of strings")`. Without this check a bare string would be iterated character by character.
- **Why it is here:** Good errors at the boundary are cheaper than mysterious failures deep inside sklearn (test `test_bad_input` checks this).

### 17. Get class probabilities  (`predict.py` lines 36–37)

```python
    model = load_model()
    probabilities = model.predict_proba(texts)              # shape (n_texts, n_classes)
```

- **In simple words:** Ask the pipeline how likely each class is for every text.
- **Exactly:** `predict_proba(texts)` runs TF-IDF and the classifier and returns a numpy array of shape `(n_texts, n_classes)` = `(1, 2)` for one text. Each row sums to 1; columns follow `model.classes_` = `['negative', 'positive']`.
- **Why it is here:** Probabilities, not just labels, give a confidence score you can threshold.

### 18. Pick the best class per text  (`predict.py` lines 38–42)

```python
    results = []
    for text, row in zip(texts, probabilities):
        best = int(row.argmax())                            # index of the highest probability
        results.append({"text": text, "label": str(model.classes_[best]), "confidence": round(float(row[best]), 3)})
    return results
```

- **In simple words:** For each sentence take the more likely class and report how likely it was.
- **Exactly:** `row.argmax()` is the index of the largest probability (numpy int, so `int(...)` converts it). `model.classes_[best]` maps the index to a label. `round(float(row[best]), 3)` gives, for "The movie was fantastic", `0.681`. Returns a list of dicts `{'text', 'label', 'confidence'}`, which are JSON-friendly for an API.
- **Why it is here:** Casting numpy scalars to Python `float` and `int` avoids JSON serialisation errors in FastAPI.

### 19. Command-line entry point  (`predict.py` lines 45–50)

```python
if __name__ == "__main__":
    inputs = sys.argv[1:] or ["The movie was fantastic", "This was a terrible waste of money"]   # sys.argv = command-line words
    for item in predict_texts(inputs):
        print(item)
    # TRY IT: predict a sentence of words the model never saw ("I really loved it") -- confidence hovers near 0.5. Also try an empty string "" -- what does the model say, and is its confidence trustworthy?
    # BREAK IT: predict_texts("a string, not a list") -> TypeError raised by our own validation (good errors are a feature).
```

- **In simple words:** When run directly, predict the sentences typed after the filename, or two defaults.
- **Exactly:** `sys.argv[1:]` is the list of words after the script name; `or [...]` falls back when empty. Real output for the defaults: `positive 0.681` and `negative 0.663`. For the unseen-words sentence `"I really loved it"` the model says `negative` with `0.501`, essentially a coin flip, because none of those words are in its vocabulary.
- **Why it is here:** Shows how a model behaves outside its data: confidence near 0.5 is the honest signal to treat the answer as unknown.

### 20. data.py docstring  (`data.py` line 1)

```python
"""Tiny sentiment dataset: 60 hand-written + 120 template-generated sentences. Real projects load thousands of rows from a CSV/DB."""
```

- **In simple words:** One line explaining the dataset is small and handmade.
- **Exactly:** 60 hand-written sentences plus 120 template-generated ones make 180 total.
- **Why it is here:** Honest labelling of toy data avoids mistaking it for a real benchmark.

### 21. 30 hand-written positive sentences  (`data.py` lines 3–34)

```python
POSITIVE = [
    "I love this phone, the battery lasts all day",
    "Great service and very friendly staff",
    "The food was delicious and fresh",
    "Absolutely fantastic movie, I enjoyed every minute",
    "Excellent quality, worth every penny",
    "The delivery was fast and the packaging was perfect",
    "Wonderful experience, I will definitely come back",
    "This laptop is fast and the screen is beautiful",
    "Amazing value for the price",
    "Super helpful support team solved my issue quickly",
    "The hotel room was clean and comfortable",
    "I am very happy with my purchase",
    "Brilliant book, I could not put it down",
    "The app is smooth and easy to use",
    "Best coffee in town, highly recommend",
    "Very pleased, it works perfectly",
    "Lovely atmosphere and great music",
    "The teacher was inspiring and kind",
    "Great product, exceeded my expectations",
    "Smooth checkout and quick refund, impressive",
    "The concert was awesome and the crowd was fun",
    "Really good headphones with clear sound",
    "Fantastic customer care, thank you",
    "The pizza tasted amazing",
    "Perfect fit and very comfortable shoes",
    "I enjoyed the course, it was clear and useful",
    "Terrific update, everything feels faster",
    "The staff were polite and helpful",
    "Good value and reliable performance",
    "Beautiful design and sturdy build",
]
```

- **In simple words:** A plain list of happy reviews.
- **Exactly:** `POSITIVE` is a `list[str]` of 30 items about phones, food, services, etc.
- **Why it is here:** Labelled examples are what supervised learning needs.

### 22. 30 hand-written negative sentences  (`data.py` lines 36–67)

```python
NEGATIVE = [
    "I hate this phone, the battery dies in an hour",
    "Terrible service and rude staff",
    "The food was cold and tasteless",
    "Awful movie, a complete waste of time",
    "Poor quality, a waste of money",
    "The delivery was late and the box was damaged",
    "Horrible experience, I will never come back",
    "This laptop is slow and the screen is dim",
    "Overpriced and disappointing",
    "Useless support team, they ignored my emails",
    "The hotel room was dirty and noisy",
    "I am very unhappy with my purchase",
    "Boring book, I could not finish it",
    "The app keeps crashing and is hard to use",
    "Worst coffee ever, do not recommend",
    "Very disappointed, it stopped working in a week",
    "Awful atmosphere and terrible music",
    "The teacher was boring and unhelpful",
    "Bad product, far below my expectations",
    "Checkout failed twice and the refund never came",
    "The concert was awful and the sound was bad",
    "Really bad headphones with a crackling sound",
    "Dreadful customer care, nobody answered",
    "The pizza tasted burnt",
    "Terrible fit and very uncomfortable shoes",
    "I disliked the course, it was confusing and useless",
    "Horrible update, everything feels slower",
    "The staff were impolite and unhelpful",
    "Bad value and unreliable performance",
    "Ugly design and flimsy build",
]
```

- **In simple words:** The matching list of unhappy reviews.
- **Exactly:** `NEGATIVE` is a `list[str]` of 30 items, written to mirror the positive ones (e.g. "Terrible service and rude staff").
- **Why it is here:** Balanced classes (30 and 30) keep accuracy meaningful.

### 23. Word lists and sentence templates  (`data.py` lines 70–76)

```python
# Template-generated sentences: a toy way to get MORE data that shares sentiment words with the hand-written
# ones.  (Real projects collect real data; synthetic data like this is only for learning the workflow.)
NOUNS = ["food", "service", "movie", "phone", "hotel", "app", "book", "staff", "delivery", "product",
         "laptop", "course", "coffee", "concert", "update"]
POS_WORDS = ["great", "excellent", "amazing", "wonderful", "fantastic", "good", "lovely", "perfect", "brilliant", "fast"]
NEG_WORDS = ["terrible", "awful", "horrible", "bad", "poor", "dreadful", "boring", "useless", "slow", "disappointing"]
TEMPLATES = ["The {noun} was {word}", "I think the {noun} is {word}", "{word} {noun}, really", "Honestly the {noun} felt {word}"]
```

- **In simple words:** Ingredients for making more sentences automatically.
- **Exactly:** 15 nouns, 10 positive words, 10 negative words and 4 templates with `{noun}` and `{word}` slots. This allows 15*10*4 = 600 distinct sentences per class.
- **Why it is here:** A toy way to get more data that shares sentiment words with the hand-written examples. Real projects collect real data.

### 24. Generate n unique template sentences  (`data.py` lines 79–85)

```python
def _generate(words, n, seed):
    import random
    rng = random.Random(seed)                      # seeded -> the same dataset on every run
    out = set()                                    # set removes duplicate sentences
    while len(out) < n:
        out.add(rng.choice(TEMPLATES).format(noun=rng.choice(NOUNS), word=rng.choice(words)))
    return sorted(out)
```

- **In simple words:** Randomly fill templates until we have n different sentences.
- **Exactly:** `random.Random(seed)` is a private seeded generator. `out = set()` drops duplicates, `rng.choice(...)` picks a template, noun and word, and `.format` fills the slots. The loop runs while `len(out) < n`. `sorted(out)` returns a stable list so the dataset is identical on every run.
- **Why it is here:** Seeds make the experiment reproducible; the set guarantees distinct rows, so train and validation do not share duplicates.

### 25. Return the full labelled dataset  (`data.py` lines 88–92)

```python
def load_dataset():
    """Return (texts, labels) where labels are the strings 'positive' / 'negative'. 60 hand-written + 120 generated."""
    pos = POSITIVE + _generate(POS_WORDS, 60, seed=1)
    neg = NEGATIVE + _generate(NEG_WORDS, 60, seed=2)
    return pos + neg, ["positive"] * len(pos) + ["negative"] * len(neg)
```

- **In simple words:** Combine hand-written and generated sentences and attach a label to each.
- **Exactly:** `pos` has 90 items (30 + 60), `neg` 90. Returns `(texts, labels)` of 180 each: texts positive-first, `labels = ['positive']*90 + ['negative']*90`. The order is not shuffled here; `train_test_split` shuffles later.
- **Why it is here:** One function is the single source of data for train, tests and demos.

### 26. flaw_demo.py docstring  (`flaw_demo.py` lines 1–10)

```python
"""
GOAL: Chapter 6 epilogue. First celebrate that the TF-IDF classifier works, then PROVE its two fatal flaws
      with numbers: (a) words have no meaning-geometry, (b) word order is thrown away.

RUN (from this folder):  python flaw_demo.py
RUN (from repo root):    .venv/bin/python 03_ml_foundations/08_sklearn_text_classifier/flaw_demo.py
EXPECTED OUTPUT: validation accuracy, then cosine 0.0 for king/queen and car/automobile, then a cosine of 1.0 for
                 'dog bites man' vs 'man bites dog'.
NEEDS: scikit-learn, numpy (no network, no files written)
"""
```

- **In simple words:** This script proves, with numbers, two weaknesses of TF-IDF.
- **Exactly:** Docstring only. Expected: cosine 0.0 for king/queen and car/automobile; cosine 1.0 for "dog bites man" vs "man bites dog".
- **Why it is here:** Motivates embeddings and transformers in module 04.

### 27. Imports  (`flaw_demo.py` lines 12–17)

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from data import load_dataset
from train import build_pipeline
```

- **In simple words:** numpy, a fresh vectoriser, the split function, and our own data and pipeline.
- **Exactly:** `from train import build_pipeline` re-uses the exact pipeline; importing `train` does not run training because of its `__main__` guard.
- **Why it is here:** Same pipeline means the "works" claim is about the same model.

### 28. A cosine similarity helper  (`flaw_demo.py` lines 20–22)

```python
def cosine(u, v):
    """cos(angle) between two vectors: 1.0 = same direction, 0.0 = unrelated (perpendicular)."""
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
```

- **In simple words:** A function giving the angle-based similarity of two vectors.
- **Exactly:** `u @ v` is the dot product; `np.linalg.norm` is the length. cosine = `u.v / (|u||v|)`, range [-1, 1] (0 to 1 for non-negative TF-IDF). `float(...)` converts numpy float64 to Python float.
- **Why it is here:** This is the standard similarity measure used again in module 04 and in RAG.

### 29. Part 1: the classifier works  (`flaw_demo.py` lines 25–31)

```python
# --- Part 1: celebrate. The classifier works. -------------------------------------------------------------------
texts, labels = load_dataset()
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.25, random_state=42, stratify=labels)
model = build_pipeline().fit(X_train, y_train)           # same pipeline as train.py, trained in memory only
print(f"PART 1  The baseline works: validation accuracy = {model.score(X_val, y_val):.2f} on {len(X_val)} unseen sentences")
print("        'great food'       ->", model.predict(["great food"])[0])
print("        'bad slow service' ->", model.predict(["bad slow service"])[0])
```

- **In simple words:** Train the pipeline again in memory and check a couple of sentences.
- **Exactly:** Same split as `train.py`. `build_pipeline().fit(...)` returns the fitted pipeline. `model.score(X_val, y_val)` is accuracy: `0.80` on `45` unseen sentences. `predict(["great food"])[0]` gives `positive` and `"bad slow service"` gives `negative`.
- **Why it is here:** Fair to first show that the baseline is decent; then the flaws mean something.

### 30. Four sentences with parallel contexts  (`flaw_demo.py` lines 33–41)

```python
# --- Part 2: flaw (a). Every word is its own column, so no two words are ever 'close'. ---------------------------
# Parallel sentences: king/queen and car/automobile appear in IDENTICAL contexts, so any real notion of meaning
# would call them near-synonyms or at least neighbours.
corpus = [
    "the king rules the land",
    "the queen rules the land",
    "my car is very fast",
    "my automobile is very fast",
]
```

- **In simple words:** King/queen and car/automobile appear in identical contexts, so a meaning-aware model would see them as near.
- **Exactly:** `corpus` is a list of 4 strings.
- **Why it is here:** Controlled data isolates the semantic gap.

### 31. Vectorise the corpus and inspect the shape  (`flaw_demo.py` lines 42–45)

```python
vec = TfidfVectorizer()
M = vec.fit_transform(corpus).toarray()                  # shape: (n_docs=4, vocab_size)  one COLUMN per word
vocab = vec.vocabulary_                                  # word -> column index
print("\nPART 2  Flaw (a): words have no semantics. TF-IDF matrix shape (docs, vocab):", M.shape)
```

- **In simple words:** Turn the four sentences into a table of numbers, one column per word.
- **Exactly:** `fit_transform` returns a sparse matrix; `.toarray()` makes it dense `M` of shape `(4, 11)`: 4 documents and 11 unique words (the, king, rules, land, queen, my, car, is, very, fast, automobile). `vec.vocabulary_` maps word to column index.
- **Why it is here:** Seeing the matrix shape makes concrete that every word is its own dimension.

### 32. Compare word columns with cosine  (`flaw_demo.py` lines 46–49)

```python
for a, b in [("king", "queen"), ("car", "automobile"), ("king", "king")]:
    sim = cosine(M[:, vocab[a]], M[:, vocab[b]])         # compare the two words' COLUMNS
    print(f"        cosine({a!r:>12}, {b!r:<12}) = {sim:.1f}")
print("        A car and an automobile are the same thing, yet the model sees them as 100% unrelated.")
```

- **In simple words:** Check how similar the model thinks word pairs are.
- **Exactly:** `M[:, vocab[a]]` is the column for word a, shape `(4,)`. Real output: `king` vs `queen` 0.0, `car` vs `automobile` 0.0, `king` vs `king` 1.0. Distinct columns are orthogonal because the words never share a document.
- **Why it is here:** This is flaw (a): no notion of meaning. Embeddings (step 04.03) fix it.

### 33. Two sentences with the same words in another order  (`flaw_demo.py` lines 51–54)

```python
# --- Part 3: flaw (b). A bag of words forgets the order. ---------------------------------------------------------
a_sentence, b_sentence = "dog bites man", "man bites dog"
vec2 = TfidfVectorizer().fit([a_sentence, b_sentence])
va, vb = vec2.transform([a_sentence]).toarray()[0], vec2.transform([b_sentence]).toarray()[0]   # shape: (3,) each
```

- **In simple words:** Same words, different story.
- **Exactly:** `vec2 = TfidfVectorizer().fit([...])` learns vocabulary `['bites', 'dog', 'man']`. `.transform([a_sentence]).toarray()[0]` is the vector for one sentence, shape `(3,)`.
- **Why it is here:** Sets up flaw (b).

### 34. Show the vectors are identical  (`flaw_demo.py` lines 55–60)

```python
print("\nPART 3  Flaw (b): word order is lost.")
print("        vocabulary:", sorted(vec2.vocabulary_))
print("        'dog bites man' ->", va.round(3))
print("        'man bites dog' ->", vb.round(3))
print("        identical vectors:", np.array_equal(va, vb), "| cosine =", round(cosine(va, vb), 1))
print("        One headline is routine, the other is news, and the classifier cannot tell them apart.")
```

- **In simple words:** Print both vectors and compare.
- **Exactly:** Both are `[0.577 0.577 0.577]` (each of three words has weight 1/sqrt(3) after L2 normalisation). `np.array_equal` is `True` and cosine is `1.0`.
- **Why it is here:** Word order is lost. A bag of words cannot tell "dog bites man" from "man bites dog".

### 35. Bigrams as a partial patch  (`flaw_demo.py` lines 62–65)

```python
# Honest footnote: bigrams (ngram_range=(1, 2)) patch this for ADJACENT words only.
vec3 = TfidfVectorizer(ngram_range=(1, 2)).fit([a_sentence, b_sentence])
v3a, v3b = vec3.transform([a_sentence]).toarray()[0], vec3.transform([b_sentence]).toarray()[0]
print(f"        with bigrams cosine = {cosine(v3a, v3b):.2f}: better, but each new n-gram is yet another unrelated column.")
```

- **In simple words:** Add word pairs as features so some order is captured.
- **Exactly:** `ngram_range=(1, 2)` adds features like `dog bites` and `bites man`. Real cosine becomes `0.43`: better, but each bigram is just another unrelated column.
- **Why it is here:** Honest footnote: n-grams help only for adjacent words and blow up the vocabulary.

### 36. Pointer to the next module, and TRY/BREAK IT  (`flaw_demo.py` lines 67–70)

```python
print("\nNEXT: we need a representation where similar words are NEARBY and order matters. First, how do we feed raw text in at all?")
print("      -> ../../04_llm_fundamentals/00_what_is_an_llm/ and ../../04_llm_fundamentals/01_tokenizer/")
# TRY IT: add "the monarch rules the land" to the corpus. 'king' vs 'monarch' is still 0.0, however close the meaning.
# BREAK IT: ask for vocab["automobile"] after fitting on only the first two sentences -> KeyError: that word is not a column.
```

- **In simple words:** A teaser sending you to the LLM fundamentals module.
- **Exactly:** Prints a message; comments suggest adding "monarch" (still cosine 0.0) and a KeyError experiment.
- **Why it is here:** Leads into module 04.

### 37. Imports for the tests  (`test_predict.py` lines 1–3)

```python
"""pytest tests: run `pytest -q` in this folder after train.py. Shows how ML code gets tested."""
import pytest
from predict import predict_texts
```

- **In simple words:** Load pytest and the function we will test.
- **Exactly:** `from predict import predict_texts` needs `train.py` to have created the artifacts first.
- **Why it is here:** Tests run against the real saved model, like a smoke test of the deployed artifact.

### 38. Test that clear sentences get the right label  (`test_predict.py` lines 6–9)

```python
def test_positive_and_negative():
    out = predict_texts(["I love it, great and excellent", "terrible awful waste"])
    assert out[0]["label"] == "positive"
    assert out[1]["label"] == "negative"
```

- **In simple words:** One obviously good and one obviously bad sentence should be classified correctly.
- **Exactly:** `predict_texts` returns a list of 2 dicts; `out[0]["label"] == "positive"` and `out[1]["label"] == "negative"`. `assert` fails the test if false.
- **Why it is here:** Catches a trained model that has somehow flipped or broken.

### 39. Test that confidence is a probability  (`test_predict.py` lines 12–13)

```python
def test_confidence_range():
    assert 0.0 <= predict_texts(["fine"])[0]["confidence"] <= 1.0
```

- **In simple words:** Confidence must lie between 0 and 1.
- **Exactly:** Chained comparison `0.0 <= x <= 1.0`. Real run: `pytest -q` reports `3 passed`.
- **Why it is here:** Guards against returning raw scores or percentages by mistake.

### 40. Test that bad input raises TypeError  (`test_predict.py` lines 16–18)

```python
def test_bad_input():
    with pytest.raises(TypeError):
        predict_texts("not a list")
```

- **In simple words:** Passing a plain string instead of a list must raise a clear error.
- **Exactly:** `pytest.raises(TypeError)` passes only if the block raises that exception.
- **Why it is here:** Proves the input validation in `predict_texts` works.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `@lru_cache(maxsize=1)` | decorator: cache return value so the model loads once | `load_model()` twice → disk read once |
| `Path(__file__).resolve().parent / "artifacts"` | path relative to this file; `/` joins path parts | works from any cwd |
| `sys.argv[1:]` | command-line arguments | `python predict.py a b` → `["a","b"]` |
| `x or default` | fallback if empty | |
| `isinstance(t, str)` | type check | |
| `raise TypeError(...)` | signal invalid input | |
| `with pytest.raises(...)` | assert that an error occurs | |
| dict literal / `json.dumps(indent=2)` | build and serialise metadata | |

## Libraries & functions
| Name | Origin | Does | Important args | Returns |
|---|---|---|---|---|
| `TfidfVectorizer` | scikit-learn | text → numeric vectors (word importance) | `ngram_range`, `lowercase` | transformer |
| `LogisticRegression` | scikit-learn | linear classifier with probabilities | `max_iter` | estimator |
| `Pipeline([...])` | scikit-learn | chain steps into one object | list of (name, step) | Pipeline |
| `train_test_split` | scikit-learn | split data | `test_size`, `random_state`, `stratify` | 4 lists |
| `.fit(X, y)` / `.predict` / `.predict_proba` | scikit-learn | learn / labels / probabilities | | |
| `classification_report` | scikit-learn | precision/recall/F1 table | `output_dict` | str or dict |
| `joblib.dump / load` | joblib | save / load model file | path | |
| `json`, `datetime`, `pathlib` | stdlib | metadata, timestamp, paths | | |

## Execution trace (real run)
Input: 180 sentences (60 hand-written + 120 template-generated), 135 train / 45 validation.
Processing: TF-IDF learns a vocabulary from train only → logistic regression learns a weight per word → metrics on validation → `joblib.dump`.
```
              precision    recall  f1-score   support
    negative       0.77      0.87      0.82        23
    positive       0.84      0.73      0.78        22
    accuracy                           0.80        45
saved model -> artifacts/model.joblib  | metadata -> {'version': 'v1', ... 'val_accuracy': 0.8 ...}
{'text': 'great food', 'label': 'positive', 'confidence': 0.675}
{'text': 'bad slow service', 'label': 'negative', 'confidence': 0.703}
```
`pytest -q` → `3 passed`.

## Experiments
1. `ngram_range=(1, 1)` vs `(1, 2)`.
2. Predict `"I really loved it"` (unseen words) → confidence ≈ 0.5: the model doesn't know; **confidence is not correctness**.
3. Set `MODEL_VERSION = "v2"`, retrain, compare metadata.

## Debugging practice
Delete `model.fit(...)` → `sklearn.exceptions.NotFittedError: This Pipeline instance is not fitted yet`. Also run `predict.py` before `train.py` → our own `FileNotFoundError: ... Run train.py first.` Good error messages tell the next person what to do.

## Rebuild checklist
- [ ] Pipeline of vectorizer + classifier; fit; predict.
- [ ] Save and load with joblib; explain why the vectorizer must be saved too.
- [ ] Explain train/inference separation (training script vs `predict_texts`).

## The fatal flaw
The pipeline works, so celebrate it. Then run the proof that it is a dead end:
```bash
python flaw_demo.py
```
Real output (trimmed):
```
PART 1  The baseline works: validation accuracy = 0.80 on 45 unseen sentences
PART 2  Flaw (a): words have no semantics. TF-IDF matrix shape (docs, vocab): (4, 11)
        cosine(      'king', 'queen'     ) = 0.0
        cosine(       'car', 'automobile') = 0.0
        cosine(      'king', 'king'      ) = 1.0
PART 3  Flaw (b): word order is lost.
        'dog bites man' -> [0.577 0.577 0.577]
        'man bites dog' -> [0.577 0.577 0.577]
        identical vectors: True | cosine = 1.0
        with bigrams cosine = 0.43: better, but each new n-gram is yet another unrelated column.
```
1. **No semantics.** Each word owns its own column, so every pair of different words is perpendicular (cosine 0.0), even `car` and `automobile` in identical sentences. Cosine of a word with itself is 1.0, so the measurement works; the representation just has no notion of meaning.
2. **No order.** TF-IDF is a bag of words. `dog bites man` and `man bites dog` give identical vectors, so no model on top can ever tell them apart. Bigrams patch adjacent pairs only, and each bigram is another unrelated column.

Experiment: add `"the monarch rules the land"` to the corpus in `flaw_demo.py`: `king` vs `monarch` is still 0.0. Break it: look up `vocab["automobile"]` after fitting on text without that word and you get `KeyError: 'automobile'`; unseen words simply do not exist for TF-IDF (step 08 Experiment 2).

## The new problem (cliffhanger)
TF-IDF works, but it cannot see that "king" and "queen" are related and it throws word order away. We need a representation where similar words sit *near* each other and order can matter. That is the job of an LLM, and the story continues in [Act II](../../04_llm_fundamentals/BRIDGE_FROM_03.md). Start with [00_what_is_an_llm](../../04_llm_fundamentals/00_what_is_an_llm/), then the first practical question: how do we feed raw text in at all? That is [01_tokenizer](../../04_llm_fundamentals/01_tokenizer/).

## Say it aloud
> **Say it aloud (interview box):** I wrapped TF-IDF and logistic regression in one sklearn Pipeline so preprocessing at inference matches training, checked precision and recall on a validation split, and saved the whole thing with joblib plus a metadata file. It works, but TF-IDF gives every word its own column, so king and queen have cosine 0.0 and "dog bites man" equals "man bites dog". That is why we move to learned dense representations, starting with tokenization and embeddings.
