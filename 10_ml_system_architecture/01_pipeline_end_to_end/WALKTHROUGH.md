# Step 01 — The end-to-end ML pipeline, stage by stage

## The story so far
Module 09 ended with an API that serves one request correctly. But the model behind it came from a notebook, and the question is now how the whole journey from raw data to answer fits together and stays correct over months. Previous step: [09_api_development/10_streaming_sse](../../09_api_development/10_streaming_sse/).

## The pain point
A notebook mixes cleaning, features, training and evaluation in cells run in whatever order you remembered. Nobody can reproduce it, and test data quietly leaks into training.

## The spark
Name every stage (collect, clean, features, train, evaluate, save, serve), run them in a fixed order in one script, and bundle preprocessing with the model in one scikit-learn Pipeline.

## Goal
Walk collect → preprocess → features → train → evaluate → serialize → load → predict with real code, so every box in an architecture diagram has code behind it.

## Run it
```bash
cd 10_ml_system_architecture/01_pipeline_end_to_end && python main.py
```
Real output (shortened):
```
[1 collect]    200 train texts, 8 test texts; first = 'The book was boring.'
[2 preprocess] 'The movie was GREAT!!  ' -> 'the movie was great'
[3 features]   matrix shape (200, 32) = (n_texts, vocabulary_size); sample words: ['the', 'book', 'was', 'boring', 'honestly']
[4 train]      classes learned: ['negative', 'positive']
[5 evaluate]   accuracy = 1.00
[6 serialize]  saved 2481 bytes to model.joblib
[7 predict]    'Honestly the delivery was TERRIBLE!!' -> negative  probs={'negative': 0.726, 'positive': 0.274}
```
(`data.py` generates the dataset from templates; seeds make it reproducible. The test set is hand-written and never used in training.)

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Preprocessing | cleaning, normalisation, text cleaning | Near-synonyms | Preprocessing is the umbrella; lowercasing/stripping punctuation is one kind of it. |
| Feature engineering | featurisation, vectorisation, feature extraction | Mostly | Turning raw input into numbers; TF-IDF vectorising is one technique of it. |
| `fit` vs `transform` vs `fit_transform` | learn vs apply vs both in one call | No | `fit` learns state (the vocabulary) from data; `transform` only applies it; `fit_transform` = both on the same data. |
| Train / test split | held-out set, validation set (loosely) | Not quite | Test = final honest score, never tuned on; validation = used to choose settings. |
| Model artifact | serialised model, checkpoint, model file | Near-synonyms | The saved file (here `model.joblib`) that the serving code loads. |
| Inference | prediction, serving, scoring | Near-synonyms | Running a trained model on new input; no learning happens. |
| `Pipeline` | sklearn pipeline vs ML pipeline (stage diagram) | No | The sklearn class is one object chaining steps; the ML pipeline is the whole system from data collection to serving. |

## Code, step by step

### 1. Module docstring: why a shared dataset exists  (`data.py` lines 1–6)

```python
"""
data.py -- a tiny, deterministic sentiment dataset used by every step in module 10.

(Module 03 builds a text classifier too; this file keeps module 10 self-contained and runnable
on its own. The ideas -- collect, clean, vectorize, train, evaluate -- are identical.)
"""
```

- **In simple words:** A note at the top saying this file makes a tiny fake sentiment dataset that every later step reuses.
- **Exactly:** A string literal as the first statement is the module docstring (`data.__doc__`). It only documents; nothing runs.
- **Why it is here:** Keeps module 10 runnable without the other modules.

### 2. Import the random module  (`data.py` lines 8–8)

```python
import random
```

- **In simple words:** We need a random number generator to build fake sentences.
- **Exactly:** `random` is the standard library module; we will use the `random.Random` class (an independent generator object).
- **Why it is here:** Without it we could not generate varied sentences.

### 3. Word lists and sentence templates  (`data.py` lines 10–13)

```python
NOUNS = ["movie", "phone", "service", "food", "book", "hotel", "app", "game", "teacher", "delivery"]
POS = ["great", "excellent", "amazing", "wonderful", "fantastic", "lovely", "brilliant", "superb"]
NEG = ["terrible", "awful", "horrible", "disappointing", "dreadful", "poor", "boring", "worthless"]
TEMPLATES = ["The {n} was {a}.", "I thought the {n} was really {a}!", "What a {a} {n}.", "Honestly, a {a} {n}."]
```

- **In simple words:** Boxes of building blocks: nouns, happy words, unhappy words, and sentence moulds with holes.
- **Exactly:** `NOUNS` has 10 words, `POS` 8, `NEG` 8, `TEMPLATES` 4 strings with `{n}` and `{a}` placeholders for `str.format`. So 10 x 16 x 4 = 640 distinct sentences are possible.
- **Why it is here:** Defining them at module level (UPPER_CASE = constants) lets both functions and other files see them.

### 4. Eight handwritten test sentences  (`data.py` lines 15–25)

```python
# Handwritten sentences the model never trains on -> an honest test set.
TEST_SET = [
    ("The hotel was lovely and the staff were amazing", "positive"),
    ("Absolutely brilliant game, loved every minute", "positive"),
    ("Superb food and excellent service", "positive"),
    ("Fantastic delivery, arrived early", "positive"),
    ("The book was boring and awful", "negative"),
    ("Terrible app, it keeps crashing, worthless", "negative"),
    ("Horrible service and dreadful food", "negative"),
    ("A poor, disappointing phone", "negative"),
]
```

- **In simple words:** A small exam the model has never seen, written by hand so it cannot have been memorised.
- **Exactly:** A `list` of 8 `(text, label)` tuples, 4 positive and 4 negative. They use phrasing and words ('Absolutely', 'loved') that templates never produce.
- **Why it is here:** If test data came from the same generator as training data, accuracy would be flattering and meaningless.

### 5. Start get_training_data with a private RNG  (`data.py` lines 28–31)

```python
def get_training_data(n: int = 200, seed: int = 0) -> tuple[list[str], list[str]]:
    """Generate n labelled sentences from templates. Same seed -> same data (reproducibility!)."""
    rng = random.Random(seed)                      # a private random generator so we never disturb global state
    texts, labels = [], []
```

- **In simple words:** Set up a dice roller with a fixed starting point so the same 'random' sentences come out every time.
- **Exactly:** `random.Random(seed)` creates its own generator; the same seed gives the identical sequence. Signature returns `tuple[list[str], list[str]]`: (texts, labels). Two empty lists collect results.
- **Why it is here:** Using the global `random` could be disturbed by other code and break reproducibility.

### 6. Loop: build one labelled sentence per iteration  (`data.py` lines 32–38)

```python
    for _ in range(n):
        positive = rng.random() < 0.5              # coin flip decides the class
        adjective = rng.choice(POS if positive else NEG)
        text = rng.choice(TEMPLATES).format(n=rng.choice(NOUNS), a=adjective)
        texts.append(text)
        labels.append("positive" if positive else "negative")
    return texts, labels
```

- **In simple words:** Flip a coin, pick a matching adjective, drop it and a noun into a random template, and record the answer.
- **Exactly:** `rng.random() < 0.5` is True with probability 0.5. `rng.choice(seq)` picks one element. `.format(n=..., a=...)` fills the placeholders. Run result: first text is `'The book was boring.'`. Returns the two parallel lists (200 each for n=200).
- **Why it is here:** Text and label are created together, so they can never disagree.

### 7. Split the test pairs into texts and labels  (`data.py` lines 41–43)

```python
def get_test_data() -> tuple[list[str], list[str]]:
    texts, labels = zip(*TEST_SET)                 # unzip list of pairs into two tuples
    return list(texts), list(labels)
```

- **In simple words:** Turn a list of pairs into two separate lists, the shape sklearn wants.
- **Exactly:** `zip(*TEST_SET)` unpacks the 8 pairs as arguments to `zip`, which regroups them into two tuples (all texts, all labels); `list(...)` converts them.
- **Why it is here:** `fit`/`predict` take X and y separately, not pairs.

### 8. Docstring: goal, how to run, expected output  (`main.py` lines 1–18)

```python
"""
GOAL:   Walk the ENTIRE ML pipeline once, stage by stage, with real code:
        collect -> preprocess -> features -> train -> evaluate -> serialize -> load -> predict.

RUN:    cd 10_ml_system_architecture/01_pipeline_end_to_end && python main.py
        (from repo root: python 10_ml_system_architecture/01_pipeline_end_to_end/main.py)

EXPECTED OUTPUT (shape):
        [1 collect]   200 train texts, 8 test texts ...
        [2 preprocess] 'The movie was GREAT!!  ' -> 'the movie was great'
        [3 features]  matrix shape (200, 32) ...
        [4 train]     ...
        [5 evaluate]  accuracy = 1.00 ...
        [6 serialize] saved ... bytes
        [7 predict]   ...

NEEDS:  scikit-learn, joblib (installed via requirements.txt)
"""
```

- **In simple words:** The instruction label on the box.
- **Exactly:** Docstring listing the 7 stages and the shape of the output.
- **Why it is here:** Anyone opening the file knows what it is for and how to run it.

### 9. Imports: regex, temp folder, joblib, sklearn pieces, our data  (`main.py` lines 20–30)

```python
import re
import tempfile
from pathlib import Path

import joblib                                           # saves/loads Python objects (good with numpy-heavy ones)
from sklearn.feature_extraction.text import TfidfVectorizer   # text -> numeric feature matrix
from sklearn.linear_model import LogisticRegression           # a simple, fast classifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline                         # chains steps so they travel together

from data import get_test_data, get_training_data
```

- **In simple words:** Gather the tools: a text-cleaning tool, a place to save files, the model-saving tool, and the machine-learning parts.
- **Exactly:** `re` = regex, `tempfile` = throwaway folder, `Path` = file paths, `joblib` saves objects, `TfidfVectorizer` text-to-numbers, `LogisticRegression` classifier, `accuracy_score`/`classification_report` metrics, `Pipeline` chains steps. `from data import ...` works because data.py is in the same folder as the script.
- **Why it is here:** Each import is one stage of the pipeline; missing one is a `NameError` at that stage.

### 10. preprocess(): lowercase, strip punctuation, tidy spaces  (`main.py` lines 33–37)

```python
def preprocess(text: str) -> str:
    """Stage 2: clean raw text. lowercase, drop punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)             # regex: delete anything that is not a letter/digit/space
    return re.sub(r"\s+", " ", text).strip()
```

- **In simple words:** A cleaning function: SHOUTING becomes quiet, punctuation vanishes, extra spaces collapse.
- **Exactly:** `re.sub(r"[^a-z0-9\s]", "", text)` deletes every char that is not a-z, 0-9 or whitespace (`^` inside `[]` = NOT). `re.sub(r"\s+", " ", ...)` turns runs of whitespace into one space; `.strip()` trims ends. Real run: `'The movie was GREAT!!  '` -> `'the movie was great'`.
- **Why it is here:** Without consistent cleaning, 'Great' and 'great!' would be different features.

### 11. Stage 1 COLLECT: load train and test texts  (`main.py` lines 40–44)

```python
if __name__ == "__main__":
    # ---- 1. COLLECT: where data comes from (files, DB, API, user feedback). Here: generated. ----
    X_train_raw, y_train = get_training_data(n=200)
    X_test_raw, y_test = get_test_data()
    print(f"[1 collect]    {len(X_train_raw)} train texts, {len(X_test_raw)} test texts; first = {X_train_raw[0]!r}")
```

- **In simple words:** Get the raw data: 200 practice sentences and 8 exam sentences.
- **Exactly:** `if __name__ == "__main__":` runs the block only when the file is executed directly. `get_training_data(n=200)` returns two lists of length 200. Real output: `200 train texts, 8 test texts; first = 'The book was boring.'`. `!r` in the f-string prints with quotes (repr).
- **Why it is here:** Everything downstream depends on data; the print confirms sizes before spending time on training.

### 12. Stage 2 PREPROCESS: clean every text  (`main.py` lines 46–49)

```python
    # ---- 2. PREPROCESS: make inputs consistent. ----
    print("[2 preprocess]", repr("The movie was GREAT!!  "), "->", repr(preprocess("The movie was GREAT!!  ")))
    X_train = [preprocess(t) for t in X_train_raw]      # list comprehension: apply preprocess to every text
    X_test = [preprocess(t) for t in X_test_raw]
```

- **In simple words:** Run the cleaning function on all sentences.
- **Exactly:** List comprehension `[f(x) for x in xs]` applies `preprocess` to each element. `X_train` and `X_test` remain lists of 200 and 8 strings.
- **Why it is here:** Train and test must go through the same cleaning.

### 13. Stage 3 FEATURES: TF-IDF turns text into a matrix  (`main.py` lines 51–55)

```python
    # ---- 3. FEATURE ENGINEERING: turn text into numbers a model can use. ----
    vectorizer = TfidfVectorizer()                      # TF-IDF: words frequent in a doc but rare overall score high
    X_train_vec = vectorizer.fit_transform(X_train)     # fit = learn the vocabulary FROM TRAIN ONLY; transform = apply
    X_test_vec = vectorizer.transform(X_test)           # only transform on test: never fit on data you evaluate on!
    print(f"[3 features]   matrix shape {X_train_vec.shape} = (n_texts, vocabulary_size); sample words: {list(vectorizer.vocabulary_)[:5]}")
```

- **In simple words:** Count words, but give rare words more weight, so each sentence becomes a row of numbers.
- **Exactly:** TF-IDF weight = term frequency x log(inverse document frequency). `fit_transform` learns the vocabulary from train and returns a sparse matrix; `transform` reuses that vocabulary for test. Real output: shape `(200, 32)` = 200 sentences x 32 distinct words; vocabulary_ is a dict word -> column, sample `['the', 'book', 'was', 'boring', 'honestly']`.
- **Why it is here:** Models need numbers. Fitting on test data would leak test information into training.

### 14. Stage 4 TRAIN: fit logistic regression  (`main.py` lines 57–60)

```python
    # ---- 4. TRAIN ----
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)                       # learns one weight per word per class
    print(f"[4 train]      classes learned: {[str(c) for c in clf.classes_]}")
```

- **In simple words:** Let the model look at 200 labelled examples and adjust one weight per word.
- **Exactly:** `LogisticRegression(max_iter=1000)` is a linear classifier; `max_iter` caps the optimiser's iterations. `fit(X, y)` learns a coefficient matrix. Real output: `classes_` = `['negative', 'positive']` (sorted alphabetically).
- **Why it is here:** This is where the 'learning' happens; the whole pipeline exists to feed this one call.

### 15. Stage 5 EVALUATE on unseen test data  (`main.py` lines 62–65)

```python
    # ---- 5. EVALUATE on UNSEEN data ----
    pred = clf.predict(X_test_vec)
    print(f"[5 evaluate]   accuracy = {accuracy_score(y_test, pred):.2f}")
    print(classification_report(y_test, pred, zero_division=0))
```

- **In simple words:** Give the exam and mark it.
- **Exactly:** `predict` returns an array of 8 labels. `accuracy_score` = correct/total; real result `1.00` (8/8). `classification_report` prints precision/recall/f1 per class (4 and 4 support). `zero_division=0` avoids warnings if a class is never predicted.
- **Why it is here:** A training score is not trustworthy; only unseen data shows whether the model generalises.

### 16. Stage 6 SERIALIZE: bundle cleaning, features, model and save  (`main.py` lines 67–76)

```python
    # ---- 6. SERIALIZE: bundle preprocessing-features-model so serving can reuse EXACTLY the same steps ----
    # TfidfVectorizer already lowercases and strips punctuation while tokenizing, so the cleaning lives INSIDE the
    # pipeline using only built-in sklearn behaviour. (A custom function like preprocess() would be pickled BY
    # NAME and break when loaded from another script -- see step 03.)
    pipeline = Pipeline([("tfidf", TfidfVectorizer()),
                         ("clf", LogisticRegression(max_iter=1000))])
    pipeline.fit(X_train_raw, y_train)                  # raw text in; the pipeline cleans + vectorizes + trains
    out = Path(tempfile.mkdtemp()) / "model.joblib"
    joblib.dump(pipeline, out)
    print(f"[6 serialize]  saved {out.stat().st_size} bytes to {out.name}")
```

- **In simple words:** Pack the cleaning, vectoriser and classifier into one box and save it as a file the server can open later.
- **Exactly:** `Pipeline([(name, step), ...])` runs steps in order; fitting on raw text means the vectoriser's own lowercase/tokenise is inside. `tempfile.mkdtemp()` makes a temp dir; `joblib.dump(obj, path)` writes it. Real output: `saved 2481 bytes to model.joblib`. The step list is refitted because the earlier `vectorizer`/`clf` were separate objects.
- **Why it is here:** The server must apply exactly the same steps as training. Shipping the model without the vectoriser is the classic training/serving skew bug (step 02).

### 17. Stage 7 LOAD and PREDICT one new sentence  (`main.py` lines 78–82)

```python
    # ---- 7. LOAD + PREDICT (this is what the API does at startup and per request) ----
    loaded = joblib.load(out)
    sample = "Honestly the delivery was TERRIBLE!!"
    probs = loaded.predict_proba([sample])[0]
    print(f"[7 predict]    {sample!r} -> {loaded.predict([sample])[0]}  probs={ {str(c): round(float(p), 3) for c, p in zip(loaded.classes_, probs)} }")
```

- **In simple words:** Open the saved box like the API does and ask it about a new review.
- **Exactly:** `joblib.load` rebuilds the pipeline. `predict_proba([sample])` returns shape (1, 2) of class probabilities in `classes_` order; `[0]` takes the first row. Real output: `negative`, `{'negative': 0.726, 'positive': 0.274}`. Note the input is a list of one string, since sklearn expects a batch.
- **Why it is here:** Proves the saved artifact works on its own, without any training code.

### 18. Comments: things to try and break  (`main.py` lines 83–85)

```python
    # TRY IT: change n=200 to n=10 and watch accuracy / confidence change.
    # TRY IT: predict on 'GREAT!!!' vs 'great' -- same answer? Why? (hint: TfidfVectorizer lowercases and tokenizes)
    # BREAK IT: call vectorizer.transform(...) BEFORE fit_transform and read the NotFittedError.
```

- **In simple words:** Suggested experiments.
- **Exactly:** Comment lines only; they do not run.
- **Why it is here:** They point to experiments that teach the pipeline (see Experiments below).


## Syntax explained
| Syntax | Meaning | Example |
|---|---|---|
| `[preprocess(t) for t in X]` | List comprehension | `[len(w) for w in words]` |
| `re.sub(r"[^a-z0-9\s]", "", text)` | Regex replace: `r"..."` raw string; `[^...]` = "anything except" | — |
| `random.Random(seed)` | Private random generator → reproducible data | — |
| `zip(*pairs)` | Unzip a list of pairs into two sequences | `zip(*[(1,'a'),(2,'b')])` |
| `f"{x:.2f}"` | f-string with 2 decimals | `f"{0.5:.2f}"` → `0.50` |
| `Path(tempfile.mkdtemp()) / "model.joblib"` | Temp folder + path join | — |

## Libraries & functions
| Name | Kind | Does | Returns |
|---|---|---|---|
| `TfidfVectorizer()` | scikit-learn | Text → numeric matrix; weights words frequent in a doc but rare overall. Lowercases and tokenizes by default | estimator |
| `.fit_transform(train)` | sklearn | Learn vocabulary **and** transform | sparse matrix (n_texts × n_words) |
| `.transform(test)` | sklearn | Apply the **already learned** vocabulary | sparse matrix |
| `LogisticRegression(max_iter=1000)` | sklearn | Linear classifier; learns a weight per word per class | estimator |
| `.fit(X, y)` / `.predict(X)` / `.predict_proba(X)` | sklearn | train / labels / class probabilities | — |
| `accuracy_score(y_true, y_pred)` | sklearn | fraction correct | float |
| `Pipeline([("tfidf", ...), ("clf", ...)])` | sklearn | Chain steps into one object (`fit`, `predict` run all steps) | estimator |
| `joblib.dump / load` | 3rd-party | Save / load object to a file | — |

## Execution trace
`predict_proba(["Honestly the delivery was TERRIBLE!!"])`: Pipeline → TfidfVectorizer lowercases + tokenizes → `['honestly','the','delivery','was','terrible']` → looks up each in the *training* vocabulary → builds a 1×32 vector → LogisticRegression multiplies by learned weights → softmax-like squash → `[0.726, 0.274]` for `[negative, positive]`.

## Why each stage exists (and what breaks it)
| Stage | Purpose | Typical failure |
|---|---|---|
| Collect | get representative data | biased/stale data → model that's wrong on real users |
| Preprocess | consistent clean input | different cleaning in training vs serving (step 02) |
| Features | numbers the model understands | fitting the vectorizer on test data = **data leakage** |
| Train | learn weights | overfitting; use validation data |
| Evaluate | honest estimate on unseen data | evaluating on training data; test set too small (8 here is a toy!) |
| Serialize | ship the exact fitted object | pickle/library version mismatch (step 03) |
| Predict | serve | cold start, latency (steps 06, 08) |

## Experiments
1. `n=10` instead of `200` → watch confidence fall and the vocabulary shrink.
2. Predict `'GREAT!!!'` vs `'great'` — same answer because the vectorizer lowercases/tokenizes.
3. Predict a sentence of words never seen in training: `'The zorp was blorf'` → probabilities near 0.5 (the model knows nothing).

## Debugging practice
Call `vectorizer.transform(X_test)` before `fit_transform` → `NotFittedError: This TfidfVectorizer instance is not fitted yet`. The message names the exact fix (`fit` first). Reading library errors fully usually saves a trip to a search engine.

## Rebuild checklist
- [ ] Rebuild the Pipeline and train/predict in ≤15 lines.
- [ ] Say which stage each of these belongs to: `re.sub`, `fit_transform`, `joblib.dump`, `accuracy_score`.
- [ ] Explain why you never `fit` on test data.

## The new problem (cliffhanger)
One script that trains and predicts hides a trap: in production training and serving are two different programs. Do they still agree? Next: [02_training_vs_inference_pipelines](../02_training_vs_inference_pipelines/).

## Say it aloud
> **Say it aloud (interview box):** An ML system is a pipeline: collect data, clean it, build features, train, evaluate on held-out data, save the fitted artifact, then load and serve it. I bundle preprocessing and the model in one scikit-learn Pipeline so training and serving cannot drift apart. The catch is that a pipeline hides steps I must still inspect when debugging, and a high score on a tiny template-based test set says little about real data.
