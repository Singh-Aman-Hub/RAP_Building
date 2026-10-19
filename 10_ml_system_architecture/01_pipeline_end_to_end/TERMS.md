# Terms — Step 01: The end-to-end ML pipeline

Folder: `10_ml_system_architecture/01_pipeline_end_to_end` · Files covered: `main.py`, `data.py`, `WALKTHROUGH.md`
Back to module: [10 ML system architecture](../README.md)

**Terms in this folder:** [ML pipeline (end to end)](#ml-pipeline-end-to-end-concept) · [Data collection](#data-collection-concept) · [Reproducibility with a seed](#reproducibility-with-a-seed-concept) · [random.Random(seed)](#randomrandomseed-standard-library) · [rng.random() and rng.choice()](#rngrandom-and-rngchoice-standard-library) · [Template sentence with str.format](#template-sentence-with-strformat-python-built-in) · [zip(*pairs) unzip idiom](#zippairs-unzip-idiom-python-built-in) · [Type hint tuple[list[str], list[str]]](#type-hint-tupleliststr-liststr-syntax) · [Held-out test set](#held-out-test-set-concept) · [Preprocessing](#preprocessing-concept) · [re.sub](#resub-standard-library) · [Regular expression and raw string r"..."](#regular-expression-and-raw-string-r-syntax) · [str methods lower() and strip()](#str-methods-lower-and-strip-python-built-in) · [List comprehension](#list-comprehension-syntax) · [Feature engineering](#feature-engineering-concept) · [TF-IDF (TfidfVectorizer)](#tf-idf-tfidfvectorizer-third-party-scikit-learn) · [fit_transform vs transform](#fit_transform-vs-transform-third-party-scikit-learn) · [Data leakage](#data-leakage-concept) · [vocabulary_](#vocabulary_-third-party-scikit-learn) · [Sparse matrix shape (n_texts, vocabulary_size)](#sparse-matrix-shape-n_texts-vocabulary_size-concept) · [LogisticRegression](#logisticregression-third-party-scikit-learn) · [fit / predict / predict_proba](#fit-predict-predict_proba-third-party-scikit-learn) · [classes_](#classes_-third-party-scikit-learn) · [accuracy_score](#accuracy_score-third-party-scikit-learn) · [classification_report](#classification_report-third-party-scikit-learn) · [Pipeline (sklearn)](#pipeline-sklearn-third-party-scikit-learn) · [Model serialization](#model-serialization-concept) · [joblib.dump / joblib.load](#joblibdump-joblibload-third-party-joblib) · [Custom functions pickled by name](#custom-functions-pickled-by-name-concept) · [tempfile.mkdtemp](#tempfilemkdtemp-standard-library) · [Path.stat().st_size](#pathstatst_size-standard-library) · [Dict comprehension for probabilities](#dict-comprehension-for-probabilities-syntax) · [f-string format specs (:.2f, !r)](#f-string-format-specs-2f-r-syntax) · [NotFittedError](#notfittederror-third-party-scikit-learn) · [Unseen words](#unseen-words-concept) · [Why each stage fails differently](#why-each-stage-fails-differently-concept)

---

## ML pipeline (end to end)  `(concept)`
**1. Definition:** The ordered sequence of stages a machine-learning system goes through: collect data, preprocess, engineer features, train, evaluate, serialize, load and predict.
**2. In plain words:** It is the assembly line that turns raw data into a served prediction. Every box in an architecture diagram has code behind it, and a failure in any stage (biased data, inconsistent cleaning, leakage) shows up as a bad model or a bad service. Thinking in stages helps you debug and explain a system.
**3. Example to quote:**
> "My pipeline is: generate 200 labelled sentences, clean them, TF-IDF them, train logistic regression, evaluate on 8 held-out sentences, save with joblib, then load and predict."

**Seen in:** whole step

## Data collection  `(concept)`
**1. Definition:** Gathering the raw examples (from files, databases, APIs, user feedback or generation) that the model will learn from.
**2. In plain words:** Your model can only be as good as the data it sees. If the data is biased or stale, the model will be wrong on real users. Here the data is generated from templates, which is fine for a demo but unrealistically easy.
**3. Example to quote:**
> "In production I would pull labelled feedback from a database; here `get_training_data` generates 200 sentences from templates."

**Seen in:** `data.py`, stage `[1 collect]`

## Reproducibility with a seed  `(concept)`
**1. Definition:** Fixing the starting state of a random number generator (a seed) so the same random sequence, and therefore the same data, is produced every run.
**2. In plain words:** Without a seed your results change each run, so you cannot tell if a change helped. With `seed=0` the dataset is identical every time, so accuracy numbers are comparable.
**3. Example to quote:**
> "`get_training_data(n=200, seed=0)` always gives the same 200 sentences, so I can compare runs fairly."

**Seen in:** `get_training_data`

## random.Random(seed)  `(standard library)`
**1. Definition:** Creates a private random number generator object initialised with a seed; its methods (`random()`, `choice()`) do not affect the global random state.
**2. In plain words:** Using your own generator means other code calling `random` cannot disturb your reproducible sequence, and you do not disturb theirs.
**3. Example to quote:**
> "`rng = random.Random(seed)` gives me a generator only my data function uses."

```python
import random
r = random.Random(0)
print(r.random() == random.Random(0).random())   # True
```

**Seen in:** `get_training_data`

## rng.random() and rng.choice()  `(standard library)`
**1. Definition:** `random()` returns a float in [0, 1); `choice(seq)` returns a random element of a non-empty sequence.
**2. In plain words:** `rng.random() < 0.5` is a coin flip that decides positive or negative; `rng.choice(POS)` picks an adjective. Together they build each template sentence.
**3. Example to quote:**
> "A coin flip picks the class, then `rng.choice(POS if positive else NEG)` picks a matching adjective."

**Seen in:** `get_training_data`

## Template sentence with str.format  `(Python built-in)`
**1. Definition:** `"The {n} was {a}.".format(n="movie", a="great")` replaces named placeholders in a string with the given values.
**2. In plain words:** It builds many different sentences from a few patterns. Placeholders are named, so the order of arguments does not matter.
**3. Example to quote:**
> "`rng.choice(TEMPLATES).format(n=..., a=...)` produces text like 'What a brilliant hotel.'"

```python
print("The {n} was {a}.".format(n="movie", a="great"))
```

**Seen in:** `get_training_data`

## zip(*pairs) unzip idiom  `(Python built-in)`
**1. Definition:** `zip(*list_of_pairs)` transposes a list of pairs into separate sequences, here all texts and all labels.
**2. In plain words:** `TEST_SET` is a list of `(text, label)` tuples; unzipping gives one tuple of texts and one of labels, ready for the model and the metric. Wrap with `list(...)` if you need lists.
**3. Example to quote:**
> "`texts, labels = zip(*TEST_SET)` splits my eight pairs into a texts tuple and a labels tuple."

```python
print(list(zip(*[(1, "a"), (2, "b")])))   # [(1, 2), ('a', 'b')]
```

**Seen in:** `get_test_data`

## Type hint tuple[list[str], list[str]]  `(syntax)`
**1. Definition:** An annotation stating that a function returns a tuple holding two lists of strings; it is documentation, not enforced.
**2. In plain words:** It tells the reader that `get_training_data` returns `(texts, labels)`. Editors use it for autocompletion.
**3. Example to quote:**
> "`-> tuple[list[str], list[str]]` says I return a list of texts and a list of labels."

**Seen in:** `data.py`

## Held-out test set  `(concept)`
**1. Definition:** A set of labelled examples never used during training and used only to estimate how the model performs on unseen data.
**2. In plain words:** Testing on training data tells you how well the model memorised, not how well it generalises. Here eight hand-written sentences serve as the test set; eight is a toy size, so a perfect score means little.
**3. Example to quote:**
> "My test sentences are handwritten and never trained on, so accuracy 1.00 is an honest, though tiny, check."

**Seen in:** `TEST_SET`, stage `[5 evaluate]`

## Preprocessing  `(concept)`
**1. Definition:** Cleaning and normalising raw input before modelling, for example lowercasing, removing punctuation and collapsing whitespace.
**2. In plain words:** Models treat "GREAT!!" and "great" as different unless you clean them. The key rule is that exactly the same cleaning must run in training and in serving, otherwise you get training/serving skew (step 02).
**3. Example to quote:**
> "`preprocess('The movie was GREAT!!  ')` returns `'the movie was great'`."

**Seen in:** `preprocess`, stage `[2 preprocess]`

## re.sub  `(standard library)`
**1. Definition:** `re.sub(pattern, replacement, text)` replaces every match of a regular expression in `text` with `replacement`.
**2. In plain words:** `re.sub(r"[^a-z0-9\s]", "", text)` deletes everything that is not a lowercase letter, digit or whitespace. `re.sub(r"\s+", " ", text)` collapses runs of whitespace into one space.
**3. Example to quote:**
> "I use `re.sub` twice: once to strip punctuation and once to collapse repeated spaces."

```python
import re
print(re.sub(r"[^a-z]", "", "a1b2"))   # ab
```

**Seen in:** `preprocess`

## Regular expression and raw string r"..."  `(syntax)`
**1. Definition:** A regular expression is a pattern language for matching text; a raw string `r"..."` stops Python from interpreting backslashes so the pattern reaches `re` unchanged.
**2. In plain words:** `[^a-z0-9\s]` means "any character that is NOT a lowercase letter, digit or whitespace". Without the `r` prefix you would need to double every backslash.
**3. Example to quote:**
> "`r\"[^a-z0-9\\s]\"` matches any punctuation, and the `r` prefix means I don't have to escape the `\\s`."

**Seen in:** `preprocess`

## str methods lower() and strip()  `(Python built-in)`
**1. Definition:** `lower()` returns a lowercase copy of a string; `strip()` removes leading and trailing whitespace.
**2. In plain words:** They are the first, simplest normalisation steps. Strings are immutable, so each returns a new string.
**3. Example to quote:**
> "`text.lower()` makes 'GREAT' into 'great' and `.strip()` removes trailing spaces."

```python
print("  Hi ".strip().lower())   # hi
```

**Seen in:** `preprocess`

## List comprehension  `(syntax)`
**1. Definition:** `[expr for item in iterable]` builds a list by evaluating `expr` for each item.
**2. In plain words:** `[preprocess(t) for t in X_train_raw]` cleans every text in one line. It is the Pythonic version of a loop that appends to a list.
**3. Example to quote:**
> "`X_train = [preprocess(t) for t in X_train_raw]` applies my cleaning function to all 200 texts."

**Seen in:** `main.py` stage 2

## Feature engineering  `(concept)`
**1. Definition:** Transforming raw data into numeric features a model can use, such as word counts or TF-IDF weights for text.
**2. In plain words:** Models need numbers, not sentences. How you turn text into numbers strongly affects what the model can learn. The vectorizer here builds a table with one column per vocabulary word.
**3. Example to quote:**
> "My feature matrix is `(200, 32)`: 200 texts, each described by weights for 32 vocabulary words."

**Seen in:** stage `[3 features]`

## TF-IDF (TfidfVectorizer)  `(third-party: scikit-learn)`
**1. Definition:** `TfidfVectorizer` converts text into a sparse matrix of TF-IDF weights, which score words high when they are frequent in a document but rare across all documents; it lowercases and tokenizes by default.
**2. In plain words:** Common words like "the" get low weight and distinctive words like "terrible" get high weight. It gives the classifier a useful signal without any deep learning. Unknown words at prediction time are simply ignored.
**3. Example to quote:**
> "`TfidfVectorizer()` turned 200 sentences into a 200 by 32 matrix, and it lowercased the text for me."

```python
from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer().fit(["a good day", "a bad day"])
print(sorted(v.vocabulary_))   # ['bad', 'day', 'good']
```

**Seen in:** `main.py` stage 3

## fit_transform vs transform  `(third-party: scikit-learn)`
**1. Definition:** `fit_transform(train)` learns the vectorizer's vocabulary from the training data and applies it; `transform(other)` applies the already learned vocabulary without learning anything new.
**2. In plain words:** Fit on training data only. For the test set (and every live request) you only `transform`, so the model sees features built from the same vocabulary it learned from. Calling `transform` before `fit` raises `NotFittedError`.
**3. Example to quote:**
> "I call `fit_transform` on train and just `transform` on test, so the test set never influences the vocabulary."

**Seen in:** `main.py` stage 3

## Data leakage  `(concept)`
**1. Definition:** Information from outside the training data (such as the test set) accidentally influencing the model or its preprocessing, making evaluation look better than it really is.
**2. In plain words:** If you fit the vectorizer on test data, the model has indirectly "seen" the test words, so your score is optimistic. Rule: anything that learns (`fit`) sees training data only.
**3. Example to quote:**
> "Fitting the vectorizer on the test set would be leakage, so I only call `transform` on it."

**Seen in:** stage 3, `WALKTHROUGH.md` table

## vocabulary_  `(third-party: scikit-learn)`
**1. Definition:** An attribute of a fitted `TfidfVectorizer`: a dict mapping each learned word to its column index.
**2. In plain words:** It shows which words the model can "see" and where each lives in the matrix. Words outside it are invisible at prediction time.
**3. Example to quote:**
> "`list(vectorizer.vocabulary_)[:5]` showed words like 'the', 'book', 'was', 'boring' and 'honestly'."

**Seen in:** stage `[3 features]`

## Sparse matrix shape (n_texts, vocabulary_size)  `(concept)`
**1. Definition:** The `.shape` of the feature matrix: rows are documents and columns are vocabulary words; most entries are zero, so it is stored sparsely.
**2. In plain words:** `(200, 32)` means 200 sentences described by 32 possible words. A large vocabulary makes a very wide, mostly empty table, which is why sparse storage is used.
**3. Example to quote:**
> "`X_train_vec.shape` is `(200, 32)`: one row per sentence, one column per word."

**Seen in:** stage `[3 features]`

## LogisticRegression  `(third-party: scikit-learn)`
**1. Definition:** A linear classification model that learns one weight per feature per class and converts the weighted sum into class probabilities; `max_iter` limits optimisation steps.
**2. In plain words:** It is a simple, fast, interpretable baseline. For text it learns which words push toward "positive" or "negative". `max_iter=1000` avoids a convergence warning on small data.
**3. Example to quote:**
> "I use `LogisticRegression(max_iter=1000)` as a quick baseline; it learns a weight per word per class."

**Seen in:** `main.py` stage 4

## fit / predict / predict_proba  `(third-party: scikit-learn)`
**1. Definition:** `fit(X, y)` trains the estimator, `predict(X)` returns class labels, `predict_proba(X)` returns class probabilities per row.
**2. In plain words:** This three-method API is shared by almost all scikit-learn models, so you can swap models without changing the surrounding code. `predict_proba` gives confidence, not just a label.
**3. Example to quote:**
> "`pipeline.fit(X_train_raw, y_train)` trains, and `predict_proba` later returns `[0.726, 0.274]` for `[negative, positive]`."

**Seen in:** stages 4, 5, 7

## classes_  `(third-party: scikit-learn)`
**1. Definition:** An attribute of a fitted classifier listing its class labels in the column order of `predict_proba`.
**2. In plain words:** It is how you know which probability belongs to which label. Here the order is `['negative', 'positive']`.
**3. Example to quote:**
> "I zip `loaded.classes_` with the probabilities to print `{'negative': 0.726, 'positive': 0.274}`."

**Seen in:** stages 4 and 7

## accuracy_score  `(third-party: scikit-learn)`
**1. Definition:** `accuracy_score(y_true, y_pred)` returns the fraction of predictions that match the true labels.
**2. In plain words:** Accuracy is the simplest quality number but can mislead on imbalanced data or tiny test sets. Eight test sentences means one mistake changes it by 12.5 points.
**3. Example to quote:**
> "Accuracy 1.00 on 8 sentences sounds great, but it is only a smoke check, not strong evidence."

**Seen in:** stage 5

## classification_report  `(third-party: scikit-learn)`
**1. Definition:** Prints precision, recall, F1-score and support per class for a set of predictions.
**2. In plain words:** It breaks quality down by class, so you can see if the model is good at one label but poor at the other. `zero_division=0` avoids warnings when a class is never predicted.
**3. Example to quote:**
> "The report lists precision and recall for 'negative' and 'positive' separately, with 4 test examples each."

**Seen in:** stage 5

## Pipeline (sklearn)  `(third-party: scikit-learn)`
**1. Definition:** `Pipeline([(name, step), ...])` chains transformers and a final estimator so `fit` and `predict` run all steps in order on one object.
**2. In plain words:** The vectorizer and classifier travel together as a single artifact, so serving automatically applies the same preprocessing as training. That prevents many training/serving mismatches.
**3. Example to quote:**
> "My `Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())])` takes raw text in and gives labels out."

**Seen in:** stage 6

## Model serialization  `(concept)`
**1. Definition:** Saving a fitted model object to a file so it can be loaded later by a different process.
**2. In plain words:** Training happens once; serving happens many times in other processes. The saved file is the handoff. The loader needs compatible library versions, and custom functions pickled by name break when loaded from another script.
**3. Example to quote:**
> "I `joblib.dump` the whole pipeline to `model.joblib`; the API loads it once at startup."

**Seen in:** stage `[6 serialize]`

## joblib.dump / joblib.load  `(third-party: joblib)`
**1. Definition:** `joblib.dump(obj, path)` saves a Python object to disk and `joblib.load(path)` restores it; it is efficient for objects containing large numpy arrays.
**2. In plain words:** The standard way to save scikit-learn models. Like pickle, only load files you trust, because loading can run code.
**3. Example to quote:**
> "`joblib.dump(pipeline, out)` wrote 2,481 bytes; `joblib.load(out)` returned an identical working pipeline."

**Seen in:** stages 6 and 7

## Custom functions pickled by name  `(concept)`
**1. Definition:** When a pickled object refers to a function you defined (like `preprocess`), pickle stores only its name and module, not its code, so loading in another script fails if that function is not importable there.
**2. In plain words:** That is why the cleaning here lives inside the pipeline using only built-in scikit-learn behaviour (the vectorizer lowercases itself). A hidden dependency on your own function would break the API when it loads the model.
**3. Example to quote:**
> "I don't put my own `preprocess` function in the saved pipeline, because the API wouldn't be able to find it when it loads the file."

**Seen in:** stage 6 comment

## tempfile.mkdtemp  `(standard library)`
**1. Definition:** Creates a new unique temporary directory and returns its path as a string; it is not deleted automatically.
**2. In plain words:** The demo saves the model into a scratch folder so it does not leave files in your project.
**3. Example to quote:**
> "`Path(tempfile.mkdtemp()) / 'model.joblib'` gives a throwaway file location."

**Seen in:** stage 6

## Path.stat().st_size  `(standard library)`
**1. Definition:** `Path.stat()` returns file metadata; `.st_size` is the file size in bytes.
**2. In plain words:** It confirms that the file was written and shows how big the model artifact is.
**3. Example to quote:**
> "`out.stat().st_size` reported 2,481 bytes for my model file."

**Seen in:** stage 6

## Dict comprehension for probabilities  `(syntax)`
**1. Definition:** `{k: v for k, v in zip(a, b)}` builds a dict by pairing keys and values from two sequences.
**2. In plain words:** It labels each probability with its class name so the output is readable. `str(c)` and `float(p)` convert numpy types to plain Python.
**3. Example to quote:**
> "`{str(c): round(float(p), 3) for c, p in zip(loaded.classes_, probs)}` prints probabilities with label names."

```python
print({k: v for k, v in zip("ab", [1, 2])})   # {'a': 1, 'b': 2}
```

**Seen in:** stage 7

## f-string format specs (:.2f, !r)  `(syntax)`
**1. Definition:** Inside an f-string, `{x:.2f}` formats a number to 2 decimals and `{s!r}` inserts `repr(s)`, which includes quotes.
**2. In plain words:** `.2f` makes accuracy print as `1.00`; `!r` shows the sample sentence quoted so spaces and edge cases are visible.
**3. Example to quote:**
> "`f'accuracy = {accuracy_score(y_test, pred):.2f}'` prints `accuracy = 1.00`."

```python
print(f"{0.5:.2f}")   # 0.50
```

**Seen in:** stages 1, 5, 7

## NotFittedError  `(third-party: scikit-learn)`
**1. Definition:** An exception raised when you use an estimator (for example `transform` or `predict`) before calling `fit`.
**2. In plain words:** The message names the fix: fit first. It is also a reminder that a loaded model must be a fitted one.
**3. Example to quote:**
> "Calling `vectorizer.transform(X_test)` before `fit_transform` raises `NotFittedError: This TfidfVectorizer instance is not fitted yet`."

**Seen in:** `BREAK IT` comment, `WALKTHROUGH.md`

## Unseen words  `(concept)`
**1. Definition:** Words absent from the training vocabulary are ignored by the vectorizer, so a sentence made only of them yields an all-zero feature vector.
**2. In plain words:** The classifier then relies on its bias and outputs probabilities near 0.5. It shows the model knows nothing about words it never saw, which is why training data must resemble real input.
**3. Example to quote:**
> "'The zorp was blorf' gets probabilities close to 0.5 because the model has never seen 'zorp' or 'blorf'."

**Seen in:** `WALKTHROUGH.md` experiment 3

## Why each stage fails differently  `(concept)`
**1. Definition:** Each pipeline stage has its own typical failure: biased data (collect), train/serve mismatch (preprocess), leakage (features), overfitting (train), wrong evaluation (evaluate), version mismatch (serialize), latency (predict).
**2. In plain words:** Knowing the typical failure per stage turns "the model is bad" into a short checklist you can walk through.
**3. Example to quote:**
> "If accuracy is great offline but poor live, I first suspect different preprocessing in serving, then data drift."

**Seen in:** `WALKTHROUGH.md` table
