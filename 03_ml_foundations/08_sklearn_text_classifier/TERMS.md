# Terms — scikit-learn text classifier

Folder: `03_ml_foundations/08_sklearn_text_classifier` · Files covered: `train.py`, `predict.py`, `test_predict.py`, `data.py`, `WALKTHROUGH.md` (the shared module-level `train.py`/`predict.py` are copies of the same ideas)
Back to module: [ML foundations README](../README.md)

**Terms in this folder:** [Text classification](#text-classification--concept) · [TF-IDF (TfidfVectorizer)](#tf-idf-tfidfvectorizer--third-party-scikit-learn) · [n-gram (`ngram_range=(1, 2)`)](#n-gram-ngram_range1-2--concept) · [Logistic regression](#logistic-regression--third-party-scikit-learn) · [Pipeline](#pipeline--third-party-scikit-learn) · [train_test_split()](#train_test_split--third-party-scikit-learn) · [fit()](#fit--third-party-scikit-learn) · [predict() and predict_proba()](#predict-and-predict_proba--third-party-scikit-learn) · [classification_report](#classification_report--third-party-scikit-learn) · [Stratified split](#stratified-split--concept) · [Model serialisation (joblib.dump / joblib.load)](#model-serialisation-joblibdump--joblibload--third-party-joblib) · [Model artifact and metadata.json](#model-artifact-and-metadatajson--concept) · [pathlib.Path(__file__)](#pathlibpath__file__--standard-library-pathlib) · [mkdir(exist_ok=True)](#mkdirexist_oktrue--standard-library-pathlib) · [json.dumps / json.loads](#jsondumps--jsonloads--standard-library-json) · [datetime with timezone](#datetime-with-timezone--standard-library-datetime) · [@lru_cache](#lru_cache--syntax) · [Decorator](#decorator--syntax) · [sys.argv](#sysargv--standard-library-sys) · [isinstance() and input validation](#isinstance-and-input-validation--python-built-in) · [argmax](#argmax--third-party-numpy) · [model.classes_](#modelclasses_--third-party-scikit-learn) · [pytest (assert, pytest.raises)](#pytest-assert-pytestraises--third-party-pytest) · [Confidence (probability of chosen label)](#confidence-probability-of-chosen-label--concept) · [Train/serve consistency (preprocessing must match)](#trainserve-consistency-preprocessing-must-match--concept) · [Dataset (hand-written plus template-generated)](#dataset-hand-written-plus-template-generated--concept)

---

## Text classification  `(concept)`
**1. Definition:** Text classification assigns a piece of text to one of a fixed set of labels, such as positive or negative sentiment.
**2. In plain words:** Computers need numbers, so the text is first converted into numeric features, then a classifier learns which features signal each label. This folder builds the full flow: data, train, evaluate, save, load, predict.
**3. Example to quote:**
> "My model reads 'great food' and answers positive with 0.675 confidence."

**Seen in:** `train.py`, `predict.py`

## TF-IDF (TfidfVectorizer)  `(third-party: scikit-learn)`
**1. Definition:** `TfidfVectorizer` converts text into numeric vectors where each word's weight is its frequency in the document multiplied by how rare it is across all documents (term frequency × inverse document frequency).
**2. In plain words:** Words that appear in every text ("the") get low weight; distinctive words ("terrible") get high weight. It learns its vocabulary from the training texts only. Unseen words are ignored, which is why very unfamiliar text gives near-50% confidence.
**3. Example to quote:**
> "`TfidfVectorizer` turns each sentence into a vector where rare, informative words count for more than common ones."

**Seen in:** `train.py`, `build_pipeline`

## n-gram (`ngram_range=(1, 2)`)  `(concept)`
**1. Definition:** An n-gram is a sequence of n consecutive words; `ngram_range=(1, 2)` makes the vectoriser use single words and pairs of adjacent words as features.
**2. In plain words:** Pairs capture short phrases such as "not good" that single words miss. The cost is a larger vocabulary.
**3. Example to quote:**
> "With bigrams, 'not good' becomes one feature, so the model can learn that it means something different from 'good'."

**Seen in:** `train.py`, `TfidfVectorizer(ngram_range=(1, 2))`

## Logistic regression  `(third-party: scikit-learn)`
**1. Definition:** `LogisticRegression` is a linear classifier that applies a sigmoid or softmax to a weighted sum of features to output class probabilities.
**2. In plain words:** Despite the name it is classification, not regression. It learns one weight per word; positive weights push towards one label. It is simple, fast and a strong baseline. `max_iter=1000` allows enough optimisation steps to converge.
**3. Example to quote:**
> "`LogisticRegression` learns a weight for every word, and `predict_proba` turns the weighted sum into class probabilities."

**Seen in:** `train.py`, `build_pipeline`

## Pipeline  `(third-party: scikit-learn)`
**1. Definition:** `Pipeline([(name, step), ...])` chains preprocessing steps and a final model into one object with a single `fit` and `predict`.
**2. In plain words:** Preprocessing and model travel together, so inference does exactly the same transformation as training, preventing train/serve skew. It also lets you save one file that contains everything.
**3. Example to quote:**
> "My pipeline is TF-IDF then logistic regression, saved as one object, so the API cannot forget the preprocessing step."

**Seen in:** `train.py`, `build_pipeline`

## train_test_split()  `(third-party: scikit-learn)`
**1. Definition:** `train_test_split(X, y, test_size, random_state, stratify)` randomly splits data into training and validation (test) subsets.
**2. In plain words:** `test_size=0.25` holds back a quarter of the data. `random_state=42` makes the split repeatable. `stratify=labels` keeps the class proportions equal in both parts.
**3. Example to quote:**
> "I use `test_size=0.25` and `stratify=labels`, so 135 samples train and 45 validate with balanced classes."

**Seen in:** `train.py`, `main`

## fit()  `(third-party: scikit-learn)`
**1. Definition:** `estimator.fit(X, y)` learns the model's parameters from the training data.
**2. In plain words:** For the pipeline, it learns the TF-IDF vocabulary and the classifier weights from the training texts only. Calling `predict` before `fit` raises `NotFittedError`.
**3. Example to quote:**
> "`model.fit(X_train, y_train)` learns from training data only, never from validation."

**Seen in:** `train.py`, `model.fit`

## predict() and predict_proba()  `(third-party: scikit-learn)`
**1. Definition:** `predict(X)` returns the predicted class label for each input; `predict_proba(X)` returns an array of class probabilities per input.
**2. In plain words:** `predict_proba` has shape `(n_texts, n_classes)`; the API uses it to report confidence. `argmax` of each row picks the label.
**3. Example to quote:**
> "`predict_proba(['great food'])` returns something like `[[0.325, 0.675]]`, and I report the larger as the confidence."

**Seen in:** `train.py`, `predict.py`

## classification_report  `(third-party: scikit-learn)`
**1. Definition:** `classification_report(y_true, y_pred)` produces a per-class table of precision, recall, F1 and support, plus accuracy and averages; `output_dict=True` returns it as a dictionary.
**2. In plain words:** The one-call summary of how the classifier does. `zero_division=0` avoids warnings when a class is never predicted. Here: accuracy 0.80 on 45 validation samples.
**3. Example to quote:**
> "The report shows precision 0.77 and recall 0.87 for negative, and overall accuracy of 0.80."

**Seen in:** `train.py`, `main`

## Stratified split  `(concept)`
**1. Definition:** A stratified split preserves the proportion of each class in both the training and validation subsets.
**2. In plain words:** Without it, a random split might put most of one class in validation. It matters most for small or imbalanced datasets.
**3. Example to quote:**
> "`stratify=labels` ensures each split has roughly half positive and half negative samples."

**Seen in:** `train.py`, `train_test_split`

## Model serialisation (joblib.dump / joblib.load)  `(third-party: joblib)`
**1. Definition:** `joblib.dump(obj, path)` writes a Python object, such as a trained scikit-learn pipeline, to disk; `joblib.load(path)` reads it back.
**2. In plain words:** Train once, use many times: the API loads the file instead of retraining. The security caveat: joblib/pickle files can execute code when loaded, so only load files you trust. Library versions must match between save and load.
**3. Example to quote:**
> "I save the whole pipeline with `joblib.dump(model, 'model.joblib')` and load it in the API at startup."

**Seen in:** `train.py`, `predict.py`

## Model artifact and metadata.json  `(concept)`
**1. Definition:** An artifact is a file produced by training, here `model.joblib`; `metadata.json` records its version, training time, labels, sizes and validation metrics.
**2. In plain words:** Metadata tells you which model you are running and how good it was. Module 10 builds a registry on this idea.
**3. Example to quote:**
> "`metadata.json` says version `v1`, validation accuracy 0.8, trained at a given time, so I can trace which model is deployed."

**Seen in:** `train.py`, `metadata`

## pathlib.Path(__file__)  `(standard library: pathlib)`
**1. Definition:** `Path(__file__).resolve().parent / "artifacts"` builds a path relative to the current script's folder; `/` joins path parts.
**2. In plain words:** Using the script's own location instead of the current working directory means the code works no matter where you run it from. This avoids the most common "file not found" bug.
**3. Example to quote:**
> "I build `ARTIFACT_DIR` from `__file__`, so `python predict.py` works from any folder."

**Seen in:** `train.py`, `predict.py`

## mkdir(exist_ok=True)  `(standard library: pathlib)`
**1. Definition:** `Path.mkdir(exist_ok=True)` creates a directory and does not raise an error if it already exists.
**2. In plain words:** Lets `train.py` be run repeatedly without crashing on the second run.
**3. Example to quote:**
> "`ARTIFACT_DIR.mkdir(exist_ok=True)` makes `artifacts/` if it is missing."

**Seen in:** `train.py`, `main`

## json.dumps / json.loads  `(standard library: json)`
**1. Definition:** `json.dumps(obj, indent=2)` converts a Python dict to a JSON string; `json.loads(text)` parses JSON text back into Python objects.
**2. In plain words:** JSON is the universal, human-readable format for configuration and API data. `write_text` and `read_text` on a `Path` save and load it as text.
**3. Example to quote:**
> "I write the metadata with `json.dumps(metadata, indent=2)` and read it back with `json.loads`."

**Seen in:** `train.py`, `predict.py`

## datetime with timezone  `(standard library: datetime)`
**1. Definition:** `datetime.now(timezone.utc).isoformat(timespec="seconds")` returns the current UTC time as an ISO-8601 string.
**2. In plain words:** Using UTC avoids timezone confusion when models train on one machine and are read on another. ISO format sorts correctly as text.
**3. Example to quote:**
> "`trained_at` is stored as a UTC ISO timestamp so every machine reads it the same way."

**Seen in:** `train.py`, `metadata`

## @lru_cache  `(syntax)`
**1. Definition:** `functools.lru_cache(maxsize=1)` is a decorator that caches a function's return value, so repeated calls with the same arguments reuse it.
**2. In plain words:** The model is loaded from disk once and then reused on every later call. In a web server this is essential: reloading the model for every request would be very slow.
**3. Example to quote:**
> "`@lru_cache(maxsize=1)` on `load_model` means the file is read once, and later calls return the same model object."

**Seen in:** `predict.py`, `load_model`

## Decorator  `(syntax)`
**1. Definition:** A decorator, written `@name` above a function, wraps the function to add behaviour without changing its code.
**2. In plain words:** Here it adds caching. Web frameworks use decorators the same way to attach routes to functions.
**3. Example to quote:**
> "The `@lru_cache` line above `def load_model()` wraps it so it remembers its result."

**Seen in:** `predict.py`, `load_model`

## sys.argv  `(standard library: sys)`
**1. Definition:** `sys.argv` is the list of command-line arguments; `sys.argv[0]` is the script name and `sys.argv[1:]` are the user's arguments.
**2. In plain words:** It lets `python predict.py "I loved it"` classify the given text. `sys.argv[1:] or [defaults]` falls back to example sentences if none are given.
**3. Example to quote:**
> "`python predict.py a b` makes `sys.argv[1:]` equal `['a', 'b']`."

**Seen in:** `predict.py`, `__main__` block

## isinstance() and input validation  `(Python built-in)`
**1. Definition:** `isinstance(value, type)` checks that a value is of a given type; combined with `raise TypeError(...)` it validates inputs.
**2. In plain words:** `predict_texts` checks it received a list of strings and fails fast with a clear message otherwise. Good error messages are part of production quality.
**3. Example to quote:**
> "If someone passes a plain string instead of a list, I raise `TypeError('texts must be a list of strings')` right away."

**Seen in:** `predict.py`, `predict_texts`

## argmax  `(third-party: numpy)`
**1. Definition:** `row.argmax()` returns the index of the largest value in an array.
**2. In plain words:** Picks the most probable class from a row of probabilities. `model.classes_[best]` converts that index back to a label.
**3. Example to quote:**
> "`row.argmax()` is 1 when the second class has the highest probability, which maps to 'positive'."

**Seen in:** `predict.py`, `predict_texts`

## model.classes_  `(third-party: scikit-learn)`
**1. Definition:** After fitting, `classes_` is the array of class labels in the order matching the columns of `predict_proba`.
**2. In plain words:** It maps probability column numbers back to label names, here `['negative', 'positive']`.
**3. Example to quote:**
> "`model.classes_[1]` is 'positive', because column 1 of `predict_proba` belongs to it."

**Seen in:** `train.py`, `predict.py`

## pytest (assert, pytest.raises)  `(third-party: pytest)`
**1. Definition:** pytest discovers functions named `test_*`, runs them, and reports failures; `with pytest.raises(Error):` asserts that the block raises that error.
**2. In plain words:** ML code needs tests too: positive and negative samples classify correctly, confidence stays between 0 and 1, bad input raises `TypeError`. Run with `pytest -q`; the walkthrough shows 3 passed.
**3. Example to quote:**
> "I test that passing a string instead of a list raises `TypeError`, using `pytest.raises`."

**Seen in:** `test_predict.py`

## Confidence (probability of chosen label)  `(concept)`
**1. Definition:** Confidence here is the probability the model assigns to its predicted label, a number between 0 and 1.
**2. In plain words:** It is not a guarantee of correctness. On words the model has never seen, or empty text, confidence hovers near 0.5. Treat it as a hint, not a truth.
**3. Example to quote:**
> "The model says 'bad slow service' is negative with confidence 0.703, but for unseen words it would hover near 0.5."

**Seen in:** `predict.py`, `predict_texts`

## Train/serve consistency (preprocessing must match)  `(concept)`
**1. Definition:** Inference must apply exactly the same preprocessing as training; bundling them in a Pipeline guarantees this.
**2. In plain words:** If the API lowercased text differently or built a different vocabulary, predictions would be silently wrong. Module 10 calls this train/serve skew.
**3. Example to quote:**
> "Because the vectoriser is inside the saved pipeline, the API preprocesses text exactly the way training did."

**Seen in:** `train.py`, `build_pipeline` docstring

## Dataset (hand-written plus template-generated)  `(concept)`
**1. Definition:** `data.py` provides 60 hand-written and 120 template-generated labelled sentences through `load_dataset()`.
**2. In plain words:** A toy dataset so the whole workflow runs in seconds. Real projects load thousands of rows from CSV or a database, and a toy set gives only about 80% accuracy.
**3. Example to quote:**
> "The dataset is only 180 sentences, so the 0.80 accuracy is a demonstration, not a production result."

**Seen in:** `data.py`, `train.py`
