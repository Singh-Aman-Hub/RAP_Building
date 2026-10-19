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
