# Step 08 — A complete mini ML pipeline: train → save → load → predict

## Goal
Build the first end-to-end ML artifact: text classifier pipeline, saved with joblib, loaded for inference. **This is reused** by [09_api_development](../../09_api_development/) (served over HTTP), [10_ml_system_architecture](../../10_ml_system_architecture/) (versioning, monitoring) and [13_capstone_project](../../13_capstone_project/).

## Run it
```bash
cd 03_ml_foundations/08_sklearn_text_classifier
python train.py                       # creates artifacts/model.joblib + metadata.json
python predict.py "great food" "bad slow service"
pytest -q                             # tests
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

## Say it aloud
"I wrapped TF-IDF and logistic regression in one sklearn Pipeline so preprocessing at inference is identical to training, trained it on a train split, checked precision/recall on validation, and saved it with joblib plus a metadata file recording version and metrics. A separate predict module loads it once and returns label plus confidence. Limitation: the data here is tiny and synthetic, so the metrics mostly show the workflow, not real quality; confidence is not the same as correctness."
