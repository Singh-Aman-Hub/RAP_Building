"""
GOAL: Train a text classifier (TF-IDF + logistic regression), evaluate it, and SAVE it as a
      versioned artifact that the API (module 09), architecture notes (module 10) and capstone (13) load.

RUN (from this folder):  python train.py
RUN (from repo root):    .venv/bin/python 03_ml_foundations/08_sklearn_text_classifier/train.py
EXPECTED OUTPUT: validation metrics and "saved model -> artifacts/model.joblib".
NEEDS: scikit-learn, joblib
ARTIFACTS: artifacts/model.joblib (the pipeline) and artifacts/metadata.json (version, metrics, labels)
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib                                              # saves/loads Python objects (sklearn models) to disk
from sklearn.feature_extraction.text import TfidfVectorizer  # text -> numeric feature vectors
from sklearn.linear_model import LogisticRegression         # the classifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split         # random split into train / validation
from sklearn.pipeline import Pipeline                        # chains preprocessing + model into ONE object

from data import load_dataset

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"   # always relative to THIS file, not the cwd
MODEL_VERSION = "v1"


def build_pipeline():
    """Preprocessing and model travel together, so inference does exactly the same transformation as training."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),  # words and word pairs, weighted by rarity
        ("clf", LogisticRegression(max_iter=1000)),                      # outputs a probability per class
    ])


def main():
    texts, labels = load_dataset()
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels)  # stratify keeps class balance equal

    model = build_pipeline()
    model.fit(X_train, y_train)                              # learn vocabulary + weights from TRAIN data only

    predictions = model.predict(X_val)                       # evaluate on unseen data
    report = classification_report(y_val, predictions, output_dict=True, zero_division=0)
    print(classification_report(y_val, predictions, zero_division=0))

    ARTIFACT_DIR.mkdir(exist_ok=True)                        # create artifacts/ if missing
    joblib.dump(model, ARTIFACT_DIR / "model.joblib")        # serialise the whole pipeline
    metadata = {
        "version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "labels": list(model.classes_),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "val_accuracy": report["accuracy"],
        "val_f1_macro": report["macro avg"]["f1-score"],
    }
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print("saved model -> artifacts/model.joblib  | metadata ->", metadata)


if __name__ == "__main__":
    main()
    # TRY IT: change ngram_range to (1, 1) and compare val_accuracy. Then set MODEL_VERSION = "v2" (model versioning in module 10).
    # BREAK IT: delete the line `model.fit(...)` and read the NotFittedError when .predict runs.
