"""
GOAL: Load the saved model and predict.  Exposes load_model() and predict_texts() for reuse by the API in module 09.

RUN (from this folder):  python predict.py "I really loved it"      (run train.py first)
RUN (from repo root):    .venv/bin/python 03_ml_foundations/08_sklearn_text_classifier/predict.py "I really loved it"
EXPECTED OUTPUT: [{'text': 'I really loved it', 'label': 'positive', 'confidence': 0.6...}]
NEEDS: scikit-learn, joblib; artifacts/ created by train.py
"""

import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


@lru_cache(maxsize=1)                                       # decorator: load from disk ONCE, then reuse (important in servers!)
def load_model():
    path = ARTIFACT_DIR / "model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run train.py first.")
    return joblib.load(path)


def load_metadata():
    return json.loads((ARTIFACT_DIR / "metadata.json").read_text())


def predict_texts(texts):
    """Input: list[str].  Output: list of dicts with label and confidence (probability of the chosen label)."""
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise TypeError("texts must be a list of strings")
    model = load_model()
    probabilities = model.predict_proba(texts)              # shape (n_texts, n_classes)
    results = []
    for text, row in zip(texts, probabilities):
        best = int(row.argmax())                            # index of the highest probability
        results.append({"text": text, "label": str(model.classes_[best]), "confidence": round(float(row[best]), 3)})
    return results


if __name__ == "__main__":
    inputs = sys.argv[1:] or ["I really loved it", "This was a terrible waste of money"]   # sys.argv = command-line words
    for item in predict_texts(inputs):
        print(item)
    # TRY IT: predict an empty string "" -- what does the model say, and is its confidence trustworthy?
    # BREAK IT: predict_texts("a string, not a list") -> TypeError raised by our own validation (good errors are a feature).
