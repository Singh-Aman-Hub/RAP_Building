"""
train.py -- the TRAINING pipeline: OFFLINE, runs occasionally, heavy, produces an artifact.

    data -> split -> fit pipeline -> evaluate -> save model.joblib + metadata.json

Run alone:  python train.py            (writes ./artifacts/model.joblib)
"""

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

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


def train(n_samples: int = 200, C: float = 1.0) -> dict:
    """Train, evaluate, save. Returns the metadata dict. C = inverse regularization strength."""
    X_train, y_train = get_training_data(n=n_samples)
    X_test, y_test = get_test_data()

    pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(C=C, max_iter=1000))])
    pipeline.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, pipeline.predict(X_test))

    ARTIFACT_DIR.mkdir(exist_ok=True)                       # create folder if missing (no error if present)
    joblib.dump(pipeline, ARTIFACT_DIR / "model.joblib")
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


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
