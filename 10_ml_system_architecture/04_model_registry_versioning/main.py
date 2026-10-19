"""
GOAL:   Practise MODEL VERSIONING: save every trained model as an immutable numbered version with metadata,
        promote one to "production" with an alias, load by latest / alias / pinned version, then ROLL BACK.

RUN:    cd 10_ml_system_architecture/04_model_registry_versioning && python main.py
        (recreates ./registry_demo each run)

EXPECTED OUTPUT (shape):
        saved v0001 accuracy=...   saved v0002 accuracy=...
        production -> v0001 ... after promote -> v0002 ... after rollback -> v0001

NEEDS:  scikit-learn, joblib
"""

import shutil
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss   # log_loss: lower = more confident AND correct
from sklearn.pipeline import Pipeline

from registry import ModelRegistry

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_pipeline_end_to_end"))
from data import get_test_data, get_training_data  # noqa: E402

ROOT = Path(__file__).resolve().parent / "registry_demo"


def train_and_register(registry: ModelRegistry, n: int, C: float) -> str:
    X, y = get_training_data(n=n)
    X_test, y_test = get_test_data()
    model = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(C=C, max_iter=1000))]).fit(X, y)
    acc = accuracy_score(y_test, model.predict(X_test))
    ll = log_loss(y_test, model.predict_proba(X_test), labels=list(model.classes_))   # accuracy ties here; log loss does not
    version = registry.save(model, metrics={"accuracy": acc, "log_loss": round(ll, 3)}, params={"n": n, "C": C}, training_texts=X)
    print(f"saved {version}  n={n:<4} C={C:<5} accuracy={acc:.2f} log_loss={ll:.3f}")
    return version


if __name__ == "__main__":
    shutil.rmtree(ROOT, ignore_errors=True)               # start clean so the demo is repeatable
    reg = ModelRegistry(ROOT)

    v1 = train_and_register(reg, n=20, C=0.1)             # a weak first model
    v2 = train_and_register(reg, n=300, C=10.0)           # a stronger second model

    reg.promote(v1, "production")                          # v1 is live
    print("\nproduction ->", reg.resolve("production"), "| latest ->", reg.resolve("latest"))

    model, meta = reg.load("production")
    print("loaded", meta["version"], "metrics", meta["metrics"], "data fingerprint", meta["data_fingerprint"])

    reg.promote(v2, "production")                          # roll FORWARD: one JSON edit, no code change
    print("after promote  ->", reg.resolve("production"))
    reg.promote(v1, "production")                          # ROLL BACK when v2 misbehaves in production
    print("after rollback ->", reg.resolve("production"))
    print("pinned load of v0002 still works:", reg.load("v0002")[1]["params"])
    try:
        reg.load("v0099")                                  # BREAK IT: read the KeyError message
    except KeyError as err:
        print("unknown version ->", err)
    print("\nFiles:", sorted(str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file()))
    # TRY IT: add a promote(..., "staging") alias and load("staging").
    # TRY IT: make train_and_register refuse to promote if accuracy < the current production accuracy.
