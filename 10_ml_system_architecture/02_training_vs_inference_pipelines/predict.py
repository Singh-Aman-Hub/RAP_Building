"""
predict.py -- the INFERENCE pipeline: ONLINE, runs on every request, light, only LOADS the artifact.

    request -> validate -> (preprocess + features + model, all inside the loaded pipeline) -> postprocess -> response

Note what is NOT here: no fit(), no training data. Inference code must never learn anything.
"""

import json
from pathlib import Path

import joblib

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


def load_artifact(artifact_dir: Path = ARTIFACT_DIR):
    """Load once at startup (slow), reuse for every request (fast)."""
    model = joblib.load(artifact_dir / "model.joblib")
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    return model, metadata


def predict_text(model, text: str, threshold: float = 0.6) -> dict:
    """Validate -> predict -> postprocess. Postprocessing here = turn probabilities into a decision."""
    if not isinstance(text, str) or not text.strip():            # 1. validate input
        raise ValueError("text must be a non-empty string")
    probs = model.predict_proba([text])[0]                       # 2. model (preprocess+features inside)
    best = int(probs.argmax())
    confidence = float(probs[best])
    label = str(model.classes_[best]) if confidence >= threshold else "uncertain"   # 3. postprocess rule
    return {"label": label, "confidence": round(confidence, 3)}


if __name__ == "__main__":
    model, meta = load_artifact()
    print("loaded model trained at", meta["created_at"])
    print(predict_text(model, "What a superb hotel"))
