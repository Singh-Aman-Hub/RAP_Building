"""
model_loader.py -- everything about getting a model and asking it questions.

Both flask_app.py and fastapi_app.py import THIS file, so the two APIs behave identically
and the only difference between them is the web framework (that is the point of this step).

THE MODEL CONTRACT (anything with these works):
    model.predict_proba(list_of_texts) -> list of probability lists   (one row per text)
    model.classes_                     -> list of label names, same order as the probability columns
scikit-learn Pipelines (like the one from module 03) satisfy this. So does KeywordSentimentModel below.

SWAPPING IN THE REAL MODEL:
    MODEL_PATH=/path/to/model.joblib python flask_app.py
"""

import os
import time

from config import get_settings


class KeywordSentimentModel:
    """Dependency-free fallback so the API runs even before you have trained anything."""

    classes_ = ["negative", "positive"]
    POSITIVE = {"good", "great", "love", "excellent", "amazing", "nice", "happy", "best"}
    NEGATIVE = {"bad", "terrible", "hate", "awful", "poor", "worst", "sad", "boring"}

    def predict_proba(self, texts):
        rows = []
        for text in texts:
            words = text.lower().split()
            pos = sum(w.strip(".,!?") in self.POSITIVE for w in words)  # True counts as 1
            neg = sum(w.strip(".,!?") in self.NEGATIVE for w in words)
            p_pos = (pos + 1) / (pos + neg + 2)          # add-one smoothing: never exactly 0 or 1
            rows.append([1 - p_pos, p_pos])
        return rows


def load_model():
    """Return (model, source_description). Real file if MODEL_PATH is set, else the fallback."""
    path = get_settings().model_path
    if path:
        import joblib                                    # lazy import: only needed for real model files
        return joblib.load(path), f"file:{os.path.basename(path)}"   # raises FileNotFoundError if wrong path
    return KeywordSentimentModel(), "fallback-keyword"


def predict_one(model, text: str) -> tuple[str, float]:
    """Run the model on ONE text and return (label, confidence)."""
    probs = list(model.predict_proba([text])[0])         # list() so numpy arrays and lists behave the same
    best = max(range(len(probs)), key=probs.__getitem__) # index of the largest probability
    return str(model.classes_[best]), round(float(probs[best]), 4)


# ------------------------- optional text generation (lazy) -------------------------
_GENERATOR = None                                        # cache: load the big model once, on first use


def generate_text(prompt: str, max_new_tokens: int = 20) -> str:
    """Generate text with distilgpt2 (downloads ~330 MB the first time; see 04_llm_fundamentals step 06 and 08, module 05)."""
    global _GENERATOR
    if _GENERATOR is None:                               # LAZY LOADING: pay the cost only if someone calls /generate
        from transformers import pipeline                # heavy import, so keep it inside the function
        _GENERATOR = pipeline("text-generation", model="distilgpt2")
    out = _GENERATOR(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    return out[0]["generated_text"]


def timed(fn, *args):
    """Call fn(*args) and return (result, elapsed_ms). Handy for latency fields."""
    start = time.perf_counter()
    result = fn(*args)
    return result, round((time.perf_counter() - start) * 1000, 2)
