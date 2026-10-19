"""
GOAL: A tiny REAL ML model (scikit-learn TF-IDF + logistic regression) that the UIs and the mock API share.

RUN: cd 12_streamlit_gradio/common && ../../.venv/bin/python model.py
EXPECTED OUTPUT: three predictions, e.g. I love it -> ('positive', 0.548)
NEEDS: scikit-learn. Trains in well under a second; no download.

Why train at import/first-use? Keeps the folder tiny (no .pkl file) and shows the pattern "load model ONCE,
reuse for every request" - which is what st.cache_resource does for Streamlit later.
"""
from sklearn.feature_extraction.text import TfidfVectorizer   # text -> numeric feature vectors
from sklearn.linear_model import LogisticRegression           # simple classifier with probabilities
from sklearn.pipeline import make_pipeline                    # chains vectorizer + classifier into one object

TRAIN = [
    ("I love this, it is great", "positive"), ("what an excellent product", "positive"),
    ("so happy with the result", "positive"), ("really good quality and fast", "positive"),
    ("absolutely wonderful experience", "positive"), ("I like it a lot", "positive"),
    ("this is terrible", "negative"), ("awful, I hate it", "negative"),
    ("very bad and slow", "negative"), ("so sad and disappointing", "negative"),
    ("worst purchase ever", "negative"), ("I dislike it, poor quality", "negative"),
]

_model = None   # module-level cache: None until first use


def load_model():
    """Train (once) and return the pipeline."""
    global _model
    if _model is None:
        texts, labels = zip(*TRAIN)                       # unzip list of pairs into two tuples
        _model = make_pipeline(TfidfVectorizer(), LogisticRegression())
        _model.fit(texts, labels)                         # learn word weights
    return _model


def predict(text: str) -> tuple[str, float]:
    """Return (label, confidence) for one sentence. Raises ValueError on empty input."""
    if not text or not text.strip():
        raise ValueError("Text must not be empty")
    model = load_model()
    proba = model.predict_proba([text])[0]                # one row of class probabilities
    best = proba.argmax()                                 # index of the largest probability
    return str(model.classes_[best]), round(float(proba[best]), 3)


if __name__ == "__main__":
    for s in ["I love it", "this is awful", "it is a table"]:
        print(s, "->", predict(s))
