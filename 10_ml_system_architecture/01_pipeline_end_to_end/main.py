"""
GOAL:   Walk the ENTIRE ML pipeline once, stage by stage, with real code:
        collect -> preprocess -> features -> train -> evaluate -> serialize -> load -> predict.

RUN:    cd 10_ml_system_architecture/01_pipeline_end_to_end && python main.py
        (from repo root: python 10_ml_system_architecture/01_pipeline_end_to_end/main.py)

EXPECTED OUTPUT (shape):
        [1 collect]   200 train texts, 8 test texts ...
        [2 preprocess] 'The movie was GREAT!!  ' -> 'the movie was great'
        [3 features]  matrix shape (200, 32) ...
        [4 train]     ...
        [5 evaluate]  accuracy = 1.00 ...
        [6 serialize] saved ... bytes
        [7 predict]   ...

NEEDS:  scikit-learn, joblib (installed via requirements.txt)
"""

import re
import tempfile
from pathlib import Path

import joblib                                           # saves/loads Python objects (good with numpy-heavy ones)
from sklearn.feature_extraction.text import TfidfVectorizer   # text -> numeric feature matrix
from sklearn.linear_model import LogisticRegression           # a simple, fast classifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline                         # chains steps so they travel together

from data import get_test_data, get_training_data


def preprocess(text: str) -> str:
    """Stage 2: clean raw text. lowercase, drop punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)             # regex: delete anything that is not a letter/digit/space
    return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    # ---- 1. COLLECT: where data comes from (files, DB, API, user feedback). Here: generated. ----
    X_train_raw, y_train = get_training_data(n=200)
    X_test_raw, y_test = get_test_data()
    print(f"[1 collect]    {len(X_train_raw)} train texts, {len(X_test_raw)} test texts; first = {X_train_raw[0]!r}")

    # ---- 2. PREPROCESS: make inputs consistent. ----
    print("[2 preprocess]", repr("The movie was GREAT!!  "), "->", repr(preprocess("The movie was GREAT!!  ")))
    X_train = [preprocess(t) for t in X_train_raw]      # list comprehension: apply preprocess to every text
    X_test = [preprocess(t) for t in X_test_raw]

    # ---- 3. FEATURE ENGINEERING: turn text into numbers a model can use. ----
    vectorizer = TfidfVectorizer()                      # TF-IDF: words frequent in a doc but rare overall score high
    X_train_vec = vectorizer.fit_transform(X_train)     # fit = learn the vocabulary FROM TRAIN ONLY; transform = apply
    X_test_vec = vectorizer.transform(X_test)           # only transform on test: never fit on data you evaluate on!
    print(f"[3 features]   matrix shape {X_train_vec.shape} = (n_texts, vocabulary_size); sample words: {list(vectorizer.vocabulary_)[:5]}")

    # ---- 4. TRAIN ----
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)                       # learns one weight per word per class
    print(f"[4 train]      classes learned: {[str(c) for c in clf.classes_]}")

    # ---- 5. EVALUATE on UNSEEN data ----
    pred = clf.predict(X_test_vec)
    print(f"[5 evaluate]   accuracy = {accuracy_score(y_test, pred):.2f}")
    print(classification_report(y_test, pred, zero_division=0))

    # ---- 6. SERIALIZE: bundle preprocessing-features-model so serving can reuse EXACTLY the same steps ----
    # TfidfVectorizer already lowercases and strips punctuation while tokenizing, so the cleaning lives INSIDE the
    # pipeline using only built-in sklearn behaviour. (A custom function like preprocess() would be pickled BY
    # NAME and break when loaded from another script -- see step 03.)
    pipeline = Pipeline([("tfidf", TfidfVectorizer()),
                         ("clf", LogisticRegression(max_iter=1000))])
    pipeline.fit(X_train_raw, y_train)                  # raw text in; the pipeline cleans + vectorizes + trains
    out = Path(tempfile.mkdtemp()) / "model.joblib"
    joblib.dump(pipeline, out)
    print(f"[6 serialize]  saved {out.stat().st_size} bytes to {out.name}")

    # ---- 7. LOAD + PREDICT (this is what the API does at startup and per request) ----
    loaded = joblib.load(out)
    sample = "Honestly the delivery was TERRIBLE!!"
    probs = loaded.predict_proba([sample])[0]
    print(f"[7 predict]    {sample!r} -> {loaded.predict([sample])[0]}  probs={ {str(c): round(float(p), 3) for c, p in zip(loaded.classes_, probs)} }")
    # TRY IT: change n=200 to n=10 and watch accuracy / confidence change.
    # TRY IT: predict on 'GREAT!!!' vs 'great' -- same answer? Why? (hint: TfidfVectorizer lowercases and tokenizes)
    # BREAK IT: call vectorizer.transform(...) BEFORE fit_transform and read the NotFittedError.
