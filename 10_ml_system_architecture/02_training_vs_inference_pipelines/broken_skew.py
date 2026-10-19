"""
broken_skew.py -- EXPECTED TO CRASH at the end. Demonstrates TRAINING/SERVING SKEW, the #1 silent ML bug.

RUN: python broken_skew.py

Part A (silent!): training used lowercase text, serving forgot to lowercase -> the model quietly gets
                  worse. No error, no warning. You only notice through monitoring (step 07).
Part B (loud):    refitting the vectorizer at inference time builds a DIFFERENT vocabulary, so the
                  model's weights no longer line up with the columns -> a crash.
"""

import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_pipeline_end_to_end"))
from data import get_test_data, get_training_data  # noqa: E402

X_train, y_train = get_training_data()
X_test, y_test = get_test_data()

# ---- Part A: preprocessing mismatch ----
train_texts = [t.lower() for t in X_train]                  # TRAINING lowercases ...
vec = TfidfVectorizer(lowercase=False)                      # ... and the vectorizer does NOT (we did it by hand)
clf = LogisticRegression(max_iter=1000).fit(vec.fit_transform(train_texts), y_train)

live = [t.upper() for t in X_test]                          # real users sometimes SHOUT
right = clf.predict(vec.transform([t.lower() for t in live]))     # serving applies the same cleaning as training
wrong = clf.predict(vec.transform(live))                           # serving FORGETS to lowercase -> every word is unseen
acc = lambda pred: sum(p == y for p, y in zip(pred, y_test)) / len(y_test)   # lambda = tiny anonymous function
print(f"A) accuracy with consistent preprocessing: {acc(right):.2f}   with skew: {acc(wrong):.2f}  (no error raised!)")

# ---- Part B: refit at inference ----
new_vec = TfidfVectorizer().fit(X_test)                     # BUG: fit on the test data -> a different vocabulary
clf.predict(new_vec.transform(X_test))                      # <- ValueError: X has N features, but LogisticRegression is expecting M
