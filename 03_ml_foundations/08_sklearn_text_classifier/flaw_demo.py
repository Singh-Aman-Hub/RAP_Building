"""
GOAL: Chapter 6 epilogue. First celebrate that the TF-IDF classifier works, then PROVE its two fatal flaws
      with numbers: (a) words have no meaning-geometry, (b) word order is thrown away.

RUN (from this folder):  python flaw_demo.py
RUN (from repo root):    .venv/bin/python 03_ml_foundations/08_sklearn_text_classifier/flaw_demo.py
EXPECTED OUTPUT: validation accuracy, then cosine 0.0 for king/queen and car/automobile, then a cosine of 1.0 for
                 'dog bites man' vs 'man bites dog'.
NEEDS: scikit-learn, numpy (no network, no files written)
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from data import load_dataset
from train import build_pipeline


def cosine(u, v):
    """cos(angle) between two vectors: 1.0 = same direction, 0.0 = unrelated (perpendicular)."""
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


# --- Part 1: celebrate. The classifier works. -------------------------------------------------------------------
texts, labels = load_dataset()
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.25, random_state=42, stratify=labels)
model = build_pipeline().fit(X_train, y_train)           # same pipeline as train.py, trained in memory only
print(f"PART 1  The baseline works: validation accuracy = {model.score(X_val, y_val):.2f} on {len(X_val)} unseen sentences")
print("        'great food'       ->", model.predict(["great food"])[0])
print("        'bad slow service' ->", model.predict(["bad slow service"])[0])

# --- Part 2: flaw (a). Every word is its own column, so no two words are ever 'close'. ---------------------------
# Parallel sentences: king/queen and car/automobile appear in IDENTICAL contexts, so any real notion of meaning
# would call them near-synonyms or at least neighbours.
corpus = [
    "the king rules the land",
    "the queen rules the land",
    "my car is very fast",
    "my automobile is very fast",
]
vec = TfidfVectorizer()
M = vec.fit_transform(corpus).toarray()                  # shape: (n_docs=4, vocab_size)  one COLUMN per word
vocab = vec.vocabulary_                                  # word -> column index
print("\nPART 2  Flaw (a): words have no semantics. TF-IDF matrix shape (docs, vocab):", M.shape)
for a, b in [("king", "queen"), ("car", "automobile"), ("king", "king")]:
    sim = cosine(M[:, vocab[a]], M[:, vocab[b]])         # compare the two words' COLUMNS
    print(f"        cosine({a!r:>12}, {b!r:<12}) = {sim:.1f}")
print("        A car and an automobile are the same thing, yet the model sees them as 100% unrelated.")

# --- Part 3: flaw (b). A bag of words forgets the order. ---------------------------------------------------------
a_sentence, b_sentence = "dog bites man", "man bites dog"
vec2 = TfidfVectorizer().fit([a_sentence, b_sentence])
va, vb = vec2.transform([a_sentence]).toarray()[0], vec2.transform([b_sentence]).toarray()[0]   # shape: (3,) each
print("\nPART 3  Flaw (b): word order is lost.")
print("        vocabulary:", sorted(vec2.vocabulary_))
print("        'dog bites man' ->", va.round(3))
print("        'man bites dog' ->", vb.round(3))
print("        identical vectors:", np.array_equal(va, vb), "| cosine =", round(cosine(va, vb), 1))
print("        One headline is routine, the other is news, and the classifier cannot tell them apart.")

# Honest footnote: bigrams (ngram_range=(1, 2)) patch this for ADJACENT words only.
vec3 = TfidfVectorizer(ngram_range=(1, 2)).fit([a_sentence, b_sentence])
v3a, v3b = vec3.transform([a_sentence]).toarray()[0], vec3.transform([b_sentence]).toarray()[0]
print(f"        with bigrams cosine = {cosine(v3a, v3b):.2f}: better, but each new n-gram is yet another unrelated column.")

print("\nNEXT: we need a representation where similar words are NEARBY and order matters. First, how do we feed raw text in at all?")
print("      -> ../../04_llm_fundamentals/00_what_is_an_llm/ and ../../04_llm_fundamentals/01_tokenizer/")
# TRY IT: add "the monarch rules the land" to the corpus. 'king' vs 'monarch' is still 0.0, however close the meaning.
# BREAK IT: ask for vocab["automobile"] after fitting on only the first two sentences -> KeyError: that word is not a column.
