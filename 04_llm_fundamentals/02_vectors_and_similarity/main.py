"""
GOAL:   Understand "text as vectors" BEFORE attention: start from the TF-IDF vectors you
        met in 03_ml_foundations/08, see where they fall short, then build small DENSE
        vectors (the idea behind embeddings) and compare words with cosine similarity.

RUN:    from this folder:  python main.py
        from repo root:    .venv/bin/python 04_llm_fundamentals/02_vectors_and_similarity/main.py
EXPECTED OUTPUT (short): a sparse TF-IDF matrix, similarity 0.0 for car~automobile,
        then ~1.0 once words get dense vectors, and a final line showing that
        "driver parks car" and "car parks driver" get the SAME vector (the problem attention, step 04, solves).
NEEDS:  numpy + scikit-learn (both already in the venv). No internet.

Big idea
--------
A model can only do maths on numbers, so every piece of text must become a vector.
  TF-IDF  : one slot per vocabulary word, mostly zeros ("sparse"), no notion of meaning.
  Dense   : ~tens to thousands of numbers per word, learned so related words are close.
  Embedding (step 03) = a dense vector per token, learned by the network itself.
Neither handles word ORDER or CONTEXT.  Embeddings (step 03) then make the vectors learned, and attention (step 04) fills the context gap.
"""

import numpy as np                                        # arrays + maths
from sklearn.feature_extraction.text import TfidfVectorizer  # same tool as 03_ml_foundations/08

np.set_printoptions(precision=2, suppress=True)          # print floats compactly, no 1e-17 noise

def cosine(a, b):
    """Cosine similarity: 1 = same direction, 0 = unrelated, -1 = opposite."""
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

# ---------------------------------------------------------------------------
# 1. A tiny corpus.  NOTE: "car" and "automobile" mean the same thing but
#    never appear in the same sentence.  Same for "cat" and "kitten".
# ---------------------------------------------------------------------------
docs = [
    "the car needs fuel",       "the automobile needs fuel",
    "the car has an engine",    "the automobile has an engine",
    "the driver parks the car", "the driver parks the automobile",
    "the cat drinks milk",      "the kitten drinks milk",
    "the cat chases a mouse",   "the kitten chases a mouse",
]

# ---------------------------------------------------------------------------
# 2. TF-IDF: you already used this.  One column per vocabulary word.
# ---------------------------------------------------------------------------
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(docs).toarray()                  # shape: (n_docs, vocab_size) = (10, 16)
vocab = tfidf.get_feature_names_out()                    # shape: (vocab_size,) = (16,)
print("TF-IDF matrix shape (docs, vocab):", X.shape)
print("first doc vector:", X[0])                           # X[0] shape: (vocab_size,) = (16,)
print("share of zeros  :", round(float((X == 0).mean()), 2), "  <- 'sparse': mostly empty")

# Treat each WORD as the column of this matrix: which documents does it appear in?
# (a word's TF-IDF "profile" across documents)
word_vec = {w: X[:, i] for i, w in enumerate(vocab)}     # each value shape: (n_docs,) = (10,)
print("\ncos(car, automobile) with TF-IDF:", round(cosine(word_vec["car"], word_vec["automobile"]), 2),
      " <- same meaning, but 0.0: they never share a document")

# ---------------------------------------------------------------------------
# 3. Dense vectors from CONTEXT.  "You shall know a word by the company it keeps."
#    Count how often words appear together in a sentence, then squeeze the big
#    count table down to k numbers per word with SVD (a classic compression trick).
# ---------------------------------------------------------------------------
STOP = {"the", "a", "an"}                                # filler words appear everywhere, so they carry no signal
words = sorted({w for d in docs for w in d.split() if w not in STOP})
index = {w: i for i, w in enumerate(words)}
co = np.zeros((len(words), len(words)))                  # shape: (n_words, n_words) = (14, 14); co[i, j] = times word i and j share a sentence
for d in docs:
    toks = [t for t in d.split() if t not in STOP]
    for a in toks:
        for b in toks:
            if a != b:
                co[index[a], index[b]] += 1

U, S, Vt = np.linalg.svd(co)                             # shapes: U (14, 14), S (14,), Vt (14, 14); co = U @ diag(S) @ Vt
k = 4                                                    # keep only the 4 strongest directions
dense = U[:, :k] * S[:k]                                 # shape: (n_words, k) = (14, 4): one DENSE row per word
print("\ndense word vectors shape (words, k):", dense.shape)
print("car       :", dense[index["car"]])
print("automobile:", dense[index["automobile"]])

def sim(w1, w2):
    return round(cosine(dense[index[w1]], dense[index[w2]]), 2)

print("\ncosine on dense vectors (the company a word keeps):")
for a, b in [("car", "automobile"), ("cat", "kitten"), ("car", "kitten"), ("engine", "milk")]:
    print(f"  {a:>10} ~ {b:<10}: {sim(a, b)}")

# ---------------------------------------------------------------------------
# 4. From word vectors to a sentence vector: just average them.
#    Cheap and useful (search, clustering) - but look what it throws away.
# ---------------------------------------------------------------------------
def sentence_vec(sentence):
    # stack of word vectors (n_words_in_sentence, k) averaged over axis 0 -> shape: (k,) = (4,)
    return np.mean([dense[index[w]] for w in sentence.split() if w in index], axis=0)

a = sentence_vec("driver parks car")                      # shape: (k,) = (4,)
b = sentence_vec("car parks driver")                     # shape: (4,); opposite meaning, same words
print("\nsentence vectors identical for 'driver parks car' vs 'car parks driver':", bool(np.allclose(a, b)))
print("-> averaging forgets ORDER and who-does-what.  A word's vector is also fixed:")
print("   'bank' has ONE vector whether it means river bank or money bank.")
print("   Making each word's vector depend on the words AROUND it is what attention does (step 04).")
print("   Next: step 03 gives every token a LEARNED dense vector instead of an SVD one.")
