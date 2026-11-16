# Step 02 - Vectors and similarity (what a good vector looks like)

## The story so far
[Step 01](../01_tokenizer/WALKTHROUGH.md) ended with ids like `995` that are arbitrary integers with no geometry. A network needs vectors where *distance means something*. You already built vectors once: in [03_ml_foundations/08_sklearn_text_classifier](../../03_ml_foundations/08_sklearn_text_classifier/) TF-IDF turned text into numbers, and `flaw_demo.py` there showed its flaw with numbers: `king`/`queen` and `car`/`automobile` score cosine 0.0, and `dog bites man` equals `man bites dog`.

## The pain point
TF-IDF has one column per vocabulary word. It is sparse (74 % zeros in this tiny demo, 50,000 columns in real life), and two words count as similar only if they appear in the same documents. Synonyms that never share a sentence look 100 % unrelated.

## The spark
"You shall know a word by the company it keeps." Describe each word by the words around it, then squeeze that description into a few numbers (a **dense vector**). Words that keep the same company end up with the same vector. Compare vectors with cosine similarity. This is the idea behind embeddings; the next step makes the network learn them.

## Goal
Go from sparse TF-IDF vectors to small dense vectors (the idea behind embeddings), compare words with cosine similarity, and see exactly what is still missing: order and context.

## Run it
```bash
.venv/bin/python 04_llm_fundamentals/02_vectors_and_similarity/main.py
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| TF-IDF vector | sparse vector, bag-of-words weights | Yes | One column per vocabulary word, mostly zeros |
| Dense vector | embedding, distributed representation | Roughly | Tens to thousands of non-zero numbers per item, with meaning in the geometry |
| Cosine similarity | cosine, angle similarity | Yes | `a.b / (|a||b|)`; compares direction, ignores length |
| Dot product | inner product, scalar product | Same as cosine only for unit vectors | Cosine is the dot product of length-normalised vectors |
| SVD | singular value decomposition, LSA/LSI (when applied to text) | Related | Factorises a matrix as `U diag(S) Vt`; keeping the top k gives a low-rank approximation |
| Co-occurrence matrix | context-count matrix | Yes | `co[i, j]` = how often words i and j share a sentence |
| Word2Vec / GloVe / LSA | classic word embeddings | Different algorithms | All learn dense word vectors from context; SVD on counts (this file) is closest to LSA/GloVe |
| Distributional hypothesis | "know a word by its company" | Yes | The idea that words in similar contexts have similar meanings |

## Code, step by step

### 1. Module docstring: why vectors  (`main.py` lines 1–20)

```python
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
```

- **In simple words:** Explains that text must become numbers and compares sparse TF-IDF with dense vectors.
- **Exactly:** Docstring only. TF-IDF: one slot per word. Dense: tens to thousands of numbers per word. Neither handles order or context.
- **Why it is here:** Prepares you for embeddings (step 03) and attention (step 04).

### 2. Imports and print formatting  (`main.py` lines 22–25)

```python
import numpy as np                                        # arrays + maths
from sklearn.feature_extraction.text import TfidfVectorizer  # same tool as 03_ml_foundations/08

np.set_printoptions(precision=2, suppress=True)          # print floats compactly, no 1e-17 noise
```

- **In simple words:** Load numpy and the TF-IDF tool, and make arrays print tidily.
- **Exactly:** `np.set_printoptions(precision=2, suppress=True)` shows 2 decimals and avoids scientific notation like `1e-17`. It only changes display, not values.
- **Why it is here:** Without it, the printed vectors are noisy and hard to read.

### 3. Cosine similarity function  (`main.py` lines 27–29)

```python
def cosine(a, b):
    """Cosine similarity: 1 = same direction, 0 = unrelated, -1 = opposite."""
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
```

- **In simple words:** A number between -1 and 1 saying how aligned two vectors are.
- **Exactly:** `a @ b` is the dot product, `np.linalg.norm` is the length; cosine = `a.b / (|a||b|)`. `+ 1e-12` avoids division by zero for zero vectors. Returns a Python float. 1 = same direction, 0 = unrelated, -1 = opposite.
- **Why it is here:** Cosine is the standard similarity for embeddings, search and RAG (module 14).

### 4. A tiny corpus where synonyms never co-occur  (`main.py` lines 31–41)

```python
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
```

- **In simple words:** Ten short sentences. "car" and "automobile" mean the same but never appear in one sentence.
- **Exactly:** `docs` is `list[str]` of 10 sentences in 5 pairs, each pair identical except car/automobile or cat/kitten.
- **Why it is here:** Controlled design: any model that treats synonyms as near must learn it from shared contexts, not from sharing a sentence.

### 5. TF-IDF matrix and its sparsity  (`main.py` lines 43–51)

```python
# ---------------------------------------------------------------------------
# 2. TF-IDF: you already used this.  One column per vocabulary word.
# ---------------------------------------------------------------------------
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(docs).toarray()                  # shape: (n_docs, vocab_size) = (10, 16)
vocab = tfidf.get_feature_names_out()                    # shape: (vocab_size,) = (16,)
print("TF-IDF matrix shape (docs, vocab):", X.shape)
print("first doc vector:", X[0])                           # X[0] shape: (vocab_size,) = (16,)
print("share of zeros  :", round(float((X == 0).mean()), 2), "  <- 'sparse': mostly empty")
```

- **In simple words:** Build the one-column-per-word table and look at how empty it is.
- **Exactly:** `fit_transform(docs).toarray()` gives `X` of shape `(10, 16)`: 10 documents, 16 vocabulary words. `get_feature_names_out()` has 16 names. The first document vector has 4 non-zero entries (0.51, 0.58, 0.58, 0.25 for the, car, needs, fuel in vocabulary order: car, fuel, needs, the). `(X == 0).mean()` is 0.74: 74% zeros.
- **Why it is here:** Sparse vectors grow with vocabulary and carry no meaning geometry.

### 6. Compare words by the documents they appear in  (`main.py` lines 53–57)

```python
# Treat each WORD as the column of this matrix: which documents does it appear in?
# (a word's TF-IDF "profile" across documents)
word_vec = {w: X[:, i] for i, w in enumerate(vocab)}     # each value shape: (n_docs,) = (10,)
print("\ncos(car, automobile) with TF-IDF:", round(cosine(word_vec["car"], word_vec["automobile"]), 2),
      " <- same meaning, but 0.0: they never share a document")
```

- **In simple words:** Treat each column as a word's profile across the documents and compare car with automobile.
- **Exactly:** `X[:, i]` is the column of word i, shape `(10,)`. Real output `cos(car, automobile) = 0.0` because they never appear in the same document (orthogonal).
- **Why it is here:** Shows the first flaw: TF-IDF sees synonyms as unrelated.

### 7. Build a word-by-word co-occurrence table  (`main.py` lines 59–67)

```python
# ---------------------------------------------------------------------------
# 3. Dense vectors from CONTEXT.  "You shall know a word by the company it keeps."
#    Count how often words appear together in a sentence, then squeeze the big
#    count table down to k numbers per word with SVD (a classic compression trick).
# ---------------------------------------------------------------------------
STOP = {"the", "a", "an"}                                # filler words appear everywhere, so they carry no signal
words = sorted({w for d in docs for w in d.split() if w not in STOP})
index = {w: i for i, w in enumerate(words)}
co = np.zeros((len(words), len(words)))                  # shape: (n_words, n_words) = (14, 14); co[i, j] = times word i and j share a sentence
```

- **In simple words:** Count how often two words appear in the same sentence, ignoring filler words.
- **Exactly:** `STOP` removes "the, a, an". `words` is the sorted vocabulary of non-stop words, 14 of them. `co` is a `(14, 14)` zero matrix. The nested loops add 1 for every ordered pair `(a, b)` of different words in a sentence. `co[i, j]` is symmetric.
- **Why it is here:** Words used in the same contexts get similar rows. This is the distributional hypothesis, the core idea behind embeddings.

### 8. Fill the co-occurrence counts  (`main.py` lines 68–73)

```python
for d in docs:
    toks = [t for t in d.split() if t not in STOP]
    for a in toks:
        for b in toks:
            if a != b:
                co[index[a], index[b]] += 1
```

- **In simple words:** Go through each sentence and add up pairs.
- **Exactly:** For each doc, `toks` is the sentence without stop words; the double loop skips `a == b`. For "the car needs fuel" the pairs are car-needs, car-fuel, needs-car and so on.
- **Why it is here:** This table is the raw statistic from which the dense vectors are squeezed.

### 9. Compress to 4 dense numbers per word with SVD  (`main.py` lines 75–80)

```python
U, S, Vt = np.linalg.svd(co)                             # shapes: U (14, 14), S (14,), Vt (14, 14); co = U @ diag(S) @ Vt
k = 4                                                    # keep only the 4 strongest directions
dense = U[:, :k] * S[:k]                                 # shape: (n_words, k) = (14, 4): one DENSE row per word
print("\ndense word vectors shape (words, k):", dense.shape)
print("car       :", dense[index["car"]])
print("automobile:", dense[index["automobile"]])
```

- **In simple words:** Squeeze each row of 14 numbers into 4 numbers keeping the strongest patterns.
- **Exactly:** `np.linalg.svd(co)` returns `U (14,14)`, `S (14,)`, `Vt (14,14)` with `co = U @ diag(S) @ Vt`. With `k = 4`, `U[:, :k] * S[:k]` is shape `(14, 4)`: one dense row per word. Real output: `car [-1.96 -0. -1.47 -0.]` and `automobile [-1.96 0. -1.47 0.]`, which are identical (up to sign noise of zeros).
- **Why it is here:** Words with the same contexts end up with the same vector, even though they never share a sentence. This is what TF-IDF cannot do.

### 10. Compare dense word vectors  (`main.py` lines 82–87)

```python
def sim(w1, w2):
    return round(cosine(dense[index[w1]], dense[index[w2]]), 2)

print("\ncosine on dense vectors (the company a word keeps):")
for a, b in [("car", "automobile"), ("cat", "kitten"), ("car", "kitten"), ("engine", "milk")]:
    print(f"  {a:>10} ~ {b:<10}: {sim(a, b)}")
```

- **In simple words:** Check which words are now neighbours.
- **Exactly:** `sim` rounds the cosine of two rows. Real output: `car ~ automobile 1.0`, `cat ~ kitten 1.0`, `car ~ kitten 0.0`, `engine ~ milk 0.23`. Related words are near, unrelated words are far.
- **Why it is here:** This is the "meaning as geometry" idea that embeddings (step 03) learn directly.

### 11. Average word vectors into sentence vectors  (`main.py` lines 89–98)

```python
# ---------------------------------------------------------------------------
# 4. From word vectors to a sentence vector: just average them.
#    Cheap and useful (search, clustering) - but look what it throws away.
# ---------------------------------------------------------------------------
def sentence_vec(sentence):
    # stack of word vectors (n_words_in_sentence, k) averaged over axis 0 -> shape: (k,) = (4,)
    return np.mean([dense[index[w]] for w in sentence.split() if w in index], axis=0)

a = sentence_vec("driver parks car")                      # shape: (k,) = (4,)
b = sentence_vec("car parks driver")                     # shape: (4,); opposite meaning, same words
```

- **In simple words:** The simplest way to represent a sentence: average its word vectors.
- **Exactly:** `np.mean([...], axis=0)` stacks `(n_words, 4)` and averages down the word axis giving shape `(4,)`. The list comprehension skips words not in `index`. Both sentences use the same three words, so `a == b`.
- **Why it is here:** Averaging is cheap and useful for search, but loses order.

### 12. Show what averaging throws away  (`main.py` lines 99–103)

```python
print("\nsentence vectors identical for 'driver parks car' vs 'car parks driver':", bool(np.allclose(a, b)))
print("-> averaging forgets ORDER and who-does-what.  A word's vector is also fixed:")
print("   'bank' has ONE vector whether it means river bank or money bank.")
print("   Making each word's vector depend on the words AROUND it is what attention does (step 04).")
print("   Next: step 03 gives every token a LEARNED dense vector instead of an SVD one.")
```

- **In simple words:** "driver parks car" and "car parks driver" get the same vector, and "bank" has one vector for all meanings.
- **Exactly:** `np.allclose(a, b)` is `True` (printed as `True`). Fixed word vectors cannot depend on context; attention (step 04) makes each token's vector depend on its neighbours.
- **Why it is here:** This is the motivation for the next two steps: learned embeddings, then attention.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `a @ b` | dot product of two vectors | `np.array([1,2]) @ np.array([3,4]) == 11` |
| `np.linalg.norm(a)` | length of a vector | `norm([3, 4]) == 5.0` |
| `X[:, i]` | the whole i-th column | one word's values across all documents |
| `U[:, :k] * S[:k]` | keep the first k columns, scale each by its strength | gives k numbers per row |
| `{w for d in docs for w in d.split()}` | set comprehension: unique words | `{"a", "b"}` |
| `(X == 0).mean()` | fraction of True values | share of zeros in a matrix |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `TfidfVectorizer` | scikit-learn | text -> TF-IDF vectors (same as in 03/08) | `lowercase`, `ngram_range` | transformer |
| `.fit_transform(docs)` | scikit-learn | learn vocabulary and convert | list of strings | sparse matrix |
| `.get_feature_names_out()` | scikit-learn | the vocabulary, in column order | - | array of words |
| `np.linalg.svd` | numpy | split a matrix into strength-ordered directions | matrix | `U, S, Vt` |
| `np.allclose(a, b)` | numpy | equal up to rounding? | arrays | bool |
| `np.set_printoptions` | numpy | change how arrays print | `precision`, `suppress` | None |

## The three ideas, in order

**1. Sparse vectors (TF-IDF).** One column per vocabulary word; 74 % of this tiny matrix is zeros, and a real vocabulary has 50,000 columns. Great for search and simple classifiers. Weakness: two words are "similar" only if they appear in the same documents, so *car* and *automobile* score 0.0 even though they mean the same thing.

**2. Dense vectors from context.** Count which words share sentences, then compress with SVD to 4 numbers per word. Now *car* and *automobile* get identical vectors because they keep the same company ("needs fuel", "has an engine"), while *car* and *kitten* are unrelated (0.0). Real embeddings (step 03) are learned by gradient descent instead of SVD, are 768+ numbers long, and pick up far richer relations - but the principle is the same: **meaning = the pattern of surrounding words, stored as a short dense vector.**

**3. Cosine similarity.** The angle between two vectors: 1 = same direction, 0 = unrelated. It ignores vector length, so a long and a short document about the same topic still match. The same measure is used later for embedding search/RAG and, with a twist, inside attention (a scaled dot product, step 04).

## What plain vectors still cannot do
- **Order:** averaging word vectors gives `'driver parks car'` and `'car parks driver'` the same vector.
- **Context:** *bank* has one vector for river bank and money bank.
Fixing the first is step 03's job (positions). Fixing the second needs a mechanism that rebuilds each word's vector **using the other words around it**: self-attention, step 04.

## Execution trace (real output)
```
TF-IDF matrix shape (docs, vocab): (10, 16)
first doc vector: [0.   0.   0.51 0.   0.   0.   0.   0.   0.58 0.   0.   0.   0.   0.58
 0.   0.25]
share of zeros  : 0.74   <- 'sparse': mostly empty

cos(car, automobile) with TF-IDF: 0.0  <- same meaning, but 0.0: they never share a document

dense word vectors shape (words, k): (14, 4)
car       : [-1.96 -0.   -1.47 -0.  ]
automobile: [-1.96  0.   -1.47  0.  ]

cosine on dense vectors (the company a word keeps):
         car ~ automobile: 1.0
         cat ~ kitten    : 1.0
         car ~ kitten    : 0.0
      engine ~ milk      : 0.23

sentence vectors identical for 'driver parks car' vs 'car parks driver': True
-> averaging forgets ORDER and who-does-what.  A word's vector is also fixed:
   'bank' has ONE vector whether it means river bank or money bank.
   Making each word's vector depend on the words AROUND it is what attention does (step 04).
   Next: step 03 gives every token a LEARNED dense vector instead of an SVD one.
```
The corpus is deliberately tiny and designed (car/automobile never co-occur), so 1.0 and 0.0 are cleaner than real life. Real similarity values are rarely exactly 0 or 1.

## Experiments (change ONE thing)
- Set `k = 2`, then `k = 8`. Fewer numbers = more squeezing; what happens to car~kitten?
- Remove the `STOP` filter. "the" appears in every sentence; how do the similarities change? (This is why TF-IDF down-weights common words.)
- Add `"the car and the automobile"` to `docs`. Does TF-IDF cos(car, automobile) leave 0.0?
- Replace averaging with a position-weighted sum (`1*v1 + 2*v2 + 3*v3`). Do the two sentences now differ? (Yes - this is a crude hint at why position information is added in step 03.)

## Debugging practice
Call `sentence_vec("zebra")`. No word is in the vocabulary, so `np.mean([])` returns `nan` with a RuntimeWarning. Meaning: an unseen word has no vector - the same *out-of-vocabulary* problem that sub-word tokenisation (step 01) solves.

## Rebuild checklist
- [ ] Write `cosine(a, b)` from memory.
- [ ] Build a co-occurrence matrix for a 5-sentence corpus.
- [ ] Reduce it with SVD and read off a dense vector for one word.
- [ ] Explain, using one example, what averaging loses.

## The new problem (cliffhanger)
Our dense vectors came from an SVD of counts on a 14-word toy corpus. A real model needs a vector for each of its 50,257 token ids, and it should *learn* them for the task, not compute them once from counts. The vectors are also still context-free and order-blind. Next: [Step 03 - Embeddings](../03_embeddings/WALKTHROUGH.md), where the vectors become a learned lookup table, and position is added.

> **Say it aloud (interview box):** TF-IDF gives sparse vectors with one slot per word, so synonyms like car and automobile look unrelated, while short dense vectors built from a word's context put them next to each other, which I measure with cosine similarity. Meaning here is the pattern of surrounding words stored as a few numbers. But each word still gets one fixed vector and averaging loses order, so the next steps make the vectors learned (embeddings) and context-aware (attention).
