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
