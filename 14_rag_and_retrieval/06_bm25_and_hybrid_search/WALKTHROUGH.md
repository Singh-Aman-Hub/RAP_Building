# Step 06 — BM25 and hybrid search

## The story so far
[Step 05](../05_ann_ivf_and_hnsw/WALKTHROUGH.md) made vector search fast: IVF, graph search and PQ find almost all true neighbours while touching a few percent of the data. But "nearest in embedding space" is only as good as the embedding.

## The pain point
Embeddings capture meaning but blur exact strings. A product code like `CK-TULSI-100`, a rare word like "window" in a section heading, or two-word jargon like "pin codes" gives a 384-dim sentence vector little to hold on to. In step 04 the right chunk for "Do my reward points run out?" came second. Keyword search engines solved exact matching decades ago, and they need no model at all.

## The spark
Write the classic keyword scorer, BM25, by hand: count query words in each chunk, weight rare words more (idf), stop rewarding repeats after a few (k1), and penalise long chunks slightly (b). Then compare it with dense search on 24 labelled questions, by question type, and fuse the two rankings.

## Goal
Tokenise, compute idf and one BM25 score by hand (and match the library class), evaluate BM25, dense, RRF hybrid and weighted-blend hybrid with hit@1, hit@3 and MRR, and see where each one wins.

## The theory (read this before the interview)
**Sparse retrieval and the inverted index.** A text is a *sparse* vector over the vocabulary: almost every word has weight 0. Search engines (Lucene, Elasticsearch, OpenSearch) store an **inverted index**: term → posting list of (doc id, term frequency, sometimes positions). A query only touches the posting lists of its few terms, not every document, which is why keyword search stays fast at billions of documents. Our `BM25` class loops over all chunks for clarity; a real engine would not.

**TF-IDF.** Score = term frequency in the document × inverse document frequency, `idf = log(N / df)`. Words that appear everywhere ("tea" in a tea shop) carry little information; rare words carry a lot.

**BM25 (Okapi BM25)** improves TF-IDF with saturation and length normalisation:
```
score(q, d) = Σ over query terms t:  idf(t) · tf(t,d) · (k1 + 1) / ( tf(t,d) + k1 · (1 − b + b · |d| / avgdl) )
idf(t)      = ln( 1 + (N − df(t) + 0.5) / (df(t) + 0.5) )      # Lucene's variant, always positive
```
- `tf(t,d)`: occurrences of t in d; `|d|`: document length in tokens; `avgdl`: average length; `N`: number of documents; `df(t)`: documents containing t.
- **k1** controls term-frequency saturation: the 1st occurrence adds a lot, the 5th very little; as tf → ∞ the term's contribution approaches `idf · (k1 + 1)`. k1 = 0 means "present or absent" only. Typical 1.2–2.0 (Lucene default 1.2; we use 1.5).
- **b** controls length normalisation: b = 1 fully normalises by length, b = 0 ignores length. Typical 0.75 (used here).
- Real numbers: with N = 36 chunks, idf('ck-tulsi-100') = 3.21 (in 1 chunk), idf('refund') = 2.36 (3 chunks), idf('days') = idf('tea') = 1.47 (8 chunks each).

**Text processing matters as much as the formula.** Tokenisation (here: lower-case, keep hyphenated codes like `ck-masala-250` as one token), stopword removal (what, is, do, i, a ...), and optionally stemming or lemmatisation ("refunds" → "refund"). Without stemming, "refunds" and "refund" do not match, one of the reasons BM25 misses paraphrases.

**Strengths of BM25:** exact terms, product codes, names, IDs, error strings, rare words, new vocabulary the embedding model never saw; no training, no GPU; explainable (you can show which words matched); cheap and mature at scale.
**Weaknesses:** vocabulary mismatch: synonyms and paraphrases share no words ("money back" vs "refund"), no understanding of word order or negation, sensitive to tokenisation choices.

**Dense retrieval** (steps 03–05) has the mirror image: strong on paraphrase, synonyms and cross-lingual queries; weak on exact codes, rare entities and out-of-domain jargon; needs a model and is harder to explain.

**Hybrid search** runs both and fuses the rankings, because their errors are partly independent: one retriever's miss is often the other's hit.

**Reciprocal Rank Fusion (RRF)** (Cormack, Clarke and Büttcher, 2009):
```
RRF(d) = Σ over rankings r:  1 / (k + rank_r(d))        ranks start at 1, k = 60 by convention
```
- Uses only **ranks**, so BM25 scores (0 to about 7 here) and cosines (about 0.2 to 0.6) never need to be put on a common scale.
- **k = 60** damps the gap between rank 1 and rank 2 (1/61 vs 1/62), so a document ranked well by *both* retrievers beats one ranked first by only one. Small k makes the top rank dominate; large k flattens everything. It was found to work well empirically and is rarely tuned.
- No training, robust, the default in many vector databases and search engines.

**Weighted score fusion** (convex combination): normalise each retriever's scores per query (min-max to [0, 1], or z-score), then `alpha · dense + (1 − alpha) · sparse`. It can beat RRF when tuned, but needs labelled data to choose alpha, and min-max is sensitive to outliers and to how scores are distributed for each query.

**Learned sparse retrieval.** SPLADE-style models use a transformer to output a weight for every vocabulary term, including expansion terms not in the text ("money back" can activate "refund"). The result is still a sparse vector, served from an inverted index, combining keyword-style efficiency with some semantic matching.

**Metrics in the table.** **hit@k**: fraction of questions whose gold chunk appears in the top k. **MRR** (mean reciprocal rank): average of 1/rank of the gold chunk (1, 1/2, 1/3 ...), 0 if it is not in the returned list (top 20 here). hit@1 = "is the very first result right"; MRR rewards near misses partially.

**The honest result here.** On this tiny, clean knowledge base (36 chunks, 24 questions) dense already beats BM25 (hit@1 0.83 vs 0.71, MRR 0.90 vs 0.79). RRF only **ties** dense on hit@1 (0.83) and is slightly lower on hit@3 (0.92 vs 0.96) and MRR (0.88 vs 0.90). The weighted blend wins every keyword question (1.00) but loses paraphrases (0.50). With only 3 exact-code questions, all four methods score 1.00 there, so this eval cannot show BM25's best case. Hybrid still matters in production because real corpora are large and messy: thousands of SKUs, people's names, ticket IDs, error messages, acronyms and domain jargon the embedding model never trained on. There dense alone misses exact matches, BM25 alone misses paraphrases, and fusion is a cheap, robust hedge. Always measure on your own labelled questions.

## Run it
```bash
cd 14_rag_and_retrieval/06_bm25_and_hybrid_search
python main.py       # about 6 s in this run (cached MiniLM); timings vary
```

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", s)` | words, keeping hyphenated codes whole | `"ck-masala-250"` stays one token |
| `set("""a an and ...""".split())` | build a stopword set from a string | `"the" in STOPWORDS` |
| `Counter(tokenize(t))` | term -> count for one chunk | `Counter(["a", "a"])` -> `{"a": 2}` |
| `Counter(term for d in docs for term in d)` | nested generator: document frequency | each term counted once per chunk |
| `next(j for j, c in enumerate(chunks) if ...)` | first index that matches | find the refund-window chunk |
| `f"{q!r}"` | insert `repr(q)`, with quotes | `'How many days ...'` |
| `{"name": lambda qq: ...}` | dict of functions: one loop evaluates every method | `methods.items()` |
| `order.index(g) + 1 if g in order else None` | 1-based rank, or None if missing | |
| `enumerate(ranking, start=1)` | ranks start at 1 (RRF needs this) | |
| `fused.most_common(top_n)` | the n highest-scoring keys of a Counter | |
| `np.mean([r == 1 for r in ranks])` | fraction of True = hit@1 | |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `tokenize(text)` | local `common.py` | lower-case tokens, codes kept, stopwords dropped | str | list of str |
| `BM25(texts, k1, b)` | local `common.py` | Okapi BM25; `.scores(q)`, `.search(q, k)` | `k1=1.5`, `b=0.75` | array / list of (idx, score) |
| `rrf_fuse(rankings, k, top_n)` | local `common.py` | Reciprocal Rank Fusion | `k=60` | list of (idx, score) |
| `Embedder().encode` | local `common.py` | MiniLM unit vectors | list of str | `(n, 384)` |
| `load_eval_set()` | local `common.py` | 29 questions; 24 have a gold chunk | – | list of dicts |
| `math.log` | standard library | natural log for idf | – | float |
| `collections.Counter` | standard library | counting terms and fused scores | – | dict subclass |
| `np.argsort(-s)[:k]` | numpy | top-k ids | – | ints |

## Execution trace (real run)
- `tokens: ['ck-masala-250', 'many', 'days', 'refund']` from "What is in CK-MASALA-250? How many days do I have for a refund?": the code survives as one token, stopwords are gone.
- `idf: {'refund': 2.36, 'days': 1.47, 'tea': 1.47, 'ck-tulsi-100': 3.21}`: the code appears in 1 chunk, 'refund' in 3, 'days' and 'tea' in 8 each (from the idf formula with N = 36).
- BM25 by hand for "How many days do I have for a refund?" vs the refund-window chunk: **3.735**, library class: **3.735**.
- The comparison on 24 labelled questions (11 keyword, 10 paraphrase, 3 exact_code):
  ```
  method                hit@1  hit@3    MRR        keyword  paraphrase  exact_code   (hit@1 by type)
  BM25 (sparse)          0.71   0.88   0.79           0.82        0.50        1.00
  dense (MiniLM)         0.83   0.96   0.90           0.91        0.70        1.00
  hybrid RRF k=60        0.83   0.92   0.88           0.91        0.70        1.00
  hybrid blend a=0.5     0.79   0.96   0.88           1.00        0.50        1.00
  ```
  Dense is best overall; RRF ties it on hit@1 and is slightly behind on hit@3 and MRR; the blend is perfect on keyword questions but weak on paraphrases.
- Where BM25 wins:
  ```
  Q: What is the money back window after the tea arrives?
     BM25 top: refunds_and_returns#Refund window
     dense top: refunds_and_returns#How to request a refund
  Q: pin codes
     BM25 top: shipping_and_delivery#Delivery times
     dense top: loyalty_program#Redeeming points
  ```
  Q1: the heading word "window" matches exactly (dense ranked that chunk 2nd). Q2: "pin codes" appears verbatim in the delivery-times text; the two-word query gives the embedding too little meaning.
- And where BM25 loses (checked separately): "How many days do I have to get a refund?" puts `loyalty_program#Earning points` first for BM25, while dense gets the refund window right.

## Experiments (change ONE thing)
1. `b=0` (no length normalisation) and `b=1` in `BM25(...)`. Which questions change rank?
2. `alpha` in `blend_rank` to 0.2 and 0.8, then RRF `k` to 1 and 1000. Which fusion is more robust to its knob?
3. Remove the stopword filter in `common.tokenize` (only in a copy): 'the', 'is', 'do' now add noise to BM25.
4. Add a naive stemmer (strip a trailing "s") in `tokenize`: does BM25's paraphrase score move?
5. Fuse with RRF using top 5 instead of top 20 from each retriever: does hit@3 change?

## Debugging practice
BREAK IT from `main.py`: make `rrf_fuse` count ranks from 0 (`enumerate(ranking, start=0)`). Confirmed in a temporary copy with a local override of `rrf_fuse`:
- With the default `k=60`: **nothing visible changes**, the table is identical (hit@1 0.83, hit@3 0.92, MRR 0.88). Using 1/(60+0) instead of 1/(60+1) shifts every score by almost the same amount, so the ranking survives. Off-by-one bugs like this are silent.
- With `k=0` as well:
  ```
      fused[idx] += 1.0 / (k + rank)
                    ~~~~^~~~~~~~~~~~
  ZeroDivisionError: float division by zero
  ```
  The top-ranked item has rank 0 and k + rank = 0. Lesson: ranks in RRF start at 1, and k > 0 also guards against division by zero.

## Rebuild checklist
- [ ] `tokenize`: lower-case regex, keep hyphenated codes, drop stopwords
- [ ] per chunk `Counter` of terms, lengths, `avg_len`, document frequencies
- [ ] `idf = ln(1 + (N − df + 0.5)/(df + 0.5))`
- [ ] BM25 sum with k1 saturation and b length normalisation; check hand score = class score
- [ ] eval loop: rank of the gold chunk -> hit@1, hit@3, MRR, broken down by question type
- [ ] RRF: `1/(k + rank)` with ranks from 1, summed over retrievers
- [ ] blend: min-max normalise both, `alpha · dense + (1 − alpha) · bm25`

## The new problem (cliffhanger)
Look at the gap between hit@3 (0.96 for dense) and hit@1 (0.83): the right chunk is usually *in* the top few, just not first. Both retrievers score the question and the chunk *separately* (a vector each, or word counts), so neither ever reads them side by side. Next: [step 07, re-ranking with a cross-encoder](../07_reranking_cross_encoder/WALKTHROUGH.md).

> **Say it aloud (interview box):** BM25 is a sparse keyword scorer served from an inverted index: it sums idf-weighted term frequencies with k1 controlling saturation and b controlling length normalisation, so it nails exact codes, names and rare words but misses synonyms and paraphrases. Dense retrieval is the mirror image, so I combine them with Reciprocal Rank Fusion, which adds one over sixty plus the rank from each list and needs no score calibration, or with a tuned weighted blend of normalised scores. On our small clean knowledge base dense alone already scored hit@1 0.83 and hybrid only tied it, but on large corpora full of product codes and jargon hybrid is the robust default, and I always decide with a labelled eval set.
