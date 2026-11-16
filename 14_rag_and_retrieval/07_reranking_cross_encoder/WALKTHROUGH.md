# Step 07 — Re-ranking with a cross-encoder

## The story so far
[Step 06](../06_bm25_and_hybrid_search/WALKTHROUGH.md) fused BM25 and dense search with Reciprocal Rank Fusion. Hybrid search finds the right chunk for almost every question somewhere in its top 20, but not always at position 1.

## The pain point
The LLM only reads the top 3 or so chunks. Hybrid RRF puts the gold chunk first for 20 of 24 questions (hit@1 0.83). For "Is anyone there to help me on Sunday?" the right section sits at rank 12, so it never reaches the prompt. The bi-encoder squeezed every chunk into one 384-number vector before it ever saw the question, so it cannot weigh "Sunday" against the exact wording of the opening-hours section.

## The spark
Split the job in two. Stage 1 (cheap, aims for **recall**) pulls 20 candidates from 36 chunks. Stage 2 (expensive, aims for **precision**) lets a cross-encoder read the question and each candidate *together*, word against word, and re-order only those 20. Twenty forward passes are affordable; one pass per chunk in the whole corpus is not.

## Goal
Build retrieve-then-rerank, measure what it gains (hit@1, MRR), what it costs (milliseconds per query), and see whether the re-rank score can tell us "the answer is not here".

## The theory (read this before the interview)
**Bi-encoder vs cross-encoder.**
- A *bi-encoder* (step 03) encodes the query and the document **separately**: `score = embed(q) · embed(d)`. Document vectors are computed once, offline, and indexed (steps 04 and 05), so a query costs one encoder pass plus a vector search. The price: the document vector was frozen without knowing the question, so every nuance must fit in one fixed vector.
- A *cross-encoder* feeds the **concatenated pair** `[CLS] question [SEP] passage [SEP]` through one transformer. Self-attention runs across both texts in every layer, so each question token can attend to each passage token ("Sunday" meets "closed on Sundays"). A classification head on top outputs one relevance logit.
- Why more accurate: joint attention models term interactions, negation and word order that a single dot product between two pooled vectors cannot express.
- Why it cannot be precomputed: the passage's representation depends on the question. There is nothing to store ahead of time, so every (query, passage) pair costs a full forward pass at query time.

**Two-stage retrieve-then-rerank.**
- Stage 1 = recall stage: "is the answer anywhere in the candidates?" Here recall@20 is 1.00, so the re-ranker always has the gold chunk to work with. A re-ranker can never recover a chunk stage 1 missed.
- Stage 2 = precision stage: "is the best chunk first?" Measured with hit@1 and MRR.
- The candidate count N is the dial. Larger N raises recall but cost grows linearly with N. Smaller N is faster but caps quality at stage 1's recall@N. Typical values are tens to a few hundred.

**Cost arithmetic (from this run, CPU).** Query embedding ~6 ms. Cross-encoder over 20 candidates ~92 ms, so ~4.6 ms per pair. Over 1 million chunks that is ~4,600 s, about 1.3 hours per query. Bi-encoder document vectors are computed once; cross-encoder scores are computed every query. That asymmetry is the whole reason for two stages.

**Late interaction (ColBERT), the middle ground.** ColBERT stores one vector per *token* of each document (precomputed). At query time each query token finds its most similar document token (MaxSim) and those maxima are summed. You get token-level matching like a cross-encoder with precomputable document side like a bi-encoder. Cost: much more storage (one vector per token instead of per chunk).

**LLM-based rerankers.** A general LLM can rerank: *pointwise* (ask "is this passage relevant? yes/no" per passage and use the probability of "yes"), *pairwise* (which of A or B is better), or *listwise* (show the list and ask for a permutation, as in RankGPT-style methods). Stronger reasoning, much higher cost and latency, and listwise prompts suffer position bias. Hosted rerank APIs and open rerankers (for example the BGE reranker family) are the usual production choice.

**Rerank scores as an abstention signal, and calibration.** The logit is a relevance score, not a probability. Applying a sigmoid gives a number between 0 and 1 but it is not calibrated for your data. Use the best score as an "is the answer here?" signal only after choosing the threshold on labelled answerable and unanswerable questions (step 09 does this). Here: "Do you sell gift cards?" scores -7.06, but the answerable London question scores -10.94, *lower*. A single threshold cannot separate them perfectly.

**Ordering the context: "lost in the middle".** Research on long contexts (Liu et al., 2023) found models use information at the start and end of the prompt better than in the middle. Practical rules: send few, well-ranked chunks; put the best one first; do not pad the prompt with weak candidates.

**Be honest about this run.** Re-ranking raised hit@1 from 0.83 to 0.88 and MRR from 0.88 to 0.92 (a net gain of one question out of 24). Underneath, 3 questions moved up (Sunday 12 -> 1, phone 2 -> 1, personal data 2 -> 1) and 3 moved **down**: the refund-days question 1 -> 2, the caffeine question 1 -> 2, and London fell from rank 8 to **20**, last of the candidates. The MS MARCO-trained model does not connect "cousin in London" with "international shipping". Re-rankers are better on average, not on every question.

## Run it
```bash
# from the repo root
.venv/bin/python 14_rag_and_retrieval/07_reranking_cross_encoder/main.py
# or from this folder
python main.py          # ~10-20 s on CPU, offline (models from the local Hugging Face cache)
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Reranker vs cross-encoder vs second-stage ranker | re-ranker, "rerank" | Mostly | Cross-encoder is the most common kind of reranker; others include late-interaction models (ColBERT) and LLM rerankers |
| Bi-encoder vs cross-encoder | retriever vs reranker | No | Bi-encoder embeds texts separately (precompute, fast); cross-encoder reads the pair together (accurate, one pass per pair) |
| Stage 1 vs stage 2 | retrieve then rerank, funnel | Same pattern | Stage 1 optimises recall over everything; stage 2 optimises precision over a few candidates |
| Logit vs probability vs score | raw model output | No | A logit is the raw number before sigmoid; higher = more relevant; not a calibrated probability |
| recall@20 vs hit@1 | candidates contain the answer vs answer is first | No | recall@20 limits what stage 2 can ever do; hit@1 is the final quality |
| top-k vs top-n | candidate count | Same | Naming differs: k candidates from stage 1, n kept after reranking |

## Code, step by step

### 1. Docstring  (`main.py` lines 1–14)
```python
"""
GOAL:  Two-stage retrieval. Stage 1 (fast, recall): hybrid BM25 + dense picks 20 candidates.
       Stage 2 (slow, precision): a cross-encoder reads (question, chunk) TOGETHER and re-orders them.
       Measure the quality gain, the latency cost, and use the re-ranker score to say "no answer".

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 14_rag_and_retrieval/07_reranking_cross_encoder/main.py
EXPECTED OUTPUT (shortened):
    stage 1 only (hybrid RRF): hit@1 0.83  MRR 0.88
    stage 1 + cross-encoder  : hit@1 0.88  MRR 0.92
    latency per query: bi-encoder ~x ms | cross-encoder on 20 candidates ~y ms
    unanswerable 'Do you sell gift cards?': best re-rank logit -7.06 (but answerable London: -10.94!)
NEEDS: torch, transformers; cached all-MiniLM-L6-v2 and cross-encoder/ms-marco-MiniLM-L-6-v2.  ~20 s.
"""
```
- **In simple words:** The two-stage idea in one paragraph.
- **Exactly:** Stage 1 hybrid BM25 + dense picks 20 candidates (recall). Stage 2 cross-encoder reorders them (precision). Measures quality gain, latency cost, and the reranker score as a "no answer" signal.
- **Why it is here:** This is the standard production retrieval funnel.

### 2. Imports  (`main.py` lines 16–24)
```python
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (BM25, CrossEncoderReranker, Embedder, build_chunks, chunk_for_embedding,   # noqa: E402
                    load_eval_set, rrf_fuse)
```
- **In simple words:** Load tools and the helpers from `common.py`.
- **Exactly:** `CrossEncoderReranker` and the retrieval helpers; `time` for latency.
- **Why it is here:** Reuse beats rewriting.

### 3. Build the retrievers once  (`main.py` lines 26–30)
```python
chunks = build_chunks()
texts = [chunk_for_embedding(c) for c in chunks]
ids = [c["chunk_id"] for c in chunks]
bm25, embedder, reranker = BM25(texts), Embedder(), CrossEncoderReranker()
matrix = embedder.encode(texts)
```
- **In simple words:** Prepare chunks, BM25, the embedder, the reranker and the vector matrix.
- **Exactly:** `texts` use the contextual header; `matrix` shape `(36, 384)`; the cross-encoder loads `ms-marco-MiniLM-L-6-v2`.
- **Why it is here:** Model loading takes seconds and must not happen per query.

### 4. stage1: hybrid candidates  (`main.py` lines 33–37)
```python
def stage1(question, k=20):
    """Hybrid retrieval: high RECALL is the goal here; order matters less."""
    s = matrix @ embedder.encode([question])[0]
    dense = [(int(j), float(s[j])) for j in np.argsort(-s)[:k]]
    return rrf_fuse([bm25.search(question, k), dense], top_n=k)
```
- **In simple words:** Get 20 good candidates using both search types.
- **Exactly:** Dense top 20 as `(row, score)`, plus `bm25.search(question, k)`, fused by `rrf_fuse(..., top_n=k)`. Returns `(row, rrf_score)` pairs.
- **Why it is here:** Goal is recall: the right chunk must be somewhere among the 20.

### 5. stage2: cross-encoder reorder  (`main.py` lines 40–44)
```python
def stage2(question, candidates):
    """Cross-encoder: one forward pass per (question, chunk) pair -> one relevance logit each."""
    logits = reranker.score(question, [texts[j] for j, _ in candidates])
    order = np.argsort(-logits)
    return [(candidates[o][0], float(logits[o])) for o in order]
```
- **In simple words:** Score each candidate together with the question, then sort.
- **Exactly:** `reranker.score(question, passages)` returns logits of shape `(n_candidates,)`; `argsort(-logits)` orders them; returns `(row, logit)` pairs in the new order. One forward pass per pair.
- **Why it is here:** Pairs must stay aligned with `candidates` (see BREAK IT).

### 6. One example before and after  (`main.py` lines 47–53)
```python
# ---- 1. One example, before and after -------------------------------------------------------------------
q = "What is the money back window after the tea arrives?"
first = stage1(q)
second = stage2(q, first)
print(f"Q: {q}")
print("  stage 1 top 3:", [ids[j] for j, _ in first[:3]])
print("  re-ranked top 3:", [(ids[j], round(s, 2)) for j, s in second[:3]])
```
- **In simple words:** Show the top three for one question before and after reranking.
- **Exactly:** Real: stage 1 top 3: Refund window, How to request a refund, Earning points. After reranking: Refund window `3.1`, How to request a refund `-2.79`, Earning points `-9.77`. The gap in logits is large: the cross-encoder is confident.
- **Why it is here:** Seeing scores makes "re-rank" concrete.

### 7. Setup for quality measurement  (`main.py` lines 55–62)
```python
# ---- 2. Quality on the 24 answerable questions ----------------------------------------------------------
eval_set = load_eval_set()
answerable = [x for x in eval_set if x["gold"]]


def rank_of(order, gold):
    order = [ids[j] for j, _ in order]
    return order.index(gold) + 1 if gold in order else None
```
- **In simple words:** Take the 24 answerable questions and write a helper that finds the rank of the right chunk.
- **Exactly:** `rank_of(order, gold)` converts `(row, score)` pairs to chunk ids and returns the 1-based position or `None`.
- **Why it is here:** Rank is the basis of hit@1 and MRR.

### 8. Compare stage 1 with stage 1 + stage 2  (`main.py` lines 65–77)
```python
r1, r2, recall20 = [], [], []
for x in answerable:
    c = stage1(x["question"])
    r1.append(rank_of(c, x["gold"][0]))
    r2.append(rank_of(stage2(x["question"], c), x["gold"][0]))
    recall20.append(r1[-1] is not None)
summ = lambda r: (np.mean([v == 1 for v in r]), np.mean([1 / v if v else 0 for v in r]))
print(f"\nstage 1 recall@20 (is the answer among the candidates at all?): {np.mean(recall20):.2f}")
print(f"stage 1 only (hybrid RRF): hit@1 {summ(r1)[0]:.2f}  MRR {summ(r1)[1]:.2f}")
print(f"stage 1 + cross-encoder  : hit@1 {summ(r2)[0]:.2f}  MRR {summ(r2)[1]:.2f}")
moved = [(x["question"], a, b) for x, a, b in zip(answerable, r1, r2) if a != b]
for question, a, b in moved:
    print(f"  rank {a} -> {b}: {question}")
```
- **In simple words:** Measure both and list the questions whose rank changed.
- **Exactly:** `recall20` = answer among the 20 candidates: real 1.00. Stage 1 only: hit@1 0.83, MRR 0.88. With the cross-encoder: hit@1 0.88, MRR 0.92. Several moved: "phone" 2 to 1, "Sunday" 12 to 1, "personal information" 2 to 1; but "refund days" 1 to 2 and "awake" 1 to 2 and "London" 8 to 20 got worse.
- **Why it is here:** Rerankers help on average but are not magic; always check the questions that got worse.

### 9. Latency: why not rerank everything  (`main.py` lines 79–90)
```python
# ---- 3. Latency: why we do NOT run the cross-encoder over the whole corpus --------------------------------
t0 = time.perf_counter()
for x in answerable:
    embedder.encode([x["question"]])
bi_ms = (time.perf_counter() - t0) / len(answerable) * 1000
t0 = time.perf_counter()
for x in answerable:
    reranker.score(x["question"], [texts[j] for j, _ in stage1(x["question"])])
ce_ms = (time.perf_counter() - t0) / len(answerable) * 1000
print(f"\nlatency per query (CPU): bi-encoder query embedding {bi_ms:.0f} ms | cross-encoder on 20 candidates {ce_ms:.0f} ms")
print(f"cross-encoder over 1M chunks would be ~{ce_ms / 20 * 1e6 / 1000 / 3600:.1f} hours PER QUERY; "
      "bi-encoder vectors are computed once, offline.")
```
- **In simple words:** Time a query embedding and a cross-encoder pass over 20 candidates.
- **Exactly:** Real (CPU): bi-encoder query embedding 7 ms; cross-encoder on 20 candidates 301 ms (about 15 ms per pair). Extrapolation: 1M chunks would take about 4.2 hours per query. Bi-encoder document vectors are computed once, offline.
- **Why it is here:** The reason for two stages: accuracy is expensive, so it is spent only on a shortlist.

### 10. The score as an "is the answer here?" signal  (`main.py` lines 92–98)
```python
# ---- 4. The re-ranker score as an 'is the answer here?' signal -------------------------------------------
print("\nbest re-rank logit per question (higher = more relevant):")
by_id = {x["id"]: x for x in eval_set}
for x in [by_id["q01"], by_id["q07"], by_id["q25"], by_id["q28"]]:
    best = stage2(x["question"], stage1(x["question"]))[0]
    print(f"  {best[1]:+6.2f}  {x['question']!r:<52} -> {ids[best[0]]}")
print("A threshold (e.g. logit < 0 -> 'I don't know') is calibrated on labelled questions, never guessed.")
```
- **In simple words:** Look at the best logit for answerable and unanswerable questions.
- **Exactly:** Real: `+1.58` for "How many days do I have to get a refund?", `-10.94` for the answerable London question, `-7.06` for the unanswerable gift cards, `-10.41` for coffee beans. A threshold would separate some but mislabel London.
- **Why it is here:** Thresholds must be calibrated on labelled data, never guessed.

### 11. Exercises as comments  (`main.py` lines 100–104)
```python
# TRY IT: re-rank only the top 5 of stage 1 instead of 20: faster, but does recall suffer?
# TRY IT: re-rank DENSE-only candidates instead of hybrid ones; compare hit@1.
# TRY IT: pass the raw chunk text instead of chunk_for_embedding(...) to the re-ranker. Does the title help?
# BREAK IT: give reranker.score the passages in a different order than `candidates` -> the ranks silently
#           point at the wrong chunks. Always keep (index, score) pairs together.
```
- **In simple words:** Ideas to try and break.
- **Exactly:** BREAK IT: scoring passages in a different order than `candidates` silently attaches scores to the wrong chunks.
- **Why it is here:** Keep `(index, score)` pairs together.


## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `np.argsort(-logits)` | indices that sort descending (negate, then ascending sort) | `np.argsort(-np.array([1, 3, 2]))` -> `[1, 2, 0]` |
| `[(candidates[o][0], float(logits[o])) for o in order]` | keep the chunk index and its score together while re-ordering | |
| `lambda r: (a, b)` | small anonymous function returning a tuple | `summ(r1)[0]` = hit@1 |
| `[v == 1 for v in r]` then `np.mean` | share of True values = hit@1 | `np.mean([True, False])` -> `0.5` |
| `1 / v if v else 0` | reciprocal rank, 0 when the gold chunk is missing (`None`) | |
| `order.index(gold) + 1` | 1-based rank of the gold chunk | |
| `zip(answerable, r1, r2)` | walk three lists in lockstep | |
| `f"{x:+6.2f}"` | always show the sign, width 6, 2 decimals | `-10.94`, `+1.58` |
| `time.perf_counter()` | high-resolution clock for timing | |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `CrossEncoderReranker()` | `common.py` (transformers) | loads `cross-encoder/ms-marco-MiniLM-L-6-v2` | `model_name` | object |
| `reranker.score(q, passages)` | `common.py` | tokenises `(q, p)` pairs together, one logit per pair | `max_length=512`, `truncation=True` | ndarray `(n,)` |
| `AutoModelForSequenceClassification` | transformers | encoder + classification head (1 output here) | model name | model |
| `tok([q]*n, passages, ...)` | transformers | tokeniser in *pair* mode: builds `[CLS] q [SEP] p [SEP]` | `padding`, `truncation`, `return_tensors="pt"` | batch dict |
| `Embedder().encode` | `common.py` | bi-encoder, unit vectors (step 03) | list of texts | `(n, 384)` |
| `BM25(...).search`, `rrf_fuse` | `common.py` | stage 1 hybrid retrieval (step 06) | `k`, `top_n` | list of `(index, score)` |
| `load_eval_set()` | `common.py` | 29 labelled questions (24 with a gold chunk) | – | list of dicts |
| `time.perf_counter()` | standard library | wall-clock timer | – | float seconds |

## Execution trace (real run)
Timings vary by machine and run; quality numbers are deterministic.
```
Q: What is the money back window after the tea arrives?
  stage 1 top 3: ['refunds_and_returns#Refund window', 'refunds_and_returns#How to request a refund', 'loyalty_program#Earning points']
  re-ranked top 3: [('refunds_and_returns#Refund window', 3.1), ('refunds_and_returns#How to request a refund', -2.79), ('loyalty_program#Earning points', -9.77)]

stage 1 recall@20 (is the answer among the candidates at all?): 1.00
stage 1 only (hybrid RRF): hit@1 0.83  MRR 0.88
stage 1 + cross-encoder  : hit@1 0.88  MRR 0.92
  rank 1 -> 2: How many days do I have to get a refund?
  rank 8 -> 20: Can you send tea to my cousin in London?
  rank 1 -> 2: Which teas will not keep me awake at night?
  rank 2 -> 1: Can I call you on the phone?
  rank 12 -> 1: Is anyone there to help me on Sunday?
  rank 2 -> 1: How do I remove my personal information from your systems?

latency per query (CPU): bi-encoder query embedding 6 ms | cross-encoder on 20 candidates 92 ms
cross-encoder over 1M chunks would be ~1.3 hours PER QUERY; bi-encoder vectors are computed once, offline.

best re-rank logit per question (higher = more relevant):
   +1.58  'How many days do I have to get a refund?'           -> refunds_and_returns#How to request a refund
  -10.94  'Can you send tea to my cousin in London?'           -> refunds_and_returns#How to request a refund
   -7.06  'Do you sell gift cards?'                            -> refunds_and_returns#Gift orders
  -10.41  'Do you sell coffee beans?'                          -> privacy_and_data#Data sharing
```
What to read from it:
- The example question: the re-ranker agrees with stage 1 but the *gap* is now visible (3.1 vs -2.79 vs -9.77). RRF scores never tell you how confident to be.
- hit@1 0.83 -> 0.88 is 20/24 -> 21/24. One question. Treat it as a direction, not proof (step 09 explains the noise).
- London is the cautionary tale: from rank 8 to 20, and its best logit (-10.94) is lower than two unanswerable questions. With a "refuse below 0" rule it would be refused.
- The best chunk for "How many days do I have to get a refund?" is now *How to request a refund*, not the gold *Refund window* (that is the 1 -> 2 move).

## Experiments (change ONE thing)
1. Re-rank only the top 5 of stage 1 instead of 20. Checked with a scratch script using the same models: hit@1 0.83 and MRR 0.88, the same as no re-ranking, so on this data the 20-candidate pool is what lets the re-ranker rescue the Sunday question (it was at rank 12).
2. Re-rank dense-only candidates instead of hybrid ones. Compare hit@1; stage-1 recall@20 is the ceiling.
3. Pass the raw chunk text instead of `chunk_for_embedding(...)` (no title and section header). Does "Opening hours" in the header matter for the Sunday question?
4. Print `1 / (1 + exp(-logit))` (sigmoid) next to each logit. Note that 0.5 is not automatically the right refusal threshold.

## Debugging practice
**BREAK IT (confirmed in a scratch copy):** give `reranker.score` the passages in a different order than `candidates`, for example `[texts[j] for j, _ in candidates][::-1]`. Nothing crashes. The logits are computed for reversed passages but paired with the original indices:
```
  re-ranked top 3: [('allergens_and_ingredients#Sesame and gluten', 3.1), ('refunds_and_returns#Gift orders', -2.79), ('product_catalog#Seasonal blends', -9.77)]
stage 1 + cross-encoder  : hit@1 0.00  MRR 0.07
  rank 1 -> 19: How many days do I have to get a refund?
```
Same scores as the healthy run (3.1, -2.79, -9.77), attached to the wrong chunks. This is the most dangerous kind of bug: plausible numbers, silent failure. Fix: never separate an index from its score; build `(index, score)` pairs in one comprehension, and keep an evaluation that would catch a drop from 0.88 to 0.00.

## Rebuild checklist
- [ ] stage 1: hybrid retrieve 20 candidates (`rrf_fuse` of BM25 and dense)
- [ ] stage 2: `reranker.score(question, [texts[j] for j, _ in candidates])`
- [ ] sort by logit descending, keep `(chunk index, logit)` pairs together
- [ ] measure recall@20 (stage-1 ceiling), then hit@1 and MRR before and after
- [ ] time both stages; extrapolate the cross-encoder cost to 1M chunks
- [ ] look at the best logit for answerable vs unanswerable questions before inventing a threshold

## The new problem (cliffhanger)
We now have good chunks in a good order and a score that hints "nothing relevant here". But nobody has answered the customer yet. Next: wire retrieval, re-ranking, refusal, prompt and generation into one pipeline with citations and see how it fails: [step 08, the RAG pipeline end to end](../08_rag_pipeline_end_to_end/WALKTHROUGH.md).

> **Say it aloud (interview box):** A bi-encoder embeds the query and documents separately so documents can be precomputed and searched fast, while a cross-encoder reads the query and a passage together with full attention, which is more accurate but must run per pair at query time. So I retrieve maybe 20 to 100 candidates for recall and re-rank only those for precision; here that took hit@1 from 0.83 to 0.88 for about 90 ms per query on CPU. I always check per-question movements and calibrate any refusal threshold on labelled data, because the re-ranker also pushed one valid question from rank 8 to rank 20.
