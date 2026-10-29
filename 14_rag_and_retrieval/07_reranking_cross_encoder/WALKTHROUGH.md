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
