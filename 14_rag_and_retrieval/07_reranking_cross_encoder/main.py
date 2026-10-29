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

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (BM25, CrossEncoderReranker, Embedder, build_chunks, chunk_for_embedding,   # noqa: E402
                    load_eval_set, rrf_fuse)

chunks = build_chunks()
texts = [chunk_for_embedding(c) for c in chunks]
ids = [c["chunk_id"] for c in chunks]
bm25, embedder, reranker = BM25(texts), Embedder(), CrossEncoderReranker()
matrix = embedder.encode(texts)


def stage1(question, k=20):
    """Hybrid retrieval: high RECALL is the goal here; order matters less."""
    s = matrix @ embedder.encode([question])[0]
    dense = [(int(j), float(s[j])) for j in np.argsort(-s)[:k]]
    return rrf_fuse([bm25.search(question, k), dense], top_n=k)


def stage2(question, candidates):
    """Cross-encoder: one forward pass per (question, chunk) pair -> one relevance logit each."""
    logits = reranker.score(question, [texts[j] for j, _ in candidates])
    order = np.argsort(-logits)
    return [(candidates[o][0], float(logits[o])) for o in order]


# ---- 1. One example, before and after -------------------------------------------------------------------
q = "What is the money back window after the tea arrives?"
first = stage1(q)
second = stage2(q, first)
print(f"Q: {q}")
print("  stage 1 top 3:", [ids[j] for j, _ in first[:3]])
print("  re-ranked top 3:", [(ids[j], round(s, 2)) for j, s in second[:3]])

# ---- 2. Quality on the 24 answerable questions ----------------------------------------------------------
eval_set = load_eval_set()
answerable = [x for x in eval_set if x["gold"]]


def rank_of(order, gold):
    order = [ids[j] for j, _ in order]
    return order.index(gold) + 1 if gold in order else None


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

# ---- 4. The re-ranker score as an 'is the answer here?' signal -------------------------------------------
print("\nbest re-rank logit per question (higher = more relevant):")
by_id = {x["id"]: x for x in eval_set}
for x in [by_id["q01"], by_id["q07"], by_id["q25"], by_id["q28"]]:
    best = stage2(x["question"], stage1(x["question"]))[0]
    print(f"  {best[1]:+6.2f}  {x['question']!r:<52} -> {ids[best[0]]}")
print("A threshold (e.g. logit < 0 -> 'I don't know') is calibrated on labelled questions, never guessed.")

# TRY IT: re-rank only the top 5 of stage 1 instead of 20: faster, but does recall suffer?
# TRY IT: re-rank DENSE-only candidates instead of hybrid ones; compare hit@1.
# TRY IT: pass the raw chunk text instead of chunk_for_embedding(...) to the re-ranker. Does the title help?
# BREAK IT: give reranker.score the passages in a different order than `candidates` -> the ranks silently
#           point at the wrong chunks. Always keep (index, score) pairs together.
