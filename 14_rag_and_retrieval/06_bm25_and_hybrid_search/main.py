"""
GOAL:  Keyword search with BM25 written out by hand, then compare it with dense search on 24 labelled
       questions, broken down by question type (keyword / paraphrase / exact product code).
       Finally fuse both rankings with Reciprocal Rank Fusion (RRF) and with a weighted score blend.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 14_rag_and_retrieval/06_bm25_and_hybrid_search/main.py
EXPECTED OUTPUT (shortened):
    idf('refund') = ...  idf('days') = ...   (rare words count more)
    method          hit@1  hit@3   MRR   | keyword paraphrase exact_code (hit@1)
    BM25 ...  dense ...  hybrid RRF ...
NEEDS: torch, transformers (cached MiniLM).  ~10 s.
"""

import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BM25, Embedder, build_chunks, chunk_for_embedding, load_eval_set, rrf_fuse, tokenize   # noqa: E402

chunks = build_chunks()
texts = [chunk_for_embedding(c) for c in chunks]

# ---- 1. Tokens: what BM25 actually compares ---------------------------------------------------------------
print("tokens:", tokenize("What is in CK-MASALA-250? How many days do I have for a refund?"))

# ---- 2. BM25 by hand for one (query, chunk) pair ------------------------------------------------------------
docs = [Counter(tokenize(t)) for t in texts]
N = len(docs)
avg_len = np.mean([sum(d.values()) for d in docs])
df = Counter(term for d in docs for term in d)                 # document frequency: chunks containing the term


def idf(term):
    """Rare terms carry more information. +1 inside the log keeps it positive."""
    return math.log(1 + (N - df[term] + 0.5) / (df[term] + 0.5))


print("\nidf (rare words weigh more):", {t: round(idf(t), 2) for t in ("refund", "days", "tea", "ck-tulsi-100")})


def bm25_one(query, i, k1=1.5, b=0.75):
    doc_len = sum(docs[i].values())
    score = 0.0
    for term in tokenize(query):
        tf = docs[i].get(term, 0)                              # how often the term occurs in this chunk
        if tf == 0:
            continue
        # tf saturation: the 5th 'refund' adds much less than the 1st (k1);
        # length normalisation: matches in long chunks count a little less (b)
        score += idf(term) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avg_len))
    return score


q = "How many days do I have for a refund?"
i = next(j for j, c in enumerate(chunks) if c["chunk_id"] == "refunds_and_returns#Refund window")
print(f"BM25 by hand for {q!r} vs refund-window chunk: {bm25_one(q, i):.3f} "
      f"| library class: {BM25(texts).scores(q)[i]:.3f}")

# ---- 3. Three retrievers on the labelled questions -----------------------------------------------------------
bm25 = BM25(texts)
embedder = Embedder()
matrix = embedder.encode(texts)
eval_set = [x for x in load_eval_set() if x["gold"]]
ids = [c["chunk_id"] for c in chunks]


def dense_rank(query, k=20):
    s = matrix @ embedder.encode([query])[0]
    top = np.argsort(-s)[:k]
    return [(int(j), float(s[j])) for j in top]


def blend_rank(query, alpha=0.5, k=20):
    """Weighted score fusion: min-max normalise both score lists, then alpha*dense + (1-alpha)*bm25."""
    d = matrix @ embedder.encode([query])[0]
    s = bm25.scores(query)
    norm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
    mix = alpha * norm(d) + (1 - alpha) * norm(s)
    top = np.argsort(-mix)[:k]
    return [(int(j), float(mix[j])) for j in top]


methods = {
    "BM25 (sparse)": lambda qq: bm25.search(qq, 20),
    "dense (MiniLM)": dense_rank,
    "hybrid RRF k=60": lambda qq: rrf_fuse([bm25.search(qq, 20), dense_rank(qq)], k=60, top_n=20),
    "hybrid blend a=0.5": blend_rank,
}
kinds = ["keyword", "paraphrase", "exact_code"]
print(f"\n{'method':<20}{'hit@1':>7}{'hit@3':>7}{'MRR':>7}   " + "".join(f"{k:>12}" for k in kinds) + "   (hit@1 by type)")
for name, fn in methods.items():
    ranks = []
    for x in eval_set:
        order = [ids[j] for j, _ in fn(x["question"])]
        ranks.append(order.index(x["gold"][0]) + 1 if x["gold"][0] in order else None)
    hit1 = np.mean([r == 1 for r in ranks])
    hit3 = np.mean([r is not None and r <= 3 for r in ranks])
    mrr = np.mean([1 / r if r else 0 for r in ranks])          # Mean Reciprocal Rank: 1, 1/2, 1/3 ... or 0
    by_kind = [np.mean([r == 1 for r, x in zip(ranks, eval_set) if x["kind"] == k]) for k in kinds]
    print(f"{name:<20}{hit1:>7.2f}{hit3:>7.2f}{mrr:>7.2f}   " + "".join(f"{v:>12.2f}" for v in by_kind))

# ---- 4. Where each one wins -----------------------------------------------------------------------------------
print()
for qq in ["What is the money back window after the tea arrives?", "pin codes"]:
    b_top = ids[bm25.search(qq, 1)[0][0]]
    d_top = ids[dense_rank(qq, 1)[0][0]]
    print(f"Q: {qq}\n   BM25 top: {b_top}\n   dense top: {d_top}")
print("\nBoth questions: BM25 wins. Q1: the heading word 'window' matched exactly; dense ranked that chunk 2nd.")
print("Q2: short jargon gives dense too little meaning to work with. Dense still wins more paraphrases overall (table).")
print("On this tiny, clean knowledge base dense alone is already strong, so hybrid only ties it on hit@1.")
print("Hybrid pays off on big corpora full of product codes, names and error strings,")
print("where one retriever's miss is often the other one's hit.")

# TRY IT: set b=0 in BM25 (no length normalisation) and b=1. Which questions change rank?
# TRY IT: change alpha in blend_rank to 0.2 and 0.8. Then RRF k to 1 and 1000. Which is more robust?
# TRY IT: remove the stopword filter in common.tokenize: 'the', 'is', 'do' now add noise to BM25.
# BREAK IT: call rrf_fuse with ranks starting at 0 instead of 1 (enumerate(..., start=0)): with k=0 -> ZeroDivisionError.
