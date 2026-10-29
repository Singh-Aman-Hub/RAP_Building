"""
GOAL:  Four "advanced RAG" techniques, each shown on a question the basic pipeline gets wrong or wastes work on:
         A) multi-query rewriting + RRF   (the query's words do not match the document's words)
         B) HyDE: search with a hypothetical ANSWER instead of the question
         C) small-to-big (parent document) retrieval: match small sentences, return the whole section
         D) semantic cache: reuse an answer when a new question means the same as an old one
       The rewrites and the hypothetical answer are hand-written here; in production an LLM writes them.
       Vector databases (FAISS, Chroma, pgvector, Qdrant) are in vector_db_reference.py (read only).

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 14_rag_and_retrieval/10_advanced_rag_and_vector_dbs/main.py
EXPECTED OUTPUT (shortened):
    A) original question rank of 'International shipping': 8 -> multi-query RRF rank: 1
    B) HyDE rank: 1
    C) sentence hit -> parent section returned
    D) cache HIT for paraphrases, MISS for 'delivery hours' -- but the margin is small
NEEDS: torch, transformers (cached MiniLM).  ~10 s.
"""

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BM25, Embedder, build_chunks, chunk_for_embedding, rrf_fuse   # noqa: E402

chunks = build_chunks()
texts = [chunk_for_embedding(c) for c in chunks]
ids = [c["chunk_id"] for c in chunks]
embedder, bm25 = Embedder(), BM25(texts)
matrix = embedder.encode(texts)
GOLD = "shipping_and_delivery#International shipping"


def hybrid(query, k=20):
    s = matrix @ embedder.encode([query])[0]
    dense = [(int(j), float(s[j])) for j in np.argsort(-s)[:k]]
    return rrf_fuse([bm25.search(query, k), dense], top_n=k)


def rank_of(ranking, gold=GOLD):
    order = [ids[j] for j, _ in ranking]
    return order.index(gold) + 1 if gold in order else None


# =========================================================================================================
# A) Multi-query: ask an LLM for 3 rewrites, retrieve for each, fuse the rankings with RRF.
# =========================================================================================================
question = "Can you send tea to my cousin in London?"
rewrites = [                                                   # what an LLM returns for "rewrite this for search"
    "Do you ship internationally?",
    "Shipping to the United Kingdom or outside India",
    "Delivery to other countries",
]
print("A) MULTI-QUERY")
print(f"   original question -> rank of the gold chunk: {rank_of(hybrid(question))}")
for r in rewrites:
    print(f"   rewrite {r!r:<52} -> rank {rank_of(hybrid(r))}")
fused = rrf_fuse([hybrid(question)] + [hybrid(r) for r in rewrites], top_n=20)
print(f"   all four fused with RRF -> rank {rank_of(fused)}   (cost: 1 extra LLM call + 3 extra searches)")

# =========================================================================================================
# B) HyDE (Hypothetical Document Embeddings): an answer looks more like the document than a question does.
# =========================================================================================================
hypothetical = ("We only deliver within India and do not ship tea abroad. International customers can order "
                "through a partner store.")                   # an LLM's guess; it may be WRONG, that is fine
s_question = matrix @ embedder.encode([question])[0]
s_hyde = matrix @ embedder.encode([hypothetical])[0]
rank = lambda s: int(np.where(np.argsort(-s) == ids.index(GOLD))[0][0]) + 1
print("\nB) HyDE (dense only)")
print(f"   embed the question            -> rank {rank(s_question)} (cosine {s_question[ids.index(GOLD)]:.2f})")
print(f"   embed a hypothetical answer   -> rank {rank(s_hyde)} (cosine {s_hyde[ids.index(GOLD)]:.2f})")
print("   Risk: if the guess invents facts in the wrong direction it pulls the wrong chunks. Never show the guess to users.")

# =========================================================================================================
# C) Small-to-big: index SENTENCES (precise match), but give the LLM the whole PARENT section (enough context).
# =========================================================================================================
sentences, parent = [], []
for ci, c in enumerate(chunks):
    for s in re.split(r"(?<=[.!?])\s+", c["text"]):
        sentences.append(f"{c['section']}: {s}")
        parent.append(ci)                                      # child sentence -> parent chunk index
sent_matrix = embedder.encode(sentences)
q = "Do I need to send the pouch back to get my money?"
best = int(np.argmax(sent_matrix @ embedder.encode([q])[0]))
print(f"\nC) SMALL-TO-BIG: {len(sentences)} sentence vectors, each pointing to one of {len(chunks)} parent sections")
print(f"   Q: {q}")
print(f"   best child sentence : {sentences[best]!r}")
print(f"   parent sent to LLM  : {ids[parent[best]]} ({len(chunks[parent[best]]['text'].split())} words)")
print("   Small chunks embed ONE idea (sharp vectors); big chunks give the LLM the surrounding conditions.")

# =========================================================================================================
# D) Semantic cache: store (question vector, answer). A new question close enough reuses the answer.
# =========================================================================================================
cache = []                                                     # list of (unit vector, answer)


def cached_answer(question, threshold=0.70):
    v = embedder.encode([question])[0]
    for vec, ans in cache:
        sim = float(vec @ v)
        if sim >= threshold:
            return ans, sim
    return None, max((float(vec @ v) for vec, _ in cache), default=0.0)


cache.append((embedder.encode(["What are your customer support hours?"])[0], "Monday to Saturday, 10am to 6pm IST."))
print("\nD) SEMANTIC CACHE (threshold 0.70)")
for q in ["What are your customer support hours please?", "What time is customer support open?", "What are your delivery hours?"]:
    ans, sim = cached_answer(q)
    print(f"   {q!r:<46} cos {sim:.2f} -> {'HIT: ' + ans if ans else 'MISS: run the full pipeline'}")
print("   The gap between a true paraphrase and a different question is small (compare the last two cosines).")
print("   Too low a threshold returns the WRONG cached answer; invalidate entries when the documents change.")

# TRY IT: remove the original question from the RRF list in A (only rewrites). Does the rank change?
# TRY IT: write a deliberately WRONG hypothetical answer ("We ship to London in 5 days.") for HyDE. What rank now?
# TRY IT: in D set threshold=0.6 and ask "What are your delivery hours?" -> it now returns the support hours: a wrong hit.
# BREAK IT: in C, append to `sentences` but forget `parent.append(ci)` -> the lists drift apart and the wrong
#           parent is returned (or IndexError). Keep child -> parent mappings in ONE structure.
