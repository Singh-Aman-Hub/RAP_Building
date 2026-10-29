"""
SOLUTION for the step 02/03 exercise "does chunk size change retrieval quality?"

For several chunk sizes: split every document by packing whole sentences up to N words, embed the chunks with
MiniLM, and measure dense hit@1 / hit@3 / MRR on the 24 answerable questions.
A chunk counts as relevant if it comes from the gold document AND contains the gold answer string.

RUN (from repo root):  .venv/bin/python 14_rag_and_retrieval/solutions/chunk_size_sweep.py
NEEDS: torch, transformers (cached MiniLM).  ~15 s.
"""

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import Embedder, load_documents, load_eval_set   # noqa: E402


def sentence_pack(text, max_words):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    chunks, cur = [], []
    for s in sents:
        if cur and len(" ".join(cur + [s]).split()) > max_words:
            chunks.append(" ".join(cur))
            cur = []
        cur.append(s)
    return chunks + ([" ".join(cur)] if cur else [])


docs = load_documents()
bodies = {d["doc_id"]: " ".join(l.strip() for l in d["text"].splitlines() if l.strip() and not l.startswith("#"))
          for d in docs}
titles = {d["doc_id"]: d["title"] for d in docs}
questions = [q for q in load_eval_set() if q["gold"]]
embedder = Embedder()
q_vecs = embedder.encode([q["question"] for q in questions])

print(f"{'max words':>9} {'chunks':>7} {'avg words':>10} {'hit@1':>6} {'hit@3':>6} {'MRR':>6}")
for size in (12, 25, 50, 100, 200):
    chunks = [(doc_id, c) for doc_id, body in bodies.items() for c in sentence_pack(body, size)]
    matrix = embedder.encode([f"{titles[d]}: {c}" for d, c in chunks])
    ranks = []
    for q, qv in zip(questions, q_vecs):
        gold_doc = q["gold"][0].split("#")[0]
        order = np.argsort(-(matrix @ qv))
        hit = [i for i, j in enumerate(order, 1)
               if chunks[j][0] == gold_doc and q["answer"].lower() in chunks[j][1].lower()]
        ranks.append(hit[0] if hit else None)
    avg = np.mean([len(c.split()) for _, c in chunks])
    print(f"{size:>9} {len(chunks):>7} {avg:>10.1f} {np.mean([r == 1 for r in ranks]):>6.2f} "
          f"{np.mean([r is not None and r <= 3 for r in ranks]):>6.2f} {np.mean([1 / r if r else 0 for r in ranks]):>6.2f}")
print("Read it carefully: at 200 words each document is ONE chunk, so there are only 9 candidates and retrieval")
print("becomes 'pick the right document' (random guessing already gets hit@1 = 0.11). The LLM then has to find")
print("the fact inside ~107 words, and every prompt pays for them. Compare heading-aware chunks (36 chunks,")
print("~29 words): dense hit@1 0.83 in step 06. Chunk BOUNDARIES that follow the document's structure beat any")
print("single size. Always judge chunking end to end (step 09), not by retrieval metrics alone.")
