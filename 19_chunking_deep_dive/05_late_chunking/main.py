"""
GOAL:  Late chunking: run the embedding model ONCE over a whole window of text, then mean-pool the token vectors
       that belong to each chunk. Each chunk vector has "seen" the text around it. Compare with naive chunk
       embeddings (embed each chunk alone) on pronoun / reference questions and on the whole eval set,
       and sweep the window size.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 19_chunking_deep_dive/05_late_chunking/main.py
EXPECTED OUTPUT (shortened):
    A) one window: token vectors (T, 384) -> 3 chunk vectors; cosine(naive, late) per chunk
    B) a pronoun-chunk: similarity to the right question vs a distractor, naive vs late
    C) retrieval table: naive vs late for window = 40 / 128 / 256 / 480 tokens
NEEDS: torch, transformers (cached MiniLM).  ~25 s.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common19 import RawChunk, count_tokens, embed, evaluate, get_embedder, load_docs, load_eval, sentence_spans, table  # noqa: E402


def late_vectors(window_text, char_spans):
    """One forward pass over `window_text`; one unit vector per (start, end) character span inside it."""
    emb = get_embedder()
    enc = emb.tok(window_text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=512)
    offs = enc.pop("offset_mapping")[0].tolist()             # [(char_start, char_end)] per token; (0, 0) for [CLS]/[SEP]
    with torch.no_grad():
        H = emb.model(**enc).last_hidden_state[0]            # (T, 384): token vectors that have attended to the WHOLE window
    out = []
    for a, b in char_spans:
        idx = [i for i, (s, e) in enumerate(offs) if e > s and s >= a and e <= b]   # tokens inside this chunk, no specials
        v = H[idx].mean(dim=0)                               # mean-pool ONLY this chunk's tokens -> (384,)
        out.append(torch.nn.functional.normalize(v, dim=0).numpy())
    return np.vstack(out), H.shape[0]


def pack(doc, spans, budget):
    """Greedy sentence packing: consecutive sentence spans -> groups of <= budget tokens."""
    groups, cur = [], []
    for sp in spans:
        if cur and count_tokens(doc["text"][cur[0][0]:sp[1]]) > budget:
            groups.append(cur)
            cur = []
        cur.append(sp)
    return groups + ([cur] if cur else [])


def build(doc, window_tokens, chunk_tokens=40):
    """Windows of <= window_tokens made of small chunks of <= chunk_tokens. Returns chunks + (naive, late) vectors."""
    sents = sentence_spans(doc["text"])
    chunks, naive_texts, late = [], [], []
    for win in pack(doc, sents, window_tokens):
        w0, w1 = win[0][0], win[-1][1]
        small = pack(doc, win, chunk_tokens)
        spans = [(g[0][0] - w0, g[-1][1] - w0) for g in small]          # chunk spans relative to the window text
        vecs, _ = late_vectors(doc["text"][w0:w1], spans)
        for g, v in zip(small, vecs):
            s, e = g[0][0], g[-1][1]
            chunks.append(RawChunk(doc["doc_id"], s, e, doc["text"][s:e]))
            late.append(v)
    return chunks, np.vstack(late)


def main():
    docs = load_docs()
    hb = next(d for d in docs if d["doc_id"] == "employee_handbook")

    # ---- A) one window, step by step ----------------------------------------------------------------------------------
    start = hb["text"].index("#### 2.3.2 Secondary caregiver")
    end = hb["text"].index("### 3. Expenses") if "### 3. Expenses" in hb["text"] else hb["text"].index("## 3. Expenses")
    window = hb["text"][start:end]
    sents = sentence_spans(window)
    small = pack({"text": window}, sents, 25)
    spans = [(g[0][0], g[-1][1]) for g in small]
    vecs, T = late_vectors(window, spans)
    naive = embed([window[a:b] for a, b in spans])
    print(f"A) window = parental-leave subsections: {count_tokens(window)} tokens -> model sees T={T} tokens (incl. [CLS], [SEP]); "
          f"token matrix (T, 384); {len(spans)} chunks")
    for (a, b), n, l in zip(spans, naive, vecs):
        print(f"   cos(naive, late) = {n @ l:.3f}   {window[a:b][:58]!r}")

    # ---- B) a chunk that is meaningless alone --------------------------------------------------------------------------------
    q = "How long must each block of secondary caregiver leave be?"
    target = next(i for i, (a, b) in enumerate(spans) if "Each block" in window[a:b])
    qv = embed([q])[0]
    print(f"\nB) question {q!r}")
    print(f"   chunk under test: {window[spans[target][0]:spans[target][1]]!r}")
    for name, M in (("naive (chunk alone)", naive), ("late  (chunk in window)", vecs)):
        sims = M @ qv
        order = np.argsort(-sims)
        print(f"   {name:<24} sim to target {sims[target]:.3f}, rank {int(np.where(order == target)[0][0]) + 1} of {len(spans)} in this window")

    # ---- C) retrieval over the whole corpus, window sweep ----------------------------------------------------------------------
    qs = load_eval()
    rows = []
    for W in (40, 128, 256, 480):
        all_chunks, all_late = [], []
        for d in docs:
            c, v = build(d, W)
            all_chunks += c
            all_late.append(v)
        late_m = np.vstack(all_late)
        naive_m = embed([c.text for c in all_chunks])
        for name, M in (("naive", naive_m), ("late", late_m)):
            r = evaluate(all_chunks, qs, docs, vectors=M)
            ref = [x["hit@1"] for x in r["dense"]["rows"] if x["kind"] == "reference"]
            rows.append([W, name, len(all_chunks), f"{r['dense']['hit@1']:.2f}", f"{r['dense']['hit@3']:.2f}",
                         f"{r['dense']['mrr']:.2f}", f"{np.mean(ref):.2f} (n={len(ref)})"])
    print("\nC) chunks of <= 40 tokens; the window is how much text the model reads at once (40 = the chunk itself)")
    print(table(rows, ["window", "vectors", "chunks", "hit@1", "hit@3", "MRR", "reference-q hit@1"]))


if __name__ == "__main__":
    main()

# TRY IT: change chunk_tokens=40 to 20 in build(); smaller chunks have less of their own meaning, so context should matter more.
# TRY IT: in late_vectors, include the [CLS] token in the pool (drop the `e > s` test); does the vector change much?
# BREAK IT: window of 600 tokens with truncation=True silently drops the tail -> H[idx] is empty -> mean of nothing = nan.
