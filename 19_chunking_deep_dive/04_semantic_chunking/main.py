"""
GOAL:  Semantic chunking from scratch: embed sentences, measure the cosine distance between neighbours, cut where the
       distance spikes. Three breakpoint rules (percentile, standard deviation, gradient), the buffered-window trick,
       min/max size guards, a text sparkline of the distance curve, the cost, and an honest comparison with plain
       recursive chunking.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 19_chunking_deep_dive/04_semantic_chunking/main.py
EXPECTED OUTPUT (shortened):
    A) distance curve of the headingless postmortem with the cuts marked
    B) how many cuts each rule makes on the same distances
    C) cost: tokens embedded just to find boundaries
    D) retrieval: semantic variants vs recursive vs sentence-pack, overall and on the no-heading document
NEEDS: torch, transformers (cached MiniLM).  ~20 s.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common19 import (_CACHE, RawChunk, count_tokens, embed, evaluate, load_docs, load_step, make_chunk,  # noqa: E402
                      sentence_spans, sparkline, table)


def neighbour_distances(texts, buffer=1):
    """d[i] = 1 - cos(window around sentence i, window around sentence i+1). `buffer` sentences each side are
    glued to every sentence before embedding, so one short sentence ('Yes.') cannot cause a fake spike."""
    windows = [" ".join(texts[max(0, i - buffer): i + buffer + 1]) for i in range(len(texts))]
    E = embed(windows)                                       # (n_sentences, 384), unit vectors
    return 1.0 - (E[:-1] * E[1:]).sum(axis=1)                # (n_sentences - 1,)


def threshold(d, method, param):
    if method == "percentile":                               # cut at the top (100-param)% biggest distances
        return np.percentile(d, param)
    if method == "stdev":                                    # mean + param standard deviations
        return d.mean() + param * d.std()
    if method == "iqr":                                      # mean + param * interquartile range
        q1, q3 = np.percentile(d, [25, 75])
        return d.mean() + param * (q3 - q1)
    raise ValueError(method)


def breakpoints(d, method="percentile", param=90):
    if method == "gradient":                                 # cut where the distance CHANGES fastest (good when
        g = np.gradient(d)                                   # the whole document is similar: absolute values mislead)
        return set(np.where(g > np.percentile(g, param))[0].tolist())
    return set(np.where(d > threshold(d, method, param))[0].tolist())   # break AFTER sentence i if d[i] > threshold


def enforce_sizes(cuts, d, tok, min_tokens, max_tokens):
    """tok[i] = tokens of sentence i. Remove weak cuts that leave tiny chunks; add cuts inside oversize chunks."""
    def groups(cs):
        edges = [-1] + sorted(cs) + [len(tok) - 1]
        return [(a + 1, b) for a, b in zip(edges, edges[1:])]            # inclusive sentence ranges
    cuts = set(cuts)
    changed = True
    while changed:
        changed = False
        for a, b in groups(cuts):
            if sum(tok[a:b + 1]) < min_tokens and len(groups(cuts)) > 1:
                near = [c for c in (a - 1, b) if c in cuts]              # the cuts touching this tiny group
                cuts.remove(min(near, key=lambda c: d[c]))               # drop the WEAKER boundary
                changed = True
                break
    changed = True
    while changed:
        changed = False
        for a, b in groups(cuts):
            if sum(tok[a:b + 1]) > max_tokens and b > a:
                cuts.add(a + int(np.argmax(d[a:b])))                     # split at the biggest distance inside
                changed = True
                break
    return cuts


def semantic_chunks(doc, method="percentile", param=90, buffer=1, min_tokens=30, max_tokens=200):
    spans = sentence_spans(doc["text"])
    texts = [doc["text"][s:e] for s, e in spans]
    if len(spans) < 3:
        return [make_chunk(doc, 0, len(doc["text"]))]
    d = neighbour_distances(texts, buffer)
    tok = [count_tokens(t) for t in texts]
    cuts = enforce_sizes(breakpoints(d, method, param), d, tok, min_tokens, max_tokens)
    edges = [-1] + sorted(cuts) + [len(spans) - 1]
    return [make_chunk(doc, spans[a + 1][0], spans[b][1]) for a, b in zip(edges, edges[1:])]


def sentence_pack(doc, max_tokens=100):
    """Baseline with the SAME sentence units but no embeddings: pack sentences up to a token budget."""
    out, cur = [], []
    for s, e in sentence_spans(doc["text"]):
        if cur and count_tokens(doc["text"][cur[0][0]:e]) > max_tokens:
            out.append(make_chunk(doc, cur[0][0], cur[-1][1]))
            cur = []
        cur.append((s, e))
    return out + ([make_chunk(doc, cur[0][0], cur[-1][1])] if cur else [])


def main():
    docs = load_docs()
    pm = next(d for d in docs if d["doc_id"] == "incident_postmortem")

    # ---- A) the distance curve ------------------------------------------------------------------------------------
    spans = sentence_spans(pm["text"])
    texts = [pm["text"][s:e] for s, e in spans]
    d = neighbour_distances(texts, buffer=1)
    cuts = breakpoints(d, "percentile", 85)
    print(f"A) postmortem: {len(texts)} sentences, {len(d)} neighbour distances: min {d.min():.2f} mean {d.mean():.2f} max {d.max():.2f}")
    print("   distance: " + sparkline(d))
    print("   cut here: " + "".join("^" if i in cuts else " " for i in range(len(d))) + f"   ({len(cuts)} cuts at the 85th percentile)")
    for i in sorted(cuts)[:3]:
        print(f"   cut after sentence {i}: {texts[i][:50]!r} | {texts[i + 1][:46]!r}  d={d[i]:.2f}")

    # ---- B) rules compared on identical distances -------------------------------------------------------------------
    rows = []
    for method, param in (("percentile", 95), ("percentile", 85), ("percentile", 70), ("stdev", 1.0), ("stdev", 3.0),
                          ("iqr", 1.5), ("gradient", 90)):
        c = breakpoints(d, method, param)
        rows.append([f"{method} {param}", len(c), len(c) + 1])
    print("\nB) number of breakpoints per rule on the same postmortem distances (before min/max size guards)")
    print(table(rows, ["rule", "cuts", "chunks"]))
    print(f"   stdev k=3 on a short document: threshold {threshold(d, 'stdev', 3):.2f} vs max distance {d.max():.2f} -> "
          f"{'no cut at all' if threshold(d, 'stdev', 3) >= d.max() else 'some cuts'}")

    # ---- C) cost ---------------------------------------------------------------------------------------------------------
    tot_sent = sum(len(sentence_spans(x["text"])) for x in docs)
    tok_doc = sum(count_tokens(x["text"]) for x in docs)
    tok_win = sum(count_tokens(" ".join(x["text"][s:e] for s, e in sentence_spans(x["text"])[max(0, i - 1): i + 2]))
                  for x in docs for i in range(len(sentence_spans(x["text"]))))
    rec = load_step("02_recursive_splitter_rebuilt")
    t0 = time.perf_counter()
    [rec.RecursiveSplitter(100, 0).split_document(x) for x in docs]
    t_rec = time.perf_counter() - t0
    _CACHE.clear(); t0 = time.perf_counter()
    [semantic_chunks(x, "percentile", 90) for x in docs]                # cache cleared: the honest cost of buffer=1
    t_sem = time.perf_counter() - t0
    print(f"\nC) cost on {len(docs)} docs ({tok_doc} tokens, {tot_sent} sentences)")
    print(f"   recursive: 0 model calls, {t_rec * 1000:.0f} ms | semantic: {tot_sent} sentence-window embeddings, "
          f"~{tok_win} tokens through the model (buffer=1 reads every token ~3x), {t_sem * 1000:.0f} ms")
    print("   ...and the final chunks must still be embedded AGAIN for the index.")

    # ---- D) retrieval ------------------------------------------------------------------------------------------------------
    rows, results = [], {}

    def add(name, fn):
        chunks = [c for x in docs for c in fn(x)]
        r = evaluate(chunks, docs=docs)
        results[name] = r
        sub = [x for x in r["dense"]["rows"] if x["id"].startswith("n") and next(q for q in qs if q["id"] == x["id"])["doc_id"] == "incident_postmortem"]
        pm_h3 = np.mean([x["hit@3"] for x in sub])
        dn = r["dense"]
        rows.append([name, len(chunks), f"{r['mean_tok']:.0f}", f"{dn['hit@1']:.2f}", f"{dn['hit@3']:.2f}", f"{dn['mrr']:.2f}",
                     f"{r['hybrid']['hit@3']:.2f}", f"{pm_h3:.2f}"])

    from common19 import load_eval
    qs = load_eval()
    add("recursive 100 tok", lambda x: rec.RecursiveSplitter(100, 0).split_document(x))
    add("sentence-pack 100 tok", sentence_pack)
    add("semantic pct 95", lambda x: semantic_chunks(x, "percentile", 95))
    add("semantic pct 85", lambda x: semantic_chunks(x, "percentile", 85))
    add("semantic pct 70", lambda x: semantic_chunks(x, "percentile", 70))
    add("semantic pct 85, buffer 0", lambda x: semantic_chunks(x, "percentile", 85, buffer=0))
    add("semantic gradient 90", lambda x: semantic_chunks(x, "gradient", 90))
    print("\nD) retrieval, 13 documents, 74 questions (last column: dense hit@3 on the headingless postmortem only)")
    print(table(rows, ["chunker", "chunks", "mean tok", "dense h@1", "dense h@3", "MRR", "hybrid h@3", "postmortem h@3"]))


if __name__ == "__main__":
    main()

# TRY IT: change buffer=1 to 0 and 3 in section A; watch the sparkline get noisier / smoother.
# TRY IT: raise min_tokens to 80 in semantic_chunks; how many chunks remain at percentile 85?
# BREAK IT: threshold(d, "median", 50) -> ValueError: median (the rule name is not one of percentile / stdev / iqr).
