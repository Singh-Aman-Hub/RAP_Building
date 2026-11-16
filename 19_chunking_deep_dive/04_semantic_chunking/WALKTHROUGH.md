# Step 04 — Semantic chunking

## The story so far
[Step 03](../03_structure_aware_markdown/WALKTHROUGH.md) chunked by the structure the author wrote: headings, tables, code fences. Its limit: when there is no structure (an incident postmortem written as flowing prose, a transcript, an exported chat), the heading tree is flat and the breadcrumb is empty. Back to the [module README](../README.md).

## The pain point
"Where does the topic change?" is exactly what a fixed size or a separator cannot answer. A 100-token window can straddle two topics (the vector becomes a blend, close to neither), and a separator-based splitter happily keeps two unrelated paragraphs together because they are short.

## The spark
Use the embedding model as a **topic-change detector**. Embed each sentence, compute the cosine distance between each sentence and the next, and cut where the distance spikes. Distances are noisy (one short sentence like "Yes." looks unrelated to everything), so first glue `buffer` neighbouring sentences around each sentence before embedding, and finally guard the result with minimum and maximum token sizes.

## Goal
Implement semantic chunking from scratch and be able to say, with numbers, what it costs and when it does *not* beat simple sentence packing:
1. The distance curve of the headingless postmortem as a text sparkline, with the cuts marked (A).
2. Four breakpoint rules (percentile, standard deviation, IQR, gradient) on the same distances (B).
3. The ingestion cost: embeddings spent only to find boundaries (C).
4. Retrieval against recursive chunking and a no-embedding sentence-pack baseline, overall and on the headingless document (D).

## The theory (read this before the interview)
```
sentences s_0 ... s_{n-1}
window_i   = s_{i-b} ... s_{i+b}                       b = buffer (0 = the sentence alone)
e_i        = embed(window_i)                           unit vector, 384 dims
d_i        = 1 - cos(e_i, e_{i+1}) = 1 - e_i . e_{i+1}  distance between neighbour windows,  i = 0 .. n-2
cut after sentence i  if  d_i > threshold
```
**Breakpoint rules** (the names follow LangChain's `SemanticChunker` `breakpoint_threshold_type`):
```
percentile  threshold = the p-th percentile of all d     "cut at the top (100-p)% of distances"      default p=95
stdev       threshold = mean(d) + k * std(d)             "standard_deviation", default k=3
iqr         threshold = mean(d) + k * (Q3 - Q1)          "interquartile"
gradient    cut where np.gradient(d) is in the top p%    cut where the distance CHANGES fastest
```
The percentile rule **always cuts**: at p=85 about 15% of the sentence boundaries become cuts *whatever the text says*. A one-topic document is cut anyway. The stdev rule with `k=3` cuts very little on a short document (here exactly 1 cut; it makes none when no distance exceeds `mean + 3 std`). Gradient helps when all distances are high or all low, because it looks at change rather than level.

**Buffer.** `buffer=1` embeds `[s_{i-1}, s_i, s_{i+1}]` as one text. It smooths the curve and costs about 3x the tokens through the model (every token is read three times).

**Size guards.** After the cuts: remove the *weaker* of the two cuts around any chunk smaller than `min_tokens`; split any chunk larger than `max_tokens` at its largest internal distance.

**Cost.** The boundary search embeds every sentence window once; the final chunks must then be embedded *again* for the index. Recursive chunking calls no model during chunking.

**Evidence and status.** Semantic chunking is widely implemented (LangChain `SemanticChunker`, LlamaIndex `SemanticSplitterNodeParser`), but independent evaluations (for example a 2024 paper titled "Is Semantic Chunking Worth the Computational Cost?", which found no consistent gain) found it often does *not* beat fixed or sentence-based chunking on retrieval, because the chunks end up small and uneven. Treat it as an option to *measure on your data*, most useful for unstructured prose with real topic shifts. That is *current practice with mixed evidence*, not a default.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| semantic chunking | embedding-based chunking, topic segmentation, similarity-based splitting, `SemanticChunker`, `SemanticSplitterNodeParser` | near | All cut where neighbour similarity drops; thresholds and buffers differ |
| breakpoint | boundary, cut point, split point, topic shift | yes | Index between two sentences where a cut is made |
| buffer | `buffer_size`, window of neighbours, context window of sentences | yes | Sentences glued on each side before embedding |
| percentile threshold | `breakpoint_threshold_type="percentile"`, top-k% rule | yes | Cut at the largest (100-p)% distances |
| standard deviation threshold | `"standard_deviation"`, z-score rule | yes | Mean + k std |
| interquartile threshold | `"interquartile"`, IQR rule | yes | Mean + k * IQR |
| gradient threshold | `"gradient"`, change-point rule | near | Uses the slope of the distance curve |
| TextTiling | classical topic segmentation (Hearst, 1997) | no | Same idea with word-count vectors instead of embeddings |
| cosine distance | 1 - cosine similarity | yes | Zero for identical direction; for unit vectors `1 - a . b` |

## Run it
```bash
cd 19_chunking_deep_dive/04_semantic_chunking
python main.py
# from the repo root:
HF_HUB_OFFLINE=1 .venv/bin/python 19_chunking_deep_dive/04_semantic_chunking/main.py
```
About 20 to 30 s of CPU time (several thousand short sentence windows are embedded). Needs `torch` + `transformers` and the cached MiniLM; no downloads.

## Code, step by step

### 1. Header docstring and imports  (`main.py` lines 1–24)

```python
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
```

- **In simple words:** goal, run lines, and the shared helpers. `_CACHE` is imported so section C can clear it for an honest timing.
- **Exactly:** `sentence_spans` (sentence boundaries with offsets), `make_chunk` (slice + strip), `sparkline` (text bars) and `embed` (cached MiniLM vectors) are explained in the [step 01 tour](../01_tokens_not_characters/WALKTHROUGH.md#the-shared-helpers-once-a-tour-of-common19py).
- **Why it is here:** this file imports no new library; the "semantics" come from `embed`.

### 2. `neighbour_distances`: the distance curve  (`main.py` lines 27–32)

```python
def neighbour_distances(texts, buffer=1):
    """d[i] = 1 - cos(window around sentence i, window around sentence i+1). `buffer` sentences each side are
    glued to every sentence before embedding, so one short sentence ('Yes.') cannot cause a fake spike."""
    windows = [" ".join(texts[max(0, i - buffer): i + buffer + 1]) for i in range(len(texts))]
    E = embed(windows)                                       # (n_sentences, 384), unit vectors
    return 1.0 - (E[:-1] * E[1:]).sum(axis=1)                # (n_sentences - 1,)
```

- **In simple words:** for each sentence build a window of itself plus `buffer` neighbours on each side, embed all windows, and return the distance between each window and the next.
- **Exactly:** `texts[max(0, i - buffer): i + buffer + 1]` clamps at the start and end of the document. `embed(windows)` returns an `(n, 384)` matrix of unit vectors, so `E[:-1] * E[1:]` multiplies row `i` with row `i+1` element-wise and `.sum(axis=1)` makes the dot product (= cosine). `1.0 - ...` turns similarity into distance. The result has `n - 1` entries.
- **Why it is here:** the whole method is this function. Everything else decides where to cut on this curve.

### 3. `threshold` and `breakpoints`: the rules  (`main.py` lines 35–50)

```python
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
```

- **In simple words:** turn the curve into a set of cut positions with one of four rules.
- **Exactly:** `np.percentile(d, p)` is the value below which `p`% of distances fall. `d.mean() + k * d.std()` is the z-score rule. `np.gradient(d)` is the discrete slope of the curve (central differences); the gradient rule cuts where the slope is in its top `(100-p)`%. `np.where(d > t)[0]` lists the indices above the threshold; a cut at index `i` means "break *after* sentence `i`". An unknown rule name raises `ValueError(method)`.
- **Why it is here:** the four rules are the vocabulary you need for the interview; here they run on identical distances so the only difference is the rule.

### 4. `enforce_sizes`: min and max guards  (`main.py` lines 53–76)

```python
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
```

- **In simple words:** kill cuts that leave a tiny chunk and add cuts inside an oversize one.
- **Exactly:** `groups(cs)` converts cuts to inclusive sentence ranges `(a, b)` using sentinels at `-1` and the last sentence. In the first loop, a group with fewer than `min_tokens` tokens removes the **weaker** of its two neighbouring cuts (`min(near, key=lambda c: d[c])` is the smallest distance); the `while changed` loop repeats until nothing changes. In the second loop, a group above `max_tokens` is cut at `a + argmax(d[a:b])`, its biggest internal distance.
- **Why it is here:** raw breakpoints produce chunks of 5 and of 900 tokens; production splitters always add size guards.

### 5. `semantic_chunks`: the whole pipeline for one document  (`main.py` lines 79–88)

```python
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
```

- **In simple words:** sentences, then distances, then cuts, then size guards, then chunks.
- **Exactly:** documents with fewer than 3 sentences are returned as one chunk. `edges` pairs consecutive cuts into inclusive sentence ranges and `make_chunk` turns each range into a `RawChunk` with exact offsets.
- **Why it is here:** the function step 10 imports (`semantic_chunks(doc, method, param, buffer, ...)`); step 08 imports `sentence_pack`.

### 6. `sentence_pack`: the no-embedding baseline  (`main.py` lines 91–99)

```python
def sentence_pack(doc, max_tokens=100):
    """Baseline with the SAME sentence units but no embeddings: pack sentences up to a token budget."""
    out, cur = [], []
    for s, e in sentence_spans(doc["text"]):
        if cur and count_tokens(doc["text"][cur[0][0]:e]) > max_tokens:
            out.append(make_chunk(doc, cur[0][0], cur[-1][1]))
            cur = []
        cur.append((s, e))
    return out + ([make_chunk(doc, cur[0][0], cur[-1][1])] if cur else [])
```

- **In simple words:** the same sentences packed up to 100 tokens with no model at all.
- **Exactly:** the token count is recomputed on the span from the first to the current sentence's end. Returns chunks via `make_chunk`.
- **Why it is here:** the control experiment. If semantic chunking cannot beat *this*, the embeddings were not worth their cost.

### 7. Section A: the distance curve of the postmortem  (`main.py` lines 102–115)

```python
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
```

- **In simple words:** embed the postmortem sentences, draw the distance as a sparkline, mark the cuts under it with `^`, and print the first three cuts with the sentences on each side.
- **Exactly:** `sparkline(d)` maps each distance to one of `▁▂▃▄▅▆▇█`. The `^` row is aligned character by character, so a `^` sits under a tall bar. `breakpoints(d, "percentile", 85)` gives the cut set.
- **Why it is here:** you see where the model thinks the topics change, and whether that matches your reading.

### 8. Section B: the rules on identical distances  (`main.py` lines 117–126)

```python
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
```

- **In simple words:** count the cuts each rule makes, then compare the `stdev k=3` threshold with the largest distance.
- **Exactly:** each row is `(rule, number of cuts, number of chunks = cuts + 1)`; these counts are *before* the size guards. The last print compares the threshold with the maximum distance (`no cut at all` if the threshold is not below the maximum, else `some cuts`).
- **Why it is here:** "which threshold do I use?" depends on the document: a percentile always cuts, a z-score cuts only on outliers and may never cut.

### 9. Section C: what the boundary search costs  (`main.py` lines 128–143)

```python
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
```

- **In simple words:** count how many sentence-window embeddings and tokens the cut-finding needs and time it against recursive splitting.
- **Exactly:** `tok_win` is the number of tokens in all the 3-sentence windows (each sentence is read about three times at `buffer=1`). `_CACHE.clear()` empties the embedding cache so the semantic timing includes the model calls instead of re-using vectors from section A. `time.perf_counter()` is the wall-clock timer.
- **Why it is here:** the headline cost: a model forward pass for every sentence, then the final chunks must be embedded again to be indexed. Recursive chunking does neither.

### 10. Section D: retrieval comparison  (`main.py` lines 145–168)

```python
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
```

- **In simple words:** seven strategies on the 74 questions; the last column restricts to the questions about the headingless postmortem.
- **Exactly:** `add(name, fn)` builds chunks for all 13 documents, runs `evaluate`, and also computes hit@3 only for questions whose gold document is `incident_postmortem` (`x["id"].startswith("n")` selects the new questions; the second condition checks the document). `qs` is created just above `add` is first called (the late `from common19 import load_eval` placement works because `add` looks `qs` up when it runs).
- **Why it is here:** the honest verdict. Read the table before believing the method.

### 11. Entry point and the TRY IT / BREAK IT lines  (`main.py` lines 171–176)

```python
if __name__ == "__main__":
    main()

# TRY IT: change buffer=1 to 0 and 3 in section A; watch the sparkline get noisier / smoother.
# TRY IT: raise min_tokens to 80 in semantic_chunks; how many chunks remain at percentile 85?
# BREAK IT: threshold(d, "median", 50) -> ValueError: median (the rule name is not one of percentile / stdev / iqr).
```

- **In simple words:** guard and exercises.
- **Exactly:** `threshold(d, "median", 50)` raises `ValueError: median`.
- **Why it is here:** `ValueError(method)` with the bad name is the only validation; a typo in a rule name fails loudly, not silently.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `texts[max(0, i - buffer): i + buffer + 1]` | Clamped slice around index `i` |
| `E[:-1] * E[1:]` then `.sum(axis=1)` | Row-wise dot products of consecutive rows |
| `set(np.where(cond)[0].tolist())` | Indices where a boolean array is true, as a Python set |
| `min(near, key=lambda c: d[c])` | Item with the smallest `d[c]` |
| `"".join("^" if i in cuts else " " for i in range(n))` | Build the marker row with a generator |
| `[...]; _CACHE.clear(); t0 = ...` (semicolons) | Several short statements on one line |
| `lambda x: semantic_chunks(x, "percentile", 95)` | A strategy as a function of one document |

## Libraries & functions
| Function | What it does | Internally |
|---|---|---|
| `np.percentile(d, p)` | The p-th percentile | Sort-and-interpolate |
| `np.gradient(d)` | Discrete derivative | Central differences, one-sided at the ends |
| `np.argmax` | Index of the maximum | Linear scan |
| `embed(list)` | Unit vectors | Tokenise (truncate at 256), BERT forward pass, mean-pool, L2-normalise; cached per text |
| `time.perf_counter()` | High-resolution wall time | Monotonic clock |
| `evaluate` | Scores chunks | See the [step 01 tour](../01_tokens_not_characters/WALKTHROUGH.md#the-shared-helpers-once-a-tour-of-common19py) |

## Execution trace (real run)
Real output of `python main.py` (progress bar removed; timings vary by machine).
```

A) postmortem: 43 sentences, 42 neighbour distances: min 0.02 mean 0.24 max 0.73
   distance: ▁▃█▄▆▁▂▅▃▃▅▃▄▂▁▃▂▂▁▄▃▂▃▃▃▄▂▁▂▄▄▂▃▃▂▃▄▁▄▄▂▂
   cut here:   ^ ^  ^  ^                  ^        ^^     (7 cuts at the 85th percentile)
   cut after sentence 2: 'This document describes what happened, why it happ' | 'It is written without blame.'  d=0.73
   cut after sentence 4: 'The goal is to understand the system, not to find ' | 'The first alert fired at 08:12 UTC.'  d=0.47
   cut after sentence 7: 'The on-call engineer, Mira, acknowledged the page ' | 'At 08:31 she upgraded it to SEV1 because the s'  d=0.42

B) number of breakpoints per rule on the same postmortem distances (before min/max size guards)
rule           cuts  chunks
-------------  ----  ------
percentile 95     3       4
percentile 85     7       8
percentile 70    13      14
stdev 1.0         5       6
stdev 3.0         1       2
iqr 1.5           1       2
gradient 90       5       6
   stdev k=3 on a short document: threshold 0.63 vs max distance 0.73 -> some cuts

C) cost on 13 docs (4987 tokens, 375 sentences)
   recursive: 0 model calls, 77 ms | semantic: 375 sentence-window embeddings, ~14665 tokens through the model (buffer=1 reads every token ~3x), 17809 ms
   ...and the final chunks must still be embedded AGAIN for the index.

D) retrieval, 13 documents, 74 questions (last column: dense hit@3 on the headingless postmortem only)
chunker                    chunks  mean tok  dense h@1  dense h@3   MRR  hybrid h@3  postmortem h@3
-------------------------  ------  --------  ---------  ---------  ----  ----------  --------------
recursive 100 tok              68        73       0.74       0.91  0.83        0.95            0.82
sentence-pack 100 tok          60        83       0.66       0.92  0.80        0.95            0.91
semantic pct 95                52        96       0.72       0.88  0.81        0.91            0.73
semantic pct 85                61        82       0.72       0.89  0.81        0.88            0.73
semantic pct 70                81        62       0.73       0.92  0.83        0.92            0.73
semantic pct 85, buffer 0      64        78       0.77       0.95  0.85        0.96            0.91
semantic gradient 90           63        79       0.72       0.86  0.81        0.95            0.73
```

How to read it:
- **A:** 43 sentences give 42 distances (min 0.02, mean 0.24, max 0.73). The 85th-percentile rule makes 7 cuts. The biggest spike (0.73) is right after the first sentence of boilerplate ("This document describes ..." | "It is written without blame."): the model reacts to a change of *register*, not necessarily of topic. The cut after sentence 4 (0.47, into "The first alert fired at 08:12 UTC.") is a real topic shift from the introduction to the timeline.
- **B:** the rule decides how many chunks you get: 4 chunks (percentile 95), 8 (85), 14 (70), 6 (stdev 1.0), 2 (stdev 3.0 and IQR 1.5), 6 (gradient 90). The `stdev k=3` threshold (0.63) is below the maximum distance (0.73), so it makes exactly one cut here; on a flatter document it could make none.
- **C:** the boundary search embeds 375 sentence windows and pushes about 14,665 tokens through the model for a 4,987-token corpus (each token read about 3 times at `buffer=1`), 17.8 s against 77 ms for recursive chunking in this run (a busy shared laptop; the ratio is the point, not the seconds). The final chunks still have to be embedded again.
- **D:** semantic chunking does **not** beat the no-embedding baseline here. Sentence-pack at 100 tokens has dense hit@3 0.92 against 0.88, 0.89, 0.92 for semantic at percentile 95, 85, 70, and the gradient rule is lowest at 0.86. The best row, `semantic pct 85, buffer 0`, reaches hit@3 0.95 and hit@1 0.77, but the same method with `buffer=1` gets 0.89 and 0.72, so a smoothing parameter moves the result more than the idea does. On the headingless postmortem (11 questions, so one question is 0.09) sentence-pack and buffer 0 score 0.91, recursive 0.82, buffer-1 variants 0.73: one to two questions apart. Honest reading: semantic chunking is an option to test, not an upgrade to assume.

## Experiments (change ONE thing)
| Change | What you will see | Why |
|---|---|---|
| `buffer` 0, 1, 3 in section A | The sparkline is jagged at buffer 0, smoother at 1; at the 85th percentile all three give 7 cuts (measured) | The buffer averages away single-sentence noise; the percentile rule fixes the share of cuts |
| `min_tokens=80` in `semantic_chunks` | Fewer, larger chunks; weak cuts are removed first | `enforce_sizes` drops the weaker neighbouring cut |
| `("percentile", 95)` vs `("percentile", 70)` | Chunks get larger vs smaller | The percentile fixes the cut rate |
| Add a `"median"` rule (median + 3 * MAD) | Threshold 0.44 and 2 cuts on the postmortem (stdev k=1 gives 5); a robust rule that ignores the few huge spikes; [solutions](../solutions/solutions.md) | Median and median absolute deviation are not dragged by outliers |
| Run section A on `employee_handbook` instead of the postmortem | Compare where the `^` marks fall with where the headings are (run it and look) | The model sees heading words change, but nothing forces a cut at a heading |

## Debugging practice
**BREAK IT:** `threshold(d, "median", 50)`.
```
ValueError: median
```
The rule name is not one of `percentile`, `stdev`, `iqr`. The message is just the bad name because the code does `raise ValueError(method)`; a friendlier version lists the valid names. **Fix:** add the rule (median + k * MAD, see the solutions) or use an existing one. A *silent* bug of the same family: a `stdev` rule with a large `k` on a short document returns almost no cut (1 here, none if no distance is an outlier) and you get one giant chunk; always print the number of cuts and the chunk size distribution.

## Common mistakes
- Assuming the method "finds topics". It finds *places where embeddings change*, which also happens at a list, a code example or a change of register.
- Using a percentile rule on a one-topic document: it cuts anyway.
- Using `stdev k=3` on a short document: it cuts only on outliers, often once or never.
- Embedding single sentences with `buffer=0`: a sentence like "Yes." or "See above." produces a fake spike.
- Forgetting the min/max guards: you get 8-token chunks and 600-token chunks.
- Forgetting that the final chunks are embedded a second time for the index: the true ingestion cost is the boundary pass *plus* the index pass.
- Comparing only against recursive chunking. Add the sentence-pack baseline, or you cannot tell whether the semantics helped or just the sentence boundaries.

## Rebuild checklist
1. `neighbour_distances(texts, buffer)`: windows, `embed`, `1 - (E[:-1] * E[1:]).sum(1)`.
2. `threshold` for percentile / stdev / IQR; `breakpoints` with the gradient rule.
3. `enforce_sizes`: `groups`; drop the weaker cut around tiny groups; split oversize groups at `argmax` of `d`.
4. `semantic_chunks`: sentences, distances, tokens per sentence, cuts, edges, `make_chunk`.
5. `sentence_pack` baseline with the same sentences.
6. Time the cost with the cache cleared; compare on hit@1, hit@3, MRR and the headingless slice.

## Say it aloud
"Semantic chunking embeds each sentence, usually with a buffer of neighbours, takes the cosine distance between consecutive windows and cuts where it spikes, using a percentile, standard-deviation, IQR or gradient rule, then enforces minimum and maximum chunk sizes. It is the only method here that finds boundaries in text with no structure, but it costs a forward pass per sentence before the chunks are embedded again for the index, and a percentile rule cuts even a single-topic document. I always compare it to sentence packing at the same size, because in independent evaluations it often fails to beat it."

## The new problem (cliffhanger)
Every chunker so far embeds each chunk *alone*. A chunk that says "Each block must be at least five days." is a fragment: its vector does not know it is about secondary-caregiver leave, no matter how cleverly it was cut. What if the model read the whole passage first and *then* we took each chunk's part of the result? Next: [step 05, late chunking](../05_late_chunking/WALKTHROUGH.md).
