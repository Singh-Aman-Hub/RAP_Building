# Step 10 — Chunking evaluation

## The story so far
Steps [01](../01_tokens_not_characters/WALKTHROUGH.md) to [09](../09_layout_aware_chunking_from_parsed_docs/WALKTHROUGH.md) each produced a table from a different script. This step is the scoreboard: every strategy of steps 01 to 08 on the **same** 13 documents and the **same** 74 questions, with quality, cost and noise side by side. Back to the [module README](../README.md).

**Relation to module 14's evaluation.** [14/09 (RAG evaluation)](../../14_rag_and_retrieval/09_rag_evaluation/WALKTHROUGH.md) teaches the *metrics* (hit@k, recall@k, MRR, nDCG), golden sets, hard negatives, and the retrieval-versus-generation split, and scores a *fixed* chunking by chunk id. This step reuses those metrics but changes the unit under test: the *chunker*. Because the chunks differ between strategies, relevance is defined by **character offsets** (a chunk is relevant if its source span covers the answer sentence), not by chunk ids, and the table adds the things a chunker changes: tokens sent to the LLM, index size, ingestion time and LLM calls. Read 14/09 first if hit@k and MRR are new to you; here they are defined again in one line each.

## The pain point
"Semantic chunking is better" and "512 tokens with overlap works fine" are claims, not measurements. Three traps:
- **Unfair comparison:** a 40-token chunk and a 250-token section cannot be compared with chunk-id labels; and hit@3 rewards big chunks (they cover more text) while costing more prompt tokens.
- **Hidden costs:** LLM calls at ingestion, index size, ingestion time.
- **Noise:** with 74 questions one question moves a metric by 0.014; most differences between strategies are inside the noise, and people ship the noise.

## The spark
One harness and four habits:
1. Define relevance by offsets so any chunker is scorable.
2. Report quality **and** cost in one table (hit@k, MRR, recall@5, context tokens, context precision, index size, ingestion seconds, LLM calls).
3. Slice (by corpus and by question kind) because an average hides where a strategy fails.
4. Put a **bootstrap confidence interval** on the headline metric and only trust big gaps.

## Goal
Run 19 strategies through `evaluate` and print T1 (quality + cost, dense), T2 (hybrid and slices) and T3 (95% intervals and quality per context token), then decide how to choose.

## The theory (read this before the interview)
**Metrics (per question, then averaged over the questions):**
```
hit@k      1 if any top-k context fully covers any answer sentence      (a "success rate at k")
recall@k   share of the answer sentences covered by the union of the top-k contexts   (matters for multi-evidence questions)
MRR        1 / rank of the first covering context     (0 if none)
ctx tok@3  tokens of the top-3 contexts' context_text: what you pay to send to the LLM
ctx prec   share of the characters sent (top-3, same document) that are answer evidence: low = you pay for text that does not help
```
**Index and ingestion cost:** `chunks`, `stored tok` (tokens of `embed_text` that must be embedded and stored), `ingest s` (build + embedding time on this CPU), `LLM calls` (contextual retrieval, RAPTOR summaries, propositions, agentic chunking).

**The three retrievers.** `dense` (cosine over MiniLM vectors), `bm25` (lexical), `hybrid` (reciprocal-rank fusion of both, as in module 14). A chunking that wins on dense retrieval may lose on hybrid, because BM25 repairs some of what dense misses.

**Slices.** `corpus` (kb = the nine clean module-14 documents; new = four messy ones) and `kind` (table, code, reference, multi, keyword, paraphrase). Slices are *small* (a handful of questions); treat them as hypotheses to check, not conclusions.

**Bootstrap interval.** Resample the 74 per-question scores *with replacement* 1000 times, take the mean each time, and report the 2.5th and 97.5th percentiles: an approximate 95% interval for the metric if the questions were a random sample. If two strategies' intervals overlap heavily, the data cannot tell them apart. (This ignores that the *same* questions are used for both, so it is conservative for differences; a paired bootstrap on per-question differences is sharper.)

**Quality per context token.** `hit@3 / ctx tok x 100` rewards strategies that reach the answer with fewer prompt tokens. It favours small contexts, so look at it *alongside* quality, never alone.

**Choosing a strategy (procedure, not a name):**
1. Start from the structure-aware or recursive baseline with a breadcrumb at 150 to 250 tokens.
2. Write 50 to 200 questions *from your own documents*, including the ugly ones (tables, pronouns, multi-evidence), labelled by answer span.
3. Change one thing at a time; keep a change only if its gain exceeds the noise and its cost is acceptable.
4. Re-run when the embedding model, the document types or the parser change.

**Status.** An evaluation set built from your own data is *fundamental practice*; the specific strategies that win depend on corpus and model. Public chunking benchmarks exist but rarely match your documents.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| hit@k | recall@k (single-answer), success@k, top-k accuracy, HitRate@k | near | Any relevant context in the top k |
| recall@k | coverage@k | near | Fraction of answer pieces found; equals hit@k when there is one |
| MRR | mean reciprocal rank | yes | Average of 1/rank of the first relevant result |
| context precision | chunk attribution, evidence share, context utilisation (RAGAS-style names vary) | near | Share of sent text that is evidence; here computed from character overlap |
| golden set | eval set, test set, labelled questions, ground truth | yes | Questions with known answers |
| bootstrap CI | percentile bootstrap interval, resampling interval | yes | Interval from resampled means |
| ingestion cost | indexing cost, build cost, preprocessing cost | yes | Time, tokens and LLM calls spent before any query |
| offset-based relevance | span-level labels, character-span ground truth | near | Relevance defined by source text, so chunkers are comparable |
| paired bootstrap | paired resampling, per-question differences | near | Bootstrap the difference between two systems on the same questions |

## Run it
```bash
cd 19_chunking_deep_dive/10_chunking_evaluation
python main.py
# from the repo root:
HF_HUB_OFFLINE=1 .venv/bin/python 19_chunking_deep_dive/10_chunking_evaluation/main.py
```
About 1.5 minutes of CPU time when the machine is free (19 strategies, each embedded from scratch for fair timing). Progress goes to stderr. Needs `torch`, `transformers`, `scikit-learn` and the cached MiniLM; no downloads.

## Code, step by step

### 1. Header docstring, imports, loading every earlier step  (`main.py` lines 1–30)

```python
"""
GOAL:  One scoreboard for every chunking strategy of steps 01-08 on the SAME 74 labelled questions over the SAME
       13 documents: hit@k, MRR, recall@5, context tokens sent per query, context precision, index size
       (#chunks, stored tokens), ingestion time and LLM calls. Then: noise (bootstrap interval), slices
       (by question kind and by corpus), and how to pick.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 19_chunking_deep_dive/10_chunking_evaluation/main.py
EXPECTED OUTPUT (shortened):
    T1  quality + cost per strategy (dense retrieval)
    T2  hybrid (BM25 + dense, RRF) and slices by corpus / question kind
    T3  95% bootstrap interval on hit@3: which differences are real?
NEEDS: torch, transformers, scikit-learn (cached MiniLM).  ~90 s on a laptop CPU.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common19  # noqa: E402
from common19 import count_tokens, embed, evaluate, load_docs, load_eval, load_step, table  # noqa: E402

docs = load_docs()
questions = load_eval()
S01, S02, S03, S04 = (load_step(n) for n in ("01_tokens_not_characters", "02_recursive_splitter_rebuilt",
                                             "03_structure_aware_markdown", "04_semantic_chunking"))
S05, S06, S07, S08 = (load_step(n) for n in ("05_late_chunking", "06_contextual_retrieval",
                                             "07_hierarchical_parent_child", "08_propositions_and_agentic_chunking"))
```

- **In simple words:** load the corpus and questions once, and import the chunkers written in steps 01 to 08 as modules.
- **Exactly:** `load_step(name)` imports `<name>/main.py` by path ([step 01 tour](../01_tokens_not_characters/WALKTHROUGH.md#the-shared-helpers-once-a-tour-of-common19py)); each `S0n` module exposes the functions written there (`S01.fixed_token_windows`, `S02.RecursiveSplitter`, `S03.StructureChunker`, `S04.semantic_chunks`, `S05.build`, `S06.situate`, `S07.build_nodes`, `S08.proposition_chunks`). The module-level `docs` and `questions` are shared by every strategy. `import common19` (as well as `from common19 import ...`) is needed to reach the private `common19._CACHE` below.
- **Why it is here:** *no chunker is re-implemented*. The scoreboard measures the code you have already read.

### 2. Builders: `plain`, `late`, `contextual`  (`main.py` lines 33–53)

```python
def plain(fn):
    """Strategy that is just 'document -> chunks'."""
    return lambda: {"chunks": [c for d in docs for c in fn(d)]}


def late():
    chunks, vecs = [], []
    for d in docs:
        c, v = S05.build(d, 128)
        chunks += c
        vecs.append(v)
    return {"chunks": chunks, "vectors": np.vstack(vecs)}


def contextual():
    by_id = {d["doc_id"]: d for d in docs}
    base = [c for d in docs for c in S02.RecursiveSplitter(80, 0).split_document(d)]
    texts = [f"{S06.situate(by_id[c.doc_id]['text'], c.text)}\n{c.text}" for c in base]       # 1 LLM call per chunk
    for c, t in zip(base, texts):
        c.embed_text = t
    return {"chunks": base, "llm_calls": len(base), "bm25_texts": texts}
```

- **In simple words:** each strategy is a function that returns a small dict: the chunks and, optionally, special vectors, extra BM25 text, LLM call count or a postprocess.
- **Exactly:** `plain(fn)` wraps "document to chunks". `late()` collects chunks and late vectors from `S05.build(d, 128)` and stacks them (`np.vstack`). `contextual()` makes 80-token recursive chunks, calls the (mock) LLM once per chunk, sets `embed_text` to `context + text`, and returns the BM25 texts and `llm_calls = len(base)`.
- **Why it is here:** the dict contract (`chunks`, `vectors`, `bm25_texts`, `postprocess`, `llm_calls`) lets *one* loop evaluate strategies with very different mechanics.

### 3. Builders: `hierarchical`, `llm_based`  (`main.py` lines 56–85)

```python
def hierarchical(kind):
    def build():
        nodes = S07.build_nodes(docs)
        leaves = [S07.as_chunk(n) for n in nodes.values() if n.level == 0]
        if kind == "s2b_mid":
            return {"chunks": leaves, "postprocess": S07.small_to_big(leaves, nodes, 1)}
        if kind == "s2b_sec":
            return {"chunks": leaves, "postprocess": S07.small_to_big(leaves, nodes, 2)}
        if kind == "auto":
            return {"chunks": leaves, "postprocess": S07.auto_merge(leaves, nodes)}
        if kind == "window":
            sc = [common19.RawChunk(d["doc_id"], s, e, d["text"][s:e]) for d in docs for s, e in common19.sentence_spans(d["text"])]
            return {"chunks": sc, "postprocess": S07.sentence_window(sc, 2)}
        if kind == "raptor":
            summ, calls = [], 0
            for d in docs:
                s, k = S07.build_raptor(d, [c for c in leaves if c.doc_id == d["doc_id"]])
                summ += s
                calls += k
            return {"chunks": leaves + summ, "llm_calls": calls}
    return build


def llm_based(fn):
    def build():
        stats, chunks = {"calls": 0, "in_tok": 0, "out_tok": 0}, []
        for d in docs:
            chunks += fn(d, stats)
        return {"chunks": chunks, "llm_calls": stats["calls"]}
    return build
```

- **In simple words:** the step-07 variants and the step-08 LLM-based chunkers as builders.
- **Exactly:** `hierarchical(kind)` returns a function that builds the nodes and leaves and returns the matching postprocess (`small_to_big`, `auto_merge`, `sentence_window`) or, for RAPTOR, leaves plus summaries with the number of summariser calls. `llm_based(fn)` runs `fn(doc, stats)` for every document with a shared `stats` dict and reports `calls`.
- **Why it is here:** the same dict contract; the `postprocess` hook is where parents and merged contexts enter the evaluation.

### 4. `STRATEGIES`: the 19 contestants  (`main.py` lines 88–108)

```python
STRATEGIES = [
    ("fixed 128 tok, no overlap", plain(lambda d: S01.fixed_token_windows(d, 128, 0))),
    ("fixed 128 tok, overlap 19", plain(lambda d: S01.fixed_token_windows(d, 128, 19))),
    ("fixed 256 tok, overlap 38", plain(lambda d: S01.fixed_token_windows(d, 256, 38))),
    ("recursive 100 tok", plain(lambda d: S02.RecursiveSplitter(100, 0).split_document(d))),
    ("recursive 100, markdown seps", plain(lambda d: S02.RecursiveSplitter(100, 0, separators=S02.MARKDOWN).split_document(d))),
    ("sentence-pack 100", plain(lambda d: S04.sentence_pack(d, 100))),
    ("structure 200 + breadcrumb", plain(lambda d: S03.StructureChunker(200, 40).split_document(d))),
    ("structure 100 + breadcrumb", plain(lambda d: S03.StructureChunker(100, 30).split_document(d))),
    ("semantic pct85 (buffer 1)", plain(lambda d: S04.semantic_chunks(d, "percentile", 85))),
    ("semantic pct85 (buffer 0)", plain(lambda d: S04.semantic_chunks(d, "percentile", 85, buffer=0))),
    ("late chunking (128 win)", late),
    ("contextual (mock LLM)", contextual),
    ("small-to-big -> mid", hierarchical("s2b_mid")),
    ("small-to-big -> section", hierarchical("s2b_sec")),
    ("sentence window +-2", hierarchical("window")),
    ("auto-merging", hierarchical("auto")),
    ("RAPTOR tree (mock)", hierarchical("raptor")),
    ("propositions (mock)", llm_based(lambda d, s: S08.proposition_chunks(d, s))),
    ("agentic (mock)", llm_based(S08.agentic_chunks)),
]
```

- **In simple words:** a list of `(name, builder)` pairs, the whole scoreboard.
- **Exactly:** fixed windows (3), recursive (2), sentence-pack, structure-aware (2), semantic (2), late chunking, contextual (mock), small-to-big (2), sentence window, auto-merging, RAPTOR (mock), propositions (mock), agentic (mock). `lambda d: ...` captures the parameters of each.
- **Why it is here:** adding a strategy to the scoreboard is one line, which is the experiment in TRY IT.

### 5. `bootstrap`  (`main.py` lines 111–115)

```python
def bootstrap(values, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.array(values)
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])
```

- **In simple words:** estimate how much a mean would wobble if you had asked different questions.
- **Exactly:** `rng.integers(0, len(v), len(v))` draws `len(v)` random indices *with replacement*; `v[idx].mean()` is one resampled mean; 1000 of them form the distribution; `np.percentile(means, [2.5, 97.5])` returns the interval. `seed=0` makes it reproducible.
- **Why it is here:** the difference between "0.89 vs 0.91" and a real effect.

### 6. `main`, part 1: run every strategy  (`main.py` lines 118–136)

```python
def main():
    results = {}
    embed(["warm-up so the first strategy does not pay for loading the model"])
    for name, builder in STRATEGIES:
        common19._CACHE.clear()                                  # fair ingestion timing: no embeddings reused across strategies
        t0 = time.perf_counter()
        spec = builder()
        t_build = time.perf_counter() - t0
        chunks = spec["chunks"]
        t0 = time.perf_counter()
        vectors = spec.get("vectors")
        if vectors is None:
            vectors = embed([c.embed_text for c in chunks])      # the embedding pass for the index
        t_embed = time.perf_counter() - t0
        r = evaluate(chunks, questions, docs, vectors=vectors, bm25_texts=spec.get("bm25_texts"), postprocess=spec.get("postprocess"))
        r["ingest_s"] = t_build + t_embed
        r["llm_calls"] = spec.get("llm_calls", 0)
        results[name] = r
        print(f"  ... {name:<30} {len(chunks):>4} chunks, {r['dense']['hit@3']:.2f} hit@3", file=sys.stderr)
```

- **In simple words:** for each strategy time the build, time the embedding pass, evaluate, and keep the results.
- **Exactly:** one warm-up `embed([...])` loads the model so the first strategy does not pay for it. **`common19._CACHE.clear()`** empties the shared embedding cache before every strategy, so each pays for its own embeddings: this is what makes `ingest s` comparable. `spec.get("vectors")` uses late-chunking vectors when present, otherwise `embed([c.embed_text ...])`. `evaluate(chunks, questions, docs, vectors=..., bm25_texts=..., postprocess=...)` is the harness from the [step 01 tour](../01_tokens_not_characters/WALKTHROUGH.md#the-shared-helpers-once-a-tour-of-common19py). The progress line goes to `stderr`, so `stdout` stays a clean table.
- **Why it is here:** the loop is where "same questions, same documents, same harness" is enforced.

### 7. T1: quality and cost, dense retrieval  (`main.py` lines 138–144)

```python
    print("T1) retrieval quality and cost, DENSE retrieval, 74 questions over 13 documents")
    rows = []
    for name, r in results.items():
        d = r["dense"]
        rows.append([name, r["n_chunks"], r["stored_tok"], f"{r['ingest_s']:.1f}", r["llm_calls"], f"{d['hit@1']:.2f}", f"{d['hit@3']:.2f}",
                     f"{d['mrr']:.2f}", f"{d['recall@5']:.2f}", f"{d['ctx_tok']:.0f}", f"{d['prec3']:.2f}"])
    print(table(rows, ["strategy", "chunks", "stored tok", "ingest s", "LLM calls", "hit@1", "hit@3", "MRR", "recall@5", "ctx tok@3", "ctx prec"]))
```

- **In simple words:** one row per strategy: chunks, stored tokens, ingestion seconds, LLM calls, hit@1, hit@3, MRR, recall@5, context tokens, context precision.
- **Exactly:** `d = r["dense"]`; `r["n_chunks"]`, `r["stored_tok"]` and `r["ingest_s"]` are index and cost numbers; `table(rows, headers)` right-aligns the numbers.
- **Why it is here:** quality and cost in the same row is the whole message.

### 8. T2: hybrid retrieval and slices  (`main.py` lines 146–158)

```python
    print("\nT2) hybrid retrieval (BM25 + dense, RRF) and dense hit@3 by slice")
    rows = []
    for name, r in results.items():
        rr = r["dense"]["rows"]
        sl = lambda f: np.mean([x["hit@3"] for x in rr if f(x)]) if any(f(x) for x in rr) else float("nan")
        rows.append([name, f"{r['hybrid']['hit@1']:.2f}", f"{r['hybrid']['hit@3']:.2f}", f"{r['hybrid']['mrr']:.2f}",
                     f"{sl(lambda x: x['corpus'] == 'kb'):.2f}", f"{sl(lambda x: x['corpus'] == 'new'):.2f}",
                     f"{sl(lambda x: x['kind'] == 'table'):.2f}", f"{sl(lambda x: x['kind'] == 'code'):.2f}",
                     f"{sl(lambda x: x['kind'] == 'reference'):.2f}", f"{sl(lambda x: x['kind'] == 'multi'):.2f}"])
    print(table(rows, ["strategy", "hyb h@1", "hyb h@3", "hyb MRR", "kb", "new", "table", "code", "reference", "multi"]))
    print(f"   slice sizes: kb={sum(q['corpus'] == 'kb' for q in questions)}, new={sum(q['corpus'] == 'new' for q in questions)}, "
          f"table={sum(q['kind'] == 'table' for q in questions)}, code={sum(q['kind'] == 'code' for q in questions)}, "
          f"reference={sum(q['kind'] == 'reference' for q in questions)}, multi={sum(q['kind'] == 'multi' for q in questions)}")
```

- **In simple words:** hybrid hit@1, hit@3 and MRR, then dense hit@3 for each slice (clean corpus, messy corpus, tables, code, reference, multi-evidence), and the number of questions in each slice.
- **Exactly:** `sl(f)` averages `hit@3` over the rows satisfying the predicate `f` (NaN if the slice is empty). The last print shows the slice sizes so you can see which numbers are based on very few questions.
- **Why it is here:** averages hide failure modes: a strategy can be fine overall and fail on tables.

### 9. T3: noise and quality per context token  (`main.py` lines 160–170)

```python
    print("\nT3) how big is the noise? 95% bootstrap interval of dense hit@3 (74 questions)")
    rows = []
    for name, r in results.items():
        h = [x["hit@3"] for x in r["dense"]["rows"]]
        lo, hi = bootstrap(h)
        rows.append([name, f"{np.mean(h):.2f}", f"[{lo:.2f}, {hi:.2f}]", f"{r['dense']['hit@3'] / r['dense']['ctx_tok'] * 100:.2f}"])
    print(table(rows, ["strategy", "hit@3", "95% CI", "hit@3 per 100 ctx tokens"]))
    best = max(results, key=lambda n: results[n]["dense"]["hit@3"])
    cheap = max(results, key=lambda n: results[n]["dense"]["hit@3"] / results[n]["dense"]["ctx_tok"])
    print(f"   best hit@3: {best} ({results[best]['dense']['hit@3']:.2f}); best quality per context token: {cheap}")
    print("   intervals overlap almost everywhere: with 74 questions only large gaps (>~0.10) are evidence.")
```

- **In simple words:** the 95% interval of dense hit@3 for each strategy and hit@3 per 100 context tokens; name the best strategy and the best per token.
- **Exactly:** `h` is the list of 74 per-question hit@3 values; `bootstrap(h)` gives `lo, hi`; `r['dense']['hit@3'] / r['dense']['ctx_tok'] * 100` is the efficiency. `max(results, key=...)` finds the winner.
- **Why it is here:** the last printed line says it: with 74 questions only large gaps are evidence.

### 10. Entry point and the TRY IT / BREAK IT lines  (`main.py` lines 173–178)

```python
if __name__ == "__main__":
    main()

# TRY IT: add a strategy "recursive 60 tok" and "recursive 200 tok" to STRATEGIES; where does the quality / ctx-token curve peak?
# TRY IT: remove the 'common19._CACHE.clear()' line: ingestion times of later strategies shrink. Why is that unfair?
# BREAK IT: pass vectors from a different chunk list (len mismatch) to evaluate -> IndexError in dense_search.
```

- **In simple words:** guard plus exercises.
- **Exactly:** vectors of the wrong length make `dense_search` index out of range.
- **Why it is here:** `evaluate(vectors=...)` trusts its caller; a shape check would be a good first improvement.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `STRATEGIES = [(name, builder), ...]` | Data-driven loop: a list of named functions |
| `def plain(fn): return lambda: {...}` | A function that returns a function (a factory) |
| `spec.get("vectors")` | Dict lookup with default `None` |
| `print(..., file=sys.stderr)` | Progress on the error stream; `stdout` stays clean |
| `sl = lambda f: np.mean([...]) if any(...) else float("nan")` | Slice mean with an empty-slice guard |
| `rng.integers(0, n, n)` | Random indices with replacement |
| `max(results, key=lambda n: ...)` | Best item by a computed score |
| `S01, S02, S03, S04 = (load_step(n) for n in (...))` | Unpack a generator of modules |

## Libraries & functions
| Function | What it does | Internally |
|---|---|---|
| `np.random.default_rng(0)` | Seeded random generator | PCG64 bit generator |
| `np.percentile(a, [2.5, 97.5])` | Interval of a distribution | Sort-and-interpolate |
| `time.perf_counter()` | Timer | Monotonic high-resolution clock |
| `evaluate` | Score chunks | Embeds, builds BM25, runs 74 questions, three retrievers, metrics |
| `common19._CACHE.clear()` | Forget cached embeddings | Empties the text-to-vector dict, so each strategy re-embeds |

## Execution trace (real run)
Real output of `python main.py` (stderr progress lines and the model loader bar removed; ingestion seconds depend on the machine and on other programs running, here the shared laptop).
```
T1) retrieval quality and cost, DENSE retrieval, 74 questions over 13 documents
strategy                      chunks  stored tok  ingest s  LLM calls  hit@1  hit@3   MRR  recall@5  ctx tok@3  ctx prec
----------------------------  ------  ----------  --------  ---------  -----  -----  ----  --------  ---------  --------
fixed 128 tok, no overlap         48        4987       9.2          0   0.72   0.89  0.81      0.91        337      0.02
fixed 128 tok, overlap 19         51        5710       8.3          0   0.68   0.88  0.79      0.96        341      0.02
fixed 256 tok, overlap 38         26        5481       5.5          0   0.70   0.95  0.82      0.96        657      0.01
recursive 100 tok                 68        4987       8.8          0   0.74   0.91  0.83      0.95        223      0.03
recursive 100, markdown seps      71        4987      13.0          0   0.78   0.92  0.86      0.96        214      0.04
sentence-pack 100                 60        4987       5.0          0   0.66   0.92  0.80      0.97        250      0.03
structure 200 + breadcrumb        56        5328      13.6          0   0.78   0.91  0.86      0.96        315      0.02
structure 100 + breadcrumb        92        6098      15.7          0   0.82   0.91  0.88      0.93        199      0.05
semantic pct85 (buffer 1)         61        4987      31.4          0   0.72   0.89  0.81      0.96        264      0.03
semantic pct85 (buffer 0)         64        4987      54.6          0   0.77   0.95  0.85      0.95        264      0.03
late chunking (128 win)          165        4987      55.1          0   0.58   0.85  0.73      0.91         91      0.07
contextual (mock LLM)             87        8321      11.9         87   0.69   0.86  0.79      0.89        176      0.04
small-to-big -> mid              156        4515      11.3          0   0.86   0.93  0.91      0.96        197      0.04
small-to-big -> section          156        4515      18.4          0   0.91   0.95  0.93      0.97        329      0.03
sentence window +-2              375        4987      52.3          0   0.86   0.95  0.91      0.96        197      0.05
auto-merging                     156        4515      13.8          0   0.88   0.93  0.92      0.96        137      0.06
RAPTOR tree (mock)               217        9195      77.0         61   0.78   0.88  0.85      0.92        125      0.08
propositions (mock)              303        4594       3.4        140   0.62   0.84  0.74      0.91         54      0.13
agentic (mock)                   224        4987       7.0        362   0.73   0.89  0.82      0.93         94      0.09

T2) hybrid retrieval (BM25 + dense, RRF) and dense hit@3 by slice
strategy                      hyb h@1  hyb h@3  hyb MRR    kb   new  table  code  reference  multi
----------------------------  -------  -------  -------  ----  ----  -----  ----  ---------  -----
fixed 128 tok, no overlap        0.80     0.95     0.87  0.88  0.90   0.75  1.00       1.00   1.00
fixed 128 tok, overlap 19        0.70     0.92     0.82  0.88  0.88   0.75  1.00       0.75   1.00
fixed 256 tok, overlap 38        0.78     0.93     0.87  0.88  0.98   1.00  1.00       1.00   1.00
recursive 100 tok                0.76     0.95     0.85  0.92  0.90   0.75  1.00       0.75   1.00
recursive 100, markdown seps     0.80     0.96     0.87  0.92  0.92   0.75  1.00       1.00   1.00
sentence-pack 100                0.72     0.95     0.83  0.88  0.94   1.00  1.00       1.00   1.00
structure 200 + breadcrumb       0.77     0.96     0.86  0.92  0.90   1.00  1.00       1.00   1.00
structure 100 + breadcrumb       0.81     0.96     0.88  0.92  0.90   0.75  1.00       0.75   1.00
semantic pct85 (buffer 1)        0.72     0.88     0.81  0.83  0.92   1.00  1.00       1.00   1.00
semantic pct85 (buffer 0)        0.80     0.96     0.88  0.92  0.96   1.00  1.00       1.00   1.00
late chunking (128 win)          0.68     0.86     0.78  0.79  0.88   0.75  0.50       0.75   1.00
contextual (mock LLM)            0.65     0.91     0.78  0.83  0.88   0.75  1.00       0.75   1.00
small-to-big -> mid              0.77     0.91     0.85  0.88  0.96   0.75  1.00       1.00   1.00
small-to-big -> section          0.84     0.95     0.90  0.88  0.98   1.00  1.00       1.00   1.00
sentence window +-2              0.72     0.95     0.84  0.96  0.94   0.75  0.50       1.00   1.00
auto-merging                     0.80     0.89     0.86  0.88  0.96   0.75  1.00       1.00   1.00
RAPTOR tree (mock)               0.74     0.85     0.81  0.88  0.88   0.25  0.50       1.00   1.00
propositions (mock)              0.54     0.77     0.68  0.79  0.86   1.00  0.50       0.50   1.00
agentic (mock)                   0.64     0.82     0.76  0.92  0.88   0.50  0.50       1.00   1.00
   slice sizes: kb=24, new=50, table=4, code=2, reference=4, multi=1

T3) how big is the noise? 95% bootstrap interval of dense hit@3 (74 questions)
strategy                      hit@3        95% CI  hit@3 per 100 ctx tokens
----------------------------  -----  ------------  ------------------------
fixed 128 tok, no overlap      0.89  [0.82, 0.96]                      0.26
fixed 128 tok, overlap 19      0.88  [0.80, 0.95]                      0.26
fixed 256 tok, overlap 38      0.95  [0.89, 0.99]                      0.14
recursive 100 tok              0.91  [0.84, 0.96]                      0.41
recursive 100, markdown seps   0.92  [0.85, 0.97]                      0.43
sentence-pack 100              0.92  [0.85, 0.97]                      0.37
structure 200 + breadcrumb     0.91  [0.82, 0.96]                      0.29
structure 100 + breadcrumb     0.91  [0.84, 0.96]                      0.45
semantic pct85 (buffer 1)      0.89  [0.81, 0.96]                      0.34
semantic pct85 (buffer 0)      0.95  [0.88, 0.99]                      0.36
late chunking (128 win)        0.85  [0.77, 0.93]                      0.93
contextual (mock LLM)          0.86  [0.78, 0.93]                      0.49
small-to-big -> mid            0.93  [0.86, 0.99]                      0.47
small-to-big -> section        0.95  [0.89, 0.99]                      0.29
sentence window +-2            0.95  [0.88, 0.99]                      0.48
auto-merging                   0.93  [0.86, 0.99]                      0.68
RAPTOR tree (mock)             0.88  [0.80, 0.95]                      0.70
propositions (mock)            0.84  [0.76, 0.92]                      1.54
agentic (mock)                 0.89  [0.81, 0.96]                      0.95
   best hit@3: fixed 256 tok, overlap 38 (0.95); best quality per context token: propositions (mock)
   intervals overlap almost everywhere: with 74 questions only large gaps (>~0.10) are evidence.
```

How to read it (ingestion seconds in this run are inflated and noisy because other jobs were using the laptop; use them only for the ordering of orders of magnitude):
- **T1, quality:** the hierarchical family leads on hit@1 and MRR: small-to-big to the section 0.91 / 0.93, auto-merging 0.88 / 0.92, sentence window and small-to-big to the mid 0.86 / 0.91. Next come structure-aware 100 + breadcrumb (0.82 / 0.88), recursive with Markdown separators and structure 200 (0.78 / 0.86). At the bottom: late chunking with MiniLM (0.58 / 0.73), propositions (0.62 / 0.74) and the mock contextual run (0.69 / 0.79).
- **T1, cost:** the winners cost different things. Small-to-big to the section sends 329 context tokens for the top 3, auto-merging 137, structure 100 + breadcrumb only 199, while fixed 256 sends 657 and gets the same hit@3 (0.95) by brute force. LLM-based methods are the expensive ones: 87 calls for contextual (and 67% more stored tokens: 8,321), 61 for RAPTOR (9,195), 140 for propositions, 362 for agentic. Semantic chunking and late chunking cost about 4 to 6 times the ingestion time of recursive in this run. `ctx prec` (share of sent characters that are evidence) is tiny for everyone (0.01 to 0.13) because each answer is one sentence in a context of 100 to 650 tokens; use it to *compare* strategies, not as an absolute.
- **T2:** hybrid retrieval lifts nearly every row to 0.95 / 0.96 hit@3 and the weak rows (late 0.86, propositions 0.77, RAPTOR 0.85) stay weak. The slices are tiny (table 4 questions, code 2, reference 4, multi 1): every slice number is a hypothesis. For example the RAPTOR collapsed tree gets 0.25 on table questions, which is one question out of four.
- **T3:** the 95% intervals of dense hit@3 are about plus or minus 0.07 wide and overlap for almost every pair: 0.95 [0.89, 0.99] for the best against 0.89 [0.82, 0.96] for fixed 128. Only the largest gaps in this table (for example 0.95 against 0.84 for propositions) are real evidence, and hit@1 differences such as 0.91 against 0.74 (small-to-big section against recursive 100) are better supported than any hit@3 gap. The `hit@3 per 100 context tokens` column crowns propositions (1.54) because their contexts are tiny, which shows why it must never be read alone: they have the lowest quality.
- **Answer to "which one?":** on this corpus, structure-aware or recursive chunking plus a breadcrumb as the baseline, and **small-to-big or auto-merging** as the first upgrade. LLM-based and embedding-based chunkers did not earn their ingestion cost here.

## Experiments (change ONE thing)
| Change | What you will see | Why |
|---|---|---|
| Add `recursive 60` and `recursive 200` to `STRATEGIES` | hit@1 peaks at 60 tokens (0.81), hit@3 plateaus at 0.91 from 100 tokens; context tokens grow from 84 (40) to 662 (300) and context precision falls from 0.08 to 0.01 (measured) | Small chunks are precise but fragmented, large chunks hit more but cost more; [solutions](../solutions/solutions.md) |
| Bootstrap hit@1 at 95% instead of hit@3 | A different interval width: hit@1 has more room to move | Variance depends on the metric |
| Remove `common19._CACHE.clear()` | Ingestion seconds of later strategies shrink (their texts were embedded already) | Unfair: the first strategy pays for embeddings that later ones reuse |
| Set `STRATEGIES` to two items and run twice | Identical quality numbers, different seconds | Quality is deterministic here; timings are not |
| Change the questions to your own 20 | Different winners | The ranking is a property of the corpus + questions + model |

## Debugging practice
**BREAK IT:** pass `vectors` computed for a different chunk list (wrong length) to `evaluate`.

```
IndexError: ...  (raised inside dense_search / the coverage check)
```
`matrix` has fewer rows than `chunks`, so a ranked index points past the end of `chunks`. **Fix:** assert `vectors.shape[0] == len(chunks)` at the top of `evaluate`. A worse variant is *same length, wrong order*: no error, quietly wrong numbers; guard it by building vectors from the chunk list in the same function.

## Common mistakes
- Comparing chunkers with chunk-id labels (impossible when the chunks differ) or by eyeballing examples.
- Reporting hit@k without the context-token cost: bigger chunks look better.
- Treating a 0.02 difference on 74 questions as a result.
- Averaging over question types and missing a collapse on tables or pronouns.
- Letting a cache or a warm model flatter one strategy's ingestion time.
- Evaluating with the same questions used to tune parameters (overfitting); hold out a part.
- Using LLM-generated questions without checking them: they echo the chunk's wording and flatter lexical overlap.
- Ignoring generation: a chunking can raise hit@k and still lower answer quality if the context is noisy (see [14/09](../../14_rag_and_retrieval/09_rag_evaluation/WALKTHROUGH.md)).

## Rebuild checklist
1. Define relevance by offsets: `covers(chunk, doc, span)` and answer spans per question.
2. `char_precision(ctxs, answer_spans, doc_id)`: merge spans, intersect with gold, divide by sent characters (rebuild this one from the step-01 tour).
3. A builder contract: `{chunks, vectors?, bm25_texts?, postprocess?, llm_calls?}`.
4. The loop with `_CACHE.clear()`, a build timer and an embedding timer.
5. T1, T2 (slices), T3 (bootstrap).
6. Name the best strategy by quality and by quality per context token; state the noise level.

## Say it aloud
"I evaluate chunkers on the same documents and questions, with relevance defined by character offsets so a 40-token chunk and a section are comparable, and I report hit@k, MRR and recall next to what the chunking costs: tokens sent to the LLM, index size, ingestion time and LLM calls. hit@k alone is misleading because bigger chunks cover more text. I slice by question type, and I put a bootstrap interval on the headline metric: with 74 questions most differences are inside the noise, so I only trust big gaps and re-run on my own labelled questions before changing production."

## The new problem (cliffhanger)
We have a measured shortlist, and one more problem that no experiment solves: **production**. Chunk ids that change when a sentence is edited; an index that must be rebuilt when the chunker is fixed, without downtime; a document too large for memory; Chinese text with no spaces; a team that wants to add a fourth chunker without touching the other three. Next: [step 11, the chunker as a production design](../11_chunker_lld/WALKTHROUGH.md).
