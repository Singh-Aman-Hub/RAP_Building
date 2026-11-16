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


def bootstrap(values, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.array(values)
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


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

    print("T1) retrieval quality and cost, DENSE retrieval, 74 questions over 13 documents")
    rows = []
    for name, r in results.items():
        d = r["dense"]
        rows.append([name, r["n_chunks"], r["stored_tok"], f"{r['ingest_s']:.1f}", r["llm_calls"], f"{d['hit@1']:.2f}", f"{d['hit@3']:.2f}",
                     f"{d['mrr']:.2f}", f"{d['recall@5']:.2f}", f"{d['ctx_tok']:.0f}", f"{d['prec3']:.2f}"])
    print(table(rows, ["strategy", "chunks", "stored tok", "ingest s", "LLM calls", "hit@1", "hit@3", "MRR", "recall@5", "ctx tok@3", "ctx prec"]))

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


if __name__ == "__main__":
    main()

# TRY IT: add a strategy "recursive 60 tok" and "recursive 200 tok" to STRATEGIES; where does the quality / ctx-token curve peak?
# TRY IT: remove the 'common19._CACHE.clear()' line: ingestion times of later strategies shrink. Why is that unfair?
# BREAK IT: pass vectors from a different chunk list (len mismatch) to evaluate -> IndexError in dense_search.
