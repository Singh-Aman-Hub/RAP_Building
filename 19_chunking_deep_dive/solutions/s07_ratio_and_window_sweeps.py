"""Step 07 exercises: auto-merging ratio and sentence-window size sweeps.
RUN: HF_HUB_OFFLINE=1 ../../.venv/bin/python s07_ratio_and_window_sweeps.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common19 import RawChunk, evaluate, load_docs, load_eval, load_step, sentence_spans  # noqa: E402

s7 = load_step("07_hierarchical_parent_child")
docs, qs = load_docs(), load_eval()
nodes = s7.build_nodes(docs)
leaves = [s7.as_chunk(n) for n in nodes.values() if n.level == 0]
for ratio in (0.3, 0.5, 0.8):
    d = evaluate(leaves, qs, docs, postprocess=s7.auto_merge(leaves, nodes, ratio=ratio))["dense"]
    print(f"auto-merge ratio {ratio}: hit@1 {d['hit@1']:.2f} hit@3 {d['hit@3']:.2f} ctx tok@3 {d['ctx_tok']:.0f}")
sc = [RawChunk(d["doc_id"], s, e, d["text"][s:e]) for d in docs for s, e in sentence_spans(d["text"])]
for w in (0, 2, 5):
    d = evaluate(sc, qs, docs, postprocess=s7.sentence_window(sc, w))["dense"]
    print(f"sentence window w={w}: hit@1 {d['hit@1']:.2f} hit@3 {d['hit@3']:.2f} ctx tok@3 {d['ctx_tok']:.0f}")
