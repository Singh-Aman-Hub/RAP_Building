"""Step 10 exercise: where does recursive chunking peak? sizes 40 .. 300 tokens.
RUN: HF_HUB_OFFLINE=1 ../../.venv/bin/python s10_add_recursive_sizes.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common19 import evaluate, load_docs, load_eval, load_step, table  # noqa: E402

s2 = load_step("02_recursive_splitter_rebuilt")
docs, qs = load_docs(), load_eval()
rows = []
for size in (40, 60, 100, 200, 300):
    chunks = [c for d in docs for c in s2.RecursiveSplitter(size, 0).split_document(d)]
    r = evaluate(chunks, qs, docs)["dense"]
    rows.append([size, len(chunks), f"{r['hit@1']:.2f}", f"{r['hit@3']:.2f}", f"{r['mrr']:.2f}", f"{r['ctx_tok']:.0f}", f"{r['prec3']:.2f}"])
print(table(rows, ["size", "chunks", "hit@1", "hit@3", "MRR", "ctx tok@3", "ctx precision"]))
