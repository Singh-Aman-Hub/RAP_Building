"""Step 03 exercises: Errors table at max 60 tokens (is the header repeated?) and the breadcrumb ablation on reference questions.
RUN: HF_HUB_OFFLINE=1 ../../.venv/bin/python s03_breadcrumb_and_table.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common19 import evaluate, load_docs, load_eval, load_step  # noqa: E402

s3 = load_step("03_structure_aware_markdown")
docs, qs = load_docs(), load_eval()
api = next(d for d in docs if d["doc_id"] == "api_reference")
pieces = [c for c in s3.StructureChunker(60, 10).split_document(api) if "Errors" in c.section_path and "table" in c.meta["kinds"]]
print(f"Errors table -> {len(pieces)} chunks; every chunk starts with the header row: {all(c.text.startswith('| HTTP status') for c in pieces)}")
for bc in (True, False):
    r = evaluate([c for d in docs for c in s3.StructureChunker(200, 40, breadcrumb=bc).split_document(d)], qs, docs)
    ref = [x for x in r["dense"]["rows"] if x["kind"] == "reference"]
    print(f"breadcrumb={bc}: overall hit@1 {r['dense']['hit@1']:.2f}; reference questions (n={len(ref)}) hit@1 {np.mean([x['hit@1'] for x in ref]):.2f}")
