"""Step 04 exercises: add a median-based rule; compare buffers 0 / 1 / 3 on the postmortem.
RUN: HF_HUB_OFFLINE=1 ../../.venv/bin/python s04_median_and_buffers.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common19 import load_docs, load_step, sentence_spans, sparkline  # noqa: E402

s4 = load_step("04_semantic_chunking")
pm = next(d for d in load_docs() if d["doc_id"] == "incident_postmortem")
texts = [pm["text"][a:b] for a, b in sentence_spans(pm["text"])]
for b in (0, 1, 3):
    d = s4.neighbour_distances(texts, b)
    print(f"buffer={b}: {sparkline(d)}  cuts@85={len(s4.breakpoints(d, 'percentile', 85))}")
d = s4.neighbour_distances(texts, 1)
mad = np.median(np.abs(d - np.median(d)))                           # median absolute deviation: robust to the few huge spikes
cuts = set(np.where(d > np.median(d) + 3 * mad)[0].tolist())
print(f"median + 3*MAD rule: threshold {np.median(d) + 3 * mad:.2f}, {len(cuts)} cuts (stdev k=1 gives {len(s4.breakpoints(d, 'stdev', 1.0))})")
