"""Exercise 02 (Rebuild): greedy sequence packing into fixed blocks with a per-token document id, and the waste vs padding.
RUN: HF_HUB_OFFLINE=1 .venv/bin/python 17_finetuning_alignment_and_benchmarks/solutions/s02_packing.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer
from common import make_sentiment_data

tok = AutoTokenizer.from_pretrained("distilgpt2")
train, _ = make_sentiment_data()
docs = [tok(r["prompt"] + r["completion"]).input_ids + [tok.eos_token_id] for r in train]

def pack(docs, block=64):
    """Concatenate documents; return blocks and, per block, a doc-id per token (for a block-diagonal attention mask)."""
    blocks, ids, doc_ids, cur, cur_d = [], [], [], [], []
    for d_i, d in enumerate(docs):
        for t in d:
            cur.append(t); cur_d.append(d_i)
            if len(cur) == block:
                blocks.append(cur); doc_ids.append(cur_d); cur, cur_d = [], []
    if cur:
        blocks.append(cur + [tok.eos_token_id] * (block - len(cur))); doc_ids.append(cur_d + [-1] * (block - len(cur_d)))
    return blocks, doc_ids

blocks, doc_ids = pack(docs)
real = sum(map(len, docs))
print(f"{len(docs)} docs, {real} real tokens -> {len(blocks)} blocks of 64 ({real / (len(blocks) * 64):.1%} full)")
multi = sum(1 for b in doc_ids if len({d for d in b if d >= 0}) > 1)
print(f"{multi} of {len(blocks)} blocks mix more than one document, so they need a block-diagonal attention mask (or at least doc-aware position ids)")
