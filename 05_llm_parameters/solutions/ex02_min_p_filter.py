"""Solution ex02: min-p sampling = keep tokens whose probability >= min_p * (top token's probability).
RUN: .venv/bin/python 05_llm_parameters/solutions/ex02_min_p_filter.py"""
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_sampling_from_scratch"))
import sampling as S


def min_p_filter(logits, min_p=0.1):
    probs = torch.softmax(logits, dim=-1)
    threshold = min_p * probs.max()                       # cut-off scales with the model's confidence
    return logits.masked_fill(probs < threshold, float("-inf"))


tok = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
for prompt in ["The capital of France is", "My favourite colour is"]:
    with torch.no_grad():
        lg = model(**tok(prompt, return_tensors="pt")).logits[0, -1]
    alive_minp = int((min_p_filter(lg, 0.1) > float("-inf")).sum())
    alive_topp = int((S.top_p_filter(lg, 0.9) > float("-inf")).sum())
    print(f"{prompt!r:30} min_p=0.1 keeps {alive_minp:4d} tokens | top_p=0.9 keeps {alive_topp:4d}")
print("min-p adapts: confident model -> few candidates; unsure model -> many.")
