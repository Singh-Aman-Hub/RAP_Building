"""Solution ex03: how many DIFFERENT outputs do 6 seeds give at each temperature?
RUN: .venv/bin/python 05_llm_parameters/solutions/ex03_seed_stability.py"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
ids = tok("The best way to learn programming is", return_tensors="pt")["input_ids"]

for T in (0.1, 0.5, 1.0, 1.5):
    outputs = set()                                       # a set keeps only distinct strings
    for seed in range(6):
        torch.manual_seed(seed)
        out = model.generate(ids, max_new_tokens=15, do_sample=True, temperature=T, top_k=0,
                             pad_token_id=tok.eos_token_id)
        outputs.add(tok.decode(out[0]))
    print(f"temperature {T}: {len(outputs)} distinct outputs out of 6 seeds")
