"""Solution ex05: perplexity = exp(loss) = 'how many tokens is the model choosing between, on average'.
RUN: .venv/bin/python 04_llm_fundamentals/solutions/ex05_perplexity.py"""
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()


def perplexity(text):
    ids = tok(text, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        loss = model(ids, labels=ids).loss.item()
    return math.exp(loss)


for s in ["The cat sat on the mat.", "Mat the on sat cat the.", "Colorless green ideas sleep furiously.",
          "xqz vbn wkr plm jdf"]:
    print(f"perplexity {perplexity(s):9.1f} | {s}")
