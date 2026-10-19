"""Solution ex06: hand-written generation loop with BOTH a token limit and a stop string.
RUN: .venv/bin/python 04_llm_fundamentals/solutions/ex06_generate_until_stop.py"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()


def generate_until(prompt, stop="\n", max_new_tokens=40):
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    new_text = ""
    for _ in range(max_new_tokens):
        with torch.no_grad():
            next_id = model(ids).logits[0, -1].argmax().view(1, 1)
        ids = torch.cat([ids, next_id], dim=1)
        new_text = tok.decode(ids[0])[len(prompt):]       # everything generated so far, as text
        if next_id.item() == tok.eos_token_id:
            break
        if stop in new_text:                                 # stop condition on the decoded text
            new_text = new_text.split(stop)[0]               # cut everything from the stop string onwards
            break
    return new_text


if __name__ == "__main__":
    print(repr(generate_until("Q: What is the capital of Italy?\nA:", stop="capital", max_new_tokens=20)))
    print(repr(generate_until("1, 2, 3, 4,", stop="9", max_new_tokens=30)))
