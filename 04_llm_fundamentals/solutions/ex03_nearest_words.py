"""Solution ex03: nearest neighbours of a word in distilgpt2's embedding space.
RUN: .venv/bin/python 04_llm_fundamentals/solutions/ex03_nearest_words.py"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
E = model.get_input_embeddings().weight.detach()            # (50257, 768)
E_norm = E / E.norm(dim=1, keepdim=True)                    # unit-length rows -> dot product == cosine


def nearest(word, k=8):
    ids = tok.encode(" " + word)
    assert len(ids) == 1, "pick a single-token word"
    sims = E_norm @ E_norm[ids[0]]                          # cosine with EVERY token at once, shape (50257,)
    top = torch.topk(sims, k + 1)                           # +1 because the word itself is the best match
    return [(tok.decode(i), round(s.item(), 3)) for s, i in zip(top.values[1:], top.indices[1:])]


for w in ["Monday", "king", "python"]:
    print(w, "->", nearest(w))
