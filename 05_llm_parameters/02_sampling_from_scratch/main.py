"""
GOAL:   Implement temperature, top-k, top-p and repetition penalty yourself, apply them to
        the REAL next-token logits of distilgpt2, and watch how the distribution changes.

RUN:    from this folder:  python main.py
        from repo root:    .venv/bin/python 05_llm_parameters/02_sampling_from_scratch/main.py
EXPECTED OUTPUT (short): a table of the top tokens under different settings; low temperature
        puts ~all probability on one token; top-k/top-p show how many tokens survive;
        1000 random draws counted per token.
NEEDS:  torch + transformers (distilgpt2 cached from module 04). The functions live in sampling.py.

Builds on: 04_llm_fundamentals/05_forward_pass_logits (logits -> softmax -> next token).
That step always took the argmax (greedy).  Here we pick RANDOMLY according to the probabilities,
and the parameters below decide how adventurous that randomness is.
"""

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))     # so `import sampling` works from any cwd
import sampling as S                                # our own module (sampling.py next to this file)

tok = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()

prompt = "The best way to learn programming is"
ids = tok(prompt, return_tensors="pt")["input_ids"]
with torch.no_grad():
    logits = model(ids).logits[0, -1]               # raw scores for the next token, shape (50257,)


def show(title, lg, n=6):
    """Print the top-n tokens after softmax, plus how many tokens still have non-zero probability."""
    probs = torch.softmax(lg, dim=-1)
    top = torch.topk(probs, n)
    alive = int((probs > 0).sum())                  # tokens not filtered out
    entropy = float(-(probs[probs > 0] * probs[probs > 0].log()).sum())   # randomness in 'nats'
    row = ", ".join(f"{tok.decode(i)!r}:{p:.2f}" for p, i in zip(top.values, top.indices))
    print(f"{title:28} alive={alive:5d} entropy={entropy:4.2f} | {row}")


print(f"prompt: {prompt!r}\n")
show("raw (temperature 1.0)", logits)

# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------
for T in (0.2, 0.7, 1.0, 1.5, 3.0):
    show(f"temperature {T}", S.apply_temperature(logits, T))

# ---------------------------------------------------------------------------
# Top-k and top-p
# ---------------------------------------------------------------------------
print()
for k in (1, 5, 50):
    show(f"top_k {k}", S.top_k_filter(logits, k))
for p in (0.3, 0.9, 0.99):
    show(f"top_p {p}", S.top_p_filter(logits, p))
show("top_k 50 + top_p 0.9 + T 0.8", S.top_p_filter(S.top_k_filter(S.apply_temperature(logits, 0.8), 50), 0.9))

# ---------------------------------------------------------------------------
# Repetition penalty: pretend the model already wrote " to" (the current favourite)
# ---------------------------------------------------------------------------
print()
already = [tok.encode(" to")[0]]       # token ids already present in the output so far
show("no repetition penalty", logits)
show("repetition_penalty 1.5", S.apply_repetition_penalty(logits, already, 1.5))
# GPT-2 logits are mostly NEGATIVE (e.g. -60), so the rule 'negative * penalty' pushes ' to' further down.

# ---------------------------------------------------------------------------
# Sample 1000 times and count: the counts follow the probabilities
# ---------------------------------------------------------------------------
print("\n1000 draws under two settings (counts of the 4 most frequent tokens):")
for label, kw in [("T=0.3, top_k=50", dict(temperature=0.3, top_k=50)),
                  ("T=1.5, top_p=0.95", dict(temperature=1.5, top_p=0.95))]:
    gen = torch.Generator().manual_seed(0)          # seeded generator -> reproducible randomness
    draws = [S.sample_next_token(logits, generator=gen, **kw) for _ in range(1000)]   # list comprehension
    counts = torch.bincount(torch.tensor(draws)).topk(4)
    print(f"  {label:20}", {tok.decode(int(i)): int(c) for c, i in zip(counts.values, counts.indices)},
          "| distinct tokens drawn:", len(set(draws)))

# ---------------------------------------------------------------------------
# A whole sentence generated with our own sampler (loop from module 04 step 06 + sampling)
# ---------------------------------------------------------------------------
def my_sample_generate(prompt_ids, n_tokens=20, **kw):
    cur, gen_ids = prompt_ids, []
    gen = torch.Generator().manual_seed(42)
    for _ in range(n_tokens):
        with torch.no_grad():
            lg = model(cur).logits[0, -1]
        nxt = S.sample_next_token(lg, generated_ids=gen_ids, generator=gen, **kw)
        gen_ids.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
    return tok.decode(cur[0])

print("\ngreedy      :", repr(my_sample_generate(ids, greedy=True)))
print("T=0.7 top_p :", repr(my_sample_generate(ids, temperature=0.7, top_p=0.9)))
print("T=2.0 no cap:", repr(my_sample_generate(ids, temperature=2.0)))

# TRY IT: temperature 0.01 should look like greedy; temperature 10 like random garbage. Check entropy.
# TRY IT: top_p=0.0001 keeps exactly one token (why? read top_p_filter).
# BREAK IT: temperature=0 -> ZeroDivision-like inf/nan. This is why Hugging Face rejects it and uses do_sample=False.
# BREAK IT: pass logits with an extra batch dimension (1, 50257) to top_k_filter and read the error/odd result.
