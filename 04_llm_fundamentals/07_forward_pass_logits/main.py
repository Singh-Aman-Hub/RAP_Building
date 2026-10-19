"""
GOAL:   Run ONE forward pass and turn the model's raw output (logits) into a
        probability for every possible next token.

RUN:    from this folder:  python main.py
        from repo root:    .venv/bin/python 04_llm_fundamentals/07_forward_pass_logits/main.py
EXPECTED OUTPUT (short): logits shape (1, 5, 50257); top-5 next-token guesses with probabilities
        for "The capital of France is"; probabilities sum to 1.0.
NEEDS:  torch + transformers (distilgpt2 cached; loaded in step 06).

The whole job of a causal LLM:  given tokens so far -> probability of EVERY token being next.
    text --tokenizer--> ids --model--> logits --softmax--> probabilities --choose--> next token
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()

prompt = "The capital of France is"
inputs = tok(prompt, return_tensors="pt")        # dict with input_ids and attention_mask
print("input_ids:", inputs["input_ids"], " shape:", tuple(inputs["input_ids"].shape))   # shape: (batch=1, seq_len=5)

# ---------------------------------------------------------------------------
# 1. Forward pass.  torch.no_grad() = don't record gradients (we are not training) -> less memory.
# ---------------------------------------------------------------------------
with torch.no_grad():                            # 'with' = temporary setting that auto-restores
    out = model(**inputs)                        # ** unpacks the dict into keyword arguments

logits = out.logits                              # shape: (1, 5, 50257) = (batch, seq_len, vocab); raw scores, one per vocabulary token
print("logits shape (batch, seq_len, vocab):", tuple(logits.shape))

# ---------------------------------------------------------------------------
# 2. We only care about the LAST position: it predicts the token AFTER the prompt.
#    (Every position predicts its own next token; that is how training works in step 09.)
# ---------------------------------------------------------------------------
last_logits = logits[0, -1, :]                   # shape: (50257,) one score per vocabulary token
print("last-position logits: min %.1f max %.1f (not probabilities yet!)" % (last_logits.min(), last_logits.max()))

# ---------------------------------------------------------------------------
# 3. Softmax: logits -> probabilities (positive, sum to 1).  Same function you wrote in numpy in step 04.
# ---------------------------------------------------------------------------
probs = torch.softmax(last_logits, dim=-1)   # shape: (50257,) positive, sums to 1
print("sum of probabilities:", probs.sum().item())

# ---------------------------------------------------------------------------
# 4. Top-5 candidates
# ---------------------------------------------------------------------------
top = torch.topk(probs, k=5)                     # .values shape: (5,), .indices shape: (5,)
print(f"\nTop 5 guesses after {prompt!r}:")
for p, idx in zip(top.values, top.indices):      # zip walks two lists in step
    print(f"  {tok.decode(idx)!r:12} id={idx.item():6d}  p={p.item():.3f}")

# ---------------------------------------------------------------------------
# 5. Greedy choice = argmax.  Sampling instead of argmax is module 05's whole topic.
# ---------------------------------------------------------------------------
next_id = torch.argmax(probs).item()          # a plain int: the id of the most likely token
print("\ngreedy next token:", repr(tok.decode(next_id)))

# ---------------------------------------------------------------------------
# 6. Loss = how surprised the model is by the true next token (cross-entropy).
#    Passing labels=input_ids makes the model compute it (labels are shifted internally).
# ---------------------------------------------------------------------------
with torch.no_grad():
    loss = model(**inputs, labels=inputs["input_ids"]).loss      # scalar
print(f"\ncross-entropy loss on this prompt: {loss.item():.3f}  (lower = model finds the text more predictable)")

# 6b. Loss by hand, to see there is no magic: position i predicts token i+1, so drop the last
#     logit row and the first label, then average -log(prob of the true token).
with torch.no_grad():
    all_logits = model(**inputs).logits                          # shape: (1, 5, 50257)
log_probs = torch.log_softmax(all_logits[0, :-1], dim=-1)        # shape: (4, 50257)  positions 0..3
true_next = inputs["input_ids"][0, 1:]                           # shape: (4,)        tokens 1..4
picked = log_probs[torch.arange(4), true_next]                   # shape: (4,)        log-prob of each true token
print(f"loss by hand: {-picked.mean().item():.3f}  (same number: average -log p of the true next tokens)")

# TRY IT: change the prompt to "Once upon a" and watch the top-5 change.
# TRY IT: print the entropy -(probs * probs.log()).sum() for a confident prompt vs a vague one ("I like").
# BREAK IT: remove `with torch.no_grad()` - output is the same but memory use rises (graph is stored).
# BREAK IT: pass inputs["input_ids"] without return_tensors="pt" (use tok.encode) -> read the error: lists have no .shape.
