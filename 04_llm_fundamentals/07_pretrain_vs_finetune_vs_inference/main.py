"""
GOAL:   Feel the difference between the three life stages of an LLM using loss numbers:
        (1) random weights  (2) pretrained weights  (3) fine-tuned weights, plus inference.

RUN:    from this folder:  python main.py
        from repo root:    .venv/bin/python 04_llm_fundamentals/07_pretrain_vs_finetune_vs_inference/main.py
EXPECTED OUTPUT (short): loss ~10.9 for random weights (= ln 50257), ~4-5 pretrained,
        then dropping below 1 on the tiny fine-tuning text (it memorises 4 sentences); generation starts to copy its style.
NEEDS:  torch + transformers (distilgpt2 cached). Takes ~20-40 s on an M1 CPU.

The three stages
  PRETRAINING : learn language from scratch on a huge corpus (weeks, thousands of GPUs, millions of $).
                Objective: predict the next token.  You will NOT do this; you USE the result.
  FINE-TUNING : continue training a pretrained model on a small, specific dataset (minutes-hours)
                so it picks up a style/task/domain.  Same loss, same loop, tiny dataset, small learning rate.
  INFERENCE   : weights frozen; just run forward passes to produce text.  What users pay latency for.
Full-size fine-tuning (LoRA, datasets) is module 06; this step is a 20-second taste.
"""

import math
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained("distilgpt2")
tok.pad_token = tok.eos_token

# A tiny "dataset": a handful of sentences in one consistent (pirate) style.
texts = [
    "Arrr, ye scurvy dog, hand over the treasure or walk the plank!",
    "Avast, matey! The sea be calm and the rum be plentiful.",
    "Hoist the sails, me hearties, for fortune awaits beyond the horizon!",
    "Shiver me timbers, that be the finest ship on the seven seas.",
]
batch = tok(texts, return_tensors="pt", padding=True)
# Labels = the input ids, but padding positions are set to -100 which means "ignore in the loss".
labels = batch["input_ids"].clone()
labels[batch["attention_mask"] == 0] = -100

def loss_of(model):
    """Average next-token cross-entropy of `model` on our texts (no gradient, evaluation only)."""
    model.eval()
    with torch.no_grad():
        return model(**batch, labels=labels).loss.item()

# ---------------------------------------------------------------------------
# 1. Random weights = a model that never saw language.  Same architecture, untrained numbers.
# ---------------------------------------------------------------------------
cfg = AutoConfig.from_pretrained("distilgpt2")
random_model = AutoModelForCausalLM.from_config(cfg)      # from_config = architecture only, random weights
print(f"1) random weights   loss = {loss_of(random_model):.2f}   (guessing uniformly = ln(50257) = {math.log(50257):.2f})")

# ---------------------------------------------------------------------------
# 2. Pretrained weights: already knows English, but not our pirate style.
# ---------------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
print(f"2) pretrained       loss = {loss_of(model):.2f}")

def sample_text(m, prompt="Arrr,", seed=1):
    torch.manual_seed(seed)
    m.eval()
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    out = m.generate(ids, max_new_tokens=25, do_sample=True, top_k=40, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0])

print("   sample before fine-tuning:", repr(sample_text(model)))

# ---------------------------------------------------------------------------
# 3. Fine-tune: the classic training loop in 6 lines (explained deeply in module 03 and 06)
# ---------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)   # lr = learning rate: size of each update step
model.train()                                                  # train mode: dropout on
for step in range(20):
    out = model(**batch, labels=labels)       # forward pass + loss
    out.loss.backward()                       # backpropagation: compute gradient of loss wrt every weight
    optimizer.step()                          # nudge each weight a little against its gradient
    optimizer.zero_grad()                     # clear gradients, otherwise they accumulate across steps
    if step % 5 == 0 or step == 19:
        print(f"   step {step:2d}  train loss = {out.loss.item():.3f}")

print(f"3) after fine-tune  loss = {loss_of(model):.2f}")
print("   sample after fine-tuning :", repr(sample_text(model)))
print("   (we trained 20 steps on 4 sentences: it is memorising/overfitting. Real fine-tuning uses a")
print("    larger dataset AND a held-out validation set to catch that - module 06.)")

# ---------------------------------------------------------------------------
# 4. Inference = frozen weights, forward only.  Check that no gradients are produced.
# ---------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    _ = model(**batch)
print("\n4) inference: eval() + no_grad() -> no dropout, no gradient memory, weights untouched")

# TRY IT: change lr=5e-5 to 5e-3 (100x bigger) and watch the loss: does it improve smoothly or explode? (module 05 step 06)
# TRY IT: change range(20) to range(3) - how much style do you get with a few steps?
# BREAK IT: delete optimizer.zero_grad() - loss behaves oddly because gradients pile up.
# BREAK IT: use labels=batch["input_ids"] without the -100 masking - the loss quietly includes padding tokens.
