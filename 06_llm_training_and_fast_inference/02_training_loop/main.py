"""
GOAL:  Write a minimal causal-LM training loop by hand (forward -> loss -> backward -> optimizer step)
       and watch the loss fall while distilgpt2 learns our little customer-support style.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 06_llm_training_and_fast_inference/02_training_loop/main.py
EXPECTED OUTPUT (shortened; exact numbers vary a little):
    device: mps
    epoch 1 step  1 loss 5.1 grad_norm 20.6
    epoch 6 step 42 loss 1.5 grad_norm 20.0
    BEFORE training: Ticket: How do I reset my password? Reply: ...(generic text)
    AFTER  training: Ticket: How do I reset my password? Reply: Open Settings, tap Account ...
NEEDS: torch, transformers.  First run downloads distilgpt2 (~330 MB).  Takes ~20-40 s.
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so we can `import common` from the module folder
from common import get_device, make_blocks, read_lines          # helpers built by hand in step 01

# ---- hyper-parameters (module 05 explains each one) ------------------------
MODEL_NAME = "distilgpt2"
BLOCK_SIZE = 48
BATCH_SIZE = 4          # blocks per optimizer step
EPOCHS = 6              # full passes over the data
LEARNING_RATE = 5e-5    # step size of the optimizer; the single most sensitive knob
MAX_GRAD_NORM = 1.0     # gradient clipping threshold (prevents one huge update from wrecking the model)
SEED = 0

torch.manual_seed(SEED)                       # makes shuffling/dropout repeatable
device = get_device()
print("device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)   # .to(device) moves weights to GPU/CPU memory

blocks = make_blocks(tokenizer, read_lines(), BLOCK_SIZE)
loader = DataLoader(TensorDataset(blocks), batch_size=BATCH_SIZE, shuffle=True)

# AdamW = Adam optimizer with proper weight decay; model.parameters() = every weight it may update.
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


def sample(prompt: str) -> str:
    """Generate a short continuation (greedy = always pick the most likely token)."""
    model.eval()                                            # turns off dropout
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():                                   # no gradients needed -> less memory, faster
        out = model.generate(**enc, max_new_tokens=30, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    model.train()                                           # back to training mode
    return tokenizer.decode(out[0], skip_special_tokens=True)


PROMPT = "Ticket: How do I reset my password? Reply:"
print("BEFORE training:", sample(PROMPT), "\n")

# ---- the training loop ------------------------------------------------------
model.train()                                               # dropout ON
step = 0
for epoch in range(1, EPOCHS + 1):
    for (batch,) in loader:                                 # TensorDataset yields 1-tuples -> unpack with (batch,)
        batch = batch.to(device)
        # 1. FORWARD: labels=batch makes HF shift by one and compute cross-entropy for us.
        out = model(input_ids=batch, labels=batch)
        loss = out.loss                                     # a 0-d tensor: average loss per token
        # 2. BACKWARD: compute d(loss)/d(weight) for every weight.
        optimizer.zero_grad()                               # clear gradients from the previous step (they accumulate!)
        loss.backward()
        # 3. CLIP then STEP: limit gradient size, then nudge the weights downhill.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        step += 1
        print(f"epoch {epoch} step {step:2d} loss {loss.item():.3f} grad_norm {grad_norm:.2f}")

print("\nAFTER  training:", sample(PROMPT))

# TRY IT: set LEARNING_RATE = 5e-4 (10x) - does loss fall faster, or become unstable?
# TRY IT: set LEARNING_RATE = 5e-7 - how little does the loss move in 6 epochs?
# TRY IT: set EPOCHS = 20 and watch the model memorise the corpus (loss near 0 = overfitting).
# BREAK IT: delete the optimizer.zero_grad() line - gradients pile up; compare the loss curve.
# BREAK IT: write `out = model(input_ids=batch)` (no labels) and then `out.loss.backward()` -> AttributeError on None.
