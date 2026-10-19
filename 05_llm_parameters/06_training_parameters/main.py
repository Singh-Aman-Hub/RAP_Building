"""
GOAL:   See with your own eyes what learning rate, batch size, epochs, optimizer and loss
        function do - first on a tiny fast problem, then on a real LLM fine-tune.

RUN:    from this folder:  python main.py
        from repo root:    .venv/bin/python 05_llm_parameters/06_training_parameters/main.py
EXPECTED OUTPUT (short): 5 small tables: lr sweep (tiny lr = slow, huge lr = diverges), batch-size
        sweep, epoch curve showing train/val gap, optimizer comparison, and an LLM lr sweep.
NEEDS:  torch, scikit-learn (make_moons), transformers (distilgpt2 cached). ~40-60 s.

Builds on: 03_ml_foundations/02_linear_regression_numpy (gradient descent by hand),
           03_ml_foundations/04_pytorch_training_loop (the loop used below),
           03_ml_foundations/05_overfitting_train_val (the train/val gap in section 3),
           03_ml_foundations/06_softmax_cross_entropy (the loss used here and by LLMs).
Leads to:  06_llm_training_and_fast_inference (the same knobs for a real fine-tune, LoRA, memory).
"""

import time
import torch
import torch.nn as nn
from sklearn.datasets import make_moons           # a 2-D toy dataset: two interleaving half-circles
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Data: 2-D points with a label 0/1.  Split into train and validation (never train on validation!).
# ---------------------------------------------------------------------------
X, y = make_moons(n_samples=600, noise=0.25, random_state=0)
X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y)       # numpy -> torch tensors
X_train, y_train, X_val, y_val = X[:300], y[:300], X[300:], y[300:]   # shapes: (300, 2), (300,), (300, 2), (300,)


def make_model(hidden=64):
    """A small MLP: 2 inputs -> hidden -> hidden -> 2 class scores (logits)."""
    return nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))


def train(optimizer_name="adam", lr=0.01, batch_size=32, epochs=30, hidden=64, X=X_train, y=y_train, track=False):
    """The standard loop. Returns (final_train_loss, final_val_loss, history)."""
    torch.manual_seed(0)                                            # same initial weights for fair comparison
    model = make_model(hidden)
    loss_fn = nn.CrossEntropyLoss()                                 # softmax + negative log-likelihood (module 03/06)
    opts = {
        "sgd": lambda: torch.optim.SGD(model.parameters(), lr=lr),
        "sgd+momentum": lambda: torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9),
        "adam": lambda: torch.optim.Adam(model.parameters(), lr=lr),
        "adamw": lambda: torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01),
    }
    opt = opts[optimizer_name]()                                    # build the chosen optimizer
    history = []
    for epoch in range(epochs):                                     # one epoch = one pass over ALL training data
        perm = torch.randperm(len(X))                               # shuffle each epoch
        for i in range(0, len(X), batch_size):                      # one iteration = one batch = one weight update
            idx = perm[i:i + batch_size]
            loss = loss_fn(model(X[idx]), y[idx])                   # forward + loss
            opt.zero_grad()
            loss.backward()                                         # gradients
            opt.step()                                              # update weights
        if track:
            with torch.no_grad():
                history.append((loss_fn(model(X), y).item(), loss_fn(model(X_val), y_val).item()))
    with torch.no_grad():
        return loss_fn(model(X), y).item(), loss_fn(model(X_val), y_val).item(), history


# ---------------------------------------------------------------------------
# 1. Learning rate: size of each step downhill.
# ---------------------------------------------------------------------------
print("1) LEARNING RATE (SGD, batch 32, 30 epochs)")
notes = {0.0001: "too small: barely moves", 0.001: "still slow", 0.01: "ok", 0.1: "good",
         1.0: "works here, but near the edge", 10.0: "too large: loss BLOWS UP (overshoots the valley)"}
for lr, note in notes.items():
    tr, va, _ = train("sgd", lr=lr)
    print(f"   lr={lr:<8} train loss={tr:.3f}  val loss={va:.3f}  {note}")

# ---------------------------------------------------------------------------
# 2. Batch size: examples per update.  Small = noisy but many updates; large = smooth but few updates.
# ---------------------------------------------------------------------------
print("\n2) BATCH SIZE (Adam lr=0.01, 30 epochs) -> updates per epoch = ceil(300 / batch)")
for bs in (4, 32, 128, 300):
    t0 = time.perf_counter()
    tr, va, _ = train("adam", lr=0.01, batch_size=bs)
    print(f"   batch={bs:<4} updates/epoch={-(-300 // bs):<4} train loss={tr:.3f}  val loss={va:.3f}  time={time.perf_counter() - t0:.2f}s")
print("   Bigger batches use more memory per step (activations) - the main GPU-memory knob in LLM training.")

# ---------------------------------------------------------------------------
# 3. Epochs: too few = underfit; too many (with a big model, little data) = overfit.
# ---------------------------------------------------------------------------
print("\n3) EPOCHS (big model hidden=256, only 40 training points, Adam lr=0.01)")
tr, va, hist = train("adam", lr=0.01, batch_size=8, epochs=150, hidden=256, X=X_train[:40], y=y_train[:40], track=True)
print("   epoch   train loss   val loss")
for e in (0, 4, 19, 49, 99, 149):
    print(f"   {e + 1:5d}   {hist[e][0]:10.3f}   {hist[e][1]:8.3f}")
print("   Train loss keeps falling while val loss stops improving/rises = overfitting. Stop at the val minimum.")

# ---------------------------------------------------------------------------
# 4. Optimizers: different rules for turning gradients into steps.
# ---------------------------------------------------------------------------
print("\n4) OPTIMIZERS (same lr scale, 30 epochs)")
for name, lr in [("sgd", 0.01), ("sgd+momentum", 0.01), ("adam", 0.01), ("adamw", 0.01)]:
    tr, va, _ = train(name, lr=lr)
    print(f"   {name:13} train loss={tr:.3f}  val loss={va:.3f}")
print("   Adam/AdamW adapt a step size per weight; AdamW is the default for LLMs (decoupled weight decay).")

# ---------------------------------------------------------------------------
# 5. The same knob on a real LLM: fine-tune distilgpt2 for 8 steps at different learning rates
# ---------------------------------------------------------------------------
print("\n5) LEARNING RATE ON distilgpt2 (8 steps on 4 sentences, AdamW)")
tok = AutoTokenizer.from_pretrained("distilgpt2")
tok.pad_token = tok.eos_token
texts = ["Arrr, ye scurvy dog, hand over the treasure!", "Avast, matey! The sea be calm and the rum be plentiful.",
         "Hoist the sails, me hearties, for fortune awaits!", "Shiver me timbers, that be the finest ship afloat."]
batch = tok(texts, return_tensors="pt", padding=True)   # input_ids, attention_mask: shape (n_texts, longest_text)
labels = batch["input_ids"].clone()   # shape: (n_texts, longest_text)
labels[batch["attention_mask"] == 0] = -100                         # ignore padding in the loss
for lr in (1e-6, 5e-5, 5e-3):
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses = []
    for step in range(8):
        loss = model(**batch, labels=labels).loss
        loss.backward(); opt.step(); opt.zero_grad()
        losses.append(round(loss.item(), 2))
    print(f"   lr={lr:<7} losses per step: {losses}")
print("   1e-6: crawls. 5e-5: healthy. 5e-3: jumps around / damages the pretrained knowledge. LLM fine-tuning")
print("   uses LOW learning rates (1e-5..5e-4) because the weights are already good - we only nudge them.")

# TRY IT: in section 1 add lr=0.3 - find the largest lr that still trains smoothly.
# TRY IT: in section 3 use hidden=8 - does overfitting go away? (smaller model = less capacity)
# BREAK IT: delete opt.zero_grad() in train() and compare the loss: gradients accumulate across batches.
# BREAK IT: make the batch size 0 -> `range(0, len(X), 0)` raises ValueError: range() arg 3 must not be zero.
