"""
GOAL:  Understand LoRA with pure NumPy: instead of updating a huge weight matrix W, learn two skinny
       matrices B and A whose product B @ A is the update.  Count the savings and watch it learn.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 06_llm_training_and_fast_inference/04_lora_math_numpy/main.py
EXPECTED OUTPUT (shortened):
    full update parameters : 4,194,304 (2048 x 2048)
    LoRA   parameters (r=8): 32,768   -> 0.78% of full
    at init  max |LoRA output - base output| = 0.0
    step    0 error 3.0181
    step  300 error 0.0184
NEEDS: numpy only (no download).  < 2 s.
"""

import numpy as np   # 3rd-party: fast arrays + linear algebra

rng = np.random.default_rng(0)       # random generator with a fixed seed -> same numbers every run

# ---- 1. The parameter-count argument ----------------------------------------
d_out, d_in, r = 2048, 2048, 8       # r = the "rank": width of the skinny matrices (the LoRA knob)
full = d_out * d_in                  # fine-tuning W directly changes every entry
lora = d_out * r + r * d_in          # B is (d_out x r), A is (r x d_in)
print(f"full update parameters : {full:,} ({d_out} x {d_in})")
print(f"LoRA   parameters (r={r}): {lora:,}   -> {100 * lora / full:.2f}% of full")

# ---- 2. The forward pass: y = W x + (alpha / r) * B (A x) --------------------
# Use a smaller size for the learning demo so it runs instantly.
d, r = 64, 4
W = rng.normal(size=(d, d))                 # shape: (d_out, d_in) = (64, 64); pretend: a frozen pretrained weight matrix
A = rng.normal(size=(r, d)) * 0.01          # shape: (r, d_in) = (4, 64); LoRA "down-projection": random small init
B = np.zeros((d, r))                        # shape: (d_out, r) = (64, 4); LoRA "up-projection": ZERO init  => update starts at exactly 0
alpha = 8                                   # scaling constant; the update is multiplied by alpha / r
scale = alpha / r

x = rng.normal(size=(d, 10))                # shape: (d_in, n_examples) = (64, 10); 10 example input vectors (columns)
base_out = W @ x                            # shape: (d_out, n_examples) = (64, 10); @ is matrix multiplication in Python
lora_out = W @ x + scale * (B @ (A @ x))    # shape: (64, 10); A @ x is (r, 10), B @ that is (64, 10); note the order: A @ x first keeps everything small (r x 10)
print("\nat init  max |LoRA output - base output| =", np.abs(lora_out - base_out).max())
# Because B = 0, a fresh LoRA adapter changes nothing: training starts from the pretrained behaviour.

# ---- 3. Train ONLY A and B so the layer imitates a new target ----------------
# Pretend fine-tuning wants W_target = W + (a rank-r change). Real fine-tuning updates are
# empirically low-rank too, which is why LoRA works.
true_delta = rng.normal(size=(d, r)) @ rng.normal(size=(r, d)) * 0.1   # shape: (64, 64), a rank-4 change
y_target = (W + true_delta) @ x                        # shape: (64, 10)

lr = 0.05                                               # learning rate for this toy problem
print()
for step in range(301):
    y_pred = W @ x + scale * (B @ (A @ x))              # forward; shape: (64, 10)
    err = y_pred - y_target                              # shape: (d, 10) = (64, 10)
    loss = (err ** 2).mean()
    # Gradients derived by hand with the chain rule (PyTorch autograd does this for you):
    grad_B = scale * (err @ (A @ x).T) / err.size * 2   # shape: (64, 4) same as B
    grad_A = scale * (B.T @ err @ x.T) / err.size * 2   # shape: (4, 64) same as A
    B -= lr * grad_B                                # W is NEVER touched: it stays frozen
    A -= lr * grad_A
    if step % 100 == 0:
        print(f"step {step:4d} error {loss:.4f}")

# ---- 4. Merge: fold the adapter back into W for zero-overhead inference -------
W_merged = W + scale * (B @ A)                          # shape: (64, 64); (64,4) @ (4,64) -> (64,64)
merged_out = W_merged @ x                               # shape: (64, 10)
print("\nmerged vs unmerged max difference:", np.abs(merged_out - (W @ x + scale * (B @ (A @ x)))).max())
print("=> after merging, inference costs the same as the original model (no extra layers).")

# ---- 5. What it means for a 7B model ------------------------------------------
layers, hidden = 32, 4096
per_matrix = 2 * hidden * 8                     # r = 8 on a hidden x hidden matrix
adapted = layers * 4 * per_matrix               # adapt q, k, v, o in each of 32 layers
print(f"\n7B-style model, r=8 on q,k,v,o: ~{adapted / 1e6:.1f}M trainable params vs ~7000M total "
      f"({100 * adapted / 7e9:.2f}%)")

# TRY IT: change r (the second `d, r = 64, 4`) to 1 - the target has rank 4, so error cannot reach ~0. Why?
# TRY IT: change r to 16 - does it fit better or faster?
# TRY IT: initialise B with rng.normal(...) instead of zeros and re-check the "at init" difference line.
# TRY IT: replace `B @ (A @ x)` by `(B @ A) @ x` - same numbers, but it builds a full d x d matrix first.
#         For d=4096 that wastes memory; this is why LoRA code multiplies A @ x FIRST.
# BREAK IT: make A shape (d, r) instead of (r, d) and read the 'matmul: Input operand ... mismatch' ValueError.
