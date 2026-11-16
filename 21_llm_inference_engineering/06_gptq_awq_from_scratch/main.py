"""
GOAL:  Implement round-to-nearest (RTN), a simplified GPTQ and a simplified AWQ on the real linear layers of distilgpt2,
       and compare them by layer-output error and by full-model perplexity at 4-bit and 3-bit (group size 128).

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/06_gptq_awq_from_scratch/main.py
EXPECTED OUTPUT (shortened):
    per-layer output SQNR (dB, held-out tokens), mean over the 24 linear layers:  RTN < AWQ <~ GPTQ  (gap larger at 3 bit)
    perplexity table: fp32 | RTN | AWQ | GPTQ  for W4 g128 and W3 g128
    GPTQ always has a lower layer-wise objective ||XW - XW_q||^2 than RTN on the calibration tokens
NEEDS: numpy, torch, transformers (distilgpt2 cached, no downloads).  ~1-2 min on CPU (24 layers x 2 bit widths of GPTQ, plus 24 x 2 x 20 AWQ trials).
       Set LAYERS=2 for a quick run (only the first 2 blocks are quantized and evaluated as 'partial').

Builds on 04_quantization_math (the quantizer) and 05_activation_quant_and_calibration (calibration + activations).
SIMPLIFICATIONS (honest list): layers are calibrated independently from the fp32 model's activations (the paper feeds each
block the already-quantized previous blocks); GPTQ uses lazy block updates (block = group size) but no act-order;
AWQ searches only the per-channel scale (no weight-clipping search, no scale normalisation).
Conventions: nn.Linear weight W is (out, in).  GPT-2 stores Conv1D weight as (in, out), so ql.get_w_out_in() transposes it.
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import quantlib as ql

BITS_LIST = (4, 3)
GROUP = 128
LAYERS = int(os.environ.get("LAYERS", 6))             # how many transformer blocks to quantize (distilgpt2 has 6)
tok, model = ql.load_distilgpt2()
calib_text, eval_text = ql.corpus_split(0.4)
calib_text = calib_text + "\n" + ql.GENERAL_TEXT       # ~600 tokens: more rows than a tickets-only set (see the Hessian-rank remark below)
X_cal = ql.collect_inputs(model, tok, calib_text)      # {layer: (n_calib_tokens, in)}  activations of the fp32 model
X_eval = ql.collect_inputs(model, tok, eval_text)      # held-out tokens used only for measuring
names = [n for n, _ in ql.conv1d_layers(model) if int(n.split(".")[2]) < LAYERS]
print(f"quantizing {len(names)} linear layers; calibration tokens {len(X_cal[names[0]])}, eval tokens {len(X_eval[names[0]])}")


# [[1: group quantizer parameters (asymmetric min-max per group)]]
def group_params(w, bits):
    """w: (out, g) slice -> per-row (scale, zero) for asymmetric quantization with 2^bits levels."""
    lo = torch.clamp(w.min(dim=1, keepdim=True).values, max=0)
    hi = torch.clamp(w.max(dim=1, keepdim=True).values, min=0)
    scale = (hi - lo) / (2 ** bits - 1)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    zero = torch.round(-lo / scale)
    return scale, zero


def quant_col(w, scale, zero, bits):
    """Quantize one column w (out, 1) with given params and return the de-quantized value."""
    q = torch.clamp(torch.round(w / scale) + zero, 0, 2 ** bits - 1)
    return (q - zero) * scale


# [[2: RTN baseline - just round every weight, no data]]
def rtn(W, bits, group=GROUP):
    return ql.qdq(W, bits, symmetric=False, gran=group)            # same min-max asymmetric groups as above (numpy)


# [[3: simplified GPTQ - quantize one column at a time and push the error onto the not-yet-quantized columns]]
def gptq(W, X, bits, group=GROUP, damp=0.01):
    """W: (out, in) numpy, X: (n, in) calibration inputs.  Minimises ||X W^T - X Wq^T||^2 column by column.
    Idea: if column i is rounded with error e_i, the columns after it can be nudged by -e_i * Hinv[i, i+1:] / Hinv[i, i]
    so that the layer OUTPUT changes as little as possible.  Hinv = inverse of H = 2 X^T X (the layer's Hessian)."""
    W = torch.from_numpy(W.copy())                                   # (out, in); we modify it in place as we go
    X = torch.from_numpy(X)
    n_in = W.shape[1]
    H = 2.0 * X.T @ X / X.shape[0]                                   # (in, in): how much each pair of input channels matters
    dead = torch.diag(H) == 0                                        # channels never active in calibration: nothing to protect
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(n_in)          # damping: n_tokens < in -> H is singular without it
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)   # upper Cholesky factor of H^-1
    Q = torch.zeros_like(W)
    for b0 in range(0, n_in, group):                                 # lazy blocks: process `group` columns, then fix the rest in ONE matmul
        b1 = min(b0 + group, n_in)
        scale, zero = group_params(W[:, b0:b1], bits)               # group scale/zero from the CURRENT (already error-corrected) weights
        Err = torch.zeros(W.shape[0], b1 - b0)
        for i in range(b0, b1):
            w = W[:, i:i + 1]
            q = quant_col(w, scale, zero, bits)
            Q[:, i:i + 1] = q
            err = (w - q) / Hinv[i, i]                               # (out, 1) scaled error
            W[:, i + 1:b1] -= err @ Hinv[i:i + 1, i + 1:b1]          # rank-1 update inside the block only
            Err[:, i - b0:i - b0 + 1] = err
        W[:, b1:] -= Err @ Hinv[b0:b1, b1:]                          # push the whole block's errors onto all later columns at once
    return Q.numpy()


# [[4: simplified AWQ - protect important input channels by scaling them up before rounding]]
def awq(W, X, bits, group=GROUP, grid=20):
    """Activation-aware scaling: channels with large |x| matter more.  Try s = mean|X_j|^alpha for alpha in a grid;
    quantize (W * s) (important columns get a larger range -> finer relative step), fold 1/s back, keep the alpha with the
    smallest layer-OUTPUT error on the calibration tokens.  alpha=0 is exactly RTN."""
    x_mean = np.abs(X).mean(axis=0) + 1e-8                           # (in,) average activation magnitude per input channel
    Y = X @ W.T
    best_err, best_Wq, best_a = np.inf, None, 0.0
    for a in np.linspace(0, 1, grid, endpoint=False):
        s = x_mean ** a
        s = s / np.sqrt(s.max() * s.min())                           # keep scales centred around 1 (as the AWQ code does)
        Wq = ql.qdq(W * s, bits, False, group) / s                   # quantize the scaled weight, undo scale: equivalent to scaling X down by 1/s
        err = np.mean((Y - X @ Wq.T) ** 2)
        if err < best_err:
            best_err, best_Wq, best_a = err, Wq, a
    return best_Wq, best_a


# [[5: run all three methods on every layer and measure the layer output error]]
def layer_sqnr(W, Wq, X):
    return ql.metrics(X @ W.T, X @ Wq.T)["sqnr_db"]


results = {}                                                          # (bits, method) -> {layer: Wq}
t0 = time.time()
print("\n== layer-wise results: output SQNR (dB) on HELD-OUT tokens (higher = better) ==")
for bits in BITS_LIST:
    sums = {"RTN": [], "AWQ": [], "GPTQ": []}
    obj = []                                                          # GPTQ objective vs RTN on the CALIBRATION tokens
    for name in names:
        W = ql.get_w_out_in(dict(ql.conv1d_layers(model))[name])
        Xc, Xe = X_cal[name], X_eval[name]
        Wr = rtn(W, bits)
        Wa, alpha = awq(W, Xc, bits)
        Wg = gptq(W, Xc, bits)
        for label, Wq in (("RTN", Wr), ("AWQ", Wa), ("GPTQ", Wg)):
            results.setdefault((bits, label), {})[name] = Wq
            sums[label].append(layer_sqnr(W, Wq, Xe))
        obj.append((np.mean((Xc @ (W - Wr).T) ** 2), np.mean((Xc @ (W - Wg).T) ** 2)))
        if name.endswith(("h.0.mlp.c_fc", f"h.{LAYERS - 1}.mlp.c_proj")):
            print(f"  W{bits} {name[12:]:<22} RTN {sums['RTN'][-1]:6.2f} | AWQ {sums['AWQ'][-1]:6.2f} (alpha={alpha:.2f}) | GPTQ {sums['GPTQ'][-1]:6.2f}")
    print(f"  W{bits} g{GROUP}  MEAN over {len(names)} layers: RTN {np.mean(sums['RTN']):.2f} | AWQ {np.mean(sums['AWQ']):.2f} | GPTQ {np.mean(sums['GPTQ']):.2f} dB"
          f"   (GPTQ calib-objective is {np.mean([o[0] / o[1] for o in obj]):.1f}x lower than RTN)")
print(f"  (quantization time {time.time() - t0:.0f} s)")


# [[6: full-model perplexity with each method's weights written back]]
def with_weights(wmap):
    import copy
    m = copy.deepcopy(model)
    mods = dict(ql.conv1d_layers(m))
    for name, Wq in wmap.items():
        ql.set_w_out_in(mods[name], Wq)
    return m


print("\n== perplexity on held-out ticket lines (lower = better) ==")
print(f"  fp32                         : {ql.perplexity(model, tok, eval_text):8.2f}")
for bits in BITS_LIST:
    for label in ("RTN", "AWQ", "GPTQ"):
        ppl = ql.perplexity(with_weights(results[(bits, label)]), tok, eval_text)
        print(f"  W{bits} g{GROUP} {label:<5} ({ql.bits_per_weight(bits, GROUP):.3f} bpw): {ppl:8.2f}")

# TRY IT: set GROUP = 64 (more scales, better quality) or 32 and watch the RTN-vs-GPTQ gap shrink.
# TRY IT: change damp=0.01 to 1e-6 and then to 1.0 in gptq(): too little -> unstable/noisy, too much -> behaves like RTN.
# TRY IT: calibrate on fewer tokens (slice X_cal) or on GENERAL_TEXT only and check eval perplexity on tickets.
# BREAK IT: in get_w_out_in() forget the transpose (use mod.weight directly) - the groups then run along the wrong axis and shapes mismatch (X @ W.T fails).
# BREAK IT: in gptq() skip the damping line - the Cholesky fails with 'linear algebra error: ... not positive-definite' because H is singular.
