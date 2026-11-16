# Step 06 — GPTQ and AWQ from scratch

## The story so far
[Step 05](../05_activation_quant_and_calibration/WALKTHROUGH.md) showed that activations carry outlier channels and that a calibration set tells you what ranges to expect. It also showed that weight-only quantization (W4A16, W8A16) is the safe, widely deployed choice because weights are tame. But plain rounding to 4 bits (RTN) still hurts. This step makes the rounding smarter by using the calibration data.

## The pain point
Round-to-nearest treats every weight as equally important and every rounding error as independent. At 4 bits perplexity rises noticeably; at 3 bits the model is badly damaged. Yet the errors do not all matter equally: an error on a weight that multiplies a huge activation channel changes the output a lot; an error that can be compensated by another weight in the same row costs nothing.

## The spark
Quantization is an optimisation problem per layer: minimise `||X W^T - X Wq^T||^2` over the calibration inputs X. Two ideas solve it differently. **GPTQ** (2022): round one column at a time and push each column's rounding error onto the not-yet-rounded columns using the inverse of the layer Hessian `H = 2 X^T X`. **AWQ** (2023): do not change how rounding works; instead scale important input channels (those with large activations) up before rounding so they get a relatively finer grid.

## Goal
On the 24 real linear layers of distilgpt2 implement RTN, simplified GPTQ and simplified AWQ with the same asymmetric g=128 grid, compare layer-output SQNR on held-out tokens and full-model perplexity at 4-bit and 3-bit, and learn what the real libraries do.

## The theory (read this before the interview)
**Layer-wise objective.** For a linear layer `y = W x` (nn.Linear layout: W is (out, in)) and calibration inputs X (n x in): `min_Wq ||X W^T - X Wq^T||_F^2`. Each output row is independent and shares the same Hessian `H = 2 X^T X / n` (in x in).

**GPTQ** (OBS / OBQ lineage, made fast). For a row w, if the weight at column i is rounded with error `e = w_i - q_i`, the optimal compensation for the remaining weights is `delta = -(e / Hinv[i,i]) * Hinv[i, :]`. Doing this column after column with one shared ordering means a single Cholesky factor of `H^-1` serves every step.

```
H = 2 X^T X / n  +  damp * mean(diag H) * I        (damping: X^T X is singular when n < in)
U = upper-Cholesky(H^-1)
for each block of 128 columns:
    scale, zero from the (already corrected) weights of the block
    for each column i in the block:
        q   = quantize(W[:, i]);  err = (W[:, i] - q) / U[i,i]
        W[:, i+1:block_end] -= err * U[i, i+1:block_end]
    W[:, block_end:] -= Err_block @ U[block, block_end:]      <- "lazy batch update"
```
Real implementations also offer act-order (quantize high-Hessian-diagonal columns first), per-group scales, and fused int4 CUDA kernels.

**AWQ** (Activation-aware Weight Quantization). Observation: ~1% of input channels (those with large average |x|) are "salient". Keeping them in fp16 works but is a mixed-precision kernel. Instead scale: `Wq' = Q(W * s) / s` with `s_j = mean|X_j|^alpha`; equivalently multiply W by s and divide X by s (the division is folded into the preceding LayerNorm or linear layer, so it is free). Search alpha in [0, 1) by grid (alpha = 0 is RTN) minimising output error on calibration data. The real AWQ also searches a weight-clipping value per group.

**What GPTQ and AWQ are not**: they do not change the stored format. Both output ordinary int4 weights + group scales (+ zeros); the difference is only which integers were chosen. Data requirement: GPTQ needs enough calibration tokens to estimate H (hundreds to thousands of sequences in practice) and can overfit a narrow calibration domain; AWQ only needs per-channel activation magnitudes, so it is more robust to the calibration set.

**HQQ** (Half-Quadratic Quantization): calibration-free. It optimises scale/zero-point of each group to minimise a robust (sparsity-promoting) reconstruction loss |W - Wq| with a half-quadratic solver. No data, minutes for a 70B model; quality between RTN and GPTQ/AWQ.

**Tools** (see `reference_tools.py`, REFERENCE, not run here):
- **AutoGPTQ / GPTQModel**: implement GPTQ with CUDA int4 kernels and checkpoint format; GPTQModel is the maintained successor family (check project status).
- **AutoAWQ**: implements AWQ; the standalone repository has been reported as archived, with AWQ support continuing through other projects (check status before depending on it).
- **llm-compressor** (vLLM project): recipes (`GPTQModifier`, `AWQModifier`, FP8/INT8 `QuantizationModifier`) that produce checkpoints in the compressed-tensors format served directly by vLLM; the common tool in 2026 for vLLM-targeted checkpoints.
- **bitsandbytes**: no calibration; NF4/FP4 (QLoRA) and LLM.int8() with on-the-fly dequantization; great for fine-tuning, slower for serving.
- **torchao**: PyTorch-native quantize_ API (int4/int8 weight-only, dynamic activation, float8) with tensor subclasses and torch.compile kernels.

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| RTN | round-to-nearest, naive quantization, data-free rounding | Yes | Each weight independently to the nearest grid point. |
| GPTQ | OBQ-style second-order PTQ, "gptq" checkpoints | Yes | Rounds column by column and compensates using H^-1 of calibration activations. |
| AWQ | activation-aware weight quantization | Yes | Per-input-channel scale chosen from activation magnitudes; no weight updates. |
| Hessian H | 2 X^T X, input covariance, second-moment matrix | In this context yes | Second derivative of the layer output error with respect to a weight row. |
| Damping | percdamp, ridge, regularisation | Yes | Add a fraction of the mean diagonal to H so it can be inverted. |
| Calibration data | calibration set, calib samples | Yes | The few hundred sequences whose activations estimate H or the channel scales. |
| W4A16 g128 | int4 weight-only, group size 128 | Yes | Stored int4 with one fp16 scale per 128 weights; activations stay in 16 bit. |
| Act-order | desc_act | Yes | Quantize columns in descending Hessian-diagonal order (not used here). |
| HQQ | half-quadratic quantization | Different method | Needs no calibration data. |

## Run it
```bash
cd 21_llm_inference_engineering/06_gptq_awq_from_scratch
python main.py                  # all 6 blocks of distilgpt2
LAYERS=2 python main.py         # quick: only the first two blocks
python reference_tools.py       # prints REFERENCE recipes only (never runs them)
# from the repo root
HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/06_gptq_awq_from_scratch/main.py
```
Runtime about 30 s of quantization and about 1 minute wall clock in total on this machine (varies; measured peak RAM 1.6 GB). Simplifications: layers are calibrated independently from the fp32 model's activations (the paper feeds each block the already-quantized earlier blocks); no act-order; AWQ searches only the scale (no clipping search). The calibration set is about 700 tokens, fewer than the 768 or 3072 input features of the layers, so H is rank deficient and damping matters.

## Code, step by step
Shared helpers (`ql.qdq`, `ql.perplexity`, `ql.collect_inputs`, ...) are in [`../quantlib.py`](../quantlib.py). Remember: GPT-2 `Conv1D` weights are stored (in, out); `ql.get_w_out_in` hands us the (out, in) view that GPTQ/AWQ code assumes.

### 1. Header, imports, calibration data  (`main.py` lines 1–39)

```python
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
```

- **In simple words:** Load distilgpt2, collect the inputs of every linear layer for ~600 calibration tokens (ticket calibration lines + a few general paragraphs) and for held-out evaluation tokens.
- **Exactly:** `X_cal[name]` is an array `(n_tokens, in_features)` of fp32 activations that the layer received when the unquantized model processed the calibration text (hooks, see 05). `LAYERS` limits how many of the 6 blocks are quantized. Note: with n_tokens (~600) smaller than in_features (768 or 3072) the matrix X^T X has rank at most n_tokens, hence is singular; GPTQ's damping handles that.
- **Why it is here:** GPTQ and AWQ are DATA-DEPENDENT: they minimise the error on real layer inputs, not on weights alone.

### 2. Group quantizer parameters  (`main.py` lines 42–56)

```python
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
```

- **In simple words:** For each group of 128 weights per output row, find the minimum and maximum and derive a scale and zero-point (asymmetric 3- or 4-bit grid).
- **Exactly:** `lo`/`hi` are clamped to contain 0; `scale = (hi-lo)/(2^bits-1)`, `zero = round(-lo/scale)`. `quant_col` returns `(clip(round(w/scale)+zero, 0, 2^bits-1) - zero) * scale`, i.e. the de-quantized value of one column of shape (out, 1).
- **Why it is here:** Gives the same asymmetric min-max group grid that `ql.qdq(.., symmetric=False, gran=128)` uses, so RTN, AWQ and GPTQ differ only in HOW values are rounded, not in the grid.

### 3. RTN baseline  (`main.py` lines 59–61)

```python
# [[2: RTN baseline - just round every weight, no data]]
def rtn(W, bits, group=GROUP):
    return ql.qdq(W, bits, symmetric=False, gran=group)            # same min-max asymmetric groups as above (numpy)
```

- **In simple words:** Round each weight to the nearest grid point independently, with no use of data.
- **Exactly:** One call to `ql.qdq` with groups of 128 along the input dimension (last axis of the (out, in) matrix).
- **Why it is here:** The reference every method must beat; at 4 bits it is already decent, at 3 bits it breaks.

### 4. Simplified GPTQ  (`main.py` lines 64–91)

```python
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
```

- **In simple words:** Quantize one input column at a time; after rounding a column, adjust the remaining columns so that the layer's output on the calibration data changes as little as possible.
- **Exactly:** Objective per layer: minimise ||X W^T - X Wq^T||^2. `H = 2 X^T X / n` is the Hessian of that objective with respect to one row of W. `damp * mean(diag H)` is added to the diagonal because H is singular. `torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)` gives an upper-triangular factor U of H^-1 (the trick that makes the column-by-column update numerically stable, from the paper). Loop `i`: round column `i` to `q`, error `err = (w-q)/U[i,i]`, then `W[:, i+1:] -= err @ U[i:i+1, i+1:]` pushes the error onto later columns. A new scale/zero is computed from the already-corrected weights at the start of each group.
- **Why it is here:** This is the core of GPTQ (Frantar et al., 2022; an optimal-brain-surgeon style second-order method). Real implementations add lazy block updates, optional act-order and fused int4 kernels, but the math is these 20 lines.

### 5. Simplified AWQ  (`main.py` lines 94–109)

```python
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
```

- **In simple words:** Find a per-input-channel scale so that channels with large activations get more precision, by trying a grid of exponents.
- **Exactly:** `x_mean` = average |activation| per input channel. For alpha in 20 steps from 0 to 0.95: `s = x_mean**alpha`, normalised by `sqrt(max*min)`, quantize `W * s` with the group grid then divide by `s` (equivalent to multiplying X by 1/s), measure the layer output error on calibration data, keep the best alpha. alpha=0 is RTN, so AWQ can never be worse than RTN on the calibration data.
- **Why it is here:** AWQ (2023) is the 'protect the salient channels' method: no weight updates, no Hessian, only a scale search; fast and robust to the calibration set.

### 6. Run all three on every layer and measure layer-output error  (`main.py` lines 112–137)

```python
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
```

- **In simple words:** For each of the 24 layers and for 4-bit and 3-bit, quantize with the three methods and report the output SQNR on held-out tokens.
- **Exactly:** `layer_sqnr` compares `X_eval @ W.T` with `X_eval @ Wq.T`. `results[(bits, method)][layer]` stores every quantized matrix for the perplexity step. The ratio of calibration objectives (RTN error / GPTQ error) is printed too: GPTQ optimises that objective directly, so it should always be above 1.
- **Why it is here:** Layer error is cheap and shows who optimises what; perplexity (next block) shows whether it matters end to end.

### 7. Full-model perplexity  (`main.py` lines 140–155)

```python
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
```

- **In simple words:** Write each method's quantized matrices back into a copy of the model and measure perplexity on held-out ticket text.
- **Exactly:** `ql.set_w_out_in` transposes (out, in) back into the GPT-2 Conv1D layout (in, out) and assigns a new tensor to `mod.weight.data`. `ql.perplexity` is exp of the mean cross-entropy over 128-token windows.
- **Why it is here:** The only number that matters for a deployed model; differences between 4-bit methods are small, 3-bit separates them.

### 8. TRY IT / BREAK IT  (`main.py` lines 157–161)

```python
# TRY IT: set GROUP = 64 (more scales, better quality) or 32 and watch the RTN-vs-GPTQ gap shrink.
# TRY IT: change damp=0.01 to 1e-6 and then to 1.0 in gptq(): too little -> unstable/noisy, too much -> behaves like RTN.
# TRY IT: calibrate on fewer tokens (slice X_cal) or on GENERAL_TEXT only and check eval perplexity on tickets.
# BREAK IT: in get_w_out_in() forget the transpose (use mod.weight directly) - the groups then run along the wrong axis and shapes mismatch (X @ W.T fails).
# BREAK IT: in gptq() skip the damping line - the Cholesky fails with 'linear algebra error: ... not positive-definite' because H is singular.
```

- **In simple words:** Experiments to run after reading.
- **Exactly:** Comments only.
- **Why it is here:** Group size, damping and calibration set are the knobs of every GPTQ run in practice.


## Syntax explained

| Syntax | Meaning |
|---|---|
| `W[:, i:i + 1]` | A column kept 2-D with shape (out, 1) so it broadcasts against scale and zero of shape (out, 1). |
| `err @ Hinv[i:i + 1, i + 1:b1]` | Outer product: (out,1) @ (1, k) gives an (out, k) correction. |
| `torch.linalg.cholesky(A, upper=True)` | Factor A = U^T U with U upper triangular. |
| `torch.cholesky_inverse(L)` | Inverse of A given its lower Cholesky factor L (cheaper and more stable than `inverse`). |
| `H[dead, dead] = 1.0` | Boolean-mask indexing on both axes sets the diagonal entries of dead channels. |
| `np.linspace(0, 1, grid, endpoint=False)` | 20 alpha values 0, 0.05, ..., 0.95. |
| `results.setdefault(key, {})[name] = Wq` | Create the inner dict on first use, then store. |

## Libraries & functions

| Function | What it does | Internally |
|---|---|---|
| `ql.collect_inputs` | Layer inputs for a text | Forward hooks on the 24 Conv1D modules, `torch.cat` per layer |
| `ql.qdq(W, bits, False, g)` | Group asymmetric RTN | See step 04 |
| `torch.linalg.cholesky` | Cholesky factor | LAPACK `potrf`; fails with an error if the matrix is not positive definite |
| `ql.set_w_out_in` | Write a quantized matrix back | Transpose to (in, out), assign `mod.weight.data` |
| `ql.perplexity` | Held-out perplexity | exp of mean cross-entropy over 128-token windows |

## Execution trace (real run)
```
quantizing 24 linear layers; calibration tokens 712, eval tokens 727

== layer-wise results: output SQNR (dB) on HELD-OUT tokens (higher = better) ==
  W4 h.0.mlp.c_fc           RTN  25.42 | AWQ  26.20 (alpha=0.30) | GPTQ  31.22
  W4 h.5.mlp.c_proj         RTN  16.17 | AWQ  18.14 (alpha=0.40) | GPTQ  23.03
  W4 g128  MEAN over 24 layers: RTN 22.17 | AWQ 23.16 | GPTQ 27.54 dB   (GPTQ calib-objective is 14.6x lower than RTN)
  W3 h.0.mlp.c_fc           RTN  18.87 | AWQ  19.58 (alpha=0.30) | GPTQ  24.50
  W3 h.5.mlp.c_proj         RTN   9.68 | AWQ  11.01 (alpha=0.25) | GPTQ  16.32
  W3 g128  MEAN over 24 layers: RTN 15.79 | AWQ 16.66 | GPTQ 20.79 dB   (GPTQ calib-objective is 12.7x lower than RTN)
  (quantization time 30 s)

== perplexity on held-out ticket lines (lower = better) ==
  fp32                         :    25.44
  W4 g128 RTN   (4.125 bpw):    29.40
  W4 g128 AWQ   (4.125 bpw):    28.42
  W4 g128 GPTQ  (4.125 bpw):    26.16
  W3 g128 RTN   (3.125 bpw):    42.24
  W3 g128 AWQ   (3.125 bpw):    39.14
  W3 g128 GPTQ  (3.125 bpw):    29.00
```

How to read it:
- **GPTQ wins clearly at the layer level**: mean output SQNR over 24 layers at 4-bit g128 is RTN 22.17, AWQ 23.16, GPTQ 27.54 dB (the same gap, +5 dB, holds at 3-bit: 15.79 / 16.66 / 20.79). GPTQ's calibration objective is 14.6x lower than RTN's, as it must be since it optimises that objective directly.
- **Perplexity follows**: fp32 25.44; at 4-bit g128 RTN 29.40, AWQ 28.42, GPTQ 26.16 (only 0.7 above fp32). At 3-bit RTN 42.24, AWQ 39.14, GPTQ 29.00. The difference between methods is small at 4 bits and large at 3 bits, which is why 4-bit RTN is acceptable for big models and 3-bit needs GPTQ-like methods.
- **AWQ here gives a modest gain** (alpha 0.25-0.40 found) because distilgpt2 has weak activation outliers (max/median up to about 59 in step 05) and our simplified AWQ lacks the clipping search. GPTQ gains more in this setup. Do not generalise: on large models with strong salient channels AWQ is competitive with GPTQ and cheaper and less prone to calibration overfitting.
- **Caveat**: tiny model (82M), tiny perplexity text (727 tokens), calibration partly from the same ticket domain. Treat the ordering as meaningful and the exact numbers as illustrative.

## Experiments (change ONE thing)

| Change | What you will see | Why |
|---|---|---|
| `GROUP = 64` | All methods improve; the GPTQ-RTN gap shrinks | More scales means RTN is already finer. |
| `damp=1e-6` in `gptq` | Possible instability / worse held-out SQNR | Near-singular H amplifies compensation noise. |
| `damp=1.0` | GPTQ approaches RTN | A huge diagonal makes H almost identity, so no cross-column compensation. |
| Calibrate on `GENERAL_TEXT` only | Ticket perplexity worse than with ticket-domain calibration | GPTQ fits the calibration distribution. |
| `LAYERS=2` | Faster; partial model | Only 2 blocks are quantized. |

## Debugging practice
**BREAK IT:** in `get_w_out_in` return `mod.weight` without the `.T`. The code then treats (in, out) as (out, in): groups run along the wrong axis and `X @ W.T` fails with a shape error (`mat1 and mat2 shapes cannot be multiplied`) or silently gives garbage when in == out. Fix: transpose Conv1D weights.
**BREAK IT:** delete the damping line in `gptq`. Cholesky raises `linalg.cholesky: The factorization could not be completed because the input is not positive-definite` because X^T X has rank <= n_tokens < in_features. Fix: damping (`percdamp` in the libraries) or more calibration tokens.

## Common mistakes
- Evaluating on the calibration tokens (hides overfitting); we use held-out tokens.
- Forgetting that GPTQ and AWQ output the same format as RTN; the speed-up comes from the int4 kernel, not from the algorithm.
- Quantizing the embedding / lm_head with the same recipe (usually left in 16 bit).
- Calibrating with a tiny or off-domain set and trusting the result.
- Comparing methods at different group sizes.
- Assuming AWQ and GPTQ package statuses without checking the projects' current release pages.

## Rebuild checklist
1. Load activations per layer with hooks; split calibration from evaluation tokens.
2. Write `group_params` and `quant_col` (asymmetric min-max).
3. RTN with `ql.qdq(..., gran=128)`.
4. GPTQ: H, dead columns, damping, Cholesky of H^-1, block loop with rank-1 update and lazy update.
5. AWQ: alpha grid, scale `x_mean**alpha` normalised, quantize `W*s`, divide by `s`, pick lowest output error.
6. Compare layer SQNR on held-out tokens, then write weights back and compute perplexity.

## Say it aloud
- "RTN rounds every weight independently; GPTQ rounds column by column and uses the inverse Hessian of the calibration activations to push each rounding error onto the remaining weights, minimising the layer output error."
- "AWQ keeps standard rounding but scales salient input channels up by mean|x|^alpha before quantization and folds the inverse scale into the previous op; no weight updates, so it is more robust to the calibration set."
- "On distilgpt2 at 3 bits g128 perplexity went RTN 42, AWQ 39, GPTQ 29 against 25.4 in fp32; at 4 bits all three are within a few points."
- "GPTQ and AWQ produce the same int4 format; for vLLM I would make the checkpoint with llm-compressor, and HQQ or bitsandbytes when I have no calibration data."

## The new problem (cliffhanger)
Weights are now small, but at long context or high batch the KV cache, not the weights, fills the GPU (recall the formula in [step 03](../03_gpu_memory_calculator/WALKTHROUGH.md)). Can the cache be quantized the same way, given that it is produced on the fly and K behaves differently from V? That is [step 07: KV-cache quantization](../07_kv_cache_quantization/WALKTHROUGH.md).
