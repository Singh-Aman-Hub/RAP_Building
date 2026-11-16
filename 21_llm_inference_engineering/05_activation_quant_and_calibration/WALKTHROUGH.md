# Step 05 — Activation quantization and calibration

## The story so far
[Step 04](../04_quantization_math/WALKTHROUGH.md) built the quantizer and ended on the observation that weights are static and tame. This step quantizes the other operand of every matmul, the activations, and meets the problem that makes LLM quantization hard. (Prerequisite: [05_llm_parameters/05_quantization](../../05_llm_parameters/05_quantization/WALKTHROUGH.md) for symmetric int8 and group-wise scales.)

## The pain point
Weight-only quantization (W4A16, W8A16) shrinks memory and the bytes read per decoded token, but the matmul still runs in 16-bit. To use fast integer or FP8 tensor cores you must also quantize activations (W8A8, FP8). Activations change with every input and a few channels are far larger than the rest. One scale for the whole tensor wastes most of the 256 levels on the giants and rounds everything else to a few codes.

## The spark
Treat the outlier problem and the range problem separately. Outliers: give each token (or channel) its own scale (dynamic, per-token), move the difficulty into the weights (SmoothQuant), or handle the outlier channels separately in higher precision (LLM.int8()). Ranges: estimate them from a calibration set (static) or compute them at run time (dynamic), with a range rule (min-max, percentile, MSE). And if PTQ is not enough, train with the quantizer in the loop (QAT).

## Goal
Measure real activation outliers in distilgpt2, compare W4A16 / W8A16 / W8A8 per-tensor / per-token / per-channel on one layer, try LLM.int8()-style decomposition, compare calibration rules and calibration sets, apply SmoothQuant, show PTQ vs QAT (straight-through estimator) on a tiny MLP, and measure perplexity for fake-quantized W8A8 variants of the whole model.

## The theory (read this before the interview)
**Naming**: `W8A8` = 8-bit weights, 8-bit activations; `W4A16` = 4-bit weights, 16-bit activations (weight-only: dequantize on the fly, compute in 16-bit); `FP8` usually means FP8 weights and activations (W8A8 with floats), hardware path on Hopper/Ada/Blackwell GPUs.

**Why activations are harder**: (1) their values depend on the input, so ranges must be predicted or computed on the fly; (2) LLM activations have systematic outlier channels (a handful of hidden dimensions with magnitudes 10-100x the median, appearing for every token, especially after LayerNorm) while weights are close to Gaussian. Per-channel activation scales cannot be applied after an integer GEMM (the scale would have to be inside the sum over channels), so engines use per-tensor or per-token activation scales and per-channel weight scales: `y = (s_x[token] * s_w[channel]) * (q_x @ q_w^T)`.

**Dynamic vs static activation quantization**: static = ranges fixed offline from a calibration set (no run-time reduction, fastest, fragile when real data exceeds the range); dynamic = compute absmax of the actual tensor (per tensor or per token) at run time (adaptive, small extra pass).

**Calibration**: run a few hundred representative samples and record activation statistics. Range rules: min-max (never clips, hostage to one outlier), percentile (clip the top 0.1%), MSE / entropy (pick the range that minimises error). Too few samples or the wrong domain gives ranges that do not cover the traffic.

**SmoothQuant** (W8A8 for LLMs): `y = X W^T = (X diag(1/s)) (W diag(s))^T` with `s_j = max|X_j|^alpha / max|W_j|^(1-alpha)`. Outlier activation channels shrink, the matching weight columns grow; alpha (typically 0.5) balances the difficulty. The `1/s` is folded into the preceding LayerNorm, so it is free at inference. Offline, once, using calibration activations.

**LLM.int8()** (bitsandbytes): find the few input channels with magnitude > 6 (about 0.1% of dimensions but systematic), multiply those in fp16, everything else in int8 with per-row/per-column scales, sum the two parts. Accurate but needs a mixed-precision kernel and was slower than plain fp16 in practice.

**PTQ vs QAT**: PTQ (post-training quantization, offline quantization) quantizes a finished model, needs no training, costs minutes. QAT (quantization-aware training) keeps training while the forward pass uses quantized weights/activations; since `round()` has zero gradient almost everywhere, backward uses the straight-through estimator (STE): treat round as identity. QAT recovers accuracy at very low bit widths but needs training data and compute (QLoRA-style fine-tuning is related but quantizes only the frozen base).

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| PTQ | post-training quantization, offline quantization | Yes | Quantize a trained model without training. |
| QAT | quantization-aware training, fake-quant training | Yes | Train with simulated quantization in the forward pass. |
| STE | straight-through estimator | Yes | Backward pretends round() is identity. |
| Weight-only quantization | W4A16, W8A16, WxA16 | Yes | Activations stay 16-bit; memory-bound savings. |
| W8A8 | int8 weights and activations, "int8 inference" | Yes | Both operands int8, integer tensor core GEMM. |
| FP8 W8A8 | float8 inference, e4m3 activations | Related | Same idea, float grid; needs hardware support. |
| Dynamic quantization | run-time / on-the-fly activation scaling | Yes | Range computed from the live tensor. |
| Static quantization | calibrated, offline ranges | Yes | Range fixed from calibration data. |
| Calibration set | calibration data, calib samples | Yes | Representative inputs used to estimate ranges (or Hessians in step 06). |
| Per-token | row-wise, dynamic per-row | Yes (activations) | One scale per token row. |
| SmoothQuant | activation-to-weight difficulty migration | Yes | Equivalent rescaling before quantization. |
| LLM.int8() | outlier-aware mixed-precision decomposition | Yes | Outlier channels in fp16, rest int8. |

## Run it
```bash
cd 21_llm_inference_engineering/05_activation_quant_and_calibration
python main.py
BITS=3 python main.py          # QAT/PTQ bit width for the tiny MLP
# from the repo root
HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/05_activation_quant_and_calibration/main.py
```
Runtime about 15 s of CPU (measured user time 11 s; peak RAM 1.6 GB) plus any wait for the shared model lock; wall time varies. Uses cached distilgpt2; no downloads. Everything is fake quantization (float simulation).

## Code, step by step
Helpers (`ql.collect_inputs`, `ql.qdq`, `ql.perplexity`, ...) are in [`../quantlib.py`](../quantlib.py).

### 1. Header, imports, and collecting real activations  (`main.py` lines 1–38)

```python
"""
GOAL:  Understand why quantizing ACTIVATIONS is harder than quantizing weights, and the standard fixes: calibration
       (min-max / percentile / MSE), SmoothQuant, outlier decomposition (LLM.int8), and QAT vs PTQ.  Measured on real distilgpt2.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/05_activation_quant_and_calibration/main.py
EXPECTED OUTPUT (shortened):
    real activation outliers: per-channel max/median ratio of layer inputs reaches ~60x (LayerNorm outputs feeding c_attn / c_fc)
    per-tensor W8A8 on that layer is far worse than per-token; SmoothQuant and LLM.int8 decomposition recover most of it
    calibration: MSE range beats min-max; 99.9th-percentile clipping HURTS (it clips the one meaningful outlier channel)
    QAT (straight-through estimator) beats PTQ at 2 bits on a tiny MLP
    perplexity: fp32 vs W8A16 vs several W8A8 variants (a table)
NEEDS: numpy, torch, transformers (distilgpt2 cached, no downloads).  ~40 s on CPU.

Builds on 04_quantization_math (the quantizer itself) and 05_llm_parameters/05_quantization (weight-only int8/int4).
Everything is FAKE quantization (quantize then de-quantize in float) so that errors can be measured on a laptop.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import quantlib as ql

torch.manual_seed(0)
np.random.seed(0)
tok, model = ql.load_distilgpt2()
calib_text, eval_text = ql.corpus_split(0.4)        # disjoint halves of the PixelNote ticket corpus
layers = dict(ql.conv1d_layers(model))
X_cal = ql.collect_inputs(model, tok, calib_text)    # {layer: (n_tokens, in_features)} activations seen by each linear layer
X_eval = ql.collect_inputs(model, tok, eval_text)
print(f"calibration tokens {len(X_cal['transformer.h.0.attn.c_attn'])}, eval tokens {len(X_eval['transformer.h.0.attn.c_attn'])}")


def sqnr_y(Y, Yh):
    return ql.metrics(Y, Yh)["sqnr_db"]
```

- **In simple words:** Load distilgpt2 and record, for every linear layer, the numbers that flow INTO it for two disjoint pieces of text (calibration and evaluation).
- **Exactly:** `ql.corpus_split(0.4)` splits the 39 ticket lines of `06_llm_training_and_lora/data/tiny_corpus.txt` into a calibration part (40%) and a held-out part (60%). `ql.collect_inputs` registers a PyTorch forward hook (a function PyTorch calls with `(module, input, output)` every time the layer runs) on the 24 `Conv1D` layers and stacks the inputs into a `(n_tokens, in_features)` array per layer. `sqnr_y` is a one-line wrapper around `ql.metrics`.
- **Why it is here:** Activation quantization cannot be studied on weights alone; we need real activations, and the held-out split makes calibration honest (no peeking).

### 2. How bad are real activation outliers?  (`main.py` lines 41–53)

```python
# [[1: how bad are real activation outliers?]]
# Per input channel j: absmax over tokens.  Ratio = (largest channel absmax) / (median channel absmax).
# One huge channel forces a per-tensor scale that wastes the integer grid on every other channel.
print("\n== 1. Real activation outliers in distilgpt2 (inputs of each linear layer) ==")
print(f"  {'layer':<28}{'max|X|':>9}{'median ch absmax':>18}{'max/median':>12}{'worst channel':>15}")
for name in ("transformer.h.0.attn.c_attn", "transformer.h.3.attn.c_attn", "transformer.h.3.mlp.c_fc", "transformer.h.3.mlp.c_proj", "transformer.h.5.mlp.c_fc"):
    ch = np.abs(X_cal[name]).max(axis=0)                      # shape (in_features,): per-channel absmax
    print(f"  {name[12:]:<28}{ch.max():9.2f}{np.median(ch):18.3f}{ch.max() / np.median(ch):12.1f}{int(ch.argmax()):15d}")
ratios = [np.abs(X_cal[n]).max(0).max() / np.median(np.abs(X_cal[n]).max(0)) for n in X_cal]
print(f"  over all {len(ratios)} linear layers: max/median ratio ranges {min(ratios):.1f} .. {max(ratios):.1f}")
W = ql.get_w_out_in(layers["transformer.h.3.mlp.c_fc"])      # (out, in)
print(f"  weights of the same layer: absmax {np.abs(W).max():.2f}, per-input-channel absmax max/median = "
      f"{(np.abs(W).max(0).max() / np.median(np.abs(W).max(0))):.1f}  (weights are far tamer than activations)")
```

- **In simple words:** For each input channel we take the largest absolute value over all tokens and compare the biggest channel with the typical one.
- **Exactly:** `np.abs(X).max(axis=0)` has shape `(in_features,)`: one absmax per channel. The ratio `max/median` of that vector measures how dominated the layer is by a single channel. Weights get the same statistic for comparison.
- **Why it is here:** One channel 10-1000x bigger than the rest is the reason a single per-tensor int8 scale for activations wastes almost all of its 256 levels.

### 3. Weight-only vs weight+activation on one layer  (`main.py` lines 55–82)

```python
# [[2: weight-only vs weight+activation quantization on one layer]]
LN = "transformer.h.3.mlp.c_fc"
Xe, Xc = X_eval[LN], X_cal[LN]
Y = Xe @ W.T                                                 # reference fp32 output (bias ignored): (n_tokens, out)


def act_qdq(X, bits, mode, amax=None):
    """Quantize activations X (n_tokens, in). mode: 'tensor' | 'token' | 'channel' (dynamic) | 'static' (fixed range `amax`)."""
    qmax = 2 ** (bits - 1) - 1
    if mode == "static":
        s = amax / qmax
    elif mode == "tensor":
        s = np.abs(X).max() / qmax
    elif mode == "token":
        s = np.abs(X).max(axis=1, keepdims=True) / qmax        # one scale per row = per token
    else:
        s = np.abs(X).max(axis=0, keepdims=True) / qmax        # one scale per input channel (cannot be folded into the matmul!)
    return np.clip(np.round(X / s), -qmax, qmax) * s


print("\n== 2. Output SQNR (dB) of layer", LN[12:], "- higher is better ==")
W4 = ql.qdq(W, 4, True, 128)
W8 = ql.qdq(W, 8, True, "row")
print(f"  W4A16 (int4 g128 weights, fp acts)      : {sqnr_y(Y, Xe @ W4.T):6.2f}")
print(f"  W8A16 (int8 per-row weights, fp acts)   : {sqnr_y(Y, Xe @ W8.T):6.2f}")
for mode in ("tensor", "token", "channel"):
    print(f"  W8A8, activations int8 per-{mode:<8}     : {sqnr_y(Y, act_qdq(Xe, 8, mode) @ W8.T):6.2f}")
print(f"  W8A4, activations int4 per-token        : {sqnr_y(Y, act_qdq(Xe, 4, 'token') @ W8.T):6.2f}")
```

- **In simple words:** Quantize only the weights (W4A16, W8A16), or both (W8A8) with the activation scale chosen per tensor, per token or per channel.
- **Exactly:** `act_qdq` quantizes an activation matrix `X (n_tokens, in)`: `tensor` = one scale for everything, `token` = one scale per row (computed at run time, cheap), `channel` = one scale per column (accurate but cannot be applied after an integer matmul, because the scale would sit inside the sum over channels). `Y = Xe @ W.T` is the fp32 reference; each variant is compared to it.
- **Why it is here:** Shows the ranking W8A16 > per-token W8A8 > per-tensor W8A8, and why engines use per-tensor/per-token activation scales plus per-channel weight scales.

### 4. LLM.int8() outlier decomposition  (`main.py` lines 84–91)

```python
# [[3: LLM.int8() outlier decomposition]]
# Keep the few outlier channels (|x| > threshold) in high precision, quantize the rest.  y = X_out @ W_out + Q(X_rest) @ Q(W_rest)
print("\n== 3. LLM.int8()-style mixed-precision decomposition (same layer) ==")
thr = 6.0                                                    # bitsandbytes default outlier threshold
out_ch = np.where(np.abs(Xc).max(axis=0) > thr)[0]           # outlier channels decided on calibration data (LLM.int8 decides per batch)
rest = np.setdiff1d(np.arange(Xe.shape[1]), out_ch)
Ydec = Xe[:, out_ch] @ W[:, out_ch].T + act_qdq(Xe[:, rest], 8, "token") @ ql.qdq(W[:, rest], 8, True, "row").T
print(f"  outlier channels (calib max > {thr}): {len(out_ch)} of {Xe.shape[1]} -> kept in fp; SQNR {sqnr_y(Y, Ydec):.2f} dB (per-token W8A8 w/o decomposition: {sqnr_y(Y, act_qdq(Xe, 8, 'token') @ W8.T):.2f})")
```

- **In simple words:** Pull the few outlier channels out of the int8 matmul and do them in floating point.
- **Exactly:** Channels whose calibration absmax exceeds 6.0 (the bitsandbytes default threshold) are kept in fp: `y = X[:, outliers] @ W[:, outliers].T + Q(X[:, rest]) @ Q(W[:, rest]).T`. (The real LLM.int8() picks outlier columns per batch at run time; we use calibration data to keep the demo simple.)
- **Why it is here:** The first practical fix for the outlier problem (2022); SmoothQuant later avoided the slower mixed-precision kernel.

### 5. Static vs dynamic ranges and calibration rules  (`main.py` lines 93–114)

```python
# [[4: static vs dynamic, and how the range is calibrated]]
# static: one fixed activation range decided OFFLINE from a calibration set (fast, but must generalise);
# dynamic: range computed at RUN TIME from the actual tensor (adaptive, small extra cost).
def static_range(Xcal, method, bits=8):
    a = np.abs(Xcal)
    if method == "minmax":
        return a.max()
    if method == "percentile":
        return np.percentile(a, 99.9)
    grid = np.linspace(0.05, 1.0, 40) * a.max()              # "mse": try 40 candidate ranges, keep the lowest reconstruction MSE
    errs = [np.mean((Xcal - act_qdq(Xcal, bits, "static", g)) ** 2) for g in grid]
    return grid[int(np.argmin(errs))]


print("\n== 4. Static calibration: range method x calibration set (output SQNR on EVAL tickets, int8 per-tensor acts) ==")
gen_X = ql.collect_inputs(model, tok, ql.GENERAL_TEXT)[LN]
print(f"  dynamic per-tensor (no calibration needed): {sqnr_y(Y, act_qdq(Xe, 8, 'tensor') @ W.T):.2f} dB")
print(f"  {'calibration set':<26}{'minmax':>9}{'percentile':>12}{'mse':>9}")
for label, Xcal in (("tickets, 8 tokens", Xc[:8]), ("tickets, 32 tokens", Xc[:32]), ("tickets, 128 tokens", Xc[:128]),
                    (f"tickets, all {len(Xc)} tokens", Xc), (f"general text, {len(gen_X)} tokens", gen_X)):
    res = [sqnr_y(Y, act_qdq(Xe, 8, "static", static_range(Xcal, m)) @ W.T) for m in ("minmax", "percentile", "mse")]
    print(f"  {label:<26}{res[0]:9.2f}{res[1]:12.2f}{res[2]:9.2f}")
```

- **In simple words:** Static quantization fixes the activation range offline from a calibration set; dynamic computes it at run time. We compare three rules for the static range and different calibration sets.
- **Exactly:** `static_range` returns the clipping value A for symmetric int8: min-max (absmax), percentile (99.9th percentile of |x|), or MSE (grid of 40 candidate A's, keep the one with the lowest reconstruction MSE). `act_qdq(X, 8, 'static', A)` then uses `s = A/127` and clips anything beyond A. Calibration sets differ in size (8..all tokens) and domain (tickets vs general English).
- **Why it is here:** This is exactly what 'calibration' means in GPTQ/AWQ/llm-compressor pipelines: a small dataset to estimate ranges (and Hessians); too small or from the wrong domain gives a bad range.

### 6. SmoothQuant  (`main.py` lines 116–138)

```python
# [[5: SmoothQuant - move the activation outliers into the weights]]
# y = X W = (X diag(1/s)) (diag(s) W).  Choose s_j = max|X_j|^a / max|W_j|^(1-a): outlier channels of X shrink, the matching
# weight columns grow.  Mathematically identical in fp, but both sides become easier to quantize.
print("\n== 5. SmoothQuant on the same layer (per-tensor W8A8: the cheap, fast-kernel setting) ==")
xmax = np.abs(Xc).max(axis=0)                                # per-input-channel absmax of ACTIVATIONS (calibration)
wmax = np.abs(W).max(axis=0)                                 # per-input-channel absmax of WEIGHTS: W is (out, in) so axis 0 -> shape (in,)


def smooth(alpha):
    s = np.maximum(xmax, 1e-5) ** alpha / np.maximum(wmax, 1e-5) ** (1 - alpha)
    return s


print(f"  no smoothing: per-tensor W8A8 SQNR {sqnr_y(Y, act_qdq(Xe, 8, 'tensor') @ ql.qdq(W, 8, True, 'tensor').T):.2f} dB")
best = (None, -1e9)
for alpha in (0.3, 0.5, 0.7, 0.9):
    s = smooth(alpha)
    Xs, Ws = Xe / s, W * s                                   # broadcast over the 'in' axis
    assert np.allclose(Xs @ Ws.T, Y, rtol=1e-3, atol=1e-3)   # identical before quantization
    val = sqnr_y(Y, act_qdq(Xs, 8, "tensor") @ ql.qdq(Ws, 8, True, "tensor").T)
    print(f"  SmoothQuant alpha={alpha}: SQNR {val:.2f} dB   (max/median of X after: {np.abs(Xs).max() / np.median(np.abs(Xs).max(0)):.1f})")
    if val > best[1]:
        best = (alpha, val)
```

- **In simple words:** Divide the outlier activation channels by a number and multiply the matching weight columns by the same number; the product is unchanged but both sides are easier to quantize.
- **Exactly:** `s_j = max|X_j|^alpha / max|W_j|^(1-alpha)`; `Xs = X / s`, `Ws = W * s` (broadcasting over the input axis; W is (out, in) so its columns are input channels). The `assert np.allclose` proves the float product is identical. alpha trades difficulty between activations (alpha near 1 moves everything to the weights) and weights.
- **Why it is here:** SmoothQuant made per-tensor W8A8 viable for large LLMs; in real deployments the `1/s` is folded into the previous LayerNorm so it costs nothing at run time.

### 7. PTQ vs QAT on a tiny MLP  (`main.py` lines 140–191)

```python
# [[6: PTQ vs QAT on a tiny MLP]]
# PTQ = quantize a finished model (no training).  QAT = keep training while the forward pass uses quantized weights;
# the gradient of round() is 0, so the straight-through estimator (STE) pretends round() is the identity in backward.
print("\n== 6. PTQ vs QAT (straight-through estimator) on a 16-32-32-3 MLP, weights quantized per-tensor ==")
BITS = int(os.environ.get("BITS", 2))
g = torch.Generator().manual_seed(1)
centers = torch.randn(3, 16, generator=g) * 1.2
ytr = torch.randint(0, 3, (2000,), generator=g)
xtr = centers[ytr] + torch.randn(2000, 16, generator=g)
yte = torch.randint(0, 3, (1000,), generator=g)
xte = centers[yte] + torch.randn(1000, 16, generator=g)


def fq(w, bits):
    """fake-quantize with STE: forward = quantized value, backward = identity (the .detach() hides rounding from autograd)."""
    qmax = 2 ** (bits - 1) - 1
    s = w.abs().max() / qmax
    wq = torch.clamp(torch.round(w / s), -qmax, qmax) * s
    return w + (wq - w).detach()


class MLP(torch.nn.Module):
    def __init__(self, bits=None):
        super().__init__()
        self.l1, self.l2, self.l3 = torch.nn.Linear(16, 32), torch.nn.Linear(32, 32), torch.nn.Linear(32, 3)
        self.bits = bits                                      # None -> float; int -> quantize weights in forward

    def forward(self, x):
        w = (lambda l: l.weight) if self.bits is None else (lambda l: fq(l.weight, self.bits))
        x = torch.relu(torch.nn.functional.linear(x, w(self.l1), self.l1.bias))
        x = torch.relu(torch.nn.functional.linear(x, w(self.l2), self.l2.bias))
        return torch.nn.functional.linear(x, w(self.l3), self.l3.bias)


def train(m, steps, lr):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(steps):
        loss = torch.nn.functional.cross_entropy(m(xtr), ytr)
        opt.zero_grad(); loss.backward(); opt.step()


def acc(m):
    with torch.no_grad():
        return (m(xte).argmax(1) == yte).float().mean().item()


net = MLP(); train(net, 300, 1e-2)
print(f"  float32 test accuracy        : {acc(net):.3f}")
ptq = MLP(BITS); ptq.load_state_dict(net.state_dict())
print(f"  PTQ int{BITS} (no training)       : {acc(ptq):.3f}")
qat = MLP(BITS); qat.load_state_dict(net.state_dict()); train(qat, 200, 3e-3)
print(f"  QAT int{BITS} (200 STE steps)     : {acc(qat):.3f}")
```

- **In simple words:** Train a small network, then either round its weights (PTQ) or keep training while rounding in the forward pass (QAT).
- **Exactly:** `fq` returns `w + (wq - w).detach()`: the forward value is `wq` but autograd sees the identity, which is the straight-through estimator (STE) (the true derivative of `round` is 0 almost everywhere). `MLP(bits)` applies `fq` to each weight matrix in `forward`. QAT starts from the trained float weights and runs 200 Adam steps at a small learning rate. The task is 3 Gaussian blobs in 16 dimensions.
- **Why it is here:** Shows the two ends of the cost/quality spectrum: PTQ costs seconds and no training data pipeline, QAT recovers accuracy at very low bit widths but needs training.

### 8. Perplexity of fake-quantized distilgpt2 variants  (`main.py` lines 193–248)

```python
# [[7: perplexity of fake-quantized distilgpt2 variants]]
class QConv(torch.nn.Module):
    """Drop-in for GPT-2's Conv1D that fake-quantizes weight (per output channel) and, optionally, the input activation."""

    def __init__(self, conv, w_bits=None, a_bits=None, a_mode="tensor", a_amax=None, smooth_s=None, w_group=None):
        super().__init__()
        W = ql.get_w_out_in(conv)                              # (out, in)
        if smooth_s is not None:
            W = W * smooth_s
        if w_bits:
            W = ql.qdq(W, w_bits, True, w_group or "row")
        self.W = torch.from_numpy(np.ascontiguousarray(W.T))   # back to Conv1D layout (in, out)
        self.b = conv.bias.detach().clone()
        self.a_bits, self.a_mode, self.a_amax = a_bits, a_mode, a_amax
        self.s = None if smooth_s is None else torch.from_numpy(smooth_s.astype(np.float32))

    def forward(self, x):
        if self.s is not None:
            x = x / self.s
        if self.a_bits:
            shp = x.shape
            x = torch.from_numpy(act_qdq(x.reshape(-1, shp[-1]).numpy(), self.a_bits, self.a_mode, self.a_amax)).reshape(shp).float()
        return x @ self.W + self.b


def quantized_copy(**kw):
    import copy
    m = copy.deepcopy(model)
    for name, conv in ql.conv1d_layers(m):
        extra = {}
        if kw.get("smooth"):
            xm = np.abs(X_cal[name]).max(axis=0)
            wm = np.abs(ql.get_w_out_in(conv)).max(axis=0)
            extra["smooth_s"] = np.maximum(xm, 1e-5) ** kw["smooth"] / np.maximum(wm, 1e-5) ** (1 - kw["smooth"])
        if kw.get("static"):
            extra["a_amax"] = static_range(X_cal[name] / (extra["smooth_s"] if "smooth_s" in extra else 1), kw["static"])
        parent = m.get_submodule(name.rsplit(".", 1)[0])
        setattr(parent, name.rsplit(".", 1)[1], QConv(conv, kw.get("w_bits"), kw.get("a_bits"),
                                                       "static" if kw.get("static") else kw.get("a_mode", "tensor"), w_group=kw.get("w_group"), **extra))
    return m


print("\n== 7. Perplexity on the held-out ticket lines (lower is better; fp32 baseline first) ==")
base = ql.perplexity(model, tok, eval_text)
print(f"  fp32                                      : {base:8.2f}")
variants = [
    ("W8A16 (int8 weights per-row)", dict(w_bits=8)),
    ("W4A16 (int4 g128)", dict(w_bits=4, w_group=128)),
    ("W8A8 dynamic per-tensor", dict(w_bits=8, a_bits=8, a_mode="tensor")),
    ("W8A8 dynamic per-token", dict(w_bits=8, a_bits=8, a_mode="token")),
    ("W8A8 static minmax", dict(w_bits=8, a_bits=8, static="minmax")),
    ("W8A8 static percentile(99.9)", dict(w_bits=8, a_bits=8, static="percentile")),
    ("W8A8 per-tensor + SmoothQuant a=0.5", dict(w_bits=8, a_bits=8, a_mode="tensor", smooth=0.5)),
]
for label, kw in variants:
    print(f"  {label:<42}: {ql.perplexity(quantized_copy(**kw), tok, eval_text):8.2f}")
```

- **In simple words:** Replace every linear layer with a wrapper that quantizes weights (and optionally activations) and measure perplexity on held-out text.
- **Exactly:** `QConv` mimics GPT-2's `Conv1D` (`y = x @ W + b`, W stored (in, out)). It quantizes W once in `__init__` (per output channel or per group), and quantizes `x` in `forward` using `act_qdq`. `smooth_s` divides the activation and multiplies W before quantization. `quantized_copy` deep-copies the model and swaps each layer with `setattr` on its parent module; `ql.perplexity` computes exp(mean token cross-entropy) over 128-token windows.
- **Why it is here:** Layer-level SQNR is a proxy; perplexity is the first end-to-end number. Gaps between variants here are what you would defend in an interview.

### 9. TRY IT / BREAK IT  (`main.py` lines 250–254)

```python
# TRY IT: BITS=3 python main.py (or BITS=4) - when does PTQ catch up with QAT?
# TRY IT: in section 7 add a variant with smooth=0.8, or w_bits=4 plus a_bits=8 (W4A8).
# TRY IT: in section 4 change the percentile from 99.9 to 99 - does it still help?
# BREAK IT: in SmoothQuant divide the weights instead of multiplying (Ws = W / s): the assert fails because X/s @ (W/s) != X @ W.
# BREAK IT: in fq() remove the .detach() trick (use wq directly): gradients are zero (round has 0 slope) and QAT does not train.
```

- **In simple words:** Experiments to run after reading.
- **Exactly:** Comments only.
- **Why it is here:** Each one changes a single choice (bits, alpha, percentile, the STE trick).


## Syntax explained

| Syntax | Meaning |
|---|---|
| `torch.manual_seed(0)`, `np.random.seed(0)` | Fix randomness so the printed numbers repeat. |
| `X.max(axis=0)` | Reduce over tokens, one value per channel. |
| `np.abs(X).max(axis=1, keepdims=True)` | One value per row kept as a column so it broadcasts. |
| `w + (wq - w).detach()` | Straight-through trick: value equals `wq`, gradient flows to `w`. |
| `lambda l: l.weight` | Small anonymous function selecting plain or fake-quantized weights. |
| `setattr(parent, 'c_fc', QConv(...))` | Replace a sub-module of a model in place. |
| `copy.deepcopy(model)` | Independent copy so each variant starts from fp32 weights. |
| `assert np.allclose(a, b, rtol=1e-3, atol=1e-3)` | Check the float identity before quantizing. |

## Libraries & functions

| Function | What it does | Internally |
|---|---|---|
| `module.register_forward_hook(fn)` | Call `fn(module, inputs, output)` after each forward | PyTorch stores the hook in the module and invokes it during `__call__` |
| `ql.collect_inputs` | Layer inputs for a text | Hooks, `torch.cat` of 128-token windows |
| `torch.nn.functional.linear(x, w, b)` | `x @ w.T + b` | One GEMM |
| `torch.optim.Adam` | Adaptive-gradient optimizer | Keeps running averages of gradient and its square per parameter |
| `ql.perplexity` | exp(mean cross-entropy) | Windows of 128 tokens, summed NLL over predicted tokens |
| `np.percentile(a, 99.9)` | Value below which 99.9% of the data lies | Sorts / partitions the array |

## Execution trace (real run)
```
calibration tokens 477, eval tokens 727

== 1. Real activation outliers in distilgpt2 (inputs of each linear layer) ==
  layer                          max|X|  median ch absmax  max/median  worst channel
  h.0.attn.c_attn                  0.65             0.332         2.0            674
  h.3.attn.c_attn                  6.62             0.723         9.2             64
  h.3.mlp.c_fc                     8.90             0.474        18.8            138
  h.3.mlp.c_proj                   4.59             0.914         5.0           2166
  h.5.mlp.c_fc                     2.41             0.443         5.4            640
  over all 24 linear layers: max/median ratio ranges 2.0 .. 58.8
  weights of the same layer: absmax 1.21, per-input-channel absmax max/median = 2.3  (weights are far tamer than activations)

== 2. Output SQNR (dB) of layer h.3.mlp.c_fc - higher is better ==
  W4A16 (int4 g128 weights, fp acts)      :  22.99
  W8A16 (int8 per-row weights, fp acts)   :  46.54
  W8A8, activations int8 per-tensor       :  23.68
  W8A8, activations int8 per-token        :  37.00
  W8A8, activations int8 per-channel      :  43.65
  W8A4, activations int4 per-token        :  13.33

== 3. LLM.int8()-style mixed-precision decomposition (same layer) ==
  outlier channels (calib max > 6.0): 1 of 768 -> kept in fp; SQNR 37.80 dB (per-token W8A8 w/o decomposition: 37.00)

== 4. Static calibration: range method x calibration set (output SQNR on EVAL tickets, int8 per-tensor acts) ==
  dynamic per-tensor (no calibration needed): 23.71 dB
  calibration set              minmax  percentile      mse
  tickets, 8 tokens             24.20       22.27    24.38
  tickets, 32 tokens            24.20       21.83    25.12
  tickets, 128 tokens           24.20       21.59    25.83
  tickets, all 477 tokens       23.87       21.69    25.83
  general text, 234 tokens      24.32       21.86    25.83

== 5. SmoothQuant on the same layer (per-tensor W8A8: the cheap, fast-kernel setting) ==
  no smoothing: per-tensor W8A8 SQNR 23.53 dB
  SmoothQuant alpha=0.3: SQNR 30.82 dB   (max/median of X after: 7.1)
  SmoothQuant alpha=0.5: SQNR 31.88 dB   (max/median of X after: 4.0)
  SmoothQuant alpha=0.7: SQNR 28.79 dB   (max/median of X after: 2.3)
  SmoothQuant alpha=0.9: SQNR 24.19 dB   (max/median of X after: 1.7)

== 6. PTQ vs QAT (straight-through estimator) on a 16-32-32-3 MLP, weights quantized per-tensor ==
  float32 test accuracy        : 0.998
  PTQ int2 (no training)       : 0.619
  QAT int2 (200 STE steps)     : 1.000

== 7. Perplexity on the held-out ticket lines (lower is better; fp32 baseline first) ==
  fp32                                      :    25.44
  W8A16 (int8 weights per-row)              :    25.42
  W4A16 (int4 g128)                         :    31.74
  W8A8 dynamic per-tensor                   :    32.97
  W8A8 dynamic per-token                    :    25.55
  W8A8 static minmax                        :    32.55
  W8A8 static percentile(99.9)              :   141.68
  W8A8 per-tensor + SmoothQuant a=0.5       :    30.79
```

How to read it:
- **Outliers are real but moderate here**: across the 24 layers the per-channel max/median ratio of the inputs reaches 58.8x, with a single dominant channel (channel 138 in `h.3.mlp.c_fc`, ratio 18.8), while the same layer's weights have ratio 2.3. In billion-parameter models the ratio is larger.
- **On `h.3.mlp.c_fc`**: W8A16 46.5 dB; W8A8 with per-tensor activations only 23.7 dB (the outlier sets the scale), per-token 37.0, per-channel 43.7 (not usable with integer GEMM). W4A16 g128 (symmetric) 23.0 dB. LLM.int8-style decomposition (1 of 768 channels in fp) lifts per-token W8A8 from 37.0 to 37.8 dB. int4 activations (W8A4) fall to 13.3 dB.
- **Calibration**: the MSE rule is the best (25.8 dB) and beats min-max (24.2); the 99.9th-percentile rule is WORSE (21.7) than min-max because it clips the one meaningful outlier channel, a warning that clipping rules designed for noisy outliers can destroy real signal. Calibration size matters little here (8 tokens gives 24.2 min-max; the MSE rule needs about 128 tokens to settle at 25.8) and general-text calibration transferred as well as in-domain tickets; the sets are tiny, so do not over-read it.
- **SmoothQuant** raises per-tensor W8A8 from 23.5 dB to 31.9 dB at alpha 0.5 (30.8 at 0.3, 28.8 at 0.7, 24.2 at 0.9): too much migration just moves the problem into the weights.
- **QAT vs PTQ** (2-bit per-tensor weights): float 0.998 accuracy, PTQ 0.619, QAT 1.000.
- **Perplexity** (fp32 25.44): W8A16 25.42 (free); W8A8 dynamic per-token 25.55 (almost free); W8A8 dynamic per-tensor 32.97; static min-max 32.55; static percentile 141.68 (catastrophic, same clipping-the-outlier effect); per-tensor + SmoothQuant 30.79; W4A16 g128 symmetric RTN 31.74.

## Experiments (change ONE thing)

| Change | What you will see | Why |
|---|---|---|
| `BITS=4` | PTQ closes the gap to QAT | 15 levels are enough for a tiny MLP. |
| `BITS=3` | In between | The gap shrinks as bits grow. |
| Section 7: add `smooth=0.8` | Perplexity changes (alpha trade-off) | Difficulty moves toward weights. |
| Percentile 99.9 -> 100 | Becomes min-max | Sanity check. |
| `w_bits=4, a_bits=8` (W4A8) | Perplexity close to W4A16 | Activations at 8 bit are mild if per-token. |

## Debugging practice
**BREAK IT:** in SmoothQuant use `Ws = W / s` instead of `W * s`. The `assert np.allclose` fails (X/s @ (W/s)^T is not X @ W^T); the layer would silently change the model. Fix: divide X and multiply W by the same s.
**BREAK IT:** in `fq()` return `wq` directly (no `.detach()` trick). `round` has zero gradient, so QAT does not train (loss flat; accuracy stays at the PTQ level). Fix: `w + (wq - w).detach()` (STE).
**A real bug we hit while writing this step**: a shared helper had called `torch.set_grad_enabled(False)` globally; QAT then died with `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`. Fix: use `torch.no_grad()` only around evaluation.

## Common mistakes
- Quantizing activations per-channel and expecting an integer GEMM to apply the scale.
- Using percentile clipping on LLM activations without checking that the outlier channels carry signal.
- Calibrating on data unlike production traffic.
- Comparing static and dynamic numbers without saying which ranges were used.
- Forgetting that W8A8 fake quantization here only measures error; real speed needs integer or FP8 kernels.
- Running QAT with a quantizer that has no STE.

## Rebuild checklist
1. Hook the linear layers and stack their inputs; split calibration / eval text.
2. Print per-channel absmax max/median.
3. Write `act_qdq` with tensor / token / channel / static modes.
4. Write the three range rules and test on different calibration sets.
5. SmoothQuant: scales, assert identity, compare SQNR.
6. Tiny MLP: `fq` with STE; PTQ vs 200 QAT steps.
7. `QConv` wrapper and perplexity table.

## Say it aloud
- "Activations are harder than weights because their ranges depend on the input and a few channels are 10-100x larger; one per-tensor scale then wastes the integer grid."
- "Engines use per-channel weight scales and per-tensor or per-token activation scales, because per-channel activation scales cannot be applied after an integer GEMM."
- "SmoothQuant divides outlier activation channels by s and multiplies the matching weight columns by s, an exact identity in float that makes both sides quantizable; LLM.int8() instead keeps outlier channels in fp16."
- "On distilgpt2 W8A8 per-token cost 0.1 perplexity, per-tensor cost 7, SmoothQuant recovered a third of that, and percentile clipping of the activation range was catastrophic."

## The new problem (cliffhanger)
Per-token or per-tensor scales are crude. A 4-bit weight with one scale per 128 weights was already visible in step 04, but plain rounding at 4 and 3 bits still costs perplexity (31.7 above). Can we choose which integers to store, using the calibration data, so the layer output barely changes? That is [step 06: GPTQ and AWQ from scratch](../06_gptq_awq_from_scratch/WALKTHROUGH.md).
