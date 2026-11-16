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

# [[3: LLM.int8() outlier decomposition]]
# Keep the few outlier channels (|x| > threshold) in high precision, quantize the rest.  y = X_out @ W_out + Q(X_rest) @ Q(W_rest)
print("\n== 3. LLM.int8()-style mixed-precision decomposition (same layer) ==")
thr = 6.0                                                    # bitsandbytes default outlier threshold
out_ch = np.where(np.abs(Xc).max(axis=0) > thr)[0]           # outlier channels decided on calibration data (LLM.int8 decides per batch)
rest = np.setdiff1d(np.arange(Xe.shape[1]), out_ch)
Ydec = Xe[:, out_ch] @ W[:, out_ch].T + act_qdq(Xe[:, rest], 8, "token") @ ql.qdq(W[:, rest], 8, True, "row").T
print(f"  outlier channels (calib max > {thr}): {len(out_ch)} of {Xe.shape[1]} -> kept in fp; SQNR {sqnr_y(Y, Ydec):.2f} dB (per-token W8A8 w/o decomposition: {sqnr_y(Y, act_qdq(Xe, 8, 'token') @ W8.T):.2f})")

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

# TRY IT: BITS=3 python main.py (or BITS=4) - when does PTQ catch up with QAT?
# TRY IT: in section 7 add a variant with smooth=0.8, or w_bits=4 plus a_bits=8 (W4A8).
# TRY IT: in section 4 change the percentile from 99.9 to 99 - does it still help?
# BREAK IT: in SmoothQuant divide the weights instead of multiplying (Ws = W / s): the assert fails because X/s @ (W/s) != X @ W.
# BREAK IT: in fq() remove the .detach() trick (use wq directly): gradients are zero (round has 0 slope) and QAT does not train.
