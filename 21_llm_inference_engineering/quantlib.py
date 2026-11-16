"""
quantlib.py - small shared helpers for module 21 steps 04-07 (quantization).  NOT a step; import it:

    import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import quantlib as ql

Everything is "fake quantization" (quantize -> de-quantize, kept as float) so error can be measured on a laptop.
Prerequisite lesson: 05_llm_parameters/05_quantization (symmetric int8, group-wise idea).
Sections: metrics | affine int quantizer | int4 packing | FP8 / FP4 simulation | NF4 | bits-per-weight | distilgpt2 helpers.
"""
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


# ----------------------------------------------------------------------------- metrics
def metrics(x, xh):
    """Compare original x with reconstruction xh.  Returns dict(mse, sqnr_db, max_abs, cos)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    xh = np.asarray(xh, dtype=np.float64).ravel()
    err = x - xh
    mse = float(np.mean(err ** 2))
    sqnr = float(10 * np.log10(np.sum(x ** 2) / max(np.sum(err ** 2), 1e-30)))   # signal-to-quantization-noise ratio, dB
    cos = float(x @ xh / (np.linalg.norm(x) * np.linalg.norm(xh) + 1e-30))
    return dict(mse=mse, sqnr_db=sqnr, max_abs=float(np.abs(err).max()), cos=cos)


# ----------------------------------------------------------------------------- affine integer quantizer
def group_view(x, gran):
    """Reshape a 2-D array (R, C) so that EACH ROW of the view shares one scale.
    gran: "tensor" -> (1, R*C);  "row" -> (R, C) (per output channel);  int g -> (R*C/g, g) (groups along the last dim)."""
    x = np.asarray(x, dtype=np.float32)
    if gran == "tensor":
        return x.reshape(1, -1)
    if gran == "row":
        return x.reshape(x.shape[0], -1)
    assert x.shape[-1] % gran == 0, "last dim must be divisible by the group size"
    return x.reshape(-1, gran)


def qdq(x, bits=8, symmetric=True, gran="tensor", clip=1.0):
    """Quantize then de-quantize x (2-D). Returns array of x's shape.
    symmetric (absmax):   s = clip*max|x| / (2^(b-1)-1),  q = round(x/s) in [-qmax, qmax],  x_hat = q*s
    asymmetric (min-max): s = (hi-lo)/(2^b-1), z = round(-lo/s),  q = round(x/s)+z in [0, 2^b-1],  x_hat = (q-z)*s
    clip < 1 shrinks the range (outliers saturate, everything else gets a finer step)."""
    v = group_view(x, gran)
    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        s = np.abs(v).max(axis=1, keepdims=True) * clip / qmax
        s = np.where(s == 0, 1.0, s)
        q = np.clip(np.round(v / s), -qmax, qmax)
        out = q * s
    else:
        lo = np.minimum(v.min(axis=1, keepdims=True) * clip, 0.0)     # range must contain 0 so that 0.0 is exactly representable
        hi = np.maximum(v.max(axis=1, keepdims=True) * clip, 0.0)
        s = (hi - lo) / (2 ** bits - 1)
        s = np.where(s == 0, 1.0, s)
        z = np.round(-lo / s)
        q = np.clip(np.round(v / s) + z, 0, 2 ** bits - 1)
        out = (q - z) * s
    return out.reshape(np.asarray(x).shape).astype(np.float32)


def mse_optimal_clip(x, bits=4, symmetric=True, gran="tensor", grid=None):
    """Grid-search ONE clip factor in (0,1] minimising reconstruction MSE.  Returns (best_clip, best_mse, mse_at_clip1)."""
    grid = np.linspace(0.3, 1.0, 36) if grid is None else grid
    mses = [metrics(x, qdq(x, bits, symmetric, gran, c))["mse"] for c in grid]
    i = int(np.argmin(mses))
    return float(grid[i]), mses[i], mses[-1]


# ----------------------------------------------------------------------------- int4 packing
def pack_int4(q):
    """q: integer array in [-7, 7] with an even last dim.  Stores two 4-bit values per uint8 (offset by 8 so they are 1..15)."""
    u = (np.asarray(q).astype(np.int16) + 8).astype(np.uint8)
    return (u[..., 0::2] | (u[..., 1::2] << 4)).astype(np.uint8)       # low nibble = even index, high nibble = odd index


def unpack_int4(p):
    lo = (p & 0x0F).astype(np.int16) - 8
    hi = (p >> 4).astype(np.int16) - 8
    out = np.empty(p.shape[:-1] + (p.shape[-1] * 2,), dtype=np.int8)
    out[..., 0::2], out[..., 1::2] = lo, hi
    return out


# ----------------------------------------------------------------------------- FP8 / FP4 simulation
FP8 = {  # (exp_bits, man_bits, bias, max finite value) ; OCP FP8 definitions
    "e4m3": (4, 3, 7, 448.0),
    "e5m2": (5, 2, 15, 57344.0),
}


def fp_round(x, fmt="e4m3"):
    """Round float values to the nearest representable value of a tiny float format (saturating, no inf/nan).
    Spacing between neighbours = 2^(exponent - man_bits); below the smallest normal exponent the spacing stays fixed (subnormals)."""
    eb, mb, bias, vmax = FP8[fmt]
    x = np.asarray(x, dtype=np.float64)
    a = np.minimum(np.abs(x), vmax)
    e = np.floor(np.log2(np.where(a > 0, a, 1.0)))
    e = np.maximum(e, 1 - bias)                                      # subnormal range shares the smallest exponent
    step = 2.0 ** (e - mb)
    r = np.minimum(np.round(a / step) * step, vmax)
    return (np.sign(x) * r).astype(np.float32)


def fp8_qdq(x, fmt="e4m3", gran="tensor"):
    """Scale so absmax maps to the format's max, round to FP8 values, scale back (what an FP8 W8A8 engine does per tensor)."""
    v = group_view(x, gran)
    s = np.abs(v).max(axis=1, keepdims=True) / FP8[fmt][3]
    s = np.where(s == 0, 1.0, s)
    return (fp_round(v / s, fmt) * s).reshape(np.asarray(x).shape).astype(np.float32)


FP4_POS = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)    # all non-negative FP4 E2M1 values
FP4_LEVELS = np.concatenate([-FP4_POS[:0:-1], FP4_POS])               # 15 distinct values (+0 and -0 merged)


def _nearest(norm, levels):
    """Index of the closest level for every value (levels sorted ascending) via midpoints + searchsorted."""
    mid = (levels[1:] + levels[:-1]) / 2
    return np.searchsorted(mid, norm)


def fp4_block_qdq(x, block=32, scale="e8m0"):
    """Block-scaled FP4 (E2M1 values).  scale="e8m0": MXFP4 (OCP microscaling, block 32, power-of-two scale).
    scale="e4m3": NVFP4 (NVIDIA Blackwell, block 16, FP8 E4M3 block scale + one FP32 per-tensor scale)."""
    v = np.asarray(x, dtype=np.float32).reshape(-1, block)
    amax = np.abs(v).max(axis=1, keepdims=True)
    amax = np.where(amax == 0, 1e-12, amax)
    if scale == "e8m0":
        s = 2.0 ** (np.floor(np.log2(amax)) - 2)                    # MX spec: shared exponent = floor(log2 amax) - emax(E2M1)=2
    else:
        ts = np.abs(v).max() / (6.0 * 448.0)                         # per-tensor FP32 scale so block scales fit E4M3's range
        s = fp_round(amax / 6.0 / ts, "e4m3") * ts
        s = np.where(s == 0, ts, s)
    idx = _nearest(np.clip(v / s, -6, 6), FP4_LEVELS)
    return (FP4_LEVELS[idx] * s).reshape(np.asarray(x).shape).astype(np.float32)


# ----------------------------------------------------------------------------- NF4
def nf4_levels():
    """The 16 NormalFloat4 levels: quantiles of N(0,1) (asymmetric split so that 0 is exactly representable), scaled to [-1, 1].
    This is the QLoRA recipe (bitsandbytes create_normal_map)."""
    from scipy.stats import norm
    off = 0.9677083
    pos = norm.ppf(np.linspace(off, 0.5, 9)[:-1])        # 8 positive levels
    neg = -norm.ppf(np.linspace(off, 0.5, 8)[:-1])       # 7 negative levels
    v = np.sort(np.concatenate([pos, [0.0], neg]))
    return (v / np.abs(v).max()).astype(np.float32)


def nf4_qdq(x, block=64, double_quant=False):
    """Blockwise NF4: per block of `block` values, scale = absmax, snap x/scale to the nearest NF4 level.
    double_quant=True also stores the per-block scales as int8 (blocks of 256 scales, mean removed) - QLoRA 'double quantization'."""
    lv = nf4_levels()
    v = np.asarray(x, dtype=np.float32).reshape(-1, block)
    s = np.abs(v).max(axis=1, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    if double_quant:
        flat = s.ravel()
        mean = flat.mean()
        pad = (-len(flat)) % 256
        c = np.pad(flat - mean, (0, pad)).reshape(-1, 256)
        cs = np.abs(c).max(axis=1, keepdims=True) / 127
        cs = np.where(cs == 0, 1.0, cs)
        c = np.round(c / cs) * cs
        s = (c.ravel()[:len(flat)] + mean).reshape(-1, 1).astype(np.float32)
    idx = _nearest(v / s, lv)
    return (lv[idx] * s).reshape(np.asarray(x).shape).astype(np.float32)


# ----------------------------------------------------------------------------- bits per weight
def bits_per_weight(bits, group=None, scale_bits=16, zero_bits=0):
    """Average storage per weight including scale (and zero-point) overhead.  group=None -> overhead ignored (per-tensor)."""
    return bits if group is None else bits + (scale_bits + zero_bits) / group


# ----------------------------------------------------------------------------- distilgpt2 helpers (need transformers + cached model)
GENERAL_TEXT = (
    "The city council met on Tuesday evening to discuss the new bicycle lanes along the river. Several residents "
    "spoke in favour, saying the lanes would make the commute safer, while shop owners worried about losing parking spaces. "
    "After two hours of debate the council voted to run a six month trial and review the results next spring.\n"
    "Photosynthesis is the process by which green plants turn sunlight, water and carbon dioxide into sugar and oxygen. "
    "It takes place in the chloroplasts, where chlorophyll absorbs mostly red and blue light and reflects green. "
    "The sugar is used by the plant as energy, and the oxygen is released into the air.\n"
    "To bake a simple loaf of bread, mix flour, water, yeast and a pinch of salt, then knead the dough until it is smooth. "
    "Let it rest in a warm place until it doubles in size, shape it, and bake it in a hot oven until the crust turns golden brown.\n"
    "The football match ended in a draw after both teams missed several chances in the second half. The goalkeeper made "
    "a spectacular save in the final minute, and the crowd rose to applaud both sides as the players walked off the pitch.\n"
)


def load_distilgpt2():
    """(tokenizer, model) for the cached distilgpt2, float32, eval mode.  Never downloads (offline)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
    return tok, model


def corpus_split(calib_fraction=0.4):
    """tiny_corpus.txt lines split into (calibration text, evaluation text) - disjoint, so perplexity is on unseen lines."""
    lines = (REPO / "06_llm_training_and_lora/data/tiny_corpus.txt").read_text().strip().split("\n")
    k = int(len(lines) * calib_fraction)
    return "\n".join(lines[:k]), "\n".join(lines[k:])


def perplexity(model, tok, text, ctx=128):
    """exp(mean next-token cross-entropy) over non-overlapping windows of `ctx` tokens."""
    import torch
    ids = tok(text, return_tensors="pt")["input_ids"][0]
    total, count = 0.0, 0
    torch.set_grad_enabled(False)                                    # evaluation only (restored below)
    for i in range(0, len(ids) - 1, ctx):
        chunk = ids[i:i + ctx + 1].unsqueeze(0)
        if chunk.shape[1] < 2:
            break
        logits = model(chunk[:, :-1]).logits
        nll = torch.nn.functional.cross_entropy(logits[0], chunk[0, 1:], reduction="sum")
        total += nll.item()
        count += chunk.shape[1] - 1
    torch.set_grad_enabled(True)
    return float(np.exp(total / count))


def conv1d_layers(model):
    """[(name, module)] for the 4 Conv1D layers per block of GPT-2/distilgpt2 (c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj).
    GPT-2's Conv1D stores weight as (in_features, out_features) and computes y = x @ W + b (transposed vs nn.Linear)."""
    keep = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")
    return [(n, m) for n, m in model.named_modules() if n.endswith(keep)]


def get_w_out_in(mod):
    """Weight in nn.Linear convention (out, in) as float32 numpy (a copy)."""
    return mod.weight.detach().numpy().T.copy()


def set_w_out_in(mod, w_out_in):
    import torch
    mod.weight.data = torch.from_numpy(np.ascontiguousarray(w_out_in.T.astype(np.float32)))


def collect_inputs(model, tok, text, ctx=128, max_tokens=2048):
    """Run `text` through the model with forward hooks; return {layer_name: X of shape (n_tokens, in_features)}."""
    import torch
    store, hooks = {}, []
    for name, m in conv1d_layers(model):
        def hook(mod, inp, out, name=name):
            store.setdefault(name, []).append(inp[0].reshape(-1, inp[0].shape[-1]).clone())
        hooks.append(m.register_forward_hook(hook))
    ids = tok(text, return_tensors="pt")["input_ids"][0][:max_tokens]
    with torch.no_grad():
        for i in range(0, len(ids), ctx):
            model(ids[i:i + ctx].unsqueeze(0))
    for h in hooks:
        h.remove()
    return {k: torch.cat(v).numpy() for k, v in store.items()}
