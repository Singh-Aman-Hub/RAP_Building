"""
GOAL:  Capture a REAL KV cache (past_key_values) from distilgpt2, quantize K and V per-token vs per-channel at int8 / int4 /
       FP8 / int2, and measure (a) reconstruction error, (b) the change in next-token logits, (c) greedy continuation agreement
       when the de-quantized cache is plugged back in.  Shows why K wants per-channel scales and V per-token (the KIVI idea).

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/07_kv_cache_quantization/main.py
EXPECTED OUTPUT (shortened):
    cache: 6 layers, K shape (1, 12, T, 64); distilgpt2's K channel max/median is only ~2-3x (mild; Qwen/Llama-class models are stronger)
    int8 is almost lossless; int4/int2 per-channel K has higher SQNR and lower KL than per-token K; greedy agreement is a noisy metric
    each config: SQNR K / SQNR V / KL of last-token logits / top-1 same / greedy agreement of 24 tokens
    memory: bytes of this cache measured vs the formula of 21/03, then the same formula at 32k context for a Llama-3-8B shape
NEEDS: torch + transformers (distilgpt2 cached; Qwen2.5-0.5B only if its weights are cached on your machine: MODEL=Qwen/Qwen2.5-0.5B), numpy.  ~30-60 s on CPU.

Builds on 06_gptq_awq_from_scratch (weights are done; now the other memory hog) and the canonical memory formula in
../03_gpu_memory_calculator (imported here, not re-implemented).  Only the PROMPT part of the cache is quantized; tokens generated
afterwards are appended in full precision (real engines quantize them too, in small groups).
"""
import importlib.util
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import quantlib as ql

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL = os.environ.get("MODEL", "distilgpt2")
NEW = int(os.environ.get("NEW", 24))                  # greedy continuation length
torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
cfg = model.config
prompt = ql.GENERAL_TEXT.split("\n")[0] + " " + ql.GENERAL_TEXT.split("\n")[1]       # two paragraphs, ~100 tokens
ids = tok(prompt, return_tensors="pt")["input_ids"]
T = ids.shape[1]


# [[1: capture the real cache]]
# prefill all prompt tokens EXCEPT the last one; the cache then holds K,V of T-1 tokens, shape (batch, kv_heads, T-1, head_dim) per layer.
def cache_to_lists(cache):
    """Return ([K_l], [V_l]) torch tensors per layer for transformers v4 and v5 cache layouts."""
    if hasattr(cache, "layers"):                                     # transformers >= 4.56 / 5.x
        return [l.keys for l in cache.layers], [l.values for l in cache.layers]
    return list(cache.key_cache), list(cache.value_cache)


def lists_to_cache(ks, vs):
    c = DynamicCache()
    for i, (k, v) in enumerate(zip(ks, vs)):
        c.update(k, v, i)
    return c


out = model(ids[:, :-1], use_cache=True)
Ks, Vs = cache_to_lists(out.past_key_values)
print(f"== 1. real cache from {MODEL}: {len(Ks)} layers, K shape {tuple(Ks[0].shape)} = (batch, kv_heads, tokens, head_dim) ==")
kv_heads, head_dim, n_layers = Ks[0].shape[1], Ks[0].shape[3], len(Ks)


# [[2: do K and V really differ? look at channel magnitudes]]
def channel_ratio(t):
    """max over channels of (absmax over tokens) / median over channels - >> 1 means a few outlier channels dominate."""
    a = t[0].abs().amax(dim=1)                                       # (heads, head_dim): per-channel absmax over tokens
    return (a.amax(dim=-1) / a.median(dim=-1).values).mean().item()


print("== 2. channel outlier ratio (avg over heads) ==")
for l in (0, n_layers // 2, n_layers - 1):
    print(f"  layer {l:2d}:  K {channel_ratio(Ks[l]):6.1f}   V {channel_ratio(Vs[l]):6.1f}")


# [[3: quantizers for a cache tensor of shape (1, heads, tokens, dim)]]
def kv_qdq(t, mode, bits=8, fmt="int"):
    """mode 'token': one scale per (head, token) over the channel dim;  'channel': one scale per (head, channel) over the tokens."""
    a = t[0].numpy()                                                 # (H, T, D)
    if mode == "channel":
        a = a.transpose(0, 2, 1)                                     # (H, D, T): now the last dim is tokens
    H = a.shape[0]
    flat = a.reshape(-1, a.shape[-1])                                # each row = one group sharing a scale
    if fmt == "fp8":
        r = ql.fp8_qdq(flat, "e4m3", "row")
    else:
        r = ql.qdq(flat, bits, symmetric=False, gran="row")          # asymmetric min-max (cache values are not zero-centred)
    r = r.reshape(a.shape)
    if mode == "channel":
        r = r.transpose(0, 2, 1)
    return torch.from_numpy(np.ascontiguousarray(r)).unsqueeze(0)


def chan_sqnr(t, q):
    """SQNR computed PER CHANNEL then averaged: whole-tensor SQNR is dominated by the energy of the outlier channels and hides damage to the rest."""
    x, e = t[0].double(), (t[0] - q[0]).double()
    return (10 * torch.log10((x ** 2).sum(1) / (e ** 2).sum(1).clamp_min(1e-30))).mean().item()


# [[4: plug the quantized cache back and compare the model's behaviour]]
def run(kmode, vmode, bits, fmt="int"):
    ks = [kv_qdq(k, kmode, bits, fmt) for k in Ks]
    vs = [kv_qdq(v, vmode, bits, fmt) for v in Vs]
    sk = np.mean([ql.metrics(k.numpy(), q.numpy())["sqnr_db"] for k, q in zip(Ks, ks)])
    sv = np.mean([ql.metrics(v.numpy(), q.numpy())["sqnr_db"] for v, q in zip(Vs, vs)])
    sc = np.mean([chan_sqnr(k, q) for k, q in zip(Ks, ks)])
    cache = lists_to_cache(ks, vs)
    logits = model(ids[:, -1:], past_key_values=cache, use_cache=True).logits[0, -1]       # next-token logits using the quantized cache
    kl = torch.nn.functional.kl_div(torch.log_softmax(logits, -1), torch.log_softmax(base_logits, -1), log_target=True, reduction="sum").item()
    toks = []
    for _ in range(NEW):                                             # greedy continuation, one token at a time, cache grows in fp32
        nxt = logits.argmax().view(1, 1)
        toks.append(nxt.item())
        logits = model(nxt, past_key_values=cache, use_cache=True).logits[0, -1]
    return sk, sc, sv, kl, toks


# baseline: fp32 cache
cache0 = lists_to_cache(Ks, Vs)
base_logits = model(ids[:, -1:], past_key_values=cache0, use_cache=True).logits[0, -1]
lg = base_logits
base_toks = []
for _ in range(NEW):
    nxt = lg.argmax().view(1, 1)
    base_toks.append(nxt.item())
    lg = model(nxt, past_key_values=cache0, use_cache=True).logits[0, -1]
print(f"\n== 3/4. effect of cache quantization (baseline continuation: {tok.decode(base_toks)!r}) ==")
print(f"  {'K / V scheme':<34}{'SQNR K':>8}{'K/chan':>8}{'SQNR V':>8}{'KL(nats)':>10}{'top1 same':>11}{'greedy agree':>14}")
configs = [("int8 K token / V token", "token", "token", 8, "int"), ("int8 K channel / V token", "channel", "token", 8, "int"),
           ("FP8 E4M3 K token / V token", "token", "token", 8, "fp8"),
           ("int4 K token / V token", "token", "token", 4, "int"), ("int4 K channel / V token  (KIVI)", "channel", "token", 4, "int"),
           ("int4 K token / V channel (reverse)", "token", "channel", 4, "int"),
           ("int2 K token / V token", "token", "token", 2, "int"), ("int2 K channel / V token  (KIVI)", "channel", "token", 2, "int")]
for label, km, vm, bits, fmt in configs:
    sk, sc, sv, kl, toks = run(km, vm, bits, fmt)
    agree = sum(a == b for a, b in zip(toks, base_toks))
    top1 = toks[0] == base_toks[0]
    print(f"  {label:<34}{sk:8.2f}{sc:8.2f}{sv:8.2f}{kl:10.4f}{str(top1):>11}{agree:>10d}/{NEW}")

# [[5: memory - the canonical formula from 21/03 (imported)]]
spec = importlib.util.spec_from_file_location("gpu_mem", os.path.join(HERE, "..", "03_gpu_memory_calculator", "main.py"))
gm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gm)                                          # safe: that file runs its CLI only under __main__
measured = sum(k.numel() * 4 + v.numel() * 4 for k, v in zip(Ks, Vs))
formula = gm.kv_cache_bytes(n_layers, kv_heads, head_dim, T - 1, 1, 4)
print(f"\n== 5. memory: measured fp32 cache {measured:,} B | formula (21/03) {formula:,} B | equal: {measured == formula}")
print("  Llama-3-8B shape (32 layers, 8 KV heads, head_dim 128), 32k context, batch 1:")
for label, bpv in (("fp16", 2.0), ("fp8 / int8", 1.0), ("int4 + fp16 scale & zero per 32", ql.bits_per_weight(4, 32, 16, 16) / 8)):
    print(f"    {label:<30}{gm.kv_cache_bytes(32, 8, 128, 32768, 1, bpv) / 1e9:6.2f} GB")

# TRY IT: MODEL=Qwen/Qwen2.5-0.5B python main.py (needs its weights cached; GQA with 2 KV heads, known K outlier channels) and compare the K/V ratios.
# TRY IT: NEW=64 - longer continuations amplify small cache errors, agreement falls.
# TRY IT: change the prompt to a long repeated text and see int4 per-token K degrade as T grows.
# BREAK IT: in kv_qdq swap which axis the 'channel' mode transposes (skip the transpose): it silently becomes per-token; the K channel result equals the token result.
# BREAK IT: pass the cache tensors without .unsqueeze(0) in kv_qdq: the cache gets shape (heads, T, D) and attention raises a shape error.
