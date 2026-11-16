# Step 07 — KV-cache quantization

## The story so far
[Step 06](../06_gptq_awq_from_scratch/WALKTHROUGH.md) shrank the weights to about 4 bits. The [memory calculator in step 03](../03_gpu_memory_calculator/WALKTHROUGH.md) already warned that weights are only half of serving memory: the KV cache (key-value cache, `past_key_values`) grows with every token and every concurrent request, and its table showed that quantizing only the weights leaves the cache at fp16.

## The pain point
At long context or high concurrency the cache dominates: for a Llama-3-8B shape, 32k tokens cost 4.29 GB of fp16 cache per sequence (computed below with the step-03 formula), more than the 4-bit weights themselves for a handful of users. Unlike weights, the cache is produced at run time, so you cannot calibrate it offline in the GPTQ sense, and K and V do not look alike.

## The spark
Quantize the cache on the fly with cheap min-max scales, choosing the GROUPING by tensor behaviour: K has a few channels that are large for every token (so give each channel its own scale, quantizing over tokens), V has no such channels but token magnitudes vary (so give each token its own scale). This is the idea of KIVI (2-bit asymmetric KV cache: K per-channel, V per-token). Production engines mostly use the simpler, hardware-friendly option: FP8 cache with scales.

## Goal
Capture a real cache from distilgpt2, quantize K and V per-token vs per-channel at int8, FP8-simulated, int4 and int2, then measure reconstruction SQNR, KL divergence of the next-token distribution, and greedy continuation agreement after plugging the de-quantized cache back in. Finish with memory arithmetic using the canonical formula imported from step 03.

## The theory (read this before the interview)
**Cache shape**: per layer, K and V each `(batch, kv_heads, tokens, head_dim)`. Memory = `2 x layers x kv_heads x head_dim x tokens x batch x bytes` (formula and presets: [step 03](../03_gpu_memory_calculator/WALKTHROUGH.md); this step imports `kv_cache_bytes` from it rather than re-deriving it).

**Grouping choices** (a group shares one scale/zero-point):
```
per-token    one scale per (head, token) over the head_dim channels     -> easy: complete as soon as the token is produced
per-channel  one scale per (head, channel) over many tokens             -> better for K's outlier channels,
                                                                           but needs a group of tokens first (KIVI keeps the newest
                                                                           ~R tokens in full precision, quantizes in token groups)
```
**KIVI (idea)**: asymmetric 2-bit, K per-channel, V per-token, plus a small full-precision residual window; no tuning needed. Related ideas: KVQuant (non-uniform, outlier handling), per-head scales.

**FP8 KV cache (current industry practice)**: vLLM `--kv-cache-dtype fp8` stores K,V as FP8 E4M3 with scales (per head in recent versions) and, per the vLLM project's April 2026 write-up, can run attention itself in FP8 end to end; halves cache memory against fp16, roughly doubling the tokens or concurrent sequences that fit; some hybrid models need `--kv-cache-dtype-skip-layers`. **NVFP4 KV cache** (4-bit block-scaled FP4, 16-value blocks with FP8 scales) is emerging: reported in recent vLLM releases but treat it as experimental and check the project's current documentation. Accuracy must be validated per model, because attention is sensitive to errors in keys (they set the softmax) more than in values.

**Where the error goes**: a perturbation of K changes the attention logits `q.k / sqrt(d)` before the softmax (amplified), a perturbation of V is averaged linearly by the attention weights. That is why K usually needs more care than V.

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| KV cache | key-value cache, attention cache, `past_key_values` | Yes | Stored K and V of all past tokens, per layer. |
| KV-cache quantization | KV quant, cache compression, `--kv-cache-dtype` | Yes | Store K,V in fewer bits. |
| Per-token quantization (cache) | token-wise, row-wise | Yes | Scale per token vector. |
| Per-channel quantization (cache) | channel-wise | Yes | Scale per channel across tokens (not the same "channel" as weight rows). |
| KIVI | tuning-free asymmetric 2-bit KV | Specific method | K per-channel + V per-token + fp residual window. |
| FP8 KV cache | e4m3 KV | Yes | 1 byte per value with scales; supported in vLLM, TensorRT-LLM, SGLang. |
| NVFP4 KV | 4-bit FP KV cache | Emerging | Block-scaled FP4 cache; check status. |
| Greedy agreement | token match rate, exact-match continuation | Yes | Fraction of positions where the quantized continuation equals the baseline. |
| KL divergence | KL(p_fp || p_quant) | Yes | Distance between the two next-token distributions, in nats. |

## Run it
```bash
cd 21_llm_inference_engineering/07_kv_cache_quantization
python main.py
NEW=48 python main.py                   # longer greedy continuation
# from the repo root
HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/07_kv_cache_quantization/main.py
```
Runtime about 10 s of CPU (measured user time 7 s; peak RAM about 0.9 GB) plus any wait for the shared model lock. The script targets Qwen2.5-0.5B-style GQA models too (`MODEL=...`), but on this machine only the tokenizer and config of Qwen2.5-0.5B are cached (no weights), so the course uses distilgpt2 (12 KV heads, plain multi-head attention) and never downloads anything.

## Code, step by step
Helpers (`ql.qdq`, `ql.fp8_qdq`, `ql.bits_per_weight`) are in [`../quantlib.py`](../quantlib.py). Only the prompt part of the cache is quantized; tokens generated afterwards are appended in fp32 (real engines quantize those too, in small token groups).

### 1. Header, imports, model and prompt  (`main.py` lines 1–41)

```python
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
```

- **In simple words:** Load distilgpt2 (cached) in float32 and tokenize a ~100-token prompt.
- **Exactly:** `AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)` loads the 82M-parameter model. `torch.set_grad_enabled(False)` disables autograd globally (inference only). distilgpt2 uses plain multi-head attention: 12 heads, so 12 KV heads of dimension 64, 6 layers. (Qwen2.5-0.5B, a GQA model with 2 KV heads, would be the more modern choice, but only its tokenizer/config are cached on this machine, so the weights cannot load offline; set `MODEL=` if yours are cached.)
- **Why it is here:** A real model gives real K/V statistics; the script is model-agnostic so a bigger one can be dropped in.

### 2. Capture the real cache  (`main.py` lines 44–63)

```python
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
```

- **In simple words:** Run the prompt (except its last token) through the model and take the keys and values it stores.
- **Exactly:** `model(ids[:, :-1], use_cache=True).past_key_values` is a `DynamicCache`. `cache_to_lists` reads `cache.layers[i].keys/.values` (transformers 4.56+/5.x) with a fallback for the older `key_cache` lists; each tensor has shape `(batch, kv_heads, tokens, head_dim)`. `lists_to_cache` builds a new `DynamicCache` with `cache.update(k, v, layer_idx)`.
- **Why it is here:** We need to be able to take the cache apart, damage it, and put it back to measure what the damage does to the model.

### 3. Do K and V really differ? Channel magnitudes  (`main.py` lines 66–75)

```python
# [[2: do K and V really differ? look at channel magnitudes]]
def channel_ratio(t):
    """max over channels of (absmax over tokens) / median over channels - >> 1 means a few outlier channels dominate."""
    a = t[0].abs().amax(dim=1)                                       # (heads, head_dim): per-channel absmax over tokens
    return (a.amax(dim=-1) / a.median(dim=-1).values).mean().item()


print("== 2. channel outlier ratio (avg over heads) ==")
for l in (0, n_layers // 2, n_layers - 1):
    print(f"  layer {l:2d}:  K {channel_ratio(Ks[l]):6.1f}   V {channel_ratio(Vs[l]):6.1f}")
```

- **In simple words:** For each head and channel compute the largest |value| over tokens, then compare the biggest channel with the median channel.
- **Exactly:** `t[0].abs().amax(dim=1)` reduces over the token axis giving `(heads, head_dim)`; ratio of max over channels to median over channels, averaged over heads.
- **Why it is here:** This is the measured reason behind KIVI's design: K has a few channels that are huge for every token (so they should be scaled per channel), V does not.

### 4. Quantizers for a cache tensor, and the per-channel SQNR metric  (`main.py` lines 78–99)

```python
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
```

- **In simple words:** Two ways to group the numbers: all channels of one token share a scale (per-token), or all tokens of one channel share a scale (per-channel).
- **Exactly:** `kv_qdq` reshapes a `(H, T, D)` array so each row is one group: for 'token' rows are `(head, token)` over D; for 'channel' we first transpose to `(H, D, T)` so rows are `(head, channel)` over T. `fmt='fp8'` uses FP8 E4M3 with one scale per row, otherwise asymmetric min-max integers. `chan_sqnr` computes the SQNR of each channel separately and averages: whole-tensor SQNR is dominated by the energy of the outlier channels, so it hides damage to the other channels.
- **Why it is here:** Per-channel scales over tokens need the tokens to exist, so real systems (KIVI) keep a short window of recent tokens in full precision and quantize in groups of tokens.

### 5. Plug the quantized cache back and compare behaviour  (`main.py` lines 102–140)

```python
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
```

- **In simple words:** Quantize all 24 layers of the cache, feed the last prompt token with this damaged cache, and see how the next-token distribution and a 24-token greedy continuation change.
- **Exactly:** `run()` returns the mean SQNRs, `KL(p_fp || p_quantized)` of the next-token distribution (`F.kl_div` with `log_target=True`, `reduction='sum'`), and the greedy tokens. The loop appends one new token per step; the cache object grows in place (new K/V in fp32). `greedy agree` counts positions where the continuation equals the fp32 continuation (once a token differs, later positions usually differ too).
- **Why it is here:** Reconstruction error is only a proxy; what matters is whether the model still says the same thing.

### 6. Memory: the canonical formula, imported  (`main.py` lines 142–151)

```python
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
```

- **In simple words:** Check that bytes of this cache equal the formula from step 03, then apply the formula to a Llama-3-8B shape at 32k context.
- **Exactly:** `importlib.util.spec_from_file_location` loads `../03_gpu_memory_calculator/main.py` as a module without running its CLI (it is guarded by `if __name__ == '__main__'`), so we call its `kv_cache_bytes(layers, kv_heads, head_dim, tokens, batch, bytes_per_value)`. Bytes per value for int4 with an fp16 scale and fp16 zero per 32 values = `ql.bits_per_weight(4, 32, 16, 16)/8` = 0.625.
- **Why it is here:** Never re-derive the formula: link and import, so the repository has one source of truth for memory arithmetic.

### 7. TRY IT / BREAK IT  (`main.py` lines 153–157)

```python
# TRY IT: MODEL=Qwen/Qwen2.5-0.5B python main.py (needs its weights cached; GQA with 2 KV heads, known K outlier channels) and compare the K/V ratios.
# TRY IT: NEW=64 - longer continuations amplify small cache errors, agreement falls.
# TRY IT: change the prompt to a long repeated text and see int4 per-token K degrade as T grows.
# BREAK IT: in kv_qdq swap which axis the 'channel' mode transposes (skip the transpose): it silently becomes per-token; the K channel result equals the token result.
# BREAK IT: pass the cache tensors without .unsqueeze(0) in kv_qdq: the cache gets shape (heads, T, D) and attention raises a shape error.
```

- **In simple words:** Experiments to run after reading.
- **Exactly:** Comments only.
- **Why it is here:** Model, continuation length, prompt length and the axis choice are the real levers.


## Syntax explained

| Syntax | Meaning |
|---|---|
| `t[0].abs().amax(dim=1)` | Drop the batch axis, absolute value, maximum over the token axis. |
| `a.transpose(0, 2, 1)` | Swap the last two axes: (H, T, D) to (H, D, T). |
| `np.ascontiguousarray` | Make a transposed view a normal contiguous array before `torch.from_numpy`. |
| `.unsqueeze(0)` | Add a leading batch axis of size 1. |
| `F.kl_div(logq, logp, log_target=True, reduction='sum')` | `sum p (log p - log q)` when both arguments are log-probabilities. Here input = quantized, target = baseline. |
| `importlib.util.spec_from_file_location` | Import a Python file by path even though its folder name starts with a digit. |
| `logits.argmax().view(1, 1)` | Greedy next token as a (1, 1) input for the next step. |

## Libraries & functions

| Function | What it does | Internally |
|---|---|---|
| `DynamicCache()` and `.update(k, v, layer_idx)` | Growable KV cache | Appends K,V along the token axis for that layer and returns the concatenated tensors |
| `model(ids, past_key_values=cache, use_cache=True)` | One forward step reusing the cache | Attention uses cached K,V for past tokens, computes K,V only for the new ones, then appends |
| `gm.kv_cache_bytes(...)` | Canonical memory formula (step 03) | `2 * layers * kv_heads * head_dim * tokens * batch * bytes` |
| `ql.fp8_qdq(flat, 'e4m3', 'row')` | FP8 simulation with a scale per row | Scale so absmax maps to 448, round to E4M3, scale back |

## Execution trace (real run)
```
== 1. real cache from distilgpt2: 6 layers, K shape (1, 12, 124, 64) = (batch, kv_heads, tokens, head_dim) ==
== 2. channel outlier ratio (avg over heads) ==
  layer  0:  K    1.7   V    2.5
  layer  3:  K    3.0   V    1.8
  layer  5:  K    2.1   V    1.9

== 3/4. effect of cache quantization (baseline continuation: ' The carbon dioxide is released into the air, and the oxygen is released into the air. The carbon dioxide is released into') ==
  K / V scheme                        SQNR K  K/chan  SQNR V  KL(nats)  top1 same  greedy agree
  int8 K token / V token               44.54   42.52   44.78    0.0000       True        24/24
  int8 K channel / V token             48.69   47.15   44.78    0.0000       True        24/24
  FP8 E4M3 K token / V token           32.20   31.79   32.18    0.0001       True        24/24
  int4 K token / V token               19.92   17.90   20.17    0.0017       True        24/24
  int4 K channel / V token  (KIVI)     24.07   22.54   20.17    0.0010       True        20/24
  int4 K token / V channel (reverse)   19.92   17.90   20.64    0.0018       True        24/24
  int2 K token / V token                6.35    4.43    6.32    0.1640       True        24/24
  int2 K channel / V token  (KIVI)     10.11    8.58    6.32    0.0967       True         1/24

== 5. memory: measured fp32 cache 4,571,136 B | formula (21/03) 4,571,136 B | equal: True
  Llama-3-8B shape (32 layers, 8 KV heads, head_dim 128), 32k context, batch 1:
    fp16                            4.29 GB
    fp8 / int8                      2.15 GB
    int4 + fp16 scale & zero per 32  1.34 GB
```

How to read it:
- **Shape**: 6 layers, K `(1, 12, 124, 64)`; our formula gives exactly the measured 4,571,136 bytes for the fp32 cache, so the step-03 formula holds.
- **distilgpt2's channel outliers are mild** (K max/median 1.7-3.0, V 1.8-2.5), much weaker than in larger LLMs; so the per-channel advantage we measure is real but modest. Per-channel scales raise K's SQNR from 19.9 to 24.1 dB at int4 and from 6.4 to 10.1 dB at int2, and lower the KL of the next-token distribution (int4: 0.0017 to 0.0010; int2: 0.164 to 0.097). Applying per-channel to V instead (reverse row) does not help V (20.6 vs 20.2 dB, noise level), which is why V stays per-token.
- **int8 and FP8 are essentially free** (KL about 0, all 24 greedy tokens identical); FP8 E4M3 has lower SQNR (32 dB) than int8 (44.5 dB) because it has fewer mantissa bits, yet it is the engine default because it needs no zero-point, scales cheaply and hardware accelerates it.
- **Greedy agreement is a brittle metric**: the baseline continuation degenerates into repetition ("...released into the air. The carbon dioxide is released into"), so a single near-tie flip changes everything after it. That is why int4 K-per-channel shows 20/24 while int4 K-per-token shows 24/24 although its KL is lower, and why int2 KIVI shows 1/24 despite a lower KL than int2 per-token (24/24): do not over-read it on 24 tokens of one prompt; use KL or perplexity on a held-out set, and a task benchmark before shipping.
- **Memory**: a Llama-3-8B-shaped cache at 32k tokens is 4.29 GB in fp16, 2.15 GB in fp8/int8 and 1.34 GB in int4 with fp16 scale and zero per group of 32 (0.625 bytes per value, so the group overhead adds 25% to the ideal 0.5 bytes).

## Experiments (change ONE thing)

| Change | What you will see | Why |
|---|---|---|
| `NEW=48` | Agreement counts fall | Errors compound over the continuation. |
| Another prompt (a long repeated text) | int4 per-token K degrades with length | More tokens, more chances for a wrong attention peak. |
| `MODEL=Qwen/Qwen2.5-0.5B` (if its weights are cached) | Much larger K channel ratio, bigger per-channel advantage | Qwen has strong K outlier channels. |
| Remove the `transpose` for 'channel' | Channel row equals token row | It silently becomes per-token. |
| Add bits=3 | Between int4 and int2 | Fewer levels, more error. |

## Debugging practice
**BREAK IT:** drop `.unsqueeze(0)` in `kv_qdq`. The rebuilt cache tensors lose the batch axis, `DynamicCache.update` accepts them but the attention matmul then fails with a shape error (expected 4-D `(batch, heads, tokens, dim)`). Fix: restore the batch axis.
**BREAK IT:** skip the transpose in the 'channel' branch. No error appears; K per-channel numbers become identical to per-token (the silent bug class). Fix: assert the group axis, e.g. check that scales vary along the intended axis.
**Real bug we hit**: the original draft defaulted to Qwen2.5-0.5B. Only its config and tokenizer were cached, so loading raised `OSError: ... does not appear to have a file named pytorch_model.bin or model.safetensors` (an offline cache miss, not a download). Fix: default to distilgpt2.

## Common mistakes
- Quantizing K per-token and V per-channel (the reverse of KIVI).
- Judging by whole-tensor SQNR, which is dominated by outlier channels (see the K/chan column; exercise Q6 shows normal channels at 0.4 dB vs 19 dB).
- Judging by greedy token agreement on a single prompt.
- Forgetting scales cost memory (int4 with group 32 is 5 bits per value with fp16 scale and zero).
- Assuming an FP8 or NVFP4 KV cache is accuracy-free for every model; validate.
- Mixing "channel" meanings: for weights it is an output row; for K/V cache it is a head_dim position.

## Rebuild checklist
1. Prefill with `use_cache=True`, split `past_key_values` into per-layer K and V.
2. Write `kv_qdq` for token / channel grouping with a transpose.
3. Rebuild a `DynamicCache`, run the last token and compute KL against the fp cache.
4. Run a greedy loop with each cache and count agreement.
5. Import `kv_cache_bytes` from step 03 and check it equals the measured size.

## Say it aloud
- "The KV cache is 2 x layers x kv_heads x head_dim x tokens x batch x bytes; for Llama-3-8B at 32k context that is about 4.3 GB per sequence in fp16, so cache dtype matters as much as weight dtype at long context."
- "K has outlier channels, so quantize it per channel; V has varying token magnitudes, so quantize it per token; that is KIVI."
- "FP8 KV cache is current practice in vLLM and halves cache memory; 4-bit block-scaled caches such as NVFP4 are emerging."
- "I evaluate cache quantization with KL or perplexity on held-out text rather than greedy match on one prompt."

## The new problem (cliffhanger)
We now know how to make weights and the cache small, but we have been working on paper and with simulation. How do real Hugging Face generation calls behave with batching, padding and the cache, and how fast do they actually run on CPU? That is [step 08: HF inference, batching and KV cache](../08_hf_inference_batching_kvcache/WALKTHROUGH.md).
