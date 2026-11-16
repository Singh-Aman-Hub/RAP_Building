# Step 04 — Attention variants (MHA, MQA, GQA, MLA) and the KV cache

## The story so far
[Step 03](../03_norms_and_ffn/WALKTHROUGH.md) made the block stable and the MLP efficient. Attention is the other half. [Module 21, step 08](../../21_llm_inference_engineering/08_hf_inference_batching_kvcache/WALKTHROUGH.md) showed that a KV cache turns generation from recomputing everything to computing one new token per step, and [Module 21, step 11](../../21_llm_inference_engineering/11_paged_attention_fragmentation/WALKTHROUGH.md) showed how paging manages its memory. This step is about making the cache **smaller by design**.

## The pain point
During generation the model keeps `K` and `V` for every past token in every layer. For Llama-2-7B that is 512 KB per token: a single 128k-token conversation needs 68.7 GB of cache, more than the weights. The cache decides how many users fit on a GPU and how long contexts can be. Memory bandwidth to read it every step, not arithmetic, limits decoding speed.

## The spark
Queries are needed per head, but keys and values can be shared or compressed. **MQA** shares one K/V head among all query heads. **GQA** shares one K/V head per *group* of query heads. **MLA** (DeepSeek) stores one small latent vector per token and rebuilds K and V from it. Sliding-window layers throw away old tokens altogether.

## Goal
Implement MHA, GQA and MQA in one class, a teaching version of MLA, compute KV-cache bytes per token for real configs, draw sliding-window and sink masks, and measure the attention-sink effect in GPT-2.

## The theory (read this before the interview)
**Multi-head attention (MHA).** `h` query heads, `h` key heads, `h` value heads, each of size `hd = d_model / h`. Cache per token per layer: `2 · h · hd` numbers (K and V).

**MQA** (Shazeer 2019): `h` query heads, **1** K/V head. Cache divided by `h`. Quality drops slightly and training can be less stable.

**GQA** (Ainslie et al. 2023): `h` query heads, `g` K/V heads (`g` divides `h`); each K/V head serves `h/g` query heads. `g = h` is MHA, `g = 1` is MQA. Llama-3-8B: 32 query heads, 8 KV heads. Qwen2.5-0.5B: 14 and 2. It is the 2023-2026 default because quality is close to MHA and the cache is `h/g` times smaller. A GQA checkpoint can be made by mean-pooling the K/V heads of an MHA one and fine-tuning briefly (uptraining).

**KV cache size formula**
```
bytes per token = 2 (K and V) * n_layers * n_kv_heads * head_dim * bytes_per_value
cache for a sequence = bytes per token * sequence length * batch size
```
bytes_per_value: 2 for bf16/fp16, 1 for fp8/int8, 0.5 for int4. KV-cache quantisation shrinks it again.

This is only the derivation. The canonical, full memory calculator (weights + KV cache + activations + overhead for a given GPU) is [21_llm_inference_engineering/03_gpu_memory_calculator](../../21_llm_inference_engineering/03_gpu_memory_calculator/WALKTHROUGH.md); use it for capacity planning and come back here for the architecture-side reasons the number is small or large.

**MLA** (DeepSeek-V2/V3). Project the hidden state down to a latent `c = W_down · x` of size `kv_lora_rank` (512 in DeepSeek-V3), cache only `c` (plus a small decoupled rotary key of 64 numbers), and compute `K = W_up_k · c`, `V = W_up_v · c` when needed. Because the up-projections are linear they can be merged ("absorbed") into the query and output projections at inference, so K and V are never materialised. Per token per layer DeepSeek-V3 caches 512 + 64 = 576 values, versus 128 heads x (192 + 128) = 40,960 for plain MHA. MLA keeps MHA-like quality (the DeepSeek-V2 paper reports it better than MHA) and needs the *decoupled RoPE* trick because RoPE (position-dependent rotation, step 02) does not commute with the merge. Kimi K2 and GLM-5 also use MLA.

**Sliding-window / local attention.** A layer attends only to the last `W` tokens: cost and cache are capped at `W`, not growing with the sequence. Information still travels beyond W through depth (receptive field ≈ layers x W). Mistral 7B used it; gpt-oss alternates windows of 128 with full layers; Gemma 3 uses 5 local layers (window 1,024) per 1 global; Llama 4 uses chunked local attention of 8,192 with every 4th layer a global NoPE layer.

**Attention sinks** (Xiao et al. 2023). In trained LLMs the first token receives a large share of attention from all later tokens even when it is semantically empty: softmax must put its mass somewhere. If you evict the first tokens to bound the cache the model breaks; keeping a few "sink" tokens plus a recent window (StreamingLLM) keeps generation stable on very long streams. gpt-oss trains a learned sink logit per head for this purpose. You will measure it below.

**Linear / hybrid attention (preview, details in step 10).** Replace the cache with a fixed-size state updated per token (Mamba, DeltaNet), mixed with a few full-attention layers. Qwen3.5 uses 3 Gated DeltaNet layers per 1 full-attention layer, so only a quarter of layers keep a growing cache.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| MHA / MQA / GQA / MLA | multi-head / multi-query / grouped-query / multi-head latent attention | No | number of K/V heads: h, 1, g, or a compressed latent |
| KV cache | `past_key_values`, `use_cache`, attention cache | Yes | stored K and V of earlier tokens so they are not recomputed |
| `num_key_value_heads` | `n_kv_heads`, `kv_heads`, `num_key_value_heads` in HF configs | Yes | the `g` in GQA; if absent in a config the model is MHA |
| `head_dim` | `d_head`, `hidden_size / num_attention_heads` (unless set explicitly) | Mostly | Qwen3 and gpt-oss set it explicitly; it is not always `d_model / heads` |
| Sliding-window attention | local attention, chunked attention, windowed attention | Related | sliding keeps the last W tokens for every query; chunked splits the sequence into blocks |
| Attention sink | sink token, StreamingLLM | Yes | first tokens soak attention; keep them when evicting |
| Prefill vs decode | prompt processing vs generation | No | prefill is compute-bound (many tokens at once); decode is memory-bound (one token, read the whole cache) |
| FlashAttention vs PagedAttention vs GQA | | No | kernel (step 05) vs memory manager (module 21) vs architecture |

## Run it
```bash
cd 16_modern_llm_architecture/04_attention_variants_and_kv_cache
HF_HUB_OFFLINE=1 python main.py     # ~5 s on CPU (loads cached gpt2 for the sink measurement at the end)
```

## Code, step by step
### 1. Imports and seed  (`main.py` lines 12–16)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
```
- **In simple words:** torch for tensors, `F` for functions, a fixed seed for repeatable random weights.
- **Exactly:** `nn.Module` is the base class for layers; `F.scaled_dot_product_attention` is used in the MLA version.
- **Why it is here:** same numbers in every run.
### 2. Attention constructor: sizes of the four projections  (`main.py` lines 20–28)
```python
class Attention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert n_heads % n_kv_heads == 0                               # each KV head is shared by a whole group of Q heads
        self.h, self.kvh, self.hd = n_heads, n_kv_heads, d_model // n_heads
        self.wq = nn.Linear(d_model, n_heads * self.hd, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.hd, bias=False)   # fewer K/V heads -> smaller projection AND cache
        self.wv = nn.Linear(d_model, n_kv_heads * self.hd, bias=False)
        self.wo = nn.Linear(n_heads * self.hd, d_model, bias=False)
```
- **In simple words:** one class that can be MHA, GQA or MQA depending on how many key/value heads you ask for.
- **Exactly:** `wq: d_model -> n_heads·hd` (all query heads); `wk`, `wv: d_model -> n_kv_heads·hd` (fewer when grouped); `wo` merges heads back. `assert n_heads % n_kv_heads == 0` guarantees equal groups. With d=512, 8 heads, hd = 64.
- **Why it is here:** the **only** difference between MHA, GQA and MQA is the output size of `wk` and `wv`, which also sets the cache size.
### 3. Attention forward pass  (`main.py` lines 30–41)
```python
    def forward(self, x, mask=None):                                   # x: (batch, seq, d_model)
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.h, self.hd).transpose(1, 2)     # (batch, heads, seq, hd)
        k = self.wk(x).view(b, t, self.kvh, self.hd).transpose(1, 2)   # (batch, kv_heads, seq, hd)  <- this is what is cached
        v = self.wv(x).view(b, t, self.kvh, self.hd).transpose(1, 2)
        k = k.repeat_interleave(self.h // self.kvh, dim=1)             # broadcast each KV head to its group: (batch, heads, seq, hd)
        v = v.repeat_interleave(self.h // self.kvh, dim=1)
        if mask is None:
            mask = torch.ones(t, t, dtype=torch.bool).tril()           # causal
        att = (q @ k.transpose(-1, -2)) / self.hd ** 0.5               # (batch, heads, seq, seq)
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)
        return self.wo((att @ v).transpose(1, 2).reshape(b, t, -1))
```
- **In simple words:** each query head looks at the keys of its group, mixes the values, and the heads are glued together.
- **Exactly:** shapes: `q (batch, heads, seq, hd)`, `k, v (batch, kv_heads, seq, hd)`. `repeat_interleave(h // kvh, dim=1)` copies each KV head to its group so shapes become `(batch, heads, seq, hd)`; (real kernels avoid the copy). `att = q @ k^T / sqrt(hd)` is `(batch, heads, seq, seq)`; masked positions get `-inf` so softmax gives 0; `att @ v` gives `(batch, heads, seq, hd)`; `transpose(1, 2).reshape(b, t, -1)` merges heads to `(batch, seq, h·hd)` and `wo` projects to `d_model`.
- **Why it is here:** the cached tensors are `k` and `v` *before* the `repeat_interleave`: that is where GQA saves memory.
### 4. Cache size per token per layer  (`main.py` lines 43–44)
```python
    def cache_floats_per_token(self):
        return 2 * self.kvh * self.hd                                  # K and V, per layer
```
- **Exactly:** `2 · kv_heads · head_dim` numbers (K and V).
- **In simple words:** how many numbers one layer must remember per token.
- **Why it is here:** the whole point of the variants; the table below shows it.
### 5. Compare MHA, GQA-2 and MQA  (`main.py` lines 47–53)
```python
D, H = 512, 8
x = torch.randn(2, 10, D)
print(f"{'variant':<10}{'n_kv_heads':>11}{'params':>10}{'cache floats/token/layer':>26}")
for name, kvh in [("MHA", 8), ("GQA-2", 2), ("MQA", 1)]:
    m = Attention(D, H, kvh)
    assert m(x).shape == (2, 10, D)
    print(f"{name:<10}{kvh:>11}{sum(p.numel() for p in m.parameters()):>10,}{m.cache_floats_per_token():>26,}")
```
- **In simple words:** build the three variants and count parameters and cache numbers.
- **Exactly:** MHA (8 KV heads): 1,048,576 params, 1,024 cache numbers/token/layer. GQA-2: 655,360 params, 256 (4x less). MQA: 589,824 params, 128 (8x less). The saving also reduces parameters in `wk`/`wv`.
- **Why it is here:** the trade-off in numbers: cache shrinks with `h/g` and parameters shrink a little too.
### 6. MHA is just GQA with g = h  (`main.py` lines 56–58)
```python
a, b = Attention(D, H, 8), Attention(D, H, 8)
b.load_state_dict(a.state_dict())
print("same weights, same output:", torch.allclose(a(x), b(x)))
```
- **Exactly:** two modules with identical weights give identical output (`True`); with `n_kv_heads == n_heads` the `repeat_interleave` copies each head once, i.e. nothing.
- **In simple words:** MHA, GQA and MQA are one family, not three different algorithms.
- **Why it is here:** proves the "knob" interpretation.
### 7. Simplified MLA  (`main.py` lines 62–80)
```python
class MLASimple(nn.Module):
    """Teaching version: no decoupled RoPE branch (the real one adds a small separate rotary key part)."""
    def __init__(self, d_model, n_heads, d_latent):
        super().__init__()
        self.h, self.hd = n_heads, d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.down = nn.Linear(d_model, d_latent, bias=False)           # compress: this (batch, seq, d_latent) IS the cache
        self.up_k = nn.Linear(d_latent, d_model, bias=False)           # expand latent -> per-head keys
        self.up_v = nn.Linear(d_latent, d_model, bias=False)           # expand latent -> per-head values
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        b, t, _ = x.shape
        latent = self.down(x)                                          # (batch, seq, d_latent)  cached
        q = self.wq(x).view(b, t, self.h, self.hd).transpose(1, 2)
        k = self.up_k(latent).view(b, t, self.h, self.hd).transpose(1, 2)
        v = self.up_v(latent).view(b, t, self.h, self.hd).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(o.transpose(1, 2).reshape(b, t, -1))
```
- **In simple words:** instead of storing keys and values, store one short summary vector per token and rebuild keys and values from it when needed.
- **Exactly:** `down: 512 -> 64` is the latent (this `(batch, seq, 64)` tensor is the cache); `up_k`, `up_v: 64 -> 512` rebuild per-head K and V; `F.scaled_dot_product_attention(q, k, v, is_causal=True)` does the masked softmax attention. Teaching simplification: real MLA also compresses the query, absorbs the up-projections into `wq`/`wo`, and adds a separate 64-dim RoPE key that is cached too.
- **Why it is here:** shows that K and V can be low-rank functions of a small latent, the idea behind DeepSeek's large cache reductions (71x versus plain MHA in the table below).
### 8. MLA cache size  (`main.py` lines 83–84)
```python
mla = MLASimple(D, H, d_latent=64)
print(f"\nMLA: out {tuple(mla(x).shape)}, cache floats/token/layer = 64 (MHA would be {2 * D}) -> {2 * D / 64:.0f}x smaller")
```
- **Exactly:** output shape `(2, 10, 512)`; cache 64 numbers per token per layer vs 1,024 for MHA: 16x smaller.
- **In simple words:** a 16-number summary replaces a 16-times-bigger record.
- **Why it is here:** quantifies the compression for the toy.
### 9. The KV-cache formula as code  (`main.py` lines 89–90)
```python
def kv_bytes_per_token(layers, kv_heads, head_dim, dtype_bytes=2):
    return 2 * layers * kv_heads * head_dim * dtype_bytes
```
- **Exactly:** `2 · layers · kv_heads · head_dim · dtype_bytes`; default 2 bytes (bf16).
- **In simple words:** two things (K and V) x layers x heads x size x bytes.
- **Why it is here:** the formula every inference-capacity question uses.
### 10. Real configs table  (`main.py` lines 93–108)
```python
configs = [  # (name, layers, kv_heads, head_dim)  all read from the public config.json files
    ("GPT-2 small (MHA)", 12, 12, 64),
    ("Llama-2-7B (MHA)", 32, 32, 128),
    ("Llama-3-8B (GQA)", 32, 8, 128),
    ("Llama-3-70B (GQA)", 80, 8, 128),
    ("Qwen2.5-0.5B (GQA)", 24, 2, 64),
]
print(f"\n{'model':<22}{'layers':>7}{'kv_heads':>9}{'head_dim':>9}{'KB/token':>10}{'GB @128k tokens':>17}")
for name, L, kv, hd in configs:
    b = kv_bytes_per_token(L, kv, hd)
    print(f"{name:<22}{L:>7}{kv:>9}{hd:>9}{b / 1024:>10.1f}{b * 131072 / 1e9:>17.1f}")
# MLA (DeepSeek-V3): per layer cache = kv_lora_rank 512 + rope key dim 64 = 576 values (not x2: one shared latent)
mla_b = 61 * (512 + 64) * 2
print(f"{'DeepSeek-V3 (MLA)':<22}{61:>7}{'latent':>9}{'576':>9}{mla_b / 1024:>10.1f}{mla_b * 131072 / 1e9:>17.1f}")
mha_equiv = 61 * 128 * (192 + 128) * 2                  # layers x heads x (K dim 128+64 rope, V dim 128) x 2 bytes (bf16)
print(f"  (DeepSeek-V3 with plain MHA, 128 heads x (192 K + 128 V) dims, would need {mha_equiv / 1024:.0f} KB/token = {mha_equiv / mla_b:.0f}x more)")
```
- **In simple words:** plug public numbers from model configs into the formula.
- **Exactly:** GPT-2 36 KB/token (4.8 GB at 128k, if it could); Llama-2-7B (MHA) 512 KB/token = 68.7 GB at 128k; Llama-3-8B (GQA, 8 KV heads) 128 KB = 17.2 GB; Llama-3-70B 320 KB = 42.9 GB; Qwen2.5-0.5B 12 KB = 1.6 GB; DeepSeek-V3 (MLA, 61 layers x 576 values) 68.6 KB = 9.2 GB, versus 4,880 KB/token (71x more) if it used plain MHA with 128 heads of (128+64 key, 128 value) dims. Going from Llama-2-7B to Llama-3-8B (MHA to GQA) cut the cache 4x for a bigger model.
- **Why it is here:** this is the capacity-planning arithmetic: `GB at context = KB/token x tokens`. Weights of the 7B model are about 14 GB, so a 128k MHA cache is 5x the weights.
### 11. Sliding-window mask  (`main.py` lines 111–113)
```python
def sliding_window_mask(t, window):
    i, j = torch.arange(t)[:, None], torch.arange(t)[None, :]
    return (j <= i) & (j > i - window)                                 # causal AND within the last `window` tokens
```
- **In simple words:** a token may look at itself and the previous W-1 tokens only.
- **Exactly:** `(j <= i)` is causal; `(j > i - window)` is within the window; `&` combines them. Shape `(t, t)` boolean.
- **Why it is here:** the mask that bounds both compute and cache per layer.
### 12. Mask picture and local:global saving  (`main.py` lines 116–123)
```python
print("\nsliding window 3, T=8 (1 = visible):")
print(sliding_window_mask(8, 3).int())
# Gemma-3 style: 5 local layers (window 1024) for every 1 global layer -> cache of local layers is capped
T, win, layers, glob_every = 128_000, 1024, 48, 6
n_global = layers // glob_every
local_tokens = min(T, win)
frac = (n_global * T + (layers - n_global) * local_tokens) / (layers * T)
print(f"5:1 local:global, T={T:,}, window={win}: cache is {frac:.1%} of an all-global model")
```
- **Exactly:** the printed 8x8 matrix is a diagonal band 3 wide. With 48 layers, 1 global per 6 and window 1,024 at 128,000 tokens the cache is 17.3% of an all-global model (Gemma 3 style).
- **In simple words:** five of every six layers forget almost everything, which cuts the cache by 5x.
- **Why it is here:** the memory argument for hybrid local/global stacks.
### 13. Sliding window plus sink tokens  (`main.py` lines 126–129)
```python
def sink_mask(t, window, n_sink):
    m = sliding_window_mask(t, window)
    m[:, :n_sink] |= torch.arange(t)[:, None] >= torch.arange(n_sink)[None, :]   # always allow the first n_sink tokens (causally)
    return m
```
- **In simple words:** like the sliding window but the first `n_sink` tokens always stay visible.
- **Exactly:** `m[:, :n_sink] |= (row index >= column index)` marks column 0 visible for all later rows (still causal).
- **Why it is here:** the StreamingLLM recipe.
### 14. Print the sink mask  (`main.py` lines 132–133)
```python
print("\nwindow 3 + 1 sink token, T=8:")
print(sink_mask(8, 3, 1).int())
```
- **Exactly:** column 0 is 1 in every row; from row 4 on a gap of zeros separates it from the 3-wide band (row 4 reads `1 0 1 1 1 0 0 0`).
- **In simple words:** the first token is never forgotten.
- **Why it is here:** makes the mask concrete.
### 15. Imports for the GPT-2 measurement  (`main.py` lines 135–137)
```python
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoModelForCausalLM, AutoTokenizer
```
- **Exactly:** offline flag set before importing transformers.
- **In simple words:** we now check the sink claim on a real model.
- **Why it is here:** claims about LLMs should be measured, not just stated.
### 16. Run GPT-2 and get attention weights  (`main.py` lines 138–141)
```python
tok, lm = AutoTokenizer.from_pretrained("gpt2"), AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager").eval()
ids = tok("The quick brown fox jumps over the lazy dog because it was a sunny day in the park.", return_tensors="pt")
with torch.no_grad():
    attn = lm(**ids, output_attentions=True).attentions               # tuple of 12 x (1, heads, seq, seq)
```
- **Exactly:** `attn_implementation="eager"` is required, because fused kernels (SDPA) do not return the attention matrix; `output_attentions=True` returns a tuple of 12 tensors, each `(1, 12 heads, seq, seq)`.
- **In simple words:** run the model and keep every attention table so we can inspect them.
- **Why it is here:** the fast path hides the weights, so we use the slow path for inspection.
### 17. Attention paid to token 0, per layer  (`main.py` lines 142–144)
```python
seq = ids["input_ids"].shape[1]
first = torch.stack([a[0, :, 1:, 0].mean() for a in attn])            # mean weight on token 0 from later tokens, per layer
print(f"gpt2: attention paid to token 0 by later tokens, per layer: {first.numpy().round(2)} (uniform attention would give {sum(1 / (i + 1) for i in range(1, seq)) / (seq - 1):.2f})")
```
- **Exactly:** `a[0, :, 1:, 0].mean()` = average over heads and over later query positions of the weight on key 0. Result per layer: `[0.19 0.37 0.35 0.52 0.55 0.71 0.7 0.76 0.71 0.79 0.79 0.65]`, versus 0.14 if attention were uniform. In middle and late layers about 70-79% of all attention goes to the very first token.
- **In simple words:** the first word of the sentence is used as a "dump" for attention that has nothing better to do.
- **Why it is here:** evidence for sinks: if you evict token 0 the softmax mass has nowhere to go and the model degrades.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `x.view(b, t, h, hd)` | reshape without copying | split `d_model` into heads |
| `.transpose(1, 2)` | swap two axes | `(b, t, h, hd) -> (b, h, t, hd)` |
| `k.repeat_interleave(r, dim=1)` | repeat each slice r times in place | `[a b] -> [a a b b]` |
| `mask.tril()` | lower triangle | causal mask |
| `masked_fill(~mask, -inf)` | set masked positions to -inf | before softmax |
| `x \|= y` | in-place logical OR | add sink columns |
| `assert cond` | stop if false | head divisibility |
| `a[0, :, 1:, 0]` | indexing: batch 0, all heads, queries 1.., key 0 | attention to the first token |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `F.scaled_dot_product_attention` | torch | fused attention | `q, k, v, is_causal, attn_mask` | `(b, h, t, hd)` |
| `nn.Linear(in, out, bias=False)` | torch | linear map | sizes | module |
| `torch.stack` | torch | stack tensors on a new axis | list, dim | tensor |
| `AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager")` | transformers | load with the explicit attention code path | | model |
| `output_attentions=True` | transformers | return attention weights | | tuple per layer |

## Execution trace (real run)
```text
variant    n_kv_heads    params  cache floats/token/layer
MHA                 8 1,048,576                     1,024
GQA-2               2   655,360                       256
MQA                 1   589,824                       128
same weights, same output: True

MLA: out (2, 10, 512), cache floats/token/layer = 64 (MHA would be 1024) -> 16x smaller

model                  layers kv_heads head_dim  KB/token  GB @128k tokens
GPT-2 small (MHA)          12       12       64      36.0              4.8
Llama-2-7B (MHA)           32       32      128     512.0             68.7
Llama-3-8B (GQA)           32        8      128     128.0             17.2
Llama-3-70B (GQA)          80        8      128     320.0             42.9
Qwen2.5-0.5B (GQA)         24        2       64      12.0              1.6
DeepSeek-V3 (MLA)          61   latent      576      68.6              9.2
  (DeepSeek-V3 with plain MHA, 128 heads x (192 K + 128 V) dims, would need 4880 KB/token = 71x more)

sliding window 3, T=8 (1 = visible):
tensor([[1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 1]], dtype=torch.int32)
5:1 local:global, T=128,000, window=1024: cache is 17.3% of an all-global model

window 3 + 1 sink token, T=8:
tensor([[1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 0, 0, 0],
        [1, 0, 0, 1, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 1, 1, 0],
        [1, 0, 0, 0, 0, 1, 1, 1]], dtype=torch.int32)
gpt2: attention paid to token 0 by later tokens, per layer: [0.19 0.37 0.35 0.52 0.55 0.71 0.7  0.76 0.71 0.79 0.79 0.65] (uniform attention would give 0.14)
```
Reading it: the cache column is exactly `2 · kv_heads · hd`. Notice that GQA changes parameters only a little (1.05M to 0.66M) but the cache 4x. The DeepSeek-V3 row has 61 layers but only 68.6 KB/token because one 576-value latent replaces K and V for 128 heads.

## Experiments (change ONE thing)
Every row below was run on a temporary copy of `main.py` (the real file is untouched). To keep the runs fast, the copies stop before the gpt2 attention-sink block at the end (it loads a model and none of these changes affect it). Only the named line changed.

| Change | What you will see | Why |
|---|---|---|
| add `("GQA-4", 4)` to the variant list (measured) | params 786,432 and 512 cache floats/token/layer, exactly between MHA (1,048,576 / 1,024) and GQA-2 (655,360 / 256) | Cache and the K/V projections both scale linearly with `n_kv_heads`. The query and output projections stay the same size, which is why parameters shrink less than the cache. |
| `dtype_bytes=1` in `kv_bytes_per_token` (measured) | Llama-3-8B 64.0 KB/token (was 128.0), 8.6 GB at 128k (was 17.2); Llama-3-70B 160.0 KB/token (was 320.0) | Cache bytes are linear in bytes per value: an 8-bit cache halves bf16/fp16. Module 21 (inference engineering) covers quantised KV caches. |
| add a row "Llama-3-8B if MHA" with 32 kv heads (measured) | 512.0 KB/token and 68.7 GB at 128k, 4x the real GQA row (128.0 KB, 17.2 GB) | Going from 8 KV heads to 32 quadruples the cache; GQA's 4x saving is bought with sharing K/V across 4 query heads each. |
| `d_latent=128` instead of 64 for MLA (measured) | cache floats/token/layer 128 and "8x smaller" (was 64, "16x smaller") | MLA's cache is the latent width, so the saving is `2 * d_model / d_latent`. A wider latent keeps more information, a narrower one saves more memory. |
| local window 4096 instead of 1024 in the 5:1 local:global layout (measured) | cache is 19.3% of an all-global model (was 17.3%) | Local layers cache only `window` tokens, global layers cache all `T`. Four times the window means more cached tokens in 5 of every 6 layers, but the global layers still dominate. |

## Debugging practice
**BREAK IT drill 1: heads that cannot be grouped.** Build a GQA layer with 3 KV heads for 8 query heads:

```diff
-    m = Attention(D, H, kvh)
+    m = Attention(D, H, 3)
```

Symptom (real run):

```text
    assert n_heads % n_kv_heads == 0   # each KV head is shared by a whole group of Q heads
AssertionError
```

Root cause: `8 % 3 = 2`, so the query heads cannot be split into equal groups. Fix: choose `n_kv_heads` that divides `n_heads` (1, 2, 4, 8 here). The assert is deliberate: without it `repeat_interleave(8 // 3 = 2)` would silently build 6 heads and fail later with a confusing shape error.

**BREAK IT drill 2: delete the causal mask (a silent bug).**

```diff
-        att = att.masked_fill(~mask, float("-inf")).softmax(-1)
+        att = att.softmax(-1)
```

Symptom (real run, with a check line I added to the copy that changes only the last token and looks at the output at token 0): the correct file gives `0.0`; the broken one gives `0.1538`. No error is raised, shapes are all fine.

Root cause: without the mask each token attends to the future, so token 0's output depends on the last token. A language model trained this way would cheat on next-token prediction and then fail at generation time, when the future does not exist. Fix: restore `masked_fill(~mask, -inf)` before the softmax. Cheap guard: this "perturb the last token" test costs three lines and catches every causal-mask bug.

**BREAK IT drill 3: wrong grouping of KV heads (also silent).**

```diff
-        k = k.repeat_interleave(self.h // self.kvh, dim=1)
+        k = k.repeat(1, self.h // self.kvh, 1, 1)
```

Symptom (real run): no error and the same shapes. The GQA-2 output checksum changes from `65.1073` to `61.9098`.

Root cause: `repeat_interleave` gives `[k0 k0 k0 k0 k1 k1 k1 k1]` (each KV head serves a block of neighbouring query heads); `repeat` gives `[k0 k1 k0 k1 ...]`, and `v` still uses the other layout, so keys and values are mismatched. Real checkpoints were trained with the interleaved layout, so loading them would give garbage text. Fix: use the same `repeat_interleave` for K and V.

## Common mistakes
- Quoting KV-cache size per layer instead of per token across all layers (the formula multiplies by `n_layers`), or forgetting the factor 2 for K and V (MLA is the exception: one shared latent).
- Using `n_heads` instead of `n_kv_heads` in the cache formula for GQA models. That overstates Llama-3-8B by 4x.
- Thinking GQA or MQA reduces compute of the score matrix: it does not, every query head still does its own `q @ k^T`. It shrinks cache and memory traffic.
- Forgetting that a sliding-window layer still sees older information indirectly, through the layers below it (the receptive field grows with depth).
- Evicting the first tokens when you slide the window: attention sinks need to stay, or quality collapses.
- Reading `head_dim` as `d_model / n_kv_heads`. It is `d_model / n_heads` (and for some models it is set explicitly in the config).

## Rebuild checklist
- [ ] attention with `n_kv_heads` and `repeat_interleave` to the group size
- [ ] the cache formula `2 * L * kv_heads * hd * bytes`
- [ ] explain what MLA caches (latent + small RoPE key) and why RoPE needs a decoupled part
- [ ] a sliding-window mask `(j <= i) & (j > i - W)`
- [ ] why keeping the first tokens prevents breakage when evicting

## The new problem (cliffhanger)
GQA and MLA shrink *what we store*. But even the attention computation itself has a problem: the naive code builds a `(seq, seq)` score matrix. At 128k tokens that is 16 billion entries per head, 32 GB in fp16, and the time goes into moving it between GPU memory and the chip rather than arithmetic. Next: [Step 05 — FlashAttention and the online softmax](../05_flash_attention_online_softmax/WALKTHROUGH.md).

> **Say it aloud (interview box):** "The KV cache costs 2 x layers x kv_heads x head_dim x bytes per token. GQA shares each KV head across a group of query heads, cutting the cache by h/g with small quality loss; MQA is the g=1 limit; MLA caches one low-rank latent per token instead, which is why DeepSeek-V3 needs only about 70 KB per token. Sliding-window layers cap the cache, and attention sinks explain why you must keep the first tokens when evicting."
