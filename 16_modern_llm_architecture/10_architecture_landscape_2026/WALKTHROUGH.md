# Step 10 — The architecture landscape (2024-2026) and what comes next

## The story so far
[Step 08](../08_read_a_real_config/WALKTHROUGH.md) closed the loop: the decoder built in steps 01-07 matches Hugging Face's Llama and we can read a real `config.json`. This last step is a reference: how do the real models differ, and which of the ideas you built show up where? It also covers three topics that sit around the architecture: reasoning models (test-time compute), long context, and non-attention (state-space and linear) layers.

## The pain point
Model announcements read like a wall of acronyms: "671B-A37B, MLA, 256 experts, MTP, 128k context". Without a way to place each number you cannot compare models, estimate their memory, or answer "which would you deploy and why?".

## The spark
Almost every open model is the same skeleton (step 07) with a small set of switches: attention type (GQA, MLA, sliding/chunked, linear hybrid), FFN type (dense SwiGLU or MoE with shared and routed experts), position scheme (RoPE plus scaling, sometimes NoPE layers), and norm placement. Read a config as a list of switch settings, then compute the two numbers that matter operationally: weights memory (total parameters) and KV cache per token.

## Goal
Print a table computed from a dictionary of public config values: parameters (total/active), KV-cache bytes per token and at 128k tokens, with `n/a` wherever a number was not verified; list what changed since GPT-2; quantify the MoE bargain.

## The theory (read this before the interview)
**The 2026 default decoder** (what step 07 builds, plus options): byte-level BPE with a 128k-260k vocabulary (01); RoPE with a large base and YaRN-style scaling (02); pre-norm RMSNorm, sometimes QK-norm and post-sublayer norm (03); SwiGLU/GeGLU or MoE (03, 06); GQA, MLA or hybrid local/global attention (04); FlashAttention/SDPA kernels (05); no biases (except sometimes q, k, v); tied embeddings only in small models (07); optional multi-token-prediction heads (09).

**What each model changed (facts from the model cards and configs checked in Nov 2026, see References)**
| Model | Notable choices |
|---|---|
| GPT-2 (2019) | learned absolute positions (1,024), LayerNorm, GELU MLP, MHA, tied embeddings, BPE 50,257 |
| Llama 3.1 8B (2024) | GQA 32 query / 8 KV heads, SwiGLU (intermediate 14,336), RoPE theta 500,000 with `llama3` scaling (factor 8 over 8,192) to 131k, vocab 128,256, RMSNorm |
| Qwen3 (2025) | dense 8B: 36 layers, 32/8 heads, head_dim 128, vocab 151,936; MoE 235B-A22B: 94 layers, 64 query / 4 KV heads, 128 experts top-8, QK-norm; RoPE theta 1,000,000 |
| DeepSeek-V3 (Dec 2024) | 671B total / 37B active; 61 layers, hidden 7,168, MLA (kv_lora_rank 512, 128 heads), first 3 layers dense then MoE with 256 routed + 1 shared expert, top-8, sigmoid routing, aux-loss-free balancing, multi-token prediction, FP8 training, YaRN to 163,840 positions |
| DeepSeek-R1 (Jan 2025) | the DeepSeek-V3 architecture trained with large-scale reinforcement learning to produce long reasoning traces; an *training* change, not an architecture change |
| Llama 4 Scout (Apr 2025) | 109B total / 17B active, 16 experts top-1 plus shared expert in every layer, GQA 40/8, chunked local attention (8,192) in three layers out of four and a global NoPE (no RoPE) layer every fourth layer, QK-norm, 10M-token configured context, natively multimodal |
| Gemma 3 27B (2025) | 62 layers, 32 query / 16 KV heads, head_dim 128, 5 local (sliding window 1,024) : 1 global layers, different RoPE bases for local (10,000) and global (1,000,000), linear rope scaling factor 8 for 128k, vocab 262k |
| gpt-oss-120b (Aug 2025) | 116.8B total / 5.1B active; 36 layers; 128 experts, top-4; GQA 64/8 with head_dim 64; alternating sliding-window (128) and full-attention layers; YaRN factor 32 over 4,096 to 131,072; attention biases; MXFP4-quantised MoE weights (fits one 80 GB GPU) |
| Kimi K2 (2025) | 1T total / 32B active; DeepSeek-V3 architecture scaled up: 61 layers, 384 routed + 1 shared expert, top-8, MLA, 64 heads, 128k context; vocabulary 163,840 |
| Qwen3.5-397B-A17B (Feb 2026) | 397B total / 17B active; 60 layers in 15 groups of (3 Gated DeltaNet + MoE, then 1 Gated Attention + MoE); 512 experts, 10 routed + 1 shared; gated attention uses 32 query / 2 KV heads, head_dim 256; 262,144 native context, extensible to about 1M; Apache-2.0; natively multimodal |
| GLM-5 (Feb 2026) | 744B total / 40B active; 78 layers; MLA plus DeepSeek Sparse Attention; 256 experts, 8 routed + 1 shared (per Raschka's survey of the model) |
| DeepSeek-V4-Pro (Apr 2026 preview) | 1.6T total / 49B active; hybrid of Compressed Sparse Attention and Heavily Compressed Attention; 1M-token context; FP4 expert weights + FP8 elsewhere; manifold-constrained hyper-connections; Muon optimizer; MIT licence (per the model card; layer and expert counts not stated there) |
| Kimi K2.5 (Jan 2026) | 1T total, MLA-based DeepSeek-V3-style, natively multimodal (per Raschka's survey; active parameters not stated there) |
Smaller 2026 data points from the same survey: Arcee Trinity Large 400B/13B active (3:1 local:global, NoPE in global layers), Step 3.5 Flash 196B/11B active (multi-token prediction with 3 extra tokens), Qwen3-Coder-Next 80B/3B active (3:1 Gated DeltaNet : gated attention), MiniMax M2.5 230B (plain GQA), Nanbeige 4.1 3B (Llama-3.2-like).

**Trends worth being able to state.** (1) Sparsity everywhere: MoE with fine-grained experts and a few shared ones; active parameters 3-6% of total in the largest models. (2) KV-cache engineering: GQA to MLA to compressed/sparse attention, local:global patterns, linear-attention hybrids. (3) Positions: RoPE + YaRN; some layers drop RoPE (NoPE) to extrapolate. (4) Quantised weights in the checkpoint (FP8, MXFP4, FP4). (5) Multi-token prediction as both a training signal and a draft head for speculative decoding. (6) Architecture is converging; the differences in quality come mostly from data, post-training and RL, not the block design.

**Reasoning models and test-time compute.** A reasoning model is an ordinary decoder (often the same architecture as its non-reasoning sibling) post-trained with reinforcement learning on verifiable tasks so that it generates a long chain of thought before the answer (OpenAI o1, DeepSeek-R1, Qwen3 "thinking" mode, DeepSeek-V4 "Think High/Max" modes). Accuracy then scales with the number of generated tokens ("test-time compute"), so serving cost scales with output length, which makes decode speed, KV-cache size and speculative decoding (step 09) more important. Controls: a thinking budget, `reasoning_effort` in some APIs, or separate "thinking/non-thinking" modes in one model. Training signal: outcome rewards on math/code; DeepSeek-R1 used GRPO (a critic-free policy-gradient method), see module 17 for fine-tuning and alignment.

**Long context.** Three separate problems: (a) *position handling* (RoPE scaling, step 02), (b) *compute and memory* (FlashAttention, GQA/MLA, local layers, paged cache; steps 04-05), (c) *quality* (the model must be trained on long documents; "needle in a haystack" tests check recall, harder tasks check reasoning over the whole input). A configured 1M-token window does not guarantee 1M tokens of reliable use. For RAG vs long context see [module 14](../../14_rag_and_retrieval/README.md).

**State-space and hybrid models.** Attention costs O(N) per generated token in time and memory (it reads the whole cache). A **state-space model** such as **Mamba** (Gu and Dao 2023) replaces attention by a recurrence with a fixed-size state: O(1) per token and no growing cache; the selective (input-dependent) state update lets it match transformers on some language tasks, but exact recall of specific earlier tokens is weaker. **Linear attention / DeltaNet** variants are in the same family: a matrix-valued state updated per token with a gated delta rule. **Hybrids** keep a few full-attention layers for recall and use linear/SSM layers for the rest: Jamba (AI21), Qwen3-Next/Qwen3.5 (3 Gated DeltaNet : 1 attention), Ling/Ring 2.5 (Lightning Attention + MLA). In the table the cache of Qwen3.5-397B is small because only 15 of 60 layers keep a per-token cache (the DeltaNet layers hold a fixed-size state, which the table does not count).

**Always-current caveat.** The 2026 list changes monthly. Treat the table as a snapshot, re-check numbers on the model card before quoting them, and prefer the *reasoning* (why a design reduces cache or cost) over memorising numbers.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Active vs total parameters | "A37B", activated params, "671B-A37B" | No | active = used per token; total = must be stored |
| MoE | sparse model, experts model, "sparse FFN" | Yes | many MLPs, few active |
| Dense model | non-MoE | No | all parameters used for every token |
| Test-time compute | inference-time scaling, thinking budget, "reasoning" | Related | more generated tokens at inference for a better answer |
| Reasoning model | thinking model, "R1-style", chain-of-thought model | Mostly | same architecture, RL-post-trained to think long |
| Context window | context length, `max_position_embeddings`, "1M context" | Mostly | configured limit, not guaranteed quality |
| SSM | state-space model, Mamba-style, recurrent layer | Related | fixed-size state instead of a growing cache |
| Hybrid model | hybrid attention, linear + full attention | Related | mixes attention with SSM/linear layers |
| Open-weight vs open-source | open weights | No | weights downloadable (licence varies) vs fully open code/data/licence |

## Run it
```bash
cd 16_modern_llm_architecture/10_architecture_landscape_2026
python main.py       # < 1 s, standard library only
```

## Code, step by step
### 1. The data: public config numbers (first rows)  (`main.py` lines 17–20)
```python
MODELS = {
    "GPT-2 small (2019)":    dict(total=0.124e9, active=0.124e9, L=12, kv_heads=12, head_dim=64, attn="MHA", ffn="GELU MLP", pos="learned abs", ctx=1_024, norm="LayerNorm pre"),
    "Llama 3.1 8B (2024)":   dict(total=8.0e9, active=8.0e9, L=32, kv_heads=8, head_dim=128, attn="GQA", ffn="SwiGLU", pos="RoPE 5e5 + scaling", ctx=131_072, norm="RMSNorm pre"),
    "Qwen2.5-0.5B (2024)":   dict(total=0.494e9, active=0.494e9, L=24, kv_heads=2, head_dim=64, attn="GQA", ffn="SwiGLU", pos="RoPE 1e6", ctx=32_768, norm="RMSNorm pre"),
```
- **In simple words:** a dictionary with one entry per model, holding only numbers that were read from a config or model card.
- **Exactly:** keys: `total`, `active` (parameters), `L` (layers), `kv_heads`, `head_dim`, text fields `attn`, `ffn`, `pos`, `norm`, `ctx`; optional `global_layers` (layers with full attention), `local_window`, and `mla_dim` for MLA models (`kv_lora_rank + rope dim` = 512 + 64). Rows GPT-2, Llama 3.1 8B and Qwen2.5-0.5B.
- **Why it is here:** one place to edit when a model card changes; the arithmetic below never repeats a number by hand.
### 2. Qwen3 rows  (`main.py` lines 21–22)
```python
    "Qwen3-8B (2025)":       dict(total=8.2e9, active=8.2e9, L=36, kv_heads=8, head_dim=128, attn="GQA + QK-norm", ffn="SwiGLU", pos="RoPE 1e6", ctx=40_960, norm="RMSNorm pre"),
    "Qwen3-235B-A22B":       dict(total=235e9, active=22e9, L=94, kv_heads=4, head_dim=128, attn="GQA + QK-norm", ffn="MoE 128e top-8", pos="RoPE 1e6", ctx=40_960, norm="RMSNorm pre"),
```
- **Exactly:** Qwen3-8B: 36 layers, 8 KV heads, head_dim 128 (total 8.2B derived from the config: 36 layers x (attention 41.9M + MLP 151M) + embeddings 2 x 151,936 x 4,096 = 8.2B). Qwen3-235B-A22B: 94 layers, 4 KV heads, head_dim 128, 235B/22B.
- **In simple words:** the dense and the MoE member of the same family.
- **Why it is here:** same family, 2.3% vs 100% of parameters active.
### 3. gpt-oss row  (`main.py` lines 23–24)
```python
    "gpt-oss-120b (2025)":   dict(total=116.8e9, active=5.1e9, L=36, kv_heads=8, head_dim=64, attn="GQA, alt. sliding(128)/full", ffn="MoE 128e top-4", pos="RoPE+YaRN", ctx=131_072, norm="RMSNorm pre",
                                  global_layers=18, local_window=128),
```
- **Exactly:** 116.8B total, 5.1B active, 36 layers, 8 KV heads, head_dim 64, window 128 in the 18 sliding layers and full attention in the other 18.
- **In simple words:** half the layers only remember the last 128 tokens.
- **Why it is here:** the cache is mostly paid by the 18 global layers.
### 4. Gemma 3 row  (`main.py` lines 25–26)
```python
    "Gemma 3 27B (2025)":    dict(total=27e9, active=27e9, L=62, kv_heads=16, head_dim=128, attn="GQA, 5 local(1024):1 global", ffn="GeGLU dense", pos="RoPE (local 1e4 / global 1e6)", ctx=131_072, norm="RMSNorm pre+post",
                                  global_layers=10, local_window=1024),
```
- **Exactly:** 62 layers, 16 KV heads, head_dim 128; 10 global layers (every 6th) and 52 local layers with window 1,024 (config `sliding_window_pattern` = 6).
- **In simple words:** five local layers for each global one.
- **Why it is here:** the Gemma pattern from step 04.
### 5. Llama 4 Scout row  (`main.py` lines 27–28)
```python
    "Llama 4 Scout (2025)":  dict(total=109e9, active=17e9, L=48, kv_heads=8, head_dim=128, attn="GQA, chunked(8192) + 1/4 NoPE global", ffn="MoE 16e top-1 (+shared)", pos="RoPE / NoPE", ctx=10_485_760, norm="RMSNorm pre + QK-norm",
                                  global_layers=12, local_window=8192),
```
- **Exactly:** 48 layers, 8 KV heads, head_dim 128; 12 layers (every 4th, NoPE) attend to everything, 36 layers use chunked attention of 8,192.
- **In simple words:** a few layers see the whole 10M window, the rest see local chunks.
- **Why it is here:** iRoPE-style design (interleaved NoPE and RoPE layers).
### 6. DeepSeek-V3 row  (`main.py` lines 29–29)
```python
    "DeepSeek-V3 (2024-12)": dict(total=671e9, active=37e9, L=61, mla_dim=512 + 64, attn="MLA", ffn="MoE 256e+1 shared top-8", pos="RoPE+YaRN (decoupled)", ctx=163_840, norm="RMSNorm pre"),
```
- **Exactly:** 671B total, 37B active, 61 layers, `mla_dim = 512 + 64` (kv_lora_rank + decoupled RoPE key), no `kv_heads` since the latent is shared across heads.
- **In simple words:** one 576-number record per token per layer.
- **Why it is here:** MLA is computed with a different formula in `kv_bytes`.
### 7. Kimi K2 row  (`main.py` lines 30–30)
```python
    "Kimi K2 (2025)":        dict(total=1.0e12, active=32e9, L=61, mla_dim=512 + 64, attn="MLA (DeepSeek-V3 design)", ffn="MoE 384e+1 shared top-8", pos="RoPE+YaRN", ctx=131_072, norm="RMSNorm pre"),
```
- **Exactly:** 1T total, 32B active, 61 layers, the same MLA dimensions as DeepSeek-V3 (the model is a `DeepseekV3ForCausalLM` in its config, with 384 experts and 64 heads).
- **In simple words:** the DeepSeek-V3 design with more experts and fewer heads.
- **Why it is here:** shows how a design is reused and scaled.
### 8. Qwen3.5 row (hybrid)  (`main.py` lines 31–32)
```python
    "Qwen3.5-397B-A17B":     dict(total=397e9, active=17e9, L=60, kv_heads=2, head_dim=256, attn="3 Gated DeltaNet : 1 gated attn", ffn="MoE 512e+1 shared, 10 routed", pos="RoPE (attn layers)", ctx=262_144, norm="RMSNorm",
                                  global_layers=15, local_window=0),         # DeltaNet layers keep a fixed-size state: 0 per-token cache
```
- **Exactly:** 60 layers of which 15 are gated full attention (2 KV heads, head_dim 256); the other 45 are Gated DeltaNet layers whose state size does not grow with context, so `local_window=0` means "no per-token cache" for them. 397B total, 17B active.
- **In simple words:** three quarters of the layers keep a fixed-size memory instead of a growing cache.
- **Why it is here:** the hybrid-attention example; the DeltaNet state is not counted in our KV number.
### 9. Rows with unverified fields  (`main.py` lines 33–35)
```python
    "GLM-5 (2026)":          dict(total=744e9, active=40e9, L=78, attn="MLA + DeepSeek Sparse Attention", ffn="MoE 256e, 8 routed + 1 shared", pos=None, ctx=None, norm=None),
    "DeepSeek-V4-Pro (2026)": dict(total=1.6e12, active=49e9, L=None, attn="Compressed Sparse + Heavily Compressed attn", ffn="MoE", pos=None, ctx=1_000_000, norm=None),
    "Kimi K2.5 (2026)":      dict(total=1.0e12, active=None, L=None, attn="MLA (K2-style) + vision", ffn="MoE", pos=None, ctx=None, norm=None),  }
```
- **Exactly:** GLM-5 744B/40B, 78 layers; DeepSeek-V4-Pro 1.6T/49B, 1M context; Kimi K2.5 1T total. Fields we could not verify (kv dims, layers, active parameters, norm and positions) are `None`, which the printer shows as `n/a`.
- **In simple words:** we leave blanks instead of guessing.
- **Why it is here:** honesty rule: a wrong number in a study table is worse than a missing one.
### 10. KV-cache function  (`main.py` lines 39–51)
```python
def kv_bytes(m, tokens, dtype_bytes=2):
    """Total KV-cache bytes for ONE sequence of `tokens` tokens. None when the config does not let us compute it."""
    if m.get("L") is None:
        return None
    if "mla_dim" in m:                                                            # MLA: one shared latent per layer per token
        return m["L"] * m["mla_dim"] * dtype_bytes * tokens
    if "kv_heads" not in m:
        return None
    per_layer_token = 2 * m["kv_heads"] * m["head_dim"] * dtype_bytes             # K and V
    g = m.get("global_layers", m["L"])
    local_layers = m["L"] - g
    local_tokens = min(tokens, m.get("local_window", tokens))                     # local layers never keep more than `window` tokens
    return per_layer_token * (g * tokens + local_layers * local_tokens)
```
- **Exactly:** returns `None` if layers or kv dims are unknown. MLA: `L · mla_dim · bytes · tokens`. Others: per-layer-token `2 · kv_heads · head_dim · bytes`, times `global_layers · tokens + local_layers · min(tokens, window)`.
- **In simple words:** global layers store everything, local layers store only their window.
- **Why it is here:** one function reproduces the step-04 table and extends it to hybrid and local patterns.
### 11. Format helper  (`main.py` lines 54–55)
```python
def fmt(x, unit, nd=1):
    return "n/a" if x is None else f"{x / unit:.{nd}f}"
```
- **Exactly:** returns `"n/a"` for `None`, otherwise the value divided by a unit with `nd` decimals.
- **In simple words:** show a dash instead of crashing on missing data.
- **Why it is here:** keeps the table printing loop short.
### 12. Print the table  (`main.py` lines 59–65)
```python
print(f"{'model':<25}{'total B':>9}{'active B':>9}{'active%':>8}{'layers':>7}{'KV KB/tok @1k':>15}{'KV GB @128k':>13}  attention")
for name, m in MODELS.items():
    ratio = None if m["active"] is None else m["active"] / m["total"]
    k1k = kv_bytes(m, 1_000)
    k128 = kv_bytes(m, 131_072)
    print(f"{name:<25}{fmt(m['total'], 1e9):>9}{fmt(m['active'], 1e9):>9}{('n/a' if ratio is None else f'{ratio:.0%}'):>8}"
          f"{m['L'] if m['L'] else 'n/a':>7}{fmt(None if k1k is None else k1k / 1_000, 1024):>15}{fmt(k128, 1e9):>13}  {m['attn']}")
```
- **Exactly:** each row: total and active in billions, active%, layers, KV bytes/token (decimal KB) when the context holds 1,000 tokens, KV GB at 131,072 tokens, attention type. Selected results: Llama 3.1 8B 128 KB/token, 17.2 GB at 128k; Qwen3-235B 188 KB, 25.2 GB; gpt-oss-120b 40.6 KB/token at 1k, 4.8 GB at 128k; Gemma 3 27B 496 KB at 1k (all layers full while the context is shorter than the window) but only 11.2 GB at 128k; DeepSeek-V3 and Kimi K2 68.6 KB, 9.2 GB; Qwen3.5 30 KB, 4.0 GB (excluding DeltaNet state).
- **In simple words:** the same context length costs 0.5-4x less cache in models that use GQA with local layers, MLA or hybrids.
- **Why it is here:** the operational comparison the whole module builds toward.
### 13. Table footnotes  (`main.py` lines 66–67)
```python
print("\nKV columns assume bf16/fp16 cache, batch 1.  KB/tok @1k = bytes per token when the context is 1,000 tokens (local layers fully used).")
print("n/a = the number was not stated in the config or card we checked (we do not guess).")
```
- **Exactly:** printed assumptions: bf16 cache, batch 1; `n/a` = not stated, not guessed.
- **In simple words:** the fine print.
- **Why it is here:** numbers without assumptions are useless.
### 14. What changed since GPT-2  (`main.py` lines 70–79)
```python
print("\nComponent       GPT-2 (2019)        2025-26 open models")
for row in [("tokenizer", "BPE 50k", "byte-level BPE, 128k-262k vocab"),
            ("position", "learned absolute", "RoPE (+YaRN scaling), sometimes NoPE layers"),
            ("norm", "LayerNorm", "RMSNorm, pre-norm (sometimes also post / QK-norm)"),
            ("FFN", "GELU MLP 4d", "SwiGLU/GeGLU, or MoE with shared + routed experts"),
            ("attention", "MHA", "GQA, MLA, sliding/chunked, hybrid with linear attention"),
            ("kernel", "naive softmax", "FlashAttention / SDPA / paged KV cache"),
            ("decoding", "1 token/step", "speculative decoding, multi-token prediction"),
            ("training", "next-token", "+ RL for reasoning (test-time compute)")]:
    print(f"{row[0]:<16}{row[1]:<20}{row[2]}")
```
- **Exactly:** an eight-row comparison (tokenizer, position, norm, FFN, attention, kernel, decoding, training), GPT-2 vs the 2025-26 open models.
- **In simple words:** the module in a nutshell.
- **Why it is here:** a one-screen answer to "how does a modern LLM differ from GPT-2?".
### 15. MoE bargain  (`main.py` lines 82–85)
```python
print("\nMoE bargain (memory ~ total params, compute per token ~ active params):")
for n in ("Qwen3-235B-A22B", "gpt-oss-120b (2025)", "DeepSeek-V3 (2024-12)", "Kimi K2 (2025)", "Qwen3.5-397B-A17B"):
    m = MODELS[n]
    print(f"  {n:<24} weights in bf16 ~ {m['total'] * 2 / 1e9:>6.0f} GB | FLOPs/token ~ 2 x {m['active'] / 1e9:.0f}B = {2 * m['active'] / 1e9:.0f} GFLOP | a dense model with the same FLOPs has {m['active'] / m['total']:.0%} of the weights")
```
- **Exactly:** for five MoE models: bf16 weight memory `total x 2 bytes` (Qwen3-235B 470 GB; DeepSeek-V3 1,342 GB; Kimi K2 2,000 GB) against FLOPs per token `2 x active` (22B -> 44 GFLOP, 37B -> 74 GFLOP) and the share of weights a dense model with equal FLOPs would have (9%, 4%, 6%, 3%, 4%).
- **In simple words:** you pay memory for the whole model but compute only for the active part.
- **Why it is here:** the cost model for MoE deployment (and why they are served at FP8/FP4 across many GPUs).

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `dict(total=..., L=...)` | build a dict with keyword arguments | one model row |
| `m.get("global_layers", m["L"])` | value with default | all layers global if unspecified |
| `"mla_dim" in m` | key membership test | MLA branch |
| `f"{x:>9}"` | right-align in 9 columns | table |
| `1e9`, `0.124e9` | scientific notation | parameter counts |
| `x if cond else y` | conditional expression | `n/a` handling |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `min(a, b)` | built-in | smaller value | | number |
| `dict.items()` | built-in | key/value pairs | | view |
| `str.format` specs | built-in | alignment and decimals | | str |

(No third-party library: the point is that this step is *data plus arithmetic*.)

## Execution trace (real run)
```text
model                      total B active B active% layers  KV KB/tok @1k  KV GB @128k  attention
GPT-2 small (2019)             0.1      0.1    100%     12           36.0          4.8  MHA
Llama 3.1 8B (2024)            8.0      8.0    100%     32          128.0         17.2  GQA
Qwen2.5-0.5B (2024)            0.5      0.5    100%     24           12.0          1.6  GQA
Qwen3-8B (2025)                8.2      8.2    100%     36          144.0         19.3  GQA + QK-norm
Qwen3-235B-A22B              235.0     22.0      9%     94          188.0         25.2  GQA + QK-norm
gpt-oss-120b (2025)          116.8      5.1      4%     36           40.6          4.8  GQA, alt. sliding(128)/full
Gemma 3 27B (2025)            27.0     27.0    100%     62          496.0         11.2  GQA, 5 local(1024):1 global
Llama 4 Scout (2025)         109.0     17.0     16%     48          192.0          7.7  GQA, chunked(8192) + 1/4 NoPE global
DeepSeek-V3 (2024-12)        671.0     37.0      6%     61           68.6          9.2  MLA
Kimi K2 (2025)              1000.0     32.0      3%     61           68.6          9.2  MLA (DeepSeek-V3 design)
Qwen3.5-397B-A17B            397.0     17.0      4%     60           30.0          4.0  3 Gated DeltaNet : 1 gated attn
GLM-5 (2026)                 744.0     40.0      5%     78            n/a          n/a  MLA + DeepSeek Sparse Attention
DeepSeek-V4-Pro (2026)      1600.0     49.0      3%    n/a            n/a          n/a  Compressed Sparse + Heavily Compressed attn
Kimi K2.5 (2026)            1000.0      n/a     n/a    n/a            n/a          n/a  MLA (K2-style) + vision

KV columns assume bf16/fp16 cache, batch 1.  KB/tok @1k = bytes per token when the context is 1,000 tokens (local layers fully used).
n/a = the number was not stated in the config or card we checked (we do not guess).

Component       GPT-2 (2019)        2025-26 open models
tokenizer       BPE 50k             byte-level BPE, 128k-262k vocab
position        learned absolute    RoPE (+YaRN scaling), sometimes NoPE layers
norm            LayerNorm           RMSNorm, pre-norm (sometimes also post / QK-norm)
FFN             GELU MLP 4d         SwiGLU/GeGLU, or MoE with shared + routed experts
attention       MHA                 GQA, MLA, sliding/chunked, hybrid with linear attention
kernel          naive softmax       FlashAttention / SDPA / paged KV cache
decoding        1 token/step        speculative decoding, multi-token prediction
training        next-token          + RL for reasoning (test-time compute)

MoE bargain (memory ~ total params, compute per token ~ active params):
  Qwen3-235B-A22B          weights in bf16 ~    470 GB | FLOPs/token ~ 2 x 22B = 44 GFLOP | a dense model with the same FLOPs has 9% of the weights
  gpt-oss-120b (2025)      weights in bf16 ~    234 GB | FLOPs/token ~ 2 x 5B = 10 GFLOP | a dense model with the same FLOPs has 4% of the weights
  DeepSeek-V3 (2024-12)    weights in bf16 ~   1342 GB | FLOPs/token ~ 2 x 37B = 74 GFLOP | a dense model with the same FLOPs has 6% of the weights
  Kimi K2 (2025)           weights in bf16 ~   2000 GB | FLOPs/token ~ 2 x 32B = 64 GFLOP | a dense model with the same FLOPs has 3% of the weights
  Qwen3.5-397B-A17B        weights in bf16 ~    794 GB | FLOPs/token ~ 2 x 17B = 34 GFLOP | a dense model with the same FLOPs has 4% of the weights
```
Reading it: three families of cache cost: GQA models (17-25 GB at 128k), local/global mixes (5-11 GB), MLA and hybrids (4-9 GB). The right-hand `active%` column shows why MoE models are cheap per token and expensive in memory (3-9% active). GLM-5, DeepSeek-V4 and Kimi K2.5 rows have `n/a` where public numbers were not verified.

## Experiments (change ONE thing)
Every row below was run on a temporary copy of `main.py` (the real file is untouched); only the named line changed. The script has no randomness and runs in well under a second, so the numbers are exact. Baseline rows for reference: Llama 3.1 8B 128.0 KB/token and 17.2 GB at 128k; Gemma 3 27B 496.0 KB/token and 11.2 GB; DeepSeek-V3 68.6 KB/token and 9.2 GB.

| Change | What you will see | Why |
|---|---|---|
| `dtype_bytes=1` in `kv_bytes` (fp8-sized cache) (measured) | Llama 3.1 8B 64.0 KB/token and 8.6 GB; Gemma 3 27B 248.0 KB/token and 5.6 GB; DeepSeek-V3 34.3 KB/token and 4.6 GB | The cache is linear in bytes per value, so every row halves exactly. Quantising the KV cache is an easy 2x in long-context serving. |
| Gemma 3: remove `global_layers=10` (keep `local_window=1024`) (measured) | KV GB @128k jumps from 11.2 to 66.6; the `KB/tok @1k` column is unchanged (496.0) | Without `global_layers`, `kv_bytes` treats all 62 layers as global. At 1k tokens local and global layers cost the same, which is why the short-context column hides the mistake. Only the 128k column shows it. |
| Gemma 3: `local_window=4096` instead of 1024 (measured) | KV GB @128k rises from 11.2 to 12.5 | The 52 local layers keep 4x more tokens, but the 10 global layers still dominate the total. The local window is a minor knob; the number of global layers is the major one. |
| give DeepSeek-V3 plain attention (`kv_heads=128, head_dim=128`) instead of `mla_dim` (measured) | 3904.0 KB/token and 524.0 GB @128k (was 68.6 KB and 9.2 GB), 57x more | That is exactly the saving that MLA's shared latent buys; the 128-head MHA variant is deliberately a rough what-if, not a real model. |
| add the bare name `"gpt-oss-120b"` to the MoE-bargain loop (measured) | `KeyError: 'gpt-oss-120b'` | The dict key is `"gpt-oss-120b (2025)"` with the year. Names are the lookup key. |

## Debugging practice
**BREAK IT drill 1: describe a hybrid model but forget `global_layers`.** For the Gemma 3 row:

```diff
-                                  global_layers=10, local_window=1024),
+                                  local_window=1024),
```

Symptom (real run), no error, and the short-context column is untouched:

```text
Gemma 3 27B (2025)            27.0     27.0    100%     62          496.0         66.6  GQA, 5 local(1024):1 global
```

KV GB @128k is `66.6` instead of `11.2`.

Root cause: `m.get("global_layers", m["L"])` defaults to "every layer is global", so all 62 layers are charged for 131,072 tokens. Fix: put `global_layers` (and `local_window`) into the row together, whenever the attention is not uniform. A sanity check: the 128k figure for a local/global model must be much smaller than for an all-global one of the same shape.

**BREAK IT drill 2: a model with an unknown active size reaches a formula.** In the MoE-bargain loop, include `"Kimi K2.5 (2026)"` (its `active` is `None` because we did not find a verified number):

```text
TypeError: unsupported operand type(s) for /: 'NoneType' and 'float'
```

Root cause: the table code guards against `None` (`fmt`, `ratio is None`), but the last loop does arithmetic on `m['active']` directly. The "n/a, never guess" rule has to hold in every code path. Fix: skip rows where `m['active'] is None`, or print `n/a` there.

## Common mistakes
- Quoting "total parameters" as if they set speed. For MoE models memory follows total parameters, per-token compute follows active parameters.
- Comparing KV caches without stating the context length, batch size and cache dtype (the table assumes bf16, batch 1).
- Treating a table of public numbers as permanent. Every row is a snapshot from a config or card; newer models change quickly and some values are `None` on purpose.
- Forgetting that hybrid layers change the cache: local-window layers cap at the window, linear-attention layers keep a fixed-size state (0 per-token cache here).
- Inventing a value when a config does not state it. Leave it `None`, print `n/a`.
- Reading "128 experts, top-8" as 8 experts total: top-k is how many are chosen per token.

## Rebuild checklist
- [ ] list the five switches (attention, FFN, position, norm, tokenizer) and name a model for each choice
- [ ] compute KV bytes/token from a config, including a local:global pattern and MLA
- [ ] explain total vs active parameters and what each costs
- [ ] explain reasoning models as post-training plus test-time compute
- [ ] explain why Mamba/DeltaNet layers have no growing cache and why hybrids keep some attention

## The new problem (cliffhanger)
You can now read any decoder. What about changing it? Training and aligning a model (SFT, preference optimisation, RL, evaluation benchmarks) is the next step in the story: [module 17](../../17_finetuning_alignment_and_benchmarks/) (written by a separate author; check its README for status). Also revisit [module 06](../../06_llm_training_and_lora/README.md) for serving these models (vLLM, paged attention) and [module 14](../../14_rag_and_retrieval/README.md) for the alternative to a huge context window.

> **Say it aloud (interview box):** "A 2026 open model is a pre-norm RMSNorm decoder with RoPE plus YaRN scaling and SwiGLU MLPs, usually sparse (MoE with shared experts) and with a cache-saving attention: GQA, MLA, local/global layers or a linear-attention hybrid. Total parameters set memory, active parameters set compute. Reasoning models are the same architecture trained with RL to think long, which moves cost to inference."
