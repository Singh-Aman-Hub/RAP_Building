# Terms — Attention variants and the KV cache

Folder: `16_modern_llm_architecture/04_attention_variants_and_kv_cache` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [16 README](../README.md)

**Terms in this folder:** [Multi-head attention (MHA)](#multi-head-attention-mha) · [Multi-query attention (MQA)](#multi-query-attention-mqa) · [Grouped-query attention (GQA)](#grouped-query-attention-gqa) · [num_key_value_heads](#num_key_value_heads) · [repeat_interleave](#repeat_interleave) · [KV cache](#kv-cache) · [KV cache size formula](#kv-cache-size-formula) · [Multi-head latent attention (MLA)](#multi-head-latent-attention-mla) · [Decoupled RoPE](#decoupled-rope) · [Sliding-window attention](#sliding-window-attention) · [Local:global layer pattern](#localglobal-layer-pattern) · [Attention sink](#attention-sink) · [Prefill and decode](#prefill-and-decode) · [output_attentions and eager attention](#output_attentions-and-eager-attention)

---

## Multi-head attention (MHA)  `(concept)`
**1. Definition:** h query, key and value heads, each of size d_model / h.
**2. In plain words:** Every head has its own K and V.
**3. Example to quote:** > "MHA caches 2 x h x head_dim numbers per token per layer."
**Seen in:** `main.py`, `Attention(D, H, 8)`

## Multi-query attention (MQA)  `(concept)`
**1. Definition:** All query heads share one key head and one value head.
**2. In plain words:** One shared notebook for all readers.
**3. Example to quote:** > "MQA cuts the cache by h but can hurt quality."
**Seen in:** `main.py`, `Attention(D, H, 1)`

## Grouped-query attention (GQA)  `(concept)`
**1. Definition:** Query heads are split into g groups; each group shares one K/V head. g = h is MHA and g = 1 is MQA.
**2. In plain words:** A few shared notebooks, one per group.
**3. Example to quote:** > "Llama-3-8B has 32 query heads and 8 KV heads."
**Seen in:** `main.py`, `Attention`

## num_key_value_heads  `(syntax)`
**1. Definition:** The config field holding the number of K/V heads (g).
**2. In plain words:** If absent, the model is MHA.
**3. Example to quote:** > "Qwen2.5-0.5B: 14 heads, num_key_value_heads 2."
**Seen in:** `WALKTHROUGH.md`

## repeat_interleave  `(library)`
**1. Definition:** torch function that repeats each slice along a dimension, used to copy each KV head to all query heads in its group.
**2. In plain words:** Photocopy each K/V head for its group.
**3. Example to quote:** > "k.repeat_interleave(4, dim=1) turns 2 heads into 8."
**Seen in:** `main.py`, `Attention.forward`

## KV cache  `(concept)`
**1. Definition:** Stored keys and values of all earlier tokens in every layer, reused at each decoding step.
**2. In plain words:** Do not re-read the whole book for every new word.
**3. Example to quote:** > "bytes per token = 2 x layers x kv_heads x head_dim x bytes."
**Seen in:** `main.py`, section 3

## KV cache size formula  `(concept)`
**1. Definition:** 2 (K and V) x n_layers x n_kv_heads x head_dim x bytes per value, per token.
**2. In plain words:** Llama-3-8B in bf16: 128 KB per token, 17.2 GB at 128k.
**3. Example to quote:** > "Memory for the cache grows linearly with context and batch size."
**Seen in:** `main.py`, `kv_bytes_per_token`

## Multi-head latent attention (MLA)  `(concept)`
**1. Definition:** DeepSeek's attention: cache one low-rank latent per token (kv_lora_rank 512 plus a 64-dim RoPE key) and rebuild K and V from it.
**2. In plain words:** Store a summary and expand on demand.
**3. Example to quote:** > "DeepSeek-V3 caches 576 values per layer per token, 71x less than plain MHA would."
**Seen in:** `main.py`, `MLASimple`

## Decoupled RoPE  `(concept)`
**1. Definition:** In MLA a small separate key part carries the rotary position because RoPE does not commute with the low-rank merge.
**2. In plain words:** A tiny extra key that holds position.
**3. Example to quote:** > "That is the +64 in 512 + 64."
**Seen in:** `WALKTHROUGH.md`

## Sliding-window attention  `(concept)`
**1. Definition:** Each token attends to only the last W tokens, capping compute and cache per layer.
**2. In plain words:** A short memory.
**3. Example to quote:** > "gpt-oss uses window 128 in half its layers."
**Seen in:** `main.py`, `sliding_window_mask`

## Local:global layer pattern  `(concept)`
**1. Definition:** Alternating windowed layers with a few full-attention layers; Gemma 3 uses 5 local : 1 global.
**2. In plain words:** Most layers read nearby, a few read everything.
**3. Example to quote:** > "17.3% of an all-global cache in our 48-layer example."
**Seen in:** `main.py`, section 4

## Attention sink  `(concept)`
**1. Definition:** The first tokens receive a large share of attention regardless of content; evicting them breaks the model.
**2. In plain words:** A place to dump attention that has nowhere better to go.
**3. Example to quote:** > "GPT-2 middle layers put 70-79% of attention on token 0."
**Seen in:** `main.py`, section 5

## Prefill and decode  `(concept)`
**1. Definition:** Prefill processes the whole prompt in parallel (compute-bound); decode generates one token at a time reading the cache (memory-bound).
**2. In plain words:** Read the question, then write the answer word by word.
**3. Example to quote:** > "The cache speeds up decode, not prefill."
**Seen in:** `WALKTHROUGH.md`

## output_attentions and eager attention  `(library)`
**1. Definition:** attn_implementation='eager' with output_attentions=True returns the attention weights; fused kernels do not.
**2. In plain words:** Use the slow path only to look inside.
**3. Example to quote:** > "gpt2 returned 12 tensors of shape (1, 12, seq, seq)."
**Seen in:** `main.py`, GPT-2 measurement

