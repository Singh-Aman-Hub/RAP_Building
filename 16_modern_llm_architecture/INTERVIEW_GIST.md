# Interview Gist — Modern LLM Architecture (Module 16)
### Tokens → Positions → Norm/FFN → Attention + KV cache → Flash → MoE → Decoder LLD → Real configs → Landscape  (speculative decoding: module 21 step 13)

One-page revision sheet. Numbers come from the runs in this module. §10 is a rapid-fire Q&A.

---

## 0. The skeleton (say this first)
```
ids ─embed─> x ─ N x [ x = x + Attn(RMSNorm(x)) ; x = x + FFN(RMSNorm(x)) ] ─> RMSNorm ─> lm_head ─> logits
Attn: q,k get RoPE; GQA/MLA/local; KV cache; FlashAttention kernel     FFN: SwiGLU (dense) or MoE (router + experts + shared)
```
One-sentence summary:
> "A 2026 LLM is a pre-norm decoder with RMSNorm, RoPE (plus YaRN scaling), SwiGLU or MoE feed-forward and a cache-friendly attention (GQA, MLA, local/global or a linear hybrid), using fused attention kernels and byte-level BPE with a 128k-260k vocabulary."

## 1. Tokenizer (step 01)
- Byte-level BPE: start from 256 bytes, merge the most frequent pair, repeat; no unknown token. Encode = apply merges in learned order.
- Our 300 merges on 20 KB: 2.23 bytes/token; 1,000 merges: 3.61. Real vocabularies reach about 4.
- Vocabulary size: bigger = fewer tokens and better multilingual coverage, but bigger embedding table (Qwen2.5-0.5B: 136M of 494M = 27.6%) and under-trained rare tokens. gpt2 50,257 · Llama 3 128,256 · Qwen2.5 151,665 · Gemma 3 262k.
- Qwen splits digits one by one; gpt2 groups them. Hindi: 25 tokens (gpt2) vs 15 (Qwen).
- Special tokens (`<|im_start|>`) are single ids; the chat template must match training. SentencePiece/Unigram, WordPiece, tiktoken: other algorithms/libraries.

## 2. Positions (step 02)
- Attention is order-blind. GPT-2: learned absolute table (1,024 rows: position 1,024 raises IndexError). Sinusoidal: fixed, no parameters.
- **RoPE**: rotate each (even, odd) pair of q and k by `pos x theta_i`, `theta_i = base^(-2i/d)`. Score depends on `m - n` only (verified: -0.960123 at offsets 0, 100, 1000). Applied to q and k, never v; no parameters; norm preserved.
- Bases: Llama 3 500,000, Qwen 1,000,000, gpt-oss 150,000.
- Context extension: **PI** divides positions; **NTK-aware** raises the base (`base x s^(d/(d-2))`, 41,829 for 4x on head_dim 64); **YaRN** per-frequency ramp + logit temperature `0.1 ln s + 1` (1.1386 for 4x). Used by gpt-oss (factor 32), DeepSeek-V3 (40).
- ALiBi: subtract `slope x distance` from scores (BLOOM, MPT).

## 3. Norms and FFN (step 03)
- RMSNorm `x / sqrt(mean(x^2) + eps) * g`: d params vs LayerNorm 2d; no centring.
- Pre-norm `x + f(norm(x))` (identity highway, needs a final norm) vs post-norm `norm(x + f(x))`.
- SwiGLU `down(silu(gate x) * up x)`, hidden about 8d/3 to match `2 x d x 4d`: 4,718,592 vs 4,722,432 at d=768. Llama-3 widens to 3.5d; Qwen2.5-0.5B 5.4d.
- QK-norm: RMSNorm on q and k; max logit 194.8 to 2.9. No biases in modern MLPs.

## 4. Attention and KV cache (step 04)
```
KV bytes/token = 2 * layers * kv_heads * head_dim * bytes
```
| Model | KB/token (bf16) | GB at 128k |
|---|---|---|
| Llama-2-7B (MHA, 32 kv) | 512 | 68.7 |
| Llama-3-8B (GQA, 8 kv) | 128 | 17.2 |
| Qwen2.5-0.5B (GQA, 2 kv) | 12 | 1.6 |
| DeepSeek-V3 (MLA, 576/layer) | 68.6 | 9.2 |
- MQA = 1 KV head, GQA = g groups, MLA = cached latent (kv_lora_rank 512 + RoPE key 64), needs decoupled RoPE. MLA vs MHA for V3: 71x smaller.
- Sliding window / local:global (Gemma 3 5:1, window 1,024; gpt-oss alternating, window 128; Llama 4 chunked 8,192 + NoPE global layers).
- Attention sinks: GPT-2 puts 70-79% of attention on token 0 in middle layers; keep first tokens when evicting.
- Decode is memory-bound (reads the whole cache per token); prefill is compute-bound.

## 5. FlashAttention (step 05)
- Online softmax: `m_new = max(m, max(chunk))`, `l = l*exp(m - m_new) + sum(exp(chunk - m_new))`; the output accumulator is rescaled by the same factor.
- Tiled attention equals naive attention to 5.6e-16; memory O(N) vs O(N^2) (64 GB for one head's S and P at 131k, fp16, vs 128 KB tiles). Exact, not approximate; same FLOPs, far less HBM traffic.
- SDPA = PyTorch API that picks FlashAttention/mem-efficient/math; PagedAttention = KV-cache memory manager (different thing). FA2: better parallelism; FA3: Hopper async + FP8.

## 6. MoE (step 06)
- Router (Linear d to E), softmax, top-k, renormalise, experts are SwiGLU MLPs, plus always-on shared experts.
- Total vs active: DeepSeek-V3 671B/37B; Qwen3-235B-A22B; gpt-oss-120b 116.8B/5.1B; Kimi K2 1T/32B. Memory ~ total, FLOPs ~ active.
- Imbalance: untrained router gives loads 4 to 1,263 of 4,096 slots; capacity 1.25 would drop 30%.
- Aux loss `E sum f_e P_e` (1.0 = balanced); aux-loss-free (DeepSeek-V3): bias on the selection score only, `bias -= gamma sign(load - mean)`: max/min 1.29x with the router untouched.

## 7. The decoder LLD (step 07)
- Classes: `ModelConfig`, `RMSNorm`, `RotaryEmbedding`, `KVCache`, `Attention`, `MLP`, `DecoderBlock`, `Transformer`. 738,432 parameters (tied head).
- Pitfalls: rotate q,k before caching; read `start = cache.seq_len()` once per forward (layer 0 updates the cache first); decode needs no mask, prefill needs causal.
- Tests: cached == uncached greedy ids; logits diff ~2e-6; causality; initial loss = ln(vocab) (5.537 vs 5.545).

## 8. Reading configs (step 08) and speculative decoding (now module 21 step 13)
- Qwen2.5-0.5B: d=896, 24 layers, 14 q / 2 kv heads, hd 64, ffn 4,864, vocab 151,936, tied, rope theta 1e6: 494,032,768 parameters = embedding 27.6% + attention 8.9% + MLP 63.5% + norms. Formula `V d + L (attn + 3 d ff + 2 d) + d` matches exactly. Cache 12 KB/token.
- HF names: `q_proj k_proj v_proj o_proj`, `gate_proj up_proj down_proj`, `input_layernorm`, `post_attention_layernorm`, `embed_tokens`, `lm_head`. Our model equals HF Llama to 3.6e-7.
- Speculative decoding: draft proposes gamma tokens, target verifies in one pass, accept the matching prefix (+1 token). Greedy output identical. Sampling rule: accept with `min(1, p/q)`, else resample from `norm(max(0, p - q))`. Speed-up `(1 - a^(g+1)) / ((1 - a)(g c + 1))`. Variants: Medusa heads, EAGLE, MTP.

## 9. Landscape 2026 (step 10)
- Trends: sparsity (3-9% active), cache engineering (GQA to MLA to compressed/sparse attention), YaRN positions, NoPE layers, low-precision checkpoints (FP8, MXFP4, FP4), MTP, hybrids (Qwen3.5: 3 Gated DeltaNet : 1 attention).
- Reasoning models = same architecture + RL (GRPO for R1); cost moves to inference (test-time compute).
- Long context needs positions + kernels + cache design + long-context training data; configured length is not usable length.
- Mamba/SSM: fixed-size state, O(1) per token, weaker exact recall; hybrids keep a few attention layers.

## 10. Rapid-fire Q&A
1. **KV cache formula?** `2 x layers x kv_heads x head_dim x bytes` per token.
2. **GQA vs MQA vs MHA?** g KV heads: g = h is MHA, g = 1 is MQA.
3. **Why 8/3 d?** Three matrices vs two of 4d: `3 d h = 8 d^2`.
4. **RMSNorm vs LayerNorm?** No mean, no bias; half the parameters; same effect in practice.
5. **Pre-norm vs post-norm?** Pre keeps an identity residual path; trains deeper; needs a final norm.
6. **RoPE on which tensors?** q and k, not v; before caching.
7. **How to extend context?** PI, NTK-aware, YaRN, plus a short long-context fine-tune.
8. **Is FlashAttention approximate?** No, exact; it is IO-aware and linear in memory.
9. **MoE total vs active?** Memory vs compute.
10. **Why load balance?** Expert collapse and straggler experts.
11. **Why must `start` be computed once?** Layer 0 updates the cache before layer 1 runs.
12. **What is MLA?** Cache a low-rank latent, rebuild K and V; DeepSeek-V3 caches 576 values per layer per token.
13. **What does speculative decoding guarantee?** The target's output distribution (identical for greedy).
14. **What is an attention sink?** The first token absorbs attention; keep it when evicting.
15. **Why Qwen tokenizes digits singly?** Regular arithmetic patterns.
16. **Is a reasoning model a new architecture?** Usually no, same architecture plus RL.
17. **Why do small models tie embeddings?** The table is a large fraction of parameters.
18. **How to validate a port?** Load your weights into the reference and compare logits; check causality and cached equality.
