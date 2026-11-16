# Module 16 — Modern LLM Architecture (2026)

How a 2026 language model differs from GPT-2, built piece by piece: byte-level BPE, rotary positions, RMSNorm and SwiGLU, GQA/MLA and the KV cache, FlashAttention, Mixture of Experts, a complete Llama-style decoder (the low-level design), reading a real config, and a verified map of today's open models.

## The story of this module
Module [04](../04_llm_fundamentals/README.md) built a GPT-2 style block and module [06](../06_llm_training_and_lora/README.md) trained and served small models. Production models in 2026 look different.
- **Pain point:** you can run `AutoModelForCausalLM.from_pretrained("...")` and still not be able to say why Llama-3-8B needs a quarter of the KV cache of Llama-2-7B, what "671B-A37B" means, or why `intermediate_size` is 11,008. "Walk me through a modern decoder" and "design the attention for a 128k context" are standard senior ML-engineer questions.
- **Spark:** nearly every open model is one skeleton (pre-norm blocks, RoPE, SwiGLU, a tied or untied head) with a few switches: GQA/MLA/local attention, dense or MoE FFN, position scaling. Build each switch in 30-100 lines, check it against a number, and then assemble them.
- **What you build:** a byte-pair tokenizer trainer, RoPE with a numerical relative-position test and context-extension maths, RMSNorm/SwiGLU/QK-norm, MHA/GQA/MQA/MLA modules and a KV-cache capacity table, a tiled online-softmax attention proven equal to naive attention, an MoE layer with two balancing methods, a 738k-parameter Llama (7 classes, trained and cache-verified), a parameter counter for Qwen2.5-0.5B plus an equality proof against Hugging Face's Llama, and a computed comparison table of 2024-2026 models.
- **Cliffhanger:** you can now read and build the architecture. Changing a model's behaviour (SFT, preference optimisation, RL for reasoning, benchmarks) is module 17; serving it fast (including speculative decoding, which moved out of this module) is module [21](../21_llm_inference_engineering/).

**Reading position:** after [03_ml_foundations](../03_ml_foundations/README.md), [15_neural_networks](../15_neural_networks/README.md), [04_llm_fundamentals](../04_llm_fundamentals/README.md) and [05_llm_parameters](../05_llm_parameters/README.md), and **before** [21_llm_inference_engineering](../21_llm_inference_engineering/) (recommended order: ...04 -> 05 -> **16** -> 21 -> 06 ...).

**Short on time?** Read [INTERVIEW_GIST.md](INTERVIEW_GIST.md) first. Practice: [exercises.md](exercises.md) and [solutions/solutions.md](solutions/solutions.md).

## 1. Why this matters
Architecture questions test whether you understand *cost*: parameters, FLOPs per token, KV-cache bytes per token, and where memory traffic goes. Every design in this module exists to reduce one of those, and every number you will quote (the cache of Llama-3-8B at 128k tokens is 17.2 GB, DeepSeek-V3 caches 576 values per layer per token, a SwiGLU hidden size is 8/3 of d) comes from a run in this module.

## 2. Where this fits
**Builds on**
- [04_llm_fundamentals](../04_llm_fundamentals/): [tokenizer](../04_llm_fundamentals/01_tokenizer/), [embeddings](../04_llm_fundamentals/03_embeddings/), [attention](../04_llm_fundamentals/04_attention_from_scratch/), [transformer block](../04_llm_fundamentals/05_transformer_block/), [generation loop](../04_llm_fundamentals/08_generation_loop/): the GPT-2 starting point we modernise.
- [06_llm_training_and_lora](../06_llm_training_and_lora/): [training loop](../06_llm_training_and_lora/02_training_loop/) (step 07 reuses it), [KV cache](../21_llm_inference_engineering/08_hf_inference_batching_kvcache/), [paged attention](../21_llm_inference_engineering/11_paged_attention_fragmentation/), [vLLM](../21_llm_inference_engineering/15_serving_engines_landscape/) (the engine that serves what we build), [GPU memory calculator](../21_llm_inference_engineering/03_gpu_memory_calculator/) (canonical memory maths), [LoRA](../06_llm_training_and_lora/04_lora_math_numpy/).
- [05_llm_parameters](../05_llm_parameters/): [context window](../05_llm_parameters/04_max_new_tokens_and_context_window/) (step 02's position scaling is what moves that limit).

**Leads to**
- Module 17 (fine-tuning, alignment, benchmarks; written separately): changing behaviour of the architecture built here.
- [14_rag_and_retrieval](../14_rag_and_retrieval/): the alternative to a bigger context window.
- [10_ml_system_architecture](../10_ml_system_architecture/): capacity planning uses the KV and weight-memory formulas of steps 04, 06 and 10.

## 3. Concept map
```
text ──01 byte-level BPE──> token ids (b, t)
        │
        v   embedding table (vocab x d_model)                        [01: vocab cost, 07: tied head]
   (b, t, d_model) ── N x DECODER BLOCK ──────────────────────────────────────────────────┐
        ┌───────────────────────────────────────────────────────────────────────────┐    │
        │ x ──RMSNorm (03)──> ATTENTION ──+──> x ──RMSNorm──> FFN ──+──> x            │    │
        │                      │          │                  │      │                  │    │
        │    q,k: RoPE (02)    │       residual           SwiGLU (03) or MoE (06)     │    │
        │    GQA / MQA / MLA (04)    sliding/local layers (04)    router, top-k, shared│    │
        │    KV cache stores k,v (04, 07)                                              │    │
        │    kernel: FlashAttention / SDPA, online softmax (05)                        │    │
        └───────────────────────────────────────────────────────────────────────────┘    │
        v                                                                                   │
   final RMSNorm ── lm_head (tied, 07) ──> logits (b, t, vocab) ──> next token ─────────────┘
                                                      │
              decoding speed-ups: KV cache (07); speculative decoding / MTP: module 21 step 13
real models = this skeleton + switches (08 reads a config, 10 compares 2024-2026 models,
 reasoning = same architecture + RL, long context = 02 + 04 + 05, hybrids = attention + SSM/DeltaNet)
```

## 4. Study path
| Step | Folder | You learn | Run | Time |
|---|---|---|---|---|
| 1 | [01_bpe_tokenizer_from_scratch](01_bpe_tokenizer_from_scratch/) | byte-level BPE training, encode/decode, vocab trade-offs, special and chat tokens vs gpt2/Qwen | `python main.py` | 45 min |
| 2 | [02_positional_encodings](02_positional_encodings/) | sinusoidal, learned, RoPE (relative property verified), ALiBi, PI / NTK / YaRN | `python main.py` | 60 min |
| 3 | [03_norms_and_ffn](03_norms_and_ffn/) | LayerNorm vs RMSNorm, pre- vs post-norm, GELU vs SwiGLU (8/3 rule), QK-norm | `python main.py` | 50 min |
| 4 | [04_attention_variants_and_kv_cache](04_attention_variants_and_kv_cache/) | MHA/MQA/GQA/MLA, KV bytes per token, sliding windows, attention sinks | `python main.py` | 70 min |
| 5 | [05_flash_attention_online_softmax](05_flash_attention_online_softmax/) | online softmax, tiled exact attention, IO-awareness, SDPA | `python main.py` | 60 min |
| 6 | [06_mixture_of_experts](06_mixture_of_experts/) | router, top-k, shared experts, aux loss vs bias balancing, total vs active parameters | `python main.py` | 60 min |
| 7 | [07_build_mini_llama](07_build_mini_llama/) | the LLD: 7 classes, training, KV-cache correctness | `python main.py` (1-2 min) | 90 min |
| 8 | [08_read_a_real_config](08_read_a_real_config/) | config fields and tensor names of Qwen2.5-0.5B, parameter counting, equality with HF Llama | `python main.py` | 45 min |
| 10 | [10_architecture_landscape_2026](10_architecture_landscape_2026/) | computed comparison table of 2024-2026 models, reasoning/test-time compute, long context, SSM and hybrids | `python main.py` | 50 min |

> **Speculative decoding moved to module 21 step 13** (it is a serving optimisation): [../21_llm_inference_engineering/13_speculative_decoding/](../21_llm_inference_engineering/13_speculative_decoding/). Step numbering here therefore skips 09. FlashAttention (step 5) stays in this module because it is the *algorithm* (online softmax, tiling); module 21 covers how serving engines *use* it.

Total about 9 hours (steps sum to 530 minutes) plus [exercises.md](exercises.md). Each step folder has `main.py`, `WALKTHROUGH.md` (story, theory, a block-by-block "Code, step by step", real output, experiments, a debugging drill) and `TERMS.md`.

Run from the repo root with `HF_HUB_OFFLINE=1 .venv/bin/python 16_modern_llm_architecture/<step>/main.py`. Everything is offline and CPU-only with the Qwen2.5-0.5B tokenizer/config (and `gpt2` for tokenizer comparison). Note: the cache holds only the *config* of Qwen2.5-0.5B, not its weights, so step 08 builds the architecture from the config (meta device) instead of loading weights. Step 07 bundles `corpus.txt` (128 KB of this repo's markdown) and step 01 `corpus_sample.txt`.

## 5. The concepts, one by one

### 5.1 Tokens (step 01)
Byte-level BPE: start from 256 bytes, merge the most frequent adjacent pair repeatedly. No unknown token, 1.4 to 3.6 bytes per token with 50 to 1,000 merges on our small text, about 4 for production vocabularies. Vocabulary size trades shorter sequences against a bigger embedding table (27.6% of Qwen2.5-0.5B). Special tokens and chat templates are part of the model's interface.

### 5.2 Positions (step 02)
Attention is order-blind, so position must be injected. RoPE rotates q and k pairs by `pos x theta_i`, making `q·k` depend on the distance only (verified: -0.960123 at three different absolute offsets). Long context needs scaled frequencies: PI divides positions, NTK-aware raises the base, YaRN blends per frequency.

### 5.3 Norms and FFN (step 03)
Pre-norm RMSNorm (rescale only, d parameters) with a final norm; SwiGLU `down(silu(gate x) * up x)` with hidden about 8/3 d to keep the parameters of a 4d GELU MLP; QK-norm to bound attention logits (194.8 to 2.9 in the test).

### 5.4 Attention and the cache (step 04)
KV bytes per token = `2 x layers x kv_heads x head_dim x bytes`. GQA divides it by h/g, MLA caches a latent (DeepSeek-V3: 576 values per layer per token), sliding-window layers cap it, sinks explain what must not be evicted.

### 5.5 FlashAttention (step 05)
Same maths, different loop order: tiles plus online softmax (`m`, `l`, rescaled accumulator) give identical results (5.6e-16) with O(N) memory and far less memory traffic.

### 5.6 Mixture of experts (step 06)
Router + top-k + experts (+ shared). Total parameters set memory, active parameters set compute. Balance with an auxiliary loss or with DeepSeek-V3's selection-only bias.

### 5.7 The decoder as code (step 07)
`ModelConfig`, `RMSNorm`, `RotaryEmbedding`, `KVCache`, `Attention`, `MLP`, `DecoderBlock`, `Transformer`: 738,432 parameters; cached and uncached generation identical; logits agree to ~2e-6.

### 5.8 Reading real models (steps 08 and 10)
Config fields map to the switches; tensor names map to our classes; parameters and cache follow from formulas. Landscape: GQA (Llama, Qwen3), MLA + MoE (DeepSeek-V3, Kimi K2, GLM-5), local/global (Gemma 3, gpt-oss, Llama 4), linear-attention hybrids (Qwen3.5), compressed sparse attention (DeepSeek-V4).

### 5.9 Faster decoding (moved to module 21 step 13)
A cheap draft model proposes tokens, the target verifies a block in one pass: identical output, fewer expensive passes. Gains need a cheap draft and high acceptance. MTP and Medusa/EAGLE heads replace the separate draft. Hands-on: [21 step 13](../21_llm_inference_engineering/13_speculative_decoding/).

## 6. Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| FFN | MLP, feed-forward, `intermediate`, "dense layer" (loosely) | Yes | per-token network in each block |
| KV cache | `past_key_values`, `use_cache`, `DynamicCache` | Yes | stored K and V of earlier tokens |
| MHA / MQA / GQA / MLA | multi-head / multi-query / grouped-query / latent attention | No | number of K/V heads: h, 1, g, or a compressed latent |
| RoPE | rotary embedding, rotary position embedding, `rotary_emb` | Yes | rotate q and k by position |
| Context window | context length, `max_position_embeddings`, `n_ctx`, sequence length | Mostly | max tokens configured; usable quality can be lower |
| `d_model` | `hidden_size`, `n_embd`, `dim` | Yes | residual-stream width |
| `n_layers` | `num_hidden_layers`, `n_layer`, depth | Yes | number of blocks |
| Experts | MoE, sparse model, "A22B" | Related | many MLPs, few active per token |
| Active vs total params | activated / full | No | compute-relevant vs memory-relevant size |
| FlashAttention vs SDPA vs PagedAttention | fused attention / PyTorch API / vLLM KV paging | No | kernel algorithm vs dispatcher API vs KV-cache memory manager |
| LayerNorm vs RMSNorm vs BatchNorm | | No | centre+scale vs scale only vs batch statistics |
| Pre-norm vs post-norm | pre-LN / post-LN | No | norm inside the residual branch vs after the addition |
| SwiGLU / GeGLU / GLU | gated MLP, `gate_proj/up_proj/down_proj` | Family | gate activation SiLU / GELU / sigmoid |
| BPE / WordPiece / Unigram / SentencePiece / tiktoken | subword tokenizers and libraries | No | algorithms (BPE, WordPiece, Unigram) vs libraries (SentencePiece, tiktoken) |
| Draft model / target model | assistant model, small/large model | Yes | proposer vs verifier in speculative decoding |
| Reasoning model | thinking model, test-time-compute model | Mostly | same architecture, RL-trained for long chains of thought |
| Prefill vs decode | prompt phase vs generation phase | No | parallel over the prompt (compute-bound) vs one token reading the cache (memory-bound) |

## 7. Interview Q&A
**Q: How does a modern LLM differ from GPT-2?**
A: Same decoder-only idea, different parts: byte-level BPE with 128k-260k tokens instead of 50k; RoPE (plus YaRN scaling) instead of learned positions; pre-norm RMSNorm instead of LayerNorm; SwiGLU or MoE instead of a GELU MLP; GQA/MLA/local attention instead of MHA; no biases; FlashAttention kernels; post-training with preference optimisation and RL.

**Q: Compute the KV cache for Llama-3-8B at 128k tokens.**
A: `2 x 32 layers x 8 kv heads x 128 head_dim x 2 bytes = 131,072 bytes = 128 KB/token`; times 131,072 tokens = 17.2 GB per sequence in bf16. The same model with MHA (32 KV heads) would need 68.7 GB.

**Q: Why is SwiGLU's hidden size 8/3 d?**
A: It has three matrices (gate, up, down). `3 x d x h` equals the `2 x d x 4d` of a GELU MLP when `h = 8d/3`, so the parameter count is unchanged (4,718,592 vs 4,722,432 at d=768).

**Q: Prove RoPE gives relative positions.**
A: Rotating q by angle `m theta` and k by `n theta` gives `q^T R(-m theta) R(n theta) k = q^T R((n-m) theta) k`; only `n - m` remains. We checked it numerically: the score at (5, 2), (105, 102) and (1005, 1002) is -0.960123 each.

**Q: Why does FlashAttention not change the result and why is it faster?**
A: It reorders the computation with an online softmax (running max and sum, rescaling the accumulator), so the result is exact (we saw 5.6e-16 difference). It is faster because attention is memory-bound: the N x N matrix is never written to HBM; tiles live in on-chip SRAM.

**Q: What is MoE and what are total vs active parameters?**
A: Replace the MLP by E expert MLPs plus a router choosing top-k per token (plus shared experts). Memory is set by the total (671B for DeepSeek-V3), compute per token by the active (37B).

**Q: How do you keep experts balanced?**
A: An auxiliary loss `E sum f_e P_e` (Switch), or DeepSeek-V3's bias: add a per-expert bias to the scores used only for top-k selection and nudge it down for overloaded experts, up for underloaded ones, with no extra loss.

**Q: What does the KV cache do to latency and memory?**
A: It removes the quadratic recomputation (decode becomes one token per step), but memory grows linearly with context x batch and each step reads the whole cache, so decode is memory-bound. GQA/MLA/local layers shrink it.

**Q: How would you check that your own transformer implementation is correct?**
A: Copy its weights into the reference implementation and compare logits (3.6e-7 in step 08); check causality (changing future tokens must not change past logits); check cached vs uncached greedy equality; check the initial loss is `ln(vocab)`.

**Q: When does speculative decoding help?** (hands-on in [module 21 step 13](../21_llm_inference_engineering/13_speculative_decoding/))
A: When the draft is much cheaper than the target and agrees often. Speed-up `(1 - a^(g+1)) / ((1 - a)(g c + 1))` for acceptance `a`, draft length `g`, cost ratio `c`. Output equals the target's (greedy) or has the target's distribution (sampling with the rejection rule).

**Q: What is a reasoning model, architecturally?**
A: Usually the same architecture as its base model (DeepSeek-R1 is V3), post-trained with RL to produce long reasoning before answering. Cost moves to inference: more output tokens, so decode speed and KV cache matter more.

**Q: Transformer vs Mamba?**
A: Attention keeps a growing cache and reads it all per token; Mamba keeps a fixed-size state (O(1) per token) but recalls specific past tokens less exactly. Production models in 2026 mix them (hybrids): Qwen3.5 uses 3 Gated DeltaNet layers per full-attention layer.

## 8. Self-check
- [ ] I can write the KV-cache formula and evaluate it for any config.
- [ ] I can derive the 8/3 rule and say why Llama-3 deviates.
- [ ] I can explain why RoPE is applied to q and k before caching.
- [ ] I can derive the online-softmax rescale factor from memory.
- [ ] I can state total vs active parameters for DeepSeek-V3 and what each controls.
- [ ] I can write a decoder block in torch from memory with GQA and a cache.
- [ ] I can list five differences between GPT-2 and a 2026 open model.

## References (checked Nov 2026)
Papers and primary sources
- Attention Is All You Need (sinusoidal positions): https://arxiv.org/abs/1706.03762
- RoFormer / RoPE: https://arxiv.org/abs/2104.09864
- ALiBi, "Train Short, Test Long": https://arxiv.org/abs/2108.12409
- Position Interpolation: https://arxiv.org/abs/2306.15595
- YaRN: https://arxiv.org/abs/2309.00071
- RMSNorm: https://arxiv.org/abs/1910.07467
- GLU variants (SwiGLU): https://arxiv.org/abs/2002.05202
- Multi-query attention: https://arxiv.org/abs/1911.02150
- GQA: https://arxiv.org/abs/2305.13245
- DeepSeek-V2 (MLA): https://arxiv.org/abs/2405.04434
- DeepSeek-V3 technical report: https://arxiv.org/abs/2412.19437 and model card https://huggingface.co/deepseek-ai/DeepSeek-V3
- StreamingLLM (attention sinks): https://arxiv.org/abs/2309.17453
- FlashAttention: https://arxiv.org/abs/2205.14135 · FlashAttention-2: https://arxiv.org/abs/2307.08691 · FlashAttention-3: https://arxiv.org/abs/2407.08608 · repository https://github.com/Dao-AILab/flash-attention
- Online softmax ("Online normalizer calculation for softmax"): https://arxiv.org/abs/1805.02867
- Switch Transformer: https://arxiv.org/abs/2101.03961 · Mixtral of Experts: https://arxiv.org/abs/2401.04088
- Fast inference via speculative decoding: https://arxiv.org/abs/2211.17192 · Medusa: https://arxiv.org/abs/2401.10774 · EAGLE: https://arxiv.org/abs/2401.15077
- Mamba: https://arxiv.org/abs/2312.00752
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- gpt-oss model card: https://arxiv.org/abs/2508.10925

Model cards and configs read for step 10 (values re-read Nov 2026)
- Qwen3 config: https://huggingface.co/Qwen/Qwen3-235B-A22B/raw/main/config.json and https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json
- Qwen3.5-397B-A17B: https://huggingface.co/Qwen/Qwen3.5-397B-A17B
- gpt-oss-120b config: https://huggingface.co/openai/gpt-oss-120b/raw/main/config.json
- Kimi K2: https://huggingface.co/moonshotai/Kimi-K2-Instruct
- Llama 3.1 8B config (mirror): https://huggingface.co/unsloth/Meta-Llama-3.1-8B/raw/main/config.json
- Llama 4 Scout (mirror): https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct
- Gemma 3 27B config (mirror): https://huggingface.co/unsloth/gemma-3-27b-it/raw/main/config.json
- DeepSeek-V4-Pro model card: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
- Raschka, "A Dream of Spring for Open-Weight LLMs" (GLM-5, Kimi K2.5, Qwen3.5 and others): https://magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight
- Raschka, LLM Architecture Gallery: https://sebastianraschka.com/llm-architecture-gallery/
- PyTorch SDPA docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- Hugging Face transformers docs (attention interfaces, `attn_implementation`): https://huggingface.co/docs/transformers/

Not independently verified here: details of GLM-5 and Kimi K2.5 beyond Raschka's summary, DeepSeek-V4's layer and expert counts (not stated on its card), and the current FlashAttention version and hardware support (check the repository).
