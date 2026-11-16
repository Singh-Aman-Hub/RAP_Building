# Module 16 — Solutions

Try each exercise first ([exercises.md](../exercises.md)). Runnable scripts (all executed on CPU; each prints its own numbers) are in this folder; run from the repo root with `.venv/bin/python 16_modern_llm_architecture/solutions/<script>`. Where an answer is reasoning rather than a run, it says so. Speculative decoding (old step 09) moved to [module 21 step 13](../../21_llm_inference_engineering/13_speculative_decoding/).

| Script | Covers |
|---|---|
| [s01_bpe.py](s01_bpe.py) | step 01 |
| [s02_rope.py](s02_rope.py) | step 02 |
| [s03_norms_ffn.py](s03_norms_ffn.py) | step 03 |
| [s04_kv_and_gqa.py](s04_kv_and_gqa.py) | step 04 |
| [s05_online_softmax.py](s05_online_softmax.py) | step 05 |
| [s06_moe.py](s06_moe.py) | step 06 |
| [s07_mini_llama_variants.py](s07_mini_llama_variants.py) | step 07 |
| [s08_param_count.py](s08_param_count.py) | step 08 |
| [s10_landscape_kv.py](s10_landscape_kv.py) | step 10 |

## Step 01 — BPE (`s01_bpe.py`)
- **NUM_MERGES sweep (20 KB sample):** 50 merges 1.40 bytes/token, 300 merges 2.21, 1000 merges 3.56. The gains are steepest early (the first merges glue the most frequent pairs such as `e `, `th`); each later merge covers fewer occurrences, so the curve flattens. Real vocabularies reach about 4 on large corpora.
- **Pre-tokenizer:** merges can no longer cross word boundaries: the number of learned tokens containing an inner space dropped from 15 to 0, and compression fell slightly (2.11 vs 2.21 at 300 merges). GPT-2/Llama/Qwen all use a regex pre-tokenizer for this reason (cleaner tokens, no token spanning two words).
- **Break it:** `b"".join(i ...)` raises `TypeError: sequence item 0: expected a bytes-like object, int found`: `decode` must map ids to byte strings first (`vocab[i]`), then join, then `.decode("utf-8", errors="replace")`.
- **Explain aloud:** byte-level BPE starts from all 256 byte values, so any string, in any language or emoji, is representable: no `<unk>`. Hindi costs more tokens in gpt2 because its merges were learned mostly on English; Devanagari characters are 3 UTF-8 bytes each and few of them were ever merged (step 01 measured 25 tokens in gpt2 vs 15 in Qwen).

## Step 02 — positions (`s02_rope.py`)
- **Base 1,000,000:** longest wavelength 4,080,185 tokens vs 47,117 at base 10,000 (head dim 64). Pairs with wavelength above 4,096: 16 of 32 vs 9 of 32: a bigger base keeps half the frequencies slow enough to distinguish positions far beyond 4k.
- **Relative property:** the score at distance 0 and 50 matches between positions 10 and 10,000 to 1.8e-15 and 1.9e-12; error grows with absolute position because `cos/sin` of large angles lose float precision (in bf16 much more).
- **D = 7:** pairs cannot be formed; the strided assignment fails with a broadcast `ValueError` (3 vs 4 elements). Head dimension must be even.
- **Layouts:** interleaved pairs `(x0,x1),(x2,x3)...` and rotate-half pairs `(x_i, x_{i+d/2})` give identical scores once the dimensions are permuted consistently for q and k (checked at two position pairs). HF checkpoints are converted to rotate-half layout, so the permutation must be applied when loading original Meta weights.
- **Explain aloud:** RoPE has no parameters: it is a fixed rotation by `pos x theta_i`. NTK-aware and YaRN treat high and low frequencies differently because high-frequency pairs already rotate many times within the training length (interpolating them destroys local resolution) while low-frequency pairs never completed a turn (they must be stretched to cover the longer context).

## Step 03 — norms and FFN (`s03_norms_ffn.py`)
- **eps 0.1:** output rms 0.845 instead of 1.0 for inputs with mean-square 0.25 (eps is not negligible against the variance).
- **hidden = 4d for SwiGLU:** 7,077,888 vs 4,718,592 parameters: 1.5x the GELU MLP. The 8/3 factor exists to keep `3 d h = 2 d (4d)`.
- **keepdim:** without it `mean(-1)` has shape `(4,)`, dividing `(4, 768)` raises a size-mismatch error; when the feature size equals the batch size (e.g. 5x5) it broadcasts silently along the wrong axis.
- **Block parameters at d=768** (12 MHA heads, SwiGLU 2048, 2 norms): 7,079,424 = `4 d^2 + 3 d h + 2 d`.
- **Explain aloud:** post-norm `norm(x + f(x))` puts the norm on the residual path, so gradients pass through a norm at every layer and deep stacks need warm-up; pre-norm `x + f(norm(x))` keeps an identity highway; but the stream's magnitude then grows with depth, so a final norm is applied before the head.

## Step 04 — attention variants (`s04_kv_and_gqa.py`)
- **KV table:** Llama-2-7B 512 KB/token, 68.7 GB at 128k; Llama-3-8B 128 KB, 17.2 GB; Qwen2.5-0.5B 12 KB, 1.6 GB. For other models: `2 x layers x kv_heads x head_dim x bytes`; for the full capacity planner see [21/03](../../21_llm_inference_engineering/03_gpu_memory_calculator/WALKTHROUGH.md).
- **GQA-2 vs 4 vs 8** (d=512, 8 heads): 655,360 / 786,432 / 1,048,576 parameters; cache floats per token per layer 256 / 512 / 1,024.
- **`Attention(512, 8, 3)`:** the assertion fires (`each KV head must serve a whole number of query heads`); without it `repeat_interleave(8 // 3)` would give 6 heads, not 8.
- **MQA = MHA with copies:** max difference 0.0 when the 8 K/V heads are copies of the single head.
- **Explain aloud:** GQA keeps several KV heads so different query groups can attend to different subspaces; MQA forces one shared K/V, which costs quality and training stability. MLA keeps per-head diversity by caching a low-rank latent and expanding it with (absorbable) up-projections, so it shrinks the cache without sharing heads.

## Step 05 — FlashAttention (`s05_online_softmax.py`)
- **block_k 1 / 64 / 200:** identical results (errors 1.5e-14, 6e-15, 9e-15, all float rounding). Speed: block_k=1 is far slower in Python (many tiny matmuls); there is no benefit to huge blocks on a CPU.
- **Timing (numpy, CPU):** at N=200, 800 and 3200 the tiled and naive versions are within the same order (naive 0.8 / 23 / 435 ms, tiled 1.0 / 20 / 595 ms in our run; machine dependent). The tiled code is not reliably faster on a CPU because it is a Python loop and CPUs do not have the HBM/SRAM gap. What changes is the extra memory: naive materialises an `N x N` matrix (82 MB at N=3200 in float64) while tiled needs O(N).
- **Missing rescale of `acc`:** max error 8.54 and no crash (shapes are unchanged); the sum `l` was rescaled but the accumulator was not, so earlier tiles are weighted with a stale max.
- **Explain aloud:** attention does little arithmetic per byte moved: writing and re-reading `S` and `P` (N x N) in HBM dominates. IO-aware means designing the algorithm around the number of HBM reads/writes: tile so that blocks stay in SRAM, fuse softmax and matmuls, never write `S`.

## Step 06 — MoE (`s06_moe.py`)
- **top_k=1, n_shared=0:** active fraction drops from 33.5% to 12.7% of the layer; the untrained router is very uneven in both cases (loads from about 50 to 1,779 of 4,096 slots for top-2).
- **Aux loss lr:** at lr 0.05 three experts end with 0 tokens (loads `[820, 819, 821, 821, 815, 0, 0, 0]`, loss 1.60): the step overshoots, the dead experts get no routed tokens and therefore no gradient from the expert path. At lr 0.01 all eight carry about 512 and the loss is 1.000.
- **gamma 0.1:** yes, it oscillates: max/min load at step 50 was 43 and the std over the last 50 steps 15 (vs 1.31 mean, std 0.09 at gamma 0.01). Each step moves the bias by more than the score gaps, so experts flip between over- and under-loaded.
- **top_k=9 with 8 experts:** `RuntimeError: selected index k out of range`.
- **Dispatch and loss:** see `forward` and `balance_loss` in the script: `nonzero` finds tokens routed to expert `e`; `out[tok] += w * expert(x[tok])`; loss `n * sum(f_e * P_e)` where `f_e` (non-differentiable count fraction) times `P_e` (mean probability) gives the gradient path.
- **Explain aloud:** total parameters determine memory (all experts must be resident), active parameters determine per-token FLOPs. The bias is used for selection only so that balancing does not change the mixing weights (the output stays a faithful weighted sum of the router's probabilities, and no extra loss competes with the language-model loss).

## Step 07 — mini-Llama (`s07_mini_llama_variants.py`)
- **n_kv_heads 8 / 2 / 1:** parameters 836,736 / 738,432 / 722,048; cache bytes per token in fp32 4,096 / 1,024 / 512.
- **600 steps instead of 300:** not run here (about 1 to 2 minutes of training; edit `STEPS` in step 07). Expected from the loss curve: lower training and validation loss and more word-like output, with the usual small-data caveat (128 KB corpus can be memorised).
- **Temperature sampling:** with a fixed generator seed, cached and uncached sampled sequences are identical (True in our run, on an untrained model; the equality depends only on the logits agreeing to float precision).
- **Break it, RoPE after `cache.update`:** the cache then stores unrotated keys and `rope(q, k, start)` rotates the whole cached key tensor as if it started at `start`: old keys get wrong positions. Prefill is unaffected; decode logits differ (max difference 5.5e-3 vs 2.5e-7 for the correct code on our untrained model; a trained model amplifies this).
- **Break it, `start` inside Attention:** layer 0 appends the new token before layer 1 runs, so layers 1+ would read a longer cache length and rotate by the wrong offset; the cached == uncached equality test fails. Hence `Transformer.forward` reads `start` once.
- **Rebuild and explain aloud:** reference the code of `KVCache` and `Attention.forward` in step 07. One decode step with a 20-token cache: ids `(1,1)` -> embedding `(1,1,d)`; per layer q `(1,h,1,hd)`, new k,v `(1,g,1,hd)`, RoPE at position 20, append so cache k is `(1,g,21,hd)`, `repeat_interleave` to `(1,h,21,hd)`, SDPA (no mask: the token may see all 21) gives `(1,h,1,hd)`, merge heads `(1,1,d)`, wo, MLP; final norm and head give logits `(1,1,vocab)`.

## Step 08 — real config (`s08_param_count.py`)
- **Pocket-calculator count:** embedding `151,936 x 896 = 136,134,656`; per layer attention weights `896 x 896 + 2 x 896 x 128 + 896 x 896 = 1,835,008` plus q/k/v biases 1,152 gives 1,836,160; MLP `3 x 896 x 4,864 = 13,074,432`; 2 norms 1,792; so 14,912,384 per layer; `x 24` plus embedding plus final norm 896 = **494,032,768** (matches; script asserts it).
- **`tie_word_embeddings=False` in the HF test (reasoning, not run):** the HF model gets a separate randomly initialised `lm_head` unless you copy our embedding into it too; if you do not, the logits differ by a large, non-small amount; if you copy it, the difference returns to about 1e-7. The parameter count rises by 136,134,656 (script prints 630,167,424).
- **`rope_theta` changed in HF only (reasoning, not run):** every rotation angle differs, so the scores and therefore the logits differ everywhere except position 0; the difference grows with position because the angle error is `pos x delta_theta`.
- **`mlp.gate` mapped to `up_proj` (reasoning, not run):** `load_state_dict(strict=False)` only reports missing/unexpected keys; here key names still exist and shapes match, so it loads without error but the model is silently wrong (gate and up swapped): the equality test must be the guard, and `strict=False` should be inspected for its returned `missing_keys` and `unexpected_keys`.
- **Explain aloud:** see the field list in step 08: `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `vocab_size`, `rope_theta`/`rope_scaling`, `rms_norm_eps`, `tie_word_embeddings`, `max_position_embeddings`.

## Step 09
Moved to [module 21 step 13](../../21_llm_inference_engineering/13_speculative_decoding/).

## Step 10 — landscape (`s10_landscape_kv.py`)
- **KV at 1M tokens** (bf16, one sequence): Llama-2-7B 524 GB, Llama-3-8B 131 GB, Qwen2.5-0.5B 12.3 GB, DeepSeek-V3 (MLA, 61 x 576 values) 70 GB. Beyond one 80 GB GPU with weights: all except the 0.5B model. DeepSeek-V3's cache is small relative to others but its 671 GB of FP8 weights need many GPUs.
- **Adding your own model / the gpt-oss `global_layers` bug / switch table:** do these from the model's `config.json` (use `None` for anything unverified). For gpt-oss, forgetting that only some layers are global makes the 128k estimate too large by treating every layer as full attention; the window layers cache only 128 tokens.
- **Explain aloud:** reasoning models spend more output tokens (decode speed and KV capacity dominate cost); hybrid models replace most of the cache with a fixed state, so long-context cost shifts from memory-per-token to compute and recall quality.

## Capstone
Not run here (it is a build exercise). Outline: replace `MLP` in every other `DecoderBlock` with the `MoE` of step 06 (use a smaller hidden size), add `is_causal=False` with an explicit boolean mask `i - j < 64` combined with causality for the window (note the cache then only needs the last 64 positions), re-run the cached == uncached test. Report total parameters, active parameters per token (router + top-k routed + shared + attention), and bytes per token (unchanged by MoE; capped by the window for the sliding-window layers).
