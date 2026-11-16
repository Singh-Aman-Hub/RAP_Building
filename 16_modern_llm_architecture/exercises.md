# Module 16 — Exercises

Rules: change ONE thing at a time, predict the result before running, and write down what happened. Worked answers (some as runnable scripts) are in [solutions/solutions.md](solutions/solutions.md). Do not look until you have tried.

## Step 01 — BPE tokenizer
- **Modify:** set `NUM_MERGES` to 50, 300 and 1000 and record bytes per token. Where do diminishing returns start?
- **Modify:** add a pre-tokenizer that splits the text on spaces before counting pairs (merges must not cross word boundaries). What changes in the learned merges?
- **Break it:** in `decode`, return `vocab[i]` without `b"".join` or use `i` instead of `vocab[i]`. Read the error.
- **Rebuild:** write `train_bpe(text, n_merges)` returning `(merges, vocab)`, then `encode`/`decode`, from memory.
- **Explain aloud:** why does a byte-level BPE never need an `<unk>` token, and why does Hindi cost more tokens than English in gpt2?

## Step 02 — positional encodings
- **Modify:** change the base to 1,000,000 and print the longest wavelength. What happens to the number of pairs above 4,096?
- **Modify:** test the relative-position property with distance 0 and distance 50 at positions 10 and 10,000. Is the float error larger for big positions? Why?
- **Break it:** set `D = 7`. Which line fails first and why?
- **Rebuild:** `apply_rope` from memory, then rewrite it in the rotate-half layout (as step 07) and show the two layouts give the same scores once the dimensions are permuted.
- **Explain aloud:** why does RoPE not need parameters, and why do NTK-aware and YaRN treat high and low frequencies differently?

## Step 03 — norms and FFN
- **Modify:** change the RMSNorm `eps` to 1e-1. How far from 1.0 is the output RMS?
- **Modify:** use hidden = 4·D for SwiGLU and compare the parameter ratio with the GELU MLP.
- **Break it:** drop `keepdim=True`. Read the error. When could it silently succeed?
- **Rebuild:** a pre-norm block with RMSNorm, attention stub and SwiGLU; count its parameters for d=768.
- **Explain aloud:** pre-norm vs post-norm in three sentences, including why a final norm is needed.

## Step 04 — attention variants and KV cache
- **Modify:** add the config of a model you know (copy `num_hidden_layers`, `num_key_value_heads`, `head_dim` from its `config.json`) to `configs` and compute KB per token and GB at 128k.
- **Modify:** change GQA-2 to GQA-4 and read the parameter and cache columns.
- **Break it:** `Attention(512, 8, 3)`. Why is the assertion necessary?
- **Rebuild:** `Attention` with `n_kv_heads` and `repeat_interleave` from memory; prove MQA with `n_kv_heads=1` equals MHA with all K/V heads copied from one head.
- **Explain aloud:** why GQA keeps quality close to MHA while MQA loses some, and what MLA does differently.

## Step 05 — FlashAttention
- **Modify:** set `block_k=1` and then `block_k=200`. Does the result change? Does the speed?
- **Modify:** time `naive_attention` against `flash_attention` for N = 200, 800, 3200 (numpy). Which is faster on a CPU and why is that not the point?
- **Break it:** remove the `* scale` on `acc`. How large is the error and why does it not crash?
- **Rebuild:** online softmax for one row, then the tiled forward pass, from memory.
- **Explain aloud:** why attention is memory-bound and what "IO-aware" means.

## Step 06 — Mixture of experts
- **Modify:** set `top_k=1, n_shared=0`. What happens to the active-parameter ratio and the utilisation?
- **Modify:** run the aux-loss training with `lr=0.05` and then `lr=0.01`. Explain the dead experts.
- **Modify:** set `gamma=0.1` for the bias method. Does it oscillate?
- **Break it:** `top_k=9` with 8 experts. Read the error.
- **Rebuild:** the dispatch loop with `nonzero` and indexed add; then the loss `E * sum(f * P)`.
- **Explain aloud:** total vs active parameters and why the bias is used for selection but not for weighting.

## Step 07 — mini-Llama
- **Modify:** set `n_kv_heads` to 8 and to 1. Compare parameter counts and cache bytes per token.
- **Modify:** train 600 steps instead of 300. How do validation loss and generated text change?
- **Modify:** replace greedy with temperature sampling (`softmax(logits / T)` and `torch.multinomial`). Does cached == uncached still hold with a fixed seed?
- **Break it:** move `q, k = rope(q, k, start)` after `cache.update(...)`. What breaks and why?
- **Break it:** compute `start` inside `Attention` from the cache length. Which equality test fails?
- **Rebuild:** `KVCache` and `Attention.forward` from memory.
- **Explain aloud:** the data flow of one decode step with a 20-token cache, with shapes.

## Step 08 — read a real config
- **Modify:** change `tie_word_embeddings` to `False` in the `LlamaConfig` of the equality test. What does the logit difference become and why?
- **Modify:** change `rope_theta` in the HF config only. Explain the new difference.
- **Break it:** map `mlp.gate` to `mlp.up_proj` in `rename`. Why is there no error? What does that teach about `strict=False`?
- **Rebuild:** compute the parameter count of Qwen2.5-0.5B from `config.json` with a pocket calculator and compare with 494,032,768.
- **Explain aloud:** the role of every field in a Llama-family `config.json`.

## Step 09 — moved
Speculative decoding exercises now live with the code in [module 21 step 13](../21_llm_inference_engineering/13_speculative_decoding/) (it is a serving optimisation). Step numbering in this module skips 09.

## Step 10 — landscape
- **Modify:** add a model of your choice (read its `config.json` and card) with `None` for anything you cannot verify.
- **Modify:** compute KV GB at 1M tokens for every row. Which models become impossible on one 80 GB GPU together with their weights?
- **Break it:** forget `global_layers` on gpt-oss. How wrong is the 128k cache?
- **Rebuild:** a one-page table of five switches (tokenizer, position, norm, FFN, attention) for three models from memory.
- **Explain aloud:** how reasoning models and hybrid-attention models change the cost story.

## Capstone
Take your trained step-07 model and (a) add MoE (step 06) in place of the MLP in every other block, (b) add a sliding-window mask of 64 tokens, (c) re-verify cached == uncached. Report parameters (total and active) and bytes per token of the cache.
