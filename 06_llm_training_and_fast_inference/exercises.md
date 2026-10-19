# Module 06 — Exercises

Rules: change ONE thing at a time, predict the result before running, write down what happened. Worked answers are in [solutions/](solutions/). Do not look until you have tried.

## Step 01 — dataset preparation
- **Modify:** set `BLOCK_SIZE = 16`. Predict the number of blocks (1206 // 16), then check. Then add 3 of your own lines to `data/tiny_corpus.txt` and re-run.
- **Break it:** change `.view(n_blocks, BLOCK_SIZE)` to `.view(n_blocks, BLOCK_SIZE + 1)`; read the traceback aloud and say which line and why.
- **Rebuild:** close the file; write `make_blocks(tokenizer, lines, block_size)` from memory. Test: `blocks.shape[1] == block_size`.
- **Explain aloud:** why are labels a copy of the inputs, and what does `-100` do?

## Step 02 — training loop
- **Modify:** run with `LEARNING_RATE = 5e-4`, then `5e-7`. Compare to `solutions/lr_sweep.py` output.
- **Modify:** log the loss to a list and print the average of the first 7 and last 7 steps.
- **Break it:** delete `optimizer.zero_grad()`. Does Python complain? What did you notice in the loss?
- **Rebuild:** write the 4-beat loop from memory on a 2-layer toy model with random data (module 03 `04_pytorch_training_loop` can help).
- **Explain aloud:** what does `clip_grad_norm_` return and why is it printed before clipping?

## Step 03 — evaluation and checkpoints
- **Modify:** train 12 epochs; plot (or print) train vs validation loss. At which epoch does validation stop improving?
- **Modify:** compute validation perplexity on a sentence *outside* the corpus style (e.g. a news sentence). Why is it much higher?
- **Break it:** delete the tokenizer save line, then try loading the tokenizer from the checkpoint folder. Read the error.
- **Rebuild:** write `evaluate()` from memory including `eval()` and `no_grad()`.
- **Explain aloud:** perplexity in plain English.

## Step 04 — LoRA math
- **Modify:** set `r=1`, then `r=16` in the learning demo; explain the final error each time.
- **Modify:** compute the parameter saving for a 4096x4096 layer at r=8 and r=64 by hand, then with code.
- **Break it:** initialise `B` randomly; print the "at init" difference.
- **Rebuild:** implement `lora_forward(W, A, B, x, alpha, r)` from memory.
- **Explain aloud:** why is `B` zero-initialised and `A` random?

## Step 05 — LoRA with peft
- **Modify:** r=2 then r=32; compare `trainable%` and loss (see `solutions/lora_rank_sweep.py`).
- **Modify:** adapt `["c_attn", "c_proj"]`; how many trainable parameters now?
- **Break it:** `target_modules=["q_proj"]` — read the error; print `named_modules()` to find the right name.
- **Rebuild:** set up LoRA on `distilgpt2` without looking: config, `get_peft_model`, optimizer over trainable params.
- **Explain aloud:** what is saved to disk and how you would deploy the result.

## Steps 06 and 10 — QLoRA and vLLM (read only)
- **Explain aloud:** the QLoRA recipe in three bullet points; PagedAttention vs continuous batching in two sentences each.
- **Written task:** in your own words, list what must change to run step 05 as QLoRA on a CUDA machine (answer in `solutions/solutions.md`).

## Step 07 — fast inference
- **Modify:** make the KV-cache comparison with `max_new_tokens=300`. Predict the ratio first.
- **Modify:** time `dtype=torch.bfloat16` vs fp32 generation on CPU. Surprised?
- **Break it:** use right padding for the mixed-length batch and compare with left padding.
- **Rebuild:** write a function `batched_generate(prompts)` with correct padding and `no_grad`.
- **Explain aloud:** why does the cache make generation faster but use more memory?

## Step 08 — memory calculator
- **Modify:** add a preset for a model of your choice (look up layers, kv_heads, head_dim in its `config.json`).
- **Modify:** find the largest batch that fits in 24 GB for `llama-3-8b` at int4 with seq 4096.
- **Break it:** run with `--preset gpt5` and read the argparse error.
- **Rebuild:** write `kv_cache_bytes` from memory and verify with distilgpt2.
- **Explain aloud:** why does int4 quantisation not shrink the KV cache here?

## Step 09 — benchmark
- **Modify:** add batch sizes 32 and 64. Where does throughput stop improving?
- **Modify:** run twice; how much does the same row vary? What does that say about a single measurement?
- **Break it:** `--device cuda` on a Mac; read the error.
- **Rebuild:** write the timing loop with warm-up from memory.
- **Explain aloud:** latency vs throughput with your own measured numbers.

## Integration challenge
Take your LoRA-adapted model from step 05, load it with `PeftModel.from_pretrained`, merge it, save it with `save_pretrained`, and time `generate` before and after merging (hint: adapter on = a little extra work per layer). Write 5 lines on what you measured.
