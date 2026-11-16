# Module 21 — Exercises

One section per step, in step order, graded easy to hard inside each step. Labels: **A1-A4** (steps 01-02), **Q1-Q6** (steps 04-07), **P1-P6** (steps 09-13), **D1-D6** (steps 14-16) are paper-and-code exercises with a runnable solution in `solutions/`; the unlabeled bullets are "modify / break it / rebuild / explain aloud" tasks on the step's own `main.py`. Answers and notes: [solutions/solutions.md](solutions/solutions.md). Run solution scripts from the repo root with `HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/solutions/<file>.py`.

---

## Part A — loading and storage

### Step 01 — checkpoint formats and loading ([walkthrough](01_checkpoint_formats_and_loading/WALKTHROUGH.md))
- **A1 (easy, rebuild) — write a safetensors file by hand.** Using only `json`, `struct` and NumPy, write a `.safetensors` file with three tensors (fp32, fp16, int64): 8-byte little-endian header length, JSON header with `dtype`, `shape`, `data_offsets`, then raw bytes. Prove the official `safetensors` reader accepts it. Solution: `solutions/a01_write_safetensors_by_hand.py`.
- **A2 (medium, compute) — scan a pickle without running it.** With `pickletools.genops`, list every module and name a pickle would import (handle both `GLOBAL` and `STACK_GLOBAL`), flag anything outside an allow-list, and show that a `__reduce__` pickle is flagged while a plain dict is not. Why is a scanner only a heuristic? Solution: `solutions/a02_pickle_opcode_scanner.py`.
- **A3 (medium, estimate) — plan a sharded checkpoint.** For a Llama-70B-like model in bf16, write a greedy planner that never splits a tensor and keeps each shard under 5 GB, and build the `weight_map` of `model.safetensors.index.json`. How many shards and how many GB? Solution: `solutions/a03_shard_plan.py`.
- **Modify:** in `main.py` change the synthetic checkpoint size and re-run section 3. Does eager peak RSS stay close to file size plus overhead? Does `safe_open` still touch only what you read?
- **Break it:** call `torch.load` with `weights_only=False` on the pickle from section 1 (in a throwaway script) and describe what you just allowed.
- **Explain aloud:** why can `from_pretrained` of a 328 MB safetensors model show almost no RSS growth, and why does converting to bf16 change that?

### Step 02 — GGUF and llama.cpp formats ([walkthrough](02_gguf_and_llamacpp_formats/WALKTHROUGH.md))
- **A4 (easy, compute) — block sizes and file sizes.** A Q8_0 block holds 32 weights and an fp16 scale; a Q4_0 block holds 32 four-bit codes and an fp16 scale. Derive bytes per block and bits per weight, size a 7B and a 70B model in each, and confirm with the quantizers in step 02's `main.py`. Solution: `solutions/a04_gguf_block_sizes.py`.
- **Modify:** add a tensor whose element count is not a multiple of 32 to the writer. What does llama.cpp require of block-quantized rows and what would you do about it?
- **Explain aloud:** why does GGUF store tensor dimensions reversed relative to NumPy, and why is the data section aligned to 32 bytes?

### Step 03 — GPU memory calculator ([walkthrough](03_gpu_memory_calculator/WALKTHROUGH.md))
- **Modify:** add a preset for a model of your choice (look up layers, kv_heads, head_dim in its `config.json`).
- **Modify:** find the largest batch that fits in 24 GB for `llama-3-8b` at int4 with seq 4096.
- **Break it:** run with `--preset gpt5` and read the argparse error.
- **Rebuild:** write `kv_cache_bytes` from memory and verify with distilgpt2 (formula 1,179,648 bytes at 32 tokens, fp32).
- **Explain aloud:** why does int4 weight quantisation not shrink the KV cache here?

---

## Part B — quantization

### Step 04 — quantization math ([walkthrough](04_quantization_math/WALKTHROUGH.md))
- **Q1 (easy) — affine quantization by hand.** For the range [-0.4, 3.0] and uint8: compute scale `s` and zero-point `z`, quantize 10,001 evenly spaced values, de-quantize, and check (a) worst error at most `s/2`, (b) 0.0 is represented exactly. Solution: `solutions/q01_affine_by_hand.py`.
- **Q2 (easy) — bits per weight.** Write `bpw(bits, group, scale_bits, zero_bits)`. Show int4 g=128 with an fp16 scale costs 4.125 bits per weight, then size a 70B model for int4 g128, int4 g64, int3 g128, MXFP4 (block 32, 8-bit scale) and NVFP4 (block 16, 8-bit scale). Solution: `solutions/q02_bits_per_weight.py`.
- **Q3 (medium) — pack int4 for an odd length.** Pack signed 4-bit codes two per byte, handling an ODD number of values (pad one nibble, remember the original length); verify the round trip for n = 1, 2, 7, 1000, 1001. Solution: `solutions/q03_pack_int4_odd_length.py`.
- **Q4 (medium) — MSE-optimal clipping.** For Gaussian and Student-t (df=3) data find the clip factor (fraction of absmax) that minimises MSE at int8 and at int4. Which data and which bit width want the harder clip, and why? Solution: `solutions/q04_mse_clip_search.py`.

### Step 05 — activation quantization and calibration ([walkthrough](05_activation_quant_and_calibration/WALKTHROUGH.md))
- **Modify:** `BITS=3 python main.py` (then 4): when does PTQ catch up with QAT?
- **Break it:** divide the weights by `s` instead of multiplying in SmoothQuant and read the failing assert.
- **Q5 (medium-hard) — SmoothQuant preserves the product and fixes an outlier channel.** Build X (512x256) with one channel scaled by 60 and W (128x256). Verify `(X/s) @ (W*s).T == X @ W.T` for `s_j = max|X_j|^0.5 / max|W_j|^0.5`, and report per-tensor W8A8 SQNR before and after. Solution: `solutions/q05_smoothquant_identity.py`.
- **Explain aloud:** why does a 99.9th-percentile calibration rule make perplexity worse here, when it usually helps?

### Step 06 — GPTQ and AWQ from scratch ([walkthrough](06_gptq_awq_from_scratch/WALKTHROUGH.md))
- **Modify:** set `GROUP = 64`, then 32, and watch the RTN-versus-GPTQ gap.
- **Modify:** change `damp=0.01` to `1e-6` and to `1.0`; explain both failures.
- **Break it:** skip the damping line and read the Cholesky error.
- **Explain aloud:** GPTQ versus AWQ versus RTN in three sentences: what each optimises, what each needs (calibration data? Hessian?), and which is the cheapest.

### Step 07 — KV-cache quantization ([walkthrough](07_kv_cache_quantization/WALKTHROUGH.md))
- **Modify:** quantize V per-channel instead of per-token and compare the KL of the next-token distribution.
- **Q6 (hard) — why K wants per-channel and V per-token scales.** Synthesize K (256 tokens x 64 channels, two channels 100x larger in every token) and V (token magnitudes vary, channels alike). Quantize both to int4 per-token and per-channel and compare SQNR on the NORMAL channels. Why would whole-tensor SQNR mislead? Solution: `solutions/q06_kv_per_channel_vs_token.py`.
- **Explain aloud:** why is greedy-token agreement a brittle metric for judging a quantized KV cache?

---

## Part C — performance

### Step 08 — fast inference with HF ([walkthrough](08_hf_inference_batching_kvcache/WALKTHROUGH.md))
- **Modify:** make the KV-cache comparison with `max_new_tokens=300`. Predict the ratio first (our 100-token run was 4.2 s without the cache against 1.6 s with it).
- **Modify:** time `dtype=torch.bfloat16` vs fp32 generation on CPU. Surprised?
- **Break it:** use right padding for the mixed-length batch and compare with left padding.
- **Rebuild:** write a function `batched_generate(prompts)` with correct padding and `no_grad`.
- **Explain aloud:** why does the cache make generation faster but use more memory?

### Step 09 — prefill, decode and the roofline ([walkthrough](09_prefill_decode_roofline/WALKTHROUGH.md))
- **P1 (easy) — ridge points and the decode crossover batch.** Using A100 (312 TFLOP/s dense bf16, 2.039 TB/s) and H100 (989 TFLOP/s, 3.35 TB/s): compute each ridge point, then the batch above which weight-only decode becomes compute-bound for fp16, int8 and int4 weights. What does quantisation do to that batch and why? Solution: `solutions/p01_roofline_ridge.py`.
- **Modify:** `THREADS=1 python main.py`. The compute roof drops about 4x: does batch-1 decode time change much? Why not?
- **Break it:** change `ctx=16` to 900 in `decode_sweep` and explain the rise in step time.
- **P6 (hard) — when do KV bytes dominate a decode step?** For a Llama-2-7B-like fp16 model (32 layers, 32 KV heads, head_dim 128, 6.74B parameters) compute KV bytes per token and the batch x context at which KV bytes read per step equal the weight bytes. Repeat with 8 KV heads. Which assumption of step 09 fails first? Solution: `solutions/p06_kv_vs_weight_bytes.py`.

### Step 10 — latency/throughput benchmark ([walkthrough](10_latency_throughput_benchmark/WALKTHROUGH.md))
- **P2 (easy) — metrics from timestamps.** A request starts at t=0 and its 5 tokens appear at 0.200, 0.220, 0.240, 0.360, 0.380 s. Compute TTFT, the ITL list, TPOT, E2E and ITL p95, and decide whether it meets an SLO of TTFT <= 250 ms and TPOT <= 40 ms. Why does p95 ITL differ so much from TPOT? Solution: `solutions/p02_metrics_from_timestamps.py`.
- **Modify:** add batch sizes 32 and 64 to the `for bs in (1, 2, 4, 8, 16)` loop. Where does throughput stop improving, and what happens to goodput?
- **Modify:** `--prompt-len 512`. Which metric moves (TTFT or TPOT) and why?
- **Modify:** run twice. How much does the same row vary? What does that say about a single measurement?
- **Break it:** `--device cuda` on a Mac; read the error. Then set `trials=1`, `warmup=0` and explain why the std column becomes fake confidence.
- **Rebuild:** write the timing loop with warm-up and per-token timestamps from memory.
- **Explain aloud:** latency vs throughput vs goodput with your own measured numbers.

### Step 11 — PagedAttention fragmentation ([walkthrough](11_paged_attention_fragmentation/WALKTHROUGH.md))
- **Modify:** set `BLOCK_SIZE = 128`, then `1`. Predict which direction the waste in section A moves first.
- **Modify:** change `MAX_LEN` to 4096. Which allocator suffers more, and why?
- **Break it:** make `PagedAllocator.free` forget to return blocks and read the error.
- **Rebuild:** write the free-stack allocator and the block table from memory, then re-create the A) numbers.
- **Explain aloud:** internal vs external fragmentation, and what the block table does, in three sentences.

### Step 12 — continuous batching and prefix caching ([walkthrough](12_continuous_batching_prefix_caching/WALKTHROUGH.md))
- **P5 (medium) — static vs continuous by hand.** Four requests with output lengths 2, 8, 2, 2 arrive together; the server has 2 slots and a decode step costs one unit (ignore prefill). Compute each request's finish time and the total under static and continuous batching. Solution: `solutions/p05_static_vs_continuous_by_hand.py`.
- **P4 (medium) — chained prefix hashes.** Cut a token list into full 16-token blocks and return chained hashes. Show that two prompts sharing exactly 50 tokens reuse 3 blocks (48 tokens), and construct a case where an un-chained hash gives a wrong cache hit. Solution: `solutions/p04_chained_prefix_hash.py`.
- **Modify:** `RATE=6 python main.py`. Which scheduler saturates first?
- **Modify:** change `BlockCache(16)` capacities in part C and find the capacity where LRU stops thrashing.
- **Break it:** drop the `h` from `hash((h, ...))` (un-chain the blocks) and describe the correctness bug.

### Step 13 — speculative decoding ([walkthrough](13_speculative_decoding/WALKTHROUGH.md))
- **P3 (medium) — best draft length.** For acceptance alpha and draft/target cost ratio c, speedup = `(1 - alpha^(k+1)) / ((1 - alpha)(k*c + 1))`. Find the best k for (alpha, c) in (0.6, 0.05), (0.8, 0.05), (0.8, 0.3), (0.9, 0.02), and check tokens-per-pass with a Monte Carlo at alpha 0.8, k 4. Solution: `solutions/p03_speculative_optimal_k.py`.
- **Modify:** use unpredictable prompts (random words) and gamma 16; how does acceptance respond?
- **Break it:** replace the acceptance cumprod by accepting everything and watch "identical" turn False.
- **Explain aloud:** why does speculative decoding with greedy or properly rejection-sampled drafts give the same output distribution as the target alone?

---

## Part D — engines, parallelism, cost, report

### Step 14 — parallelism and offloading ([walkthrough](14_parallelism_and_offloading/WALKTHROUGH.md))
- **D1 (easy, rebuild):** write `tp_linear(X, W, n, mode)` for `mode` in `"column"` / `"row"` and verify it equals `X @ W` for N in 1, 2, 3, 4, 6, 8. What happens with N=5 and why? Solution: `solutions/d01_tp_linear_any_n.py`.
- **D2 (easy, compute):** for a pipeline of p = 8 stages, how many micro-batches keep the bubble below 10%? Derive from `(p-1)/(m+p-1)` and verify by simulation. Solution: `solutions/d02_bubble_micro_batches.py`.
- **D3 (medium, estimate):** for an 80-layer, hidden-8192 model in fp16 at TP=8, how many bytes does each GPU communicate per token, in how many collectives, and how long does the bandwidth part take over NVLink vs 100 GbE? Solution: `solutions/d03_tp_comm_70b.py`.
- **D6 (medium, training side):** the smallest ZeRO stage that trains a 13B model on 16 GPUs with a 32 GB budget for weights + gradients + optimizer. Solution: `solutions/d06_zero_stage_for_13b.py`.
- **Modify:** change `max_memory` in section I so everything fits in RAM; then so little fits that every layer goes to disk. Time one forward in each case.
- **Explain aloud:** why does Megatron split the first MLP matrix by columns and the second by rows? What breaks if you split the first by rows?

### Step 15 — serving engines ([walkthrough](15_serving_engines_landscape/WALKTHROUGH.md))
- **Explain aloud:** PagedAttention vs continuous batching in two sentences each.
- **Explain aloud:** pick an engine for (a) a laptop with a 4-bit GGUF model and one user, (b) 500 concurrent users with a shared 2k-token system prompt, (c) an NVIDIA-only fleet where cost per token is the KPI. Defend each in two sentences using scheduler, KV management and quantization format.
- **Modify:** add a row for an engine you read about to `ENGINES`, a rule in `recommend()`, and a scenario the `assert` checks.

### Step 16 — cost per token ([walkthrough](16_cost_per_token/WALKTHROUGH.md))
- **D4 (medium, compute):** break-even volume of 2 always-on $3/h GPUs against an API at $0.60/$2.40 per 1M (80/20 mix); then with 50% extra capacity for redundancy. Solution: `solutions/d04_break_even_redundancy.py`.
- **D5 (medium, compute):** what prefix-cache hit rate cuts a request's cost by 30% when cached input costs 25% of normal input? (Prefix caching itself is step 12.) Solution: `solutions/d05_prefix_cache_hit_target.py`.
- **Modify:** put your cloud's real GPU price into the dict and re-run; which conclusion changed?
- **Break it:** set utilisation to 0 and read the error; what does that say about an idle GPU's cost per token?
- **Explain aloud:** why is an output token more expensive than an input token, and how does batching change that?

### Step 17 — inference report project ([walkthrough](17_inference_report_project/WALKTHROUGH.md))
- **Rebuild:** add a configuration axis to `main.py` (for example `--threads 1/4`), regenerate the report and update the recommendation text.
- **Break it:** remove the quality gate in `recommend()`; which config becomes "fastest", and what did it cost in perplexity?
- **Explain aloud:** present `inference_report.md` in two minutes using the rubric in [17_inference_report_project/README.md](17_inference_report_project/README.md).
