# Module 21 — Solutions and notes

Exercises: [../exercises.md](../exercises.md). Every script below was run from the repo root with `HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/solutions/<file>.py`; all 22 exit 0 and end with an OK/ok line (see the index at the bottom). Timing-based "modify" tasks vary by machine; the notes give the expected direction, not exact numbers.

## Part A — loading and storage

### Step 01
- **A1** (`a01_write_safetensors_by_hand.py`): file = 8-byte little-endian length N + N bytes of JSON (padded with spaces to 8-byte alignment) + raw data; `data_offsets` are relative to the start of the data section and contiguous. Output: 300 bytes = 8 + 176 + 116; the official reader round-trips all three tensors (fp32, fp16, int64). Takeaway: the format is just JSON plus bytes, so there is no code path to attack.
- **A2** (`a02_pickle_opcode_scanner.py`): `GLOBAL` (protocol 3 and below) carries "module name" in one argument; `STACK_GLOBAL` (protocol 4+) takes the two strings pushed just before it. The evil pickle yields `[("builtins", "print")]`, the dict yields `[]`. It is a heuristic: attackers can hide the import behind allowed callables or build the name dynamically, so scanning is defence in depth; the real fix is a format with no code (safetensors) or `weights_only=True`.
- **A3** (`a03_shard_plan.py`): 642 tensors, 141.1 GB, 30 shards of at most 5 GB, each tensor in exactly one shard, index covers all. A tensor is never split across shards (this is why the largest shard can be a bit under the cap and why a huge embedding needs a big enough cap).
- **Modify:** eager peak RSS stays near file size plus the copy overhead; `safe_open` still grows RSS only by the tensors you read. On a busy macOS machine RSS can be compressed, so read it as indicative.
- **Break it:** `weights_only=False` re-enables the `REDUCE` opcode: the file can run anything as you, at load time.
- **Explain aloud:** safetensors weights are memory-mapped, so only touched pages count towards RSS; converting dtype must read and rewrite every value, which forces a real copy.

### Step 02
- **A4** (`a04_gguf_block_sizes.py`): Q8_0 = 2 + 32 = 34 bytes per 32 weights = 8.5 bpw (7B about 7.4 GB, 70B 74.4 GB); Q4_0 = 2 + 16 = 18 bytes = 4.5 bpw (7B 3.9 GB, 70B 39.4 GB). Step 02's own quantizers produce exactly those byte counts.
- **Modify:** block formats need the row length to be a multiple of the block size (32 for Q8_0/Q4_0, 256 for k-quants); otherwise pad the row or keep that tensor in a higher precision format (llama.cpp falls back to a bigger type for such tensors).
- **Explain aloud:** GGUF stores dims fastest-varying first (ggml's convention), hence numpy `(8, 64)` becomes `[64, 8]`; 32-byte alignment lets the file be mmap-ed and read with aligned, vectorised loads.

### Step 03
- Preset: read `num_hidden_layers`, `num_key_value_heads`, `hidden_size / num_attention_heads` from `config.json`.
- Largest batch: subtract int4 weights (4 GB) and the overhead from 24 GB, divide the rest by the KV bytes for one 4096-token sequence (fp16: 4096 x 131,072 B = 0.5 GiB for Llama-3-8B). Roughly 30-35 sequences before the 10% overhead; run the calculator to get your exact figure.
- The KV cache stays fp16 in the table while only the weights are quantised; quantise the cache separately (fp8/int8 KV) to shrink it.
- `kv_cache_bytes` check: distilgpt2, 32 tokens, fp32 = 1,179,648 bytes (2 x 6 layers x 12 heads x 64 x 32 x 4).

## Part B — quantization

### Step 04
- **Q1** (`q01_affine_by_hand.py`): s = 3.4/255 = 0.013333, z = round(0.4/s) = 30; worst error = s/2 = 0.006667; 0.0 maps to code 30 and back exactly. This is why the zero-point is an integer: "no error on zero" matters for padding and ReLU outputs.
- **Q2** (`q02_bits_per_weight.py`): bpw = bits + (scale_bits + zero_bits)/group. int4 g128 = 4.125 (70B = 36.1 GB), int4 g64 = 4.25 (37.2 GB), int3 g128 = 3.125 (27.3 GB), MXFP4 = 4.25 (37.2 GB), NVFP4 = 4.5 (39.4 GB). Lesson: always count the scales.
- **Q3** (`q03_pack_int4_odd_length.py`): add 8 to map -8..7 to 0..15, pad one nibble when n is odd, keep `n`, slice the unpacked array back to `n`. Without remembering `n` the last value would come back as a phantom extra element.
- **Q4** (`q04_mse_clip_search.py`): Gaussian int8 clip 0.84 (1.2x lower MSE than absmax), Gaussian int4 0.52 (2.9x), Student-t int8 0.86 (1.2x), Student-t int4 0.30 (3.5x; this is the lower edge of the grid, so the true optimum may be lower). Heavy tails and fewer bits both want a harder clip: with only 16 levels the rounding error of the many ordinary values dominates, so saturating a few giants is worth it.

### Step 05
- Modify `BITS`: PTQ catches up with QAT as the bit width grows; at 2 bits per-tensor weights our run gave float 0.998, PTQ 0.619, QAT 1.000.
- Break it: `W / s` gives `(X/s) @ (W/s).T != X @ W.T`, so the identity assert fails; the product is only preserved when the weights get the inverse scaling.
- **Q5** (`q05_smoothquant_identity.py`): the product is identical up to float error, and per-tensor W8A8 SQNR rises from 18.8 dB to 33.0 dB because the outlier channel no longer sets the activation scale (its size moves into the weights).
- Explain aloud: percentile clipping assumes outliers are noise; in LLMs the outlier channel carries signal, so clipping it hurts (our perplexity went from 32.55 min-max to 141.68 with static percentile).

### Step 06
- Smaller groups (64, 32) improve RTN more than GPTQ, so the gap shrinks. `damp` too small makes the Hessian near-singular (Cholesky fails or the update is noisy); too large turns GPTQ back into RTN because the correction is damped away.
- Skipping the damping line: `linear algebra error ... not positive-definite`, because H = X^T X is singular when calibration tokens are fewer than input features.
- In three sentences: RTN rounds each weight independently; AWQ scales salient channels (found from activation magnitudes) before rounding, needs only activation statistics, and is cheapest; GPTQ uses the Hessian of the layer's output error to round column by column and push each column's error onto the not-yet-quantized ones, which is the most accurate and the most expensive.

### Step 07
- Per-channel V instead of per-token raises V error (our script: V per-token 20.73 dB vs per-channel 14.46 dB), confirming V's variation is across tokens, K's across channels.
- **Q6** (`q06_kv_per_channel_vs_token.py`): K normal channels: per-token 0.38 dB vs per-channel 19.17 dB. Whole-tensor SQNR is dominated by the energy of the two giant channels, which per-token scaling represents well, so it hides how badly the other 62 channels are crushed.
- Greedy agreement flips on near-ties and then everything after changes; KL of the next-token distribution is the steadier metric.

## Part C — performance

### Step 08
- Without the cache the cost per new token grows with sequence length, so the ratio grows with `max_new_tokens` (100 tokens: 4.2 s vs 1.6 s in our run, about 2.6x; expect clearly more at 300).
- bf16 on CPU is often not faster (our run for the capstone: much slower); GPUs have bf16/fp16 hardware.
- Right padding continues from pad tokens (junk output plus a warning); left padding gives the correct continuation.

### Step 09
- **P1** (`p01_roofline_ridge.py`): ridge 153 FLOP/byte (A100) and 295 (H100). Decode intensity is about `2B / bytes_per_param`, so `B* = ridge x bytes_per_param / 2`: A100 fp16 153, int8 77, int4 38; H100 295, 148, 74. Quantisation moves the crossover down 4x (fp16 to int4): fewer bytes per weight means more FLOPs per byte at the same batch.
- Modify `THREADS=1`: compute roof falls about 4x, but batch-1 decode is memory-bound so its step time barely changes.
- Break it: `ctx=900` makes attention read a larger KV cache each step, so step time rises.
- **P6** (`p06_kv_vs_weight_bytes.py`): 524,288 bytes (0.5 MiB) per token, weights 13.5 GB, equal at about 25,700 batch-x-context tokens (batch 8 at context 3,200). With 8 KV heads: 131,072 B per token, crossover at about 102,800 tokens. The "bytes per step is about the weights" approximation fails first at long context and large batch.

### Step 10
- **P2** (`p02_metrics_from_timestamps.py`): TTFT 0.200 s; ITL = [20, 20, 120, 20] ms; TPOT = (0.380 - 0.200)/4 = 45 ms; E2E 0.380 s; p95 ITL = 105 ms. TPOT 45 ms misses the 40 ms target, so the request does not count towards goodput. The mean hides the single 120 ms stall; the tail percentile shows it.
- Throughput usually flattens when the hardware becomes compute-bound (large batch) or when the KV cache stops more sequences from fitting; goodput (tokens that meet the SLO) can fall while throughput still rises.
- `--prompt-len 512` moves TTFT (prefill is compute-bound) and leaves TPOT nearly unchanged.
- Repeated runs differ by tens of percent on a laptop, so average several runs and report the spread.
- `trials=1, warmup=0`: the first measured row includes lazy initialisation and the std column is 0, a number with no uncertainty attached.

### Step 11
- Smaller blocks waste less in the last block (at most `BLOCK_SIZE - 1` slots per request) but need a longer block table; `BLOCK_SIZE = 128` makes waste visibly larger, which is why vLLM defaults to 16.
- With `MAX_LEN = 4096` the contiguous reservation doubles while real lengths stay the same, so contiguous fits half as many requests; paged barely changes because it only allocates what exists.
- The leaked-block bug ends with `RuntimeError: simulation did not finish`: no blocks return to the pool, so nothing new is admitted.

### Step 12
- **P5** (`p05_static_vs_continuous_by_hand.py`): static finishes at [8, 8, 10, 10] (total 10); continuous at [2, 8, 4, 6] (total 8). Mean latency 9 vs 5.
- **P4** (`p04_chained_prefix_hash.py`): `h = hash((h, tuple(block)))`; 50 shared tokens = 3 full blocks = 48 tokens reusable. Prompt C with a different first block but the same second block gets a wrong hit if the hash is not chained, because the KV of block 2 depends on everything before it.
- `RATE=6`: static (and dynamic) batching saturates first and the queue grows without bound; continuous degrades gracefully.
- Prefix-cache capacity: below the working set (4 prompts x 30 blocks = 120 blocks in our run) LRU thrashes (0% hits at 16 blocks); around 128 blocks it holds most prefixes (63% in our run).

### Step 13
- **P3** (`p03_speculative_optimal_k.py`): best k = 4, 8, 3, 19 with speedups 1.92x, 3.09x, 1.55x, 6.37x. A costlier draft shortens the best k; higher acceptance lengthens it. Monte Carlo 3.364 tokens per pass vs formula 3.362.
- Unpredictable prompts and gamma 16: acceptance falls sharply and long drafts mostly get thrown away.
- Accepting everything breaks the guarantee: "identical" becomes False because you keep tokens after the first mismatch.
- Aloud: each draft token is accepted with probability min(1, p/q) and a rejection resamples from the residual `max(p - q, 0)`, which makes the output distribution exactly the target's.

## Part D — engines, parallelism, cost, report

### Step 14
- **D1** (`d01_tp_linear_any_n.py`): column split (concatenate outputs) and row split (sum partial outputs) both equal `X @ W` for N in 1, 2, 3, 4, 6, 8; N=5 fails with "array split does not result in an equal division" because the dimension is not divisible by 5 (real frameworks require heads/hidden divisible by TP).
- **D2** (`d02_bubble_micro_batches.py`): `(p-1)/(m+p-1) < 0.1` with p=8 gives m >= 63 micro-batches (bubble 0.100 at 63, 0.101 at 62).
- **D3** (`d03_tp_comm_70b.py`): 2 all-reduces per layer x 80 layers = 160 all-reduces of 16 KB, 4.59 MB communicated per token per GPU; bandwidth part 10 us over NVLink (450 GB/s) vs 382 us over 100 GbE (12 GB/s), and the 160 latencies come on top, which is why TP stays inside a node.
- **D6** (`d06_zero_stage_for_13b.py`): per-GPU memory 208.0 / 61.8 / 37.4 / 13.0 GB for ZeRO-0/1/2/3, so the smallest stage that fits 32 GB is ZeRO-3.
- Modify `max_memory`: everything in RAM is fastest; forcing layers to disk makes the forward bound by disk reads (orders of magnitude slower) with logits unchanged.
- Aloud: column-splitting the first MLP matrix keeps the elementwise activation local (each GPU applies GELU to its own slice); row-splitting the second then needs only one all-reduce at the end. Splitting the first by rows would need a sum before the non-linearity, because `GELU(a + b) != GELU(a) + GELU(b)`; our run showed a "wrong" variant off by tens.

### Step 15
- (a) laptop, 4-bit GGUF, one user: llama.cpp or Ollama (GGUF native, CPU/Metal, single-stream latency). (b) 500 users sharing a 2k prompt: SGLang or vLLM with prefix caching (RadixAttention or automatic prefix caching reuses the shared KV; continuous batching keeps the GPU full). (c) NVIDIA-only, cost per token: TensorRT-LLM (or vLLM) with FP8 kernels and in-flight batching, measured with the step 16 model.
- Modify: add the engine to `ENGINES`, a rule, and a scenario; the `assert` catches rules that contradict an existing scenario.

### Step 16
- **D4** (`d04_break_even_redundancy.py`): blended API price $0.96 per 1M (0.8 x 0.60 + 0.2 x 2.40), fixed cost $4,380 per month for 2 GPUs, break-even 4,562M tokens per month; with 50% extra capacity for redundancy 6,844M. Below the break-even the API is cheaper on cost alone.
- **D5** (`d05_prefix_cache_hit_target.py`): base $0.0042 per request; a 56.0% hit rate gives -30% (cost 0.002941).
- Idle GPU: `ZeroDivisionError`, because cost per token is price / (tokens per hour x utilisation) and utilisation 0 means infinite cost.
- Output tokens cost more because decode is memory-bound and runs one token per step per sequence while prefill processes thousands of tokens per second; batching shares each weight read among many sequences, so the decode price falls with batch.

### Step 17
- A new axis (`--threads`) multiplies the config list; update `recommend()` so ties are broken by the stated rule and the recommendation sentence names the thread count.
- Without the quality gate the fastest config may be one that hurts perplexity (bf16 here was slower and slightly worse; fake-int8 kept quality).
- Two-minute presentation: question, setup (model, CPU, threads, run count), the table, the recommendation with the quality gate, caveats (CPU numbers, tiny model, noise), next step on a GPU.

## Index of solution scripts (all run, all exit 0)

| File | Exercise | Step | Result line |
|---|---|---|---|
| `a01_write_safetensors_by_hand.py` | A1 | 01 | official reader round-trips 3 tensors: OK |
| `a02_pickle_opcode_scanner.py` | A2 | 01 | evil pickle flagged, dict passes, no `pickle.loads` |
| `a03_shard_plan.py` | A3 | 01 | 642 tensors, 141.1 GB, 30 shards |
| `a04_gguf_block_sizes.py` | A4 | 02 | Q8_0 34 B (8.5 bpw), Q4_0 18 B (4.5 bpw) |
| `q01_affine_by_hand.py` | Q1 | 04 | OK |
| `q02_bits_per_weight.py` | Q2 | 04 | OK |
| `q03_pack_int4_odd_length.py` | Q3 | 04 | OK |
| `q04_mse_clip_search.py` | Q4 | 04 | OK |
| `q05_smoothquant_identity.py` | Q5 | 05 | OK |
| `q06_kv_per_channel_vs_token.py` | Q6 | 07 | OK |
| `p01_roofline_ridge.py` | P1 | 09 | OK |
| `p06_kv_vs_weight_bytes.py` | P6 | 09 | OK |
| `p02_metrics_from_timestamps.py` | P2 | 10 | OK (needs `../benchlib.py`) |
| `p04_chained_prefix_hash.py` | P4 | 12 | OK |
| `p05_static_vs_continuous_by_hand.py` | P5 | 12 | OK |
| `p03_speculative_optimal_k.py` | P3 | 13 | OK |
| `d01_tp_linear_any_n.py` | D1 | 14 | ok (N=5 fails as expected) |
| `d02_bubble_micro_batches.py` | D2 | 14 | m >= 63 |
| `d03_tp_comm_70b.py` | D3 | 14 | 4.59 MB, 160 all-reduces |
| `d06_zero_stage_for_13b.py` | D6 | 14 | ZeRO-3 |
| `d04_break_even_redundancy.py` | D4 | 16 | 4,562M / 6,844M tokens per month |
| `d05_prefix_cache_hit_target.py` | D5 | 16 | 56.0% |
