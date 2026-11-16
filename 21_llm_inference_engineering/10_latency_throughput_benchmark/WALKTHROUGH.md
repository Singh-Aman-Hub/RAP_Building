# Step 09 — Throughput benchmark: latency vs throughput vs memory

## The story so far
[Step 07](../07_hf_inference_batching_kvcache/WALKTHROUGH.md) showed that batching and caching help. "It feels faster" is not an engineering answer.

## The pain point
Latency and throughput are different goals. A bigger batch makes the whole system produce more tokens per second but each user waits longer. Quote one number without the other and you will mislead yourself and your manager.

## The spark
Run the same generation at batch sizes 1, 2, 4, 8, time it, and compute tokens per second for the whole system and for one user. Record peak memory too.

## Goal
Produce your own table showing that batching raises **throughput** (tokens/s for the whole system) while lowering **per-user speed** (what one request feels).

## Run it
```bash
cd 06_llm_training_and_fast_inference/09_throughput_benchmark
python main.py                 # CPU (reference)
python main.py --device mps    # Apple GPU: noisy for such a small model
```

## Definitions
- **Latency** = time for one request to finish (here: seconds for N new tokens). Users feel this.
- **Throughput** = total tokens generated per second across all requests. Cost per token depends on this.
- **Memory** = weights (fixed) + KV cache and activations (grow with batch and length).

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `ap.add_argument("--device", default="cpu", choices=[...])` | command-line option with allowed values | |
| `with open(path, "w", newline="") as f:` | open a file and close it automatically | |
| `csv.writer(f).writerows(rows)` | write a list of tuples as CSV | |
| `def sync():` | tiny helper that hides device differences | |
| `sys.platform == "darwin"` | detect macOS | |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does |
|---|---|---|
| `resource.getrusage(RUSAGE_SELF).ru_maxrss` | built-in (Unix) | peak memory of this process (bytes on macOS, KB on Linux) |
| `torch.mps.synchronize()` / `torch.cuda.synchronize()` | torch | wait for asynchronous GPU work to end before reading the clock |
| `model.generate(..., min_new_tokens=N, max_new_tokens=N)` | transformers | force exactly N tokens per sequence, so comparisons are fair |
| `csv` | built-in | read/write CSV |

## Execution trace (real CPU run, 40 new tokens)
| batch | latency (s) | tokens/s | per-user tok/s |
|---|---|---|---|
| 1 | 1.24 | 32.3 | 32.3 |
| 2 | 0.98 | 81.8 | 40.9 |
| 4 | 1.13 | 142.2 | 35.6 |
| 8 | 1.14 | 280.8 | 35.1 |
| 16 | 1.13 | 566.8 | 35.4 |
- 16x more requests cost about the same latency (1.13 s vs 1.24 s): throughput x17.5 in this run. Batch 1 was the slowest row in this run (probably start-up effects), so the per-user column barely falls here; in an earlier quieter run it fell from 70.7 to 42.8 tokens/s. Expect noise: warm up and repeat.
- Peak RSS stays flat (737 MB here, ~805 MB in earlier runs): weights dominate; this tiny model's KV cache is negligible (the calculator in step 08 shows when it is not).
- On `--device mps` the numbers were erratic (e.g. batch 1 faster than batch 2). Small models under-use a GPU and first-time shapes cause overhead — the lesson is to **warm up, repeat, and distrust one run**.

## Experiments (change ONE thing)
1. `--new-tokens 100`: longer outputs, bigger KV cache.
2. Add batch size 32 and 64: where does throughput flatten? (compute-bound)
3. Compare with `07`'s unbatched numbers.

## Debugging practice
`--device cuda` on a Mac: PyTorch raises an error that it was "not compiled with CUDA enabled" (or an assertion about invalid device). Check devices with `torch.cuda.is_available()` and `torch.backends.mps.is_available()` before choosing.

## Rebuild checklist
- [ ] warm-up call, then timed calls
- [ ] throughput = batch x tokens / seconds; per-user = tokens / seconds
- [ ] synchronise before reading the clock on GPUs

## The new problem (cliffhanger)
Throughput rises with batch size, then something stops it: the KV cache for every request in the batch must live in GPU memory, and naive servers reserve far more than they use. How do production engines fix that? [Step 10, vLLM](../10_vllm_reference/WALKTHROUGH.md).

> **Say it aloud (interview box):** Latency is how long one user waits, and throughput is the total tokens per second the system produces, and batching raises throughput while lowering each user's speed. I measure both with a warm-up run, at several batch sizes, on the same prompts. The point where the table stops improving shows what limits the system, and on a real GPU that is usually KV-cache memory.
