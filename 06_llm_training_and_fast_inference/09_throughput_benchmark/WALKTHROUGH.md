# Step 09 — Throughput benchmark: latency vs throughput vs memory

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
| 1 | 0.57 | 70.7 | 70.7 |
| 2 | 0.75 | 106.4 | 53.2 |
| 4 | 0.81 | 197.5 | 49.4 |
| 8 | 0.86 | 372.6 | 46.6 |
| 16 | 0.94 | 684.1 | 42.8 |
- 16x more requests cost only ~1.6x the latency: throughput x9.7, per-user speed x0.6.
- Peak RSS ~805-809 MB stays flat: weights dominate; this tiny model's KV cache is negligible (the calculator in step 08 shows when it is not).
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

## Say it aloud
"I benchmark by warming up, then timing generation for several batch sizes with a fixed number of new tokens. Batching raises throughput a lot because the hardware does more work per forward pass, but each user's tokens-per-second drops, so serving is a trade-off between cost and latency. Memory is dominated by weights for small models and by KV cache for long contexts. Production servers like vLLM add continuous batching to keep that trade-off favourable."
