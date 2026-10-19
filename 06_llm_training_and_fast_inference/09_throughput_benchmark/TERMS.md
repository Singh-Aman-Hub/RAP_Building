# Terms — Throughput Benchmark

Folder: `06_llm_training_and_fast_inference/09_throughput_benchmark` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [06 README](../README.md)

**Terms in this folder:** [Latency](#latency) · [Throughput](#throughput) · [Per-user tokens per second](#per-user-tokens-per-second) · [Latency–throughput trade-off](#latencythroughput-trade-off) · [argparse options (--device, --new-tokens)](#argparse-options---device---new-tokens) · [padding_side="left" in from_pretrained](#padding_sideleft-in-from_pretrained) · [Device synchronisation](#device-synchronisation) · [resource.getrusage and peak RSS](#resourcegetrusage-and-peak-rss) · [sys.platform](#sysplatform) · [min_new_tokens = max_new_tokens (fair comparison)](#min_new_tokens--max_new_tokens-fair-comparison) · [Warm-up and repeated measurement](#warm-up-and-repeated-measurement) · [List repetition `["x"] * n`](#list-repetition-x--n) · [csv module and writer.writerows](#csv-module-and-writerwriterows) · [`with open(...)` and newline=""](#with-open-and-newline) · [Tuple of batch sizes loop](#tuple-of-batch-sizes-loop) · [round()](#round) · [Compute-bound vs memory-bound](#compute-bound-vs-memory-bound) · [Error: Torch not compiled with CUDA](#error-torch-not-compiled-with-cuda)

---

## Latency  `(concept)`
**1. Definition:** Latency is the time one request takes from start to finish; here, seconds to generate N new tokens.
**2. In plain words:** It's what a user feels while waiting. Lower is better for interactivity.
**3. Example to quote:** > "At batch 1, my 40-token request took 0.57 s; at batch 16 it took 0.94 s."
**Seen in:** `main.py`, `run_batch`; `WALKTHROUGH.md`

## Throughput  `(concept)`
**1. Definition:** Throughput is the total number of tokens generated per second across all requests: `batch × new_tokens / seconds`.
**2. In plain words:** It drives cost per token, because a GPU is paid for whether it's busy or idle. From batch 1 to 16 throughput rose from 70.7 to 684.1 tokens/s.
**3. Example to quote:** > "Throughput went up about 9.7× from batch 1 to 16, so the same hardware serves many more tokens per second."
**Seen in:** `main.py`

## Per-user tokens per second  `(concept)`
**1. Definition:** Per-user tokens per second is `new_tokens / seconds`: the generation speed one individual request experiences.
**2. In plain words:** It's the inverse view of latency. In an earlier quiet run it fell from 70.7 to 42.8 as batch size grew to 16; in the latest run it stayed near 35.
**3. Example to quote:** > "Each user's speed dropped from 70.7 to 42.8 tokens per second at batch 16 in one run (timings vary by machine)."
**Seen in:** `main.py`

## Latency–throughput trade-off  `(concept)`
**1. Definition:** Increasing batch size raises total throughput but typically lengthens each request's latency.
**2. In plain words:** Sixteen times more requests cost only about 1.6× the latency here. Serving is about choosing a point on that curve for your cost and responsiveness needs.
**3. Example to quote:** > "Batching makes serving cheaper per token, but each user waits slightly longer, which is the core trade-off."
**Seen in:** `main.py`, final print

## argparse options (--device, --new-tokens)  `(standard library)`
**1. Definition:** `ap.add_argument("--device", default="cpu", choices=[...])` defines an option with allowed values; `--new-tokens` becomes `args.new_tokens` with underscores.
**2. In plain words:** You change the experiment from the terminal: `python main.py --device mps --new-tokens 100`.
**3. Example to quote:** > "I run `python main.py --new-tokens 100` to test whether longer outputs change the curve."
**Seen in:** `main.py`, top

## padding_side="left" in from_pretrained  `(third-party: transformers)`
**1. Definition:** Passing `padding_side="left"` to `AutoTokenizer.from_pretrained` sets the tokenizer to add padding tokens before the text.
**2. In plain words:** Decoder-only models need left padding for batched generation (step 07). Here it is set once at load time.
**3. Example to quote:** > "I load the tokenizer with `padding_side='left'` so the batch of identical prompts generates correctly."
**Seen in:** `main.py`

## Device synchronisation  `(concept)`
**1. Definition:** `torch.cuda.synchronize()` and `torch.mps.synchronize()` block until all queued GPU work has finished.
**2. In plain words:** GPUs run asynchronously, so a Python line can return before the work is done. Without synchronising, your stopwatch measures only how long it took to queue the work. The `sync()` helper hides the device differences.
**3. Example to quote:** > "I call `sync()` before and after timing so the clock measures real GPU work, not just the queueing."
**Seen in:** `main.py`, `sync()`

## resource.getrusage and peak RSS  `(standard library)`
**1. Definition:** `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` returns this process's peak resident set size (the most physical RAM it used); it's in bytes on macOS and kilobytes on Linux.
**2. In plain words:** It's a cheap way to see peak memory without a profiler. It stayed at about 805 to 809 MB because the weights dominate for this small model.
**3. Example to quote:** > "Peak RSS stayed near 805 MB across all batch sizes, since distilgpt2's KV cache is tiny next to its weights."
**Seen in:** `main.py`, `peak_rss_mb()`

## sys.platform  `(standard library)`
**1. Definition:** `sys.platform` is a string naming the operating system, e.g. `"darwin"` for macOS or `"linux"`.
**2. In plain words:** The script uses it to pick the right unit for `ru_maxrss`.
**3. Example to quote:** > "`sys.platform == 'darwin'` tells me macOS reports bytes, so I divide by 1e6; on Linux I divide the kilobytes by 1e3."
**Seen in:** `main.py`

## min_new_tokens = max_new_tokens (fair comparison)  `(third-party: transformers)`
**1. Definition:** Setting `min_new_tokens` equal to `max_new_tokens` forces every sequence to produce exactly that many tokens.
**2. In plain words:** If one sequence stopped early, the timings wouldn't compare like with like.
**3. Example to quote:** > "I fix both to 40 so every batch does exactly 40 tokens of work per sequence."
**Seen in:** `main.py`, `run_batch`

## Warm-up and repeated measurement  `(concept)`
**1. Definition:** A warm-up run precedes timing, and conclusions should rest on repeated runs, not one.
**2. In plain words:** First calls include one-time costs, and on `mps` the numbers were erratic (batch 1 even faster than batch 2). Lesson: warm up, repeat, and distrust a single run.
**3. Example to quote:** > "The MPS timings were noisy, so I wouldn't draw conclusions from one run; the CPU table is my reference."
**Seen in:** `main.py` (`run_batch(1)` before the loop); `WALKTHROUGH.md`

## List repetition `["x"] * n`  `(syntax)`
**1. Definition:** Multiplying a list by an integer repeats its contents: `["a"] * 3` is `["a", "a", "a"]`.
**2. In plain words:** It builds a batch of identical prompts of any size in one expression.
**3. Example to quote:** > "`['Once upon a time...'] * batch_size` makes a batch of 16 identical prompts."
**Seen in:** `main.py`, `run_batch`

## csv module and writer.writerows  `(standard library)`
**1. Definition:** The `csv` module reads and writes comma-separated files; `csv.writer(f)` wraps a file, `writerow` writes one row and `writerows` writes many.
**2. In plain words:** It saves the benchmark table to `results.csv` so you can open it in a spreadsheet or keep it in your notes.
**3. Example to quote:** > "I save the batch results with `csv.writer(f).writerows(rows)` so I can compare runs later."
**Seen in:** `main.py`, end

## `with open(...)` and newline=""  `(syntax)`
**1. Definition:** `with open(path, "w", newline="") as f:` opens a file for writing and guarantees it is closed afterwards; `newline=""` prevents extra blank lines in CSV output on some systems.
**2. In plain words:** The `with` block closes the file even if an error occurs, so you don't leak file handles or lose data.
**3. Example to quote:** > "I always open files with `with` so they close automatically, even if the script crashes midway."
**Seen in:** `main.py`

## Tuple of batch sizes loop  `(syntax)`
**1. Definition:** `for bs in (1, 2, 4, 8, 16):` iterates over a tuple of values.
**2. In plain words:** It's a simple way to repeat the experiment at several sizes. Add 32 to see where throughput flattens.
**3. Example to quote:** > "I loop over `(1, 2, 4, 8, 16)` and append one row of results per batch size."
**Seen in:** `main.py`

## round()  `(Python built-in)`
**1. Definition:** `round(x, n)` rounds a number to `n` decimal places.
**2. In plain words:** It keeps the saved CSV values readable.
**3. Example to quote:** > "`round(seconds, 2)` turns 0.5712345 into 0.57 in the table."
```python
round(3.14159, 2)   # 3.14
```
**Seen in:** `main.py`

## Compute-bound vs memory-bound  `(concept)`
**1. Definition:** A workload is compute-bound when arithmetic speed limits it and memory-bound when data movement limits it.
**2. In plain words:** At small batches the hardware is underused; as batches grow throughput eventually flattens when it becomes compute-bound. That is why the curve levels off at larger sizes.
**3. Example to quote:** > "If I add batch sizes 32 and 64, I expect throughput to flatten once the hardware is saturated."
**Seen in:** `WALKTHROUGH.md`, Experiments

## Error: Torch not compiled with CUDA  `(concept)`
**1. Definition:** Selecting `--device cuda` on a machine without NVIDIA/CUDA support raises an error that PyTorch was not compiled with CUDA enabled (or an invalid-device assertion).
**2. In plain words:** Check availability first with `torch.cuda.is_available()` and `torch.backends.mps.is_available()`.
**3. Example to quote:** > "On my Mac `--device cuda` fails, so I check `torch.cuda.is_available()` and pick `mps` or `cpu` instead."
**Seen in:** `WALKTHROUGH.md`, Debugging practice
