# Step 07 - Experiment logger (length, latency, memory, quality)

## The story so far
[Step 06](../06_training_parameters/WALKTHROUGH.md) taught you the dials, and by now you have a head full of runs. Which setting gave which result?

## The pain point
Without a record, tuning is superstition. You remember the output you liked, not the latency, the memory or the seed that produced it, and a teammate cannot repeat your result.

## The spark
Run each setting, measure output length, latency and memory, add a simple quality signal, and write every run as a row in a CSV and a Markdown table. Evidence beats impressions.

## Goal
Run several settings, measure output length / latency / memory, add a quality signal and save the results as CSV + Markdown - evidence instead of impressions.

## Run it
```bash
.venv/bin/python 05_llm_parameters/07_experiment_logger/main.py
```
(~1-2 min; first run downloads `gpt2`, ~500 MB.) Output files: `07_experiment_logger/results/experiments.csv` and `experiments.md`.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| list of tuples `EXPERIMENTS = [(name, model, overrides), ...]` | data-driven experiments: add a row, not code | - |
| `for name, model_name, overrides in EXPERIMENTS:` | unpack each tuple | - |
| `{**BASE, **overrides}` | merge dicts, right side wins | - |
| `kwargs.pop(k, None)` | remove key if present, no error otherwise | - |
| `sum(... for p in model.parameters())` | generator expression | - |
| `" | ".join(...)` | build a string from a list | `"-".join(["a","b"])` |
| `rows[0].keys()` | field names for the CSV header | - |
| `if sys.platform == "darwin"` | OS check (macOS reports memory in bytes, Linux in KB) | - |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `csv.DictWriter` | stdlib | write a list of dicts as CSV | `fieldnames` | writer |
| `resource.getrusage(RUSAGE_SELF).ru_maxrss` | stdlib (Unix) | peak memory of this process | - | int |
| `statistics.median` | stdlib | robust average of repeated timings | - | float |
| `time.perf_counter` | stdlib | stopwatch | - | float |
| `Path.mkdir(exist_ok=True)` | stdlib | create folder if missing | `exist_ok` | - |
| `model.generate(...)` | transformers | the thing being measured | see step 03 | ids |

## Execution trace
```
| experiment | new_tokens | latency_s | tokens_per_s | weights_MB | peak_rss_MB | distinct_ratio | auto_note |
|---|---|---|---|---|---|---|---|
| baseline (config.json) | 40 | 0.85 | 46.9 | 328 | 839 | 0.93 | varied |
| greedy | 40 | 0.62 | 64.4 | 328 | 839 | 0.93 | varied |
| temperature 0.3 | 18 | 0.79 | 22.8 | 328 | 839 | 1.0 | stopped early (EOS) |
| temperature 1.5 | 40 | 0.74 | 53.8 | 328 | 839 | 1.0 | varied |
| top_k 5 | 40 | 0.75 | 53.3 | 328 | 839 | 0.97 | varied |
| repetition_penalty 1.0 | 40 | 0.75 | 53.3 | 328 | 839 | 0.12 | repetitive |
| max_new_tokens 10 | 10 | 0.21 | 46.9 | 328 | 839 | 1.0 | varied |
| max_new_tokens 100 | 83 | 1.7 | 48.9 | 328 | 839 | 0.92 | stopped early (EOS) |
| bigger model: gpt2 | 40 | 1.38 | 29.0 | 498 | 997 | 1.0 | varied |
```
(Your numbers will differ - machine, load, randomness.)
How to read it:
- **Latency scales with output length** (10 tokens 0.21 s, 83 tokens 1.7 s) but **tokens/s stays about constant**: cost per token is steady, so shorter outputs = faster, cheaper responses.
- **Sampling parameters barely change speed**; they change *quality*. `repetition_penalty 1.0` + greedy has `distinct_ratio 0.12`: a loop. The auto-note flags it; you still READ the sample text and fill `my_notes`.
- **Bigger model (gpt2)**: +52% weights (328 -> 498 MB), ~60% slower here; peak memory jumps. Quality is usually better - the classic size/latency/quality trade-off.
- **`peak_rss_MB` is a high-water mark**: it only goes up during the process, so compare *differences*, and for a clean per-model number run each model in its own process.
- Timing noise: we do a warm-up call and take the median of 3 runs - single timings mislead.
- A **proxy metric** (`distinct_ratio`) is cheap but crude; the real quality judgement is reading samples or using a labelled test set (module 07, prompt testing).

## Experiments
- Add `("top_p 0.5", "distilgpt2", {"top_p": 0.5})` to `EXPERIMENTS`.
- Add a second prompt and average the results. Why is one prompt a weak test?
- Run with `REPEATS = 10` and compare how much the medians move.

## Debugging practice
`REPEATS = 0` -> `statistics.median` raises `StatisticsError: no median for empty data`. Trace: the error arises in `statistics`, but the cause is your constant two screens earlier. Add `assert REPEATS >= 1, "need at least one repeat"` near the constant - fail early, near the cause.

## Rebuild checklist
- [ ] Time a function with `perf_counter` and take a median of 3 runs.
- [ ] Write rows (list of dicts) to CSV with `csv.DictWriter`.
- [ ] Explain why only one parameter changes per row.
- [ ] Explain what peak RSS does and does not tell you.

## The new problem (cliffhanger)
Module 05 gave you the dials and the gauges. The next question is a hard one: will a full fine-tune of a real model even fit on our GPU? Do the arithmetic in [module 06, step 08, GPU memory calculator](../../06_llm_training_and_fast_inference/08_gpu_memory_calculator/WALKTHROUGH.md).

> **Say it aloud (interview box):** An experiment logger runs each parameter setting and records length, latency, memory and a quality score into a CSV and a Markdown table. It turns opinions about settings into evidence that someone else can reproduce. I change one thing at a time, fix the seed and test on more than one prompt.
