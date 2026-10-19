# Terms — Experiment Logger

Folder: `05_llm_parameters/07_experiment_logger` · Files covered: `main.py`, `WALKTHROUGH.md`, `results/experiments.csv`, `results/experiments.md`
Back to module: [05 LLM parameters README](../README.md)

**Terms in this folder:** [Experiment logging](#experiment-logging) · [One-parameter-at-a-time experiments](#one-parameter-at-a-time-experiments) · [csv module](#csv-module) · [csv.DictWriter](#csvdictwriter) · [resource and ru_maxrss](#resource-and-ru_maxrss) · [Peak RSS (high-water mark)](#peak-rss-high-water-mark) · [sys.platform](#sysplatform) · [statistics.median](#statisticsmedian) · [Repeats and median timing](#repeats-and-median-timing) · [Warm-up run](#warm-up-run) · [time.perf_counter](#timeperf_counter) · [Data-driven list of tuples](#data-driven-list-of-tuples) · [Tuple unpacking in a for loop](#tuple-unpacking-in-a-for-loop) · [Dict merge with {**a, **b}](#dict-merge-with-a-b) · [dict.pop(key, None)](#dictpopkey-none) · [Model cache dictionary](#model-cache-dictionary) · [Weights in MB](#weights-in-mb) · [Latency](#latency) · [Tokens per second](#tokens-per-second) · [Latency vs output length](#latency-vs-output-length) · [Distinct-token ratio](#distinct-token-ratio) · [Proxy metric](#proxy-metric) · [.tolist()](#tolist) · [Path.mkdir(exist_ok=True)](#pathmkdirexist_okTrue) · [str.join](#strjoin) · [Markdown table output](#markdown-table-output) · [Model size trade-off](#model-size-trade-off) · [StatisticsError](#statisticserror) · [Fail early with assert](#fail-early-with-assert)

---

## Experiment logging  `(concept)`
**1. Definition:** Recording the settings and measured results of each run in a structured table so runs can be compared.
**2. In plain words:** "It felt better" is not evidence. By logging output length, latency, memory and a quality signal you can answer "what happens if I raise temperature or use a bigger model?" with data.
**3. Example to quote:** > "I saved every run to `experiments.csv` and `experiments.md`, so I can show the numbers, not impressions."

**Seen in:** `main.py`, docstring.

## One-parameter-at-a-time experiments  `(concept)`
**1. Definition:** Changing exactly one setting between runs while keeping the prompt, seed and everything else fixed.
**2. In plain words:** If you change two things you cannot tell which caused the difference. Each row in `EXPERIMENTS` overrides a single value of the baseline.
**3. Example to quote:** > "Row 'temperature 0.3' differs from baseline only in temperature."

**Seen in:** `main.py`, `EXPERIMENTS`.

## csv module  `(standard library)`
**1. Definition:** The standard-library module for reading and writing comma-separated-values files.
**2. In plain words:** CSV opens in Excel or Numbers, making results easy to inspect and chart.
**3. Example to quote:** > "I wrote the results with the `csv` module so I can open them in a spreadsheet."

**Seen in:** `main.py`, imports.

## csv.DictWriter  `(standard library)`
**1. Definition:** Writes a list of dictionaries as CSV rows using `fieldnames` as columns.
**2. In plain words:** `writeheader()` writes the column names and `writerows(rows)` writes every row. Open the file with `newline=""` to avoid blank lines on Windows.
**3. Example to quote:** > "`csv.DictWriter(f, fieldnames=rows[0].keys())` turns my list of dicts into a table."
```python
writer = csv.DictWriter(f, fieldnames=rows[0].keys())
writer.writeheader(); writer.writerows(rows)
```
**Seen in:** `main.py`, saving section.

## resource and ru_maxrss  `(standard library)`
**1. Definition:** `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` returns the process's peak resident memory (Unix only).
**2. In plain words:** It tells how much RAM the program has needed at most. macOS reports bytes and Linux reports kilobytes, so the code converts.
**3. Example to quote:** > "Peak memory rose from about 865 MB to 1,390 MB after loading gpt2."

**Seen in:** `main.py`, `rss_mb`.

## Peak RSS (high-water mark)  `(concept)`
**1. Definition:** Resident set size is the RAM a process holds; its peak only ever goes up during the process.
**2. In plain words:** You cannot tell when a model released memory. Compare differences and, for a clean per-model figure, run each model in its own process.
**3. Example to quote:** > "`peak_rss_MB` is a high-water mark, so I read differences between rows."

**Seen in:** `main.py`, final print.

## sys.platform  `(standard library)`
**1. Definition:** A string naming the operating system, such as `"darwin"` for macOS.
**2. In plain words:** Used to handle the macOS-versus-Linux difference in memory units.
**3. Example to quote:** > "`sys.platform == 'darwin'` tells me to divide by 1e6 instead of 1e3."

**Seen in:** `main.py`, `rss_mb`.

## statistics.median  `(standard library)`
**1. Definition:** Returns the middle value of numeric data; it raises `StatisticsError` on empty data.
**2. In plain words:** The median is robust to a single slow outlier run, unlike the mean.
**3. Example to quote:** > "I take the median of 3 timings so one slow run does not mislead."

**Seen in:** `main.py`.

## Repeats and median timing  `(concept)`
**1. Definition:** Running the same measurement several times and using the median to reduce timing noise.
**2. In plain words:** Single timings vary with machine load. Three repeats with a median give a steadier number.
**3. Example to quote:** > "`REPEATS = 3` and `statistics.median(times)` give a more reliable latency."

**Seen in:** `main.py`.

## Warm-up run  `(concept)`
**1. Definition:** An initial throwaway call that triggers lazy initialisation and caching before measurement.
**2. In plain words:** The first call is always slower. Without a warm-up it would distort the first row of results.
**3. Example to quote:** > "I generate 5 tokens from 'warm up' once so timings after that are fair."

**Seen in:** `main.py`, model-loading block.

## time.perf_counter  `(standard library)`
**1. Definition:** A high-resolution stopwatch in seconds; subtract two readings.
**2. In plain words:** It measures each generation call.
**3. Example to quote:** > "I wrap each `generate` call between two `perf_counter()` readings."

**Seen in:** `main.py`.

## Data-driven list of tuples  `(syntax)`
**1. Definition:** Describing experiments as data, a list of `(name, model, overrides)` tuples, instead of repeating code.
**2. In plain words:** To add an experiment you add a row, not code.
**3. Example to quote:** > "I add `('top_p 0.5', 'distilgpt2', {'top_p': 0.5})` and re-run."

**Seen in:** `main.py`, `EXPERIMENTS`.

## Tuple unpacking in a for loop  `(syntax)`
**1. Definition:** `for name, model_name, overrides in EXPERIMENTS:` assigns each tuple's three parts to three names.
**2. In plain words:** It is cleaner than indexing `exp[0]`, `exp[1]`.
**3. Example to quote:** > "Each loop iteration gives me the experiment name, the model and the overrides."

**Seen in:** `main.py`.

## Dict merge with {**a, **b}  `(syntax)`
**1. Definition:** Builds a new dict from `a` and `b`, with `b`'s values winning for duplicate keys.
**2. In plain words:** Start with config defaults (`BASE`) and apply the experiment's overrides on top.
**3. Example to quote:** > "`{**BASE, **overrides}` gives the baseline settings with one value changed."
```python
{**{"x": 1}, **{"x": 2}}  # {'x': 2}
```
**Seen in:** `main.py`.

## dict.pop(key, None)  `(Python built-in)`
**1. Definition:** Removes a key and returns its value; the second argument is the default so no error is raised if the key is missing.
**2. In plain words:** Greedy decoding ignores temperature, top-k and top-p, so the code removes them to avoid library warnings.
**3. Example to quote:** > "`kwargs.pop('temperature', None)` drops it only if present."

**Seen in:** `main.py`.

## Model cache dictionary  `(concept)`
**1. Definition:** A dict mapping a model name to its loaded `(tokenizer, model, size)` so each model loads once.
**2. In plain words:** Loading takes seconds. Cache it, as a service would load a model once at startup (module 09).
**3. Example to quote:** > "`if model_name not in loaded:` ensures gpt2 is loaded only once."

**Seen in:** `main.py`.

## Weights in MB  `(concept)`
**1. Definition:** Model weight memory computed as the sum of `numel() * element_size()` over parameters, divided by 1e6.
**2. In plain words:** distilgpt2 is 328 MB and gpt2 is 498 MB, matching the parameters-times-bytes formula from module 04.
**3. Example to quote:** > "gpt2's weights are 52% bigger than distilgpt2's."

**Seen in:** `main.py`.

## Latency  `(concept)`
**1. Definition:** The time taken to complete one request, here one `generate` call.
**2. In plain words:** It is what a user waits for. It grows with output length.
**3. Example to quote:** > "Baseline latency was 0.73 s for 40 tokens."

**Seen in:** `main.py`, `latency_s`.

## Tokens per second  `(concept)`
**1. Definition:** Generated tokens divided by latency, a throughput measure of generation speed.
**2. In plain words:** It stays roughly constant for a model on a machine: cost per token is steady. gpt2 was slower per token than distilgpt2.
**3. Example to quote:** > "distilgpt2 ran about 55 to 77 tokens per second versus gpt2's 40."

**Seen in:** `main.py`.

## Latency vs output length  `(concept)`
**1. Definition:** Total latency grows roughly linearly with the number of generated tokens.
**2. In plain words:** 10 tokens took 0.18 s while 83 tokens took 1.44 s, so shorter outputs are faster and cheaper.
**3. Example to quote:** > "Latency scales with length but tokens per second stays about constant."

**Seen in:** `WALKTHROUGH.md`.

## Distinct-token ratio  `(concept)`
**1. Definition:** Unique generated tokens divided by total generated tokens; near 1 means varied, low means repetitive.
**2. In plain words:** A cheap repetition detector. Greedy with repetition penalty 1.0 scored 0.12, a loop.
**3. Example to quote:** > "A ratio of 0.12 told me the model was stuck repeating."
```python
len(set(token_ids)) / max(len(token_ids), 1)
```
**Seen in:** `main.py`, `distinct_ratio`.

## Proxy metric  `(concept)`
**1. Definition:** A cheap, indirect number standing in for something hard to measure, such as distinct ratio for quality.
**2. In plain words:** It can mislead, so you still read samples (the `my_notes` column) or use a labelled test set (module 07).
**3. Example to quote:** > "Distinct ratio is a crude proxy; the real judgement is reading the text."

**Seen in:** `WALKTHROUGH.md`.

## .tolist()  `(third-party: torch)`
**1. Definition:** Converts a tensor to a plain Python list.
**2. In plain words:** Used to slice off the new token IDs and compute with `set`.
**3. Example to quote:** > "`last_out[0][ids.shape[1]:].tolist()` gives only the generated token IDs."

**Seen in:** `main.py`.

## Path.mkdir(exist_ok=True)  `(standard library)`
**1. Definition:** Creates a directory; `exist_ok=True` suppresses the error if it already exists.
**2. In plain words:** The script can be re-run safely.
**3. Example to quote:** > "`out_dir.mkdir(exist_ok=True)` creates `results/` only if missing."

**Seen in:** `main.py`.

## str.join  `(Python built-in)`
**1. Definition:** `sep.join(list_of_strings)` concatenates strings with the separator between them.
**2. In plain words:** Used to build Markdown table rows like `| a | b |`.
**3. Example to quote:** > "`' | '.join(cols)` builds the header row."
```python
"-".join(["a", "b"])  # 'a-b'
```
**Seen in:** `main.py`, Markdown table code.

## Markdown table output  `(concept)`
**1. Definition:** A table written in Markdown syntax with pipes and a `---` separator row.
**2. In plain words:** It renders nicely in editors and on GitHub, so results are readable without a spreadsheet.
**3. Example to quote:** > "I write the same results as CSV for spreadsheets and Markdown for my notes."

**Seen in:** `main.py`.

## Model size trade-off  `(concept)`
**1. Definition:** Bigger models usually give better quality but need more memory and run slower.
**2. In plain words:** Moving from distilgpt2 to gpt2 added 52% weights and made it about 35 to 45% slower with higher peak memory.
**3. Example to quote:** > "gpt2 costs more memory and latency than distilgpt2, and usually buys better quality."

**Seen in:** `WALKTHROUGH.md`.

## StatisticsError  `(concept)`
**1. Definition:** Exception raised by the `statistics` module on invalid input such as an empty list for `median`.
**2. In plain words:** With `REPEATS = 0` the timing list is empty. The error shows in `statistics`, but the cause is the constant defined earlier.
**3. Example to quote:** > "`no median for empty data` told me my repeat count was zero."

**Seen in:** `main.py`, BREAK IT comment.

## Fail early with assert  `(syntax)`
**1. Definition:** Placing an `assert` near a configuration value so invalid settings fail immediately with a clear message.
**2. In plain words:** `assert REPEATS >= 1, "need at least one repeat"` moves the failure next to its cause.
**3. Example to quote:** > "I added an assert beside the constant so the error is obvious."

**Seen in:** `WALKTHROUGH.md`, debugging section.
