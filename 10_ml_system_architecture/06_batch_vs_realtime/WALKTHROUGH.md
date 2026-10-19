# Step 06 — Batch vs real-time inference

## The story so far
Step 05 logged predictions one request at a time. But nightly scoring of a million rows should not make a million separate calls.

## The pain point
Scoring items one by one pays fixed overhead per call, so a big job runs far slower than it needs to. Treating a user-facing request like a batch job makes the user wait.

## The spark
Pick by who is waiting: real-time when a person waits, batch when no one does, micro-batching in between. One vectorized call over many items amortises the overhead.

## Goal
Run the same model two ways and understand which metric each optimises: **latency** (real-time) vs **throughput** (batch).

## Run it
```bash
cd 10_ml_system_architecture/06_batch_vs_realtime && python main.py
```
Real output:
```
real-time: 2000 predictions, one call each   ->   497.6 ms total, 0.249 ms per item
batch    : 2000 predictions, chunks of 500   ->     5.5 ms total, 0.003 ms per item   (91x faster)
batch results written to /var/folders/.../scored.csv
```
Same model, same answers (the script asserts equality). The 91× comes from **vectorization**: one call amortizes Python and library overhead across 500 items; the same effect (even bigger) makes GPUs love batches.

## Decision table
| | Real-time (online) | Batch (offline) | Micro-batch |
|---|---|---|---|
| Trigger | each user request | schedule (nightly) | a few ms of collected requests |
| Optimises | p95 latency | throughput, cost | both |
| Infra | always-on API (module 09) | scheduled job, retry-able | inference servers (vLLM, Triton) |
| Failure handling | retry/fallback immediately | re-run the job | per-batch retry |
| Example | chatbot, fraud check | nightly recommendations, report scoring | LLM serving with continuous batching |
Typical hybrid: precompute (batch) the expensive parts, serve cached results in real time.

## Syntax explained
| Syntax | Meaning | Example |
|---|---|---|
| `def chunks(...): ... yield piece` | **Generator**: produces values one at a time, lazily | `for c in chunks(items, 500)` |
| `range(0, n, size)` | start, stop, step | `range(0, 10, 4)` → 0,4,8 |
| `items[start:start+size]` | Slice | `[1,2,3,4][1:3]` → `[2,3]` |
| `out.extend(list)` | Add all items of another list | — |
| `assert rt == bt` | Crash loudly if the two results differ | — |
| `open(f, "w", newline="")` | Required by the `csv` module for correct line endings | — |
| `zip(texts, labels)` | Pair items | — |

## Libraries & functions
| Name | Kind | Does |
|---|---|---|
| `model.predict(list_of_texts)` | sklearn | Predict labels for many texts in one call |
| `csv.writer(f).writerow / writerows` | stdlib | Write CSV rows |
| `time.perf_counter()` | stdlib | Timer |

## Execution trace
`batch(model, texts, 500)`: `chunks` yields texts[0:500], [500:1000], ... → each chunk → one `predict` → results appended → list of 2000 labels → `zip` with texts → CSV.

## Experiments
1. `chunk_size` 10 / 100 / 2000 — find where gains flatten.
2. `n=20000`: real-time grows linearly; what happens to the batch time?
3. For a large file, replace the list with reading the CSV in chunks (never load 10 GB into RAM).

## Debugging practice
`assert rt == bt` failing means batching changed answers — rare in sklearn, but real with padding/truncation in neural nets or non-deterministic sampling. Compare the first differing element to find the cause.

## Rebuild checklist
- [ ] Write a generator that yields chunks.
- [ ] State which metric each mode optimises and give one example of each.
- [ ] Explain micro-batching.

## The new problem (cliffhanger)
Whichever mode we pick, once it runs unattended we have to know whether it is healthy: slow, failing, or seeing strange inputs. Next: [07_logging_and_monitoring](../07_logging_and_monitoring/).

## Say it aloud
> **Say it aloud (interview box):** Real-time inference answers one request at a time and optimises latency, while batch inference scores many items together and optimises throughput and cost. Batching is faster because one vectorized call amortises overhead, which is also why LLM servers batch concurrent requests. The cost is waiting time per item, so interactive use needs small batch windows.
