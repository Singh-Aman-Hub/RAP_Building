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

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Latency vs throughput | response time per item vs items per second | No | Latency = how long one request waits; throughput = how many items finish per second. Batching trades the first for the second. |
| Batch inference | offline inference, bulk scoring, batch prediction | Yes | Run on a schedule over a whole table; nobody waits. |
| Real-time inference | online inference, synchronous serving | Yes | One request in, one answer out while the user waits. |
| Micro-batching | dynamic batching, near-real-time | Mostly | Servers collect requests for a few ms and score them as one batch; 'dynamic batching' is the Triton/vLLM name. |
| Generator vs list | lazy iterator vs materialised list | No | A generator (`yield`) produces items on demand; a list holds all of them in memory. |

## Code, step by step

### 1. Imports and borrowing step 01/02 code  (`main.py` lines 15–26)

```python
import csv
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_training_vs_inference_pipelines"))
from predict import load_artifact   # noqa: E402
from train import train             # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_pipeline_end_to_end"))
from data import get_training_data  # noqa: E402
```

- **In simple words:** Load the tools we need, and reuse the training code and data helper from earlier steps instead of copying them.
- **Exactly:** `sys.path.insert(0, ...)` puts the sibling step folders at the front of Python's import search list, so `from predict import load_artifact`, `from train import train` and `from data import get_training_data` resolve. `Path(__file__).resolve().parents[1]` is the module folder `10_ml_system_architecture`. `# noqa: E402` silences the linter's 'import not at top' warning, which is intentional here.
- **Why it is here:** Without the path edit the imports fail with `ModuleNotFoundError`; copying the code would let the three copies drift apart.

### 2. chunks(): a generator that yields slices  (`main.py` lines 29–33)

```python
def chunks(items: list, size: int):
    """Yield successive slices. `yield` makes this a GENERATOR: it produces one chunk at a time, so a huge
    file never has to sit in memory all at once."""
    for start in range(0, len(items), size):
        yield items[start:start + size]
```

- **In simple words:** Cut a long list into pieces of a fixed size and hand them out one at a time, like dealing cards.
- **Exactly:** `range(0, len(items), size)` gives starts 0, size, 2*size, ...; `items[start:start+size]` is a slice (the last one may be shorter). `yield` makes the function a generator: it returns an iterator and computes each slice only when asked. For 2000 items and size 500 it yields 4 lists of 500.
- **Why it is here:** Scoring a table of millions of rows in one go would exhaust memory; chunking bounds memory per call.

### 3. realtime(): one model call per text  (`main.py` lines 36–38)

```python
def realtime(model, texts):
    """What an API endpoint does: one request -> one prediction. Optimises LATENCY of each answer."""
    return [model.predict([t])[0] for t in texts]
```

- **In simple words:** This mimics an API endpoint: every request is answered on its own.
- **Exactly:** `model.predict([t])` takes a list of length 1 and returns an array of length 1; `[0]` takes that single label. The list comprehension therefore makes 2000 separate sklearn calls, each paying the full fixed overhead (input validation, TF-IDF transform setup, array building).
- **Why it is here:** It is the baseline: it shows what you pay when each item travels alone.

### 4. batch(): one model call per chunk  (`main.py` lines 41–46)

```python
def batch(model, texts, chunk_size=500):
    """What a nightly job does: score a whole table. Optimises THROUGHPUT (items per second)."""
    out = []
    for piece in chunks(texts, chunk_size):
        out.extend(model.predict(piece))            # one vectorized call per chunk
    return out
```

- **In simple words:** Score many texts in one call, 500 at a time, and collect the answers in one list.
- **Exactly:** `out.extend(...)` appends every element of the returned array to `out` (unlike `append`, which would add the array as one item). With `chunk_size=500` and 2000 texts there are 4 `predict` calls, each vectorized: the overhead is paid 4 times instead of 2000.
- **Why it is here:** This is the whole point of the step: same model, same answers, far higher throughput.

### 5. Train and load the model, create 2000 new texts  (`main.py` lines 49–52)

```python
if __name__ == "__main__":
    train()
    model, _ = load_artifact()
    texts, _ = get_training_data(n=2000, seed=1)    # pretend: 2000 new items to score
```

- **In simple words:** Make sure a trained model exists, load it, and invent 2000 items to score.
- **Exactly:** `train()` (step 02) fits a TF-IDF + logistic regression pipeline and saves it under `artifacts/`; `load_artifact()` returns `(model, metadata)`, the metadata is discarded with `_`. `get_training_data(n=2000, seed=1)` returns `(texts, labels)`: two lists of 2000 items; seed 1 differs from training so these are 'new' texts (labels ignored).
- **Why it is here:** The benchmark needs a real fitted model and enough items for timing to be meaningful.

### 6. Time both approaches  (`main.py` lines 54–55)

```python
    t0 = time.perf_counter(); rt = realtime(model, texts); rt_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter(); bt = batch(model, texts);    bt_ms = (time.perf_counter() - t0) * 1000
```

- **In simple words:** Start a stopwatch, run one approach, stop the watch, in milliseconds.
- **Exactly:** `time.perf_counter()` is a monotonic high-resolution clock (seconds, float); the difference times 1000 is ms. Two statements per line separated by `;`. In my run: real-time 929.0 ms, batch 11.3 ms (times vary by machine and run).
- **Why it is here:** `time.time()` can jump (clock adjustments); `perf_counter` is the right tool for measuring durations.

### 7. Print the comparison and assert equal answers  (`main.py` lines 56–59)

```python
    print(f"real-time: {len(rt)} predictions, one call each   -> {rt_ms:7.1f} ms total, {rt_ms/len(rt):.3f} ms per item")
    print(f"batch    : {len(bt)} predictions, chunks of 500   -> {bt_ms:7.1f} ms total, {bt_ms/len(bt):.3f} ms per item"
          f"   ({rt_ms/bt_ms:.0f}x faster)")
    assert rt == bt                                  # same model, same answers; only the calling pattern differs
```

- **In simple words:** Show total time and time per item for both, then check they gave identical predictions.
- **Exactly:** Real output: `real-time: 2000 predictions ... 929.0 ms total, 0.464 ms per item` and `batch ... 11.3 ms total, 0.006 ms per item (82x faster)`. `rt_ms/bt_ms` is the speed-up. `assert rt == bt` compares two lists of label strings element by element.
- **Why it is here:** It proves batching changes only the calling pattern, not the results; without the assert a faster-but-wrong path would go unnoticed.

### 8. Write the batch results to a CSV file  (`main.py` lines 61–66)

```python
    out_file = Path(tempfile.mkdtemp()) / "scored.csv"
    with open(out_file, "w", newline="") as f:       # newline="" is the csv module's required setting
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(zip(texts, bt))             # zip pairs each text with its label
    print("batch results written to", out_file)
```

- **In simple words:** Save every text next to its label in a temporary CSV, like a nightly job writing a result table.
- **Exactly:** `tempfile.mkdtemp()` creates a fresh directory; `newline=""` stops the csv module from doubling line endings on Windows; `writerow` writes the header; `zip(texts, bt)` pairs the i-th text with the i-th label, and `writerows` writes all 2000 pairs. The printed path was `.../tmpz06a5ld4/scored.csv`.
- **Why it is here:** Batch jobs exist to produce stored output; this shows the typical end of the pipeline.

### 9. Print the decision guide  (`main.py` lines 68–75)

```python
    print("""
WHEN TO USE WHICH
  real-time : user is waiting (chatbot, fraud check at checkout)   -> small p95 latency, always-on service, autoscale
  batch     : nobody is waiting (nightly recommendations, reports) -> big throughput, cheap scheduled job, retry-able
  near-real-time / micro-batch: collect requests for ~10-50 ms and score them together (what vLLM/Triton do for GPUs)
""")
    # TRY IT: change chunk_size to 10, 100, 2000 and plot/print how throughput changes.
    # TRY IT: use n=20000 and watch real-time grow linearly while batch stays small.
```

- **In simple words:** A cheat sheet of when to pick real-time, batch or micro-batch, followed by two exercises as comments.
- **Exactly:** A triple-quoted string printed verbatim. Micro-batching waits ~10-50 ms to collect requests and scores them together: a compromise of latency and throughput used by GPU servers such as vLLM and Triton. The final `# TRY IT:` lines are suggestions, not code.
- **Why it is here:** The measurement is only useful if it ends in a decision rule you can say in an interview.


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
