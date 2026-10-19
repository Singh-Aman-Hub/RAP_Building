# Terms — Step 06: Batch vs real-time inference

Folder: `10_ml_system_architecture/06_batch_vs_realtime` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [10 ML system architecture](../README.md)

**Terms in this folder:** [Real-time (online) inference](#real-time-online-inference-concept) · [Batch (offline) inference](#batch-offline-inference-concept) · [Latency vs throughput](#latency-vs-throughput-concept) · [Vectorization (why batching is faster)](#vectorization-why-batching-is-faster-concept) · [Micro-batching](#micro-batching-concept) · [Generator (yield)](#generator-yield-syntax) · [range(start, stop, step)](#rangestart-stop-step-python-built-in) · [List slicing items[start:start+size]](#list-slicing-itemsstartstartsize-syntax) · [list.extend](#listextend-python-built-in) · [model.predict with a list of texts](#modelpredict-with-a-list-of-texts-third-party-scikit-learn) · [List comprehension for per-item calls](#list-comprehension-for-per-item-calls-syntax) · [assert rt == bt](#assert-rt-bt-python-built-in) · [csv module (writer, writerow, writerows)](#csv-module-writer-writerow-writerows-standard-library) · [open(f, "w", newline="")](#openf-w-newline-python-built-in) · [zip(texts, labels)](#ziptexts-labels-python-built-in) · [Timing with perf_counter and semicolons](#timing-with-perf_counter-and-semicolons-standard-library) · [f-string alignment {rt_ms:7.1f}](#f-string-alignment-rt_ms71f-syntax) · [Hybrid: precompute and serve cached](#hybrid-precompute-and-serve-cached-concept) · [Batch failure handling](#batch-failure-handling-concept) · [Chunk size trade-off](#chunk-size-trade-off-concept)

---

## Real-time (online) inference  `(concept)`
**1. Definition:** Serving one prediction per incoming request, with a user or system waiting for the answer.
**2. In plain words:** It is what an API endpoint does. The goal is a small latency for every request, especially the slow tail (p95). It needs an always-on service that can scale with traffic.
**3. Example to quote:**
> "A fraud check at checkout is real-time: the customer is waiting, so I need a fast, always-on endpoint."

**Seen in:** `realtime` in `main.py`

## Batch (offline) inference  `(concept)`
**1. Definition:** Scoring many inputs together in a scheduled job, where nobody waits for individual answers.
**2. In plain words:** Nightly recommendations or reports are batch jobs. The goal is throughput and low cost, and a failed job can simply be re-run.
**3. Example to quote:**
> "I score the whole customer table every night in one batch job and store the results for the app to read."

**Seen in:** `batch` in `main.py`

## Latency vs throughput  `(concept)`
**1. Definition:** Latency is the time to complete one request; throughput is the number of items processed per unit of time.
**2. In plain words:** Real-time systems optimise latency, batch systems optimise throughput. Batching improves throughput but makes each individual item wait for its batch.
**3. Example to quote:**
> "Batch scoring took 0.003 ms per item against 0.249 ms per item one-at-a-time, but a single user waits longer for a batch window."

**Seen in:** whole step

## Vectorization (why batching is faster)  `(concept)`
**1. Definition:** Processing many items in one array-based library call, which amortises Python and library overhead across the batch and uses optimised native code.
**2. In plain words:** Calling `predict` 2000 times pays a fixed overhead 2000 times; calling it 4 times on 500 items pays it 4 times. Here batching was 91 times faster. GPUs benefit even more, which is why LLM servers batch requests.
**3. Example to quote:**
> "Real-time took 497.6 ms for 2000 items; batch in chunks of 500 took 5.5 ms for the same answers."

**Seen in:** `WALKTHROUGH.md`

## Micro-batching  `(concept)`
**1. Definition:** Collecting requests for a very short window (about 10 to 50 ms) and scoring them together.
**2. In plain words:** It combines the efficiency of batching with near-real-time responses. Inference servers such as vLLM and Triton do this for GPUs.
**3. Example to quote:**
> "My API could wait 20 ms to gather concurrent requests and run them as one batch, trading a little latency for much better throughput."

**Seen in:** `main.py` print, `WALKTHROUGH.md` table

## Generator (yield)  `(syntax)`
**1. Definition:** A function containing `yield` returns an iterator that produces one value at a time and pauses between them.
**2. In plain words:** `chunks` yields one slice at a time, so a huge file never needs to sit fully in memory. Use it in a `for` loop, and remember a generator can be consumed only once.
**3. Example to quote:**
> "`for piece in chunks(texts, 500)` gives me 500 texts at a time, so memory use stays small."

```python
def count(n):
    for i in range(n):
        yield i
print(list(count(3)))   # [0, 1, 2]
```

**Seen in:** `chunks`

## range(start, stop, step)  `(Python built-in)`
**1. Definition:** `range(0, n, size)` generates numbers from 0 up to but not including `n`, stepping by `size`.
**2. In plain words:** It produces the start index of each chunk: 0, 500, 1000, 1500 for 2000 items and a size of 500.
**3. Example to quote:**
> "`range(0, len(items), size)` gives 0, 500, 1000, 1500, the start of each chunk."

```python
print(list(range(0, 10, 4)))   # [0, 4, 8]
```

**Seen in:** `chunks`

## List slicing items[start:start+size]  `(syntax)`
**1. Definition:** A slice `items[a:b]` returns the elements from index `a` up to but not including `b`; slices past the end are simply shorter.
**2. In plain words:** It cuts one chunk out of the list. The last chunk may be smaller without needing special handling.
**3. Example to quote:**
> "`items[start:start + size]` returns the next 500 texts, or fewer at the end."

```python
print([1, 2, 3, 4][1:3])   # [2, 3]
```

**Seen in:** `chunks`

## list.extend  `(Python built-in)`
**1. Definition:** `out.extend(other)` appends every element of `other` to the list `out`.
**2. In plain words:** Unlike `append`, which adds one item (a whole list as a single element), `extend` flattens the results of each chunk into one result list.
**3. Example to quote:**
> "`out.extend(model.predict(piece))` adds 500 labels at a time to my results list."

```python
a = [1]; a.extend([2, 3]); print(a)   # [1, 2, 3]
```

**Seen in:** `batch`

## model.predict with a list of texts  `(third-party: scikit-learn)`
**1. Definition:** `model.predict(list_of_texts)` returns one label per text in a single vectorized call.
**2. In plain words:** Passing 500 texts at once is much faster than 500 calls with one text each; the answers are the same.
**3. Example to quote:**
> "Real-time does `model.predict([t])[0]` per text; batch does `model.predict(piece)` for a whole chunk."

**Seen in:** `realtime`, `batch`

## List comprehension for per-item calls  `(syntax)`
**1. Definition:** `[model.predict([t])[0] for t in texts]` builds a list by calling the model once per text.
**2. In plain words:** It models an API that gets one request at a time. It is clean code but slow, which is exactly the point of the comparison.
**3. Example to quote:**
> "The real-time function is a one-line comprehension with one model call per text, which is why it is 91 times slower."

**Seen in:** `realtime`

## assert rt == bt  `(Python built-in)`
**1. Definition:** `assert condition` raises `AssertionError` if the condition is false; here it checks that both approaches return identical predictions.
**2. In plain words:** It proves the speedup does not change the answers. A failing assert would signal that batching altered results, which can happen with padding or non-deterministic sampling in neural nets.
**3. Example to quote:**
> "`assert rt == bt` confirms the same model gives the same labels whether I call it one at a time or in chunks."

**Seen in:** `main.py`

## csv module (writer, writerow, writerows)  `(standard library)`
**1. Definition:** `csv.writer(f)` writes rows to a CSV file; `writerow` writes one row and `writerows` writes many.
**2. In plain words:** CSV is the simplest way to hand batch results to other tools. Open the file with `newline=""` so line endings are correct on every system.
**3. Example to quote:**
> "`writer.writerows(zip(texts, bt))` writes one `(text, label)` row for each scored item."

```python
import csv, sys
csv.writer(sys.stdout).writerow(["a", "b"])
```

**Seen in:** `main.py`

## open(f, "w", newline="")  `(Python built-in)`
**1. Definition:** Opens a file for writing in text mode; `newline=""` is the setting the `csv` module requires to avoid doubled line breaks.
**2. In plain words:** Without it you may get blank lines between rows on Windows. Use `with` so the file is closed automatically.
**3. Example to quote:**
> "`with open(out_file, 'w', newline='') as f:` is the right way to open a file for `csv.writer`."

**Seen in:** `main.py`

## zip(texts, labels)  `(Python built-in)`
**1. Definition:** `zip(a, b)` pairs items from two sequences into tuples, stopping at the shorter.
**2. In plain words:** It lines up each text with its predicted label to form CSV rows.
**3. Example to quote:**
> "`zip(texts, bt)` pairs each of the 2000 texts with its prediction."

```python
print(list(zip("ab", [1, 2])))   # [('a', 1), ('b', 2)]
```

**Seen in:** `main.py`

## Timing with perf_counter and semicolons  `(standard library)`
**1. Definition:** `t0 = time.perf_counter(); work(); ms = (time.perf_counter() - t0) * 1000` measures elapsed milliseconds; a semicolon separates statements on one line.
**2. In plain words:** The pattern compares the two approaches on the same machine. Semicolons are legal but uncommon; here they keep the timing code compact.
**3. Example to quote:**
> "I record `perf_counter()` before and after each approach and convert the difference to milliseconds."

**Seen in:** `main.py`

## f-string alignment {rt_ms:7.1f}  `(syntax)`
**1. Definition:** `:7.1f` formats a float to one decimal in a field 7 characters wide.
**2. In plain words:** It lines up the timing numbers in neat columns so the comparison is easy to read.
**3. Example to quote:**
> "`f'{rt_ms:7.1f} ms total'` prints 497.6 padded to the same width as 5.5."

```python
print(f"{5.5:7.1f}")
```

**Seen in:** `main.py`

## Hybrid: precompute and serve cached  `(concept)`
**1. Definition:** A design that precomputes expensive results in batch and serves the cached results in real time.
**2. In plain words:** For example, compute recommendations nightly and look them up per request in milliseconds. It combines the cost efficiency of batch with the responsiveness of an online service, at the price of staleness.
**3. Example to quote:**
> "Product recommendations are computed nightly in batch and the API only does a fast lookup per user."

**Seen in:** `WALKTHROUGH.md`

## Batch failure handling  `(concept)`
**1. Definition:** A failed batch job is re-run (ideally idempotently), whereas a failed real-time request needs an immediate retry or fallback.
**2. In plain words:** Batch tolerates delay but must not duplicate work. Real-time must degrade gracefully, for example by returning a default answer.
**3. Example to quote:**
> "If the nightly job fails I re-run it; if the API fails the client gets a retry or a safe fallback."

**Seen in:** `WALKTHROUGH.md` decision table

## Chunk size trade-off  `(concept)`
**1. Definition:** The size of each batch affects speed and memory: small chunks keep per-call overhead high, huge chunks use lots of memory and delay results.
**2. In plain words:** Gains flatten after some size. Experimenting with 10, 100, 500 and 2000 shows where. For huge files read in chunks rather than loading everything.
**3. Example to quote:**
> "I tried chunk sizes 10, 100 and 2000 to see where throughput stops improving."

**Seen in:** `TRY IT` comments, `WALKTHROUGH.md` experiments
