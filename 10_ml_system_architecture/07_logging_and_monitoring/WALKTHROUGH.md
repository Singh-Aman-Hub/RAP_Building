# Step 07 — Logging, metrics and drift monitoring

## The story so far
Step 06 chose how the model runs. Now it runs unattended, and nobody is reading its output.

## The pain point
A silent model can fail in three ways: it crashes, it gets slow for a few users, or its inputs change so its answers quietly degrade. Averages and print statements show none of this.

## The spark
Emit one structured JSON log line per request, count errors, track latency percentiles instead of averages, and compare live inputs to the training reference.

## Goal
Know what to observe in a deployed model: **logs** (what happened), **metrics** (how often/how fast), **drift** (is the world changing so the model gets worse?).

## Run it
```bash
cd 10_ml_system_architecture/07_logging_and_monitoring && python main.py
```
Real output (shortened):
```
--- structured logs ---
{"level": "INFO", "event": "prediction", "request_id": "aeb9d3bc", "label": "negative", "confidence": 0.782, "latency_ms": 0.372, "text_len": 18}
...
{"level": "WARNING", "event": "bad_input", "request_id": "29f4a92c", "error": "text must be a non-empty string"}

counters: {'requests': 60, 'label_negative': 39, 'label_positive': 21, 'errors': 1}

--- latency ---
latency ms  p50=0.252  p95=0.413  p99=0.583  max=0.655  (average hides the slow tail!)
    0.229-  0.314 ms | ################################         48
    ...
--- drift ---
drift check [normal traffic]: PSI(text length) = 0.027 -> STABLE
drift check [users now write long reviews]: PSI(text length) = 7.827 -> DRIFT
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Logs vs metrics vs traces | the three pillars of observability | No | Logs = events, metrics = aggregated numbers, traces = a request's path across services (not in this step). |
| Data drift vs concept drift vs prediction drift | covariate shift / label shift / concept shift | No | Data drift: inputs change; prediction drift: output mix changes; concept drift: input-to-label relationship changes. |
| PSI vs KS | Population Stability Index vs Kolmogorov-Smirnov | No | PSI is a binned divergence score with rules of thumb; KS is the largest CDF gap with a significance test (step 09). |
| p95 / p99 latency | tail latency, percentile latency | Yes | The latency below which 95% / 99% of requests complete. |
| Structured logging | JSON logging, key-value logging | Yes | Each log line has named fields instead of free text. |
| Monitoring vs observability | telemetry | Mostly | Monitoring watches known signals; observability lets you ask new questions from the data you emit. |

## Code, step by step

### 1. Imports and borrowed step 01/02 code  (`main.py` lines 13–28)

```python
import json
import logging                       # built-in: the standard logging framework (levels, handlers, formatters)
import sys
import time
import uuid
from collections import Counter     # built-in dict subclass for counting
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_training_vs_inference_pipelines"))
from predict import load_artifact, predict_text   # noqa: E402
from train import train                           # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_pipeline_end_to_end"))
from data import get_training_data  # noqa: E402
```

- **In simple words:** Bring in the standard-library tools (JSON, logging, ids, counting) plus numpy, and reuse earlier steps' model code.
- **Exactly:** `logging` is Python's built-in framework (levels, handlers, formatters); `uuid` makes random ids; `Counter` is a dict that counts. `sys.path.insert` makes `predict`, `train` and `data` from steps 01/02 importable.
- **Why it is here:** Reusing the real predict code means we monitor the actual service logic, not a toy.

### 2. JsonFormatter: one JSON object per log line  (`main.py` lines 31–38)

```python
# ----------------------------------- 1. structured logs ----------------------------------
class JsonFormatter(logging.Formatter):
    """Format each log record as ONE line of JSON. Machines (Datadog, ELK, CloudWatch) can then filter/aggregate by field."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {"level": record.levelname, "event": record.getMessage()}
        entry.update(getattr(record, "fields", {}))      # extra key/values passed via logger.info(..., extra={"fields": {...}})
        return json.dumps(entry)
```

- **In simple words:** Teach the logger to print each event as a single line of JSON, so machines can read it.
- **Exactly:** A subclass of `logging.Formatter` overriding `format(record)`. It starts a dict with `level` (e.g. `INFO`) and `event` (`record.getMessage()`, the text passed to `logger.info`), merges extra fields from `record.fields`, and returns `json.dumps(entry)`. Real line: `{"level": "INFO", "event": "prediction", "request_id": "b473ea35", "label": "negative", "confidence": 0.782, "latency_ms": 0.639, "text_len": 18}`.
- **Why it is here:** Plain-text logs need fragile regexes; JSON lines can be filtered by field in ELK, Datadog or CloudWatch.

### 3. Create the logger, handler and level  (`main.py` lines 41–45)

```python
logger = logging.getLogger("predict-service")
handler = logging.StreamHandler(sys.stdout)              # send logs to the terminal (containers log to stdout; see module 11)
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)                            # DEBUG < INFO < WARNING < ERROR < CRITICAL
```

- **In simple words:** Set up a named logger that prints JSON to the terminal and shows INFO and above.
- **Exactly:** `getLogger("predict-service")` returns a singleton by name; `StreamHandler(sys.stdout)` writes to stdout (what containers expect); `setFormatter` attaches our formatter; `addHandler` connects them; `setLevel(INFO)` drops DEBUG. Levels are numbers: DEBUG 10 < INFO 20 < WARNING 30 < ERROR 40 < CRITICAL 50.
- **Why it is here:** Without a handler only WARNING and above would reach stderr in the default format; without the level, INFO lines would be dropped.

### 4. Metric storage: latency list and counters  (`main.py` lines 48–50)

```python
# ----------------------------------- 2. metrics -------------------------------------------
LATENCIES: list[float] = []
COUNTERS: Counter = Counter()
```

- **In simple words:** Two global notebooks: one for every request's time, one for tallies.
- **Exactly:** `LATENCIES: list[float]` of milliseconds; `COUNTERS: Counter` where a missing key reads as 0, so `COUNTERS["errors"] += 1` works without initialising. Final counters in the run: `{'requests': 60, 'label_negative': 39, 'label_positive': 21, 'errors': 1}`.
- **Why it is here:** Metrics are numbers you aggregate; logs are single events. You need both.

### 5. monitored_predict: the observability wrapper  (`main.py` lines 53–70)

```python
def monitored_predict(model, text: str) -> dict | None:
    """Wrap prediction with a request id, timing, counters and logs -- the 'observability wrapper'."""
    request_id = uuid.uuid4().hex[:8]                    # lets you find ALL log lines for one request
    start = time.perf_counter()
    try:
        result = predict_text(model, text, threshold=0.0)
    except ValueError as err:
        COUNTERS["errors"] += 1
        logger.warning("bad_input", extra={"fields": {"request_id": request_id, "error": str(err)}})
        return None
    ms = (time.perf_counter() - start) * 1000
    LATENCIES.append(ms)
    COUNTERS["requests"] += 1
    COUNTERS[f"label_{result['label']}"] += 1
    logger.info("prediction", extra={"fields": {"request_id": request_id, "label": result["label"],
                                                "confidence": result["confidence"], "latency_ms": round(ms, 3),
                                                "text_len": len(text)}})        # log the LENGTH, not private text
    return result
```

- **In simple words:** Wrap each prediction with an id, a stopwatch, counters and a log line, and handle bad input without crashing.
- **Exactly:** `uuid4().hex[:8]` is 8 hex characters like `b473ea35`. `predict_text(..., threshold=0.0)` returns `{label, confidence}` or raises `ValueError` on empty text; on error we count it, log at WARNING and return `None`. On success: elapsed ms appended, `requests` and `label_<label>` counters incremented, one INFO line logged with `extra={"fields": {...}}` (that is how the formatter receives it). Only `len(text)` is logged, not the text (privacy).
- **Why it is here:** The request id ties all lines of one request together; timing and counters give the metrics; without the try/except one bad request would take the service down.

### 6. latency_report: percentiles and a text histogram  (`main.py` lines 73–79)

```python
def latency_report(latencies: list[float]) -> None:
    arr = np.array(latencies)
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])     # percentile: value below which q% of requests fall
    print(f"latency ms  p50={p50:.3f}  p95={p95:.3f}  p99={p99:.3f}  max={arr.max():.3f}  (average hides the slow tail!)")
    counts, edges = np.histogram(arr, bins=5)
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        print(f"  {lo:7.3f}-{hi:7.3f} ms | {'#' * int(40 * c / len(arr)):<40} {c}")
```

- **In simple words:** Summarise the latencies as p50/p95/p99 and draw a little bar chart.
- **Exactly:** `np.percentile(arr, [50,95,99])` returns the values below which 50/95/99% of requests fall. `np.histogram(arr, bins=5)` returns 5 counts and 6 edges. Each bar is `'#' * int(40*c/len(arr))`. Real run: p50=0.385 p95=0.714 p99=0.825 max=0.935 ms; first bin 42 of 60 requests.
- **Why it is here:** An average hides the slow tail that real users feel; percentiles expose it.

### 7. psi(): Population Stability Index  (`main.py` lines 82–90)

```python
# ----------------------------------- 3. drift ---------------------------------------------
def psi(reference: np.ndarray, live: np.ndarray, bins: int = 5) -> float:
    """Population Stability Index: how different two distributions are.
    Rule of thumb: < 0.1 stable, 0.1-0.25 watch, > 0.25 significant drift."""
    edges = np.histogram_bin_edges(reference, bins=bins)             # bin edges from the REFERENCE data
    ref_pct = np.histogram(reference, bins=edges)[0] / len(reference) + 1e-4     # +tiny avoids log(0)
    live_clipped = np.clip(live, edges[0], edges[-1])                # values outside the range go in the end bins
    live_pct = np.histogram(live_clipped, bins=edges)[0] / len(live) + 1e-4
    return float(np.sum((live_pct - ref_pct) * np.log(live_pct / ref_pct)))
```

- **In simple words:** Score how different the live data looks from the reference data with one number.
- **Exactly:** Bin edges come from the reference (`np.histogram_bin_edges`, 5 equal-width bins). `ref_pct`, `live_pct` are bin proportions, +1e-4 to avoid log(0). Live values outside the reference range are clipped into the end bins. Formula: PSI = sum((live% - ref%) * ln(live% / ref%)). Rule of thumb: <0.1 stable, 0.1-0.25 watch, >0.25 drift.
- **Why it is here:** Raw 'the mean moved' is hard to threshold; PSI gives one comparable score and an alert rule. Edges from the live data would hide the shift.

### 8. drift_report: apply PSI to text length  (`main.py` lines 93–98)

```python
def drift_report(name: str, ref_texts: list[str], live_texts: list[str]) -> None:
    ref_len = np.array([len(t.split()) for t in ref_texts])          # feature to watch: words per text
    live_len = np.array([len(t.split()) for t in live_texts])
    score = psi(ref_len, live_len)
    verdict = "STABLE" if score < 0.1 else "WATCH" if score < 0.25 else "DRIFT"
    print(f"drift check [{name}]: PSI(text length) = {score:.3f} -> {verdict}")
```

- **In simple words:** Count words per text for both groups and print a verdict.
- **Exactly:** `len(t.split())` is the number of whitespace-separated words, giving a 1-D array per group. Verdict thresholds: <0.1 STABLE, <0.25 WATCH, else DRIFT. Real: normal traffic PSI=0.027 STABLE.
- **Why it is here:** Text length is a cheap input feature that changes when users change behaviour.

### 9. Main: train, load, create the reference set  (`main.py` lines 101–104)

```python
if __name__ == "__main__":
    train()
    model, _ = load_artifact()
    reference_texts, _ = get_training_data(n=300, seed=0)            # what the model saw in training
```

- **In simple words:** Make the model, load it, and take 300 training-like texts as the 'what we expected' baseline.
- **Exactly:** `get_training_data(n=300, seed=0)` returns `(texts, labels)`; labels dropped. The reference is what the model saw at launch.
- **Why it is here:** Drift only exists relative to a reference.

### 10. Log a few predictions, then raise the level to WARNING  (`main.py` lines 106–114)

```python
    print("--- structured logs ---")
    normal_traffic, _ = get_training_data(n=60, seed=5)
    for t in normal_traffic[:3]:
        monitored_predict(model, t)
    logger.setLevel(logging.WARNING)                                 # level filter: from now on INFO lines are dropped
    monitored_predict(model, "   ")                                  # invalid input -> WARNING still shown
    for t in normal_traffic[3:]:
        monitored_predict(model, t)                                  # still counted + timed, just not logged
    print("\ncounters:", dict(COUNTERS))
```

- **In simple words:** Show 3 INFO log lines, then silence INFO and still feed the rest of the traffic through.
- **Exactly:** 60 normal requests (seed 5). The first 3 print JSON lines. After `setLevel(WARNING)` INFO is dropped, so the blank-text call `"   "` logs `{"level": "WARNING", "event": "bad_input", "request_id": "aa1f7737", "error": "text must be a non-empty string"}`; the remaining 57 requests are still timed and counted. Result: `requests` 60, `errors` 1.
- **Why it is here:** Shows that metrics are independent of log verbosity: log less in production, keep counting.

### 11. Print the latency report  (`main.py` lines 115–116)

```python
    print("\n--- latency ---")
    latency_report(LATENCIES)
```

- **In simple words:** Show the latency summary and histogram.
- **Exactly:** Passes the 60 recorded latencies to `latency_report`; real output is `latency ms  p50=0.385  p95=0.714  p99=0.825  max=0.935` plus a 5-bar histogram.
- **Why it is here:** Latency is the first thing to alert on.

### 12. Drift check on normal vs changed traffic  (`main.py` lines 118–122)

```python
    print("\n--- drift ---")
    drift_report("normal traffic", reference_texts, normal_traffic)
    novel = ["I ordered this item last week and honestly the whole experience from checkout to delivery was "
             "frustrating because nobody answered my emails and the package arrived damaged"] * 60
    drift_report("users now write long reviews", reference_texts, novel)
```

- **In simple words:** Compare the reference with normal traffic (should be stable) and with 60 long reviews (should drift).
- **Exactly:** `novel` is a list of one 30-word review repeated 60 times. Real: normal PSI 0.027 STABLE; long reviews PSI 7.827 DRIFT. 7.8 is huge because every live value falls in the last reference bin.
- **Why it is here:** Proves the detector is quiet when nothing changed and loud when it did.

### 13. Prediction drift: label share  (`main.py` lines 124–128)

```python
    for label in ("positive", "negative"):                           # prediction drift: label share vs expectation
        share = COUNTERS[f"label_{label}"] / COUNTERS["requests"]
        print(f"label share {label}: {share:.0%} (training ~50/50; with only 60 requests some wobble is normal, a persistent swing = investigate)")
    # TRY IT: change the novel texts to short ones ("good", "bad") and see what PSI does.
    # TRY IT: add a WARNING when p95 latency exceeds 50 ms -- that is the seed of an alert rule.
```

- **In simple words:** Compare the share of positive vs negative answers with the 50/50 training mix.
- **Exactly:** `COUNTERS['label_positive'] / COUNTERS['requests']`: 21/60 = 35% positive, 65% negative; `:.0%` formats as a percentage. With 60 requests wobble is normal; a persistent swing means investigate. The last two lines are `# TRY IT:` suggestions.
- **Why it is here:** Labels are available instantly, unlike ground truth, so label shift is the earliest signal of trouble.


## The three pillars
| Pillar | Question | Example here | Real tools |
|---|---|---|---|
| Logs | What happened to *this* request? | JSON line with `request_id` | ELK, CloudWatch, Datadog |
| Metrics | How is the *service* doing? | counters, p50/p95/p99 | Prometheus + Grafana |
| Traces | Where did time go across services? | (not demoed) | OpenTelemetry |
Plus **ML-specific** monitoring: input drift, prediction drift, and (when labels arrive later) real accuracy.

## What kind of drift?
- **Data/covariate drift:** inputs change (longer reviews). Detect by comparing live feature stats to the training reference (PSI here). Rule of thumb: <0.1 stable, 0.1–0.25 watch, >0.25 drift.
- **Prediction drift:** the share of predicted labels shifts.
- **Concept drift:** the relationship between input and correct label changes (new slang). Needs fresh labels to detect.
Response: investigate → collect/label recent data → retrain → register new version (step 04) → canary → promote.

## Syntax explained
| Syntax | Meaning | Example |
|---|---|---|
| `class JsonFormatter(logging.Formatter)` | Customize how a log record becomes text | — |
| `logger.info("msg", extra={"fields": {...}})` | Attach structured fields to a log record | — |
| `logging.INFO/WARNING` levels | `DEBUG < INFO < WARNING < ERROR < CRITICAL`; logger drops lower levels | `logger.setLevel(logging.WARNING)` |
| `Counter()` | dict that counts; missing keys start at 0 | `c["a"] += 1` |
| `uuid.uuid4().hex[:8]` | Random request id | — |
| `-> dict \| None` | Function may return None | — |
| `np.percentile(arr, [50,95,99])` | Percentiles | — |
| `getattr(record, "fields", {})` | Attribute with default if missing | — |

## Libraries & functions
| Name | Kind | Does |
|---|---|---|
| `logging.getLogger(name)`, `StreamHandler(sys.stdout)`, `.addHandler`, `.setFormatter` | stdlib | Wire logger → output → format |
| `np.histogram(arr, bins=5)` | numpy | counts per bin + edges |
| `np.histogram_bin_edges(ref, bins)` | numpy | bin edges from the reference |
| `np.clip(x, lo, hi)` | numpy | limit values to a range |
| PSI formula `Σ (live% − ref%) · ln(live% / ref%)` | — | distribution distance |

## Execution trace
`monitored_predict(model, "   ")`: request id created → `predict_text` raises `ValueError` → caught: `errors` counter +1, WARNING JSON log emitted, returns `None`. For a valid text: timed → `latencies.append`, counters updated, INFO log line with latency and `text_len` (the **length**, not the text — privacy).

## Experiments
1. Replace the long novel texts with `"good"` ×60 → PSI?
2. Alert rule: warn when p95 > 50 ms or error rate > 2%.
3. Change the logger level to `DEBUG` and add a `logger.debug` line inside `predict_text`.

## Debugging practice
No log lines appear? Checklist: logger level too high; no handler attached; logging to a file nobody is reading; in Docker, writing to a file instead of stdout (module 11: `docker logs` only shows stdout/stderr).

## Rebuild checklist
- [ ] Emit one structured log line with a request id.
- [ ] Compute p50/p95 from a list of latencies.
- [ ] Explain data drift vs concept drift and what you do when you detect drift.

## The new problem (cliffhanger)
That drift check was a quick one-feature taste. Real monitoring needs proper statistics and an alert threshold, fed from the prediction log. Next: [08_scalability_and_reliability](../08_scalability_and_reliability/), then the full drift demo in [09_drift_monitoring](../09_drift_monitoring/).

## Say it aloud
> **Say it aloud (interview box):** I log one structured JSON line per request with an id, model version, latency and outcome, and I track latency percentiles because averages hide slow tails. I also compare live input and prediction distributions with the training reference to catch drift. The cost is overhead and noisy thresholds, and true accuracy is only known once labels arrive.
