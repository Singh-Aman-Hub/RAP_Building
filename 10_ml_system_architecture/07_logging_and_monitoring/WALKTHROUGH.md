# Step 07 — Logging, metrics and drift monitoring

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

## Say it aloud
"I log one structured JSON line per request with an id, model version, latency and outcome; I track counters and latency percentiles, because averages hide slow tails; and I compare live input and prediction distributions against the training reference to catch drift. When drift shows up I investigate, retrain, version and roll out carefully. Trade-off: monitoring adds overhead and noise, thresholds need tuning, and true accuracy can only be measured once labels arrive."
