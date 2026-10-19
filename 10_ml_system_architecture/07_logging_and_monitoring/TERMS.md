# Terms — Step 07: Logging and monitoring

Folder: `10_ml_system_architecture/07_logging_and_monitoring` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [10 ML system architecture](../README.md)

**Terms in this folder:** [Observability (logs, metrics, traces)](#observability-logs-metrics-traces-concept) · [Structured logging](#structured-logging-concept) · [logging module](#logging-module-standard-library) · [Log levels](#log-levels-concept) · [logging.StreamHandler and Formatter](#loggingstreamhandler-and-formatter-standard-library) · [JsonFormatter (custom logging.Formatter)](#jsonformatter-custom-loggingformatter-syntax) · [extra={"fields": {...}}](#extrafields-standard-library) · [getattr with default](#getattr-with-default-python-built-in) · [Request id (uuid)](#request-id-uuid-standard-library) · [Counter](#counter-standard-library) · [Observability wrapper](#observability-wrapper-concept) · [Optional return type (dict | None)](#optional-return-type-dict-none-syntax) · [Percentiles (p50, p95, p99)](#percentiles-p50-p95-p99-concept) · [np.percentile](#nppercentile-third-party-numpy) · [np.histogram](#nphistogram-third-party-numpy) · [Text histogram with string multiplication](#text-histogram-with-string-multiplication-syntax) · [Drift](#drift-concept) · [Data (covariate) drift vs prediction drift vs concept drift](#data-covariate-drift-vs-prediction-drift-vs-concept-drift-concept) · [PSI (Population Stability Index)](#psi-population-stability-index-concept) · [np.histogram_bin_edges and np.clip](#nphistogram_bin_edges-and-npclip-third-party-numpy) · [Adding 1e-4 to avoid log(0)](#adding-1e-4-to-avoid-log0-concept) · [Reference distribution](#reference-distribution-concept) · [Monitored feature: text length](#monitored-feature-text-length-concept) · [Label share (prediction drift)](#label-share-prediction-drift-concept) · [Privacy: log length, not text](#privacy-log-length-not-text-concept) · [Alert rule](#alert-rule-concept) · [Prometheus and Grafana, ELK](#prometheus-and-grafana-elk-concept) · [Debug checklist: no log lines appear](#debug-checklist-no-log-lines-appear-concept) · [Reusing earlier steps (sys.path.insert)](#reusing-earlier-steps-syspathinsert-standard-library)

---

## Observability (logs, metrics, traces)  `(concept)`
**1. Definition:** The ability to understand a running system from its outputs: logs record what happened to a request, metrics summarise how the service is doing, and traces follow a request across services.
**2. In plain words:** Logs answer "what happened to this request?", metrics answer "how is the service behaving overall?", traces answer "where did the time go?". ML systems add model-specific monitoring such as drift and, when labels arrive, real accuracy.
**3. Example to quote:**
> "I emit a JSON log line per request, keep latency and error counters, and compare live inputs with the training distribution to catch drift."

**Seen in:** `WALKTHROUGH.md` "three pillars"

## Structured logging  `(concept)`
**1. Definition:** Writing each log entry as machine-readable key-value data, usually one JSON object per line, rather than free text.
**2. In plain words:** Tools like ELK, CloudWatch and Datadog can then filter and aggregate by field, for example "show all requests with latency over 50 ms". A request id lets you find every log line for one request.
**3. Example to quote:**
> "My log line is `{\"level\": \"INFO\", \"event\": \"prediction\", \"request_id\": \"aeb9d3bc\", \"label\": \"negative\", \"latency_ms\": 0.372}`."

**Seen in:** `JsonFormatter`, `monitored_predict`

## logging module  `(standard library)`
**1. Definition:** Python's standard logging framework, built from loggers (what you call), handlers (where records go) and formatters (how records look).
**2. In plain words:** It gives levels, routing and formatting out of the box, so you do not need `print`. In containers, log to stdout so `docker logs` can show them.
**3. Example to quote:**
> "I created `logging.getLogger('predict-service')`, attached a stdout handler and a JSON formatter, and set the level to INFO."

```python
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("x").info("hello")
```

**Seen in:** top of `main.py`

## Log levels  `(concept)`
**1. Definition:** Severity ranks `DEBUG < INFO < WARNING < ERROR < CRITICAL`; a logger drops records below its configured level.
**2. In plain words:** Levels let you control noise: development uses DEBUG, production INFO or WARNING. Raising the level to WARNING made routine INFO prediction lines disappear while the bad-input warning still printed.
**3. Example to quote:**
> "After `logger.setLevel(logging.WARNING)` the INFO prediction lines are dropped but the `bad_input` warning still shows."

**Seen in:** `main.py`

## logging.StreamHandler and Formatter  `(standard library)`
**1. Definition:** A `StreamHandler(sys.stdout)` sends log records to the terminal; a `Formatter` subclass controls how each record is converted to text; `handler.setFormatter(...)` and `logger.addHandler(handler)` wire them together.
**2. In plain words:** Logger decides whether to log, the handler decides where it goes, the formatter decides what it looks like. Missing any one of them is a common reason "no log lines appear".
**3. Example to quote:**
> "`handler.setFormatter(JsonFormatter())` makes every line JSON, and `logger.addHandler(handler)` sends it to stdout."

**Seen in:** top of `main.py`

## JsonFormatter (custom logging.Formatter)  `(syntax)`
**1. Definition:** A subclass of `logging.Formatter` whose `format` method returns `json.dumps(entry)` built from the record's level, message and extra fields.
**2. In plain words:** Overriding `format` is the standard way to change log output. Here each record becomes one JSON line with any extra fields added.
**3. Example to quote:**
> "My `JsonFormatter.format` returns a JSON string with `level`, `event` and the fields I passed through `extra`."

**Seen in:** `class JsonFormatter`

## extra={"fields": {...}}  `(standard library)`
**1. Definition:** The `extra` argument of logging calls attaches custom attributes to the log record; here a `fields` dict that the formatter reads with `getattr(record, "fields", {})`.
**2. In plain words:** It is how you pass structured data (request id, latency) into a log line without string formatting.
**3. Example to quote:**
> "`logger.info('prediction', extra={'fields': {'request_id': rid, 'latency_ms': 0.37}})` adds those keys to the JSON line."

**Seen in:** `monitored_predict`

## getattr with default  `(Python built-in)`
**1. Definition:** `getattr(obj, "name", default)` returns the attribute if it exists, otherwise the default.
**2. In plain words:** A log record without extra fields has no `fields` attribute; the default `{}` keeps the formatter from crashing.
**3. Example to quote:**
> "`getattr(record, 'fields', {})` returns an empty dict for plain log calls."

```python
print(getattr(object(), "x", "none"))   # none
```

**Seen in:** `JsonFormatter.format`

## Request id (uuid)  `(standard library)`
**1. Definition:** A short random identifier attached to each request (here `uuid.uuid4().hex[:8]`) and included in every log line for it.
**2. In plain words:** When many requests interleave, the id lets you find all lines that belong to one request, which is essential for debugging a specific user's problem.
**3. Example to quote:**
> "I generate `request_id = uuid.uuid4().hex[:8]` so I can grep one request's logs."

```python
import uuid
print(uuid.uuid4().hex[:8])
```

**Seen in:** `monitored_predict`

## Counter  `(standard library)`
**1. Definition:** `collections.Counter` is a dict subclass for counting; missing keys start at 0, so `c["a"] += 1` just works.
**2. In plain words:** It keeps simple metrics: requests, errors, and per-label counts such as `label_positive`. Counters only go up; rates are computed over time.
**3. Example to quote:**
> "`COUNTERS['requests'] += 1` and `COUNTERS[f'label_{label}'] += 1` give me totals by label with no setup."

```python
from collections import Counter
c = Counter(); c["a"] += 1; print(c)
```

**Seen in:** `COUNTERS`

## Observability wrapper  `(concept)`
**1. Definition:** A function (`monitored_predict`) that wraps the real prediction call to add a request id, timing, counters and logging, and to handle errors.
**2. In plain words:** Putting monitoring in one wrapper means every prediction path is observed the same way, and the model code stays clean.
**3. Example to quote:**
> "`monitored_predict` times the call, counts it, logs it, and turns bad input into a warning and a `None` result."

**Seen in:** `monitored_predict`

## Optional return type (dict | None)  `(syntax)`
**1. Definition:** `-> dict | None` says the function returns a dict or `None`.
**2. In plain words:** On invalid input the wrapper logs a warning and returns `None` instead of raising, so the caller must handle both outcomes.
**3. Example to quote:**
> "`monitored_predict(model, '   ')` returns `None` and logs `bad_input`."

**Seen in:** `monitored_predict`

## Percentiles (p50, p95, p99)  `(concept)`
**1. Definition:** The value below which a given percentage of measurements fall; p99 means 99% of requests were faster.
**2. In plain words:** The average hides slow outliers. p95 and p99 show the experience of your unluckiest users, so alerts and service-level targets use them.
**3. Example to quote:**
> "Latency shows p50=0.252 ms but p99=0.583 ms; the average alone would hide that tail."

**Seen in:** `latency_report`

## np.percentile  `(third-party: numpy)`
**1. Definition:** `np.percentile(arr, [50, 95, 99])` returns the requested percentile values of an array.
**2. In plain words:** One call computes several percentiles of the latency list, ready to print or alert on.
**3. Example to quote:**
> "`p50, p95, p99 = np.percentile(arr, [50, 95, 99])` gives me the three numbers for my report."

```python
import numpy as np
print(np.percentile([1, 2, 3, 4], 50))   # 2.5
```

**Seen in:** `latency_report`

## np.histogram  `(third-party: numpy)`
**1. Definition:** `np.histogram(arr, bins=5)` returns the count of values in each of five equal-width bins, plus the bin edges.
**2. In plain words:** It summarises a distribution; the demo prints a text bar chart with `#` characters to show how latencies spread.
**3. Example to quote:**
> "Most requests fall in the first bin, 48 of 60, with a short tail of slower ones."

**Seen in:** `latency_report`, `psi`

## Text histogram with string multiplication  `(syntax)`
**1. Definition:** `'#' * n` repeats a character n times, and `f"{...:<40}"` pads it to width 40.
**2. In plain words:** A quick terminal bar chart with no plotting library.
**3. Example to quote:**
> "`'#' * int(40 * c / len(arr))` draws a bar whose length is proportional to the count in that bin."

```python
print("#" * 5)   # #####
```

**Seen in:** `latency_report`

## Drift  `(concept)`
**1. Definition:** A change over time in the data or in the relationship between inputs and outputs, causing a deployed model to perform worse.
**2. In plain words:** The world changes: users write longer reviews, new slang appears. The model was trained on the past, so monitoring checks whether live data still looks like training data.
**3. Example to quote:**
> "When users started writing long reviews, my drift check flagged it before accuracy complaints arrived."

**Seen in:** `drift_report`

## Data (covariate) drift vs prediction drift vs concept drift  `(concept)`
**1. Definition:** Data drift is a change in the input distribution; prediction drift is a change in the share of predicted labels; concept drift is a change in the relationship between input and correct label.
**2. In plain words:** The first two can be detected without labels by comparing live statistics to a reference. Concept drift (for example new meanings of a word) needs fresh labels to detect.
**3. Example to quote:**
> "Text length shifting is data drift; a sudden 90% positive share is prediction drift; both are warnings, not proof the model is wrong."

**Seen in:** `WALKTHROUGH.md`

## PSI (Population Stability Index)  `(concept)`
**1. Definition:** A measure of how different two distributions are: the sum over bins of `(live% - ref%) * ln(live% / ref%)`. Common rule: below 0.1 stable, 0.1 to 0.25 watch, above 0.25 significant drift.
**2. In plain words:** Bin the reference (training) data, bin live data with the same edges, and compare the proportions. A small score means "looks the same"; a large score means the inputs changed. Here normal traffic scored 0.027 (stable) and long reviews 7.827 (drift).
**3. Example to quote:**
> "PSI of text length was 0.027 for normal traffic and 7.827 when users wrote long reviews, so the check fired."

**Seen in:** `psi`

## np.histogram_bin_edges and np.clip  `(third-party: numpy)`
**1. Definition:** `np.histogram_bin_edges(reference, bins)` computes bin boundaries from the reference data; `np.clip(x, lo, hi)` limits values to a range.
**2. In plain words:** The bins come from the reference only, and live values outside that range are clipped into the end bins so they are still counted.
**3. Example to quote:**
> "I clip live lengths to the reference range, so a 40-word review lands in the last bin instead of disappearing."

**Seen in:** `psi`

## Adding 1e-4 to avoid log(0)  `(concept)`
**1. Definition:** A tiny constant added to bin proportions so no proportion is exactly zero before taking a logarithm or ratio.
**2. In plain words:** An empty bin would make `log(0)` or a division by zero. A small epsilon keeps the maths finite without changing the result noticeably.
**3. Example to quote:**
> "`+ 1e-4` on each percentage makes sure an empty bin doesn't crash the PSI formula."

**Seen in:** `psi`

## Reference distribution  `(concept)`
**1. Definition:** The training data's feature statistics used as the baseline against which live data is compared.
**2. In plain words:** Drift only makes sense relative to something. The demo uses the 300 training texts as the reference for text length.
**3. Example to quote:**
> "`reference_texts` are what the model saw in training; live traffic is compared against them."

**Seen in:** `main.py`

## Monitored feature: text length  `(concept)`
**1. Definition:** A simple measurable property of each input (words per text) tracked to detect changes.
**2. In plain words:** You can monitor input length, vocabulary overlap or any cheap statistic. It is a quick proxy for "the inputs look different now".
**3. Example to quote:**
> "`len(t.split())` gives words per text, and I compare its distribution between training and live."

**Seen in:** `drift_report`

## Label share (prediction drift)  `(concept)`
**1. Definition:** The proportion of predictions assigned to each label, compared with the expected proportion.
**2. In plain words:** If training was about 50/50 and live traffic suddenly predicts 90% negative, something changed (the data, an upstream bug, or the world). With few requests some wobble is normal; a persistent swing deserves investigation.
**3. Example to quote:**
> "Of 60 requests, 39 were negative; the training split was about 50/50, so I'd watch it over time."

**Seen in:** end of `main.py`

## Privacy: log length, not text  `(concept)`
**1. Definition:** Logging derived values such as `len(text)` instead of raw user input to avoid storing personal data.
**2. In plain words:** Raw text may contain emails, names or secrets. Logging only its length still allows monitoring while reducing risk.
**3. Example to quote:**
> "My log line has `text_len: 18`, not the sentence itself, because the text could be private."

**Seen in:** `monitored_predict`

## Alert rule  `(concept)`
**1. Definition:** A condition on a metric (for example p95 latency above 50 ms or error rate above 2%) that triggers a warning or page.
**2. In plain words:** Metrics only help if somebody is told when they go wrong. Start with a few simple, meaningful rules, then tune thresholds to avoid noise.
**3. Example to quote:**
> "I'd warn when p95 latency exceeds 50 ms or the error rate passes 2%."

**Seen in:** `TRY IT` comment, `WALKTHROUGH.md`

## Prometheus and Grafana, ELK  `(concept)`
**1. Definition:** Prometheus collects and stores time-series metrics and Grafana draws dashboards from them; ELK (Elasticsearch, Logstash, Kibana) indexes and searches logs.
**2. In plain words:** They are the production tools behind the ideas in this step. The demo uses plain Python so you can see exactly what they measure.
**3. Example to quote:**
> "In production my counters and latency histograms would be scraped by Prometheus and shown in Grafana."

**Seen in:** `WALKTHROUGH.md`

## Debug checklist: no log lines appear  `(concept)`
**1. Definition:** Possible causes: logger level too high, no handler attached, logging to a file nobody reads, or in Docker writing to a file instead of stdout.
**2. In plain words:** `docker logs` only shows stdout and stderr, so file logging inside a container is invisible by default. Walk the list from logger to handler to destination.
**3. Example to quote:**
> "My container showed nothing because the app logged to a file; switching to a stdout handler fixed it."

**Seen in:** `WALKTHROUGH.md` debugging practice

## Reusing earlier steps (sys.path.insert)  `(standard library)`
**1. Definition:** The script inserts sibling step folders into `sys.path` to import `predict`, `train` and `data`.
**2. In plain words:** This reuses the inference code from step 02 and the data from step 01 instead of copying them, so monitoring wraps the same model.
**3. Example to quote:**
> "I import `predict_text` from step 02 and wrap it in `monitored_predict`."

**Seen in:** top of `main.py`
