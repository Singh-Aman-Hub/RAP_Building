# Terms — Observability and online eval hooks: traces, logs, metrics, feedback

Folder: `20_llm_system_lld/09_observability_and_eval_hooks` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [20 README](../README.md)

**Terms in this folder:** [Trace and span](#trace-and-span--concept) · [contextvars (ContextVar)](#contextvars-contextvar--library) · [Context manager (@contextmanager)](#context-manager-contextmanager--syntax) · [Structured logging (JSON lines)](#structured-logging-json-lines--concept) · [Percentile (p50, p95, p99), nearest rank](#percentile-p50-p95-p99-nearest-rank--concept) · [Metrics: counter vs histogram](#metrics-counter-vs-histogram--concept) · [Cache hit rate](#cache-hit-rate--concept) · [OpenTelemetry GenAI semantic conventions](#opentelemetry-genai-semantic-conventions--concept) · [Online evaluation hooks](#online-evaluation-hooks--concept) · [Feedback capture (thumbs up/down joined to trace)](#feedback-capture-thumbs-updown-joined-to-trace--concept) · [Time to first token (TTFT)](#time-to-first-token-ttft--concept) · [Re-raise after recording (raise)](#re-raise-after-recording-raise--syntax) · [ValueError: rating must be +1 or -1](#valueerror-rating-must-be-1-or--1--error)

---

## Trace and span  `(concept)`
**1. Definition:** A span is a named, timed unit of work with attributes, status and a parent; a trace is the tree of spans for one request.
**2. In plain words:** A receipt line per step of one order.
**3. Example to quote:** > "Spans nest through a context variable; the root span starts the trace id."
```python
with tracer.span('retrieve', k=20) as s: ...
```
**Seen in:** `ragkit/observability.py`

## contextvars (ContextVar)  `(library)`
**1. Definition:** Context-local variables that are isolated per thread and per asyncio task.
**2. In plain words:** A pocket that follows each task around.
**3. Example to quote:** > "Plain worker threads do not inherit them; use copy_context().run."
```python
_current: ContextVar = ContextVar('current_span', default=None)
```
**Seen in:** `ragkit/observability.py`

## Context manager (@contextmanager)  `(syntax)`
**1. Definition:** A generator-based `with` helper: code before yield is setup, after yield (in finally) is teardown.
**2. In plain words:** Open the door, do the work, always close it.
**3. Example to quote:** > "Spans close in `finally`, so errors are timed and the parent restored."
```python
@contextmanager
def span(...):
    try: yield s
    finally: ...
```
**Seen in:** `ragkit/observability.py`

## Structured logging (JSON lines)  `(concept)`
**1. Definition:** One JSON object per log event, with stable keys and correlation ids.
**2. In plain words:** A log you can query instead of grep.
**3. Example to quote:** > "Sorted keys and a trace_id make logs diffable and joinable."
```python
{"level": "info", "msg": "answered", "trace_id": "trace-..."}
```
**Seen in:** `ragkit/observability.py`

## Percentile (p50, p95, p99), nearest rank  `(concept)`
**1. Definition:** The value below which p% of samples fall; nearest-rank picks sorted[ceil(p/100*n)-1].
**2. In plain words:** '95 of 100 users waited this long or less.'
**3. Example to quote:** > "Mean 63 ms hid a p95 of 900 ms in the demo."
```python
xs[max(0, math.ceil(p / 100 * len(xs)) - 1)]
```
**Seen in:** `ragkit/observability.py`

## Metrics: counter vs histogram  `(concept)`
**1. Definition:** A counter only increases (requests, errors); a histogram records a distribution (latency, tokens).
**2. In plain words:** Odometer versus a log of trip times.
**3. Example to quote:** > "Avoid high-cardinality labels such as the question text."
**Seen in:** `ragkit/observability.py`

## Cache hit rate  `(concept)`
**1. Definition:** hits / (hits + misses); here counted from the answer's cache label.
**2. In plain words:** How often the shortcut worked.
**3. Example to quote:** > "Exact and semantic hits are counted together; split them for tuning."
```python
metrics.rate('cache_hit', 'cache_miss')
```
**Seen in:** `main.py`

## OpenTelemetry GenAI semantic conventions  `(concept)`
**1. Definition:** Standard gen_ai.* span/metric attribute names (operation, provider, request model, input/output tokens) for LLM calls; Development status as of Nov 2026, moved to a separate repository.
**2. In plain words:** Common labels so every tool reads your LLM traces.
**3. Example to quote:** > "Pin versions and check the repo; capturing prompt content is opt-in because of privacy."
```python
gen_ai.usage.input_tokens
```
**Seen in:** `ragkit/observability.py`

## Online evaluation hooks  `(concept)`
**1. Definition:** Cheap checks run on every live answer (citation valid, latency) whose flags become metrics.
**2. In plain words:** A smoke detector on every request.
**3. Example to quote:** > "Run expensive judges on a sample or asynchronously."
```python
flags.update(hook.evaluate(question, ans, latency_ms))
```
**Seen in:** `ragkit/observability.py`

## Feedback capture (thumbs up/down joined to trace)  `(concept)`
**1. Definition:** Storing user ratings keyed by the trace id of the answer they rate.
**2. In plain words:** A comment card stapled to the receipt.
**3. Example to quote:** > "Negative feedback becomes new evaluation cases."
```python
fb.record(trace_id, -1, 'wrong threshold')
```
**Seen in:** `ragkit/observability.py`

## Time to first token (TTFT)  `(concept)`
**1. Definition:** Latency until the first streamed token arrives; the perceived latency of a streaming answer.
**2. In plain words:** How long until the first word appears.
**3. Example to quote:** > "Measure it separately from total latency."
**Seen in:** `WALKTHROUGH.md`, theory

## Re-raise after recording (raise)  `(syntax)`
**1. Definition:** Catching an exception to record it, then raising it again unchanged.
**2. In plain words:** Noting the accident, then calling for help.
**3. Example to quote:** > "A tracer must never swallow errors."
```python
except Exception as exc:
    s.status = 'error'
    raise
```
**Seen in:** `ragkit/observability.py`

## ValueError: rating must be +1 or -1  `(error)`
**1. Definition:** Raised by FeedbackStore.record when the rating is outside the allowed values.
**2. In plain words:** A form rejecting a score of 5 on a thumbs scale.
**3. Example to quote:** > "Validate at boundaries so bad data cannot skew metrics."
**Seen in:** `WALKTHROUGH.md`, Debugging practice
