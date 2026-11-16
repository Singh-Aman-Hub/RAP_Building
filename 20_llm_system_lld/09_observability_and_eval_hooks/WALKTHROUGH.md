# Step 09 — Observability and online eval hooks: traces, logs, metrics, feedback

## The story so far
[Step 08](../08_llm_gateway_and_streaming/WALKTHROUGH.md) metered tokens and cost per call. Module [10 step 07](../../10_ml_system_architecture/07_logging_and_monitoring/WALKTHROUGH.md) covered logging and monitoring for ML services in general; module [14 step 09](../../14_rag_and_retrieval/09_rag_evaluation/WALKTHROUGH.md) measured RAG quality offline on a labelled set.

## The pain point
A user says 'the bot was slow and wrong'. Was it retrieval (6 ms or 600?), the reranker, the model, a cold cache, a retry storm? Logs without a request id cannot be joined; averages hide the 5% of requests that take ten times longer; quality regressions are invisible until a customer complains, because offline evals only cover questions you thought of.

## The spark
Instrument by **wrapping** (Decorators from step 05), not by editing: a **tracer** records nested spans with timings and attributes; a **JSON logger** stamps every line with the trace id; **metrics** aggregate latency percentiles, tokens, cost and cache hit rate; **online eval hooks** run cheap checks on every answer (citation valid? too slow?) and **feedback** (thumbs up/down) is joined to the trace so bad answers become labelled test cases.

## Goal
Build a context-managed nested tracer, a trace-correlated JSON logger, a percentile metrics store, tracing decorators with OpenTelemetry GenAI attribute names, eval hooks and a feedback store; run six requests and read the trace tree, the metrics and the join from a thumbs-down to its log line.

## The theory (read this before the interview)
### The three pillars, and what each answers
| Signal | Question | Cardinality | ragkit |
|---|---|---|---|
| **Traces** (spans) | where did *this request's* time go? what called what? | one per request | `Tracer`, `Span`, `tracer.tree()` |
| **Logs** | what exactly happened (discrete events with fields)? | many per request | `JsonLogger` (one JSON object per line, with `trace_id`) |
| **Metrics** | how is the *system* behaving over time? | aggregated | `Metrics`: counters, histograms, `percentile`, `rate` |
Traces and logs join on **trace_id**; metrics tell you *that* something is wrong, traces tell you *where*, logs tell you *what*.

### Spans
A span = a named, timed unit of work with attributes, a status and a parent. `with tracer.span('retrieve', k=20) as s:` starts it; on exit it records the end time (in `finally`, so errors are timed too) and, on exception, sets `status='error'` and **re-raises** (record, never swallow). Nesting is automatic through a `contextvars.ContextVar` holding the current span: a span created inside another becomes its child and shares its `trace_id`; leaving the block restores the parent (`reset(token)`). Context variables are per-thread/per-task: **a plain worker thread does not inherit them** (Experiments); copy the context (`contextvars.copy_context().run`) or pass the span explicitly. `asyncio` tasks do copy the context.

### Percentiles, not averages
Latency is skewed. In Experiments 94 requests at 10 ms and 6 at 900 ms: mean 63.4 ms (looks fine), p50 10 ms, **p95 900 ms**. Report p50/p95/p99. `percentile(p)` = nearest-rank: sort, take element `ceil(p/100 * n) - 1`. A histogram with fixed buckets (Prometheus-style) is the production form; computing percentiles from raw samples is fine for a demo. Watch **cardinality**: never use `question` or `user_id` as a metric label.

### What to measure for an LLM/RAG service
Latency per stage and total (p50/p95), time to first token (streaming), input/output tokens, **cost per request**, cache hit rate (exact vs semantic), refusal rate, retry and fallback rates, tool-call count, citation-valid rate, retrieval score distribution (drift signal), error rate by type.

### OpenTelemetry GenAI semantic conventions (status verified Nov 2026)
OpenTelemetry defines standard span/metric attribute names for GenAI so any backend understands them. As of the sources checked (the OpenTelemetry site now says the GenAI conventions **moved to a separate `semantic-conventions-genai` repository**; secondary sources from May-Sept 2026 describe all `gen_ai.*` items as still **Development** status, i.e. not stable), the commonly used names are:
- `gen_ai.operation.name` (`chat`, `embeddings`, `execute_tool`, `invoke_agent`, `retrieval`, ...)
- `gen_ai.provider.name` (older drafts used `gen_ai.system`)
- `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`
- span name convention for model calls: `{operation} {model}`, e.g. `chat claude-...`
`TracedGenerator` sets exactly these (names only; ragkit uses its own simple tracer, not the OTel SDK). Because the status is Development, **pin versions and check the repository before relying on a name**; instrumentation libraries may require an opt-in environment variable (check their README; I did not verify the variable). Prompt/response *content* capture is opt-in because it can contain personal data.

### Online eval hooks
Offline evaluation (labelled set, step 10) is slow and narrow; online hooks run on **every real answer** and must be cheap and non-blocking: `CitationHook` (answer has valid, supported citations unless refused), `SlowHook` (latency above a limit). Each returns `{flag: bool}`; flagged counts become metrics (`flag.bad_citation`). Heavier checks (LLM-as-judge for faithfulness) run on a **sample** or asynchronously. Alert on the *rate* of flags, and review flagged traces by hand.

### Feedback capture
`FeedbackStore.record(trace_id, +1/-1, comment)`. Because the UI returns the `trace_id` with the answer, a thumbs-down can be joined to the logged question, retrieved chunks, prompt and answer. Negative feedback becomes new `EvalCase`s (step 10): the cheapest way to grow a regression suite from reality.

### Privacy and cost of telemetry
Redact PII before logging (the chain in step 05 does it for questions); sample verbose traces; store prompts only with consent and retention limits; telemetry must never break the request (the EventBus isolates subscribers).

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Tracing | distributed tracing, spans, APM | Close | Tracing records causal, timed structure of one request; APM products add dashboards on top. |
| Observability | monitoring + logging + tracing, telemetry | No | Monitoring watches known failure signals; observability lets you ask new questions of the data. |
| Span | trace segment, timed operation | Same | A node of the trace tree: name, start, end, attributes, status, parent. |
| Structured log | JSON log, JSONL, event log | Same | One machine-parseable object per event instead of free text. |
| Percentile latency (p50/p95/p99) | tail latency, quantile | Close | p95 = 95% of requests are at least this fast. |
| Online eval | production evaluation, live monitoring of quality, guardrail metrics | Close | Cheap checks on live traffic; offline eval = labelled set run before release. |
| Feedback loop | thumbs up/down, user ratings, human-in-the-loop labels | Close | Capture signals from users and turn them into test cases. |
| OTel GenAI semantic conventions | OpenTelemetry gen_ai attributes, OTel LLM conventions | Same | Standard attribute names (`gen_ai.*`) for LLM spans and metrics; Development status as of Nov 2026. |

## Run it
```bash
# from this step folder
HF_HUB_OFFLINE=1 python main.py
# or from the repo root
HF_HUB_OFFLINE=1 .venv/bin/python 20_llm_system_lld/09_observability_and_eval_hooks/main.py
```


## Code, step by step
Two files: `main.py` and `ragkit/observability.py`. Every block is listed.

#### Walking `main.py`

### 1. Header docstring  (`main.py` lines 1–15)
```python
"""
GOAL:  See inside the RAG service: nested spans (a trace tree), structured JSON logs that join to traces, metrics
       (latency p50/p95, tokens, cost, cache hit rate), GenAI-convention attributes, online eval hooks and feedback capture.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 20_llm_system_lld/09_observability_and_eval_hooks/main.py
EXPECTED OUTPUT (shortened):
    rag.answer  ... ms
      retrieve  ... n_hits=..
      rerank    ... best_score=..
      chat mock-llm ... gen_ai.operation.name=chat gen_ai.usage.input_tokens=..
    metrics: requests=.. latency_ms.p50=.. latency_ms.p95=.. cache hit rate ..
    feedback: thumbs-down trace ids ... joined to the logged answer
NEEDS: ragkit only (HashingEmbedder, no models).
"""
```
- **In simple words:** What the file shows.
- **Exactly:** GOAL/RUN/EXPECTED/NEEDS (no models: HashingEmbedder).
- **Why it is here:** Orientation.

### 2. Imports  (`main.py` lines 16–27)
```python
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragkit.cache import CachedAnswerer, SemanticCache, TTLLRUCache
from ragkit.factory import build_system
from ragkit.ingest import DirectoryLoader, HashingEmbedder
from ragkit.observability import (CitationHook, FeedbackStore, JsonLogger, Metrics, SlowHook, TracedAnswerer,
                                  TracedGenerator, TracedReranker, TracedRetriever, Tracer)

KB = Path(__file__).resolve().parents[2] / "14_rag_and_retrieval" / "data" / "kb"
```
- **In simple words:** Tracer, logger, metrics, wrappers, hooks, plus the cache and factory from earlier steps.
- **Exactly:** All observability classes come from one module.
- **Why it is here:** Instrumentation is a separate concern from the pipeline.

### 3. Wrap the existing service with tracing decorators  (`main.py` lines 29–39)
```python
# ---- 1. Wire the instruments AROUND the existing parts (Decorators) --------------------------------------------------
system = build_system({}, DirectoryLoader(KB))
system.pipeline.run()
svc = system.service
tracer, metrics, log_lines = Tracer(), Metrics(), []
svc.retriever = TracedRetriever(svc.retriever, tracer)
svc.reranker = TracedReranker(svc.reranker, tracer)
svc.generator = TracedGenerator(svc.generator, tracer, model="mock-llm", provider="local-sim")
cached = CachedAnswerer(svc, TTLLRUCache(50, 300), SemanticCache(HashingEmbedder(128), 0.75),
                        scope=lambda: system.pipeline.index_version)
app = TracedAnswerer(cached, tracer, metrics, JsonLogger(log_lines), hooks=[CitationHook(), SlowHook(limit_ms=5.0)])
```
- **In simple words:** Put a stopwatch around each stage without editing any class.
- **Exactly:** `svc.retriever = TracedRetriever(...)`, `svc.reranker = TracedReranker(...)`, `svc.generator = TracedGenerator(..., model='mock-llm', provider='local-sim')`; the cache wraps the service; `TracedAnswerer` wraps the cache and runs hooks `CitationHook`, `SlowHook(5 ms)`.
- **Why it is here:** Order matters: the root span sits outside the cache, so cache hits are traced too.

### 4. Send six questions  (`main.py` lines 41–51)
```python
# ---- 2. Traffic -----------------------------------------------------------------------------------------------------------
questions = ["How many days do I have to get a refund?", "How many days do I have to get a refund?",
             "Is shipping free on my order?", "What is the capital of Mongolia?", "Is anyone there to help me on Sunday?",
             "how many days do i have to get a refund"]
trace_ids = {}                                                      # question -> trace id of its LAST request
trace_list = []                                                     # trace id of every request, in order
for q in questions:
    a = app.answer(q)
    trace_ids[q] = app.last_trace_id
    trace_list.append(app.last_trace_id)
    print(f"{q[:44]!r:48s} refused={a.refused!s:5s} cache={a.meta.get('cache')}")
```
- **In simple words:** Mix: repeated, paraphrased, unanswerable.
- **Exactly:** Results: `miss`, `exact`, `miss`, refused+`miss` (Mongolia), `miss`, `semantic(1.00)`. `trace_ids` keeps each request's trace id.
- **Why it is here:** Realistic variety makes the metrics non-trivial.

### 5. Print two trace trees  (`main.py` lines 53–57)
```python
# ---- 3. A trace tree: the first (cache-miss) request ------------------------------------------------------------------------
print("\ntrace tree of the first request (name, duration, attributes):")
print(tracer.tree(trace_list[0]))
print("\nthe 2nd request (exact cache hit) has only the root span:")
print(tracer.tree(trace_list[1]))
```
- **In simple words:** Show the nested spans of a cache miss and of an exact hit.
- **Exactly:** Miss: root `rag.answer` with children `retrieve` (k=20, n_hits=20), `rerank` (n_in=20, best_score=0.5), `chat mock-llm` (GenAI attributes, 223 input and 27 output tokens). The exact hit has only the root span, a fraction of a millisecond.
- **Why it is here:** A cache hit is visible as a missing subtree.

### 6. Read the metrics  (`main.py` lines 59–67)
```python
# ---- 4. Metrics ---------------------------------------------------------------------------------------------------------------
snap = metrics.snapshot()
print("\nmetrics:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in sorted(snap.items())})
print(f"cache hit rate (hits/(hits+misses)) = {metrics.rate('cache_hit', 'cache_miss'):.2f}")
gen_spans = [s for s in tracer.spans if s.name.startswith("chat ")]
tok_in = sum(s.attrs["gen_ai.usage.input_tokens"] for s in gen_spans)
tok_out = sum(s.attrs["gen_ai.usage.output_tokens"] for s in gen_spans)
print(f"LLM calls actually made: {len(gen_spans)} of {len(questions)} requests | input tokens {tok_in} output tokens {tok_out} (word-count estimates)")
print("GenAI attributes on a chat span:", json.dumps(gen_spans[0].attrs, sort_keys=True))
```
- **In simple words:** Counters, percentiles, hit rate, token totals.
- **Exactly:** `requests: 6`, `refusals: 1`, `cache_hit: 2`, `cache_miss: 4`, hit rate 0.33, `flag.slow: 1`, p50 about 1-2 ms vs p95 about 12-20 ms (varies per run); LLM calls actually made: 3 of 6; tokens 557 in / 70 out (word-count estimates).
- **Why it is here:** Only 3 of 6 requests paid for generation: the cache and the refusal saved the rest.

### 7. A JSON log line and its trace id  (`main.py` lines 69–71)
```python
# ---- 5. Structured logs ------------------------------------------------------------------------------------------------------------
print("\nfirst JSON log line:", log_lines[0])
print("log line carries the trace id of its request:", json.loads(log_lines[0])["trace_id"] == trace_list[0])
```
- **In simple words:** One line per request, joinable to the trace.
- **Exactly:** Keys: `flags`, `latency_ms`, `level`, `msg`, `refused`, `trace_id`, `ts`; `json.loads(line)['trace_id'] == trace_list[0]` is `True`.
- **Why it is here:** The join key between logs and traces.

### 8. Thumbs-down joined to its log line  (`main.py` lines 73–79)
```python
# ---- 6. Feedback joined to traces -> labelled examples for offline eval --------------------------------------------------------------
fb = FeedbackStore()
fb.record(trace_ids["What is the capital of Mongolia?"], +1, "correct refusal")
fb.record(trace_ids["Is shipping free on my order?"], -1, "wrong threshold, I live in Pune")
bad = fb.negative_trace_ids()
bad_logs = [json.loads(l) for l in log_lines if json.loads(l)["trace_id"] in bad]
print("\nthumbs-down trace ids:", bad, "->", [(l["msg"], l["refused"], l["flags"]) for l in bad_logs])
```
- **In simple words:** Record feedback on two traces and look up the bad one.
- **Exactly:** `fb.negative_trace_ids()` -> one id; the matching log line shows `('answered', False, {'bad_citation': False, 'slow': False})`.
- **Why it is here:** From a complaint to the exact request in one lookup.

### 9. Exercises  (`main.py` lines 81–83)
```python
# TRY IT: set SlowHook(limit_ms=0.01): how does flag.slow change, and is it a good alert on its own?
# TRY IT: add a second CachedAnswerer layer outside TracedAnswerer instead of inside: what does the trace tree lose?
# BREAK IT: remove `_current.reset(token)` in Tracer.span and run again: the trace tree nests every later request under the first.
```
- **In simple words:** Things to try.
- **Exactly:** Comment lines.
- **Why it is here:** See exercises.md.


#### Walking `ragkit/observability.py`

### 10. observability.py header and imports  (`ragkit/observability.py` lines 1–16)
```python
"""observability.py -- spans, structured logs, metrics, online-eval hooks, feedback (step 09)."""
from __future__ import annotations

import contextvars
import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Protocol

from .domain import Answer
from .ids import new_id
from .query import WordTokenCounter

# Attribute names follow the OpenTelemetry GenAI semantic conventions (status: Development as of Nov 2026).
```
- **In simple words:** Everything for telemetry.
- **Exactly:** Imports `contextvars`, `contextmanager`, `json`, `math`, the `Answer` type, `new_id` and `WordTokenCounter`.
- **Why it is here:** Standard library only; no OpenTelemetry SDK required.

### 11. GenAI attribute names  (`ragkit/observability.py` lines 17–18)
```python
GENAI = {"op": "gen_ai.operation.name", "model": "gen_ai.request.model", "provider": "gen_ai.provider.name",
         "in_tok": "gen_ai.usage.input_tokens", "out_tok": "gen_ai.usage.output_tokens"}
```
- **In simple words:** Constants for the OTel GenAI attribute keys.
- **Exactly:** `op`, `model`, `provider`, `in_tok`, `out_tok` map to `gen_ai.operation.name`, `gen_ai.request.model`, `gen_ai.provider.name`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`. Status: Development (Nov 2026).
- **Why it is here:** One place to change if the convention changes.

### 12. The current-span context variable  (`ragkit/observability.py` lines 20–20)
```python
_current: contextvars.ContextVar["Span | None"] = contextvars.ContextVar("current_span", default=None)
```
- **In simple words:** A hidden 'which span am I inside?' variable.
- **Exactly:** `ContextVar('current_span', default=None)`; per thread and per asyncio task.
- **Why it is here:** Gives automatic nesting without passing spans around.

### 13. Span: the record of one unit of work  (`ragkit/observability.py` lines 23–37)
```python
@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    start: float
    end: float | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        return ((self.end or self.start) - self.start) * 1000
```
- **In simple words:** Name, ids, times, attributes, status.
- **Exactly:** Fields `name, trace_id, span_id, parent_id, start, end, attrs, status ('ok'|'error'), error`; `duration_ms = (end - start) * 1000`.
- **Why it is here:** Mutable on purpose: attributes are added while the span is open.

### 14. Tracer: constructor  (`ragkit/observability.py` lines 40–44)
```python
class Tracer:
    """A span is a timed, named unit of work with attributes. Nesting is automatic via a contextvar."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter):
        self.clock, self.spans = clock, []
```
- **In simple words:** A clock and a list of finished spans.
- **Exactly:** `clock` defaults to `time.perf_counter`; tests inject a fake ticker for exact durations.
- **Why it is here:** Deterministic tests.

### 15. Tracer.span: open a span  (`ragkit/observability.py` lines 46–52)
```python
    @contextmanager
    def span(self, name: str, **attrs) -> Iterator[Span]:
        parent = _current.get()
        s = Span(name, parent.trace_id if parent else new_id("trace"), new_id("sp"),
                 parent.span_id if parent else None, self.clock(), attrs=dict(attrs))
        token = _current.set(s)                                # children created inside the `with` see s as parent
        try:
```
- **In simple words:** A context manager that starts a child of the current span.
- **Exactly:** Parent = `_current.get()`; new span reuses the parent's `trace_id` or creates `new_id('trace')`; `_current.set(s)` returns a token; `yield s` hands it to the caller.
- **Why it is here:** Root spans start a new trace; children join it.

### 16. Tracer.span: close, record errors, restore  (`ragkit/observability.py` lines 53–60)
```python
            yield s
        except Exception as exc:
            s.status, s.error = "error", f"{type(exc).__name__}: {exc}"
            raise                                              # record, never swallow
        finally:
            s.end = self.clock()
            _current.reset(token)                              # restore the parent as current
            self.spans.append(s)
```
- **In simple words:** Always finish the span.
- **Exactly:** `except Exception`: `status='error'`, `error='Type: msg'`, then `raise`. `finally`: `end = clock()`, `_current.reset(token)` (restore parent), `spans.append(s)`.
- **Why it is here:** Without `reset`, later requests would nest under the first (see BREAK IT).

### 17. Tracer.current  (`ragkit/observability.py` lines 62–64)
```python
    @staticmethod
    def current() -> Span | None:
        return _current.get()
```
- **In simple words:** Which span am I in?
- **Exactly:** Static method returning `_current.get()`.
- **Why it is here:** Used by the logger and by code that wants to add attributes.

### 18. Tracer.tree: render a trace as text  (`ragkit/observability.py` lines 66–80)
```python
    def tree(self, trace_id: str | None = None) -> str:
        spans = [s for s in self.spans if trace_id is None or s.trace_id == trace_id]
        kids: dict[str | None, list[Span]] = {}
        for s in sorted(spans, key=lambda s: s.start):
            kids.setdefault(s.parent_id, []).append(s)
        lines: list[str] = []

        def walk(pid, depth):
            for s in kids.get(pid, []):
                extra = " ".join(f"{k}={v}" for k, v in s.attrs.items())
                flag = "" if s.status == "ok" else f" [{s.status}: {s.error}]"
                lines.append(f"{'  ' * depth}{s.name:<{22 - 2 * depth}} {s.duration_ms:7.2f} ms  {extra}{flag}")
                walk(s.span_id, depth + 1)
        walk(None, 0)
        return "\n".join(lines)
```
- **In simple words:** An indented list of spans.
- **Exactly:** Groups spans by `parent_id`, sorts by start, walks depth-first with indentation; prints name, `duration_ms`, attributes and an error flag.
- **Why it is here:** Quick local view; production exports to Jaeger/Tempo/etc.

### 19. JsonLogger: constructor  (`ragkit/observability.py` lines 83–87)
```python
class JsonLogger:
    """One JSON object per line, auto-tagged with the current trace/span ids so logs and traces join."""

    def __init__(self, sink: list | None = None, clock: Callable[[], float] = time.time):
        self.sink, self.clock = sink if sink is not None else [], clock
```
- **In simple words:** Where lines go and what clock to use.
- **Exactly:** `sink` list (a file or stream in production), `clock=time.time`.
- **Why it is here:** Injectable sink: tests read lines.

### 20. JsonLogger.log: one JSON object per line  (`ragkit/observability.py` lines 89–96)
```python
    def log(self, level: str, msg: str, **fields) -> str:
        cur = _current.get()
        rec = {"ts": round(self.clock(), 3), "level": level, "msg": msg, **fields}
        if cur:
            rec["trace_id"], rec["span_id"] = cur.trace_id, cur.span_id
        line = json.dumps(rec, sort_keys=True, default=str)
        self.sink.append(line)
        return line
```
- **In simple words:** Build a record, tag it with the current trace, serialise.
- **Exactly:** `{'ts', 'level', 'msg', **fields}`; adds `trace_id`/`span_id` if inside a span; `json.dumps(sort_keys=True, default=str)`.
- **Why it is here:** Sorted keys make logs diffable; `default=str` avoids crashes on odd values.

### 21. Metrics: counters and samples  (`ragkit/observability.py` lines 99–108)
```python
class Metrics:
    def __init__(self):
        self.counters: dict[str, float] = {}
        self.samples: dict[str, list[float]] = {}

    def inc(self, name: str, n: float = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + n

    def observe(self, name: str, value: float) -> None:
        self.samples.setdefault(name, []).append(value)
```
- **In simple words:** Numbers by name.
- **Exactly:** `counters` dict and `samples` dict of lists; `inc(name, n)` and `observe(name, value)`.
- **Why it is here:** Smallest useful metrics store.

### 22. Metrics.percentile: nearest rank  (`ragkit/observability.py` lines 110–115)
```python
    def percentile(self, name: str, p: float) -> float:
        """Nearest-rank percentile: the smallest value with at least p% of samples <= it."""
        xs = sorted(self.samples.get(name, []))
        if not xs:
            return float("nan")
        return xs[max(0, math.ceil(p / 100 * len(xs)) - 1)]
```
- **In simple words:** Sort and pick.
- **Exactly:** `xs[max(0, ceil(p/100*n) - 1)]`; `nan` if no samples. For 1..100: p50=50, p95=95.
- **Why it is here:** Defines the percentile precisely; interviewers ask.

### 23. rate() and snapshot()  (`ragkit/observability.py` lines 117–128)
```python
    def rate(self, hit: str, miss: str) -> float:
        h, m = self.counters.get(hit, 0), self.counters.get(miss, 0)
        return h / (h + m) if h + m else 0.0

    def snapshot(self) -> dict:
        out = dict(self.counters)
        for k in self.samples:
            out[f"{k}.p50"], out[f"{k}.p95"] = self.percentile(k, 50), self.percentile(k, 95)
        return out


# ------------------------------------------------------------------ tracing decorators (same Protocols as the originals)
```
- **In simple words:** Hit rate and a flat dict of everything.
- **Exactly:** `rate(hit, miss) = h / (h + m)`; `snapshot` adds `<name>.p50` and `<name>.p95` for every sample series.
- **Why it is here:** What a /metrics endpoint would return.

### 24. TracedRetriever  (`ragkit/observability.py` lines 129–137)
```python
class TracedRetriever:
    def __init__(self, inner, tracer: Tracer):
        self.inner, self.tracer = inner, tracer

    def retrieve(self, query, k=10):
        with self.tracer.span("retrieve", k=k) as s:
            hits = self.inner.retrieve(query, k)
            s.attrs["n_hits"] = len(hits)
            return hits
```
- **In simple words:** Span around retrieval.
- **Exactly:** `span('retrieve', k=k)`; after the call sets `n_hits`.
- **Why it is here:** Same `Retriever` Protocol: a Decorator.

### 25. TracedReranker  (`ragkit/observability.py` lines 140–148)
```python
class TracedReranker:
    def __init__(self, inner, tracer: Tracer):
        self.inner, self.tracer = inner, tracer

    def rerank(self, query, hits, top_n):
        with self.tracer.span("rerank", n_in=len(hits)) as s:
            out = self.inner.rerank(query, hits, top_n)
            s.attrs["best_score"] = round(out[0].score, 3) if out else None
            return out
```
- **In simple words:** Span around reranking.
- **Exactly:** `span('rerank', n_in=...)`; sets `best_score`.
- **Why it is here:** Best score over time is a drift signal.

### 26. TracedGenerator: GenAI attributes  (`ragkit/observability.py` lines 151–162)
```python
class TracedGenerator:
    def __init__(self, inner, tracer: Tracer, model: str = "unknown", provider: str = "unknown"):
        self.inner, self.tracer, self.model, self.provider = inner, tracer, model, provider
        self.count = WordTokenCounter().count

    def generate(self, system, user):
        with self.tracer.span(f"chat {self.model}", **{GENAI["op"]: "chat", GENAI["model"]: self.model,
                                                       GENAI["provider"]: self.provider}) as s:
            s.attrs[GENAI["in_tok"]] = self.count(system + " " + user)
            out = self.inner.generate(system, user)
            s.attrs[GENAI["out_tok"]] = self.count(out)
            return out
```
- **In simple words:** A span named like the OTel convention.
- **Exactly:** Span `'chat <model>'` with `gen_ai.operation.name='chat'`, request model, provider name; input tokens before the call, output tokens after (word-count estimates).
- **Why it is here:** Standard names let any OTel backend chart tokens by model.

### 27. TracedAnswerer: constructor  (`ragkit/observability.py` lines 165–170)
```python
class TracedAnswerer:
    """Root span per question; feeds metrics and logs; runs online-eval hooks after the answer exists."""

    def __init__(self, inner, tracer: Tracer, metrics: Metrics, logger: JsonLogger | None = None, hooks=()):
        self.inner, self.tracer, self.metrics, self.logger, self.hooks = inner, tracer, metrics, logger, list(hooks)
        self.last_trace_id: str | None = None
```
- **In simple words:** The root-span wrapper.
- **Exactly:** Takes the inner answerer, tracer, metrics, optional logger and eval hooks; remembers `last_trace_id`.
- **Why it is here:** The UI needs the trace id to attach feedback.

### 28. TracedAnswerer.answer: root span  (`ragkit/observability.py` lines 172–177)
```python
    def answer(self, question: str) -> Answer:
        with self.tracer.span("rag.answer") as root:
            self.last_trace_id = root.trace_id
            ans = self.inner.answer(question)
            root.attrs["refused"] = ans.refused
            root.attrs["cache"] = ans.meta.get("cache", "n/a")
```
- **In simple words:** Time the whole request.
- **Exactly:** `span('rag.answer')`; copies `refused` and `cache` into attributes.
- **Why it is here:** One root per request.

### 29. TracedAnswerer.answer: metrics  (`ragkit/observability.py` lines 178–183)
```python
        self.metrics.inc("requests")
        self.metrics.observe("latency_ms", root.duration_ms)
        self.metrics.inc("refusals", ans.refused)
        cache = ans.meta.get("cache")
        if cache:
            self.metrics.inc("cache_hit" if cache != "miss" else "cache_miss")
```
- **In simple words:** Count and time every request.
- **Exactly:** `inc('requests')`, `observe('latency_ms', root.duration_ms)`, `inc('refusals', ans.refused)`, and `cache_hit`/`cache_miss` from the answer's `cache` label.
- **Why it is here:** Metrics are computed *after* the span closes so the duration is complete.

### 30. TracedAnswerer.answer: eval hooks and log  (`ragkit/observability.py` lines 184–193)
```python
        flags = {}
        for hook in self.hooks:                                # online eval: cheap checks on every answer
            flags.update(hook.evaluate(question, ans, root.duration_ms))
        for k, v in flags.items():
            if v:
                self.metrics.inc(f"flag.{k}")
        if self.logger:
            self.logger.log("info", "answered", trace_id=root.trace_id, refused=ans.refused,
                            latency_ms=round(root.duration_ms, 2), flags=flags)
        return ans
```
- **In simple words:** Run each hook; count flags; write the log.
- **Exactly:** `flags.update(hook.evaluate(question, ans, latency))`; each true flag -> `inc('flag.<name>')`; log line with `trace_id`, `refused`, `latency_ms`, `flags`.
- **Why it is here:** Online quality signals on every answer.

### 31. EvalHook Protocol  (`ragkit/observability.py` lines 196–197)
```python
class EvalHook(Protocol):
    def evaluate(self, question: str, answer: Answer, latency_ms: float) -> dict[str, bool]: ...
```
- **In simple words:** The hook contract.
- **Exactly:** `evaluate(question, answer, latency_ms) -> dict[str, bool]`.
- **Why it is here:** New checks need no change to `TracedAnswerer`.

### 32. CitationHook  (`ragkit/observability.py` lines 200–204)
```python
class CitationHook:
    """Flag answers that are not refusals but have no valid, supported citation."""
    def evaluate(self, question, answer, latency_ms):
        ok = answer.refused or (bool(answer.citations) and all(c.supported for c in answer.citations))
        return {"bad_citation": not ok}
```
- **In simple words:** Flag answers with missing/unsupported citations.
- **Exactly:** OK if refused, or citations non-empty and all supported; returns `{'bad_citation': not ok}`.
- **Why it is here:** Cheap proxy for grounding.

### 33. SlowHook  (`ragkit/observability.py` lines 207–212)
```python
class SlowHook:
    def __init__(self, limit_ms: float):
        self.limit = limit_ms

    def evaluate(self, question, answer, latency_ms):
        return {"slow": latency_ms > self.limit}
```
- **In simple words:** Flag slow answers.
- **Exactly:** `{'slow': latency_ms > limit}`.
- **Why it is here:** Alert on the flag rate, not single events.

### 34. Feedback record  (`ragkit/observability.py` lines 215–219)
```python
@dataclass
class Feedback:
    trace_id: str
    rating: int                                                # +1 thumbs up, -1 thumbs down
    comment: str = ""
```
- **In simple words:** One thumbs-up/down.
- **Exactly:** `trace_id`, `rating`, `comment`.
- **Why it is here:** Join key to everything else.

### 35. FeedbackStore  (`ragkit/observability.py` lines 222–234)
```python
class FeedbackStore:
    """Join user feedback to the trace that produced the answer -> a labelled example for offline eval."""

    def __init__(self):
        self.items: list[Feedback] = []

    def record(self, trace_id: str, rating: int, comment: str = "") -> None:
        if rating not in (-1, 1):
            raise ValueError("rating must be +1 or -1")
        self.items.append(Feedback(trace_id, rating, comment))

    def negative_trace_ids(self) -> list[str]:
        return [f.trace_id for f in self.items if f.rating < 0]
```
- **In simple words:** Collect feedback and list the bad ones.
- **Exactly:** `record` rejects ratings other than +1/-1; `negative_trace_ids()` returns the thumbs-down traces.
- **Why it is here:** Source of new regression cases.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `@contextmanager` + `yield` | Turns a generator into a `with`-block manager; code after `yield` runs on exit (use `try/finally`) |
| `ContextVar.set()` / `.reset(token)` | Set a context-local value and restore the previous one |
| `raise` inside `except` | Record the error and re-raise unchanged |
| `json.dumps(..., default=str)` | Fallback conversion for non-JSON types |
| `math.ceil` | Nearest-rank percentile index |
| `Protocol` for hooks | Structural interface: any object with `evaluate` is a hook |

## Libraries & functions
| Name | What it does here |
|---|---|
| `contextvars` | Per-task/thread context for the current span |
| `time.perf_counter` | High-resolution monotonic clock for spans |
| `json` | Structured log lines |
| `math` | Percentile index |
| OpenTelemetry GenAI conventions (reference only) | Attribute names; the OTel SDK is not installed or used here |

## Execution trace (real run)
```text
'How many days do I have to get a refund?'       refused=False cache=miss
'How many days do I have to get a refund?'       refused=False cache=exact
'Is shipping free on my order?'                  refused=False cache=miss
'What is the capital of Mongolia?'               refused=True  cache=miss
'Is anyone there to help me on Sunday?'          refused=False cache=miss
'how many days do i have to get a refund'        refused=False cache=semantic(1.00)

trace tree of the first request (name, duration, attributes):
rag.answer               18.62 ms  refused=False cache=miss
  retrieve               16.10 ms  k=20 n_hits=20
  rerank                  0.64 ms  n_in=20 best_score=0.5
  chat mock-llm           0.55 ms  gen_ai.operation.name=chat gen_ai.request.model=mock-llm gen_ai.provider.name=local-sim gen_ai.usage.input_tokens=223 gen_ai.usage.output_tokens=27

the 2nd request (exact cache hit) has only the root span:
rag.answer                0.04 ms  refused=False cache=exact

metrics: {'cache_hit': 2, 'cache_miss': 4, 'flag.slow': 1, 'latency_ms.p50': 0.712, 'latency_ms.p95': 18.62, 'refusals': 1, 'requests': 6}
cache hit rate (hits/(hits+misses)) = 0.33
LLM calls actually made: 3 of 6 requests | input tokens 557 output tokens 70 (word-count estimates)
GenAI attributes on a chat span: {"gen_ai.operation.name": "chat", "gen_ai.provider.name": "local-sim", "gen_ai.request.model": "mock-llm", "gen_ai.usage.input_tokens": 223, "gen_ai.usage.output_tokens": 27}

first JSON log line: {"flags": {"bad_citation": false, "slow": true}, "latency_ms": 18.62, "level": "info", "msg": "answered", "refused": false, "trace_id": "trace-0e3f5e028a69", "ts": 1793869389.607}
log line carries the trace id of its request: True

thumbs-down trace ids: ['trace-f1ba4c90004d'] -> [('answered', False, {'bad_citation': False, 'slow': False})]
```
- Six requests, `refused=True` only for the Mongolia question; labels: `miss`, `exact`, `miss`, `miss`, `miss`, `semantic(1.00)`.
- The miss tree shows the three child spans with attributes; the exact hit is a single root span.
- Metrics: 2 hits and 4 misses (the refusal counts as a miss), hit rate 0.33, one `slow` flag (the first, cold request builds the BM25 index), LLM called 3 times for 6 requests.
- The GenAI attribute dictionary on a `chat` span uses the OpenTelemetry names.
- The log line's `trace_id` equals the first request's trace id: logs and traces join.
- The thumbs-down trace resolves to its `answered` log record.
(Durations differ run to run; counts do not.)

## Experiments (change ONE thing)
Each snippet runs from this step folder (`cd` into it first). Outputs below are from real runs.

### Why p95 and not the mean
```python
from ragkit.observability import Metrics
m = Metrics(); data = [10] * 94 + [900] * 6
for x in data: m.observe("lat", x)
print("mean", sum(data) / len(data), "| p50", m.percentile("lat", 50), "| p95", m.percentile("lat", 95), "| p99", m.percentile("lat", 99))
```
Observed:
```text
mean 63.4 | p50 10 | p95 900 | p99 900
```
The mean (63.4 ms) hides that 6% of users wait 900 ms; p95 exposes it. Use the tail percentile for SLOs.

### Nesting: children join the parent's trace
```python
from ragkit.observability import Tracer
t = Tracer()
with t.span("a") as a:
    with t.span("b") as b:
        pass
with t.span("c") as c:
    pass
print("a/b same trace:", a.trace_id == b.trace_id, "| b.parent is a:", b.parent_id == a.span_id, "| c new trace:", c.trace_id != a.trace_id)
```
Observed:
```text
a/b same trace: True | b.parent is a: True | c new trace: True
```
`b` is a child of `a` in the same trace; `c`, started after `a` ended, is a new trace. The context variable did the nesting; no span was passed as an argument.

### contextvars do not cross into plain threads
```python
from concurrent.futures import ThreadPoolExecutor
import contextvars
from ragkit.observability import Tracer
t = Tracer()
with t.span("root") as root:
    in_thread = ThreadPoolExecutor(1).submit(Tracer.current).result()
    ctx = contextvars.copy_context()
    copied = ThreadPoolExecutor(1).submit(ctx.run, Tracer.current).result()
print("plain thread sees current span:", in_thread is not None, "| thread with copied context sees it:", copied is root)
```
Observed:
```text
plain thread sees current span: False | thread with copied context sees it: True
```
A plain worker thread sees no current span (`False`), so spans created there become orphan roots; with `copy_context().run` the thread sees the parent (`True`). Remember this when you add thread pools (e.g. parallel retrieval) or `run_in_executor`.

## Debugging practice
**BREAK IT:**
```python
import sys; sys.path.insert(0, '..')

from ragkit.observability import FeedbackStore
FeedbackStore().record("trace-1", 5)
```
Real output:
```text
Traceback (most recent call last):
  File "<string>", line 4, in <module>
  File "/Users/aman/Desktop/Project_resume/RAP_interview/20_llm_system_lld/09_observability_and_eval_hooks/../ragkit/observability.py", line 230, in record
    raise ValueError("rating must be +1 or -1")
ValueError: rating must be +1 or -1
```
`ValueError: rating must be +1 or -1`, raised by `FeedbackStore.record` (the guard we wrote). Validate at the boundary: a UI bug that sends `5` should fail loudly in the store, not silently skew your 'negative feedback' count. Read the traceback bottom-up: the message states the rule; the frame shows the exact line of the guard.

## Rebuild checklist
- [ ] Write a `Tracer.span` context manager with nesting through a `ContextVar`, recording errors and re-raising.
- [ ] Write nearest-rank `percentile` and compute p95 of 1..100.
- [ ] Write a JSON logger that tags records with the current trace id.
- [ ] List the metrics you would export for a RAG service and which one you would alert on.
- [ ] Name four `gen_ai.*` attributes, and state the standard's status and the caveat.
- [ ] Explain how a thumbs-down becomes a regression test.

## The new problem (cliffhanger)
Every piece exists: domain, ingestion, query, patterns, resilience, caching, gateway, observability. In an LLD interview you are asked to *design* with these pieces on a whiteboard in 45 minutes: requirements, entities, interfaces, diagrams, edge cases, tests. [Step 10](../10_lld_interview_problems/WALKTHROUGH.md) works through six classic problems.

> **Say it aloud (interview box):** "I instrument by wrapping: a tracer with nested spans via a context variable, JSON logs stamped with the trace id, metrics with p50 and p95 rather than means, GenAI attribute names from the OpenTelemetry conventions - which are still in Development status, so I pin versions - and cheap online eval hooks on every answer. User feedback is joined to the trace and becomes regression cases."
