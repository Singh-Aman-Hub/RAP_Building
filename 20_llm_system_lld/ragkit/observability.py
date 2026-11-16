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
GENAI = {"op": "gen_ai.operation.name", "model": "gen_ai.request.model", "provider": "gen_ai.provider.name",
         "in_tok": "gen_ai.usage.input_tokens", "out_tok": "gen_ai.usage.output_tokens"}

_current: contextvars.ContextVar["Span | None"] = contextvars.ContextVar("current_span", default=None)


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


class Tracer:
    """A span is a timed, named unit of work with attributes. Nesting is automatic via a contextvar."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter):
        self.clock, self.spans = clock, []

    @contextmanager
    def span(self, name: str, **attrs) -> Iterator[Span]:
        parent = _current.get()
        s = Span(name, parent.trace_id if parent else new_id("trace"), new_id("sp"),
                 parent.span_id if parent else None, self.clock(), attrs=dict(attrs))
        token = _current.set(s)                                # children created inside the `with` see s as parent
        try:
            yield s
        except Exception as exc:
            s.status, s.error = "error", f"{type(exc).__name__}: {exc}"
            raise                                              # record, never swallow
        finally:
            s.end = self.clock()
            _current.reset(token)                              # restore the parent as current
            self.spans.append(s)

    @staticmethod
    def current() -> Span | None:
        return _current.get()

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


class JsonLogger:
    """One JSON object per line, auto-tagged with the current trace/span ids so logs and traces join."""

    def __init__(self, sink: list | None = None, clock: Callable[[], float] = time.time):
        self.sink, self.clock = sink if sink is not None else [], clock

    def log(self, level: str, msg: str, **fields) -> str:
        cur = _current.get()
        rec = {"ts": round(self.clock(), 3), "level": level, "msg": msg, **fields}
        if cur:
            rec["trace_id"], rec["span_id"] = cur.trace_id, cur.span_id
        line = json.dumps(rec, sort_keys=True, default=str)
        self.sink.append(line)
        return line


class Metrics:
    def __init__(self):
        self.counters: dict[str, float] = {}
        self.samples: dict[str, list[float]] = {}

    def inc(self, name: str, n: float = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + n

    def observe(self, name: str, value: float) -> None:
        self.samples.setdefault(name, []).append(value)

    def percentile(self, name: str, p: float) -> float:
        """Nearest-rank percentile: the smallest value with at least p% of samples <= it."""
        xs = sorted(self.samples.get(name, []))
        if not xs:
            return float("nan")
        return xs[max(0, math.ceil(p / 100 * len(xs)) - 1)]

    def rate(self, hit: str, miss: str) -> float:
        h, m = self.counters.get(hit, 0), self.counters.get(miss, 0)
        return h / (h + m) if h + m else 0.0

    def snapshot(self) -> dict:
        out = dict(self.counters)
        for k in self.samples:
            out[f"{k}.p50"], out[f"{k}.p95"] = self.percentile(k, 50), self.percentile(k, 95)
        return out


# ------------------------------------------------------------------ tracing decorators (same Protocols as the originals)
class TracedRetriever:
    def __init__(self, inner, tracer: Tracer):
        self.inner, self.tracer = inner, tracer

    def retrieve(self, query, k=10):
        with self.tracer.span("retrieve", k=k) as s:
            hits = self.inner.retrieve(query, k)
            s.attrs["n_hits"] = len(hits)
            return hits


class TracedReranker:
    def __init__(self, inner, tracer: Tracer):
        self.inner, self.tracer = inner, tracer

    def rerank(self, query, hits, top_n):
        with self.tracer.span("rerank", n_in=len(hits)) as s:
            out = self.inner.rerank(query, hits, top_n)
            s.attrs["best_score"] = round(out[0].score, 3) if out else None
            return out


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


class TracedAnswerer:
    """Root span per question; feeds metrics and logs; runs online-eval hooks after the answer exists."""

    def __init__(self, inner, tracer: Tracer, metrics: Metrics, logger: JsonLogger | None = None, hooks=()):
        self.inner, self.tracer, self.metrics, self.logger, self.hooks = inner, tracer, metrics, logger, list(hooks)
        self.last_trace_id: str | None = None

    def answer(self, question: str) -> Answer:
        with self.tracer.span("rag.answer") as root:
            self.last_trace_id = root.trace_id
            ans = self.inner.answer(question)
            root.attrs["refused"] = ans.refused
            root.attrs["cache"] = ans.meta.get("cache", "n/a")
        self.metrics.inc("requests")
        self.metrics.observe("latency_ms", root.duration_ms)
        self.metrics.inc("refusals", ans.refused)
        cache = ans.meta.get("cache")
        if cache:
            self.metrics.inc("cache_hit" if cache != "miss" else "cache_miss")
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


class EvalHook(Protocol):
    def evaluate(self, question: str, answer: Answer, latency_ms: float) -> dict[str, bool]: ...


class CitationHook:
    """Flag answers that are not refusals but have no valid, supported citation."""
    def evaluate(self, question, answer, latency_ms):
        ok = answer.refused or (bool(answer.citations) and all(c.supported for c in answer.citations))
        return {"bad_citation": not ok}


class SlowHook:
    def __init__(self, limit_ms: float):
        self.limit = limit_ms

    def evaluate(self, question, answer, latency_ms):
        return {"slow": latency_ms > self.limit}


@dataclass
class Feedback:
    trace_id: str
    rating: int                                                # +1 thumbs up, -1 thumbs down
    comment: str = ""


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
