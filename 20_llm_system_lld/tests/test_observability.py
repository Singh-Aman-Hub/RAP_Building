import json

import pytest

from ragkit.domain import Answer, Citation
from ragkit.observability import (GENAI, CitationHook, FeedbackStore, JsonLogger, Metrics, SlowHook, TracedAnswerer,
                                  TracedGenerator, Tracer)


class Ticker:
    """Deterministic clock: each call advances 1 ms."""
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 0.001
        return self.t


def test_spans_nest_and_share_trace():
    tr = Tracer(Ticker())
    with tr.span("root") as r:
        with tr.span("child") as c:
            with tr.span("grandchild") as g:
                pass
    assert c.parent_id == r.span_id and g.parent_id == c.span_id
    assert len({s.trace_id for s in tr.spans}) == 1 and r.parent_id is None
    assert Tracer.current() is None                        # context restored after the with-blocks


def test_separate_roots_get_separate_traces():
    tr = Tracer(Ticker())
    with tr.span("a") as a:
        pass
    with tr.span("b") as b:
        pass
    assert a.trace_id != b.trace_id


def test_error_recorded_and_reraised():
    tr = Tracer(Ticker())
    with pytest.raises(ValueError):
        with tr.span("boom") as s:
            raise ValueError("bad")
    assert s.status == "error" and "ValueError: bad" in s.error and s.end is not None


def test_tree_render():
    tr = Tracer(Ticker())
    with tr.span("root"):
        with tr.span("kid", n=2):
            pass
    out = tr.tree().splitlines()
    assert out[0].startswith("root") and out[1].startswith("  kid") and "n=2" in out[1]


def test_json_log_has_trace_ids_inside_span():
    tr, lg = Tracer(Ticker()), JsonLogger(clock=lambda: 1.0)
    with tr.span("x") as s:
        line = lg.log("info", "hello", k=1)
    rec = json.loads(line)
    assert rec["trace_id"] == s.trace_id and rec["span_id"] == s.span_id and rec["k"] == 1
    assert "trace_id" not in json.loads(lg.log("info", "outside"))


def test_percentiles_nearest_rank():
    m = Metrics()
    for v in range(1, 101):
        m.observe("lat", v)
    assert m.percentile("lat", 50) == 50 and m.percentile("lat", 95) == 95 and m.percentile("lat", 100) == 100
    assert m.percentile("nothing", 50) != m.percentile("nothing", 50)     # NaN


def test_hit_rate_and_snapshot():
    m = Metrics()
    m.inc("h", 3); m.inc("m")
    m.observe("lat", 5)
    assert m.rate("h", "m") == 0.75 and m.snapshot()["lat.p95"] == 5


def test_traced_generator_sets_genai_attributes():
    tr = Tracer(Ticker())

    class G:
        def generate(self, s, u):
            return "four words in reply"
    TracedGenerator(G(), tr, model="m1", provider="mock").generate("sys", "user text")
    s = tr.spans[0]
    assert s.name == "chat m1" and s.attrs[GENAI["op"]] == "chat" and s.attrs[GENAI["out_tok"]] >= 1
    assert s.attrs[GENAI["in_tok"]] >= 1


def test_traced_answerer_metrics_hooks_feedback():
    class Inner:
        def answer(self, q):
            if q == "refuse":
                return Answer("no", refused=True, refusal_reason="x")
            if q == "nocite":
                return Answer("text without citation")
            return Answer("ok [1]", (Citation(1, "c"),), meta={"cache": "miss"})
    m, tr = Metrics(), Tracer(Ticker())
    ta = TracedAnswerer(Inner(), tr, m, JsonLogger(), [CitationHook(), SlowHook(0.0001)])
    for q in ("good", "refuse", "nocite"):
        ta.answer(q)
    snap = m.snapshot()
    assert snap["requests"] == 3 and snap["refusals"] == 1 and snap["flag.bad_citation"] == 1
    assert snap["flag.slow"] == 3 and snap["cache_miss"] == 1

    fb = FeedbackStore()
    fb.record(ta.last_trace_id, -1, "wrong")
    assert fb.negative_trace_ids() == [ta.last_trace_id]
    with pytest.raises(ValueError):
        fb.record("t", 5)
