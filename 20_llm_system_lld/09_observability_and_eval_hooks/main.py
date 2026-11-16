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

# ---- 3. A trace tree: the first (cache-miss) request ------------------------------------------------------------------------
print("\ntrace tree of the first request (name, duration, attributes):")
print(tracer.tree(trace_list[0]))
print("\nthe 2nd request (exact cache hit) has only the root span:")
print(tracer.tree(trace_list[1]))

# ---- 4. Metrics ---------------------------------------------------------------------------------------------------------------
snap = metrics.snapshot()
print("\nmetrics:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in sorted(snap.items())})
print(f"cache hit rate (hits/(hits+misses)) = {metrics.rate('cache_hit', 'cache_miss'):.2f}")
gen_spans = [s for s in tracer.spans if s.name.startswith("chat ")]
tok_in = sum(s.attrs["gen_ai.usage.input_tokens"] for s in gen_spans)
tok_out = sum(s.attrs["gen_ai.usage.output_tokens"] for s in gen_spans)
print(f"LLM calls actually made: {len(gen_spans)} of {len(questions)} requests | input tokens {tok_in} output tokens {tok_out} (word-count estimates)")
print("GenAI attributes on a chat span:", json.dumps(gen_spans[0].attrs, sort_keys=True))

# ---- 5. Structured logs ------------------------------------------------------------------------------------------------------------
print("\nfirst JSON log line:", log_lines[0])
print("log line carries the trace id of its request:", json.loads(log_lines[0])["trace_id"] == trace_list[0])

# ---- 6. Feedback joined to traces -> labelled examples for offline eval --------------------------------------------------------------
fb = FeedbackStore()
fb.record(trace_ids["What is the capital of Mongolia?"], +1, "correct refusal")
fb.record(trace_ids["Is shipping free on my order?"], -1, "wrong threshold, I live in Pune")
bad = fb.negative_trace_ids()
bad_logs = [json.loads(l) for l in log_lines if json.loads(l)["trace_id"] in bad]
print("\nthumbs-down trace ids:", bad, "->", [(l["msg"], l["refused"], l["flags"]) for l in bad_logs])

# TRY IT: set SlowHook(limit_ms=0.01): how does flag.slow change, and is it a good alert on its own?
# TRY IT: add a second CachedAnswerer layer outside TracedAnswerer instead of inside: what does the trace tree lose?
# BREAK IT: remove `_current.reset(token)` in Tracer.span and run again: the trace tree nests every later request under the first.
