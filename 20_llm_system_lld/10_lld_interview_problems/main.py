"""
GOAL:  Six worked LLD problems (see WALKTHROUGH.md for requirements -> entities -> interfaces -> diagrams -> edge cases),
       with a running mini-demo of each on top of ragkit: (1) a RAG service wired from config with cache, retry, tracing;
       (2) a chunker library; (3) an LLM rate limiter; (4) an eval harness on module 14's 29 questions;
       (5) a conversation memory store; (6) a tool-calling agent runtime.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 20_llm_system_lld/10_lld_interview_problems/main.py
EXPECTED OUTPUT (shortened):
    1) service: ... cache=miss then exact | retries=1 | trace spans ..
    2) chunker library: heading=.. fixed=.. sentence=..
    3) rate limiter: 60 allowed in burst, then waits
    4) eval: contains=.. gold_in_context=.. refusal_correct=..
    5) memory: window trimmed ... summary kept
    6) agent: step 1 search_docs -> ok ... step 2 final answer
NEEDS: ragkit + module 14 data. No models (HashingEmbedder), no network.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pydantic import BaseModel

from ragkit.agent import AgentRuntime, ToolRegistry
from ragkit.cache import CachedAnswerer, TTLLRUCache
from ragkit.chunking import FixedWordChunker, HeadingChunker, SentenceChunker
from ragkit.evalkit import (CitedGoldScorer, ContainsScorer, EvalCase, EvalRunner, RefusalScorer, RetrievalHitScorer,
                            regressions)
from ragkit.factory import CHUNKERS, build_system
from ragkit.gateway import Gateway, ScriptedProvider, ToolCall, ToolSpec
from ragkit.ingest import DirectoryLoader, MarkdownParser
from ragkit.memory import ConversationMemory
from ragkit.observability import Metrics, TracedAnswerer, Tracer
from ragkit.ratelimit import LLMRateLimiter, RateLimitedGenerator
from ragkit.resilience import RetryingGenerator, RetryPolicy, TransientError
from ragkit.testing import FakeClock

ROOT = Path(__file__).resolve().parents[2]
KB = ROOT / "14_rag_and_retrieval" / "data" / "kb"

# ======================= 1. Design a RAG service ==================================================================
system = build_system({"chunker": {"type": "heading", "max_words": 80}}, DirectoryLoader(KB))
system.pipeline.run()
retries = []
real_gen = system.service.generator
class Wrapped:                                                           # fails once on cue, then delegates
    def __init__(self):
        self.first = True
    def generate(self, s, u):
        if self.first:
            self.first = False
            raise TransientError("503")
        return real_gen.generate(s, u)
system.service.generator = RetryingGenerator(Wrapped(), RetryPolicy(max_attempts=3), FakeClock(), random.Random(1),
                                             on_retry=lambda n, e, d: retries.append(n))
tracer, metrics = Tracer(), Metrics()
app = TracedAnswerer(CachedAnswerer(system.service, TTLLRUCache(64, 600), scope=lambda: system.pipeline.index_version),
                     tracer, metrics)
for q in ["How many days do I have to get a refund?", "How many days do I have to get a refund?"]:
    a = app.answer(q)
    print(f"1) service: {a.text[:58]}... cache={a.meta['cache']} retries so far={len(retries)} | root spans={sum(s.name == 'rag.answer' for s in tracer.spans)}")

# ======================= 2. Design a document chunker library ====================================================
doc = next(d for d in DirectoryLoader(KB).load() if d.source == "refunds_and_returns.md")
elements = MarkdownParser().parse(doc)
for name in CHUNKERS.names():
    ch = CHUNKERS.create(name)
    chunks = ch.chunk(doc, elements)
    print(f"2) chunker {ch.name:28s} -> {len(chunks)} chunks, longest {max(len(c.text.split()) for c in chunks)} words")

# ======================= 3. Design a rate limiter for an LLM API ===============================================
clock = FakeClock()
limiter = LLMRateLimiter(rpm=60, tpm=6000, clock=clock)
allowed = sum(limiter.try_acquire(50) for _ in range(100))
print(f"3) rate limiter rpm=60 tpm=6000: {allowed} of 100 instant requests allowed (burst), tokens left {limiter.tokens.tokens:.0f}")
class Echo:
    def generate(self, s, u):
        return "ok"
rl = RateLimitedGenerator(Echo(), LLMRateLimiter(rpm=30, tpm=100_000, clock=clock), max_output_tokens=10)
for _ in range(40):
    rl.generate("s", "u")
print(f"   40 sequential calls at 30 rpm: simulated waiting {rl.total_wait:.1f} s (no real sleeping)")

# ======================= 4. Design an LLM eval harness ============================================================
by_key = {f"{Path(c.metadata['source']).stem}#{c.section}": c.chunk_id for c in system.store.all_chunks()}
raw = json.loads((ROOT / "14_rag_and_retrieval" / "data" / "eval_questions.json").read_text())
cases = [EvalCase(r["id"], r["question"], r.get("answer") if r["gold"] else None,
                  tuple(by_key[g] for g in r["gold"]), bool(r["gold"])) for r in raw]
scorers = [ContainsScorer(), RetrievalHitScorer(), CitedGoldScorer(), RefusalScorer()]
base = EvalRunner(system.service, scorers).run(cases)
print("4) eval on", len(cases), "questions:", {k: round(v, 2) for k, v in base.means().items() if v is not None})
print("   cases where the gold chunk never reached the prompt:", base.failures("gold_in_context"))
system.service.refusal.min_score = 0.6                                    # a "change" we want to gate in CI
new = EvalRunner(system.service, scorers).run(cases)
print("   after raising refusal threshold to 0.6 -> regressions on 'contains':", regressions(base, new, "contains")[:6],
      "| refusal_correct", round(base.mean("refusal_correct"), 2), "->", round(new.mean("refusal_correct"), 2))
system.service.refusal.min_score = 0.3

# ======================= 5. Design a conversation memory store =====================================================
mem = ConversationMemory(max_tokens=30, clock=FakeClock(),
                         summarizer=lambda old, ev: (old + " " if old else "") + "earlier: " + "; ".join(t.content[:25] for t in ev))
for i, text in enumerate(["I want to return my tea", "Which order is it?", "Order 1001, the masala blend",
                          "Is it sealed?", "Yes it is sealed", "Then you can get a refund"]):
    mem.add("sess-1", "user" if i % 2 == 0 else "assistant", text)
print("5) memory window for sess-1:")
for m in mem.window("sess-1"):
    print("     ", m["role"], "|", m["content"])
print("   other session is empty:", mem.window("sess-2") == [])

# ======================= 6. Design a tool-calling agent runtime ====================================================
class SearchArgs(BaseModel):
    query: str
    top_k: int = 2
def search_docs(query, top_k):
    hits = system.service.retriever.retrieve(query, top_k)
    return " | ".join(f"{h.chunk.section}: {h.chunk.text[:60]}" for h in hits)
tools = ToolRegistry()
tools.register(ToolSpec.from_model("search_docs", "Search the knowledge base", SearchArgs), search_docs)
script = [("", [ToolCall("c1", "search_docs", {"query": "refunds within days of delivery"})]),
          ("", [ToolCall("c2", "search_docs", {"top_k": 1})]),                     # bad arguments: the model sees the error
          ("", [ToolCall("c3", "calculator", {"x": 1})]),                            # unknown tool
          "Refunds are issued within 9 days of delivery."]
agent = AgentRuntime(Gateway({"default": [ScriptedProvider("scripted", script)]}), tools, max_steps=6)
res = agent.run("What is the refund window?")
print(f"6) agent stopped={res.stopped} after {res.steps} steps -> {res.text!r}")
for line in res.transcript:
    print("     ", line[:140])

# TRY IT: in problem 4 set refusal min_score to 0.0 and compare refusal_correct: which unanswerable questions leak through?
# TRY IT: in problem 6 use max_steps=2: read the 'max_steps' stop reason.
# BREAK IT: in problem 1 use RetryPolicy(max_attempts=1): the first TransientError reaches the caller. Where should it be caught?
