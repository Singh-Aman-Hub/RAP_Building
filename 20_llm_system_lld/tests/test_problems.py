import pytest
from pydantic import BaseModel

from ragkit.agent import AgentRuntime, ToolRegistry
from ragkit.domain import Answer, Citation
from ragkit.evalkit import (CitedGoldScorer, ContainsScorer, EvalCase, EvalRunner, RefusalScorer, RetrievalHitScorer,
                            regressions)
from ragkit.gateway import Gateway, ScriptedProvider, ToolCall, ToolSpec
from ragkit.memory import ConversationMemory
from ragkit.ratelimit import LLMRateLimiter, RateLimitedGenerator, TenantLimiters
from ragkit.testing import FakeClock


# ---------------- rate limiter
def test_rpm_limit_blocks_then_refills():
    clock = FakeClock()
    rl = LLMRateLimiter(rpm=60, tpm=10_000, clock=clock)           # 1 request/second, burst 60
    assert sum(rl.try_acquire(10) for _ in range(70)) == 60
    clock.advance(2)
    assert rl.try_acquire(10) and rl.try_acquire(10) and not rl.try_acquire(10)


def test_tpm_limit_independent_of_rpm():
    rl = LLMRateLimiter(rpm=1000, tpm=600, clock=FakeClock())
    assert rl.try_acquire(500) and not rl.try_acquire(500)         # tokens exhausted although requests remain


def test_failed_acquire_spends_nothing():
    rl = LLMRateLimiter(rpm=2, tpm=100, clock=FakeClock())
    assert not rl.try_acquire(500)
    assert rl.requests.tokens == 2


def test_acquire_waits_on_fake_clock_and_settle_refunds():
    clock = FakeClock()
    rl = LLMRateLimiter(rpm=60, tpm=600, clock=clock)              # tokens refill at 10/s
    rl.acquire(600)
    assert rl.acquire(100) == pytest.approx(10.0, rel=1e-3)
    rl.settle(100, 40)
    assert rl.tokens.tokens == pytest.approx(60, abs=1e-3)


def test_tenant_isolation():
    t = TenantLimiters(rpm=1, tpm=1000, clock=FakeClock())
    assert t.for_tenant("a").try_acquire(1) and not t.for_tenant("a").try_acquire(1)
    assert t.for_tenant("b").try_acquire(1)


def test_rate_limited_generator():
    class G:
        def generate(self, s, u):
            return "out"
    clock = FakeClock()
    g = RateLimitedGenerator(G(), LLMRateLimiter(rpm=60, tpm=60_000, clock=clock))
    for _ in range(61):
        g.generate("s", "u")
    assert g.total_wait > 0


# ---------------- memory
def test_memory_window_trims_oldest():
    m = ConversationMemory(max_tokens=10, clock=FakeClock())
    for i in range(6):
        m.add("s", "user", f"message number {i} here")             # 4 words = 6 tokens each
    w = m.window("s")
    assert len(w) == 1 and "5" in w[0]["content"]


def test_memory_summary_and_isolation_and_ttl():
    clock = FakeClock()
    m = ConversationMemory(max_tokens=10, ttl=60, clock=clock,
                           summarizer=lambda old, ev: (old + " | " if old else "") + "; ".join(t.content for t in ev))
    m.add("a", "user", "first message of ours"); m.add("a", "assistant", "second message of ours")
    w = m.window("a")
    assert w[0]["role"] == "system" and "first message" in w[0]["content"]
    assert m.window("b") == []
    clock.advance(61)
    assert m.window("a") == []


def test_memory_keeps_at_least_latest_turn():
    m = ConversationMemory(max_tokens=1, clock=FakeClock())
    m.add("s", "user", "a very long single message exceeding the budget")
    assert len(m.window("s")) == 1


def test_memory_lru_sessions():
    m = ConversationMemory(max_sessions=2, clock=FakeClock())
    for s in "abc":
        m.add(s, "user", "hello")
    assert m.window("a") == [] and len(m.window("c")) == 1


# ---------------- eval
class FakeAnswerer:
    def answer(self, q):
        if "gift" in q:
            return Answer("I don't know", refused=True, refusal_reason="x")
        if q == "crash":
            raise RuntimeError("boom")
        return Answer("Refunds take 9 days [1]", (Citation(1, "c1"),), context_chunk_ids=("c1", "c2"))


CASES = [EvalCase("q1", "refund days?", "9 days", ("c1",)), EvalCase("q2", "gift cards?", None, (), False),
         EvalCase("q3", "refund wrong?", "14 days", ("c2",))]


def test_eval_runner_scores_and_failures():
    rep = EvalRunner(FakeAnswerer(), [ContainsScorer(), RefusalScorer(), RetrievalHitScorer(), CitedGoldScorer()]).run(CASES)
    m = rep.means()
    assert m["contains"] == 0.5 and m["refusal_correct"] == 1.0
    assert m["gold_in_context"] == 1.0 and m["cites_gold"] == 0.5
    assert rep.failures("contains") == ["q3"]


def test_eval_runner_survives_crash_and_regressions():
    base = EvalRunner(FakeAnswerer(), [ContainsScorer()]).run(CASES[:1])
    bad = EvalRunner(FakeAnswerer(), [ContainsScorer()]).run([EvalCase("q1", "crash", "9 days")])
    assert bad.rows[0]["error"] == 1.0 and regressions(base, bad, "contains") == ["q1"]


# ---------------- agent
class Args(BaseModel):
    city: str


def make_agent(replies, max_steps=5):
    reg = ToolRegistry()
    reg.register(ToolSpec.from_model("weather", "weather of a city", Args), lambda city: f"sunny in {city}")
    reg.register(ToolSpec.from_model("broken", "always fails", Args), lambda city: 1 / 0)
    return AgentRuntime(Gateway({"default": [ScriptedProvider("p", replies)]}), reg, max_steps=max_steps)


def test_agent_tool_then_final():
    r = make_agent([("", [ToolCall("1", "weather", {"city": "Pune"})]), "It is sunny in Pune."]).run("weather?")
    assert r.stopped == "final" and r.steps == 2 and "sunny in Pune" in r.transcript[0]


def test_agent_tool_errors_are_fed_back_not_raised():
    r = make_agent([("", [ToolCall("1", "weather", {})]), ("", [ToolCall("2", "nope", {})]),
                    ("", [ToolCall("3", "broken", {"city": "x"})]), "done"]).run("go")
    assert r.stopped == "final" and sum("ERROR" in line for line in r.transcript) == 3


def test_agent_loop_guard_and_step_limit():
    same = ("", [ToolCall("1", "weather", {"city": "A"})])
    assert make_agent([same, same]).run("x").stopped == "loop"
    varied = [("", [ToolCall(str(i), "weather", {"city": str(i)})]) for i in range(10)]
    r = make_agent(varied, max_steps=3).run("x")
    assert r.stopped == "max_steps" and r.steps == 3
