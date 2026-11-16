import pytest

from ragkit.gateway import BudgetExceeded, LLMResponse, Message, Role, Usage
from ragkit.governance import ContextBudget, Ledger, LedgerRow, TenantGate
from ragkit.resilience import PermanentError, RateLimitedError
from ragkit.testing import FakeClock


def turns(n):
    return [Message(Role.USER, f"turn {i} " + "word " * 20) for i in range(n)]


def test_context_plan_keeps_newest_and_reserves_answer():
    cb = ContextBudget(window=300, answer_reserve=100)
    plan = cb.plan("system prompt here", "question?", turns(10))
    assert plan.history_kept[-1].content.startswith("turn 9")
    assert plan.history_dropped == 10 - len(plan.history_kept) > 0
    assert plan.prompt_budget == 300 - 100 - plan.history_tokens


def test_context_plan_no_history_gives_all_room():
    plan = ContextBudget(300, 100).plan("sys", "q", [])
    assert plan.history_tokens == 0 and plan.prompt_budget == 200


def test_context_too_small_raises_permanent():
    with pytest.raises(PermanentError, match="context_window_too_small"):
        ContextBudget(window=50, answer_reserve=40).plan("word " * 40, "q", [])


def test_output_limit_is_min_of_ask_cap_and_room():
    cb = ContextBudget(window=500, answer_reserve=100)
    assert cb.output_limit(100, 50) == 50
    assert cb.output_limit(100, 9999, hard_cap=300) == 300
    assert cb.output_limit(450, 300) == 50
    assert cb.output_limit(600, 300) == 0


def resp(model="small", i=100, o=50):
    return LLMResponse("x", Usage(i, o), "p", model)


def test_gate_rate_limit_per_tenant_and_refill():
    clock = FakeClock()
    g = TenantGate(60, 600, 1.0, clock)
    g.admit("a", 400)
    with pytest.raises(RateLimitedError) as e:
        g.admit("a", 400)
    assert e.value.retry_after > 0
    g.admit("b", 400)                                   # other tenant has its own bucket
    clock.advance(30)
    g.admit("a", 400)


def test_gate_request_bigger_than_bucket_is_permanent():
    with pytest.raises(PermanentError):
        TenantGate(60, 600, 1.0, FakeClock()).admit("a", 601)


def test_gate_spend_ceiling_and_settle_refunds_overreservation():
    g = TenantGate(600, 6000, 0.0001, FakeClock(), prices={"large": (1000.0, 1000.0)})
    g.admit("a", 1000)
    before = g.limiters.for_tenant("a").tokens.tokens
    cost = g.settle("a", 1000, resp("large", 20, 10))
    assert cost == pytest.approx(30 * 1000 / 1e6)
    assert g.limiters.for_tenant("a").tokens.tokens > before        # unused 970 tokens given back
    with pytest.raises(BudgetExceeded):
        g.admit("a", 10)
    g.admit("other", 10)


def test_ledger_aggregates():
    L = Ledger()
    L.add(LedgerRow("1", "t1", "stable", "f", 10, 5, 0.5, "ok"))
    L.add(LedgerRow("2", "t1", "canary", "g", 10, 5, 1.0, "ok"))
    L.add(LedgerRow("3", "t2", "stable", "f", 0, 0, 0.0, "blocked_input"))
    assert L.by_tenant() == {"t1": 1.5, "t2": 0.0}
    assert L.by_release()["stable"] == (2, 0.5)
