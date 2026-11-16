import random
import time

import pytest

from ragkit.resilience import (BreakerGenerator, CallTimeout, CircuitBreaker, CircuitOpenError, FallbackGenerator,
                               IdempotencyStore, PermanentError, RateLimitedError, RetryingGenerator, RetryPolicy,
                               State, TimeoutGenerator, TokenBucket, TransientError, call_with_timeout, idempotency_key,
                               retry_call)
from ragkit.testing import FakeClock, FlakyGenerator


def boom(exc):
    raise exc


def test_backoff_bounds_grow_and_cap():
    p, rng = RetryPolicy(base_delay=1, max_delay=8), random.Random(1)
    for attempt, cap in [(0, 1), (1, 2), (2, 4), (3, 8), (6, 8)]:
        assert all(0 <= p.backoff(attempt, rng) <= cap for _ in range(50))


def test_retry_succeeds_after_transients():
    clock = FakeClock()
    g = RetryingGenerator(FlakyGenerator([TransientError, TransientError]), RetryPolicy(max_attempts=4), clock)
    assert g.generate("s", "u") == "ok-reply" and g.inner.calls == 3 and len(clock.sleeps) == 2


def test_retry_gives_up_and_raises_last_error():
    g = RetryingGenerator(FlakyGenerator([TransientError] * 5), RetryPolicy(max_attempts=3), FakeClock())
    with pytest.raises(TransientError):
        g.generate("s", "u")
    assert g.inner.calls == 3


def test_permanent_error_not_retried():
    g = RetryingGenerator(FlakyGenerator([PermanentError]), clock=FakeClock())
    with pytest.raises(PermanentError):
        g.generate("s", "u")
    assert g.inner.calls == 1


def test_retry_after_header_is_respected():
    clock = FakeClock()
    calls = iter([RateLimitedError("429", retry_after=5.0), "done"])

    def fn():
        x = next(calls)
        if isinstance(x, Exception):
            raise x
        return x
    assert retry_call(fn, RetryPolicy(base_delay=0.1), clock, random.Random(0)) == "done"
    assert clock.sleeps[0] >= 5.0


def test_timeout_raises_calltimeout():
    with pytest.raises(CallTimeout):
        call_with_timeout(lambda: time.sleep(0.3), 0.05)
    assert call_with_timeout(lambda: 7, 1) == 7

    class Slow:
        def generate(self, s, u):
            time.sleep(0.3)
    with pytest.raises(CallTimeout):
        TimeoutGenerator(Slow(), 0.05).generate("s", "u")


def test_circuit_breaker_full_cycle():
    clock = FakeClock()
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=10, clock=clock)
    g = BreakerGenerator(FlakyGenerator([TransientError, TransientError, "ok", TransientError]), cb)
    for _ in range(2):
        with pytest.raises(TransientError):
            g.generate("s", "u")
    assert cb.state is State.OPEN
    with pytest.raises(CircuitOpenError):                      # fails fast: inner NOT called
        g.generate("s", "u")
    assert g.inner.calls == 2
    clock.advance(10)
    assert g.generate("s", "u") == "ok-reply" and cb.state is State.CLOSED   # half-open probe succeeded


def test_half_open_failure_reopens():
    clock = FakeClock()
    cb = CircuitBreaker(1, 5, clock)
    with pytest.raises(TransientError):
        cb.call(lambda: boom(TransientError()))
    clock.advance(5)
    with pytest.raises(TransientError):
        cb.call(lambda: boom(TransientError()))
    assert cb.state is State.OPEN


def test_permanent_errors_do_not_trip_breaker():
    cb = CircuitBreaker(1, 5, FakeClock())
    with pytest.raises(PermanentError):
        cb.call(lambda: boom(PermanentError()))
    assert cb.state is State.CLOSED


def test_token_bucket_burst_then_steady_rate():
    clock = FakeClock()
    tb = TokenBucket(rate=2, capacity=4, clock=clock)
    assert [tb.try_acquire() for _ in range(5)] == [True] * 4 + [False]   # burst of 4
    assert tb.wait_time() == pytest.approx(0.5)
    clock.advance(1)                                                      # refills 2 tokens
    assert tb.try_acquire(2) and not tb.try_acquire()
    clock.advance(100)
    assert tb.try_acquire(4) and not tb.try_acquire()                     # capped at capacity 4


def test_token_bucket_acquire_blocks_via_clock():
    clock = FakeClock()
    tb = TokenBucket(rate=1, capacity=1, clock=clock)
    tb.acquire()
    assert tb.acquire() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        tb.acquire(5)


def test_fallback_goes_to_second_provider():
    chain = FallbackGenerator([("primary", FlakyGenerator([TransientError] * 9)), ("backup", FlakyGenerator([], "B"))])
    assert chain.generate("s", "u") == "B"
    assert chain.chain.last_used == "backup" and chain.chain.failures == [("primary", "TransientError")]


def test_fallback_all_fail():
    chain = FallbackGenerator([("a", FlakyGenerator([TransientError])), ("b", FlakyGenerator([CircuitOpenError]))])
    with pytest.raises(TransientError, match="all providers failed"):
        chain.generate("s", "u")


def test_fallback_does_not_swallow_permanent():
    chain = FallbackGenerator([("a", FlakyGenerator([PermanentError])), ("b", FlakyGenerator([]))])
    with pytest.raises(PermanentError):
        chain.generate("s", "u")


def test_idempotency_runs_once():
    store, k = IdempotencyStore(), idempotency_key("charge", "order-1")
    n = []
    assert store.run_once(k, lambda: n.append(1) or "r1") == "r1"
    assert store.run_once(k, lambda: n.append(1) or "r2") == "r1"
    assert len(n) == 1 and store.executions == 1 and len(k) == 24
