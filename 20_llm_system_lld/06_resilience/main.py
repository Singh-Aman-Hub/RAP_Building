"""
GOAL:  Keep an LLM app alive when providers fail: retry with exponential backoff + full jitter, timeout, circuit breaker,
       token-bucket rate limiter, provider fallback, idempotency keys. All built from scratch; a FakeClock makes the
       "waiting" instant and the printed numbers exact.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 20_llm_system_lld/06_resilience/main.py
EXPECTED OUTPUT (shortened):
    backoff caps: attempt0 <=0.5 ... attempt5 <=8.0 | retry: 2 failures then ok, sleeps=[...]
    breaker: closed -> open -> fail-fast (inner calls still 2) -> closed   (the probe call ran in half_open)
    token bucket: burst of 4 allowed, 5th denied, wait 0.5 s
    fallback: primary failed (TransientError) -> answered by backup
NEEDS: ragkit only (stdlib). Deterministic: seeded RNG + fake clock.
"""
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragkit.resilience import (BreakerGenerator, CallTimeout, CircuitBreaker, CircuitOpenError, FallbackGenerator,
                               IdempotencyStore, PermanentError, RateLimitedError, RetryingGenerator, RetryPolicy,
                               TimeoutGenerator, TokenBucket, TransientError, idempotency_key)
from ragkit.testing import FakeClock, FlakyGenerator

# ---- 1. Backoff schedule: the CAP doubles each attempt; full jitter picks a random delay under the cap -------------
policy, rng = RetryPolicy(max_attempts=6, base_delay=0.5, max_delay=8.0), random.Random(42)
print("backoff caps  :", [min(policy.max_delay, policy.base_delay * 2 ** a) for a in range(6)])
print("jittered delay:", [round(policy.backoff(a, rng), 3) for a in range(6)], "(uniform(0, cap), seed 42)")

# ---- 2. Retry: two transient failures, then success ------------------------------------------------------------------------
clock, log = FakeClock(), []
flaky = FlakyGenerator([TransientError, RateLimitedError("429", retry_after=2.0)])
g = RetryingGenerator(flaky, RetryPolicy(max_attempts=4, base_delay=0.5), clock, random.Random(7),
                      on_retry=lambda n, exc, d: log.append(f"retry {n} after {type(exc).__name__}, waiting {d:.2f}s"))
print("retry result  :", g.generate("sys", "hi"), "| calls:", flaky.calls, "| log:", log)
print("fake clock advanced by", round(clock.now() - 1000.0, 2), "s without actually sleeping")

# ---- 3. Permanent errors are NOT retried -------------------------------------------------------------------------------------
bad = FlakyGenerator([PermanentError("401 invalid api key")])
try:
    RetryingGenerator(bad, clock=FakeClock()).generate("s", "u")
except PermanentError as e:
    print("permanent     :", e, "| calls:", bad.calls, "(no retry)")

# ---- 4. Timeout (real wall-clock, tiny) -------------------------------------------------------------------------------------
class Slow:
    def generate(self, s, u):
        time.sleep(0.5)
        return "late"


t0 = time.perf_counter()
try:
    TimeoutGenerator(Slow(), seconds=0.1).generate("s", "u")
except CallTimeout as e:
    print(f"timeout       : {e} after {time.perf_counter() - t0:.2f}s of waiting")

# ---- 5. Circuit breaker: closed -> open -> half_open -> closed ------------------------------------------------------------------
clock = FakeClock()
cb = CircuitBreaker(failure_threshold=2, recovery_timeout=30, clock=clock)
inner = FlakyGenerator([TransientError, TransientError, "ok"])
bg, states = BreakerGenerator(inner, cb), []
for _ in range(2):
    try:
        bg.generate("s", "u")
    except TransientError:
        pass
states.append(cb.state.value)
try:
    bg.generate("s", "u")
except CircuitOpenError:
    states.append("fail-fast (inner calls still %d)" % inner.calls)
clock.advance(30)
bg.generate("s", "u")
states.append(cb.state.value)
print("breaker       : closed ->", " -> ".join(states))

# ---- 6. Token bucket -------------------------------------------------------------------------------------------------------------
clock = FakeClock()
tb = TokenBucket(rate=2, capacity=4, clock=clock)
print("token bucket  :", [tb.try_acquire() for _ in range(5)], "| wait for next token:", tb.wait_time(), "s")
clock.advance(1.0)
print("                after 1 s (+2 tokens):", [tb.try_acquire() for _ in range(3)])
t2 = TokenBucket(1, 1, FakeClock())
t2.acquire()                                                        # takes the only token
print("                second acquire() blocks via the clock and waited", round(t2.acquire(), 3), "s")

# ---- 7. Fallback between providers -------------------------------------------------------------------------------------------------
fb = FallbackGenerator([("primary", FlakyGenerator([TransientError] * 9)), ("backup", FlakyGenerator([], "answer from backup"))])
print("fallback      :", fb.generate("s", "u"), "| used:", fb.chain.last_used, "| failures:", fb.chain.failures)

# ---- 8. Idempotency: a retried 'charge' runs once ---------------------------------------------------------------------------------------
store, key = IdempotencyStore(), idempotency_key("charge-user-42", "order-1001")
runs = []
for attempt in range(3):                                            # client retried because the first response was lost
    store.run_once(key, lambda: runs.append("charged") or "receipt-1")
print("idempotency   : key", key, "| attempts 3 | real executions", store.executions)

# TRY IT: set the breaker's failure_threshold to 1 and recovery_timeout to 0: does it still open? What does half_open do then?
# TRY IT: change jitter to 'equal' (delay = cap/2 + uniform(0, cap/2)) in RetryPolicy.backoff. Which is better under a 1000-client stampede?
# BREAK IT: make retry_on=(Exception,) and retry a PermanentError: how many pointless calls hit a revoked key?
