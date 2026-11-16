# Step 06 — Resilience: retries, timeouts, circuit breaker, rate limiter, fallbacks, idempotency

## The story so far
[Step 05](../05_design_patterns_in_llm_apps/WALKTHROUGH.md) made every part swappable, which also lets us wrap any part with a Decorator. Module [08 step 06](../../08_agent_harness/README.md) showed an agent retrying a failed tool call once.

## The pain point
LLM APIs fail constantly in ordinary ways: 429 rate limited, 500/504/529 overloaded or timing out, a connection reset halfway through a stream, a key that is revoked. Naive code either crashes on the first 503 or retries instantly in a loop and makes the outage worse (a retry storm). A slow provider holds your threads hostage. A duplicated 'charge the customer' call double-charges.

## The spark
Treat failure as a normal input and give each failure kind its own tool: **retry with exponential backoff and jitter** for transient errors, a **timeout** so waiting has a limit, a **circuit breaker** to stop hammering a dead provider, a **token-bucket rate limiter** to stay under quotas, a **fallback** to a second provider, and **idempotency keys** so a retried side effect happens once. Test all of it with a **fake clock** and a **fake flaky provider**, with no real sleeping and no network.

## Goal
Implement and test each mechanism from scratch in `ragkit/resilience.py`, understand the error taxonomy (retry or not), read the numbers (backoff caps, breaker states, bucket refill), and explain how the pieces compose around a `Generator`.

## The theory (read this before the interview)
### Error taxonomy: what is worth retrying?
| Kind | Examples (Claude API) | Retry? | ragkit type |
|---|---|---|---|
| Transient | 500 `api_error`, 504 `timeout_error`, 529 `overloaded_error`, connection reset | yes, with backoff | `TransientError` |
| Rate limited | 429 `rate_limit_error` (may carry `retry-after`) | yes, wait at least `retry-after` | `RateLimitedError(retry_after)` |
| Permanent | 400 `invalid_request_error`, 401 bad key, 403, 404, 413 | **no** - same call fails again | `PermanentError` |
| Spend cap | a tier spend-cap 429 has **no** `retry-after` and keeps failing | no (alert a human) | treat as permanent |
Anthropic's SDK itself retries transient failures twice by default with exponential backoff, honouring `retry-after` (see the API errors page, References). So application-level retry is for *wider* policies (more attempts, other providers), and you must avoid **stacking retries** (3 layers x 3 attempts = 27 calls).

### Retry with exponential backoff and jitter
`delay(attempt) = random.uniform(0, min(max_delay, base * 2**attempt))` ("full jitter"). Without jitter, 1000 clients that failed together all retry at the same instant (the **thundering herd**). In Experiments: no jitter -> 1000 retries in one 100 ms slot; full jitter -> about 100 per slot. Other variants: *equal jitter* (`cap/2 + uniform(0, cap/2)`), *decorrelated jitter*. Also: cap the number of attempts, honour `retry-after`, retry only **idempotent** operations (a read-only completion is; a "send email" tool call is not unless it has an idempotency key).

### Timeouts
Every network call needs one: connect timeout, read timeout, and an overall **deadline** for the whole request (including retries). `call_with_timeout` runs the call in a worker thread and stops *waiting* after N seconds; it cannot kill the thread (Python limitation), so real HTTP clients must also set socket timeouts. For streams, use a per-chunk idle timeout plus a total deadline. Mid-stream errors can arrive after a 200 response.

### Circuit breaker
States: **closed** (normal) -> after `failure_threshold` consecutive failures -> **open** (reject immediately, `CircuitOpenError`, no call made) -> after `recovery_timeout` -> **half-open** (allow one probe) -> success closes it, failure re-opens it. Protects the provider from your retries and your users from waiting on a dead service (fail fast). `PermanentError` does not count: a bad request is not a health signal.

### Token bucket rate limiter
A bucket holds up to `capacity` tokens and refills at `rate` tokens/second. A request takes tokens if available. Allows **bursts** up to `capacity`, then a sustained `rate`. Implemented with timestamps (no background thread): refill on demand `tokens = min(capacity, tokens + elapsed * rate)`. LLM providers limit requests per minute **and** tokens per minute (step 10 uses two buckets). Alternatives: fixed window (boundary bursts), sliding window log (exact, memory heavy), leaky bucket (smooths output).

### Fallbacks
Try providers in order (`primary -> backup`), falling over only on transient/open-circuit errors, never on permanent ones (a 400 would fail on the backup too, or worse succeed with different semantics). Record which provider answered and why others failed. Caveats: different models answer differently (log it), different prompt formats and context limits, cost differences; degrade gracefully (a smaller model, a cached answer, an honest 'try again later').

### Idempotency keys
The client sends a unique key per logical operation; the server stores the result under it and returns the stored result for any repeat. Retries after a lost response then cannot double-execute. `idempotency_key(*parts)` hashes the logical operation (`charge-user-42`, `order-1001`). A production store is a table with a unique constraint on the key, and handles the **in-flight** case (second request arrives while the first is running).

### Composition order (outermost first)
`rate limiter -> retry -> circuit breaker -> timeout -> provider`: the limiter admits a request; retry re-tries failures; each attempt goes through the breaker (which can short-circuit retries when the provider is down) and a per-attempt timeout. Put a fallback chain *around* per-provider stacks. Emit events at each decision (step 09).

### Testing without sleeping
`FakeClock` has `now()`, `sleep()` (advances time instantly) and `advance()`. Every time-dependent class takes an injectable `clock`, so a 30-second breaker timeout is tested in microseconds, deterministically. `FlakyGenerator` follows a script (`[TransientError, TransientError, 'ok']`).

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Retry / backoff / jitter | exponential backoff, retry policy, randomised delay | Close | Backoff = growing delay; jitter = randomness added so clients do not synchronise. |
| Timeout / deadline | read timeout, socket timeout, request budget | Close | A timeout bounds one wait; a deadline bounds the whole operation including retries. |
| Circuit breaker | fail-fast guard, health gate | Mostly | Stops calls to a failing dependency for a while; unlike retry it reduces load. |
| Rate limiter / throttle | token bucket, leaky bucket, quota, back-pressure | No | Rate limiter is the goal; token bucket is one algorithm; back-pressure pushes the slowdown upstream. |
| Fallback / failover | backup provider, degradation, router | Mostly | Fallback = try the next option on failure; routing = choose by rule *before* calling. |
| Idempotency key | dedup key, request id, exactly-once effect | Close | The key identifies one logical operation so repeats have no extra effect. |
| Transient vs permanent error | retryable vs non-retryable, 5xx vs 4xx | Mostly | Whether trying the identical request again can succeed. |

## Run it
```bash
# from this step folder
HF_HUB_OFFLINE=1 python main.py
# or from the repo root
HF_HUB_OFFLINE=1 .venv/bin/python 20_llm_system_lld/06_resilience/main.py
```


## Code, step by step
Three files: `main.py`, the mechanisms in `ragkit/resilience.py` and the test doubles in `ragkit/testing.py`. Every block is listed.

#### Walking `main.py`

### 1. Header docstring  (`main.py` lines 1–14)
```python
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
```
- **In simple words:** What the file shows.
- **Exactly:** GOAL/RUN/EXPECTED/NEEDS; deterministic (seeded RNG, fake clock).
- **Why it is here:** Orientation.

### 2. Imports  (`main.py` lines 15–24)
```python
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragkit.resilience import (BreakerGenerator, CallTimeout, CircuitBreaker, CircuitOpenError, FallbackGenerator,
                               IdempotencyStore, PermanentError, RateLimitedError, RetryingGenerator, RetryPolicy,
                               TimeoutGenerator, TokenBucket, TransientError, idempotency_key)
from ragkit.testing import FakeClock, FlakyGenerator
```
- **In simple words:** Everything from resilience and the fakes.
- **Exactly:** `FakeClock` and `FlakyGenerator` from `ragkit.testing`.
- **Why it is here:** No third-party libraries.

### 3. The backoff schedule  (`main.py` lines 26–29)
```python
# ---- 1. Backoff schedule: the CAP doubles each attempt; full jitter picks a random delay under the cap -------------
policy, rng = RetryPolicy(max_attempts=6, base_delay=0.5, max_delay=8.0), random.Random(42)
print("backoff caps  :", [min(policy.max_delay, policy.base_delay * 2 ** a) for a in range(6)])
print("jittered delay:", [round(policy.backoff(a, rng), 3) for a in range(6)], "(uniform(0, cap), seed 42)")
```
- **In simple words:** Print the cap for each attempt and a random delay under it.
- **Exactly:** Caps `[0.5, 1.0, 2.0, 4.0, 8.0, 8.0]` (doubling, then capped at `max_delay=8`); jittered delays with seed 42 `[0.32, 0.025, 0.55, 0.893, 5.892, 5.414]`, each below its cap.
- **Why it is here:** Shows both ideas: exponential growth and randomisation.

### 4. Retry through two failures  (`main.py` lines 31–37)
```python
# ---- 2. Retry: two transient failures, then success ------------------------------------------------------------------------
clock, log = FakeClock(), []
flaky = FlakyGenerator([TransientError, RateLimitedError("429", retry_after=2.0)])
g = RetryingGenerator(flaky, RetryPolicy(max_attempts=4, base_delay=0.5), clock, random.Random(7),
                      on_retry=lambda n, exc, d: log.append(f"retry {n} after {type(exc).__name__}, waiting {d:.2f}s"))
print("retry result  :", g.generate("sys", "hi"), "| calls:", flaky.calls, "| log:", log)
print("fake clock advanced by", round(clock.now() - 1000.0, 2), "s without actually sleeping")
```
- **In simple words:** The provider fails twice (one is a 429 with retry-after 2 s), then succeeds.
- **Exactly:** Result `ok-reply`, `calls: 3`; log: `retry 1 after TransientError, waiting 0.16s`, `retry 2 after RateLimitedError, waiting 2.00s` (retry_after beat the jittered delay). The fake clock advanced 2.16 s with no real sleeping.
- **Why it is here:** Demonstrates `retry_after` handling and instant testing.

### 5. A permanent error is not retried  (`main.py` lines 39–44)
```python
# ---- 3. Permanent errors are NOT retried -------------------------------------------------------------------------------------
bad = FlakyGenerator([PermanentError("401 invalid api key")])
try:
    RetryingGenerator(bad, clock=FakeClock()).generate("s", "u")
except PermanentError as e:
    print("permanent     :", e, "| calls:", bad.calls, "(no retry)")
```
- **In simple words:** A bad API key: trying again is pointless.
- **Exactly:** `PermanentError: 401 invalid api key`, `calls: 1`.
- **Why it is here:** Retrying a 401 wastes quota and delays the real fix.

### 6. A real (tiny) timeout  (`main.py` lines 46–57)
```python
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
```
- **In simple words:** A provider that takes 0.5 s, with a 0.1 s limit.
- **Exactly:** `CallTimeout: no answer within 0.1s after 0.10s of waiting`.
- **Why it is here:** The caller regains control after the limit, not after the slow call.

### 7. Open the breaker with two failures  (`main.py` lines 59–63)
```python
# ---- 5. Circuit breaker: closed -> open -> half_open -> closed ------------------------------------------------------------------
clock = FakeClock()
cb = CircuitBreaker(failure_threshold=2, recovery_timeout=30, clock=clock)
inner = FlakyGenerator([TransientError, TransientError, "ok"])
bg, states = BreakerGenerator(inner, cb), []
```
- **In simple words:** Threshold 2, recovery 30 s: fail twice and watch it open.
- **Exactly:** After two `TransientError`s the state is `open`.
- **Why it is here:** A dead provider should stop receiving traffic.

### 8. Fail fast, then recover  (`main.py` lines 64–77)
```python
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
```
- **In simple words:** While open, calls are rejected without touching the provider; after 30 s one probe succeeds.
- **Exactly:** `CircuitOpenError` raised and the inner call count stays 2; after `clock.advance(30)` the next call runs (half-open probe), succeeds, and the breaker is `closed` again. Printed: `closed -> open -> fail-fast (inner calls still 2) -> closed`.
- **Why it is here:** Shows all three behaviours: trip, reject, heal.

### 9. A bucket of 4 refilling at 2 per second  (`main.py` lines 79–87)
```python
# ---- 6. Token bucket -------------------------------------------------------------------------------------------------------------
clock = FakeClock()
tb = TokenBucket(rate=2, capacity=4, clock=clock)
print("token bucket  :", [tb.try_acquire() for _ in range(5)], "| wait for next token:", tb.wait_time(), "s")
clock.advance(1.0)
print("                after 1 s (+2 tokens):", [tb.try_acquire() for _ in range(3)])
t2 = TokenBucket(1, 1, FakeClock())
t2.acquire()                                                        # takes the only token
print("                second acquire() blocks via the clock and waited", round(t2.acquire(), 3), "s")
```
- **In simple words:** Allow a burst of 4, deny the 5th, wait.
- **Exactly:** `[True, True, True, True, False]`; `wait_time` for the next token 0.5 s; after `advance(1.0)` two more succeed: `[True, True, False]`; a second `acquire()` on a 1-token bucket waited 1.0 s (via the clock).
- **Why it is here:** Burst then sustained rate, in numbers.

### 10. Primary fails, backup answers  (`main.py` lines 89–91)
```python
# ---- 7. Fallback between providers -------------------------------------------------------------------------------------------------
fb = FallbackGenerator([("primary", FlakyGenerator([TransientError] * 9)), ("backup", FlakyGenerator([], "answer from backup"))])
print("fallback      :", fb.generate("s", "u"), "| used:", fb.chain.last_used, "| failures:", fb.chain.failures)
```
- **In simple words:** Try providers in order.
- **Exactly:** Reply `answer from backup`, `used: backup`, `failures: [('primary', 'TransientError')]`.
- **Why it is here:** The system stays up and records what happened.

### 11. A retried charge runs once  (`main.py` lines 93–98)
```python
# ---- 8. Idempotency: a retried 'charge' runs once ---------------------------------------------------------------------------------------
store, key = IdempotencyStore(), idempotency_key("charge-user-42", "order-1001")
runs = []
for attempt in range(3):                                            # client retried because the first response was lost
    store.run_once(key, lambda: runs.append("charged") or "receipt-1")
print("idempotency   : key", key, "| attempts 3 | real executions", store.executions)
```
- **In simple words:** The client retried three times because the first response was lost.
- **Exactly:** `attempts 3`, `real executions 1`; key `847abf01da26c37e0c805e96` (24 hex chars of a content hash).
- **Why it is here:** Prevents double side effects.

### 12. Exercises  (`main.py` lines 100–102)
```python
# TRY IT: set the breaker's failure_threshold to 1 and recovery_timeout to 0: does it still open? What does half_open do then?
# TRY IT: change jitter to 'equal' (delay = cap/2 + uniform(0, cap/2)) in RetryPolicy.backoff. Which is better under a 1000-client stampede?
# BREAK IT: make retry_on=(Exception,) and retry a PermanentError: how many pointless calls hit a revoked key?
```
- **In simple words:** Things to try.
- **Exactly:** Comment lines.
- **Why it is here:** See exercises.md.


#### Walking `ragkit/resilience.py`

### 13. resilience.py header and imports  (`ragkit/resilience.py` lines 1–17)
```python
"""resilience.py -- what keeps an LLM app alive when providers fail (step 06). Everything from scratch, injectable clock."""
from __future__ import annotations

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Sequence, TypeVar

from .ids import content_hash

T = TypeVar("T")


# ------------------------------------------------------------------ error taxonomy: what is worth retrying?
```
- **In simple words:** All mechanisms in one file, with an injectable clock.
- **Exactly:** Imports `random`, `threading`, `time`, `ThreadPoolExecutor`, `dataclass`, `Enum`, and `content_hash`.
- **Why it is here:** Standard library only.

### 14. Error base and TransientError  (`ragkit/resilience.py` lines 18–23)
```python
class LLMError(Exception):
    pass


class TransientError(LLMError):
    """Network blip, 500, 503, overloaded: trying again may work."""
```
- **In simple words:** A family of exceptions for LLM calls.
- **Exactly:** `LLMError(Exception)` is the root; `TransientError(LLMError)` = worth retrying.
- **Why it is here:** `except TransientError` can express policy; `except Exception` cannot.

### 15. RateLimitedError carries retry_after  (`ragkit/resilience.py` lines 26–29)
```python
class RateLimitedError(TransientError):
    def __init__(self, msg: str = "rate limited", retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after                         # the server's Retry-After header, if it sent one
```
- **In simple words:** A transient error that can say how long to wait.
- **Exactly:** Subclass of `TransientError` with `retry_after: float | None` (from the `retry-after` header).
- **Why it is here:** Honouring the server's hint beats guessing.

### 16. PermanentError, CircuitOpenError, CallTimeout  (`ragkit/resilience.py` lines 32–41)
```python
class PermanentError(LLMError):
    """400 bad request, 401 bad key, content policy: retrying the same call cannot help."""


class CircuitOpenError(LLMError):
    pass


class CallTimeout(TransientError):
    pass
```
- **In simple words:** The other error kinds.
- **Exactly:** `PermanentError` = never retry; `CircuitOpenError` = rejected without calling; `CallTimeout(TransientError)` = retryable.
- **Why it is here:** Each exception type maps to one decision.

### 17. SystemClock: the real clock  (`ragkit/resilience.py` lines 44–49)
```python
class SystemClock:
    now = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)


# ------------------------------------------------------------------ retry with exponential backoff + jitter
```
- **In simple words:** Wraps the real time functions.
- **Exactly:** `now = time.monotonic` (cannot go backwards), `sleep = time.sleep`, as static methods.
- **Why it is here:** Same interface as `FakeClock`, so tests swap them.

### 18. RetryPolicy: the settings  (`ragkit/resilience.py` lines 50–55)
```python
@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 4
    base_delay: float = 0.5
    max_delay: float = 8.0
    retry_on: tuple = (TransientError,)
```
- **In simple words:** How many attempts, how long to wait, which errors count.
- **Exactly:** Frozen dataclass: `max_attempts=4`, `base_delay=0.5`, `max_delay=8.0`, `retry_on=(TransientError,)`.
- **Why it is here:** A policy object can be shared, logged and tested.

### 19. RetryPolicy.backoff: full jitter  (`ragkit/resilience.py` lines 57–59)
```python
    def backoff(self, attempt: int, rng: random.Random) -> float:
        """'Full jitter': uniform(0, min(max_delay, base * 2**attempt)). attempt is 0 for the first retry."""
        return rng.uniform(0, min(self.max_delay, self.base_delay * 2 ** attempt))
```
- **In simple words:** Pick a random delay below a growing cap.
- **Exactly:** `rng.uniform(0, min(max_delay, base_delay * 2 ** attempt))` with `attempt` starting at 0.
- **Why it is here:** Spreads clients out in time (Experiments: 1000 -> ~100 per slot).

### 20. retry_call: signature and defaults  (`ragkit/resilience.py` lines 62–64)
```python
def retry_call(fn: Callable[[], T], policy: RetryPolicy, clock=None, rng: random.Random | None = None,
               on_retry: Callable[[int, Exception, float], None] | None = None) -> T:
    clock, rng = clock or SystemClock, rng or random.Random()
```
- **In simple words:** Call a function, retrying per policy.
- **Exactly:** `fn` takes no arguments (wrap with `lambda`); injectable `clock`, `rng` (seed it for tests), and `on_retry` callback.
- **Why it is here:** Injection makes retry deterministic and observable.

### 21. retry_call: the loop  (`ragkit/resilience.py` lines 65–71)
```python
    for attempt in range(policy.max_attempts):
        try:
            return fn()
        except policy.retry_on as exc:
            if attempt == policy.max_attempts - 1:
                raise                                          # out of attempts: surface the last error
            delay = policy.backoff(attempt, rng)
```
- **In simple words:** Try; on a retryable error either give up or wait.
- **Exactly:** `except policy.retry_on`: if this was the last attempt, `raise` (the original exception, traceback intact); otherwise compute the delay.
- **Why it is here:** Re-raising the last real error keeps the traceback honest.

### 22. retry_call: honour retry-after, then sleep  (`ragkit/resilience.py` lines 72–77)
```python
            if isinstance(exc, RateLimitedError) and exc.retry_after is not None:
                delay = max(delay, exc.retry_after)            # never retry sooner than the server asked
            if on_retry:
                on_retry(attempt + 1, exc, delay)
            clock.sleep(delay)
    raise AssertionError("unreachable")
```
- **In simple words:** Never retry sooner than the server asked.
- **Exactly:** `delay = max(delay, exc.retry_after)`; `on_retry(attempt+1, exc, delay)`; `clock.sleep(delay)`; trailing `raise AssertionError('unreachable')` satisfies type checkers.
- **Why it is here:** A 429 retried too early just earns another 429.

### 23. RetryingGenerator: retry as a Decorator  (`ragkit/resilience.py` lines 80–91)
```python
class RetryingGenerator:
    """Decorator: Generator + retry policy."""

    def __init__(self, inner, policy: RetryPolicy | None = None, clock=None, rng=None, on_retry=None):
        self.inner, self.policy = inner, policy or RetryPolicy()
        self.clock, self.rng, self.on_retry = clock, rng or random.Random(0), on_retry

    def generate(self, system: str, user: str) -> str:
        return retry_call(lambda: self.inner.generate(system, user), self.policy, self.clock, self.rng, self.on_retry)


# ------------------------------------------------------------------ timeout
```
- **In simple words:** Wrap any Generator with the policy.
- **Exactly:** Same `generate(system, user)` signature; delegates through `retry_call(lambda: inner.generate(...))`; default `random.Random(0)` for reproducibility.
- **Why it is here:** Retry becomes a layer you add in the composition root.

### 24. call_with_timeout: bound the wait  (`ragkit/resilience.py` lines 92–102)
```python
def call_with_timeout(fn: Callable[[], T], seconds: float) -> T:
    """Run fn in a worker thread and stop WAITING after `seconds`. (Python cannot kill a thread: the worker keeps
    running in the background, so real clients should also set a socket/HTTP timeout.)"""
    pool = ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(fn)
    try:
        return fut.result(timeout=seconds)
    except FutureTimeout:
        raise CallTimeout(f"no answer within {seconds}s") from None
    finally:
        pool.shutdown(wait=False)
```
- **In simple words:** Run in a worker thread and stop waiting.
- **Exactly:** `ThreadPoolExecutor(1).submit(fn)`, `future.result(timeout=seconds)`; on `FutureTimeout` raise `CallTimeout`; `pool.shutdown(wait=False)` so we do not block on the stuck thread.
- **Why it is here:** A stuck call cannot be killed in Python, only abandoned; set real socket timeouts too.

### 25. TimeoutGenerator  (`ragkit/resilience.py` lines 105–113)
```python
class TimeoutGenerator:
    def __init__(self, inner, seconds: float):
        self.inner, self.seconds = inner, seconds

    def generate(self, system, user):
        return call_with_timeout(lambda: self.inner.generate(system, user), self.seconds)


# ------------------------------------------------------------------ circuit breaker
```
- **In simple words:** The timeout as a Decorator.
- **Exactly:** Wraps `inner.generate` in `call_with_timeout`.
- **Why it is here:** Per-attempt timeout inside the retry layer.

### 26. State: the breaker's three states  (`ragkit/resilience.py` lines 114–117)
```python
class State(str, Enum):
    CLOSED = "closed"          # normal: calls pass through
    OPEN = "open"              # failing: calls are rejected immediately (fail fast)
    HALF_OPEN = "half_open"    # probing: let ONE call through to test recovery
```
- **In simple words:** closed, open, half_open.
- **Exactly:** `str` Enum so states print and compare as text.
- **Why it is here:** Named states beat booleans.

### 27. CircuitBreaker: settings and state  (`ragkit/resilience.py` lines 120–123)
```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 30.0, clock=None):
        self.threshold, self.recovery, self.clock = failure_threshold, recovery_timeout, clock or SystemClock
        self.state, self.failures, self.opened_at = State.CLOSED, 0, 0.0
```
- **In simple words:** Remember failures and when we opened.
- **Exactly:** `failure_threshold=3`, `recovery_timeout=30`, `state`, `failures`, `opened_at`, injectable `clock`.
- **Why it is here:** All time-dependent logic uses `clock.now()`.

### 28. CircuitBreaker.call: gate the call  (`ragkit/resilience.py` lines 125–132)
```python
    def call(self, fn: Callable[[], T]) -> T:
        if self.state is State.OPEN:
            if self.clock.now() - self.opened_at >= self.recovery:
                self.state = State.HALF_OPEN                   # cool-down over: allow one probe
            else:
                raise CircuitOpenError("circuit open")
        try:
            result = fn()
```
- **In simple words:** If open and not yet time, reject; if open and time is up, allow one probe.
- **Exactly:** OPEN and `now - opened_at >= recovery` -> `HALF_OPEN`; else raise `CircuitOpenError('circuit open')` without calling `fn`.
- **Why it is here:** Fail fast protects both sides.

### 29. CircuitBreaker.call: record the outcome  (`ragkit/resilience.py` lines 133–141)
```python
        except PermanentError:
            raise                                              # caller's fault, not a health signal
        except Exception:
            self.failures += 1
            if self.state is State.HALF_OPEN or self.failures >= self.threshold:
                self.state, self.opened_at = State.OPEN, self.clock.now()
            raise
        self.failures, self.state = 0, State.CLOSED            # success closes the circuit
        return result
```
- **In simple words:** Failures count; permanent errors do not; success heals.
- **Exactly:** Other exceptions: `failures += 1`; if half-open or `failures >= threshold`: state OPEN, `opened_at = now`; re-raise. Success: `failures = 0`, state CLOSED.
- **Why it is here:** Counting only health-related failures avoids tripping on a user's bad request.

### 30. BreakerGenerator  (`ragkit/resilience.py` lines 144–152)
```python
class BreakerGenerator:
    def __init__(self, inner, breaker: CircuitBreaker):
        self.inner, self.breaker = inner, breaker

    def generate(self, system, user):
        return self.breaker.call(lambda: self.inner.generate(system, user))


# ------------------------------------------------------------------ token bucket rate limiter
```
- **In simple words:** Breaker as a Decorator.
- **Exactly:** `breaker.call(lambda: inner.generate(...))`.
- **Why it is here:** One breaker per provider.

### 31. TokenBucket: state  (`ragkit/resilience.py` lines 153–159)
```python
class TokenBucket:
    """Capacity `capacity`, refilled at `rate` units/second. Allows bursts up to capacity, then a steady `rate`."""

    def __init__(self, rate: float, capacity: float, clock=None):
        self.rate, self.capacity, self.clock = rate, capacity, clock or SystemClock
        self.tokens, self.updated = float(capacity), self.clock.now()
        self._lock = threading.Lock()
```
- **In simple words:** A bucket that starts full.
- **Exactly:** `rate` tokens/second, `capacity`, `tokens = capacity`, `updated = clock.now()`, a `threading.Lock` for thread safety.
- **Why it is here:** A lock keeps concurrent callers from over-spending.

### 32. TokenBucket._refill: lazy refill  (`ragkit/resilience.py` lines 161–164)
```python
    def _refill(self) -> None:
        now = self.clock.now()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now
```
- **In simple words:** Add the tokens earned since last time.
- **Exactly:** `tokens = min(capacity, tokens + (now - updated) * rate)`; `updated = now`.
- **Why it is here:** No timer thread; accuracy comes from timestamps.

### 33. TokenBucket.try_acquire  (`ragkit/resilience.py` lines 166–172)
```python
    def try_acquire(self, n: float = 1) -> bool:
        with self._lock:
            self._refill()
            if n <= self.tokens:
                self.tokens -= n
                return True
            return False
```
- **In simple words:** Take n tokens if there are enough.
- **Exactly:** Refill, then `if n <= tokens: tokens -= n; True` else `False` (nothing taken).
- **Why it is here:** Non-blocking: the caller decides to reject or wait.

### 34. TokenBucket.wait_time  (`ragkit/resilience.py` lines 174–177)
```python
    def wait_time(self, n: float = 1) -> float:
        with self._lock:
            self._refill()
            return max(0.0, (n - self.tokens) / self.rate)
```
- **In simple words:** How long until n tokens exist?
- **Exactly:** `max(0, (n - tokens) / rate)`; e.g. 0.5 s for 1 token at rate 2.
- **Why it is here:** Lets callers sleep exactly as long as needed.

### 35. TokenBucket.acquire: block until available  (`ragkit/resilience.py` lines 179–191)
```python
    def acquire(self, n: float = 1) -> float:
        """Block until n units are available; return how long we waited."""
        if n > self.capacity:
            raise ValueError("request larger than bucket capacity can never succeed")
        waited = 0.0
        while not self.try_acquire(n):
            w = self.wait_time(n)
            self.clock.sleep(w)
            waited += w
        return waited


# ------------------------------------------------------------------ fallback between providers
```
- **In simple words:** Wait (via the clock) and retry.
- **Exactly:** Raises `ValueError` if `n > capacity` (could never succeed); loops `try_acquire` / `clock.sleep(wait_time)`; returns total waited.
- **Why it is here:** Blocking admission with a guaranteed end.

### 36. FallbackChain: ordered options  (`ragkit/resilience.py` lines 192–199)
```python
class FallbackChain:
    """Try callables in order; return the first success. Remembers which one answered and why others failed."""

    def __init__(self, names_and_fns: Sequence[tuple[str, Callable[..., Any]]],
                 fall_on: tuple = (TransientError, CircuitOpenError)):
        self.options, self.fall_on = list(names_and_fns), fall_on
        self.last_used: str | None = None
        self.failures: list[tuple[str, str]] = []
```
- **In simple words:** A list of (name, callable).
- **Exactly:** `fall_on=(TransientError, CircuitOpenError)`; remembers `last_used` and `failures`.
- **Why it is here:** Only infrastructure errors trigger fallback.

### 37. FallbackChain.__call__  (`ragkit/resilience.py` lines 201–210)
```python
    def __call__(self, *args, **kwargs):
        self.failures = []
        for name, fn in self.options:
            try:
                result = fn(*args, **kwargs)
                self.last_used = name
                return result
            except self.fall_on as exc:
                self.failures.append((name, type(exc).__name__))
        raise TransientError(f"all providers failed: {self.failures}")
```
- **In simple words:** Try each in turn.
- **Exactly:** Returns the first success; collects `(name, ExceptionName)`; if all fail raises `TransientError('all providers failed: [...]')`. Permanent errors pass through.
- **Why it is here:** A 400 on the primary must not silently succeed on a different model.

### 38. FallbackGenerator  (`ragkit/resilience.py` lines 213–221)
```python
class FallbackGenerator:
    def __init__(self, generators: Sequence[tuple[str, Any]]):
        self.chain = FallbackChain([(n, g.generate) for n, g in generators])

    def generate(self, system, user):
        return self.chain(system, user)


# ------------------------------------------------------------------ idempotency
```
- **In simple words:** Fallback as a Generator.
- **Exactly:** Builds a `FallbackChain` of each generator's `.generate`.
- **Why it is here:** Plugs straight into `RagService`.

### 39. idempotency_key  (`ragkit/resilience.py` lines 222–223)
```python
def idempotency_key(*parts: str) -> str:
    return content_hash(*parts, length=24)
```
- **In simple words:** Make a stable key for an operation.
- **Exactly:** `content_hash(*parts, length=24)`: same parts, same key.
- **Why it is here:** Derived, not random: a retry recomputes the same key.

### 40. IdempotencyStore: state  (`ragkit/resilience.py` lines 226–232)
```python
class IdempotencyStore:
    """Run an operation at most once per key; later calls with the same key get the stored result."""

    def __init__(self):
        self._done: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.executions = 0
```
- **In simple words:** Remember results by key.
- **Exactly:** `_done` dict, a `Lock`, `executions` counter.
- **Why it is here:** The counter proves how many real executions happened.

### 41. IdempotencyStore.run_once  (`ragkit/resilience.py` lines 234–241)
```python
    def run_once(self, key: str, fn: Callable[[], T]) -> T:
        with self._lock:                                       # (a real system would use a DB row with a unique key)
            if key in self._done:
                return self._done[key]
            result = fn()
            self._done[key] = result
            self.executions += 1
            return result
```
- **In simple words:** Run at most once per key.
- **Exactly:** Under the lock: if the key exists return the stored result, else run `fn`, store, count.
- **Why it is here:** Holding the lock while running serialises everything; a real system stores an 'in progress' marker instead.


#### Walking `ragkit/testing.py`

### 42. testing.py header  (`ragkit/testing.py` lines 1–2)
```python
"""testing.py -- fakes used by tests AND demos: a controllable clock and providers that fail on cue (step 06)."""
from __future__ import annotations
```
- **In simple words:** Fakes used by the tests AND the demos.
- **Exactly:** Docstring.
- **Why it is here:** Shipping fakes in the package keeps demos and tests identical.

### 43. FakeClock  (`ragkit/testing.py` lines 5–19)
```python
class FakeClock:
    """Time that only moves when you say so: sleep(2) advances now() by 2 instantly. Makes backoff tests instant + exact."""

    def __init__(self, start: float = 1000.0):
        self.t, self.sleeps = start, []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds

    def advance(self, seconds: float) -> None:
        self.t += seconds
```
- **In simple words:** Time you control.
- **Exactly:** `t` starts at 1000.0; `now()` returns it; `sleep(s)` records `s` and advances `t`; `advance(s)` moves time without recording.
- **Why it is here:** Turns waits into instant arithmetic.

### 44. FlakyGenerator: a provider that fails on cue  (`ragkit/testing.py` lines 22–35)
```python
class FlakyGenerator:
    """Generator that follows a script: each item is 'ok', an exception instance/class, or a string reply."""

    def __init__(self, script, reply: str = "ok-reply"):
        self.script, self.reply, self.calls = list(script), reply, 0

    def generate(self, system: str, user: str) -> str:
        self.calls += 1
        step = self.script.pop(0) if self.script else "ok"
        if step == "ok":
            return self.reply
        if isinstance(step, str):
            return step
        raise step() if isinstance(step, type) else step
```
- **In simple words:** Follow a script of outcomes.
- **Exactly:** Each script item: `'ok'`, a string reply, an exception class or instance; empty script -> `'ok'`. Counts `calls`.
- **Why it is here:** Reproduces any failure sequence deterministically.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `except policy.retry_on as exc` | Catch a tuple of exception classes stored in a variable |
| `raise` (bare) | Re-raise the exception being handled, keeping its traceback |
| `lambda: self.inner.generate(...)` | Zero-argument wrapper so `retry_call` can call it repeatedly |
| `time.monotonic` | Clock that only moves forward (safe for timeouts) |
| `with self._lock:` | Mutual exclusion for shared state |
| `str, Enum` | Enum whose values print as text |

## Libraries & functions
| Name | What it does here |
|---|---|
| `random.Random(seed)` | Seedable RNG; `uniform(a, b)` |
| `concurrent.futures.ThreadPoolExecutor` | Worker thread for `call_with_timeout`; `Future.result(timeout=)` |
| `threading.Lock` | Thread-safe token bucket |
| `time.monotonic`, `time.sleep` | Real clock |
| `hashlib` via `content_hash` | Idempotency keys |

## Execution trace (real run)
```text
backoff caps  : [0.5, 1.0, 2.0, 4.0, 8.0, 8.0]
jittered delay: [0.32, 0.025, 0.55, 0.893, 5.892, 5.414] (uniform(0, cap), seed 42)
retry result  : ok-reply | calls: 3 | log: ['retry 1 after TransientError, waiting 0.16s', 'retry 2 after RateLimitedError, waiting 2.00s']
fake clock advanced by 2.16 s without actually sleeping
permanent     : 401 invalid api key | calls: 1 (no retry)
timeout       : no answer within 0.1s after 0.11s of waiting
breaker       : closed -> open -> fail-fast (inner calls still 2) -> closed
token bucket  : [True, True, True, True, False] | wait for next token: 0.5 s
                after 1 s (+2 tokens): [True, True, False]
                second acquire() blocks via the clock and waited 1.0 s
fallback      : answer from backup | used: backup | failures: [('primary', 'TransientError')]
idempotency   : key 847abf01da26c37e0c805e96 | attempts 3 | real executions 1
```
- **Backoff**: caps double and stop at 8; jittered delays are random numbers below the caps (0.025 s is possible; that is the point of *full* jitter).
- **Retry**: one transient error then a 429 asking for 2 s. The second wait is exactly 2.00 s because `retry_after` is larger than the jittered value. Total fake time 2.16 s.
- **Permanent**: exactly one call.
- **Timeout**: the caller got control back after about 0.10 s although the provider needed 0.5 s.
- **Breaker**: the inner provider was called only twice even though three calls were made; the third was rejected instantly. After 30 fake seconds a probe succeeded and the breaker closed.
- **Bucket**: 4 allowed, then the fifth denied, 0.5 s to the next token; after 1 s two more.
- **Fallback** and **idempotency**: backup answered; three attempts, one execution.

## Experiments (change ONE thing)
Each snippet runs from this step folder (`cd` into it first). Outputs below are from real runs.

### Thundering herd: 1000 clients retry once. No jitter vs full jitter
```python
import random
from collections import Counter
rng = random.Random(0)
cap = 1.0
no_jitter = Counter(round(cap, 1) for _ in range(1000))
full = Counter(int(rng.uniform(0, cap) * 10) / 10 for _ in range(1000))
print("no jitter : retries per 100 ms slot:", dict(no_jitter))
print("full jitter: retries per 100 ms slot:", dict(sorted(full.items())))
```
Observed:
```text
no jitter : retries per 100 ms slot: {1.0: 1000}
full jitter: retries per 100 ms slot: {0.0: 114, 0.1: 87, 0.2: 95, 0.3: 102, 0.4: 110, 0.5: 89, 0.6: 109, 0.7: 104, 0.8: 86, 0.9: 104}
```
With no jitter all 1000 clients retry in the same 100 ms slot (1000 requests at once); with full jitter they spread over ten slots, roughly 90-115 each. A recovering server survives 100 requests per slot far better than 1000.

### Total wait of 4 attempts with a fake clock
```python
import random
from ragkit.resilience import *
from ragkit.testing import FakeClock, FlakyGenerator
for seed in (1, 2, 3):
    c = FakeClock()
    try:
        RetryingGenerator(FlakyGenerator([TransientError] * 9), RetryPolicy(max_attempts=4), c, random.Random(seed)).generate("s", "u")
    except TransientError:
        print("seed", seed, "sleeps", [round(s, 2) for s in c.sleeps], "total", round(sum(c.sleeps), 2))
```
Observed:
```text
seed 1 sleeps [0.07, 0.85, 1.53] total 2.44
seed 2 sleeps [0.48, 0.95, 0.11] total 1.54
seed 3 sleeps [0.12, 0.54, 0.74] total 1.4
```
Total wait before the last attempt varies by seed (1.4-2.4 s here). Worst case for 4 attempts with base 0.5 is 0.5 + 1 + 2 = 3.5 s; average about half. Budget the *worst case* against your request deadline.

### Breaker threshold 1 vs 3 on the same failure stream
```python
from ragkit.resilience import *
from ragkit.testing import FakeClock
for th in (1, 3):
    cb = CircuitBreaker(th, 30, FakeClock()); rejected = reached = 0
    for _ in range(6):
        try:
            cb.call(lambda: (_ for _ in ()).throw(TransientError()))
            
        except CircuitOpenError:
            rejected += 1
        except TransientError:
            reached += 1
    print("threshold", th, "-> calls that reached the failing provider:", reached, "| rejected fast:", rejected)
```
Observed:
```text
threshold 1 -> calls that reached the failing provider: 1 | rejected fast: 5
threshold 3 -> calls that reached the failing provider: 3 | rejected fast: 3
```
Threshold 1 sends one call to the failing provider then rejects five; threshold 3 sends three and rejects three. A lower threshold protects the provider faster but trips on a single blip. Choose by how costly a false trip is.

## Debugging practice
**BREAK IT:**
```python
import sys; sys.path.insert(0, '..')

import random
from ragkit.resilience import *
from ragkit.testing import FakeClock, FlakyGenerator
g = FlakyGenerator([PermanentError("401 invalid api key")] * 5)
RetryingGenerator(g, RetryPolicy(max_attempts=4, retry_on=(Exception,)), FakeClock(), random.Random(0)).generate("s", "u")
```
Real output:
```text
sers/aman/Desktop/Project_resume/RAP_interview/20_llm_system_lld/06_resilience/../ragkit/testing.py", line 35, in generate
    raise step() if isinstance(step, type) else step
  File "/Users/aman/Desktop/Project_resume/RAP_interview/20_llm_system_lld/06_resilience/../ragkit/resilience.py", line 67, in retry_call
    return fn()
           ^^^^
  File "/Users/aman/Desktop/Project_resume/RAP_interview/20_llm_system_lld/06_resilience/../ragkit/resilience.py", line 88, in <lambda>
    return retry_call(lambda: self.inner.generate(system, user), self.policy, self.clock, self.rng, self.on_retry)
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/aman/Desktop/Project_resume/RAP_interview/20_llm_system_lld/06_resilience/../ragkit/testing.py", line 35, in generate
    raise step() if isinstance(step, type) else step
ragkit.resilience.PermanentError: 401 invalid api key
```
The traceback ends in `PermanentError: 401 invalid api key`, raised on the *first* call and then... retried: `retry_on=(Exception,)` made the policy retry a permanent error, so the provider was called 4 times (`max_attempts=4`) before the same error surfaced. Reading it: the frames show `retry_call` -> `RetryingGenerator.generate` -> `FlakyGenerator.generate`, i.e. the error came from the provider and passed through the retry layer unchanged. Fix: list only `TransientError` in `retry_on`.

## Rebuild checklist
- [ ] Write `retry_call` with full jitter and `retry-after` handling from memory.
- [ ] Draw the circuit breaker state diagram and write `call()`.
- [ ] Write a token bucket with lazy refill and explain burst vs sustained rate.
- [ ] Explain why fallback must not trigger on a `PermanentError`.
- [ ] Describe the order of limiter, retry, breaker and timeout and why.
- [ ] Explain how an idempotency key prevents a double charge and what an in-flight duplicate needs.

## The new problem (cliffhanger)
Retry, breaker and fallback all assumed that identical requests give identical answers. If they do, we should not call the model twice at all. How do you cache LLM answers safely: exact match, semantic match, TTL, eviction, and what must invalidate them when the index changes? [Step 07](../07_caching/WALKTHROUGH.md).

> **Say it aloud (interview box):** "I classify errors first: transient and rate-limit errors get retries with exponential backoff and full jitter, honouring retry-after; permanent errors never retry. Each attempt has a timeout, a circuit breaker per provider fails fast and heals through a half-open probe, a token bucket keeps me under quota, a fallback chain moves to a backup on infrastructure errors only, and idempotency keys make retried side effects happen once. All of it takes an injectable clock, so I test it without sleeping."
