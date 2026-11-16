"""resilience.py -- what keeps an LLM app alive when providers fail (step 06). Everything from scratch, injectable clock."""
from __future__ import annotations

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence, TypeVar

from .ids import content_hash

T = TypeVar("T")


# ------------------------------------------------------------------ error taxonomy: what is worth retrying?
class LLMError(Exception):
    pass


class TransientError(LLMError):
    """Network blip, 500, 503, overloaded: trying again may work."""


class RateLimitedError(TransientError):
    def __init__(self, msg: str = "rate limited", retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after                         # the server's Retry-After header, if it sent one


class PermanentError(LLMError):
    """400 bad request, 401 bad key, content policy: retrying the same call cannot help."""


class CircuitOpenError(LLMError):
    pass


class CallTimeout(TransientError):
    pass


class SystemClock:
    now = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)


# ------------------------------------------------------------------ retry with exponential backoff + jitter
@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 4
    base_delay: float = 0.5
    max_delay: float = 8.0
    retry_on: tuple = (TransientError,)

    def backoff(self, attempt: int, rng: random.Random) -> float:
        """'Full jitter': uniform(0, min(max_delay, base * 2**attempt)). attempt is 0 for the first retry."""
        return rng.uniform(0, min(self.max_delay, self.base_delay * 2 ** attempt))


def retry_call(fn: Callable[[], T], policy: RetryPolicy, clock=None, rng: random.Random | None = None,
               on_retry: Callable[[int, Exception, float], None] | None = None) -> T:
    clock, rng = clock or SystemClock, rng or random.Random()
    for attempt in range(policy.max_attempts):
        try:
            return fn()
        except policy.retry_on as exc:
            if attempt == policy.max_attempts - 1:
                raise                                          # out of attempts: surface the last error
            delay = policy.backoff(attempt, rng)
            if isinstance(exc, RateLimitedError) and exc.retry_after is not None:
                delay = max(delay, exc.retry_after)            # never retry sooner than the server asked
            if on_retry:
                on_retry(attempt + 1, exc, delay)
            clock.sleep(delay)
    raise AssertionError("unreachable")


class RetryingGenerator:
    """Decorator: Generator + retry policy."""

    def __init__(self, inner, policy: RetryPolicy | None = None, clock=None, rng=None, on_retry=None):
        self.inner, self.policy = inner, policy or RetryPolicy()
        self.clock, self.rng, self.on_retry = clock, rng or random.Random(0), on_retry

    def generate(self, system: str, user: str) -> str:
        return retry_call(lambda: self.inner.generate(system, user), self.policy, self.clock, self.rng, self.on_retry)


# ------------------------------------------------------------------ timeout
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


class TimeoutGenerator:
    def __init__(self, inner, seconds: float):
        self.inner, self.seconds = inner, seconds

    def generate(self, system, user):
        return call_with_timeout(lambda: self.inner.generate(system, user), self.seconds)


# ------------------------------------------------------------------ circuit breaker
class State(str, Enum):
    CLOSED = "closed"          # normal: calls pass through
    OPEN = "open"              # failing: calls are rejected immediately (fail fast)
    HALF_OPEN = "half_open"    # probing: let ONE call through to test recovery


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 30.0, clock=None):
        self.threshold, self.recovery, self.clock = failure_threshold, recovery_timeout, clock or SystemClock
        self.state, self.failures, self.opened_at = State.CLOSED, 0, 0.0

    def call(self, fn: Callable[[], T]) -> T:
        if self.state is State.OPEN:
            if self.clock.now() - self.opened_at >= self.recovery:
                self.state = State.HALF_OPEN                   # cool-down over: allow one probe
            else:
                raise CircuitOpenError("circuit open")
        try:
            result = fn()
        except PermanentError:
            raise                                              # caller's fault, not a health signal
        except Exception:
            self.failures += 1
            if self.state is State.HALF_OPEN or self.failures >= self.threshold:
                self.state, self.opened_at = State.OPEN, self.clock.now()
            raise
        self.failures, self.state = 0, State.CLOSED            # success closes the circuit
        return result


class BreakerGenerator:
    def __init__(self, inner, breaker: CircuitBreaker):
        self.inner, self.breaker = inner, breaker

    def generate(self, system, user):
        return self.breaker.call(lambda: self.inner.generate(system, user))


# ------------------------------------------------------------------ token bucket rate limiter
class TokenBucket:
    """Capacity `capacity`, refilled at `rate` units/second. Allows bursts up to capacity, then a steady `rate`."""

    def __init__(self, rate: float, capacity: float, clock=None):
        self.rate, self.capacity, self.clock = rate, capacity, clock or SystemClock
        self.tokens, self.updated = float(capacity), self.clock.now()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = self.clock.now()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now

    def try_acquire(self, n: float = 1) -> bool:
        with self._lock:
            self._refill()
            if n <= self.tokens:
                self.tokens -= n
                return True
            return False

    def wait_time(self, n: float = 1) -> float:
        with self._lock:
            self._refill()
            return max(0.0, (n - self.tokens) / self.rate)

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
class FallbackChain:
    """Try callables in order; return the first success. Remembers which one answered and why others failed."""

    def __init__(self, names_and_fns: Sequence[tuple[str, Callable[..., Any]]],
                 fall_on: tuple = (TransientError, CircuitOpenError)):
        self.options, self.fall_on = list(names_and_fns), fall_on
        self.last_used: str | None = None
        self.failures: list[tuple[str, str]] = []

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


class FallbackGenerator:
    def __init__(self, generators: Sequence[tuple[str, Any]]):
        self.chain = FallbackChain([(n, g.generate) for n, g in generators])

    def generate(self, system, user):
        return self.chain(system, user)


# ------------------------------------------------------------------ idempotency
def idempotency_key(*parts: str) -> str:
    return content_hash(*parts, length=24)


class IdempotencyStore:
    """Run an operation at most once per key; later calls with the same key get the stored result."""

    def __init__(self):
        self._done: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.executions = 0

    def run_once(self, key: str, fn: Callable[[], T]) -> T:
        with self._lock:                                       # (a real system would use a DB row with a unique key)
            if key in self._done:
                return self._done[key]
            result = fn()
            self._done[key] = result
            self.executions += 1
            return result
