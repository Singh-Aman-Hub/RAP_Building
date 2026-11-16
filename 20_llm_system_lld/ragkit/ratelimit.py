"""ratelimit.py -- an LLM-API rate limiter: requests/min AND tokens/min, per tenant (step 10, problem 3)."""
from __future__ import annotations

from .query import WordTokenCounter
from .resilience import SystemClock, TokenBucket


class LLMRateLimiter:
    """LLM providers limit BOTH requests per minute (RPM) and tokens per minute (TPM). Both buckets must have room."""

    def __init__(self, rpm: int, tpm: int, clock=None):
        self.clock = clock or SystemClock
        self.requests = TokenBucket(rpm / 60, rpm, self.clock)       # refill rate per second, burst = one minute's worth
        self.tokens = TokenBucket(tpm / 60, tpm, self.clock)

    def try_acquire(self, est_tokens: int) -> bool:
        if self.requests.wait_time(1) > 0 or self.tokens.wait_time(est_tokens) > 0:
            return False                                              # check BOTH first so we never spend one bucket for nothing
        return self.requests.try_acquire(1) and self.tokens.try_acquire(est_tokens)

    def acquire(self, est_tokens: int) -> float:
        waited = 0.0
        while not self.try_acquire(est_tokens):
            w = max(self.requests.wait_time(1), self.tokens.wait_time(est_tokens), 1e-6)
            self.clock.sleep(w)
            waited += w
        return waited

    def settle(self, estimated: int, actual: int) -> None:
        """After the call we know the real token count: give back what we over-reserved."""
        if actual < estimated:
            self.tokens.tokens = min(self.tokens.capacity, self.tokens.tokens + (estimated - actual))


class TenantLimiters:
    """One limiter per tenant, created on first use, so one noisy tenant cannot starve the others."""

    def __init__(self, rpm: int, tpm: int, clock=None):
        self.rpm, self.tpm, self.clock = rpm, tpm, clock
        self._by: dict[str, LLMRateLimiter] = {}

    def for_tenant(self, tenant: str) -> LLMRateLimiter:
        if tenant not in self._by:
            self._by[tenant] = LLMRateLimiter(self.rpm, self.tpm, self.clock)
        return self._by[tenant]


class RateLimitedGenerator:
    """Decorator: wait for capacity, call, settle the token estimate."""

    def __init__(self, inner, limiter: LLMRateLimiter, max_output_tokens: int = 200):
        self.inner, self.limiter, self.max_out = inner, limiter, max_output_tokens
        self.count = WordTokenCounter().count
        self.total_wait = 0.0

    def generate(self, system: str, user: str) -> str:
        est = self.count(system + " " + user) + self.max_out         # reserve worst case: input + max output
        self.total_wait += self.limiter.acquire(est)
        out = self.inner.generate(system, user)
        self.limiter.settle(est, self.count(system + " " + user) + self.count(out))
        return out
