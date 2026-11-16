"""Solutions for step 06: 6.1 (equal vs full vs decorrelated jitter), 6.2 (break: retry permanent), 6.3 (token bucket)."""
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragkit.resilience import PermanentError, RetryingGenerator, RetryPolicy, TokenBucket
from ragkit.testing import FakeClock, FlakyGenerator

# 6.1 1000 clients, 4th retry (cap = min(8, 0.5 * 2**3) = 4 s). Full vs equal jitter: how spread out are the retries?
rng = random.Random(3)
cap, base, n = 4.0, 0.5, 1000
def peak(delays): return max(Counter(int(d / 0.5) for d in delays).values())
full = [rng.uniform(0, cap) for _ in range(n)]
equal = [cap / 2 + rng.uniform(0, cap / 2) for _ in range(n)]
print(f"6.1 peak retries per 0.5 s slot (1000 clients): none={n} full={peak(full)} equal={peak(equal)}")
print(f"    mean wait: full={sum(full)/n:.2f}s equal={sum(equal)/n:.2f}s (equal jitter waits longer on average: it keeps half of the cap)")

# 6.2 break: retry_on=(Exception,) retries a 401
g = FlakyGenerator([PermanentError("401")] * 9)
try:
    RetryingGenerator(g, RetryPolicy(max_attempts=4, retry_on=(Exception,)), FakeClock(), random.Random(0)).generate("s", "u")
except PermanentError:
    pass
print("6.2 provider calls with retry_on=(Exception,):", g.calls, "(should be 1)")
g2 = FlakyGenerator([PermanentError("401")] * 9)
try:
    RetryingGenerator(g2, RetryPolicy(max_attempts=4), FakeClock(), random.Random(0)).generate("s", "u")
except PermanentError:
    pass
print("    with the default retry_on=(TransientError,):", g2.calls)

# 6.3 token bucket from scratch (what you write on a whiteboard)
class Bucket:
    def __init__(self, rate, cap, now): self.rate, self.cap, self.tokens, self.t = rate, cap, cap, now
    def take(self, n, now):
        self.tokens = min(self.cap, self.tokens + (now - self.t) * self.rate); self.t = now
        if n <= self.tokens:
            self.tokens -= n
            return True
        return False
b = Bucket(rate=2, cap=4, now=0.0)
print("6.3 from scratch:", [b.take(1, 0.0) for _ in range(5)], "then at t=1.0:", [b.take(1, 1.0) for _ in range(3)])
