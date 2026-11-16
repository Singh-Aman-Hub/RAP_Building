"""Solutions for step 07: 7.1 (LRU from scratch), 7.2 (prompt-caching break-even), 7.3 (empty container is falsy)."""
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragkit.cache import SemanticCache, prompt_cache_cost
from ragkit.ingest import HashingEmbedder

# 7.1 LRU cache in 12 lines
class LRU:
    def __init__(self, cap): self.cap, self.d = cap, OrderedDict()
    def get(self, k):
        if k not in self.d: return None
        self.d.move_to_end(k); return self.d[k]
    def put(self, k, v):
        self.d[k] = v; self.d.move_to_end(k)
        if len(self.d) > self.cap: self.d.popitem(last=False)
c = LRU(2); c.put("a", 1); c.put("b", 2); c.get("a"); c.put("c", 3)
print("7.1 LRU keys after put a,b / get a / put c:", list(c.d))

# 7.2 how many calls until prompt caching pays off? (write 1.25x, read 0.1x)
for prefix in (1000, 10_000):
    n = next(k for k in range(1, 50) if prompt_cache_cost(prefix, 200, k)["with_cache_usd"] < prompt_cache_cost(prefix, 200, k)["without_cache_usd"])
    print(f"7.2 prefix {prefix:6d} tokens: caching wins from call number {n}")
print("    analytic: write premium 0.25*P is repaid by 0.9*P saved on every later call -> from the 2nd call whenever 0.9*P > 0.25*P")

# 7.3 empty containers are falsy
sc = SemanticCache(HashingEmbedder(8))
print("7.3 bool(empty SemanticCache) =", bool(sc), "-> `if self.semantic:` would skip it; use `is not None`")
