import pytest

from ragkit.cache import (MISS, CachedAnswerer, CachingGenerator, SemanticCache, TTLLRUCache, cache_key, normalize,
                          prompt_cache_cost)
from ragkit.domain import Answer, Citation
from ragkit.ingest import HashingEmbedder
from ragkit.testing import FakeClock


def test_normalize_and_key_stability():
    assert normalize("  Hello   WORLD ") == "hello world"
    k1 = cache_key("s", "Hi  there", {"a": 1, "b": 2}, "v1")
    assert k1 == cache_key("s", "hi there", {"b": 2, "a": 1}, "v1")
    assert k1 != cache_key("s", "hi there", {"a": 1, "b": 2}, "v2")        # scope changes the key
    assert k1 != cache_key("s", "hi there", {"a": 1, "b": 3}, "v1")        # params change the key


def test_ttl_expiry():
    clock = FakeClock()
    c = TTLLRUCache(10, ttl=60, clock=clock)
    c.put("k", "v")
    assert c.get("k") == "v"
    clock.advance(61)
    assert c.get("k") is MISS and c.stats.expirations == 1 and len(c) == 0


def test_lru_eviction_order():
    c = TTLLRUCache(2, ttl=None, clock=FakeClock())
    c.put("a", 1); c.put("b", 2)
    c.get("a")                                             # a is now newer than b
    c.put("c", 3)                                          # evicts b
    assert c.get("b") is MISS and c.get("a") == 1 and c.get("c") == 3 and c.stats.evictions == 1


def test_none_is_cacheable():
    c = TTLLRUCache(2, None, FakeClock())
    c.put("k", None)
    assert c.get("k") is None


def test_hit_rate():
    c = TTLLRUCache(2, None, FakeClock())
    c.put("a", 1)
    c.get("a"); c.get("zzz")
    assert c.stats.hit_rate == 0.5


def test_semantic_cache_hit_miss_scope_ttl():
    clock = FakeClock()
    sc = SemanticCache(HashingEmbedder(128), threshold=0.7, ttl=100, clock=clock)
    sc.put("how many days do I have for a refund", "A", scope="v1")
    value, sim = sc.lookup("How many days do I have for a refund?", "v1")
    assert value == "A" and sim > 0.95
    assert sc.lookup("what is the tracking link", "v1") is None
    assert sc.lookup("how many days do I have for a refund", "v2") is None   # other index version: no cross-over
    clock.advance(101)
    assert sc.lookup("how many days do I have for a refund", "v1") is None and len(sc) == 0


def test_semantic_cache_capacity():
    sc = SemanticCache(HashingEmbedder(64), capacity=2, ttl=None, clock=FakeClock())
    for q in ("alpha beta", "gamma delta", "epsilon zeta"):
        sc.put(q, q)
    assert len(sc) == 2 and sc.stats.evictions == 1


def test_caching_generator():
    class G:
        n = 0
        def generate(self, s, u):
            G.n += 1
            return "out"
    ver = ["v1"]
    g = CachingGenerator(G(), TTLLRUCache(8, None, FakeClock()), scope=lambda: ver[0])
    g.generate("s", "q"); g.generate("s", " Q ")
    assert G.n == 1
    ver[0] = "v2"
    g.generate("s", "q")
    assert G.n == 2


class CountingAnswerer:
    def __init__(self):
        self.n = 0
        self.refuse = False

    def answer(self, q):
        self.n += 1
        if self.refuse:
            return Answer("no", refused=True, refusal_reason="x")
        return Answer(f"answer to {q} [1]", (Citation(1, "c"),))


def test_cached_answerer_layers_and_invalidation():
    clock, inner, ver = FakeClock(), CountingAnswerer(), ["v1"]
    ca = CachedAnswerer(inner, TTLLRUCache(8, 300, clock), SemanticCache(HashingEmbedder(128), 0.8, 8, 300, clock),
                        scope=lambda: ver[0])
    assert ca.answer("how many days for a refund").meta["cache"] == "miss"
    assert ca.answer("how many days for a refund").meta["cache"] == "exact"
    assert ca.answer("how many days for a refund?").meta["cache"].startswith("semantic")   # '?' changes the exact key only
    assert inner.n == 1
    r = ca.answer("how many days for a refund please")
    assert r.meta["cache"].startswith("semantic") and inner.n == 1
    ver[0] = "v2"                                                  # re-index happened
    assert ca.answer("how many days for a refund").meta["cache"] == "miss" and inner.n == 2


def test_refusals_not_cached():
    inner = CountingAnswerer()
    inner.refuse = True
    ca = CachedAnswerer(inner, TTLLRUCache(8, None, FakeClock()))
    ca.answer("q"); ca.answer("q")
    assert inner.n == 2


def test_prompt_cache_cost():
    r = prompt_cache_cost(prefix_tokens=10_000, suffix_tokens=200, calls=10)
    assert r["with_cache_usd"] < r["without_cache_usd"] and 70 < r["saving_pct"] < 90
    one = prompt_cache_cost(10_000, 200, 1)
    assert one["with_cache_usd"] > one["without_cache_usd"]         # a single call only pays the write premium
