"""
GOAL:  Three caches for an LLM app: exact-match (hash of normalised prompt + params + scope), semantic (embedding
       similarity >= threshold) and provider-side prompt caching (cost arithmetic), with TTL, LRU eviction and
       invalidation when the index version changes.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 20_llm_system_lld/07_caching/main.py
EXPECTED OUTPUT (shortened):
    key stability ... same: True
    q1 miss -> q1 exact -> paraphrase semantic(0.9x) -> after re-index: miss
    TTL: hit at 59 s, expired at 61 s ; LRU: 'b' evicted
    prompt caching: 10 calls with a 10k-token prefix saves ~77%
NEEDS: ragkit only. HashingEmbedder (offline, instant) + FakeClock.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragkit.cache import MISS, CachedAnswerer, SemanticCache, TTLLRUCache, cache_key, normalize, prompt_cache_cost
from ragkit.factory import build_system
from ragkit.ingest import DirectoryLoader, HashingEmbedder
from ragkit.testing import FakeClock

KB = Path(__file__).resolve().parents[2] / "14_rag_and_retrieval" / "data" / "kb"

# ---- 1. Exact cache key: what must be in the hash ---------------------------------------------------------------
k1 = cache_key("sys", "How many days for a refund?", {"temperature": 0, "max_tokens": 200}, scope="idx-v1")
k2 = cache_key("sys", "  how many   DAYS for a refund? ", {"max_tokens": 200, "temperature": 0}, scope="idx-v1")
print("normalised:", repr(normalize("  how many   DAYS for a refund? ")))
print("key stable under whitespace/case/param order:", k1 == k2, "|", k1)
print("different temperature ->", cache_key("sys", "How many days for a refund?", {"temperature": 0.7, "max_tokens": 200}, "idx-v1") != k1,
      "| different index version ->", cache_key("sys", "How many days for a refund?", {"temperature": 0, "max_tokens": 200}, "idx-v2") != k1)

# ---- 2. TTL and LRU on the cache itself -------------------------------------------------------------------------------------
clock = FakeClock()
c = TTLLRUCache(capacity=2, ttl=60, clock=clock)
c.put("a", 1)
clock.advance(59); print("TTL: at 59 s ->", c.get("a"))
clock.advance(2);  print("     at 61 s ->", "MISS" if c.get("a") is MISS else "hit", "| expirations:", c.stats.expirations)
c.put("a", 1); c.put("b", 2); c.get("a"); c.put("c", 3)           # 'b' is the least recently used
print("LRU: capacity 2, touched a, added c ->", {k: ("kept" if c.get(k) is not MISS else "evicted") for k in "abc"}, "| evictions:", c.stats.evictions)

# ---- 3. The full stack on a real RAG service ------------------------------------------------------------------------------------
system = build_system({}, DirectoryLoader(KB))
system.pipeline.run()
emb, clock = HashingEmbedder(128), FakeClock()
calls = {"n": 0}
inner = system.service
real_answer = inner.answer
def counting_answer(q):
    calls["n"] += 1
    return real_answer(q)
inner.answer = counting_answer                                      # count how often the real pipeline runs
version = {"v": system.pipeline.index_version}
cached = CachedAnswerer(inner, TTLLRUCache(100, 300, clock), SemanticCache(emb, threshold=0.75, ttl=300, clock=clock),
                        scope=lambda: version["v"])
for q in ["How many days do I have to get a refund?", "How many days do I have to get a refund?",
          "how many days do i have to get a refund", "How many days do I have to get my refund?",
          "Is shipping free?"]:
    a = cached.answer(q)
    print(f"   {q!r:46s} -> cache={a.meta['cache']:16s} pipeline runs so far={calls['n']}")
print("   hit rates: exact", round(cached.exact.stats.hit_rate, 2), "| semantic", round(cached.semantic.stats.hit_rate, 2))

# ---- 4. Invalidation: the index changed, so every old answer is suspect ---------------------------------------------------------------
version["v"] = "idx-v2"                                             # what a re-index with a new embedder/chunker does
print("   after re-index ->", cached.answer("How many days do I have to get a refund?").meta["cache"], "| pipeline runs:", calls["n"])

# ---- 5. The semantic-cache danger: a high similarity is not the same meaning --------------------------------------------------------------
sc = SemanticCache(emb, threshold=0.8, clock=FakeClock())
sc.put("Is shipping free on my order", "Yes, free above 799 rupees.")
for q in ["Is shipping free on my order today", "Is shipping not free on my order"]:
    r = sc.lookup(q)
    print(f"   {q!r:42s} -> {'HIT sim=%.2f' % r[1] if r else 'miss'}")

# ---- 6. Provider-side prompt caching: arithmetic only (no API call) ---------------------------------------------------------------------------
for calls_n in (1, 2, 10, 100):
    r = prompt_cache_cost(prefix_tokens=10_000, suffix_tokens=200, calls=calls_n)
    print(f"   prompt caching, 10k-token prefix, {calls_n:3d} calls: ${r['without_cache_usd']:.4f} -> ${r['with_cache_usd']:.4f} ({r['saving_pct']:+.0f}% saving)")

# TRY IT: set the semantic threshold to 0.95 and rerun section 3: which question stops being a semantic hit?
# TRY IT: set capacity=2 in TTLLRUCache for the exact cache and ask 3 different questions, then the first again.
# BREAK IT: in CachedAnswerer change `is not None` back to `if self.semantic:`; the semantic cache is EMPTY on first use, so it is falsy. Why?
