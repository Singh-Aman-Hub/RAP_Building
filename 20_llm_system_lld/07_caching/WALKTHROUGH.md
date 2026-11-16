# Step 07 — Caching: exact, semantic, TTL, LRU, invalidation, and provider-side prompt caching

## The story so far
[Step 06](../06_resilience/WALKTHROUGH.md) made calls survive failure. The cheapest call is the one you never make. Module [14 step 10](../../14_rag_and_retrieval/10_advanced_rag_and_vector_dbs/WALKTHROUGH.md) sketched a semantic cache as a dictionary of vectors.

## The pain point
Users ask the same question again and again ("how many days for a refund?"), each time paying 300 ms of retrieval, a cross-encoder pass and an LLM call. A cache that is too naive is worse than none: it serves yesterday's answer after the policy changed, mixes answers between index versions or tenants, grows without bound, or answers "is shipping NOT free?" with the answer for "is shipping free?".

## The spark
Three layers, each with a different risk: an **exact cache** keyed by a hash of everything that can change the answer; a **semantic cache** that matches *near* questions by embedding similarity above a threshold; and **provider-side prompt caching**, where the API reuses the processed *prefix* of a long prompt. All client-side caches need **TTL** (staleness limit), **LRU** (size limit) and a **scope in the key** (index version, model, tenant) so changes invalidate automatically.

## Goal
Implement `TTLLRUCache`, `SemanticCache`, `CachingGenerator` and `CachedAnswerer`; see the hit/miss/expire/evict numbers; invalidate by changing the index version; measure the semantic-threshold trade-off; compute prompt-caching savings and know the provider's rules.

## The theory (read this before the interview)
### Where to cache (cheapest, safest first)
| Layer | Key | Hits when | Risk | Saves |
|---|---|---|---|---|
| Embedding cache | hash(text) | same text embedded again | none (pure function) | embedding calls |
| Exact answer cache | hash(normalised prompt, params, **scope**) | identical request | stale answer after data change | whole pipeline + LLM |
| Semantic cache | embedding of the question, similarity >= threshold, same **scope** | paraphrases | **wrong answer for a near-miss question** | whole pipeline + LLM |
| Provider prompt cache | exact byte prefix of the prompt | same long prefix (system prompt, tools, documents) | none for correctness; only cost/latency | input-token cost, time to first token |
| Retrieval cache | hash(query, k, filters, index version) | same query | stale hits | retrieval only |

### What must be in an exact key
`cache_key = sha256(json{normalised system, normalised prompt, sampling params, scope})`. Include **temperature/top_p/max_tokens/model** (a temperature-1 sample must not be served for a temperature-0 request), and the **scope**: index version, model id, tenant/permission group. Normalise whitespace (and case, for questions); `sort_keys=True` so dict order does not matter. Only cache deterministic calls (temperature 0) unless you accept replaying one random sample.

### TTL and LRU
- **TTL (time to live)**: each entry expires after N seconds. Bounds *staleness* when you cannot detect changes. Implemented lazily: checked on read (`len()` may count expired entries until they are touched; Experiments).
- **LRU (least recently used)**: bounded size; on overflow evict the entry unused the longest. An `OrderedDict` with `move_to_end` on every hit gives O(1) get/put. Other policies: LFU (least frequently used), FIFO, size-aware. 
- `MISS` is a unique sentinel object so `None` can be a legitimate cached value.

### Invalidation
"There are only two hard things in computer science: cache invalidation and naming things." Strategies: **versioned keys** (the scope contains `index_version`, so re-indexing makes old keys unreachable: they simply age out; no scan needed), **TTL**, **explicit purge** (by document id), **event-driven** (subscribe to the ingestion pipeline's events). Never cache refusals: tomorrow's index may contain the answer.

### Semantic cache
`lookup(question)`: embed, compare with every entry *in the same scope*, return the best if `cosine >= threshold`. Threshold trade-off (Experiments): high = few hits, few wrong answers; low = more hits, more wrong answers. Embedding similarity measures **topic similarity, not equivalence**: "is shipping free" and "is shipping not free" are close in vector space (0.93 in the demo) but have opposite answers. Mitigations: high threshold (0.9+), embeddings plus a cheap verification step, restrict to FAQ-style questions, include negation checks, never use across permission scopes (**ACL leak**: user A's cached answer shown to user B), and log hits for audit. Cost: one embedding per lookup plus a vector search; for large caches use an ANN index.

### Provider-side prompt caching (concept; verified in the Claude docs, Nov 2026)
The API stores the processed **prefix** of your prompt (order: tools, then system, then messages) up to a marked `cache_control` breakpoint (`{"type": "ephemeral"}`; up to 4 breakpoints, or one top-level automatic setting). A later request whose prefix is **byte-identical** reads the cache instead of recomputing it.
- Default lifetime **5 minutes**, refreshed on each hit; optional **1 hour** (`"ttl": "1h"`).
- Pricing relative to base input price: 5-minute write **1.25x**, 1-hour write **2x**, cache read **0.1x** (some newer models list lower read multipliers; check the pricing page and model card).
- Minimum cacheable prompt length depends on the model (512 to 4,096 tokens per the docs); shorter prompts are processed uncached without an error.
- The response `usage` reports `cache_creation_input_tokens`, `cache_read_input_tokens` and `input_tokens` (tokens after the last breakpoint).
- Design consequence: put **stable content first** (tools, system prompt, instructions, big documents) and **volatile content last** (the user question). A timestamp or request id at the top of the system prompt destroys every hit.
- It does not change outputs, only cost and latency. It is *not* an answer cache.
`prompt_cache_cost` reproduces the arithmetic: one call costs *more* (write premium); from the second call you win; 10 calls with a 10k-token prefix save about 77%.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Exact cache | response cache, memoisation, key-value cache | Close | Memoisation = caching a pure function's result; an LLM response cache is only that when the call is deterministic. |
| Semantic cache | similarity cache, fuzzy cache, GPTCache-style | Mostly | Hit by closeness in embedding space rather than by equal bytes. |
| Prompt caching | prefix caching, context caching, KV-cache reuse | Close | Provider-side reuse of the processed prompt prefix (KV cache); different providers use different names and rules. |
| TTL | expiry, max-age, time-to-live | Same | Seconds until an entry is considered stale. |
| LRU | eviction policy, cache replacement | No | LRU is one eviction policy; LFU and FIFO are others. |
| Cache invalidation | purge, bust, flush, versioned keys | Close | Making sure stale entries are not served; versioned keys avoid deleting anything. |
| Cache scope / namespace | tenant key, partition, cache key prefix | Close | The part of the key that stops entries from crossing index versions, models or users. |
| KV cache (inside the model) | attention key-value cache | No | Per-request tensors reused while decoding tokens (module 06); prompt caching persists a prefix's KV state across requests. |

## Run it
```bash
# from this step folder
HF_HUB_OFFLINE=1 python main.py
# or from the repo root
HF_HUB_OFFLINE=1 .venv/bin/python 20_llm_system_lld/07_caching/main.py
```


## Code, step by step
Two files: `main.py` and `ragkit/cache.py`. Every block is listed.

#### Walking `main.py`

### 1. Header docstring  (`main.py` lines 1–14)
```python
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
```
- **In simple words:** What the file shows.
- **Exactly:** GOAL/RUN/EXPECTED/NEEDS.
- **Why it is here:** Orientation.

### 2. Imports  (`main.py` lines 15–24)
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragkit.cache import MISS, CachedAnswerer, SemanticCache, TTLLRUCache, cache_key, normalize, prompt_cache_cost
from ragkit.factory import build_system
from ragkit.ingest import DirectoryLoader, HashingEmbedder
from ragkit.testing import FakeClock

KB = Path(__file__).resolve().parents[2] / "14_rag_and_retrieval" / "data" / "kb"
```
- **In simple words:** The cache classes, the factory and fakes.
- **Exactly:** `MISS` sentinel, `CachedAnswerer`, `SemanticCache`, `TTLLRUCache`, `cache_key`, `normalize`, `prompt_cache_cost`; `FakeClock` for time.
- **Why it is here:** Time is injected so TTL demos are instant.

### 3. What goes into the exact key  (`main.py` lines 26–32)
```python
# ---- 1. Exact cache key: what must be in the hash ---------------------------------------------------------------
k1 = cache_key("sys", "How many days for a refund?", {"temperature": 0, "max_tokens": 200}, scope="idx-v1")
k2 = cache_key("sys", "  how many   DAYS for a refund? ", {"max_tokens": 200, "temperature": 0}, scope="idx-v1")
print("normalised:", repr(normalize("  how many   DAYS for a refund? ")))
print("key stable under whitespace/case/param order:", k1 == k2, "|", k1)
print("different temperature ->", cache_key("sys", "How many days for a refund?", {"temperature": 0.7, "max_tokens": 200}, "idx-v1") != k1,
      "| different index version ->", cache_key("sys", "How many days for a refund?", {"temperature": 0, "max_tokens": 200}, "idx-v2") != k1)
```
- **In simple words:** Show that spacing, case and parameter order do not matter, but temperature and index version do.
- **Exactly:** `normalize('  how many   DAYS for a refund? ')` -> `'how many days for a refund?'`; both spellings give key `46d3a56af5f210f64673286c940e3d0c`; changing temperature or `scope` changes the key.
- **Why it is here:** A key that ignores parameters serves wrong samples; one that ignores whitespace misses needlessly.

### 4. TTL and LRU on the bare cache  (`main.py` lines 34–41)
```python
# ---- 2. TTL and LRU on the cache itself -------------------------------------------------------------------------------------
clock = FakeClock()
c = TTLLRUCache(capacity=2, ttl=60, clock=clock)
c.put("a", 1)
clock.advance(59); print("TTL: at 59 s ->", c.get("a"))
clock.advance(2);  print("     at 61 s ->", "MISS" if c.get("a") is MISS else "hit", "| expirations:", c.stats.expirations)
c.put("a", 1); c.put("b", 2); c.get("a"); c.put("c", 3)           # 'b' is the least recently used
print("LRU: capacity 2, touched a, added c ->", {k: ("kept" if c.get(k) is not MISS else "evicted") for k in "abc"}, "| evictions:", c.stats.evictions)
```
- **In simple words:** A capacity-2 cache with a 60-second lifetime.
- **Exactly:** At 59 s the value is returned (`1`); at 61 s `MISS` and `expirations: 1`. Then `put a, put b, get a, put c`: `b` (least recently used) is evicted: `{'a': 'kept', 'b': 'evicted', 'c': 'kept'}`, `evictions: 1`.
- **Why it is here:** Proves both limits before building on them.

### 5. Build a real service and a cached wrapper  (`main.py` lines 43–56)
```python
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
```
- **In simple words:** A RAG service from step 05's factory, wrapped by the cache; we count how often the real pipeline runs.
- **Exactly:** `inner.answer = counting_answer` replaces the method with a counting wrapper (monkey patch); `version = {'v': index_version}` is read through `scope=lambda: version['v']` on every request; semantic threshold 0.75, TTL 300.
- **Why it is here:** The lambda scope is how invalidation reaches the cache without the cache knowing about indexing.

### 6. Five questions  (`main.py` lines 57–62)
```python
for q in ["How many days do I have to get a refund?", "How many days do I have to get a refund?",
          "how many days do i have to get a refund", "How many days do I have to get my refund?",
          "Is shipping free?"]:
    a = cached.answer(q)
    print(f"   {q!r:46s} -> cache={a.meta['cache']:16s} pipeline runs so far={calls['n']}")
print("   hit rates: exact", round(cached.exact.stats.hit_rate, 2), "| semantic", round(cached.semantic.stats.hit_rate, 2))
```
- **In simple words:** Same question twice, a case/punctuation variant, a paraphrase, an unrelated question.
- **Exactly:** Outputs: `miss` (runs=1), `exact` (runs=1), `semantic(1.00)` for the lower-cased version without '?' (the exact key differs, the embedding is identical), `semantic(0.90)` for 'get my refund', `miss` for 'Is shipping free?' (runs=2). Hit rates: exact 0.2, semantic 0.5.
- **Why it is here:** Three outcomes of the lookup order: exact, then semantic, then compute.

### 7. Change the index version  (`main.py` lines 64–66)
```python
# ---- 4. Invalidation: the index changed, so every old answer is suspect ---------------------------------------------------------------
version["v"] = "idx-v2"                                             # what a re-index with a new embedder/chunker does
print("   after re-index ->", cached.answer("How many days do I have to get a refund?").meta["cache"], "| pipeline runs:", calls["n"])
```
- **In simple words:** Pretend a re-index happened.
- **Exactly:** `version['v'] = 'idx-v2'`; the same question is a `miss` and the pipeline runs a third time.
- **Why it is here:** Old entries became unreachable without being deleted; they age out by TTL/LRU.

### 8. A near-miss that means the opposite  (`main.py` lines 68–73)
```python
# ---- 5. The semantic-cache danger: a high similarity is not the same meaning --------------------------------------------------------------
sc = SemanticCache(emb, threshold=0.8, clock=FakeClock())
sc.put("Is shipping free on my order", "Yes, free above 799 rupees.")
for q in ["Is shipping free on my order today", "Is shipping not free on my order"]:
    r = sc.lookup(q)
    print(f"   {q!r:42s} -> {'HIT sim=%.2f' % r[1] if r else 'miss'}")
```
- **In simple words:** Store 'Is shipping free on my order' and look up two variants.
- **Exactly:** '...order today' -> `HIT sim=0.93` (fine); 'Is shipping not free on my order' -> also `HIT sim=0.93` (WRONG: opposite meaning).
- **Why it is here:** Concrete evidence that similarity is not equivalence.

### 9. The savings arithmetic  (`main.py` lines 75–78)
```python
# ---- 6. Provider-side prompt caching: arithmetic only (no API call) ---------------------------------------------------------------------------
for calls_n in (1, 2, 10, 100):
    r = prompt_cache_cost(prefix_tokens=10_000, suffix_tokens=200, calls=calls_n)
    print(f"   prompt caching, 10k-token prefix, {calls_n:3d} calls: ${r['without_cache_usd']:.4f} -> ${r['with_cache_usd']:.4f} ({r['saving_pct']:+.0f}% saving)")
```
- **In simple words:** Compute cost with and without prefix caching for 1, 2, 10, 100 calls.
- **Exactly:** 10k-token prefix + 200-token suffix at $3/Mtok: 1 call $0.0306 -> $0.0381 (-25%, a loss: write premium); 2 calls +32%; 10 calls $0.3060 -> $0.0705 (+77%); 100 calls +87%.
- **Why it is here:** Prompt caching pays off from the second call within the TTL; the saving tends to 90% of the prefix share.

### 10. Exercises  (`main.py` lines 80–82)
```python
# TRY IT: set the semantic threshold to 0.95 and rerun section 3: which question stops being a semantic hit?
# TRY IT: set capacity=2 in TTLLRUCache for the exact cache and ask 3 different questions, then the first again.
# BREAK IT: in CachedAnswerer change `is not None` back to `if self.semantic:`; the semantic cache is EMPTY on first use, so it is falsy. Why?
```
- **In simple words:** Things to try.
- **Exactly:** Comment lines.
- **Why it is here:** See exercises.md.


#### Walking `ragkit/cache.py`

### 11. cache.py header and imports  (`ragkit/cache.py` lines 1–15)
```python
"""cache.py -- exact cache, semantic cache, TTL + LRU, invalidation by index version (step 07)."""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np

from .domain import Answer
from .ids import content_hash
from .resilience import SystemClock

MISS = object()                                               # sentinel: lets None be a legitimate cached value
```
- **In simple words:** All caches in one file.
- **Exactly:** Imports `OrderedDict`, `dataclass`, `replace`, numpy, `Answer`, `content_hash` and `SystemClock` (the injectable clock from step 06).
- **Why it is here:** Reuses step 06's clock so one `FakeClock` drives everything.

### 12. MISS sentinel and normalize()  (`ragkit/cache.py` lines 18–20)
```python
def normalize(text: str) -> str:
    """Collapse whitespace + casefold. (Casefold is safe for questions; do NOT do it for case-sensitive code prompts.)"""
    return " ".join(text.split()).casefold()
```
- **In simple words:** A unique 'not found' marker, and a text normaliser.
- **Exactly:** `MISS = object()`; `' '.join(text.split()).casefold()` collapses all whitespace and ignores case.
- **Why it is here:** `None` must be cacheable; normalisation raises the hit rate.

### 13. cache_key: hash everything that matters  (`ragkit/cache.py` lines 23–28)
```python
def cache_key(system: str, user: str, params: dict | None = None, scope: str = "") -> str:
    """Hash of everything that can change the answer: normalized prompt, sampling params, and the SCOPE
    (index version, model, tenant). Different scope -> different key -> automatic invalidation."""
    payload = json.dumps({"s": normalize(system), "u": normalize(user), "p": params or {}, "scope": scope},
                         sort_keys=True)                      # sort_keys: {"a":1,"b":2} and {"b":2,"a":1} hash equal
    return content_hash(payload, length=32)
```
- **In simple words:** One stable fingerprint of a request.
- **Exactly:** `json.dumps({'s','u','p','scope'}, sort_keys=True)` then `content_hash(..., length=32)` (128 bits).
- **Why it is here:** Anything that can change the answer but is not in the key is a correctness bug.

### 14. CacheStats: counters  (`ragkit/cache.py` lines 31–41)
```python
@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
```
- **In simple words:** Hits, misses, evictions, expirations.
- **Exactly:** Dataclass with a `hit_rate` property = `hits / (hits + misses)`, 0 if no traffic.
- **Why it is here:** A cache you cannot measure cannot be tuned.

### 15. TTLLRUCache: state  (`ragkit/cache.py` lines 44–50)
```python
class TTLLRUCache:
    """Bounded cache: entries expire after `ttl` seconds, and the least recently USED goes first when full."""

    def __init__(self, capacity: int = 128, ttl: float | None = 300.0, clock=None):
        self.capacity, self.ttl, self.clock = capacity, ttl, clock or SystemClock
        self._data: OrderedDict[str, tuple[Any, float | None]] = OrderedDict()   # order = recency (last = newest)
        self.stats = CacheStats()
```
- **In simple words:** An ordered dictionary with limits.
- **Exactly:** `OrderedDict`: key -> `(value, expires_at or None)`; order = recency (last = newest). `capacity=128`, `ttl=300.0`, injectable clock.
- **Why it is here:** `ttl=None` = never expire.

### 16. TTLLRUCache.get  (`ragkit/cache.py` lines 52–65)
```python
    def get(self, key: str):
        item = self._data.get(key)
        if item is None:
            self.stats.misses += 1
            return MISS
        value, expires = item
        if expires is not None and self.clock.now() >= expires:
            del self._data[key]                                # lazy expiry: checked on read
            self.stats.expirations += 1
            self.stats.misses += 1
            return MISS
        self._data.move_to_end(key)                            # touched -> most recently used
        self.stats.hits += 1
        return value
```
- **In simple words:** Return the value or `MISS`; expire old entries when touched.
- **Exactly:** Absent -> miss. Present but `now >= expires` -> delete, count `expirations` and a miss. Else `move_to_end(key)` (mark recently used), count a hit.
- **Why it is here:** Lazy expiry: no background thread; cost paid only when an entry is read.

### 17. TTLLRUCache.put  (`ragkit/cache.py` lines 67–75)
```python
    def put(self, key: str, value: Any) -> None:
        self._data[key] = (value, None if self.ttl is None else self.clock.now() + self.ttl)
        self._data.move_to_end(key)
        while len(self._data) > self.capacity:
            self._data.popitem(last=False)                     # drop the least recently used
            self.stats.evictions += 1

    def __len__(self):
        return len(self._data)
```
- **In simple words:** Store and evict the oldest if over capacity.
- **Exactly:** Writes `(value, now + ttl)`, `move_to_end`, then `while len > capacity: popitem(last=False)` (removes the least recently used) counting `evictions`.
- **Why it is here:** Bounded memory.

### 18. SemanticCache: state  (`ragkit/cache.py` lines 78–86)
```python
class SemanticCache:
    """Hit when a NEW question is close enough (cosine >= threshold) to an OLD one in the same scope."""

    def __init__(self, embedder, threshold: float = 0.85, capacity: int = 128, ttl: float | None = 300.0, clock=None):
        self.embedder, self.threshold, self.capacity, self.ttl = embedder, threshold, capacity, ttl
        self.clock = clock or SystemClock
        self._entries: OrderedDict[int, tuple[str, np.ndarray, Any, float | None]] = OrderedDict()
        self._next_id = 0
        self.stats = CacheStats()
```
- **In simple words:** Entries with a vector, a scope, a value and an expiry.
- **Exactly:** `embedder`, `threshold=0.85`, `capacity`, `ttl`; `_entries: OrderedDict[id -> (scope, vector, value, expires)]`.
- **Why it is here:** Same limits as the exact cache.

### 19. SemanticCache.lookup: embed and drop expired  (`ragkit/cache.py` lines 88–93)
```python
    def lookup(self, query: str, scope: str = ""):
        """Return (value, similarity) or None."""
        now, qv = self.clock.now(), self.embedder.embed([normalize(query)])[0]
        for eid in [e for e, (_, _, _, exp) in self._entries.items() if exp is not None and now >= exp]:
            del self._entries[eid]
            self.stats.expirations += 1
```
- **In simple words:** Turn the question into a vector and clean out expired entries.
- **Exactly:** `embedder.embed([normalize(query)])[0]` has shape `(dim,)`; expired entries are removed eagerly here (counted in `expirations`).
- **Why it is here:** Expired entries must not be matched.

### 20. SemanticCache.lookup: best match in scope  (`ragkit/cache.py` lines 94–106)
```python
        best_id, best_sim = None, -1.0
        for eid, (sc, vec, _, _) in self._entries.items():
            if sc != scope:                                    # never cross scopes (index version / tenant / ACL)
                continue
            sim = float(vec @ qv)
            if sim > best_sim:
                best_id, best_sim = eid, sim
        if best_id is not None and best_sim >= self.threshold:
            self._entries.move_to_end(best_id)
            self.stats.hits += 1
            return self._entries[best_id][2], best_sim
        self.stats.misses += 1
        return None
```
- **In simple words:** Compare to every entry of the same scope; keep the best.
- **Exactly:** `sim = float(vec @ qv)` (dot product of unit vectors = cosine); entries with another scope are skipped; the best is a hit only if `best_sim >= threshold`, then `move_to_end` and return `(value, sim)`.
- **Why it is here:** Skipping other scopes prevents cross-version and cross-tenant leaks.

### 21. SemanticCache.put  (`ragkit/cache.py` lines 108–118)
```python
    def put(self, query: str, value: Any, scope: str = "") -> None:
        vec = self.embedder.embed([normalize(query)])[0]
        exp = None if self.ttl is None else self.clock.now() + self.ttl
        self._entries[self._next_id] = (scope, vec, value, exp)
        self._next_id += 1
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)
            self.stats.evictions += 1

    def __len__(self):
        return len(self._entries)
```
- **In simple words:** Add an entry and evict if full.
- **Exactly:** Stores `(scope, vector, value, expires)` under an incrementing id; evicts oldest beyond `capacity`.
- **Why it is here:** Linear scan lookup is fine for thousands of entries; use an ANN index beyond that.

### 22. CachingGenerator: exact cache around a Generator  (`ragkit/cache.py` lines 121–134)
```python
class CachingGenerator:
    """Decorator: exact-match cache in front of any Generator. Only sensible at temperature 0."""

    def __init__(self, inner, cache: TTLLRUCache, scope: Callable[[], str] = lambda: "", params: dict | None = None):
        self.inner, self.cache, self.scope, self.params = inner, cache, scope, params or {}

    def generate(self, system: str, user: str) -> str:
        key = cache_key(system, user, self.params, self.scope())
        hit = self.cache.get(key)
        if hit is not MISS:
            return hit
        out = self.inner.generate(system, user)
        self.cache.put(key, out)
        return out
```
- **In simple words:** A Decorator that memoises `generate`.
- **Exactly:** Key = `cache_key(system, user, params, scope())`; `MISS` -> call inner and store.
- **Why it is here:** Cheap saving at the LLM call level (inside retries, outside the provider).

### 23. CachedAnswerer: constructor  (`ragkit/cache.py` lines 137–143)
```python
class CachedAnswerer:
    """Decorator for the whole RAG service: exact cache first, then semantic cache, then the real pipeline.
    `scope()` is called on EVERY request (usually returns the index version), so re-indexing invalidates old entries."""

    def __init__(self, inner, exact: TTLLRUCache, semantic: SemanticCache | None = None,
                 scope: Callable[[], str] = lambda: ""):
        self.inner, self.exact, self.semantic, self.scope = inner, exact, semantic, scope
```
- **In simple words:** A cache around the whole RAG service.
- **Exactly:** Holds the inner `Answerer`, the exact cache, an optional semantic cache and a `scope` callable.
- **Why it is here:** `scope()` is called per request so a changed index version applies immediately.

### 24. CachedAnswerer.answer: exact, semantic, compute  (`ragkit/cache.py` lines 145–160)
```python
    def answer(self, question: str) -> Answer:
        scope = self.scope()
        key = cache_key("", question, None, scope)
        hit = self.exact.get(key)
        if hit is not MISS:
            return replace(hit, meta={**hit.meta, "cache": "exact"})
        if self.semantic is not None:
            found = self.semantic.lookup(question, scope)
            if found:
                return replace(found[0], meta={**found[0].meta, "cache": f"semantic({found[1]:.2f})"})
        ans = self.inner.answer(question)
        if not ans.refused:                                    # never cache refusals: the index may gain the answer tomorrow
            self.exact.put(key, ans)
            if self.semantic is not None:
                self.semantic.put(question, ans, scope)
        return replace(ans, meta={**ans.meta, "cache": "miss"})
```
- **In simple words:** Three-step lookup, tagging each result.
- **Exactly:** Exact hit -> `replace(hit, meta={..., 'cache': 'exact'})`; semantic hit -> `'semantic(0.90)'`; else compute, and store only non-refused answers; always returns the `cache` label (`miss`, `exact`, `semantic(x)`).
- **Why it is here:** `if self.semantic is not None` (not `if self.semantic`): an EMPTY cache defines `__len__` and is falsy. See Debugging.

### 25. prompt_cache_cost: provider caching arithmetic  (`ragkit/cache.py` lines 163–171)
```python
def prompt_cache_cost(prefix_tokens: int, suffix_tokens: int, calls: int, price_per_mtok: float = 3.0,
                      write_mult: float = 1.25, read_mult: float = 0.1) -> dict:
    """Provider-side prompt caching arithmetic (the provider keeps the processed KV state of a repeated PREFIX).
    First call writes the cache (write_mult x), later calls read it (read_mult x); the suffix is always full price.
    Multipliers default to Anthropic's published 5-minute-cache numbers (check the current pricing page)."""
    p = price_per_mtok / 1e6
    without = calls * (prefix_tokens + suffix_tokens) * p
    with_cache = (prefix_tokens * write_mult + (calls - 1) * prefix_tokens * read_mult + calls * suffix_tokens) * p
    return {"without_cache_usd": without, "with_cache_usd": with_cache, "saving_pct": 100 * (1 - with_cache / without)}
```
- **In simple words:** Compare costs with and without a cached prefix.
- **Exactly:** `without = calls * (prefix + suffix) * p`; `with = (prefix*write_mult + (calls-1)*prefix*read_mult + calls*suffix) * p`; defaults 1.25 and 0.1 (Anthropic's 5-minute numbers; some models differ).
- **Why it is here:** Shows why one-shot prompts lose and repeated long prefixes win.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `OrderedDict.move_to_end(k)` / `popitem(last=False)` | Mark most recently used / remove the oldest |
| `dataclasses.replace(obj, meta=...)` | Copy a frozen dataclass with one field changed |
| `{**hit.meta, 'cache': 'exact'}` | Dict merge with an override |
| `scope: Callable[[], str] = lambda: ''` | A function default evaluated at call time |
| `if x is not None` vs `if x` | Truthiness uses `__len__`: an empty container is falsy |
| `json.dumps(..., sort_keys=True)` | Canonical text for hashing |

## Libraries & functions
| Name | What it does here |
|---|---|
| `collections.OrderedDict` | Recency order for LRU |
| `json` | Canonical key payload |
| `numpy` | Vector dot product in the semantic cache |
| `dataclasses.replace` | Tag answers with their cache source |
| `ragkit.ingest.HashingEmbedder` | Offline embedder used for the semantic cache in the demo |

## Execution trace (real run)
```text
normalised: 'how many days for a refund?'
key stable under whitespace/case/param order: True | 46d3a56af5f210f64673286c940e3d0c
different temperature -> True | different index version -> True
TTL: at 59 s -> 1
     at 61 s -> MISS | expirations: 1
LRU: capacity 2, touched a, added c -> {'a': 'kept', 'b': 'evicted', 'c': 'kept'} | evictions: 1
   'How many days do I have to get a refund?'     -> cache=miss             pipeline runs so far=1
   'How many days do I have to get a refund?'     -> cache=exact            pipeline runs so far=1
   'how many days do i have to get a refund'      -> cache=semantic(1.00)   pipeline runs so far=1
   'How many days do I have to get my refund?'    -> cache=semantic(0.90)   pipeline runs so far=1
   'Is shipping free?'                            -> cache=miss             pipeline runs so far=2
   hit rates: exact 0.2 | semantic 0.5
   after re-index -> miss | pipeline runs: 3
   'Is shipping free on my order today'       -> HIT sim=0.93
   'Is shipping not free on my order'         -> HIT sim=0.93
   prompt caching, 10k-token prefix,   1 calls: $0.0306 -> $0.0381 (-25% saving)
   prompt caching, 10k-token prefix,   2 calls: $0.0612 -> $0.0417 (+32% saving)
   prompt caching, 10k-token prefix,  10 calls: $0.3060 -> $0.0705 (+77% saving)
   prompt caching, 10k-token prefix, 100 calls: $3.0600 -> $0.3945 (+87% saving)
```
- The two spellings of one question hash to the same key; temperature and scope each change it.
- TTL: value at 59 s, `MISS` at 61 s. LRU: after touching `a` and adding `c`, `b` was the one evicted.
- Five questions: `miss`, `exact`, `semantic(1.00)`, `semantic(0.90)`, `miss`; the real pipeline ran only twice (`runs=2`). Exact hit rate is 0.2 because 4 of 5 lookups missed the exact layer (two of them were rescued by the semantic layer).
- After the version change the same question is a `miss` again (runs 3): invalidation by scope worked.
- The danger demo: the "not free" question hit with similarity 0.93, the same as the harmless variant. A threshold alone cannot tell them apart.
- Prompt caching: 1 call is a loss (-25%), the saving rises with the number of reuses within the TTL.

## Experiments (change ONE thing)
Each snippet runs from this step folder (`cd` into it first). Outputs below are from real runs.

### Semantic threshold sweep on paraphrases and non-paraphrases
```python
from ragkit.cache import SemanticCache
from ragkit.ingest import HashingEmbedder
from ragkit.testing import FakeClock
tests = [("How many days do I have to get my refund?", True), ("What is the refund window in days?", True), ("Is shipping free on my order?", False),
         ("How many days until my parcel arrives?", False), ("Is shipping not free on my order", False)]
for th in (0.95, 0.85, 0.75, 0.6):
    sc = SemanticCache(HashingEmbedder(128), th, clock=FakeClock()); sc.put("How many days do I have to get a refund?", "A")
    res = [(sc.lookup(q) is not None, want) for q, want in tests]
    print(f"threshold {th}: hits={[int(h) for h, _ in res]} wanted={[int(w) for _, w in res]}  wrong={sum(h != w for h, w in res)}")
```
Observed:
```text
threshold 0.95: hits=[0, 0, 0, 0, 0] wanted=[1, 1, 0, 0, 0]  wrong=2
threshold 0.85: hits=[1, 0, 0, 0, 0] wanted=[1, 1, 0, 0, 0]  wrong=1
threshold 0.75: hits=[1, 0, 0, 0, 0] wanted=[1, 1, 0, 0, 0]  wrong=1
threshold 0.6: hits=[1, 0, 0, 0, 0] wanted=[1, 1, 0, 0, 0]  wrong=1
```
With the offline hashing embedder (lexical, not semantic) the real paraphrase 'What is the refund window in days?' never hits (0 at every threshold) while the unrelated 'How many days until my parcel arrives?' stays a miss. At 0.95 even the near-identical question misses. So: threshold choice depends on the embedder; calibrate it on labelled paraphrase/non-paraphrase pairs from your own traffic, and expect a real sentence embedder to do better and also to fail on negation.

### Key without sampling params
```python
from ragkit.cache import cache_key
a = cache_key("s", "q", {"temperature": 0.0}); b = cache_key("s", "q", {"temperature": 1.0}); c = cache_key("s", "q")
print("temp 0 vs temp 1 differ:", a != b, "| params ignored would give identical keys:", c == cache_key("s", "q", None))
```
Observed:
```text
temp 0 vs temp 1 differ: True | params ignored would give identical keys: True
```
Two requests that differ only in temperature get different keys (`True`); a key built without params would not. Always hash sampling parameters.

### TTL: lazy expiry means len() can lie
```python
from ragkit.cache import TTLLRUCache
from ragkit.testing import FakeClock
c = FakeClock(); cache = TTLLRUCache(10, 60, c); cache.put("a", 1); c.advance(100)
print("len before any get:", len(cache), "| get:", cache.get("a") is not None, "| len after:", len(cache))
```
Observed:
```text
len before any get: 1 | get: True | len after: 0
```
`len(cache)` was 1 although the entry expired 40 s ago; it dropped to 0 only after the `get`. Lazy expiry means size metrics can over-count; a periodic sweep or a size gauge on `get` calls fixes dashboards.

## Debugging practice
**BREAK IT:**
```python
import sys; sys.path.insert(0, '..')

from ragkit.cache import *
from ragkit.ingest import HashingEmbedder
sc = SemanticCache(HashingEmbedder(8))
print("empty semantic cache is truthy?", bool(sc))
class Svc:
    def answer(self, q): raise RuntimeError("x")
print("`if self.semantic:` would skip an EMPTY cache forever ->", "skipped" if not sc else "used")
raise SystemExit("an empty container is falsy: use `is not None`")
```
Real output:
```text
an empty container is falsy: use `is not None`
```
No traceback: this bug is *silent*, which is why it is instructive. `SemanticCache` defines `__len__`, so an empty cache is falsy; `if self.semantic:` would skip it on the first request, never `put` anything, and the cache would stay empty forever, with hit rate 0 and no error. We hit exactly this while writing the module (the test `test_cached_answerer_layers_and_invalidation` failed with `cache == 'miss'`). Rule: test optional collaborators with `is not None`.

## Rebuild checklist
- [ ] Write `cache_key` and list what must be in it.
- [ ] Write `TTLLRUCache` with lazy expiry and LRU eviction using `OrderedDict`.
- [ ] Write a semantic lookup with scope isolation and a threshold; explain the negation failure.
- [ ] Explain how versioned keys invalidate without scanning.
- [ ] Explain provider prompt caching: prefix order, TTL, write/read multipliers, how to order a prompt for hits.
- [ ] Compute the saving for a 10k-token prefix over 10 calls.

## The new problem (cliffhanger)
We have wrapped and cached `Generator.generate(system, user) -> str`, but real providers return more than a string: token usage, finish reasons, tool calls, a stream of pieces. How do you design one gateway type system over many providers, stream results, validate structured output, and account for cost? [Step 08](../08_llm_gateway_and_streaming/WALKTHROUGH.md).

> **Say it aloud (interview box):** "I cache in layers: exact match keyed by a hash of the normalised prompt, sampling parameters and a scope containing the index version, so re-indexing invalidates by construction; TTL plus LRU for staleness and size; a semantic cache only with a high threshold and the same scope, because similarity is not equivalence; and provider-side prompt caching with stable content first. I never cache refusals and I always measure hit rate."
