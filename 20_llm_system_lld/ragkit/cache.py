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


def normalize(text: str) -> str:
    """Collapse whitespace + casefold. (Casefold is safe for questions; do NOT do it for case-sensitive code prompts.)"""
    return " ".join(text.split()).casefold()


def cache_key(system: str, user: str, params: dict | None = None, scope: str = "") -> str:
    """Hash of everything that can change the answer: normalized prompt, sampling params, and the SCOPE
    (index version, model, tenant). Different scope -> different key -> automatic invalidation."""
    payload = json.dumps({"s": normalize(system), "u": normalize(user), "p": params or {}, "scope": scope},
                         sort_keys=True)                      # sort_keys: {"a":1,"b":2} and {"b":2,"a":1} hash equal
    return content_hash(payload, length=32)


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


class TTLLRUCache:
    """Bounded cache: entries expire after `ttl` seconds, and the least recently USED goes first when full."""

    def __init__(self, capacity: int = 128, ttl: float | None = 300.0, clock=None):
        self.capacity, self.ttl, self.clock = capacity, ttl, clock or SystemClock
        self._data: OrderedDict[str, tuple[Any, float | None]] = OrderedDict()   # order = recency (last = newest)
        self.stats = CacheStats()

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

    def put(self, key: str, value: Any) -> None:
        self._data[key] = (value, None if self.ttl is None else self.clock.now() + self.ttl)
        self._data.move_to_end(key)
        while len(self._data) > self.capacity:
            self._data.popitem(last=False)                     # drop the least recently used
            self.stats.evictions += 1

    def __len__(self):
        return len(self._data)


class SemanticCache:
    """Hit when a NEW question is close enough (cosine >= threshold) to an OLD one in the same scope."""

    def __init__(self, embedder, threshold: float = 0.85, capacity: int = 128, ttl: float | None = 300.0, clock=None):
        self.embedder, self.threshold, self.capacity, self.ttl = embedder, threshold, capacity, ttl
        self.clock = clock or SystemClock
        self._entries: OrderedDict[int, tuple[str, np.ndarray, Any, float | None]] = OrderedDict()
        self._next_id = 0
        self.stats = CacheStats()

    def lookup(self, query: str, scope: str = ""):
        """Return (value, similarity) or None."""
        now, qv = self.clock.now(), self.embedder.embed([normalize(query)])[0]
        for eid in [e for e, (_, _, _, exp) in self._entries.items() if exp is not None and now >= exp]:
            del self._entries[eid]
            self.stats.expirations += 1
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


class CachedAnswerer:
    """Decorator for the whole RAG service: exact cache first, then semantic cache, then the real pipeline.
    `scope()` is called on EVERY request (usually returns the index version), so re-indexing invalidates old entries."""

    def __init__(self, inner, exact: TTLLRUCache, semantic: SemanticCache | None = None,
                 scope: Callable[[], str] = lambda: ""):
        self.inner, self.exact, self.semantic, self.scope = inner, exact, semantic, scope

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


def prompt_cache_cost(prefix_tokens: int, suffix_tokens: int, calls: int, price_per_mtok: float = 3.0,
                      write_mult: float = 1.25, read_mult: float = 0.1) -> dict:
    """Provider-side prompt caching arithmetic (the provider keeps the processed KV state of a repeated PREFIX).
    First call writes the cache (write_mult x), later calls read it (read_mult x); the suffix is always full price.
    Multipliers default to Anthropic's published 5-minute-cache numbers (check the current pricing page)."""
    p = price_per_mtok / 1e6
    without = calls * (prefix_tokens + suffix_tokens) * p
    with_cache = (prefix_tokens * write_mult + (calls - 1) * prefix_tokens * read_mult + calls * suffix_tokens) * p
    return {"without_cache_usd": without, "with_cache_usd": with_cache, "saving_pct": 100 * (1 - with_cache / without)}
