"""decorators.py -- Decorator pattern: wrap an object, implement the SAME Protocol, add one behaviour (step 05)."""
from __future__ import annotations

import functools
import time
from typing import Callable, Sequence

import numpy as np

from .ids import content_hash


def timed(emit: Callable[[str, dict], None], label: str):
    """Function decorator: report how long each call took."""
    def deco(fn):
        @functools.wraps(fn)                                   # keep fn.__name__ / docstring
        def wrapper(*a, **kw):
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:                                           # also times the calls that raise
                emit("timing", {"label": label, "seconds": time.perf_counter() - t0})
        return wrapper
    return deco


class TimedGenerator:
    def __init__(self, inner, emit: Callable[[str, dict], None]):
        self.inner, self.emit = inner, emit

    def generate(self, system: str, user: str) -> str:
        t0 = time.perf_counter()
        try:
            return self.inner.generate(system, user)
        finally:
            self.emit("generate_timing", {"seconds": time.perf_counter() - t0})


class CachingEmbedder:
    """Memoise embeddings per text. Same name/dim as the inner embedder, so it is a drop-in replacement."""

    def __init__(self, inner):
        self.inner = inner
        self.name, self.dim = inner.name, inner.dim
        self._cache: dict[str, np.ndarray] = {}
        self.hits = self.misses = 0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        keys = [content_hash(t, length=32) for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in self._cache]
        if missing:
            vecs = self.inner.embed([t for _, t in missing])    # ONE batched call for all misses
            for (k, _), v in zip(missing, vecs):
                self._cache[k] = v
        self.misses += len(missing)
        self.hits += len(texts) - len(missing)
        return np.stack([self._cache[k] for k in keys])
