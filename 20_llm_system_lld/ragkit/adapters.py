"""adapters.py -- Adapter pattern: make code we do NOT own fit the ragkit Protocols (steps 04, 08).

  Module 14's Embedder.encode()                     -> ragkit Embedder.embed()
  Module 14's CrossEncoderReranker.score()          -> ragkit Reranker.rerank()
  shared/llm_client.generate(messages, system, ...) -> ragkit Generator.generate(system, user)
Heavy imports are lazy: importing ragkit never loads torch.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from .domain import SearchHit

REPO = Path(__file__).resolve().parents[2]


def _use_module14():
    for p in (REPO / "14_rag_and_retrieval", REPO / "shared"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


class Module14Embedder:
    name = "all-MiniLM-L6-v2"

    def __init__(self):
        _use_module14()
        from common import Embedder as _Emb               # module 14's class
        self._inner = _Emb()
        self.dim = self._inner.dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self._inner.encode(list(texts))            # already (n, dim) unit rows


class Module14CrossEncoder:
    def __init__(self):
        _use_module14()
        from common import CrossEncoderReranker
        self._inner = CrossEncoderReranker()

    def rerank(self, query, hits, top_n):
        if not hits:
            return []
        logits = self._inner.score(query, [h.chunk.text for h in hits])    # (n,) raw logits, ~ -11 .. +9
        order = np.argsort(-logits)[:top_n]
        return [SearchHit(hits[i].chunk, float(logits[i]), "rerank", r) for r, i in enumerate(order, 1)]


class LLMClientGenerator:
    """Adapt shared/llm_client.generate (messages-based) to Generator.generate(system, user)."""

    def __init__(self, backend: str | None = None, mock=None, max_tokens: int = 300, temperature: float = 0.0):
        _use_module14()
        import llm_client
        self._llm, self.backend, self.mock = llm_client, backend, mock
        self.max_tokens, self.temperature = max_tokens, temperature

    def generate(self, system: str, user: str) -> str:
        return self._llm.generate([{"role": "user", "content": user}], system=system, max_tokens=self.max_tokens,
                                  temperature=self.temperature, backend=self.backend, mock=self.mock)
