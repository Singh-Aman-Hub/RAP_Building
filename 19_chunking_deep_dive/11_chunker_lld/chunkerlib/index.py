"""index.py -- idempotent re-chunking, versioned indexes and an embedding cache keyed by content."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import Chunk, sha


@dataclass
class EmbeddingCache:
    """(model_id, sha(embed_text)) -> vector. A chunker upgrade that leaves a text unchanged re-uses its vector."""
    model_id: str
    embed_fn: object                                   # callable: list[str] -> (n, d) array
    store: dict = field(default_factory=dict)
    calls: int = 0                                     # texts actually sent to the model (the cost you pay)

    def get_many(self, texts):
        miss = [t for t in dict.fromkeys(texts) if (self.model_id, sha(t)) not in self.store]
        if miss:
            for t, v in zip(miss, self.embed_fn(miss)):
                self.store[(self.model_id, sha(t))] = v
            self.calls += len(miss)
        return np.vstack([self.store[(self.model_id, sha(t))] for t in texts])


@dataclass
class Diff:
    kept: list
    added: list
    removed: list


def diff_chunks(old: list, new: list) -> Diff:
    """Idempotent re-chunking: ids are content hashes, so an edited paragraph changes only ITS chunks."""
    o, n = {c.chunk_id: c for c in old}, {c.chunk_id: c for c in new}
    return Diff([n[i] for i in n if i in o], [n[i] for i in n if i not in o], [o[i] for i in o if i not in n])


class IndexManager:
    """Named, immutable index versions + an alias. Never mutate the live index: build a new version, then switch."""

    def __init__(self, collection: str, cache: EmbeddingCache):
        self.collection, self.cache = collection, cache
        self.indexes: dict[str, dict] = {}             # index_name -> {chunk_id: (Chunk, vector)}
        self.alias: str | None = None                  # what queries read

    def index_name(self, chunker_id, chunker_version, config_hash):
        return f"{self.collection}__{chunker_id}@{chunker_version}__{config_hash[:6]}__{self.cache.model_id}"

    def build(self, name: str, chunks: list[Chunk]):
        vecs = self.cache.get_many([c.embed_text for c in chunks])
        self.indexes[name] = {c.chunk_id: (c, v) for c, v in zip(chunks, vecs)}

    def upsert_doc(self, name: str, doc_id: str, new_chunks: list[Chunk]) -> Diff:
        """Replace one document's chunks inside an index: embed only the added chunks, delete only the removed ones."""
        idx = self.indexes[name]
        old = [c for c, _ in idx.values() if c.doc_id == doc_id]
        d = diff_chunks(old, new_chunks)
        for c in d.removed:
            del idx[c.chunk_id]
        if d.added:
            vecs = self.cache.get_many([c.embed_text for c in d.added])
            for c, v in zip(d.added, vecs):
                idx[c.chunk_id] = (c, v)
        for c in d.kept:                               # positions may shift even when text is unchanged: refresh the metadata
            idx[c.chunk_id] = (c, idx[c.chunk_id][1])
        return d

    def switch(self, name: str):
        assert name in self.indexes, f"build {name} before switching the alias"
        self.alias = name

    def search(self, qvec, k=3):
        items = list(self.indexes[self.alias].values())
        scores = np.array([v @ qvec for _, v in items])
        return [(items[i][0], float(scores[i])) for i in np.argsort(-scores)[:k]]
