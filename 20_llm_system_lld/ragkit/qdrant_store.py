"""qdrant_store.py -- Adapter: Qdrant's client API -> ragkit's VectorStore Protocol (step 05).

Uses QdrantClient(":memory:") (local mode, no server, no network). Pointing the same class at
QdrantClient(url="http://...") would use a real server: only the constructor argument changes.
"""
from __future__ import annotations

import uuid
from typing import Any, Mapping, Sequence

import numpy as np
from qdrant_client import QdrantClient, models

from .domain import Chunk, Embedding, SearchHit

NS = uuid.UUID("12345678-1234-5678-1234-567812345678")


def point_id(chunk_id: str) -> str:
    """Qdrant ids must be uint or UUID, so map our hex chunk_id to a deterministic UUID5 (same chunk -> same point)."""
    return str(uuid.uuid5(NS, chunk_id))


class QdrantStore:
    def __init__(self, dim: int, client: QdrantClient | None = None, collection: str = "chunks"):
        self.client = client or QdrantClient(":memory:")
        self.collection, self.revision = collection, 0
        self.client.create_collection(collection, vectors_config=models.VectorParams(
            size=dim, distance=models.Distance.COSINE))

    @staticmethod
    def _cond(where: Mapping[str, Any] | None):
        if not where:
            return None
        must = [models.FieldCondition(key=k if k in ("doc_id", "section") else f"metadata.{k}",
                                      match=models.MatchValue(value=v)) for k, v in where.items()]
        return models.Filter(must=must)

    def upsert(self, items: Sequence[tuple[Chunk, Embedding]]) -> None:
        pts = [models.PointStruct(id=point_id(c.chunk_id), vector=list(e.vector), payload=c.to_dict())
               for c, e in items]
        if pts:
            self.client.upsert(self.collection, points=pts)
        self.revision += 1

    def delete_chunks(self, chunk_ids: Sequence[str]) -> None:
        if chunk_ids:
            self.client.delete(self.collection, points_selector=models.PointIdsList(
                points=[point_id(c) for c in chunk_ids]))
        self.revision += 1

    def _scroll(self, flt=None) -> list[Chunk]:
        out, offset = [], None
        while True:                                            # scroll = paginate through all matching points
            pts, offset = self.client.scroll(self.collection, scroll_filter=flt, limit=256, offset=offset,
                                             with_payload=True, with_vectors=False)
            out += [Chunk.from_dict(p.payload) for p in pts]
            if offset is None:
                return out

    def chunk_ids_for_doc(self, doc_id: str) -> set[str]:
        return {c.chunk_id for c in self._scroll(self._cond({"doc_id": doc_id}))}

    def delete_doc(self, doc_id: str) -> int:
        ids = list(self.chunk_ids_for_doc(doc_id))
        self.delete_chunks(ids)
        return len(ids)

    def all_chunks(self) -> list[Chunk]:
        return self._scroll()

    def search(self, vector: np.ndarray, k: int, where=None) -> list[SearchHit]:
        res = self.client.query_points(self.collection, query=[float(x) for x in vector], limit=k,
                                       query_filter=self._cond(where), with_payload=True).points
        return [SearchHit(Chunk.from_dict(p.payload), float(p.score), "dense", r) for r, p in enumerate(res, 1)]

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def clear(self) -> None:
        self.client.delete(self.collection, points_selector=models.FilterSelector(filter=models.Filter()))
        self.revision += 1
