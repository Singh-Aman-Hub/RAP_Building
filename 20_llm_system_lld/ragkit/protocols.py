"""protocols.py -- the contracts between components (steps 03-04).

typing.Protocol = STRUCTURAL typing: any class with these methods fits, no inheritance needed.
That is what lets a test fake, a Qdrant adapter and a real model all plug into the same pipeline.
"""
from __future__ import annotations

from typing import Any, Iterator, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .domain import Answer, Chunk, Document, Element, Embedding, SearchHit


@runtime_checkable
class Loader(Protocol):
    def load(self) -> Iterator[Document]: ...


@runtime_checkable
class Parser(Protocol):
    def parse(self, doc: Document) -> list[Element]: ...


@runtime_checkable
class Chunker(Protocol):
    name: str                                         # part of the index version (changing chunking = new index)
    def chunk(self, doc: Document, elements: Sequence[Element]) -> list[Chunk]: ...


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...       # (n, dim), L2-normalised rows


@runtime_checkable
class VectorStore(Protocol):
    revision: int                                      # bumps on every write (lets caches/BM25 notice changes)
    def upsert(self, items: Sequence[tuple[Chunk, Embedding]]) -> None: ...
    def delete_chunks(self, chunk_ids: Sequence[str]) -> None: ...
    def delete_doc(self, doc_id: str) -> int: ...
    def chunk_ids_for_doc(self, doc_id: str) -> set[str]: ...
    def all_chunks(self) -> list[Chunk]: ...
    def search(self, vector: np.ndarray, k: int, where: Mapping[str, Any] | None = None) -> list[SearchHit]: ...
    def count(self) -> int: ...
    def clear(self) -> None: ...


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 10) -> list[SearchHit]: ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, hits: Sequence[SearchHit], top_n: int) -> list[SearchHit]: ...


@runtime_checkable
class Generator(Protocol):
    def generate(self, system: str, user: str) -> str: ...


@runtime_checkable
class Answerer(Protocol):
    def answer(self, question: str) -> Answer: ...


@runtime_checkable
class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...
