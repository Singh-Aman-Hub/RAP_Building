"""domain.py -- the nouns of a RAG system as immutable value objects (step 02).

Rules used everywhere in ragkit:
  * frozen=True  -> nobody can change an object after creation (safe to share, cache, hash)
  * slots=True   -> no per-instance __dict__: less memory, typos in attribute names raise
  * validate in __post_init__ -> an invalid object can never exist
  * to_dict / from_dict -> JSON-friendly boundary (storage, API, logs)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .ids import content_hash


def _freeze(meta: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(meta or {}))        # read-only view of a private copy


class ElementKind(str, Enum):
    """What a parsed piece of a document is. str-mixin so it JSON-serialises as its value."""
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    CODE = "code"


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str                                      # stable identity of the SOURCE (hash of its path)
    source: str                                      # where it came from (path / URL)
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def __post_init__(self):
        if not self.text.strip():
            raise ValueError(f"Document {self.source!r} is empty")
        object.__setattr__(self, "metadata", _freeze(self.metadata))   # frozen -> must bypass __setattr__

    @classmethod
    def from_text(cls, source: str, text: str, **metadata) -> "Document":
        return cls(doc_id=content_hash(source), source=source, text=text, metadata=metadata)

    @property
    def content_hash(self) -> str:
        """Changes when the text changes; doc_id does not. This pair drives incremental re-indexing."""
        return content_hash(self.text)

    def to_dict(self) -> dict:
        return {"doc_id": self.doc_id, "source": self.source, "text": self.text, "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Document":
        return cls(d["doc_id"], d["source"], d["text"], d.get("metadata", {}))


@dataclass(frozen=True, slots=True)
class Element:
    """One structural piece of a parsed document (a heading, a paragraph, ...)."""
    kind: ElementKind
    text: str
    level: int = 0                                   # heading level (1 for '#', 2 for '##'); 0 otherwise
    position: int = 0                                # order inside the document

    def to_dict(self) -> dict:
        return {"kind": self.kind.value, "text": self.text, "level": self.level, "position": self.position}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Element":
        return cls(ElementKind(d["kind"]), d["text"], d.get("level", 0), d.get("position", 0))


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    section: str = ""
    position: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def __post_init__(self):
        if not self.text.strip():
            raise ValueError("Chunk text must not be empty")
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    @classmethod
    def create(cls, doc_id: str, text: str, section: str = "", position: int = 0, **metadata) -> "Chunk":
        """chunk_id = hash(doc_id, section, text): NOT the position, so inserting a paragraph elsewhere
        does not change the ids of the chunks around it."""
        return cls(content_hash(doc_id, section, text), doc_id, text, section, position, metadata)

    def to_dict(self) -> dict:
        return {"chunk_id": self.chunk_id, "doc_id": self.doc_id, "text": self.text, "section": self.section,
                "position": self.position, "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Chunk":
        return cls(d["chunk_id"], d["doc_id"], d["text"], d.get("section", ""), d.get("position", 0),
                   d.get("metadata", {}))


@dataclass(frozen=True, slots=True)
class Embedding:
    chunk_id: str
    model: str                                       # which model made it: vectors from different models are NOT comparable
    vector: tuple[float, ...]

    def __post_init__(self):
        if not self.vector:
            raise ValueError("empty vector")

    @classmethod
    def from_array(cls, chunk_id: str, model: str, arr: np.ndarray) -> "Embedding":
        return cls(chunk_id, model, tuple(float(x) for x in arr))

    @property
    def dim(self) -> int:
        return len(self.vector)

    def as_array(self) -> np.ndarray:
        return np.asarray(self.vector, dtype=np.float32)

    def to_dict(self) -> dict:
        return {"chunk_id": self.chunk_id, "model": self.model, "vector": list(self.vector)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Embedding":
        return cls(d["chunk_id"], d["model"], tuple(d["vector"]))


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk: Chunk
    score: float
    source: str = "dense"                            # which retriever / stage produced this score
    rank: int = 0                                    # 1-based position in the result list

    def to_dict(self) -> dict:
        return {"chunk": self.chunk.to_dict(), "score": self.score, "source": self.source, "rank": self.rank}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SearchHit":
        return cls(Chunk.from_dict(d["chunk"]), d["score"], d.get("source", "dense"), d.get("rank", 0))


@dataclass(frozen=True, slots=True)
class Citation:
    marker: int                                      # the [n] the model wrote
    chunk_id: str                                    # the chunk that [n] points to
    supported: bool = True                           # did the verifier find lexical support?

    def to_dict(self) -> dict:
        return {"marker": self.marker, "chunk_id": self.chunk_id, "supported": self.supported}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Citation":
        return cls(d["marker"], d["chunk_id"], d.get("supported", True))


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    citations: tuple[Citation, ...] = ()
    refused: bool = False
    refusal_reason: str | None = None
    context_chunk_ids: tuple[str, ...] = ()
    index_version: str = ""
    meta: Mapping[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "meta", _freeze(self.meta))
        if self.refused and self.citations:
            raise ValueError("a refusal cannot carry citations")

    def to_dict(self) -> dict:
        return {"text": self.text, "citations": [c.to_dict() for c in self.citations], "refused": self.refused,
                "refusal_reason": self.refusal_reason, "context_chunk_ids": list(self.context_chunk_ids),
                "index_version": self.index_version, "meta": dict(self.meta)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Answer":
        return cls(d["text"], tuple(Citation.from_dict(c) for c in d.get("citations", [])),
                   d.get("refused", False), d.get("refusal_reason"), tuple(d.get("context_chunk_ids", ())),
                   d.get("index_version", ""), d.get("meta", {}))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "Answer":
        return cls.from_dict(json.loads(s))
