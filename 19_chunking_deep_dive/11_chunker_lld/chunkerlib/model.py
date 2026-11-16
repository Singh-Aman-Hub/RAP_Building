"""model.py -- the data model: Chunk (what is stored), ChunkConfig (how it was made), TokenCounter (injected)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Callable, Mapping, Optional

TokenCounter = Callable[[str], int]          # text -> number of tokens. INJECTED: it must be the embedding model's tokenizer.


def sha(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


@dataclass(frozen=True)
class ChunkConfig:
    """Every knob that changes the output. Frozen + hashable, so config_hash is a stable fingerprint."""
    max_tokens: int = 200
    overlap_tokens: int = 0
    min_tokens: int = 0
    separators: tuple = ("\n\n", "\n", ". ", " ", "")      # recursive splitter hierarchy, biggest boundary first
    breadcrumb: bool = True                                # markdown strategy: prepend "Doc > Section" to embed_text

    def __post_init__(self):
        if not (0 <= self.overlap_tokens < self.max_tokens):
            raise ValueError(f"overlap_tokens ({self.overlap_tokens}) must be >= 0 and < max_tokens ({self.max_tokens})")
        if self.min_tokens < 0 or self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens must be between 0 and max_tokens")

    @property
    def config_hash(self) -> str:
        return sha(json.dumps(asdict(self), sort_keys=True))


@dataclass(frozen=True)
class Chunk:
    chunk_id: str                      # stable: hash(doc_id, chunker id/version/config, text) -- NOT the position
    doc_id: str
    doc_version: str                   # hash of the whole source text this chunk was cut from
    parent_id: Optional[str]           # section / parent node id (None for top-level); links small-to-big retrieval
    text: str                          # the exact slice of the source (what offsets point at)
    embed_text: str                    # what the embedding model and BM25 see (text + breadcrumb / context)
    display_text: str                  # what the user / LLM prompt shows (may differ: e.g. table as markdown)
    token_count: int                   # tokens of embed_text under the injected counter
    position: int                      # 0-based order inside the document
    char_start: int                    # offsets into the source: highlight and cite
    char_end: int
    section_path: tuple = ()           # ("Handbook", "2. Time off", "2.3 Parental leave")
    page: Optional[int] = None         # parsed documents only
    bbox: Optional[tuple] = None       # (x0, y0, x1, y1) in page coordinates
    metadata: Mapping = field(default_factory=dict)
    chunker_id: str = ""               # registry name, e.g. "recursive"
    chunker_version: str = ""          # bump on ANY behaviour change -> new index version

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)


def make_chunk_id(doc_id: str, chunker_id: str, chunker_version: str, config_hash: str, text: str, occurrence: int) -> str:
    """Same text in the same doc under the same chunker => same id; `occurrence` separates repeated identical texts."""
    return sha(f"{doc_id}|{chunker_id}|{chunker_version}|{config_hash}|{sha(text)}|{occurrence}", 20)
