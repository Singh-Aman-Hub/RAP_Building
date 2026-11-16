"""ingest.py -- the offline half: Loader -> Parser -> Chunker -> Embedder -> VectorStore (step 03)."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .domain import Chunk, Document, Element, ElementKind, Embedding, SearchHit
from .ids import content_hash
from .protocols import Chunker, Embedder, Loader, Parser, VectorStore


# ------------------------------------------------------------------ loaders and parser
class DirectoryLoader:
    """Yield one Document per matching file, in sorted order (deterministic)."""

    def __init__(self, directory: str | Path, pattern: str = "*.md"):
        self.directory, self.pattern = Path(directory), pattern

    def load(self) -> Iterator[Document]:
        for path in sorted(self.directory.glob(self.pattern)):
            yield Document.from_text(path.name, path.read_text(encoding="utf-8"))


class InMemoryLoader:
    def __init__(self, docs: Sequence[Document]):
        self.docs = list(docs)

    def load(self) -> Iterator[Document]:
        yield from self.docs


class MarkdownParser:
    """Split markdown into HEADING / LIST / PARAGRAPH elements (blank-line separated blocks)."""

    def parse(self, doc: Document) -> list[Element]:
        out = []
        for block in re.split(r"\n\s*\n", doc.text.strip()):
            block = block.strip()
            if not block:
                continue
            m = re.match(r"(#{1,6})\s+(.*)", block)
            if m and "\n" not in block:
                out.append(Element(ElementKind.HEADING, m.group(2).strip(), len(m.group(1)), len(out)))
            elif m:                                                # heading line followed directly by text
                head, _, rest = block.partition("\n")
                hm = re.match(r"(#{1,6})\s+(.*)", head)
                out.append(Element(ElementKind.HEADING, hm.group(2).strip(), len(hm.group(1)), len(out)))
                out.append(Element(ElementKind.PARAGRAPH, " ".join(rest.split()), 0, len(out)))
            elif re.match(r"[-*]\s", block):
                out.append(Element(ElementKind.LIST, " ".join(block.split()), 0, len(out)))
            else:
                out.append(Element(ElementKind.PARAGRAPH, " ".join(block.split()), 0, len(out)))
        return out


# ------------------------------------------------------------------ embedders
class HashingEmbedder:
    """Deterministic, offline, instant 'embedding': signed hashing of lower-cased words, L2-normalised.
    Not semantic (no synonyms) but lexically meaningful -- perfect for tests and offline demos."""

    def __init__(self, dim: int = 128):
        self.dim, self.name = dim, f"hashing-{dim}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", text.lower()):
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)   # md5, not hash(): Python's hash() is salted per run
                out[i, h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)


# ------------------------------------------------------------------ vector store (in memory)
def _matches(chunk: Chunk, where: Mapping[str, Any] | None) -> bool:
    if not where:
        return True
    for k, v in where.items():
        have = getattr(chunk, k) if k in ("doc_id", "section") else chunk.metadata.get(k)
        if have != v:
            return False
    return True


class InMemoryVectorStore:
    """Exact (brute-force) cosine search over a dict. Fine up to ~100k chunks."""

    def __init__(self):
        self._items: dict[str, tuple[Chunk, np.ndarray]] = {}
        self.revision = 0

    def upsert(self, items):
        for chunk, emb in items:
            self._items[chunk.chunk_id] = (chunk, emb.as_array())   # same id twice -> overwritten, not duplicated
        self.revision += 1

    def delete_chunks(self, chunk_ids):
        for cid in chunk_ids:
            self._items.pop(cid, None)
        self.revision += 1

    def delete_doc(self, doc_id):
        ids = self.chunk_ids_for_doc(doc_id)
        self.delete_chunks(list(ids))
        return len(ids)

    def chunk_ids_for_doc(self, doc_id):
        return {cid for cid, (c, _) in self._items.items() if c.doc_id == doc_id}

    def all_chunks(self):
        return [c for c, _ in self._items.values()]

    def search(self, vector, k, where=None):
        cands = [(c, v) for c, v in self._items.values() if _matches(c, where)]    # pre-filter, then rank
        if not cands:
            return []
        scores = np.stack([v for _, v in cands]) @ np.asarray(vector, dtype=np.float32)
        order = np.argsort(-scores)[:k]
        return [SearchHit(cands[i][0], float(scores[i]), "dense", r) for r, i in enumerate(order, 1)]

    def count(self):
        return len(self._items)

    def clear(self):
        self._items.clear()
        self.revision += 1


# ------------------------------------------------------------------ bookkeeping value objects
@dataclass(frozen=True, slots=True)
class DeadLetter:
    doc_id: str
    source: str
    stage: str
    error: str


@dataclass
class IngestReport:
    index_version: str
    docs_seen: int = 0
    docs_added: int = 0
    docs_updated: int = 0
    docs_skipped: int = 0
    docs_deleted: int = 0
    chunks_embedded: int = 0
    chunks_reused: int = 0
    chunks_deleted: int = 0
    rebuilt: bool = False
    dead_letters: list[DeadLetter] = field(default_factory=list)

    def summary(self) -> str:
        return (f"v={self.index_version} seen={self.docs_seen} added={self.docs_added} updated={self.docs_updated} "
                f"skipped={self.docs_skipped} deleted={self.docs_deleted} | chunks embedded={self.chunks_embedded} "
                f"reused={self.chunks_reused} deleted={self.chunks_deleted} | dead={len(self.dead_letters)}"
                f"{' | REBUILT' if self.rebuilt else ''}")


class Manifest:
    """What the index currently contains: {doc_id: content_hash} plus the index_version that built it."""

    def __init__(self, index_version: str = "", docs: dict[str, str] | None = None):
        self.index_version, self.docs = index_version, dict(docs or {})

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"index_version": self.index_version, "docs": self.docs}, indent=1))

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        if not path.exists():
            return cls()
        d = json.loads(path.read_text())
        return cls(d["index_version"], d["docs"])


# ------------------------------------------------------------------ the pipeline
class StageError(Exception):
    """Wraps any failure with the stage it happened in ('parse', 'chunk', 'embed', 'store'), for the dead-letter record."""

    def __init__(self, stage: str, cause: Exception):
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.stage = stage


class IngestionPipeline:
    def __init__(self, loader: Loader, parser: Parser, chunker: Chunker, embedder: Embedder,
                 store: VectorStore, manifest: Manifest | None = None, batch_size: int = 16):
        self.loader, self.parser, self.chunker, self.embedder, self.store = loader, parser, chunker, embedder, store
        self.manifest = manifest or Manifest()
        self.batch_size = batch_size
        self.dead_letters: list[DeadLetter] = []

    @property
    def index_version(self) -> str:
        """Fingerprint of everything that changes what a vector MEANS: embedder + dims + chunker settings."""
        return content_hash(self.embedder.name, str(self.embedder.dim), self.chunker.name, "schema1", length=8)

    def run(self) -> IngestReport:
        rep = IngestReport(self.index_version)
        if self.manifest.index_version and self.manifest.index_version != rep.index_version:
            self.store.clear()                                     # old vectors are meaningless under the new config
            self.manifest.docs.clear()                             # (mutate in place: other holders of the manifest see it)
            rep.rebuilt = True
        seen: set[str] = set()
        for doc in self.loader.load():
            rep.docs_seen += 1
            seen.add(doc.doc_id)
            old = self.manifest.docs.get(doc.doc_id)
            if old == doc.content_hash:
                rep.docs_skipped += 1                              # unchanged: do no work at all
                continue
            try:
                self._index_doc(doc, rep)
            except StageError as exc:                              # one bad document must not stop the batch
                self.dead_letters.append(DeadLetter(doc.doc_id, doc.source, exc.stage, str(exc)))
                rep.dead_letters.append(self.dead_letters[-1])
                continue
            self.manifest.docs[doc.doc_id] = doc.content_hash
            if old is None:
                rep.docs_added += 1
            else:
                rep.docs_updated += 1
        for gone in set(self.manifest.docs) - seen:                # document disappeared from the source
            rep.chunks_deleted += self.store.delete_doc(gone)
            del self.manifest.docs[gone]
            rep.docs_deleted += 1
        self.manifest.index_version = rep.index_version
        return rep

    def _index_doc(self, doc: Document, rep: IngestReport) -> None:
        stage = "parse"
        try:
            elements = self.parser.parse(doc)
            stage = "chunk"
            chunks = self.chunker.chunk(doc, elements)
            wanted = {c.chunk_id for c in chunks}
            existing = self.store.chunk_ids_for_doc(doc.doc_id)
            stale = existing - wanted
            if stale:
                self.store.delete_chunks(list(stale))
                rep.chunks_deleted += len(stale)
            new = [c for c in chunks if c.chunk_id not in existing]    # only chunks the store has never seen
            rep.chunks_reused += len(chunks) - len(new)
            for i in range(0, len(new), self.batch_size):              # batch: one model call per batch, not per chunk
                batch = new[i:i + self.batch_size]
                stage = "embed"
                vecs = self.embedder.embed([c.text for c in batch])
                stage = "store"
                self.store.upsert([(c, Embedding.from_array(c.chunk_id, self.embedder.name, v))
                                   for c, v in zip(batch, vecs)])
                rep.chunks_embedded += len(batch)
        except Exception as exc:
            raise StageError(stage, exc) from exc
