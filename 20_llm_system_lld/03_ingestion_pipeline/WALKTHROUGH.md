# Step 03 — The ingestion pipeline: Loader -> Parser -> Chunker -> Embedder -> VectorStore

## The story so far
[Step 02](../02_domain_model/WALKTHROUGH.md) defined the nouns and gave chunks a content-hash id. Module [14 steps 01-04](../../14_rag_and_retrieval/README.md) taught each indexing stage as a function in a script.

## The pain point
A script that loads, chunks, embeds and stores works once. Then the real problems arrive: a 3-hour re-index because one document changed; duplicates because the job ran twice; one corrupt PDF kills the whole batch at document 4,000; a new embedding model silently mixes with old vectors; nobody can tell which index version answered a question.

## The spark
Five small interfaces, one orchestrator, and four properties: **idempotent** (run twice = run once), **incremental** (only changed documents and only new chunks are embedded), **fault-isolated** (a bad document goes to a dead-letter list), **versioned** (a fingerprint of embedder + chunker; a change forces a rebuild).

## Goal
Read the five Protocols, three chunkers (Template Method), the in-memory store and `IngestionPipeline`; run five scenarios (cold, no-op, one edit, one deletion, crash, version change) and explain the numbers.

## The theory (read this before the interview)
### The offline pipeline
```
Loader --Document--> Parser --Element[]--> Chunker --Chunk[]--> (diff vs store) --new Chunk[]--> Embedder --vectors--> VectorStore
                                                                    |  stale ids -> delete                      ^
                                                                    +-- unchanged ids -> skip (no embedding) ---+
 failures at any stage -> DeadLetter(doc, stage, error); manifest not updated -> retried next run
```
| Stage | Interface | Responsibility | Swap examples |
|---|---|---|---|
| Loader | `load() -> Iterator[Document]` | where documents come from | directory, S3, Confluence API, web crawler |
| Parser | `parse(doc) -> list[Element]` | structure (headings, paragraphs, tables) | markdown, HTML, PDF layout parser (module 14 step 01) |
| Chunker | `chunk(doc, elements) -> list[Chunk]` | split into retrieval units; has a `name` | heading, fixed words, sentences, semantic |
| Embedder | `embed(texts) -> (n, dim)` | text to vectors; has `name`, `dim` | MiniLM, e5, a hosted API |
| VectorStore | `upsert, delete_*, search, ...` | store and search | memory, Qdrant, pgvector |

### Idempotent upsert via content hashes
`chunk_id = hash(doc_id, section, text)`. For each document: compute its chunk ids, ask the store which ids it already has for that `doc_id`, then
- `wanted - existing` = **new** -> embed and upsert;
- `existing - wanted` = **stale** -> delete;
- `wanted & existing` = **reused** -> do nothing.
`upsert` with an existing id overwrites, never duplicates. Running the job twice therefore gives the same store.

### Incremental re-index: the manifest
A `Manifest` records `{doc_id: content_hash}` of what the index contains. Per run: same hash -> **skip the document entirely** (no parsing, no chunking); changed hash -> diff chunks as above; a `doc_id` in the manifest but not in the source -> `delete_doc`. The manifest is updated **only after** a document succeeds, so a crash mid-document leaves it "dirty" and the next run retries it. Operations are re-runnable, which is easier than making them transactional.

### Batch embedding
Embedding models are efficient in batches (one forward pass for 16-64 texts) and hosted APIs charge latency per call. `batch_size` groups chunks per call. Too large: memory and timeouts; too small: per-call overhead (36 calls vs 9 in Experiments). Batch **across** documents in production for fuller batches; here per document for clarity.

### Dead-letter queue (DLQ)
A document that raises goes to a `DeadLetter(doc_id, source, stage, error)` list instead of aborting the run. Stage names (`parse`, `chunk`, `embed`, `store`) tell you where to look. Operations: alert on DLQ size, fix the cause, re-run (the manifest still lacks the doc). **Poison-pill** documents (always fail) need a retry limit so they do not block forever.

### Index versioning
`index_version = hash(embedder.name, embedder.dim, chunker.name, "schema1")[:8]`. Anything that changes what a vector *means* must change it. If the manifest's version differs, the store is cleared and rebuilt. In production you build the new index **beside** the old (a new collection), switch an alias when it is complete (blue/green), and keep the old one for rollback. The version is stamped on every `Answer` so you can tell which index produced it, and it keys caches (step 07).

### Template Method (preview of step 05)
`BaseChunker.chunk()` is the fixed algorithm (create chunks, dedupe by id, assign positions); subclasses implement only the hook `_groups()` that says *how to group text*. `HeadingChunker`, `FixedWordChunker`, `SentenceChunker` differ in ~15 lines.

### Failure semantics worth stating in an interview
- *At-least-once, idempotent*: an operation may run twice but the result is the same.
- *Partial failure inside one document*: stale chunks may be deleted before embedding fails; the manifest stays dirty so the next run repairs it. Strict atomicity would need a transaction or a staging collection.
- *Ordering*: process documents independently; no cross-document ordering assumptions.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Ingestion pipeline | indexing pipeline, ETL for RAG, data prep, offline pipeline | Mostly | Same job: raw files -> searchable vectors; "offline" contrasts it with the per-question "online" path. |
| Idempotent | re-runnable, safe to retry, exactly-once effect | Close | Idempotent = same result however many times you run it; exactly-once delivery is a stronger network guarantee. |
| Upsert | insert-or-update, merge | Same | Write that creates a record or overwrites the one with the same id. |
| Dead-letter queue (DLQ) | error queue, quarantine, poison-message queue | Mostly | Where items that failed processing are parked for inspection and replay. |
| Manifest | state file, checkpoint, index ledger, watermark | Close | Record of what has been indexed; a watermark is a timestamp-style manifest. |
| Index version | embedding version, collection version, schema version | Close | Fingerprint of everything that determines what vectors mean. |
| Full re-index vs incremental | rebuild vs delta indexing | No | Full: clear and redo everything (needed when the index version changes); incremental: only what changed. |

## Run it
```bash
# from this step folder
HF_HUB_OFFLINE=1 python main.py
# or from the repo root
HF_HUB_OFFLINE=1 .venv/bin/python 20_llm_system_lld/03_ingestion_pipeline/main.py
```


## Code, step by step
Four files: the demo `main.py`, then the contracts `ragkit/protocols.py`, the chunkers `ragkit/chunking.py` and the pipeline `ragkit/ingest.py`. Every block is listed.

#### Walking `main.py`

### 1. Header docstring  (`main.py` lines 1–13)
```python
"""
GOAL:  Loader -> Parser -> Chunker -> Embedder -> VectorStore as five small interfaces, then the properties that make
       an indexer production-grade: idempotent upsert, incremental re-index, batching, dead-letter, index versioning.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 20_llm_system_lld/03_ingestion_pipeline/main.py
EXPECTED OUTPUT (shortened):
    run 1 (cold)      : added=9 ... chunks embedded=36
    run 2 (no change) : skipped=9 ... embedded=0
    run 3 (1 doc edit): updated=1 skipped=8 | embedded=1 reused=3 deleted=1
    dead letter: ... run 5 (new embedder dim): REBUILT
NEEDS: ragkit + module 14's data/kb markdown files. HashingEmbedder is offline and instant (no model download).
"""
```
- **In simple words:** What this file shows.
- **Exactly:** GOAL / RUN / EXPECTED OUTPUT / NEEDS.
- **Why it is here:** Orientation.

### 2. Imports  (`main.py` lines 14–22)
```python
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragkit.chunking import HeadingChunker
from ragkit.ids import content_hash
from ragkit.ingest import DirectoryLoader, HashingEmbedder, InMemoryVectorStore, IngestionPipeline, Manifest, MarkdownParser
```
- **In simple words:** Tools: file copying, a temp folder, our pipeline parts.
- **Exactly:** `HashingEmbedder` is an offline, instant embedder (step 03 does not need a model); `content_hash` is used to check the manifest.
- **Why it is here:** Keeps the demo runnable in a second.

### 3. Copy the knowledge base to a scratch folder  (`main.py` lines 24–27)
```python
KB = Path(__file__).resolve().parents[2] / "14_rag_and_retrieval" / "data" / "kb"
work = Path(tempfile.mkdtemp())                                     # a scratch copy of the KB we are allowed to edit
for f in KB.glob("*.md"):
    shutil.copy(f, work / f.name)
```
- **In simple words:** We copy the 9 markdown files so we can edit and delete them without touching module 14's data.
- **Exactly:** `tempfile.mkdtemp()` returns a new empty directory; `shutil.copy` copies each `*.md` (9 files).
- **Why it is here:** The demo edits and deletes documents; the original data must stay intact.

### 4. An embedder that counts its own calls  (`main.py` lines 30–39)
```python
class CountingEmbedder(HashingEmbedder):
    """Decorates HashingEmbedder to count how many model calls (batches) and texts we pay for."""
    def __init__(self, dim=128):
        super().__init__(dim)
        self.calls = self.texts = 0

    def embed(self, texts):
        self.calls += 1
        self.texts += len(texts)
        return super().embed(texts)
```
- **In simple words:** A wrapper that keeps a tally of how many times the embedding service was called and with how many texts.
- **Exactly:** Subclass of `HashingEmbedder`; `embed()` increments `calls` and adds `len(texts)` to `texts`, then delegates to `super().embed`.
- **Why it is here:** Embedding calls are the money/time cost; the demo proves the later runs make fewer.

### 5. Shared store and manifest, and a pipeline factory  (`main.py` lines 42–46)
```python
store, manifest, emb = InMemoryVectorStore(), Manifest(), CountingEmbedder(128)


def pipeline(embedder=emb, batch_size=16):
    return IngestionPipeline(DirectoryLoader(work), MarkdownParser(), HeadingChunker(), embedder, store, manifest, batch_size)
```
- **In simple words:** One notebook of what is indexed (manifest), one shelf (store), and a function that builds a fresh pipeline around them.
- **Exactly:** `pipeline(embedder, batch_size)` builds `IngestionPipeline(DirectoryLoader(work), MarkdownParser(), HeadingChunker(), embedder, store, manifest, batch_size)`. The store and manifest persist across pipeline objects (like a database across job runs).
- **Why it is here:** A new pipeline object per run mimics a nightly cron job: state lives in the store and manifest, not in the object.

### 6. Run 1 (cold) and run 2 (nothing changed)  (`main.py` lines 49–52)
```python
p = pipeline()
print("index_version:", p.index_version, "| chunker:", HeadingChunker().name)
print("run 1 (cold)      :", p.run().summary(), f"| embed calls={emb.calls} (batch_size=16)")
print("run 2 (no change) :", pipeline().run().summary(), f"| embed calls still {emb.calls}")
```
- **In simple words:** First run indexes everything; the second finds nothing to do.
- **Exactly:** Run 1: `seen=9 added=9`, `chunks embedded=36 reused=0`, `embed calls=9` (one batch per document, each has fewer than 16 chunks). Run 2: `skipped=9`, `embedded=0`, calls still 9. `index_version: b69545a4`.
- **Why it is here:** Idempotency in numbers: the second run costs nothing.

### 7. Run 3: edit one document  (`main.py` lines 54–57)
```python
# ---- edit ONE document: change the refund window from 9 to 14 days ------------------------------
f = work / "refunds_and_returns.md"
f.write_text(f.read_text().replace("within 9 days", "within 14 days"))
print("run 3 (1 doc edit):", pipeline().run().summary())
```
- **In simple words:** Change the refund window from 9 to 14 days and re-run.
- **Exactly:** `updated=1 skipped=8`; `chunks embedded=1 reused=3 deleted=1`: the refunds file has 4 chunks, only the changed section got a new id, the old one was deleted, three were reused.
- **Why it is here:** Incremental indexing: 1 embedding call instead of 36.

### 8. Run 4: delete a source file  (`main.py` lines 59–61)
```python
# ---- delete a document from the source --------------------------------------------------------------------------
(work / "wholesale_and_b2b.md").unlink()
print("run 4 (doc removed):", pipeline().run().summary(), f"| store now holds {store.count()} chunks")
```
- **In simple words:** Remove a file; the index must forget it.
- **Exactly:** `deleted=1` document, `chunks deleted=4`, store now holds 32 chunks (36 - 4). `wholesale_and_b2b.md` was in the manifest but not in the loader output.
- **Why it is here:** Without delete handling, answers cite documents that no longer exist.

### 9. A picky embedder that rejects some input  (`main.py` lines 64–69)
```python
# ---- a document that crashes the embedder goes to the dead-letter list; the rest continue -------------------------
class PickyEmbedder(CountingEmbedder):
    def embed(self, texts):
        if any("gift" in t.lower() for t in texts):
            raise RuntimeError("embedding service rejected input")
        return super().embed(texts)
```
- **In simple words:** Simulates an embedding service that refuses certain text.
- **Exactly:** Raises `RuntimeError` if any text contains 'gift'.
- **Why it is here:** Gives us a deterministic failure to send to the dead-letter list.

### 10. Run with a failing document  (`main.py` lines 72–77)
```python
(work / "gifts.md").write_text("# Gifts\n\n## Gift wrap\nWe can gift wrap any order for 50 rupees.\n")
dead = pipeline(PickyEmbedder(128))
rep = dead.run()
print("dead letter:", rep.summary())
print("   ->", [(d.source, d.stage, d.error) for d in rep.dead_letters])
print("   gifts.md is NOT in the manifest (so the next run retries it):", content_hash("gifts.md") not in manifest.docs)
```
- **In simple words:** One bad file does not stop the other eight.
- **Exactly:** `added=0 skipped=8 dead=1`; dead letter `('gifts.md', 'embed', 'RuntimeError: embedding service rejected input')`. `content_hash('gifts.md') not in manifest.docs` is `True`, so the next run retries it.
- **Why it is here:** Fault isolation plus a stage name that tells you where to look.

### 11. Run 5: new embedder dimension forces a rebuild  (`main.py` lines 79–82)
```python
# ---- changing what a vector MEANS (embedder dim) invalidates the whole index ---------------------------------------
new = pipeline(CountingEmbedder(256))
print("run 5 (new embedder dim=256): version", new.index_version, "->", new.run().summary())
print("sample stored chunk:", store.all_chunks()[0].to_dict()["chunk_id"], "|", store.all_chunks()[0].section)
```
- **In simple words:** Switching to a 256-dimension embedder changes what every vector means.
- **Exactly:** Version `b69545a4` -> `bd3dbd43`; `REBUILT`, `added=9`, `embedded=33` (32 old docs' chunks + the now-working gifts.md chunk). A sample stored chunk id `d375c871a6f892f8`, section `Nuts`.
- **Why it is here:** Mixing 128-dim and 256-dim (or two models') vectors in one index gives errors or silent nonsense.

### 12. Exercises  (`main.py` lines 84–86)
```python
# TRY IT: HeadingChunker(max_words=20): how many more chunks? Is run 2 still a no-op afterwards? (the version changes)
# TRY IT: pipeline(batch_size=1) on a cold index: how many embed calls now?
# BREAK IT: make Chunk.create hash the position too (domain.py) and redo run 3: how many chunks get re-embedded?
```
- **In simple words:** Things to try.
- **Exactly:** Comment lines.
- **Why it is here:** See exercises.md.


#### Walking `ragkit/protocols.py`

### 13. protocols.py header and imports  (`ragkit/protocols.py` lines 1–12)
```python
"""protocols.py -- the contracts between components (steps 03-04).

typing.Protocol = STRUCTURAL typing: any class with these methods fits, no inheritance needed.
That is what lets a test fake, a Qdrant adapter and a real model all plug into the same pipeline.
"""
from __future__ import annotations

from typing import Any, Iterator, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .domain import Answer, Chunk, Document, Element, Embedding, SearchHit
```
- **In simple words:** The list of contracts between components.
- **Exactly:** Docstring explains structural typing; imports `Protocol`, `runtime_checkable`, domain types and numpy.
- **Why it is here:** One file shows every seam of the system.

### 14. Loader and Parser  (`ragkit/protocols.py` lines 15–22)
```python
@runtime_checkable
class Loader(Protocol):
    def load(self) -> Iterator[Document]: ...


@runtime_checkable
class Parser(Protocol):
    def parse(self, doc: Document) -> list[Element]: ...
```
- **In simple words:** Where documents come from, and how they are split into elements.
- **Exactly:** `Loader.load() -> Iterator[Document]` (a generator, so a million documents never sit in memory); `Parser.parse(doc) -> list[Element]`. Both `@runtime_checkable`.
- **Why it is here:** Smallest possible contracts: one method each (interface segregation).

### 15. Chunker: a named strategy  (`ragkit/protocols.py` lines 25–28)
```python
@runtime_checkable
class Chunker(Protocol):
    name: str                                         # part of the index version (changing chunking = new index)
    def chunk(self, doc: Document, elements: Sequence[Element]) -> list[Chunk]: ...
```
- **In simple words:** Turns a document's elements into chunks.
- **Exactly:** `name: str` attribute (part of the index version) and `chunk(doc, elements) -> list[Chunk]`.
- **Why it is here:** If chunk settings changed but the name did not, an old index would be treated as current.

### 16. Embedder: text to vectors  (`ragkit/protocols.py` lines 31–35)
```python
@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...       # (n, dim), L2-normalised rows
```
- **In simple words:** Turns many texts into a matrix of numbers.
- **Exactly:** `name`, `dim` and `embed(texts) -> np.ndarray` of shape `(n, dim)`, rows L2-normalised so dot product = cosine.
- **Why it is here:** Batch-shaped input and output is what makes embedding efficient.

### 17. VectorStore: storage and search  (`ragkit/protocols.py` lines 38–48)
```python
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
```
- **In simple words:** Everything a store must do, nothing more.
- **Exactly:** `revision` counter (bumps on every write) plus `upsert`, `delete_chunks`, `delete_doc`, `chunk_ids_for_doc`, `all_chunks`, `search(vector, k, where)`, `count`, `clear`.
- **Why it is here:** Memory and Qdrant implement the same 9 methods, so the pipeline never knows which one it uses (Liskov).

### 18. Retriever and Reranker (used in step 04)  (`ragkit/protocols.py` lines 51–58)
```python
@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 10) -> list[SearchHit]: ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, hits: Sequence[SearchHit], top_n: int) -> list[SearchHit]: ...
```
- **In simple words:** Find candidates; reorder them precisely.
- **Exactly:** `retrieve(query, k) -> list[SearchHit]`; `rerank(query, hits, top_n) -> list[SearchHit]`.
- **Why it is here:** Dense, sparse, hybrid and cross-encoder variants fit these two shapes.

### 19. Generator, Answerer, TokenCounter  (`ragkit/protocols.py` lines 61–73)
```python
@runtime_checkable
class Generator(Protocol):
    def generate(self, system: str, user: str) -> str: ...


@runtime_checkable
class Answerer(Protocol):
    def answer(self, question: str) -> Answer: ...


@runtime_checkable
class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...
```
- **In simple words:** Produce text; answer a question; count tokens.
- **Exactly:** `generate(system, user) -> str`; `answer(question) -> Answer`; `count(text) -> int`.
- **Why it is here:** `Answerer` is what caches and tracers wrap (step 07, 09).


#### Walking `ragkit/chunking.py`

### 20. chunking.py header and imports  (`ragkit/chunking.py` lines 1–8)
```python
"""chunking.py -- Template Method: BaseChunker.chunk() fixes the skeleton, subclasses fill one hook (steps 03, 05, 10)."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Sequence

from .domain import Chunk, Document, Element, ElementKind
```
- **In simple words:** Chunkers share a skeleton.
- **Exactly:** Docstring names the pattern: Template Method. Imports `ABC`, `abstractmethod`, `re`.
- **Why it is here:** Say the pattern name in the interview; it earns points.

### 21. BaseChunker: the fixed part  (`ragkit/chunking.py` lines 11–13)
```python
class BaseChunker(ABC):
    """Template Method pattern. `chunk` is the fixed algorithm; `_groups` is the hook that varies."""
    name = "base"
```
- **In simple words:** The base class knows how to make chunks; subclasses only decide how to group text.
- **Exactly:** `name = 'base'`; `chunk` is the template method.
- **Why it is here:** Shared behaviour (dedupe, ids, positions) is written once.

### 22. The template method chunk()  (`ragkit/chunking.py` lines 15–25)
```python
    def chunk(self, doc: Document, elements: Sequence[Element]) -> list[Chunk]:
        chunks, seen = [], set()
        for section, text in self._groups(elements):              # the hook: (section title, text) pairs
            if not text.strip():
                continue
            c = Chunk.create(doc.doc_id, text.strip(), section, position=len(chunks), source=doc.source)
            if c.chunk_id in seen:                                # identical text in the same section -> keep one
                continue
            seen.add(c.chunk_id)
            chunks.append(c)
        return chunks
```
- **In simple words:** For each (section, text) group, make a chunk, skip blanks and duplicates.
- **Exactly:** `_groups(elements)` yields `(section, text)` pairs; `Chunk.create(doc_id, text.strip(), section, position=len(chunks), source=doc.source)`; a `seen` set drops chunks whose `chunk_id` repeats within the document.
- **Why it is here:** Without dedupe, two identical paragraphs would collide on the same id and be inserted twice.

### 23. The abstract hook _groups  (`ragkit/chunking.py` lines 27–29)
```python
    @abstractmethod
    def _groups(self, elements: Sequence[Element]) -> list[tuple[str, str]]:
        ...
```
- **In simple words:** The one thing every subclass must supply.
- **Exactly:** `@abstractmethod` returning `list[tuple[str, str]]`.
- **Why it is here:** Forgetting it fails at instantiation (`TypeError`), not mid-run.

### 24. HeadingChunker: constructor and name  (`ragkit/chunking.py` lines 32–37)
```python
class HeadingChunker(BaseChunker):
    """One chunk per section (heading + its paragraphs); a section longer than max_words is split by paragraph."""

    def __init__(self, max_words: int = 120):
        self.max_words = max_words
        self.name = f"heading(max_words={max_words})"
```
- **In simple words:** One chunk per section, but cap long sections.
- **Exactly:** `max_words` default 120; `name = f'heading(max_words={max_words})'` so the index version changes when the cap changes.
- **Why it is here:** Heading-aware chunks keep a fact with its section title (module 14 step 02: best strategy on this KB).

### 25. HeadingChunker: group paragraphs under their heading  (`ragkit/chunking.py` lines 39–45)
```python
    def _groups(self, elements):
        groups, section, buf = [], "", []

        def flush():
            if buf:
                groups.append((section, " ".join(buf)))
                buf.clear()
```
- **In simple words:** A small helper that saves the paragraphs collected so far as one group.
- **Exactly:** `flush()` appends `(section, ' '.join(buf))` and clears `buf`.
- **Why it is here:** Used both at a new heading and when the word budget is reached.

### 26. HeadingChunker: the loop  (`ragkit/chunking.py` lines 47–56)
```python
        for el in elements:
            if el.kind is ElementKind.HEADING:
                flush()
                section = el.text
            else:
                if buf and sum(len(b.split()) for b in buf) + len(el.text.split()) > self.max_words:
                    flush()                                        # budget reached: start a new chunk, same section
                buf.append(el.text)
        flush()
        return groups
```
- **In simple words:** Walk the elements; start a new group at each heading or when too long.
- **Exactly:** A HEADING element flushes and sets `section`. Otherwise, if adding this paragraph would exceed `max_words`, flush first, then append. Final `flush()` after the loop.
- **Why it is here:** Never splits inside a paragraph, so sentences stay whole.

### 27. FixedWordChunker: sliding window settings  (`ragkit/chunking.py` lines 59–66)
```python
class FixedWordChunker(BaseChunker):
    """Sliding window of `size` words, advancing by size - overlap."""

    def __init__(self, size: int = 50, overlap: int = 10):
        if not 0 <= overlap < size:
            raise ValueError("need 0 <= overlap < size")
        self.size, self.overlap = size, overlap
        self.name = f"fixed(size={size},overlap={overlap})"
```
- **In simple words:** Chunks of N words that overlap by M words.
- **Exactly:** `size=50, overlap=10` default; `ValueError('need 0 <= overlap < size')` otherwise (else the window never advances).
- **Why it is here:** Overlap equal to size would loop forever; fail at construction.

### 28. FixedWordChunker: windows  (`ragkit/chunking.py` lines 68–72)
```python
    def _groups(self, elements):
        words = " ".join(e.text for e in elements if e.kind is not ElementKind.HEADING).split()
        step = self.size - self.overlap
        return [("", " ".join(words[i:i + self.size])) for i in range(0, max(len(words), 1), step)
                if words[i:i + self.size]]
```
- **In simple words:** Cut the text into overlapping windows.
- **Exactly:** Words of all non-heading elements; step `size - overlap`; `range(0, max(len(words),1), step)`; window `words[i:i+size]`; section is `''` because windows ignore structure.
- **Why it is here:** Baseline strategy: simple, structure-blind (and unstable under edits - see Experiments).

### 29. SentenceChunker: sentence packing settings  (`ragkit/chunking.py` lines 75–80)
```python
class SentenceChunker(BaseChunker):
    """Pack whole sentences until max_words would be exceeded."""

    def __init__(self, max_words: int = 40):
        self.max_words = max_words
        self.name = f"sentence(max_words={max_words})"
```
- **In simple words:** Pack whole sentences up to a word budget.
- **Exactly:** `max_words=40` default; name includes it.
- **Why it is here:** Never cuts mid-sentence.

### 30. SentenceChunker: split and pack  (`ragkit/chunking.py` lines 82–94)
```python
    def _groups(self, elements):
        text = " ".join(e.text for e in elements if e.kind is not ElementKind.HEADING)
        out, buf, n = [], [], 0
        for s in re.split(r"(?<=[.!?])\s+", text):
            w = len(s.split())
            if buf and n + w > self.max_words:
                out.append(("", " ".join(buf)))
                buf, n = [], 0
            buf.append(s)
            n += w
        if buf:
            out.append(("", " ".join(buf)))
        return out
```
- **In simple words:** Split on sentence ends and fill chunks.
- **Exactly:** `re.split(r'(?<=[.!?])\s+', text)` (a lookbehind keeps the punctuation); start a new chunk when `n + w > max_words`.
- **Why it is here:** A naive splitter would fail on abbreviations like 'e.g.'; use a real sentence tokenizer in production.


#### Walking `ragkit/ingest.py`

### 31. ingest.py header and imports  (`ragkit/ingest.py` lines 1–18)
```python
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
```
- **In simple words:** The offline half in one file.
- **Exactly:** Imports numpy, domain types, `content_hash` and the Protocol types used as annotations.
- **Why it is here:** Type hints reference Protocols, never concrete classes.

### 32. DirectoryLoader: files to Documents  (`ragkit/ingest.py` lines 19–27)
```python
class DirectoryLoader:
    """Yield one Document per matching file, in sorted order (deterministic)."""

    def __init__(self, directory: str | Path, pattern: str = "*.md"):
        self.directory, self.pattern = Path(directory), pattern

    def load(self) -> Iterator[Document]:
        for path in sorted(self.directory.glob(self.pattern)):
            yield Document.from_text(path.name, path.read_text(encoding="utf-8"))
```
- **In simple words:** Read every matching file in sorted order.
- **Exactly:** `Path.glob(pattern)` sorted; `Document.from_text(path.name, text)` - the *file name* is the source, so `doc_id` is stable across runs. A generator (`yield`).
- **Why it is here:** `sorted` makes runs deterministic; a generator keeps memory flat.

### 33. InMemoryLoader: documents from a list  (`ragkit/ingest.py` lines 30–35)
```python
class InMemoryLoader:
    def __init__(self, docs: Sequence[Document]):
        self.docs = list(docs)

    def load(self) -> Iterator[Document]:
        yield from self.docs
```
- **In simple words:** A loader for tests.
- **Exactly:** Stores a copy of the list and `yield from`.
- **Why it is here:** Tests need no files on disk.

### 34. MarkdownParser: split into blocks  (`ragkit/ingest.py` lines 38–46)
```python
class MarkdownParser:
    """Split markdown into HEADING / LIST / PARAGRAPH elements (blank-line separated blocks)."""

    def parse(self, doc: Document) -> list[Element]:
        out = []
        for block in re.split(r"\n\s*\n", doc.text.strip()):
            block = block.strip()
            if not block:
                continue
```
- **In simple words:** Cut the text at blank lines.
- **Exactly:** `re.split(r'\n\s*\n', text)` -> blocks; blank blocks skipped.
- **Why it is here:** Blank-line separated blocks are markdown's paragraphs.

### 35. MarkdownParser: headings  (`ragkit/ingest.py` lines 47–54)
```python
            m = re.match(r"(#{1,6})\s+(.*)", block)
            if m and "\n" not in block:
                out.append(Element(ElementKind.HEADING, m.group(2).strip(), len(m.group(1)), len(out)))
            elif m:                                                # heading line followed directly by text
                head, _, rest = block.partition("\n")
                hm = re.match(r"(#{1,6})\s+(.*)", head)
                out.append(Element(ElementKind.HEADING, hm.group(2).strip(), len(hm.group(1)), len(out)))
                out.append(Element(ElementKind.PARAGRAPH, " ".join(rest.split()), 0, len(out)))
```
- **In simple words:** Recognise `# Title` and `## Section`.
- **Exactly:** Regex `(#{1,6})\s+(.*)`: group 1 length = heading level. A heading directly followed by text (no blank line) is split into a HEADING and a PARAGRAPH element.
- **Why it is here:** Section names become chunk `section` metadata.

### 36. MarkdownParser: lists and paragraphs  (`ragkit/ingest.py` lines 55–62)
```python
            elif re.match(r"[-*]\s", block):
                out.append(Element(ElementKind.LIST, " ".join(block.split()), 0, len(out)))
            else:
                out.append(Element(ElementKind.PARAGRAPH, " ".join(block.split()), 0, len(out)))
        return out


# ------------------------------------------------------------------ embedders
```
- **In simple words:** Bullet blocks become LIST, the rest PARAGRAPH.
- **Exactly:** `' '.join(block.split())` collapses newlines and extra spaces.
- **Why it is here:** Chunk text is clean single-line prose.

### 37. HashingEmbedder: a deterministic fake embedder  (`ragkit/ingest.py` lines 63–68)
```python
class HashingEmbedder:
    """Deterministic, offline, instant 'embedding': signed hashing of lower-cased words, L2-normalised.
    Not semantic (no synonyms) but lexically meaningful -- perfect for tests and offline demos."""

    def __init__(self, dim: int = 128):
        self.dim, self.name = dim, f"hashing-{dim}"
```
- **In simple words:** Counts words into 128 buckets; same words, same vector.
- **Exactly:** `dim` default 128; `name = 'hashing-128'`.
- **Why it is here:** Offline, instant, repeatable: right for tests and for explaining the pipeline; not semantic.

### 38. HashingEmbedder.embed  (`ragkit/ingest.py` lines 70–80)
```python
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", text.lower()):
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)   # md5, not hash(): Python's hash() is salted per run
                out[i, h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)


# ------------------------------------------------------------------ vector store (in memory)
```
- **In simple words:** Make a vector for each text.
- **Exactly:** For each word: `md5` hash `h`; bucket `h % dim`; add +1 or -1 depending on bit 64 (signed hashing reduces collisions' bias). Rows divided by their L2 norm (0-norm guarded) -> shape `(n, dim)`. md5 not `hash()` because Python's `hash()` is salted per process.
- **Why it is here:** Reproducible vectors across runs; otherwise tests would be flaky.

### 39. _matches: metadata filter  (`ragkit/ingest.py` lines 81–88)
```python
def _matches(chunk: Chunk, where: Mapping[str, Any] | None) -> bool:
    if not where:
        return True
    for k, v in where.items():
        have = getattr(chunk, k) if k in ("doc_id", "section") else chunk.metadata.get(k)
        if have != v:
            return False
    return True
```
- **In simple words:** Does a chunk satisfy the `where` filter?
- **Exactly:** `doc_id` and `section` are fields; any other key is looked up in `chunk.metadata`; all conditions must be equal.
- **Why it is here:** Pre-filtering before ranking (module 14 step 04: post-filtering can return fewer than k).

### 40. InMemoryVectorStore: init and upsert  (`ragkit/ingest.py` lines 91–101)
```python
class InMemoryVectorStore:
    """Exact (brute-force) cosine search over a dict. Fine up to ~100k chunks."""

    def __init__(self):
        self._items: dict[str, tuple[Chunk, np.ndarray]] = {}
        self.revision = 0

    def upsert(self, items):
        for chunk, emb in items:
            self._items[chunk.chunk_id] = (chunk, emb.as_array())   # same id twice -> overwritten, not duplicated
        self.revision += 1
```
- **In simple words:** A dictionary of chunk id -> (chunk, vector).
- **Exactly:** `upsert` stores `emb.as_array()` under `chunk.chunk_id` (an existing id is overwritten) and bumps `revision`.
- **Why it is here:** `revision` lets BM25 and caches notice the store changed.

### 41. Deleting and listing  (`ragkit/ingest.py` lines 103–117)
```python
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
```
- **In simple words:** Remove chunks, remove a whole document, list ids.
- **Exactly:** `delete_doc` returns how many chunks it removed (used in the report); `chunk_ids_for_doc` is the set the diff uses.
- **Why it is here:** The pipeline needs exactly these three to implement incremental indexing.

### 42. search, count, clear  (`ragkit/ingest.py` lines 119–135)
```python
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
```
- **In simple words:** Exact cosine search with an optional filter.
- **Exactly:** Filter first, stack vectors to `(m, dim)`, `scores = matrix @ vector` -> `(m,)`, `argsort(-scores)[:k]`; results are `SearchHit(chunk, score, 'dense', rank)`.
- **Why it is here:** Brute force is exact and fast enough up to ~100k chunks (module 14 step 04).

### 43. DeadLetter: a failure record  (`ragkit/ingest.py` lines 136–141)
```python
@dataclass(frozen=True, slots=True)
class DeadLetter:
    doc_id: str
    source: str
    stage: str
    error: str
```
- **In simple words:** Which document failed, at which stage, with what error.
- **Exactly:** Frozen dataclass: `doc_id, source, stage, error`.
- **Why it is here:** Immutable so a record cannot be altered after the fact.

### 44. IngestReport: counters for one run  (`ragkit/ingest.py` lines 144–156)
```python
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
```
- **In simple words:** A receipt of what happened.
- **Exactly:** Counts for docs (seen/added/updated/skipped/deleted) and chunks (embedded/reused/deleted), `rebuilt` flag, `dead_letters` list.
- **Why it is here:** Reports are what you alert on and what the demo prints.

### 45. IngestReport.summary  (`ragkit/ingest.py` lines 158–162)
```python
    def summary(self) -> str:
        return (f"v={self.index_version} seen={self.docs_seen} added={self.docs_added} updated={self.docs_updated} "
                f"skipped={self.docs_skipped} deleted={self.docs_deleted} | chunks embedded={self.chunks_embedded} "
                f"reused={self.chunks_reused} deleted={self.chunks_deleted} | dead={len(self.dead_letters)}"
                f"{' | REBUILT' if self.rebuilt else ''}")
```
- **In simple words:** One line version of the receipt.
- **Exactly:** Formats all counters into the line seen in the execution trace.
- **Why it is here:** Human-readable logs.

### 46. Manifest: what the index contains  (`ragkit/ingest.py` lines 165–182)
```python
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
```
- **In simple words:** A ledger: document id -> content hash, plus the index version.
- **Exactly:** `save`/`load` use JSON; `load` of a missing file returns an empty manifest.
- **Why it is here:** Persist it next to the index; losing it only costs a full re-index.

### 47. StageError: failure plus its stage  (`ragkit/ingest.py` lines 183–188)
```python
class StageError(Exception):
    """Wraps any failure with the stage it happened in ('parse', 'chunk', 'embed', 'store'), for the dead-letter record."""

    def __init__(self, stage: str, cause: Exception):
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.stage = stage
```
- **In simple words:** Wrap any exception and remember where it happened.
- **Exactly:** `stage` attribute; the original is chained with `raise ... from exc` (`__cause__`).
- **Why it is here:** The dead letter shows `embed`, not just a stack trace.

### 48. IngestionPipeline: constructor  (`ragkit/ingest.py` lines 191–197)
```python
class IngestionPipeline:
    def __init__(self, loader: Loader, parser: Parser, chunker: Chunker, embedder: Embedder,
                 store: VectorStore, manifest: Manifest | None = None, batch_size: int = 16):
        self.loader, self.parser, self.chunker, self.embedder, self.store = loader, parser, chunker, embedder, store
        self.manifest = manifest or Manifest()
        self.batch_size = batch_size
        self.dead_letters: list[DeadLetter] = []
```
- **In simple words:** Receives all five collaborators.
- **Exactly:** Takes `loader, parser, chunker, embedder, store, manifest, batch_size` - dependency injection; defaults to a fresh `Manifest()`.
- **Why it is here:** Any combination of implementations works; tests inject fakes.

### 49. index_version  (`ragkit/ingest.py` lines 199–202)
```python
    @property
    def index_version(self) -> str:
        """Fingerprint of everything that changes what a vector MEANS: embedder + dims + chunker settings."""
        return content_hash(self.embedder.name, str(self.embedder.dim), self.chunker.name, "schema1", length=8)
```
- **In simple words:** A short fingerprint of the configuration.
- **Exactly:** `content_hash(embedder.name, str(embedder.dim), chunker.name, 'schema1', length=8)`; a `@property`, recomputed on access. `b69545a4` for 128-dim hashing + heading(max_words=120).
- **Why it is here:** Bump the `'schema1'` string when you change anything else about stored data.

### 50. run(): decide whether to rebuild  (`ragkit/ingest.py` lines 204–209)
```python
    def run(self) -> IngestReport:
        rep = IngestReport(self.index_version)
        if self.manifest.index_version and self.manifest.index_version != rep.index_version:
            self.store.clear()                                     # old vectors are meaningless under the new config
            self.manifest.docs.clear()                             # (mutate in place: other holders of the manifest see it)
            rep.rebuilt = True
```
- **In simple words:** If the configuration changed since the last run, throw the old index away.
- **Exactly:** If `manifest.index_version` is set and differs: `store.clear()`, `manifest.docs.clear()`, `rep.rebuilt = True`. (The manifest is mutated in place so other holders see it.)
- **Why it is here:** Vectors under a different embedder are meaningless; keeping them poisons search.

### 51. run(): per-document skip check  (`ragkit/ingest.py` lines 210–217)
```python
        seen: set[str] = set()
        for doc in self.loader.load():
            rep.docs_seen += 1
            seen.add(doc.doc_id)
            old = self.manifest.docs.get(doc.doc_id)
            if old == doc.content_hash:
                rep.docs_skipped += 1                              # unchanged: do no work at all
                continue
```
- **In simple words:** Go through the loader's documents; skip the unchanged ones.
- **Exactly:** `seen` collects doc ids for deletion detection; `old = manifest.docs.get(doc_id)`; if `old == doc.content_hash` -> `docs_skipped += 1; continue`.
- **Why it is here:** The cheapest document is one you never touch.

### 52. run(): index with fault isolation  (`ragkit/ingest.py` lines 218–228)
```python
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
```
- **In simple words:** Index the document; park failures.
- **Exactly:** `self._index_doc(doc, rep)` inside `try`; on `StageError` append a `DeadLetter` to both lists and `continue`. On success write the new hash into the manifest and count added vs updated.
- **Why it is here:** Order matters: the manifest is updated only after success.

### 53. run(): remove vanished documents  (`ragkit/ingest.py` lines 229–234)
```python
        for gone in set(self.manifest.docs) - seen:                # document disappeared from the source
            rep.chunks_deleted += self.store.delete_doc(gone)
            del self.manifest.docs[gone]
            rep.docs_deleted += 1
        self.manifest.index_version = rep.index_version
        return rep
```
- **In simple words:** Anything in the manifest that the loader did not return was deleted at the source.
- **Exactly:** `set(manifest.docs) - seen`; `store.delete_doc` returns removed chunk count; delete from manifest; finally stamp `manifest.index_version`.
- **Why it is here:** Otherwise deleted documents stay searchable forever.

### 54. _index_doc: parse and chunk  (`ragkit/ingest.py` lines 236–241)
```python
    def _index_doc(self, doc: Document, rep: IngestReport) -> None:
        stage = "parse"
        try:
            elements = self.parser.parse(doc)
            stage = "chunk"
            chunks = self.chunker.chunk(doc, elements)
```
- **In simple words:** Turn the document into chunks, tracking the current stage name.
- **Exactly:** `stage` variable updated before each step; `parser.parse` then `chunker.chunk`.
- **Why it is here:** `stage` is how a failure is attributed.

### 55. _index_doc: diff against the store  (`ragkit/ingest.py` lines 242–247)
```python
            wanted = {c.chunk_id for c in chunks}
            existing = self.store.chunk_ids_for_doc(doc.doc_id)
            stale = existing - wanted
            if stale:
                self.store.delete_chunks(list(stale))
                rep.chunks_deleted += len(stale)
```
- **In simple words:** Which chunk ids are wanted, which exist, which are stale?
- **Exactly:** `wanted` from the chunker, `existing = store.chunk_ids_for_doc`, `stale = existing - wanted` deleted at once.
- **Why it is here:** Stale chunks would still be retrieved and cited.

### 56. _index_doc: only new chunks  (`ragkit/ingest.py` lines 248–249)
```python
            new = [c for c in chunks if c.chunk_id not in existing]    # only chunks the store has never seen
            rep.chunks_reused += len(chunks) - len(new)
```
- **In simple words:** Chunks the store already has are not embedded again.
- **Exactly:** `new` = chunks not in `existing`; `chunks_reused += len(chunks) - len(new)`.
- **Why it is here:** This line is where idempotency saves money.

### 57. _index_doc: batch embed and store  (`ragkit/ingest.py` lines 250–257)
```python
            for i in range(0, len(new), self.batch_size):              # batch: one model call per batch, not per chunk
                batch = new[i:i + self.batch_size]
                stage = "embed"
                vecs = self.embedder.embed([c.text for c in batch])
                stage = "store"
                self.store.upsert([(c, Embedding.from_array(c.chunk_id, self.embedder.name, v))
                                   for c, v in zip(batch, vecs)])
                rep.chunks_embedded += len(batch)
```
- **In simple words:** Embed in batches and write each batch.
- **Exactly:** Slice `new[i:i+batch_size]`; `embedder.embed([texts])` -> `(b, dim)`; wrap each row in `Embedding.from_array`; `store.upsert`.
- **Why it is here:** One model call per batch.

### 58. _index_doc: attach the stage  (`ragkit/ingest.py` lines 258–259)
```python
        except Exception as exc:
            raise StageError(stage, exc) from exc
```
- **In simple words:** Convert any error into a `StageError`.
- **Exactly:** `raise StageError(stage, exc) from exc`.
- **Why it is here:** Keeps the original traceback while adding the stage.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `yield` / `yield from` | Generator: produce items one at a time |
| `Iterator[Document]` | Type of something you can loop over once, lazily |
| `{a} - {b}` | Set difference: ids in a but not in b |
| `raise X from exc` | Chain exceptions; the original stays in `__cause__` |
| `@property` | Computed attribute (`pipeline.index_version`) |
| `range(0, n, step)` | Batch start indexes |
| `(?<=[.!?])` | Regex lookbehind: split after punctuation, keep it |

## Libraries & functions
| Name | What it does here |
|---|---|
| `pathlib.Path.glob` | List files matching a pattern |
| `tempfile.mkdtemp` | Create a scratch directory |
| `shutil.copy` | Copy a file |
| `re.split`, `re.match` | Block and heading detection |
| `hashlib.md5` | Stable word hashing in `HashingEmbedder` |
| `numpy` | Matrix of vectors, `@` for dot products, `argsort` for ranking |
| `json` | Manifest save/load |

## Execution trace (real run)
```text
index_version: b69545a4 | chunker: heading(max_words=120)
run 1 (cold)      : v=b69545a4 seen=9 added=9 updated=0 skipped=0 deleted=0 | chunks embedded=36 reused=0 deleted=0 | dead=0 | embed calls=9 (batch_size=16)
run 2 (no change) : v=b69545a4 seen=9 added=0 updated=0 skipped=9 deleted=0 | chunks embedded=0 reused=0 deleted=0 | dead=0 | embed calls still 9
run 3 (1 doc edit): v=b69545a4 seen=9 added=0 updated=1 skipped=8 deleted=0 | chunks embedded=1 reused=3 deleted=1 | dead=0
run 4 (doc removed): v=b69545a4 seen=8 added=0 updated=0 skipped=8 deleted=1 | chunks embedded=0 reused=0 deleted=4 | dead=0 | store now holds 32 chunks
dead letter: v=b69545a4 seen=9 added=0 updated=0 skipped=8 deleted=0 | chunks embedded=0 reused=0 deleted=0 | dead=1
   -> [('gifts.md', 'embed', 'RuntimeError: embedding service rejected input')]
   gifts.md is NOT in the manifest (so the next run retries it): True
run 5 (new embedder dim=256): version bd3dbd43 -> v=bd3dbd43 seen=9 added=9 updated=0 skipped=0 deleted=0 | chunks embedded=33 reused=0 deleted=0 | dead=0 | REBUILT
sample stored chunk: d375c871a6f892f8 | Nuts
```
- **Run 1** embeds all 36 chunks in 9 calls: one call per document because each document has fewer than 16 chunks.
- **Run 2** skips all 9 documents; the embed-call counter does not move. That is idempotency.
- **Run 3**: only the edited document is touched, and inside it only the one changed section: `embedded=1 reused=3 deleted=1`.
- **Run 4**: deleting a source file removes its 4 chunks (36 -> 32).
- **Dead letter**: one failing document reported with its stage (`embed`) while eight others were skipped as unchanged; it stays out of the manifest.
- **Run 5**: a different embedder dimension changes the version `b69545a4` -> `bd3dbd43` and triggers `REBUILT`; all 9 documents (33 chunks including the previously failing one) are re-embedded.

## Experiments (change ONE thing)
Each snippet runs from this step folder (`cd` into it first). Outputs below are from real runs.

### A smaller max_words changes the index VERSION even when the chunk count does not
```python
from pathlib import Path
KB = Path('../../14_rag_and_retrieval/data/kb')

from ragkit.chunking import HeadingChunker
from ragkit.ingest import *
for mw in (120, 20):
    ch, n = HeadingChunker(mw), 0
    for d in DirectoryLoader(KB).load():
        n += len(ch.chunk(d, MarkdownParser().parse(d)))
    print(ch.name, "->", n, "chunks for 9 docs | index_version", IngestionPipeline(DirectoryLoader(KB), MarkdownParser(), ch, HashingEmbedder(128), InMemoryVectorStore()).index_version)
```
Observed:
```text
heading(max_words=120) -> 36 chunks for 9 docs | index_version b69545a4
heading(max_words=20) -> 36 chunks for 9 docs | index_version 62c6a880
```
Both settings gave 36 chunks (each section is one short paragraph, so the cap never binds), **but the index version changed** (`b69545a4` -> `62c6a880`) because the chunker name includes `max_words`. Changing the setting would trigger a full rebuild even though the chunks are identical. A conservative version is safe; an over-sensitive one costs money. Decide what belongs in the fingerprint.

### batch_size=1 vs 16 on a cold index
```python
from pathlib import Path
KB = Path('../../14_rag_and_retrieval/data/kb')

from ragkit.chunking import HeadingChunker
from ragkit.ingest import *
class Count(HashingEmbedder):
    calls = 0
    def embed(self, t):
        Count.calls += 1
        return super().embed(t)
for bs in (16, 1):
    Count.calls = 0
    IngestionPipeline(DirectoryLoader(KB), MarkdownParser(), HeadingChunker(), Count(128), InMemoryVectorStore(), Manifest(), bs).run()
    print("batch_size", bs, "-> embed calls", Count.calls)
```
Observed:
```text
batch_size 16 -> embed calls 9
batch_size 1 -> embed calls 36
```
9 calls with batch_size 16 vs 36 with batch_size 1: same vectors, four times the per-call overhead. With a hosted embedding API each call is a network round trip.

### Edit the START of a document: heading chunks vs fixed-window chunks
```python
from ragkit.chunking import FixedWordChunker, HeadingChunker
from ragkit.domain import Document
from ragkit.ingest import *
body = "\n\n".join(f"## S{i}\n" + " ".join(f"w{i}x{j}" for j in range(30)) for i in range(6))
v1 = Document.from_text("d.md", "# T\n\n" + body)
v2 = Document.from_text("d.md", "# T\n\n## Intro\nA brand new opening sentence here.\n\n" + body)
for ch in (HeadingChunker(), FixedWordChunker(30, 5)):
    ids1 = {c.chunk_id for c in ch.chunk(v1, MarkdownParser().parse(v1))}
    ids2 = {c.chunk_id for c in ch.chunk(v2, MarkdownParser().parse(v2))}
    print(f"{ch.name:24s} before={len(ids1)} after={len(ids2)} re-embedded={len(ids2 - ids1)} reused={len(ids2 & ids1)}")
```
Observed:
```text
heading(max_words=120)   before=6 after=7 re-embedded=1 reused=6
fixed(size=30,overlap=5) before=8 after=8 re-embedded=8 reused=0
```
Insert one new section at the start: the heading chunker re-embeds **1** new chunk and reuses 6; the fixed-window chunker re-embeds **all 8** because every window shifts. Content-hash ids only help if chunk *boundaries* are stable. This is a practical argument for structure-aware chunking.

## Debugging practice
**BREAK IT:**
```python
import sys; sys.path.insert(0, '..')

from ragkit.chunking import FixedWordChunker
FixedWordChunker(size=40, overlap=40)
```
Real output:
```text
Traceback (most recent call last):
  File "<string>", line 4, in <module>
  File "/Users/aman/Desktop/Project_resume/RAP_interview/20_llm_system_lld/03_ingestion_pipeline/../ragkit/chunking.py", line 64, in __init__
    raise ValueError("need 0 <= overlap < size")
ValueError: need 0 <= overlap < size
```
The `ValueError` is raised at construction, in `FixedWordChunker.__init__`, with a message that states the rule. Reading it: last line = error; the frame above names the file and line where the guard lives. Without the guard the loop `range(0, n, size - overlap)` would be called with step 0 and raise `ValueError: range() arg 3 must not be zero` much later and further from the cause. Validate parameters in constructors.

## Rebuild checklist
- [ ] Draw the pipeline with the five interfaces and the diff step; label what each failure does.
- [ ] Write `run()` from memory: version check, skip unchanged, isolate failures, delete vanished.
- [ ] Compute `wanted`, `existing`, `stale`, `new` for a document where one paragraph changed.
- [ ] Explain why the manifest is updated only after success.
- [ ] Explain what belongs in `index_version` and why a rebuild uses a new collection and an alias switch.
- [ ] Implement `FixedWordChunker` and explain why it is unstable under edits at the top.

## The new problem (cliffhanger)
The index is built, versioned and incremental. Now a question arrives. How does the **online** side combine dense and sparse retrieval, rerank, fit chunks in a token budget, refuse politely, and check its own citations - as classes with sequence diagrams? [Step 04](../04_query_pipeline/WALKTHROUGH.md).

> **Say it aloud (interview box):** "My ingestion pipeline has five small interfaces and one orchestrator. Chunk ids are content hashes, so it diffs the store: new ids are embedded in batches, stale ids deleted, unchanged ones skipped. A manifest of document hashes makes it incremental, failures go to a dead-letter list with their stage, and an index version of embedder plus chunker forces a rebuild when vectors would change meaning."
