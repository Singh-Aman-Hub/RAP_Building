# Terms — The ingestion pipeline: Loader -> Parser -> Chunker -> Embedder -> VectorStore

Folder: `20_llm_system_lld/03_ingestion_pipeline` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [20 README](../README.md)

**Terms in this folder:** [Idempotent upsert](#idempotent-upsert--concept) · [Incremental re-index (manifest)](#incremental-re-index-manifest--concept) · [Batch embedding](#batch-embedding--concept) · [Dead-letter queue (DLQ)](#dead-letter-queue-dlq--concept) · [Index version (fingerprint)](#index-version-fingerprint--concept) · [Loader, Parser, Chunker, Embedder, VectorStore](#loader-parser-chunker-embedder-vectorstore--concept) · [Template Method](#template-method--concept) · [Generator (yield) for streaming a source](#generator-yield-for-streaming-a-source--syntax) · [Set difference (wanted - existing)](#set-difference-wanted---existing--syntax) · [Exception chaining (raise from)](#exception-chaining-raise-from--syntax) · [Pre-filter vs post-filter](#pre-filter-vs-post-filter--concept) · [ValueError: need 0 <= overlap < size](#valueerror-need-0--overlap--size--error)

---

## Idempotent upsert  `(concept)`
**1. Definition:** A write that creates a record or overwrites the one with the same id, so repeating it leaves the same state.
**2. In plain words:** Pressing the same elevator button twice does not summon two elevators.
**3. Example to quote:** > "Content-hash ids make re-running the indexer a no-op."
**Seen in:** `ragkit/ingest.py`

## Incremental re-index (manifest)  `(concept)`
**1. Definition:** Re-indexing only documents whose content hash differs from the one recorded in a manifest, and deleting those that disappeared.
**2. In plain words:** Re-reading only the chapters that changed.
**3. Example to quote:** > "The manifest is updated only after a document succeeds, so failures are retried next run."
```python
if old == doc.content_hash: skip
```
**Seen in:** `ragkit/ingest.py`, run()

## Batch embedding  `(concept)`
**1. Definition:** Embedding many texts in one model call, in groups of batch_size.
**2. In plain words:** Washing 16 plates at once instead of one by one.
**3. Example to quote:** > "Batching cut embed calls from 36 to 9 on 36 chunks."
```python
for i in range(0, len(new), batch_size): ...
```
**Seen in:** `ragkit/ingest.py`, _index_doc

## Dead-letter queue (DLQ)  `(concept)`
**1. Definition:** A holding list for items that failed processing, with enough context (stage, error) to inspect and replay them.
**2. In plain words:** The 'return to sender' shelf in a post office.
**3. Example to quote:** > "One bad document must not abort the batch; it goes to the DLQ and the run continues."
```python
DeadLetter(doc_id, source, stage, error)
```
**Seen in:** `ragkit/ingest.py`

## Index version (fingerprint)  `(concept)`
**1. Definition:** A short hash of embedder name, dimension, chunker settings and schema; a change means stored vectors are stale.
**2. In plain words:** A recipe version number printed on every jar.
**3. Example to quote:** > "When it changes, build a new index beside the old and switch an alias (blue/green)."
```python
content_hash(embedder.name, str(embedder.dim), chunker.name, 'schema1', length=8)
```
**Seen in:** `ragkit/ingest.py`

## Loader, Parser, Chunker, Embedder, VectorStore  `(concept)`
**1. Definition:** The five stages of ingestion, each behind a one-purpose Protocol.
**2. In plain words:** An assembly line with swappable stations.
**3. Example to quote:** > "Keeping the stages separate lets me swap a PDF parser or a Qdrant store without touching the rest."
**Seen in:** `ragkit/protocols.py`

## Template Method  `(concept)`
**1. Definition:** A base class defines the fixed algorithm and calls overridable hook methods that subclasses implement.
**2. In plain words:** A form with fixed sections and one blank to fill.
**3. Example to quote:** > "BaseChunker.chunk() is fixed; subclasses supply _groups()."
```python
class BaseChunker(ABC):
    def chunk(...): ... self._groups(elements)
```
**Seen in:** `ragkit/chunking.py`

## Generator (yield) for streaming a source  `(syntax)`
**1. Definition:** A function with `yield` returns an iterator that produces items lazily.
**2. In plain words:** A conveyor belt rather than a warehouse.
**3. Example to quote:** > "Loader.load() yields documents so a million files never sit in memory."
```python
def load(self):
    for p in sorted(...): yield Document...
```
**Seen in:** `ragkit/ingest.py`

## Set difference (wanted - existing)  `(syntax)`
**1. Definition:** Items in the first set but not the second.
**2. In plain words:** Which guests are on the list but have not arrived.
**3. Example to quote:** > "existing - wanted = stale chunks to delete; wanted - existing = new chunks to embed."
```python
stale = existing - wanted
```
**Seen in:** `ragkit/ingest.py`

## Exception chaining (raise from)  `(syntax)`
**1. Definition:** `raise New from old` stores the original as `__cause__`.
**2. In plain words:** Writing 'caused by' on the incident report.
**3. Example to quote:** > "StageError keeps the real traceback and adds the stage name."
```python
raise StageError(stage, exc) from exc
```
**Seen in:** `ragkit/ingest.py`

## Pre-filter vs post-filter  `(concept)`
**1. Definition:** Applying a metadata filter before ranking (always returns up to k) versus after (may return fewer than k).
**2. In plain words:** Filter the shelf first, then pick the best books.
**3. Example to quote:** > "InMemoryVectorStore filters first, then scores."
```python
cands = [(c, v) for c, v in items if _matches(c, where)]
```
**Seen in:** `ragkit/ingest.py`

## ValueError: need 0 <= overlap < size  `(error)`
**1. Definition:** Raised by FixedWordChunker when overlap is not smaller than the window.
**2. In plain words:** A window that never slides.
**3. Example to quote:** > "Validate parameters in the constructor so the failure is early and clear."
```python
FixedWordChunker(size=40, overlap=40)
```
**Seen in:** `WALKTHROUGH.md`, Debugging practice
