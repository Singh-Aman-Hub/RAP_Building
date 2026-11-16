# Terms — Chunker Low-Level Design

Folder: `19_chunking_deep_dive/11_chunker_lld` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [19 README](../README.md)

**Terms in this folder:** [Chunk dataclass (frozen)](#chunk-dataclass-frozen) · [Content-hash chunk id](#content-hash-chunk-id) · [Chunker Protocol](#chunker-protocol) · [Strategy registry](#strategy-registry) · [ChunkConfig and config_hash](#chunkconfig-and-config_hash) · [Idempotent re-chunking](#idempotent-re-chunking) · [Versioned index and alias](#versioned-index-and-alias) · [Embedding cache](#embedding-cache) · [Streaming chunking](#streaming-chunking) · [pytest](#pytest) · [typing.Protocol and runtime_checkable](#typingprotocol-and-runtime_checkable)

---

## Chunk dataclass (frozen)  `(concept)`
**1. Definition:** Immutable record with chunk_id, doc_id, doc_version, parent_id, text, embed_text, display_text, token_count, position, offsets, section_path, page, bbox, metadata, chunker_id and version.
**2. In plain words:** Everything needed to index, cite and migrate a chunk.
**3. Example to quote:** > "Chunk dataclass: Everything needed to index, cite and migrate a chunk."
**Seen in:** chunkerlib/model.py

## Content-hash chunk id  `(concept)`
**1. Definition:** sha256 of doc_id, chunker id, version, config hash, text hash and occurrence, not of the position.
**2. In plain words:** The same text keeps its id when other parts of the document change.
**3. Example to quote:** > "Content-hash chunk id: The same text keeps its id when other parts of the document change."
**Seen in:** chunkerlib/model.py

## Chunker Protocol  `(concept)`
**1. Definition:** typing.Protocol: any class with chunker_id, version and chunk(doc_id, text, base_offset) is a Chunker.
**2. In plain words:** A socket any strategy plugs into.
**3. Example to quote:** > "Chunker Protocol: A socket any strategy plugs into."
**Seen in:** chunkerlib/registry.py

## Strategy registry  `(concept)`
**1. Definition:** @register('name') stores a class in a dict; create(name, config, counter) builds it.
**2. In plain words:** Choose chunkers from a config file.
**3. Example to quote:** > "Strategy registry: Choose chunkers from a config file."
**Seen in:** chunkerlib/registry.py

## ChunkConfig and config_hash  `(concept)`
**1. Definition:** Frozen dataclass of all knobs, validated in __post_init__, with a stable hash of its fields.
**2. In plain words:** A fingerprint of how chunks were made.
**3. Example to quote:** > "ChunkConfig and config_hash: A fingerprint of how chunks were made."
**Seen in:** chunkerlib/model.py

## Idempotent re-chunking  `(concept)`
**1. Definition:** Same input gives the same ids; diff_chunks(old, new) yields kept / added / removed.
**2. In plain words:** Re-running ingestion is safe and cheap.
**3. Example to quote:** > "Idempotent re-chunking: Re-running ingestion is safe and cheap."
**Seen in:** chunkerlib/index.py

## Versioned index and alias  `(concept)`
**1. Definition:** Index name = collection + chunker@version + config hash + embed model; build, then flip the alias; keep the old one.
**2. In plain words:** Blue-green deployment for search.
**3. Example to quote:** > "Versioned index and alias: Blue-green deployment for search."
**Seen in:** chunkerlib/index.py

## Embedding cache  `(concept)`
**1. Definition:** (model, sha(text)) -> vector; unchanged text is not re-embedded after a chunker version bump.
**2. In plain words:** Pay for each distinct text once.
**3. Example to quote:** > "Embedding cache: Pay for each distinct text once."
**Seen in:** chunkerlib/index.py

## Streaming chunking  `(concept)`
**1. Definition:** Segment the input at blank lines, chunk each segment with base_offset, yield, forget.
**2. In plain words:** Memory is one segment.
**3. Example to quote:** > "Streaming chunking: Memory is one segment."
**Seen in:** chunkerlib/streaming.py

## pytest  `(library)`
**1. Definition:** Test runner: functions named test_*, plain assert, parametrize, pytest.raises.
**2. In plain words:** How the 12 tests run.
**3. Example to quote:** > "pytest: How the 12 tests run."
```python
python -m pytest -q
```
**Seen in:** test_chunker.py

## typing.Protocol and runtime_checkable  `(syntax)`
**1. Definition:** Structural typing; @runtime_checkable lets isinstance() check the members.
**2. In plain words:** Duck typing with a contract.
**3. Example to quote:** > "typing.Protocol and runtime_checkable: Duck typing with a contract."
**Seen in:** chunkerlib/registry.py
