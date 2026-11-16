# Step 11 — The chunker as a production design (LLD)

## The story so far
[Step 10](../10_chunking_evaluation/WALKTHROUGH.md) measured the strategies and left a shortlist. A measured chunker is still not a *production* chunker. Back to the [module README](../README.md).

**Where this step sits.** This is a low-level design (LLD) of the *chunking component* in a few hundred lines, runnable and tested. The **full production RAG LLD**, with the domain model, ingestion pipeline, query pipeline, design patterns, resilience, caching and observability around the chunker, is module 20 ([20_llm_system_lld](../../20_llm_system_lld/README.md)). Its chunking layer lives in [`ragkit/chunking.py`](../../20_llm_system_lld/ragkit/chunking.py) (a Template-Method `BaseChunker` with heading, fixed-word and sentence strategies) and is explained in [20/03 ingestion pipeline](../../20_llm_system_lld/03_ingestion_pipeline/WALKTHROUGH.md). Read this step for the *chunker-specific* problems (stable ids, idempotent re-chunking, versioned indexes, streaming, multilingual); read module 20 for how a chunker plugs into a whole system.

## The pain point
A notebook chunker has none of the properties a service needs:
- **Unstable ids.** `chunk_17` becomes `chunk_18` when someone inserts a sentence at the top, so every chunk below it looks "new", and everything is re-embedded and re-indexed.
- **Silent configuration.** Nobody knows which size, overlap or separators produced the live index, so a change cannot be reproduced or rolled back.
- **Big-bang upgrades.** Fixing a chunker bug means rebuilding and replacing the live index while users query it.
- **Memory.** A 500 MB export does not fit in a Python string.
- **Language blind spots.** English separators on Chinese text cut mid-sentence.
- **Untestable sprawl.** Three chunkers copy-pasted with `if kind == ...` branches.

## The spark
Treat the chunker as a small library with a data model, a stable interface and an operational story:
```
Chunk (frozen dataclass)         immutable record with content-hash id, offsets, section path, page/bbox, versions
ChunkConfig (frozen, validated)  every knob in one hashable object -> config_hash
Chunker Protocol + registry      add a strategy without touching callers (Strategy pattern, registry by name)
content-hash chunk ids           same text under the same chunker/config -> same id: edit one sentence, change one chunk
IndexManager + alias             build index_v2 next to index_v1, switch an alias atomically, keep v1 for rollback
EmbeddingCache keyed by text     an upgrade that leaves a text unchanged re-uses its vector
chunk_stream                     chunk in segments so memory is one segment, not the file
```

## Goal
Run and read `chunkerlib/` and show with numbers: (A) registry, config validation and one chunk as JSON; (B) stable ids and edit locality; (C) a chunker version bump to a new index with an alias switch and zero wasted embeddings; (D) streaming memory; (E) a CJK example with the multilingual tokenizer.

## The theory (read this before the interview)
**Immutable value objects.** `@dataclass(frozen=True)` makes `Chunk` and `ChunkConfig` read-only and hashable: no one can mutate a chunk after it is indexed, and a config can be fingerprinted.

**Config hash.** `sha256(json.dumps(asdict(config), sort_keys=True))`: every knob that changes the output is *in* the config, so the hash identifies the behaviour. Two indexes with different config hashes are different indexes; a name that includes the hash makes that visible.

**Content-hash chunk ids.** `id = sha(doc_id | chunker id | version | config hash | sha(text) | occurrence)`. Not a position and not a random UUID. Consequences: deterministic (idempotent) re-runs; an edit changes only the ids of chunks whose text changed; a chunker or config change changes *all* ids (correct, because the output changed). `occurrence` disambiguates identical texts within one document.

**Idempotent re-chunking (edit locality).** After an edit, re-chunk the document, compare ids: `kept` = unchanged, `added`, `removed`. Embed only `added`; delete only `removed`. A position-based id scheme would invalidate every chunk after the edit.

**Versioning and zero-downtime rollout.**
```
build  index "kb__recursive@1.1.0__cfg9a3f__model"    (offline, while the alias still points at 1.0.0)
verify (offline evaluation, step 10; shadow queries)
switch alias "kb" -> the new index                     atomic pointer flip
keep the old index for a rollback window, then drop it
```
`chunker_version` is bumped on *any* behaviour change, even a bug fix; config and embedding-model id are in the index name because a different embedder needs different vectors.

**Embedding cache keyed by `(model id, sha(embed_text))`.** The expensive step is the embedding. A new chunker version that produces the same text for most chunks re-uses their vectors. (With LLM-written contexts the same trick applies to the LLM call: cache by text hash and prompt version.)

**Protocol (structural typing) vs abstract base class.** A `typing.Protocol` says "anything with `chunker_id`, `version` and `chunk(doc_id, text, base_offset)` is a chunker", with no inheritance. Module 20's `BaseChunker` uses a *Template Method* base class instead; both are valid, the Protocol keeps the library decoupled.

**Streaming.** Read lines, close a segment at the first blank line after `segment_chars`, chunk it with a `base_offset` so offsets stay global, yield, forget. Memory is about one segment. The limit: a chunk never spans two segments, and a boundary at a blank line is a heuristic (you may cut inside a code fence), so choose the segment size larger than your largest structure.

**Multilingual.** Token counts must come from the *embedder's* tokenizer, so `TokenCounter` is injected. Separators must match the script: Chinese and Japanese need `。` `，` `！` `？` before the space; Thai has no spaces at all and needs a word segmenter; Arabic and Hebrew are right-to-left but split like Latin text.

**Tests that matter for a chunker.** Offsets point at the exact text; sizes are respected; chunking is deterministic; ids change with config and version; edit locality; overlap shares text; code fences are not headings; the cache survives a version bump; streaming covers the document.

**Status.** This is the shape of the ingestion layer in mature RAG systems (idempotent upserts keyed by content hash, versioned collections behind an alias, an embedding cache). Vector databases offer *aliases* or *collection swaps* for the switch; names differ by product.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| content-hash id | content-addressed id, deterministic id, idempotency key, fingerprint | near | Id derived from content, not from position or randomness |
| alias switch | blue/green index, collection alias, index alias swap, atomic cutover | near | Queries read a name that points at the live index |
| versioned index | index per chunker version, shadow index, parallel index | near | New version built beside the old one |
| idempotent upsert | upsert by id, delta indexing, incremental indexing, change data capture | near | Re-running yields the same state, touching only changed chunks |
| Chunker Protocol | interface, abstract base class (ABC), duck typing, Strategy | near | Structural interface vs inheritance-based |
| registry | plugin registry, factory by name, service locator | near | name to class lookup, filled by a decorator |
| embedding cache | vector cache, content-addressed store | yes | Vector keyed by model + text hash |
| frozen dataclass | immutable value object, record | yes | Fields cannot be reassigned after construction |
| streaming chunking | incremental / segment-wise chunking, out-of-core chunking | near | Bounded memory, global offsets by `base_offset` |

## Run it
```bash
cd 19_chunking_deep_dive/11_chunker_lld
python main.py
python -m pytest -q          # 12 unit tests, a word counter instead of a model: instant
# from the repo root:
HF_HUB_OFFLINE=1 .venv/bin/python 19_chunking_deep_dive/11_chunker_lld/main.py
```
About 20 s of CPU time for `main.py` (MiniLM and the e5 tokenizer); the tests need no model. Needs `torch`, `transformers`, `pytest` and the cached `all-MiniLM-L6-v2` and `multilingual-e5-small`; no downloads.

## Code, step by step
The library comes first (five small files), then the `main.py` demo that exercises it.

### L1. `model.py`: `Chunk`, `ChunkConfig`, ids  (`chunkerlib/model.py` lines 1–62)

```python
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
```

- **In simple words:** the data model: how a chunk is stored, how a configuration is validated and fingerprinted, and how the id is made.
- **Exactly:** `TokenCounter = Callable[[str], int]` is the injected tokeniser contract. `ChunkConfig` is frozen; `__post_init__` raises `ValueError` if `overlap_tokens` is not in `[0, max_tokens)` or `min_tokens` is out of range, so an invalid config can never exist. `config_hash` is the first 16 hex characters of SHA-256 over the sorted-key JSON of the config (`asdict` converts the dataclass to a dict). `Chunk` carries what a production system needs: a stable `chunk_id`; `doc_id` and `doc_version` (hash of the whole source); `parent_id` (small-to-big, [step 07](../07_hierarchical_parent_child/WALKTHROUGH.md)); `text` (exact source slice), `embed_text` (searched), `display_text` (shown); `token_count`; offsets; `section_path`; `page` and `bbox` for parsed documents ([step 09](../09_layout_aware_chunking_from_parsed_docs/WALKTHROUGH.md)); free `metadata`; `chunker_id` and `chunker_version`. `to_json` serialises with sorted keys and `ensure_ascii=False` (so Chinese stays readable). `make_chunk_id` hashes document, chunker, version, config hash, text hash and occurrence.
- **Why it is here:** this file is the schema; everything else is behaviour. It is the production counterpart of step 01's `RawChunk`.

### L2. `registry.py`: the Protocol and the registry  (`chunkerlib/registry.py` lines 1–39)

```python
"""registry.py -- the Chunker Protocol and a name -> class registry (strategy pattern)."""
from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from .model import Chunk, ChunkConfig, TokenCounter


@runtime_checkable
class Chunker(Protocol):
    """Anything with these members is a chunker (structural typing: no inheritance needed)."""
    chunker_id: str
    version: str

    def chunk(self, doc_id: str, text: str, base_offset: int = 0) -> list[Chunk]: ...


_REGISTRY: dict[str, type] = {}


def register(name: str):
    """Class decorator: @register("recursive") makes the class constructible by name from a config file."""
    def deco(cls):
        if name in _REGISTRY:
            raise ValueError(f"chunker {name!r} already registered")
        cls.chunker_id = name
        _REGISTRY[name] = cls
        return cls
    return deco


def create(name: str, config: ChunkConfig, count_tokens: TokenCounter) -> Chunker:
    if name not in _REGISTRY:
        raise KeyError(f"unknown chunker {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](config, count_tokens)


def available() -> list[str]:
    return sorted(_REGISTRY)
```

- **In simple words:** an interface that every chunker satisfies, and a name-to-class dictionary filled by a decorator.
- **Exactly:** `@runtime_checkable` lets `isinstance(x, Chunker)` check for the members at run time. `register("recursive")` returns a class decorator that refuses duplicate names (`ValueError`), stamps `cls.chunker_id = name` and stores the class. `create(name, config, count_tokens)` instantiates by name and raises `KeyError` with the list of available names for an unknown one. `available()` lists the registered names.
- **Why it is here:** callers say `create("markdown_sections", cfg, counter)`; adding a fourth strategy is one decorated class in a new file and no caller changes (open/closed principle).

### L3. `strategies.py`: three chunkers behind one interface  (`chunkerlib/strategies.py` lines 1–133)

```python
"""strategies.py -- three chunkers behind the same Protocol: token windows, recursive, markdown sections."""
from __future__ import annotations

import re
from collections import Counter

from .model import Chunk, ChunkConfig, TokenCounter, make_chunk_id, sha
from .registry import register


class _Base:
    version = "1.0.0"

    def __init__(self, config: ChunkConfig, count_tokens: TokenCounter):
        self.config, self.count = config, count_tokens

    def _emit(self, doc_id, text, spans, base_offset, doc_version, section_path=(), parent_id=None, embed_prefix="", metadata=None):
        """spans: [(start, end)] relative to `text`  ->  list[Chunk] with stable ids and offsets."""
        seen, out = Counter(), []
        for pos, (s, e) in enumerate(spans):
            raw = text[s:e]
            lead = len(raw) - len(raw.lstrip())
            body = raw.strip()
            if not body:
                continue
            embed = f"{embed_prefix}\n{body}" if embed_prefix else body
            occ = seen[body]
            seen[body] += 1
            out.append(Chunk(
                chunk_id=make_chunk_id(doc_id, self.chunker_id, self.version, self.config.config_hash, body, occ),
                doc_id=doc_id, doc_version=doc_version, parent_id=parent_id, text=body, embed_text=embed, display_text=body,
                token_count=self.count(embed), position=pos, char_start=base_offset + s + lead, char_end=base_offset + s + lead + len(body),
                section_path=tuple(section_path), metadata=dict(metadata or {}), chunker_id=self.chunker_id, chunker_version=self.version))
        return out


@register("token_window")
class TokenWindowChunker(_Base):
    """Fixed windows by tokens, found by binary search on the injected counter (works with ANY tokenizer)."""

    def chunk(self, doc_id, text, base_offset=0):
        spans, i, n = [], 0, len(text)
        while i < n:
            lo, hi = i + 1, n
            while lo < hi:                                           # largest j with count(text[i:j]) <= max_tokens
                mid = (lo + hi + 1) // 2
                lo, hi = (mid, hi) if self.count(text[i:mid]) <= self.config.max_tokens else (lo, mid - 1)
            spans.append((i, lo))
            if lo >= n:
                break
            back = lo
            while back > i + 1 and self.count(text[back:lo]) < self.config.overlap_tokens:   # step back to create the overlap
                back -= 1
            i = max(back, i + 1)
        return self._emit(doc_id, text, spans, base_offset, sha(text))


@register("recursive")
class RecursiveChunker(_Base):
    """Separator hierarchy; merges small pieces; hard-cuts only when no separator is left."""

    def _pieces(self, text, start, end, seps):
        """Split text[start:end] into atomic spans <= max_tokens, using the first separator that occurs."""
        if self.count(text[start:end]) <= self.config.max_tokens:
            return [(start, end)]
        sep, rest = seps[0], seps[1:]
        if sep == "":                                                # last resort: character cut by token ratio
            step = max(1, int((end - start) * self.config.max_tokens / max(self.count(text[start:end]), 1)))
            return [(i, min(i + step, end)) for i in range(start, end, step)]
        cuts, pos = [], start
        while True:
            k = text.find(sep, pos, end)
            if k == -1:
                break
            cuts.append(k + len(sep))                                # the separator stays at the END of the left piece
            pos = k + len(sep)
        if not cuts:
            return self._pieces(text, start, end, rest)
        edges = [start] + [c for c in cuts if c < end] + [end]
        out = []
        for a, b in zip(edges, edges[1:]):
            out += self._pieces(text, a, b, rest) if self.count(text[a:b]) > self.config.max_tokens else [(a, b)]
        return out

    def _merge(self, text, pieces):
        """Greedy pack; carry the tail of each chunk into the next until `overlap_tokens` is reached."""
        cfg, groups, cur = self.config, [], []
        for p in pieces:
            if cur and self.count(text[cur[0][0]:p[1]]) > cfg.max_tokens:
                groups.append(cur)
                tail, t = [], 0
                for q in reversed(cur):
                    t += self.count(text[q[0]:q[1]])
                    if t > cfg.overlap_tokens:
                        break
                    tail.insert(0, q)
                cur = tail
                while cur and self.count(text[cur[0][0]:p[1]]) > cfg.max_tokens:
                    cur.pop(0)
            cur.append(p)
        if cur:
            groups.append(cur)
        return [(g[0][0], g[-1][1]) for g in groups]

    def chunk(self, doc_id, text, base_offset=0):
        spans = self._merge(text, self._pieces(text, 0, len(text), tuple(self.config.separators)))
        return self._emit(doc_id, text, spans, base_offset, sha(text))


@register("markdown_sections")
class MarkdownSectionChunker(RecursiveChunker):
    """Heading tree -> one chunk per section body (recursive split if too long); breadcrumb in embed_text."""
    HEAD = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.M)

    def chunk(self, doc_id, text, base_offset=0):
        heads, fence = [], False
        for m in self.HEAD.finditer(text):
            if text[:m.start()].count("```") % 2 == 0:               # ignore '# comment' lines inside code fences
                heads.append((m.start(), m.end(), len(m.group(1)), m.group(2)))
        bounds = [(h[1], heads[i + 1][0] if i + 1 < len(heads) else len(text)) for i, h in enumerate(heads)]
        path, out, dv = [], [], sha(text)
        for h, (s, e) in zip(heads, bounds):
            path = path[:h[2] - 1] + [h[3]]
            body = text[s:e]
            if not body.strip():
                continue
            crumb = " > ".join(path)
            sub = self._merge(body, self._pieces(body, 0, len(body), tuple(self.config.separators)))
            out += self._emit(doc_id, body, sub, base_offset + s, dv, tuple(path), parent_id=f"{doc_id}#{sha(crumb, 8)}",
                              embed_prefix=crumb if self.config.breadcrumb else "")
        for i, c in enumerate(out):                                  # renumber positions over the whole document
            object.__setattr__(c, "position", i)
        return out
```

- **In simple words:** a shared base class that turns spans into `Chunk` objects, plus token-window, recursive and Markdown-section strategies.
- **Exactly:** `_Base._emit(doc_id, text, spans, base_offset, ...)` takes `(start, end)` spans *relative to `text`*, strips whitespace and moves the offsets with it, builds `embed_text` (an optional prefix such as the breadcrumb, plus the body), counts occurrences of identical texts, and creates the `Chunk` with its id. `TokenWindowChunker` finds the largest `j` with `count(text[i:j]) <= max_tokens` by *binary search on the injected counter*, so it works with any tokenizer, and steps back to create overlap. `RecursiveChunker._pieces` splits on the first separator present into atoms of at most `max_tokens` (keeping the separator at the end of the left piece; the empty separator is a character cut proportional to the token ratio), `_merge` packs pieces greedily and carries the tail as overlap: the algorithm of [step 02](../02_recursive_splitter_rebuilt/WALKTHROUGH.md) with spans instead of strings. `MarkdownSectionChunker` extends it: headings found by regex *outside code fences*, a section path, `parent_id = doc#hash(breadcrumb)`, and a breadcrumb prefix in `embed_text` (the idea of [step 03](../03_structure_aware_markdown/WALKTHROUGH.md)). `object.__setattr__(c, "position", i)` renumbers positions on a frozen dataclass, the one deliberate escape hatch.
- **Why it is here:** three strategies, one contract; each is a few dozen lines because the base class and the counter are shared.

### L4. `index.py`: cache, diff, versioned indexes  (`chunkerlib/index.py` lines 1–78)

```python
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
```

- **In simple words:** an embedding cache keyed by content, a diff of old versus new chunks, and an index manager with an alias.
- **Exactly:** `EmbeddingCache.get_many` embeds only texts whose `(model_id, sha(text))` is missing and counts `calls` (texts actually sent to the model, the cost). `diff_chunks(old, new)` compares ids and returns `kept`, `added`, `removed`. `IndexManager.index_name` builds `collection__chunker@version__config6__model`; `build` embeds all chunks of a version; `upsert_doc` replaces one document's chunks in place by diff (delete removed, embed added, refresh metadata of kept); `switch` flips the alias (an assertion prevents pointing at an unbuilt index); `search` is a brute-force dot-product search over the alias target (a vector database in production).
- **Why it is here:** the operational story: edit locality, version indexes and an alias, and a cache that makes upgrades cheap.

### L5. `streaming.py` and `__init__.py`  (`chunkerlib/streaming.py` lines 1–21)

```python
"""streaming.py -- chunk a document that does not fit in memory: cut it into segments at blank lines, chunk each segment."""
from __future__ import annotations

from typing import Iterable, Iterator

from .model import Chunk
from .registry import Chunker


def chunk_stream(lines: Iterable[str], chunker: Chunker, doc_id: str, segment_chars: int = 20_000) -> Iterator[Chunk]:
    """Read line by line, close a segment at the first blank line after `segment_chars`, chunk it, yield, forget it.
    Peak memory ~ one segment, not the whole file. Offsets stay global via base_offset."""
    buf, offset, pos = [], 0, 0
    for line in lines:
        buf.append(line)
        pos += len(line)
        if pos - offset >= segment_chars and not line.strip():     # a safe boundary: never inside a paragraph
            yield from chunker.chunk(doc_id, "".join(buf), base_offset=offset)
            buf, offset = [], pos
    if buf:
        yield from chunker.chunk(doc_id, "".join(buf), base_offset=offset)
```

- **In simple words:** chunk a very large document segment by segment.
- **Exactly:** `chunk_stream(lines, chunker, doc_id, segment_chars)` accumulates lines in `buf`; at the first blank line after `segment_chars` characters it calls `chunker.chunk(doc_id, "".join(buf), base_offset=offset)`, yields the chunks, and forgets the segment. `base_offset` keeps `char_start`/`char_end` global. The remaining lines are chunked at the end. It is a generator (`yield from`), so the caller can index chunks as they appear. `__init__.py` re-exports the public API and imports `strategies` so that importing the package registers the three chunkers.
- **Why it is here:** memory is bounded by the segment, not the file.

### 1. `main.py`: header docstring, imports  (`main.py` lines 1–24)

```python
"""
GOAL:  The low-level design of a production chunker, runnable: a frozen `Chunk` dataclass, a `Chunker` Protocol with a
       strategy registry, config objects with a config hash, content-hash chunk ids, idempotent re-chunking (edit one
       sentence -> only that chunk is re-embedded), versioned indexes behind an alias with an embedding cache,
       streaming of a large document, and a multilingual (CJK) example. Library: chunkerlib/. Tests: test_chunker.py.

RUN (from this step folder):  python main.py        (tests: python -m pytest -q)
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 19_chunking_deep_dive/11_chunker_lld/main.py
EXPECTED OUTPUT (shortened):
    A) registry + one Chunk as JSON          B) ids stable across runs, edit locality (kept / added / removed)
    C) chunker version bump -> new index name, 0 re-embeddings for unchanged text, alias switch
    D) streaming: peak memory batch vs stream   E) multilingual: Chinese text, e5 tokenizer
NEEDS: torch, transformers (cached MiniLM, multilingual-e5-small).  ~20 s.
"""
import sys
import tracemalloc
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from chunkerlib import ChunkConfig, EmbeddingCache, IndexManager, available, chunk_stream, create, diff_chunks  # noqa: E402
from common19 import count_tokens, get_embedder, load_docs  # noqa: E402

```

- **In simple words:** goal, run lines and imports of the library and the shared helpers.
- **Exactly:** `HERE` is this step's folder; both it (for `chunkerlib`) and its parent (for `common19`) are put on `sys.path`. `tracemalloc` is Python's allocation tracer, used in section D.
- **Why it is here:** the demo uses the *library*, not copies of the code.

### 2. Section A: registry, config, one chunk as JSON  (`main.py` lines 26–43)

```python
def main():
    docs = {d["doc_id"]: d for d in load_docs()}
    hb = docs["employee_handbook"]
    emb = get_embedder()
    cache = EmbeddingCache("all-MiniLM-L6-v2", emb.encode)

    # ---- A) registry, config, one Chunk ------------------------------------------------------------------------------
    print("A) registered chunkers:", available())
    cfg = ChunkConfig(max_tokens=120, overlap_tokens=0)
    md = create("markdown_sections", cfg, count_tokens)
    chunks = md.chunk("employee_handbook", hb["text"])
    print(f"   markdown_sections v{md.version} config_hash={cfg.config_hash} -> {len(chunks)} chunks")
    c = chunks[3]
    print("   chunk[3] as JSON:", c.to_json()[:430], "...")
    try:
        ChunkConfig(max_tokens=64, overlap_tokens=64)
    except ValueError as e:
        print("   invalid config rejected at construction:", e)
```

- **In simple words:** list the strategies, create the Markdown chunker, chunk the handbook, print one chunk as JSON, and show an invalid config being rejected.
- **Exactly:** `EmbeddingCache("all-MiniLM-L6-v2", emb.encode)` wraps the embedder with a cache that counts calls. `create("markdown_sections", cfg, count_tokens)` is the registry in action. `c.to_json()[:430]` is the first part of the serialised `Chunk`, showing the id, the versions, offsets and section path. `ChunkConfig(max_tokens=64, overlap_tokens=64)` raises `ValueError` at construction.
- **Why it is here:** the first picture of the production record, and validation at the edge (fail at construction, not deep inside a splitter).

### 3. Section B: stable ids and edit locality  (`main.py` lines 45–58)

```python
    # ---- B) stable ids and edit locality --------------------------------------------------------------------------------
    again = md.chunk("employee_handbook", hb["text"])
    print(f"\nB) chunk it twice: same ids? {[x.chunk_id for x in chunks] == [x.chunk_id for x in again]}")
    mgr = IndexManager("kb", cache)
    name1 = mgr.index_name(md.chunker_id, md.version, cfg.config_hash)
    mgr.build(name1, chunks)
    mgr.switch(name1)
    base_calls = cache.calls
    edited = hb["text"].replace("The budget is 1,200 euros", "The budget is 1,500 euros")
    new = md.chunk("employee_handbook", edited)
    d = mgr.upsert_doc(name1, "employee_handbook", new)
    print(f"   edit one number in the handbook: kept {len(d.kept)}, added {len(d.added)}, removed {len(d.removed)}; "
          f"texts re-embedded: {cache.calls - base_calls} of {len(new)}")
    print(f"   (a position-based id scheme would have invalidated every chunk after the edit)")
```

- **In simple words:** chunk twice and compare ids; index the document; change one number in the text and see how many chunks change.
- **Exactly:** `IndexManager("kb", cache)`, `build(name1, chunks)` and `switch(name1)` create the first live index. `hb["text"].replace("The budget is 1,200 euros", "The budget is 1,500 euros")` is a one-number edit. `mgr.upsert_doc(...)` returns the `Diff`; `cache.calls - base_calls` is the number of texts that actually went through the model.
- **Why it is here:** the headline benefit of content-hash ids: one edit, one chunk re-embedded.

### 4. Section C: version bump, new index, alias switch  (`main.py` lines 60–74)

```python
    # ---- C) versioned index + alias switch ---------------------------------------------------------------------------------
    md2 = create("markdown_sections", cfg, count_tokens)
    md2.version = "1.1.0"                                          # pretend we fixed a bug in the splitter
    name2 = mgr.index_name(md2.chunker_id, md2.version, cfg.config_hash)
    before = cache.calls
    mgr.build(name2, md2.chunk("employee_handbook", hb["text"]))
    print(f"\nC) chunker 1.0.0 -> 1.1.0: new index {name2!r}")
    print(f"   chunk ids differ (new version in the hash): {not (set(mgr.indexes[name1]) & set(mgr.indexes[name2]))}; "
          f"embeddings recomputed: {cache.calls - before} (cache keyed by text hash)")
    q = emb.encode(["How many weeks of leave does the primary caregiver get?"])[0]
    top_old = mgr.search(q, 1)[0][0]
    mgr.switch(name2)                                              # atomic alias flip; the old index stays for rollback
    top_new = mgr.search(q, 1)[0][0]
    print(f"   alias {mgr.alias!r}; top hit before/after switch: {top_old.section_path[-1]!r} / {top_new.section_path[-1]!r}; "
          f"indexes kept: {len(mgr.indexes)}")
```

- **In simple words:** pretend a bug fix bumped the chunker to 1.1.0, build the new index beside the old one, compare, and flip the alias.
- **Exactly:** `md2.version = "1.1.0"` changes every chunk id (the version is in the hash) and the index name. `mgr.build(name2, ...)` embeds through the same cache, so unchanged texts cost nothing: `embeddings recomputed: 0`. `set(mgr.indexes[name1]) & set(mgr.indexes[name2])` tests that no ids are shared. `mgr.search` runs against the alias target before and after `switch`; both indexes remain, so rollback is another `switch`.
- **Why it is here:** the zero-downtime rollout story with numbers.

### 5. Section D: streaming memory  (`main.py` lines 76–87)

```python
    # ---- D) streaming -------------------------------------------------------------------------------------------------------------
    big = "\n\n".join(d["text"] for d in docs.values()) * 6
    rec = create("recursive", ChunkConfig(max_tokens=150), count_tokens)
    tracemalloc.start()
    batch = rec.chunk("big", big)
    peak_batch = tracemalloc.get_traced_memory()[1]
    tracemalloc.reset_peak()
    n = sum(1 for _ in chunk_stream(iter(big.splitlines(keepends=True)), rec, "big", segment_chars=6000))
    peak_stream = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    print(f"\nD) {len(big):,}-character document: batch -> {len(batch)} chunks, peak {peak_batch / 1e6:.1f} MB; "
          f"stream (6k-char segments) -> {n} chunks, peak {peak_stream / 1e6:.1f} MB")
```

- **In simple words:** build a big text (every document repeated six times), chunk it in one go and then streamed, and compare peak memory.
- **Exactly:** `tracemalloc.start()` then `get_traced_memory()[1]` is the peak allocated bytes since the last reset; `reset_peak()` zeroes it between the two runs; `chunk_stream(iter(big.splitlines(keepends=True)), rec, "big", segment_chars=6000)` feeds the text line by line. `sum(1 for _ in ...)` consumes the generator while counting.
- **Why it is here:** a measured claim instead of "streaming saves memory". The saving is bounded by the size of the *chunker's* own temporaries too.

### 6. Section E: multilingual  (`main.py` lines 89–101)

```python
    # ---- E) multilingual --------------------------------------------------------------------------------------------------------------
    from transformers import AutoTokenizer
    e5 = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    e5_count = lambda t: len(e5(t, add_special_tokens=False)["input_ids"])
    zh = "试用期结束后，员工每周最多可以在家远程工作三天。远程工作需要提前获得经理批准。设备预算为一千二百欧元，每三年可以使用一次。所有收据必须在三十天内上传。"
    cjk = ChunkConfig(max_tokens=18, separators=("\n\n", "\n", "。", "，", " ", ""))
    cs = create("recursive", cjk, e5_count).chunk("zh", zh)
    print(f"\nE) Chinese, e5 tokenizer ({e5_count(zh)} tokens for {len(zh)} chars), separators include '。' and '，':")
    for x in cs:
        print(f"   [{x.token_count:>2} tok] {x.text}")
    with_en = create("recursive", ChunkConfig(max_tokens=18), e5_count).chunk("zh", zh)
    print(f"   with the English-only separators the same text gives {len(with_en)} chunks of "
          f"{[x.token_count for x in with_en]} tokens; first one ends {with_en[0].text[-6:]!r} (cut mid-sentence by the character fallback)")
```

- **In simple words:** chunk a Chinese paragraph using the multilingual e5 tokenizer as the counter, once with CJK separators and once with English ones.
- **Exactly:** `e5_count` is a `TokenCounter` built from the e5 tokenizer; passing it as `count_tokens` is the whole multilingual adaptation of the counting. `ChunkConfig(max_tokens=18, separators=("\n\n", "\n", "。", "，", " ", ""))` puts CJK sentence and clause punctuation into the hierarchy. With the English-only default separators the recursive chunker falls through to the empty separator and cuts by characters, mid-sentence.
- **Why it is here:** language support is data (separators) plus an injected counter, not a code change.

### 7. Entry point and the TRY IT / BREAK IT lines  (`main.py` lines 104–109)

```python
if __name__ == "__main__":
    main()

# TRY IT: change ChunkConfig(max_tokens=120) to 60 in section A; how many chunks, and does the config_hash change?
# TRY IT: in section B edit a sentence in the FIRST section vs the LAST section; is 'added' still 1?
# BREAK IT: ChunkConfig(max_tokens=64, overlap_tokens=64) raises ValueError at construction time, not deep inside a splitter.
```

- **In simple words:** guard plus exercises.
- **Exactly:** an invalid config fails at construction, before any splitter runs.
- **Why it is here:** the earlier the failure, the cheaper the bug.

## The unit tests (`test_chunker.py`)
`python -m pytest -q` runs them in a fraction of a second because they inject a **word counter** (`lambda t: len(t.split())`) and need no model. They check: the registry lists the three names and rejects an unknown one; an invalid config raises `ValueError`; for each strategy the offsets point at the exact text; chunking is deterministic; ids change with config and with version; **editing one sentence changes exactly one chunk** (`added == 1`, `removed == 1`); overlap really shares text; a `# comment` inside a code fence is not a heading and the breadcrumb is present; a version bump re-uses cached embeddings (zero new calls) and the alias can be switched; and a streamed document is covered with exact offsets. Injecting the counter is what makes the library testable without a model: dependency injection pays for itself in tests.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `@dataclass(frozen=True)` | Immutable, hashable record |
| `from __future__ import annotations` | Annotations are stored as strings, so forward references work |
| `class Chunker(Protocol)` + `@runtime_checkable` | Structural interface; `isinstance` checks for the members |
| `def register(name): def deco(cls): ...; return deco` | A decorator factory |
| `Callable[[str], int]` | Type of a function from `str` to `int` |
| `object.__setattr__(c, "position", i)` | Bypass `frozen` to set a field (used once, deliberately) |
| `yield from chunker.chunk(...)` | Delegate to an iterable inside a generator |
| `Mapping`, `Optional[int]` | Type hints; `Optional[X]` is `X or None` |
| `sha256(...).hexdigest()[:16]` | Short fingerprint of a string |
| `with`-less `tracemalloc.start()` / `get_traced_memory()` | Manual start/stop of allocation tracing |

## Libraries & functions
| Function | What it does | Internally |
|---|---|---|
| `hashlib.sha256` | Cryptographic hash | Fixed-size digest of the bytes; same input, same output, on every machine |
| `json.dumps(sort_keys=True)` | Canonical JSON | Keys sorted so the same config gives the same string, hence the same hash |
| `dataclasses.asdict` | Dataclass to dict | Recursive copy of the fields |
| `typing.Protocol` | Structural type | Checked by static tools; `runtime_checkable` adds a member check to `isinstance` |
| `tracemalloc` | Memory profiler for Python allocations | Hooks the allocator; reports current and peak bytes |
| `pytest` | Test runner | Collects `test_*` functions, runs `assert`s, `parametrize` multiplies cases |

## Execution trace (real run)
Real output of `python main.py` (progress bar removed; the memory and time figures vary by machine).
```
A) registered chunkers: ['markdown_sections', 'recursive', 'token_window']
   markdown_sections v1.0.0 config_hash=afb6097d7d8f43f9 -> 19 chunks
   chunk[3] as JSON: {"bbox": null, "char_end": 958, "char_start": 675, "chunk_id": "3ccd22631bfa3daaf402", "chunker_id": "markdown_sections", "chunker_version": "1.0.0", "display_text": "Lumen pays for a desk setup so that you can work comfortably. The budget is 1,200 euros and it can be spent once every three years. It covers a desk, a chair and a monitor. Receipts must be uploaded to the expense tool within 30 days. Items bought after that date ...
   invalid config rejected at construction: overlap_tokens (64) must be >= 0 and < max_tokens (64)

B) chunk it twice: same ids? True
   edit one number in the handbook: kept 18, added 1, removed 1; texts re-embedded: 1 of 19
   (a position-based id scheme would have invalidated every chunk after the edit)

C) chunker 1.0.0 -> 1.1.0: new index 'kb__markdown_sections@1.1.0__afb609__all-MiniLM-L6-v2'
   chunk ids differ (new version in the hash): True; embeddings recomputed: 0 (cache keyed by text hash)
   alias 'kb__markdown_sections@1.1.0__afb609__all-MiniLM-L6-v2'; top hit before/after switch: '2.3.1 Primary caregiver' / '2.3.1 Primary caregiver'; indexes kept: 2

D) 125,724-character document: batch -> 234 chunks, peak 1.7 MB; stream (6k-char segments) -> 244 chunks, peak 1.1 MB

E) Chinese, e5 tokenizer (46 tokens for 75 chars), separators include '。' and '，':
   [15 tok] 试用期结束后，员工每周最多可以在家远程工作三天。
   [ 9 tok] 远程工作需要提前获得经理批准。
   [14 tok] 设备预算为一千二百欧元，每三年可以使用一次。
   [11 tok] 所有收据必须在三十天内上传。
   with the English-only separators the same text gives 3 chunks of [18, 18, 13] tokens; first one ends '。远程工作需' (cut mid-sentence by the character fallback)
```

And the test run, real output of `python -m pytest -q`:
```
............                                                             [100%]
12 passed in 0.27s
```

How to read it:
- **A:** three strategies are registered. The handbook becomes 19 chunks under `markdown_sections` 1.0.0 with config hash `afb6097d7d8f43f9`; the JSON shows the production record (id, offsets, `display_text`, versions, `bbox: null` because this is not a parsed document). `ChunkConfig(max_tokens=64, overlap_tokens=64)` is rejected immediately with a readable message.
- **B:** chunking twice gives the same ids. Changing one number ("1,200" to "1,500") leaves **18 kept, 1 added, 1 removed**, and exactly 1 of 19 texts is re-embedded.
- **C:** the version bump to 1.1.0 yields a new index name that contains chunker, version, config hash prefix and embedding model, **no shared chunk ids**, yet **0 embeddings recomputed** (the cache is keyed by text). After the alias switch the same question gets the same top hit and both indexes remain for rollback.
- **D:** for a 125,724-character document, batch chunking peaks at 1.7 MB and streaming in 6k-character segments at 1.1 MB. The saving is modest here because the whole input string already exists in memory in this demo; the point is that peak memory follows the segment, not the file. Streaming gives 244 chunks instead of 234, because every segment boundary forces a cut.
- **E:** the Chinese paragraph is 46 e5 tokens for 75 characters and, with CJK separators, becomes four whole-sentence chunks (9 to 15 tokens each). With the English-only separators the same text gives 3 chunks of 18, 18 and 13 tokens, the first ending mid-sentence at `'。远程工作需'`.

## Experiments (change ONE thing)
| Change | What you will see | Why |
|---|---|---|
| `ChunkConfig(max_tokens=60)` in section A | 21 chunks instead of 19 and `config_hash` `7d899763e99c70b5` instead of `afb6097d7d8f43f9` (measured) | The hash covers every field, so a changed size is a new index name |
| Edit a sentence in the first section, then in the last | Both give kept 18, added 1, removed 1 (measured) | Ids are content hashes, not positions |
| Add `("min_tokens", 20)` to the config | A new config hash and possibly different chunks | `min_tokens` is a config field |
| Chunk twice with a different `doc_id` | Different ids for identical text | `doc_id` is in the id hash |
| `segment_chars=500` in section D | More segments; peak memory lower; chunks cannot span segments | The segment bound is the memory bound |

## Debugging practice
**BREAK IT:** register two chunkers under the same name (decorate a second class with `@register("recursive")`), or build `ChunkConfig(max_tokens=64, overlap_tokens=64)`.

Real results of both experiments:
```
ValueError: chunker 'recursive' already registered
ValueError: overlap_tokens (64) must be >= 0 and < max_tokens (64)
```
The first is raised by `register` at import time, the second by `ChunkConfig.__post_init__` at construction, so both fail before any document is chunked, with a message that names the offender. **Fix:** give the second chunker a new name (or remove the duplicate import that registers the same module twice) and keep `overlap < max`. Compare with [step 02](../02_recursive_splitter_rebuilt/WALKTHROUGH.md), where the same mistake raised a bare `AssertionError`.

## Common mistakes
- Position-based ids (`doc-3`, `chunk_17`): a one-sentence edit invalidates the rest of the document.
- Random UUIDs for chunks: the same document gets new ids on every run and the index grows with duplicates.
- Forgetting to put the chunker version, the config hash or the embedding model in the index name.
- Mutating the live index in place during a chunker upgrade; build beside it and flip the alias.
- Not bumping `chunker_version` after a bug fix; stale chunks keep their old ids.
- Counting tokens with the wrong tokenizer in production (the injected counter must be the embedder's).
- English separators on CJK text; no word segmenter for Thai.
- Streaming a document and cutting inside a code fence or a table because the segment boundary was a blank line.
- A cache keyed by chunk id instead of by text: re-embeds everything on a version bump.
- Testing the chunker only with a real model: slow tests that nobody runs; inject a word counter.

## Rebuild checklist
1. `ChunkConfig` (frozen, validated in `__post_init__`, `config_hash` from sorted JSON).
2. `Chunk` (frozen) with id, offsets, section path, page/bbox, versions; `to_json`.
3. `make_chunk_id` from doc, chunker, version, config hash, text hash, occurrence.
4. `Chunker` Protocol, `register` decorator (reject duplicates), `create`, `available`.
5. `_emit` base method; `TokenWindowChunker`, `RecursiveChunker`, `MarkdownSectionChunker`.
6. `EmbeddingCache`, `diff_chunks`, `IndexManager` (name, build, upsert, switch, search).
7. `chunk_stream` with `base_offset`.
8. Tests with an injected word counter, including edit locality and cache reuse.
9. Write a fourth strategy (for example sentence-based) behind the Protocol and test it ([solution](../solutions/solutions.md)).

## Say it aloud
"In production a chunker is a small library: a frozen `Chunk` with a content-hash id built from the document, chunker version, config hash and text, so re-running is idempotent and editing one sentence changes one chunk. A validated config object with a hash, a Protocol plus a registry so strategies are pluggable, and an injected tokenizer so it is multilingual and testable. Upgrades build a new index named after chunker version, config and embedding model, verify offline, and switch an alias atomically, with an embedding cache keyed by text so unchanged chunks are not re-embedded and the old index kept for rollback. Large documents are chunked in segments with global offsets. The whole-system version of this is the RAG LLD in module 20."

## The new problem (cliffhanger)
This is the end of module 19, but not of the story. A chunker is only as good as the text it receives: for scanned documents and complex PDFs the text itself has to be recovered first, which is [module 18 (OCR and document parsing)](../../18_ocr_and_document_parsing/README.md), whose last step hands over the parsed-document JSON that [step 09](../09_layout_aware_chunking_from_parsed_docs/WALKTHROUGH.md) chunks. And the chunker has to live inside a whole service with ingestion, query, caching, resilience and observability: [module 20, the LLM system LLD](../../20_llm_system_lld/README.md). Back to the [module README](../README.md) or on to [module 20](../../20_llm_system_lld/03_ingestion_pipeline/WALKTHROUGH.md).
