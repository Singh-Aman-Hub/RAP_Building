# Step 02 — The domain model: Document, Element, Chunk, Embedding, SearchHit, Citation, Answer

## The story so far
[Step 01](../01_lld_foundations/WALKTHROUGH.md) gave us the vocabulary: Protocols, frozen dataclasses, enums, dependency injection. Module [14](../../14_rag_and_retrieval/README.md) passed plain `dict`s around (`{"chunk_id": ..., "text": ...}`) and parallel lists of indexes.

## The pain point
Dicts have no shape. `chunk["chunck_id"]` (typo) is a `KeyError` three layers away, a score list and a chunk list drift out of step, and nobody knows whether `"position"` is an int or a string. Worse, ids: if every run invents new random ids, re-indexing an unchanged document duplicates everything; if ids depend on the position, inserting one paragraph changes every id after it.

## The spark
Name every noun of the system as a small **immutable value object**, give each a deliberate **identity rule** (content hash for content, random uuid for events), validate it in `__post_init__`, and give it `to_dict` / `from_dict` so it can cross a JSON boundary. Use pydantic only where text from outside enters (HTTP, LLM JSON).

## Goal
Build and read `ragkit/ids.py`, `ragkit/domain.py` and `ragkit/schemas.py`; understand why `chunk_id = hash(doc_id, section, text)`; see immutability, validation and round-trips run; know when to pick dataclass and when pydantic v2.

## The theory (read this before the interview)
### The entities
| Object | What it is | Identity | Why a separate type |
|---|---|---|---|
| `Document` | one source file/page, whole text + metadata | `doc_id = hash(source)` (stable), `content_hash` (changes with text) | the pair tells the indexer *same document?* and *changed?* |
| `Element` | a parsed piece: heading, paragraph, list, table, code | none (position in doc) | chunkers work on structure, not raw strings |
| `Chunk` | the unit of retrieval and of prompt context | `chunk_id = hash(doc_id, section, text)` | its id must survive unrelated edits |
| `Embedding` | vector + the model that made it | `chunk_id` + `model` | vectors from different models are not comparable; the model name travels with the vector |
| `SearchHit` | chunk + score + which stage produced it + rank | none | scores from dense (cosine), BM25, RRF and a cross-encoder have different scales |
| `Citation` | `[n]` the model wrote -> the chunk it points to, `supported` flag | marker | lets code check the answer |
| `Answer` | text + citations + refusal flag + context ids + index version | none | the API response, loggable and cacheable |

### Identity: content hash vs uuid
- **Content hash** (`sha256(parts joined by \x1f)[:16]`): deterministic. Same chunk -> same id -> the vector store sees "already have it" and the indexer skips embedding (idempotent upsert, step 03). 16 hex chars = 64 bits: collision probability is negligible below ~10^8 chunks (birthday bound about n^2 / 2^65).
- **uuid4** (`new_id`): random. Right for *events* (a trace, a request, feedback) which must be unique even if their content repeats.
- **Do not hash the position**: insert one paragraph at the top and every later chunk gets a new id, so all are re-embedded (Experiments: 0 of 3 ids kept vs 3 of 3).
- **Separator matters**: `hash("ab","c")` must differ from `hash("a","bc")`; we join with the unit-separator character `\x1f`.

### Immutability, three levels
1. `frozen=True`: fields cannot be re-assigned.
2. `slots=True`: no stray attributes, smaller objects.
3. **Deep** immutability needs care: `frozen` is shallow. We store metadata as `MappingProxyType(dict(copy))` (read-only view of a private copy) and sequences as tuples. `Document.metadata["x"] = 1` -> `TypeError`.

### Validation
Validate **at construction** (`__post_init__`): empty text, a refusal that carries citations. An invalid object never exists, so downstream code needs no defensive checks. Dataclass type hints are *not* enforced: `Chunk(..., position="3")` is accepted (see output). Pydantic v2 *does* enforce and coerce.

### dataclass vs pydantic v2
| | `@dataclass(frozen, slots)` | pydantic `BaseModel` |
|---|---|---|
| Checks types at runtime | no | yes (coerces `"3"` -> `3`, errors collected in one `ValidationError`) |
| Speed / memory | fastest, smallest | slower (Rust core `pydantic-core` makes v2 fast, still overhead) |
| JSON Schema, `model_validate_json` | no | yes (`model_json_schema()`; the same schema you give an LLM for structured output, step 08) |
| Use for | internal objects we construct ourselves | **trust boundaries**: HTTP request bodies, LLM JSON, config files |
Rule: *parse with pydantic at the edge, then work with plain trusted objects inside*.

### Serialisation
`to_dict()` returns only JSON types (str, int, float, bool, list, dict); `from_dict()` rebuilds; `json.dumps(sort_keys=True)` makes output deterministic (good for hashing, diffing, golden tests). Enums are stored as `.value`; tuples become lists and back; the vector in `Embedding` is a tuple of floats. Round-trip equality (`from_dict(to_dict(x)) == x`) is a property test worth having.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Value object | immutable dataclass, record, DTO | Close | Value object = equality by fields + immutability; DTO only means "carries data". |
| Content hash id | content-addressed id, fingerprint, digest | Same idea | Id derived from the content, so identical content gets an identical id (as in git or Docker layers). |
| uuid4 / random id | surrogate id, request id | No | Random: unique per creation even when content repeats; use for events, not for content. |
| Dataclass vs pydantic model | data class, BaseModel, schema class | No | Dataclass: no runtime type checks; pydantic validates and coerces. |
| Serialisation | marshalling, to_dict, dump | Close | Turning an object into JSON-safe data; the reverse is deserialisation / parsing / `from_dict`. |
| Chunk / passage / node | segment, document fragment, node (LlamaIndex) | Mostly | Same unit of retrieval; frameworks add their own metadata and class names. |

## Run it
```bash
# from this step folder
HF_HUB_OFFLINE=1 python main.py
# or from the repo root
HF_HUB_OFFLINE=1 .venv/bin/python 20_llm_system_lld/02_domain_model/main.py
```


## Code, step by step
Three files are introduced here: `main.py` (the demo), then `ragkit/ids.py`, `ragkit/domain.py` and `ragkit/schemas.py`. Every block of all four is listed.

#### Walking `main.py`

### 1. Header docstring  (`main.py` lines 1–13)
```python
"""
GOAL:  Build the nouns of a RAG system (Document, Element, Chunk, Embedding, SearchHit, Citation, Answer) as
       immutable value objects; see content-hash ids, JSON round-trips, validation, and dataclass vs pydantic.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 20_llm_system_lld/02_domain_model/main.py
EXPECTED OUTPUT (shortened):
    doc_id=... content_hash=...   (doc_id stable, content_hash changes with the text)
    chunk ids with different position are equal: True
    round trip equal: True
    dataclass accepts position='3' silently: True | pydantic coerces to int 3
NEEDS: ragkit (stdlib + numpy + pydantic v2). No models, no network.
"""
```
- **In simple words:** What this file proves and how to run it.
- **Exactly:** Docstring with GOAL / RUN / EXPECTED OUTPUT / NEEDS.
- **Why it is here:** Orientation for the reader.

### 2. Imports  (`main.py` lines 14–24)
```python
import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pydantic import ValidationError

from ragkit.domain import Answer, Chunk, Citation, Document, Element, ElementKind, Embedding, SearchHit
from ragkit.ids import content_hash, new_id
from ragkit.schemas import ChunkModel
```
- **In simple words:** Load the tools and our domain classes.
- **Exactly:** `pydantic.ValidationError` is what pydantic raises; `FrozenInstanceError` is what frozen dataclasses raise; all domain classes come from `ragkit.domain`.
- **Why it is here:** Everything the demo needs, nothing more.

### 3. Same source -> same doc_id; changed text -> new content_hash  (`main.py` lines 26–32)
```python
# ---- 1. Ids: content-hash (deterministic) vs uuid (random) -----------------------------------------------
v1 = Document.from_text("refunds.md", "# Refunds\nWithin 9 days.", team="support")
v2 = Document.from_text("refunds.md", "# Refunds\nWithin 14 days.", team="support")
print(f"doc_id v1={v1.doc_id} v2={v2.doc_id} same source -> same id: {v1.doc_id == v2.doc_id}")
print(f"content_hash v1={v1.content_hash} v2={v2.content_hash} text changed -> hash changed: {v1.content_hash != v2.content_hash}")
r1, r2 = new_id("trace"), new_id("trace")                           # uuid4-based: different every run
print("random ids (events, traces): prefix", r1[:6], "| length", len(r1), "| two calls differ:", r1 != r2)
```
- **In simple words:** The document's name stays the same when you edit it; its fingerprint changes.
- **Exactly:** Output: `doc_id v1=c8fac47431246695 v2=c8fac47431246695 ... True`; `content_hash v1=c592d2192085d61f v2=7deea2a32cc2a502` differ. `new_id('trace')` gives `trace-` + 12 hex = length 18, different every call.
- **Why it is here:** The indexer compares `doc_id` (is it the same doc?) and `content_hash` (did it change?) - the basis of incremental re-indexing.

### 4. Chunk id ignores position  (`main.py` lines 34–37)
```python
# ---- 2. Chunk identity ignores position, so inserting a paragraph elsewhere does not change this id -------
a = Chunk.create(v1.doc_id, "Within 9 days.", "Refund window", position=0)
b = Chunk.create(v1.doc_id, "Within 9 days.", "Refund window", position=7)
print("chunk ids with different position are equal:", a.chunk_id == b.chunk_id, "->", a.chunk_id)
```
- **In simple words:** Moving a paragraph does not change what it *is*.
- **Exactly:** Two `Chunk.create` calls with `position=0` and `position=7` give the same id `116f3b4893883d7c`, because the hash covers `(doc_id, section, text)` only.
- **Why it is here:** Otherwise inserting one paragraph would re-embed every chunk after it.

### 5. Try to mutate: field assignment and metadata  (`main.py` lines 39–50)
```python
# ---- 3. Immutability: frozen, slots, and read-only metadata ------------------------------------------------
def mutate_metadata():
    v1.metadata["team"] = "sales"                                   # metadata is a read-only mappingproxy


for label, action in [("assign field", lambda: setattr(a, "text", "x")), ("mutate metadata", mutate_metadata)]:
    try:
        action()
    except (FrozenInstanceError, TypeError) as e:
        print(f"{label:16s}-> {type(e).__name__}: {e}")
twin = Chunk.create(v1.doc_id, "Within 9 days.", "Refund window", position=0)
print("equal by value:", a == twin, "| hashable, set of 2 equal chunks has", len({a, twin}), "member | a == b (position differs):", a == b)
```
- **In simple words:** Both attempts are refused.
- **Exactly:** `setattr(a, 'text', 'x')` -> `FrozenInstanceError: cannot assign to field 'text'`; `v1.metadata['team'] = ...` -> `TypeError: 'mappingproxy' object does not support item assignment`. `twin == a` is True and a set of the two has 1 member, while `a == b` is False because `position` differs (position is part of equality, not of the id).
- **Why it is here:** Shows frozen (shallow) plus our read-only metadata view. Objects can be shared and cached safely.

### 6. Invalid objects cannot be created  (`main.py` lines 52–58)
```python
# ---- 4. Validation in __post_init__: an invalid object cannot exist ----------------------------------------
for label, make in [("empty document", lambda: Document.from_text("x.md", "   ")),
                    ("refusal with citations", lambda: Answer("no", (Citation(1, "c"),), refused=True))]:
    try:
        make()
    except ValueError as e:
        print(f"{label:23s}-> ValueError: {e}")
```
- **In simple words:** A blank document or a refusal with citations is rejected the moment you try to build it.
- **Exactly:** `Document.from_text('x.md', '   ')` -> `ValueError: Document 'x.md' is empty`; `Answer(..., refused=True)` with citations -> `ValueError: a refusal cannot carry citations`.
- **Why it is here:** Business rules live in one place (the constructor) and cannot be bypassed.

### 7. Build an Answer, a hit and elements  (`main.py` lines 60–64)
```python
# ---- 5. Serialisation: to_dict / from_dict / JSON ------------------------------------------------------------
els = [Element(ElementKind.HEADING, "Refunds", 1, 0), Element(ElementKind.PARAGRAPH, "Within 9 days.", 0, 1)]
hit = SearchHit(a, 0.8731, "dense", 1)
ans = Answer("Refunds take 9 days [1].", (Citation(1, a.chunk_id),), context_chunk_ids=(a.chunk_id,), index_version="v-1a2b")
text = ans.to_json()
```
- **In simple words:** Make some objects to save.
- **Exactly:** `Answer.to_json()` produces sorted-key JSON; `SearchHit(a, 0.8731, 'dense', 1)`; `Element(ElementKind.HEADING, 'Refunds', 1, 0)`.
- **Why it is here:** Inputs for the round-trip proof below.

### 8. Round-trip: object -> JSON -> object  (`main.py` lines 65–67)
```python
print("answer as JSON:", text[:110], "...")
print("round trip equal:", Answer.from_json(text) == ans, "| hit round trip:", SearchHit.from_dict(hit.to_dict()) == hit,
      "| elements:", [Element.from_dict(e.to_dict()) for e in els] == els)
```
- **In simple words:** Save and load, and check nothing was lost.
- **Exactly:** `Answer.from_json(text) == ans` -> `True`; same for `SearchHit` and a list of `Element`s. The JSON starts `{"citations": [{"chunk_id": "116f...", "marker": 1, "supported": true}], ...`.
- **Why it is here:** Round-trip equality is the test that proves `to_dict` and `from_dict` agree.

### 9. An Embedding remembers its model  (`main.py` lines 68–69)
```python
emb = Embedding.from_array(a.chunk_id, "hashing-64", __import__("numpy").arange(4, dtype="float32"))
print("embedding dim:", emb.dim, "| model recorded:", emb.model, "| as_array dtype:", emb.as_array().dtype)
```
- **In simple words:** A vector is meaningless without knowing which model produced it.
- **Exactly:** `Embedding.from_array(id, 'hashing-64', np.arange(4))` -> `dim 4`, `model hashing-64`, `as_array().dtype float32`.
- **Why it is here:** Prevents comparing a MiniLM vector to an e5 vector (module 14 step 03: no error, meaningless number).

### 10. Dataclass accepts a wrong type; pydantic coerces or rejects  (`main.py` lines 71–80)
```python
# ---- 6. dataclass vs pydantic at a trust boundary ---------------------------------------------------------------
bad = Chunk("id", "doc", "text", position="3")                      # dataclass: type hints are NOT enforced
print("dataclass accepts position='3' silently:", bad.position == "3", "| type:", type(bad.position).__name__)
print("pydantic coerces:", ChunkModel(chunk_id="id", doc_id="doc", text="t", position="3").position)
try:
    ChunkModel(chunk_id="", doc_id="doc", text="t", position=-1)
except ValidationError as e:
    print("pydantic error count:", e.error_count(), "|", [(x["loc"][0], x["type"]) for x in e.errors()])
print("pydantic JSON schema keys:", sorted(ChunkModel.model_json_schema()["properties"]))
print("content_hash demo:", content_hash("ab", "c"), "!=", content_hash("a", "bc"))
```
- **In simple words:** The dataclass trusts you; pydantic checks you.
- **Exactly:** `Chunk(..., position='3')` is accepted and `position` stays the `str` `'3'`. `ChunkModel(position='3').position` -> `3` (int). `ChunkModel(chunk_id='', position=-1)` -> `ValidationError` with `error_count() == 2`: `chunk_id: string_too_short`, `position: greater_than_equal`. `model_json_schema()` lists the 5 properties.
- **Why it is here:** Choose per boundary: pydantic where data comes from outside, dataclass where we build it ourselves.

### 11. Exercises  (`main.py` lines 82–84)
```python
# TRY IT: change the chunk text by one character and print chunk_id again: how many hex chars change?
# TRY IT: make Chunk.create hash `position` too, re-run section 2: the ids differ. What would that cost during re-indexing?
# BREAK IT: remove `slots=True` from Chunk in domain.py: section 3 still raises for `text`, but what happens to c.txt = 1?
```
- **In simple words:** Things to try yourself.
- **Exactly:** Comment lines.
- **Why it is here:** Active practice; see exercises.md.


#### Walking `ragkit/ids.py`

### 12. ids.py header and imports  (`ragkit/ids.py` lines 1–11)
```python
"""ids.py -- how entities get their identity (step 02).

Two kinds of id:
  content_hash : deterministic. Same input -> same id. Used for chunks so re-ingesting an unchanged
                 document produces the SAME ids (idempotent upsert).
  new_id       : random (uuid4). Used for things that are events, not content: traces, requests, feedback.
"""
from __future__ import annotations

import hashlib
import uuid
```
- **In simple words:** The two ways to name things.
- **Exactly:** Docstring explains `content_hash` (deterministic) vs `new_id` (random). Imports `hashlib` and `uuid` from the standard library.
- **Why it is here:** Choosing the id rule is a design decision; the docstring records it.

### 13. content_hash: sha256 of the joined parts  (`ragkit/ids.py` lines 14–17)
```python
def content_hash(*parts: str, length: int = 16) -> str:
    """sha256 over the parts, joined with an unambiguous separator; return the first `length` hex chars."""
    joined = "\x1f".join(parts)                      # \x1f (unit separator) so ("ab","c") != ("a","bc")
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:length]
```
- **In simple words:** Turn any text into a short fixed-size fingerprint.
- **Exactly:** `parts` joined with `\x1f`, UTF-8 encoded, `sha256(...).hexdigest()[:length]` with default length 16 hex chars (64 bits). `content_hash('ab','c') != content_hash('a','bc')` (printed: `9617c506357d3d46 != 624ce4e22aa197da`).
- **Why it is here:** Without the separator, different part lists could collapse to the same string and collide.

### 14. new_id: random id with a readable prefix  (`ragkit/ids.py` lines 20–22)
```python
def new_id(prefix: str = "") -> str:
    """Random 12-hex-char id with an optional readable prefix, e.g. 'trace-3f9a1c0b77de'."""
    return f"{prefix}{'-' if prefix else ''}{uuid.uuid4().hex[:12]}"
```
- **In simple words:** A fresh name for a one-off event, e.g. `trace-4679f204d9d2`.
- **Exactly:** `uuid.uuid4().hex[:12]` = 48 random bits; prefix + `-` + hex when a prefix is given.
- **Why it is here:** Traces and requests must be unique even if their content is identical.


#### Walking `ragkit/domain.py`

### 15. domain.py header and imports  (`ragkit/domain.py` lines 1–19)
```python
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
```
- **In simple words:** The rules for every class in this file, then the tools.
- **Exactly:** Docstring lists the four rules: frozen, slots, validate in `__post_init__`, `to_dict`/`from_dict`. Imports `MappingProxyType` (read-only dict view), `Enum`, numpy (for `Embedding.as_array`) and `content_hash`.
- **Why it is here:** The rules are the design; the classes only apply them.

### 16. _freeze: a read-only copy of a mapping  (`ragkit/domain.py` lines 22–23)
```python
def _freeze(meta: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(meta or {}))        # read-only view of a private copy
```
- **In simple words:** Copy the dictionary, then lock the copy.
- **Exactly:** `MappingProxyType(dict(meta or {}))`: `dict(...)` copies so the caller's dict can change later without affecting us; the proxy forbids item assignment. `None` becomes an empty mapping.
- **Why it is here:** `frozen=True` alone would still let `doc.metadata['x'] = 1` through.

### 17. ElementKind: the allowed kinds of element  (`ragkit/domain.py` lines 26–32)
```python
class ElementKind(str, Enum):
    """What a parsed piece of a document is. str-mixin so it JSON-serialises as its value."""
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    CODE = "code"
```
- **In simple words:** A fixed menu: heading, paragraph, list, table, code.
- **Exactly:** `class ElementKind(str, Enum)`; members compare equal to their string values (`'heading'`) and serialise as plain JSON strings.
- **Why it is here:** A typo such as `'headin'` fails on construction instead of silently matching nothing.

### 18. Document: fields, validation, freezing metadata  (`ragkit/domain.py` lines 35–45)
```python
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
```
- **In simple words:** A source document: id, where it came from, its text, and extra info.
- **Exactly:** `@dataclass(frozen=True, slots=True)`. `metadata` has `hash=False, compare=False` so a dict does not break hashing or equality. `__post_init__` rejects empty text (`ValueError`) and uses `object.__setattr__` to replace `metadata` with the frozen copy (the only way to assign inside a frozen class).
- **Why it is here:** Invalid empty documents never enter the pipeline; metadata cannot be changed behind our back.

### 19. Document.from_text and content_hash  (`ragkit/domain.py` lines 47–54)
```python
    @classmethod
    def from_text(cls, source: str, text: str, **metadata) -> "Document":
        return cls(doc_id=content_hash(source), source=source, text=text, metadata=metadata)

    @property
    def content_hash(self) -> str:
        """Changes when the text changes; doc_id does not. This pair drives incremental re-indexing."""
        return content_hash(self.text)
```
- **In simple words:** A convenient constructor, and a fingerprint of the text.
- **Exactly:** `from_text(source, text, **metadata)` sets `doc_id = content_hash(source)`: stable per source path. The `content_hash` property = `content_hash(self.text)`: changes whenever the text changes.
- **Why it is here:** `doc_id` answers "which document?", `content_hash` answers "has it changed?".

### 20. Document to_dict / from_dict  (`ragkit/domain.py` lines 56–61)
```python
    def to_dict(self) -> dict:
        return {"doc_id": self.doc_id, "source": self.source, "text": self.text, "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Document":
        return cls(d["doc_id"], d["source"], d["text"], d.get("metadata", {}))
```
- **In simple words:** Convert to a plain dict and back.
- **Exactly:** `to_dict` returns `{'doc_id','source','text','metadata': dict(...)}` (the proxy becomes a normal dict so `json.dumps` works). `from_dict` is a classmethod calling the constructor, so validation runs again on load.
- **Why it is here:** Loading stored data re-checks the rules; corrupt records fail loudly.

### 21. Element: one parsed piece of a document  (`ragkit/domain.py` lines 64–77)
```python
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
```
- **In simple words:** A heading or a paragraph with its order in the document.
- **Exactly:** Fields `kind: ElementKind`, `text`, `level` (1 for `#`, 2 for `##`), `position`. Includes `to_dict`/`from_dict`; the enum is stored as `kind.value` and rebuilt with `ElementKind(d['kind'])`.
- **Why it is here:** Chunkers need structure (where a section starts) to split sensibly.

### 22. Chunk: the unit of retrieval  (`ragkit/domain.py` lines 80–92)
```python
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
```
- **In simple words:** A passage of text with the section it came from.
- **Exactly:** Fields `chunk_id`, `doc_id`, `text`, `section`, `position`, `metadata`. `__post_init__` rejects empty text and freezes metadata.
- **Why it is here:** The chunk is both the unit of search and the unit of prompt context, so it is the most important type.

### 23. Chunk.create: the id rule  (`ragkit/domain.py` lines 94–98)
```python
    @classmethod
    def create(cls, doc_id: str, text: str, section: str = "", position: int = 0, **metadata) -> "Chunk":
        """chunk_id = hash(doc_id, section, text): NOT the position, so inserting a paragraph elsewhere
        does not change the ids of the chunks around it."""
        return cls(content_hash(doc_id, section, text), doc_id, text, section, position, metadata)
```
- **In simple words:** Build a chunk and let the code compute its id.
- **Exactly:** `chunk_id = content_hash(doc_id, section, text)`; position is stored but not hashed. Extra keyword arguments (e.g. `source=...`) become metadata.
- **Why it is here:** Identical text in the same section of the same document always gets the same id -> idempotent upsert.

### 24. Chunk to_dict / from_dict  (`ragkit/domain.py` lines 100–107)
```python
    def to_dict(self) -> dict:
        return {"chunk_id": self.chunk_id, "doc_id": self.doc_id, "text": self.text, "section": self.section,
                "position": self.position, "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Chunk":
        return cls(d["chunk_id"], d["doc_id"], d["text"], d.get("section", ""), d.get("position", 0),
                   d.get("metadata", {}))
```
- **In simple words:** Plain-dict conversion for storage (the Qdrant adapter stores exactly this as the payload).
- **Exactly:** Six keys; `metadata` converted to `dict`. `from_dict` uses `.get` with defaults for optional fields.
- **Why it is here:** One serialisation format shared by JSON files, Qdrant payloads and logs.

### 25. Embedding: vector plus the model that made it  (`ragkit/domain.py` lines 110–118)
```python
@dataclass(frozen=True, slots=True)
class Embedding:
    chunk_id: str
    model: str                                       # which model made it: vectors from different models are NOT comparable
    vector: tuple[float, ...]

    def __post_init__(self):
        if not self.vector:
            raise ValueError("empty vector")
```
- **In simple words:** A list of numbers with a label saying which model produced it.
- **Exactly:** Fields `chunk_id`, `model`, `vector: tuple[float, ...]` (tuple = immutable). `__post_init__` rejects an empty vector.
- **Why it is here:** Mixing vectors from different models gives silent nonsense scores.

### 26. Embedding helpers: from_array, dim, as_array, dict  (`ragkit/domain.py` lines 120–136)
```python
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
```
- **In simple words:** Convert between numpy and the immutable tuple.
- **Exactly:** `from_array` turns a `(dim,)` ndarray into a tuple of Python floats; `dim` is `len(vector)`; `as_array()` returns `np.float32` array for fast math; `to_dict`/`from_dict` use a list.
- **Why it is here:** numpy for computation, tuples for safe storage.

### 27. SearchHit: chunk + score + origin + rank  (`ragkit/domain.py` lines 139–151)
```python
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
```
- **In simple words:** One search result.
- **Exactly:** `source` is `'dense'`, `'sparse'`, `'hybrid'` or `'rerank'`; `rank` is 1-based. Nested `Chunk.to_dict()` inside `to_dict`.
- **Why it is here:** Scores are only comparable within one `source`; carrying it prevents mixing cosine with BM25.

### 28. Citation: [n] -> chunk id, plus support flag  (`ragkit/domain.py` lines 154–165)
```python
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
```
- **In simple words:** The link from a number in the answer to the chunk it points at.
- **Exactly:** `marker` (the n), `chunk_id`, `supported` (set by the verifier in step 04).
- **Why it is here:** Makes answers checkable by code and renderable as source links in a UI.

### 29. Answer: the service's output  (`ragkit/domain.py` lines 168–181)
```python
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
```
- **In simple words:** Text, citations, whether we refused, and which index produced it.
- **Exactly:** Fields `text`, `citations: tuple[Citation,...]`, `refused`, `refusal_reason`, `context_chunk_ids`, `index_version`, `meta`. `__post_init__` freezes `meta` and enforces "a refusal cannot carry citations".
- **Why it is here:** Everything needed to log, cache (step 07), evaluate (step 10) and explain an answer in one value.

### 30. Answer serialisation and JSON helpers  (`ragkit/domain.py` lines 183–199)
```python
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
```
- **In simple words:** Save/load an answer, including nested citations.
- **Exactly:** `to_dict` nests `Citation.to_dict()`; `to_json` uses `sort_keys=True` so equal answers give identical strings; `from_json` = `from_dict(json.loads(s))`.
- **Why it is here:** Deterministic JSON is easy to diff, hash and snapshot-test.


#### Walking `ragkit/schemas.py`

### 31. schemas.py header and imports  (`ragkit/schemas.py` lines 1–9)
```python
"""schemas.py -- pydantic v2 models for data that crosses a TRUST boundary (HTTP bodies, LLM JSON output).

Rule of thumb used in ragkit:
  dataclass (domain.py)  -> objects we build ourselves, already trusted, hot paths
  pydantic  (here)       -> text from outside: parse + coerce + validate + good error messages + JSON Schema
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
```
- **In simple words:** Pydantic is for data from outside.
- **Exactly:** Docstring states the rule: dataclass inside, pydantic at trust boundaries. Imports `BaseModel`, `ConfigDict`, `Field` from pydantic v2.
- **Why it is here:** Keeps the two tools from being used interchangeably by habit.

### 32. ChunkModel: same fields as Chunk, validated  (`ragkit/schemas.py` lines 12–19)
```python
class ChunkModel(BaseModel):
    """Same fields as domain.Chunk, but validated and coerced on construction."""
    model_config = ConfigDict(frozen=True)
    chunk_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    section: str = ""
    position: int = Field(default=0, ge=0)
```
- **In simple words:** A strict twin of `Chunk` for untrusted input.
- **Exactly:** `ConfigDict(frozen=True)`; `Field(min_length=1)` rejects empty strings; `position: int = Field(default=0, ge=0)` coerces `'3'` to `3` and rejects negatives.
- **Why it is here:** Shows the difference in main.py section 6: pydantic reports both errors at once.

### 33. AskRequest: what POST /ask would accept  (`ragkit/schemas.py` lines 22–25)
```python
class AskRequest(BaseModel):
    """What an HTTP endpoint `POST /ask` would accept."""
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=20)
```
- **In simple words:** Rules for a user's question.
- **Exactly:** `question` 3-2000 chars, `top_k` between 1 and 20 (default 3). FastAPI (module 09) uses such a model directly as a request body.
- **Why it is here:** Bounds protect the service from empty or megabyte-sized questions.

### 34. StructuredAnswer: JSON we ask the LLM to produce  (`ragkit/schemas.py` lines 28–32)
```python
class StructuredAnswer(BaseModel):
    """What we ask the LLM to return as JSON (step 08)."""
    answer: str
    citations: list[int] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
```
- **In simple words:** The shape of the reply we demand from the model.
- **Exactly:** `answer: str`, `citations: list[int]`, `confidence: float` between 0 and 1. Used in step 08 for the repair loop.
- **Why it is here:** LLM output is untrusted text: validate it before use.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `object.__setattr__(self, 'metadata', ...)` | Bypasses the frozen guard; allowed inside `__post_init__` to normalise a field |
| `field(default_factory=dict, hash=False, compare=False)` | Mutable default created per instance; excluded from `__hash__` and `__eq__` |
| `@classmethod def create(cls, ...)` | Alternative constructor that returns `cls(...)` |
| `@property` | Read a computed value as an attribute (`doc.content_hash`) |
| `tuple[float, ...]` | A tuple of any length containing floats |
| `MappingProxyType(d)` | Read-only dict view |
| `str | None` | Union type: a string or None (Python 3.10+) |

## Libraries & functions
| Name | What it does here |
|---|---|
| `hashlib.sha256` | Cryptographic hash; we keep the first 16 hex chars |
| `uuid.uuid4` | Random 128-bit id |
| `dataclasses` | `dataclass`, `field`, `FrozenInstanceError` |
| `types.MappingProxyType` | Read-only mapping wrapper |
| `pydantic` v2 | `BaseModel`, `Field`, `ConfigDict`, `ValidationError`, `model_json_schema()` |
| `numpy` | `np.asarray(..., dtype=float32)` for `Embedding.as_array` |

## Execution trace (real run)
```text
doc_id v1=c8fac47431246695 v2=c8fac47431246695 same source -> same id: True
content_hash v1=c592d2192085d61f v2=7deea2a32cc2a502 text changed -> hash changed: True
random ids (events, traces): prefix trace- | length 18 | two calls differ: True
chunk ids with different position are equal: True -> 116f3b4893883d7c
assign field    -> FrozenInstanceError: cannot assign to field 'text'
mutate metadata -> TypeError: 'mappingproxy' object does not support item assignment
equal by value: True | hashable, set of 2 equal chunks has 1 member | a == b (position differs): False
empty document         -> ValueError: Document 'x.md' is empty
refusal with citations -> ValueError: a refusal cannot carry citations
answer as JSON: {"citations": [{"chunk_id": "116f3b4893883d7c", "marker": 1, "supported": true}], "context_chunk_ids": ["116f3 ...
round trip equal: True | hit round trip: True | elements: True
embedding dim: 4 | model recorded: hashing-64 | as_array dtype: float32
dataclass accepts position='3' silently: True | type: str
pydantic coerces: 3
pydantic error count: 2 | [('chunk_id', 'string_too_short'), ('position', 'greater_than_equal')]
pydantic JSON schema keys: ['chunk_id', 'doc_id', 'position', 'section', 'text']
content_hash demo: 9617c506357d3d46 != 624ce4e22aa197da
```
- `doc_id` is identical for v1 and v2 (hash of the source name) while `content_hash` differs: the pair the indexer needs.
- The two ids with different positions are equal (`116f...`): position is not part of identity.
- Two mutation attempts are refused with two different exception types (`FrozenInstanceError`, `TypeError`).
- `equal by value: True ... set of 2 equal chunks has 1 member`: frozen value objects behave like numbers.
- Two validation failures raise `ValueError` with readable messages.
- `round trip equal: True` three times: serialisation is lossless.
- The dataclass kept `'3'` as a `str`; pydantic returned `3`; pydantic reported 2 errors in one exception.

## Experiments (change ONE thing)
Each snippet runs from this step folder (`cd` into it first). Outputs below are from real runs.

### Change ONE character of the text: how different is the id?
```python
from ragkit.domain import Chunk
a = Chunk.create("doc1", "Refunds take 9 days.", "Refund window")
b = Chunk.create("doc1", "Refunds take 8 days.", "Refund window")
print(a.chunk_id, b.chunk_id, "| differing hex positions:", sum(x != y for x, y in zip(a.chunk_id, b.chunk_id)), "of", len(a.chunk_id))
```
Observed:
```text
e4fce38e0595b191 cba6e9213bfd8a20 | differing hex positions: 15 of 16
```
15 of 16 hex characters differ after a one-character edit: a hash is a fingerprint, not a similarity measure. Near-duplicate detection needs other tools (MinHash, embeddings).

### Put the position INTO the id: what happens when one paragraph is inserted at the top?
```python
from ragkit.ids import content_hash
paras = ["alpha text", "beta text", "gamma text"]
before_pos = {content_hash("d", p, str(i)) for i, p in enumerate(paras)}
after = ["NEW intro"] + paras
after_pos = {content_hash("d", p, str(i)) for i, p in enumerate(after)}
before_c = {content_hash("d", p) for p in paras}
after_c = {content_hash("d", p) for p in after}
print("id includes position : unchanged chunks keep their id:", len(before_pos & after_pos), "of", len(paras))
print("id is content-only   : unchanged chunks keep their id:", len(before_c & after_c), "of", len(paras))
```
Observed:
```text
id includes position : unchanged chunks keep their id: 0 of 3
id is content-only   : unchanged chunks keep their id: 3 of 3
```
With position inside the hash, **0 of 3** unchanged chunks keep their id after one insertion at the top, so the indexer would re-embed everything; with a content-only id **3 of 3** survive. This is a real design decision with a cost in embedding calls and money.

### Memory: slots vs no slots
```python
import sys
from dataclasses import dataclass
@dataclass(frozen=True)
class A:
    chunk_id: str; doc_id: str; text: str
@dataclass(frozen=True, slots=True)
class B:
    chunk_id: str; doc_id: str; text: str
a, b = A("x", "y", "z"), B("x", "y", "z")
print("no slots:", sys.getsizeof(a) + sys.getsizeof(a.__dict__), "bytes (object + __dict__) | slots:", sys.getsizeof(b), "bytes")
```
Observed:
```text
no slots: 344 bytes (object + __dict__) | slots: 56 bytes
```
344 bytes (object plus its `__dict__`) vs 56 bytes with slots in this tiny example. A million chunks: roughly 344 MB vs 56 MB of object overhead before the text itself.

### frozen is SHALLOW: a list field can still be mutated
```python
from dataclasses import dataclass
@dataclass(frozen=True)
class D:
    tags: list
d = D(["a"])
d.tags.append("b")
print(d)
try:
    hash(d)
except TypeError as e:
    print("hash fails:", e)
```
Observed:
```text
D(tags=['a', 'b'])
hash fails: unhashable type: 'list'
```
`frozen=True` stops re-assignment of `tags` but not `d.tags.append(...)`, and a frozen dataclass holding a list cannot be hashed. That is why ragkit stores tuples and `MappingProxyType`.

## Debugging practice
**BREAK IT:**
```python
import sys; sys.path.insert(0, '..')

from ragkit.domain import Chunk
c = Chunk.create("doc", "text")
c.text = "changed"
```
Real output:
```text
Traceback (most recent call last):
  File "<string>", line 5, in <module>
  File "<string>", line 4, in __setattr__
dataclasses.FrozenInstanceError: cannot assign to field 'text'
```
Read it from the bottom: `FrozenInstanceError: cannot assign to field 'text'`. The frame above shows the generated `__setattr__` of the dataclass - code you never wrote, produced by `frozen=True`. Fix: create a new object (`dataclasses.replace(c, text='changed')`) instead of editing. Note that `replace` on a `Chunk` keeps the OLD `chunk_id`; build with `Chunk.create` when the text really changes.

## Rebuild checklist
- [ ] Write `content_hash` with the `\x1f` separator and explain why it is there.
- [ ] Write a frozen+slots `Chunk` with `__post_init__` validation and a read-only metadata.
- [ ] Write `to_dict` / `from_dict` for `Answer` and a round-trip test.
- [ ] Explain, with the insert-a-paragraph example, why position is not in the chunk id.
- [ ] State three differences between a dataclass and a pydantic model, and when you use each.

## The new problem (cliffhanger)
We can describe the nouns. Who creates chunks, in what order, and what stops an unchanged document from being embedded again every night? [Step 03](../03_ingestion_pipeline/WALKTHROUGH.md) builds the offline ingestion pipeline.

> **Say it aloud (interview box):** "I model the domain as frozen, slotted value objects with validation in `__post_init__` and to_dict/from_dict. Chunk ids are a content hash of document, section and text - not position and not random - so unchanged content keeps its id and re-indexing is idempotent. I use pydantic only at trust boundaries like HTTP bodies and LLM JSON."
