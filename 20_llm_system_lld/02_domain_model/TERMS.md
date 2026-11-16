# Terms — The domain model: Document, Element, Chunk, Embedding, SearchHit, Citation, Answer

Folder: `20_llm_system_lld/02_domain_model` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [20 README](../README.md)

**Terms in this folder:** [Content hash (content-addressed id)](#content-hash-content-addressed-id--concept) · [uuid4 (random id)](#uuid4-random-id--concept) · [Document (doc_id, content_hash)](#document-doc_id-content_hash--concept) · [Element](#element--concept) · [Chunk](#chunk--concept) · [Embedding (vector + model name)](#embedding-vector--model-name--concept) · [SearchHit](#searchhit--concept) · [Citation and Answer](#citation-and-answer--concept) · [Immutability (frozen, slots, MappingProxyType)](#immutability-frozen-slots-mappingproxytype--concept) · [__post_init__ validation](#__post_init__-validation--syntax) · [Serialisation (to_dict, from_dict, JSON round trip)](#serialisation-to_dict-from_dict-json-round-trip--concept) · [pydantic v2 (BaseModel, Field, ValidationError)](#pydantic-v2-basemodel-field-validationerror--library) · [FrozenInstanceError](#frozeninstanceerror--error)

---

## Content hash (content-addressed id)  `(concept)`
**1. Definition:** An id computed by hashing the content (here sha256 of doc_id, section and text, first 16 hex characters).
**2. In plain words:** The same text always gets the same name, like a fingerprint.
**3. Example to quote:** > "I derive chunk ids from content, not position, so re-indexing unchanged text is a no-op."
```python
content_hash(doc_id, section, text)  # 16 hex chars
```
**Seen in:** `ragkit/ids.py`

## uuid4 (random id)  `(concept)`
**1. Definition:** A randomly generated 128-bit identifier; collision chance is negligible.
**2. In plain words:** A ticket number: unique even if two tickets say the same thing.
**3. Example to quote:** > "I use random ids for traces and requests, content hashes for chunks."
```python
uuid.uuid4().hex[:12]
```
**Seen in:** `ragkit/ids.py`

## Document (doc_id, content_hash)  `(concept)`
**1. Definition:** A source with a stable `doc_id` (hash of its path) and a `content_hash` that changes with its text.
**2. In plain words:** The book's ISBN versus a checksum of its pages.
**3. Example to quote:** > "The pair lets an indexer decide 'same document?' and 'changed?'."
**Seen in:** `ragkit/domain.py`

## Element  `(concept)`
**1. Definition:** One structural piece of a parsed document: heading, paragraph, list, table or code, with level and position.
**2. In plain words:** A paragraph or a heading, before chunking.
**3. Example to quote:** > "Chunkers should work on elements so they never cut a heading from its section."
**Seen in:** `ragkit/domain.py`

## Chunk  `(concept)`
**1. Definition:** The unit of retrieval and of prompt context, with id, doc id, text, section, position and metadata.
**2. In plain words:** One 'page' handed to the model.
**3. Example to quote:** > "A chunk id ignores position so inserting a paragraph does not re-embed its neighbours."
```python
Chunk.create(doc_id, text, section, position)
```
**Seen in:** `ragkit/domain.py`

## Embedding (vector + model name)  `(concept)`
**1. Definition:** A tuple of floats plus the name of the model that produced it.
**2. In plain words:** A coordinate with a label for which map it belongs to.
**3. Example to quote:** > "Vectors from different models are not comparable, so the model name travels with the vector."
**Seen in:** `ragkit/domain.py`

## SearchHit  `(concept)`
**1. Definition:** A chunk with a score, the stage that scored it (dense, sparse, hybrid, rerank) and a 1-based rank.
**2. In plain words:** A row in a result list.
**3. Example to quote:** > "Scores from different stages have different scales; the source tag prevents mixing them."
**Seen in:** `ragkit/domain.py`

## Citation and Answer  `(concept)`
**1. Definition:** A citation links a `[n]` marker to a chunk id with a support flag; an Answer bundles text, citations, refusal, context ids and index version.
**2. In plain words:** The reply plus its footnotes.
**3. Example to quote:** > "A refusal cannot carry citations - enforced in the constructor."
**Seen in:** `ragkit/domain.py`

## Immutability (frozen, slots, MappingProxyType)  `(concept)`
**1. Definition:** Objects whose fields cannot change after creation; frozen is shallow, so nested containers are made read-only too.
**2. In plain words:** A laminated card.
**3. Example to quote:** > "Shared, cached objects must be immutable or one caller can corrupt another."
```python
MappingProxyType(dict(meta))
```
**Seen in:** `ragkit/domain.py`

## __post_init__ validation  `(syntax)`
**1. Definition:** A dataclass hook that runs right after `__init__`; used to check invariants and normalise fields.
**2. In plain words:** A bouncer at the door.
**3. Example to quote:** > "Validate at construction so an invalid object can never exist."
```python
def __post_init__(self):
    if not self.text.strip(): raise ValueError(...)
```
**Seen in:** `ragkit/domain.py`

## Serialisation (to_dict, from_dict, JSON round trip)  `(concept)`
**1. Definition:** Converting objects to JSON-safe data and back; the property to test is from_dict(to_dict(x)) == x.
**2. In plain words:** Saving a game and loading it again.
**3. Example to quote:** > "I use sort_keys=True so equal objects produce identical JSON."
```python
Answer.from_json(ans.to_json()) == ans
```
**Seen in:** `main.py`

## pydantic v2 (BaseModel, Field, ValidationError)  `(library)`
**1. Definition:** A library that validates and coerces data by type hints and generates JSON Schema; v2 has a Rust core.
**2. In plain words:** A strict form that corrects '3' to 3 or rejects it.
**3. Example to quote:** > "Pydantic at the edge, plain value objects inside."
```python
ChunkModel(chunk_id='', doc_id='d', text='t', position=-1)  # ValidationError, 2 errors
```
**Seen in:** `ragkit/schemas.py`

## FrozenInstanceError  `(error)`
**1. Definition:** Subclass of AttributeError raised when assigning to a field of a frozen dataclass.
**2. In plain words:** 'You are not allowed to edit this.'
**3. Example to quote:** > "Fix by creating a new object, e.g. dataclasses.replace."
```python
c.text = 'x'  # FrozenInstanceError: cannot assign to field 'text'
```
**Seen in:** `WALKTHROUGH.md`, Debugging practice
