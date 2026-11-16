# Step 11 - Qdrant fundamentals: the same job in a real vector database

## The story so far
In [step 04](../04_vector_index_and_search/WALKTHROUGH.md) you wrote a flat vector index by hand (upsert, delete, filter, save, load). [Step 10](../10_advanced_rag_and_vector_dbs/WALKTHROUGH.md) listed the real tools (FAISS, Chroma, pgvector, Qdrant) as read-only reference code. This step runs one of them for real: Qdrant, with the official Python client, offline, on a local folder.

## Goal
Do everything step 04 did by hand, but in Qdrant: create a collection, add points with payloads, make re-ingestion idempotent, filter by metadata, search with `query_points`, delete by filter, scroll, retrieve by id, and reopen from disk. The knowledge base is the same 36 chunks of module 14.

## Run it
```bash
cd 14_rag_and_retrieval/11_qdrant_fundamentals
HF_HUB_OFFLINE=1 ../../.venv/bin/python main.py      # about 10 s, offline, no server needed
```
Real output of the run (the two warning lines from sentence-transformers and qdrant-client are omitted):
```
A/B) collection 'kb' created: size=384 distance=Cosine
C) 36 points upserted; upsert again -> still 36 (uuid5 ids are stable)
   id='rule_0' rejected: ValueError: Point id rule_0 is not a valid UUID
E) top-1 for 'How long does delivery take?' -> shipping_and_delivery#Tracking an order
   no filter, limit=3:
      0.471  shipping_and_delivery#Tracking an order
      0.459  shipping_and_delivery#Delivery times
      0.358  customer_support#Escalation
   must doc_id == refunds_and_returns:
      0.232  refunds_and_returns#Refund window
      0.198  refunds_and_returns#How to request a refund
   must doc_id in [loyalty_program, customer_support]:
      0.358  customer_support#Escalation
      0.328  customer_support#Response times
   n_words <= 30 AND doc_id != shipping:
      0.328  customer_support#Response times
      0.279  wholesale_and_b2b#Custom blends
   score_threshold=0.45 for an unanswerable question -> 0 results (an empty list is a refusal signal, step 08)
F) retrieve by id -> shipping_and_delivery#Delivery times
   scroll limit=10 -> 10 points, next_page offset is set
   delete doc_id == privacy_and_data -> 32 points left (how you remove a document that was taken down)
   reopened from disk -> 32 points
```
Note: the nearest chunk to "How long does delivery take?" is `Tracking an order` (0.471), only just ahead of `Delivery times` (0.459). This is the same small-embedding-model weakness you saw in steps 03 and 04; a reranker (step 07) fixes the order.

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Collection vs index vs table | one vector space | Related | A Qdrant collection holds points of one vector size and distance; like a table with a vector column and an ANN index |
| Point vs record vs document vs row | id + vector + payload | Related | A point = id, vector, payload (metadata); what other tools call a document or row |
| Payload vs metadata vs attributes | extra fields | Same idea | JSON stored next to the vector and usable in filters |
| `query_points` vs `search` | search API | No | `query_points` is the current API; `search` was deprecated and removed in newer clients |
| `:memory:` vs `path=` vs `url=` client | client modes | No | In-process and lost at exit; in-process saved to a folder; remote server over HTTP. Same API |
| Payload index vs vector index | two kinds of index | No | Payload index speeds up filters on a field; the vector (HNSW) index speeds up nearest-neighbour search |
| Upsert vs insert | write | No | Upsert inserts or replaces by id, so re-running ingestion never duplicates |

## Code, step by step

### 1. Docstring  (`main.py` lines 1–19)
```python
"""
GOAL:  Do in a real vector database (Qdrant) everything step 04 built by hand: a collection, points with
       payloads, idempotent upserts, metadata filters, scroll/retrieve/delete, persistence.
         A) the three client modes: ":memory:", local folder (path=...), server (url=...)
         B) create a collection whose size comes FROM THE MODEL, not from memory
         C) point ids: only unsigned ints or UUIDs. uuid5(chunk_id) makes re-ingestion idempotent
         D) payload indexes + filters: MatchValue, MatchAny, Range, must_not
         E) query_points (the current search API), score_threshold, with_payload
         F) upsert vs duplicate, delete by filter, scroll, retrieve by id, reopen from disk

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 14_rag_and_retrieval/11_qdrant_fundamentals/main.py
EXPECTED OUTPUT (shortened):
    A) collection 'kb' created: size=384 distance=Cosine
    C) 36 points upserted; upsert again -> still 36 (uuid5 ids are stable)
    E) top-1 for 'How long does delivery take?' -> shipping_and_delivery#Tracking an order (0.471; 'Delivery times' is 2nd, 0.459)
    F) reopened from disk -> 36 points
NEEDS: qdrant-client, sentence-transformers (cached all-MiniLM-L6-v2).  ~10 s.
"""
```
- **In simple words:** What this step covers, in six lettered parts.
- **Exactly:** A) client modes; B) collection size from the model; C) ids (uint or UUID) and idempotent upsert; D) payload indexes and filters; E) `query_points`; F) upsert, delete, scroll, retrieve, reopen from disk. Needs `qdrant-client` and cached MiniLM, no server.
- **Why it is here:** A map of the file before the details.

### 2. Standard-library imports  (`main.py` lines 21–25)
```python
import os
import sys
import tempfile
import uuid
from pathlib import Path
```
- **In simple words:** Load tools for environment, paths, temporary folders and UUIDs.
- **Exactly:** `os`, `sys`, `tempfile`, `uuid`, `Path`.
- **Why it is here:** `uuid` makes stable ids; `tempfile` keeps the demo from leaving files behind.

### 3. Offline flag and Qdrant/sentence-transformers imports  (`main.py` lines 27–30)
```python
os.environ.setdefault("HF_HUB_OFFLINE", "1")                      # cached models only

from qdrant_client import QdrantClient, models                    # noqa: E402
from sentence_transformers import SentenceTransformer             # noqa: E402
```
- **In simple words:** Work offline and load the two libraries.
- **Exactly:** `os.environ.setdefault("HF_HUB_OFFLINE", "1")` must run before `SentenceTransformer` is used so it never contacts the Hub. `models` holds all Qdrant request classes. The `# noqa: E402` comments allow imports after code.
- **Why it is here:** Without the flag a missing network leads to retries and warnings.

### 4. Shared chunks from common.py  (`main.py` lines 32–33)
```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EMBED_MODEL, build_chunks, chunk_for_embedding  # noqa: E402
```
- **In simple words:** Add the module folder to the import path and load our helpers.
- **Exactly:** `EMBED_MODEL`, `build_chunks`, `chunk_for_embedding` come from `common`.
- **Why it is here:** Same 36 chunks as every other step.

### 5. Chunks, model and vectors  (`main.py` lines 35–38)
```python
chunks = build_chunks()                                           # 36 section chunks of the Chai Kitchen KB
model = SentenceTransformer(EMBED_MODEL, device="cpu")
DIM = model.get_sentence_embedding_dimension()                    # 384 -- ask the model, never hard-code
vectors = model.encode([chunk_for_embedding(c) for c in chunks], normalize_embeddings=True)
```
- **In simple words:** Make the 36 vectors.
- **Exactly:** `SentenceTransformer(EMBED_MODEL, device="cpu")`. `get_sentence_embedding_dimension()` returns 384: ask the model, never hard-code. `encode(..., normalize_embeddings=True)` returns a `(36, 384)` array of unit vectors.
- **Why it is here:** A collection must be created with the exact size of the vectors.

### 6. embed helper  (`main.py` lines 41–42)
```python
def embed(text):
    return model.encode(text, normalize_embeddings=True).tolist()  # Qdrant wants a plain list of floats
```
- **In simple words:** Turn one text into a plain list of floats.
- **Exactly:** `model.encode(text, normalize_embeddings=True).tolist()`; Qdrant takes Python lists, not numpy arrays.
- **Why it is here:** Passing a numpy array can raise a type error.

### 7. A: create a client on a folder  (`main.py` lines 51–52)
```python
tmp = tempfile.TemporaryDirectory()
client = QdrantClient(path=tmp.name)
```
- **In simple words:** Start Qdrant inside this Python process, saving to a temporary folder.
- **Exactly:** `QdrantClient(path=tmp.name)` is local mode with persistence. Alternatives: `":memory:"` (gone at exit) or `url="http://localhost:6333"` (real server, `docker run -p 6333:6333 qdrant/qdrant`). The calling code is the same for all three.
- **Why it is here:** Develop locally, deploy against a server without changing code.

### 8. B: create the collection  (`main.py` lines 57–65)
```python
if client.collection_exists("kb"):                                # make the script safe to re-run
    client.delete_collection("kb")
client.create_collection(
    collection_name="kb",
    vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE),
)
info = client.get_collection("kb")
print(f"A/B) collection 'kb' created: size={info.config.params.vectors.size} "
      f"distance={info.config.params.vectors.distance.value}")
```
- **In simple words:** Make a collection that expects vectors of 384 numbers compared by cosine.
- **Exactly:** `collection_exists` plus `delete_collection` make the script re-runnable. `VectorParams(size=DIM, distance=Distance.COSINE)`: size and distance are fixed at creation. `get_collection` returns its config; real output `size=384 distance=Cosine`.
- **Why it is here:** A wrong size gives an error on the first upsert; the distance changes how scores are read (cosine: higher is closer).

### 9. C: deterministic point ids  (`main.py` lines 71–75)
```python
NAMESPACE = uuid.UUID("6f1c2b1e-3c1a-4c7e-9a52-2d1f0c9b7a10")    # any fixed UUID; keep it constant forever


def point_id(chunk_id):
    return str(uuid.uuid5(NAMESPACE, chunk_id))
```
- **In simple words:** Turn a readable chunk id into a UUID that never changes.
- **Exactly:** Qdrant ids must be an unsigned integer or a UUID. `uuid.uuid5(NAMESPACE, chunk_id)` is a name-based UUID: same namespace and name always give the same UUID. Keep the namespace constant forever.
- **Why it is here:** Deterministic ids make re-ingestion replace points instead of duplicating them.

### 10. C: build the points  (`main.py` lines 78–86)
```python
points = [
    models.PointStruct(
        id=point_id(c["chunk_id"]),
        vector=vectors[i].tolist(),
        payload={"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "section": c["section"],
                 "text": c["text"], "n_words": len(c["text"].split())},
    )
    for i, c in enumerate(chunks)
]
```
- **In simple words:** One point per chunk: id, vector and its facts.
- **Exactly:** `PointStruct(id=..., vector=list of 384 floats, payload={chunk_id, doc_id, section, text, n_words})`. Payload is JSON stored next to the vector; `n_words` is added for the range filter later.
- **Why it is here:** Payload is what you filter on and what you show as the citation.

### 11. C: upsert twice, count  (`main.py` lines 87–91)
```python
client.upsert(collection_name="kb", points=points, wait=True)    # wait=True: return after it is searchable
n1 = client.count("kb").count
client.upsert(collection_name="kb", points=points, wait=True)    # re-ingest the same chunks
n2 = client.count("kb").count
print(f"C) {n1} points upserted; upsert again -> still {n2} (uuid5 ids are stable)")
```
- **In simple words:** Write all points, then write them again, and count.
- **Exactly:** `upsert(..., wait=True)` returns after the points are searchable. `client.count("kb").count` is 36 both times (real output): the second upsert replaced the same ids.
- **Why it is here:** Idempotent ingestion lets you re-run a pipeline safely.

### 12. C: an invalid id is rejected  (`main.py` lines 92–95)
```python
try:
    client.upsert("kb", points=[models.PointStruct(id="rule_0", vector=vectors[0].tolist(), payload={})])
except Exception as e:                                            # noqa: BLE001 -- show the real message
    print(f"   id='rule_0' rejected: {type(e).__name__}: {str(e).splitlines()[0][:90]}")
```
- **In simple words:** Show what happens with a bad id.
- **Exactly:** `id="rule_0"` raises `ValueError: Point id rule_0 is not a valid UUID`. The `except Exception` is used on purpose to print the real message.
- **Why it is here:** Seeing the error once means you can recognise it later.

### 13. D: payload indexes  (`main.py` lines 102–103)
```python
client.create_payload_index("kb", field_name="doc_id", field_schema=models.PayloadSchemaType.KEYWORD)
client.create_payload_index("kb", field_name="n_words", field_schema=models.PayloadSchemaType.INTEGER)
```
- **In simple words:** Prepare fast filtering on two fields.
- **Exactly:** `create_payload_index(field_name, field_schema=KEYWORD or INTEGER)`. On a server, filtering on an un-indexed field scans every point; create indexes before bulk loading. Local mode accepts the call and warns that it ignores it.
- **Why it is here:** Without an index, filters get slower as the collection grows.

### 14. show helper  (`main.py` lines 106–109)
```python
def show(title, result):
    print(f"   {title}")
    for p in result.points:
        print(f"      {p.score:.3f}  {p.payload['chunk_id']}")
```
- **In simple words:** Print scores and chunk ids from a result.
- **Exactly:** `result.points` is a list of scored points; `p.score` is the cosine similarity; `p.payload` is the stored JSON.
- **Why it is here:** Keeps the later lines short.

### 15. E: unfiltered search  (`main.py` lines 116–119)
```python
q = "How long does delivery take?"
res = client.query_points("kb", query=embed(q), limit=3, with_payload=True)
print(f"E) top-1 for {q!r} -> {res.points[0].payload['chunk_id']}")
show("no filter, limit=3:", res)
```
- **In simple words:** Ask a question and print the top 3.
- **Exactly:** `query_points("kb", query=embed(q), limit=3, with_payload=True)`. Real top-1: `shipping_and_delivery#Tracking an order` 0.471; second `Delivery times` 0.459; third `customer_support#Escalation` 0.358.
- **Why it is here:** `query_points` is the current search API; `search()` was deprecated and later removed.

### 16. E: filter with one exact value  (`main.py` lines 121–122)
```python
only_refunds = models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value="refunds_and_returns"))])
show("must doc_id == refunds_and_returns:", client.query_points("kb", query=embed(q), query_filter=only_refunds, limit=2))
```
- **In simple words:** Only look inside the refunds document.
- **Exactly:** `Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=...))])`. Filtering happens inside the index search, not after, so you still get `limit` results (compare pre-filter vs post-filter in step 04). Real: `Refund window` 0.232 and `How to request a refund` 0.198.
- **Why it is here:** Low scores here are fine: the question is about delivery and we forced refunds.

### 17. E: filter with any of several values  (`main.py` lines 124–125)
```python
two_docs = models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=["loyalty_program", "customer_support"]))])
show("must doc_id in [loyalty_program, customer_support]:", client.query_points("kb", query=embed(q), query_filter=two_docs, limit=2))
```
- **In simple words:** Look in two documents.
- **Exactly:** `MatchAny(any=[...])` is "doc_id is one of". Real: `customer_support#Escalation` 0.358 then `Response times` 0.328.
- **Why it is here:** Typical for access control by group.

### 18. E: range plus exclusion  (`main.py` lines 127–131)
```python
short_not_shipping = models.Filter(
    must=[models.FieldCondition(key="n_words", range=models.Range(lte=30))],
    must_not=[models.FieldCondition(key="doc_id", match=models.MatchValue(value="shipping_and_delivery"))],
)
show("n_words <= 30 AND doc_id != shipping:", client.query_points("kb", query=embed(q), query_filter=short_not_shipping, limit=2))
```
- **In simple words:** Short chunks only, and not from shipping.
- **Exactly:** `must` = all conditions true (`Range(lte=30)` on `n_words`); `must_not` = none may be true. Real: `customer_support#Response times` 0.328, `wholesale_and_b2b#Custom blends` 0.279.
- **Why it is here:** Boolean logic on payload fields is the main reason to use a vector database over a bare ANN index.

### 19. E: score threshold as a refusal signal  (`main.py` lines 133–135)
```python
res = client.query_points("kb", query=embed("Do you sell gift cards?"), limit=5, score_threshold=0.45)
print(f"   score_threshold=0.45 for an unanswerable question -> {len(res.points)} results "
      f"(an empty list is a refusal signal, step 08)")
```
- **In simple words:** Ask an unanswerable question and ignore weak matches.
- **Exactly:** `score_threshold=0.45` drops points scoring below 0.45. For "Do you sell gift cards?" the result is `0 results`, an empty list that the pipeline can read as "I don't know" (step 08).
- **Why it is here:** Cheap guard, but a bi-encoder score threshold is less reliable than a reranker's (step 09).

### 20. F: retrieve by id  (`main.py` lines 140–141)
```python
got = client.retrieve("kb", ids=[point_id("shipping_and_delivery#Delivery times")], with_vectors=False)
print(f"F) retrieve by id -> {got[0].payload['chunk_id']}")
```
- **In simple words:** Fetch one known point directly.
- **Exactly:** `retrieve(ids=[...], with_vectors=False)` returns the points with the given ids; no search happens. The id is recomputed with `point_id(...)`.
- **Why it is here:** Look up exactly what was stored for a citation.

### 21. F: scroll through the collection  (`main.py` lines 143–144)
```python
batch, next_page = client.scroll("kb", limit=10, with_payload=["chunk_id"], with_vectors=False)
print(f"   scroll limit=10 -> {len(batch)} points, next_page offset is {'set' if next_page else 'None'}")
```
- **In simple words:** Page through points without a query.
- **Exactly:** `scroll(limit=10, with_payload=["chunk_id"], with_vectors=False)` returns `(points, next_page_offset)`; the offset is set because 26 points remain.
- **Why it is here:** How to export or audit a collection.

### 22. F: delete by filter  (`main.py` lines 146–149)
```python
client.delete("kb", points_selector=models.FilterSelector(filter=models.Filter(
    must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value="privacy_and_data"))])))
print(f"   delete doc_id == privacy_and_data -> {client.count('kb').count} points left "
      f"(how you remove a document that was taken down)")
```
- **In simple words:** Remove every chunk of one document.
- **Exactly:** `delete(points_selector=FilterSelector(filter=...))` removes points where `doc_id == privacy_and_data` (4 chunks): 36 becomes 32.
- **Why it is here:** How you take a withdrawn document out of answers.

### 23. F: close, reopen, clean up  (`main.py` lines 151–155)
```python
client.close()                                                    # releases the folder lock
client = QdrantClient(path=tmp.name)
print(f"   reopened from disk -> {client.count('kb').count} points")
client.close()
tmp.cleanup()
```
- **In simple words:** Prove the data survived on disk.
- **Exactly:** `close()` releases the folder lock (only one process may open a local folder). A new client on the same path counts 32 points. `tmp.cleanup()` deletes the temporary folder.
- **Why it is here:** Persistence that is not tested is not persistence.


