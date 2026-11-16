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

import os
import sys
import tempfile
import uuid
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")                      # cached models only

from qdrant_client import QdrantClient, models                    # noqa: E402
from sentence_transformers import SentenceTransformer             # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EMBED_MODEL, build_chunks, chunk_for_embedding  # noqa: E402

chunks = build_chunks()                                           # 36 section chunks of the Chai Kitchen KB
model = SentenceTransformer(EMBED_MODEL, device="cpu")
DIM = model.get_sentence_embedding_dimension()                    # 384 -- ask the model, never hard-code
vectors = model.encode([chunk_for_embedding(c) for c in chunks], normalize_embeddings=True)


def embed(text):
    return model.encode(text, normalize_embeddings=True).tolist()  # Qdrant wants a plain list of floats


# =========================================================================================================
# A) Client modes. Same API for all three, so code written against :memory: runs against a server.
#      QdrantClient(":memory:")              -> in this process, gone when it exits (tests, notebooks)
#      QdrantClient(path="qdrant_data")      -> in this process, saved to a folder (single-user scripts)
#      QdrantClient(url="http://localhost:6333")  -> real server: docker run -p 6333:6333 qdrant/qdrant
# =========================================================================================================
tmp = tempfile.TemporaryDirectory()
client = QdrantClient(path=tmp.name)

# =========================================================================================================
# B) Collection = one vector space. size and distance are fixed at creation and cannot be changed later.
# =========================================================================================================
if client.collection_exists("kb"):                                # make the script safe to re-run
    client.delete_collection("kb")
client.create_collection(
    collection_name="kb",
    vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE),
)
info = client.get_collection("kb")
print(f"A/B) collection 'kb' created: size={info.config.params.vectors.size} "
      f"distance={info.config.params.vectors.distance.value}")

# =========================================================================================================
# C) Points = id + vector + payload. Ids must be an unsigned int or a UUID; "rule_0" is rejected.
#    uuid5(namespace, chunk_id) is deterministic: same chunk -> same id -> upsert REPLACES, never duplicates.
# =========================================================================================================
NAMESPACE = uuid.UUID("6f1c2b1e-3c1a-4c7e-9a52-2d1f0c9b7a10")    # any fixed UUID; keep it constant forever


def point_id(chunk_id):
    return str(uuid.uuid5(NAMESPACE, chunk_id))


points = [
    models.PointStruct(
        id=point_id(c["chunk_id"]),
        vector=vectors[i].tolist(),
        payload={"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "section": c["section"],
                 "text": c["text"], "n_words": len(c["text"].split())},
    )
    for i, c in enumerate(chunks)
]
client.upsert(collection_name="kb", points=points, wait=True)    # wait=True: return after it is searchable
n1 = client.count("kb").count
client.upsert(collection_name="kb", points=points, wait=True)    # re-ingest the same chunks
n2 = client.count("kb").count
print(f"C) {n1} points upserted; upsert again -> still {n2} (uuid5 ids are stable)")
try:
    client.upsert("kb", points=[models.PointStruct(id="rule_0", vector=vectors[0].tolist(), payload={})])
except Exception as e:                                            # noqa: BLE001 -- show the real message
    print(f"   id='rule_0' rejected: {type(e).__name__}: {str(e).splitlines()[0][:90]}")

# =========================================================================================================
# D) Payload indexes. On a server, filtering on an un-indexed field scans every point (and strict mode can
#    refuse it). Create one index per field you filter on, BEFORE bulk loading. Local mode accepts and
#    ignores the call, so the same code is correct everywhere.
# =========================================================================================================
client.create_payload_index("kb", field_name="doc_id", field_schema=models.PayloadSchemaType.KEYWORD)
client.create_payload_index("kb", field_name="n_words", field_schema=models.PayloadSchemaType.INTEGER)


def show(title, result):
    print(f"   {title}")
    for p in result.points:
        print(f"      {p.score:.3f}  {p.payload['chunk_id']}")


# =========================================================================================================
# E) Search. query_points() is the current API (client.search() was deprecated in 1.10 and later removed).
#    Distance.COSINE -> score is cosine similarity: HIGHER is closer.
# =========================================================================================================
q = "How long does delivery take?"
res = client.query_points("kb", query=embed(q), limit=3, with_payload=True)
print(f"E) top-1 for {q!r} -> {res.points[0].payload['chunk_id']}")
show("no filter, limit=3:", res)

only_refunds = models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value="refunds_and_returns"))])
show("must doc_id == refunds_and_returns:", client.query_points("kb", query=embed(q), query_filter=only_refunds, limit=2))

two_docs = models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=["loyalty_program", "customer_support"]))])
show("must doc_id in [loyalty_program, customer_support]:", client.query_points("kb", query=embed(q), query_filter=two_docs, limit=2))

short_not_shipping = models.Filter(
    must=[models.FieldCondition(key="n_words", range=models.Range(lte=30))],
    must_not=[models.FieldCondition(key="doc_id", match=models.MatchValue(value="shipping_and_delivery"))],
)
show("n_words <= 30 AND doc_id != shipping:", client.query_points("kb", query=embed(q), query_filter=short_not_shipping, limit=2))

res = client.query_points("kb", query=embed("Do you sell gift cards?"), limit=5, score_threshold=0.45)
print(f"   score_threshold=0.45 for an unanswerable question -> {len(res.points)} results "
      f"(an empty list is a refusal signal, step 08)")

# =========================================================================================================
# F) CRUD around search: retrieve by id, scroll (paginate without a query), delete by filter, persistence.
# =========================================================================================================
got = client.retrieve("kb", ids=[point_id("shipping_and_delivery#Delivery times")], with_vectors=False)
print(f"F) retrieve by id -> {got[0].payload['chunk_id']}")

batch, next_page = client.scroll("kb", limit=10, with_payload=["chunk_id"], with_vectors=False)
print(f"   scroll limit=10 -> {len(batch)} points, next_page offset is {'set' if next_page else 'None'}")

client.delete("kb", points_selector=models.FilterSelector(filter=models.Filter(
    must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value="privacy_and_data"))])))
print(f"   delete doc_id == privacy_and_data -> {client.count('kb').count} points left "
      f"(how you remove a document that was taken down)")

client.close()                                                    # releases the folder lock
client = QdrantClient(path=tmp.name)
print(f"   reopened from disk -> {client.count('kb').count} points")
client.close()
tmp.cleanup()
