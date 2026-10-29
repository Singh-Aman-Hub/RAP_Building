"""
REFERENCE ONLY -- NOT EXECUTED IN THIS STUDY PACK.

The same "add vectors, search top-k, filter by metadata" job you built by hand in steps 04 and 05,
done by four real tools.  None of them is installed in this repo's .venv; read the code, then try one in a
notebook:   pip install faiss-cpu chromadb qdrant-client psycopg[binary] pgvector sentence-transformers

Check syntax without installing anything:   python -m py_compile vector_db_reference.py

Which one when?
    FAISS     library (not a server). Fastest in-process search, many index types (Flat, IVF, HNSW, PQ).
              You manage persistence, metadata and updates yourself.
    Chroma    embedded/local database. Easiest start: stores text + metadata + vectors, filters with `where`.
    pgvector  extension for PostgreSQL. Vectors live next to your normal tables: one database, SQL joins,
              transactions, backups you already have.  HNSW and IVFFlat indexes.
    Qdrant /  dedicated vector databases (servers). Horizontal scaling, payload filtering during HNSW search,
    Milvus /  replication, snapshots, hybrid (sparse + dense) search built in.  Pinecone is the hosted kind.
    Weaviate
"""

import numpy as np

texts = ["Refunds are issued within 9 days.", "Shipping is free above 799 rupees.", "Support is closed on Sundays."]
metas = [{"doc_id": "refunds"}, {"doc_id": "shipping"}, {"doc_id": "support"}]

# Embeddings: the same MiniLM model as the rest of the module, via the sentence-transformers library.
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
vectors = model.encode(texts, normalize_embeddings=True).astype(np.float32)        # (3, 384)
query = model.encode(["How long do refunds take?"], normalize_embeddings=True).astype(np.float32)

# ---------------------------------------------------------------------------------------------------------
# 1) FAISS: exact, HNSW and IVF-PQ indexes
# ---------------------------------------------------------------------------------------------------------
import faiss

flat = faiss.IndexFlatIP(384)                    # IP = inner product = cosine on unit vectors (step 04)
flat.add(vectors)
scores, idx = flat.search(query, 2)              # both arrays have shape (n_queries, k)
print("FAISS flat:", [texts[i] for i in idx[0]], scores[0])

hnsw = faiss.IndexHNSWFlat(384, 32, faiss.METRIC_INNER_PRODUCT)   # M = 32 edges per node (step 05 B)
hnsw.hnsw.efSearch = 64                          # ef: the recall/speed dial at query time
hnsw.add(vectors)

# IVF + PQ needs TRAINING data (k-means for cells and codebooks, step 05 A and C). With 3 vectors it cannot
# train, so this is the shape of the code for a real corpus of 100k+ vectors:
#   quantizer = faiss.IndexFlatIP(384)
#   ivfpq = faiss.IndexIVFPQ(quantizer, 384, 1024, 48, 8)   # nlist=1024 cells, 48 sub-vectors, 8 bits each
#   ivfpq.train(big_matrix); ivfpq.add(big_matrix); ivfpq.nprobe = 16
faiss.write_index(flat, "kb.faiss")              # persistence is a file; metadata goes in your own JSON/DB

# ---------------------------------------------------------------------------------------------------------
# 2) Chroma: text + metadata + vectors in one local store, with metadata filters
# ---------------------------------------------------------------------------------------------------------
import chromadb

client = chromadb.PersistentClient(path="chroma_db")
col = client.get_or_create_collection("kb", metadata={"hnsw:space": "cosine"})
col.upsert(ids=["r1", "s1", "c1"], documents=texts, metadatas=metas, embeddings=vectors.tolist())
res = col.query(query_embeddings=query.tolist(), n_results=2, where={"doc_id": "refunds"})   # pre-filter (step 04)
print("Chroma:", res["documents"][0], res["distances"][0])

# ---------------------------------------------------------------------------------------------------------
# 3) pgvector: vectors as a column in PostgreSQL (SQL shown as strings; run with psycopg)
# ---------------------------------------------------------------------------------------------------------
PGVECTOR_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE chunks (
    id        bigserial PRIMARY KEY,
    doc_id    text NOT NULL,
    content   text NOT NULL,
    embedding vector(384) NOT NULL,
    updated   timestamptz DEFAULT now()
);
-- HNSW index for cosine distance; m = edges per node, ef_construction = build-time beam width
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
SET hnsw.ef_search = 40;                         -- query-time recall/speed dial

-- top-5 by cosine distance (<=>), filtered by metadata in plain SQL
SELECT doc_id, content, 1 - (embedding <=> %(q)s::vector) AS cosine
FROM chunks
WHERE doc_id = 'refunds'
ORDER BY embedding <=> %(q)s::vector
LIMIT 5;
"""
# Operators: <=> cosine distance, <-> L2 distance, <#> negative inner product.

# ---------------------------------------------------------------------------------------------------------
# 4) Qdrant: a vector database server (here in-memory mode), payload filtering during search
# ---------------------------------------------------------------------------------------------------------
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

qdrant = QdrantClient(":memory:")                # production: QdrantClient(url="http://localhost:6333")
qdrant.create_collection("kb", vectors_config=VectorParams(size=384, distance=Distance.COSINE))
qdrant.upsert("kb", points=[PointStruct(id=i, vector=v.tolist(), payload={**m, "text": t})
                            for i, (v, m, t) in enumerate(zip(vectors, metas, texts))])
hits = qdrant.query_points("kb", query=query[0].tolist(), limit=2,
                           query_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value="refunds"))]))
print("Qdrant:", [(h.payload["text"], round(h.score, 3)) for h in hits.points])
