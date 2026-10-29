"""
Exercise 06: keeping index-time and query-time embeddings consistent.

Simulates two separate programs that a real team would deploy separately:
  1. an INDEXING JOB that embeds knowledge-base articles and writes the vectors + metadata to disk;
  2. a SEARCH SERVICE that loads them into Qdrant and answers queries.
Both read the shared CONFIG below (in production it would be a YAML file or environment variables).

Run:   python buggy_code.py
Needs: pip install qdrant-client sentence-transformers numpy
       (first run downloads all-MiniLM-L6-v2 (~90 MB) and paraphrase-multilingual-MiniLM-L12-v2 (~470 MB))
"""

import json
import tempfile
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

CONFIG = {
    "index_model": "sentence-transformers/all-MiniLM-L6-v2",
    "query_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "embedding_dim": 768,
    "max_seq_length": 32,          # tokens per article; lowered to make indexing faster
    "collection": "kb",
}

ARTICLES = [
    {"id": 1, "title": "Two-factor authentication",
     "body": "Two-factor authentication adds a second step to signing in. After entering your password you confirm "
             "the login in the authenticator app on your phone. If you lose your phone, sign in with one of the "
             "ten backup codes you downloaded when you enabled two-factor authentication."},
    {"id": 2, "title": "Exporting your data",
     "body": "You can export all your projects as a ZIP archive from Settings, Privacy, Export data. Large exports "
             "are prepared in the background and a download link is emailed to you within 24 hours. The link "
             "expires after seven days, after which you need to request a new export."},
    {"id": 3, "title": "Team plans and billing",
     "body": "Team plans are billed per active member per month. Members who have not logged in during a billing "
             "cycle are not charged. Admins can download invoices from the Billing page."},
    {"id": 4, "title": "Deleting your account",
     "body": "Deleting your account permanently removes all projects after a 30 day grace period. During the grace "
             "period you can restore the account by simply logging in again."},
]


# ------------------------------------------------------------------------------------------------------------
# Program 1: indexing job
# ------------------------------------------------------------------------------------------------------------
def run_indexing_job(out_dir):
    model = SentenceTransformer(CONFIG["index_model"], device="cpu")
    model.max_seq_length = CONFIG["max_seq_length"]
    texts = [f"{a['title']}. {a['body']}" for a in ARTICLES]
    vectors = model.encode(texts, normalize_embeddings=True)
    np.save(out_dir / "vectors.npy", vectors)
    (out_dir / "meta.json").write_text(json.dumps([{"id": a["id"], "title": a["title"]} for a in ARTICLES]))
    print(f"[indexer] wrote {vectors.shape[0]} vectors of dim {vectors.shape[1]}")


# ------------------------------------------------------------------------------------------------------------
# Program 2: search service
# ------------------------------------------------------------------------------------------------------------
class SearchService:
    def __init__(self, index_dir):
        self.model = SentenceTransformer(CONFIG["query_model"], device="cpu")
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            CONFIG["collection"],
            vectors_config=models.VectorParams(size=CONFIG["embedding_dim"], distance=models.Distance.COSINE),
        )
        vectors = np.load(index_dir / "vectors.npy")
        meta = json.loads((index_dir / "meta.json").read_text())
        self.client.upsert(CONFIG["collection"], points=[
            models.PointStruct(id=m["id"], vector=v.tolist(), payload=m) for v, m in zip(vectors, meta)
        ])
        print(f"[search] loaded {len(meta)} articles")

    def search(self, query, k=1):
        query_vector = self.model.encode([query], normalize_embeddings=True).tolist()
        hits = self.client.query_points(CONFIG["collection"], query=query_vector, limit=k).points
        return [(h.payload["title"], round(h.score, 3)) for h in hits]


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        index_dir = Path(tmp)
        run_indexing_job(index_dir)
        service = SearchService(index_dir)

        # Sanity check: searching with an article's exact indexed text must find that article with score 1.000.
        a = ARTICLES[1]
        indexed_text = f"{a['title']}. {a['body']}"
        print(f"self-retrieval check -> {service.search(indexed_text)}\n")

        for q in ["I lost my phone and can't log in",
                  "how long is the data download link valid?",
                  "am I charged for inactive team members?",
                  "can I undo deleting my account?"]:
            print(f"{q!r:<48} -> {service.search(q)}")
