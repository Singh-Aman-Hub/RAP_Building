"""
Exercise 07: filtered similarity search in Qdrant.

Internal policy search: the user types a question and picks a department in a dropdown; only documents from
that department that are still current (published in or after a given year) may be returned.
The documents arrive as a CSV export from the company wiki.

Run:   python buggy_code.py
Needs: pip install qdrant-client sentence-transformers
       (first run downloads sentence-transformers/all-MiniLM-L6-v2, ~90 MB)
"""

import csv
import io

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

WIKI_EXPORT_CSV = """id,title,department,year,text
1,Parental leave policy,HR,2021,Employees get 12 weeks of paid parental leave.
2,Parental leave policy,HR,2024,Employees get 26 weeks of paid parental leave and can split it into two blocks.
3,Remote work guidelines,HR,2025,Employees may work remotely up to three days a week with manager approval.
4,Deployment checklist,Engineering,2022,Deploy only on Tuesdays after a manual QA sign-off.
5,Deployment checklist,Engineering,2025,Deploy any weekday once CI is green and a reviewer has approved.
6,On-call rotation,Engineering,2024,Each engineer is on call for one week every two months.
7,Expense claims,Finance,2023,Submit expense claims within 30 days with receipts.
8,Travel booking,Finance,2025,Book flights through the travel portal at least 14 days in advance.
"""

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
client = QdrantClient(":memory:")
COLLECTION = "policies"


def ingest():
    rows = list(csv.DictReader(io.StringIO(WIKI_EXPORT_CSV)))
    vectors = model.encode([f"{r['title']}: {r['text']}" for r in rows], normalize_embeddings=True)
    client.create_collection(
        COLLECTION,
        vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.COSINE),
    )
    client.upsert(COLLECTION, points=[
        models.PointStruct(id=int(r["id"]), vector=v.tolist(), payload=r) for r, v in zip(rows, vectors)
    ])
    print(f"ingested {len(rows)} documents")


def search(question, department, min_year, k=3):
    """Top-k documents of `department` (dropdown value, any case) published in or after `min_year`."""
    query_filter = models.Filter(
        should=[
            models.FieldCondition(key="department", match=models.MatchValue(value=department)),
            models.FieldCondition(key="year", range=models.Range(gte=min_year)),
        ]
    )
    hits = client.query_points(
        COLLECTION,
        query=model.encode(question, normalize_embeddings=True).tolist(),
        query_filter=query_filter,
        limit=k,
        with_payload=False,
    ).points
    return [(h.payload["year"], h.payload["department"], h.payload["title"]) for h in hits]


if __name__ == "__main__":
    ingest()
    # The frontend dropdown sends lower-case values.
    print(search("How many weeks of parental leave do I get?", department="hr", min_year=2024))
    print(search("When am I allowed to deploy?", department="engineering", min_year=2024))
    print(search("How do I claim my expenses?", department="finance", min_year=2026))
