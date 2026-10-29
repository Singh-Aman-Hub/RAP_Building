"""
Exercise 03: the retrieval step of a RAG system, in memory.

Splits a few internal documents into paragraph chunks, embeds them once, and for each question returns the
top-k most similar chunks (with their source document) above a minimum similarity score. The returned
chunks are what would be pasted into the LLM prompt.

Run:   python buggy_code.py
Needs: pip install sentence-transformers numpy
       (first run downloads sentence-transformers/all-MiniLM-L6-v2, ~90 MB)
"""

import numpy as np
from sentence_transformers import SentenceTransformer

DOCUMENTS = [
    {
        "source": "leave_policy.md",
        "text": (
            "Full-time employees get 24 days of paid leave per calendar year.\n\n"
            "Unused paid leave can be carried over to the next year, up to a maximum of 10 days.\n\n"
            "Sick leave is separate: 12 days per year, and a doctor's note is needed for more than 2 days in a row."
        ),
    },
    {
        "source": "expense_policy.md",
        "text": (
            "Travel expenses must be submitted within 30 days of the trip with itemised receipts.\n\n"
            "Hotel stays are reimbursed up to 6,000 rupees per night in metro cities and 4,000 elsewhere.\n\n"
            "Alcohol and personal entertainment are never reimbursed."
        ),
    },
    {
        "source": "it_security.md",
        "text": (
            "Laptops must use full-disk encryption and lock automatically after 5 minutes of inactivity.\n\n"
            "Report a lost or stolen laptop to the IT helpdesk within 24 hours.\n\n"
            "Passwords must be at least 14 characters long and are rotated every 180 days."
        ),
    },
]

MIN_SCORE = 0.30          # cosine similarity below this is treated as "not relevant"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

# ---- offline: chunk + embed once --------------------------------------------------------------------------
chunks = []
for doc in DOCUMENTS:
    for paragraph in doc["text"].split("\n"):
        chunks.append({"source": doc["source"], "text": paragraph.strip()})
chunk_vectors = model.encode([c["text"] for c in chunks], normalize_embeddings=True)   # (n_chunks, 384)
print(f"{len(DOCUMENTS)} documents -> {len(chunks)} chunks")


def similarity(query_vector, matrix):
    """Cosine similarity between one unit query vector and every unit row of `matrix` (higher = closer)."""
    return np.linalg.norm(matrix - query_vector, axis=1)


def retrieve(question, k=3, min_score=MIN_SCORE):
    """Top-k chunks for `question`, most similar first, each with its source and score."""
    query_vector = model.encode(question, normalize_embeddings=True)
    scores = similarity(query_vector, chunk_vectors)
    order = np.argsort(-scores)
    hits = []
    for idx in order[1 : k + 1]:                      # position 0 is the query itself
        if scores[idx] < min_score:
            continue
        hits.append({"source": DOCUMENTS[idx]["source"], "text": chunks[idx]["text"],
                     "score": round(float(scores[idx]), 3)})
    return hits


if __name__ == "__main__":
    questions = [
        "How many days of paid leave can I carry forward?",
        "What is the hotel limit per night in Mumbai?",
        "I lost my laptop at the airport, what should I do?",
        "What is the capital of France?",                 # unrelated: should return no chunks
    ]
    for q in questions:
        print(f"\nQ: {q}")
        hits = retrieve(q)
        if not hits:
            print("   (no relevant chunks)")
        for h in hits:
            print(f"   {h['score']:.3f}  {h['source']:<18} {h['text'][:70]}")
