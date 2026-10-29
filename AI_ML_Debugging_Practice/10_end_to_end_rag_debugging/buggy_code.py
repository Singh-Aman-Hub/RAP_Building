"""
Exercise 10: an end-to-end RAG pipeline: ingest -> embed -> Qdrant -> retrieve -> prompt -> answer.

A customer-support assistant over three policy documents. Users can optionally restrict a question to one
document (e.g. the UI's "search only in: warranty.md" option).

The LLM is a deterministic offline stand-in (`fake_llm`, do not edit it):
  - it splits the question into parts on " and ";
  - for each part it answers with the SOURCES sentence sharing the most words with that part, citing its label;
  - if no sentence shares at least 2 words with any part, it invents an answer (a hallucination).

Run:   python buggy_code.py
Needs: pip install qdrant-client sentence-transformers
       (first run downloads sentence-transformers/all-MiniLM-L6-v2, ~90 MB)
"""

import re

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

DOCUMENTS = {
    "shipping.md": (
        "Standard delivery takes 3 to 5 working days anywhere in India.\n\n"
        "Express delivery takes 2 working days and costs 149 rupees.\n\n"
        "Orders above 999 rupees ship free of charge."
    ),
    "returns.md": (
        "You can return an item within 10 days of delivery if it is unused.\n\n"
        "Refunds are credited to the original payment method within 7 working days.\n\n"
        "Sale items and gift cards cannot be returned."
    ),
    "warranty.md": (
        "Electronics come with a one year manufacturer warranty.\n\n"
        "The warranty does not cover water damage or accidental drops.\n\n"
        "Warranty claims need the original invoice and the serial number."
    ),
}

COLLECTION = "support_docs"
TOP_K = 3
MIN_SCORE = 0.35            # best hit below this -> refuse instead of calling the LLM
REFUSAL = "Sorry, I can't find that in our policies."

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
client = QdrantClient(":memory:")


# ---------------------------------------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------------------------------------
def ingest():
    dim = model.encode("probe").shape[0]
    client.create_collection(COLLECTION, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
    total = 0
    for filename, text in DOCUMENTS.items():
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        vectors = model.encode(paragraphs, normalize_embeddings=True)
        client.upsert(COLLECTION, points=[
            models.PointStruct(id=i, vector=vectors[i].tolist(), payload={"doc": filename, "text": paragraphs[i]})
            for i in range(len(paragraphs))
        ])
        total += len(paragraphs)
    print(f"ingested {total} chunks; collection holds {client.count(COLLECTION).count} points")


# ---------------------------------------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------------------------------------
def retrieve(question, source=None, k=TOP_K):
    """Top-k chunks for the question, optionally only from one source document."""
    query_filter = None
    if source:
        query_filter = models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))])
    hits = client.search(
        collection_name=COLLECTION,
        query_vector=model.encode(question, normalize_embeddings=True).tolist(),
        query_filter=query_filter,
        limit=1,
    )
    return [{"source": h.payload["doc"], "text": h.payload["chunk"], "score": h.score} for h in hits]


# ---------------------------------------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------------------------------------
STOP = set("a an the is are do does i my you your can how long many what of to in for and or it be".split())


def _words(s):
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP}


def fake_llm(prompt):
    """Offline stand-in for an LLM. DO NOT EDIT. See the module docstring."""
    question = prompt.split("QUESTION:")[1].strip()
    sources = re.findall(r"^\[(\d+)\] (.*)$", prompt, flags=re.M)
    answers = []
    for part in question.rstrip("?").split(" and "):
        scored = [(len(_words(part) & _words(text)), label, text) for label, text in sources]
        overlap, label, text = max(scored, default=(0, None, None))
        if overlap >= 2:
            answers.append(f"{text} [{label}]")
    return " ".join(answers) or "Yes! That is fully covered and free of charge."


def build_prompt(question, chunks):
    sources = "\n".join(f"[{i}] {c['text']}" for i, c in enumerate(chunks, start=1))
    return ("Answer only from the sources and cite them like [1]. "
            f"If the answer is not in the sources, say: {REFUSAL}\n\nSOURCES:\n{sources}\n\nQUESTION: {question}")


def answer(question, source=None):
    chunks = retrieve(question, source=source)
    if not chunks or chunks[0]["score"] < MIN_SCORE:
        return REFUSAL, []
    return fake_llm(build_prompt(question, chunks)), [c["source"] for c in chunks]


if __name__ == "__main__":
    ingest()
    tests = [
        ("How long does standard delivery take?", None),
        ("How long does standard delivery take and how many days do I have to return an item?", None),
        ("Is water damage covered?", "warranty.md"),
        ("Do you offer gift wrapping?", None),
    ]
    for question, source in tests:
        reply, used = answer(question, source)
        print(f"\nQ: {question}" + (f"   [only {source}]" if source else ""))
        print(f"A: {reply}\n   retrieved from: {used}")
