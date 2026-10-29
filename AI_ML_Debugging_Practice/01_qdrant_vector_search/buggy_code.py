"""
Exercise 01: semantic FAQ search with Qdrant.

Embeds a small help-centre FAQ with Sentence Transformers, stores it in an in-memory Qdrant collection,
and answers user questions, optionally restricted to one FAQ category.

Run:   python buggy_code.py
Needs: pip install qdrant-client sentence-transformers
       (first run downloads sentence-transformers/all-MiniLM-L6-v2, ~90 MB)
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

COLLECTION = "faq"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE = 768

FAQ = [
    {"question": "How do I reset my password?",
     "answer": "Click 'Forgot password' on the login page and follow the link we email you.",
     "category": "account"},
    {"question": "How can I change the email address on my account?",
     "answer": "Go to Settings > Profile, edit the email field and confirm the change from your new inbox.",
     "category": "account"},
    {"question": "Can I pay with a credit card?",
     "answer": "Yes, we accept Visa, Mastercard and RuPay credit and debit cards.",
     "category": "billing"},
    {"question": "Why was I charged twice?",
     "answer": "A duplicate charge is usually a pending authorisation that disappears within 3 working days.",
     "category": "billing"},
    {"question": "How long does delivery take?",
     "answer": "Standard delivery takes 3 to 5 working days; express takes 2 working days in metro cities.",
     "category": "shipping"},
    {"question": "Do you deliver outside India?",
     "answer": "Not yet. We currently deliver only to addresses within India.",
     "category": "shipping"},
]

model = SentenceTransformer(MODEL_NAME, device="cpu")
client = QdrantClient(":memory:")


def build_index():
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    texts = [f"{item['question']} {item['answer']}" for item in FAQ]
    vectors = model.encode(texts, normalize_embeddings=True)
    points = [
        PointStruct(
            id=f"faq-{i}",
            vector=vectors[i].tolist(),
            payload={"question": item["question"], "answer": item["answer"], "category": item["category"]},
        )
        for i, item in enumerate(FAQ)
    ]
    client.upsert(collection_name=COLLECTION, points=points)
    print(f"Indexed {len(points)} FAQ entries")


def ask(user_question, category=None, top_k=3):
    query_vector = model.encode(user_question, normalize_embeddings=True).tolist()
    query_filter = None
    if category is not None:
        query_filter = Filter(must=[FieldCondition(key="category", match=category)])

    results = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
    )

    print(f"\nQ: {user_question}   (category={category})")
    for hit in results:
        print(f"   {hit.score:.3f}  [{hit.payload['category']}] {hit.payload['question']}")


if __name__ == "__main__":
    build_index()
    ask("I forgot my login password")
    ask("I got billed two times for one order", category="billing")
    ask("Can you ship to Dubai?", category="shipping", top_k=1)
