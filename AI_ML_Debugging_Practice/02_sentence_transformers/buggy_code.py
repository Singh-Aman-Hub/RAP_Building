"""
Exercise 02: semantic search over help articles with Sentence Transformers.

Encodes a handful of help-centre articles, removes duplicate articles (the CMS export contains a few),
and returns the top-k most similar articles for a user query as JSON for the frontend.

Run:   python buggy_code.py
Needs: pip install sentence-transformers numpy
       (first run downloads sentence-transformers/all-MiniLM-L6-v2, ~90 MB)
"""

import json

import numpy as np
from sentence_transformers import SentenceTransformer, util

# CMS export: (title, body). Some articles were exported twice.
ARTICLES = [
    ("Refund policy", "You can request a refund within 14 days of delivery if the item is unused."),
    ("Delivery delays", "Parcels can be delayed by up to 3 days during festivals or heavy rain."),
    ("Reset your password", "Use the 'Forgot password' link on the login page to set a new password."),
    ("Delivery delays", "Parcels can be delayed by up to 3 days during festivals or heavy rain."),
    ("Payment methods", "We accept UPI, credit cards, debit cards and net banking."),
    ("Cancel an order", "Orders can be cancelled from the Orders page until they are shipped."),
    ("Refund policy", "You can request a refund within 14 days of delivery if the item is unused."),
    ("Damaged items", "If your item arrives damaged, upload a photo within 48 hours for a free replacement."),
    ("Change delivery address", "The delivery address can be changed only before the order is shipped."),
]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

titles = [title for title, _ in ARTICLES]
bodies = list(set(body for _, body in ARTICLES))            # drop duplicate articles
body_embeddings = model.encode(bodies, normalize_embeddings=True)
print(f"{len(ARTICLES)} articles exported, {len(bodies)} unique articles indexed")


def search(query, k=3):
    """Return the k most similar articles, most similar first, as JSON-serialisable dicts."""
    query_embedding = model.encode(query, normalize_embeddings=True)
    scores = util.cos_sim(query_embedding, body_embeddings)
    top = np.argsort(scores)[:k]
    return [{"title": titles[i], "snippet": bodies[i][:50], "score": scores[i]} for i in top]


if __name__ == "__main__":
    for q in ["why is my parcel delayed?", "how do I get my money back?", "I can't log in"]:
        print(f"\nQuery: {q}")
        print(json.dumps(search(q), indent=2))
