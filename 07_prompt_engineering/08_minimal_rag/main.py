"""
GOAL   : Show why a frozen model "hallucinates" about private facts, then fix it with a minimal RAG
         pipeline: embed documents -> cosine-similarity search (NumPy) -> inject top-k into the prompt.
RUN    : cd 07_prompt_engineering/08_minimal_rag && python main.py
         (optional: RAG_EMBEDDER=minilm python main.py   -> real sentence embeddings, needs the cached
          all-MiniLM-L6-v2 model; falls back to TF-IDF automatically if it cannot be loaded offline)
EXPECTED OUTPUT: the same questions answered WITHOUT and WITH retrieved context, then two failure modes
         (retrieval miss, irrelevant chunk) and the similarity-threshold fix.
NEEDS  : numpy, scikit-learn (installed in .venv). No network. The "model" is a scripted simulator.
"""
import os
import sys
from pathlib import Path

import numpy as np                                    # all vector maths below is plain NumPy
from sklearn.feature_extraction.text import TfidfVectorizer   # turns text into word-weight vectors

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from llm_client import generate, MockLLM

# ---------------------------------------------------------------------------
# 1. The private knowledge base: facts a frozen model cannot possibly have seen.
# ---------------------------------------------------------------------------
DOCS = [
    "Refunds at Chai Kitchen are issued within 9 days of delivery if the seal is unbroken.",
    "Orders ship from the Pune warehouse and arrive in 4 working days. Shipping is free above 799 rupees.",
    "Customer support is open Monday to Saturday, 10am to 6pm IST. It is closed on Sunday.",
    "Loyalty points: customers earn 3 points per 100 rupees. 500 points give one free 250 g pack.",
    "Every blend is nut free, but the packing facility also handles sesame.",
    "Tea is packed in compostable pouches that can go into a home compost bin.",
]

# ---------------------------------------------------------------------------
# 2. Embedding: text -> vector. TF-IDF by default (always works); MiniLM if asked for AND loadable.
# ---------------------------------------------------------------------------
def build_embedder(corpus):
    if os.getenv("RAG_EMBEDDER") == "minilm":
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")             # never touch the network
            import torch
            from transformers import AutoModel, AutoTokenizer
            name = "sentence-transformers/all-MiniLM-L6-v2"
            tok, net = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name)
            def embed(texts):                                        # returns shape: (n_texts, 384)
                batch = tok(texts, padding=True, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    hidden = net(**batch).last_hidden_state          # shape: (n_texts, n_tokens, 384)
                mask = batch["attention_mask"].unsqueeze(-1)         # shape: (n_texts, n_tokens, 1)
                return ((hidden * mask).sum(1) / mask.sum(1)).numpy()  # mean-pool -> (n_texts, 384)
            embed(["warm up"])
            return embed, "MiniLM (sentence embeddings)"
        except Exception as err:                                     # model not cached, etc.
            print(f"[minilm unavailable ({type(err).__name__}); falling back to TF-IDF]")
    vec = TfidfVectorizer(stop_words="english").fit(corpus)
    return (lambda texts: vec.transform(texts).toarray()), "TF-IDF (word-weight vectors)"   # (n_texts, vocab)

embed, EMBEDDER_NAME = build_embedder(DOCS)
doc_vectors = embed(DOCS)                                  # shape: (n_docs, dim)

# ---------------------------------------------------------------------------
# 3. The ~20-line vector search: cosine similarity = dot product of unit-length vectors.
# ---------------------------------------------------------------------------
def retrieve(question, k=2, min_score=0.0):
    q = embed([question])[0]                               # shape: (dim,)
    norms = np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(q) + 1e-9   # shape: (n_docs,)
    scores = doc_vectors @ q / norms                       # cosine per document, shape: (n_docs,)
    best = np.argsort(scores)[::-1][:k]                    # indices of the k highest scores
    return [(DOCS[i], float(scores[i])) for i in best if scores[i] >= min_score]

# ---------------------------------------------------------------------------
# 4. The "model": a scripted simulator of a FROZEN LLM (honest toy, not a real model).
#    No context  -> it guesses confidently from generic habits (a hallucination).
#    With context-> it repeats the context sentence that best overlaps the question.
# ---------------------------------------------------------------------------
GUESSES = {"refund": "Refunds are usually accepted within 30 days.",
           "support": "Support is available 24/7 by phone.",
           "ship": "Delivery usually takes 2 days."}

def frozen_model(messages, system):
    prompt = messages[-1]["content"]
    question = prompt.rsplit("Question:", 1)[1].strip().lower()
    if "Context:" not in prompt:
        for key, guess in GUESSES.items():
            if key in question:
                return guess
        return "Yes, that is generally possible."
    context = prompt.split("Context:")[1].split("Question:")[0]
    stop = {"the", "a", "an", "are", "is", "what", "do", "i", "for", "of", "how", "you", "to", "have", "many", "can"}
    words = set(question.replace("?", "").split()) - stop        # content words only
    sentences = [s.strip(" -") for s in context.replace("\n", " ").split(".") if s.strip(" -")]
    if not sentences:
        return "I don't know based on the provided context."
    return max(sentences, key=lambda s: len(words & set(s.lower().split()))) + "."

LLM = MockLLM(rules=[(r"", frozen_model)])

def ask(question, use_rag, k=2, min_score=0.0):
    if not use_rag:
        prompt = f"Question: {question}"
    else:
        hits = retrieve(question, k, min_score)
        context = "\n".join(f"- {text}" for text, _ in hits)      # inject retrieved chunks into the prompt
        prompt = f"Context:\n{context}\n\nQuestion: {question}" if hits else f"Question: {question}"
        print(f"   retrieved: {[(t[:28] + '...', round(s, 2)) for t, s in hits]}")
        if not hits:
            return "I don't know: nothing relevant in the knowledge base."
    system = "Answer ONLY from the context. If it is not there, say you don't know."
    return generate([{"role": "user", "content": prompt}], system=system, mock=LLM)

if __name__ == "__main__":
    print(f"Embedder: {EMBEDDER_NAME} | {len(DOCS)} documents -> vectors {doc_vectors.shape}\n")

    for q in ["How many days do I have for a refund?", "What are the support hours?"]:
        print(f"Q: {q}")
        print("   WITHOUT retrieval:", ask(q, use_rag=False))
        print("   WITH retrieval   :", ask(q, use_rag=True))
        print()

    print("--- Failure mode 1: retrieval MISS (question uses different words than the document) ---")
    q = "What is the money back window?"
    print(f"Q: {q}")
    print("   WITH retrieval   :", ask(q, use_rag=True))

    print("\n--- Failure mode 2: irrelevant chunk (the answer is NOT in the knowledge base) ---")
    q = "Do you sell gift cards?"
    print(f"Q: {q}")
    print("   WITH retrieval   :", ask(q, use_rag=True))
    print("   same, but with a similarity threshold (min_score=0.2):")
    print("   ->", ask(q, use_rag=True, min_score=0.2))

# TRY IT: change k=2 to k=1, then k=6 (everything) - watch the answer quality and prompt size change.
# TRY IT: add a document "Gift cards are sold in 500 and 1000 rupee values." and re-ask the gift card question.
# TRY IT: RAG_EMBEDDER=minilm python main.py - the money-back question may now find the refund document.
# BREAK IT: embed with one vectorizer and query with another -> shapes differ:
#           doc_vectors @ q  ->  ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0
