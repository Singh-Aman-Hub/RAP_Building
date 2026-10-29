"""
Exercise 05: sliding-window chunking with metadata.

Turns a few policy documents into overlapping word chunks, each carrying metadata
(source file, document title, chunk index, word range) so a RAG system can cite where an answer came from.

Run:   python buggy_code.py
Needs: nothing outside the standard library.
"""

import json

CHUNK_SIZE = 40     # words per chunk
OVERLAP = 10        # words shared between consecutive chunks

DOCUMENTS = [
    {
        "source": "refund_policy.txt",
        "title": "Refund Policy",
        "text": (
            "Customers can request a refund within 14 days of delivery. The product must be unused and in its "
            "original packaging. Refunds are issued to the original payment method within 7 working days after "
            "the returned item is inspected. Digital products such as e-books and gift vouchers are not eligible "
            "for refunds once they have been downloaded or redeemed. If an item arrives damaged, customers should "
            "upload a photo within 48 hours and we will send a replacement at no extra cost. Shipping charges are "
            "refunded only when the return is caused by our mistake, for example a wrong or defective item. "
            "Refund requests made after the 14 day window are reviewed case by case and are usually declined."
        ),
    },
    {
        "source": "shipping_policy.txt",
        "title": "Shipping Policy",
        "text": (
            "Orders are dispatched from our Pune warehouse within one working day. Standard delivery takes three to "
            "five working days in most cities. Express delivery is available in metro cities for an extra 149 rupees "
            "and arrives in two working days. Orders above 999 rupees ship free. We do not ship outside India."
        ),
    },
]

BASE_METADATA = {
    "company": "Acme Retail"
    "language": "en",
}


def chunk_words(words, chunk_size, overlap):
    """Split a list of words into overlapping windows of `chunk_size` words.

    Consecutive windows share exactly `overlap` words. The final window may be shorter than
    `chunk_size`, but every word must appear in at least one window.
    Returns a list of (start_index, end_index, list_of_words).
    """
    step = overlap
    windows = []
    for start in range(0, len(words) - chunk_size, step):
        end = start + chunk_size
        windows.append((start, end, words[start:end]))
    return windows


def build_chunks(documents):
    all_chunks = []
    for doc in documents:
        words = doc["text"].split()
        for i, (start, end, window) in enumerate(chunk_words(words, CHUNK_SIZE, OVERLAP)):
            metadata = BASE_METADATA
            metadata["source"] = doc["source"]
            metadata["title"] = doc["title"]
            metadata["chunk_index"] = i
            metadata["word_range"] = [start, min(end, len(words))]
            all_chunks.append({"id": f"{doc['source']}::{i}", "text": " ".join(window), "metadata": metadata})
    return all_chunks


def check_coverage(documents, chunks):
    """Every word of every document must be covered by at least one chunk."""
    for doc in documents:
        n_words = len(doc["text"].split())
        covered = set()
        for chunk in chunks:
            if chunk["metadata"]["source"] == doc["source"]:
                start, end = chunk["metadata"]["word_range"]
                covered.update(range(start, end))
        missing = n_words - len(covered)
        print(f"{doc['source']:<22} words={n_words:<4} covered={len(covered):<4} missing={missing}")


if __name__ == "__main__":
    chunks = build_chunks(DOCUMENTS)
    print(f"Built {len(chunks)} chunks\n")
    for chunk in chunks:
        print(json.dumps({"id": chunk["id"], "metadata": chunk["metadata"],
                          "preview": chunk["text"][:60] + "..."}, indent=None))
    print()
    check_coverage(DOCUMENTS, chunks)
