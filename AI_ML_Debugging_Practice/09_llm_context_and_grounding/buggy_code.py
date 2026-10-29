"""
Exercise 09: building a grounded LLM prompt from retrieved chunks.

Retrieval has already happened: for each question we have the retrieved chunks with their cosine similarity
scores (higher = more relevant). This module decides what goes into the prompt, calls the LLM, and maps
the [n] citations in the answer back to source documents.

The "LLM" is a deterministic stand-in (`fake_llm`) so the exercise runs offline with no API key:
  - it answers using the SOURCES line that shares the most words with the question, and cites that line's label;
  - if no source shares any word with the question, it makes up a confident answer (as real LLMs do).
Do not edit `fake_llm`; treat it like an external model you cannot change.

Run:   python buggy_code.py
Needs: nothing outside the standard library.
"""

import re

MAX_CONTEXT_WORDS = 60          # word budget for the SOURCES section of the prompt
REFUSAL_THRESHOLD = 0.45        # below this best-score, retrieval found nothing relevant -> refuse
REFUSAL = "I don't know based on the available documents."

# question -> retrieved chunks (already ranked by the retriever, scores are cosine similarity)
RETRIEVED = {
    "Within how many days can items be returned?": [
        {"id": "c1", "source": "refund_policy_2025.md", "status": "active", "score": 0.78,
         "text": "Items can be returned within 30 days of delivery for a full refund. Items must be unused and in "
                 "original packaging. Refunds reach the original payment method within 7 working days."},
        {"id": "c2", "source": "refund_policy_2022.md", "status": "archived", "score": 0.74,
         "text": "Items can be returned within 14 days of delivery. Opened items are not accepted. Refunds are "
                 "issued as store credit only."},
        {"id": "c3", "source": "shipping_policy.md", "status": "active", "score": 0.41,
         "text": "Standard delivery takes three to five working days. Express delivery takes two working days in "
                 "metro cities."},
        {"id": "c4", "source": "warranty.md", "status": "active", "score": 0.38,
         "text": "Electronics carry a one year warranty against manufacturing defects. The warranty does not cover "
                 "accidental damage or water damage."},
    ],
    "How long does express delivery take?": [
        {"id": "c3", "source": "shipping_policy.md", "status": "active", "score": 0.81,
         "text": "Standard delivery takes three to five working days. Express delivery takes two working days in "
                 "metro cities."},
        {"id": "c1", "source": "refund_policy_2025.md", "status": "active", "score": 0.35,
         "text": "Items can be returned within 30 days of delivery for a full refund. Items must be unused and in "
                 "original packaging. Refunds reach the original payment method within 7 working days."},
        {"id": "c4", "source": "warranty.md", "status": "active", "score": 0.30,
         "text": "Electronics carry a one year warranty against manufacturing defects. The warranty does not cover "
                 "accidental damage or water damage."},
    ],
    "Do you sell gift cards?": [
        {"id": "c3", "source": "shipping_policy.md", "status": "active", "score": 0.31,
         "text": "Standard delivery takes three to five working days. Express delivery takes two working days in "
                 "metro cities."},
        {"id": "c4", "source": "warranty.md", "status": "active", "score": 0.29,
         "text": "Electronics carry a one year warranty against manufacturing defects. The warranty does not cover "
                 "accidental damage or water damage."},
        {"id": "c1", "source": "refund_policy_2025.md", "status": "active", "score": 0.27,
         "text": "Items can be returned within 30 days of delivery for a full refund. Items must be unused and in "
                 "original packaging. Refunds reach the original payment method within 7 working days."},
    ],
}

STOPWORDS = set("a an the is are do does you your i can be how what within of to in for and or by long".split())


def words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}


def fake_llm(prompt):
    """Stand-in for a real LLM. DO NOT EDIT. See the module docstring for its behaviour."""
    question = prompt.split("QUESTION:")[1].strip()
    best_label, best_sentence, best_overlap = None, None, 0
    for label, text in re.findall(r"^\[(\d+)\] \(.*?\) (.*)$", prompt, flags=re.M):
        for sentence in re.split(r"(?<=\.)\s+", text):
            overlap = len(words(question) & words(sentence))
            if overlap > best_overlap:
                best_label, best_sentence, best_overlap = label, sentence, overlap
    if best_label is None:
        return "Yes, absolutely! We offer this and it usually arrives within 24 hours."   # hallucination
    return f"{best_sentence} [{best_label}]"


# ----------------------------------------------------------------------------------------------------------
# Code under test
# ----------------------------------------------------------------------------------------------------------
def select_context(chunks, max_words=MAX_CONTEXT_WORDS):
    """Keep the MOST relevant chunks that fit in the word budget."""
    ordered = sorted(chunks, key=lambda c: c["score"])
    selected, used = [], 0
    for chunk in ordered:
        n = len(chunk["text"].split())
        if used + n > max_words:
            break
        selected.append(chunk)
        used += n
    return selected


def build_prompt(question, chunks):
    sources = "\n".join(f"[{i}] ({c['source']}) {c['text']}" for i, c in enumerate(chunks))
    return (
        "SYSTEM: Answer ONLY from the sources below. Cite the source number like [1]. "
        f"If the sources do not contain the answer, reply exactly: {REFUSAL}\n\n"
        f"SOURCES:\n{sources}\n\n"
        f"QUESTION: {question}"
    )


def parse_citations(answer, chunks):
    """Map every [n] in the answer back to the chunk shown to the model as [n]."""
    return [chunks[int(n) - 1]["source"] for n in re.findall(r"\[(\d+)\]", answer)]


def answer_question(question, retrieved):
    if max(c["score"] for c in retrieved) > REFUSAL_THRESHOLD:
        return REFUSAL, [], []
    current = [c for c in retrieved if c["status"] != "archived"]      # never answer from archived policies
    context = select_context(retrieved)
    prompt = build_prompt(question, context)
    answer = fake_llm(prompt)
    return answer, parse_citations(answer, context), [c["id"] for c in context]


if __name__ == "__main__":
    for question, retrieved in RETRIEVED.items():
        answer, cited, context_ids = answer_question(question, retrieved)
        print(f"Q: {question}\nA: {answer}\n   cited sources: {cited}\n   chunks in prompt: {context_ids}\n")
