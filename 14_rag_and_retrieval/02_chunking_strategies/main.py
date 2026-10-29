"""
GOAL:  Split documents into chunks five different ways and SEE the trade-offs: chunk count, chunk size,
       sentences cut in half, and whether the answer to each evaluation question survives intact.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 14_rag_and_retrieval/02_chunking_strategies/main.py
EXPECTED OUTPUT (shortened):
    strategy                     chunks  avg words  min  max  cut sentences  answers intact
    fixed 200 chars                  ...
    heading-aware (## sections)      36        ...                    0          24/24
NEEDS: Python + numpy only.  Instant.
"""

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_documents, load_eval_set   # noqa: E402

docs = load_documents()


def body_of(doc):
    """Text without markdown heading lines (the fixed-size splitters do not understand structure)."""
    return " ".join(l.strip() for l in doc["text"].splitlines() if l.strip() and not l.startswith("#"))


def sentences_of(text):
    """Naive sentence splitter: break after . ! ? followed by a space. Good enough for clean prose."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


# ---- Strategy 1: fixed number of characters (what a naive splitter does) ----------------------------
def fixed_chars(text, size=200):
    return [text[i:i + size] for i in range(0, len(text), size)]


# ---- Strategy 2: fixed number of words WITH overlap ---------------------------------------------------
def fixed_words_overlap(text, size=40, overlap=10):
    words = text.split()
    step = size - overlap                                  # each new chunk starts `step` words later
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), step)]


# ---- Strategy 3: pack whole sentences until a word budget is reached -----------------------------------
def sentence_pack(text, max_words=40):
    chunks, current = [], []
    for sent in sentences_of(text):
        if current and len(" ".join(current + [sent]).split()) > max_words:
            chunks.append(" ".join(current))
            current = []
        current.append(sent)
    if current:
        chunks.append(" ".join(current))
    return chunks


# ---- Strategy 4: recursive splitting (LangChain's RecursiveCharacterTextSplitter idea) ----------------
def recursive_split(text, max_chars=250, separators=("\n\n", "\n", ". ", " ")):
    """Try the biggest natural boundary first; only fall back to smaller ones for pieces still too long."""
    if len(text) <= max_chars:
        return [text.strip()] if text.strip() else []
    sep, *rest = separators
    parts = text.split(sep)
    if len(parts) == 1:                                    # this separator does not occur: go finer
        return recursive_split(text, max_chars, tuple(rest)) if rest else fixed_chars(text, max_chars)
    chunks, current = [], ""
    for part in parts:
        candidate = current + sep + part if current else part
        if len(candidate) <= max_chars:
            current = candidate                            # merge small neighbours back together
        else:
            if current:
                chunks.append(current)
            current = part if len(part) <= max_chars else ""
            if len(part) > max_chars:
                chunks.extend(recursive_split(part, max_chars, tuple(rest)) if rest else fixed_chars(part, max_chars))
    if current:
        chunks.append(current)
    return [c.strip() for c in chunks if c.strip()]


# ---- Strategy 5: structure-aware -- one chunk per '## ' section ------------------------------------------
def heading_aware(doc):
    out = []
    for block in re.split(r"\n(?=## )", doc["text"]):
        lines = block.strip().splitlines()
        if lines and lines[0].startswith("## "):
            out.append(lines[0][3:] + ": " + " ".join(l.strip() for l in lines[1:] if l.strip()))
    return out


STRATEGIES = {
    "fixed 200 chars": lambda d: fixed_chars(body_of(d), 200),
    "fixed 40 words, overlap 10": lambda d: fixed_words_overlap(body_of(d), 40, 10),
    "sentence pack <=40 words": lambda d: sentence_pack(body_of(d), 40),
    "recursive <=250 chars": lambda d: recursive_split(d["text"].split("\n", 1)[1], 250),
    "heading-aware (## sections)": heading_aware,
}

all_sentences = {s for d in docs for s in sentences_of(body_of(d))}
eval_set = [q for q in load_eval_set() if q["gold"]]       # 24 answerable questions
doc_by_id = {d["doc_id"]: d for d in docs}

print(f"{'strategy':<30}{'chunks':>7}{'avg words':>11}{'min':>5}{'max':>5}{'cut sentences':>15}{'answers intact':>16}")
for name, split in STRATEGIES.items():
    chunks_per_doc = {d["doc_id"]: split(d) for d in docs}
    chunks = [c for cs in chunks_per_doc.values() for c in cs]
    sizes = np.array([len(c.split()) for c in chunks])
    # a sentence is "cut" if no single chunk contains all of it
    cut = sum(1 for s in all_sentences if not any(s in c for c in chunks))
    # the answer "survives" if one chunk of the gold document contains the gold sentence's key fact
    intact = 0
    for q in eval_set:
        gold_doc = q["gold"][0].split("#")[0]
        intact += any(q["answer"].lower() in c.lower() for c in chunks_per_doc[gold_doc])
    print(f"{name:<30}{len(chunks):>7}{sizes.mean():>11.1f}{sizes.min():>5}{sizes.max():>5}{cut:>15}"
          f"{f'{intact}/{len(eval_set)}':>16}")

# ---- Show WHY fixed-size chunks hurt: the refund rule is cut in half --------------------------------
print("\nfixed 200 chars, refunds document, first two chunks:")
for c in fixed_chars(body_of(doc_by_id["refunds_and_returns"]), 200)[:2]:
    print(f"  [{c}]")
print("\nheading-aware, same document, first chunk:")
print(f"  [{heading_aware(doc_by_id['refunds_and_returns'])[0]}]")

# ---- Overlap: the price you pay ------------------------------------------------------------------------
words = sum(len(body_of(d).split()) for d in docs)
for overlap in (0, 10, 20):
    stored = sum(len(c.split()) for d in docs for c in fixed_words_overlap(body_of(d), 40, overlap))
    print(f"overlap {overlap:>2} words -> {stored:5d} words stored for {words} real words "
          f"({stored / words:.2f}x: more vectors, more duplicate hits)")

# TRY IT: change fixed_chars size to 80, then 1000. Watch 'cut sentences' and 'max words'.
# TRY IT: change sentence_pack(..., 40) to 15 (and its label in STRATEGIES); how many chunks hold a single sentence?
# TRY IT: in heading_aware, drop the "section title + ': '" prefix. Which questions get harder to match later?
# BREAK IT: set overlap=40 with size=40 in fixed_words_overlap -> step = 0 -> ValueError: range() arg 3 must not be zero.
