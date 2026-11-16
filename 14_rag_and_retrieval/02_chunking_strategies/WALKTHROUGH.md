# Step 02 — Chunking strategies

## The story so far
[Step 01](../01_documents_and_why_rag/WALKTHROUGH.md) loaded 9 clean documents with stable ids and showed that you cannot paste a knowledge base into every prompt: you must retrieve *pieces*. It ended with a question: what exactly is a piece?

## The pain point
The simplest answer, "every 200 characters", cuts text wherever the counter runs out. The refund rule "The money is returned to the original payment method within 5 working days" becomes `...returned to the origin` in one chunk and `al payment method within 5 working days...` in the next. Neither chunk holds the whole fact, so neither will be retrieved for it, and the LLM never sees it.

## The spark
Cut along the boundaries the author already wrote: sentences, paragraphs, sections. Compare five splitters on the same 9 documents with three numbers: how many sentences each one cuts in half, how big the chunks are, and whether the answer to each of the 24 evaluation questions survives inside one chunk.

## Goal
Implement fixed-character, fixed-word-with-overlap, sentence-packing, recursive and heading-aware chunking, measure their trade-offs, and see what overlap really costs.

## The theory (read this before the interview)
**Why chunk at all?**
- *Embedding input limit*: an embedding model reads a limited number of tokens. all-MiniLM-L6-v2 was trained on sequences of up to 128 tokens and sentence-transformers truncates its input at 256 by default; everything after that is ignored (proved in step 03).
- *One idea per vector*: a vector is an average of meaning. A whole document about refunds, shipping and loyalty points becomes a blurry vector that is "a bit similar" to everything and very similar to nothing.
- *LLM context budget*: you send k chunks to the LLM. Smaller chunks mean you can include more distinct sources for the same token budget.
- *Precision vs context*: the chunk you retrieve is also the text the LLM reads. It must be small enough to match precisely and large enough to be understood on its own.

**Strategies, from naive to clever.**
| Strategy | How it cuts | Strength | Weakness |
|---|---|---|---|
| Fixed characters / tokens | every N characters or tokens | trivial, predictable size | cuts words, sentences, facts |
| Fixed + overlap | windows of N that share M units with the previous one | a fact on a boundary appears whole in at least one window (if it is shorter than the overlap) | more vectors, duplicate hits |
| Sentence packing | add whole sentences until a budget is full | never cuts a sentence | ignores topic boundaries |
| Recursive | try `\n\n`, then `\n`, then `. `, then ` `; only go finer for pieces still too long | respects the biggest natural boundary available; the common default (LangChain's `RecursiveCharacterTextSplitter`) | sizes vary |
| Structure / heading-aware | one chunk per section (Markdown `##`, HTML `<h2>`, PDF headings) | chunks match the author's topics; heading gives context | sections can be huge or tiny; needs a second split for long sections |
| Semantic chunking | embed sentences, start a new chunk where similarity between neighbours drops | follows topic shifts without headings | slower (embeds everything twice), needs a threshold |
| Late chunking | run a long-context embedding model over the whole document first, then pool token vectors per chunk | each chunk vector "knows" its surrounding document | needs a long-context model with token-level output |

**Size trade-offs.** Small chunks are precise but lose context ("it is valid for 9 days": what is "it"?). Large chunks keep context but dilute the vector and waste the LLM's context on irrelevant sentences. Common starting points are roughly 200 to 500 tokens with 10 to 20% overlap, then **tune on your own evaluation set**; there is no universal best size.

**Contextual headers.** Prepend the document title and section heading to each chunk before embedding, for example `Refunds and Returns - Refund window: ...`. A chunk that says only "within 9 days" then still matches "refund" queries. (`common.chunk_for_embedding` does this from step 03 on.) A stronger variant asks an LLM to write a one-sentence context for each chunk.

**Special cases.**
- *Tables*: never split rows from their header; keep a table whole, or turn each row into a sentence that repeats the column names.
- *Code*: split on functions or classes (language-aware splitters), not on character counts.
- *Lists and FAQs*: one question with its answer per chunk.
- *Parent–child*: retrieve small chunks for precision, then pass their larger parent section to the LLM for context.

Always store `chunk_id`, `doc_id`, section and position with each chunk, so you can cite it and rebuild context.

## Run it
```bash
cd 14_rag_and_retrieval/02_chunking_strategies
python main.py       # instant, Python + numpy only
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Chunk vs passage vs segment vs node | split piece, snippet | Same idea | The unit of text that is embedded and retrieved; "node" is LlamaIndex wording |
| Chunk size vs window size | `chunk_size`, `max_words` | Related | Size is how long a chunk is; overlap is how much two neighbours share |
| Overlap vs stride | `overlap`, `step = size - overlap` | Related | Stride = how far the next chunk starts; overlap = size minus stride |
| Fixed-size vs recursive vs semantic vs structure-aware | splitter types | No | Cut by count; by natural boundaries in order; by meaning shifts; by headings |
| Recursive splitter vs LangChain `RecursiveCharacterTextSplitter` | same idea | Same idea | `recursive_split` here is a simplified teaching version |
| Heading-aware vs layout-aware | structure chunking | Related | Heading-aware uses Markdown `##`; layout-aware uses PDF or HTML structure (tables, columns) |

## Code, step by step

### 1. Docstring  (`main.py` lines 1–12)
```python
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
```
- **In simple words:** Explains what is compared: five ways to split text.
- **Exactly:** Columns printed: chunks, average words, min, max, cut sentences, answers intact. Needs only numpy.
- **Why it is here:** You will measure the trade-off, not just read about it.

### 2. Imports and loading  (`main.py` lines 14–23)
```python
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_documents, load_eval_set   # noqa: E402

docs = load_documents()
```
- **In simple words:** Load tools and the documents.
- **Exactly:** `from common import load_documents, load_eval_set` (module folder added to `sys.path`). `docs` is the list of 9 documents.
- **Why it is here:** All five strategies work on the same documents, so the comparison is fair.

### 3. body_of: text without headings  (`main.py` lines 26–28)
```python
def body_of(doc):
    """Text without markdown heading lines (the fixed-size splitters do not understand structure)."""
    return " ".join(l.strip() for l in doc["text"].splitlines() if l.strip() and not l.startswith("#"))
```
- **In simple words:** Flatten a document to one line of text without the `#` lines.
- **Exactly:** Keeps non-empty lines not starting with `#`, strips them and joins with spaces. Returns a `str`.
- **Why it is here:** Fixed-size splitters do not understand headings, so they get plain text.

### 4. sentences_of: a simple sentence splitter  (`main.py` lines 31–33)
```python
def sentences_of(text):
    """Naive sentence splitter: break after . ! ? followed by a space. Good enough for clean prose."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
```
- **In simple words:** Cut text into sentences.
- **Exactly:** `re.split(r"(?<=[.!?])\s+", text)`: split on whitespace that follows `.`, `!` or `?` (look-behind keeps the punctuation). Empty strings dropped.
- **Why it is here:** Needed to count "cut sentences" and for sentence packing. It is naive: "Dr. Smith" would split, fine for clean prose.

### 5. Strategy 1: fixed characters  (`main.py` lines 36–38)
```python
# ---- Strategy 1: fixed number of characters (what a naive splitter does) ----------------------------
def fixed_chars(text, size=200):
    return [text[i:i + size] for i in range(0, len(text), size)]
```
- **In simple words:** Cut every 200 characters, no matter what.
- **Exactly:** `text[i:i+size] for i in range(0, len(text), size)`. Chunks can end mid-word. Real: 31 chunks, 20 sentences cut, answers intact 23/24.
- **Why it is here:** The naive baseline; it shows the damage that structure-blind splitting does.

### 6. Strategy 2: fixed words with overlap  (`main.py` lines 41–45)
```python
# ---- Strategy 2: fixed number of words WITH overlap ---------------------------------------------------
def fixed_words_overlap(text, size=40, overlap=10):
    words = text.split()
    step = size - overlap                                  # each new chunk starts `step` words later
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), step)]
```
- **In simple words:** Cut every 40 words, but let neighbours share 10 words.
- **Exactly:** `step = size - overlap` (30). Chunk `i` covers words `[i, i+40)`. `range(0, max(len(words) - overlap, 1), step)` avoids a tiny last chunk that is fully covered by the previous one. Real: 33 chunks, 7 cut sentences, 24/24.
- **Why it is here:** Overlap reduces the chance that an answer is split across a boundary, at a storage cost (see the end).

### 7. Strategy 3: pack whole sentences  (`main.py` lines 48–58)
```python
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
```
- **In simple words:** Add sentences to a chunk until the word budget would be exceeded.
- **Exactly:** `current` is a list of sentences. If adding the next sentence exceeds `max_words=40`, close the chunk and start a new one. A sentence longer than the budget still becomes its own chunk. Real: 33 chunks, 0 cut sentences.
- **Why it is here:** Never cuts a sentence in half, and chunk sizes stay under a budget.

### 8. Strategy 4: recursive splitting  (`main.py` lines 61–83)
```python
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
```
- **In simple words:** Try the biggest natural break first, and only cut finer where a piece is still too long.
- **Exactly:** Separators in order: paragraph, line, sentence end, space. If the text fits `max_chars=250` return it. Otherwise split on the first separator; if it does not occur, recurse with the next one; if none left, fall back to `fixed_chars`. Small neighbours are merged back until adding the next would exceed the limit. Real: 38 chunks, 1 cut sentence.
- **Why it is here:** Keeps paragraphs together when it can, and always terminates because separators run out.

### 9. Strategy 5: one chunk per section  (`main.py` lines 86–93)
```python
# ---- Strategy 5: structure-aware -- one chunk per '## ' section ------------------------------------------
def heading_aware(doc):
    out = []
    for block in re.split(r"\n(?=## )", doc["text"]):
        lines = block.strip().splitlines()
        if lines and lines[0].startswith("## "):
            out.append(lines[0][3:] + ": " + " ".join(l.strip() for l in lines[1:] if l.strip()))
    return out
```
- **In simple words:** Use the document's own headings as chunk boundaries.
- **Exactly:** `re.split(r"\n(?=## )", ...)` splits before each `## `. Each chunk is `"Section title: body"`. Real: 36 chunks (9 documents times 4 sections), 0 cut sentences, 24/24, sizes 19 to 49 words.
- **Why it is here:** Humans wrote the headings to mark topics, so these are the most meaningful boundaries.

### 10. The strategy table  (`main.py` lines 96–102)
```python
STRATEGIES = {
    "fixed 200 chars": lambda d: fixed_chars(body_of(d), 200),
    "fixed 40 words, overlap 10": lambda d: fixed_words_overlap(body_of(d), 40, 10),
    "sentence pack <=40 words": lambda d: sentence_pack(body_of(d), 40),
    "recursive <=250 chars": lambda d: recursive_split(d["text"].split("\n", 1)[1], 250),
    "heading-aware (## sections)": heading_aware,
}
```
- **In simple words:** A menu of the five splitters under readable names.
- **Exactly:** A dict of name to function. Each lambda takes a document and returns a list of chunk strings. The recursive one removes the `# Title` line first.
- **Why it is here:** A loop can then run them all with the same code.

### 11. Reference data for the metrics  (`main.py` lines 104–106)
```python
all_sentences = {s for d in docs for s in sentences_of(body_of(d))}
eval_set = [q for q in load_eval_set() if q["gold"]]       # 24 answerable questions
doc_by_id = {d["doc_id"]: d for d in docs}
```
- **In simple words:** Prepare the facts needed to judge each strategy.
- **Exactly:** `all_sentences` is the set of every sentence in the corpus. `eval_set` keeps the 24 questions that have a gold chunk. `doc_by_id` maps id to document.
- **Why it is here:** Metrics need a ground truth.

### 12. Run all strategies and print the table  (`main.py` lines 108–121)
```python
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
```
- **In simple words:** Split everything with each strategy and measure it.
- **Exactly:** `chunks_per_doc` keeps the per-document chunks. `sizes` is words per chunk (mean, min, max). A sentence is "cut" if no single chunk contains it entirely. An answer is "intact" if the question's `answer` string occurs in one chunk of the gold document. Real table: fixed 200 chars 31 chunks, 20 cut, 23/24; fixed 40 words overlap 10: 33, 7, 24/24; sentence pack: 33, 0, 24/24; recursive: 38, 1, 24/24; heading-aware: 36, 0, 24/24.
- **Why it is here:** Numbers make the trade-off arguable in an interview.

### 13. Show a cut rule with your own eyes  (`main.py` lines 123–128)
```python
# ---- Show WHY fixed-size chunks hurt: the refund rule is cut in half --------------------------------
print("\nfixed 200 chars, refunds document, first two chunks:")
for c in fixed_chars(body_of(doc_by_id["refunds_and_returns"]), 200)[:2]:
    print(f"  [{c}]")
print("\nheading-aware, same document, first chunk:")
print(f"  [{heading_aware(doc_by_id['refunds_and_returns'])[0]}]")
```
- **In simple words:** Print the refund rule as fixed-size and as a section.
- **Exactly:** The fixed 200-char chunks end in "...returned to the origin" and begin with "al payment method...": the word "original" is cut in half. The heading-aware chunk keeps the full rule: `Refund window: ... within 5 working days after we approve the request.`
- **Why it is here:** One concrete example is more convincing than any statistic.

### 14. The price of overlap  (`main.py` lines 130–135)
```python
# ---- Overlap: the price you pay ------------------------------------------------------------------------
words = sum(len(body_of(d).split()) for d in docs)
for overlap in (0, 10, 20):
    stored = sum(len(c.split()) for d in docs for c in fixed_words_overlap(body_of(d), 40, overlap))
    print(f"overlap {overlap:>2} words -> {stored:5d} words stored for {words} real words "
          f"({stored / words:.2f}x: more vectors, more duplicate hits)")
```
- **In simple words:** Count how many words we store for each overlap setting.
- **Exactly:** Real: overlap 0 stores 964 words for 964 real words (1.00x), overlap 10 stores 1204 (1.25x), overlap 20 stores 1644 (1.71x). The formula is roughly `size / (size - overlap)`.
- **Why it is here:** More storage, more vectors, and duplicate hits in search results.

### 15. Exercises as comments  (`main.py` lines 137–140)
```python
# TRY IT: change fixed_chars size to 80, then 1000. Watch 'cut sentences' and 'max words'.
# TRY IT: change sentence_pack(..., 40) to 15 (and its label in STRATEGIES); how many chunks hold a single sentence?
# TRY IT: in heading_aware, drop the "section title + ': '" prefix. Which questions get harder to match later?
# BREAK IT: set overlap=40 with size=40 in fixed_words_overlap -> step = 0 -> ValueError: range() arg 3 must not be zero.
```
- **In simple words:** Ideas to try and break.
- **Exactly:** BREAK IT: `overlap=40` with `size=40` gives `step = 0`, so `range()` raises `ValueError: range() arg 3 must not be zero`.
- **Why it is here:** Read the error and you know the parameter relation.


## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `text[i:i + size]` | slice; safe past the end (returns what is left) | `"abcdef"[4:8]` -> `"ef"` |
| `range(0, n, step)` | start, stop, step; `step` must not be 0 | `range(0, 10, 3)` -> 0, 3, 6, 9 |
| `re.split(r"(?<=[.!?])\s+", text)` | split on whitespace **preceded by** `.`, `!` or `?` (lookbehind keeps the punctuation) | `"A. B."` -> `["A.", "B."]` |
| `re.split(r"\n(?=## )", text)` | split on a newline **followed by** `## ` (lookahead keeps the heading) | one block per section |
| `sep, *rest = separators` | unpack: first item, then the rest as a list | `("\n\n", "\n", ". ", " ")` |
| `recursive_split(...)` calling itself | recursion with a finer separator list | |
| `lambda d: fixed_chars(body_of(d), 200)` | a tiny anonymous function | the `STRATEGIES` dict |
| `{s for d in docs for s in ...}` | set comprehension (unique sentences) | `all_sentences` |
| `any(s in c for c in chunks)` | True if at least one chunk contains `s` | sentence "cut" test |
| `intact += any(...)` | True counts as 1, False as 0 | |
| `f"{name:<30}{n:>7}"` | left/right aligned columns | the results table |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `re.split(pattern, s)` | built-in `re` | split text on a regex | lookahead `(?=...)`, lookbehind `(?<=...)` | `list[str]` |
| `str.split()` | built-in | split on any whitespace (crude word count) | separator | `list[str]` |
| `str.split("\n", 1)[1]` | built-in | drop the first line (the `# Title`) | `maxsplit=1` | `str` |
| `" ".join(words)` | built-in | glue words back into text | – | `str` |
| `np.array(sizes).mean()/.min()/.max()` | numpy | chunk size statistics | – | numbers |
| `load_documents()` | `common.py` | step 01's loader | – | list of doc dicts |
| `load_eval_set()` | `common.py` | 29 questions with `gold` chunk ids and `answer` strings (24 answerable) | – | list of dicts |

## Execution trace (real run)
```
strategy                       chunks  avg words  min  max  cut sentences  answers intact
fixed 200 chars                    31       31.6    1   42             20           23/24
fixed 40 words, overlap 10         33       36.5   11   40              7           24/24
sentence pack <=40 words           33       29.2    6   39              0           24/24
recursive <=250 chars              38       28.5    3   46              1           24/24
heading-aware (## sections)        36       29.1   19   49              0           24/24
```
- **Fixed 200 chars** cuts 20 sentences and loses one answer (23/24). A 1-word minimum means some chunks are just a sentence's last fragment.
- **Overlap** rescues the answers (24/24) but still cuts 7 sentences, because overlap only helps facts shorter than the overlap.
- **Sentence pack** and **heading-aware** cut 0 sentences. Heading-aware also has the tightest size range (19 to 49 words) and one chunk per topic: 36 chunks = 9 documents x 4 sections.
- **Recursive** cuts 1 sentence: a piece longer than 250 characters had to fall back to the `" "` separator.
- The refund document under fixed 200 chars: chunk 1 ends `...returned to the origin`, chunk 2 starts `al payment method within 5 working days...`. Heading-aware keeps it whole: `Refund window: Refunds at Chai Kitchen are issued within 9 days ... within 5 working days after we approve the request.`
- **Overlap cost:** 964 real words become 964 / 1,204 / 1,644 stored words for overlap 0 / 10 / 20 (1.00x / 1.25x / 1.71x). Every extra word is extra embedding work, storage, and a chance that the top-k fills with near-duplicates.

## Experiments (change ONE thing)
1. `fixed_chars` size 80: 74 chunks, 62 cut sentences, 21/24 answers intact. Size 1000: 9 chunks (one per document), 0 cut, 24/24, but each chunk averages 107 words of mixed topics, which is the diluted-vector problem.
2. `sentence_pack(..., 15)`: 79 chunks and 78 of them hold a single sentence (the table label still says `<=40`; only the number changed). Sentences longer than the budget are kept whole, so the max is 22 words.
3. In `heading_aware`, drop the `section title + ": "` prefix: still 36 chunks, 0 cut, 24/24, only the average falls from 29.1 to 26.8 words. This table cannot see the damage. It shows up at retrieval time (step 04 onwards): a query phrased like the heading ("refund window", "international shipping") loses its best matching words. Expect that, then measure it.

## Debugging practice
BREAK IT: in `STRATEGIES`, call `fixed_words_overlap(body_of(d), 40, 40)` (overlap equal to size). Then `step = size - overlap = 0`:
```
  File ".../main.py", line 45, in fixed_words_overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), step)]
ValueError: range() arg 3 must not be zero
```
Read the traceback bottom-up: the error is in `range`, argument 3 is `step`, and `step` came from `size - overlap`. Validate inputs at the top of the function: `assert 0 <= overlap < size`. (Overlap larger than size gives a negative step and silently returns no chunks, which is worse.)

## Rebuild checklist
- [ ] `fixed_chars`: slices of `size`
- [ ] `fixed_words_overlap`: `step = size - overlap`, guard `overlap < size`
- [ ] `sentence_pack`: regex sentence split, add sentences until the budget, flush
- [ ] `recursive_split`: biggest separator first, merge small neighbours, recurse on long pieces
- [ ] `heading_aware`: split before `## `, keep the heading as a prefix
- [ ] measure: chunk count, size spread, cut sentences, answers intact

## The new problem (cliffhanger)
We now have 36 clean chunks, one per topic. But how does the computer decide that "How long can I wait before asking for my money back?" belongs to the "Refund window" chunk when they share almost no words? We need vectors that capture meaning, not spelling. Next: [step 03, embeddings with a bi-encoder](../03_embeddings_bi_encoder/WALKTHROUGH.md).

Deep dive: every 2026 chunking technique (token-based, semantic, late, contextual, hierarchical, propositions, layout-aware) with evaluation and LLD → [19_chunking_deep_dive](../../19_chunking_deep_dive/README.md); this step's recursive splitter is rebuilt and extended in [19/02](../../19_chunking_deep_dive/02_recursive_splitter_rebuilt/WALKTHROUGH.md)

> **Say it aloud (interview box):** I chunk because embedding models truncate long input and because one vector should represent one idea, while the LLM needs each chunk to make sense on its own. I prefer structure-aware or recursive splitting that never cuts a sentence, add a contextual header with the document title and section, and use 10 to 20 percent overlap only when I must use fixed windows, since overlap multiplies storage and duplicate hits. I tune chunk size on an evaluation set rather than trusting a default, and treat tables and code specially.
