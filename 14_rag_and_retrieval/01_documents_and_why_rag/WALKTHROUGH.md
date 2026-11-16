# Step 01 — Documents, and why RAG exists

## The story so far
In [07_prompt_engineering/08_minimal_rag](../../07_prompt_engineering/08_minimal_rag/WALKTHROUGH.md) you built RAG in 20 lines: six one-sentence documents, TF-IDF vectors, top-k by cosine, paste into the prompt. It worked because the documents were tiny, clean and already split. This module rebuilds that toy piece by piece until it is something you could defend in production. It starts with the raw material: the documents.

## The pain point
Real knowledge does not arrive as six tidy sentences. It arrives as HTML with `&nbsp;` and `<div>` tags, PDFs with "Page 3 of 12" on every page, and wikis with thousands of pages. And the obvious shortcut, "just paste all of it into the prompt", fails on arithmetic: a mid-size wiki is millions of tokens, and even a small knowledge base pasted into every request makes every request expensive.

## The spark
Treat documents as data. Load each file into a record with a stable id and metadata, clean away boilerplate, and count everything in **tokens**, the unit context windows and bills use. Then do the sums: 9 small documents are already 1,502 tokens, so 20,000 pages would be about 3.3M tokens. You cannot send everything. You must **retrieve** the few pieces that matter. That is the R in RAG.

## Goal
Load the Chai Kitchen knowledge base (9 markdown files) with metadata, clean a messy HTML snippet, count tokens with the embedding model's tokenizer, and compute why "stuff everything into the prompt" does not scale.

## The theory (read this before the interview)
**What RAG is.** Retrieval-Augmented Generation answers a question in three moves:
1. **Retrieve**: search an external store for the passages most relevant to the question.
2. **Augment**: put those passages into the prompt as context, with an instruction such as "answer only from the context".
3. **Generate**: the LLM writes the answer, ideally citing which passage it used.
The model's weights never change. You change what it reads.

**Why an LLM needs it.**
- *Knowledge cutoff*: the model only knows what was in its training data up to some date.
- *Private data*: it never saw your company's policies, tickets or contracts.
- *Hallucination*: asked about something it does not know, it produces the most typical-sounding answer instead of "I don't know". Grounding it in retrieved text reduces this (it does not eliminate it).
- *Citations and traceability*: each answer can point to a `doc_id`, so a human can check it.
- *Access control*: you can filter retrieval by the user's permissions, so the model only sees documents that user may read. Weights cannot "forget" for one user.

**RAG vs fine-tuning vs long context.**

| | RAG | Fine-tuning | Long context (paste everything) |
|---|---|---|---|
| Freshness | update the index in minutes | retrain to add facts | always fresh, but you resend it every call |
| Cost per query | small prompt (a few chunks) | small prompt | huge prompt, paid on every request |
| Up-front cost | build an index | training run + evaluation | none |
| Teaches | **knowledge** (facts to look up) | **behaviour** (style, format, task skill) | knowledge, while it fits |
| Citations / access control | natural | hard | possible but you send everything |
| Scale limit | millions of documents | facts blur into weights | context window size; quality can drop for facts buried mid-context |

Rule of thumb: fine-tune for *how* to answer, retrieve for *what* to answer. They combine well.

**Two pipelines.**
```
OFFLINE (indexing, runs when documents change)
  sources -> parse -> clean -> chunk -> embed -> store vectors + text + metadata in an index

ONLINE (query, runs per request)
  question -> embed -> search index (top-k) -> [re-rank] -> build prompt -> LLM -> answer + citations
```

**Ingestion concerns (where real projects spend their time).**
- *Parsing*: PDFs lose reading order, tables and headings; HTML carries navigation and scripts. Garbage in, garbage retrieved.
- *Cleaning*: remove boilerplate (headers, footers, "Page 3 of 12", cookie banners) or it gets into every chunk and makes unrelated chunks look similar.
- *Metadata*: id, title, source path, section, date, owner, permissions. Used for citations, filtering and freshness.
- *Deduplication*: the same paragraph copied across pages fills the top-k with duplicates.
- *Stable ids*: derive ids from the source (file name, URL), not from list position, so an update replaces the right record.
- *Incremental re-indexing*: re-embed only new or changed documents (compare a content hash) and delete vectors of removed ones.
- *Tokens, not words*: budgets are in tokens. English averages a little more than one token per word; other languages and code can use many more.

## Run it
```bash
cd 14_rag_and_retrieval/01_documents_and_why_rag
python main.py       # ~5 s, loads only the cached MiniLM tokenizer
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Document vs source vs record | file, page, item | Related | A document is one whole file we load; "source" is its path kept as metadata for citations |
| Metadata vs payload vs fields | doc_id, title, source | Same idea | Extra facts stored next to the text; vector DBs call them payload |
| Token vs word vs character | tokenisation | No | A token is a model-specific piece (often a word piece); the knowledge base here is 1,502 MiniLM tokens for about 964 words |
| Context window vs max sequence length | limit | Related | Context window = how much an LLM can read; max sequence length (256 for this embedder) = how much the embedder reads before cutting |
| Cleaning vs normalising vs preprocessing | text prep | Overlapping | Cleaning removes junk (HTML, boilerplate); normalising makes forms consistent (whitespace, unicode) |
| RAG vs "stuff everything in the prompt" | long-context prompting | No | RAG sends only the retrieved pieces; stuffing sends all text every time |

## Code, step by step

### 1. Docstring: goal, run, expected output  (`main.py` lines 1–13)
```python
"""
GOAL:  Load a private knowledge base, clean it, attach metadata, and count its tokens.  Then do the
       arithmetic that motivates RAG: why not paste every document into every prompt?

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 14_rag_and_retrieval/01_documents_and_why_rag/main.py
EXPECTED OUTPUT (shortened):
    loaded 9 documents from data/kb
    cleaned text: 'Refund policy: Refunds are issued within 9 days. Contact help@... for help.'
    whole knowledge base: ~1,5xx tokens
    a company wiki of 20,000 such pages: ~3.xM tokens -> does not fit any context window
NEEDS: transformers (only the cached MiniLM tokenizer).  A few seconds.
"""
```
- **In simple words:** A note on what the file does and what it prints.
- **Exactly:** Goal: load a knowledge base, clean it, count tokens, then do the arithmetic that motivates RAG. Needs only the cached MiniLM tokenizer.
- **Why it is here:** Orients you before the code.

### 2. Imports and path setup  (`main.py` lines 15–23)
```python
import html
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import KB_DIR, EMBED_MODEL   # noqa: E402

from transformers import AutoTokenizer   # noqa: E402
```
- **In simple words:** Load tools, and tell Python where `common.py` lives.
- **Exactly:** `html`, `re`, `sys`, `Path` are standard library. `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` adds the module folder `14_rag_and_retrieval` to the import path. `from common import KB_DIR, EMBED_MODEL` takes the knowledge-base folder and the model name. `AutoTokenizer` comes from `transformers`.
- **Why it is here:** Folder names starting with digits cannot be imported as packages, so the path is inserted by hand.

### 3. Load every file as a record with metadata  (`main.py` lines 25–35)
```python
# ---- 1. Load: every file becomes a dict with text + metadata ------------------------------------------
docs = []
for path in sorted(KB_DIR.glob("*.md")):                 # sorted -> same order every run
    text = path.read_text(encoding="utf-8")
    docs.append({
        "doc_id": path.stem,                              # stable id: used for citations and updates
        "title": text.splitlines()[0].lstrip("# ").strip(),
        "source": str(path.relative_to(KB_DIR.parents[1])),
        "n_sections": text.count("\n## "),
        "text": text,
    })
```
- **In simple words:** Read each Markdown file and put its text with a few facts about it into a dictionary.
- **Exactly:** `sorted(KB_DIR.glob("*.md"))` gives a fixed order. `path.stem` is the file name without `.md`. Title is the first line minus `# `. `source` is the path relative to the repo module folder. `text.count("\n## ")` counts sections. `read_text(encoding="utf-8")` is explicit, so Windows defaults cannot corrupt characters. Result: a list of 9 dicts.
- **Why it is here:** The `doc_id` and `source` let you cite where an answer came from; sorting keeps results identical on every machine.

### 4. Print what was loaded  (`main.py` lines 36–38)
```python
print(f"loaded {len(docs)} documents from {KB_DIR.relative_to(KB_DIR.parents[1])}")
for d in docs:
    print(f"  {d['doc_id']:<28} title={d['title']!r:<34} sections={d['n_sections']}")
```
- **In simple words:** Show the list of documents.
- **Exactly:** `loaded 9 documents from data/kb` then one line per document, each with `sections=4`. The `!r` format shows the title with quotes; `:<28` pads to 28 characters.
- **Why it is here:** A quick sanity check that loading worked before anything is built on it.

### 5. A deliberately messy sample  (`main.py` lines 40–43)
```python
# ---- 2. Clean: real documents come from HTML, PDFs and wikis and are messy ----------------------------
messy = """<div class="policy"><h2>Refund&nbsp;policy:</h2>   Refunds are issued
within 9 days.   Contact <a href="mailto:help@chaikitchen.example">help@chaikitchen.example</a>
for help.</div>   Page 3 of 12   """
```
- **In simple words:** A small dirty text like real web pages produce.
- **Exactly:** A string with HTML tags, an `&nbsp;` entity, extra spaces and the footer "Page 3 of 12".
- **Why it is here:** Real data is never clean; this shows what cleaning must handle.

### 6. The clean function, five steps  (`main.py` lines 46–52)
```python
def clean(raw: str) -> str:
    text = html.unescape(raw)                             # &nbsp; -> real character
    text = re.sub(r"<[^>]+>", " ", text)                  # drop HTML tags, keep their inner text
    text = re.sub(r"Page \d+ of \d+", " ", text)          # boilerplate that would pollute every chunk
    text = text.replace("\u00a0", " ")                    # non-breaking spaces -> normal spaces
    text = re.sub(r"\s+", " ", text)                      # collapse all whitespace runs
    return text.strip()
```
- **In simple words:** Remove the junk in five small steps.
- **Exactly:** `html.unescape` turns `&nbsp;` into a non-breaking space character. `re.sub(r"<[^>]+>", " ", ...)` replaces tags with a space (keeps inner text). `Page \d+ of \d+` removes the footer. `replace(" ", " ")` turns non-breaking spaces into plain ones. `re.sub(r"\s+", " ", ...)` collapses all whitespace runs, then `strip()`.
- **Why it is here:** Without it every chunk would start with tags and end with "Page 3 of 12", polluting embeddings and search.

### 7. Show before and after  (`main.py` lines 55–56)
```python
print("\nraw text    :", repr(messy[:70]) + "...")
print("cleaned text:", repr(clean(messy)))
```
- **In simple words:** Print the messy and the clean version.
- **Exactly:** Real output: `cleaned text: 'Refund policy: Refunds are issued within 9 days. Contact help@chaikitchen.example for help.'`
- **Why it is here:** Always look at your cleaned text: that is the cheapest RAG debugging there is.

### 8. Count tokens  (`main.py` lines 58–65)
```python
# ---- 3. Count tokens: the unit that context windows and API bills are measured in ---------------------
tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
token_counts = {d["doc_id"]: len(tok(d["text"])["input_ids"]) for d in docs}
total = sum(token_counts.values())
print(f"\nwhole knowledge base: {total:,} tokens (MiniLM tokenizer; an LLM tokenizer gives similar numbers)")
print(f"largest document    : {max(token_counts, key=token_counts.get)} with {max(token_counts.values())} tokens")
print(f"tokenizer limit: {tok.model_max_length} tokens, but sentence-transformers truncates this model at 256 "
      "(it was trained on 128) -> longer text is silently cut off")
```
- **In simple words:** Count how many tokens the whole knowledge base has.
- **Exactly:** `AutoTokenizer.from_pretrained(EMBED_MODEL)` loads MiniLM's WordPiece tokenizer. `tok(text)["input_ids"]` is the token list (including `[CLS]` and `[SEP]`). Real: total `1,502` tokens, largest document `product_catalog` with 231. `model_max_length` prints 512, but sentence-transformers truncates at 256.
- **Why it is here:** Tokens are the unit of context windows and bills; text beyond the embedder's limit is silently cut off.

### 9. Scale up to a company wiki  (`main.py` lines 67–70)
```python
# ---- 4. Why not just paste everything into the prompt? -------------------------------------------------
pages = 20_000                                              # a mid-size company wiki
wiki_tokens = total / len(docs) * pages
print(f"\na company wiki of {pages:,} such pages: ~{wiki_tokens / 1e6:.1f}M tokens -> does not fit any context window")
```
- **In simple words:** How big would a real company's wiki be?
- **Exactly:** Average tokens per document (`1502 / 9`, about 167) times 20,000 pages = about 3.3M tokens (real output). Larger than any context window.
- **Why it is here:** Shows that "just paste everything" cannot work at real scale.

### 10. Cost of stuffing vs retrieving  (`main.py` lines 72–79)
```python
price_per_m = 3.0                                           # illustrative $ per million input tokens
queries_per_day = 10_000
stuff_cost = total * queries_per_day * price_per_m / 1e6
rag_tokens = 3 * 120                                        # RAG: about 3 chunks of ~120 tokens
rag_cost = rag_tokens * queries_per_day * price_per_m / 1e6
print(f"even our tiny KB, pasted into {queries_per_day:,} prompts/day: ${stuff_cost:,.0f}/day "
      f"vs RAG with 3 chunks: ${rag_cost:,.0f}/day")
print("=> retrieve only the few relevant pieces; that is the R in RAG.")
```
- **In simple words:** Compare the daily cost of sending everything with sending 3 chunks.
- **Exactly:** Illustrative price of $3 per million input tokens and 10,000 queries a day: stuffing the whole KB costs `1502 * 10000 * 3 / 1e6 = $45/day`; RAG with 3 chunks of about 120 tokens costs `360 * 10000 * 3 / 1e6 = $11/day` (real output: 45 and 11, rounded). The price is illustrative; check current provider pricing.
- **Why it is here:** Even a tiny KB shows the cost and latency gap that grows with size: that is the R in RAG.

### 11. Exercises as comments  (`main.py` lines 81–85)
```python
# TRY IT: add a new file data/kb/gift_cards.md with a '# Gift Cards' title and two '## ' sections; re-run.
# TRY IT: change `pages` to 500. Would 'stuff everything' now fit a 200k-token context window? Is it still wise?
# BREAK IT: remove `sorted(...)` around glob - the order may change between machines, so ids/positions are unstable.
# BREAK IT: add a '₹' sign to a KB file and read it with encoding="cp1252" (a common Windows default): no error,
#           just mojibake ('â‚¹'). With encoding="ascii" you get UnicodeDecodeError. Always pass encoding="utf-8".
```
- **In simple words:** Ideas to try and to break.
- **Exactly:** TRY IT: add a file; change `pages`. BREAK IT: drop `sorted`, or read with the wrong encoding (mojibake `â‚¹`, or `UnicodeDecodeError` with ascii).
- **Why it is here:** Encoding bugs are silent; knowing them saves hours.


## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `sorted(KB_DIR.glob("*.md"))` | all `.md` files, in a fixed alphabetical order | `brewing_guide.md` before `customer_support.md` |
| `path.stem` | file name without folder and extension | `refunds_and_returns.md` -> `refunds_and_returns` |
| `text.splitlines()[0].lstrip("# ").strip()` | first line, strip leading `#` and spaces | `"# Brewing Guide"` -> `"Brewing Guide"` |
| `text.count("\n## ")` | count section headings | 4 for every file here |
| `path.relative_to(KB_DIR.parents[1])` | path relative to the module folder | `data/kb/brewing_guide.md` |
| `f"{d['doc_id']:<28}"` | left-align in a 28-character column | table output |
| `{x!r}` | `repr()` inside an f-string, shows quotes | `'Brewing Guide'` |
| `re.sub(r"<[^>]+>", " ", text)` | replace every HTML tag with a space | `<h2>Refund</h2>` -> ` Refund ` |
| `re.sub(r"\s+", " ", text)` | collapse runs of whitespace (spaces, newlines, tabs) | `"a \n  b"` -> `"a b"` |
| `{k: v for ...}` | dict comprehension | `token_counts` |
| `max(d, key=d.get)` | the key with the largest value | `product_catalog` |
| `20_000` | underscores in number literals for readability | same as `20000` |
| `f"{x:,.0f}"` | thousands separator, no decimals | `45.06` -> `45` |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `Path.glob(pattern)` | built-in `pathlib` | list matching files (arbitrary order) | `"*.md"` | generator of `Path` |
| `Path.read_text(encoding=...)` | built-in `pathlib` | read a whole file as text | always pass `encoding="utf-8"` | `str` |
| `html.unescape(s)` | built-in `html` | turn HTML entities into characters | – | `str` (`&nbsp;` -> U+00A0) |
| `re.sub(pattern, repl, s)` | built-in `re` | regex replace | pattern, replacement | `str` |
| `str.replace(old, new)` | built-in | literal replace (here: non-breaking space -> space) | – | `str` |
| `AutoTokenizer.from_pretrained(name)` | transformers | load the tokenizer that matches a model | model name (cached offline) | tokenizer |
| `tok(text)["input_ids"]` | transformers | text -> token ids (adds `[CLS]`/`[SEP]`) | `truncation`, `max_length` | dict of lists |
| `tok.model_max_length` | transformers | max input length the tokenizer reports | – | `int` (512 here) |

## Execution trace (real run)
- `loaded 9 documents from data/kb`: from `allergens_and_ingredients` to `wholesale_and_b2b`, each with `sections=4` (so 36 sections in total, used in step 02).
- Cleaning: `'<div class="policy"><h2>Refund&nbsp;policy:</h2>   Refunds are issued\n'...` becomes `'Refund policy: Refunds are issued within 9 days. Contact help@chaikitchen.example for help.'` Tags, entity, page footer and extra whitespace are gone; the link text survives.
- `whole knowledge base: 1,502 tokens`, `largest document: product_catalog with 231 tokens`.
- `tokenizer limit: 512 tokens, but sentence-transformers truncates this model at 256 (it was trained on 128)`. 512 is what the tokenizer reports (BERT's position limit). sentence-transformers truncates this model at 256 by default (step 03).
- `a company wiki of 20,000 such pages: ~3.3M tokens -> does not fit any context window`.
- Cost at an illustrative $3 per million input tokens and 10,000 queries/day: pasting the whole KB `$45/day` vs RAG with 3 chunks of ~120 tokens `$11/day`. The gap grows linearly with the size of the knowledge base; the RAG cost does not.

## Experiments (change ONE thing)
1. Add `data/kb/gift_cards.md` with a `# Gift Cards` title and two `## ` sections: you get 10 documents, the new one with `sections=2`, and a higher token total. (Delete it afterwards: later steps and the eval set assume 9 files.)
2. `pages = 500`: 1,502 / 9 x 500 is about 83,000 tokens, which fits a 200k context window. It is still a bad idea: you pay for all of it on every query, latency grows, and models can miss facts buried in the middle of a long prompt.
3. Remove the `Page \d+ of \d+` line from `clean()`: the footer survives in the cleaned text. In a real PDF it would appear in every chunk.

## Debugging practice
BREAK IT 1: remove `sorted(...)` around `KB_DIR.glob("*.md")`. No error. On this Mac the order became:
```
loaded 9 documents from data/kb
  customer_support             title='Customer Support'                 sections=4
  privacy_and_data             title='Privacy and Your Data'            sections=4
  refunds_and_returns          title='Refunds and Returns'              sections=4
  ...
```
The order depends on the file system, so "document number 3" means different things on different machines. Anything keyed by list position (vector row -> document) breaks silently. Fix: sort, and key everything by `doc_id`.

BREAK IT 2: read a file containing `₹` without `encoding="utf-8"`. The KB files have no `₹` today, so this was tested on a scratch file containing `Masala chai costs ₹250 per pouch.`:
- with `encoding="ascii"`: `UnicodeDecodeError: 'ascii' codec can't decode byte 0xe2 in position 37: ordinal not in range(128)`
- with `encoding="cp1252"` (a common Windows default): **no error**, the text silently becomes `'Masala chai costs â‚¹250 per pouch.'` (mojibake). That is worse than a crash: the wrong text gets embedded. Always pass `encoding="utf-8"`.

## Rebuild checklist
- [ ] `sorted(glob)` and a stable `doc_id` from the file name
- [ ] record = `doc_id`, `title`, `source`, section count, `text`
- [ ] `clean()`: unescape entities, strip tags, strip boilerplate, collapse whitespace
- [ ] count tokens with the tokenizer, not words
- [ ] do the context-window and cost arithmetic out loud

## The new problem (cliffhanger)
We know we must retrieve *pieces*, not whole documents. But what is a piece? A fixed number of characters, a sentence, a section? Cut badly and the refund rule is split across two pieces, so no single piece contains the answer. Next: [step 02, chunking strategies](../02_chunking_strategies/WALKTHROUGH.md).

> **Say it aloud (interview box):** RAG retrieves the few passages relevant to a question, adds them to the prompt and lets the LLM answer from them, which gives it fresh and private knowledge, fewer hallucinations, citations and per-user access control without retraining. I would fine-tune to change behaviour and use RAG to supply knowledge, because pasting a whole knowledge base into every prompt does not fit the context window at scale and costs money on every request. Most of the real work is ingestion: parsing, cleaning boilerplate, metadata, stable ids and incremental re-indexing.
