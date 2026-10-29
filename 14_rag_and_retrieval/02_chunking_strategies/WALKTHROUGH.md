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

> **Say it aloud (interview box):** I chunk because embedding models truncate long input and because one vector should represent one idea, while the LLM needs each chunk to make sense on its own. I prefer structure-aware or recursive splitting that never cuts a sentence, add a contextual header with the document title and section, and use 10 to 20 percent overlap only when I must use fixed windows, since overlap multiplies storage and duplicate hits. I tune chunk size on an evaluation set rather than trusting a default, and treat tables and code specially.
