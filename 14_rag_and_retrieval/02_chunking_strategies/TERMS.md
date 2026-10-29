# Terms — Chunking Strategies

Folder: `14_rag_and_retrieval/02_chunking_strategies` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [14 README](../README.md)

**Terms in this folder:** [Chunk](#chunk) · [Why chunk (embedding limit, one idea per vector)](#why-chunk-embedding-limit-one-idea-per-vector) · [Fixed-size chunking](#fixed-size-chunking) · [Chunk overlap](#chunk-overlap) · [Sentence splitting and packing](#sentence-splitting-and-packing) · [Recursive splitting](#recursive-splitting) · [Structure-aware (heading-aware) chunking](#structure-aware-heading-aware-chunking) · [Semantic chunking](#semantic-chunking) · [Late chunking](#late-chunking) · [Chunk size trade-off](#chunk-size-trade-off) · [Contextual header](#contextual-header) · [Parent–child retrieval](#parentchild-retrieval) · [Tables and code in chunking](#tables-and-code-in-chunking) · [Evaluation set (gold, answer)](#evaluation-set-gold-answer) · [Cut sentences and answers intact](#cut-sentences-and-answers-intact) · [Regex lookahead and lookbehind](#regex-lookahead-and-lookbehind) · [Slicing past the end](#slicing-past-the-end) · [range(start, stop, step)](#rangestart-stop-step) · [Star unpacking `sep, *rest`](#star-unpacking-sep-rest) · [Recursion](#recursion) · [lambda and a dict of strategies](#lambda-and-a-dict-of-strategies) · [any() and set comprehensions](#any-and-set-comprehensions) · [ValueError: range() arg 3 must not be zero](#valueerror-range-arg-3-must-not-be-zero)

---

## Chunk  `(concept)`
**1. Definition:** A chunk is a contiguous piece of a document that is embedded, indexed and retrieved as one unit.
**2. In plain words:** It is the "page" the LLM gets to read. Here the 9 documents become between 31 and 38 chunks depending on the strategy; heading-aware gives 36.
**3. Example to quote:** > "The chunk is both the unit of search and the unit of context, so it must match precisely and also make sense on its own."
**Seen in:** `main.py`, all strategies

## Why chunk (embedding limit, one idea per vector)  `(concept)`
**1. Definition:** Documents are chunked because embedding models accept a limited number of tokens, because one vector represents one averaged meaning, and because the LLM context budget is limited.
**2. In plain words:** all-MiniLM-L6-v2 was trained on up to 128 tokens and is truncated at 256 by default; a whole document's vector would be a blur of refunds, shipping and loyalty points.
**3. Example to quote:** > "A vector is an average; average too many topics and it matches nothing well."
**Seen in:** `WALKTHROUGH.md`, theory

## Fixed-size chunking  `(concept)`
**1. Definition:** Fixed-size chunking cuts text every N characters, words or tokens regardless of content.
**2. In plain words:** Fast and predictable, but blind: 200-character chunks cut 20 sentences and lost 1 of 24 answers (23/24). The refund rule broke at `origin` / `al payment method`.
**3. Example to quote:** > "Fixed-size splitting is a baseline; it cuts facts in half, so I only use it with overlap or as a last resort."
```python
def fixed_chars(text, size=200):
    return [text[i:i + size] for i in range(0, len(text), size)]
```
**Seen in:** `main.py`, strategy 1

## Chunk overlap  `(concept)`
**1. Definition:** Overlap makes consecutive chunks share their last/first M units, so each window starts `size - overlap` units after the previous one.
**2. In plain words:** A fact on a boundary appears whole in one of the two windows, if it is shorter than the overlap. It cost us 964 -> 1,204 words stored at overlap 10 (1.25x) and 1,644 at overlap 20 (1.71x). With 40-word windows and 10 overlap, 7 sentences were still cut.
**3. Example to quote:** > "Overlap of 10 to 20 percent is a common default, but it multiplies storage and fills the top-k with near-duplicates."
**Seen in:** `main.py`, strategy 2 and the overlap cost loop

## Sentence splitting and packing  `(concept)`
**1. Definition:** Sentence packing splits text into sentences and adds whole sentences to a chunk until a size budget would be exceeded.
**2. In plain words:** Never cuts a sentence (0 cut, 24/24) with 33 chunks. The regex splitter here is naive (abbreviations like "e.g." would fool it); libraries such as NLTK or spaCy split more robustly.
**3. Example to quote:** > "Packing whole sentences gives near-fixed sizes without ever breaking a sentence."
**Seen in:** `main.py`, strategy 3

## Recursive splitting  `(concept)`
**1. Definition:** Recursive splitting tries a list of separators from largest to smallest (paragraph, line, sentence, word), splitting further only those pieces that are still too long, and merges small neighbours up to the size limit.
**2. In plain words:** It is the idea behind LangChain's `RecursiveCharacterTextSplitter`. Here: 38 chunks, 1 cut sentence (where a piece had to fall back to splitting on spaces), 24/24.
**3. Example to quote:** > "Recursive splitting respects the biggest natural boundary available and only goes finer when it has to."
**Seen in:** `main.py`, strategy 4

## Structure-aware (heading-aware) chunking  `(concept)`
**1. Definition:** Structure-aware chunking uses document structure such as Markdown headings, HTML tags or PDF section titles as chunk boundaries.
**2. In plain words:** One chunk per `## ` section: 36 chunks (9 x 4), 0 cut sentences, 24/24, sizes 19 to 49 words. Each chunk is one topic the author defined. Long sections would need a second split.
**3. Example to quote:** > "When documents have headings, I chunk by section and fall back to recursive splitting inside long sections."
**Seen in:** `main.py`, strategy 5; `common.split_by_heading`

## Semantic chunking  `(concept)`
**1. Definition:** Semantic chunking embeds consecutive sentences and starts a new chunk where the similarity between neighbouring sentences drops below a threshold.
**2. In plain words:** It finds topic shifts even in text without headings, at the cost of embedding every sentence and tuning a threshold. Not implemented in this step.
**3. Example to quote:** > "Semantic chunking cuts where the meaning changes rather than where a counter runs out."
**Seen in:** `WALKTHROUGH.md`, theory

## Late chunking  `(concept)`
**1. Definition:** Late chunking runs a long-context embedding model over the whole document first, then mean-pools the token vectors of each chunk's span to form chunk embeddings.
**2. In plain words:** Each chunk vector was computed while the model could see the rest of the document, so "it" in a chunk can carry the meaning of the earlier noun. Needs a long-context model.
**3. Example to quote:** > "Late chunking embeds first and splits second, so chunk vectors keep document context."
**Seen in:** `WALKTHROUGH.md`, theory

## Chunk size trade-off  `(concept)`
**1. Definition:** Smaller chunks give more precise matches but less self-contained context; larger chunks give more context but diluted vectors and more irrelevant text in the prompt.
**2. In plain words:** At 1,000 characters we got 9 chunks averaging 107 words of mixed topics; at 80 characters, 74 chunks and 62 cut sentences. Common starting points are 200 to 500 tokens, then tune on an evaluation set.
**3. Example to quote:** > "There is no universal best chunk size; I pick a starting point and tune it against retrieval metrics on my own questions."
**Seen in:** `WALKTHROUGH.md`, Experiments and theory

## Contextual header  `(concept)`
**1. Definition:** A contextual header is text such as the document title and section heading prepended to a chunk before embedding (and often before showing it to the LLM).
**2. In plain words:** `heading_aware` prefixes `Refund window: `; from step 03 on `chunk_for_embedding` adds `Refunds and Returns - Refund window: `. Short chunks then still carry their topic.
**3. Example to quote:** > "I prepend title and section to every chunk so a chunk that only says 'within 9 days' still matches refund questions."
**Seen in:** `main.py`, `heading_aware`; `common.chunk_for_embedding`

## Parent–child retrieval  `(concept)`
**1. Definition:** Parent–child (small-to-big) retrieval indexes small chunks for matching but returns their larger parent section or document to the LLM.
**2. In plain words:** Search with a sharp knife, read with a wide lens. Store the `parent_id` with every small chunk.
**3. Example to quote:** > "I match on sentence-sized chunks and pass the whole section to the model."
**Seen in:** `WALKTHROUGH.md`, theory

## Tables and code in chunking  `(concept)`
**1. Definition:** Tables and source code have structure that generic text splitters break; they need dedicated handling.
**2. In plain words:** Keep a table whole or turn each row into a sentence with its column names; split code on function or class boundaries.
**3. Example to quote:** > "A table row without its header is just numbers, so I never split a table from its header."
**Seen in:** `WALKTHROUGH.md`, theory

## Evaluation set (gold, answer)  `(concept)`
**1. Definition:** An evaluation set is a list of questions with known correct sources (`gold` chunk ids) and expected answer strings, used to measure retrieval and answer quality.
**2. In plain words:** `data/eval_questions.json` has 29 questions; 24 have a gold chunk (the rest are deliberately unanswerable). This step uses them to check that each answer survives inside one chunk.
**3. Example to quote:** > "I judge chunking by whether answers to real questions survive, not by how the chunks look."
**Seen in:** `main.py`, `load_eval_set()`

## Cut sentences and answers intact  `(concept)`
**1. Definition:** Two metrics in this step: a sentence is "cut" if no single chunk contains all of it; an answer is "intact" if some chunk of the gold document contains the answer string.
**2. In plain words:** Heading-aware: 0 cut, 24/24 intact. Fixed 200 chars: 20 cut, 23/24 intact. These are cheap proxies; real retrieval quality is measured with recall@k in later steps.
**3. Example to quote:** > "Before any embedding I check that my splitter does not destroy the answers."
**Seen in:** `main.py`, results table

## Regex lookahead and lookbehind  `(syntax)`
**1. Definition:** `(?=X)` matches a position followed by X and `(?<=X)` a position preceded by X, without consuming X.
**2. In plain words:** `re.split(r"(?<=[.!?])\s+", t)` splits after punctuation but keeps the full stop with its sentence; `re.split(r"\n(?=## )", t)` splits before each heading but keeps the `## ` in the next block.
**3. Example to quote:** > "Lookarounds let me split at a boundary without throwing the boundary characters away."
**Seen in:** `main.py`, `sentences_of`, `heading_aware`

## Slicing past the end  `(syntax)`
**1. Definition:** A Python slice `s[i:j]` with `j` beyond the length returns everything up to the end without an error.
**2. In plain words:** That is why `fixed_chars` needs no special case for the last, shorter chunk, and why fixed 200 chars can produce a 1-word final chunk.
**3. Example to quote:** > "`'abcdef'[4:8]` is `'ef'`, no IndexError."
**Seen in:** `main.py`, `fixed_chars`, `fixed_words_overlap`

## range(start, stop, step)  `(standard library)`
**1. Definition:** `range(start, stop, step)` produces integers from `start` up to but not including `stop`, in increments of `step`; `step` must be non-zero.
**2. In plain words:** The window starts: 0, 30, 60, ... for size 40 and overlap 10.
**3. Example to quote:** > "`range(0, len(words), size - overlap)` gives the start of every overlapping window."
**Seen in:** `main.py`, strategies 1 and 2

## Star unpacking `sep, *rest`  `(syntax)`
**1. Definition:** Extended iterable unpacking assigns the first element to `sep` and the remaining elements, as a list, to `rest`.
**2. In plain words:** `sep` is the separator to try now, `rest` the finer ones to try next.
**3. Example to quote:** > "`sep, *rest = ('\n\n', '\n', '. ', ' ')` gives `sep='\n\n'` and `rest=['\n', '. ', ' ']`."
**Seen in:** `main.py`, `recursive_split`

## Recursion  `(concept)`
**1. Definition:** Recursion is a function calling itself on a smaller problem until a base case is reached.
**2. In plain words:** `recursive_split` stops when a piece fits (`len(text) <= max_chars`) and otherwise calls itself with the next, finer separator. If none are left it falls back to `fixed_chars`.
**3. Example to quote:** > "The base case is 'the piece already fits'; every recursive call uses a finer separator, so it always terminates."
**Seen in:** `main.py`, `recursive_split`

## lambda and a dict of strategies  `(syntax)`
**1. Definition:** `lambda args: expression` creates a small anonymous function; storing functions as dict values lets you loop over interchangeable implementations.
**2. In plain words:** Every strategy becomes "a function from a doc to a list of chunks", so one loop can compare all five.
**3. Example to quote:** > "A dict of name -> splitter function makes adding a sixth strategy a one-line change."
**Seen in:** `main.py`, `STRATEGIES`

## any() and set comprehensions  `(standard library)`
**1. Definition:** `any(iterable)` returns True if any element is truthy; `{expr for ...}` builds a set of unique values.
**2. In plain words:** `all_sentences` is the set of every sentence in the KB; `any(s in c for c in chunks)` asks "does any chunk contain this whole sentence?" Adding a boolean to an int counts True as 1.
**3. Example to quote:** > "`cut = sum(1 for s in all_sentences if not any(s in c for c in chunks))`."
**Seen in:** `main.py`, results loop

## ValueError: range() arg 3 must not be zero  `(concept)`
**1. Definition:** Python raises this `ValueError` when `range` is called with a step of 0.
**2. In plain words:** It happens when `overlap == size` in `fixed_words_overlap`, because `step = size - overlap = 0`. A negative step (overlap > size) gives no error but an empty list.
**3. Example to quote:** > "I validate `0 <= overlap < size` at the top of the splitter instead of trusting the caller."
**Seen in:** `WALKTHROUGH.md`, Debugging practice
