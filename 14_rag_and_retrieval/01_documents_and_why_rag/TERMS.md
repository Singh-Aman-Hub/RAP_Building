# Terms — Documents and Why RAG

Folder: `14_rag_and_retrieval/01_documents_and_why_rag` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [14 README](../README.md)

**Terms in this folder:** [RAG (retrieval-augmented generation)](#rag-retrieval-augmented-generation) · [Knowledge base](#knowledge-base) · [Knowledge cutoff](#knowledge-cutoff) · [Hallucination and grounding](#hallucination-and-grounding) · [RAG vs fine-tuning vs long context](#rag-vs-fine-tuning-vs-long-context) · [Indexing pipeline vs query pipeline](#indexing-pipeline-vs-query-pipeline) · [Document metadata](#document-metadata) · [Stable document id (doc_id)](#stable-document-id-doc_id) · [sys.path.insert and common.py](#syspathinsert-and-commonpy) · [pathlib: glob, stem, relative_to](#pathlib-glob-stem-relative_to) · [sorted(glob)](#sortedglob) · [read_text(encoding="utf-8")](#read_textencodingutf-8) · [Text cleaning and boilerplate](#text-cleaning-and-boilerplate) · [html.unescape](#htmlunescape) · [re.sub](#resub) · [Non-breaking space](#non-breaking-space) · [Token](#token) · [AutoTokenizer.from_pretrained](#autotokenizerfrom_pretrained) · [model_max_length and truncation](#model_max_length-and-truncation) · [Context window](#context-window) · [Cost per query arithmetic](#cost-per-query-arithmetic) · [Incremental re-indexing](#incremental-re-indexing)

---

## RAG (retrieval-augmented generation)  `(concept)`
**1. Definition:** RAG is a pattern in which a system retrieves passages relevant to a query from an external store, inserts them into the LLM prompt as context, and has the model generate an answer grounded in them.
**2. In plain words:** The model does not memorise your documents; it reads the right page just before answering. Here even the tiny KB is 1,502 tokens, so we send only about 3 chunks (~360 tokens) instead of everything.
**3. Example to quote:** > "RAG is retrieve, augment, generate: find the few relevant chunks, paste them into the prompt, and let the model answer from them with citations."
**Seen in:** `main.py`, section 4; `WALKTHROUGH.md`, theory

## Knowledge base  `(concept)`
**1. Definition:** The knowledge base is the collection of source documents a RAG system can retrieve from.
**2. In plain words:** For this module it is 9 markdown files about the made-up shop Chai Kitchen in `data/kb`, each with a title and 4 sections.
**3. Example to quote:** > "My knowledge base is 9 documents, 36 sections, 1,502 tokens; a real company wiki would be thousands of pages."
**Seen in:** `main.py`, section 1

## Knowledge cutoff  `(concept)`
**1. Definition:** The knowledge cutoff is the date after which no data was included in a model's training set.
**2. In plain words:** Anything newer than that date, or anything private, the model simply never saw. Retrieval brings it in at question time.
**3. Example to quote:** > "Prices change weekly, but the model's knowledge stopped at its cutoff, so I retrieve the current price list instead."
**Seen in:** `WALKTHROUGH.md`, theory

## Hallucination and grounding  `(concept)`
**1. Definition:** A hallucination is fluent output that is not supported by facts; grounding means constraining the answer to provided source text.
**2. In plain words:** Without context a model answers "refunds within 30 days" because that is typical. With the refund chunk in the prompt it can read "9 days". Grounding reduces hallucination but does not guarantee zero.
**3. Example to quote:** > "I instruct the model to answer only from the retrieved context and to say 'I don't know' if the context lacks the answer."
**Seen in:** `WALKTHROUGH.md`, theory

## RAG vs fine-tuning vs long context  `(concept)`
**1. Definition:** Three ways to give a model new information: retrieve it per query (RAG), change the weights by training (fine-tuning), or put all of it in the prompt (long context).
**2. In plain words:** RAG is cheap to update and gives citations; fine-tuning is best for behaviour, style and format; long context is simple but you pay for every token every time. Our numbers: whole KB in 10,000 prompts/day costs $45/day vs $11/day with RAG at $3 per million tokens.
**3. Example to quote:** > "Fine-tune for how to answer, retrieve for what to answer."
**Seen in:** `WALKTHROUGH.md`, theory

## Indexing pipeline vs query pipeline  `(concept)`
**1. Definition:** The indexing (offline) pipeline parses, cleans, chunks, embeds and stores documents; the query (online) pipeline embeds a question, searches the index, builds a prompt and calls the LLM.
**2. In plain words:** One runs when documents change, the other on every user request. This step is the very start of the offline side.
**3. Example to quote:** > "Indexing is a batch job; querying is latency-critical, so everything expensive belongs offline."
**Seen in:** `WALKTHROUGH.md`, theory

## Document metadata  `(concept)`
**1. Definition:** Metadata is structured information stored alongside a document's text, such as id, title, source, date or permissions.
**2. In plain words:** It powers citations ("from Refunds and Returns"), filtering (only this user's documents) and updates. Each record here has `doc_id`, `title`, `source`, `n_sections` and `text`.
**3. Example to quote:** > "I keep `source` and `doc_id` next to every chunk so the answer can cite where it came from."
**Seen in:** `main.py`, section 1

## Stable document id (doc_id)  `(concept)`
**1. Definition:** A stable id is an identifier derived from the document itself (file name, URL, database key) that does not change between runs.
**2. In plain words:** `path.stem` gives `refunds_and_returns`. If you used "document number 6" instead, re-ordering files would point vectors at the wrong documents.
**3. Example to quote:** > "Ids come from the file name, not the list position, so re-indexing replaces the right record."
**Seen in:** `main.py`, section 1

## sys.path.insert and common.py  `(standard library)`
**1. Definition:** `sys.path.insert(0, folder)` adds a folder to the front of Python's module search path so `import` can find files there.
**2. In plain words:** It lets every step folder import `KB_DIR` and `EMBED_MODEL` from the shared `14_rag_and_retrieval/common.py`. `Path(__file__).resolve().parents[1]` is "the folder above this script". Importing `common` also sets `HF_HUB_OFFLINE=1`.
**3. Example to quote:** > "Shared constants live in common.py; each step adds the module folder to sys.path and imports them."
```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import KB_DIR, EMBED_MODEL
```
**Seen in:** `main.py`, imports

## pathlib: glob, stem, relative_to  `(standard library)`
**1. Definition:** `pathlib.Path` represents file paths; `.glob("*.md")` finds matching files, `.stem` is the name without extension, `.relative_to(base)` strips a base folder.
**2. In plain words:** Three one-liners that turn a folder into a list of records: find the files, name them, record where they came from (`data/kb/brewing_guide.md`).
**3. Example to quote:** > "`path.stem` gives me the doc id and `relative_to` gives a short source path for citations."
**Seen in:** `main.py`, section 1

## sorted(glob)  `(standard library)`
**1. Definition:** `sorted(iterable)` returns a new list in ascending order; `glob` itself returns files in file-system order, which is not guaranteed.
**2. In plain words:** Without `sorted`, this Mac listed `customer_support` first instead of `allergens_and_ingredients`. No error, just a different order on a different machine.
**3. Example to quote:** > "I always sort the file list so positions are reproducible, and I key everything by id anyway."
**Seen in:** `main.py`, section 1; `WALKTHROUGH.md`, Debugging practice

## read_text(encoding="utf-8")  `(standard library)`
**1. Definition:** `Path.read_text(encoding=...)` reads a whole file and decodes its bytes with the given character encoding.
**2. In plain words:** Without an explicit encoding Python uses the platform default (often cp1252 on Windows). A `₹` then becomes `â‚¹` silently; with ASCII it crashes with `UnicodeDecodeError: 'ascii' codec can't decode byte 0xe2`.
**3. Example to quote:** > "I always pass encoding='utf-8'; a wrong default can silently corrupt text before it is embedded."
**Seen in:** `main.py`, section 1; `WALKTHROUGH.md`, Debugging practice

## Text cleaning and boilerplate  `(concept)`
**1. Definition:** Cleaning removes markup, repeated template text (boilerplate) and formatting noise from extracted text before chunking.
**2. In plain words:** "Page 3 of 12" or a cookie banner would appear in every chunk and make unrelated chunks look alike. `clean()` turned the messy snippet into `'Refund policy: Refunds are issued within 9 days. Contact help@chaikitchen.example for help.'`
**3. Example to quote:** > "Garbage in, garbage retrieved: I strip tags, footers and whitespace before I chunk."
**Seen in:** `main.py`, section 2

## html.unescape  `(standard library)`
**1. Definition:** `html.unescape(s)` converts HTML character references such as `&nbsp;` or `&amp;` into the characters they represent.
**2. In plain words:** `Refund&nbsp;policy` becomes `Refund` + non-breaking space + `policy`, which the next line turns into a normal space.
**3. Example to quote:** > "`html.unescape('&amp;')` returns `'&'`."
**Seen in:** `main.py`, `clean()`

## re.sub  `(standard library)`
**1. Definition:** `re.sub(pattern, replacement, text)` replaces every match of a regular expression.
**2. In plain words:** `<[^>]+>` means "a `<`, then anything that is not `>`, then `>`", so it removes tags but keeps the text between them. `\s+` collapses whitespace runs into one space.
**3. Example to quote:** > "`re.sub(r'\s+', ' ', text)` normalises all whitespace in one line."
**Seen in:** `main.py`, `clean()`

## Non-breaking space  `(concept)`
**1. Definition:** The non-breaking space (U+00A0) is a space character that prevents a line break; HTML writes it as `&nbsp;`.
**2. In plain words:** It looks like a space but is a different character, so `"a b" == "a b"` is False and tokenizers may treat it differently. `main.py` replaces it with a literal invisible character in the source; writing `" "` would be clearer.
**3. Example to quote:** > "Invisible characters like non-breaking spaces break exact matching, so I normalise them during cleaning."
**Seen in:** `main.py`, `clean()`

## Token  `(concept)`
**1. Definition:** A token is the unit of text a model's tokenizer produces: a word, part of a word, or a symbol, each mapped to an integer id.
**2. In plain words:** Context windows, embedding limits and API prices are all measured in tokens, not words or characters. The whole KB is 1,502 MiniLM tokens; the largest document, `product_catalog`, is 231.
**3. Example to quote:** > "I budget in tokens; different tokenizers give different counts, so I count with the tokenizer of the model I will use."
**Seen in:** `main.py`, section 3

## AutoTokenizer.from_pretrained  `(third-party: transformers)`
**1. Definition:** `AutoTokenizer.from_pretrained(name)` loads the tokenizer that belongs to a given pretrained model.
**2. In plain words:** It loads the all-MiniLM-L6-v2 tokenizer from the local cache (offline). Calling `tok(text)["input_ids"]` gives the token ids, including the special `[CLS]` and `[SEP]` tokens.
**3. Example to quote:** > "`len(tok(text)['input_ids'])` is how I count tokens for the embedding model."
**Seen in:** `main.py`, section 3

## model_max_length and truncation  `(concept)`
**1. Definition:** `tokenizer.model_max_length` is the longest input the tokenizer is configured for; with `truncation=True` longer inputs are cut, and the cut text is ignored.
**2. In plain words:** The tokenizer reports 512 tokens here. Text past the limit is not "summarised"; it is simply never seen by the model. Step 03 proves this.
**3. Example to quote:** > "Embedding models silently truncate, so I chunk before I embed."
**Seen in:** `main.py`, section 3

## Context window  `(concept)`
**1. Definition:** The context window is the maximum number of tokens (prompt plus output) an LLM can process in one call.
**2. In plain words:** 20,000 pages like ours would be about 3.3M tokens, too much for any context window. With `pages = 500` it is about 83,000 tokens, which fits a 200k window but still costs on every request.
**3. Example to quote:** > "Even when everything fits, sending everything is slow and expensive, and facts in the middle of long prompts can be missed."
**Seen in:** `main.py`, section 4

## Cost per query arithmetic  `(concept)`
**1. Definition:** Input cost = tokens per prompt x queries x price per token.
**2. In plain words:** 1,502 tokens x 10,000 queries/day x $3 per million = about $45/day; 3 chunks x 120 tokens = 360 tokens gives about $11/day. The price is illustrative; the ratio is the point.
**3. Example to quote:** > "Stuffing scales with the size of the knowledge base; RAG scales with k times the chunk size."
**Seen in:** `main.py`, section 4

## Incremental re-indexing  `(concept)`
**1. Definition:** Incremental re-indexing updates an index by processing only documents that were added, changed (for example detected by a content hash) or deleted, instead of rebuilding everything.
**2. In plain words:** If one policy page changes, re-chunk and re-embed only that page and replace its vectors by `doc_id`. This is why stable ids matter.
**3. Example to quote:** > "I hash each document; if the hash changes I delete its old chunks by doc_id and insert the new ones."
**Seen in:** `WALKTHROUGH.md`, theory
