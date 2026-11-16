# Module 19 — Exercises

Rules: change ONE thing at a time, predict the result before you run, and write down what happened. Worked answers: [solutions/solutions.md](solutions/solutions.md) (runnable scripts in [solutions/](solutions/)). Do not look until you have tried.

## Step 01 — tokens, not characters
- **Modify:** in `main.py` section C use `size=64, overlap=10`. How many windows, and what is the largest window in tokens?
- **Modify:** add a Japanese and an Arabic sample to `samples`. Which tokenizer is cheapest for each? How many `[UNK]` does MiniLM give?
- **Break it:** call `fixed_token_windows(doc, size=128, overlap=128)`. Read the traceback: what exactly stops it, and why is that better than a silent infinite loop?
- **Rebuild:** write `fixed_token_windows` from memory using `offset_mapping`.
- **Explain aloud:** why does a text of 481 tokens embedded with `max_length=256` give exactly the same vector as its first 256 tokens?

## Step 02 — recursive splitter rebuilt
- **Modify:** run the splitter with `length_function=len` and `chunk_size=400`. How many chunks exceed 128 tokens?
- **Modify:** add a `"\n- "` separator before `"\n"` for bullet lists. Which documents change?
- **Break it:** `RecursiveSplitter(64, 64)`. Then set `keep_separator="end"` and look at the first chunk boundaries of the Markdown splitter: what is wrong with them?
- **Rebuild:** write `merge()` (greedy pack with overlap carried from the front) without looking.
- **Explain aloud:** why does the Python separator `"\n\tdef "` miss methods in a file indented with spaces? How do you fix it? (Check `splitter.used`: on a small example both lists give the same pieces because blank lines separate the methods, so the observable difference is which separator was *selected*, not the output.)

## Step 03 — structure-aware Markdown
- **Modify:** `StructureChunker(60, 10)` on `api_reference`: what does the Errors table become? Is the header in every piece?
- **Modify:** set `breadcrumb=False` and compare dense hit@1 on the `reference` questions.
- **Break it:** delete the fenced-code branch of `parse_blocks`. What does section A print, and which section path does the "SEV1" paragraph get?
- **Rebuild:** write `build_tree` (heading stack) from memory.
- **Explain aloud:** why is the breadcrumb in `embed_text` and `context_text` but not in `text`?

## Step 04 — semantic chunking
- **Modify:** plot (print) the distance sparkline for buffers 0, 1, 3. Which is smoothest, and how many cuts does each give at the 85th percentile?
- **Modify:** raise `min_tokens` to 80. How many chunks remain in total?
- **Break it:** call `threshold(d, "median", 50)`; add a `"median"` rule that works. (Also note that `stdev` with `k=3` is not "never cuts": on the postmortem it makes exactly one cut.)
- **Rebuild:** write the gradient rule from the description only.
- **Explain aloud:** why can percentile breakpoints cut a document that has only one topic?

## Step 05 — late chunking
- **Modify:** `build(d, 128, chunk_tokens=20)`. Does the gap between late and naive move?
- **Modify:** also pool the `[CLS]` token. Does the chunk vector move a lot?
- **Break it:** run `late_vectors` on a 700-word text with a span at the very end. What do you get, and why?
- **Rebuild:** `late_vectors` from memory: forward pass, offsets, pooling.
- **Explain aloud:** why does late chunking need a model trained for it, and which model families advertise it?

## Step 06 — contextual retrieval
- **Modify:** use `RecursiveSplitter(200, 0)` as the baseline. Does the contextual gain shrink?
- **Modify:** replace `heuristic_context` by one that returns only the document title. How much of the effect is the title alone?
- **Break it:** remove the `<chunk>` tags from `PROMPT`. Read the traceback.
- **Rebuild:** the cost formula with and without prompt caching, from the numbers of tokens.
- **Explain aloud:** contextual embeddings vs contextual BM25: what does each fix?

## Step 07 — hierarchical retrieval
- **Modify:** auto-merging with `ratio` 0.3 and 0.8. Compare hit@1 and context tokens.
- **Modify:** sentence window with `w` 0, 2, 5.
- **Break it:** remove the de-duplication in `small_to_big`. What happens to ctx tokens and why?
- **Rebuild:** `build_nodes` with parent/child ids.
- **Explain aloud:** what lives in the vector DB, what in the docstore?

## Step 08 — propositions and agentic chunking
- **Modify:** agentic SAME/NEW threshold 0.06 vs 0.3.
- **Modify:** remove the pronoun replacement from `rule_propositions`.
- **Break it:** make the mock return non-JSON and read the error.
- **Rebuild:** compute the ingestion cost (calls, tokens in and out) for a 1M-token corpus from the ratio in step 08.
- **Explain aloud:** propositions vs sentences vs semantic chunks.

## Step 09 — layout-aware chunking
- **Modify:** `LayoutChunker(60)`: which chunks change?
- **Modify:** remove `merge_continued_tables`. What does the milestone question return?
- **Break it:** remove the `sorted(...)` on reading order.
- **Rebuild:** a function that turns a table into row groups with the header repeated.
- **Explain aloud:** how does a chunk keep its citation (page, bbox) through merging?

## Step 10 — evaluation
- **Modify:** add `recursive 60` and `recursive 200` to `STRATEGIES`. Where does hit@1 peak? What does ctx tokens do?
- **Modify:** change the bootstrap to 95% on hit@1 instead of hit@3.
- **Break it:** drop the cache clearing. Which timings change?
- **Rebuild:** `char_precision` in `common19.py`.
- **Explain aloud:** why is hit@k alone misleading when comparing 40-token and 250-token chunks?

## Step 11 — chunker LLD
- **Modify:** `ChunkConfig(max_tokens=60)`: chunks and `config_hash`.
- **Modify:** edit a sentence in the first and then the last section; count `added`.
- **Break it:** register two chunkers under one name.
- **Rebuild:** write a fourth strategy `SentenceChunker` behind the Protocol and add a test.
- **Explain aloud:** describe the zero-downtime rollout of a new chunker version.
