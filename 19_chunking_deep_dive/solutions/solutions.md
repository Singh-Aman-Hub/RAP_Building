# Module 19 — Solutions

Back to the [exercises](../exercises.md) and the [module README](../README.md). Try the exercise first. Every number below is from a real run of the script named in the table (CPU, MiniLM, offline); runs print the same numbers on any machine, only the timings differ. Run any script with `HF_HUB_OFFLINE=1 ../../.venv/bin/python <file>` from this folder.

| Exercise step | Runnable solution | Answer in one line |
|---|---|---|
| 01 tokens | [s01_windows_and_unk.py](s01_windows_and_unk.py) | see below |
| 02 recursive splitter | [s02_python_separator_fix.py](s02_python_separator_fix.py) | see below |
| 03 structure-aware | [s03_breadcrumb_and_table.py](s03_breadcrumb_and_table.py) | see below |
| 04 semantic | [s04_median_and_buffers.py](s04_median_and_buffers.py) | see below |
| 05 late chunking | (answered in the [step 05 walkthrough](../05_late_chunking/WALKTHROUGH.md)) | see below |
| 06 contextual | [s06_cost_formula.py](s06_cost_formula.py) | see below |
| 07 hierarchical | [s07_ratio_and_window_sweeps.py](s07_ratio_and_window_sweeps.py) | see below |
| 08 propositions / agentic | (answered in the [step 08 walkthrough](../08_propositions_and_agentic_chunking/WALKTHROUGH.md)) | see below |
| 09 layout-aware | (answered in the [step 09 walkthrough](../09_layout_aware_chunking_from_parsed_docs/WALKTHROUGH.md)) | see below |
| 10 evaluation | [s10_add_recursive_sizes.py](s10_add_recursive_sizes.py) | see below |
| 11 chunker LLD | [s11_sentence_chunker.py](s11_sentence_chunker.py) | see below |

## Step 01
- `size=64, overlap=10` on the handbook: **20 windows**, largest **64 tokens** (the default 128 / 20 gives 10 windows).
- Japanese sample (29 characters): MiniLM 29 tokens with **12 `[UNK]`**, e5-small **19** tokens, GPT-2 47. Arabic sample (70 characters): MiniLM 58 tokens with 0 `[UNK]` (WordPiece falls back to single characters), e5-small **16**, GPT-2 70. e5-small is the cheapest for both.
- `overlap == size` raises `AssertionError` before any loop starts; without the guard the step would be 0 (`range()` raises `ValueError: range() arg 3 must not be zero`).
- Rebuild: see the checklist in the [step 01 walkthrough](../01_tokens_not_characters/WALKTHROUGH.md).
- Explain aloud: the embedder truncates to `max_length=256` tokens before pooling, so tokens 257 to 481 never enter the mean; the two inputs are the same 256 tokens, hence the same vector (cosine 1.000000 in step 01 section D).

## Step 02
- `length_function=len, chunk_size=400` on the handbook: 15 chunks, **0** above 128 tokens (400 characters of this prose is about 90 to 100 tokens; on JSON the answer would differ).
- The tab separator `"\n\tdef "` is never *selected* on a space-indented class: `sp.used` is `{'\n\n': 1}` for the original list and `{'\\n[ \\t]*def ': 1}` for `[r"\nclass ", r"\n[ \t]*def ", "\n\n", "\n", " ", ""]`. On the small example in the script both lists produce the same 3 pieces (blank lines already separate the methods), so the *observable* difference is the chosen separator, not the output.
- `RecursiveSplitter(64, 64)` raises a bare `AssertionError`. With `keep_separator="end"` the heading marker (`####`) ends the *previous* chunk, severed from its own section.
- Also measured: `keep_separator=False` with the Markdown regexes leaves the literal text `#{1,6}` in 6 of 23 chunks, a wart of using the regex as the joiner.

## Step 03
- `StructureChunker(60, 10)` on `api_reference`: the Errors table becomes **7 chunks, every one starting with the header row** (`True` in the script).
- `breadcrumb=False` vs `True` at 200 tokens: overall dense hit@1 **0.73 vs 0.78**; on the 4 `reference` questions it is 0.75 in both, so that slice is too small to show the effect.
- Deleting the fenced-code branch of `parse_blocks` turns the `# Report a suspected incident...` comment into a heading (section A prints `True`) and moves the SEV1 paragraph under a section that does not exist in the document.
- Explain aloud: the breadcrumb is in `embed_text` (so the vector and BM25 know the topic) and `context_text` (so the LLM knows where the passage comes from) but not in `text`, which must stay the exact source slice for offsets and citations.

## Step 04
- Buffers 0 / 1 / 3 on the postmortem: the curve is jagged at 0 and smoother at 1 and 3; at the 85th percentile every buffer gives **7 cuts**.
- Median rule (median + 3 x MAD): threshold **0.44**, **2 cuts** (stdev k=1 gives 5). `threshold(d, "median", 50)` raises `ValueError: median`.
- Why a percentile cuts a one-topic document: it fixes the *share* of boundaries that become cuts (top 15% at p=85), whatever the text says.
- `min_tokens=80`: weak cuts around small groups are removed first (`enforce_sizes`), leaving fewer, larger chunks; run it to count them.

## Step 05
- Late vs naive with `chunk_tokens=20`, and pooling `[CLS]`: for a 35-token chunk the cosine between the pooled vector with and without `[CLS]` is **0.998** (measured), so it barely moves.
- With a 700-word text and a span at the end the window is truncated to 512 tokens, the span matches no token, and the pooled vector is **`nan`** (no error, no warning).
- Why a trained model matters: with MiniLM late chunking is *worse* than naive from window 128 up (hit@1 0.58 against 0.68); long-context models such as Jina v2/v3 advertise late chunking.

## Step 06
- Cost formula (`contextual_cost`): a 1,000-token document with 12 chunks costs 13,788 input tokens uncached and 4,138 cached (**3.3x cheaper**); 10,000 tokens and 120 chunks: **8.2x**; 100,000 tokens and 1,200 chunks: **9.8x**. The ratio tends to `1 / 0.10 = 10x` because the document part dominates only for long documents.
- Removing the `<chunk>` tags makes the mock's `re.search` return `None`: `AttributeError: 'NoneType' object has no attribute 'group'`.
- Contextual embeddings fix the *meaning* gap in the vector; contextual BM25 fixes the *vocabulary* gap in the lexical index.

## Step 07
- Auto-merging: ratio 0.3 gives hit@1 0.89, hit@3 0.95 and **284** context tokens; 0.5 gives 0.88 / 0.93 / **137**; 0.8 gives 0.82 / 0.92 / **105**.
- Sentence window: `w=0` hit@1 0.61, hit@3 0.78, 46 tokens (it equals flat sentences); `w=2` 0.86 / 0.95 / 197; `w=5` 0.92 / 0.97 / **415**. More window buys quality at a roughly linear token cost.
- Removing the `seen` de-duplication in `small_to_big` (section level): context tokens rise from 329 to 345, hit@3 unchanged; silent waste, no error.
- The vector DB holds only the leaves (id, doc id, parent id, offsets); the docstore holds every node by id and is where parents are fetched from.

## Step 08
- Agentic threshold 0.06 to 0.3: more `NEW` decisions, so smaller chunks (try it; the default run gives 25 chunks of 5 to 119 tokens on the postmortem).
- Without the `PRONOUN.sub` line the facts start with "It must be ..." again. Even with it the mock replaces "These weeks" by the section topic ("The secondary caregiver weeks"), a reminder to keep the source span.
- A non-JSON reply makes `extract_json` raise `ValueError: No JSON object found in model reply`.
- Ingestion cost: the run gives 140 calls and 20,204 input + 6,301 output tokens for a 4,987-token corpus, **5.3x the corpus size**; for 1M tokens that is about 28 thousand calls at the same ratio (140 calls per 4,987 tokens) and 5.3M LLM tokens.
- Propositions vs sentences vs semantic chunks: propositions rewrite the *content* into atomic facts, sentences are the *source* units, semantic chunks are *groups of sentences* cut by embedding distance.

## Step 09
- `LayoutChunker(60)`: 16 chunks instead of 10; the tables split into row groups with the header repeated and the same citation box.
- Removing `merge_continued_tables`: 11 chunks; the milestone table splits into a page-2 chunk with its caption and a page-3 chunk without it.
- Removing the `sorted(...)`: in this sample the output is unchanged, because a table followed by its caption still binds through the "caption below" branch; it matters when the order separates elements the rules look at together.
- Removing `reading_order` from one element raises `KeyError: 'reading_order'`. Fallback sort key: `(page, bbox top, bbox left)`.
- Table to row groups with the header repeated: see `object_chunks` in step 09 (`md_table(header, group)` for each group).
- Citation through merging: keep `element_ids` and one union box per page (`union_boxes`); a merged table keeps the boxes of both pages (`p.2-3`).

## Step 10
- Recursive sizes (dense): 40 tokens hit@1 0.68, ctx 84; **60 tokens hit@1 0.81, hit@3 0.86**, ctx 134; 100 tokens 0.74 / 0.91 / 223; 200 tokens 0.72 / 0.91 / 450; 300 tokens 0.72 / 0.91 / 662. hit@1 peaks at 60 tokens, hit@3 plateaus from 100 tokens, context tokens grow linearly, and context precision falls from 0.08 to 0.01.
- Dropping `common19._CACHE.clear()`: later strategies reuse vectors for identical texts, so their ingestion times shrink; unfair because the first strategy pays for the shared embeddings.
- Hit@k alone is misleading because a 250-token chunk covers more text than a 40-token one: hit@k rises while the LLM's context and the bill grow. Report context tokens and precision next to it.

## Step 11
- `ChunkConfig(max_tokens=60)`: **21 chunks** instead of 19 and config hash `7d899763e99c70b5` instead of `afb6097d7d8f43f9`.
- Editing one sentence in the first or in the last section: kept 18, added 1, removed 1 in both cases.
- A duplicate registration raises `ValueError: chunker 'recursive' already registered`.
- A fourth strategy behind the Protocol: [s11_sentence_chunker.py](s11_sentence_chunker.py) registers `sentence_pack` (whole sentences up to `max_tokens`), checks that offsets reproduce the text and ids are deterministic, and shows the duplicate guard. Add a pytest case like `test_chunks_respect_max_tokens_and_offsets` parametrised with `"sentence_pack"`.
- Zero-downtime rollout: build a new index named after chunker version, config hash and embedding model beside the live one; the embedding cache (keyed by text hash) keeps unchanged texts free; verify offline; flip the alias atomically; keep the old index for rollback, then drop it.
