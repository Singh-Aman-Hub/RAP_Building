# Interview Gist — Chunking Deep Dive (Module 19)
### Tokens → Recursive → Structure → Semantic → Late → Contextual → Hierarchical → Propositions → Layout → Evaluate → Design

One-page revision sheet. Numbers come from this module's own runs (13 documents, 74 questions, MiniLM, CPU). Details: [README](README.md).

## 0. The one-sentence answer
> "I measure in the embedding model's tokens, chunk along the document's own structure where it has one (otherwise recursively at 100 to 200 tokens), put a breadcrumb in the embedded text, retrieve small and return the bigger parent when precision and context conflict, add LLM-written context only after a failure analysis shows chunks that cannot stand alone, and I choose between all of that on my own labelled questions with offsets as ground truth, watching context tokens and ingestion cost as well as hit@k."

## 1. Why chunk (recap of [14/02](../14_rag_and_retrieval/02_chunking_strategies/WALKTHROUGH.md))
Embedding input limit (tokens), one idea per vector, LLM context budget, precision vs context. The chunk is the unit of search and the unit of context.

## 2. Tokens, not characters (step 01)
- Limits and prices are in **tokens** of the model's tokenizer. Count with it.
- MiniLM (WordPiece): English prose about 5.4 chars/token on a sentence, whole documents 3.4 (tables/code) to 4.7; JSON 1.7; URL + hash 1.6; German 2.9; Hindi 1.5; Chinese 1.0 with 19 of 24 characters `[UNK]` (wrong model). e5-small's SentencePiece: Hindi 3.9, German 4.5.
- `max_seq_length` truncates **silently**: text of 481 tokens at max 256 -> 225 tokens unread, vector identical to the truncated text's. Beyond the 512 positions: `RuntimeError` (tensor a (964) vs b (512)).
- Token windows via `offset_mapping` give exact character slices. 512-character windows varied 104 to 132 tokens.

## 3. Recursive splitter (step 02)
```
split_text(text, separators):
    sep = first separator that occurs in text          # coarse -> fine
    pieces = split(text, sep, keep_separator)
    small pieces  -> merge(...) up to chunk_size, carrying chunk_overlap into the next chunk
    big piece     -> recurse with the remaining (finer) separators
```
- `length_function` decides what "size" means: pass the token counter.
- `keep_separator` start/end/False decides where `## ` goes; "end" glued `####` to the previous chunk.
- Language separators: Markdown (`\n#{1,6} `, fences, rules), Python (`\nclass `, `\ndef `, `\n\tdef `). Gotcha: space-indented methods miss `\n\tdef `.
- Overlap: carried pieces total at most `chunk_overlap` tokens (max shared 16 tokens at overlap 24). Cost: 1.12x stored tokens at size 96, overlap 24.

## 4. Structure-aware chunking (step 03)
- Parse blocks (heading, paragraph, list, table, code) -> heading stack -> sections. Fenced code is one block so `# comment` is not a heading.
- One chunk per section body, `embed_text = "Doc > H1 > H2\n" + text`. Split oversize sections at block boundaries, then inside a block: sentences, table rows with the **header repeated**, code lines with the fence re-closed. Merge tiny siblings.
- Damage of a structure-blind recursive splitter on the handbook + API doc: 4 cut code fences, 2 headerless table fragments; structure-aware: 0 and 0.
- Breadcrumb: dense hit@1 0.73 -> 0.78 at 200 tokens.

## 5. Semantic chunking (step 04)
- Embed sentences (buffer = neighbours glued on), `d[i] = 1 - cos(e_i, e_{i+1})`, cut where `d` exceeds: percentile p, mean + k std, mean + k IQR, or percentile of the gradient. Guard min/max chunk tokens.
- Cost: 375 sentence-windows, about 14,665 tokens through the model (3x the corpus with buffer 1) before the chunks are embedded again.
- Result here: no better than recursive or sentence-packing (dense hit@3 0.86 to 0.95 vs 0.91 to 0.92, within noise). Postmortem without headings: sentence-pack 0.91, semantic 0.73 (buffer 1). Literature: benefits are inconsistent (Qu et al. 2024). Percentile always cuts something, even in a one-topic text. A stdev rule (k=3) cuts only outliers: exactly 1 cut on the postmortem.

## 6. Late chunking (step 05)
Run the model over the whole window, `H = last_hidden_state` (T, d), mean-pool the token vectors of each chunk span. Needs a model trained for it with a long context (Jina: jina-embeddings-v3, 8,192 tokens, `late_chunking` parameter; paper arXiv 2409.04701). With plain MiniLM: one pronoun chunk improved (rank 5 -> 3 in its window) but hit@1 over the corpus fell 0.69 -> 0.54 to 0.58 as the window grew. Window equal to the chunk reproduces naive exactly (sanity check).

## 7. Contextual retrieval (step 06)
LLM writes 1 to 2 situating sentences per chunk; prepend before embedding **and** before BM25. Anthropic reports 35% fewer top-20 failures from contextual embeddings alone, 49% with contextual BM25, 67% with reranking. Cost: one call per chunk with the whole document in the prompt; **prompt caching** makes the document prefix cheap (here 71,850 input tokens uncached vs 22,811 token-equivalents cached, 3.1x). The index grows by the context length (+67% stored tokens). Our mock heuristic gave no gain when headings stay in the chunk text; a small gain when they were lost.

## 8. Hierarchical retrieval (step 07)
Store nodes with `parent_id` and child ids; **only leaves are embedded**; parents live in a docstore.
- Small-to-big: match leaf, send parent (de-duplicate). Sentence window: send +-w sentences. Auto-merging: if more than 50% of a parent's children are retrieved, send the parent. RAPTOR: cluster, summarise, repeat; index all levels.
- Here (hit@1): flat leaves 0.78, small-to-big mid 0.86, section 0.91, sentence window 0.86, auto-merging 0.88, RAPTOR (mock extractive) 0.78. Section context cost 329 tokens @3 vs 88 for flat leaves. RAPTOR helps thematic questions; this eval is fact lookup.

## 9. Propositions and agentic (step 08)
Propositions (Dense X, arXiv 2312.06648): atomic, self-contained facts, pronouns resolved, one LLM call per passage. Here 140 calls and 5.3x the corpus in tokens at ingestion. Agentic: one SAME/NEW LLM call per sentence (362 calls). The mock-LLM versions did not beat sentence-packing (hit@1 0.62 and 0.73 vs 0.66; hit@3 0.84 and 0.89 vs 0.92) but sent the fewest context tokens (54 and 94).

## 10. Layout-aware chunking (step 09)
Input: elements `{type, text, page, bbox, reading_order}`. Sort by `(page, reading_order)` (array order is not reading order); drop page headers/footers; a section header starts a chunk; keep table whole or row groups with header repeated; merge a table continued on the next page; bind caption to its table/figure; keep page + union bbox per page. Result: 9 chunks, 6/7 answered with a page + bbox citation vs 0/7 for flatten-and-window. Tools: Docling `HybridChunker` (token-aware refinement over document hierarchy, `merge_peers`), Unstructured `chunk_by_title` (title starts a chunk; `max_characters`, `new_after_n_chars`, `combine_text_under_n_chars`).

## 11. Evaluation (step 10)
- Ground truth = answer **character spans**; a context is relevant if its source span covers the answer. Works for any chunker.
- Metrics: hit@k, recall@k (multi-evidence), MRR, **context tokens sent**, context precision (answer chars / chars sent), index size, ingestion time, LLM calls.
- Noise: 74 questions give a 95% interval of about +-0.07 on hit@3: most rows overlap. Compare on hit@1/MRR and cost, slice by question kind and corpus.
- Bigger chunks make hit@k easier and the LLM's job harder: always report context tokens.
- Attribution beyond retrieval: which chunk supports which answer sentence (citation precision/recall) needs a generation step; context precision is the cheap proxy.

## 12. Production design (step 11)
```
Chunk(frozen): chunk_id, doc_id, doc_version, parent_id, text, embed_text, display_text,
               token_count, position, char_start, char_end, section_path, page, bbox, metadata,
               chunker_id, chunker_version
chunk_id = hash(doc_id, chunker_id, chunker_version, config_hash, text, occurrence)   # not the position
Chunker (Protocol): chunk(doc_id, text, base_offset) -> list[Chunk];  registry: name -> class
ChunkConfig(frozen): max/overlap/min tokens, separators ... -> config_hash
```
- **Idempotent re-chunking:** same input -> same ids; edit one number -> kept 18, added 1, removed 1, 1 text re-embedded of 19.
- **Versioned indexes:** `collection__chunker@version__confighash__embedmodel`; build new, flip alias, keep old for rollback; embedding cache keyed by text hash (0 re-embeddings for a version bump that left texts unchanged).
- **Streaming:** segment at blank lines, chunk each segment, `base_offset` keeps offsets global (peak 1.1 MB vs 1.7 MB on 126k chars).
- **Multilingual:** inject a multilingual tokenizer; CJK separators `。` `，`; English separators on Chinese fall back to character cuts mid-sentence.
- **Tests:** determinism, offsets reproduce text, max tokens, config validation, edit locality, overlap, code fences, cache reuse, stream coverage (12 pytest tests).

## 13. Same thing, different names (cheat list)
chunk = passage = node = split = segment. overlap = stride complement. parent-child = small-to-big = hierarchical = (auto-merging decides when). contextual retrieval = contextual chunk headers = contextual embeddings. Semantic = where to cut; late = how to embed; proposition = what to index. `chunk_size` in characters is not tokens.

## 14. Quick-fire Q&A
**Q: How do you choose chunk size?** A: Start at 100 to 250 tokens along structure; sweep on your own questions; watch hit@1, MRR and context tokens; bigger always raises hit@k.
**Q: Why tokens?** A: The model's limit and truncation are in tokens; characters per token ranges 1.0 (Chinese in MiniLM) to 5+ (English prose).
**Q: What happens past max_seq_length?** A: Silent truncation to 256 here (error beyond 512 positions); text after it never reaches the vector.
**Q: What is overlap for?** A: A fact on a boundary appears whole in one chunk if it is shorter than the overlap. It costs storage and duplicate hits.
**Q: Recursive splitter in one breath?** A: Coarsest separator that occurs, merge small pieces to the size limit, recurse into oversize pieces, carry overlap.
**Q: How do you chunk a table?** A: Whole if it fits; else row groups with the header row repeated; embed rows as "Column: value".
**Q: Semantic chunking, worth it?** A: Costs a model pass over every sentence; often ties recursive; use only if you measure a gain on your documents.
**Q: Late chunking vs contextual retrieval?** A: Late = pool token vectors from a full-context pass (no LLM, needs a long-context model trained for it). Contextual = LLM writes context text (LLM cost, works with any embedder, helps BM25 too).
**Q: Small-to-big?** A: Embed small leaves, send their parent; store `parent_id`; de-duplicate parents.
**Q: Propositions?** A: Atomic decontextualised facts; strong precision, expensive to produce, can lose nuance.
**Q: How do you cite a page?** A: Keep page and bbox per element through chunking; union boxes per page on the chunk.
**Q: How do you evaluate chunkers with different boundaries?** A: Offset-based relevance: a context counts if its source span covers the answer sentence.
**Q: What goes in a chunk id?** A: Doc id, chunker id and version, config hash, content hash, occurrence; not the position.
**Q: Roll out a new chunker?** A: Build a new versioned index in the background with the shared embedding cache, compare on the eval set, flip the alias, keep the old index for rollback.
**Q: Is 14/02's recursive splitter the same as LangChain's?** A: Same idea, but it counts characters, has no overlap, no keep_separator and no offsets; step 02 here adds them.
