# Module 19 — Chunking Deep Dive

Every chunking technique used in RAG systems in 2026, how each one works inside, each rebuilt from scratch, all scored on the same labelled questions, and the low-level design of a production chunker.

## The story of this module
[Module 14, step 02](../14_rag_and_retrieval/02_chunking_strategies/WALKTHROUGH.md) compared five basic splitters by counting characters and words. It ended with "tune on your own evaluation set". This module is that tuning, and much more.
- **Pain point:** a chunk is both the unit of search and the unit of context. Cut it badly and the fact is split, the heading is lost, the table loses its header, the pronoun has no noun, or the vector is a blur. No retriever, embedding model or re-ranker can fix text that was cut wrongly at ingestion.
- **Spark:** measure in the embedding model's own tokens, respect the structure the author wrote, add context where a chunk lost it, retrieve small and return big where precision and context pull in different directions, and score every idea on one evaluation harness instead of arguing about it.
- **What you build:** a token counter and token windows, a LangChain-style recursive splitter, a Markdown structure chunker with breadcrumbs, semantic chunking, late chunking, contextual retrieval, four hierarchical retrievers (small-to-big, sentence window, auto-merging, RAPTOR-style), proposition and agentic chunking with a mock LLM, a layout-aware chunker for parsed PDFs, a scoreboard of 19 strategies, and a production `Chunk` / `Chunker` / index-versioning design with pytest tests.
- **Cliffhanger:** the chunker is only as good as the text it gets. For scanned or complex PDFs, the text itself has to be recovered first: OCR and document parsing (module 18, written in parallel; this module's step 09 works on a sample parsed document and does not depend on it).

**Short on time?** Read [INTERVIEW_GIST.md](INTERVIEW_GIST.md) first.

## 1. Why this matters
"How do you chunk documents?" is the most common follow-up to "build a RAG chatbot". A weak answer is "500 tokens with 50 overlap". A strong answer measures size in tokens of the embedding model, picks the chunker by document type, names the failure mode of each technique, adds context to chunks that lost it, and says how the choice was evaluated and what it costs at ingestion. Every number below comes from a real run on a laptop CPU. The corpus is small (13 documents, about 5,000 tokens, 74 questions), so use the numbers to understand the *mechanisms*, not as benchmark truth (see step 10 for the noise level).

## 2. Where this fits
**Builds on**
- [14_rag_and_retrieval](../14_rag_and_retrieval/README.md): `common.py` (Embedder with all-MiniLM-L6-v2, BM25, RRF), the 9-document Chai Kitchen knowledge base and its 24 labelled questions. Step [14/02](../14_rag_and_retrieval/02_chunking_strategies/WALKTHROUGH.md) is the five-splitter overview this module goes beyond; step [14/10](../14_rag_and_retrieval/10_advanced_rag_and_vector_dbs/WALKTHROUGH.md) showed small-to-big in one example.
- [04_llm_fundamentals](../04_llm_fundamentals/README.md): tokenizers, embeddings.
- [shared/llm_client.py](../shared/llm_client.py): the mock / local / Anthropic LLM interface used by steps 06 and 08.

**Leads to**
- [18_ocr_and_document_parsing](../18_ocr_and_document_parsing/README.md): its last step, [10 parsed document hand-off](../18_ocr_and_document_parsing/10_parsed_document_handoff/WALKTHROUGH.md), writes the parsed-document JSON (same schema as step 09's [sample file](09_layout_aware_chunking_from_parsed_docs/sample_parsed_document.json)) that step 09 chunks.
- [20_llm_system_lld](../20_llm_system_lld/README.md): the production RAG LLD around the chunker; its chunking layer is [`ragkit/chunking.py`](../20_llm_system_lld/ragkit/chunking.py) (see step 11).
- The production RAG pipeline of [14/08](../14_rag_and_retrieval/08_rag_pipeline_end_to_end/WALKTHROUGH.md): drop `chunkerlib` (step 11) in as its ingestion stage.

## 3. Concept map
```
 raw document ──► parse (text only | Markdown | parsed elements with page + bbox)
        │
        ▼
 measure in TOKENS of the embedding model  (01)  ◄── max_seq_length: silent truncation
        │
        ├── fixed token windows (+overlap)                       (01)
        ├── recursive: separator hierarchy + merge               (02)
        ├── structure-aware: heading tree, atomic blocks         (03)  ──► breadcrumb
        ├── semantic: embed sentences, cut at distance spikes    (04)
        ├── layout-aware: reading order, tables, figures, bbox   (09)
        └── LLM-driven: propositions, agentic boundaries         (08)
        │
        ▼
 make each chunk understandable ALONE
        ├── breadcrumb / contextual header (03)   ├── late chunking: pool token vectors (05)
        └── contextual retrieval: LLM-written context, + BM25 (06)
        │
        ▼
 decide what to INDEX vs what to SEND                           (07)
        small-to-big · sentence window · auto-merging · RAPTOR tree
        │
        ▼
 evaluate on one harness (10)  ──►  production design: Chunk, Chunker Protocol, ids, versions (11)
```

## 4. Study path
| Step | Folder | You learn | Run | Time |
|---|---|---|---|---|
| 01 | [01_tokens_not_characters](01_tokens_not_characters/WALKTHROUGH.md) | token counting with the real tokenizer, chars/token by language and content, token windows, silent truncation | `python main.py` | ~25 s |
| 02 | [02_recursive_splitter_rebuilt](02_recursive_splitter_rebuilt/WALKTHROUGH.md) | the LangChain recursive splitter rebuilt: separators, merge + overlap, keep_separator, Markdown / Python separators | `python main.py` | ~20 s |
| 03 | [03_structure_aware_markdown](03_structure_aware_markdown/WALKTHROUGH.md) | Markdown block parser, heading tree, breadcrumbs, atomic blocks, tiny-section merging | `python main.py` | ~20 s |
| 04 | [04_semantic_chunking](04_semantic_chunking/WALKTHROUGH.md) | embedding-distance breakpoints (percentile, stdev, IQR, gradient), buffers, cost, when it loses | `python main.py` | ~30 s |
| 05 | [05_late_chunking](05_late_chunking/WALKTHROUGH.md) | embed the whole window once, pool token vectors per chunk; long-context embedders | `python main.py` | ~30 s |
| 06 | [06_contextual_retrieval](06_contextual_retrieval/WALKTHROUGH.md) | LLM-written chunk context, contextual embeddings + BM25, prompt-caching cost | `python main.py` | ~25 s |
| 07 | [07_hierarchical_parent_child](07_hierarchical_parent_child/WALKTHROUGH.md) | small-to-big, sentence window, auto-merging, RAPTOR-style tree, parent links | `python main.py` | ~30 s |
| 08 | [08_propositions_and_agentic_chunking](08_propositions_and_agentic_chunking/WALKTHROUGH.md) | atomic-fact chunks and LLM-guided boundaries, their ingestion cost | `python main.py` | ~25 s |
| 09 | [09_layout_aware_chunking_from_parsed_docs](09_layout_aware_chunking_from_parsed_docs/WALKTHROUGH.md) | chunking parsed elements: tables, captions, cross-page merge, page + bbox citations | `python main.py` | ~10 s |
| 10 | [10_chunking_evaluation](10_chunking_evaluation/WALKTHROUGH.md) | one scoreboard: hit@k, MRR, recall, context tokens, index size, ingestion cost, noise, how to pick | `python main.py` | ~1.5 min |
| 11 | [11_chunker_lld](11_chunker_lld/WALKTHROUGH.md) | production design: Chunk dataclass, Protocol + registry, ids, idempotent re-chunking, versioned indexes, streaming, tests | `python main.py` | ~20 s |

All runs: `HF_HUB_OFFLINE=1 .venv/bin/python 19_chunking_deep_dive/<step>/main.py` from the repo root. Times are CPU time on a quiet laptop.

## 5. The test bed
- **13 documents:** the 9 Chai Kitchen knowledge-base files from module 14 (clean, short, `##` sections) plus four deliberately messy new ones in [data/](data/): `employee_handbook.md` (nested headings 1 to 4 levels, pronoun-heavy sections, a code fence containing a `# comment`), `api_reference.md` (two tables, two code blocks), `support_faq.md` (Q/A pairs, no sub-headings), `incident_postmortem.txt` (flowing prose, no structure at all).
- **74 questions** (24 from module 14, 50 new in [data/eval_questions.json](data/eval_questions.json)); kinds: keyword, paraphrase, reference (the answer sentence relies on a pronoun or heading), table, code, multi (two evidence sentences).
- **Relevance by character offsets.** Each question has the exact answer sentence(s). A returned chunk or context is relevant if its source span fully covers the answer sentence. This is what makes a 40-token chunk, a 250-token section, a proposition and a summary comparable.
- **Metrics** (full definitions in step 10): hit@k, recall@k, MRR, context tokens sent per query, context precision, index size, ingestion time, LLM calls.

## 6. The headline result (step 10, dense MiniLM retrieval, 74 questions)
| Strategy | Chunks | LLM calls | hit@1 | hit@3 | MRR | ctx tokens @3 |
|---|---|---|---|---|---|---|
| fixed 128 tokens, no overlap | 48 | 0 | 0.72 | 0.89 | 0.81 | 337 |
| fixed 256 tokens, overlap 38 | 26 | 0 | 0.70 | 0.95 | 0.82 | 657 |
| recursive 100 tokens | 68 | 0 | 0.74 | 0.91 | 0.83 | 223 |
| recursive 100, Markdown separators | 71 | 0 | 0.78 | 0.92 | 0.86 | 214 |
| structure-aware 100 + breadcrumb | 92 | 0 | 0.82 | 0.91 | 0.88 | 199 |
| semantic, percentile 85 (buffer 1) | 61 | 0 | 0.72 | 0.89 | 0.81 | 264 |
| late chunking (MiniLM, 128-token window) | 165 | 0 | 0.58 | 0.85 | 0.73 | 91 |
| contextual (mock LLM) | 87 | 87 | 0.69 | 0.86 | 0.79 | 176 |
| small-to-big, return section (250) | 156 | 0 | 0.91 | 0.95 | 0.93 | 329 |
| sentence window +-2 | 375 | 0 | 0.86 | 0.95 | 0.91 | 197 |
| auto-merging | 156 | 0 | 0.88 | 0.93 | 0.92 | 137 |
| propositions (mock LLM) | 303 | 140 | 0.62 | 0.84 | 0.74 | 54 |
| agentic SAME/NEW (mock LLM) | 224 | 362 | 0.73 | 0.89 | 0.82 | 94 |

**How to read it.** The 95% bootstrap interval of hit@3 is about plus or minus 0.07, so most rows overlap: only large gaps (greater than about 0.10) are evidence. What does stand out: the **hierarchical** family (retrieve small, send bigger) wins on hit@1 and MRR; **structure-aware with a breadcrumb** beats structure-blind splitters at the same size; **semantic chunking does not beat** simple recursive or sentence packing; **late chunking with a small model not trained for it hurts**; the **mock-LLM** chunkers (contextual, propositions, agentic) cost many LLM calls here without a retrieval gain (their real-LLM versions may do better; the mock is a stand-in, see steps 06 and 08).

## 7. The concepts, one by one
### Tokens, not characters (step 01)
Embedding models read tokens, limit input in tokens (`max_seq_length`) and truncate silently. English prose is about 4 to 5 characters per token with MiniLM's WordPiece tokenizer, but JSON, URLs and hashes give 1.6 to 1.7, Hindi 1.5 and Chinese 1.0 (and 19 of 24 Chinese characters are `[UNK]` for MiniLM: the wrong model for that language). Count with the tokenizer of the model you embed with.

### Recursive splitting (step 02)
Try the coarsest separator that occurs; merge small pieces up to `chunk_size`; recurse only into pieces still too long; carry `chunk_overlap` into the next chunk. With `length_function` = the token counter, "128" means 128 tokens. Language-specific separator lists (Markdown, Python) turn it into a syntax-aware splitter. Gotcha: Python methods indented with spaces are not matched by the `"\n\tdef "` separator.

### Structure-aware chunking (step 03)
Parse blocks, build the heading tree, chunk by section, add a breadcrumb to the embedded text, never cut a table or code fence, split oversize sections inside the block (rows with a repeated header, sentences, code lines), merge tiny siblings. Structure-blind splitting cut 4 code fences and 2 table fragments on two documents; structure-aware cut none.

### Semantic chunking (step 04)
Distance between neighbouring sentence embeddings, breakpoints by percentile, standard deviation, IQR or gradient. It costs a model pass over every sentence (about 3x the tokens with a buffer) and, on this corpus, did not beat sentence packing. Literature agrees that its gains are inconsistent (see References).

### Late chunking (step 05)
Embed the whole window once, pool token vectors per chunk. Needs a model trained for it (long context, contextual token vectors). With plain MiniLM it fixed one hand-picked pronoun chunk (rank 5 to 3) but lowered overall hit@1.

### Contextual retrieval (step 06)
Anthropic's method: an LLM writes situating context per chunk; prepend it before embedding and before BM25. Caching the document prompt makes it affordable (3.1x fewer input-token equivalents here). The mock heuristic gave no gain on a corpus whose chunks keep their headings and a small gain when headings were lost.

### Hierarchical retrieval (step 07)
Small chunks match precisely; their parents answer. `parent_id` links in a docstore, only leaves embedded. Small-to-big, sentence window and auto-merging were the best on hit@1.

### Propositions and agentic chunking (step 08)
Atomic facts (Dense X Retrieval) and LLM-guided boundaries. Strong on paper for dense retrieval; the real cost is ingestion: 140 calls and 5.3x the corpus in tokens for propositions.

### Layout-aware chunking (step 09)
Chunk parsed elements: reading order, drop headers/footers, section start at headers, whole tables or row groups with header, cross-page table merge, caption binding, page + bbox kept. 6 of 7 questions answered with a page and box citation; the flatten-and-window baseline had none.

### Evaluation (step 10) and design (step 11)
One harness, offsets as ground truth, intervals for noise. A frozen `Chunk` with a content-hash id, a `Chunker` Protocol and registry, config hashes, idempotent re-chunking (one edited number re-embedded exactly 1 of 19 chunks), versioned indexes with an alias flip and an embedding cache, streaming, CJK separators.

## 8. Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| chunk | passage, node (LlamaIndex), split, segment, document fragment, text unit, window | Yes, same idea | LlamaIndex nodes also carry relationships (parent, previous, next); "passage" is older IR vocabulary; "document" in LangChain is a chunk with metadata |
| overlap | stride (step = size - overlap), window slide, chunk_overlap | Overlap and stride are two views of one number | overlap is what is shared; stride is how far the window moves |
| parent-child | small-to-big, parent-document retrieval, hierarchical chunking, auto-merging | Related | all index small pieces and send bigger ones; auto-merging decides *when* to swap children for their parent |
| sentence-window retrieval | context-window retrieval | Same | neighbours by position, not by tree |
| contextual retrieval | contextual chunk headers, contextual embeddings, chunk enrichment | Mostly | Anthropic's name for an LLM-written header; a breadcrumb from the heading path is the cheap, non-LLM cousin |
| semantic chunking vs late chunking vs proposition chunking | | No | semantic = where to cut (by embedding distance); late = how to embed (pool after a full-context pass); proposition = what to index (rewritten atomic facts) |
| recursive splitter | RecursiveCharacterTextSplitter, hierarchical separators | Yes | "character" in the name means the default length is characters; pass a token counter |
| chunk_size in characters vs tokens | length_function, max_tokens | No | 512 characters is about 110 to 130 English tokens but about 300 for JSON; the model limit is tokens |
| RAPTOR | recursive summary tree, collapsed-tree retrieval | Same | summaries of clusters at several levels, indexed together with the leaves |
| layout-aware chunking | document-structure chunking, Docling HybridChunker, Unstructured chunk_by_title | Family | all consume parsed elements; they differ in token awareness and table handling |

## 9. Decision guide: which chunker for which document
| Your document | Start with | Why | Watch out for |
|---|---|---|---|
| Clean Markdown / HTML with real headings (docs, wikis, handbooks) | Structure-aware, section chunks of 100 to 250 tokens, breadcrumb in `embed_text` | the author already chose the boundaries; breadcrumb restores context | tiny sections (merge them), huge sections (split inside blocks) |
| Reference with tables and code (API docs, specs) | Structure-aware with atomic blocks; tables whole or row groups with repeated header; code by function | splitters cut fences and header rows | embed linearised rows; keep the header in every table chunk |
| FAQ, Q/A pairs | One chunk per question + answer | the pair is the unit | do not merge neighbouring Q/As |
| Plain prose with no structure (reports, transcripts, postmortems) | Sentence-packing or recursive at 100 to 200 tokens, 10 to 15% overlap | nothing to follow; try semantic only if you can measure a gain | semantic chunking costs a model pass and often ties recursive |
| Mixed-length answers: precise match but wide context needed | Small-to-big or auto-merging over structure-aware sections | best hit@1 and MRR here | needs a parent store and ids; send de-duplicated parents |
| Chunks that say "it", "this policy", "the limit" | Breadcrumb first; then contextual retrieval (LLM + prompt caching); late chunking only with a long-context model made for it | a chunk must make sense alone | LLM cost per chunk; re-run when the chunker or prompt changes |
| Scanned or PDF documents, tables, figures, citations to pages | Parse first, then layout-aware chunking on elements; keep page + bbox | text-only extraction loses reading order and tables | merge tables across pages; drop headers and footers |
| Code repositories | Language-aware separators (class / def) or AST-based chunks | functions are the unit | spaces vs tabs in separator regexes |
| Multilingual (CJK, Hindi) | Multilingual tokenizer for counting; punctuation separators of that language | whitespace splitting fails; MiniLM maps CJK to `[UNK]` | different chars/token per language |
| Very large corpus, tight ingestion budget | Recursive or structure-aware, no LLM calls | LLM-based chunkers cost 5x the corpus in tokens | measure before paying |

If unsure: **structure-aware where structure exists, otherwise recursive on a token counter at 100 to 200 tokens, plus a breadcrumb, evaluated on your own questions.** Add hierarchy (small-to-big) next; add LLM-based techniques only when a failure analysis shows a context problem.

## 10. Common mistakes and gotchas
- Counting characters when the model limit is tokens; or counting with a different tokenizer than the embedder's.
- Not noticing silent truncation: a 481-token text embedded at `max_length=256` had 225 tokens never read, and its vector was identical to the truncated text's.
- Overlap equal to or larger than size (step 0 or negative).
- Treating a `# comment` inside a code fence as a heading.
- Splitting a table so that rows lose the header row.
- Position-based chunk ids: one inserted paragraph renames every later chunk and forces a full re-embed.
- Declaring a winner on 74 questions: differences below about 0.10 are noise.
- Evaluating chunkers by chunk-id gold labels: they differ per chunker; use answer spans.
- Evaluating hit@k only: a bigger chunk makes hit@k easier while making the LLM's job and the bill worse; also track context tokens and precision.
- Using a mock LLM result as if it were an LLM result (steps 06 and 08 are honest about this).

## 11. Self-check
1. Why is `chunk_size=512` with the default `len` not 512 tokens? What changes with `length_function=count_tokens`?
2. What happens to tokens 257 to 481 of a text embedded with `max_length=256`? How would you detect it?
3. Walk the recursive splitter on a 300-token paragraph with chunk_size 128: which separators fire, in which order?
4. Why does `merge` keep the pieces in `current_doc` when carrying overlap, and when does it drop them?
5. Name three things a breadcrumb gives you. Where does it go: `text`, `embed_text`, `display_text`?
6. How does a heading tree handle `# comment` lines inside a code fence?
7. Compare percentile, standard-deviation and gradient breakpoints. Which fails on a short document?
8. Why does late chunking need a model trained for it?
9. What does contextual retrieval cost without prompt caching, and with it? What is cached?
10. In small-to-big, what is embedded, what is stored in the docstore, what is sent to the LLM?
11. How does auto-merging decide to return a parent?
12. Why are propositions expensive at ingestion? What do you store besides the proposition?
13. How do you keep a citation (page, bbox) through chunking of a parsed PDF?
14. How do you evaluate chunkers whose chunks differ in text and boundaries?
15. Why is a content-hash chunk id better than a position-based one? What exactly goes in the hash?
16. A new chunker version ships. Describe the zero-downtime migration.

## 12. Interview questions (short answers in [INTERVIEW_GIST.md](INTERVIEW_GIST.md))
1. How would you choose a chunk size?
2. Why measure in tokens? How do you count them for a given embedder?
3. Explain recursive character splitting and what `chunk_overlap` does.
4. How do you chunk Markdown, tables, code, and PDFs?
5. What is semantic chunking and when does it lose?
6. What is late chunking? Contextual retrieval? Compare them.
7. Explain parent-document retrieval, sentence-window and auto-merging retrieval.
8. What are propositions, and what do they cost?
9. How would you evaluate a chunking strategy?
10. Design the data model of a chunk for production. What goes in the id?
11. How do you roll out a new chunker without downtime?
12. How do you handle multilingual or CJK text?

## References (checked Nov 2026)
- Anthropic, *Introducing Contextual Retrieval* (Sept 2024; 49% fewer failed top-20 retrievals combining contextual embeddings and BM25, 67% with reranking): <https://www.anthropic.com/engineering/contextual-retrieval>. The $1.02 per million document tokens figure appears only in secondary summaries in my search; check the post and the current pricing page.
- Günther et al., *Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models* (Jina AI): <https://arxiv.org/abs/2409.04701>
- Jina, *jina-embeddings-v3* (8,192-token context; `late_chunking` request parameter, total 8,192 tokens per request when enabled): <https://jina.ai/news/jina-embeddings-v3-a-frontier-multilingual-embedding-model>. jina-embeddings-v4 (3.8B, 32,768-token text input per the MRL 2025 paper) and jina-embeddings-v5-text (Feb 2026): I could not confirm whether they expose late chunking natively; check their model cards.
- Qu, Tu, Bao, *Is Semantic Chunking Worth the Computational Cost?* (2024; benefits "inconsistent and often insufficient to justify the computational cost"): <https://arxiv.org/abs/2410.13070>
- Chen et al., *Dense X Retrieval: What Retrieval Granularity Should We Use?* (propositions): <https://arxiv.org/abs/2312.06648>
- Sarthi et al., *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval*: <https://arxiv.org/abs/2401.18059>
- LlamaIndex node parser modules (SentenceWindowNodeParser, HierarchicalNodeParser, AutoMergingRetriever, SemanticSplitterNodeParser): <https://developers.llamaindex.ai/python/framework/module_guides/loading/node_parsers/modules/>
- LangChain text splitters (`langchain-text-splitters`; RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter): <https://reference.langchain.com/python/langchain-text-splitters/langchain_text_splitters>
- Docling chunking concepts and HybridChunker: <https://docling-project.github.io/docling/concepts/chunking/>
- Unstructured `chunk_by_title` (max_characters, new_after_n_chars, combine_text_under_n_chars, multipage_sections; the sources disagree on the `multipage_sections` default, check your version): <https://docs.unstructured.io/api-reference/workflow/nodes/chunker/chunker-chunk-by-title>
- Chonkie (TokenChunker, RecursiveChunker, optional semantic extras): <https://github.com/chonkie-inc/chonkie>; semchunk: <https://pypi.org/project/semchunk>
- Smith and Troynikov, *Evaluating Chunking Strategies for Retrieval* (Chroma Research, 2024): I could not reach the primary page in my search, only aggregator summaries, so no numbers from it are used here.
