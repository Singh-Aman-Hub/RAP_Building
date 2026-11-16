<!-- vocabulary draft: 19_chunking_deep_dive -->

### Chunk (passage, node, split, segment)
<!-- cat: llm -->
- **Technical definition:** A contiguous piece of a document that is embedded, indexed and retrieved as one unit. Libraries call it a chunk, passage, node (LlamaIndex), split or segment.
- **Plain English:** The "page" of your knowledge base that the search engine can hand to the LLM.
- **Example / analogy:** Cutting a long scroll into index cards. Each card must make sense when read alone.
- **Snippet:**
```python
chunk = {"doc_id": "handbook", "start": 675, "end": 958, "text": doc[675:958]}
```
- **Snippet explained:** A chunk is best stored as offsets into the source plus its text, so you can cite and highlight it later.
- **Related terms:** [[Chunking]], [[Chunk overlap (stride, window)]], [[Chunk ID (content hash)]]

### Chunking
<!-- cat: llm -->
- **Technical definition:** The preprocessing step of a RAG pipeline that splits documents into chunks sized for the embedding model and the LLM context, ideally along natural boundaries.
- **Plain English:** Deciding where to cut a document so that each piece holds one idea.
- **Example / analogy:** A good editor cuts at paragraph and section breaks, not in the middle of a sentence.
- **Snippet:**
```python
chunks = RecursiveSplitter(chunk_size=100, chunk_overlap=0).split_document(doc)
```
- **Snippet explained:** Split a document into pieces of at most 100 tokens, preferring paragraph, then line, then word boundaries.
- **Related terms:** [[Recursive character splitting]], [[Structure-aware chunking (heading-aware, markdown-aware)]], [[Semantic chunking (breakpoint, percentile, standard deviation, gradient)]]

### Token (vs character vs word)
<!-- cat: llm -->
- **Technical definition:** The unit a model's tokenizer produces (a sub-word piece for WordPiece/BPE/SentencePiece). Model input limits and costs are counted in tokens, not characters.
- **Plain English:** The model does not read letters or words; it reads pieces, and the number of pieces is what is limited and billed.
- **Example / analogy:** English averages about 4-5 characters per token in MiniLM's tokenizer; JSON, URLs and Chinese often give 1-2.
- **Snippet:**
```python
ids = tokenizer(text, add_special_tokens=False)["input_ids"]
n_tokens = len(ids)
```
- **Snippet explained:** Count tokens with the SAME tokenizer as the embedding model, never with len(text).
- **Related terms:** [[max_seq_length (silent truncation)]], [[Token-based chunking (fixed-size windows)]]

### max_seq_length (silent truncation)
<!-- cat: llm -->
- **Technical definition:** The maximum number of tokens an embedding model reads per input (all-MiniLM-L6-v2: 256 in sentence-transformers by default, 512 positions in the architecture). Longer input is truncated without an error.
- **Plain English:** The model stops reading after a fixed number of tokens and does not tell you.
- **Example / analogy:** Handing a 5-page letter to someone who only reads the first page and never says so.
- **Snippet:**
```python
batch = tok(texts, truncation=True, max_length=256, padding=True, return_tensors="pt")
```
- **Snippet explained:** truncation=True cuts everything after token 256; a fact at token 400 never reaches the vector.
- **Related terms:** [[Token (vs character vs word)]], [[Late chunking]]

### Token-based chunking (fixed-size windows)
<!-- cat: llm -->
- **Technical definition:** Cutting text into windows of N tokens, each starting N - overlap tokens after the previous one, using tokenizer offsets to return exact character slices.
- **Plain English:** Cut every N tokens no matter what the text says.
- **Example / analogy:** Slicing bread with a ruler.
- **Snippet:**
```python
spans = tok(text, return_offsets_mapping=True)["offset_mapping"]
window = spans[i:i + 128]
chunk = text[window[0][0]:window[-1][1]]
```
- **Snippet explained:** offset_mapping gives (char_start, char_end) per token, so token windows map back to text exactly.
- **Related terms:** [[Chunk overlap (stride, window)]], [[Recursive character splitting]]

### Chunk overlap (stride, window)
<!-- cat: llm -->
- **Technical definition:** Consecutive chunks share the last M units of the previous chunk; the step between chunk starts (stride) is size - overlap.
- **Plain English:** Each chunk repeats a little of the previous one, so a fact on the border appears whole somewhere.
- **Example / analogy:** Subtitles that repeat the last words of the previous line.
- **Snippet:**
```python
step = size - overlap          # overlap must be < size
```
- **Snippet explained:** overlap == size gives step 0 (infinite loop or ValueError); validate it.
- **Related terms:** [[Token-based chunking (fixed-size windows)]], [[Recursive character splitting]]

### Recursive character splitting
<!-- cat: llm -->
- **Technical definition:** A splitter that tries a list of separators from coarse to fine (paragraph, line, word, character), merges small pieces up to chunk_size and recurses with finer separators only into pieces that are still too long. LangChain's RecursiveCharacterTextSplitter is the reference.
- **Plain English:** Cut at the biggest natural break that makes the pieces small enough.
- **Example / analogy:** Cut a cake at the big slices first; only slice a piece further if it is too big for the plate.
- **Snippet:**
```python
splitter = RecursiveSplitter(128, 20, separators=["\n\n", "\n", " ", ""])
```
- **Snippet explained:** Size and overlap are measured by the length function (use a token counter).
- **Related terms:** [[length_function]], [[keep_separator]], [[Language-specific separators]]

### length_function
<!-- cat: python -->
- **Technical definition:** The callable a splitter uses to measure text size. Default len (characters); should be the embedding model's token counter.
- **Plain English:** The ruler. If it measures characters, your "128" is not 128 tokens.
- **Example / analogy:** A shop that sells by weight but measures by the number of items.
- **Snippet:**
```python
RecursiveSplitter(128, 0, length_function=count_tokens)
```
- **Snippet explained:** Now chunk_size=128 means at most 128 MiniLM tokens.
- **Related terms:** [[Token (vs character vs word)]], [[Recursive character splitting]]

### keep_separator
<!-- cat: python -->
- **Technical definition:** Option of recursive splitters controlling whether the separator is dropped, kept at the start of the next piece ("start") or at the end of the previous one ("end").
- **Plain English:** Whether the "## " or the blank line stays with the text it introduced.
- **Example / analogy:** A heading should stay with its section, not with the section above.
- **Snippet:**
```python
parts = re.split(f"({sep})", text)
pieces = [parts[0]] + [parts[i] + parts[i + 1] for i in range(1, len(parts), 2)]   # keep="start"
```
- **Snippet explained:** A capture group keeps the separators in the split result; they are then glued to the following piece.
- **Related terms:** [[Recursive character splitting]], [[Language-specific separators]]

### Language-specific separators
<!-- cat: llm -->
- **Technical definition:** Separator lists tuned to a syntax: Markdown headings and fences, Python "class "/"def " boundaries, CJK punctuation ("。") for languages without spaces.
- **Plain English:** Tell the splitter where the natural cut points are in this kind of text.
- **Example / analogy:** A butcher cuts at joints; a baker cuts at layers.
- **Snippet:**
```python
PYTHON = [r"\nclass ", r"\ndef ", r"\n\tdef ", "\n\n", "\n", " ", ""]
```
- **Snippet explained:** Note that methods indented with spaces are not matched by "\n\tdef "; they fall through to "\n\n".
- **Related terms:** [[Recursive character splitting]], [[Structure-aware chunking (heading-aware, markdown-aware)]]

### Structure-aware chunking (heading-aware, markdown-aware)
<!-- cat: llm -->
- **Technical definition:** Chunking along the document's own structure: parse headings, paragraphs, lists, tables and code into blocks; make one chunk per section; never split an atomic block; split only oversize sections.
- **Plain English:** Respect the author's sections, tables and code blocks.
- **Example / analogy:** Tear a book at chapter breaks, not at page 37 line 12.
- **Snippet:**
```python
blocks = parse_blocks(text)
tree = build_tree(blocks, title)
chunks = StructureChunker(200, 40).split_document(doc)
```
- **Snippet explained:** Parse blocks, build the heading tree, emit one chunk per section with a breadcrumb.
- **Related terms:** [[Breadcrumb header (section path)]], [[Atomic block]], [[Heading tree]]

### Heading tree
<!-- cat: python -->
- **Technical definition:** A tree whose nodes are sections: a heading of level L closes all open sections of level >= L and becomes a child of the nearest shallower section.
- **Plain English:** The outline of the document as a tree.
- **Example / analogy:** The table of contents with indentation.
- **Snippet:**
```python
while stack[-1].level >= b.level:
    stack.pop()
sec = Section(b.level, b.title, parent=stack[-1])
```
- **Snippet explained:** A stack of open sections; pop until the parent is shallower than the new heading.
- **Related terms:** [[Structure-aware chunking (heading-aware, markdown-aware)]], [[Breadcrumb header (section path)]]

### Breadcrumb header (section path)
<!-- cat: llm -->
- **Technical definition:** The chain of headings above a chunk ("Doc > 2. Time off > 2.3 Parental leave"), prepended to the text that is embedded and indexed, and stored as metadata.
- **Plain English:** A label that tells the embedding model and the reader where the chunk lives.
- **Example / analogy:** The address on a letter: a sentence like "It is 18 weeks" means little without it.
- **Snippet:**
```python
embed_text = f"{' > '.join(path)}\n{chunk_text}"
```
- **Snippet explained:** The header goes into embed_text (and usually display_text) but the stored `text` stays the exact source slice.
- **Related terms:** [[Contextual retrieval (contextual chunk headers, contextual embeddings)]], [[embed_text vs display_text]]

### Atomic block
<!-- cat: llm -->
- **Technical definition:** A piece of a document that should never be cut inside: a table, a fenced code block, a list item run, a figure with its caption.
- **Plain English:** Things that stop making sense when halved.
- **Example / analogy:** Do not cut a recipe's ingredient list in half.
- **Snippet:**
```python
elif stripped.startswith("```"):          # fenced code: runs to the closing fence
    j = i + 1
    while j < len(lines) and not lines[j].strip().startswith("```"):
        j += 1
```
- **Snippet explained:** The whole fence is one block, so a '# comment' inside it is never taken for a heading.
- **Related terms:** [[Structure-aware chunking (heading-aware, markdown-aware)]], [[Table chunking (row-wise with repeated header)]]

### Table chunking (row-wise with repeated header)
<!-- cat: llm -->
- **Technical definition:** Keeping a table in one chunk if it fits, else splitting into row groups and repeating the header row in each, optionally linearising rows into "Column: value" text for embedding.
- **Plain English:** Every piece of a big table still knows what its columns mean.
- **Example / analogy:** Photocopy the column titles onto every page of a long spreadsheet printout.
- **Snippet:**
```python
"; ".join(f"{h}: {v}" for h, v in zip(header, row))
```
- **Snippet explained:** Turns ['Pro', '600'] into 'Plan: Pro; Requests per minute: 600', which embedding models read better than pipes.
- **Related terms:** [[Atomic block]], [[Layout-aware chunking]]

### Semantic chunking (breakpoint, percentile, standard deviation, gradient)
<!-- cat: llm -->
- **Technical definition:** Embed sentences (optionally with neighbour windows), compute cosine distance between consecutive sentences, and cut where the distance exceeds a threshold: a percentile of all distances, mean + k standard deviations, mean + k IQR, or the percentile of the distance gradient.
- **Plain English:** Cut where the topic seems to change according to the embedding model.
- **Example / analogy:** Noticing that the conversation moved on because the words suddenly differ.
- **Snippet:**
```python
d = 1 - (E[:-1] * E[1:]).sum(axis=1)
cuts = set(np.where(d > np.percentile(d, 85))[0])
```
- **Snippet explained:** E holds unit sentence vectors; d[i] is the distance between sentence i and i+1; the top 15% of distances become cuts.
- **Related terms:** [[Buffered window (semantic chunking)]], [[Late chunking]], [[Proposition chunking]]

### Buffered window (semantic chunking)
<!-- cat: llm -->
- **Technical definition:** Gluing buffer_size neighbouring sentences on each side to a sentence before embedding it, so one very short sentence does not create a fake distance spike.
- **Plain English:** Look at a sentence together with its neighbours before judging the topic.
- **Example / analogy:** A moving average for text.
- **Snippet:**
```python
windows = [" ".join(texts[max(0, i - b): i + b + 1]) for i in range(len(texts))]
```
- **Snippet explained:** Each sentence is replaced by a (2b+1)-sentence window; it triples the tokens embedded when b=1.
- **Related terms:** [[Semantic chunking (breakpoint, percentile, standard deviation, gradient)]]

### Late chunking
<!-- cat: llm -->
- **Technical definition:** Run a long-context embedding model once over the whole text, keep the token-level output vectors, and mean-pool the vectors of each chunk's token span, so every chunk vector has attended to the surrounding text (Jina AI, arXiv 2409.04701).
- **Plain English:** Read the whole page first, then summarise each paragraph into a vector.
- **Example / analogy:** Understanding "It lasts 18 weeks" because you read the heading before it.
- **Snippet:**
```python
H = model(**enc).last_hidden_state[0]          # (T, d) token vectors for the whole window
v = H[token_idx_of_chunk].mean(dim=0)          # pool only this chunk's tokens
```
- **Snippet explained:** One forward pass; pooling happens after, per chunk span.
- **Related terms:** [[Contextual retrieval (contextual chunk headers, contextual embeddings)]], [[max_seq_length (silent truncation)]]

### Contextual retrieval (contextual chunk headers, contextual embeddings)
<!-- cat: llm -->
- **Technical definition:** Anthropic's method: for each chunk an LLM writes 1-2 sentences situating it in its document; the text is prepended before embedding (contextual embeddings) and before BM25 indexing (contextual BM25). Prompt caching makes the per-chunk calls cheap.
- **Plain English:** Give every chunk a short note saying what it is about and where it comes from.
- **Example / analogy:** A sticky note on a loose page: "From the 2024 lease, section on deposits."
- **Snippet:**
```python
ctx = llm(f"<document>{doc}</document> <chunk>{chunk}</chunk> Situate this chunk ...")
embed_text = ctx + "\n" + chunk
```
- **Snippet explained:** One LLM call per chunk; the context is part of what is embedded and BM25-indexed but not necessarily shown to the user.
- **Related terms:** [[Breadcrumb header (section path)]], [[Prompt caching (for chunk contextualisation)]]

### Prompt caching (for chunk contextualisation)
<!-- cat: api -->
- **Technical definition:** Provider feature that stores a long, repeated prompt prefix so later calls read it at a fraction of the input price. When contextualising N chunks of one document, the whole document is the shared prefix.
- **Plain English:** Pay full price to send the document once, then a small fee each time you re-use it.
- **Example / analogy:** A library card: the first time you copy the book, later you just look at it.
- **Snippet:**
```python
cost = doc_tokens * 1.25 + doc_tokens * (n_chunks - 1) * 0.10   # write 1.25x, read 0.10x (check current pricing)
```
- **Snippet explained:** Multipliers come from Anthropic's prompt-caching docs; verify current prices before budgeting.
- **Related terms:** [[Contextual retrieval (contextual chunk headers, contextual embeddings)]]

### Parent-document retrieval (small-to-big, hierarchical)
<!-- cat: llm -->
- **Technical definition:** Index small child chunks for precise matching, but return their larger parent (section or document) to the LLM. Needs parent_id links and a docstore.
- **Plain English:** Find with a small net, hand over the whole page.
- **Example / analogy:** Search a book index for a word, then read the whole paragraph.
- **Snippet:**
```python
parent = nodes[nodes[leaf_id].parent_id]
context = parent.text
```
- **Snippet explained:** Only leaves are embedded; parents live in a key-value store and are fetched by id; de-duplicate parents.
- **Related terms:** [[Sentence-window retrieval]], [[Auto-merging retrieval]], [[RAPTOR (recursive summary tree)]]

### Sentence-window retrieval
<!-- cat: llm -->
- **Technical definition:** Index single sentences; at query time return the matched sentence plus w sentences on each side.
- **Plain English:** Match on one sentence, read a few around it.
- **Example / analogy:** Highlighter hit, then reading the surrounding lines.
- **Snippet:**
```python
window = sentences[max(0, j - 2): j + 3]
```
- **Snippet explained:** w=2 gives five sentences of context for each hit.
- **Related terms:** [[Parent-document retrieval (small-to-big, hierarchical)]]

### Auto-merging retrieval
<!-- cat: llm -->
- **Technical definition:** Retrieve leaf chunks of a hierarchy; if more than a ratio (e.g. 50%) of a parent's children are in the result, replace them by the parent, recursively (LlamaIndex AutoMergingRetriever).
- **Plain English:** If you found most of a section's pieces, just send the section.
- **Example / analogy:** If you collected 3 of the 4 puzzle pieces of a corner, take the whole corner.
- **Snippet:**
```python
if len(kids) / len(nodes[p].child_ids) > 0.5:
    result.add(p)
```
- **Snippet explained:** The parent inherits the best child rank; the fragments are dropped.
- **Related terms:** [[Parent-document retrieval (small-to-big, hierarchical)]]

### RAPTOR (recursive summary tree)
<!-- cat: llm -->
- **Technical definition:** Build a tree over chunks: cluster chunks, summarise each cluster with an LLM, embed the summaries, repeat until one root; retrieve from all levels together (arXiv 2401.18059).
- **Plain English:** Summaries of summaries, so broad questions can match a high-level node and detailed ones a leaf.
- **Example / analogy:** Chapter summaries and a book summary stored next to the pages.
- **Snippet:**
```python
labels = KMeans(k).fit_predict(vecs)
summary = summarise([texts[i] for i in members])
```
- **Snippet explained:** One summariser (LLM) call per cluster per level.
- **Related terms:** [[Parent-document retrieval (small-to-big, hierarchical)]]

### Proposition chunking
<!-- cat: llm -->
- **Technical definition:** Rewriting passages into atomic, self-contained factual statements (propositions) with pronouns resolved, and indexing those instead of passages (Dense X Retrieval, arXiv 2312.06648).
- **Plain English:** One fact per chunk, written so it makes sense alone.
- **Example / analogy:** Flash cards instead of paragraphs.
- **Snippet:**
```python
props = json.loads(llm(PROP_PROMPT.format(passage=passage)))["propositions"]
```
- **Snippet explained:** One LLM call per passage; the output can be 5x the input in tokens over a whole corpus.
- **Related terms:** [[Agentic chunking]], [[Semantic chunking (breakpoint, percentile, standard deviation, gradient)]]

### Agentic chunking
<!-- cat: llm -->
- **Technical definition:** Using an LLM to decide chunk boundaries or grouping, e.g. asking for each new sentence whether it belongs to the current chunk or starts a new one.
- **Plain English:** Let a model act as the editor that decides where to cut.
- **Example / analogy:** A human editor reading the text and marking cuts.
- **Snippet:**
```python
verdict = llm(AGENT_PROMPT.format(chunk=current, sentence=s))   # "SAME" or "NEW"
```
- **Snippet explained:** At least one LLM call per sentence; quality depends on the model and the prompt, cost scales with document length.
- **Related terms:** [[Proposition chunking]], [[Semantic chunking (breakpoint, percentile, standard deviation, gradient)]]

### Layout-aware chunking
<!-- cat: llm -->
- **Technical definition:** Chunking a parsed document (list of elements with type, text, page, bbox, reading order): sort by reading order, drop headers and footers, start chunks at section headers, keep tables and figures with captions, merge tables across pages, keep page and bounding box for citations. Docling's HybridChunker and Unstructured's chunk_by_title implement versions of this.
- **Plain English:** Chunk what the page looks like, not just the text it contains.
- **Example / analogy:** Cutting a newspaper by articles and photos, not by line number.
- **Snippet:**
```python
els = sorted(doc["elements"], key=lambda e: (e["page"], e["reading_order"]))
```
- **Snippet explained:** Array order from a parser is not reading order.
- **Related terms:** [[Table chunking (row-wise with repeated header)]], [[Bounding box (bbox)]]

### Bounding box (bbox)
<!-- cat: ml -->
- **Technical definition:** A rectangle [x0, y0, x1, y1] in page coordinates (PDF points, origin top-left here) locating an element on a page.
- **Plain English:** Where on the page the text sits.
- **Example / analogy:** The coordinates you need to draw a yellow highlight over a cited sentence.
- **Snippet:**
```python
box = [min(x0s), min(y0s), max(x1s), max(y1s)]    # union of element boxes on a page
```
- **Snippet explained:** A chunk spanning several elements keeps one union box per page.
- **Related terms:** [[Layout-aware chunking]]

### hit@k, MRR, recall@k (retrieval metrics)
<!-- cat: ml -->
- **Technical definition:** hit@k = 1 if any relevant item is in the top k; recall@k = share of all relevant items found in the top k; MRR = mean of 1/rank of the first relevant item.
- **Plain English:** Did the answer show up in the first few results, and how high?
- **Example / analogy:** A quiz where you get 1 point if the right answer is in your top 3 guesses.
- **Snippet:**
```python
rr = 1 / first_rank if first_rank else 0.0
```
- **Snippet explained:** Averaged over all questions; with 74 questions one question is worth 0.0135.
- **Related terms:** [[Context precision (chunk attribution)]], [[Bootstrap confidence interval]]

### Context precision (chunk attribution)
<!-- cat: ml -->
- **Technical definition:** The share of the characters (or tokens) sent to the LLM that belong to the answer evidence. Low values mean you pay for text that does not help; attribution metrics also ask which chunk an answer sentence is supported by.
- **Plain English:** How much of what you sent was actually useful.
- **Example / analogy:** Mailing a 300-page book to deliver one sentence.
- **Snippet:**
```python
precision = gold_chars_in_context / total_context_chars
```
- **Snippet explained:** Computed from character offsets, so it works for any chunker.
- **Related terms:** [[hit@k, MRR, recall@k (retrieval metrics)]]

### Bootstrap confidence interval
<!-- cat: ml -->
- **Technical definition:** Estimating the uncertainty of a metric by resampling the evaluation questions with replacement many times and taking the 2.5th and 97.5th percentiles of the resampled means.
- **Plain English:** How much would my score change if I had asked slightly different questions?
- **Example / analogy:** Re-running a poll on random re-draws of the same respondents.
- **Snippet:**
```python
means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(1000)]
lo, hi = np.percentile(means, [2.5, 97.5])
```
- **Snippet explained:** With 74 questions the interval of hit@3 is about +-0.07, so small differences are noise.
- **Related terms:** [[hit@k, MRR, recall@k (retrieval metrics)]]

### Chunk ID (content hash)
<!-- cat: python -->
- **Technical definition:** A stable identifier derived from the chunk's content and the chunker's identity (hash of doc_id, chunker id, version, config hash, text, occurrence) rather than its position.
- **Plain English:** The same text always gets the same ID, so editing one paragraph does not rename every chunk after it.
- **Example / analogy:** A fingerprint instead of a queue number.
- **Snippet:**
```python
chunk_id = sha256(f"{doc_id}|{chunker}|{version}|{config_hash}|{sha(text)}|{occ}")[:20]
```
- **Snippet explained:** Enables idempotent re-chunking and diffing: only added/removed ids need embedding/deleting.
- **Related terms:** [[Idempotent re-chunking]], [[Versioned index (alias switch)]]

### Idempotent re-chunking
<!-- cat: llm -->
- **Technical definition:** Running the chunker again on unchanged input yields identical chunks and ids; on changed input only the affected chunks differ, so only they are re-embedded.
- **Plain English:** Re-running ingestion is safe and cheap.
- **Example / analogy:** `make` rebuilding only the files that changed.
- **Snippet:**
```python
d = diff_chunks(old_chunks, new_chunks)    # kept / added / removed by chunk_id
```
- **Snippet explained:** Upsert d.added, delete d.removed, leave d.kept untouched.
- **Related terms:** [[Chunk ID (content hash)]], [[Versioned index (alias switch)]]

### Versioned index (alias switch)
<!-- cat: llm -->
- **Technical definition:** Naming each index by collection, chunker id and version, config hash and embedding model; building a new version in the background, then atomically flipping an alias; keeping the old one for rollback. An embedding cache keyed by text hash avoids re-embedding unchanged text.
- **Plain English:** Never edit the live index in place; build a new one and swap.
- **Example / analogy:** Blue-green deployment for your search index.
- **Snippet:**
```python
name = f"{collection}__{chunker_id}@{version}__{config_hash[:6]}__{embed_model}"
```
- **Snippet explained:** Any change to chunker behaviour, config or embedding model produces a new name, so versions never mix.
- **Related terms:** [[Idempotent re-chunking]], [[embed_text vs display_text]]

### embed_text vs display_text
<!-- cat: llm -->
- **Technical definition:** Two views of a chunk: embed_text is what the embedder and BM25 index see (text plus breadcrumb/context), display_text is what is shown to the user or put in the LLM prompt (often cleaner, or a table as markdown).
- **Plain English:** What you search with and what you show can differ.
- **Example / analogy:** A library catalogue card versus the book.
- **Snippet:**
```python
Chunk(text=body, embed_text=f"{crumb}\n{body}", display_text=body)
```
- **Snippet explained:** Keep `text` as the exact source slice for offsets; vary the other two per strategy.
- **Related terms:** [[Breadcrumb header (section path)]], [[Chunk ID (content hash)]]

### Strategy registry (Protocol)
<!-- cat: python -->
- **Technical definition:** A typing.Protocol describing the chunker interface plus a name-to-class dict filled by a decorator, so strategies are selected from config by name.
- **Plain English:** A menu of chunkers that all plug into the same socket.
- **Example / analogy:** USB: any device that follows the standard works.
- **Snippet:**
```python
@register("recursive")
class RecursiveChunker: ...
chunker = create("recursive", config, count_tokens)
```
- **Snippet explained:** The decorator stores the class under a name; create() builds it with the config and the injected token counter.
- **Related terms:** [[Chunk ID (content hash)]], [[Chunking]]
