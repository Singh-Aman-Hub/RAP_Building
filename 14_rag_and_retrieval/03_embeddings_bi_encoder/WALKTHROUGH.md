# Step 03 — Embeddings with a bi-encoder

## The story so far
[Step 02](../02_chunking_strategies/WALKTHROUGH.md) turned 9 documents into 36 heading-aware chunks that never cut a sentence and keep every answer intact. Now each chunk needs a representation a computer can compare with a question.

## The pain point
The minimal RAG in module 07 used TF-IDF, which only counts shared words. "How long can I wait before asking for my money back?" and "Refunds are issued within 9 days..." share no words, so TF-IDF scores them 0.00. Users never phrase questions the way the documentation is written.

## The spark
Use a neural **embedding model** that maps text to a vector where *meaning*, not spelling, decides closeness. A bi-encoder encodes each text separately, so all 36 chunk vectors can be computed once, offline, and only the question is encoded at query time. The same pair scores 0.46 with MiniLM.

## Goal
Build the embedding pipeline by hand (tokens -> hidden states -> mean pooling -> L2 normalisation), compare it with TF-IDF on five sentence pairs, prove that truncation silently drops text, see why e5-style models want `query:` / `passage:` prefixes, and embed the whole knowledge base.

## The theory (read this before the interview)
**What an embedding is.** A function `f(text) -> vector` (here 384 numbers) trained so that texts with similar meaning get vectors pointing in similar directions. Similarity is measured with cosine: `cos(a, b) = a·b / (|a| |b|)`, from -1 to 1. If every vector is normalised to length 1, cosine is just the dot product, which is what makes search a single matrix multiplication.

**Bi-encoder architecture.**
```
query   -> encoder -> pool -> normalise -> q  (384,)  \
                                                       >  score = q · d
passage -> encoder -> pool -> normalise -> d  (384,)  /
```
Query and passage never see each other inside the model. That is the weakness (no word-by-word interaction, so details blur) and the strength (passage vectors are precomputed once; a query costs one encoder pass plus a dot product per passage). A **cross-encoder** reads query and passage together and is more accurate but must run once per pair, so it is used to re-rank a short list (step 07).

**Pooling.** The transformer outputs one vector per token, shape `(batch, tokens, 384)`. Pooling reduces that to one vector per text:
- *mean pooling*: average the token vectors, using the attention mask so padding does not count (all-MiniLM-L6-v2, e5);
- *CLS pooling*: take the first token's vector (many BERT-style and bge models).
Use the pooling the model was **trained** with; the wrong one still produces numbers, just worse ones.

**How bi-encoders are trained: contrastive learning.** Collect (query, relevant passage) pairs. In a batch of N pairs, each query's own passage is the positive and the other N-1 passages are **in-batch negatives** (free negatives, which is why large batches help). **Hard negatives** are passages that look relevant but are not (often mined with BM25 or an earlier model); they teach fine distinctions. The usual loss is InfoNCE, a softmax cross-entropy over similarities with temperature τ:

  `L = -log( exp(sim(q, d+)/τ) / Σ_j exp(sim(q, d_j)/τ) )`

It pulls the positive closer and pushes the negatives away. all-MiniLM-L6-v2 was trained this way on a large set of sentence pairs.

**Dense vs sparse.**
| | Dense (embeddings) | Sparse (TF-IDF, BM25) |
|---|---|---|
| Vector | ~384–1,024 numbers, all non-zero | vocabulary-sized, mostly zeros |
| Matches | meaning, paraphrases, synonyms | exact words, codes, names, numbers |
| Fails on | exact identifiers, rare terms, "refunds" vs "shipping" details | paraphrases with no shared words |
| Needs training | yes | no |
Each covers the other's blind spot, hence hybrid search (step 06).

**Symmetric vs asymmetric.** Symmetric models compare texts of the same kind (sentence vs sentence, duplicate detection). Asymmetric retrieval compares a short question with a longer passage. Some models are trained with role markers and expect them at inference: e5 uses `query: ` and `passage: `, bge recommends an instruction prefix on retrieval queries, nomic uses `search_query: ` / `search_document: `. Read the model card and use the same format at indexing and query time.

**Truncation.** Every encoder has a maximum input length (MiniLM: truncated at 256 tokens by default, trained on 128). Text past it is dropped without warning. That is why you chunk *before* you embed.

**Dimensions and memory.** Memory = number of vectors x dimension x bytes per number. 384 float32 numbers are 1,536 bytes, so 1M chunks need about 1.5 GB. Larger dimensions (768, 1,024, more) can be more accurate but cost memory and search time. Mitigations: float16 or int8 quantisation, and **Matryoshka** embeddings, which are trained so that a prefix of the vector (for example the first 256 numbers) is itself a usable, slightly less accurate embedding.

**Choosing a model.** Start from the MTEB leaderboard (the retrieval tasks, not the overall average), then check: your domain (legal, medical, code), languages (multilingual?), dimension and memory, max input length, latency on your hardware, licence, and whether it needs prefixes. Then evaluate the top candidates on **your own** questions; leaderboard rank does not transfer automatically.

**Same model for index and query, always.** Different models (even with the same dimension) live in unrelated vector spaces. If you change the model, or even its version, re-embed the entire corpus. Store the model name and version with the index.

## Run it
```bash
cd 14_rag_and_retrieval/03_embeddings_bi_encoder
python main.py       # ~10 s on CPU; two cached models (MiniLM, multilingual-e5-small), no network
```

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `with torch.no_grad():` | inference only, no gradient bookkeeping | wraps `model(**batch)` |
| `model(**batch)` | unpack `input_ids`, `attention_mask`, ... as keyword arguments | |
| `.last_hidden_state` | one vector per token from the last layer | shape `(2, 15, 384)` |
| `mask.unsqueeze(-1)` | add a trailing axis so the mask broadcasts over 384 numbers | `(2, 15)` -> `(2, 15, 1)` |
| `(hidden * mask).sum(dim=1) / mask.sum(dim=1)` | masked mean over the token axis | padding contributes 0 |
| `a @ b` | dot product of two 1-D arrays; matrix x vector for many | `matrix @ qv` -> 36 scores |
| `np.argsort(-scores)[:3]` | indices of the 3 largest scores | negate for descending |
| `"text " * 60` | repeat a string | ~600 tokens of filler |
| `["passage: " + p for p in passages]` | list comprehension adding a prefix | |
| `p.numel()` | number of elements in a tensor | parameter count |
| `float(x)` | 0-d array or tensor -> Python float | for formatting |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `AutoTokenizer.from_pretrained(name)` | transformers | tokenizer for the model | model name | tokenizer |
| `tok(texts, ...)` | transformers | texts -> padded id tensors + attention mask | `padding=True`, `truncation=True`, `max_length=256`, `return_tensors="pt"` | `BatchEncoding` |
| `AutoModel.from_pretrained(name).eval()` | transformers | the bare encoder (no task head), inference mode | model name | model |
| `torch.nn.functional.normalize(x, dim=1)` | torch | divide each row by its L2 norm | `dim` | tensor of unit rows |
| `tensor.norm(dim=1)` | torch | length of each row | `dim` | tensor |
| `TfidfVectorizer().fit(texts)` | scikit-learn | learn vocabulary and IDF weights | – | vectorizer |
| `tfidf.transform([a]).toarray()` | scikit-learn | text -> sparse TF-IDF vector -> dense array | – | ndarray |
| `np.linalg.norm(v)` | numpy | vector length | – | float |
| `matrix.nbytes` | numpy | memory used by the array | – | int |
| `build_chunks()`, `chunk_for_embedding(c)` | `common.py` | 36 heading-aware chunks; prepend `title - section: ` | – | list / str |

## Execution trace (real run)
- `model: sentence-transformers/all-MiniLM-L6-v2 | 22.7M params | dim 384`.
- `hidden states (2, 15, 384) -> pooled (2, 384) -> norms [1. 1.]`: 2 texts padded to 15 tokens, one 384-vector per token, averaged to one per text, normalised to length 1.
- Meaning vs words:

| pair | MiniLM | TF-IDF |
|---|---|---|
| paraphrase, no shared words | 0.46 | 0.00 |
| paraphrase, shared words | 0.60 | 0.06 |
| same words, different meaning | 0.56 | 0.50 |
| needs world knowledge (caffeine free vs "keep me awake") | 0.47 | 0.00 |
| unrelated | 0.07 | 0.00 |

  Dense wins on paraphrase and world knowledge; but "Refunds are issued within 9 days" vs "Shipping is issued within 9 days" still scores 0.56. Embeddings blur exact details.
- Truncation: `cosine(text WITH the answer at the end, text WITHOUT it) = 1.0000`. The sentence after ~600 tokens of filler was never read.
- e5 prefixes: without `refund 0.793 | shipping 0.766 | gap 0.027`; with `refund 0.770 | shipping 0.738 | gap 0.032`. The absolute scores fell, but the **gap** between the right and the wrong passage grew a little. Ranking depends on gaps, not absolute values; on two passages this is a small, single-example effect, not a benchmark.
- Knowledge base: `36 chunks -> matrix (36, 384), 55,296 bytes (float32)`; `1M chunks would need 1.54 GB`.
- "Which teas will not keep me awake at night?" -> `0.570 allergens_and_ingredients#Caffeine` first. A dense win.
- "Can you send tea to my cousin in London?" -> top 3 are `product_catalog#Green and herbal teas` 0.375, `product_catalog#Classic blends` 0.374, `refunds_and_returns#How to request a refund` 0.374. The `shipping_and_delivery#International shipping` chunk is missing (it says "outside India", not "London"). Note how flat the scores are: the model is not confident about anything.

## Experiments (change ONE thing)
1. CLS pooling (`pooled = hidden[:, 0]`) on the five pairs gives 0.77, 0.82, 0.91, 0.69, 0.60 instead of 0.46, 0.60, 0.56, 0.47, 0.07. Even "unrelated" scores 0.60: everything looks similar, because MiniLM was trained with mean pooling. Wrong pooling does not crash; it quietly ruins the scale.
2. Remove the `normalize` line: vector lengths differ (4.65, 6.41, 7.28 for three of the test sentences), so the dot product mixes "how long is the vector" with "which direction does it point". In this run the *shortest* sentence had the *largest* norm, so do not assume length grows with text length; just normalise and compare directions.
3. A Hindi question ("How many days do I get for a refund?") against the English refund rule: e5 (multilingual) scores refund 0.832 vs shipping 0.746; English-only MiniLM scores the refund pair -0.031. Pick a multilingual model when users are.

## Debugging practice
BREAK IT: compare a MiniLM query vector with e5 passage vectors. Both are 384-dimensional, so there is **no error**:
```
shapes (384,) (2, 384)
MiniLM query vs e5 passages : refund 0.088 | shipping 0.080
MiniLM query vs MiniLM pass.: refund 0.155 | shipping 0.085
```
The cross-model numbers are meaningless: the axes of the two spaces have nothing to do with each other. Matching shapes prove nothing. Defence: store the embedding model name and version with the index and assert it at query time; re-embed everything when the model changes.

## Rebuild checklist
- [ ] tokenize with `padding=True, truncation=True, max_length=256`
- [ ] `model(**batch).last_hidden_state` under `torch.no_grad()`
- [ ] masked mean pooling, then L2 normalise
- [ ] cosine = dot product of unit vectors; score all chunks with `matrix @ q`
- [ ] same model (and prefix format) for chunks and queries
- [ ] chunk before embedding; know the memory bill (n x dim x 4 bytes)

## The new problem (cliffhanger)
We scored 36 chunks with one matrix multiplication. A real index has millions of vectors, needs to add and delete documents, and must return the top-k fast and in the right order. How do we store vectors and search them properly? Next: [step 04, vector index and search](../04_vector_index_and_search/WALKTHROUGH.md).

> **Say it aloud (interview box):** A bi-encoder embeds the query and each passage separately into normalised vectors, usually by mean-pooling token states, so passage vectors are precomputed and search is a dot product. It is trained contrastively with in-batch and hard negatives using an InfoNCE loss, which makes it good at paraphrases but blurry on exact details, so I pair it with BM25 and a cross-encoder re-ranker. I pick the model from MTEB retrieval scores plus my own evaluation, use its expected prefixes and pooling, chunk before embedding because of truncation, and re-embed the whole corpus whenever the model changes.
