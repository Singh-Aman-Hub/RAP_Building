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

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Embedding vs vector vs representation | "dense vector", "sentence embedding", "latent" | Same idea | A list of numbers (384 here) that stands for the meaning of a text |
| Bi-encoder vs sentence transformer vs dual encoder | two-tower | Mostly | Encodes query and document separately with the same (or paired) network, so document vectors can be precomputed |
| Mean pooling vs CLS pooling vs max pooling | token to sentence vector | No | Mean = average of real token vectors (used here); CLS = take the first token; max = element-wise maximum |
| Cosine similarity vs dot product | similarity score | Same on unit vectors | After L2 normalisation, dot product equals cosine; without it, vector length contaminates the dot product |
| Dense vs sparse (TF-IDF, BM25) | semantic vs lexical | No | Dense: every dimension has a value, similar meaning is near; sparse: mostly zeros, counts shared words |
| `query:` / `passage:` prefixes vs instructions | asymmetric models (e5, bge) | Related | Prefixes tell the model whether the text is a question or a document; some models use longer instruction strings |

## Code, step by step

### 1. Docstring  (`main.py` lines 1–13)
```python
"""
GOAL:  Turn text into meaning vectors with a bi-encoder (all-MiniLM-L6-v2): tokens -> hidden states ->
       mean pooling -> L2 normalisation.  Compare with TF-IDF on paraphrases, see truncation, and see why
       some models need "query:" / "passage:" prefixes.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 14_rag_and_retrieval/03_embeddings_bi_encoder/main.py
EXPECTED OUTPUT (shortened):
    hidden states (2, 15, 384) -> pooled (2, 384) -> norms [1. 1.]
    pair                                         MiniLM  TF-IDF
    paraphrase, no shared words                     0.46    0.0x
NEEDS: torch, transformers, scikit-learn; cached all-MiniLM-L6-v2 and multilingual-e5-small.  ~10 s.
"""
```
- **In simple words:** What the file shows: how a sentence becomes a vector and where that goes wrong.
- **Exactly:** Pipeline: tokens to hidden states to mean pooling to L2 normalisation. Compares with TF-IDF, shows truncation and prefixes. Needs torch, transformers, scikit-learn.
- **Why it is here:** Maps the five sections of the file.

### 2. Imports and path setup  (`main.py` lines 15–24)
```python
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EMBED_MODEL, build_chunks, chunk_for_embedding   # noqa: E402  (also sets HF offline)
```
- **In simple words:** Load numeric and model libraries and our shared constants.
- **Exactly:** `torch` runs the network; `TfidfVectorizer` is the sparse baseline; `AutoTokenizer` and `AutoModel` load MiniLM. `from common import EMBED_MODEL, build_chunks, chunk_for_embedding` also sets the Hugging Face offline flags.
- **Why it is here:** Importing `common` before loading models keeps the run offline.

### 3. Load tokenizer and model  (`main.py` lines 26–28)
```python
tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
model = AutoModel.from_pretrained(EMBED_MODEL).eval()
print(f"model: {EMBED_MODEL} | {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params | dim {model.config.hidden_size}")
```
- **In simple words:** Download (from cache) the dictionary and the neural network.
- **Exactly:** `AutoTokenizer.from_pretrained(EMBED_MODEL)` loads the WordPiece vocabulary; `AutoModel...eval()` loads weights and switches dropout off. Real output: `22.7M params | dim 384`.
- **Why it is here:** `.eval()` makes outputs deterministic; without it dropout adds noise.

### 4. The embed function, tensor by tensor  (`main.py` lines 31–41)
```python
# ---- 1. The pipeline, one tensor at a time ------------------------------------------------------------
def embed(texts, max_length=256, verbose=False):
    batch = tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state                  # (n_texts, n_tokens, 384): one vector PER TOKEN
    mask = batch["attention_mask"].unsqueeze(-1).float()           # (n_texts, n_tokens, 1): 0 on padding
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)          # mean over REAL tokens -> (n_texts, 384)
    unit = torch.nn.functional.normalize(pooled, dim=1)            # length 1 -> dot product == cosine
    if verbose:
        print(f"hidden states {tuple(hidden.shape)} -> pooled {tuple(pooled.shape)} -> norms {unit.norm(dim=1).numpy().round(3)}")
    return unit.numpy()
```
- **In simple words:** Turn a list of sentences into one unit-length vector each.
- **Exactly:** `tok(...)` returns `input_ids` and `attention_mask` of shape `(n_texts, n_tokens)`; padding fills the shorter ones. `hidden` has shape `(n_texts, n_tokens, 384)`: one vector per token. `mask` shape `(n_texts, n_tokens, 1)` is 0 on padding. `(hidden*mask).sum(1)/mask.sum(1)` is the mean over real tokens: `(n_texts, 384)`. `normalize(dim=1)` gives length 1. Real: `hidden states (2, 15, 384) -> pooled (2, 384) -> norms [1. 1.]`.
- **Why it is here:** Mask and normalise are the two lines people forget; without them similarity is wrong.

### 5. Run it on two sentences  (`main.py` lines 44–44)
```python
embed(["Refunds are issued within 9 days.", "Shipping is free above 799 rupees for every order."], verbose=True)
```
- **In simple words:** Embed two example sentences with printing switched on.
- **Exactly:** Prints the shapes shown above. The two sentences have 15 tokens after padding to the longest.
- **Why it is here:** Seeing the shapes once makes the rest of the file easy to read.

### 6. Five test pairs  (`main.py` lines 47–54)
```python
refund = "Refunds are issued within 9 days of delivery if the seal is unbroken."
pairs = [
    ("How long can I wait before asking for my money back?", refund, "paraphrase, no shared words"),
    ("How many days do I have for a refund?", refund, "paraphrase, shared words"),
    ("Refunds are issued within 9 days.", "Shipping is issued within 9 days.", "same words, different meaning"),
    ("Which teas will not keep me awake?", "Chamomile Calm is caffeine free.", "needs world knowledge"),
    ("How do I brew green tea?", "Our office is closed on Sundays.", "unrelated"),
]
```
- **In simple words:** Pairs of sentences with a label for what each pair is testing.
- **Exactly:** Paraphrase without shared words; paraphrase with shared words; same words but different meaning (refund vs shipping); needs world knowledge (awake vs caffeine free); unrelated.
- **Why it is here:** Each pair probes a different strength or weakness of embeddings.

### 7. Compare MiniLM with TF-IDF  (`main.py` lines 55–63)
```python
tfidf = TfidfVectorizer().fit([t for p in pairs for t in p[:2]])
print(f"\n{'pair':<48}{'MiniLM':>8}{'TF-IDF':>8}")
for a, b, label in pairs:
    dense = float(embed([a])[0] @ embed([b])[0])
    va, vb = tfidf.transform([a]).toarray()[0], tfidf.transform([b]).toarray()[0]
    sparse = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
    print(f"{label:<48}{dense:>8.2f}{sparse:>8.2f}")
print("Dense vectors score paraphrases and world knowledge; TF-IDF needs shared words (rows 1 and 4 score 0).")
print("Row 3: dense similarity stays high for 'refunds' vs 'shipping' -> embeddings blur exact details.")
```
- **In simple words:** Score every pair with dense vectors and with word counts.
- **Exactly:** `TfidfVectorizer().fit(...)` learns a vocabulary from the ten sentences; cosine of TF-IDF vectors is `va @ vb / (|va||vb| + 1e-9)`. Real: row 1 MiniLM 0.46 vs TF-IDF 0.00; row 2 0.60 vs 0.06; row 3 0.56 vs 0.50; row 4 0.47 vs 0.00; row 5 0.07 vs 0.00. Row 3 shows dense vectors blur exact details.
- **Why it is here:** Shows both why dense retrieval exists (rows 1 and 4) and why it needs help (row 3).

### 8. Truncation  (`main.py` lines 65–71)
```python
# ---- 3. Truncation: text beyond max_length is silently ignored --------------------------------------
filler = "Our teas are packed with care in Pune. " * 60                  # ~600 tokens of filler
tail = " The refund window is 9 days."
v_full = embed([filler + tail], max_length=256)[0]
v_without = embed([filler], max_length=256)[0]
print(f"\ntruncation: cosine(text WITH the answer at the end, text WITHOUT it) = {float(v_full @ v_without):.4f}")
print("1.0000 means the model never saw the last sentence. Chunk BEFORE embedding; keep chunks short.")
```
- **In simple words:** Put the answer at the end of a very long text and see whether the model notices.
- **Exactly:** `filler` is about 600 tokens; `max_length=256` cuts everything after token 256. The cosine between "with answer" and "without answer" is `1.0000`: the model never saw the tail.
- **Why it is here:** Long chunks silently lose their end. Chunk before embedding.

### 9. Load a second model (e5)  (`main.py` lines 73–75)
```python
# ---- 4. Asymmetric models want prefixes (e5, bge, nomic ...) -----------------------------------------
e5_name = "intfloat/multilingual-e5-small"
e5_tok, e5 = AutoTokenizer.from_pretrained(e5_name), AutoModel.from_pretrained(e5_name).eval()
```
- **In simple words:** Load an embedding model that expects prefixes.
- **Exactly:** `intfloat/multilingual-e5-small`, also 384-dim, loaded the same way.
- **Why it is here:** To show that different models have different input formats.

### 10. e5_embed: the same pipeline in compact form  (`main.py` lines 78–83)
```python
def e5_embed(texts):
    b = e5_tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        h = e5(**b).last_hidden_state
    m = b["attention_mask"].unsqueeze(-1).float()
    return torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=1).numpy()
```
- **In simple words:** The same embedding steps in one function for e5.
- **Exactly:** Tokenise, forward, mean-pool with mask, normalise. Same recipe as `embed`.
- **Why it is here:** The model changes, the recipe does not.

### 11. Prefixes: with and without  (`main.py` lines 86–94)
```python
q = "What is the money back window?"
passages = [refund, "Shipping is free on orders above 799 rupees."]
plain = e5_embed([q] + passages)
prefixed = e5_embed(["query: " + q] + ["passage: " + p for p in passages])
print(f"\ne5 without prefixes: refund {plain[0] @ plain[1]:.3f} | shipping {plain[0] @ plain[2]:.3f} "
      f"| gap {plain[0] @ plain[1] - plain[0] @ plain[2]:.3f}")
print(f"e5 with prefixes   : refund {prefixed[0] @ prefixed[1]:.3f} | shipping {prefixed[0] @ prefixed[2]:.3f} "
      f"| gap {prefixed[0] @ prefixed[1] - prefixed[0] @ prefixed[2]:.3f}")
print("Use the format the model was trained with; mixing formats between indexing and querying hurts.")
```
- **In simple words:** Embed the same texts with and without `query:` and `passage:`.
- **Exactly:** Real: without prefixes refund 0.793, shipping 0.766, gap 0.027; with prefixes refund 0.770, shipping 0.738, gap 0.032. The gap between right and wrong passage grows slightly with prefixes on this tiny example. Use the format used in training; follow the model card.
- **Why it is here:** Wrong prefixes quietly reduce quality; no error is raised.

### 12. Embed the whole knowledge base  (`main.py` lines 96–100)
```python
# ---- 5. Embed the whole knowledge base -----------------------------------------------------------------
chunks = build_chunks()
matrix = embed([chunk_for_embedding(c) for c in chunks])               # (36, 384)
print(f"\nknowledge base: {len(chunks)} chunks -> matrix {matrix.shape}, {matrix.nbytes:,} bytes (float32)")
print(f"1M chunks would need {1_000_000 * matrix.shape[1] * 4 / 1e9:.2f} GB of float32 vectors")
```
- **In simple words:** Turn all 36 chunks into vectors and see how much memory that takes.
- **Exactly:** `chunk_for_embedding` adds title and section. Matrix shape `(36, 384)`, `55,296` bytes (`36 * 384 * 4`). 1M chunks would need `1_000_000 * 384 * 4 / 1e9 = 1.54 GB`.
- **Why it is here:** Memory planning starts with this multiplication.

### 13. Two real queries  (`main.py` lines 101–107)
```python
for question in ["Which teas will not keep me awake at night?", "Can you send tea to my cousin in London?"]:
    qv = embed([question])[0]                                            # (384,)
    print(f"Q: {question}")
    for i in np.argsort(-(matrix @ qv))[:3]:
        print(f"  {float(matrix[i] @ qv):.3f}  {chunks[i]['chunk_id']}")
print("The second question misses 'International shipping' (the chunk says 'outside India', the query says 'London').")
print("Fixes come later: hybrid search (06), re-ranking (07), query rewriting (10).")
```
- **In simple words:** Ask two questions and print the three most similar chunks.
- **Exactly:** `matrix @ qv` is `(36, 384) @ (384,)` = 36 cosine scores; `argsort(-scores)[:3]`. Q1 top: `allergens_and_ingredients#Caffeine` 0.570. Q2 "cousin in London" does not find International shipping (top score 0.375, wrong chunks): the chunk says "outside India", the query says "London".
- **Why it is here:** Shows a real failure that later steps fix (hybrid search, reranking, query rewriting).

### 14. Exercises as comments  (`main.py` lines 109–114)
```python
# TRY IT: in embed(), replace mean pooling by the first token only: pooled = hidden[:, 0]. Do the scores change?
# TRY IT: remove the normalize line and print the vector norms: they differ (4.65 to 7.28 in one run), so raw dot
#         products mix 'how similar' with 'how long the vector is'. Normalising removes that.
# TRY IT: add your own pair to `pairs`, e.g. a question in Hindi vs the English refund rule (use e5_embed: it is multilingual).
# BREAK IT: compare a MiniLM vector with an e5 vector (both 384-dim!). No error, but the score is meaningless:
#           different models live in different vector spaces, so the index and the query MUST use the same model.
```
- **In simple words:** Ideas to try and break.
- **Exactly:** BREAK IT: comparing MiniLM and e5 vectors (both 384-dim) gives no error but a meaningless score; different models live in different vector spaces.
- **Why it is here:** Index and query must use the same model and the same version.


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
