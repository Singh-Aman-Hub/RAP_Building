# `common.py` walkthrough: the shared toolbox of module 14

## Goal
Steps 01 to 09 each teach one idea, then import the finished helper from `common.py` so later steps stay short. This page walks `common.py` once, top to bottom, so no helper is a black box. Each helper is taught properly in its own step (named in the heading); here you get the exact meaning of every line.

## Run it
`common.py` is a library, not a script. Check it from this folder:
```bash
cd 14_rag_and_retrieval
HF_HUB_OFFLINE=1 ../.venv/bin/python -c "import common as c; print(len(c.load_documents()), len(c.build_chunks()))"
```
Real values from a run: 9 documents, 36 chunks (first chunk id `allergens_and_ingredients#Nuts`), `Embedder().dim == 384`, `encode(["hello","refund policy"])` has shape `(2, 384)`, dtype `float32`, row norms `[1. 1.]`; `tokenize("How do I order ck-masala-250 and what is the refund policy?")` returns `['order', 'ck-masala-250', 'refund', 'policy']`; BM25 over the 36 chunks has average length 19.31 and 408 distinct terms; `rrf_fuse` of rankings `[1,2,3]` and `[2,1,4]` gives `[(1, 0.0325), (2, 0.0325), (3, 0.0159)]`; the cross-encoder scores "we refund in 14 days" `1.57` against "tea is hot" `-10.92` for the query "refund?"; the eval file has 29 questions (24 with a gold chunk).

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Embedder vs bi-encoder vs sentence-transformers | text embedder | Same idea | `Embedder` re-implements the bi-encoder recipe by hand (tokenise, run the model, mean-pool, normalise); `sentence-transformers` does the same in one call |
| `dense_search` vs vector DB search | nearest-neighbour search | No | `dense_search` is exact brute force over a NumPy matrix; a vector DB adds ANN indexes, filters and storage |
| BM25 vs TF-IDF | lexical score | Related | BM25 adds term-frequency saturation (`k1`) and length normalisation (`b`) to TF-IDF |
| `rrf_fuse` vs score fusion | merge rankings | No | RRF uses only ranks (`1/(k+rank)`), so it needs no score calibration; score fusion adds weighted raw scores |
| Bi-encoder vs `CrossEncoderReranker` | retriever vs reranker | No | Bi-encoder: two separate vectors; cross-encoder: reads query and passage together and outputs one score |
| Chunk vs document | unit of retrieval | No | A document is a whole file; a chunk is the piece we embed and retrieve |

## Code, step by step

### 1. Module docstring: who uses what  (`common.py` lines 1–15)
```python
"""
common.py -- small helpers shared by the steps of module 14 (RAG and retrieval).

Every helper here is TAUGHT first, line by line, inside one step folder. Later steps import the
finished version from here so they can focus on the next idea:

    load_documents, split_by_heading   -> taught in 01 and 02
    Embedder                           -> taught in 03
    dense_search                       -> taught in 04
    BM25, rrf_fuse                     -> taught in 06
    CrossEncoderReranker               -> taught in 07
    load_eval_set                      -> used in 06, 07, 09

Everything runs offline on a CPU. Models come from the local Hugging Face cache.
"""
```
- **In simple words:** A map saying which step teaches which helper.
- **Exactly:** Lists `load_documents, split_by_heading` (steps 01 and 02), `Embedder` (03), `dense_search` (04), `BM25, rrf_fuse` (06), `CrossEncoderReranker` (07), `load_eval_set` (06, 07, 09). Everything runs offline on CPU.
- **Why it is here:** Tells you where to read the lesson before reading the code.

### 2. Imports and offline settings  (`common.py` lines 17–27)
```python
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")          # never touch the network; use cached models only
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
```
- **In simple words:** Load tools and tell Hugging Face not to use the internet.
- **Exactly:** Standard library: `json`, `math`, `os`, `re`, `Counter`, `Path`; third-party `numpy`. `os.environ.setdefault("HF_HUB_OFFLINE", "1")` makes `from_pretrained` use only the local cache (it does not override a value you set). `TOKENIZERS_PARALLELISM=false` silences the fork warning of Rust tokenizers.
- **Why it is here:** Without the offline flag a missing network makes model loading hang on retries.

### 3. Paths and model names  (`common.py` lines 29–34)
```python
MODULE_DIR = Path(__file__).resolve().parent
KB_DIR = MODULE_DIR / "data" / "kb"                   # the knowledge base: 9 markdown files
EVAL_FILE = MODULE_DIR / "data" / "eval_questions.json"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384-dim bi-encoder, 22M parameters
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"    # cross-encoder trained on MS MARCO
```
- **In simple words:** Where the data is and which two models we use.
- **Exactly:** `MODULE_DIR` is the folder of this file; `KB_DIR` is `data/kb` (9 Markdown files); `EVAL_FILE` is `data/eval_questions.json`. `EMBED_MODEL` is the 384-dimensional MiniLM bi-encoder (22M parameters); `RERANK_MODEL` is the MS MARCO cross-encoder.
- **Why it is here:** Paths built from `__file__` work from any working directory.

### 4. load_documents (step 01)  (`common.py` lines 40–47)
```python
def load_documents(kb_dir=KB_DIR):
    """Read every .md file -> list of {"doc_id", "title", "text"} sorted by doc_id."""
    docs = []
    for path in sorted(Path(kb_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip()   # first line is "# Title"
        docs.append({"doc_id": path.stem, "title": title, "text": text})
    return docs
```
- **In simple words:** Read every knowledge-base file into a list of records.
- **Exactly:** `sorted(Path(kb_dir).glob("*.md"))` gives a stable order. Title = first line without `# `. Returns `{"doc_id": file stem, "title", "text"}`. Real result: 9 documents, the first is `allergens_and_ingredients` with title `Allergens and Ingredients`.
- **Why it is here:** A stable id and order make citations and tests repeatable.

### 5. split_by_heading (step 02)  (`common.py` lines 50–61)
```python
def split_by_heading(doc):
    """One chunk per '## Section'. chunk_id = 'doc_id#Section'. The doc title is kept as metadata."""
    chunks = []
    for block in re.split(r"\n(?=## )", doc["text"]):         # split right before every '## '
        lines = block.strip().splitlines()
        if not lines or not lines[0].startswith("## "):
            continue                                           # skip the '# Title' block
        section = lines[0][3:].strip()
        body = " ".join(line.strip() for line in lines[1:] if line.strip())
        chunks.append({"chunk_id": f"{doc['doc_id']}#{section}", "doc_id": doc["doc_id"],
                       "title": doc["title"], "section": section, "text": body})
    return chunks
```
- **In simple words:** Cut each document into one chunk per `##` section.
- **Exactly:** `re.split(r"\n(?=## )", text)` splits right before each `## ` (a look-ahead, the heading stays with its block). The `# Title` block is skipped. The body joins the non-empty lines with spaces. `chunk_id = "doc_id#Section"`. Result: 4 chunks per document, 36 in total.
- **Why it is here:** Keeps each rule with its heading, and the id is human readable for citations.

### 6. build_chunks  (`common.py` lines 64–66)
```python
def build_chunks():
    """All section chunks of the whole knowledge base."""
    return [c for d in load_documents() for c in split_by_heading(d)]
```
- **In simple words:** Chunks for the entire knowledge base in one call.
- **Exactly:** A nested list comprehension over documents then their chunks: 36 dicts.
- **Why it is here:** Convenience used by steps 03 to 10.

### 7. chunk_for_embedding  (`common.py` lines 69–71)
```python
def chunk_for_embedding(chunk):
    """Prepend the document title and section heading ("contextual header") before embedding."""
    return f"{chunk['title']} - {chunk['section']}: {chunk['text']}"
```
- **In simple words:** The text we actually embed: heading words first, then the body.
- **Exactly:** Returns `"{title} - {section}: {text}"`, e.g. `Allergens and Ingredients - Nuts: Every blend except Winter Kahwa is nut free...`. This is a simple "contextual header".
- **Why it is here:** A body such as "9 days" has no topic; the header gives the vector its context.

### 8. Embedder.__init__ (step 03)  (`common.py` lines 77–87)
```python
class Embedder:
    """Bi-encoder: text -> one L2-normalised vector. Mean pooling over tokens, like sentence-transformers."""

    def __init__(self, model_name=EMBED_MODEL, max_length=256):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()
        self.max_length = max_length
        self.dim = self.model.config.hidden_size
```
- **In simple words:** Load the model and the tokenizer once.
- **Exactly:** `AutoTokenizer.from_pretrained` loads the WordPiece vocabulary; `AutoModel.from_pretrained(...).eval()` loads the transformer and switches off dropout. `max_length=256` is the truncation limit. `self.dim = model.config.hidden_size` is 384. `torch` and `transformers` are imported inside the constructor so that importing `common` stays fast for steps that do not need them.
- **Why it is here:** Loading is slow (seconds); doing it per query would be wasteful.

### 9. Embedder.encode: batches, pooling, normalising  (`common.py` lines 89–100)
```python
    def encode(self, texts, batch_size=32):
        out = []
        for i in range(0, len(texts), batch_size):
            batch = self.tok(texts[i:i + batch_size], padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt")
            with self.torch.no_grad():
                hidden = self.model(**batch).last_hidden_state          # (b, tokens, dim)
            mask = batch["attention_mask"].unsqueeze(-1).float()       # (b, tokens, 1)
            pooled = (hidden * mask).sum(1) / mask.sum(1)              # mean over real tokens -> (b, dim)
            pooled = self.torch.nn.functional.normalize(pooled, dim=1)  # unit length -> dot = cosine
            out.append(pooled.numpy())
        return np.vstack(out).astype(np.float32)                       # (n_texts, dim)
```
- **In simple words:** Turn a list of texts into a matrix of unit-length vectors.
- **Exactly:** For each batch of 32 texts: tokenise with padding and truncation (`return_tensors="pt"`), run the model without gradients, take `last_hidden_state` of shape `(b, tokens, 384)`. `attention_mask.unsqueeze(-1)` has shape `(b, tokens, 1)`; `(hidden * mask).sum(1) / mask.sum(1)` is the mean over REAL tokens only, shape `(b, 384)`. `normalize(dim=1)` makes each vector length 1, so a dot product equals cosine similarity. Output `np.vstack(...)` has shape `(n_texts, 384)`, `float32` (real: norms `[1. 1.]`).
- **Why it is here:** Without the mask, padding tokens drag the average; without normalising, dot product is not cosine.

### 10. dense_search (step 04)  (`common.py` lines 103–109)
```python
def dense_search(query_vec, matrix, k=5):
    """Exact (brute-force) top-k by dot product. Rows of `matrix` and `query_vec` are unit vectors."""
    scores = matrix @ query_vec                                        # (n,)
    k = min(k, len(scores))
    top = np.argpartition(-scores, k - 1)[:k]                          # k best, unordered, O(n)
    top = top[np.argsort(-scores[top])]                                # order only those k
    return [(int(i), float(scores[i])) for i in top]
```
- **In simple words:** Find the k most similar vectors by checking all of them.
- **Exactly:** `matrix @ query_vec` is `(n, 384) @ (384,)` = scores of shape `(n,)`. `argpartition(-scores, k-1)[:k]` finds the k best in O(n) without a full sort; `argsort` orders only those k. Returns a list of `(row_index, score)` tuples. Real: querying a matrix with itself returns `(0, 1.0)` first (a vector is perfectly similar to itself).
- **Why it is here:** Exact brute force is the correct baseline for ANN comparisons (step 05).

### 11. STOPWORDS and tokenize (step 06)  (`common.py` lines 115–121)
```python
STOPWORDS = set("""a an and are as at be by can do does for from has have how i if in is it its
my of on or our the this to was we what when where which who will with you your me am""".split())


def tokenize(text):
    """Lower-case word tokens; keeps codes like 'ck-masala-250' as one token; drops stopwords."""
    return [t for t in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower()) if t not in STOPWORDS]
```
- **In simple words:** Break text into lowercase words and drop filler words.
- **Exactly:** The regex `[a-z0-9]+(?:-[a-z0-9]+)*` keeps hyphenated codes such as `ck-masala-250` as ONE token. Real: `tokenize("How do I order ck-masala-250 and what is the refund policy?")` returns `['order', 'ck-masala-250', 'refund', 'policy']`.
- **Why it is here:** Product codes and SKUs are exactly what keyword search is good at; splitting them would lose that.

### 12. BM25.__init__: statistics of the corpus  (`common.py` lines 124–134)
```python
class BM25:
    """Okapi BM25 over a list of texts. k1 = term-frequency saturation, b = length normalisation."""

    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avg_len = self.lengths.mean()
        n = len(texts)
        df = Counter(term for d in self.docs for term in d)            # in how many docs each term appears
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
```
- **In simple words:** Count the words of every chunk and how rare each word is.
- **Exactly:** `docs` is a `Counter` of tokens per text. `lengths` is the token count per text, `avg_len` its mean (19.31 on 36 chunks). `df` counts in how many texts each term appears. `idf = log(1 + (N - df + 0.5) / (df + 0.5))`: rare terms get a high weight (408 distinct terms here). `k1=1.5` controls term-frequency saturation, `b=0.75` length normalisation.
- **Why it is here:** Rare words carry the signal; "the" does not.

### 13. BM25.scores: score every text for a query  (`common.py` lines 136–146)
```python
    def scores(self, query):
        s = np.zeros(len(self.docs))
        for term in tokenize(query):
            if term not in self.idf:
                continue
            for i, d in enumerate(self.docs):
                tf = d.get(term, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_len)
                    s[i] += self.idf[term] * tf * (self.k1 + 1) / norm
        return s
```
- **In simple words:** Add up, for each query word, how well it matches each chunk.
- **Exactly:** For each query term in the vocabulary and each text with `tf > 0`: `idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avg_len))`. Terms not in the corpus are skipped. Returns an array of shape `(n_texts,)`.
- **Why it is here:** Saturation stops a word repeated 10 times scoring 10 times higher; length normalisation stops long chunks winning by size.

### 14. BM25.search  (`common.py` lines 148–151)
```python
    def search(self, query, k=5):
        s = self.scores(query)
        top = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in top]
```
- **In simple words:** The k best-scoring chunks.
- **Exactly:** `argsort(-s)[:k]`; returns `(index, score)` tuples. Real: `search("refund", 2)` gives `[(12, 2.714), (25, 2.241)]`.
- **Why it is here:** Same return shape as `dense_search`, so fusion code treats both alike.

### 15. rrf_fuse (step 06)  (`common.py` lines 154–160)
```python
def rrf_fuse(rankings, k=60, top_n=5):
    """Reciprocal Rank Fusion: score(doc) = sum over rankings of 1 / (k + rank). Ranks start at 1."""
    fused = Counter()
    for ranking in rankings:                                           # each ranking: list of (index, score)
        for rank, (idx, _) in enumerate(ranking, start=1):
            fused[idx] += 1.0 / (k + rank)
    return [(idx, score) for idx, score in fused.most_common(top_n)]
```
- **In simple words:** Merge several ranked lists into one.
- **Exactly:** Each document gets `1 / (k + rank)` from every list it appears in (ranks start at 1, `k=60`), summed in a `Counter`. `most_common(top_n)` returns the best. Real: lists `[1,2,3]` and `[2,1,4]` give docs 1 and 2 `0.0325` each (`1/61 + 1/62`) and doc 3 `0.0159`.
- **Why it is here:** Uses only ranks, so no need to make cosine and BM25 scores comparable.

### 16. CrossEncoderReranker.__init__ (step 07)  (`common.py` lines 166–174)
```python
class CrossEncoderReranker:
    """Reads (query, passage) TOGETHER and outputs one relevance logit. Slower but more accurate."""

    def __init__(self, model_name=RERANK_MODEL):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
```
- **In simple words:** Load the reranking model.
- **Exactly:** `AutoModelForSequenceClassification` with the cross-encoder checkpoint (one output logit); `.eval()` disables dropout.
- **Why it is here:** A different model type from the Embedder: it outputs a score, not a vector.

### 17. CrossEncoderReranker.score  (`common.py` lines 176–180)
```python
    def score(self, query, passages):
        batch = self.tok([query] * len(passages), passages, padding=True, truncation=True,
                         max_length=512, return_tensors="pt")
        with self.torch.no_grad():
            return self.model(**batch).logits.squeeze(-1).numpy()      # (n_passages,)
```
- **In simple words:** Score how well each passage answers the query.
- **Exactly:** The tokenizer receives `[query] * n` and `passages` as pairs, producing `[CLS] query [SEP] passage [SEP]` with padding up to 512 tokens. `logits.squeeze(-1)` has shape `(n_passages,)`. Real: for "refund?", "we refund in 14 days" scores `1.57` and "tea is hot" `-10.92`. Higher = more relevant; values are logits, not probabilities.
- **Why it is here:** Reading query and passage together is what makes it more accurate than a bi-encoder, at the cost of one forward pass per pair.

### 18. load_eval_set  (`common.py` lines 186–187)
```python
def load_eval_set():
    return json.loads(EVAL_FILE.read_text(encoding="utf-8"))
```
- **In simple words:** Read the list of test questions.
- **Exactly:** `json.loads(...)` returns a list of dicts with `id`, `question`, `gold` (list of chunk ids), `answer`, `kind`. Real: 29 questions, 24 with a non-empty `gold`.
- **Why it is here:** A fixed question set is what makes retrieval quality measurable.


