# Step 04 — Vector index and exact search

## The story so far
[Step 03](../03_embeddings_bi_encoder/WALKTHROUGH.md) turned every chunk into a 384-dimensional unit vector with a bi-encoder (all-MiniLM-L6-v2). Two texts that mean the same thing now sit close together, so "find relevant chunks" becomes "find the nearest vectors".

## The pain point
A pile of vectors is not a search system. We need somewhere to keep each vector *together with* its chunk id, document id and text, a way to get the top-k quickly, a way to say "only search the loyalty document", a way to update a chunk when the policy changes without creating a duplicate, and a way to save it all to disk so we do not re-embed on every restart. That bundle is what people call a vector database.

## The spark
Strip a vector database down to its core and it is a matrix plus a list: `vectors` of shape (n, dim) and `meta`, one dict per row. Search is one matrix-vector product, `vectors @ query`, followed by "pick the k largest". Everything else (filters, upserts, deletes, persistence) is bookkeeping on that list. Build this "flat" index by hand, then measure where brute force stops being good enough.

## Goal
Build a `FlatIndex` with upsert, delete, filtered top-k search, save and load. Then time brute force on 100,000 random unit vectors and prove that cosine, dot product and L2 distance give the same ranking on unit vectors.

## The theory (read this before the interview)
**What a vector database is.** A vector database stores (1) vectors, (2) metadata (payload) per vector, (3) an index that makes nearest-neighbour search fast, usually approximate (ANN, step 05), and supports (4) CRUD by id: insert, upsert, delete, (5) metadata filtering ("doc_id = X", "date > Y", "tenant = Z"), and (6) persistence, replication and snapshots. Our `FlatIndex` does 1, 2, 4, 5 and 6 with an *exact* index; FAISS `IndexFlatIP` is the same idea in C++.

**Exact kNN cost.** Brute force scores every stored vector: n dot products of length d, so O(n·d) per query. For n = 100,000 and d = 384 that is 38.4 million multiply-adds. It is perfectly accurate (recall = 1.0 by definition), embarrassingly parallel, and BLAS makes it fast, but it grows linearly with n.

**Similarity metrics.**
- Cosine similarity: `a·b / (||a|| ||b||)`, the angle only, ignores length.
- Dot (inner) product: `a·b`, angle *and* length.
- L2 (Euclidean) distance: `||a − b||`, smaller is closer.
- For unit vectors, `||a − b||² = ||a||² + ||b||² − 2a·b = 2 − 2cos(a, b)`. So L2 is a monotone function of cosine, and dot product *equals* cosine. All three give the same ranking. The script checks this: `||a-b||^2 = 2.0094` and `2 - 2cos = 2.0094`.
- When dot ≠ cosine: if vectors are **not normalised**, longer vectors win dot-product contests regardless of angle. Some models are trained with a dot-product objective and their vector length carries meaning (for example a "how much content / how confident" signal), so normalising would throw information away. Rule: use the metric the embedding model was trained with, and if you normalise, normalise both documents and queries.

**Top-k selection.** After scoring we need the k largest of n scores. A full `argsort` is O(n log n). `argpartition` uses a selection algorithm (introselect) that puts the k best in front, unordered, in O(n) on average; then you sort only those k, O(k log k). For k = 10 and n = 100,000 the saving is real: 8.2 ms vs 19.1 ms per query in our run (both timings include the matrix product).

**Filtering: pre vs post vs in-index.**
- *Post-filtering*: search top-k, then drop rows that fail the filter. Simple, but can return fewer than k results, even zero (our run: 0 results for `brewing_guide`). The usual patch is to over-fetch (ask for k × some factor), which still fails for very selective filters.
- *Pre-filtering*: select rows that pass the filter first, then rank only those. Always returns k if k rows match. With a flat index it is trivial; with an ANN graph it is hard, because removing most nodes can disconnect the graph.
- *Filtered ANN* (e.g. filter-aware HNSW traversal): the engine checks the filter while walking the index and skips failing nodes, and many engines fall back to brute force over the matching subset when the filter is very selective. This is a major practical differentiator between vector databases.

**Updates.** Documents change. An *upsert* replaces the row with the same id instead of appending a duplicate (our `keep` logic). In a flat index delete is just dropping rows. In ANN indexes deletes are usually *tombstones* (mark as deleted, skip at query time) followed by periodic compaction or re-indexing; IVF centroids may also need retraining when the data distribution drifts. Re-embedding everything is required whenever you change the embedding model, because vectors from different models are not comparable.

**Memory arithmetic.** float32 costs n × d × 4 bytes. 100,000 × 384 × 4 = 153.6 MB (printed as 154 MB); 10 million would be 15.4 GB. float16 halves that (77 MB here), int8 scalar quantisation quarters it, binary quantisation (1 bit per dimension) divides by 32, usually followed by re-scoring the shortlist with fuller vectors.

**When brute force is enough.** Up to roughly 100k–1M vectors on a CPU, exact search is often fast enough (ours: about 8 ms for 100k) and needs no tuning, no training and gives perfect recall. Beyond that, or with tight latency budgets and many queries per second, linear cost hurts: the script extrapolates to about 817 ms per query for 10 million vectors. That is the motivation for ANN.

## Run it
```bash
cd 14_rag_and_retrieval/04_vector_index_and_search
python main.py       # about 10 s on a laptop CPU (cached MiniLM); timings vary per run
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Flat index vs brute force vs exact search | `IndexFlatIP`, "kNN" | Same thing | Compare the query against every vector; 100 percent recall, cost grows with N |
| Exact kNN vs ANN | approximate nearest neighbour | No | ANN (step 05) skips most vectors to be faster and may miss a true neighbour |
| Pre-filter vs post-filter | metadata filtering | No | Pre: restrict rows first, then rank; post: rank top-k first, then drop non-matching (can return fewer than k) |
| Upsert vs insert vs update | write to the index | Related | Upsert = insert or replace by id, so no duplicates |
| Cosine vs dot vs L2 | metrics | Equivalent here | On unit vectors all three give the same ranking (`||a-b||^2 = 2 - 2cos`) |
| Vector store vs vector database vs index | FAISS, Qdrant | No | Index = structure; store/DB = index plus persistence, filtering, CRUD and a server |

## Code, step by step

### 1. Docstring  (`main.py` lines 1–14)
```python
"""
GOAL:  Build the thing a vector database is at its core: a "flat" index that stores vectors + metadata,
       answers top-k queries exactly, filters by metadata, upserts/deletes by id, and saves to disk.
       Then measure how brute force scales, and prove cosine / dot / L2 give the same ranking on unit vectors.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 14_rag_and_retrieval/04_vector_index_and_search/main.py
EXPECTED OUTPUT (shortened):
    index: 36 vectors x 384 dims
    filtered to doc_id=loyalty_program: ...
    brute force over 100,000 vectors: ~x ms per query (argpartition) vs ~y ms (full argsort)
    rankings identical for cosine, dot and L2 on unit vectors: True
NEEDS: torch, transformers (cached MiniLM), numpy.  ~10 s.
"""
```
- **In simple words:** What this file builds: the core of a vector database in about 40 lines.
- **Exactly:** A flat index with metadata, exact top-k, filtering, upsert, delete, save/load, a speed test, and a proof that three metrics agree.
- **Why it is here:** A black box (a vector DB) becomes ordinary code.

### 2. Imports  (`main.py` lines 16–25)
```python
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import Embedder, build_chunks, chunk_for_embedding   # noqa: E402
```
- **In simple words:** Load the standard modules, numpy and our embedder.
- **Exactly:** `json` and `Path` for saving; `tempfile` for a safe temporary folder; `time` for timing; `numpy` for matrices; `Embedder`, `build_chunks`, `chunk_for_embedding` from `common`.
- **Why it is here:** Only what is needed, no vector DB library.

### 3. FlatIndex: create an empty index  (`main.py` lines 28–34)
```python
class FlatIndex:
    """Exact nearest-neighbour index. vectors: (n, dim) float32 unit rows; meta: one dict per row."""

    def __init__(self, dim):
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.meta = []
```
- **In simple words:** A table of vectors plus a list of facts, one per row.
- **Exactly:** `vectors` is `(0, dim)` float32; `meta` is a list of dicts. Row `i` of `vectors` belongs to `meta[i]`.
- **Why it is here:** Keeping the two aligned is the whole job of an index.

### 4. upsert  (`main.py` lines 36–41)
```python
    def upsert(self, vectors, metas):
        """Insert, or replace rows whose chunk_id already exists (documents change over time)."""
        ids = {m["chunk_id"] for m in metas}
        keep = [i for i, m in enumerate(self.meta) if m["chunk_id"] not in ids]   # drop old versions
        self.vectors = np.vstack([self.vectors[keep], vectors.astype(np.float32)])
        self.meta = [self.meta[i] for i in keep] + list(metas)
```
- **In simple words:** Add new rows or replace rows with the same `chunk_id`.
- **Exactly:** `ids` is the set of incoming chunk ids; `keep` lists old rows not being replaced; `np.vstack([kept, new])`; meta is rebuilt in the same order. Real: after one upsert the count does not grow.
- **Why it is here:** Documents change; without replace-by-id you get duplicates (a stale rule and a new one).

### 5. delete by document  (`main.py` lines 43–45)
```python
    def delete(self, doc_id):
        keep = [i for i, m in enumerate(self.meta) if m["doc_id"] != doc_id]
        self.vectors, self.meta = self.vectors[keep], [self.meta[i] for i in keep]
```
- **In simple words:** Remove all chunks of a document.
- **Exactly:** Keeps rows whose `doc_id` differs. Real: deleting `brewing_guide` removes 4 rows.
- **Why it is here:** When a document is withdrawn it must disappear from answers.

### 6. search with a metadata filter  (`main.py` lines 47–58)
```python
    def search(self, query_vec, k=3, where=None):
        """Top-k by dot product. `where` = metadata filter applied BEFORE ranking (pre-filtering)."""
        rows = np.arange(len(self.meta))
        if where:
            rows = np.array([i for i in rows if all(self.meta[i].get(f) == v for f, v in where.items())], dtype=int)
        if len(rows) == 0:
            return []
        scores = self.vectors[rows] @ query_vec                      # (len(rows),)
        k = min(k, len(rows))
        top = np.argpartition(-scores, k - 1)[:k]                    # O(n): the k best, unordered
        top = top[np.argsort(-scores[top])]                          # O(k log k): order just those
        return [(float(scores[j]), self.meta[rows[j]]) for j in top]
```
- **In simple words:** Find the k closest vectors, optionally only among rows with given facts.
- **Exactly:** `rows` are the candidate row numbers. `where={"doc_id": "x"}` keeps rows where every key equals the value (pre-filter). `scores = vectors[rows] @ query_vec` has shape `(len(rows),)`. `argpartition(-scores, k-1)[:k]` finds the k best in O(n); `argsort` orders only those k. Returns `(score, meta)` pairs.
- **Why it is here:** Pre-filtering always returns k results when k exist; post-filtering can return fewer.

### 7. save  (`main.py` lines 60–63)
```python
    def save(self, folder):
        folder = Path(folder)
        np.save(folder / "vectors.npy", self.vectors)                # binary float32 matrix
        (folder / "meta.json").write_text(json.dumps(self.meta))     # metadata as JSON
```
- **In simple words:** Write the index to disk.
- **Exactly:** `np.save` writes `vectors.npy` (binary float32); `meta.json` holds metadata. Real size: 59,176 bytes for 32 vectors.
- **Why it is here:** A restart should not mean re-embedding everything.

### 8. load  (`main.py` lines 65–71)
```python
    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        vectors = np.load(folder / "vectors.npy")
        index = cls(vectors.shape[1])
        index.vectors, index.meta = vectors, json.loads((folder / "meta.json").read_text())
        return index
```
- **In simple words:** Read the index back from disk.
- **Exactly:** `@classmethod` builds an instance without needing an existing object; dimension from `vectors.shape[1]`.
- **Why it is here:** Round trip must give identical results (checked in section 4).

### 9. Build the index from the knowledge base  (`main.py` lines 74–79)
```python
# ---- 1. Build the index from the knowledge base --------------------------------------------------------
embedder = Embedder()
chunks = build_chunks()
index = FlatIndex(embedder.dim)
index.upsert(embedder.encode([chunk_for_embedding(c) for c in chunks]), chunks)
print(f"index: {len(index.meta)} vectors x {index.dim} dims")
```
- **In simple words:** Embed all 36 chunks and put them in.
- **Exactly:** `Embedder()` (384-dim); `encode` returns `(36, 384)`; the chunks serve as metadata (each has `chunk_id`, `doc_id`, ...). Real: `index: 36 vectors x 384 dims`.
- **Why it is here:** The same flow a vector DB does on `upsert`.

### 10. First search  (`main.py` lines 81–84)
```python
q = embedder.encode(["Do my reward points run out?"])[0]
print("\nQ: Do my reward points run out?")
for score, m in index.search(q, k=3):
    print(f"  {score:.3f}  {m['chunk_id']}")
```
- **In simple words:** Ask "Do my reward points run out?" and print the top 3.
- **Exactly:** Real: `0.593 loyalty_program#Earning points`, `0.547 ...#Points expiry`, `0.531 ...#Redeeming points`. The "Points expiry" chunk is the exact answer but ranks second: a small embedding model is not always right; step 07 fixes the order.
- **Why it is here:** Shows both that search works and that ranking is imperfect.

### 11. Pre-filter versus post-filter  (`main.py` lines 86–93)
```python
# ---- 2. Metadata filtering -----------------------------------------------------------------------------
print("\nsame query, filtered to doc_id=loyalty_program:")
for score, m in index.search(q, k=3, where={"doc_id": "loyalty_program"}):
    print(f"  {score:.3f}  {m['chunk_id']}")
# POST-filtering (search top-k first, filter afterwards) can return FEWER than k results, or none:
post = [r for r in index.search(q, k=3) if r[1]["doc_id"] == "brewing_guide"]
pre = index.search(q, k=3, where={"doc_id": "brewing_guide"})
print(f"filter doc_id=brewing_guide -> post-filter returns {len(post)} results, pre-filter returns {len(pre)}")
```
- **In simple words:** Restrict to one document, and compare two ways of filtering.
- **Exactly:** Filtered to `loyalty_program` gives the same three hits. For `brewing_guide`: post-filter (search top 3 then keep brewing_guide) returns 0; pre-filter returns 3.
- **Why it is here:** In production, filter before the nearest-neighbour search or you can lose all results.

### 12. Upsert and delete in action  (`main.py` lines 95–100)
```python
# ---- 3. Upsert and delete: the knowledge base changes -------------------------------------------------
new = dict(chunks[0], chunk_id="allergens_and_ingredients#Nuts",
           text="UPDATED: From 2027 every blend, including Winter Kahwa, is nut free.")
index.upsert(embedder.encode([chunk_for_embedding(new)]), [new])
index.delete("brewing_guide")
print(f"\nafter 1 upsert (replaced, not duplicated) and deleting brewing_guide: {len(index.meta)} vectors")
```
- **In simple words:** Replace one chunk's text and delete a document.
- **Exactly:** `dict(chunks[0], chunk_id=..., text=...)` copies a dict with overrides. The upsert replaces the existing `allergens_and_ingredients#Nuts` row. 36 vectors minus 4 deleted = 32 (real output).
- **Why it is here:** Proves the knowledge base can change without rebuilding.

### 13. Save and reload round trip  (`main.py` lines 102–108)
```python
# ---- 4. Save / load round trip -------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    index.save(tmp)
    size = sum(p.stat().st_size for p in Path(tmp).iterdir())
    again = FlatIndex.load(tmp)
    same = again.search(q, k=1)[0][1]["chunk_id"] == index.search(q, k=1)[0][1]["chunk_id"]
    print(f"saved to disk: {size:,} bytes | reloaded index gives the same top hit: {same}")
```
- **In simple words:** Save to a temporary folder, reload, and check the same answer comes back.
- **Exactly:** `TemporaryDirectory()` deletes itself at the end. Real: `saved to disk: 59,176 bytes | reloaded index gives the same top hit: True`.
- **Why it is here:** Persistence that is not tested is not persistence.

### 14. A big random matrix  (`main.py` lines 110–116)
```python
# ---- 5. How brute force scales ------------------------------------------------------------------------
rng = np.random.default_rng(0)
N, D = 100_000, 384
big = rng.normal(size=(N, D)).astype(np.float32)
big /= np.linalg.norm(big, axis=1, keepdims=True)                   # unit rows, like real embeddings
query = big[123] + 0.05 * rng.normal(size=D).astype(np.float32)
query /= np.linalg.norm(query)
```
- **In simple words:** Make 100,000 random unit vectors to test speed.
- **Exactly:** `default_rng(0)` seeds randomness. `big` has shape `(100000, 384)` float32 (154 MB). Rows divided by their norms to be unit length. `query` is vector 123 plus small noise, normalised.
- **Why it is here:** Random unit vectors behave like embeddings for a timing test.

### 15. Time the search  (`main.py` lines 118–128)
```python
def timed(fn, repeats=5):
    fn()                                                            # warm-up
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats * 1000

t_part = timed(lambda: np.argpartition(-(big @ query), 9)[:10])
t_sort = timed(lambda: np.argsort(-(big @ query))[:10])
print(f"\nbrute force over {N:,} vectors ({big.nbytes / 1e6:.0f} MB float32): {t_part:.1f} ms per query "
      f"(argpartition) vs {t_sort:.1f} ms (full argsort)")
```
- **In simple words:** Measure how long one query takes.
- **Exactly:** `timed` runs once as warm-up, then averages 5 runs in milliseconds. `argpartition` over `big @ query` takes 11.6 ms; full `argsort` takes 20.5 ms (your numbers vary).
- **Why it is here:** The O(n) partition beats the O(n log n) sort, and both grow linearly with N.

### 16. Extrapolate  (`main.py` lines 129–130)
```python
print(f"float16 storage would be {big.astype(np.float16).nbytes / 1e6:.0f} MB; "
      f"10M vectors would be {10_000_000 * D * 4 / 1e9:.1f} GB float32 and ~{t_part * 100:.0f} ms per query -> ANN (step 05)")
```
- **In simple words:** Estimate what 10 million vectors would cost.
- **Exactly:** float16 would halve memory (77 MB). 10M vectors is `10_000_000 * 384 * 4 / 1e9 = 15.4 GB` and about 1155 ms per query (100 times the time): too slow, so use ANN (step 05).
- **Why it is here:** Motivates approximate search.

### 17. Cosine, dot and L2 agree  (`main.py` lines 133–139)
```python
sample = big[:2000]
cos = sample @ query / (np.linalg.norm(sample, axis=1) * np.linalg.norm(query))
dot = sample @ query
l2 = np.linalg.norm(sample - query, axis=1)                         # ||a-b||^2 = 2 - 2 cos(a,b) for unit vectors
same_rank = (np.argsort(-cos)[:10] == np.argsort(-dot)[:10]).all() and (np.argsort(-cos)[:10] == np.argsort(l2)[:10]).all()
print(f"rankings identical for cosine, dot and L2 on unit vectors: {bool(same_rank)}")
print(f"check the identity on one row: ||a-b||^2 = {l2[0] ** 2:.4f}, 2 - 2cos = {2 - 2 * cos[0]:.4f}")
```
- **In simple words:** Prove the three similarity measures give the same ranking on unit vectors.
- **Exactly:** `cos`, `dot` and `l2` over 2000 vectors; the top 10 by `-cos`, `-dot` and `+l2` are identical. Identity: `||a-b||^2 = 2 - 2cos(a,b)` for unit vectors; real check on one row: 2.0094 both.
- **Why it is here:** Lets you choose dot product (cheapest) without worrying about the metric.

### 18. Exercises as comments  (`main.py` lines 141–144)
```python
# TRY IT: N = 1_000_000 (needs ~1.5 GB RAM). Is the per-query time roughly 10x?
# TRY IT: search with where={"doc_id": "product_catalog"} for "cheapest tea" and print all four results.
# TRY IT: remove the `keep` logic in upsert and upsert the same chunk twice; search returns duplicates.
# BREAK IT: skip the normalisation of `big` and compare argsort of cos and dot: the rankings now differ.
```
- **In simple words:** Ideas to try and break.
- **Exactly:** BREAK IT: skip the normalisation and the rankings of cosine and dot differ.
- **Why it is here:** Normalisation is what makes the metrics equivalent.


## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `class FlatIndex:` + `__init__` | an object that keeps state (`vectors`, `meta`) | `index = FlatIndex(384)` |
| `@classmethod` + `cls(...)` | an alternative constructor called on the class, not an instance | `FlatIndex.load(folder)` |
| `{m["chunk_id"] for m in metas}` | set comprehension: fast "is this id in there?" | `"a" in {"a", "b"}` |
| `self.vectors[keep]` | fancy indexing: keep only the listed rows | `x[[0, 2]]` |
| `all(cond for f, v in where.items())` | True only if every filter field matches | `all([True, False])` -> False |
| `dict(chunks[0], chunk_id=...)` | copy a dict and override some keys | `dict({"a": 1}, a=2)` -> `{"a": 2}` |
| `with tempfile.TemporaryDirectory() as tmp:` | a folder that is deleted automatically afterwards | |
| `lambda: ...` | tiny anonymous function, passed to `timed` | `timed(lambda: big @ query)` |
| `big /= np.linalg.norm(big, axis=1, keepdims=True)` | divide each row by its own length -> unit rows | `keepdims` keeps shape `(n, 1)` for broadcasting |
| `f"{N:,}"`, `f"{x:.1f}"` | thousands separators, one decimal | `100,000`, `8.2` |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `Embedder().encode(texts)` | local `common.py` | MiniLM bi-encoder, mean pooling, L2 normalise | list of str | float32 `(n, 384)` |
| `build_chunks()`, `chunk_for_embedding(c)` | local `common.py` | 36 section chunks; prepends "title - section:" | – | list of dicts / str |
| `np.argpartition(a, kth)` | numpy | k best positions, unordered, O(n) | `kth = k-1` on `-scores` | index array |
| `np.argsort(a)` | numpy | full sort order, O(n log n) | negate for descending | index array |
| `np.vstack([a, b])` | numpy | stack rows | list of arrays | ndarray |
| `np.save` / `np.load` | numpy | binary `.npy` file for the matrix | path, array | – / ndarray |
| `json.dumps` / `json.loads` | standard library | metadata to and from text | – | str / list |
| `Path.stat().st_size` | standard library (`pathlib`) | file size in bytes | – | int |
| `time.perf_counter()` | standard library | high-resolution timer | – | float seconds |
| `np.linalg.norm(x, axis=1)` | numpy | length of each row | `axis`, `keepdims` | ndarray |
| `.astype(np.float16)`, `.nbytes` | numpy | change precision; bytes used | dtype | ndarray / int |

## Execution trace (real run)
- `index: 36 vectors x 384 dims`: the 36 section chunks of the 9-document knowledge base.
- Query "Do my reward points run out?":
  ```
  0.593  loyalty_program#Earning points
  0.547  loyalty_program#Points expiry
  0.531  loyalty_program#Redeeming points
  ```
  Honest note: the chunk that actually answers it, `Points expiry`, is 2nd, not 1st. The search is exact; the *embedding* is the weak link. Keep this in mind for steps 06 and 07.
- Filter `doc_id=loyalty_program`: same three results (they were already all loyalty chunks).
- Filter `doc_id=brewing_guide`: **post-filter returns 0 results, pre-filter returns 3**. The global top-3 contained no brewing chunks, so filtering afterwards left nothing.
- Upsert of the existing id `allergens_and_ingredients#Nuts` replaced it (no duplicate), then deleting `brewing_guide` removed 4 chunks: **32 vectors** left.
- Save/load: **59,176 bytes** on disk (32 × 384 × 4 = 49,152 bytes of vectors + `.npy` header + JSON metadata); reloaded index gives the same top hit: `True`.
- Brute force over 100,000 random unit vectors (**154 MB** float32): **8.2 ms** per query with `argpartition` vs **19.1 ms** with full `argsort`. Timings vary per run and machine.
- float16 would be **77 MB**; 10M vectors would be **15.4 GB** float32 and **~817 ms** per query (linear extrapolation).
- `rankings identical for cosine, dot and L2 on unit vectors: True`, and the identity check `||a-b||^2 = 2.0094, 2 - 2cos = 2.0094`.
- Whole script: about 10 s wall time (9.9 s in this run), most of it loading the model.

## Experiments (change ONE thing)
1. `N = 1_000_000` in section 5 (needs ~1.5 GB RAM for the matrix). Predict: per-query time roughly 10x, because cost is O(n·d).
2. Search `"cheapest tea"` with `where={"doc_id": "product_catalog"}` and `k=10`. Only the catalogue chunks come back; count them.
3. Delete the `keep` lines in `upsert` and upsert the same chunk twice: search now returns the same chunk id twice, wasting a top-k slot.
4. Replace pre-filtering with post-filtering plus over-fetch (`index.search(q, k=12)` then filter). How large must the over-fetch be before `brewing_guide` returns 3?
5. Change `D = 384` to `768`: the time per query and the MB both roughly double.

## Debugging practice
BREAK IT from `main.py`: skip the normalisation of `big` (comment out `big /= np.linalg.norm(...)`). Confirmed in a temporary copy:
```
rankings identical for cosine, dot and L2 on unit vectors: False
check the identity on one row: ||a-b||^2 = 389.7090, 2 - 2cos = 2.0526
```
No exception, just a wrong answer. The raw Gaussian rows have lengths of about 17 to 22 (roughly √384 ≈ 19.6), so the dot product now rewards long vectors: the top-10 by cosine started `123 1569 1551 1134 ...` while the top-10 by dot started `123 1551 1569 1134 ...` and pulled in different rows (578, 1739). The identity `2 − 2cos` only holds for unit vectors. Lesson: silent metric bugs do not crash; assert `np.allclose(np.linalg.norm(vectors, axis=1), 1)` after loading.

## Rebuild checklist
- [ ] `vectors` `(0, dim)` float32 + `meta` list, one dict per row
- [ ] `upsert`: drop rows whose `chunk_id` is in the new batch, then `vstack`
- [ ] `delete(doc_id)`: keep rows with a different `doc_id`
- [ ] `search`: optional pre-filter rows -> `vectors[rows] @ q` -> `argpartition` -> sort the k
- [ ] `save` / `load`: `.npy` for vectors, `.json` for metadata
- [ ] time `argpartition` vs `argsort`; check `||a-b||² = 2 − 2cos` on unit vectors

## The new problem (cliffhanger)
Exact search is perfect but linear: 8 ms for 100k vectors becomes about 817 ms and 15.4 GB for 10 million. We need to look at only a tiny fraction of the vectors and still find almost all of the true neighbours. Next: [step 05, ANN with IVF, HNSW and PQ](../05_ann_ivf_and_hnsw/WALKTHROUGH.md).

> **Say it aloud (interview box):** At its core a vector database is a matrix of embeddings plus metadata, with top-k search, filtering, upserts, deletes and persistence on top; brute-force search is one matrix-vector product and an argpartition, O(n·d) per query, with perfect recall. For unit vectors cosine equals the dot product and squared L2 distance equals two minus two cosine, so all three rank identically, but if vectors are not normalised dot product also rewards length. I pre-filter rather than post-filter so a selective filter cannot return empty results, and I stay with brute force up to roughly a million vectors before paying for an ANN index.
