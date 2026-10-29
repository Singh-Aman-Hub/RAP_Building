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
