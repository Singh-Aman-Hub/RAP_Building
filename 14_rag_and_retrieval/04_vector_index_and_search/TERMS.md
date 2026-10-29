# Terms — Vector Index and Exact Search

Folder: `14_rag_and_retrieval/04_vector_index_and_search` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [14 README](../README.md)

**Terms in this folder:** [Vector database](#vector-database) · [Flat index](#flat-index) · [Exact (brute-force) kNN](#exact-brute-force-knn) · [Metadata](#metadata) · [Top-k](#top-k) · [np.argpartition](#npargpartition) · [np.argsort](#npargsort) · [Cosine similarity](#cosine-similarity) · [Dot product](#dot-product) · [L2 (Euclidean) distance](#l2-euclidean-distance) · [Unit vectors and normalisation](#unit-vectors-and-normalisation) · [Pre-filtering vs post-filtering](#pre-filtering-vs-post-filtering) · [Upsert](#upsert) · [Delete and tombstones](#delete-and-tombstones) · [Persistence (np.save and json)](#persistence-npsave-and-json) · [@classmethod](#classmethod) · [tempfile.TemporaryDirectory](#tempfiletemporarydirectory) · [time.perf_counter](#timeperf_counter) · [Memory arithmetic and float16](#memory-arithmetic-and-float16) · [ANN (approximate nearest neighbour)](#ann-approximate-nearest-neighbour)

---

## Vector database  `(concept)`
**1. Definition:** A vector database stores embedding vectors with metadata and serves nearest-neighbour queries through an index, with insert/upsert/delete by id, metadata filtering and persistence.
**2. In plain words:** It is "a matrix of vectors plus a list of labels" with a search button. Our `FlatIndex` holds 36 vectors of 384 dims and their chunk metadata; FAISS, Qdrant, Milvus, Weaviate and pgvector add fast ANN indexes, replication and scaling on top.
**3. Example to quote:** > "Strip a vector database down and it is vectors, metadata, a top-k search and CRUD; my FlatIndex does all of that in about 50 lines."
**Seen in:** `main.py`, `class FlatIndex`

## Flat index  `(concept)`
**1. Definition:** A flat index stores vectors uncompressed in one array and answers queries by comparing the query with every stored vector.
**2. In plain words:** No clustering, no graph, no compression: just the raw matrix. FAISS calls it `IndexFlatIP` (inner product) or `IndexFlatL2`. It is the gold standard that ANN indexes are measured against.
**3. Example to quote:** > "I use a flat index as ground truth; any ANN index is judged by how many of the flat index's top-10 it recovers."
**Seen in:** `main.py`, `FlatIndex`

## Exact (brute-force) kNN  `(concept)`
**1. Definition:** Exact k-nearest-neighbour search computes the similarity between the query and all n stored vectors and returns the k best, at O(n·d) cost per query.
**2. In plain words:** Perfect accuracy, linear cost. 100,000 × 384 took 8.2 ms per query in our run; extrapolated, 10 million would be about 817 ms.
**3. Example to quote:** > "Brute force over 100k 384-dim vectors is about 8 ms on a laptop, fine up to roughly a million vectors, but it grows linearly."
**Seen in:** `main.py`, section 5

## Metadata  `(concept)`
**1. Definition:** Metadata (or payload) is the structured data stored alongside each vector, such as `chunk_id`, `doc_id`, title, section and text.
**2. In plain words:** The vector finds the row; the metadata tells you what the row *is*, lets you filter, and gives the text to send to the LLM. Here it is one dict per row in `self.meta`.
**3. Example to quote:** > "Each vector carries its chunk_id, doc_id and text as metadata, so I can filter by document and cite the source."
**Seen in:** `main.py`, `self.meta`

## Top-k  `(concept)`
**1. Definition:** Top-k retrieval returns the k items with the highest similarity score to the query, in descending order.
**2. In plain words:** We ask for k=3: the three most similar chunks. For "Do my reward points run out?" they were Earning points 0.593, Points expiry 0.547, Redeeming points 0.531.
**3. Example to quote:** > "Top-3 for the points question were all loyalty chunks, though the true answer, Points expiry, came second."
**Seen in:** `main.py`, section 1

## np.argpartition  `(third-party: numpy)`
**1. Definition:** `np.argpartition(a, kth)` returns indices such that the element at position `kth` is where it would be in sorted order, with all smaller elements before it, using an O(n) selection algorithm.
**2. In plain words:** It finds the k best *without* sorting everything. Negate the scores to get the largest, take the first k, then sort only those k. 8.2 ms vs 19.1 ms for a full sort here.
**3. Example to quote:** > "`np.argpartition(-scores, k-1)[:k]` gives the top-k unordered in linear time, then I sort just those k."
```python
top = np.argpartition(-scores, k - 1)[:k]
top = top[np.argsort(-scores[top])]
```
**Seen in:** `main.py`, `search` and section 5

## np.argsort  `(third-party: numpy)`
**1. Definition:** `np.argsort(a)` returns the indices that would sort `a` in ascending order, O(n log n).
**2. In plain words:** Sort everything, then slice. Correct but wasteful when you only need 10 of 100,000: 19.1 ms vs 8.2 ms in our run.
**3. Example to quote:** > "A full argsort is n log n; for top-10 out of 100k I use argpartition instead."
**Seen in:** `main.py`, section 5 and 6

## Cosine similarity  `(concept)`
**1. Definition:** Cosine similarity is `a·b / (||a|| ||b||)`, the cosine of the angle between two vectors, ranging from −1 to 1.
**2. In plain words:** It compares direction only, so a long and a short vector pointing the same way score 1. Our MiniLM scores (0.593 etc.) are cosines because the vectors are unit length.
**3. Example to quote:** > "Cosine ignores vector length; on unit vectors it is exactly the dot product."
**Seen in:** `main.py`, section 6

## Dot product  `(concept)`
**1. Definition:** The dot (inner) product `a·b = Σ aᵢbᵢ` combines the angle and the lengths of two vectors.
**2. In plain words:** It is what `vectors @ query` computes. If vectors are not normalised, longer vectors score higher; some models are trained with dot product deliberately, so use the metric the model was trained with.
**3. Example to quote:** > "I normalise everything so the dot product equals cosine; with unnormalised vectors the rankings stopped matching."
**Seen in:** `main.py`, `search`, section 6

## L2 (Euclidean) distance  `(concept)`
**1. Definition:** L2 distance is `||a − b|| = sqrt(Σ (aᵢ − bᵢ)²)`, the straight-line distance; smaller means more similar.
**2. In plain words:** For unit vectors `||a − b||² = 2 − 2cos(a, b)`, so ranking by smallest L2 equals ranking by largest cosine. The script printed 2.0094 for both sides.
**3. Example to quote:** > "On unit vectors squared L2 is two minus two cosine, so cosine, dot and L2 all give the same ranking."
**Seen in:** `main.py`, section 6

## Unit vectors and normalisation  `(concept)`
**1. Definition:** Normalising divides a vector by its L2 norm so it has length 1.
**2. In plain words:** Do it to both documents and queries and the three metrics agree. Skipping it (the BREAK IT) made the rankings differ and the identity fail: 389.7090 vs 2.0526.
**3. Example to quote:** > "I assert all stored norms are 1, because a forgotten normalisation does not crash, it silently changes the ranking."
```python
big /= np.linalg.norm(big, axis=1, keepdims=True)
```
**Seen in:** `main.py`, section 5; `WALKTHROUGH.md`, Debugging practice

## Pre-filtering vs post-filtering  `(concept)`
**1. Definition:** Pre-filtering restricts the candidate set by metadata before ranking; post-filtering ranks first and removes non-matching results afterwards.
**2. In plain words:** Filtering to `brewing_guide`: post-filter returned 0 results, pre-filter returned 3. Post-filtering can come back empty; pre-filtering is easy on a flat index but hard inside an ANN graph, which is why vector databases implement filter-aware search.
**3. Example to quote:** > "Post-filtering the top-3 to brewing_guide gave zero results; pre-filtering gave three."
**Seen in:** `main.py`, section 2

## Upsert  `(concept)`
**1. Definition:** Upsert is "update or insert": if a record with the same id exists it is replaced, otherwise it is added.
**2. In plain words:** When the nuts policy changes we re-embed that chunk and upsert it under the same `chunk_id`, so the index does not hold an old and a new copy.
**3. Example to quote:** > "I upsert by chunk_id so an edited document replaces its old vectors instead of duplicating them."
**Seen in:** `main.py`, `upsert`, section 3

## Delete and tombstones  `(concept)`
**1. Definition:** A tombstone is a marker that flags a record as deleted so searches skip it, with physical removal deferred to a later compaction or rebuild.
**2. In plain words:** In our flat index delete really removes rows (36 → 32 after dropping brewing_guide's 4 chunks). ANN indexes usually cannot cheaply remove a node, so they tombstone and clean up later.
**3. Example to quote:** > "Flat indexes delete rows directly; HNSW-style indexes tombstone and compact later."
**Seen in:** `main.py`, `delete`, section 3

## Persistence (np.save and json)  `(third-party: numpy)`
**1. Definition:** `np.save` writes an array to a binary `.npy` file and `np.load` reads it back; `json.dumps`/`json.loads` serialise the metadata.
**2. In plain words:** Embedding is the expensive part, so save once and reload. Our 32-vector index was 59,176 bytes on disk and reloaded with the same top hit.
**3. Example to quote:** > "Vectors go to .npy, metadata to JSON; reloading gives the same top hit, so I never re-embed on restart."
**Seen in:** `main.py`, `save`, `load`, section 4

## @classmethod  `(syntax)`
**1. Definition:** `@classmethod` makes a method receive the class (`cls`) instead of an instance, commonly used for alternative constructors.
**2. In plain words:** `FlatIndex.load(folder)` builds a new index from files without first creating an empty one yourself.
**3. Example to quote:** > "`load` is a classmethod, so `FlatIndex.load(path)` returns a ready index."
**Seen in:** `main.py`, `FlatIndex.load`

## tempfile.TemporaryDirectory  `(standard library)`
**1. Definition:** `tempfile.TemporaryDirectory()` creates a temporary folder that is deleted automatically when the `with` block ends.
**2. In plain words:** The save/load test leaves no files behind.
**3. Example to quote:** > "I test persistence inside a TemporaryDirectory so the demo cleans up after itself."
**Seen in:** `main.py`, section 4

## time.perf_counter  `(standard library)`
**1. Definition:** `time.perf_counter()` returns a high-resolution monotonic clock value in seconds, suited to measuring short durations.
**2. In plain words:** `timed()` runs once to warm up, then averages 5 runs. Results like 8.2 ms vary from run to run and machine to machine.
**3. Example to quote:** > "I warm up once and average five runs with perf_counter so one slow call does not skew the timing."
**Seen in:** `main.py`, `timed`

## Memory arithmetic and float16  `(concept)`
**1. Definition:** A float32 embedding matrix needs n × d × 4 bytes; float16 needs 2 bytes per value, int8 one, binary one bit.
**2. In plain words:** 100,000 × 384 × 4 = 154 MB; float16 is 77 MB; 10 million vectors would be 15.4 GB in float32. Lower precision costs a little accuracy, usually recovered by re-scoring a shortlist.
**3. Example to quote:** > "Ten million 384-dim float32 vectors are about 15 GB, so at that scale I quantise to float16 or int8."
**Seen in:** `main.py`, section 5

## ANN (approximate nearest neighbour)  `(concept)`
**1. Definition:** Approximate nearest-neighbour search uses an index that examines only a fraction of the vectors and returns most, but not always all, of the true nearest neighbours.
**2. In plain words:** It trades a little recall for a lot of speed. The 817 ms per query extrapolated for 10M vectors is why step 05 exists.
**3. Example to quote:** > "Beyond about a million vectors I switch from flat to an ANN index like HNSW and measure recall against the flat index."
**Seen in:** `main.py`, section 5 print; next step
