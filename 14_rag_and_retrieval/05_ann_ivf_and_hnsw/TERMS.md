# Terms — ANN: IVF, HNSW and PQ

Folder: `14_rag_and_retrieval/05_ann_ivf_and_hnsw` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [14 README](../README.md)

**Terms in this folder:** [ANN (approximate nearest neighbour)](#ann-approximate-nearest-neighbour) · [Curse of dimensionality](#curse-of-dimensionality) · [Recall@k](#recallk) · [Ground truth (brute force)](#ground-truth-brute-force) · [k-means (Lloyd's algorithm)](#k-means-lloyds-algorithm) · [IVF (inverted file index)](#ivf-inverted-file-index) · [nlist and nprobe](#nlist-and-nprobe) · [Border problem](#border-problem) · [kNN graph](#knn-graph) · [NSW (navigable small world)](#nsw-navigable-small-world) · [HNSW](#hnsw) · [M, efConstruction and efSearch](#m-efconstruction-and-efsearch) · [Greedy best-first graph search](#greedy-best-first-graph-search) · [Visited set](#visited-set) · [Product quantisation (PQ)](#product-quantisation-pq) · [Codebook and codes](#codebook-and-codes) · [Asymmetric distance computation (ADC)](#asymmetric-distance-computation-adc) · [Re-ranking with full vectors](#re-ranking-with-full-vectors) · [Scalar and binary quantisation](#scalar-and-binary-quantisation) · [IVF-PQ and DiskANN](#ivf-pq-and-diskann) · [Broadcasting](#broadcasting) · [np.uint8](#npuint8)

---

## ANN (approximate nearest neighbour)  `(concept)`
**1. Definition:** Approximate nearest-neighbour search returns most of the true nearest neighbours while examining only a small fraction of the stored vectors.
**2. In plain words:** Give up a little recall for a lot of speed. IVF with `nprobe=4` scanned 1.6% of 50,000 vectors and still found all of the true top-10 (recall 1.000) at 0.17 ms vs 3.83 ms for brute force.
**3. Example to quote:** > "ANN trades exactness for speed; I accept it only after measuring recall against brute force."
**Seen in:** `main.py`, docstring and all sections

## Curse of dimensionality  `(concept)`
**1. Definition:** The curse of dimensionality refers to effects in high-dimensional spaces such as distances between points becoming similar, which make space-partitioning structures like k-d trees nearly as slow as a linear scan.
**2. In plain words:** In 128 or 384 dimensions "near" and "far" are less different, so a tree cannot prune much. ANN methods use data structure (clusters, neighbour graphs) instead.
**3. Example to quote:** > "k-d trees degrade towards a linear scan at hundreds of dimensions, which is why vector search uses IVF or graph indexes."
**Seen in:** `WALKTHROUGH.md`, theory

## Recall@k  `(concept)`
**1. Definition:** Recall@k is the fraction of the exact top-k neighbours that the approximate method also returns, averaged over queries.
**2. In plain words:** If brute force says these 10 are nearest and the index returns 9 of them, recall@10 is 0.9. Always quote it next to a cost: NSW reached 0.978 while computing 10.4% of distances.
**3. Example to quote:** > "I report recall@10 and the fraction scanned together; recall alone or speed alone means nothing."
```python
np.mean([len(set(f[:k]) & set(t)) / k for f, t in zip(found, truth)])
```
**Seen in:** `main.py`, `recall_at_k`

## Ground truth (brute force)  `(concept)`
**1. Definition:** Ground truth is the exact top-k for each query, computed by brute force, used as the reference for measuring ANN recall.
**2. In plain words:** `truth = np.argsort(-(queries @ data.T))[:, :10]`; it cost 3.83 ms per query and scans 100% of vectors.
**3. Example to quote:** > "I compute exact neighbours for a few hundred queries once and evaluate every index setting against them."
**Seen in:** `main.py`, top (`truth`, `gtruth`)

## k-means (Lloyd's algorithm)  `(concept)`
**1. Definition:** k-means partitions points into k clusters by alternating between assigning each point to its nearest centroid and moving each centroid to the mean of its points.
**2. In plain words:** It is the training step of both IVF (256 cells, 0.6 s) and PQ (16 codebooks of 256 centroids, 8.4 s).
**3. Example to quote:** > "IVF and PQ both need a k-means training step on a representative sample before you can add vectors."
**Seen in:** `main.py`, `kmeans`

## IVF (inverted file index)  `(concept)`
**1. Definition:** An IVF index clusters vectors with k-means into `nlist` cells, stores each vector in its cell's inverted list, and at query time scans only the lists of the `nprobe` nearest centroids.
**2. In plain words:** Like searching only the nearest few shelves of a library. 256 cells of about 195 vectors; `nprobe=1` gave recall 0.919 scanning 0.5%.
**3. Example to quote:** > "IVF with 256 cells hit recall 0.986 at nprobe 2 while scanning under 1% of the vectors."
**Seen in:** `main.py`, section A

## nlist and nprobe  `(concept)`
**1. Definition:** `nlist` is the number of IVF cells (centroids); `nprobe` is how many of the nearest cells are searched per query.
**2. In plain words:** `nprobe` is the speed/recall dial: 1 → 0.919, 2 → 0.986, 4 → 1.000 in our run. `nlist` sets cell size; FAISS suggests roughly 4√n to 16√n cells.
**3. Example to quote:** > "I tune nprobe at query time to hit a recall target, and pick nlist around a few times the square root of n."
**Seen in:** `main.py`, `NLIST`, nprobe loop

## Border problem  `(concept)`
**1. Definition:** The border problem is that a query's true neighbours can lie in an adjacent cell just across a partition boundary, which a small `nprobe` never searches.
**2. In plain words:** It is why recall at `nprobe=1` was 0.919, not 1.0. Probing more cells fixes it at the cost of more scanning.
**3. Example to quote:** > "Vectors near a cell border get missed at nprobe 1, so I probe a few neighbouring cells."
**Seen in:** `main.py`, print after section A

## kNN graph  `(concept)`
**1. Definition:** A k-nearest-neighbour graph connects every node to its M most similar nodes with directed edges.
**2. In plain words:** Built here with M=16 for 10,000 nodes in 1.1 s. On its own it is a poor search structure: greedy walks get trapped, recall only 0.354 at ef=100.
**3. Example to quote:** > "A plain kNN graph lacks long-range links, so a greedy search from a random entry gets stuck in a local optimum."
**Seen in:** `main.py`, section B (`knn`, `directed`)

## NSW (navigable small world)  `(concept)`
**1. Definition:** A navigable small-world graph is a proximity graph with both short local edges and some long-range edges, so greedy routing reaches any region in few hops.
**2. In plain words:** Making edges two-way and adding 4 random long-range edges per node lifted recall from 0.354 to 0.978 at ef=100.
**3. Example to quote:** > "Long-range shortcuts make the graph small-world; that is the core idea HNSW builds on."
**Seen in:** `main.py`, `nsw`

## HNSW  `(concept)`
**1. Definition:** Hierarchical Navigable Small World is a graph ANN index with multiple layers: sparse upper layers hold long-range links, layer 0 holds all nodes, and search descends greedily from the top before a beam search at the bottom.
**2. In plain words:** A skip list made of graphs. Fast, high recall, insert-friendly and needs no training, but costs RAM for links and handles deletes poorly. Our NSW is the single-layer version of the idea.
**3. Example to quote:** > "HNSW is the default index in most vector databases: great recall and latency, incremental inserts, but memory-hungry and awkward with deletes."
**Seen in:** `main.py`, section B comments; `WALKTHROUGH.md`, theory

## M, efConstruction and efSearch  `(concept)`
**1. Definition:** In HNSW, `M` is the maximum number of links per node per layer, `efConstruction` is the candidate-list size used while building, and `efSearch` is the candidate-list size used at query time (must be at least k).
**2. In plain words:** Our `ef` plays the role of `efSearch`: NSW recall 0.356 at ef=10, 0.911 at 50, 0.978 at 100, with work growing from 217 to 1038 distances.
**3. Example to quote:** > "M and efConstruction set graph quality and memory at build time; efSearch is the knob I turn per query for recall versus latency."
**Seen in:** `main.py`, `M`, `ef` loop

## Greedy best-first graph search  `(concept)`
**1. Definition:** Best-first search repeatedly expands the most promising unexpanded node, scoring its neighbours and keeping the `ef` best found so far, and stops when no frontier node can improve that set.
**2. In plain words:** Walk towards whatever neighbour looks closest to the query, keeping a short list of the best seen. Real libraries use heaps; the script uses a sorted list for clarity.
**3. Example to quote:** > "Graph search is a beam search: expand the closest candidate, keep the ef best, stop when the frontier is worse than the worst kept."
**Seen in:** `main.py`, `graph_search`

## Visited set  `(concept)`
**1. Definition:** A visited set records nodes already scored so a graph search never processes the same node twice.
**2. In plain words:** Without it (the BREAK IT) recall at ef=100 fell from 0.978 to 0.016 and distance computations rose from 1038 to 2855, because duplicates filled the result list.
**3. Example to quote:** > "Removing the visited check did not crash; recall went to almost zero because the result list filled with duplicates."
**Seen in:** `main.py`, `graph_search`; `WALKTHROUGH.md`, Debugging practice

## Product quantisation (PQ)  `(concept)`
**1. Definition:** Product quantisation splits each vector into m sub-vectors, quantises each with its own k-means codebook, and stores only the centroid ids.
**2. In plain words:** 128 dims = 16 pieces of 8 dims; each piece becomes one byte. 512 bytes → 16 bytes per vector, 32x smaller.
**3. Example to quote:** > "PQ with 16 sub-spaces of 256 centroids turns a 512-byte vector into 16 bytes."
**Seen in:** `main.py`, section C

## Codebook and codes  `(concept)`
**1. Definition:** A PQ codebook is the set of 256 centroids learned for one sub-space; a code is the index (0–255, one `uint8`) of the nearest centroid for a sub-vector.
**2. In plain words:** 16 codebooks × 256 centroids are stored once; each vector stores just 16 small integers.
**3. Example to quote:** > "256 centroids per sub-space is chosen because an id then fits in exactly one byte."
**Seen in:** `main.py`, `codebooks`, `codes`

## Asymmetric distance computation (ADC)  `(concept)`
**1. Definition:** ADC estimates the similarity between an unquantised query and PQ-encoded vectors by precomputing, per sub-space, the query's similarity to every centroid, then summing table entries selected by each vector's codes.
**2. In plain words:** One small table (16 × 256) per query, then each database vector costs 16 lookups and adds. "Asymmetric" because only the database side is compressed.
**3. Example to quote:** > "With ADC the query stays full precision; scoring a PQ vector is just m table lookups."
```python
tables = np.stack([codebooks[s] @ q[s*ds:(s+1)*ds] for s in range(SUB)])   # (16, 256)
approx = tables[np.arange(SUB), codes].sum(axis=1)                         # (N,)
```
**Seen in:** `main.py`, section C

## Re-ranking with full vectors  `(concept)`
**1. Definition:** Re-ranking takes a shortlist produced by a compressed or approximate index and re-scores it with exact (full-precision) vectors.
**2. In plain words:** PQ alone gave recall 0.362; re-scoring its top-100 with full vectors gave 0.952. The compressed index finds candidates, exact maths orders them.
**3. Example to quote:** > "PQ alone hit 0.36 recall; re-ranking a 100-item shortlist with full vectors brought it to 0.95."
**Seen in:** `main.py`, `found_rerank`

## Scalar and binary quantisation  `(concept)`
**1. Definition:** Scalar quantisation maps each float to a low-bit integer (typically int8) using a per-dimension range; binary quantisation keeps one bit (the sign) per dimension and compares with Hamming distance.
**2. In plain words:** int8 is 4x smaller with little recall loss; binary is 32x smaller and usually needs re-scoring. Both are common add-ons to HNSW.
**3. Example to quote:** > "For RAM I'd start with int8 scalar quantisation on HNSW, and use binary plus re-scoring only for very large corpora."
**Seen in:** `WALKTHROUGH.md`, theory

## IVF-PQ and DiskANN  `(concept)`
**1. Definition:** IVF-PQ combines an IVF coarse quantiser with PQ-encoded residuals inside each cell; DiskANN is a graph index designed to keep the graph and full vectors on SSD with PQ codes in RAM.
**2. In plain words:** The script ends with "Production combines them: IVF-PQ (FAISS), HNSW + PQ/SQ". These combinations are how billion-scale search fits on few machines.
**3. Example to quote:** > "IVF-PQ gives coarse partitioning plus compression; DiskANN pushes a graph index to SSD for billion-scale data."
**Seen in:** `main.py`, last print; `WALKTHROUGH.md`, theory

## Broadcasting  `(third-party: numpy)`
**1. Definition:** Broadcasting lets numpy combine arrays of different shapes by virtually repeating size-1 dimensions.
**2. In plain words:** `part[:, None, :] - cb[None]` subtracts every one of 256 centroids from every sub-vector at once, giving `(n, 256, 8)`, then `.sum(-1)` gives squared distances.
**3. Example to quote:** > "I compute all point-to-centroid distances with one broadcast instead of a double loop."
**Seen in:** `main.py`, `kmeans`, section C

## np.uint8  `(third-party: numpy)`
**1. Definition:** `np.uint8` is an unsigned 8-bit integer type holding values 0–255.
**2. In plain words:** Exactly one byte, which is why a PQ code array of shape `(50000, 16)` is 16 bytes per vector.
**3. Example to quote:** > "Codes are uint8, so 16 sub-spaces cost 16 bytes per vector."
**Seen in:** `main.py`, `codes = np.zeros((N, SUB), dtype=np.uint8)`
