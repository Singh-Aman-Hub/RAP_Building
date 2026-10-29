# Step 05 — ANN: IVF, graph search (HNSW idea) and product quantisation

## The story so far
[Step 04](../04_vector_index_and_search/WALKTHROUGH.md) built an exact flat index. It is perfect but linear: about 8 ms per query for 100k vectors, an extrapolated ~817 ms and 15.4 GB of float32 for 10 million.

## The pain point
Every query touches every vector. More data means proportionally more time and more RAM, and a production RAG service may need hundreds of queries per second over tens of millions of chunks. We cannot afford to look at everything, and we cannot afford 1.5 KB of float32 per 384-dim vector either.

## The spark
Real embeddings are not uniform noise: they cluster by topic, and their neighbours are neighbours of each other. Exploit that structure three ways: (A) **partition** the space and search only the nearby partitions (IVF), (B) **link** each vector to its neighbours and walk the graph towards the query (NSW/HNSW), (C) **compress** each vector to a few bytes and compare with lookup tables (PQ). Judge every trick by **recall@10** against brute force: speed only counts if the answers stay right.

## Goal
Implement IVF, a navigable small-world graph search and product quantisation from scratch in numpy on 50,000 clustered 128-dim vectors, and print the recall/work trade-off of each.

## The theory (read this before the interview)
**Why ANN.** Exact kNN is O(n·d) per query. Classic space-partitioning trees (k-d trees, ball trees) work in low dimensions but suffer the **curse of dimensionality**: in hundreds of dimensions distances concentrate (nearest and farthest points are at similar distances), so a tree must visit most leaves and degrades to a linear scan. ANN gives up guaranteed exactness to touch only a small fraction of the data.

**Measuring quality: recall@k.** recall@k = |approx top-k ∩ exact top-k| / k, averaged over queries. Always report it *together with* a cost (latency, QPS, or fraction of vectors scanned / distance computations). One number without the other is meaningless: brute force has recall 1.0, returning random ids is instant.

**IVF (inverted file index).**
- Training: run k-means on (a sample of) the data to get `nlist` centroids, the *coarse quantiser*. Each vector is stored in the inverted list of its nearest centroid.
- Query: find the `nprobe` nearest centroids, scan only their lists exactly.
- Dials: larger `nprobe` = higher recall, more work. `nlist` sets cell size; the FAISS guideline is on the order of 4√n to 16√n cells (√50,000 ≈ 224; we use 256).
- The **border problem**: a true neighbour can sit just across a cell boundary from the query, in a cell that `nprobe=1` never opens. That is why recall is below 1 at small `nprobe`.
- Needs a training step on representative data; if the data distribution drifts, cells become unbalanced and you retrain. Inserts are cheap (assign to nearest centroid). Memory overhead is small (centroids + ids).

**HNSW (hierarchical navigable small world).**
- Build a proximity graph: each node links to up to `M` close neighbours (often 2·M on the bottom layer). Edges are chosen with a diversity heuristic so they point in different directions, not just to the M closest.
- *Small world*: a mix of short local edges and some long-range edges lets a greedy walk cross the space in few hops. A plain kNN graph lacks long links and the greedy walk gets stuck in a local optimum (our run shows exactly this).
- *Hierarchy*: each node gets a random maximum level with exponentially decaying probability, like a skip list. Upper layers are sparse and contain long-range links; layer 0 contains everyone.
- Search: start at the entry point on the top layer, greedily descend layer by layer to get close, then on layer 0 run a beam (best-first) search keeping the `efSearch` best candidates. `efSearch` must be ≥ k; larger means higher recall and slower. Search cost grows roughly logarithmically with n.
- Build: `efConstruction` is the beam width used while inserting; larger gives a better graph and a slower build.
- Trade-offs: excellent recall/latency, supports incremental inserts without training, but memory overhead for the links (about 2·M neighbour ids per node on the bottom layer, plus a little for the upper layers, on top of the vectors), the whole graph usually lives in RAM, and **deletes are hard** (removing a node can break paths), so engines tombstone and repair or rebuild.

**PQ (product quantisation).**
- Split each d-dim vector into m sub-vectors (here 128 = 16 × 8). For each sub-space train k-means with 256 centroids, a *codebook*. Store each sub-vector as the id of its nearest centroid: 256 centroids fit in 1 byte, so a vector becomes m bytes (512 → 16 bytes, 32x smaller).
- **Asymmetric distance computation (ADC)**: the query is *not* quantised. For each sub-space precompute a table of query·centroid for all 256 centroids (m × 256 numbers); the approximate score of any database vector is then m table lookups and adds, no floating-point dot products over d dims.
- Cost: quantisation error lowers recall (0.362 alone in our run), so in practice PQ produces a shortlist that is **re-ranked with full or less-compressed vectors** (0.952 after re-ranking 100 here). OPQ adds a learned rotation first to spread information evenly across sub-spaces.

**Other compression.** Scalar quantisation (SQ): each float becomes an int8 (or int4) with a per-dimension range, 4x smaller, small recall loss. Binary quantisation: keep only the sign bit per dimension, 32x smaller, compare with Hamming distance, then re-score the shortlist with fuller vectors; works best with high-dimensional embeddings.

**Combinations used in practice.** IVF-PQ (FAISS `IndexIVFPQ`): IVF cells, PQ-encoded residuals (vector minus its centroid) inside each cell, optional re-rank. HNSW + SQ/PQ, or HNSW over float16 vectors, to cut RAM. **DiskANN** (Vamana graph): keeps the graph and full vectors on SSD and only compressed PQ codes in RAM, aimed at billion-scale data on one machine.

**Which index when.**
| Situation | Index | Memory | Build | Updates | Recall |
|---|---|---|---|---|---|
| < ~1M vectors, or need exact | Flat | n·d·4 bytes | none | trivial | 1.0 |
| Up to tens of millions, RAM available, low latency | HNSW | vectors + links | slow-ish, no training | inserts easy, deletes hard | very high, tune `efSearch` |
| Large, memory-limited, batch-built | IVF-PQ | a few bytes to tens of bytes per vector | k-means training | append easy, retrain on drift | medium; re-rank to recover |
| Medium-large, simple, filter-heavy | IVF-Flat | ≈ flat | k-means training | append easy | high with enough `nprobe` |
| Billion-scale, one machine | DiskANN-style | PQ codes in RAM, rest on SSD | long | limited | high |

## Run it
```bash
cd 14_rag_and_retrieval/05_ann_ivf_and_hnsw
python main.py       # numpy only; about 15 s in this run (docstring says 20-40 s); timings vary
```

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `x[:, None, :] - cent[None]` | broadcasting: every vector minus every centroid -> `(n, k, d)` | used for L2 in PQ k-means |
| `a if cond else b` with `\` | conditional expression continued on the next line | dot-product vs L2 assignment |
| `np.where(labels == c)[0]` | ids of vectors in cell c | the inverted list |
| `np.concatenate([...])` | join the lists of the probed cells | candidate ids |
| `set(f[:k]) & set(t)` | set intersection = how many true neighbours were found | recall@k |
| `np.fill_diagonal(sims, -np.inf)` | stop a node choosing itself as a neighbour | building the kNN graph |
| `list(map(int, row))` | numpy ints -> plain Python ints | graph adjacency lists |
| `cand.sort(reverse=True); cand.pop(0)` | take the best frontier node (real code uses a heap) | best-first search |
| `tables[np.arange(SUB), codes]` | advanced indexing: one lookup per sub-space for every vector | PQ scoring, `(N, SUB)` |
| `np.uint8` | 1-byte unsigned integer, 0..255 | one PQ code |
| `f"{x:>10.3f}"` | right-align in 10 characters, 3 decimals | the result tables |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `np.random.default_rng(0)` | numpy | seeded generator: same data every run | seed | Generator |
| `unit(x)` | local helper | divide by L2 norm | – | unit vectors |
| `kmeans(x, k, iters, seed)` | local helper | Lloyd's k-means (dot assignment, or L2 for small sub-vectors) | `k`, `iters` | centroids, labels |
| `recall_at_k(found)` | local helper | mean overlap with exact top-10 | `k=10` | float |
| `graph_search(graph, q, ef, entry)` | local helper | best-first walk keeping `ef` best | `ef`, `entry` | top-10 ids, distance count |
| `np.argsort(..., axis=1)` | numpy | exact top-10 per query (ground truth) | `axis` | `(NQ, K)` ints |
| `np.argpartition(-sims, M, axis=1)` | numpy | M nearest neighbours per node | kth, axis | indices |
| `np.argmax` / `np.argmin` | numpy | nearest centroid by dot / by L2 | `axis=1` | labels |
| `time.perf_counter()` | standard library | timing | – | seconds |

## Execution trace (real run)
- `data (50000, 128), 200 queries | exact brute force: 3.83 ms/query, scans 100% of vectors`
- **A) IVF**, 256 cells trained in 0.6 s, avg 195 vectors per cell:
  ```
  nprobe  recall@10  % scanned  ms/query
       1      0.919       0.5%      0.06
       2      0.986       0.8%      0.10
       4      1.000       1.6%      0.17
       8      1.000       3.2%      0.33
      16      1.000       6.3%      0.63
      32      1.000      12.6%      1.28
  ```
  With `nprobe=4` IVF found every true neighbour while scanning 1.6% of the vectors, about 20x faster than brute force here. Recall is so high because our data has 200 clean topic clusters; real data is messier.
- **B) Graph**, 10,000 nodes, 16 neighbours each, built in 1.1 s:
  ```
  graph                               ef  recall@10  distances  % of n
  directed kNN only                   10      0.045         74    0.7%
  directed kNN only                   50      0.179        229    2.3%
  directed kNN only                  100      0.354        421    4.2%
  NSW: two-way + 4 long-range edges   10      0.356        217    2.2%
  NSW: two-way + 4 long-range edges   50      0.911        561    5.6%
  NSW: two-way + 4 long-range edges  100      0.978       1038   10.4%
  ```
  The plain kNN graph gets trapped (0.354 even at ef=100). Two-way edges plus 4 random long-range edges per node lift recall to 0.978 while computing distances for 10.4% of the nodes. `ef` is the recall/speed dial.
- **C) PQ**, 16 codebooks × 256 centroids trained in 8.4 s: **512 bytes -> 16 bytes per vector (32x smaller)**. recall@10 **PQ only: 0.362** | **PQ shortlist of 100 + exact re-rank: 0.952**.
- Whole script: 15.2 s wall time in this run, most of it PQ training. Timings vary by run and machine; recalls repeat exactly because data, k-means and graph entry points all come from seeded generators.

## Experiments (change ONE thing)
1. `NLIST = 64` and `1024`: fewer, bigger cells give higher recall at `nprobe=1` but scan more; more cells scan less but suffer more border misses.
2. `M = 4` and `32` in section B: fewer edges, worse recall; more edges, better recall and more memory.
3. Uniform data `data = unit(rng.normal(size=(N, D)))`: no cluster structure, so IVF recall at small `nprobe` drops sharply.
4. Long-range edges `4` -> `0`, `1`, `8` in `nsw`: with 0 you are left with a two-way kNN graph; watch how recall depends on these shortcuts.
5. PQ shortlist `100` -> `20` and `500`: re-ranking a longer shortlist recovers more recall at more cost.

## Debugging practice
BREAK IT from `main.py`: in `graph_search` remove the `if nb in visited: continue` check. Confirmed in a temporary copy (run through section B only):
```
graph                               ef  recall@10  distances  % of n
directed kNN only                   10      0.001        207    2.1%
directed kNN only                  100      0.003       1744   17.4%
NSW: two-way + 4 long-range edges   50      0.015       1460   14.6%
NSW: two-way + 4 long-range edges  100      0.016       2855   28.6%
```
No crash and no infinite loop (the stopping rule still fires), but recall collapses from 0.978 to 0.016 and the work almost triples (2855 vs 1038 distances at ef=100). Why: the same good nodes are reached again through other neighbours, re-scored, and appended to `best` as duplicate copies, so `best` fills up with repeats of a few nodes and the returned "top-10" contains very few distinct ids. Lesson: every graph search needs a visited set; if recall is near zero while work goes up, suspect duplicates.

## Rebuild checklist
- [ ] exact ground truth with brute force; `recall_at_k` = mean overlap / k
- [ ] IVF: k-means `nlist` centroids -> inverted lists -> probe `nprobe` nearest cells -> exact scan of candidates
- [ ] graph: M nearest neighbours per node, make edges two-way, add a few long-range edges
- [ ] best-first search with a `visited` set, a frontier and the `ef` best; stop when the frontier cannot improve
- [ ] PQ: m sub-spaces × 256-centroid codebooks -> `uint8` codes -> per-query lookup tables -> sum
- [ ] re-rank the PQ shortlist with full vectors

## The new problem (cliffhanger)
Fast or exact, every method here still ranks by embedding similarity. In step 04 the true answer to "Do my reward points run out?" came *second*, and embeddings are weak at exact strings such as product codes and rare jargon. A perfect ANN index cannot fix a ranking signal that misses the keyword. Next: [step 06, BM25 and hybrid search](../06_bm25_and_hybrid_search/WALKTHROUGH.md).

> **Say it aloud (interview box):** Approximate nearest-neighbour indexes avoid the linear scan by exploiting structure, and I always judge them by recall@k against brute force together with latency or work done. IVF clusters vectors with k-means and probes only the nprobe nearest cells, HNSW walks a layered small-world graph greedily with efSearch controlling recall, and product quantisation compresses each vector to a few bytes using per-subspace codebooks and lookup tables. In production I combine them, for example IVF-PQ or HNSW with scalar quantisation, and re-rank the shortlist with fuller vectors to win back recall.
