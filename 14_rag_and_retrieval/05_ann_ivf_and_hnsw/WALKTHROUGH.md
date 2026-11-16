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

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| ANN vs kNN vs exact search | approximate vs exact nearest neighbour | No | kNN is the exact answer; ANN returns almost the same neighbours much faster |
| IVF vs IVF-Flat vs IVF-PQ | inverted file | Related | IVF = cluster and search only some cells; Flat keeps full vectors in cells; PQ compresses them |
| `nlist` vs `nprobe` | cells vs cells searched | No | `nlist` = how many clusters at build time; `nprobe` = how many are scanned per query |
| `M` vs `ef` vs `efConstruction` (HNSW) | graph parameters | No | `M` = edges per node; `efConstruction` = beam width while building; `ef`/`efSearch` = beam width while querying |
| NSW vs HNSW | navigable small world, hierarchical | No | HNSW adds sparse upper layers with long-range links on top of an NSW-style graph |
| PQ vs scalar quantisation (SQ) vs binary quantisation | vector compression | No | PQ stores codebook ids per sub-vector; SQ stores each number in fewer bits (int8, float16); binary keeps 1 bit per dimension |
| Recall@k vs precision@k | accuracy of search | No | Here recall@10 = share of the true top-10 that the ANN also returned |

## Code, step by step

### 1. Docstring: three ANN ideas  (`main.py` lines 1–16)
```python
"""
GOAL:  Approximate nearest-neighbour (ANN) search from scratch, the three ideas inside FAISS / Milvus /
       Qdrant / pgvector:
         A) IVF  (inverted file)       : cluster the vectors, search only the nprobe closest clusters
         B) Graph search (NSW / HNSW)  : walk a k-nearest-neighbour graph greedily towards the query
         C) PQ   (product quantisation): compress each vector to a few bytes, compare via lookup tables
       Every method is scored by recall@10 against exact brute force: speed is only worth it if recall holds.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 14_rag_and_retrieval/05_ann_ivf_and_hnsw/main.py
EXPECTED OUTPUT (shortened, numbers depend on the machine):
    A) IVF nlist=256:  nprobe 1 -> recall@10 0.92 scanning 0.5% ... nprobe 4 -> recall 1.00 scanning 1.6%
    B) graph search: plain kNN graph -> recall stuck low; NSW with long-range edges -> recall 0.9+ scanning ~6-11% of n
    C) PQ: 512 bytes -> 16 bytes per vector (32x smaller), recall@10 ~0.4 alone, ~0.95 with exact re-rank
NEEDS: numpy only.  ~15-20 s.
"""
```
- **In simple words:** Introduces the three tricks behind every big vector database.
- **Exactly:** IVF (cluster and search only near clusters), graph search (HNSW idea) and product quantisation (compression). Each is scored by recall@10 against exact brute force. Needs only numpy, about 40 s in the real run on this machine.
- **Why it is here:** Speed is useless if the right answer is lost, so recall is measured every time.

### 2. Imports and random generator  (`main.py` lines 18–22)
```python
import time

import numpy as np

rng = np.random.default_rng(0)
```
- **In simple words:** Load tools and fix the random numbers.
- **Exactly:** `default_rng(0)` makes every run reproducible.
- **Why it is here:** Comparable runs.

### 3. unit(): scale to length 1  (`main.py` lines 25–26)
```python
def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)
```
- **In simple words:** A helper that turns vectors into unit vectors.
- **Exactly:** Divides by `np.linalg.norm(x, axis=-1, keepdims=True)`; works on one vector or a matrix.
- **Why it is here:** With unit vectors, dot product equals cosine (step 04).

### 4. Simulated clustered data  (`main.py` lines 32–35)
```python
N, D, TOPICS, NQ, K = 50_000, 128, 200, 200, 10
centers = unit(rng.normal(size=(TOPICS, D)))
data = unit(centers[rng.integers(TOPICS, size=N)] + 0.35 * rng.normal(size=(N, D)) / np.sqrt(D) * 4).astype(np.float32)
queries = unit(centers[rng.integers(TOPICS, size=NQ)] + 0.35 * rng.normal(size=(NQ, D)) / np.sqrt(D) * 4).astype(np.float32)
```
- **In simple words:** Make 50,000 fake embeddings that group into 200 topics, plus 200 queries.
- **Exactly:** `N, D, TOPICS, NQ, K = 50_000, 128, 200, 200, 10`. `centers` has shape `(200, 128)`. Each data vector is a random centre plus Gaussian noise, normalised: `data` shape `(50000, 128)` float32; `queries` shape `(200, 128)` from the same topics.
- **Why it is here:** Real embeddings cluster by topic; uniform random noise would make IVF look bad (TRY IT at the bottom).

### 5. Exact answers as ground truth  (`main.py` lines 37–40)
```python
t0 = time.perf_counter()
truth = np.argsort(-(queries @ data.T), axis=1)[:, :K]             # exact top-10 for every query: (NQ, K)
exact_ms = (time.perf_counter() - t0) / NQ * 1000
print(f"data {data.shape}, {NQ} queries | exact brute force: {exact_ms:.2f} ms/query, scans 100% of vectors")
```
- **In simple words:** Find the real top 10 for every query by checking everything.
- **Exactly:** `queries @ data.T` is `(200, 50000)` dot products; `argsort(-...)[:, :K]` gives `truth` of shape `(200, 10)`. Real: 6.17 ms per query, scanning 100 percent.
- **Why it is here:** You cannot measure recall without the true answers.

### 6. recall_at_k  (`main.py` lines 43–45)
```python
def recall_at_k(found, k=K):
    """Fraction of the true top-k that the approximate method also returned (averaged over queries)."""
    return np.mean([len(set(f[:k]) & set(t)) / k for f, t in zip(found, truth)])
```
- **In simple words:** Score how many of the true neighbours an approximate method found.
- **Exactly:** For each query, `len(set(found[:k]) & set(truth)) / k`, averaged. 1.0 = perfect.
- **Why it is here:** The single number that says whether an ANN index is safe to use.

### 7. k-means clustering  (`main.py` lines 48–59)
```python
def kmeans(x, k, iters=10, seed=0):
    """Plain Lloyd's k-means with dot-product assignment (x are unit vectors). Returns centroids, labels."""
    r = np.random.default_rng(seed)
    cent = x[r.choice(len(x), k, replace=False)].copy()
    for _ in range(iters):
        labels = np.argmax(x @ cent.T, axis=1) if x.shape[1] > 16 else \
            np.argmin(((x[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)   # small sub-vectors (PQ): use L2
        for c in range(k):
            members = x[labels == c]
            if len(members):
                cent[c] = members.mean(0)
    return cent, labels
```
- **In simple words:** Group vectors into k clusters by repeating "assign, then re-centre".
- **Exactly:** Start from k random data points. Each iteration: assign every vector to the nearest centroid (max dot product for 128-dim; L2 distance for the small 8-dim PQ sub-vectors), then set each centroid to the mean of its members (skipping empty clusters). 10 iterations by default.
- **Why it is here:** Clusters are the cells of IVF and the codebooks of PQ.

### 8. IVF: build the cells  (`main.py` lines 65–70)
```python
NLIST = 256
t0 = time.perf_counter()
centroids, labels = kmeans(data, NLIST)
centroids = unit(centroids)
lists = [np.where(labels == c)[0] for c in range(NLIST)]              # the "inverted lists": cell -> vector ids
print(f"\nA) IVF: trained {NLIST} cells in {time.perf_counter() - t0:.1f} s (avg {N // NLIST} vectors per cell)")
```
- **In simple words:** Cluster the data into 256 cells and remember which vectors are in which.
- **Exactly:** `centroids` shape `(256, 128)`, re-normalised. `lists[c]` is the array of row ids with label `c`: the "inverted list". Real: trained in 0.8 s, about 195 vectors per cell.
- **Why it is here:** This is the index; queries will only open a few lists.

### 9. IVF: search with different nprobe  (`main.py` lines 71–82)
```python
print(f"   {'nprobe':>6} {'recall@10':>10} {'% scanned':>10} {'ms/query':>9}")
for nprobe in (1, 2, 4, 8, 16, 32):
    found, scanned = [], 0
    t0 = time.perf_counter()
    for q in queries:
        cells = np.argsort(-(centroids @ q))[:nprobe]                 # nearest cells to the query
        cand = np.concatenate([lists[c] for c in cells])
        scanned += len(cand)
        found.append(cand[np.argsort(-(data[cand] @ q))[:K]])
    ms = (time.perf_counter() - t0) / NQ * 1000
    print(f"   {nprobe:>6} {recall_at_k(found):>10.3f} {100 * scanned / (NQ * N):>9.1f}% {ms:>9.2f}")
print("   nprobe is the speed/recall dial. Vectors near a cell border are missed at small nprobe.")
```
- **In simple words:** For each query, look only into the nearest few cells and see how good the result is.
- **Exactly:** `argsort(-(centroids @ q))[:nprobe]` finds the nearest cells; candidates are the concatenated lists; the top 10 are chosen among candidates by exact dot product. Real: nprobe 1 gives recall 0.919 scanning 0.5 percent; nprobe 2: 0.986 at 0.8; nprobe 4: 1.000 at 1.6 percent in 0.19 ms (versus 6.17 ms exact).
- **Why it is here:** `nprobe` is the speed versus recall dial; vectors near a cell border are missed when it is small.

### 10. Graph: smaller data and its truth  (`main.py` lines 88–90)
```python
GN, M = 10_000, 16                                                   # smaller set: building the graph is O(n^2) here
gdata = data[:GN]
gtruth = np.argsort(-(queries @ gdata.T), axis=1)[:, :K]
```
- **In simple words:** Use 10,000 vectors for the graph part because building the graph is slow here.
- **Exactly:** `GN, M = 10_000, 16`; `gdata` is the first 10,000 vectors; `gtruth` recomputes the exact top 10 on that subset.
- **Why it is here:** Recall must be measured against the same set the graph contains.

### 11. Graph: build an exact kNN graph  (`main.py` lines 91–97)
```python
t0 = time.perf_counter()
knn = np.zeros((GN, M), dtype=np.int64)
for start in range(0, GN, 2000):                                     # exact kNN graph, in blocks to save memory
    sims = gdata[start:start + 2000] @ gdata.T
    np.fill_diagonal(sims[:, start:start + 2000], -np.inf)
    knn[start:start + 2000] = np.argpartition(-sims, M, axis=1)[:, :M]
print(f"\nB) graph: {GN:,} nodes, {M} nearest neighbours each, built in {time.perf_counter() - t0:.1f} s")
```
- **In simple words:** For every vector, find its 16 nearest neighbours.
- **Exactly:** In blocks of 2000 rows: `sims` is `(2000, 10000)`; `fill_diagonal(..., -inf)` stops a vector being its own neighbour; `argpartition(-sims, M)[:, :M]` takes the M best. `knn` shape `(10000, 16)`. Real: built in 3.0 s.
- **Why it is here:** This graph is what the search walks along.

### 12. Graph: two versions, directed and NSW  (`main.py` lines 99–106)
```python
# Graph 1: plain directed kNN graph.  Graph 2 (NSW, "navigable small world"): edges made two-way, plus a few
# random LONG-RANGE edges per node, so a walk can jump between distant regions instead of getting stuck.
directed = [list(map(int, row)) for row in knn]
undirected = [set(row) for row in directed]
for i, row in enumerate(directed):
    for j in row:
        undirected[j].add(i)                                         # reverse edge j -> i
nsw = [list(nbrs) + list(map(int, rng.integers(GN, size=4))) for nbrs in undirected]
```
- **In simple words:** Make the graph two-way and add a few random long jumps.
- **Exactly:** `directed` is the kNN lists. `undirected` adds the reverse edge `j -> i`. `nsw` = undirected neighbours plus 4 random long-range edges per node (`rng.integers(GN, size=4)`).
- **Why it is here:** Long-range links are what make a graph "small world" and navigable.

### 13. Graph: greedy best-first search  (`main.py` lines 109–131)
```python
def graph_search(graph, q, ef, entry):
    """Best-first search. Returns (top-K ids, number of distance computations)."""
    visited = {entry}
    cand = [(float(gdata[entry] @ q), entry)]                         # frontier to expand
    best = list(cand)                                                 # the ef best seen so far
    computed = 1
    while cand:
        cand.sort(reverse=True)
        score, node = cand.pop(0)                                     # most promising node
        if len(best) >= ef and score < min(best)[0]:
            break                                                     # nothing on the frontier can improve `best`
        for nb in graph[node]:
            if nb in visited:
                continue
            visited.add(nb)
            s = float(gdata[nb] @ q)
            computed += 1
            if len(best) < ef or s > min(best)[0]:
                cand.append((s, nb))
                best.append((s, nb))
                if len(best) > ef:
                    best.remove(min(best))
    return [n for _, n in sorted(best, reverse=True)[:K]], computed
```
- **In simple words:** Start somewhere, keep walking to the most promising neighbour, stop when nothing looks better.
- **Exactly:** `visited` set avoids repeats; `cand` is the frontier (score, node); `best` holds the `ef` best seen. Pop the best candidate; if `best` is full and the candidate scores below the worst in `best`, stop. Otherwise score its unvisited neighbours (one dot product each, `computed += 1`) and add those that improve `best`; if `best` grows past `ef`, drop the worst. Returns the top K ids and the number of distance computations.
- **Why it is here:** `ef` is the beam width: higher means more recall and more work.

### 14. Graph: compare the two graphs  (`main.py` lines 134–143)
```python
print(f"   {'graph':<34}{'ef':>4} {'recall@10':>10} {'distances':>10} {'% of n':>7}")
for label, graph in (("directed kNN only", directed), ("NSW: two-way + 4 long-range edges", nsw)):
    for ef in (10, 50, 100):
        res = [graph_search(graph, q, ef, entry=int(rng.integers(GN))) for q in queries]
        rec = np.mean([len(set(f) & set(t)) / K for (f, _), t in zip(res, gtruth)])
        comp = np.mean([c for _, c in res])
        print(f"   {label:<34}{ef:>4} {rec:>10.3f} {comp:>10.0f} {100 * comp / GN:>6.1f}%")
print("   The plain kNN graph traps the greedy walk in the wrong neighbourhood (a local optimum).")
print("   Long-range edges make the graph 'small world'. HNSW goes further: sparse UPPER layers hold the")
print("   long-range links, the search starts at the top and descends, and ef trades recall for speed.")
```
- **In simple words:** Run both graphs at three `ef` values and print recall and cost.
- **Exactly:** Real: directed kNN only: ef 10 gives recall 0.045, ef 100 gives 0.354 (4.2 percent of vectors). NSW: ef 10 gives 0.356, ef 50 gives 0.911 (5.6 percent), ef 100 gives 0.978 (10.4 percent). Random entry points per query.
- **Why it is here:** Shows why a plain kNN graph is not enough and why HNSW adds layers.

### 15. PQ: split each vector into 16 parts  (`main.py` lines 149–150)
```python
SUB, KS = 16, 256
ds = D // SUB
```
- **In simple words:** Cut each 128-number vector into 16 pieces of 8 numbers.
- **Exactly:** `SUB, KS = 16, 256`; `ds = D // SUB = 8`.
- **Why it is here:** Small pieces can be coded with a small codebook.

### 16. PQ: train codebooks and encode every vector  (`main.py` lines 151–157)
```python
t0 = time.perf_counter()
codebooks, codes = [], np.zeros((N, SUB), dtype=np.uint8)
for s in range(SUB):
    part = data[:, s * ds:(s + 1) * ds]
    cb, _ = kmeans(part[:5_000], KS, iters=6, seed=s)              # train on a sample, like FAISS does
    codebooks.append(cb)
    codes[:, s] = np.argmin(((part[:, None, :] - cb[None]) ** 2).sum(-1), axis=1)
```
- **In simple words:** For each piece, learn 256 typical shapes and store only the number of the closest one.
- **Exactly:** For each sub-space: k-means on a sample of 5,000 (like FAISS) gives `cb` of shape `(256, 8)`. `codes[:, s]` stores the nearest centroid id (`uint8`, 1 byte). `codes` shape `(50000, 16)`. Real: trained in 24 s.
- **Why it is here:** This is the compression.

### 17. PQ: memory saved  (`main.py` lines 158–159)
```python
print(f"\nC) PQ: trained {SUB} codebooks x {KS} centroids in {time.perf_counter() - t0:.1f} s")
print(f"   memory: {data.nbytes / N:.0f} bytes -> {codes.nbytes / N:.0f} bytes per vector ({data.nbytes / codes.nbytes:.0f}x smaller)")
```
- **In simple words:** Print how much smaller the data is.
- **Exactly:** 512 bytes (128 float32) becomes 16 bytes per vector: 32 times smaller.
- **Why it is here:** Lets 10 times more vectors fit in memory.

### 18. PQ: asymmetric distance search and re-rank  (`main.py` lines 161–168)
```python
found_pq, found_rerank = [], []
for q in queries:
    # asymmetric distance: build one lookup table per sub-space (q_sub . centroid), then sum table entries
    tables = np.stack([codebooks[s] @ q[s * ds:(s + 1) * ds] for s in range(SUB)])   # (SUB, KS)
    approx = tables[np.arange(SUB), codes].sum(axis=1)                               # (N,) approximate dot products
    short = np.argsort(-approx)[:100]
    found_pq.append(short[:K])
    found_rerank.append(short[np.argsort(-(data[short] @ q))[:K]])                   # re-score 100 with full vectors
```
- **In simple words:** Score everything quickly from lookup tables, then recheck the best 100 with full vectors.
- **Exactly:** `tables` shape `(16, 256)` holds the dot product of each query piece with each centroid. `tables[np.arange(16), codes]` picks, for every vector, 16 table entries and `.sum(axis=1)` gives approximate dot products of shape `(50000,)`. The top 100 are re-scored with the true vectors.
- **Why it is here:** Compressed search is fast but lossy; re-ranking recovers accuracy.

### 19. PQ: results  (`main.py` lines 169–170)
```python
print(f"   recall@10 PQ only: {recall_at_k(found_pq):.3f} | PQ shortlist of 100 + exact re-rank: {recall_at_k(found_rerank):.3f}")
print("   Production combines them: IVF-PQ (FAISS), HNSW + PQ/SQ, or HNSW over float16 vectors.")
```
- **In simple words:** Print recall without and with re-rank.
- **Exactly:** Real: recall@10 0.362 for PQ alone, 0.952 with a shortlist of 100 and exact re-rank.
- **Why it is here:** In production this is combined with IVF or HNSW.

### 20. Exercises as comments  (`main.py` lines 172–176)
```python
# TRY IT: change NLIST to 64 and 1024. Fewer, bigger cells -> higher recall at nprobe=1 but more scanning.
# TRY IT: change M (edges per node) to 4 and 32 in B. Fewer edges -> worse recall; more edges -> more memory.
# TRY IT: use uniform random data (data = unit(rng.normal(size=(N, D)))). IVF recall drops: no cluster structure.
# TRY IT: change the 4 random long-range edges in `nsw` to 0, 1 and 8. Watch recall and the distance count.
# BREAK IT: in graph_search remove the `if nb in visited` check -> the same nodes are scored again and again.
```
- **In simple words:** Ideas to try and break.
- **Exactly:** BREAK IT: remove the `if nb in visited` check and the same nodes are scored again and again.
- **Why it is here:** Visited-set logic is what makes graph search finite and cheap.


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
