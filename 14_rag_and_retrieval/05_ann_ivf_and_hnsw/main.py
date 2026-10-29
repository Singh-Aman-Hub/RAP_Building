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

import time

import numpy as np

rng = np.random.default_rng(0)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------------------------------------
# Data: real embeddings are CLUSTERED (topics), not uniform noise. Simulate 200 topics in 128 dims.
# ---------------------------------------------------------------------------------------------------------
N, D, TOPICS, NQ, K = 50_000, 128, 200, 200, 10
centers = unit(rng.normal(size=(TOPICS, D)))
data = unit(centers[rng.integers(TOPICS, size=N)] + 0.35 * rng.normal(size=(N, D)) / np.sqrt(D) * 4).astype(np.float32)
queries = unit(centers[rng.integers(TOPICS, size=NQ)] + 0.35 * rng.normal(size=(NQ, D)) / np.sqrt(D) * 4).astype(np.float32)

t0 = time.perf_counter()
truth = np.argsort(-(queries @ data.T), axis=1)[:, :K]             # exact top-10 for every query: (NQ, K)
exact_ms = (time.perf_counter() - t0) / NQ * 1000
print(f"data {data.shape}, {NQ} queries | exact brute force: {exact_ms:.2f} ms/query, scans 100% of vectors")


def recall_at_k(found, k=K):
    """Fraction of the true top-k that the approximate method also returned (averaged over queries)."""
    return np.mean([len(set(f[:k]) & set(t)) / k for f, t in zip(found, truth)])


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


# =========================================================================================================
# A) IVF: k-means into nlist cells; at query time scan only the nprobe nearest cells
# =========================================================================================================
NLIST = 256
t0 = time.perf_counter()
centroids, labels = kmeans(data, NLIST)
centroids = unit(centroids)
lists = [np.where(labels == c)[0] for c in range(NLIST)]              # the "inverted lists": cell -> vector ids
print(f"\nA) IVF: trained {NLIST} cells in {time.perf_counter() - t0:.1f} s (avg {N // NLIST} vectors per cell)")
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

# =========================================================================================================
# B) Graph search (the idea inside HNSW): connect each vector to its M nearest neighbours,
#    then walk greedily from an entry point, keeping the `ef` best candidates found so far.
# =========================================================================================================
GN, M = 10_000, 16                                                   # smaller set: building the graph is O(n^2) here
gdata = data[:GN]
gtruth = np.argsort(-(queries @ gdata.T), axis=1)[:, :K]
t0 = time.perf_counter()
knn = np.zeros((GN, M), dtype=np.int64)
for start in range(0, GN, 2000):                                     # exact kNN graph, in blocks to save memory
    sims = gdata[start:start + 2000] @ gdata.T
    np.fill_diagonal(sims[:, start:start + 2000], -np.inf)
    knn[start:start + 2000] = np.argpartition(-sims, M, axis=1)[:, :M]
print(f"\nB) graph: {GN:,} nodes, {M} nearest neighbours each, built in {time.perf_counter() - t0:.1f} s")

# Graph 1: plain directed kNN graph.  Graph 2 (NSW, "navigable small world"): edges made two-way, plus a few
# random LONG-RANGE edges per node, so a walk can jump between distant regions instead of getting stuck.
directed = [list(map(int, row)) for row in knn]
undirected = [set(row) for row in directed]
for i, row in enumerate(directed):
    for j in row:
        undirected[j].add(i)                                         # reverse edge j -> i
nsw = [list(nbrs) + list(map(int, rng.integers(GN, size=4))) for nbrs in undirected]


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

# =========================================================================================================
# C) Product quantisation: split each 128-dim vector into 16 sub-vectors of 8 dims; replace each
#    sub-vector by the id (1 byte) of its nearest of 256 sub-centroids. 512 bytes -> 16 bytes.
# =========================================================================================================
SUB, KS = 16, 256
ds = D // SUB
t0 = time.perf_counter()
codebooks, codes = [], np.zeros((N, SUB), dtype=np.uint8)
for s in range(SUB):
    part = data[:, s * ds:(s + 1) * ds]
    cb, _ = kmeans(part[:5_000], KS, iters=6, seed=s)              # train on a sample, like FAISS does
    codebooks.append(cb)
    codes[:, s] = np.argmin(((part[:, None, :] - cb[None]) ** 2).sum(-1), axis=1)
print(f"\nC) PQ: trained {SUB} codebooks x {KS} centroids in {time.perf_counter() - t0:.1f} s")
print(f"   memory: {data.nbytes / N:.0f} bytes -> {codes.nbytes / N:.0f} bytes per vector ({data.nbytes / codes.nbytes:.0f}x smaller)")

found_pq, found_rerank = [], []
for q in queries:
    # asymmetric distance: build one lookup table per sub-space (q_sub . centroid), then sum table entries
    tables = np.stack([codebooks[s] @ q[s * ds:(s + 1) * ds] for s in range(SUB)])   # (SUB, KS)
    approx = tables[np.arange(SUB), codes].sum(axis=1)                               # (N,) approximate dot products
    short = np.argsort(-approx)[:100]
    found_pq.append(short[:K])
    found_rerank.append(short[np.argsort(-(data[short] @ q))[:K]])                   # re-score 100 with full vectors
print(f"   recall@10 PQ only: {recall_at_k(found_pq):.3f} | PQ shortlist of 100 + exact re-rank: {recall_at_k(found_rerank):.3f}")
print("   Production combines them: IVF-PQ (FAISS), HNSW + PQ/SQ, or HNSW over float16 vectors.")

# TRY IT: change NLIST to 64 and 1024. Fewer, bigger cells -> higher recall at nprobe=1 but more scanning.
# TRY IT: change M (edges per node) to 4 and 32 in B. Fewer edges -> worse recall; more edges -> more memory.
# TRY IT: use uniform random data (data = unit(rng.normal(size=(N, D)))). IVF recall drops: no cluster structure.
# TRY IT: change the 4 random long-range edges in `nsw` to 0, 1 and 8. Watch recall and the distance count.
# BREAK IT: in graph_search remove the `if nb in visited` check -> the same nodes are scored again and again.
