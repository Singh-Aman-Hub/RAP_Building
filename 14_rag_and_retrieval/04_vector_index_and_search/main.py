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

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import Embedder, build_chunks, chunk_for_embedding   # noqa: E402


class FlatIndex:
    """Exact nearest-neighbour index. vectors: (n, dim) float32 unit rows; meta: one dict per row."""

    def __init__(self, dim):
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.meta = []

    def upsert(self, vectors, metas):
        """Insert, or replace rows whose chunk_id already exists (documents change over time)."""
        ids = {m["chunk_id"] for m in metas}
        keep = [i for i, m in enumerate(self.meta) if m["chunk_id"] not in ids]   # drop old versions
        self.vectors = np.vstack([self.vectors[keep], vectors.astype(np.float32)])
        self.meta = [self.meta[i] for i in keep] + list(metas)

    def delete(self, doc_id):
        keep = [i for i, m in enumerate(self.meta) if m["doc_id"] != doc_id]
        self.vectors, self.meta = self.vectors[keep], [self.meta[i] for i in keep]

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

    def save(self, folder):
        folder = Path(folder)
        np.save(folder / "vectors.npy", self.vectors)                # binary float32 matrix
        (folder / "meta.json").write_text(json.dumps(self.meta))     # metadata as JSON

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        vectors = np.load(folder / "vectors.npy")
        index = cls(vectors.shape[1])
        index.vectors, index.meta = vectors, json.loads((folder / "meta.json").read_text())
        return index


# ---- 1. Build the index from the knowledge base --------------------------------------------------------
embedder = Embedder()
chunks = build_chunks()
index = FlatIndex(embedder.dim)
index.upsert(embedder.encode([chunk_for_embedding(c) for c in chunks]), chunks)
print(f"index: {len(index.meta)} vectors x {index.dim} dims")

q = embedder.encode(["Do my reward points run out?"])[0]
print("\nQ: Do my reward points run out?")
for score, m in index.search(q, k=3):
    print(f"  {score:.3f}  {m['chunk_id']}")

# ---- 2. Metadata filtering -----------------------------------------------------------------------------
print("\nsame query, filtered to doc_id=loyalty_program:")
for score, m in index.search(q, k=3, where={"doc_id": "loyalty_program"}):
    print(f"  {score:.3f}  {m['chunk_id']}")
# POST-filtering (search top-k first, filter afterwards) can return FEWER than k results, or none:
post = [r for r in index.search(q, k=3) if r[1]["doc_id"] == "brewing_guide"]
pre = index.search(q, k=3, where={"doc_id": "brewing_guide"})
print(f"filter doc_id=brewing_guide -> post-filter returns {len(post)} results, pre-filter returns {len(pre)}")

# ---- 3. Upsert and delete: the knowledge base changes -------------------------------------------------
new = dict(chunks[0], chunk_id="allergens_and_ingredients#Nuts",
           text="UPDATED: From 2027 every blend, including Winter Kahwa, is nut free.")
index.upsert(embedder.encode([chunk_for_embedding(new)]), [new])
index.delete("brewing_guide")
print(f"\nafter 1 upsert (replaced, not duplicated) and deleting brewing_guide: {len(index.meta)} vectors")

# ---- 4. Save / load round trip -------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    index.save(tmp)
    size = sum(p.stat().st_size for p in Path(tmp).iterdir())
    again = FlatIndex.load(tmp)
    same = again.search(q, k=1)[0][1]["chunk_id"] == index.search(q, k=1)[0][1]["chunk_id"]
    print(f"saved to disk: {size:,} bytes | reloaded index gives the same top hit: {same}")

# ---- 5. How brute force scales ------------------------------------------------------------------------
rng = np.random.default_rng(0)
N, D = 100_000, 384
big = rng.normal(size=(N, D)).astype(np.float32)
big /= np.linalg.norm(big, axis=1, keepdims=True)                   # unit rows, like real embeddings
query = big[123] + 0.05 * rng.normal(size=D).astype(np.float32)
query /= np.linalg.norm(query)

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
print(f"float16 storage would be {big.astype(np.float16).nbytes / 1e6:.0f} MB; "
      f"10M vectors would be {10_000_000 * D * 4 / 1e9:.1f} GB float32 and ~{t_part * 100:.0f} ms per query -> ANN (step 05)")

# ---- 6. cosine vs dot vs L2: identical ranking for unit vectors ------------------------------------------
sample = big[:2000]
cos = sample @ query / (np.linalg.norm(sample, axis=1) * np.linalg.norm(query))
dot = sample @ query
l2 = np.linalg.norm(sample - query, axis=1)                         # ||a-b||^2 = 2 - 2 cos(a,b) for unit vectors
same_rank = (np.argsort(-cos)[:10] == np.argsort(-dot)[:10]).all() and (np.argsort(-cos)[:10] == np.argsort(l2)[:10]).all()
print(f"rankings identical for cosine, dot and L2 on unit vectors: {bool(same_rank)}")
print(f"check the identity on one row: ||a-b||^2 = {l2[0] ** 2:.4f}, 2 - 2cos = {2 - 2 * cos[0]:.4f}")

# TRY IT: N = 1_000_000 (needs ~1.5 GB RAM). Is the per-query time roughly 10x?
# TRY IT: search with where={"doc_id": "product_catalog"} for "cheapest tea" and print all four results.
# TRY IT: remove the `keep` logic in upsert and upsert the same chunk twice; search returns duplicates.
# BREAK IT: skip the normalisation of `big` and compare argsort of cos and dot: the rankings now differ.
