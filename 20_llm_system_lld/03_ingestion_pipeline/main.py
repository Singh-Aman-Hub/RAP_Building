"""
GOAL:  Loader -> Parser -> Chunker -> Embedder -> VectorStore as five small interfaces, then the properties that make
       an indexer production-grade: idempotent upsert, incremental re-index, batching, dead-letter, index versioning.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 20_llm_system_lld/03_ingestion_pipeline/main.py
EXPECTED OUTPUT (shortened):
    run 1 (cold)      : added=9 ... chunks embedded=36
    run 2 (no change) : skipped=9 ... embedded=0
    run 3 (1 doc edit): updated=1 skipped=8 | embedded=1 reused=3 deleted=1
    dead letter: ... run 5 (new embedder dim): REBUILT
NEEDS: ragkit + module 14's data/kb markdown files. HashingEmbedder is offline and instant (no model download).
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragkit.chunking import HeadingChunker
from ragkit.ids import content_hash
from ragkit.ingest import DirectoryLoader, HashingEmbedder, InMemoryVectorStore, IngestionPipeline, Manifest, MarkdownParser

KB = Path(__file__).resolve().parents[2] / "14_rag_and_retrieval" / "data" / "kb"
work = Path(tempfile.mkdtemp())                                     # a scratch copy of the KB we are allowed to edit
for f in KB.glob("*.md"):
    shutil.copy(f, work / f.name)


class CountingEmbedder(HashingEmbedder):
    """Decorates HashingEmbedder to count how many model calls (batches) and texts we pay for."""
    def __init__(self, dim=128):
        super().__init__(dim)
        self.calls = self.texts = 0

    def embed(self, texts):
        self.calls += 1
        self.texts += len(texts)
        return super().embed(texts)


store, manifest, emb = InMemoryVectorStore(), Manifest(), CountingEmbedder(128)


def pipeline(embedder=emb, batch_size=16):
    return IngestionPipeline(DirectoryLoader(work), MarkdownParser(), HeadingChunker(), embedder, store, manifest, batch_size)


p = pipeline()
print("index_version:", p.index_version, "| chunker:", HeadingChunker().name)
print("run 1 (cold)      :", p.run().summary(), f"| embed calls={emb.calls} (batch_size=16)")
print("run 2 (no change) :", pipeline().run().summary(), f"| embed calls still {emb.calls}")

# ---- edit ONE document: change the refund window from 9 to 14 days ------------------------------
f = work / "refunds_and_returns.md"
f.write_text(f.read_text().replace("within 9 days", "within 14 days"))
print("run 3 (1 doc edit):", pipeline().run().summary())

# ---- delete a document from the source --------------------------------------------------------------------------
(work / "wholesale_and_b2b.md").unlink()
print("run 4 (doc removed):", pipeline().run().summary(), f"| store now holds {store.count()} chunks")


# ---- a document that crashes the embedder goes to the dead-letter list; the rest continue -------------------------
class PickyEmbedder(CountingEmbedder):
    def embed(self, texts):
        if any("gift" in t.lower() for t in texts):
            raise RuntimeError("embedding service rejected input")
        return super().embed(texts)


(work / "gifts.md").write_text("# Gifts\n\n## Gift wrap\nWe can gift wrap any order for 50 rupees.\n")
dead = pipeline(PickyEmbedder(128))
rep = dead.run()
print("dead letter:", rep.summary())
print("   ->", [(d.source, d.stage, d.error) for d in rep.dead_letters])
print("   gifts.md is NOT in the manifest (so the next run retries it):", content_hash("gifts.md") not in manifest.docs)

# ---- changing what a vector MEANS (embedder dim) invalidates the whole index ---------------------------------------
new = pipeline(CountingEmbedder(256))
print("run 5 (new embedder dim=256): version", new.index_version, "->", new.run().summary())
print("sample stored chunk:", store.all_chunks()[0].to_dict()["chunk_id"], "|", store.all_chunks()[0].section)

# TRY IT: HeadingChunker(max_words=20): how many more chunks? Is run 2 still a no-op afterwards? (the version changes)
# TRY IT: pipeline(batch_size=1) on a cold index: how many embed calls now?
# BREAK IT: make Chunk.create hash the position too (domain.py) and redo run 3: how many chunks get re-embedded?
