"""Solutions for step 03: 3.1 (manifest persisted to disk), 3.2 (break: overlap), 3.3 (rebuild: the diff)."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragkit.chunking import FixedWordChunker, HeadingChunker
from ragkit.ingest import (DirectoryLoader, HashingEmbedder, InMemoryVectorStore, IngestionPipeline, Manifest,
                           MarkdownParser)

KB = Path(__file__).resolve().parents[2] / "14_rag_and_retrieval" / "data" / "kb"

# 3.1 Persist the manifest: a SECOND PROCESS (simulated by a new store + loaded manifest) would skip everything,
#     but only if the vector store is persistent too. Show both halves of that truth.
tmp = Path(tempfile.mkdtemp())
store, manifest = InMemoryVectorStore(), Manifest()
IngestionPipeline(DirectoryLoader(KB), MarkdownParser(), HeadingChunker(), HashingEmbedder(64), store, manifest).run()
manifest.save(tmp / "manifest.json")
loaded = Manifest.load(tmp / "manifest.json")
same_store = IngestionPipeline(DirectoryLoader(KB), MarkdownParser(), HeadingChunker(), HashingEmbedder(64), store, loaded).run()
fresh_store = IngestionPipeline(DirectoryLoader(KB), MarkdownParser(), HeadingChunker(), HashingEmbedder(64),
                                InMemoryVectorStore(), loaded).run()
print("3.1 loaded manifest + SAME store  ->", f"skipped={same_store.docs_skipped} embedded={same_store.chunks_embedded}")
print("    loaded manifest + EMPTY store ->", f"skipped={fresh_store.docs_skipped} embedded={fresh_store.chunks_embedded}",
      "(manifest and store must be persisted together, else the index silently stays empty)")

# 3.2 break: overlap == size
try:
    FixedWordChunker(size=40, overlap=40)
except ValueError as e:
    print("3.2 ValueError:", e)

# 3.3 rebuild the diff: wanted / existing / stale / new
existing = {"a", "b", "c"}
wanted = {"b", "c", "d"}
print("3.3 stale =", sorted(existing - wanted), "| new =", sorted(wanted - existing), "| reused =", sorted(wanted & existing))
