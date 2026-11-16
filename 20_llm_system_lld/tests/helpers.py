"""Shared builders for tests and demos."""
from ragkit.chunking import HeadingChunker
from ragkit.domain import Document
from ragkit.ingest import HashingEmbedder, InMemoryLoader, InMemoryVectorStore, IngestionPipeline, Manifest, MarkdownParser

DOC_A = Document.from_text("refunds.md", "# Refunds\n\n## Refund window\nRefunds are issued within 9 days of delivery if the seal is unbroken.\n\n## How to request\nOpen the order and choose Request refund. Agents review within 48 hours.")
DOC_B = Document.from_text("shipping.md", "# Shipping\n\n## Free shipping\nShipping is free above 799 rupees. Delivery takes 3 to 5 working days.\n\n## Tracking\nYou receive a tracking link by email.")


def make_pipeline(docs, store=None, embedder=None, chunker=None, manifest=None, **kw):
    store = store if store is not None else InMemoryVectorStore()
    return IngestionPipeline(InMemoryLoader(docs), MarkdownParser(), chunker or HeadingChunker(),
                             embedder or HashingEmbedder(64), store, manifest or Manifest(), **kw), store
