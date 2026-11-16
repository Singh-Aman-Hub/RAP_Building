import numpy as np
import pytest

from ragkit.chunking import FixedWordChunker, HeadingChunker, SentenceChunker
from ragkit.domain import Document
from ragkit.ingest import HashingEmbedder, InMemoryLoader, InMemoryVectorStore, MarkdownParser, Manifest
from ragkit.protocols import Chunker, Embedder, Loader, Parser, VectorStore
from helpers import DOC_A, DOC_B, make_pipeline


class CountingEmbedder(HashingEmbedder):
    def __init__(self, dim=64):
        super().__init__(dim)
        self.calls, self.texts = 0, 0

    def embed(self, texts):
        self.calls += 1
        self.texts += len(texts)
        return super().embed(texts)


def test_protocols_are_structural():
    assert isinstance(InMemoryLoader([]), Loader) and isinstance(MarkdownParser(), Parser)
    assert isinstance(HeadingChunker(), Chunker) and isinstance(HashingEmbedder(), Embedder)
    assert isinstance(InMemoryVectorStore(), VectorStore)


def test_hashing_embedder_unit_norm_and_deterministic():
    e = HashingEmbedder(64)
    v = e.embed(["refund window", "refund window"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0) and np.allclose(v[0], v[1])


def test_parser_finds_headings():
    els = MarkdownParser().parse(DOC_A)
    assert [e.text for e in els if e.kind.value == "heading"] == ["Refunds", "Refund window", "How to request"]


def test_heading_chunker_one_per_section():
    chunks = HeadingChunker().chunk(DOC_A, MarkdownParser().parse(DOC_A))
    assert [c.section for c in chunks] == ["Refund window", "How to request"]


def test_other_chunkers_and_validation():
    els = MarkdownParser().parse(DOC_B)
    assert len(FixedWordChunker(8, 2).chunk(DOC_B, els)) > 1
    assert len(SentenceChunker(8).chunk(DOC_B, els)) >= 2
    with pytest.raises(ValueError):
        FixedWordChunker(5, 5)


def test_idempotent_second_run_does_nothing():
    emb = CountingEmbedder()
    p, store = make_pipeline([DOC_A, DOC_B], embedder=emb)
    r1 = p.run()
    n = store.count()
    assert r1.docs_added == 2 and r1.chunks_embedded == n
    calls = emb.calls
    r2 = p.run()
    assert r2.docs_skipped == 2 and emb.calls == calls and store.count() == n


def test_incremental_only_changed_doc_embedded():
    emb = CountingEmbedder()
    manifest, store = Manifest(), InMemoryVectorStore()
    make_pipeline([DOC_A, DOC_B], store=store, embedder=emb, manifest=manifest)[0].run()
    changed = Document.from_text("refunds.md", DOC_A.text.replace("9 days", "14 days"))
    p2, _ = make_pipeline([changed, DOC_B], store=store, embedder=emb, manifest=manifest)
    r = p2.run()
    assert r.docs_updated == 1 and r.docs_skipped == 1
    assert r.chunks_embedded == 1 and r.chunks_reused == 1 and r.chunks_deleted == 1


def test_deleted_document_removed():
    manifest, store = Manifest(), InMemoryVectorStore()
    make_pipeline([DOC_A, DOC_B], store=store, manifest=manifest)[0].run()
    r = make_pipeline([DOC_A], store=store, manifest=manifest)[0].run()
    assert r.docs_deleted == 1 and all(c.doc_id == DOC_A.doc_id for c in store.all_chunks())


def test_dead_letter_and_batch_continues():
    class Boom(HashingEmbedder):
        def embed(self, texts):
            if any("tracking" in t.lower() for t in texts):
                raise RuntimeError("embedder exploded")
            return super().embed(texts)
    p, store = make_pipeline([DOC_A, DOC_B], embedder=Boom(64))
    r = p.run()
    assert r.docs_added == 1 and len(r.dead_letters) == 1
    assert r.dead_letters[0].source == "shipping.md" and "exploded" in r.dead_letters[0].error
    assert r.dead_letters[0].stage == "embed"
    assert DOC_B.doc_id not in p.manifest.docs                 # not marked done -> retried next run


def test_index_version_change_forces_rebuild():
    manifest, store = Manifest(), InMemoryVectorStore()
    p1, _ = make_pipeline([DOC_A], store=store, manifest=manifest)
    p1.run()
    p2, _ = make_pipeline([DOC_A], store=store, manifest=manifest, embedder=HashingEmbedder(32))
    assert p2.index_version != p1.index_version
    r = p2.run()
    assert r.rebuilt and r.docs_added == 1


def test_batching():
    emb = CountingEmbedder()
    long = Document.from_text("long.md", "# T\n\n" + "\n\n".join(f"## S{i}\nthis is section number {i} text." for i in range(10)))
    p, _ = make_pipeline([long], embedder=emb, batch_size=4)
    p.run()
    assert emb.calls == 3 and emb.texts == 10


def test_store_filter_and_search():
    p, store = make_pipeline([DOC_A, DOC_B])
    p.run()
    e = HashingEmbedder(64)
    hits = store.search(e.embed(["refunds issued within 9 days of delivery"])[0], 3)
    assert hits[0].chunk.section == "Refund window" and hits[0].rank == 1
    only_b = store.search(e.embed(["refund"])[0], 5, where={"doc_id": DOC_B.doc_id})
    assert {h.chunk.doc_id for h in only_b} == {DOC_B.doc_id}
