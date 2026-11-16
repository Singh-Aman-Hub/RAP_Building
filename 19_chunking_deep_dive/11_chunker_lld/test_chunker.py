"""pytest for chunkerlib. Uses a WORD counter so the tests are instant and need no model download.
Run:  cd 19_chunking_deep_dive/11_chunker_lld && ../../.venv/bin/python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunkerlib import (ChunkConfig, EmbeddingCache, IndexManager, available, chunk_stream, create, diff_chunks,  # noqa: E402
                        sha)

words = lambda t: len(t.split())                                  # injected token counter for tests
DOC = "# Guide\n\n## Alpha\n" + " ".join(f"alpha{i}" for i in range(60)) + ".\n\n## Beta\n" + " ".join(f"beta{i}" for i in range(60)) + ".\n"


def mk(name="recursive", **kw):
    return create(name, ChunkConfig(**{"max_tokens": 20, **kw}), words)


def test_registry_lists_and_rejects_unknown():
    assert {"token_window", "recursive", "markdown_sections"} <= set(available())
    with pytest.raises(KeyError):
        create("nope", ChunkConfig(), words)


def test_config_validation():
    with pytest.raises(ValueError):
        ChunkConfig(max_tokens=10, overlap_tokens=10)


@pytest.mark.parametrize("name", ["token_window", "recursive", "markdown_sections"])
def test_chunks_respect_max_tokens_and_offsets(name):
    chunks = mk(name).chunk("d", DOC)
    assert chunks
    for c in chunks:
        assert DOC[c.char_start:c.char_end] == c.text                 # offsets point at the exact text
        assert words(c.text) <= 20 or name == "token_window"           # (word counter != char windows: tolerance)


def test_deterministic_and_idempotent():
    a, b = mk().chunk("d", DOC), mk().chunk("d", DOC)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_ids_change_with_config_and_version():
    a = mk(max_tokens=20).chunk("d", DOC)[0].chunk_id
    assert a != mk(max_tokens=25).chunk("d", DOC)[0].chunk_id
    ch = mk()
    ch.version = "2.0.0"
    assert a != ch.chunk("d", DOC)[0].chunk_id


def test_edit_locality_only_touched_chunks_change():
    ch = mk("markdown_sections")
    old = ch.chunk("d", DOC)
    new = ch.chunk("d", DOC.replace("beta5 ", "BETA-EDIT "))
    d = diff_chunks(old, new)
    assert len(d.added) == 1 and len(d.removed) == 1 and len(d.kept) == len(old) - 1


def test_overlap_shares_text():
    chunks = mk(overlap_tokens=6).chunk("d", DOC)
    assert any(a.char_end > b.char_start for a, b in zip(chunks, chunks[1:]))


def test_markdown_breadcrumb_and_code_fence():
    text = "# T\n\n## A\n\n```bash\n# not a heading\nls\n```\n\ntext after\n"
    cs = create("markdown_sections", ChunkConfig(max_tokens=50), words).chunk("d", text)
    assert [c.section_path for c in cs] == [("T", "A")] or all(c.section_path[-1] == "A" for c in cs)
    assert cs[0].embed_text.startswith("T > A")


def test_embedding_cache_reuses_vectors_across_chunker_versions():
    emb = lambda texts: np.ones((len(texts), 4), dtype=np.float32)
    cache = EmbeddingCache("m", emb)
    mgr = IndexManager("c", cache)
    v1 = mk("recursive").chunk("d", DOC)
    mgr.build("v1", v1)
    first = cache.calls
    ch2 = mk("recursive")
    ch2.version = "1.1.0"                                              # new ids, SAME texts
    mgr.build("v2", ch2.chunk("d", DOC))
    assert cache.calls == first                                        # nothing re-embedded
    mgr.switch("v2")
    assert mgr.alias == "v2"


def test_stream_covers_whole_document():
    big = "\n\n".join(DOC for _ in range(30))
    ch = mk("recursive")
    streamed = list(chunk_stream(iter(big.splitlines(keepends=True)), ch, "d", segment_chars=800))
    covered = sum(c.char_end - c.char_start for c in streamed)
    assert covered >= 0.8 * len(big)                                   # nearly every character belongs to some chunk
    assert all(big[c.char_start:c.char_end] == c.text for c in streamed)
