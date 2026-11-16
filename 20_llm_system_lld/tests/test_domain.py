import dataclasses
import json

import pytest
from pydantic import ValidationError

from ragkit.domain import Answer, Chunk, Citation, Document, Element, ElementKind, Embedding, SearchHit
from ragkit.ids import content_hash, new_id
from ragkit.schemas import ChunkModel


def test_content_hash_deterministic_and_separator_safe():
    assert content_hash("a", "b") == content_hash("a", "b")
    assert content_hash("ab", "c") != content_hash("a", "bc")


def test_new_id_unique():
    assert new_id("t") != new_id("t") and new_id("t").startswith("t-")


def test_document_frozen_and_validated():
    d = Document.from_text("a.md", "hello", team="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.text = "other"
    with pytest.raises(TypeError):
        d.metadata["team"] = "y"                         # read-only mapping
    with pytest.raises(ValueError):
        Document.from_text("b.md", "   ")
    assert Document.from_dict(d.to_dict()) == d


def test_doc_id_stable_content_hash_changes():
    a, b = Document.from_text("a.md", "v1"), Document.from_text("a.md", "v2")
    assert a.doc_id == b.doc_id and a.content_hash != b.content_hash


def test_chunk_id_ignores_position():
    a = Chunk.create("d", "same text", "S", position=0)
    b = Chunk.create("d", "same text", "S", position=7)
    assert a.chunk_id == b.chunk_id


def test_slots_block_typos():
    c = Chunk.create("d", "t")
    with pytest.raises((AttributeError, TypeError)):
        c.txt = "x"


def test_round_trips():
    c = Chunk.create("d", "text", "S", 2, k="v")
    assert Chunk.from_dict(c.to_dict()) == c
    e = Embedding("id", "m", (0.1, 0.2))
    assert Embedding.from_dict(e.to_dict()) == e and e.dim == 2
    el = Element(ElementKind.HEADING, "Hi", 2, 0)
    assert Element.from_dict(el.to_dict()) == el
    h = SearchHit(c, 0.5, "dense", 1)
    assert SearchHit.from_dict(h.to_dict()) == h
    a = Answer("x [1]", (Citation(1, c.chunk_id),), context_chunk_ids=(c.chunk_id,), index_version="v1")
    assert Answer.from_json(a.to_json()) == a
    json.loads(a.to_json())


def test_refusal_cannot_have_citations():
    with pytest.raises(ValueError):
        Answer("no", (Citation(1, "x"),), refused=True)


def test_pydantic_coerces_dataclass_does_not():
    assert ChunkModel(chunk_id="a", doc_id="b", text="t", position="3").position == 3
    with pytest.raises(ValidationError):
        ChunkModel(chunk_id="", doc_id="b", text="t")
