"""
GOAL:  Build the nouns of a RAG system (Document, Element, Chunk, Embedding, SearchHit, Citation, Answer) as
       immutable value objects; see content-hash ids, JSON round-trips, validation, and dataclass vs pydantic.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 20_llm_system_lld/02_domain_model/main.py
EXPECTED OUTPUT (shortened):
    doc_id=... content_hash=...   (doc_id stable, content_hash changes with the text)
    chunk ids with different position are equal: True
    round trip equal: True
    dataclass accepts position='3' silently: True | pydantic coerces to int 3
NEEDS: ragkit (stdlib + numpy + pydantic v2). No models, no network.
"""
import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pydantic import ValidationError

from ragkit.domain import Answer, Chunk, Citation, Document, Element, ElementKind, Embedding, SearchHit
from ragkit.ids import content_hash, new_id
from ragkit.schemas import ChunkModel

# ---- 1. Ids: content-hash (deterministic) vs uuid (random) -----------------------------------------------
v1 = Document.from_text("refunds.md", "# Refunds\nWithin 9 days.", team="support")
v2 = Document.from_text("refunds.md", "# Refunds\nWithin 14 days.", team="support")
print(f"doc_id v1={v1.doc_id} v2={v2.doc_id} same source -> same id: {v1.doc_id == v2.doc_id}")
print(f"content_hash v1={v1.content_hash} v2={v2.content_hash} text changed -> hash changed: {v1.content_hash != v2.content_hash}")
r1, r2 = new_id("trace"), new_id("trace")                           # uuid4-based: different every run
print("random ids (events, traces): prefix", r1[:6], "| length", len(r1), "| two calls differ:", r1 != r2)

# ---- 2. Chunk identity ignores position, so inserting a paragraph elsewhere does not change this id -------
a = Chunk.create(v1.doc_id, "Within 9 days.", "Refund window", position=0)
b = Chunk.create(v1.doc_id, "Within 9 days.", "Refund window", position=7)
print("chunk ids with different position are equal:", a.chunk_id == b.chunk_id, "->", a.chunk_id)

# ---- 3. Immutability: frozen, slots, and read-only metadata ------------------------------------------------
def mutate_metadata():
    v1.metadata["team"] = "sales"                                   # metadata is a read-only mappingproxy


for label, action in [("assign field", lambda: setattr(a, "text", "x")), ("mutate metadata", mutate_metadata)]:
    try:
        action()
    except (FrozenInstanceError, TypeError) as e:
        print(f"{label:16s}-> {type(e).__name__}: {e}")
twin = Chunk.create(v1.doc_id, "Within 9 days.", "Refund window", position=0)
print("equal by value:", a == twin, "| hashable, set of 2 equal chunks has", len({a, twin}), "member | a == b (position differs):", a == b)

# ---- 4. Validation in __post_init__: an invalid object cannot exist ----------------------------------------
for label, make in [("empty document", lambda: Document.from_text("x.md", "   ")),
                    ("refusal with citations", lambda: Answer("no", (Citation(1, "c"),), refused=True))]:
    try:
        make()
    except ValueError as e:
        print(f"{label:23s}-> ValueError: {e}")

# ---- 5. Serialisation: to_dict / from_dict / JSON ------------------------------------------------------------
els = [Element(ElementKind.HEADING, "Refunds", 1, 0), Element(ElementKind.PARAGRAPH, "Within 9 days.", 0, 1)]
hit = SearchHit(a, 0.8731, "dense", 1)
ans = Answer("Refunds take 9 days [1].", (Citation(1, a.chunk_id),), context_chunk_ids=(a.chunk_id,), index_version="v-1a2b")
text = ans.to_json()
print("answer as JSON:", text[:110], "...")
print("round trip equal:", Answer.from_json(text) == ans, "| hit round trip:", SearchHit.from_dict(hit.to_dict()) == hit,
      "| elements:", [Element.from_dict(e.to_dict()) for e in els] == els)
emb = Embedding.from_array(a.chunk_id, "hashing-64", __import__("numpy").arange(4, dtype="float32"))
print("embedding dim:", emb.dim, "| model recorded:", emb.model, "| as_array dtype:", emb.as_array().dtype)

# ---- 6. dataclass vs pydantic at a trust boundary ---------------------------------------------------------------
bad = Chunk("id", "doc", "text", position="3")                      # dataclass: type hints are NOT enforced
print("dataclass accepts position='3' silently:", bad.position == "3", "| type:", type(bad.position).__name__)
print("pydantic coerces:", ChunkModel(chunk_id="id", doc_id="doc", text="t", position="3").position)
try:
    ChunkModel(chunk_id="", doc_id="doc", text="t", position=-1)
except ValidationError as e:
    print("pydantic error count:", e.error_count(), "|", [(x["loc"][0], x["type"]) for x in e.errors()])
print("pydantic JSON schema keys:", sorted(ChunkModel.model_json_schema()["properties"]))
print("content_hash demo:", content_hash("ab", "c"), "!=", content_hash("a", "bc"))

# TRY IT: change the chunk text by one character and print chunk_id again: how many hex chars change?
# TRY IT: make Chunk.create hash `position` too, re-run section 2: the ids differ. What would that cost during re-indexing?
# BREAK IT: remove `slots=True` from Chunk in domain.py: section 3 still raises for `text`, but what happens to c.txt = 1?
