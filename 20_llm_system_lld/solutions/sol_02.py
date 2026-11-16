"""Solutions for step 02: 2.1 (modify: position in the id), 2.2 (break: mutate), 2.3 (rebuild: Answer round trip)."""
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragkit.domain import Answer, Chunk, Citation
from ragkit.ids import content_hash

# 2.1 How many chunks must be re-embedded after inserting ONE paragraph at the top, with and without position in the id?
paras = [f"paragraph number {i} with some text" for i in range(20)]
def ids(ps, with_pos):
    return {content_hash("doc", p, str(i) if with_pos else "") for i, p in enumerate(ps)}
before_c, after_c = ids(paras, False), ids(["NEW opening paragraph"] + paras, False)
before_p, after_p = ids(paras, True), ids(["NEW opening paragraph"] + paras, True)
print(f"2.1 20 paragraphs, 1 inserted at top: content-only ids re-embed {len(after_c - before_c)}; position in id re-embed {len(after_p - before_p)}")

# 2.2 mutate: the right way to 'change' a frozen object is dataclasses.replace -- but the chunk_id then goes STALE
c = Chunk.create("doc", "old text", "S")
try:
    c.text = "new text"
except dataclasses.FrozenInstanceError as e:
    print("2.2 frozen:", e)
c2 = dataclasses.replace(c, text="new text")
print("    replace keeps the OLD id:", c2.chunk_id == c.chunk_id, "| Chunk.create gives a NEW id:", Chunk.create("doc", "new text", "S").chunk_id != c.chunk_id)

# 2.3 Answer round trip, including a refusal and nested citations
a = Answer("Refunds take 9 days [1].", (Citation(1, "abc123"),), context_chunk_ids=("abc123",), index_version="v1", meta={"cache": "miss"})
r = Answer("I don't know.", refused=True, refusal_reason="no_hits")
for x in (a, r):
    assert Answer.from_json(x.to_json()) == x
print("2.3 round trips OK; JSON:", a.to_json()[:90], "...")
