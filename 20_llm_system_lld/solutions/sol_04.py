"""Solutions for step 04: 4.1 (stop-at-first vs skip-and-continue packing), 4.2 (rrf by hand), 4.3 (verifier cases)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragkit.domain import Chunk, SearchHit
from ragkit.query import CitationVerifier, PromptBuilder, WordTokenCounter, rrf

def hit(text, score=1.0):
    return SearchHit(Chunk.create("d", text), score)

# 4.1 module 14's packer STOPPED at the first chunk that did not fit; ours skips it and keeps trying smaller ones.
hits = [hit("word " * 200), hit("a short but useful fact"), hit("another short fact")]
count = WordTokenCounter().count
def stop_at_first(hits, budget, base=40):
    used, total = [], base
    for h in hits:
        cost = count(h.chunk.text) + 8
        if total + cost > budget:
            break
        used.append(h); total += cost
    return used
stopped = stop_at_first(hits, 150)
skipped = PromptBuilder(budget=150 + 0).build("q", hits).used
print(f"4.1 stop-at-first packs {len(stopped)} chunk(s); skip-and-continue packs {len(skipped)} chunk(s)")

# 4.2 RRF by hand: chunk X is rank 1 in list A and rank 3 in list B
X, Y, Z, W = hit("x"), hit("y"), hit("z"), hit("w")
fused = rrf([[X, Y, Z], [W, Y, X]])
by = {h.chunk.text: round(h.score, 5) for h in fused}
print("4.2 rrf scores:", by, "| X by hand = 1/61 + 1/63 =", round(1 / 61 + 1 / 63, 5), "| Y (rank 2 and 2) =", round(2 / 62, 5))

# 4.3 what does the verifier prove?
used = [hit("Refunds are issued within 9 days of delivery.")]
for text in ["Refunds are issued within 9 days. [1]", "Refunds are issued within 30 days. [1]", "Nothing relevant here. [1]"]:
    v = CitationVerifier().verify(text, used)
    print(f"4.3 {text!r:46s} ok={v.ok}")
print("    note: '30 days' is WRONG but still 'supported' by word overlap -> lexical support is not faithfulness")
