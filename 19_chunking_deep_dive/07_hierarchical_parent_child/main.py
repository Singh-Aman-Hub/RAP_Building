"""
GOAL:  Hierarchical retrieval: match SMALL pieces (precise), return BIG ones (context). Four variants built on one
       node store with parent/child ids: small-to-big (parent-document retrieval), sentence-window retrieval,
       auto-merging retrieval, and a RAPTOR-style recursive summary tree (mock extractive summariser).

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 19_chunking_deep_dive/07_hierarchical_parent_child/main.py
EXPECTED OUTPUT (shortened):
    A) how ids / parent links are stored (a leaf, its parent, the vector-DB payload)
    B) tree shape per level     C) auto-merging example (3 leaves -> 1 parent)
    D) table: flat 40 | small-to-big (mid) | small-to-big (section) | sentence window | auto-merge | RAPTOR
NEEDS: torch, transformers, scikit-learn (cached MiniLM).  ~25 s.
"""
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common19 import RawChunk, count_tokens, embed, evaluate, load_docs, load_step, sentence_spans, table  # noqa: E402


@dataclass
class Node:
    node_id: str                  # "<doc_id>:<level>:<n>"  stable as long as the chunker is deterministic
    doc_id: str
    level: int                    # 0 = leaf (small), 1 = mid, 2 = section (big)
    start: int
    end: int
    text: str
    parent_id: str = None         # the ONE link upward ...
    child_ids: list = field(default_factory=list)   # ... and the list downward (kept so auto-merging can count children)
    spans: list = field(default_factory=list)


def pack(doc, spans, budget):
    groups, cur = [], []
    for sp in spans:
        if cur and count_tokens(doc["text"][cur[0][0]:sp[1]]) > budget:
            groups.append(cur)
            cur = []
        cur.append(sp)
    return groups + ([cur] if cur else [])


def build_nodes(docs, section_tokens=250, mid_tokens=110, leaf_tokens=40):
    """Sections (from the step-03 structure chunker) -> mids -> leaves. Returns {node_id: Node}."""
    st = load_step("03_structure_aware_markdown")
    nodes, counter = {}, defaultdict(int)

    def new(doc, level, start, end, parent, spans=None):
        nid = f"{doc['doc_id']}:{level}:{counter[(doc['doc_id'], level)]}"
        counter[(doc["doc_id"], level)] += 1
        n = Node(nid, doc["doc_id"], level, start, end, doc["text"][start:end], parent.node_id if parent else None,
                 spans=spans or [(start, end)])
        nodes[nid] = n
        if parent:
            parent.child_ids.append(nid)
        return n

    for doc in docs:
        for sec in st.StructureChunker(section_tokens, 0, breadcrumb=False).split_document(doc):
            sn = new(doc, 2, sec.start, sec.end, None, sec.spans)
            sents = [s for s in sentence_spans(doc["text"][sec.start:sec.end], base=sec.start)]
            for mid in pack(doc, sents, mid_tokens):
                mn = new(doc, 1, mid[0][0], mid[-1][1], sn)
                for leaf in pack(doc, mid, leaf_tokens):
                    new(doc, 0, leaf[0][0], leaf[-1][1], mn)
    return nodes


def as_chunk(n, context=None):
    """Node -> RawChunk. `context` = the node whose text is SENT to the LLM (default: the node itself)."""
    c = context or n
    return RawChunk(n.doc_id, n.start, n.end, n.text, context_text=c.text, spans=list(c.spans),
                    meta={"node_id": n.node_id, "ctx_id": c.node_id})


# ---------------------------------------------------------------------------------------------------------------
# post-processors: ranked leaf ids -> the context objects that would be sent to the LLM
# ---------------------------------------------------------------------------------------------------------------
def small_to_big(leaves, nodes, up=1):
    """Replace every hit by its ancestor `up` levels higher and drop duplicates (two hits in one section = one context)."""
    def post(order):
        seen, out = set(), []
        for j in order:
            n = nodes[leaves[j].meta["node_id"]]
            for _ in range(up):
                n = nodes[n.parent_id] if n.parent_id else n
            if n.node_id not in seen:
                seen.add(n.node_id)
                out.append(as_chunk(n))
        return out
    return post


def auto_merge(leaves, nodes, top=12, ratio=0.5):
    """LlamaIndex AutoMergingRetriever idea: if more than `ratio` of a parent's children are among the top hits,
    send the PARENT instead of its fragments (repeat upward)."""
    def post(order):
        picked = [leaves[j].meta["node_id"] for j in order[:top]]          # rank order of the retrieved leaves
        rank = {nid: r for r, nid in enumerate(picked)}
        changed = True
        while changed:
            changed = False
            by_parent = defaultdict(list)
            for nid in list(rank):
                p = nodes[nid].parent_id
                if p:
                    by_parent[p].append(nid)
            for p, kids in by_parent.items():
                if len(kids) / len(nodes[p].child_ids) > ratio:
                    rank[p] = min(rank[k] for k in kids)                  # the parent inherits the best child rank
                    for k in kids:
                        rank.pop(k)
                    changed = True
                    break
        return [as_chunk(nodes[nid]) for nid in sorted(rank, key=rank.get)]
    return post


def sentence_window(sent_chunks, w=2):
    """Index single sentences; send the sentence plus w neighbours on each side (same document)."""
    def post(order):
        out = []
        for j in order:
            c = sent_chunks[j]
            lo, hi = max(0, j - w), min(len(sent_chunks), j + w + 1)
            nb = [x for x in sent_chunks[lo:hi] if x.doc_id == c.doc_id]
            out.append(RawChunk(c.doc_id, nb[0].start, nb[-1].end, c.text, context_text=" ".join(x.text for x in nb),
                                spans=[(nb[0].start, nb[-1].end)]))
        return out
    return post


# ---------------------------------------------------------------------------------------------------------------
# RAPTOR-style tree: cluster, summarise each cluster, repeat. The summariser is a plug-in (LLM in production).
# ---------------------------------------------------------------------------------------------------------------
def extractive_summary(texts, spans_list, vecs):
    """MOCK summariser: the 2 members closest to the cluster centroid, in document order. No new words are invented,
    so `spans` stay honest: a summary 'contains' only the source sentences it quotes."""
    centroid = vecs.mean(axis=0)
    best = sorted(np.argsort(-(vecs @ centroid))[:2])
    return " ".join(texts[i] for i in best), [sp for i in best for sp in spans_list[i]]   # union of the quoted spans


def build_raptor(doc, leaves, group=4):
    """Per document: level 0 = leaves; cluster (KMeans, ~`group` per cluster) -> summary nodes; repeat until 1 node."""
    from sklearn.cluster import KMeans
    level_nodes = [(c.text, c.spans, c.start, c.end) for c in leaves]
    summaries, calls = [], 0
    while len(level_nodes) > 1:
        vecs = embed([t for t, *_ in level_nodes])
        k = max(1, int(np.ceil(len(level_nodes) / group)))
        labels = KMeans(k, n_init=3, random_state=0).fit_predict(vecs) if k > 1 else np.zeros(len(level_nodes), int)
        nxt = []
        for lab in range(k):
            idx = [i for i, l in enumerate(labels) if l == lab]
            text, spans = extractive_summary([level_nodes[i][0] for i in idx], [level_nodes[i][1] for i in idx], vecs[idx])
            calls += 1
            s0, e1 = min(level_nodes[i][2] for i in idx), max(level_nodes[i][3] for i in idx)
            nxt.append((text, spans, s0, e1))
            summaries.append(RawChunk(doc["doc_id"], s0, e1, text, spans=spans, meta={"summary": True}))
        if k == 1:
            break
        level_nodes = nxt
    return summaries, calls


def main():
    docs = load_docs()
    nodes = build_nodes(docs)
    by_level = defaultdict(list)
    for n in nodes.values():
        by_level[n.level].append(n)
    leaves = [as_chunk(n) for n in by_level[0]]
    mids = [as_chunk(n) for n in by_level[1]]

    print("A) one leaf, followed up the tree (this is what you store; the docstore is just {node_id: Node})")
    leaf = by_level[0][30]
    for n in (leaf, nodes[leaf.parent_id], nodes[nodes[leaf.parent_id].parent_id]):
        print(f"   level {n.level}: {n.node_id:<28} parent={n.parent_id!s:<26} children={len(n.child_ids):>2}  {count_tokens(n.text):>3} tok  {n.text[:40]!r}")
    print("   vector-DB point payload for the leaf:", json.dumps({"id": leaf.node_id, "doc_id": leaf.doc_id, "parent_id": leaf.parent_id,
                                                                    "start": leaf.start, "end": leaf.end}))
    print("   only LEAVES are embedded; parents live in a key-value docstore and are fetched by parent_id.")

    print("\nB) tree shape over 13 documents")
    for lvl, name in ((2, "sections"), (1, "mids"), (0, "leaves")):
        t = [count_tokens(n.text) for n in by_level[lvl]]
        print(f"   level {lvl} {name:<9}: {len(t):>3} nodes, tokens min {min(t)} mean {np.mean(t):.0f} max {max(t)}")

    # ---- C) auto-merging on one question -----------------------------------------------------------------------------
    q = "How long is parental leave for the primary and for the secondary caregiver?"
    M = embed([c.text for c in leaves])
    qv = embed([q])[0]
    order = list(np.argsort(-(M @ qv))[:12])
    merged = auto_merge(leaves, nodes)(order)
    print(f"\nC) question: {q!r}")
    print(f"   top-12 leaves -> after auto-merging: {len(merged)} contexts: "
          f"{[(m.meta['ctx_id'], count_tokens(m.context_text)) for m in merged[:4]]}")

    # ---- D) the comparison ---------------------------------------------------------------------------------------------------
    sent_chunks = [RawChunk(d["doc_id"], s, e, d["text"][s:e]) for d in docs for s, e in sentence_spans(d["text"])]
    raptor, calls = [], 0
    for d in docs:
        lv = [c for c in leaves if c.doc_id == d["doc_id"]]
        s, k = build_raptor(d, lv)
        raptor += s
        calls += k
    print(f"\n   RAPTOR-style tree: {len(raptor)} summary nodes from {len(leaves)} leaves = {calls} summariser calls "
          f"(an LLM call each in production); first summary: {raptor[0].text[:80]!r}")

    rows = []

    def add(name, chunks, post=None, vec=None):
        r = evaluate(chunks, docs=docs, postprocess=post, vectors=vec)
        d = r["dense"]
        rows.append([name, len(chunks), f"{r['mean_tok']:.0f}", f"{d['hit@1']:.2f}", f"{d['hit@3']:.2f}", f"{d['mrr']:.2f}",
                     f"{r['hybrid']['hit@3']:.2f}", f"{d['ctx_tok']:.0f}"])

    add("flat leaves (40 tok)", leaves)
    add("small-to-big -> mid (110)", leaves, small_to_big(leaves, nodes, 1))
    add("small-to-big -> section (250)", leaves, small_to_big(leaves, nodes, 2))
    add("sentence window +-2", sent_chunks, sentence_window(sent_chunks, 2))
    add("auto-merging (ratio 0.5)", leaves, auto_merge(leaves, nodes))
    add("RAPTOR collapsed tree", leaves + raptor)
    print("\nD) retrieval over 13 documents, 74 questions (ctx tok@3 = tokens sent to the LLM for the top 3 contexts)")
    print(table(rows, ["strategy", "indexed", "tok/unit", "dense h@1", "dense h@3", "MRR", "hybrid h@3", "ctx tok@3"]))


if __name__ == "__main__":
    main()

# TRY IT: auto_merge(..., ratio=0.3) then 0.8: how does the number of merged contexts and ctx tok@3 change?
# TRY IT: sentence_window(sent_chunks, w=0) must equal 'flat sentences'; w=5 sends far more tokens. Check hit@3 and ctx tok.
# BREAK IT: in small_to_big, remove the `seen` de-duplication: nothing crashes, the top-3 sometimes repeats a section (ctx tok@3 329 -> 345 at section level). Silent waste, not an error.
