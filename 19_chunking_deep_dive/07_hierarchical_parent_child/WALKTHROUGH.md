# Step 07 — Hierarchical and parent-child retrieval

## The story so far
[Step 06](../06_contextual_retrieval/WALKTHROUGH.md) wrote the missing context into the chunk itself, paid for with one LLM call per chunk. There is a cheaper way to give the LLM context, with no extra model calls: keep the chunk *small for search* but hand the LLM something *bigger*. Back to the [module README](../README.md). The module-14 single example of this idea is in [14/10 advanced RAG](../../14_rag_and_retrieval/10_advanced_rag_and_vector_dbs/WALKTHROUGH.md); this step builds four variants properly.

## The pain point
Chunk size pulls in two directions:
```
small chunk (40 tokens):  vector is focused (precise to search)         but the LLM gets a fragment
large chunk (250 tokens): the LLM gets the whole passage                but the vector is a blur, and the prompt is full of text it did not need
```
A single size cannot be both the **unit of search** and the **unit of context**.

## The spark
Decouple them. Index small pieces; store the big ones in a plain key-value store; link each small piece to its parent by id. At query time search the small pieces, then *replace* each hit by its parent (or its neighbours, or a merged group) before building the prompt. Four ways to do that:

| Variant | Index | What the LLM gets |
|---|---|---|
| **small-to-big** (parent-document retrieval) | leaves (40 tokens) | the parent at level 1 (110 tokens) or level 2 (250, a whole section) |
| **sentence window** | single sentences | the sentence plus `w` neighbours on each side |
| **auto-merging** | leaves | if more than `ratio` of a parent's children were hit, the parent replaces them |
| **RAPTOR-style tree** | leaves *and* summaries of clusters of leaves | whichever nodes match best: detail or the big picture |

## Goal
Build one node store with parent/child ids, the four retrievers on top of it, and compare them on the same 74 questions: quality *and* tokens sent to the LLM.

## The theory (read this before the interview)
```
SECTION (level 2, ~250 tok)  <- parent_id
   MID  (level 1, ~110 tok)  <- parent_id          Only leaves are embedded.
      LEAF (level 0, ~40 tok)  <- vector DB point   Parents live in a docstore {node_id: Node}.
```
**Node.** `node_id`, `doc_id`, `level`, character span, text, `parent_id` (one link up), `child_ids` (links down; needed so auto-merging can count how many children were hit).

**Small-to-big (parent-document retriever).** `hits = search(leaves)`, then `contexts = dedupe(parent(hit) for hit in hits)`. De-duplication matters: two hits in the same section must send that section once.

**Sentence window.** Embed one sentence per vector, send `sentence[j-w .. j+w]` (same document only). The window is pure position arithmetic; no tree.

**Auto-merging (LlamaIndex `AutoMergingRetriever`).** Take the top `k` leaves. If more than `ratio` of a parent's children are among them, the evidence is spread across that parent: replace those children with the parent, inheriting the best child's rank, and repeat upward. Logic: "if most of this section matched, the whole section is relevant".

**RAPTOR (Sarthi et al., 2024).** Recursively: embed the pieces, cluster them, *summarise* each cluster (an LLM in the paper), and treat the summaries as new pieces, until one root remains. At query time search the **collapsed tree**: leaves and summaries together in one index, so a broad question can match a summary and a narrow one a leaf. The paper clusters with UMAP + Gaussian mixtures; this step uses KMeans and an **extractive mock summariser** (the two member sentences closest to the cluster centroid, no invented words), so honesty about source spans is preserved.

**Metrics here.** `ctx tok@3` is the tokens sent to the LLM for the top 3 contexts (a cost), not just hit@k. A bigger context raises recall and raises the bill. A fair reading compares quality *per context token* (step 10).

**Status.** Small-to-big and sentence-window are *current industry practice* (LangChain `ParentDocumentRetriever`, LlamaIndex sentence-window and auto-merging). RAPTOR is *emerging*: strong on whole-document questions, more machinery to maintain (summaries must be rebuilt when documents change). Always store `parent_id` so you can switch strategies without re-chunking.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| small-to-big | parent-document retrieval, parent-child chunking, "retrieve small, return big", `ParentDocumentRetriever` | yes | Search leaves, send ancestors |
| sentence-window retrieval | window retrieval, `SentenceWindowNodeParser`, context window expansion | near | Neighbours by position, not by tree |
| auto-merging retrieval | hierarchical retrieval, `AutoMergingRetriever`, `HierarchicalNodeParser` | near | Merge only when enough children matched |
| RAPTOR | recursive abstractive processing for tree-organised retrieval, summary tree, collapsed tree | no | Adds LLM summary nodes that *new text* the leaves do not contain |
| docstore | key-value store of parents, parent store | yes | Holds big nodes by id; not searched |
| vector store payload | metadata, point payload | near | Per-vector fields incl. `parent_id` for the lookup |
| leaf / child / parent | level 0 / level 1+ nodes | yes | Position in the tree |

## Run it
```bash
cd 19_chunking_deep_dive/07_hierarchical_parent_child
python main.py
# from the repo root:
HF_HUB_OFFLINE=1 .venv/bin/python 19_chunking_deep_dive/07_hierarchical_parent_child/main.py
```
About 25 to 30 s of CPU time. Needs `torch`, `transformers`, `scikit-learn` and the cached MiniLM; no downloads.

## Code, step by step

### 1. Header docstring and imports  (`main.py` lines 1–23)

```python
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
```

- **In simple words:** the usual helpers, `dataclasses` for the node, `defaultdict` for grouping, `json` to print a vector-DB payload.
- **Exactly:** `asdict` is imported but only `dataclass` and `field` are used in this file; `load_step` brings in step 03's `StructureChunker`.
- **Why it is here:** the tree is built on structure-aware sections ([step 03](../03_structure_aware_markdown/WALKTHROUGH.md)), not on a fresh splitter.

### 2. `Node`: one record per level  (`main.py` lines 26–36)

```python
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
```

- **In simple words:** a node knows its level, its span, its text, one parent and its children.
- **Exactly:** `node_id = "<doc_id>:<level>:<n>"` is stable as long as the chunker is deterministic (step 11 replaces this by content hashes). `child_ids` is kept so auto-merging can ask "how many of this parent's children were hit?". `spans` lists the source regions contained.
- **Why it is here:** one dataclass is the whole data model of the hierarchy.

### 3. `pack`: greedy grouping by tokens  (`main.py` lines 39–46)

```python
def pack(doc, spans, budget):
    groups, cur = [], []
    for sp in spans:
        if cur and count_tokens(doc["text"][cur[0][0]:sp[1]]) > budget:
            groups.append(cur)
            cur = []
        cur.append(sp)
    return groups + ([cur] if cur else [])
```

- **In simple words:** the same sentence packer as before.
- **Exactly:** groups consecutive spans while the token count of the joined text stays within `budget`.
- **Why it is here:** used twice, to make mids from a section's sentences and leaves from a mid's sentences.

### 4. `build_nodes`: sections to mids to leaves  (`main.py` lines 49–72)

```python
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
```

- **In simple words:** take the structure chunker's sections (up to 250 tokens), pack each section's sentences into mids of at most 110 tokens, and each mid's sentences into leaves of at most 40 tokens, linking every child to its parent.
- **Exactly:** the nested function `new(...)` allocates the id from a per-(document, level) counter, creates the `Node`, stores it in the dict `nodes`, and appends the id to the parent's `child_ids`. `StructureChunker(section_tokens, 0, breadcrumb=False)` yields sections without a breadcrumb because here the section text is what the LLM will read. `sentence_spans(..., base=sec.start)` keeps offsets in the document's frame. Returns `{node_id: Node}`.
- **Why it is here:** levels are *nested* (a leaf lies within one mid, a mid within one section), which is what makes parent lookup and merging well-defined.

### 5. `as_chunk`: node to `RawChunk`  (`main.py` lines 75–79)

```python
def as_chunk(n, context=None):
    """Node -> RawChunk. `context` = the node whose text is SENT to the LLM (default: the node itself)."""
    c = context or n
    return RawChunk(n.doc_id, n.start, n.end, n.text, context_text=c.text, spans=list(c.spans),
                    meta={"node_id": n.node_id, "ctx_id": c.node_id})
```

- **In simple words:** convert a node into the standard chunk record, with an optional different node as the *context to send*.
- **Exactly:** `text` is the node's own text (what is embedded when it is a leaf); `context_text` and `spans` come from the node `c` that will be sent (default: itself). `meta` records both ids.
- **Why it is here:** `RawChunk.context_text` is the field that separates "searched" from "sent" (explained in the [step 01 tour](../01_tokens_not_characters/WALKTHROUGH.md#the-shared-helpers-once-a-tour-of-common19py)); the evaluation harness uses `spans` to decide if the *sent* context covers the answer.

### 6. `small_to_big`  (`main.py` lines 82–97)

```python
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
```

- **In simple words:** for each retrieved leaf walk up `up` levels and keep each parent once.
- **Exactly:** `post(order)` takes the ranked leaf ids from the harness; `seen` is the set of already-emitted node ids, so two hits in one section produce one context; the result list keeps the rank order. This is a closure: `small_to_big` returns the function `post`.
- **Why it is here:** the postprocess hook of `evaluate` was designed for this: "ranked leaf ids to the contexts actually sent".

### 7. `auto_merge`  (`main.py` lines 100–122)

```python
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
```

- **In simple words:** among the top 12 leaves, whenever more than half of one parent's children are present, replace them by the parent, then check again one level up.
- **Exactly:** `rank` maps node id to the best (smallest) original rank. `by_parent` groups the present nodes by parent; the test `len(kids) / len(nodes[p].child_ids) > ratio` is the merge rule; `rank[p] = min(rank[k] for k in kids)` makes the parent inherit the best child's rank; the children are removed; the `while changed` loop repeats because a merged parent may itself complete its own parent. The final list is sorted by rank and converted with `as_chunk`.
- **Why it is here:** this is the LlamaIndex idea in 20 lines. `ratio` is the knob: low merges eagerly (more tokens), high rarely merges.

### 8. `sentence_window`  (`main.py` lines 125–136)

```python
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
```

- **In simple words:** index one sentence per vector; send it with `w` sentences on each side.
- **Exactly:** `lo, hi` clamp the window; the filter `x.doc_id == c.doc_id` stops the window from crossing into the next document; the new chunk's `text` is the matched sentence, its `context_text` the joined window, and its `spans` the window's source range.
- **Why it is here:** the simplest hierarchy: no tree, only a position. `w=0` must equal the flat sentence index.

### 9. RAPTOR part 1: the mock summariser  (`main.py` lines 139–147)

```python
# ---------------------------------------------------------------------------------------------------------------
# RAPTOR-style tree: cluster, summarise each cluster, repeat. The summariser is a plug-in (LLM in production).
# ---------------------------------------------------------------------------------------------------------------
def extractive_summary(texts, spans_list, vecs):
    """MOCK summariser: the 2 members closest to the cluster centroid, in document order. No new words are invented,
    so `spans` stay honest: a summary 'contains' only the source sentences it quotes."""
    centroid = vecs.mean(axis=0)
    best = sorted(np.argsort(-(vecs @ centroid))[:2])
    return " ".join(texts[i] for i in best), [sp for i in best for sp in spans_list[i]]   # union of the quoted spans
```

- **In simple words:** pick the two cluster members closest to the cluster centroid and return them in document order as the "summary".
- **Exactly:** `centroid = vecs.mean(axis=0)`; `vecs @ centroid` is the similarity of each member to the centroid; `np.argsort(-...)[:2]` takes the best two; `sorted(...)` restores document order. It returns the joined text and the union of the members' source spans. No new words are invented, so `spans` stay honest: the summary "contains" only what it quotes.
- **Why it is here:** a stand-in for an LLM summariser with the same signature, so a real one can be dropped in (`texts, spans, vecs -> text, spans`). Real RAPTOR summaries are abstractive, which is the point and the risk (they can drift from the source).

### 10. RAPTOR part 2: `build_raptor`  (`main.py` lines 150–170)

```python
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
```

- **In simple words:** per document, cluster the leaves with KMeans (about 4 per cluster), summarise each cluster, then repeat on the summaries until one node remains.
- **Exactly:** `KMeans(k, n_init=3, random_state=0).fit_predict(vecs)` assigns each node a cluster label (scikit-learn runs Lloyd's algorithm: assign to the nearest centre, recompute the centres, repeat; three restarts keep the best). `k = ceil(n / group)`. Each cluster becomes one summary `RawChunk` spanning the min start to the max end of its members, with `meta={"summary": True}`. `calls` counts summariser calls: in production, one LLM call each.
- **Why it is here:** shows the tree shape and its cost: roughly `n/group + n/group^2 + ...` extra nodes, each an LLM call.

### 11. Section A and B: ids, links, the vector-DB payload, tree shape  (`main.py` lines 173–193)

```python
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
```

- **In simple words:** pick leaf number 30, print it and its two ancestors, show the JSON payload you would store in a vector DB, and print the node count and token sizes per level.
- **Exactly:** `by_level` groups nodes by level; `json.dumps({...})` shows what a vector DB point contains: id, doc id, parent id and offsets, but not the parent's text. Only leaves are embedded; parents live in a key-value docstore keyed by `parent_id`.
- **Why it is here:** "what lives in the vector DB and what in the docstore" is the standard interview question for this step.

### 12. Section C: auto-merging on one question  (`main.py` lines 195–203)

```python
    # ---- C) auto-merging on one question -----------------------------------------------------------------------------
    q = "How long is parental leave for the primary and for the secondary caregiver?"
    M = embed([c.text for c in leaves])
    qv = embed([q])[0]
    order = list(np.argsort(-(M @ qv))[:12])
    merged = auto_merge(leaves, nodes)(order)
    print(f"\nC) question: {q!r}")
    print(f"   top-12 leaves -> after auto-merging: {len(merged)} contexts: "
          f"{[(m.meta['ctx_id'], count_tokens(m.context_text)) for m in merged[:4]]}")
```

- **In simple words:** retrieve the top 12 leaves for a question that needs two subsections (primary and secondary caregiver leave) and show what auto-merging turns them into.
- **Exactly:** `M @ qv` scores all leaves; `np.argsort(-...)[:12]` is the ranked top 12; `auto_merge(leaves, nodes)(order)` applies the default `ratio=0.5`; the printout lists `(context node id, tokens)` for the first four contexts.
- **Why it is here:** a concrete case where several leaves of one section merge into the parent.

### 13. Section D: the comparison  (`main.py` lines 205–231)

```python
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
```

- **In simple words:** build the sentence index and the RAPTOR tree, then score six strategies with the same harness.
- **Exactly:** `evaluate(chunks, postprocess=post)` calls `post` with the ranked ids to obtain the contexts; `RAPTOR collapsed tree` indexes `leaves + raptor` together with no postprocess. Columns: units indexed, tokens per unit, dense hit@1 and hit@3, MRR, hybrid hit@3, and tokens sent to the LLM.
- **Why it is here:** quality and context cost side by side: the trade-off is the result.

### 14. Entry point and the TRY IT / BREAK IT lines  (`main.py` lines 234–239)

```python
if __name__ == "__main__":
    main()

# TRY IT: auto_merge(..., ratio=0.3) then 0.8: how does the number of merged contexts and ctx tok@3 change?
# TRY IT: sentence_window(sent_chunks, w=0) must equal 'flat sentences'; w=5 sends far more tokens. Check hit@3 and ctx tok.
# BREAK IT: in small_to_big, remove the `seen` de-duplication: nothing crashes, the top-3 sometimes repeats a section (ctx tok@3 329 -> 345 at section level). Silent waste, not an error.
```

- **In simple words:** guard plus exercises.
- **Exactly:** without the `seen` set nothing crashes; the top-3 can repeat a section.
- **Why it is here:** a silent-waste bug is more dangerous than a crash.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `def small_to_big(...): def post(order): ... return post` | A closure: the inner function remembers `leaves`, `nodes`, `up` |
| `defaultdict(list)` | Dictionary that creates an empty list on first access |
| `for _ in range(up): n = nodes[n.parent_id] if n.parent_id else n` | Walk up the tree, staying at the root if there is no parent |
| `sorted(rank, key=rank.get)` | Sort node ids by their rank value |
| `[sp for i in best for sp in spans_list[i]]` | Nested comprehension: flatten spans of the chosen members |
| `labels = ... if k > 1 else np.zeros(n, int)` | Skip clustering when only one cluster is needed |
| `@dataclass` + `field(default_factory=list)` | Generated class; separate list per instance |

## Libraries & functions
| Function | What it does | Internally |
|---|---|---|
| `KMeans(k, n_init=3, random_state=0).fit_predict(X)` | Cluster labels for each row | Lloyd's algorithm: assign to the nearest centroid, recompute centroids, repeat until stable; best of 3 starts |
| `np.argsort(-scores)[:k]` | Indices of the top `k` scores | Sort descending |
| `vecs @ centroid` | Similarity of each row to the centroid | Matrix-vector product (cosine for unit vectors, roughly for a mean) |
| `json.dumps(dict)` | JSON string | Serialises Python types |
| `evaluate(..., postprocess=)` | Score the *contexts*, not the chunks | Applies the function to each ranked list before checking coverage |

## Execution trace (real run)
Real output of `python main.py` (progress bar removed; timings vary by machine).
```
A) one leaf, followed up the tree (this is what you store; the docstore is just {node_id: Node})
   level 0: refunds_and_returns:0:1      parent=refunds_and_returns:1:0    children= 0   19 tok  'The money is returned to the original pa'
   level 1: refunds_and_returns:1:0      parent=refunds_and_returns:2:0    children= 2   55 tok  'Refunds at Chai Kitchen are issued withi'
   level 2: refunds_and_returns:2:0      parent=None                       children= 1   55 tok  'Refunds at Chai Kitchen are issued withi'
   vector-DB point payload for the leaf: {"id": "refunds_and_returns:0:1", "doc_id": "refunds_and_returns", "parent_id": "refunds_and_returns:1:0", "start": 205, "end": 309}
   only LEAVES are embedded; parents live in a key-value docstore and are fetched by parent_id.

B) tree shape over 13 documents
   level 2 sections :  69 nodes, tokens min 21 mean 65 max 249
   level 1 mids     :  83 nodes, tokens min 21 mean 54 max 110
   level 0 leaves   : 156 nodes, tokens min 3 mean 29 max 62

C) question: 'How long is parental leave for the primary and for the secondary caregiver?'
   top-12 leaves -> after auto-merging: 10 contexts: [('employee_handbook:2:8', 32), ('employee_handbook:2:7', 41), ('employee_handbook:2:5', 33), ('employee_handbook:2:9', 48)]

   RAPTOR-style tree: 61 summary nodes from 156 leaves = 61 summariser calls (an LLM call each in production); first summary: 'Every blend except Winter Kahwa is nut free. Winter Kahwa contains almond sliver'

D) retrieval over 13 documents, 74 questions (ctx tok@3 = tokens sent to the LLM for the top 3 contexts)
strategy                       indexed  tok/unit  dense h@1  dense h@3   MRR  hybrid h@3  ctx tok@3
-----------------------------  -------  --------  ---------  ---------  ----  ----------  ---------
flat leaves (40 tok)               156        29       0.78       0.89  0.85        0.82         88
small-to-big -> mid (110)          156        29       0.86       0.93  0.91        0.91        197
small-to-big -> section (250)      156        29       0.91       0.95  0.93        0.95        329
sentence window +-2                375        13       0.86       0.95  0.91        0.95        197
auto-merging (ratio 0.5)           156        29       0.88       0.93  0.92        0.89        137
RAPTOR collapsed tree              217        42       0.78       0.88  0.85        0.85        125
```

How to read it:
- **A:** leaf `refunds_and_returns:0:1` (19 tokens) has parent `...:1:0` (a mid of 55 tokens with 2 children), whose parent `...:2:0` is the section. Here the section is so short (55 tokens) that mid and section have the **same text**: when the document is smaller than the section budget the levels collapse, and parent-lookup then returns the same text twice. The vector-DB payload holds only ids, parent id and offsets; the parent text sits in the docstore.
- **B:** 69 sections (mean 65 tokens), 83 mids (mean 54), 156 leaves (mean 29, min 3). The corpus is small, so most "levels" are barely bigger than each other; on long documents the levels separate properly.
- **C:** the top-12 leaves for the two-part parental-leave question merge into 10 contexts; the printed ones are whole handbook sections of 32 to 48 tokens, i.e. leaves of the same section were promoted to their parent.
- **D:** the hierarchical family is the clear winner on hit@1. Flat 40-token leaves: 0.78. Small-to-big to the section: **0.91** (hit@3 0.95, MRR 0.93) but 329 context tokens for the top 3 against 88. Sentence window +-2: 0.86 / 0.95 / 0.91 at 197 tokens. Auto-merging: 0.88 / 0.93 / 0.92 at only 137 tokens, the cheapest way to most of the gain. The RAPTOR collapsed tree (leaves plus 61 summary nodes, 61 summariser calls) does not help here (0.78, 0.88): extractive mock summaries add no new information and a corpus of short documents has little for a summary level to say. The differences between 0.78 and 0.91 are about 10 questions: larger than the noise level reported in [step 10](../10_chunking_evaluation/WALKTHROUGH.md), though the hit@3 gaps are not.

## Experiments (change ONE thing)
| Change | What you will see | Why |
|---|---|---|
| `auto_merge(..., ratio=0.3)` then `0.8` | 0.3 merges more (fewer, larger contexts, more tokens); 0.8 rarely merges and approaches flat leaves | `ratio` is the merge threshold on the share of children hit |
| `sentence_window(sent_chunks, w=0)`, then `2`, then `5` | `w=0` equals flat sentences; larger `w` sends many more tokens | The window adds `2w` neighbours per hit |
| `small_to_big(..., up=2)` vs `up=1` | Section-level contexts (250 tokens) vs mid-level (110) | Parent size is the knob |
| `group=2` in `build_raptor` | A deeper tree with more summary nodes and more (mock) LLM calls | Fewer members per cluster |
| `leaf_tokens=20` in `build_nodes` | More leaves; each is more specific | Leaf size controls search precision; parent size controls context |

## Debugging practice
**BREAK IT:** in `small_to_big`, delete the `seen` de-duplication (`if n.node_id not in seen:` and the set).

Symptom: nothing crashes. The top-3 contexts sometimes contain the same section twice, `ctx tok@3` rises (329 to 345 at section level in a real run, with hit@3 unchanged at 0.95) and the third slot that could have shown a different section is wasted. **Fix:** de-duplicate by `node_id` after the parent lookup, and add an assertion in the evaluation that contexts are unique. The lesson: a retrieval bug that costs tokens and quality but raises no error is found only by measuring `ctx tok` next to hit@k.

## Common mistakes
- Storing parents in the vector DB too. Only leaves are searched; parents belong in a docstore, otherwise parents compete with their own children.
- Forgetting to de-duplicate parents after the lookup.
- Using a parent size larger than the LLM can afford. `ctx tok` is the budget: check it.
- Judging a variant on hit@k alone. A bigger context almost always raises hit@k; compare quality per context token.
- Building a RAPTOR tree and never rebuilding it when a document changes (summaries go stale).
- Using abstractive summaries without checking them against the source: the summary node may state something the document does not.
- Windows that cross document boundaries (keep the same-document filter).

## Rebuild checklist
1. `Node` with `node_id`, `level`, span, `parent_id`, `child_ids`.
2. `build_nodes`: sections to mids to leaves, ids from counters, link both directions.
3. `as_chunk(node, context=None)` with `context_text` and `spans` of the sent node.
4. `small_to_big(up)`: walk up `up` levels, dedupe, keep rank order.
5. `auto_merge(top, ratio)`: group present nodes by parent, merge on `> ratio`, repeat until stable.
6. `sentence_window(w)`: clamp, same-document filter, joined text.
7. RAPTOR: KMeans per level, extractive summary, union spans, repeat to one root.
8. Compare on hit@k **and** `ctx tok@3`.

## Say it aloud
"Hierarchical retrieval decouples the unit of search from the unit of context: I embed small leaves for precision, store bigger parents in a key-value docstore with a `parent_id` link, and after retrieval replace hits by their parent, de-duplicated, so the LLM gets whole passages. The variants are small-to-big, sentence window, and auto-merging, which swaps a group of leaves for their parent when more than a ratio of the children matched. RAPTOR adds cluster summaries to the index so broad questions can hit a summary, at the price of LLM calls and stale summaries. I judge them on quality per context token, because a bigger context raises recall and the bill together."

## The new problem (cliffhanger)
All of this assumes sentences and sections are the right atoms. But a sentence is often a bad atom for retrieval: "It must be approved by the manager, and the budget is 1,200 euros" is two facts joined by a comma and a pronoun. What if we let an LLM rewrite the text into self-contained *atomic facts* first, or let it decide the boundaries itself? Next: [step 08, propositions and agentic chunking](../08_propositions_and_agentic_chunking/WALKTHROUGH.md).
