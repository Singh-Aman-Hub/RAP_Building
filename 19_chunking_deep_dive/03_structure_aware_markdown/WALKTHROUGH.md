# Step 03 — Structure-aware Markdown chunking

## The story so far
[Step 02](../02_recursive_splitter_rebuilt/WALKTHROUGH.md) rebuilt the recursive splitter: biggest separator first, merge up to a token limit. Its last output showed the weakness: with `keep_separator="end"` a heading marker was left dangling at the end of the previous chunk, and a regex has no idea that a heading *owns* the text under it. Back to the [module README](../README.md).

## The pain point
A structure-blind splitter damages the things authors took care to build:
- a **table** is cut between rows, so the second half has numbers with no column names;
- a **code block** is cut in half (odd number of ` ``` ` fences), or a `# comment` inside a fence is mistaken for a heading;
- a paragraph that says "It must be approved by the manager" ends up in a chunk with no heading, so "It" has no noun and the vector does not know the topic is *remote work*;
- a tiny section of 12 tokens becomes its own chunk with a vector that means nothing.

## The spark
Parse the Markdown into **blocks** (heading, paragraph, list, table, code), build the **heading tree** with a stack, and chunk *by section*:
1. Never split a table, a code fence or a list item run in the middle (they are atomic blocks).
2. Group whole blocks up to the size budget; split *inside* a too-big block only along its own structure (sentences, table rows with the header repeated, code lines with the fence re-closed).
3. Prepend a **breadcrumb** (`Doc > Section > Subsection`) to what is embedded and to what the LLM reads, so the chunk carries its own context.
4. Merge tiny sibling sections.

## Goal
Build a block parser, a heading-tree builder and a `StructureChunker`, then measure (A) which block kinds exist, (B) the heading tree with token counts, (C) the chunks with their breadcrumbs, (D) the structural damage a generic splitter does, and (E) the retrieval effect of the breadcrumb.

## The theory (read this before the interview)
**Markdown block grammar (the subset we need).**
```
heading     ^#{1,6}\s+title                       level = number of #
fenced code ``` ... ```                           ONE block; everything inside is literal
table       consecutive lines starting with |     line 1 header, line 2 separator |---|, then rows
list        lines starting with - * + or 1.       plus indented continuation lines
paragraph   anything else, until blank line or another block starts
```
**Heading tree with a stack.** Keep a stack of open sections. A new heading of level `L` closes every open section whose level is `>= L`, then becomes a child of the section now on top, and is pushed. The first `# Title` becomes the document root. Every non-heading block is appended to the section on top of the stack. The section's **path** is the titles from the root down.

**Breadcrumb / contextual header.** `Doc > Parental leave > Secondary caregiver`. It is prepended to `embed_text` (so the vector and BM25 know the topic) and to `context_text` (so the LLM knows where the passage is from) but *not* to `text` (the exact source slice used for offsets and citations).

**Atomic blocks and internal splitting.**
```
paragraph / list -> pack whole sentences (or list lines)
table            -> row groups, the 2-line header repeated in every group
code             -> line groups, wrapped in a fresh ``` fence
```
**Tiny sections.** A 12-token chunk has a vector dominated by its heading words. Neighbouring tiny sibling sections (same parent) are merged and the merged section's heading is kept *inside the text* (`Heading: body`), and the chunk keeps all the source spans it came from.

**Metadata.** Each chunk keeps `section_path`, the block kinds it contains and the heading level: used for filters ("only the API reference"), citations and later for parent-child retrieval ([step 07](../07_hierarchical_parent_child/WALKTHROUGH.md)).

**Status.** Structure-aware chunking of Markdown / HTML / docx is *current industry practice* (LangChain `MarkdownHeaderTextSplitter`, LlamaIndex `MarkdownNodeParser`, Unstructured `by_title`, Docling's hierarchical chunker). The heading tree and breadcrumb are the parts that last; the exact block grammar differs by library.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| structure-aware chunking | document-structure chunking, header-based splitting, section chunking, `MarkdownHeaderTextSplitter`, `by_title` | near | Same idea; tools differ in how they merge and split big sections |
| breadcrumb | contextual header, heading path, section path, title prefix, "contextual chunk headers" | near | A deterministic prefix from the structure; [step 06](../06_contextual_retrieval/WALKTHROUGH.md)'s contextual retrieval writes a prefix with an LLM instead |
| heading tree | document outline, section hierarchy, TOC tree | yes | Parent/child sections built with a stack |
| atomic block | unsplittable unit, non-splittable element | near | A table / code fence / list kept whole when it fits |
| row-group splitting | table chunking with repeated header | near | Split a table by rows and repeat the header in each piece |
| tiny-chunk merging | small-chunk coalescing, `combine_text_under_n_chars` (Unstructured) | near | Merge neighbours below a minimum size |
| element-based chunking | partition then chunk (Unstructured) | no | Works on parsed elements of any file type; [step 09](../09_layout_aware_chunking_from_parsed_docs/WALKTHROUGH.md) |

## Run it
```bash
cd 19_chunking_deep_dive/03_structure_aware_markdown
python main.py
# from the repo root:
HF_HUB_OFFLINE=1 .venv/bin/python 19_chunking_deep_dive/03_structure_aware_markdown/main.py
```
About 15 to 20 s of CPU time. Needs `torch` + `transformers` and the cached MiniLM; no downloads.

## Code, step by step

### 1. Header docstring, imports, the two regexes  (`main.py` lines 1–25)

```python
"""
GOAL:  Parse Markdown into blocks (heading / paragraph / list / table / code) and a heading tree, then chunk BY SECTION:
       breadcrumb headers ("Doc > Section > Subsection"), never split a table / code block / list item run,
       split long sections at block boundaries (then sentences / table rows / code lines), merge tiny sections,
       attach metadata. Measure the damage a structure-blind splitter does, and the retrieval effect.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 19_chunking_deep_dive/03_structure_aware_markdown/main.py
EXPECTED OUTPUT (shortened):
    A) block kinds found per document      B) heading tree of the handbook (with token counts)
    C) first chunks with breadcrumbs       D) structural damage: generic recursive vs structure-aware
    E) retrieval: with / without breadcrumb, vs recursive 100 tokens
NEEDS: torch, transformers (cached MiniLM).  ~15 s.
"""
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common19 import RawChunk, count_tokens, evaluate, load_docs, load_step, sentence_spans, table  # noqa: E402

HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
LIST_ITEM = re.compile(r"^\s*([-*+]|\d+[.)])\s+")
```

- **In simple words:** goal, imports and two compiled regexes: one recognises a heading line, one a list item.
- **Exactly:** `HEADING = ^(#{1,6})\s+(.*?)\s*#*\s*$` captures the `#` run (its length is the level) and the title, ignoring trailing `#` marks. `LIST_ITEM = ^\s*([-*+]|\d+[.)])\s+` matches `-`, `*`, `+`, `1.` or `1)` bullets with optional indentation. `load_step` is the helper that imports another step's `main.py` ([step 01 tour](../01_tokens_not_characters/WALKTHROUGH.md#the-shared-helpers-once-a-tour-of-common19py)).
- **Why it is here:** two cheap regexes are enough for a block parser; a full Markdown library would add dependencies for no gain here.

### 2. `Block` and `Section` data classes  (`main.py` lines 28–46)

```python
@dataclass
class Block:
    kind: str            # heading | paragraph | list | table | code
    start: int           # char offsets in the document
    end: int
    level: int = 0       # heading level (1-6)
    title: str = ""


@dataclass
class Section:
    level: int
    title: str
    blocks: list = field(default_factory=list)       # body blocks BEFORE the first child heading
    children: list = field(default_factory=list)
    parent: "Section" = None

    def path(self):
        return ([] if self.parent is None else self.parent.path()) + ([self.title] if self.level else [])
```

- **In simple words:** a `Block` is one piece of the document with a kind and a character span; a `Section` is a heading with its own body blocks and child sections.
- **Exactly:** `field(default_factory=list)` gives every instance its own list (a bare `[]` default would be shared). `parent: "Section" = None` is a quoted forward reference to the class being defined. `path()` walks up the parent chain recursively and returns the titles from the root down (the root, level 0, contributes none).
- **Why it is here:** the tree is the data structure the whole step rests on: a `Section.blocks` list is "the body before the first child heading".

### 3. `parse_blocks`: text to blocks  (`main.py` lines 49–94)

```python
# ---------------------------------------------------------------------------------------------------------------
# 1. text -> blocks. A fenced code block is ONE block, so '# comment' lines inside it are never mistaken for headings.
# ---------------------------------------------------------------------------------------------------------------
def parse_blocks(text):
    blocks, pos, i = [], 0, 0
    lines = text.splitlines(keepends=True)
    offs = []                                                # offs[i] = char offset where line i starts
    for ln in lines:
        offs.append(pos)
        pos += len(ln)
    offs.append(pos)
    while i < len(lines):
        ln, stripped = lines[i], lines[i].strip()
        if not stripped:
            i += 1
        elif stripped.startswith("```"):                     # fenced code: runs to the closing fence
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("```"):
                j += 1
            j = min(j, len(lines) - 1)
            blocks.append(Block("code", offs[i], offs[j] + len(lines[j].rstrip("\n"))))
            i = j + 1
        elif HEADING.match(stripped):
            m = HEADING.match(stripped)
            blocks.append(Block("heading", offs[i], offs[i] + len(ln.rstrip("\n")), len(m.group(1)), m.group(2)))
            i += 1
        elif stripped.startswith("|"):                       # table: consecutive lines starting with |
            j = i
            while j + 1 < len(lines) and lines[j + 1].strip().startswith("|"):
                j += 1
            blocks.append(Block("table", offs[i], offs[j] + len(lines[j].rstrip("\n"))))
            i = j + 1
        elif LIST_ITEM.match(ln):                            # list: item lines plus indented continuations
            j = i
            while j + 1 < len(lines) and lines[j + 1].strip() and (LIST_ITEM.match(lines[j + 1]) or lines[j + 1].startswith((" ", "\t"))):
                j += 1
            blocks.append(Block("list", offs[i], offs[j] + len(lines[j].rstrip("\n"))))
            i = j + 1
        else:                                                # paragraph: until blank line or another block starts
            j = i
            while j + 1 < len(lines) and lines[j + 1].strip() and not lines[j + 1].strip().startswith(("```", "|", "#")) \
                    and not LIST_ITEM.match(lines[j + 1]):
                j += 1
            blocks.append(Block("paragraph", offs[i], offs[j] + len(lines[j].rstrip("\n"))))
            i = j + 1
    return blocks
```

- **In simple words:** read the document line by line and decide for each line which kind of block starts there, then consume all lines of that block.
- **Exactly:** `splitlines(keepends=True)` keeps the newline characters so `offs[i]` (the running sum of line lengths) is the exact character offset of line `i`. The `while` loop checks, in order: a blank line (skip), a ` ``` ` fence (scan to the closing fence; the whole fence is **one** block, so `# comment` lines inside are never seen as headings), a heading, a table (all consecutive `|` lines), a list (item lines plus indented continuations), and finally a paragraph (until a blank line or the start of another block). Each block records `(kind, start, end)` with the trailing newline excluded.
- **Why it is here:** the fence check comes *before* the heading check. Reorder them and a comment in a bash snippet becomes a fake section.

### 4. `build_tree` and `walk`: blocks to a heading tree  (`main.py` lines 97–122)

```python
# ---------------------------------------------------------------------------------------------------------------
# 2. blocks -> heading tree (a stack of open sections; a heading closes every section at the same or deeper level)
# ---------------------------------------------------------------------------------------------------------------
def build_tree(blocks, doc_title):
    root = Section(0, doc_title)
    stack = [root]
    first_h1_seen = False
    for b in blocks:
        if b.kind == "heading":
            if b.level == 1 and not first_h1_seen and not root.blocks:
                root.title, first_h1_seen = b.title, True    # the first '# Title' IS the document root
                continue
            while stack[-1].level >= b.level:
                stack.pop()
            sec = Section(b.level, b.title, parent=stack[-1])
            stack[-1].children.append(sec)
            stack.append(sec)
        else:
            stack[-1].blocks.append(b)
    return root


def walk(sec):
    yield sec
    for c in sec.children:
        yield from walk(c)
```

- **In simple words:** a stack of open sections. A heading closes every section at the same or a deeper level, then opens a new child; all other blocks are added to the section on top.
- **Exactly:** the first level-1 heading (while the root has no body yet) renames the root and is not a section itself. `while stack[-1].level >= b.level: stack.pop()` is the closing rule. `walk` is a recursive generator (`yield from`) that visits the tree depth-first in document order.
- **Why it is here:** the stack gives the `path` of every block in one pass, which is what the breadcrumb is made of.

### 5. `StructureChunker`: config and `split_block` (inside one oversize block)  (`main.py` lines 125–141)

```python
class StructureChunker:
    def __init__(self, max_tokens=200, min_tokens=40, breadcrumb=True):
        self.max, self.min, self.breadcrumb = max_tokens, min_tokens, breadcrumb

    # -- 3. split ONE oversize block without breaking its internal structure ------------------------------------------
    def split_block(self, doc, b, budget):
        text = doc["text"]
        if b.kind in ("paragraph", "list"):                  # pack whole sentences (list items are lines = sentences)
            parts = sentence_spans(text[b.start:b.end], base=b.start)
            return self.pack(doc, parts, budget, join="\n" if b.kind == "list" else " ")
        if b.kind == "table":                                # row groups, header (2 lines) repeated in every group
            rows = sentence_spans(text[b.start:b.end], base=b.start)
            header = text[rows[0][0]:rows[1][1]] if len(rows) > 2 else ""
            groups = self.pack(doc, rows[2:], budget - count_tokens(header), join="\n")
            return [(f"{header}\n{t}", s, e) for t, s, e in groups]
        lines = sentence_spans(text[b.start:b.end], base=b.start)        # code: group lines, re-close the fence
        return [(f"```\n{t}\n```", s, e) for t, s, e in self.pack(doc, lines[1:-1], budget - 6, join="\n")]
```

- **In simple words:** when a single block is bigger than the budget, split it *along its own structure* rather than at an arbitrary character.
- **Exactly:** for a paragraph or list, `sentence_spans` gives sentence (or line) spans which `pack` groups up to `budget` tokens. A table: `rows[0]` and `rows[1]` are the header and separator lines; `header` is their text; the remaining rows are packed into groups with the budget reduced by the header's tokens; every group is returned as `header + rows`, so every piece has column names. Code: the first and last line are the fences; inner lines are packed into groups (budget minus 6 tokens for the new fences), and each group is wrapped in a fresh fence so it stays valid Markdown. Each return value is a list of `(text, start, end)`.
- **Why it is here:** these three branches are the "never cut a table row from its header, never leave a code fence open" guarantee.

### 6. `pack`: greedy sentence packing  (`main.py` lines 143–152)

```python
    def pack(self, doc, spans, budget, join):
        out, cur = [], []
        for sp in spans:
            if cur and count_tokens(join.join(doc["text"][a:b] for a, b in cur + [sp])) > budget:
                out.append((join.join(doc["text"][a:b] for a, b in cur), cur[0][0], cur[-1][1]))
                cur = []
            cur.append(sp)
        if cur:
            out.append((join.join(doc["text"][a:b] for a, b in cur), cur[0][0], cur[-1][1]))
        return out
```

- **In simple words:** add spans to the current group until the next one would exceed the budget, then start a new group.
- **Exactly:** the token count is recomputed on the *joined* text (`join.join(...)`), so the joiner's tokens count. Returns `(text, start, end)` tuples where the offsets are the first span's start and the last span's end.
- **Why it is here:** the same 10 lines are the core of sentence-pack, late chunking and the hierarchical builder in the next steps.

### 7. `section_units`: group whole blocks inside one section  (`main.py` lines 154–174)

```python
    # -- 4. one section body -> list of (text, start, end, kinds): whole blocks grouped up to the budget -----------------
    def section_units(self, doc, sec, budget):
        units, cur = [], []                                  # cur = blocks being grouped

        def flush():
            if cur:
                units.append((doc["text"][cur[0].start:cur[-1].end], cur[0].start, cur[-1].end, sorted({b.kind for b in cur})))
                cur.clear()

        for b in sec.blocks:
            t = doc["text"][b.start:b.end]
            if count_tokens(t) > budget:                     # a single block too big: split INSIDE it, structure-aware
                flush()
                units += [(t2, s, e, [b.kind]) for t2, s, e in self.split_block(doc, b, budget)]
            elif cur and count_tokens(doc["text"][cur[0].start:b.end]) > budget:
                flush()
                cur.append(b)
            else:
                cur.append(b)
        flush()
        return units
```

- **In simple words:** walk a section's blocks and pack them into units up to the budget; a block that alone exceeds the budget is split by `split_block`.
- **Exactly:** `cur` is the group being built; `flush()` (a nested function) turns it into one unit `(text, start, end, kinds)`, where `kinds` is the sorted set of block types. A group closes when adding the next block would exceed the budget (`count_tokens(text[cur[0].start:b.end])` includes the gap text between the blocks).
- **Why it is here:** it keeps a heading's intro paragraph with its list and a table with its caption paragraph whenever they fit together.

### 8. `split_document`: sections to chunks with breadcrumbs  (`main.py` lines 176–197)

```python
    def split_document(self, doc):
        root = build_tree(parse_blocks(doc["text"]), doc["title"])
        doc_title = root.title
        raw = []                                             # dicts: path, text, start, end, kinds, level
        for sec in walk(root):
            if not sec.blocks:
                continue                                     # a heading with no own body (only children) makes no chunk
            path = [doc_title] + sec.path()
            crumb = " > ".join(path)
            budget = self.max - (count_tokens(crumb) + 1 if self.breadcrumb else 0)
            for t, s, e, kinds in self.section_units(doc, sec, budget):
                raw.append({"path": path, "text": t, "start": s, "end": e, "kinds": kinds, "level": sec.level})
        chunks = []
        for r in self.merge_tiny(raw):
            crumb = " > ".join(r["path"])
            c = RawChunk(doc["doc_id"], r["start"], r["end"], r["text"], section_path=crumb,
                         spans=r.get("spans") or [(r["start"], r["end"])],
                         meta={"level": r["level"], "kinds": r["kinds"], "doc_title": doc_title})
            if self.breadcrumb:
                c.embed_text = c.context_text = f"{crumb}\n{r['text']}"   # header is embedded AND sent to the LLM
            chunks.append(c)
        return chunks
```

- **In simple words:** for every section with a body, make the breadcrumb, shrink the token budget by the breadcrumb's length, split the body into units, merge tiny ones, and build the chunk records.
- **Exactly:** `path = [doc_title] + sec.path()`, `crumb = " > ".join(path)`. The budget is `max - (tokens(crumb) + 1)` so the *embedded text* (crumb + body) stays within `max_tokens`. A section with no own body (only children) produces no chunk. After `merge_tiny`, each chunk becomes a `RawChunk` whose `text` is the exact body, `section_path` the crumb, `spans` the source spans, and `meta` the heading level, block kinds and document title. With `breadcrumb=True`, `embed_text` and `context_text` are both set to `crumb + "\n" + text`.
- **Why it is here:** this is the "header in the embedding and in the prompt, offsets from the exact text" design. `text` stays the true source slice, which is what citations and the evaluation use.

### 9. `merge_tiny`: merge small sibling sections  (`main.py` lines 199–213)

```python
    # -- 5. merge neighbours that are both tiny and share the same parent heading ---------------------------------------
    def merge_tiny(self, raw):
        out = []
        for r in raw:
            if out and count_tokens(out[-1]["text"]) < self.min and out[-1]["path"][:-1] == r["path"][:-1] \
                    and count_tokens(out[-1]["text"] + r["text"]) + 20 <= self.max:
                p, label = out[-1], r["path"][-1]            # keep the merged section's heading inside the text
                p["text"] = p["text"] + f"\n\n{label}: " + r["text"]
                p["spans"] = (p.get("spans") or [(p["start"], p["end"])]) + [(r["start"], r["end"])]
                p["end"] = r["end"]
                p["kinds"] = sorted(set(p["kinds"]) | set(r["kinds"]))
                p["path"] = p["path"][:-1] + [p["path"][-1] + " + " + label]
            else:
                out.append(dict(r))
        return out
```

- **In simple words:** if the previous chunk is under `min` tokens, has the same parent heading as this one, and they fit together, glue them.
- **Exactly:** `out[-1]["path"][:-1] == r["path"][:-1]` means same parent. The merged text keeps the second section's heading inside (`"\n\nLabel: body"`); `spans` records both source regions; the path becomes `A + B` (two sibling titles); `kinds` is the union. The `+ 20` leaves room for the label and the crumb.
- **Why it is here:** a 12-token chunk is a poor retrieval unit and costs a vector slot.

### 10. Section A and B: block kinds and the heading tree  (`main.py` lines 216–234)

```python
def main():
    docs = load_docs()
    hb = next(d for d in docs if d["doc_id"] == "employee_handbook")
    api = next(d for d in docs if d["doc_id"] == "api_reference")

    print("A) block kinds per document")
    rows = []
    for d in [x for x in docs if x["corpus"] == "new"]:
        k = Counter(b.kind for b in parse_blocks(d["text"]))
        rows.append([d["doc_id"]] + [k.get(x, 0) for x in ("heading", "paragraph", "list", "table", "code")])
    print(table(rows, ["document", "heading", "paragraph", "list", "table", "code"]))
    fake = [b for b in parse_blocks(hb["text"]) if b.kind == "heading" and b.title.startswith("Report a suspected")]
    print(f"   '# Report a suspected incident...' is a comment inside a ```bash fence: parsed as a heading? {bool(fake)}")

    print("\nB) heading tree of the handbook (tokens of each section's own body)")
    root = build_tree(parse_blocks(hb["text"]), hb["title"])
    for s in walk(root):
        body = sum(count_tokens(hb["text"][b.start:b.end]) for b in s.blocks)
        print(f"   {'  ' * s.level}{s.title[:44]:<44} {body:>4} tok")
```

- **In simple words:** count the block kinds per new document, check the fake-heading trap, and print the handbook's tree with the token size of each section's own body.
- **Exactly:** `Counter(b.kind for b in parse_blocks(...))` tallies the kinds. `fake` looks for a heading titled "Report a suspected incident..." which only exists as a `# comment` inside a bash fence; the expected answer is `False`. In B, `'  ' * s.level` indents by depth.
- **Why it is here:** it proves the parser sees code fences as atoms and shows how uneven section sizes are (some are 10 tokens, some 100).

### 11. Section C: first chunks, a split table, the FAQ  (`main.py` lines 236–247)

```python
    print("\nC) first chunks (max 200 tokens, min 40): breadcrumb + body")
    ch = StructureChunker().split_document(hb)
    print(f"   handbook -> {len(ch)} chunks")
    for c in ch[2:5]:
        print(f"   [{count_tokens(c.text):>3} tok] {c.section_path}\n          {c.text[:70]!r}")
    big = StructureChunker(max_tokens=70, min_tokens=20).split_document(api)
    print(f"   api_reference with max 70 tokens -> {len(big)} chunks; the Errors table became:")
    for c in big:
        if "Errors" in c.section_path and "table" in c.meta["kinds"]:
            print("   ", repr(c.text[:100] + " ..."))
    faq = StructureChunker().split_document(next(d for d in docs if d["doc_id"] == "support_faq"))
    print(f"   FAQ (no sub-headings): {len(faq)} chunks, kinds {faq[0].meta['kinds']}, first path {faq[0].section_path!r}")
```

- **In simple words:** print three handbook chunks with their breadcrumbs, show the API errors table when the budget is tiny (70 tokens), and chunk the FAQ, which has no sub-headings.
- **Exactly:** `ch[2:5]` is a slice of the chunk list; `c.section_path` is the breadcrumb; `c.meta["kinds"]` lists the block types in the chunk.
- **Why it is here:** the tiny-budget run shows header repetition in action; the FAQ shows the graceful case where there is no hierarchy to exploit.

### 12. Section D: structural damage, generic vs structure-aware  (`main.py` lines 249–263)

```python
    print("\nD) structural damage on the two structured documents (api_reference + handbook)")
    rec = load_step("02_recursive_splitter_rebuilt")

    def damage(chunks):
        odd_fence = sum(c.text.count("```") % 2 for c in chunks)           # a code block cut in half has an odd fence count
        headless = sum(1 for c in chunks if c.text.lstrip().startswith("|") and not re.match(r"\|\s*(Plan|HTTP)", c.text.lstrip()))
        return odd_fence, headless

    rows = []
    for name, fn in (("recursive 100 tok (generic)", lambda d: rec.RecursiveSplitter(100, 0).split_document(d)),
                     ("structure-aware max 100", lambda d: StructureChunker(100, 30).split_document(d))):
        cs = [c for d in (api, hb) for c in fn(d)]
        f, h = damage(cs)
        rows.append([name, len(cs), f, h])
    print(table(rows, ["chunker", "chunks", "code fences cut (odd ```)", "table fragments w/o header"]))
```

- **In simple words:** count code blocks cut in half and table fragments without a header, for the generic recursive splitter and for the structure chunker, on the two structured documents.
- **Exactly:** an odd number of ` ``` ` fences in a chunk means a code block was cut; a chunk that starts with `|` but not with the header row (`| Plan` or `| HTTP`) is a headerless table fragment. `load_step` imports step 02's splitter, so nothing is re-implemented.
- **Why it is here:** damage counts make the argument without needing a retrieval model.

### 13. Section E: retrieval with and without breadcrumb  (`main.py` lines 265–278)

```python
    print("\nE) retrieval over all 13 documents, 74 questions")
    rows = []

    def add(name, chunks):
        r = evaluate(chunks, docs=docs)
        d = r["dense"]
        rows.append([name, len(chunks), f"{r['mean_tok']:.0f}", f"{d['hit@1']:.2f}", f"{d['hit@3']:.2f}", f"{d['mrr']:.2f}",
                     f"{r['bm25']['hit@3']:.2f}", f"{r['hybrid']['hit@3']:.2f}", f"{d['ctx_tok']:.0f}"])

    add("recursive 100 tok", [c for d in docs for c in rec.RecursiveSplitter(100, 0).split_document(d)])
    add("structure 200, no breadcrumb", [c for d in docs for c in StructureChunker(200, 40, breadcrumb=False).split_document(d)])
    add("structure 200 + breadcrumb", [c for d in docs for c in StructureChunker(200, 40).split_document(d)])
    add("structure 100 + breadcrumb", [c for d in docs for c in StructureChunker(100, 30).split_document(d)])
    print(table(rows, ["chunker", "chunks", "mean tok", "dense h@1", "dense h@3", "MRR", "bm25 h@3", "hybrid h@3", "ctx tok@3"]))
```

- **In simple words:** four configurations, scored on the 74 questions.
- **Exactly:** `add` builds the chunks, runs `evaluate`, and appends one table row. Columns: chunks, mean tokens, dense hit@1 and hit@3, MRR, BM25 hit@3, hybrid hit@3, context tokens for the top 3.
- **Why it is here:** the breadcrumb is a cheap, deterministic way to add context; this is its measured effect.

### 14. Entry point and the TRY IT / BREAK IT lines  (`main.py` lines 281–286)

```python
if __name__ == "__main__":
    main()

# TRY IT: StructureChunker(max_tokens=60, min_tokens=10) on api_reference: how many chunks does the error table become?
# TRY IT: set breadcrumb=False and compare the "reference"-kind questions in the retrieval table.
# BREAK IT: delete the fenced-code branch in parse_blocks -> '# Report a suspected...' becomes a fake heading (section A shows True).
```

- **In simple words:** the `__main__` guard (so step 07 and step 10 can import `StructureChunker`) and the exercises.
- **Exactly:** removing the fenced-code branch from `parse_blocks` makes the shell comment a heading.
- **Why it is here:** the exercise shows that the parser order is part of the correctness.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `@dataclass` with `field(default_factory=list)` | Generated class; each instance gets its own list |
| `parent: "Section" = None` | Forward reference in quotes: the class does not exist yet at definition time |
| `yield from walk(c)` | Delegate to a sub-generator: recursive depth-first traversal |
| `lines[i].strip().startswith("```")` | Fence detection ignoring indentation |
| `str.startswith((" ", "\t"))` | Tuple form: any of several prefixes |
| `def flush(): ... cur.clear()` (nested) | Inner function closes over `units` and `cur` |
| `sorted({b.kind for b in cur})` | Set comprehension then sort: the distinct kinds in stable order |
| `f"{header}\n{t}"` | Header row text prepended to every table group |
| `c.embed_text = c.context_text = ...` | Chained assignment: both fields get the same string |

## Libraries & functions
| Function | What it does | Internally |
|---|---|---|
| `re.compile(...).match(s)` | Anchored match at the start of `s` | Returns a match object with groups or `None` |
| `str.splitlines(keepends=True)` | Lines including newline characters | Lets offsets be the exact running sum of lengths |
| `Counter` | Tally of block kinds | `dict` with default 0 |
| `sentence_spans(text, base)` | Sentence/line spans | Regex boundary; see the [step 01 tour](../01_tokens_not_characters/WALKTHROUGH.md#the-shared-helpers-once-a-tour-of-common19py) |
| `count_tokens` | Embedder tokens of a string | Cached tokenizer call |
| `evaluate` | Score chunks on 74 questions | Same harness for every step |

## Execution trace (real run)
Real output of `python main.py` (progress bar removed; timings vary).
```
A) block kinds per document
document             heading  paragraph  list  table  code
-------------------  -------  ---------  ----  -----  ----
api_reference              7         10     0      2     2
employee_handbook         25         19     0      0     1
incident_postmortem        0         10     0      0     0
support_faq                1         15     0      0     0
   '# Report a suspected incident...' is a comment inside a ```bash fence: parsed as a heading? False

B) heading tree of the handbook (tokens of each section's own body)
   Lumen Cloud Employee Handbook                  35 tok
       1. Working at Lumen                             0 tok
         1.1 Working hours                              38 tok
         1.2 Remote work                                 0 tok
           1.2.1 Eligibility                              34 tok
           1.2.2 Home office equipment                    66 tok
           1.2.3 Working from abroad                      45 tok
       2. Time off                                     0 tok
         2.1 Annual leave                               33 tok
         2.2 Sick leave                                 39 tok
         2.3 Parental leave                              0 tok
           2.3.1 Primary caregiver                        41 tok
           2.3.2 Secondary caregiver                      32 tok
           2.3.3 Returning to work                        48 tok
       3. Expenses and travel                          0 tok
         3.1 Travel booking                             51 tok
         3.2 Meals and per diem                         38 tok
         3.3 Submitting expenses                        41 tok
       4. Information security                         0 tok
         4.1 Passwords and devices                      56 tok
         4.2 Incident reporting                        131 tok
         4.3 Acceptable use                             35 tok
       5. Learning and development                     0 tok
         5.1 Learning budget                            34 tok
         5.2 Conferences                                28 tok

C) first chunks (max 200 tokens, min 40): breadcrumb + body
   handbook -> 13 chunks
   [109 tok] Lumen Cloud Employee Handbook > 1. Working at Lumen > 1.2 Remote work > 1.2.1 Eligibility + 1.2.2 Home office equipment
          'Every employee may work remotely up to three days per week after the s'
   [ 45 tok] Lumen Cloud Employee Handbook > 1. Working at Lumen > 1.2 Remote work > 1.2.3 Working from abroad
          'You may work from another country for up to 20 days per calendar year.'
   [ 78 tok] Lumen Cloud Employee Handbook > 2. Time off > 2.1 Annual leave + 2.2 Sick leave
          'Full-time employees receive 28 days of paid annual leave per year. Unu'
   api_reference with max 70 tokens -> 25 chunks; the Errors table became:
    '| HTTP status | Code | Meaning | What to do |\n|-------------|------|---------|------------|\n| 400 |  ...'
    '| HTTP status | Code | Meaning | What to do |\n|-------------|------|---------|------------|\n| 401 |  ...'
    '| HTTP status | Code | Meaning | What to do |\n|-------------|------|---------|------------|\n| 404 |  ...'
    '| HTTP status | Code | Meaning | What to do |\n|-------------|------|---------|------------|\n| 409 |  ...'
    '| HTTP status | Code | Meaning | What to do |\n|-------------|------|---------|------------|\n| 422 |  ...'
    '| HTTP status | Code | Meaning | What to do |\n|-------------|------|---------|------------|\n| 429 |  ...'
    '| HTTP status | Code | Meaning | What to do |\n|-------------|------|---------|------------|\n| 503 |  ...'
   FAQ (no sub-headings): 4 chunks, kinds ['paragraph'], first path 'Lumen Cloud Support FAQ'

D) structural damage on the two structured documents (api_reference + handbook)
chunker                      chunks  code fences cut (odd ```)  table fragments w/o header
---------------------------  ------  -------------------------  --------------------------
recursive 100 tok (generic)      27                          4                           2
structure-aware max 100          41                          0                           0

E) retrieval over all 13 documents, 74 questions

chunker                       chunks  mean tok  dense h@1  dense h@3   MRR  bm25 h@3  hybrid h@3  ctx tok@3
----------------------------  ------  --------  ---------  ---------  ----  --------  ----------  ---------
recursive 100 tok                 68        73       0.74       0.91  0.83      0.85        0.95        223
structure 200, no breadcrumb      54        85       0.73       0.89  0.82      0.88        0.96        288
structure 200 + breadcrumb        56        95       0.78       0.91  0.86      0.89        0.96        315
structure 100 + breadcrumb        92        66       0.82       0.91  0.88      0.86        0.96        199
```

How to read it:
- **A:** the parser finds 2 tables and 2 code blocks in the API reference, 1 code block in the handbook, and in the postmortem 10 paragraphs and **0 headings**: there is no structure to exploit there. The trap check prints `False`: the `# Report a suspected incident...` comment inside the bash fence is *not* a heading.
- **B:** the handbook tree. Level-1 sections such as "2. Time off" have a body of 0 tokens (they only contain children); most subsections are 28 to 66 tokens, far below the 200-token budget, and the largest, "4.2 Incident reporting" (131 tokens), contains the code fence. Many sections are under the 40-token minimum, which is why merging exists.
- **C:** the handbook becomes 13 chunks. The first printed chunk is two sibling sections merged (`1.2.1 Eligibility + 1.2.2 Home office equipment`, 109 tokens) and the third merges `2.1 Annual leave + 2.2 Sick leave`. With a 70-token budget the Errors table becomes **7 chunks, each starting with the header row** (one status code per chunk here). The FAQ has one heading, so its 4 chunks have only the document title as breadcrumb.
- **D:** the generic splitter at 100 tokens cuts **4 code blocks in half and leaves 2 table fragments without a header**; the structure-aware chunker (41 chunks) leaves 0 and 0.
- **E:** at the same 200-token budget the breadcrumb lifts dense hit@1 from 0.73 to 0.78 and MRR from 0.82 to 0.86 and BM25 hit@3 from 0.88 to 0.89, for about 10 extra tokens per chunk (mean 85 to 95). Dense hit@3 is 0.89 to 0.91 for all four rows: no difference at that level. The best row on hit@1 and MRR is `structure 100 + breadcrumb` (0.82 and 0.88) and it also sends the fewest context tokens (199 against 223 for recursive 100). Differences of 0.04 to 0.09 in hit@1 are 3 to 7 questions out of 74, so call it a consistent hint, not proof ([step 10](../10_chunking_evaluation/WALKTHROUGH.md) puts intervals on this).

## Experiments (change ONE thing)
| Change | What you will see | Why |
|---|---|---|
| `StructureChunker(60, 10)` on `api_reference` | The Errors table becomes 7 chunks (one row each at a 70-token budget) and each starts with the header row `\| HTTP status ...` | `split_block` repeats the 2-line header in every row group |
| `breadcrumb=False` | Overall dense hit@1 falls from 0.78 to 0.73, but on the 4 `reference` questions it stays 0.75 in both runs (measured: [solutions](../solutions/solutions.md)); the slice is too small to show the effect | Without the heading words the vector for "It must be approved by the manager" has no topic |
| `StructureChunker(400, 40)` | Fewer, larger chunks, one per section; context tokens per query rise | The budget is per chunk; more of each section fits |
| `min_tokens=0` | No tiny-section merging: more chunks of 10 to 30 tokens | `merge_tiny` never fires |
| `max_tokens=100` instead of 200 (section E, last row) | More chunks, lower context tokens, quality changes within noise | A smaller budget splits more sections; breadcrumbs repeat in each piece |

## Debugging practice
**BREAK IT:** delete the fenced-code branch (`elif stripped.startswith("```"): ...`) in `parse_blocks`.

Symptom: section A's last line prints `True` for "parsed as a heading?": the `# Report a suspected incident...` comment inside the ` ```bash ` fence becomes a level-1 heading, closes every open section, and the paragraph after it (the SEV1 text) lands under a section that does not exist in the document. Nothing crashes. **Fix:** restore the fence branch *before* the heading branch, and test with a fixture that has a `#` comment inside a code fence. A real parser has the same trap with `~~~` fences and with fences indented inside list items.

## Common mistakes
- Letting a regex treat `# comment` inside code as a heading.
- Splitting a table by tokens (headerless fragments are meaningless: "99.97 | 99.99" with no column names).
- Putting the breadcrumb in `text`: offsets, citations and the displayed quote then contain text that is not in the source. Keep source and decoration separate.
- Counting the breadcrumb's tokens *after* the budget decision: the embedded text silently exceeds `max_tokens`.
- Merging tiny sections without keeping their headings: the merged text loses what it was about (here the label stays in the text).
- Believing the breadcrumb helps equally everywhere: on a headingless document (the FAQ, the postmortem) there is nothing to prepend.
- Using one maximum for all content: a code block of 300 tokens may be better whole than split to fit a 200 limit.

## Rebuild checklist
1. `HEADING` and `LIST_ITEM` regexes; `Block` and `Section` dataclasses (`path()` recursive).
2. `parse_blocks`: line offsets, then in order: blank, fence, heading, table, list, paragraph.
3. `build_tree`: stack; the first `# ` is the root; pop while `stack[-1].level >= b.level`.
4. `pack` (greedy by tokens of the joined text).
5. `split_block`: paragraph/list by sentences; table by rows with the header repeated; code by lines, re-fenced.
6. `section_units`, `merge_tiny` (same parent, both small, keep the heading inside the text and all spans).
7. `split_document`: breadcrumb into `embed_text`/`context_text` only, budget minus crumb tokens.
8. Section D: count odd fences and headerless table fragments.

## Say it aloud
"I parse the document into blocks and a heading tree with a stack, then chunk by section: tables, code fences and lists are atomic, an oversize block is split only along its own structure with the table header repeated, and tiny sibling sections are merged. Every chunk carries a breadcrumb, `Doc > Section > Subsection`, in the text I embed and in the text I send to the LLM, but not in the stored source slice, so offsets and citations stay exact. A generic splitter cuts code fences and table rows; the structure-aware version cut none in the test bed. The breadcrumb matters most for chunks whose meaning depends on the heading: pronouns, short answers, tables."

## The new problem (cliffhanger)
Structure helps only when the author wrote structure. The incident postmortem in our test bed is flowing prose with no headings at all, and a heading tree cannot cut it. Could the *embedding model itself* tell us where the topic changes? Next: embed every sentence and cut where the distance between neighbours spikes: [step 04, semantic chunking](../04_semantic_chunking/WALKTHROUGH.md).
