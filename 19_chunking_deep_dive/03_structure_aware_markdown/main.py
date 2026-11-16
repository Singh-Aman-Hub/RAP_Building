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


if __name__ == "__main__":
    main()

# TRY IT: StructureChunker(max_tokens=60, min_tokens=10) on api_reference: how many chunks does the error table become?
# TRY IT: set breadcrumb=False and compare the "reference"-kind questions in the retrieval table.
# BREAK IT: delete the fenced-code branch in parse_blocks -> '# Report a suspected...' becomes a fake heading (section A shows True).
