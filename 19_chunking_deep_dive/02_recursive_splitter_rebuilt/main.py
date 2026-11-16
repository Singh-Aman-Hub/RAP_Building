"""
GOAL:  Rebuild the LangChain-style RecursiveCharacterTextSplitter from first principles, with the token counter as
       its length function: separator hierarchy, merge-with-overlap logic, keep_separator, language-specific
       separators (markdown, python). Compare it with the 40-line version in module 14 step 02.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 19_chunking_deep_dive/02_recursive_splitter_rebuilt/main.py
EXPECTED OUTPUT (shortened):
    A) which separator level did the work (counts) for the handbook
    B) keep_separator False / "start" / "end": how the first chunk-starts differ
    C) overlap really is <= chunk_overlap tokens; sizes never exceed chunk_size
    D) python separators cut a code string at def boundaries
    E) retrieval: module-14 version vs rebuilt (generic / markdown separators), 74 questions
NEEDS: torch, transformers (cached MiniLM).  ~15 s.
"""
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common19 import RawChunk, count_tokens, evaluate, load_docs, table  # noqa: E402

# Separator hierarchies (same idea as LangChain's Language.MARKDOWN / Language.PYTHON; regexes, biggest first).
GENERIC = ["\n\n", "\n", " ", ""]
MARKDOWN = [r"\n#{1,6} ", r"```\n", r"\n\*\*\*+\n", r"\n---+\n", r"\n___+\n", "\n\n", "\n", " ", ""]
PYTHON = [r"\nclass ", r"\ndef ", r"\n\tdef ", "\n\n", "\n", " ", ""]


def split_keep(text, sep, keep):
    """Split `text` on `sep` (a regex). keep = False | "start" | "end": which neighbour keeps the separator."""
    if sep == "":
        return list(text)                                    # last resort: one piece per character
    if not keep:
        pieces = re.split(sep, text)
    else:
        parts = re.split(f"({sep})", text)                   # capture group -> separators stay in the list
        if keep == "end":                                    # [t0, s1, t1, s2, t2] -> [t0+s1, t1+s2, t2]
            pieces = [parts[i] + parts[i + 1] for i in range(0, len(parts) - 1, 2)] + [parts[-1]]
        else:                                                # "start": [t0, s1+t1, s2+t2]
            pieces = [parts[0]] + [parts[i] + parts[i + 1] for i in range(1, len(parts), 2)]
    return [p for p in pieces if p != ""]


class RecursiveSplitter:
    def __init__(self, chunk_size=128, chunk_overlap=0, length_function=count_tokens,
                 separators=None, keep_separator="start", strip_whitespace=True):
        assert 0 <= chunk_overlap < chunk_size
        self.size, self.overlap, self.length = chunk_size, chunk_overlap, length_function
        self.separators = separators or GENERIC
        self.keep, self.strip = keep_separator, strip_whitespace
        self.used = Counter()                                # how often each separator level did the splitting

    # -- step 1: choose the separator and recurse ---------------------------------------------------------
    def split_text(self, text, separators=None):
        separators = self.separators if separators is None else separators
        sep, rest = separators[-1], []                       # default: the last (finest) separator
        for i, cand in enumerate(separators):
            if cand == "" or re.search(cand, text):          # first separator that actually occurs in the text
                sep, rest = cand, separators[i + 1:]
                break
        self.used[sep] += 1
        pieces = split_keep(text, sep, self.keep)
        joiner = "" if self.keep else sep                    # a kept separator is already inside the pieces
        chunks, good = [], []                                # `good` = consecutive small pieces waiting to be merged
        for p in pieces:
            if self.length(p) < self.size:
                good.append(p)
                continue
            if good:                                         # a big piece ends the run of small ones: merge them
                chunks += self.merge(good, joiner)
                good = []
            if not rest:
                chunks.append(p)                             # nothing finer left: keep oversize piece as is
            else:
                chunks += self.split_text(p, rest)           # recurse into the oversize piece with finer separators
        if good:
            chunks += self.merge(good, joiner)
        return chunks

    # -- step 2: glue small pieces into chunks of <= size tokens, carrying `overlap` tokens into the next chunk
    def merge(self, pieces, joiner):
        jlen = self.length(joiner)
        docs, cur, total = [], [], 0
        for p in pieces:
            plen = self.length(p)
            if total + plen + (jlen if cur else 0) > self.size:      # adding p would overflow: close the chunk
                if cur:
                    docs.append(self._join(cur, joiner))
                    # drop pieces from the FRONT until what remains is <= overlap (and p fits)
                    while total > self.overlap or (total + plen + (jlen if cur else 0) > self.size and total > 0):
                        total -= self.length(cur[0]) + (jlen if len(cur) > 1 else 0)
                        cur = cur[1:]
            cur.append(p)
            total += plen + (jlen if len(cur) > 1 else 0)
        if cur:
            docs.append(self._join(cur, joiner))
        return [d for d in docs if d]

    def _join(self, parts, joiner):
        text = joiner.join(parts)
        return text.strip() if self.strip else text

    # -- step 3: strings -> RawChunk with character offsets in the original document ----------------------------
    def split_document(self, doc):
        text, chunks, cursor = doc["text"], [], 0
        for c in self.split_text(text):
            i = text.find(c, max(cursor - len(c), 0))        # overlap means the next chunk may start before `cursor`
            i = text.find(c) if i == -1 else i               # (fallback for pathological separators)
            chunks.append(RawChunk(doc["doc_id"], i, i + len(c), c))
            cursor = i + 1
        return chunks


def recursive_split_14(text, max_chars=250, separators=("\n\n", "\n", ". ", " ")):
    """Module 14 step 02's version, copied: characters, no overlap, no offsets, no token counter."""
    if len(text) <= max_chars:
        return [text.strip()] if text.strip() else []
    sep, *rest = separators
    parts = text.split(sep)
    if len(parts) == 1:
        return recursive_split_14(text, max_chars, tuple(rest)) if rest else [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
    chunks, current = [], ""
    for part in parts:
        cand = current + sep + part if current else part
        if len(cand) <= max_chars:
            current = cand
        else:
            if current:
                chunks.append(current)
            current = part if len(part) <= max_chars else ""
            if len(part) > max_chars:
                chunks.extend(recursive_split_14(part, max_chars, tuple(rest)) if rest else [part])
    if current:
        chunks.append(current)
    return [c.strip() for c in chunks if c.strip()]


def main():
    docs = load_docs()
    hb = next(d for d in docs if d["doc_id"] == "employee_handbook")

    # ---- A) which separator did the work ---------------------------------------------------------------------
    sp = RecursiveSplitter(48, 0)
    chunks = sp.split_document(hb)
    print(f"A) handbook, size=48 tokens, generic separators -> {len(chunks)} chunks; levels used: "
          f"{{{', '.join(f'{k!r}: {v}' for k, v in sp.used.items())}}}")
    sp = RecursiveSplitter(48, 0, separators=MARKDOWN)
    chunks = sp.split_document(hb)
    print(f"   same with MARKDOWN separators -> {len(chunks)} chunks; levels used: "
          f"{{{', '.join(f'{k!r}: {v}' for k, v in sp.used.items())}}}")

    # ---- B) keep_separator ---------------------------------------------------------------------------------------
    print("\nB) keep_separator on the first 3 chunks of the handbook (markdown separators, size=64): where does '## ' go?")
    for keep in (False, "start", "end"):
        cs = RecursiveSplitter(64, 0, separators=MARKDOWN, keep_separator=keep).split_text(hb["text"])
        print(f"   keep={keep!s:<6}", " | ".join(repr(c[:22]) for c in cs[:3]), "...", repr(cs[1][-14:]))

    # ---- C) overlap and size guarantees ------------------------------------------------------------------------------
    sp = RecursiveSplitter(96, 24)
    cs = sp.split_document(hb)
    sizes = [count_tokens(c.text) for c in cs]
    shared = []
    for a, b in zip(cs, cs[1:]):                              # tokens that chunk a and b share
        lo, hi = b.start, min(a.end, b.end)
        shared.append(count_tokens(hb["text"][lo:hi]) if hi > lo else 0)
    print(f"\nC) size=96 overlap=24: {len(cs)} chunks, token sizes min {min(sizes)} / max {max(sizes)}; "
          f"shared tokens between neighbours: max {max(shared)}, mean {sum(shared) / len(shared):.1f}")
    print(f"   stored tokens {sum(sizes)} for {count_tokens(hb['text'])} real tokens = {sum(sizes) / count_tokens(hb['text']):.2f}x")

    # ---- D) language-specific separators --------------------------------------------------------------------------------
    code = ("import requests\n\nclass Client:\n    def __init__(self, key):\n        self.key = key\n\n"
            "    def get(self, path):\n        return requests.get(path, headers={'k': self.key}).json()\n\n"
            "def verify(body, sig):\n    return sig == body[::-1]\n\ndef main():\n    print(Client('x').get('/ping'))\n")
    pieces = RecursiveSplitter(30, 0, separators=PYTHON, strip_whitespace=False).split_text(code)
    print(f"\nD) python separators, size=30 tokens -> {len(pieces)} pieces, each starts at a boundary:")
    for p in pieces:
        print("   ", repr(p[:46]))

    # ---- E) retrieval vs the module-14 splitter -----------------------------------------------------------------------------
    def old14(d):                                             # module 14 version cut into RawChunks (offsets via find)
        out, cur = [], 0
        for c in recursive_split_14(d["text"], 400):
            i = d["text"].find(c, cur)
            i = d["text"].find(c) if i == -1 else i
            out.append(RawChunk(d["doc_id"], i, i + len(c), c))
            cur = i + 1
        return out
    configs = {
        "module 14: 400 chars, no overlap": old14,
        "rebuilt: 100 tok, generic": lambda d: RecursiveSplitter(100, 0).split_document(d),
        "rebuilt: 100 tok, overlap 20": lambda d: RecursiveSplitter(100, 20).split_document(d),
        "rebuilt: 100 tok, markdown seps": lambda d: RecursiveSplitter(100, 0, separators=MARKDOWN).split_document(d),
        "rebuilt: 200 tok, markdown seps": lambda d: RecursiveSplitter(200, 0, separators=MARKDOWN).split_document(d),
    }
    rows = []
    for name, fn in configs.items():
        ch = [c for d in docs for c in fn(d)]
        r = evaluate(ch, docs=docs)
        d = r["dense"]
        rows.append([name, len(ch), f"{r['mean_tok']:.0f}", f"{d['hit@1']:.2f}", f"{d['hit@3']:.2f}", f"{d['mrr']:.2f}", f"{r['hybrid']['hit@3']:.2f}"])
    print("\nE) retrieval, 13 documents, 74 questions")
    print(table(rows, ["splitter", "chunks", "mean tok", "dense hit@1", "dense hit@3", "dense MRR", "hybrid hit@3"]))


if __name__ == "__main__":
    main()

# TRY IT: set keep_separator="end" in section A's splitter and check that chunk texts still concatenate back to the doc.
# TRY IT: pass length_function=len (characters) with chunk_size=400; count how many chunks exceed 128 TOKENS.
# BREAK IT: RecursiveSplitter(64, 64) -> AssertionError (overlap must be < size).
