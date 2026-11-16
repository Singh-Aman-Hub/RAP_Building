# Step 02 — The recursive splitter, rebuilt

## The story so far
[Step 01](../01_tokens_not_characters/WALKTHROUGH.md) gave us the ruler (tokens of the embedding model, with exact character offsets) and a chunker that cuts every N tokens regardless of meaning. It ended with the obvious fix: prefer natural boundaries. Back to the [module README](../README.md).

**Relation to module 14.** This step is *not* a duplicate of [14/02 chunking strategies](../../14_rag_and_retrieval/02_chunking_strategies/WALKTHROUGH.md). That step wrote a 25-line recursive splitter in characters to compare five strategies on nine documents. Here we **rebuild and extend** the same idea to the level of LangChain's `RecursiveCharacterTextSplitter`: a token-based length function, overlap carried between chunks, `keep_separator`, language-specific separator lists (Markdown, Python), and character offsets back into the source. The module-14 version is copied into this file (`recursive_split_14`) so the two can be compared on the same test bed in section E.

## The pain point
A fixed window cuts a sentence in half, and "split on `\n\n`" alone makes one chunk 20 tokens and the next 900. The two common fixes are both wrong in isolation: "split on sentences" ignores paragraphs and structure, and "split on the biggest separator only" ignores size. We want a rule that respects structure when it can and size when it must.

## The spark
Keep an ordered list of separators, from the biggest boundary to the smallest: paragraph break, line break, space, and finally nothing (single characters). Split on the first separator that actually occurs. Pieces that are small enough get **merged back together** up to the size limit; a piece that is still too big is split again, *recursively*, with the remaining finer separators. That is the whole algorithm: split, merge small, recurse into large.

## Goal
Rebuild the splitter step by step and prove its properties:
1. Which separator level did the work, for generic and Markdown separator lists (section A).
2. What `keep_separator` = `False` / `"start"` / `"end"` does to the chunk boundaries (B).
3. Sizes never exceed `chunk_size`, and overlap is at most `chunk_overlap` tokens (C).
4. A Python separator list cuts code at `class` / `def` boundaries (D).
5. Compare with the module-14 splitter on 74 questions (E).

## The theory (read this before the interview)
```
split_text(text, separators = [s1, s2, ..., ""]):
    s = the first separator that occurs in text
    pieces = text.split(s)                  (optionally keeping s attached to a neighbour)
    for each piece:
        small (< size)  -> collect into a run
        big             -> merge the run so far, then split_text(piece, separators after s)   <- recursion
    merge(run):  greedily pack consecutive pieces until adding one more would exceed `size`;
                 start the next chunk with the LAST pieces of the previous one, as many as fit in `overlap`
```
**Why recursion?** A document is a tree (sections, paragraphs, lines, words). Splitting the biggest unit first keeps whole paragraphs together; recursing only into the pieces that are too big means a small paragraph is never sliced.

**The merge step is where overlap lives.** Overlap here is made of *whole pieces* (whole lines, whole sentences), so it is at most `overlap` tokens and usually less (in section C: at most 16 tokens for a requested 24). A character splitter would cut overlap mid-word.

**`keep_separator`.** When you split on `"\n## "` the separator itself carries information (it is the heading marker). Options: drop it (`False`), attach it to the *start* of the next piece (`"start"`: the heading marker stays with its section; the default here), or to the *end* of the previous piece (`"end"`: the heading marker is left dangling at the end of the *previous* chunk, which is wrong for headings and right for sentence punctuation such as `". "`).

**Language-specific separators.** The same machinery with a different list:
```
MARKDOWN: "\n#{1,6} "  ```\n  horizontal rules  "\n\n"  "\n"  " "  ""      headings first
PYTHON:   "\nclass "   "\ndef "  "\n\tdef "  "\n\n"  "\n"  " "  ""        definitions first
```
Separators are **regexes** here (as in LangChain's `is_separator_regex=True` path), which is also why `"\n\tdef "` (a tab) silently misses methods indented with spaces: the pattern must match whatever indentation the file uses.

**Length function.** The only difference between "character splitter" and "token splitter" is the function passed as `length_function`. Passing `count_tokens` ([step 01](../01_tokens_not_characters/WALKTHROUGH.md)) makes `chunk_size=100` mean 100 tokens.

**Status.** The recursive splitter is *current industry practice* and the default in most RAG frameworks. It is a strong baseline, and the rest of the module is about when to leave it.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| recursive splitter | `RecursiveCharacterTextSplitter` (LangChain), `RecursiveTextSplitter`, hierarchical separator splitting | near | Same algorithm; LangChain also has `from_tiktoken_encoder` and `Language.*` presets |
| separator hierarchy | separator list, `separators=[...]`, delimiter cascade | yes | Ordered biggest-boundary-first list |
| chunk_size / chunk_overlap | `max_tokens` / `overlap_tokens`, window / stride | near | Size is an upper bound on the merged chunk; overlap is an upper bound on shared text |
| keep_separator | separator attachment, `"start"` / `"end"` | yes | Which neighbour keeps the delimiter |
| length function | `length_function`, token counter, sizer | yes | Anything `str -> int`; tokens, chars or words |
| language-aware splitting | `Language.PYTHON`, `Language.MARKDOWN` presets, code splitter | near | Different separator list, same algorithm |
| greedy merge | bin packing (greedy, in order), pack-to-size | near | Consecutive pieces only; order is never changed |

## Run it
```bash
cd 19_chunking_deep_dive/02_recursive_splitter_rebuilt
python main.py
# from the repo root:
HF_HUB_OFFLINE=1 .venv/bin/python 19_chunking_deep_dive/02_recursive_splitter_rebuilt/main.py
```
About 15 s of CPU time (it embeds the chunks of five configurations). In this course's run script the wall time was longer only because it waited for the shared "heavy run" lock. Needs `torch` + `transformers` and the cached MiniLM; no downloads.

## Code, step by step

### 1. Header docstring, imports  (`main.py` lines 1–22)

```python
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
```

- **In simple words:** goal and run lines, then standard-library imports (`re` for regexes, `Counter` for tallies) and the shared helpers.
- **Exactly:** `RawChunk`, `count_tokens`, `evaluate`, `load_docs`, `table` come from `common19`, explained once in the [step 01 tour](../01_tokens_not_characters/WALKTHROUGH.md#the-shared-helpers-once-a-tour-of-common19py).
- **Why it is here:** nothing new; the point is that this file needs no model except through `count_tokens` and `evaluate`.

### 2. The three separator hierarchies  (`main.py` lines 24–27)

```python
# Separator hierarchies (same idea as LangChain's Language.MARKDOWN / Language.PYTHON; regexes, biggest first).
GENERIC = ["\n\n", "\n", " ", ""]
MARKDOWN = [r"\n#{1,6} ", r"```\n", r"\n\*\*\*+\n", r"\n---+\n", r"\n___+\n", "\n\n", "\n", " ", ""]
PYTHON = [r"\nclass ", r"\ndef ", r"\n\tdef ", "\n\n", "\n", " ", ""]
```

- **In simple words:** three ordered lists, biggest boundary first.
- **Exactly:** these are Python strings used as **regular expressions**: `r"\n#{1,6} "` matches a newline, one to six `#`, a space (a heading line start); `r"```\n"` a code-fence line; `"\n\n"` is a literal blank line (also a valid regex). `""` (empty) is the last resort and means "one piece per character".
- **Why it is here:** this is the entire "language support" of the splitter: a different list per file type. LangChain ships the same lists as `Language.MARKDOWN` and `Language.PYTHON`.

### 3. `split_keep`: split and decide who keeps the separator  (`main.py` lines 30–42)

```python
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
```

- **In simple words:** cut `text` at every `sep`, and either drop the separator or glue it to the start of the next piece or the end of the previous one.
- **Exactly:** `re.split(sep, text)` returns the pieces without the separator. Wrapping the pattern in a capture group, `re.split(f"({sep})", text)`, makes the separators part of the list: `[t0, s1, t1, s2, t2]`. For `"end"` each text is paired with the separator after it (`t0+s1`, `t1+s2`, `t2`); for `"start"` each separator is paired with the text after it (`t0`, `s1+t1`, `s2+t2`). Empty pieces are removed. With `sep == ""` the function returns `list(text)`, i.e. single characters.
- **Why it is here:** this tiny function decides whether a heading marker stays with its section. Either choice concatenates back to the original text (checked in the experiments below).

### 4. `RecursiveSplitter.__init__`  (`main.py` lines 45–52)

```python
class RecursiveSplitter:
    def __init__(self, chunk_size=128, chunk_overlap=0, length_function=count_tokens,
                 separators=None, keep_separator="start", strip_whitespace=True):
        assert 0 <= chunk_overlap < chunk_size
        self.size, self.overlap, self.length = chunk_size, chunk_overlap, length_function
        self.separators = separators or GENERIC
        self.keep, self.strip = keep_separator, strip_whitespace
        self.used = Counter()                                # how often each separator level did the splitting
```

- **In simple words:** store the size, the overlap, the length function and the separators.
- **Exactly:** `assert 0 <= chunk_overlap < chunk_size` rejects an overlap that would stop progress. `length_function` defaults to `count_tokens`, so a `chunk_size` of 100 means 100 tokens; pass `len` for characters. `Counter()` named `used` counts how many times each separator level did the work (section A reads it).
- **Why it is here:** the constructor is the whole configuration surface; step 11's `ChunkConfig` turns exactly these fields into a frozen config object.

### 5. `split_text`: choose the separator, split, recurse  (`main.py` lines 54–79)

```python
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
```

- **In simple words:** find the biggest separator that exists in this text, cut on it, keep small pieces for merging, and recurse into pieces that are still too big using the finer separators.
- **Exactly:** the `for` loop returns the first candidate separator for which `re.search(cand, text)` finds a match (the empty string always qualifies); `rest` is the list of finer separators. `self.used[sep] += 1` records the choice. `joiner` is `""` when a separator is kept (it is already inside the pieces), otherwise the separator text itself. `good` accumulates consecutive pieces that are `< self.size` tokens. When a big piece appears, `merge(good, joiner)` closes the run, and the big piece is either appended as is (nothing finer left, `not rest`) or split by `split_text(p, rest)`: the recursion. Whatever is left in `good` at the end is merged too. The return value is a list of strings.
- **Why it is here:** this is the algorithm. Note the two edge behaviours: a piece that no separator can shrink is kept oversize (so `chunk_size` is a target for the splitting, not a guarantee in the pathological case), and when `keep_separator=False` and the separator is a *regex*, the joiner is the pattern text itself. That is a known wart of this design: in our check, the literal text `#{1,6}` ended up in 6 of 23 chunks of the handbook with `keep_separator=False` and Markdown separators. Use `"start"` (the default here) for regex separators.

### 6. `merge` and `_join`: pack pieces with overlap  (`main.py` lines 81–102)

```python
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
```

- **In simple words:** glue small pieces into chunks of at most `size` tokens. When a chunk is full, start the next one with the last few pieces of the old one, as many as fit in the overlap budget.
- **Exactly:** `total` tracks the token length of `cur`, including `jlen` (the length of the joiner) between pieces. If adding `p` would exceed `size`, the finished chunk is stored and the `while` loop drops pieces **from the front** of `cur` until what remains is at most `overlap` tokens *and* the new piece fits. Then `p` is appended. `_join` joins the parts and strips outer whitespace if `strip_whitespace` is true. Overlap is therefore a set of whole pieces, never a partial piece.
- **Why it is here:** the only place overlap is created. Dropping from the front keeps the *end* of the previous chunk, the text closest to the cut, as the shared context.

### 7. `split_document`: strings to `RawChunk` with offsets  (`main.py` lines 104–112)

```python
    # -- step 3: strings -> RawChunk with character offsets in the original document ----------------------------
    def split_document(self, doc):
        text, chunks, cursor = doc["text"], [], 0
        for c in self.split_text(text):
            i = text.find(c, max(cursor - len(c), 0))        # overlap means the next chunk may start before `cursor`
            i = text.find(c) if i == -1 else i               # (fallback for pathological separators)
            chunks.append(RawChunk(doc["doc_id"], i, i + len(c), c))
            cursor = i + 1
        return chunks
```

- **In simple words:** turn the list of chunk strings into chunk records that know where they came from.
- **Exactly:** each chunk string is searched in the document with `text.find(c, start)`. The search starts at `max(cursor - len(c), 0)` because overlap means a chunk may begin *before* the end of the previous one. If the string cannot be found (separators were dropped and pieces were re-joined) the fallback searches from the beginning. `cursor = i + 1` guarantees the next search moves forward.
- **Why it is here:** the evaluation harness needs character offsets. This `find`-based mapping is simple but assumes chunk text occurs in the source verbatim, which holds for `keep_separator="start"/"end"` and for strip-only differences; step 11's production chunker tracks offsets directly instead of searching.

### 8. `recursive_split_14`: the module-14 splitter, copied  (`main.py` lines 115–136)

```python
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
```

- **In simple words:** the 25-line version from [14/02](../../14_rag_and_retrieval/02_chunking_strategies/WALKTHROUGH.md), kept here unchanged for comparison.
- **Exactly:** characters only, separators `("\n\n", "\n", ". ", " ")`, no overlap, no offsets, no length function, and the separator is dropped. It splits on the first separator, packs parts up to `max_chars`, and recurses into parts that are too long.
- **Why it is here:** the same idea at a different level of completeness: section E shows what the extra machinery buys.

### 9. Section A: which separator level did the work  (`main.py` lines 139–151)

```python
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
```

- **In simple words:** split the handbook into 48-token chunks with generic and with Markdown separators and print how many times each level was used.
- **Exactly:** `sp.used` is the `Counter`; `{k!r}` prints the separator with Python quoting. Counting happens on every call of `split_text`, including recursive calls.
- **Why it is here:** it shows the recursion at work: the generic list uses `\n\n` once, `\n` five times and the space four times, because the long paragraphs need finer cuts; the Markdown list finds headings first.

### 10. Section B: `keep_separator`  (`main.py` lines 153–157)

```python
    # ---- B) keep_separator ---------------------------------------------------------------------------------------
    print("\nB) keep_separator on the first 3 chunks of the handbook (markdown separators, size=64): where does '## ' go?")
    for keep in (False, "start", "end"):
        cs = RecursiveSplitter(64, 0, separators=MARKDOWN, keep_separator=keep).split_text(hb["text"])
        print(f"   keep={keep!s:<6}", " | ".join(repr(c[:22]) for c in cs[:3]), "...", repr(cs[1][-14:]))
```

- **In simple words:** print the first three chunk starts and the end of the second chunk for the three settings.
- **Exactly:** `cs[:3]` are the first three chunk strings, shortened to 22 characters by `c[:22]`; `cs[1][-14:]` shows how chunk 2 ends. `{keep!s:<6}` formats the setting left-aligned.
- **Why it is here:** with `False` the `###` / `####` heading markers vanish; with `"start"` they stay at the start of their section; with `"end"` the marker `####` is stuck at the end of the *previous* chunk (`'ote work\n\n####'`), a chunk that now ends with a heading that belongs to the next one.

### 11. Section C: size and overlap guarantees  (`main.py` lines 159–169)

```python
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
```

- **In simple words:** check that no chunk exceeds 96 tokens, and measure how many tokens neighbours really share.
- **Exactly:** `shared` counts the tokens in the character range where chunk `b` starts before chunk `a` ends (`hi = min(a.end, b.end)`). `stored tokens / real tokens` is the index-size cost of the overlap.
- **Why it is here:** the overlap you ask for is an upper bound; here 24 requested gives at most 16 shared and 8.7 on average, because overlap is made of whole pieces.

### 12. Section D: language-specific separators  (`main.py` lines 171–178)

```python
    # ---- D) language-specific separators --------------------------------------------------------------------------------
    code = ("import requests\n\nclass Client:\n    def __init__(self, key):\n        self.key = key\n\n"
            "    def get(self, path):\n        return requests.get(path, headers={'k': self.key}).json()\n\n"
            "def verify(body, sig):\n    return sig == body[::-1]\n\ndef main():\n    print(Client('x').get('/ping'))\n")
    pieces = RecursiveSplitter(30, 0, separators=PYTHON, strip_whitespace=False).split_text(code)
    print(f"\nD) python separators, size=30 tokens -> {len(pieces)} pieces, each starts at a boundary:")
    for p in pieces:
        print("   ", repr(p[:46]))
```

- **In simple words:** cut a small Python file with the `PYTHON` list.
- **Exactly:** `strip_whitespace=False` keeps the leading newline so each piece visibly starts at its boundary. Pieces begin with `\nclass`, `\ndef` or `\n\n    def` (the last is a `\n\n` split, because the methods are indented with spaces and the `"\n\tdef "` pattern needs a tab).
- **Why it is here:** one list swap gives a code-aware splitter. The `\n\tdef` miss is the classic bug; the fix is `r"\n[ \t]*def "` (see Debugging practice).

### 13. Section E: retrieval against the module-14 splitter  (`main.py` lines 180–203)

```python
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
```

- **In simple words:** score five configurations on the 74 questions: the old module-14 splitter and four settings of the rebuilt one.
- **Exactly:** `old14` wraps the module-14 function to produce `RawChunk` offsets by `find`. `configs` maps a label to a function `doc -> chunks`; `lambda d: ...` closes over the settings. `r['mean_tok']` is the average tokens per chunk; `r['hybrid']['hit@3']` is the BM25 + dense (RRF) score.
- **Why it is here:** an honest comparison. Read the numbers in the trace before drawing conclusions.

### 14. Entry point and the TRY IT / BREAK IT lines  (`main.py` lines 206–211)

```python
if __name__ == "__main__":
    main()

# TRY IT: set keep_separator="end" in section A's splitter and check that chunk texts still concatenate back to the doc.
# TRY IT: pass length_function=len (characters) with chunk_size=400; count how many chunks exceed 128 TOKENS.
# BREAK IT: RecursiveSplitter(64, 64) -> AssertionError (overlap must be < size).
```

- **In simple words:** run `main()` only when executed; the comments are the exercises.
- **Exactly:** `RecursiveSplitter(64, 64)` fails at the first `assert`.
- **Why it is here:** step 10 imports this class with `load_step`, so the guard keeps the experiments from running on import.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `re.split(f"({sep})", text)` | Split and keep the matches because the pattern has a capture group |
| `[parts[i] + parts[i + 1] for i in range(0, len(parts) - 1, 2)]` | Pair every text with the separator after it (stride 2) |
| `for i, cand in enumerate(separators): ... break` | Linear search for the first separator that occurs |
| `chunks += self.merge(good, joiner)` | Extend a list in place with another list |
| `self.length(p)` | Call the injected length function (default `count_tokens`) |
| `while total > self.overlap or (...)` | Drop pieces from the front until the leftover fits the overlap budget |
| `Counter()` + `self.used[sep] += 1` | Dictionary that counts, defaulting to 0 |
| `lambda d: RecursiveSplitter(100, 0).split_document(d)` | Anonymous function used as a configuration |
| `{keep!s:<6}` | `!s` converts to `str`, `<6` pads left-aligned to width 6 |

## Libraries & functions
| Function | What it does | Internally |
|---|---|---|
| `re.search(pattern, text)` | First regex match or `None` | Compiles (cached) and scans |
| `re.split(pattern, text)` | List of pieces between matches | Returns the capture groups too when the pattern has any |
| `str.find(sub, start)` | Index of the first occurrence at or after `start` | Plain substring search, `-1` if absent |
| `collections.Counter` | Tally dictionary | `dict` subclass with default count 0 |
| `evaluate` | Score chunks on 74 questions | See the [step 01 tour](../01_tokens_not_characters/WALKTHROUGH.md#the-shared-helpers-once-a-tour-of-common19py) |

## Execution trace (real run)
Real output of `python main.py` (the model-loading progress bar is removed; timings vary by machine).
```
A) handbook, size=48 tokens, generic separators -> 33 chunks; levels used: {'\n\n': 1, '\n': 5, ' ': 4}
   same with MARKDOWN separators -> 39 chunks; levels used: {'\\n#{1,6} ': 1, '\n\n': 9, '\n': 4, ' ': 4, '```\\n': 1}

B) keep_separator on the first 3 chunks of the handbook (markdown separators, size=64): where does '## ' go?
   keep=False  '# Lumen Cloud Employee' | '1.1 Working hours\n\nCor' | '1.2.1 Eligibility\n\nEve' ... '.2 Remote work'
   keep=start  '# Lumen Cloud Employee' | '### 1.1 Working hours\n' | '#### 1.2.1 Eligibility' ... '.2 Remote work'
   keep=end    '# Lumen Cloud Employee' | '1.1 Working hours\n\nCor' | '1.2.1 Eligibility\n\nEve' ... 'ote work\n\n####'

C) size=96 overlap=24: 16 chunks, token sizes min 35 / max 95; shared tokens between neighbours: max 16, mean 8.7
   stored tokens 1176 for 1046 real tokens = 1.12x

D) python separators, size=30 tokens -> 6 pieces, each starts at a boundary:
    'import requests\n'
    '\nclass Client:\n    def __init__(self, key):\n  '
    '\n\n    def get(self, path):'
    "\n        return requests.get(path, headers={'k"
    '\ndef verify(body, sig):\n    return sig == body'
    "\ndef main():\n    print(Client('x').get('/ping'"


E) retrieval, 13 documents, 74 questions
splitter                          chunks  mean tok  dense hit@1  dense hit@3  dense MRR  hybrid hit@3
--------------------------------  ------  --------  -----------  -----------  ---------  ------------
module 14: 400 chars, no overlap      70        71         0.76         0.92       0.84          0.97
rebuilt: 100 tok, generic             68        73         0.74         0.91       0.83          0.95
rebuilt: 100 tok, overlap 20          71        74         0.77         0.91       0.85          0.96
rebuilt: 100 tok, markdown seps       71        70         0.78         0.92       0.86          0.96
rebuilt: 200 tok, markdown seps       35       142         0.72         0.92       0.81          0.96
```

How to read it:
- **A:** the generic list used `\n\n` once and `\n` and space several times: the recursion went down only where paragraphs were too long. With Markdown separators there are 39 chunks (the heading separator and code fence were used; paragraphs `\n\n` nine times).
- **B:** `keep=False` drops `###`; `"start"` keeps `### 1.1 Working hours` at the head of its chunk; `"end"` leaves `####` dangling at the end of the previous chunk (`'ote work\n\n####'`).
- **C:** every chunk is `<= 95` tokens (limit 96), but neighbours share at most 16 tokens (mean 8.7) although 24 were requested, and storage is 1.12x the text.
- **D:** six pieces; the file is cut at `class` and `def` boundaries, but the indented methods are separated only by `\n\n` (the tab pattern does not match spaces).
- **E:** hit@3 for dense retrieval is 0.91 to 0.92 for all five configurations, and hybrid is 0.95 to 0.97. The rebuilt splitter with token sizes and Markdown separators gains a little in hit@1 (0.78 vs 0.76) and MRR (0.86 vs 0.84), which is one question or two, inside the noise ([step 10](../10_chunking_evaluation/WALKTHROUGH.md)). The honest reading: on clean documents the recursive splitter is as good as the module-14 version; its value is guaranteed token sizes, overlap, offsets and the ability to switch separator lists.

## Experiments (change ONE thing)
| Change | What you will see | Why |
|---|---|---|
| `keep_separator=False` with `MARKDOWN` | The literal text `#{1,6}` appears inside 6 of 23 chunks of the handbook (size 64) | The joiner becomes the regex source when the separator is not kept: a bug-shaped wart; use `"start"` |
| `keep_separator="end"` or `"start"` in section A's splitter | `"".join(chunks) == text` holds in both cases (with `strip_whitespace=False`); 37 pieces for size 48 | Kept separators are still inside the pieces, nothing is lost |
| `RecursiveSplitter(400, 0, length_function=len)` on the handbook | 15 chunks and none above 128 tokens | 400 characters is about 90 to 100 tokens on this prose. On JSON or code the same 400 characters would exceed the limit |
| `RecursiveSplitter(96, 0)` vs `(96, 24)` | With overlap 24 the stored tokens are 1.12x the text (trace C); sizes stay `<= 96` | Overlap pieces are copied into the next chunk |
| Replace `r"\n\tdef "` by `r"\n[ \t]*def "` in `PYTHON` and print `sp.used` on a space-indented class | Same 3 pieces in a small example, but `used` is `{'\n\n': 1}` for the original and `{'\\n[ \\t]*def ': 1}` for the fixed list: the tab pattern is never selected, so method boundaries are found only by accident (blank lines) | The pattern must match the file's indentation; the effect shows when methods are not separated by blank lines or exceed the budget (solution: [solutions](../solutions/solutions.md)) |

## Debugging practice
**BREAK IT:** `RecursiveSplitter(64, 64)`.
```
AssertionError
```
The `assert` has no message, so the traceback shows only the failing line `assert 0 <= chunk_overlap < chunk_size`. A better version raises `ValueError(f"overlap {overlap} must be < size {size}")`; [step 11](../11_chunker_lld/WALKTHROUGH.md) does exactly that in `ChunkConfig.__post_init__`.

**Second bug to look at:** set `keep_separator="end"` and print the first chunk boundaries of the Markdown splitter. Each heading marker (`####`) is attached to the end of the chunk *before* its heading, so a chunk ends with a heading that belongs to the next section. **Fix:** use `"start"` for heading-like separators and `"end"` only for sentence punctuation.

## Common mistakes
- Treating `chunk_size` as a hard cap. A piece that no separator can split (a 600-token line of base64) stays oversize.
- Using `keep_separator=False` with regex separators and then seeing separator text in the output.
- Assuming overlap equals `chunk_overlap`. It is at most that, in whole pieces.
- Writing the Python separator for one indentation style (tab) and running it on another (spaces).
- Using characters as the length function and the embedder's limit in tokens ([step 01](../01_tokens_not_characters/WALKTHROUGH.md)).
- Believing a recursive splitter understands headings. It sees only separators; a heading and its paragraph can still land in different chunks (fixed in [step 03](../03_structure_aware_markdown/WALKTHROUGH.md)).
- Mapping chunks back to offsets with `find` when the text has repeated passages: the first match may be the wrong one.

## Rebuild checklist
1. `split_keep(text, sep, keep)` with a capture group, `"start"` and `"end"` pairing, drop empty pieces, `""` returns characters.
2. `split_text`: pick the first separator present, split, accumulate small pieces, merge on a big piece, recurse with `separators[i+1:]`, merge the remainder.
3. `merge(pieces, joiner)`: greedy pack by token length; on overflow, store the chunk and drop from the front until the leftover is `<= overlap` and the new piece fits.
4. `split_document`: offsets by `find` from `cursor - len(c)`.
5. Check: no chunk above `size`, shared tokens at most `overlap`, `"".join(chunks) == text` for kept separators.
6. Build `MARKDOWN` and `PYTHON` lists (regexes, biggest first).

## Say it aloud
"A recursive splitter tries the biggest separator first, merges the small pieces back up to the size limit, and recurses into any piece that is still too big with the next finer separator, so it respects paragraphs and lines when it can and falls back to words and characters only when it must. I give it the embedder's token counter as the length function, and I choose `keep_separator='start'` so heading markers stay with their section. Overlap is made of whole pieces, so it is at most the requested amount. On clean documents it is a strong baseline, as good as the character version, and in my test bed the differences were inside the noise. The value is predictable sizes and offsets, not magic recall."

## The new problem (cliffhanger)
The splitter sees `\n#{1,6} ` as just another regex. It does not know that a heading *owns* the text under it, that a table must never be cut between rows, or that a code fence is one unit. In the run above `'ote work\n\n####'` shows a heading marker severed from its section. Next: parse the Markdown into blocks and a heading tree and chunk by section, with a breadcrumb: [step 03, structure-aware Markdown](../03_structure_aware_markdown/WALKTHROUGH.md).
