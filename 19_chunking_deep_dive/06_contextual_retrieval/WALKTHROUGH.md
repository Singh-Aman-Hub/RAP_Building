# Step 06 — Contextual retrieval

## The story so far
[Step 05](../05_late_chunking/WALKTHROUGH.md) put context into the *vector* of each chunk by pooling after a long forward pass. BM25 and the LLM that reads the chunk got nothing, because the chunk *text* was unchanged. Back to the [module README](../README.md).

## The pain point
A retrieved chunk often cannot say what it is about. Take "The company's revenue grew by 3% over the previous quarter." Which company? Which quarter? Or in our corpus: "Each block must be at least five days." A dense retriever has to guess from a weak vector; BM25 cannot match a query that says "secondary caregiver" because those words are not in the chunk; and the LLM that receives the chunk may answer about the wrong thing.

## The spark
Let an LLM **write the missing context in words** for every chunk and store it with the chunk. The LLM is shown the whole document and the chunk, and asked for one or two sentences that situate the chunk in the document. That sentence is prepended to the chunk text, and the combination is used in two places: the text that is embedded (**contextual embeddings**) and the text that is BM25-indexed (**contextual BM25**). This is Anthropic's *Contextual Retrieval* (September 2024). It is also a technique for the LLM-facing text: the prepended context can be sent along with the chunk.

## Goal
1. See the prompt, and three chunks with their generated context (A).
2. Measure the four variants: baseline, contextual embeddings only, contextual BM25 only, both (B).
3. Test the case where it should matter most: the chunker never saw the headings (B2).
4. Count the ingestion cost with and without prompt caching (C).

A deterministic **mock** stands in for the LLM so everything runs offline and for free. The same prompt goes to a real model if you set `LLM_BACKEND=anthropic` and an API key (optional, never needed for this course).

## The theory (read this before the interview)
```
for every chunk c of document D:
    ctx  = LLM( <document> D </document>  <chunk> c </chunk>  "give a short succinct context to situate this chunk ..." )
    embed_text = ctx + "\n" + c      -> dense vector (contextual embedding)
    bm25_text  = ctx + "\n" + c      -> BM25 index   (contextual BM25)
    stored/sent to LLM: c (optionally with ctx)
```
**Reported effect (Anthropic's post, their datasets):** contextual embeddings cut the top-20 retrieval *failure rate* by 35% (5.7% to 3.7%); adding contextual BM25 cut it by 49% (to 2.9%); adding a re-ranker on top, 67% (to 1.9%). These are the vendor's numbers on their own datasets, quoted to show scale; the run below uses a tiny corpus.

**Why both indexes?** Embeddings capture meaning but blur exact tokens; BM25 captures exact tokens but has no meaning. The context text adds the *document's vocabulary* (title, section, entity names) to both.

**Cost.** One LLM call per chunk, and every call includes the whole document, so uncached input tokens are `chunks x (document + chunk + prompt)`. **Prompt caching** (a provider feature: the document prefix is stored after the first call) makes later calls read the document from cache at a fraction of the price. With Anthropic's published multipliers (a 5-minute cache write is 1.25x the base input price, a cache read is 0.10x), the document part of the bill falls to about `1.25 + 0.10 x (chunks - 1)` document-lengths instead of `chunks` document-lengths, so a long document with many chunks approaches a 10x cheaper input bill. Check the provider's current pricing; the formula is the lasting part. This is a **one-off ingestion cost**, paid again only when a document changes.

**When it matters most.** Chunks that lost their headings (flat text exports, PDF extraction), pronoun-heavy text, short chunks. For a structure-aware chunker whose breadcrumb already carries the heading ([step 03](../03_structure_aware_markdown/WALKTHROUGH.md)), a large part of the gain is already captured for free. The honest comparison is against a *cheap deterministic header*, not against nothing.

**Mock caveat.** The mock here builds the context from the document's heading path (or the preceding sentence). A real LLM can also resolve what a chunk refers to ("the March checkout outage") from anywhere in the document, which the mock cannot. Real gains differ.

**Status.** Contextual retrieval is *current industry practice* for high-value corpora (cost is acceptable, quality matters). Contextual chunk headers by rules are the cheap alternative. Related: Anthropic describes it as a preprocessing step to combine with a re-ranker.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| contextual retrieval | contextual chunk augmentation, chunk context enrichment, "situating" chunks with an LLM | near | An LLM writes the prefix; Anthropic popularised the name |
| contextual embeddings | context-prefixed embeddings | yes | The prefix is embedded together with the chunk |
| contextual BM25 | context-prefixed lexical index | yes | The same text goes into the BM25 index |
| contextual chunk headers | breadcrumb, title / section prefix, heading path | near | Deterministic prefix from metadata, no LLM ([step 03](../03_structure_aware_markdown/WALKTHROUGH.md)) |
| prompt caching | prefix caching, context caching (provider terms differ) | near | Provider stores a repeated prompt prefix so repeats are cheap; a different feature from the server-side KV prefix cache of an inference engine |
| document summary prepend | doc-level context | no | One summary per document, not one context per chunk |
| late chunking | chunk-after-embed | no | Context in vector space, [step 05](../05_late_chunking/WALKTHROUGH.md); no extra text, no LLM calls |

## Run it
```bash
cd 19_chunking_deep_dive/06_contextual_retrieval
python main.py
# from the repo root:
HF_HUB_OFFLINE=1 .venv/bin/python 19_chunking_deep_dive/06_contextual_retrieval/main.py
# optional, real LLM (costs money, never required):
# LLM_BACKEND=anthropic ANTHROPIC_API_KEY=... HF_HUB_OFFLINE=1 .venv/bin/python 19_chunking_deep_dive/06_contextual_retrieval/main.py
```
About 20 to 25 s of CPU time with the mock backend. Needs `torch` + `transformers`, the cached MiniLM, and `shared/llm_client.py`; no downloads.

## Code, step by step

### 1. Header docstring and imports  (`main.py` lines 1–23)

```python
"""
GOAL:  Anthropic-style Contextual Retrieval: for every chunk an LLM writes 1-2 sentences that situate the chunk in its
       document; that text is prepended before embedding (contextual embeddings) and before BM25 indexing
       (contextual BM25). Uses shared/llm_client.py: the MOCK backend by default (a deterministic heuristic stands in
       for the LLM, so this runs offline), LLM_BACKEND=anthropic for a real model. Measures the retrieval change and
       the cost with and without prompt caching.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 19_chunking_deep_dive/06_contextual_retrieval/main.py
EXPECTED OUTPUT (shortened):
    A) the prompt (shortened) and 3 example chunks: raw vs contextualised
    B) retrieval: baseline | contextual embeddings | contextual BM25 | both   (dense, BM25, hybrid)
    C) cost: LLM input tokens without caching vs with caching
NEEDS: torch, transformers (cached MiniLM).  ~20 s.  Real LLM: LLM_BACKEND=anthropic + ANTHROPIC_API_KEY.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from common19 import count_tokens, embed, evaluate, load_docs, load_step, sentence_spans, table  # noqa: E402
from llm_client import MockLLM, generate  # noqa: E402
```

- **In simple words:** the usual helpers plus the course's tiny LLM client.
- **Exactly:** `shared/llm_client.py` exposes `generate(messages, ...)`, which routes to a backend chosen by `LLM_BACKEND` (`mock` by default, `local`, `anthropic`), and `MockLLM(rules=[(regex, reply_fn)])`, a scripted fake: the first rule whose regex matches the last user message produces the reply. The second `sys.path.insert` adds the repository's `shared/` folder.
- **Why it is here:** the code under test (`situate`) is identical for the mock and a real model; only the backend changes.

### 2. The prompt  (`main.py` lines 25–36)

```python
# The prompt from Anthropic's "Introducing Contextual Retrieval" post (wording shortened a little).
PROMPT = """<document>
{doc}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving
search retrieval of the chunk. Answer only with the succinct context and nothing else."""

HEAD = re.compile(r"^(#{1,6})\s+(.*)$", re.M)
```

- **In simple words:** the instruction sent for every chunk: the whole document, then the chunk, then the request.
- **Exactly:** a `str.format` template with `{doc}` and `{chunk}`. The wording follows Anthropic's published contextual-retrieval prompt (shortened a little). The `<document>` and `<chunk>` tags are structural: the mock (and a real model) uses them to find which text is which. `HEAD` is a multiline regex for Markdown headings.
- **Why it is here:** the stable part of the prompt is the *document*, which is exactly what prompt caching can store.

### 3. `heuristic_context`: the deterministic stand-in for the LLM  (`main.py` lines 39–61)

```python
def heuristic_context(doc_text, chunk_text, pos=None):
    """Deterministic stand-in for the LLM. Finds the chunk in the document, reads its heading path (or, in a document
    without headings, the sentence just before it) and writes 1-2 situating sentences."""
    pos = doc_text.find(chunk_text[:60]) if pos is None else pos
    title = doc_text.splitlines()[0].lstrip("# ").strip()
    path = {}
    for m in HEAD.finditer(doc_text[:pos + 1]):
        if doc_text[:m.start()].count("```") % 2:                # a '# comment' inside a code fence is not a heading
            continue
        lvl = len(m.group(1))
        path = {k: v for k, v in path.items() if k < lvl}
        path[lvl] = m.group(2)
    trail = [path[k] for k in sorted(path) if path[k] != title]
    if trail:
        section_first = doc_text[doc_text.find(trail[-1]) + len(trail[-1]):].strip().split(". ")[0]
        topic = " ".join(section_first.split()[:16])
        return f"This passage is from '{title}', section '{' > '.join(trail)}', which begins: \"{topic}...\"."
    q = [m for m in re.finditer(r"\*\*Q: (.*?)\*\*", doc_text[:pos + 1])]
    if q:
        return f"This passage is from '{title}' and answers the question \"{q[-1].group(1)}\"."
    before = sentence_spans(doc_text[:pos])
    prev = doc_text[before[-1][0]:before[-1][1]] if before else ""
    return f"This passage is from '{title}'. It follows: \"{' '.join(prev.split()[:14])}...\"." if prev else f"This passage opens '{title}'."
```

- **In simple words:** find where the chunk sits in the document and write one sentence using the document title, the heading path above it, and the first words of the section; for a document with Q/A pairs use the question; for a headingless document use the sentence before the chunk.
- **Exactly:** `pos` is where the chunk starts (found from its first 60 characters unless given). The loop over `HEAD.finditer(doc_text[:pos + 1])` builds `path`, a dict `level -> heading`; a heading is dropped when it sits inside a code fence (`count("```") % 2`), and deeper levels are cleared when a shallower heading appears. If a trail exists, the sentence is `This passage is from '<title>', section '<a > b>', which begins: "..."`. Otherwise the code looks for the last `**Q: ...**` marker or the previous sentence.
- **Why it is here:** deterministic and free, so the experiment reproduces. It uses *only the document's own structure*, so it is a weaker contextualiser than a real LLM: that is a fact to keep in mind when reading the gain.

### 4. Mock rule and `situate`: one call per chunk  (`main.py` lines 64–77)

```python
def _mock_rule(messages, system):
    text = messages[-1]["content"]
    doc = re.search(r"<document>\n(.*?)\n</document>", text, re.S).group(1)
    chunk = re.search(r"<chunk>\n(.*?)\n</chunk>", text, re.S).group(1)
    return heuristic_context(doc, chunk)


MOCK = MockLLM(rules=[(r"succinct context", _mock_rule)])     # LLM_BACKEND=mock (default) routes here


def situate(doc_text, chunk_text):
    """One LLM call per chunk. With LLM_BACKEND=anthropic the same call goes to a real model."""
    msg = [{"role": "user", "content": PROMPT.format(doc=doc_text, chunk=chunk_text)}]
    return generate(msg, max_tokens=120, temperature=0.0, mock=MOCK).strip()
```

- **In simple words:** the mock extracts the document and chunk from the prompt and calls `heuristic_context`; `situate` builds the prompt and calls `generate`.
- **Exactly:** `re.search(r"<document>\n(.*?)\n</document>", text, re.S)` captures lazily across newlines (`re.S` = dot matches newline). `MockLLM(rules=[(r"succinct context", _mock_rule)])` routes any prompt containing "succinct context" to that function. `generate(msg, max_tokens=120, temperature=0.0, mock=MOCK)` returns the reply string (temperature 0 for determinism).
- **Why it is here:** the seam between mock and real: change the backend, keep the function.

### 5. Section A: baseline chunks, contexts, examples  (`main.py` lines 80–92)

```python
def main():
    docs = load_docs()
    rec = load_step("02_recursive_splitter_rebuilt")
    by_id = {d["doc_id"]: d for d in docs}
    base = [c for d in docs for c in rec.RecursiveSplitter(80, 0).split_document(d)]      # context-free baseline chunks
    print(f"baseline: {len(base)} plain recursive chunks of <= 80 tokens\n")

    ctx = [situate(by_id[c.doc_id]["text"], c.text) for c in base]                           # N LLM calls
    print("A) the prompt sent per chunk (document + chunk + instruction); three examples:")
    print(f"   prompt overhead without the document/chunk: {count_tokens(PROMPT.format(doc='', chunk=''))} tokens")
    for i in (5, 40, 62):
        print(f"   chunk : {base[i].text[:70]!r}")
        print(f"   context: {ctx[i]}\n")
```

- **In simple words:** chunk the corpus with the step-02 recursive splitter at 80 tokens (no overlap), call `situate` once per chunk, and print three examples.
- **Exactly:** `load_step("02_recursive_splitter_rebuilt")` reuses the splitter. `base` is a list of `RawChunk`; `ctx` is one string per chunk (`N` LLM calls). `count_tokens(PROMPT.format(doc='', chunk=''))` is the fixed overhead of the instruction text.
- **Why it is here:** the chunks come from a context-free splitter on purpose: contextual retrieval is a layer on top of any chunker.

### 6. Section B: the four variants  (`main.py` lines 94–112)

```python
    rows = []

    plain = [c.text for c in base]
    withctx = [f"{x}\n{c.text}" for x, c in zip(ctx, base)]
    M_plain, M_ctx = embed(plain), embed(withctx)
    variants = [("baseline (plain / plain)", M_plain, plain),
                ("contextual embeddings only", M_ctx, plain),
                ("contextual BM25 only", M_plain, withctx),
                ("contextual embeddings + BM25", M_ctx, withctx)]
    for name, M, bm in variants:
        r = evaluate(base, docs=docs, vectors=M, bm25_texts=bm)
        rows.append([name, f"{r['dense']['hit@1']:.2f}", f"{r['dense']['hit@5']:.2f}", f"{r['bm25']['hit@5']:.2f}",
                     f"{r['hybrid']['hit@1']:.2f}", f"{r['hybrid']['hit@5']:.2f}", f"{r['hybrid']['mrr']:.2f}"])
    print("B) retrieval, 74 questions (hit@k = answer sentence inside a top-k chunk)")
    print(table(rows, ["variant", "dense h@1", "dense h@5", "bm25 h@5", "hybrid h@1", "hybrid h@5", "hybrid MRR"]))
    b = evaluate(base, docs=docs, vectors=M_plain, bm25_texts=plain)["hybrid"]["hit@5"]
    f_ = evaluate(base, docs=docs, vectors=M_ctx, bm25_texts=withctx)["hybrid"]["hit@5"]
    print(f"   hybrid top-5 FAILURE rate: {1 - b:.1%} -> {1 - f_:.1%} "
          f"(Anthropic reports 5.7% -> 2.9% for top-20 on their datasets; ours is a tiny corpus)")
```

- **In simple words:** embed and BM25-index the plain chunks or the context-prefixed chunks in all four combinations, and print retrieval scores plus the failure rate.
- **Exactly:** `plain` is the chunk texts; `withctx` is `ctx + "\n" + text`. `M_plain`, `M_ctx` are the two `(n, 384)` embedding matrices. In `evaluate(base, docs=docs, vectors=M, bm25_texts=bm)`, `vectors` decides what the dense search uses and `bm25_texts` what BM25 indexes, while the *chunks being scored* stay the same plain `base`: the context improves retrieval of the chunk but does not change what counts as the answer. The failure rate is `1 - hit@5` of the hybrid retriever.
- **Why it is here:** the four-way split separates the two effects (dense vs lexical) that Anthropic's post reports together.

### 7. Section B2: chunks whose headings were lost  (`main.py` lines 114–128)

```python
    # ---- B2) chunks whose headings were lost (plain-text export / PDF extraction) ----------------------------------------
    blank = [dict(d, text=HEAD.sub(lambda m: " " * len(m.group(0)), d["text"])) for d in docs]   # same offsets, headings gone
    base2 = [c for d in blank for c in rec.RecursiveSplitter(80, 0).split_document(d)]
    plain2 = [c.text for c in base2]
    # context from the pipeline's own metadata (title + section path read from the ORIGINAL document at the chunk offset)
    ctx2 = [heuristic_context(by_id[c.doc_id]["text"], c.text, pos=c.start) for c in base2]
    with2 = [f"{x}\n{c.text}" for x, c in zip(ctx2, base2)]
    rows2 = []
    for name, M, bm in (("headings lost: plain", embed(plain2), plain2),
                        ("headings lost: + context (metadata)", embed(with2), with2)):
        r = evaluate(base2, docs=docs, vectors=M, bm25_texts=bm)
        rows2.append([name, f"{r['dense']['hit@1']:.2f}", f"{r['dense']['hit@5']:.2f}", f"{r['bm25']['hit@5']:.2f}",
                      f"{r['hybrid']['hit@1']:.2f}", f"{r['hybrid']['hit@5']:.2f}", f"{r['hybrid']['mrr']:.2f}"])
    print("\nB2) same, but the chunker never saw the headings (they were blanked out before chunking)")
    print(table(rows2, ["variant", "dense h@1", "dense h@5", "bm25 h@5", "hybrid h@1", "hybrid h@5", "hybrid MRR"]))
```

- **In simple words:** blank out the Markdown headings (same offsets), chunk again, and give the contextual variant the context from the *original* document's metadata.
- **Exactly:** `HEAD.sub(lambda m: " " * len(m.group(0)), d["text"])` replaces each heading line with spaces so every character offset stays valid (the questions still point at the right place). `heuristic_context(..., pos=c.start)` reads title and section path from the original document at the chunk's offset, i.e. it simulates a pipeline that kept heading metadata outside the text.
- **Why it is here:** this is where contextual retrieval is supposed to help: the chunker lost the structure, the pipeline still knows it. Note the comparison: *plain text export* against *plain text + context from metadata*. A [structure-aware chunker with a breadcrumb](../03_structure_aware_markdown/WALKTHROUGH.md) gives the same information with no LLM.

### 8. Section C: cost with and without prompt caching  (`main.py` lines 130–150)

```python
    # ---- C) cost ---------------------------------------------------------------------------------------------------------
    per_chunk_overhead = count_tokens(PROMPT.format(doc="", chunk=""))
    uncached = cached_write = cached_read = out_tok = 0
    for d in docs:
        cs = [c for c in base if c.doc_id == d["doc_id"]]
        dt = count_tokens(d["text"])
        for c in cs:
            uncached += dt + count_tokens(c.text) + per_chunk_overhead       # whole doc re-sent for EVERY chunk
        cached_write += dt                                                   # first call writes the doc into the cache
        cached_read += dt * (len(cs) - 1)                                    # every later call reads it
    out_tok = sum(count_tokens(x) for x in ctx)
    chunk_tok = sum(count_tokens(c.text) + per_chunk_overhead for c in base)
    # price multipliers from Anthropic's prompt-caching docs: 5-minute cache write 1.25x, cache read 0.1x of the base input price
    cost_cached = cached_write * 1.25 + cached_read * 0.10 + chunk_tok
    print(f"\nC) LLM input tokens to contextualise the whole corpus ({sum(count_tokens(d['text']) for d in docs)} tokens, {len(base)} chunks)")
    print(f"   no caching : {uncached:>8} input tokens (document re-sent per chunk) + {out_tok} output tokens")
    print(f"   prompt cache: {cost_cached:>8.0f} input-token-equivalents (write 1.25x, read 0.10x, chunk part full price) -> "
          f"{uncached / cost_cached:.1f}x cheaper")
    print(f"   index grows: {sum(count_tokens(x) for x in ctx)} extra stored tokens "
          f"(+{sum(count_tokens(x) for x in ctx) / sum(count_tokens(c.text) for c in base):.0%})")
    print(f"   mock backend made {len(MOCK.calls)} calls (a real run would make the same {len(base)} requests)")
```

- **In simple words:** count the input tokens needed to contextualise the whole corpus if the document is re-sent for each chunk, and the cost-equivalent if the document is cached.
- **Exactly:** per document `dt` is its token count and `cs` its chunks. Uncached: `dt + chunk_tokens + overhead` per chunk. Cached: the first call writes the document (`dt`, billed 1.25x), every later call reads it (`dt x (chunks - 1)`, billed 0.10x), and the chunk-and-prompt part is always full price (`chunk_tok`). `uncached / cost_cached` is the saving. `out_tok` is the generated context (output tokens are priced higher than input per token; check the provider). `MOCK.calls` is the list of calls the mock recorded.
- **Why it is here:** contextual retrieval is cheap *only* with caching, and this is the formula to quote in the interview.

### 9. Entry point and the TRY IT / BREAK IT lines  (`main.py` lines 153–158)

```python
if __name__ == "__main__":
    main()

# TRY IT: change RecursiveSplitter(80, 0) to (200, 0): fewer, bigger chunks need less context; does the gain shrink?
# TRY IT: replace heuristic_context with a function that returns only the document title; how much of the gain is just the title?
# BREAK IT: remove the <chunk> tags from PROMPT -> _mock_rule's re.search returns None -> AttributeError: 'NoneType' has no attribute 'group'.
```

- **In simple words:** guard plus exercises.
- **Exactly:** removing the `<chunk>` tags makes `_mock_rule`'s `re.search` return `None`.
- **Why it is here:** the tags are a contract between the prompt builder and the (mock) responder.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `PROMPT.format(doc=..., chunk=...)` | Fill the `{doc}` and `{chunk}` placeholders |
| `re.search(r"...(.*?)...", text, re.S).group(1)` | Lazy capture across lines (`re.S`: `.` matches newline); `.group(1)` is the captured text |
| `{k: v for k, v in path.items() if k < lvl}` | Dict comprehension: keep only shallower headings |
| `MockLLM(rules=[(pattern, fn)])` | Scripted fake model: a regex on the last user message selects the reply function |
| `lambda m: " " * len(m.group(0))` | Replacement function for `re.sub`: spaces of equal length |
| `dict(d, text=...)` | Copy of a dict with one field replaced |
| `f"{b:.1%}"` | Percentage with one decimal |

## Libraries & functions
| Function | What it does | Internally |
|---|---|---|
| `generate(messages, mock=MOCK)` | One LLM completion | Picks the backend from `LLM_BACKEND`; for `mock` runs the first matching rule |
| `re.sub(pattern, fn, text)` | Replace matches by `fn(match)` | Same offsets if the replacement has equal length |
| `embed(texts)` | Cached MiniLM vectors | See the [step 01 tour](../01_tokens_not_characters/WALKTHROUGH.md#the-shared-helpers-once-a-tour-of-common19py) |
| `evaluate(..., vectors=, bm25_texts=)` | Score with chosen vectors and BM25 texts | Same harness as every step |
| real backend: `LLM_BACKEND=anthropic` | Sends the same messages to a Claude model | The Anthropic Messages API; optional, never required |

## Execution trace (real run)
Real output of `python main.py` (mock backend; progress bar removed; timings vary by machine).
```
baseline: 87 plain recursive chunks of <= 80 tokens

A) the prompt sent per chunk (document + chunk + instruction); three examples:
   prompt overhead without the document/chunk: 69 tokens
   chunk : '# Customer Support\n\n## Opening hours\nCustomer support is open Monday t'
   context: This passage opens 'Customer Support'.

   chunk : '# Lumen Cloud Employee Handbook\n\nThis handbook applies to every employ'
   context: This passage opens 'Lumen Cloud Employee Handbook'.

   chunk : 'the lack of a single dashboard that shows queue depth next to lock wai'
   context: This passage is from 'Postmortem: the March checkout outage (INC-20481)'. It follows: "Looking back, both guesses were reasonable given the graphs we had, and...".

B) retrieval, 74 questions (hit@k = answer sentence inside a top-k chunk)
variant                       dense h@1  dense h@5  bm25 h@5  hybrid h@1  hybrid h@5  hybrid MRR
----------------------------  ---------  ---------  --------  ----------  ----------  ----------
baseline (plain / plain)           0.76       0.89      0.92        0.69        0.95        0.81
contextual embeddings only         0.69       0.89      0.92        0.72        0.95        0.82
contextual BM25 only               0.76       0.89      0.91        0.68        0.93        0.80
contextual embeddings + BM25       0.69       0.89      0.91        0.65        0.95        0.78
   hybrid top-5 FAILURE rate: 5.4% -> 5.4% (Anthropic reports 5.7% -> 2.9% for top-20 on their datasets; ours is a tiny corpus)

B2) same, but the chunker never saw the headings (they were blanked out before chunking)
variant                              dense h@1  dense h@5  bm25 h@5  hybrid h@1  hybrid h@5  hybrid MRR
-----------------------------------  ---------  ---------  --------  ----------  ----------  ----------
headings lost: plain                      0.72       0.91      0.91        0.65        0.95        0.78
headings lost: + context (metadata)       0.73       0.92      0.89        0.69        0.96        0.81

C) LLM input tokens to contextualise the whole corpus (4987 tokens, 87 chunks)
   no caching :    71850 input tokens (document re-sent per chunk) + 3334 output tokens
   prompt cache:    22811 input-token-equivalents (write 1.25x, read 0.10x, chunk part full price) -> 3.1x cheaper
   index grows: 3334 extra stored tokens (+67%)
   mock backend made 87 calls (a real run would make the same 87 requests)
```

How to read it:
- **A:** 87 chunks, so 87 mock LLM calls; the prompt overhead is 69 tokens. Look at the contexts: the mock only has the document's own structure to work with, so a chunk at the top of a document gets "This passage opens 'Customer Support'" (no information beyond the title), and a headingless chunk gets only "It follows: ...". A real LLM would write much richer text.
- **B:** on this corpus contextual retrieval gives **no gain**. Hybrid top-5 failure rate is 5.4% before and after; contextual embeddings alone *lower* dense hit@1 from 0.76 to 0.69 (and raise hybrid hit@1 from 0.69 to 0.72); contextual BM25 changes nothing. These are differences of 1 to 5 questions out of 74. Reasons, in order of likelihood: the documents are tiny and the chunks mostly keep their headings, the mock context repeats words the chunk's neighbours already contain, and the top-5 ceiling is already 0.95. Anthropic's 5.7% to 2.9% is for top-20 on their large datasets with a real LLM. Do not read our table as a refutation, and do not quote it as support.
- **B2:** when the chunker never saw the headings, contextual text from the pipeline's metadata helps a little in the places that matter: hybrid h@1 0.65 to 0.69, hybrid MRR 0.78 to 0.81, hybrid h@5 0.95 to 0.96 (and dense h@1 0.72 to 0.73, BM25 h@5 down 0.91 to 0.89). Small and inside the noise, but in the expected direction for the realistic case.
- **C:** without caching the corpus costs 71,850 input tokens (14 times the corpus size, because each document is re-sent for every one of its chunks) plus 3,334 output tokens; with caching 22,811 input-token-equivalents, 3.1x cheaper. The ratio is far below the 10x limit because the documents are short (about 6 chunks each): each call still pays for its chunk and the 69-token prompt. On long documents with hundreds of chunks it approaches 10x ([solutions](../solutions/solutions.md)). The index grows by 67% (3,334 stored tokens of context).

## Experiments (change ONE thing)
| Change | What you will see | Why |
|---|---|---|
| Baseline `RecursiveSplitter(200, 0)` | Fewer, bigger chunks; the gain from context is expected to shrink | Bigger chunks carry more of their own context |
| `heuristic_context` returning only the document title | Most of the lexical gain can already come from the title, if you test it | Measures how much of the effect is just the title |
| Contextual BM25 only vs contextual embeddings only (rows of table B) | The two effects are separable | Dense and lexical retrieval fail in different ways |
| Remove the caching terms: `uncached / uncached` | Ratio 1.0x | The cache is the whole saving |
| `LLM_BACKEND=anthropic` (optional) | Real, richer contexts; the same code path; a cost | Only `generate` changes |

## Debugging practice
**BREAK IT:** remove the `<chunk>` tags from `PROMPT`.

```
AttributeError: 'NoneType' object has no attribute 'group'
```
`_mock_rule` runs `re.search(r"<chunk>\n(.*?)\n</chunk>", text, re.S).group(1)`; with no tags the search returns `None`, and `.group` fails. **Fix:** the pattern and the template must change together, and a real parser should check for `None` and raise a clear error ("prompt template no longer contains <chunk> tags"). With a *real* LLM the failure is quieter: the model gets an unmarked blob and the context quality drops with no exception.

## Common mistakes
- Comparing contextual retrieval to *nothing* when a free breadcrumb gives most of the benefit. Always include the deterministic-header baseline.
- Re-sending the full document for every chunk without prompt caching and being surprised by the bill.
- Asking the LLM for long contexts: 50 to 100 tokens is the target; longer text dilutes the chunk.
- Re-contextualising the whole corpus when one document changes. Cache by `(document version, chunk text)` ([step 11](../11_chunker_lld/WALKTHROUGH.md)).
- Letting the LLM *invent* facts in the context. The context should only situate; validate on a sample.
- Forgetting that documents longer than the LLM context need a truncated or windowed document in the prompt.
- Expecting the mock's numbers to match a real LLM's.

## Rebuild checklist
1. Write the prompt with `<document>`, `<chunk>` and the instruction; measure its overhead tokens.
2. `situate(doc, chunk)`: format, `generate(temperature=0)`, strip.
3. Contexts for every chunk; `withctx = ctx + "\n" + text`.
4. Four combinations through `evaluate(vectors=..., bm25_texts=...)` with the *same* chunks.
5. Cost: uncached `sum(doc + chunk + overhead)`; cached `1.25 x doc + 0.10 x doc x (n - 1) + n x (chunk + overhead)` per document.
6. The headings-lost experiment as the realistic case.

## Say it aloud
"Contextual retrieval has an LLM write one or two sentences that situate each chunk in its document, then prepends that to the chunk before embedding and before BM25 indexing, so both retrievers and the generator see a chunk that can stand alone. Anthropic reported the top-20 failure rate falling from 5.7% to 2.9% with both indexes, on their datasets. Without caching you pay for the whole document once per chunk; with prompt caching the document is read at about a tenth of the price, which turns it into a modest one-off ingestion cost, repeated only when a document changes. The honest baseline is a free deterministic heading prefix, so I measure the gain over that, not over nothing."

## The new problem (cliffhanger)
Everything so far fights one tension from the *indexing* side: small chunks are precise to search but lack context; big chunks have context but blur the vector and waste tokens in the prompt. What if we stop choosing: **search small, answer with big**? Index the little pieces, and when one matches, hand the LLM its parent section. Next: [step 07, hierarchical and parent-child retrieval](../07_hierarchical_parent_child/WALKTHROUGH.md).
