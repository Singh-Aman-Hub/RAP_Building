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


if __name__ == "__main__":
    main()

# TRY IT: change RecursiveSplitter(80, 0) to (200, 0): fewer, bigger chunks need less context; does the gain shrink?
# TRY IT: replace heuristic_context with a function that returns only the document title; how much of the gain is just the title?
# BREAK IT: remove the <chunk> tags from PROMPT -> _mock_rule's re.search returns None -> AttributeError: 'NoneType' has no attribute 'group'.
