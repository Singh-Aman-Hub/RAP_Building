"""Step 06 exercise: the contextualisation cost formula, with and without prompt caching, as a function you can reuse.
RUN: python s06_cost_formula.py        (pure Python, no model)"""


def contextual_cost(doc_tokens, n_chunks, chunk_tokens=80, prompt_overhead=69, write=1.25, read=0.10):
    """Input-token equivalents to contextualise ONE document. write/read multipliers: check the provider's current pricing."""
    uncached = n_chunks * (doc_tokens + chunk_tokens + prompt_overhead)
    cached = doc_tokens * write + doc_tokens * read * (n_chunks - 1) + n_chunks * (chunk_tokens + prompt_overhead)
    return uncached, cached


for doc_tokens, n in ((1000, 12), (10_000, 120), (100_000, 1200)):
    u, c = contextual_cost(doc_tokens, n)
    print(f"{doc_tokens:>7} token document, {n:>4} chunks: uncached {u:>11,.0f}  cached {c:>11,.0f}  -> {u / c:.1f}x cheaper")
print("The ratio tends to 1/read = 10x for long documents; it is below that because each call still pays for its chunk + prompt.")
