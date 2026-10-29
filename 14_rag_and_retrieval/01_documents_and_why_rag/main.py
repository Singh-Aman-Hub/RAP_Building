"""
GOAL:  Load a private knowledge base, clean it, attach metadata, and count its tokens.  Then do the
       arithmetic that motivates RAG: why not paste every document into every prompt?

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 14_rag_and_retrieval/01_documents_and_why_rag/main.py
EXPECTED OUTPUT (shortened):
    loaded 9 documents from data/kb
    cleaned text: 'Refund policy: Refunds are issued within 9 days. Contact help@... for help.'
    whole knowledge base: ~1,5xx tokens
    a company wiki of 20,000 such pages: ~3.xM tokens -> does not fit any context window
NEEDS: transformers (only the cached MiniLM tokenizer).  A few seconds.
"""

import html
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import KB_DIR, EMBED_MODEL   # noqa: E402

from transformers import AutoTokenizer   # noqa: E402

# ---- 1. Load: every file becomes a dict with text + metadata ------------------------------------------
docs = []
for path in sorted(KB_DIR.glob("*.md")):                 # sorted -> same order every run
    text = path.read_text(encoding="utf-8")
    docs.append({
        "doc_id": path.stem,                              # stable id: used for citations and updates
        "title": text.splitlines()[0].lstrip("# ").strip(),
        "source": str(path.relative_to(KB_DIR.parents[1])),
        "n_sections": text.count("\n## "),
        "text": text,
    })
print(f"loaded {len(docs)} documents from {KB_DIR.relative_to(KB_DIR.parents[1])}")
for d in docs:
    print(f"  {d['doc_id']:<28} title={d['title']!r:<34} sections={d['n_sections']}")

# ---- 2. Clean: real documents come from HTML, PDFs and wikis and are messy ----------------------------
messy = """<div class="policy"><h2>Refund&nbsp;policy:</h2>   Refunds are issued
within 9 days.   Contact <a href="mailto:help@chaikitchen.example">help@chaikitchen.example</a>
for help.</div>   Page 3 of 12   """


def clean(raw: str) -> str:
    text = html.unescape(raw)                             # &nbsp; -> real character
    text = re.sub(r"<[^>]+>", " ", text)                  # drop HTML tags, keep their inner text
    text = re.sub(r"Page \d+ of \d+", " ", text)          # boilerplate that would pollute every chunk
    text = text.replace("\u00a0", " ")                    # non-breaking spaces -> normal spaces
    text = re.sub(r"\s+", " ", text)                      # collapse all whitespace runs
    return text.strip()


print("\nraw text    :", repr(messy[:70]) + "...")
print("cleaned text:", repr(clean(messy)))

# ---- 3. Count tokens: the unit that context windows and API bills are measured in ---------------------
tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
token_counts = {d["doc_id"]: len(tok(d["text"])["input_ids"]) for d in docs}
total = sum(token_counts.values())
print(f"\nwhole knowledge base: {total:,} tokens (MiniLM tokenizer; an LLM tokenizer gives similar numbers)")
print(f"largest document    : {max(token_counts, key=token_counts.get)} with {max(token_counts.values())} tokens")
print(f"tokenizer limit: {tok.model_max_length} tokens, but sentence-transformers truncates this model at 256 "
      "(it was trained on 128) -> longer text is silently cut off")

# ---- 4. Why not just paste everything into the prompt? -------------------------------------------------
pages = 20_000                                              # a mid-size company wiki
wiki_tokens = total / len(docs) * pages
print(f"\na company wiki of {pages:,} such pages: ~{wiki_tokens / 1e6:.1f}M tokens -> does not fit any context window")

price_per_m = 3.0                                           # illustrative $ per million input tokens
queries_per_day = 10_000
stuff_cost = total * queries_per_day * price_per_m / 1e6
rag_tokens = 3 * 120                                        # RAG: about 3 chunks of ~120 tokens
rag_cost = rag_tokens * queries_per_day * price_per_m / 1e6
print(f"even our tiny KB, pasted into {queries_per_day:,} prompts/day: ${stuff_cost:,.0f}/day "
      f"vs RAG with 3 chunks: ${rag_cost:,.0f}/day")
print("=> retrieve only the few relevant pieces; that is the R in RAG.")

# TRY IT: add a new file data/kb/gift_cards.md with a '# Gift Cards' title and two '## ' sections; re-run.
# TRY IT: change `pages` to 500. Would 'stuff everything' now fit a 200k-token context window? Is it still wise?
# BREAK IT: remove `sorted(...)` around glob - the order may change between machines, so ids/positions are unstable.
# BREAK IT: add a '₹' sign to a KB file and read it with encoding="cp1252" (a common Windows default): no error,
#           just mojibake ('â‚¹'). With encoding="ascii" you get UnicodeDecodeError. Always pass encoding="utf-8".
