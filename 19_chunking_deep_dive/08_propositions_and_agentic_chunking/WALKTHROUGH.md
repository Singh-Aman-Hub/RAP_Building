# Step 08 — Propositions and agentic chunking

## The story so far
[Step 07](../07_hierarchical_parent_child/WALKTHROUGH.md) kept the atoms (leaves, sentences) fixed and varied what we *send*. But a sentence is a poor atom when it packs two facts together, hides its subject behind "it", or relies on a heading. Back to the [module README](../README.md).

## The pain point
Take: "It must be approved by the manager, and the budget is 1,200 euros." One sentence, two independent facts, and the subject "It" is unresolved. A sentence-level vector blends both facts; a question about only the budget matches a vector that is half about approval; and the retrieved sentence is useless in isolation because "It" refers to remote work, three sentences earlier.

## The spark
Let an LLM do the editing that a splitter cannot:
1. **Proposition chunking.** Rewrite each passage as **propositions**: atomic, self-contained facts, one claim each, with pronouns resolved ("The remote work policy must be approved by the manager."). Index those. (Dense X Retrieval, Chen et al., 2023, which also trained a small "propositionizer" model so you need not call a large LLM.)
2. **Agentic chunking.** Walk the text sentence by sentence and ask an LLM, for each sentence, "same topic as the current chunk, or a new one?" The LLM decides the boundaries.

Both put an LLM *inside the chunker*, so both cost ingestion calls. This step counts them.

## Goal
Implement both with deterministic rules standing in for the LLM (so everything runs offline), run the same prompts through the course's LLM client (a real model can replace the mock), and measure:
1. A passage before and after proposition chunking (A).
2. Agentic SAME/NEW decisions on the headingless postmortem (B).
3. Retrieval, and ingestion cost in calls and tokens, against sentence packing (C and D).

## The theory (read this before the interview)
**Proposition** (as defined for retrieval): a minimal unit of meaning that is (1) *atomic* (one fact), (2) *self-contained* (understandable without the surrounding text: pronouns and references resolved, necessary context added) and (3) faithful to the source (numbers, names and units unchanged).

```
sentence:     "It must be approved by the manager, and the budget is 1,200 euros."
propositions: "Remote work must be approved by the manager."
              "The remote work budget is 1,200 euros."
```
**Two indexing choices.** Embed the proposition and *return the proposition* (small, precise, but a bare fact); or embed the proposition and *return its source sentence or passage* (retrieve small, return big, like [step 07](../07_hierarchical_parent_child/WALKTHROUGH.md)). The step measures both.

**Agentic chunking** (popularised as the top level of Greg Kamradt's "5 levels of text splitting"): an LLM loop maintains the current chunk, reads the next sentence, and outputs `SAME` or `NEW`. A hard token cap closes oversize chunks. The prompt must include the current chunk, so input tokens grow with chunk size; calls are `sentences - 1` per document.

**Ingestion cost.** Proposition chunking: one LLM call per passage; input is the passage plus the instruction, output is roughly the passage again, so the cost is a multiple of the corpus size: **5.3 times the corpus in tokens** in the run below (20,204 input + 6,301 output tokens for a 4,987-token corpus, because every passage is sent with a 100-token instruction and comes back as JSON). A distilled propositionizer model would be cheaper per token. Agentic chunking: one short call per sentence, with input that includes the growing chunk. Compare with recursive chunking: zero calls.

**Faithfulness risk.** An LLM rewriting text can drop a qualifier ("up to", "after probation") or merge numbers. Keep the source span of every proposition (this step does), validate on a sample, and prefer *return the source sentence* when exactness matters (legal, medical, finance).

**Mock caveat.** The mock here uses rules: split at `;` and `, and` / `, but`, replace a leading pronoun by the section topic, and a lexical-overlap test for SAME/NEW. It shows the *mechanism and cost accounting*; a real LLM paraphrases better and also errs differently.

**Status.** Propositions are *research with practical adoption* in high-precision question answering over dense factual text; agentic chunking is *emerging / experimental* (expensive, slow, non-deterministic). For most corpora, structure-aware chunking plus a cheap header or contextual retrieval gives more per dollar.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| proposition | atomic fact, claim, "fact chunk", decontextualised sentence | near | Atomic **and** self-contained; a short sentence is not necessarily either |
| propositionizer / proposition chunking | Dense X Retrieval, fact extraction, claim extraction | near | The model or step that rewrites text into propositions |
| decontextualisation | coreference resolution, pronoun resolution, "make it stand alone" | near | Replacing "it" and "this" by what they refer to |
| agentic chunking | LLM-based chunking, LLM-guided splitting, "level 5" splitting | near | An LLM decides every boundary |
| return the fact vs return the source | small-to-big on propositions, retrieve-fact-return-sentence | no | Same index, different context sent |
| SAME / NEW decision | topic continuation test, boundary classification | yes | The binary question put to the LLM per sentence |
| semantic chunking | embedding-distance chunking | no | [Step 04](../04_semantic_chunking/WALKTHROUGH.md): the embedding model, not an LLM, decides |

## Run it
```bash
cd 19_chunking_deep_dive/08_propositions_and_agentic_chunking
python main.py
# from the repo root:
HF_HUB_OFFLINE=1 .venv/bin/python 19_chunking_deep_dive/08_propositions_and_agentic_chunking/main.py
# optional, real LLM (costs money, never required): LLM_BACKEND=anthropic ANTHROPIC_API_KEY=...
```
About 20 to 25 s of CPU time with the mock. Needs `torch` + `transformers`, the cached MiniLM and `shared/llm_client.py`; no downloads.

## Code, step by step

### 1. Header docstring and imports  (`main.py` lines 1–25)

```python
"""
GOAL:  Two LLM-in-the-loop chunkers, with the LLM replaced by deterministic rules so everything runs offline:
         1) PROPOSITION chunking (Dense X Retrieval): rewrite each passage as atomic, self-contained facts and index those
         2) AGENTIC chunking: walk the text sentence by sentence and ask "same topic as the current chunk, or start a new one?"
       Counts the LLM calls and tokens each would cost in production and measures retrieval against sentence-packing.
       The same prompts go to a real model with LLM_BACKEND=anthropic (shared/llm_client.py).

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 19_chunking_deep_dive/08_propositions_and_agentic_chunking/main.py
EXPECTED OUTPUT (shortened):
    A) a passage -> propositions (before / after)         B) agentic decisions on the postmortem
    C) ingestion cost: LLM calls and tokens               D) retrieval table
NEEDS: torch, transformers (cached MiniLM).  ~20 s.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from common19 import RawChunk, count_tokens, evaluate, load_docs, load_step, make_chunk, sentence_spans, table  # noqa: E402
from llm_client import MockLLM, extract_json, generate  # noqa: E402
```

- **In simple words:** helpers plus the LLM client (`MockLLM`, `generate`, `extract_json`).
- **Exactly:** `extract_json(text)` pulls the first `{...}` out of an LLM reply (models add chatter around JSON) and raises `ValueError("No JSON object found in model reply")` if there is none. `make_chunk` and `sentence_spans` come from the [step 01 tour](../01_tokens_not_characters/WALKTHROUGH.md#the-shared-helpers-once-a-tour-of-common19py).
- **Why it is here:** the prompts and parsing are the same for the mock and a real model.

### 2. The two prompts  (`main.py` lines 27–38)

```python
PROP_PROMPT = """Decompose the passage into clear, simple propositions (atomic facts) that can be understood on their own.
1. Split compound sentences into simple sentences; keep the original wording where possible.
2. Replace pronouns and vague references ("it", "this", "these") with the full noun they refer to.
3. Keep numbers, names and units exactly.
Return JSON: {{"propositions": ["...", "..."]}}
Section: {section}
Passage: {passage}"""

AGENT_PROMPT = """You are building topic-coherent chunks. Current chunk:
{chunk}
Next sentence: {sentence}
Answer with exactly one word: SAME if the sentence continues the topic of the current chunk, NEW if it starts a new topic."""
```

- **In simple words:** the proposition prompt asks for JSON with a list of atomic facts; the agentic prompt asks for one word, `SAME` or `NEW`.
- **Exactly:** the doubled braces `{{ ... }}` are literal braces in a `str.format` template. `{section}`, `{passage}`, `{chunk}`, `{sentence}` are the placeholders. The three numbered rules in the first prompt (split compounds, replace pronouns, keep numbers exactly) are the definition of a proposition turned into instructions.
- **Why it is here:** the prompts are the product. A good prompt asks for a *structured* answer (JSON, one word) that code can parse and validate.

### 3. Rules that stand in for the LLM: constants and `topic_of`  (`main.py` lines 40–45)

```python
PRONOUN = re.compile(r"^(It|This|They|These|Those)\b")
STOP = set("a an the of to in on for and or is are was were be it this that these those with at by as from we our you your".split())


def topic_of(heading):
    return re.sub(r"^[\d.]+\s*", "", heading).strip().lower()           # "2.3.2 Secondary caregiver" -> "secondary caregiver"
```

- **In simple words:** a regex for sentence-initial pronouns, a stop-word list, and a function that turns a heading like `2.3.2 Secondary caregiver` into `secondary caregiver`.
- **Exactly:** `PRONOUN` matches a sentence starting with It / This / They / These / Those. `topic_of` removes leading section numbers with `re.sub(r"^[\d.]+\s*", "", heading)` and lower-cases.
- **Why it is here:** the "section topic" is the noun phrase the mock substitutes for a pronoun.

### 4. `split_compound` and `rule_propositions`  (`main.py` lines 48–66)

```python
def split_compound(sent):
    """Rule 1: split at ';' and at ', and' / ', but' when both halves are long enough to stand alone."""
    parts = re.split(r";\s+|,\s+(?:and|but)\s+", sent)
    return parts if len(parts) > 1 and all(len(p.split()) >= 4 for p in parts) else [sent]


def rule_propositions(passage, section):
    """MOCK 'LLM': split compounds, replace a leading pronoun by the section topic, keep each piece's source."""
    out = []
    for s, e in sentence_spans(passage):
        sent = passage[s:e]
        if sent.startswith(("#", "|", "`")):
            continue                                                    # headings / tables / code are not prose
        for piece in split_compound(sent):
            piece = piece.strip().rstrip(".") + "."
            piece = PRONOUN.sub(lambda m: f"The {topic_of(section)}" if section else m.group(0), piece, count=1)
            piece = piece[0].upper() + piece[1:]
            out.append(piece)
    return out
```

- **In simple words:** split compound sentences, replace a leading pronoun with the section topic, ensure each piece ends with a full stop and starts with a capital.
- **Exactly:** `re.split(r";\s+|,\s+(?:and|but)\s+", sent)` splits at semicolons and at `, and` / `, but`; the split is accepted only if *every* piece has at least four words (so "A, and B" is not cut into fragments). Headings, table rows and code lines (`#`, `|`, `` ` ``) are skipped as non-prose. `PRONOUN.sub(..., count=1)` replaces only the leading pronoun by `The <topic>`.
- **Why it is here:** the three proposition rules in code: atomic (split), self-contained (pronoun), faithful (otherwise the wording is untouched).

### 5. The mock model: `_prop_rule`, `_agent_rule`, `MOCK`  (`main.py` lines 69–85)

```python
def _prop_rule(messages, system):
    text = messages[-1]["content"]
    section = re.search(r"Section: (.*)", text).group(1)
    passage = text.split("Passage: ", 1)[1]
    return json.dumps({"propositions": rule_propositions(passage, section)})


def _agent_rule(messages, system):
    text = messages[-1]["content"]
    chunk = re.search(r"Current chunk:\n(.*)\nNext sentence:", text, re.S).group(1)
    sent = text.split("Next sentence: ", 1)[1].split("\nAnswer with")[0]
    a = {w for w in re.findall(r"[a-z0-9]+", chunk.lower()) if w not in STOP}
    b = {w for w in re.findall(r"[a-z0-9]+", sent.lower()) if w not in STOP}
    return "SAME" if len(a & b) / max(len(b), 1) >= 0.06 else "NEW"      # lexical overlap stands in for 'topic'


MOCK = MockLLM(rules=[(r"Decompose the passage", _prop_rule), (r"building topic-coherent", _agent_rule)])
```

- **In simple words:** two functions that read the prompt and produce what an LLM would: a JSON string of propositions, or `SAME`/`NEW`.
- **Exactly:** `re.search(r"Section: (.*)", text)` and `text.split("Passage: ", 1)[1]` parse the prompt the code itself built. The agentic rule treats the stop-word-free words of the current chunk and of the next sentence as sets `a` and `b`; if `|a & b| / |b| >= 0.06` the sentence is `SAME`, else `NEW` (lexical overlap stands in for "topic"). `MockLLM(rules=[(regex, fn), ...])` dispatches on the prompt text.
- **Why it is here:** deterministic and free; a real model would reason about topic, the mock only counts shared words, so its boundaries are cruder.

### 6. `nearest_heading` and `paragraphs`  (`main.py` lines 88–94)

```python
def nearest_heading(text, pos):
    heads = [m for m in re.finditer(r"^#{1,6}\s+(.*)$", text[:pos + 1], re.M) if text[:m.start()].count("```") % 2 == 0]
    return heads[-1].group(1) if heads else ""


def paragraphs(text):
    return [(m.start(), m.end()) for m in re.finditer(r"\S[^\n]*(?:\n(?!\s*\n)[^\n]*)*", text)]
```

- **In simple words:** find the heading above a position (ignoring `#` comments inside code fences) and split a text into paragraphs.
- **Exactly:** the heading search keeps only headings preceded by an even number of fences. The paragraph regex `\S[^\n]*(?:\n(?!\s*\n)[^\n]*)*` matches a non-space start, the rest of the line, and following lines until a blank line (`(?!\s*\n)` is a negative lookahead for an empty line). Returns `(start, end)` pairs.
- **Why it is here:** the proposition call needs the passage and its section name.

### 7. `proposition_chunks`: one LLM call per paragraph  (`main.py` lines 97–121)

```python
def proposition_chunks(doc, stats, return_source=False):
    """One LLM call per paragraph -> propositions. Each proposition keeps the span of the sentence it came from."""
    out = []
    for ps, pe in paragraphs(doc["text"]):
        passage = doc["text"][ps:pe]
        section = topic_of(nearest_heading(doc["text"], ps))
        prompt = PROP_PROMPT.format(section=section, passage=passage)
        reply = generate([{"role": "user", "content": prompt}], mock=MOCK)
        stats["calls"] += 1
        stats["in_tok"] += count_tokens(prompt)
        stats["out_tok"] += count_tokens(reply)
        props = extract_json(reply)["propositions"]
        src = [(ps + s, ps + e) for s, e in sentence_spans(passage) if not passage[s:e].startswith(("#", "|", "`"))]
        k = 0
        for s, e in src:                                               # attach propositions to their source sentence
            for _ in split_compound(doc["text"][s:e]):
                if k < len(props):
                    sent_text = doc["text"][s:e]
                    out.append(RawChunk(doc["doc_id"], s, e, props[k], context_text=sent_text if return_source else props[k],
                                        meta={"source": sent_text}))
                    k += 1
        # non-prose blocks (tables, code, headings) are kept whole as one unit each
        if passage.startswith(("|", "`")):
            out.append(make_chunk(doc, ps, pe))
    return out
```

- **In simple words:** for each paragraph ask for propositions, parse them, and attach each proposition to the sentence it came from, so every fact keeps a source.
- **Exactly:** `generate([...], mock=MOCK)` returns a JSON string; `extract_json(reply)["propositions"]` parses it. `stats` accumulates calls and input/output tokens for the cost report. `src` lists the paragraph's prose sentences with absolute offsets; the double loop (`for s, e in src: for _ in split_compound(...)`) walks sentences and the pieces each produced, assigning `props[k]` to each. The `RawChunk` gets the proposition as `text` (what is embedded) and, if `return_source`, the *sentence* as `context_text` (what the LLM receives). Tables and code paragraphs are kept whole.
- **Why it is here:** the **source span** per proposition is the safeguard against LLM rewriting errors and what makes citations possible. `return_source=True/False` is the "return fact vs return sentence" choice.

### 8. `agentic_chunks`: one LLM call per sentence  (`main.py` lines 124–138)

```python
def agentic_chunks(doc, stats, max_tokens=150):
    """Sentence by sentence: ask the LLM SAME/NEW; a hard token cap closes oversize chunks."""
    spans = sentence_spans(doc["text"])
    chunks, cur = [], [spans[0]]
    for sp in spans[1:]:
        chunk_text = doc["text"][cur[0][0]:cur[-1][1]]
        prompt = AGENT_PROMPT.format(chunk=chunk_text, sentence=doc["text"][sp[0]:sp[1]])
        verdict = generate([{"role": "user", "content": prompt}], mock=MOCK).strip()
        stats["calls"] += 1
        stats["in_tok"] += count_tokens(prompt)
        if verdict == "NEW" or count_tokens(doc["text"][cur[0][0]:sp[1]]) > max_tokens:
            chunks.append(make_chunk(doc, cur[0][0], cur[-1][1]))
            cur = []
        cur.append(sp)
    return chunks + [make_chunk(doc, cur[0][0], cur[-1][1])]
```

- **In simple words:** start with the first sentence; for each next sentence ask SAME/NEW; on NEW (or if adding it would exceed 150 tokens) close the chunk.
- **Exactly:** the prompt includes the *whole current chunk*, so input tokens per call grow with chunk size; `stats["calls"]` is `sentences - 1` per document. `make_chunk` turns a closed range into a `RawChunk`; the last open chunk is appended after the loop.
- **Why it is here:** it makes the cost visible: a model call and a growing prompt for every boundary decision.

### 9. Section A and B: before / after, and agentic decisions  (`main.py` lines 141–161)

```python
def main():
    docs = load_docs()
    by_id = {d["doc_id"]: d for d in docs}
    hb = by_id["employee_handbook"]

    print("A) a passage before and after proposition chunking (mock LLM = rules; a real LLM paraphrases better)")
    ps, pe = next((a, b) for a, b in paragraphs(hb["text"]) if hb["text"][a:b].startswith("The secondary caregiver"))
    print("   before:", hb["text"][ps:pe])
    stats = {"calls": 0, "in_tok": 0, "out_tok": 0}
    one = [c for c in proposition_chunks({"doc_id": "x", "text": hb["text"][:pe]}, stats) if c.start >= ps]
    print("   after :")
    for c in one:
        print("     -", c.text)

    pm = by_id["incident_postmortem"]
    st = {"calls": 0, "in_tok": 0, "out_tok": 0}
    ag = agentic_chunks(pm, st)
    print(f"\nB) agentic chunking of the headingless postmortem: {st['calls']} SAME/NEW calls -> {len(ag)} chunks "
          f"(token sizes {[count_tokens(c.text) for c in ag]})")
    for c in ag[:3]:
        print(f"   {c.text[:60]!r} ... {c.text[-40:]!r}")
```

- **In simple words:** print one handbook passage and its propositions, then agentic-chunk the postmortem and show chunk sizes and edges.
- **Exactly:** `ps, pe` locate the paragraph that starts with "The secondary caregiver". Running `proposition_chunks` on a one-passage fake document `{"doc_id": "x", "text": hb["text"][:pe]}` with a fresh `stats` and keeping chunks with `c.start >= ps` isolates that paragraph's propositions. The B line prints the call count (`SAME/NEW calls`), chunk count and the token size of each chunk.
- **Why it is here:** a visible before/after, and the first look at the unevenness of LLM-chosen boundaries.

### 10. Section C and D: retrieval and ingestion cost  (`main.py` lines 163–184)

```python
    # ---- C/D) the full comparison ------------------------------------------------------------------------------------------
    sem = load_step("04_semantic_chunking")
    rows, costs = [], {}

    def add(name, fn):
        stats = {"calls": 0, "in_tok": 0, "out_tok": 0}
        chunks = [c for d in docs for c in fn(d, stats)]
        r = evaluate(chunks, docs=docs)
        dn = r["dense"]
        costs[name] = stats
        rows.append([name, len(chunks), f"{r['mean_tok']:.0f}", f"{dn['hit@1']:.2f}", f"{dn['hit@3']:.2f}", f"{dn['mrr']:.2f}",
                     f"{r['hybrid']['hit@3']:.2f}", f"{dn['ctx_tok']:.0f}", stats["calls"], stats["in_tok"] + stats["out_tok"]])

    add("sentence-pack 100 tok", lambda d, s: sem.sentence_pack(d, 100))
    add("propositions (return fact)", lambda d, s: proposition_chunks(d, s))
    add("propositions (return sentence)", lambda d, s: proposition_chunks(d, s, return_source=True))
    add("agentic SAME/NEW", agentic_chunks)
    print("\nC+D) 13 documents, 74 questions; last two columns = LLM calls and LLM tokens (in+out) spent at INGESTION")
    print(table(rows, ["chunker", "chunks", "tok/unit", "dense h@1", "dense h@3", "MRR", "hybrid h@3", "ctx tok@3", "LLM calls", "LLM tokens"]))
    pc = costs["propositions (return fact)"]
    print(f"   proposition ingestion: {pc['calls']} calls, {pc['in_tok']} input + {pc['out_tok']} output tokens for a "
          f"{sum(count_tokens(d['text']) for d in docs)}-token corpus -> {(pc['in_tok'] + pc['out_tok']) / sum(count_tokens(d['text']) for d in docs):.1f}x the corpus size")
```

- **In simple words:** four strategies on all 13 documents: sentence packing, propositions (return the fact), propositions (return the source sentence), and agentic. Print quality plus the LLM calls and tokens each cost, and the proposition cost as a multiple of the corpus.
- **Exactly:** `add(name, fn)` builds chunks for every document with a fresh `stats` dict, runs `evaluate`, and stores the stats. The last two columns are the calls and the total LLM tokens (input + output). `sem.sentence_pack` is imported from step 04 as the no-LLM control.
- **Why it is here:** cost next to quality: the question to ask of any LLM-in-the-loop chunker is "what did it buy per call?".

### 11. Entry point and the TRY IT / BREAK IT lines  (`main.py` lines 187–192)

```python
if __name__ == "__main__":
    main()

# TRY IT: change the 0.06 overlap threshold in _agent_rule to 0.3; chunks get smaller. Does hit@1 move?
# TRY IT: in rule_propositions delete the PRONOUN.sub line; the 'Each block must be...' style facts lose their subject.
# BREAK IT: make the mock return text that is not JSON -> extract_json raises ValueError: No JSON object found in model reply.
```

- **In simple words:** guard plus exercises.
- **Exactly:** a mock reply that is not JSON makes `extract_json` raise `ValueError: No JSON object found in model reply`.
- **Why it is here:** a real model sometimes ignores the format; the parser must fail loudly or retry.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `"""...{{"propositions": [...]}}"""` + `.format` | Doubled braces print literal `{` `}` in a format string |
| `re.split(r";\s+|,\s+(?:and|but)\s+", s)` | Split on `;` or `, and` / `, but`; `(?:...)` is a non-capturing group |
| `all(len(p.split()) >= 4 for p in parts)` | Every piece has at least four words |
| `PRONOUN.sub(lambda m: ..., piece, count=1)` | Replace only the first match using a function |
| `text.split("Passage: ", 1)[1]` | Split once and take what follows the marker |
| `stats["calls"] += 1` on a dict | Mutable accumulator passed into the function |
| `(?!\s*\n)` | Negative lookahead: not followed by an empty line |
| `for _ in split_compound(...)` | `_` = loop variable we do not need |

## Libraries & functions
| Function | What it does | Internally |
|---|---|---|
| `generate(messages, mock=MOCK)` | One completion | Routes to mock / local / Anthropic by `LLM_BACKEND` |
| `extract_json(reply)` | First JSON object in a reply | Greedy regex `\{.*\}` then `json.loads`; raises `ValueError` |
| `MockLLM(rules=[...])` | Deterministic fake LLM | First regex matching the last user message wins |
| `re.finditer` | Iterate matches with positions | Lazy scan |
| `count_tokens` | Embedder tokens | Used here to cost LLM input/output (a rough proxy for the LLM's own tokenizer) |

## Execution trace (real run)
Real output of `python main.py` (mock rules instead of an LLM; progress bar removed; timings vary by machine).
```
A) a passage before and after proposition chunking (mock LLM = rules; a real LLM paraphrases better)
   before: The secondary caregiver receives 6 weeks of fully paid leave. These weeks may be split into two blocks. Each block must be at least two weeks long.
   after :
     - The secondary caregiver receives 6 weeks of fully paid leave.
     - The secondary caregiver weeks may be split into two blocks.
     - Each block must be at least two weeks long.

B) agentic chunking of the headingless postmortem: 42 SAME/NEW calls -> 25 chunks (token sizes [37, 15, 6, 16, 10, 25, 104, 65, 36, 54, 7, 39, 27, 10, 43, 19, 31, 81, 6, 119, 5, 31, 11, 24, 59])
   'Postmortem: the March checkout outage (INC-20481)\n\nOn the mo' ... 'te new deployments for almost two hours.'
   'This document describes what happened, why it happened and w' ... 'hy it happened and what we are changing.'
   'It is written without blame.' ... 'It is written without blame.'

C+D) 13 documents, 74 questions; last two columns = LLM calls and LLM tokens (in+out) spent at INGESTION
chunker                         chunks  tok/unit  dense h@1  dense h@3   MRR  hybrid h@3  ctx tok@3  LLM calls  LLM tokens
------------------------------  ------  --------  ---------  ---------  ----  ----------  ---------  ---------  ----------
sentence-pack 100 tok               60        83       0.66       0.92  0.80        0.95        250          0           0
propositions (return fact)         303        15       0.62       0.84  0.74        0.77         54        140       26505
propositions (return sentence)     303        15       0.62       0.84  0.74        0.77         58        140       26505
agentic SAME/NEW                   224        22       0.73       0.89  0.82        0.82         94        362       28741
   proposition ingestion: 140 calls, 20204 input + 6301 output tokens for a 4987-token corpus -> 5.3x the corpus size
```

How to read it:
- **A:** the rules produce three facts from three sentences. The first and third are fine. The second shows the mock's limit and the faithfulness risk: "**These** weeks may be split into two blocks." became "The secondary caregiver weeks may be split into two blocks.": the pronoun was replaced by the *section topic*, not by its real referent ("The six weeks"), so the fact is now slightly wrong. A real LLM does better, and also errs differently, which is why every proposition keeps its source sentence.
- **B:** 42 SAME/NEW calls produce 25 chunks with very uneven sizes (5 to 119 tokens, several under 10). The lexical-overlap rule calls most short sentences a topic change. The first cut isolates the one-line "It is written without blame." as a 6-token chunk.
- **C+D:** propositions are **worse** than sentence-pack on dense retrieval here: hit@1 0.62 against 0.66, hit@3 0.84 against 0.92, hybrid hit@3 0.77 against 0.95, with 303 tiny units (15 tokens each). Returning the source sentence instead of the bare fact changes nothing in quality (identical 0.62 / 0.84) and costs 4 more context tokens (58 against 54), because here a sentence is only slightly bigger than its fact. The context sent to the LLM is very small (54 tokens against 250) but so is the information. Agentic SAME/NEW is on par with recursive (hit@1 0.73, hit@3 0.89) at 362 calls.
- **Cost:** propositions need 140 calls and 26,505 LLM tokens (5.3x the corpus size); agentic needs 362 calls and 28,741 tokens. Sentence-pack needs none and has the best hit@3. With a mock LLM the quality numbers say little about a real LLM; the call and token counts are the transferable result.

## Experiments (change ONE thing)
| Change | What you will see | Why |
|---|---|---|
| Agentic threshold `0.06` to `0.3` in `_agent_rule` | More `NEW` decisions, so more and smaller chunks | A higher overlap requirement makes "same topic" harder to satisfy |
| Delete the `PRONOUN.sub` line | Facts begin with "It must be ..." again; the vector for the fact has no topic | The pronoun replacement is what makes the proposition self-contained |
| `max_tokens=60` in `agentic_chunks` | The token cap closes chunks, not the model | The hard cap is a safety net |
| `return_source=True` vs `False` | Quality changes with the context sent; context tokens change | Sentence vs bare fact as the evidence |
| Corpus of 1M tokens: scale the ratio from section C | Calls and tokens scale linearly with the corpus: do the arithmetic from the table | Cost is per passage, not per query |

## Debugging practice
**BREAK IT:** make the mock return text that is not JSON.

```
ValueError: No JSON object found in model reply: 'Sure! Here are the facts ...'
```
`extract_json` searched the reply for `{...}` and found none. Real models do this when they add a preamble or refuse. **Fix:** retry once with a stricter instruction ("Return ONLY JSON"), validate the schema (`propositions` is a list of non-empty strings), fall back to the source sentence for that passage, and count failures as a metric. Never let a malformed LLM reply drop a passage from the index silently.

## Common mistakes
- Trusting proposition rewrites without keeping the source span: errors become invisible and uncitable.
- Returning bare facts when the user needs the surrounding sentence, or the source sentence when you only need the fact: decide per use case and measure.
- Calling an LLM per sentence over a large corpus without estimating cost first (agentic chunking: calls grow with sentences, input with chunk size).
- Using a *propositionizer* trained on Wikipedia-style prose on code or tables.
- Resolving pronouns with a section title that is not the referent (the mock's weakness; a real LLM reads the preceding text).
- Re-running the whole LLM ingestion when one document changes. Cache by `(model, prompt version, text hash)` ([step 11](../11_chunker_lld/WALKTHROUGH.md)).
- Forgetting that the LLM output is *new text*: it changes BM25 and can drop qualifiers ("up to", "after probation").

## Rebuild checklist
1. Write the two prompts; make the proposition prompt return JSON and the agentic prompt one word.
2. `rule_propositions` (or a real LLM call) with split-compound and pronoun replacement.
3. A mock with `MockLLM(rules=...)` that parses the prompt the way the real one would be called.
4. `proposition_chunks`: per paragraph, call, `extract_json`, attach each fact to its source sentence, keep tables/code whole, count calls and tokens.
5. `agentic_chunks`: SAME/NEW per sentence, token cap, last chunk.
6. Cost table: calls, input tokens, output tokens; extrapolate to a corpus.

## Say it aloud
"Proposition chunking has an LLM rewrite each passage into atomic, self-contained facts, with pronouns resolved and numbers untouched, and indexes those, ideally returning the source sentence to the generator. Agentic chunking asks an LLM, sentence by sentence, whether the next sentence continues the current chunk. Both move an LLM into ingestion: roughly one call per passage, or one per sentence with a growing prompt, and the rewritten text can drop qualifiers, so I keep source spans and validate on a sample. They pay off on dense factual text where precision matters; for most corpora structure-aware chunking with a header or contextual retrieval gives more per dollar."

## The new problem (cliffhanger)
Every chunker so far starts from a *string*. But a PDF is not a string: it has pages, columns, tables that continue on the next page, headers and footers that repeat on every page, figures with captions, and bounding boxes that let you highlight the answer. Flatten it to text and all of that is gone. Next: chunk a **parsed document**, a list of layout elements: [step 09, layout-aware chunking](../09_layout_aware_chunking_from_parsed_docs/WALKTHROUGH.md).
