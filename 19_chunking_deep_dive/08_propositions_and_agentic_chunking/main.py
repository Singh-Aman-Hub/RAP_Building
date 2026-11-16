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

PRONOUN = re.compile(r"^(It|This|They|These|Those)\b")
STOP = set("a an the of to in on for and or is are was were be it this that these those with at by as from we our you your".split())


def topic_of(heading):
    return re.sub(r"^[\d.]+\s*", "", heading).strip().lower()           # "2.3.2 Secondary caregiver" -> "secondary caregiver"


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


def nearest_heading(text, pos):
    heads = [m for m in re.finditer(r"^#{1,6}\s+(.*)$", text[:pos + 1], re.M) if text[:m.start()].count("```") % 2 == 0]
    return heads[-1].group(1) if heads else ""


def paragraphs(text):
    return [(m.start(), m.end()) for m in re.finditer(r"\S[^\n]*(?:\n(?!\s*\n)[^\n]*)*", text)]


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


if __name__ == "__main__":
    main()

# TRY IT: change the 0.06 overlap threshold in _agent_rule to 0.3; chunks get smaller. Does hit@1 move?
# TRY IT: in rule_propositions delete the PRONOUN.sub line; the 'Each block must be...' style facts lose their subject.
# BREAK IT: make the mock return text that is not JSON -> extract_json raises ValueError: No JSON object found in model reply.
