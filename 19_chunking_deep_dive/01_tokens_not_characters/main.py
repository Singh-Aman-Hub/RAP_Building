"""
GOAL:  Stop counting characters. Measure text with the REAL tokenizer of the embedding model, cut fixed windows
       by TOKENS (with overlap), compare chars-per-token across text types/languages/tokenizers, and see what
       silently happens to text past the model's max_seq_length.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 19_chunking_deep_dive/01_tokens_not_characters/main.py
EXPECTED OUTPUT (shortened):
    A) per document: chars, words, tokens, chars/token
    B) chars per token for English / code / JSON / German / Hindi / Chinese / emoji on 3 tokenizers
    C) token windows: every window <= N tokens; character windows of the "same" size vary wildly in tokens
    D) a fact after token 256 is invisible to the vector (cosine to truncated == cosine to full)
    E) retrieval hit@3 for token windows of 64 / 128 / 256 tokens with 0 and ~15% overlap
NEEDS: torch, transformers (cached MiniLM, multilingual-e5-small, gpt2).  ~20 s.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common19 import (RawChunk, answer_spans, count_tokens, embed, evaluate, get_tokenizer,  # noqa: E402
                      load_docs, load_eval, table, token_spans)


def fixed_token_windows(doc, size=128, overlap=20):
    """Cut `doc` into windows of at most `size` tokens; each window starts `size - overlap` tokens after the last.
    Offsets from the tokenizer turn token indices back into exact character slices of the original text."""
    assert 0 <= overlap < size, "overlap must be smaller than size (step would be 0 or negative)"
    spans = token_spans(doc["text"])                       # [(char_start, char_end)] per token
    step, chunks = size - overlap, []
    for i in range(0, max(len(spans) - overlap, 1), step):
        window = spans[i:i + size]                          # token indices i .. i+size-1
        start, end = window[0][0], window[-1][1]            # first token's start, last token's end
        chunks.append(RawChunk(doc["doc_id"], start, end, doc["text"][start:end],
                               meta={"tok_start": i, "tok_end": i + len(window)}))
        if i + size >= len(spans):                          # this window already reached the end: stop
            break
    return chunks


def fixed_char_windows(doc, size_chars):
    """The naive version: every `size_chars` characters. Used only to show how unstable its token size is."""
    t = doc["text"]
    return [RawChunk(doc["doc_id"], i, min(i + size_chars, len(t)), t[i:i + size_chars]) for i in range(0, len(t), size_chars)]


def main():
    docs = load_docs()
    new_docs = [d for d in docs if d["corpus"] == "new"]

    # ---- A) characters vs words vs tokens -----------------------------------------------------------------
    print("A) the same documents, three ways to measure them")
    rows = []
    for d in new_docs:
        c, w, t = len(d["text"]), len(d["text"].split()), count_tokens(d["text"])
        rows.append([d["doc_id"], c, w, t, f"{c / t:.2f}", f"{t / w:.2f}"])
    print(table(rows, ["document", "chars", "words", "tokens", "chars/token", "tokens/word"]))

    # ---- B) chars per token depends on content and on the tokenizer ----------------------------------------
    from transformers import AutoTokenizer
    toks = {"MiniLM (WordPiece)": get_tokenizer(),
            "e5-small (SentencePiece)": AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small"),
            "gpt2 (byte BPE)": AutoTokenizer.from_pretrained("gpt2")}
    samples = {
        "English prose": "Every employee may work remotely up to three days per week after the probation period.",
        "Python code": "def list_projects(page_size=50):\n    resp = requests.get(url, params={'page_size': page_size})\n    return resp.json()['items']",
        "JSON": '{"code": "rate_limited", "retry_after": 30, "request_id": "9f8c2e1a-77b4-4d0e"}',
        "URL + hash": "https://api.lumen.example/v2/deployments/3f2a9c4e7b1d8a60c5e2f9d1b7a34c88",
        "German": "Mitarbeitende dürfen nach der Probezeit bis zu drei Tage pro Woche im Homeoffice arbeiten.",
        "Hindi": "कर्मचारी परिवीक्षा अवधि के बाद सप्ताह में तीन दिन तक घर से काम कर सकते हैं।",
        "Chinese": "试用期结束后，员工每周最多可以在家远程工作三天。",
        "Emoji": "Great job team 🎉🎉🚀🔥 see you at the offsite 🍕",
    }
    print("\nB) characters per token (higher = cheaper). [UNK] = token the model cannot represent at all")
    rows = []
    for label, s in samples.items():
        row = [label, len(s)]
        for name, tk in toks.items():
            ids = tk(s, add_special_tokens=False)["input_ids"]
            row.append(f"{len(s) / len(ids):.1f} ({len(ids)})")
        unk = sum(i == toks["MiniLM (WordPiece)"].unk_token_id for i in toks["MiniLM (WordPiece)"](s, add_special_tokens=False)["input_ids"])
        row.append(unk)
        rows.append(row)
    print(table(rows, ["text", "chars"] + list(toks) + ["MiniLM [UNK]"]))

    # ---- C) token windows vs character windows -------------------------------------------------------------
    doc = next(d for d in new_docs if d["doc_id"] == "employee_handbook")
    tw = fixed_token_windows(doc, size=128, overlap=20)
    sizes = [count_tokens(c.text) for c in tw]
    print(f"\nC) handbook ({count_tokens(doc['text'])} tokens): token windows size=128 overlap=20 -> {len(tw)} chunks")
    print(f"   token sizes: min {min(sizes)}, max {max(sizes)}  (every one <= 128; exact slices: "
          f"{all(doc['text'][c.start:c.end] == c.text for c in tw)})")
    cw = fixed_char_windows(doc, 128 * 4)                    # 'a token is about 4 characters' rule of thumb
    cs = [count_tokens(c.text) for c in cw]
    print(f"   character windows of 512 chars -> {len(cw)} chunks, token sizes: min {min(cs)}, max {max(cs)}, "
          f"{sum(s > 128 for s in cs)} of them above 128")
    print(f"   overlap check: window 0 ends at token {tw[0].meta['tok_end']}, window 1 starts at token {tw[1].meta['tok_start']}")

    # ---- D) what silently happens past max_seq_length ---------------------------------------------------------
    by_id = {d["doc_id"]: d for d in docs}
    tok = get_tokenizer()
    ids_all = tok(doc["text"], add_special_tokens=False, return_offsets_mapping=True)
    offs = ids_all["offset_mapping"]
    q = next(q for q in load_eval() if q["doc_id"] == "employee_handbook"
             and 300 < next(i for i, (s, e) in enumerate(offs) if e > answer_spans(q, by_id)[0][0]) < 440)
    a_tok = next(i for i, (s, e) in enumerate(offs) if e > answer_spans(q, by_id)[0][0])
    prefix = doc["text"][:offs[480][1]]                       # first 481 tokens of the handbook
    trunc = doc["text"][:offs[255][1]]                        # first 256 tokens only
    around = doc["text"][offs[a_tok - 40][0]:offs[a_tok + 40][1]]  # 81-token window centred on the answer
    qv, v_full, v_trunc, v_around = embed([q["question"], prefix, trunc, around])
    print(f"\nD) question: {q['question']!r}  (answer starts at token {a_tok})")
    print(f"   embedder max_length = 256; input has {count_tokens(prefix)} tokens -> {count_tokens(prefix) - 256} tokens never read")
    print(f"   cos(question, whole 481-token text)    = {qv @ v_full:.4f}")
    print(f"   cos(question, first 256 tokens only)   = {qv @ v_trunc:.4f}   <- identical vector: cos(full, trunc) = {v_full @ v_trunc:.6f}")
    print(f"   cos(question, 81-token window on answer) = {qv @ v_around:.4f}")
    long_ids = tok(prefix * 2, return_tensors="pt", truncation=False)   # ~960 tokens: beyond BERT's 512 positions
    try:
        get_embedder_model().model(**long_ids)
    except Exception as e:                                   # the hard failure that truncation normally hides
        print(f"   untruncated {long_ids['input_ids'].shape[1]}-token input -> {type(e).__name__}: {str(e)[:70]}")

    # ---- E) retrieval quality of token windows --------------------------------------------------------------
    print("\nE) retrieval over all 13 documents (74 questions), dense MiniLM")
    rows = []
    for size in (64, 128, 256):
        for ov in (0, int(size * 0.15)):
            chunks = [c for d in docs for c in fixed_token_windows(d, size, ov)]
            r = evaluate(chunks, docs=docs)["dense"]
            rows.append([size, ov, len(chunks), f"{r['hit@1']:.2f}", f"{r['hit@3']:.2f}", f"{r['mrr']:.2f}", f"{r['ctx_tok']:.0f}"])
    print(table(rows, ["size", "overlap", "chunks", "hit@1", "hit@3", "MRR", "ctx tokens@3"]))


def get_embedder_model():
    from common19 import get_embedder
    return get_embedder()


if __name__ == "__main__":
    main()

# TRY IT: change size=128 to 64 in section C; how many windows do you get and what is the new max token size?
# TRY IT: add a sample in your own language to `samples`; which tokenizer is cheapest for it, and is MiniLM [UNK] > 0?
# BREAK IT: call fixed_token_windows(doc, size=128, overlap=128) -> AssertionError (step would be 0).
