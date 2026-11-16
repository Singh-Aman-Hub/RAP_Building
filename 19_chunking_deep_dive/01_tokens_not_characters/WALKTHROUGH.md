# Step 01 — Tokens, not characters

## The story so far
This is the first step of module 19. Module 14 [step 02 (chunking strategies)](../../14_rag_and_retrieval/02_chunking_strategies/WALKTHROUGH.md) compared five splitters by counting *characters* and *words*, and ended with "tune on your own evaluation set". Before we can tune anything we have to fix the ruler: an embedding model does not read characters, it reads **tokens**, and it silently throws away everything past its limit. Back to the [module README](../README.md).

## The pain point
"Chunk size 500" is meaningless until you say 500 *what*. Five hundred characters is about 110 tokens of English prose but about 300 tokens of JSON and over 400 tokens of Chinese. If the chunks are cut by characters, their token sizes wander, some blow past the embedder's 256-token limit, and the part beyond the limit is **never read**. The vector looks fine, the pipeline raises no error, and a fact at the tail of the chunk is simply invisible to search.

## The spark
Measure with the **real tokenizer of the embedding model** (here the WordPiece tokenizer of `all-MiniLM-L6-v2`). Ask the tokenizer for `offset_mapping` (the character span of every token), cut the *token list* into windows, and map each window back to an exact slice of the original text. Now every chunk is guaranteed `<= N` tokens, the text is untouched, and we know the character offsets for citations and for the evaluation harness used in the rest of the module.

## Goal
1. Count characters, words and tokens for the same documents and see that they disagree by 2x or more.
2. Compare characters per token across content types and languages, on three tokenizers (WordPiece, SentencePiece, byte BPE).
3. Build `fixed_token_windows` (size + overlap) and show it keeps every window under the limit, unlike character windows.
4. Prove the silent truncation: a 481-token text embeds to *exactly* the same vector as its first 256 tokens.
5. Score token windows of 64 / 128 / 256 tokens, with and without overlap, on the 74 labelled questions.
6. Read, once, the helper module `common19.py` that every later step imports.

## The theory (read this before the interview)
**Token** = the unit a model's vocabulary is built from. A tokenizer maps text to a list of integer ids (`input_ids`). Three families you will meet:
```
WordPiece (BERT, MiniLM)      "unaffordable" -> un ##afford ##able     ## marks a continuation; unknown pieces -> [UNK]
SentencePiece (e5 / XLM-R: Unigram; Llama-2: BPE)  "_un afford able"  "_" marks a word start; works on raw text, no pre-split
byte-level BPE (GPT-2/3/4, Llama-3, Qwen)    bytes merged by frequency  never [UNK]: any string is bytes
```
**Characters per token** is not a constant. Rules of thumb (about 4 chars per token for English) fail on code, JSON, URLs, hashes and non-Latin scripts, and differ by tokenizer. The only safe unit is the token count *of the model you will embed with*.

**Why the unit matters for chunking.** Three hard limits are all in tokens:
```
embedder max_seq_length   (all-MiniLM-L6-v2: 256 in sentence-transformers; BERT position table: 512)
LLM context window        (what the retrieved chunks must fit into)
chunk quality sweet spot  (a vector averages everything it reads; long chunks blur, short ones lack context)
```
**Silent truncation.** `tokenizer(text, truncation=True, max_length=256)` keeps the first 256 tokens and drops the rest without a warning. With a mean-pooling model, the vector is the mean of the *kept* token vectors, so a chunk of 481 tokens and its first 256 tokens give the same vector. Without truncation, a BERT-type model raises an error once the input is longer than its position table (512).

**Token windows with overlap.** Window `k` covers tokens `[k*(size-overlap), k*(size-overlap)+size)`. The number of windows is about `ceil((T - overlap) / (size - overlap))`, and the stored tokens are about `T * size/(size-overlap)` (overlap costs index size: 15% overlap stores about 1.18x the text). `overlap < size` is required or the step is zero.

**Offsets.** A fast tokenizer returns `offset_mapping = [(char_start, char_end), ...]`, one pair per token. Cutting at token `i` and `j` then means `text[offs[i][0] : offs[j-1][1]]`. This keeps the original text exact (no detokenisation artefacts such as lower-casing or lost whitespace).

**Status (fundamentals vs practice).** Counting in the embedder's tokens is *fundamental*. Fixed windows with 10-20% overlap are the *common baseline* (a fine place to start, rarely the end state). Newer long-context embedders (8k tokens and more) move the limit but do not remove it.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| token | subword, piece, wordpiece, BPE token, `input_id` | yes | One vocabulary entry; not a word and not a character |
| max sequence length | `max_seq_length`, `max_length`, context length of the *embedder*, "token limit" | near | Embedder limit (256 here) is not the same as the LLM context window (thousands) |
| silent truncation | truncation, clipping, "tail drop" | yes | The input beyond `max_length` is dropped with no error |
| offset mapping | `return_offsets_mapping`, character spans, alignment | yes | (start, end) in the *original* string for every token |
| fixed-size chunking | fixed windows, sliding window, token windows | near | Cut every N units; "sliding window" implies overlap |
| overlap | stride, `chunk_overlap`, sliding step = size - overlap | near | Overlap is shared tokens; stride is the step between starts |
| WordPiece / BPE / Unigram | subword tokenizers | no | Different algorithms: likelihood merges, frequency merges, probabilistic segmentation |
| [UNK] | unknown token, out-of-vocabulary | yes | A piece the vocabulary cannot represent: information lost |

## Run it
```bash
cd 19_chunking_deep_dive/01_tokens_not_characters
python main.py
# from the repo root:
HF_HUB_OFFLINE=1 .venv/bin/python 19_chunking_deep_dive/01_tokens_not_characters/main.py
```
About 25-30 s on a laptop CPU (it loads the cached MiniLM model and scores 6 chunkings of 74 questions). Needs `torch` + `transformers` and the cached models `all-MiniLM-L6-v2`, `multilingual-e5-small`, `gpt2`; nothing is downloaded (`HF_HUB_OFFLINE=1`).

## Code, step by step
The order is: setup, the two chunkers, then sections A to E of `main()`. After the `main.py` blocks comes a tour of `common19.py`, the helper module shared by the whole module; later steps just link back here.

### 1. Header docstring, imports and the helper module  (`main.py` lines 1–23)

```python
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
```

- **In simple words:** the docstring is the file's contract (goal, how to run, expected output, needs); then we pull in the shared helpers. `sys.path.insert` lets Python find `common19.py`, which sits one folder above this step.
- **Exactly:** `Path(__file__).resolve().parents[1]` is the module folder `19_chunking_deep_dive`. Importing `common19` also sets `HF_HUB_OFFLINE=1` (so nothing can download) and imports module 14's `common.py` (Embedder, BM25, RRF). The `# noqa: E402` comment tells the linter "yes, an import after code is intended".
- **Why it is here:** folder names start with a digit (`01_...`), so they cannot be imported as packages. Every step adds the parent folder to `sys.path` and imports from `common19` instead.

### 2. `fixed_token_windows`: cut by tokens, return exact slices  (`main.py` lines 26–39)

```python
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
```

- **In simple words:** turn the document into a list of tokens, then take windows of `size` tokens, each starting `size - overlap` tokens after the previous one, and map each window back to the original characters.
- **Exactly:** `token_spans(text)` returns one `(char_start, char_end)` per token (see the tour below). The loop `range(0, max(len(spans) - overlap, 1), step)` visits the window starts; `window[0][0]` is the first token's start and `window[-1][1]` the last token's end, so `doc["text"][start:end]` is an exact slice. A `RawChunk` stores `doc_id`, the offsets, the text and the token range in `meta`. The `if i + size >= len(spans): break` guard stops as soon as a window reaches the end, so we never emit a tail window made only of overlap. The `assert` makes `overlap == size` fail immediately instead of looping forever with step 0.
- **Why it is here:** this is the reference chunker for the module and the baseline in the scoreboard of [step 10](../10_chunking_evaluation/WALKTHROUGH.md). Offsets, not detokenised strings, are what let step 10 compare a 40-token chunk and a 250-token section on the same footing.

### 3. `fixed_char_windows`: the naive version  (`main.py` lines 42–45)

```python
def fixed_char_windows(doc, size_chars):
    """The naive version: every `size_chars` characters. Used only to show how unstable its token size is."""
    t = doc["text"]
    return [RawChunk(doc["doc_id"], i, min(i + size_chars, len(t)), t[i:i + size_chars]) for i in range(0, len(t), size_chars)]
```

- **In simple words:** cut every `size_chars` characters, whatever they contain.
- **Exactly:** a list comprehension over `range(0, len(t), size_chars)`; each piece is `t[i:i+size_chars]` with `min` protecting the last piece. No tokenizer is involved.
- **Why it is here:** only to show how unstable its token size is (section C). Never use it for embedding.

### 4. Section A: characters, words, tokens for the same documents  (`main.py` lines 48–58)

```python
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
```

- **In simple words:** for the four new messy documents print chars, whitespace words, real tokens, and two ratios.
- **Exactly:** `count_tokens(text)` is the number of WordPiece tokens (no `[CLS]`/`[SEP]`). `table(rows, headers)` prints a fixed-width table (see the tour). `c / t` is characters per token, `t / w` tokens per word.
- **Why it is here:** the first evidence that "1 token = 4 characters" is only an average. The API reference (tables, code) is much more expensive per character than the prose.

### 5. Section B (setup): three tokenizers and eight sample texts  (`main.py` lines 60–74)

```python
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
```

- **In simple words:** load two more tokenizers and prepare short samples in different content types and languages.
- **Exactly:** `AutoTokenizer.from_pretrained(name)` loads the tokenizer files (vocabulary + rules) from the local Hugging Face cache. `toks` maps a label to a tokenizer object. The samples cover prose, code, JSON, a URL with a hash, German, Hindi, Chinese and emoji.
- **Why it is here:** the same text costs a different number of tokens on each tokenizer, and the only way to know is to count with the one you will embed with.

### 6. Section B (loop): characters per token and [UNK] count  (`main.py` lines 75–85)

```python
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
```

- **In simple words:** for every sample and tokenizer print "chars per token (token count)", and count how many pieces MiniLM could not represent.
- **Exactly:** `tk(s, add_special_tokens=False)["input_ids"]` is the list of ids. `len(s) / len(ids)` is the ratio (higher is cheaper). `unk_token_id` is the id of `[UNK]`; the sum counts how many tokens equal it.
- **Why it is here:** shows two separate problems: cost (Hindi costs 115 tokens on GPT-2's byte BPE for 75 characters) and loss (19 of 24 Chinese tokens are `[UNK]` on MiniLM: that model cannot read Chinese at all).

### 7. Section C: token windows vs character windows  (`main.py` lines 87–98)

```python
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
```

- **In simple words:** chunk the handbook both ways with "the same" size and compare the real token sizes.
- **Exactly:** `next(...)` picks the handbook. `fixed_token_windows(doc, 128, 20)` gives windows of at most 128 tokens. `128 * 4 = 512` characters is the "4 characters per token" rule of thumb applied by a naive pipeline. `all(doc["text"][c.start:c.end] == c.text ...)` confirms offsets give exact slices. The last print shows the overlap: window 0 ends at token 128 and window 1 starts at token 108, so 20 tokens are shared.
- **Why it is here:** token windows have a hard upper bound; character windows only have an average.

### 8. Section D: what silently happens past `max_seq_length`  (`main.py` lines 100–121)

```python
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
```

- **In simple words:** take a question whose answer sits at token ~400 of the handbook, embed (a) the first 481 tokens, (b) the first 256 tokens, (c) a small 81-token window centred on the answer, and compare each with the question.
- **Exactly:** `ids_all["offset_mapping"]` gives token spans; `a_tok` is the index of the first token whose end is after the answer's start character. The conditional `300 < a_tok < 440` chooses a question whose answer is beyond token 256 but inside the 481-token prefix. `embed([...])` (see the tour) returns unit vectors, so `@` is cosine similarity. The embedder truncates at 256 tokens, so `v_full` and `v_trunc` are the same vector (cosine 1.000000). The last `try` feeds 964 tokens straight to the model *without* truncation and shows the hard failure (BERT has 512 position embeddings) that normal truncation hides.
- **Why it is here:** this is the most important picture of the step. The question vs the whole text and vs the first 256 tokens are the same number (0.1543): the answer is invisible. The small window on the answer scores 0.5946. Cutting around the answer beats stuffing more text in.

### 9. Section E: retrieval quality of token windows  (`main.py` lines 123–131)

```python
    # ---- E) retrieval quality of token windows --------------------------------------------------------------
    print("\nE) retrieval over all 13 documents (74 questions), dense MiniLM")
    rows = []
    for size in (64, 128, 256):
        for ov in (0, int(size * 0.15)):
            chunks = [c for d in docs for c in fixed_token_windows(d, size, ov)]
            r = evaluate(chunks, docs=docs)["dense"]
            rows.append([size, ov, len(chunks), f"{r['hit@1']:.2f}", f"{r['hit@3']:.2f}", f"{r['mrr']:.2f}", f"{r['ctx_tok']:.0f}"])
    print(table(rows, ["size", "overlap", "chunks", "hit@1", "hit@3", "MRR", "ctx tokens@3"]))
```

- **In simple words:** build chunks for all 13 documents at 3 sizes, each with no overlap and with about 15% overlap, and score them.
- **Exactly:** `int(size * 0.15)` is 9, 19 and 38 tokens. `evaluate(chunks, docs=docs)` embeds every chunk, ranks the chunks for each of the 74 questions with dense search, and returns metrics (see the tour); `["dense"]` selects the dense-retriever row. `ctx_tok` is the number of tokens of the top-3 chunks, i.e. what you would send to the LLM.
- **Why it is here:** a first baseline, and an honest warning: the differences between rows are small (one question is 0.014; see step 10 for confidence intervals).

### 10. Helper, entry point and the TRY IT / BREAK IT lines  (`main.py` lines 134–144)

```python
def get_embedder_model():
    from common19 import get_embedder
    return get_embedder()


if __name__ == "__main__":
    main()

# TRY IT: change size=128 to 64 in section C; how many windows do you get and what is the new max token size?
# TRY IT: add a sample in your own language to `samples`; which tokenizer is cheapest for it, and is MiniLM [UNK] > 0?
# BREAK IT: call fixed_token_windows(doc, size=128, overlap=128) -> AssertionError (step would be 0).
```

- **In simple words:** a tiny accessor for the embedder model, the `if __name__` guard, and the exercises as comments.
- **Exactly:** `get_embedder_model()` imports `get_embedder` lazily and returns the cached `Embedder` (so its `.model` can be called directly in section D). The guard makes `main()` run only when the file is executed, not when a later step imports `fixed_token_windows` with `load_step`.
- **Why it is here:** steps 02 to 10 import `fixed_token_windows` from this file, so the module must not run its experiments at import time.

## The shared helpers, once: a tour of `common19.py`
Everything below is imported by later steps without re-explanation. If a later step uses a helper in a new way, that step says so.

### T1. Offline setup and the module-14 helpers  (`common19.py` lines 31–40)

```python

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
M14 = REPO / "14_rag_and_retrieval"
sys.path.insert(0, str(M14))
import common as c14   # noqa: E402  (module 14's helpers: Embedder, BM25, rrf_fuse, load_documents)

DATA = HERE / "data"
EMBED_MODEL = c14.EMBED_MODEL

```

- **In simple words:** make sure nothing can download, and borrow module 14's embedding and retrieval code.
- **Exactly:** `os.environ.setdefault` sets `HF_HUB_OFFLINE=1` and `TOKENIZERS_PARALLELISM=false` only if they are not set already. `import common as c14` is [14_rag_and_retrieval/common.py](../../14_rag_and_retrieval/COMMON_WALKTHROUGH.md): `Embedder` (MiniLM, mean pooling, L2-normalised, `max_length` 256), `BM25`, `dense_search`, `rrf_fuse`, `load_documents`, `load_eval_set`.
- **Why it is here:** one embedding model, one BM25 and one fusion rule for the whole module, so strategies differ only in how they chunk.

### T2. `get_tokenizer`, `count_tokens`, `token_spans`  (`common19.py` lines 43–61)

```python
@lru_cache(maxsize=1)
def get_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    tok.model_max_length = 10**9            # silence the "longer than max length" warning; we measure on purpose
    return tok


@lru_cache(maxsize=200_000)
def count_tokens(text: str) -> int:
    """Number of WordPiece tokens the embedding model sees for `text` (special [CLS]/[SEP] not counted)."""
    return len(get_tokenizer()(text, add_special_tokens=False)["input_ids"])


def token_spans(text: str):
    """[(char_start, char_end), ...] one pair per token: lets us cut by tokens but return exact text slices."""
    enc = get_tokenizer()(text, add_special_tokens=False, return_offsets_mapping=True)
    return enc["offset_mapping"]

```

- **In simple words:** the ruler. `count_tokens` answers "how many tokens does the embedder see for this text?", `token_spans` says where each token sits in the string.
- **Exactly:** `lru_cache` on `get_tokenizer` loads the tokenizer once; `count_tokens` is cached on the text, so counting the same string twice is free. `tok(text, add_special_tokens=False)` returns a dict with `input_ids`; `return_offsets_mapping=True` adds `offset_mapping`, a list of `(char_start, char_end)`. A fast (Rust) tokenizer computes this in one pass. `model_max_length = 10**9` only silences the "longer than the maximum" warning because we measure long text on purpose.
- **Why it is here:** every chunker in the module sizes its pieces with `count_tokens`.

### T3. `RawChunk` and `make_chunk`  (`common19.py` lines 64–91)

```python
@dataclass
class RawChunk:
    doc_id: str
    start: int                    # char offset in the ORIGINAL document where the source region starts
    end: int                      # ... and ends (exclusive)
    text: str                     # what was cut out
    embed_text: str = ""          # what gets embedded / BM25-indexed (text + header/context); default = text
    context_text: str = ""        # what is sent to the LLM when retrieved; default = text (parent text for small-to-big)
    spans: list = field(default_factory=list)   # source spans actually contained in context_text
    section_path: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.embed_text = self.embed_text or self.text
        self.context_text = self.context_text or self.text
        self.spans = self.spans or [(self.start, self.end)]


def make_chunk(doc, start, end, **kw):
    """Chunk from a slice of the document; strips whitespace and moves the offsets with it."""
    raw = doc["text"][start:end]
    lead = len(raw) - len(raw.lstrip())
    text = raw.strip()
    return RawChunk(doc["doc_id"], start + lead, start + lead + len(text), text, **kw)


# ------------------------------------------------------------------ sentences --------------------------
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`\"(*|])|\n+")
```

- **In simple words:** the light record every chunker returns, plus a helper that cuts a slice and trims whitespace.
- **Exactly:** a `dataclass` is a class whose `__init__`, `__repr__` and `__eq__` are generated from the annotated fields. `start`/`end` are character offsets in the *original* document. Three texts can differ: `text` (the cut), `embed_text` (what is embedded and BM25-indexed, e.g. text plus a heading) and `context_text` (what is sent to the LLM, e.g. a parent). `spans` lists the source regions that `context_text` really contains. `__post_init__` fills the defaults. `make_chunk` strips whitespace and shifts the offsets by the number of stripped leading characters so the offsets still point at the text.
- **Why it is here:** separating *what is searched* from *what is shown* is the idea behind contextual headers (step 03), small-to-big (step 07) and propositions (step 08). Step 11 builds the production version of this record.

### T4. `sentence_spans`  (`common19.py` lines 94–106)

```python
def sentence_spans(text: str, base=0):
    """[(start, end)] of sentences; boundaries = end punctuation + space + capital, or any newline."""
    spans, pos = [], 0
    for m in list(_SENT.finditer(text)) + [None]:
        end = m.start() if m else len(text)
        seg = text[pos:end]
        if seg.strip():
            lead = len(seg) - len(seg.lstrip())
            spans.append((base + pos + lead, base + pos + lead + len(seg.strip())))
        if m:
            pos = m.end()
    return spans

```

- **In simple words:** a regex sentence splitter that remembers where each sentence is.
- **Exactly:** `_SENT` matches a boundary: end punctuation, whitespace, then a capital letter / digit / quote / bracket (lookbehind and lookahead keep the punctuation and the capital in the sentences), or any newline. The loop turns the boundaries into `(start, end)` pairs, trims whitespace and adds `base` so spans can be relative to a larger text. Returns `[(start, end), ...]`.
- **Why it is here:** steps 03, 04, 05, 07, 08 all work on sentences; this is deliberately simple (no abbreviation handling such as "Dr. Smith"); a production system would use a library splitter.

### T5. `load_docs`, `load_eval`, `answer_spans`  (`common19.py` lines 109–147)

```python
def load_docs():
    """9 module-14 documents + 4 new messy documents. Each: {doc_id, title, text, corpus}."""
    docs = [dict(d, corpus="kb") for d in c14.load_documents()]
    for p in sorted(DATA.glob("*")):
        if p.suffix in (".md", ".txt"):
            t = p.read_text(encoding="utf-8")
            first = t.splitlines()[0]
            docs.append({"doc_id": p.stem, "title": first.lstrip("# ").strip(), "text": t, "corpus": "new"})
    return docs


def load_eval():
    """All questions in one format: {id, question, doc_id, answers:[exact strings], kind, corpus}."""
    docs = {d["doc_id"]: d["text"] for d in load_docs()}
    out = []
    for q in c14.load_eval_set():
        if not q["gold"]:
            continue                                         # skip the unanswerable ones: no span to find
        doc_id, section = q["gold"][0].split("#")
        text = docs[doc_id]
        s0 = text.index("## " + section)                     # restrict the search to the gold section
        s1 = text.find("\n## ", s0 + 3)
        body = text[s0: s1 if s1 != -1 else len(text)]
        i = body.lower().find(q["answer"].lower())
        out.append({"id": q["id"], "question": q["question"], "doc_id": doc_id,
                    "answers": [body[i:i + len(q["answer"])]], "kind": q["kind"], "corpus": "kb"})
    for q in json.loads((DATA / "eval_questions.json").read_text(encoding="utf-8")):
        out.append(dict(q, corpus="new"))
    return out


def answer_spans(q, docs_by_id):
    text = docs_by_id[q["doc_id"]]["text"]
    return [(text.index(a), text.index(a) + len(a)) for a in q["answers"]]


# ------------------------------------------------------------------ embedding + retrieval --------------
_EMB = None
_CACHE = {}
```

- **In simple words:** the test bed: 13 documents and 74 questions in one format.
- **Exactly:** `load_docs` returns dicts `{doc_id, title, text, corpus}`: the nine module-14 knowledge-base files (`corpus="kb"`) plus the four new `.md`/`.txt` files in `data/` (`corpus="new"`). `load_eval` converts module 14's 24 answerable questions (the gold section is searched for the answer string to get the exact answer text) and appends the 50 questions of `data/eval_questions.json`; each has `answers`, a list of exact strings (two for "multi" questions). `answer_spans` turns answer strings into `(start, end)` character spans.
- **Why it is here:** relevance is defined by *character offsets*, not by chunk ids, so any chunker can be scored against the same labels.

### T6. `embed`, `get_embedder`, `covers`  (`common19.py` lines 150–168)

```python
def get_embedder():
    global _EMB
    if _EMB is None:
        _EMB = c14.Embedder(max_length=256)
    return _EMB


def embed(texts):
    """Embed with a per-process cache keyed by text (the same string is never embedded twice)."""
    need = [t for t in dict.fromkeys(texts) if t not in _CACHE]
    if need:
        for t, v in zip(need, get_embedder().encode(need)):
            _CACHE[t] = v
    return np.vstack([_CACHE[t] for t in texts])


def covers(chunk, doc_id, span):
    return chunk.doc_id == doc_id and any(s <= span[0] and span[1] <= e for s, e in chunk.spans)

```

- **In simple words:** embed text once, remember it, and decide whether a chunk contains an answer.
- **Exactly:** `get_embedder()` creates module 14's `Embedder(max_length=256)` once. `embed(texts)` returns an `(n, 384)` float32 matrix of unit vectors; `_CACHE` maps text to vector, so the same string is embedded once per process (step 10 clears the cache on purpose when it times ingestion). `covers(chunk, doc_id, span)` is true when some source span of the chunk fully contains the answer span.
- **Why it is here:** `covers` is the whole definition of "relevant" in the module: the chunk's source span must include the complete answer sentence. A chunk that only holds half the answer does not count.

### T7. `char_precision` and `evaluate`: the one harness  (`common19.py` lines 170–233)

```python
def _merge(spans):
    out = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def char_precision(ctxs, answer_spans_, doc_id):
    """Context precision: share of the characters sent to the LLM (top contexts of the right document, source spans only)
    that are the answer evidence. Low = you pay for text that does not help. (A cheap chunk-attribution metric.)"""
    ctx = _merge([sp for c in ctxs if c.doc_id == doc_id for sp in c.spans])
    total = sum(e - s for s, e in ctx) + sum(len(c.context_text) for c in ctxs if c.doc_id != doc_id)
    gold = sum(max(0, min(e, ge) - max(s, gs)) for s, e in ctx for gs, ge in _merge(answer_spans_))
    return gold / total if total else 0.0


def evaluate(chunks, questions=None, docs=None, ks=(1, 3, 5), vectors=None, bm25_texts=None, postprocess=None):
    """Score a list of RawChunk on all questions. Returns {retriever: {metric: value}} plus size numbers.

    hit@k    : 1 if ANY answer sentence is covered by a top-k chunk
    recall@k : share of the answer sentences covered by the union of the top-k chunks
    MRR      : 1 / rank of the first covering chunk
    ctx_tok  : tokens of the top-3 context_text strings (what you would pay to send to the LLM)
    """
    docs = docs or load_docs()
    questions = questions or load_eval()
    by_id = {d["doc_id"]: d for d in docs}
    matrix = vectors if vectors is not None else embed([c.embed_text for c in chunks])   # (n_chunks, 384); late chunking passes its own
    bm25 = c14.BM25(bm25_texts or [c.embed_text for c in chunks])   # contextual BM25 passes its own texts
    qvecs = embed([q["question"] for q in questions])
    rows = {"dense": [], "bm25": [], "hybrid": []}
    per_q = {}
    for q, qv in zip(questions, qvecs):
        spans = answer_spans(q, by_id)
        dense = c14.dense_search(qv, matrix, k=min(20, len(chunks)))
        sparse = bm25.search(q["question"], 20)
        hybrid = c14.rrf_fuse([sparse, dense], top_n=20)
        for name, ranking in (("dense", dense), ("bm25", sparse), ("hybrid", hybrid)):
            order = [j for j, _ in ranking]
            # postprocess (optional) turns the ranked chunk ids into the CONTEXT objects actually sent to the LLM:
            # parent lookup + de-duplication (small-to-big), merging siblings (auto-merging), ...
            ctxs = postprocess(order) if postprocess else [chunks[j] for j in order]
            cov = [[covers(c, q["doc_id"], sp) for sp in spans] for c in ctxs]  # (rank, span)
            first = next((r for r, c in enumerate(cov, 1) if any(c)), None)
            row = {"mrr": 1 / first if first else 0.0, "rank": first,
                   "ctx_tok": sum(count_tokens(c.context_text) for c in ctxs[:3]),
                   "prec3": char_precision(ctxs[:3], spans, q["doc_id"])}
            for k in ks:
                row[f"hit@{k}"] = float(any(any(c) for c in cov[:k]))
                row[f"recall@{k}"] = float(np.mean([any(c[i] for c in cov[:k]) for i in range(len(spans))]))
            rows[name].append(dict(row, kind=q["kind"], corpus=q["corpus"], id=q["id"]))
        per_q[q["id"]] = rows["dense"][-1]["rank"]
    res = {}
    for name, rs in rows.items():
        res[name] = {m: float(np.mean([r[m] for r in rs])) for m in rs[0] if m not in ("kind", "corpus", "id", "rank")}
        res[name]["rows"] = rs
    res["n_chunks"] = len(chunks)
    res["stored_tok"] = sum(count_tokens(c.embed_text) for c in chunks)
    res["mean_tok"] = res["stored_tok"] / max(len(chunks), 1)
    res["per_q_rank"] = per_q
    return res
```

- **In simple words:** the function that scores any list of chunks. It builds a dense index and a BM25 index, runs all 74 questions, and averages the metrics.
- **Exactly:** `matrix` is `(n_chunks, 384)`; `vectors=` lets late chunking pass its own, `bm25_texts=` lets contextual BM25 index different text, `postprocess=` turns a ranked list of chunk ids into the *context objects* sent to the LLM (parents, merged siblings). Three retrievers are scored: `dense` (cosine via `dense_search`), `bm25` (module 14's BM25), and `hybrid` (`rrf_fuse`, reciprocal rank fusion of both rankings). Per question and retriever: `hit@k` (any top-k context covers any answer span), `recall@k` (share of answer spans covered; matters for two-evidence questions), `mrr` (1/rank of the first covering context), `ctx_tok` (tokens of the top-3 `context_text`, i.e. what you pay for), `prec3` (character precision of the top-3: the share of sent characters that are answer evidence). Then it averages. `n_chunks`, `stored_tok` and `mean_tok` describe the index size.
- **Why it is here:** one harness for every strategy means "which chunker is better?" becomes a measurement. Step 10 [explains the metrics and their noise](../10_chunking_evaluation/WALKTHROUGH.md); the retrieval-metrics theory itself is in [14/09](../../14_rag_and_retrieval/09_rag_evaluation/WALKTHROUGH.md).

### T8. `table`, `chunk_id`, `load_step`, `sparkline`  (`common19.py` lines 236–261)

```python
def table(rows, headers):
    """Tiny fixed-width table printer: rows = list of lists."""
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = lambda r: "  ".join(str(x).ljust(w[i]) if i == 0 else str(x).rjust(w[i]) for i, x in enumerate(r))
    return "\n".join([line(headers), line(["-" * x for x in w])] + [line(r) for r in rows])


def chunk_id(doc_id, start, end, text):
    return hashlib.sha1(f"{doc_id}|{start}|{end}|{text}".encode()).hexdigest()[:12]


def load_step(folder):
    """Import <folder>/main.py by file path (so step 10 can reuse the chunkers written in steps 01-08)."""
    path = HERE / folder / "main.py"
    spec = importlib.util.spec_from_file_location("m_" + folder, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def sparkline(values, lo=None, hi=None):
    bars = "▁▂▃▄▅▆▇█"
    lo = min(values) if lo is None else lo
    hi = max(values) if hi is None else hi
    return "".join(bars[min(7, int((v - lo) / (hi - lo + 1e-9) * 8))] for v in values)
```

- **In simple words:** four small utilities.
- **Exactly:** `table` prints a fixed-width text table (first column left-aligned, others right-aligned). `chunk_id` is a short SHA-1 of doc id, span and text (used before step 11 builds proper ids). `load_step("02_recursive_splitter_rebuilt")` imports that folder's `main.py` as a module with `importlib` (needed because the folder name starts with a digit), registers it in `sys.modules`, and returns it, which is how step 10 reuses the chunkers of steps 01 to 08. `sparkline(values)` maps numbers to the eight block characters `▁..█`.
- **Why it is here:** reuse instead of copy: no chunker is implemented twice in this module.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `Path(__file__).resolve().parents[1]` | Absolute path of the module folder (the parent of this step's folder) |
| `from x import (a, b,  # noqa: E402` | Multi-line import; the comment silences the "import not at top" lint |
| `tok(text, return_offsets_mapping=True)` | Fast tokenizer call that also returns `(char_start, char_end)` per token |
| `spans[i:i + size]` | List slice: tokens `i` .. `i+size-1` (shorter at the end) |
| `f"{x:.2f}"`, `f"{name:<30}"` | Format: 2 decimals; left-align in 30 columns |
| `next(gen, default)` | First item of a generator (or the default); used to find "the first question whose answer is deep in the text" |
| `a @ b` | Dot product; for unit vectors it is cosine similarity |
| `@lru_cache` | Memoise a function on its arguments |
| `try: ... except Exception as e:` | Show the hard failure that truncation normally hides |

## Libraries & functions
| Function | What it does | Internally |
|---|---|---|
| `AutoTokenizer.from_pretrained(name)` | Loads a tokenizer from the local HF cache | Reads `tokenizer.json` / vocab files; builds a Rust `tokenizers` object (fast tokenizer) |
| `tok(text, add_special_tokens=False)` | Text to `input_ids` | Normalise, pre-split, apply WordPiece/BPE/Unigram merges, map pieces to ids |
| `return_offsets_mapping=True` | Adds character spans per token | The Rust tokenizer tracks the original offsets through normalisation |
| `Embedder.encode` (module 14) | Texts to unit vectors | Tokenise with truncation at 256, run BERT, mean-pool real tokens, L2-normalise |
| `model(**ids)` on 964 tokens | Raw forward pass | Fails: `position_embeddings` has only 512 rows, so the add of token + position embeddings has mismatched sizes |
| `np.percentile`, `np.mean` | Aggregates | Plain NumPy |

## Execution trace (real run)
This is real output from `python main.py` (the progress bar lines of the model loader are removed; times vary by machine).
```
A) the same documents, three ways to measure them
document             chars  words  tokens  chars/token  tokens/word
-------------------  -----  -----  ------  -----------  -----------
api_reference         3180    489     933         3.41         1.91
employee_handbook     4592    787    1046         4.39         1.33
incident_postmortem   4139    716     880         4.70         1.23
support_faq           2628    465     644         4.08         1.38

B) characters per token (higher = cheaper). [UNK] = token the model cannot represent at all
text           chars  MiniLM (WordPiece)  e5-small (SentencePiece)  gpt2 (byte BPE)  MiniLM [UNK]
-------------  -----  ------------------  ------------------------  ---------------  ------------
English prose     86            5.4 (16)                  4.8 (18)         5.4 (16)             0
Python code      126            2.5 (50)                  2.5 (50)         2.7 (46)             0
JSON              79            1.7 (46)                  1.7 (47)         2.0 (40)             0
URL + hash        73            1.6 (47)                  1.6 (46)         1.6 (45)             0
German            90            2.9 (31)                  4.5 (20)         2.9 (31)             0
Hindi             75            1.5 (49)                  3.9 (19)        0.7 (115)             1
Chinese           24            1.0 (24)                  1.6 (15)         0.5 (48)            19
Emoji             44            4.0 (11)                  2.8 (16)         1.8 (24)             2

C) handbook (1046 tokens): token windows size=128 overlap=20 -> 10 chunks
   token sizes: min 74, max 128  (every one <= 128; exact slices: True)
   character windows of 512 chars -> 9 chunks, token sizes: min 104, max 132, 1 of them above 128
   overlap check: window 0 ends at token 128, window 1 starts at token 108


D) question: 'How many weeks off does the main caregiver of a new baby get?'  (answer starts at token 403)
   embedder max_length = 256; input has 481 tokens -> 225 tokens never read
   cos(question, whole 481-token text)    = 0.1543
   cos(question, first 256 tokens only)   = 0.1543   <- identical vector: cos(full, trunc) = 1.000000
   cos(question, 81-token window on answer) = 0.5946
   untruncated 964-token input -> RuntimeError: The size of tensor a (964) must match the size of tensor b (512) at no

E) retrieval over all 13 documents (74 questions), dense MiniLM
size  overlap  chunks  hit@1  hit@3   MRR  ctx tokens@3
----  -------  ------  -----  -----  ----  ------------
64          0      86   0.68   0.85  0.78           180
64          9      93   0.68   0.91  0.80           186
128         0      48   0.72   0.89  0.81           337
128        19      51   0.68   0.88  0.79           341
256         0      25   0.65   0.88  0.77           623
256        38      26   0.70   0.95  0.82           657
```

How to read it:
- **A:** the API reference is 3.41 characters per token because of tables, code and `|` separators; the prose postmortem is 4.70. A single "4 characters per token" constant is wrong by 30% inside one corpus.
- **B:** JSON and a URL with a hash cost about 1.6 to 1.7 characters per token. German is 2.9 on MiniLM but 4.5 on e5 (a multilingual vocabulary). Chinese is 1.0 on MiniLM with 19 of 24 tokens `[UNK]`: MiniLM *cannot read* Chinese. GPT-2's byte BPE needs 115 tokens for the 75-character Hindi sentence.
- **C:** token windows are all `<= 128` and exact slices (`True`). The character windows of "the same" size reach 132 tokens and 1 of 9 is over the limit. Window 0 ends at token 128 and window 1 starts at token 108: 20 tokens of overlap, as asked.
- **D:** 225 tokens of the 481 are never read. The question's cosine with the whole text and with the first 256 tokens is identical (0.1543, and cos(full, trunc) = 1.000000), while an 81-token window on the answer gives 0.5946. Without truncation the model fails with `RuntimeError` (964 tokens against 512 positions).
- **E:** hit@3 moves between 0.85 and 0.95 across sizes and overlaps, but not monotonically (size 64: 0.85 without overlap, 0.91 with; size 256: 0.88 and 0.95). With 74 questions one question is 0.014, so most of these gaps are noise; the clear signal is context cost: 180 tokens for 64-token chunks, 623 to 657 for 256-token chunks.

## Experiments (change ONE thing)
| Change | What you will see | Why |
|---|---|---|
| Section C with `size=64, overlap=10` | 20 windows instead of 10, largest window 64 tokens (real run: [solutions](../solutions/solutions.md)) | The bound is the `size` argument; the number of windows is about `T / (size - overlap)` |
| Add a Japanese and an Arabic sample to `samples` | e5-small is cheapest for both (Japanese 19 tokens vs 29 MiniLM vs 47 GPT-2; Arabic 16 vs 58 vs 70); MiniLM gives 12 `[UNK]` for the Japanese sentence and 0 for Arabic (WordPiece falls back to single characters) | MiniLM's vocabulary is English-centric; a multilingual vocabulary has these scripts. Worked answer: [solutions s01](../solutions/solutions.md) |
| Section E with `overlap = 50%` of size | More chunks and about 2x stored tokens (run it for the exact numbers) | Overlap duplicates text; it helps only when a fact lies across a cut, and costs index size |
| `max_length=128` in `common19.get_embedder` | Section D truncates earlier; 353 tokens never read instead of 225 | The truncation point is the embedder's `max_length`, not the chunker |
| Replace MiniLM by `intfloat/multilingual-e5-small` in `EMBED_MODEL` | Different token counts for every chunk and different windows | The ruler changes with the model: always count with the model you embed with |

## Debugging practice
**BREAK IT:** `fixed_token_windows(doc, size=128, overlap=128)`.
```
AssertionError: overlap must be smaller than size (step would be 0 or negative)
```
The guard fires before any loop starts. Without the `assert`, `range(0, n, 0)` would raise `ValueError: range() arg 3 must not be zero`, and a negative overlap-minus-size step would make `range` silently produce no windows at all. **Fix:** keep `0 <= overlap < size` (typically 10-20% of `size`). A related silent bug: counting with the wrong tokenizer (for example GPT-2's counts for a MiniLM index). Nothing crashes, and every chunk size is off by 10 to 100%.

## Common mistakes
- Setting chunk size in characters or words and never checking the embedder's token limit.
- Counting with `tiktoken` / GPT tokenizers for a different embedding model. The budget is the *embedder's* tokenizer.
- Assuming the tail of a long chunk is embedded. With `truncation=True` it is dropped without a warning; mean pooling hides the loss.
- Forgetting that `max_seq_length` (256 here) is smaller than the model's position limit (512), so "it did not crash" proves nothing.
- Using overlap to hide a bad cut. Overlap duplicates text (storage, retrieval of near-duplicates) and is not a substitute for cutting at natural boundaries ([step 02](../02_recursive_splitter_rebuilt/WALKTHROUGH.md)).
- Detokenising to build the chunk text (lower-cased, whitespace changed); use offsets into the original.
- Forgetting special tokens: the real input is `count + 2` (`[CLS]`, `[SEP]`).

## Rebuild checklist
1. `get_tokenizer()` (cached) and `count_tokens(text)` with `add_special_tokens=False`.
2. `token_spans(text)` with `return_offsets_mapping=True`.
3. `fixed_token_windows(doc, size, overlap)`: assert `overlap < size`; `step = size - overlap`; loop `range(0, max(len - overlap, 1), step)`; slice spans; offsets from first start to last end; break when a window reaches the end.
4. Verify the two invariants: every window `<= size` tokens, and `text[start:end] == chunk.text`.
5. Section D: find a question whose answer starts beyond token 256; show `cos(full) == cos(first 256)`.
6. Section E: `evaluate` for 3 sizes x 2 overlaps.

## Say it aloud
"A chunk size without a unit is meaningless: I measure in tokens of the embedding model, because that is what it reads and what it truncates. Characters per token ranges from about 5 for English prose down to 1.6 for hashes and 1.0 for Chinese on an English WordPiece vocabulary, so a character window has no guaranteed token size. Past `max_seq_length` a bi-encoder silently drops the tail, and mean pooling hides it, so a fact beyond the limit is invisible to search. I cut with the tokenizer's offset mapping so every chunk is at most N tokens and is an exact slice of the source, and I use 10 to 20% overlap only as a cheap hedge."

## The new problem (cliffhanger)
Fixed token windows are exact about size and blind to meaning: they cut in the middle of sentences, in the middle of a list, between a heading and its paragraph. The classic fix is to try the biggest natural boundary first (paragraph), then smaller ones (line, sentence, word) only when needed: the recursive splitter. Next: [step 02, the recursive splitter rebuilt](../02_recursive_splitter_rebuilt/WALKTHROUGH.md).
