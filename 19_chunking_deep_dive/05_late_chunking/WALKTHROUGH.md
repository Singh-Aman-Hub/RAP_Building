# Step 05 — Late chunking

## The story so far
[Step 04](../04_semantic_chunking/WALKTHROUGH.md) improved *where* we cut. It did nothing about a different problem: after cutting, every chunk is embedded **alone**. A chunk such as "Each block must be at least five days." is a fragment whose vector cannot know it is about secondary-caregiver leave. Back to the [module README](../README.md).

## The pain point
Cutting first and embedding second throws context away at the exact moment it is needed. Pronouns ("it", "this"), omitted subjects and "see above" make small chunks ambiguous, and small chunks are what give precise retrieval. We want chunk-sized vectors that still carry the surrounding text.

## The spark
Reverse the order: **embed first, chunk second**. Run the embedding model *once* over a whole window of text. Every token vector comes out of self-attention already conditioned on the full window. Then cut the window into chunks and mean-pool only the token vectors that belong to each chunk. Each chunk vector is chunk-sized but has "seen" its neighbourhood. This is **late chunking** (also called chunk-after-embed, or long-context embedding with late pooling), introduced by Jina AI in 2024 together with their 8192-token embedding models.

## Goal
Implement `late_vectors` (one forward pass, offsets, pooling), and measure what it changes:
1. Per-chunk cosine between the naive vector (chunk alone) and the late vector (A).
2. A pronoun-style chunk against its question: similarity and rank, naive vs late (B).
3. Retrieval over the whole corpus for window sizes 40 / 128 / 256 / 480 tokens (C), where a window of 40 tokens equals the chunk itself and therefore must reproduce the naive result.

## The theory (read this before the interview)
```
naive:  chunk_i  --embed-->  v_i = mean( H(chunk_i alone) )        attention sees only the chunk
late :  window W --embed ONCE--> H = [h_1 ... h_T]                  attention sees the whole window
        chunk_i = tokens a..b of W  -->  v_i = mean( h_a ... h_b )   pool only this chunk's tokens
```
- `H` has shape `(T, d)`: one contextual vector per token (`last_hidden_state`). Mean pooling is exactly what the embedding model does anyway; late chunking only changes **which tokens are pooled** and **what text the attention saw**.
- The result is then L2-normalised so that dot product equals cosine.
- **Window limit.** The window must fit the model's position limit (512 for MiniLM's BERT; 8192 tokens for Jina v2/v3 and some others). Longer documents are processed in overlapping windows.
- **Needs a model trained for it?** The mechanics work with any mean-pooled encoder (this step uses MiniLM). But the benefit depends on how the model was trained: models with long context and a training regime where token vectors stay meaningful on their own (the Jina long-context models advertise late chunking) gain more. A short-context model not trained on long inputs may gain nothing, or lose a little.
- **Cost.** The same tokens go through the model once (plus a longer attention window), so ingestion is about as cheap as naive embedding, unlike contextual retrieval (an LLM call per chunk, [step 06](../06_contextual_retrieval/WALKTHROUGH.md)). The index is the same size: one vector per chunk.
- **What it cannot do.** It adds *implicit* context only inside the window; it does not fix a chunk that is simply cut badly, and BM25 gets nothing (the chunk text is unchanged).

**Status.** Late chunking is *emerging* (2024 research from Jina AI; it is also exposed as an option in their embeddings API, and the idea is easy to reimplement, as here). Evidence is encouraging but model-dependent, so treat it as something to measure, as here.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| late chunking | chunk-after-embed, late pooling, long-context embedding pooling, contextual chunk embeddings | near | Embed the window first, pool per chunk afterwards |
| naive chunking | chunk-then-embed, independent chunk embeddings | yes | Each chunk is embedded alone |
| token embeddings / hidden states | `last_hidden_state`, contextual token vectors | yes | `(T, d)` matrix, one row per token |
| mean pooling | average pooling, mean of token vectors (SBERT default) | yes | Average over the (masked) tokens |
| `[CLS]` pooling | first-token pooling | no | Uses the first token only; some models are trained that way, MiniLM uses mean pooling |
| late interaction (ColBERT) | multi-vector retrieval, MaxSim | no | Keeps *all* token vectors and scores token by token; late chunking keeps one pooled vector per chunk |
| contextual retrieval | LLM-written context prefix | no | Adds context in *text* (works for BM25 too), late chunking adds it in *vector space* |

## Run it
```bash
cd 19_chunking_deep_dive/05_late_chunking
python main.py
# from the repo root:
HF_HUB_OFFLINE=1 .venv/bin/python 19_chunking_deep_dive/05_late_chunking/main.py
```
About 25 to 30 s of CPU time. Needs `torch` + `transformers` and the cached MiniLM; no downloads.

## Code, step by step

### 1. Header docstring and imports  (`main.py` lines 1–22)

```python
"""
GOAL:  Late chunking: run the embedding model ONCE over a whole window of text, then mean-pool the token vectors
       that belong to each chunk. Each chunk vector has "seen" the text around it. Compare with naive chunk
       embeddings (embed each chunk alone) on pronoun / reference questions and on the whole eval set,
       and sweep the window size.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 19_chunking_deep_dive/05_late_chunking/main.py
EXPECTED OUTPUT (shortened):
    A) one window: token vectors (T, 384) -> 3 chunk vectors; cosine(naive, late) per chunk
    B) a pronoun-chunk: similarity to the right question vs a distractor, naive vs late
    C) retrieval table: naive vs late for window = 40 / 128 / 256 / 480 tokens
NEEDS: torch, transformers (cached MiniLM).  ~25 s.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common19 import RawChunk, count_tokens, embed, evaluate, get_embedder, load_docs, load_eval, sentence_spans, table  # noqa: E402
```

- **In simple words:** goal and run lines, plus the shared helpers; `torch` is imported because we now touch the model's hidden states directly.
- **Exactly:** `get_embedder` returns module 14's `Embedder` (which holds `.tok`, the tokenizer, and `.model`, the BERT encoder); `embed` is the cached "naive" embedding path. Explained in the [step 01 tour](../01_tokens_not_characters/WALKTHROUGH.md#the-shared-helpers-once-a-tour-of-common19py).
- **Why it is here:** the naive path uses the library helper; the late path reaches under it.

### 2. `late_vectors`: one forward pass, pool per chunk  (`main.py` lines 25–37)

```python
def late_vectors(window_text, char_spans):
    """One forward pass over `window_text`; one unit vector per (start, end) character span inside it."""
    emb = get_embedder()
    enc = emb.tok(window_text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=512)
    offs = enc.pop("offset_mapping")[0].tolist()             # [(char_start, char_end)] per token; (0, 0) for [CLS]/[SEP]
    with torch.no_grad():
        H = emb.model(**enc).last_hidden_state[0]            # (T, 384): token vectors that have attended to the WHOLE window
    out = []
    for a, b in char_spans:
        idx = [i for i, (s, e) in enumerate(offs) if e > s and s >= a and e <= b]   # tokens inside this chunk, no specials
        v = H[idx].mean(dim=0)                               # mean-pool ONLY this chunk's tokens -> (384,)
        out.append(torch.nn.functional.normalize(v, dim=0).numpy())
    return np.vstack(out), H.shape[0]
```

- **In simple words:** read the whole window once; for every chunk, average the token vectors that lie inside its character span.
- **Exactly:** `emb.tok(window_text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=512)` returns `input_ids`, `attention_mask` and the per-token character offsets; `enc.pop("offset_mapping")` removes the offsets so they are not passed to the model. `emb.model(**enc).last_hidden_state[0]` has shape `(T, 384)` (batch of one): each row is a token vector that attended to every token in the window. `offs` holds `(char_start, char_end)` per token and `(0, 0)` for `[CLS]` and `[SEP]`; the test `e > s` skips them. `idx` are the token indices fully inside the span `(a, b)`. `H[idx].mean(dim=0)` pools them to shape `(384,)`, and `torch.nn.functional.normalize(v, dim=0)` makes the result unit length. The function returns an `(n_chunks, 384)` NumPy matrix and `T`.
- **Why it is here:** this is the whole technique in 12 lines. `torch.no_grad()` turns off gradient tracking (inference only).

### 3. `pack`: sentences to groups under a budget  (`main.py` lines 40–48)

```python
def pack(doc, spans, budget):
    """Greedy sentence packing: consecutive sentence spans -> groups of <= budget tokens."""
    groups, cur = [], []
    for sp in spans:
        if cur and count_tokens(doc["text"][cur[0][0]:sp[1]]) > budget:
            groups.append(cur)
            cur = []
        cur.append(sp)
    return groups + ([cur] if cur else [])
```

- **In simple words:** the greedy sentence packer used twice, once to build windows and once to build the small chunks inside a window.
- **Exactly:** consecutive `(start, end)` sentence spans are grouped until the token count of the joined text would exceed `budget`.
- **Why it is here:** same ten lines as the packers of steps 03, 04 and 07 (kept local so each file runs alone).

### 4. `build`: windows made of small chunks  (`main.py` lines 51–64)

```python
def build(doc, window_tokens, chunk_tokens=40):
    """Windows of <= window_tokens made of small chunks of <= chunk_tokens. Returns chunks + (naive, late) vectors."""
    sents = sentence_spans(doc["text"])
    chunks, naive_texts, late = [], [], []
    for win in pack(doc, sents, window_tokens):
        w0, w1 = win[0][0], win[-1][1]
        small = pack(doc, win, chunk_tokens)
        spans = [(g[0][0] - w0, g[-1][1] - w0) for g in small]          # chunk spans relative to the window text
        vecs, _ = late_vectors(doc["text"][w0:w1], spans)
        for g, v in zip(small, vecs):
            s, e = g[0][0], g[-1][1]
            chunks.append(RawChunk(doc["doc_id"], s, e, doc["text"][s:e]))
            late.append(v)
    return chunks, np.vstack(late)
```

- **In simple words:** cut the document into windows of at most `window_tokens`; cut each window into small chunks of at most `chunk_tokens`; embed each window once and return the chunks and their late vectors.
- **Exactly:** `pack(doc, sents, window_tokens)` makes windows; `pack(doc, win, chunk_tokens)` splits one window's sentences into chunks; `spans` converts chunk offsets to be relative to the window text; `late_vectors(window_text, spans)` returns one vector per chunk. Each chunk is a `RawChunk` over its own sentences, so offsets and evaluation are unchanged. With `window_tokens == chunk_tokens`, a window *is* a chunk and the late vector equals the naive vector.
- **Why it is here:** `window_tokens` is the experimental knob: how much surrounding text the model may read while the retrieval unit stays small. Step 10 reuses `build(d, 128)`.

### 5. Section A: one window, step by step  (`main.py` lines 67–83)

```python
def main():
    docs = load_docs()
    hb = next(d for d in docs if d["doc_id"] == "employee_handbook")

    # ---- A) one window, step by step ----------------------------------------------------------------------------------
    start = hb["text"].index("#### 2.3.2 Secondary caregiver")
    end = hb["text"].index("### 3. Expenses") if "### 3. Expenses" in hb["text"] else hb["text"].index("## 3. Expenses")
    window = hb["text"][start:end]
    sents = sentence_spans(window)
    small = pack({"text": window}, sents, 25)
    spans = [(g[0][0], g[-1][1]) for g in small]
    vecs, T = late_vectors(window, spans)
    naive = embed([window[a:b] for a, b in spans])
    print(f"A) window = parental-leave subsections: {count_tokens(window)} tokens -> model sees T={T} tokens (incl. [CLS], [SEP]); "
          f"token matrix (T, 384); {len(spans)} chunks")
    for (a, b), n, l in zip(spans, naive, vecs):
        print(f"   cos(naive, late) = {n @ l:.3f}   {window[a:b][:58]!r}")
```

- **In simple words:** take the parental-leave subsections of the handbook as one window, cut it into tiny (25-token) chunks, embed it late and naive, and print their cosine.
- **Exactly:** `T` is the number of tokens the model saw, including `[CLS]` and `[SEP]`. `naive = embed([...])` embeds each chunk alone. `n @ l` is the cosine between the naive and the late vector of the same chunk: 1.0 would mean context made no difference.
- **Why it is here:** the first picture: the late vector moves away from the naive one by exactly the amount the context mattered.

### 6. Section B: a chunk that is meaningless alone  (`main.py` lines 85–94)

```python
    # ---- B) a chunk that is meaningless alone --------------------------------------------------------------------------------
    q = "How long must each block of secondary caregiver leave be?"
    target = next(i for i, (a, b) in enumerate(spans) if "Each block" in window[a:b])
    qv = embed([q])[0]
    print(f"\nB) question {q!r}")
    print(f"   chunk under test: {window[spans[target][0]:spans[target][1]]!r}")
    for name, M in (("naive (chunk alone)", naive), ("late  (chunk in window)", vecs)):
        sims = M @ qv
        order = np.argsort(-sims)
        print(f"   {name:<24} sim to target {sims[target]:.3f}, rank {int(np.where(order == target)[0][0]) + 1} of {len(spans)} in this window")
```

- **In simple words:** a question about "each block of secondary caregiver leave", and a target chunk that only says "Each block ..." (without naming the topic). Compare the target's similarity and rank among the window's chunks under both embeddings.
- **Exactly:** `target` is the index of the chunk containing "Each block". `M @ qv` gives the cosine of every chunk with the question; `np.argsort(-sims)` ranks them; the printed rank is the target's position among the chunks of this one window.
- **Why it is here:** a deliberately chosen example that shows the *mechanism*. Do not generalise from one hand-picked chunk; section C does the honest measurement.

### 7. Section C: corpus-wide retrieval, window sweep  (`main.py` lines 96–113)

```python
    # ---- C) retrieval over the whole corpus, window sweep ----------------------------------------------------------------------
    qs = load_eval()
    rows = []
    for W in (40, 128, 256, 480):
        all_chunks, all_late = [], []
        for d in docs:
            c, v = build(d, W)
            all_chunks += c
            all_late.append(v)
        late_m = np.vstack(all_late)
        naive_m = embed([c.text for c in all_chunks])
        for name, M in (("naive", naive_m), ("late", late_m)):
            r = evaluate(all_chunks, qs, docs, vectors=M)
            ref = [x["hit@1"] for x in r["dense"]["rows"] if x["kind"] == "reference"]
            rows.append([W, name, len(all_chunks), f"{r['dense']['hit@1']:.2f}", f"{r['dense']['hit@3']:.2f}",
                         f"{r['dense']['mrr']:.2f}", f"{np.mean(ref):.2f} (n={len(ref)})"])
    print("\nC) chunks of <= 40 tokens; the window is how much text the model reads at once (40 = the chunk itself)")
    print(table(rows, ["window", "vectors", "chunks", "hit@1", "hit@3", "MRR", "reference-q hit@1"]))
```

- **In simple words:** for four window sizes build all chunks of all documents with naive and with late vectors, and score both.
- **Exactly:** for each window size `W`, `build` produces `all_chunks` and `late_m` (their late vectors). `naive_m = embed([c.text ...])` embeds the same chunks alone. `evaluate(all_chunks, qs, docs, vectors=M)` accepts the vectors directly (the `vectors=` argument of the harness), so both rows rank *identical chunks*, differing only in the vector. The last column is hit@1 on the `reference` kind: questions whose answer sentence depends on a pronoun or heading.
- **Why it is here:** a controlled comparison: same chunks, same questions, only the vector differs.

### 8. Entry point and the TRY IT / BREAK IT lines  (`main.py` lines 116–121)

```python
if __name__ == "__main__":
    main()

# TRY IT: change chunk_tokens=40 to 20 in build(); smaller chunks have less of their own meaning, so context should matter more.
# TRY IT: in late_vectors, include the [CLS] token in the pool (drop the `e > s` test); does the vector change much?
# BREAK IT: window of 600 tokens with truncation=True silently drops the tail -> H[idx] is empty -> mean of nothing = nan.
```

- **In simple words:** guard plus exercises.
- **Exactly:** `truncation=True, max_length=512` silently drops the tail of a window longer than 512 tokens; a span beyond it then matches no token (see Debugging practice).
- **Why it is here:** step 10 calls `build`, so `main()` must not run at import.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `enc.pop("offset_mapping")[0].tolist()` | Remove a key from the batch dict and take item 0 of the batch as a Python list |
| `with torch.no_grad():` | Do not record operations for backpropagation: less memory, faster |
| `H[idx].mean(dim=0)` | Fancy indexing with a list of rows, then average over rows |
| `torch.nn.functional.normalize(v, dim=0)` | Divide a vector by its L2 norm |
| `np.vstack(list_of_vectors)` | Stack `(d,)` arrays into an `(n, d)` matrix |
| `for (a, b), n, l in zip(spans, naive, vecs)` | Iterate three sequences in parallel, unpacking a tuple |
| `np.argsort(-sims)` | Indices from highest to lowest similarity |
| `evaluate(..., vectors=M)` | Score with externally computed vectors |

## Libraries & functions
| Function | What it does | Internally |
|---|---|---|
| `tok(..., return_offsets_mapping=True, return_tensors="pt")` | Tokenise to PyTorch tensors plus offsets | Fast tokenizer; offsets track original characters |
| `model(**enc).last_hidden_state` | Contextual token vectors, shape `(batch, T, 384)` | 6 transformer layers of self-attention over all `T` tokens |
| `Tensor.mean(dim=0)` | Average rows | Sum and divide |
| `torch.nn.functional.normalize` | Unit-length vector | `v / max(norm, eps)` |
| `evaluate` | Scoring harness | See the [step 01 tour](../01_tokens_not_characters/WALKTHROUGH.md#the-shared-helpers-once-a-tour-of-common19py) |

## Execution trace (real run)
Real output of `python main.py` (progress bar removed; timings vary by machine).
```

A) window = parental-leave subsections: 105 tokens -> model sees T=107 tokens (incl. [CLS], [SEP]); token matrix (T, 384); 6 chunks
   cos(naive, late) = 0.721   '#### 2.3.2 Secondary caregiver'
   cos(naive, late) = 0.913   'The secondary caregiver receives 6 weeks of fully paid lea'
   cos(naive, late) = 0.498   'Each block must be at least two weeks long.\n\n#### 2.3.3 Re'
   cos(naive, late) = 0.846   'After the leave, employees can return part-time for up to '
   cos(naive, late) = 0.593   'During this time they keep their full salary for the hours'
   cos(naive, late) = 0.660   'Please tell your manager at least eight weeks before you r'

B) question 'How long must each block of secondary caregiver leave be?'
   chunk under test: 'Each block must be at least two weeks long.\n\n#### 2.3.3 Returning to work'
   naive (chunk alone)      sim to target 0.406, rank 5 of 6 in this window
   late  (chunk in window)  sim to target 0.688, rank 3 of 6 in this window

C) chunks of <= 40 tokens; the window is how much text the model reads at once (40 = the chunk itself)
window  vectors  chunks  hit@1  hit@3   MRR  reference-q hit@1
------  -------  ------  -----  -----  ----  -----------------
40        naive     155   0.69   0.86  0.79         0.50 (n=4)
40         late     155   0.69   0.86  0.79         0.50 (n=4)
128       naive     165   0.68   0.84  0.78         0.50 (n=4)
128        late     165   0.58   0.85  0.73         0.00 (n=4)
256       naive     157   0.69   0.86  0.79         0.50 (n=4)
256        late     157   0.57   0.82  0.70         0.50 (n=4)
480       naive     156   0.68   0.86  0.79         0.25 (n=4)
480        late     156   0.54   0.78  0.67         0.50 (n=4)
```

How to read it:
- **A:** the 105-token window is seen as `T=107` tokens (with `[CLS]` and `[SEP]`) and cut into 6 small chunks. The cosine between naive and late vectors goes from 0.913 for a self-contained sentence ("The secondary caregiver receives 6 weeks ...") down to 0.498 for the fragment "Each block must be at least two weeks long." plus the next heading. Late chunking moves a vector exactly as much as the context mattered.
- **B:** for the question about "each block of secondary caregiver leave", the context-dependent fragment goes from similarity 0.406 and rank 5 of 6 (naive) to 0.688 and rank 3 of 6 (late). The mechanism works on a hand-picked example, and even there the target is not rank 1.
- **C, the control:** with `window = 40` the late and naive rows are identical (155 chunks, hit@1 0.69, hit@3 0.86, MRR 0.79), as they must be: when the window is the chunk there is no outside context to pool.
- **C, the verdict for MiniLM:** with wider windows late chunking is **worse**, not better: hit@1 0.58 (window 128), 0.57 (256), 0.54 (480) against 0.68 to 0.69 for naive embeddings of the same chunks, and hit@3 falls to 0.78 at window 480. The `reference` slice has only 4 questions (0.00 against 0.50 at window 128 is two questions, and it reverses at 480), so it says nothing. A plausible reason: MiniLM was trained on short inputs, so token vectors read in a long window drift towards the topic of the whole window and chunks of the same window become more alike, which hurts ranking between them. This is why the technique is tied to long-context models trained for it (Jina v2/v3 style), and why you test your own model before adopting it.

## Experiments (change ONE thing)
| Change | What you will see | Why |
|---|---|---|
| `build(d, 128, chunk_tokens=20)` | Smaller chunks, more of them; compare the late-vs-naive gap with the 40-token table | A smaller chunk carries less of its own meaning, so context should matter more |
| Pool the `[CLS]` token too (drop `e > s`) | The cosine between the two pooled vectors is 0.998 for a 35-token chunk (measured) | One extra token among dozens is averaged in; `[CLS]` is itself a summary of the whole window |
| Window 512 instead of 480 | Right at the limit: check whether any chunk's `idx` comes back empty | `max_length=512` includes `[CLS]` and `[SEP]`, so the usable text is 510 tokens |
| Swap in a long-context embedder (not cached here) | Larger windows become usable; the effect is model-dependent | Long-context models are trained so that token vectors remain meaningful when pooled |
| `late_vectors` with `chunk_tokens` equal to the window | late == naive (cosine 1.000) | Nothing outside the chunk to attend to |

## Debugging practice
**BREAK IT:** call `late_vectors` on a 700-word text with a span at the very end.

Real result (checked in a run):
```
T = 512   nan in first span: False   nan in last span: True   (no warning, no error)
```
The window is longer than 512 tokens, so `truncation=True, max_length=512` silently drops the tail. A span near the end matches **no** token, `idx` is empty, `H[[]].mean(dim=0)` is the mean of nothing, and the vector is `nan` (normalising keeps it `nan`). Nothing crashes. A `nan` vector then poisons every similarity it touches (comparisons with `nan` are false, so that chunk can never be retrieved). **Fix:** check `T` against the window length before pooling (or cap windows at 510 tokens, as `build` does through `pack`), `assert idx`, and `assert not np.isnan(vecs).any()` after pooling.

## Common mistakes
- Comparing late and naive chunks that were cut differently. Use the *same* chunk boundaries, as `evaluate(..., vectors=M)` does here.
- Assuming a bigger window is always better. Past the model's trained length, quality can fall.
- Letting truncation drop the tail silently, then pooling an empty set of tokens (a `nan` vector).
- Forgetting to skip `[CLS]`/`[SEP]` (offset `(0, 0)`) when selecting tokens.
- Expecting BM25 to benefit. The chunk *text* is unchanged; only the dense vector gained context.
- Expecting it to work equally with every encoder. Test your model: the mechanism is general, the benefit is not.
- Storing the late vectors but sending the *chunk* text to the LLM: fine, but the LLM still sees the fragment. If the LLM needs context, add it to the text too ([step 06](../06_contextual_retrieval/WALKTHROUGH.md), [step 07](../07_hierarchical_parent_child/WALKTHROUGH.md)).

## Rebuild checklist
1. Tokenise the window with offsets and `truncation=True, max_length=512`; pop the offsets.
2. One forward pass under `torch.no_grad()`: `H = model(**enc).last_hidden_state[0]`.
3. For each chunk span, `idx` = tokens with `e > s and s >= a and e <= b`.
4. `H[idx].mean(0)`, L2-normalise, stack.
5. `build`: windows from packed sentences, chunks from packed sentences inside each window, spans relative to the window.
6. Evaluate naive vs late on *identical* chunks via `vectors=`; sweep the window size.

## Say it aloud
"Late chunking flips chunk-then-embed: I run the embedding model once over a long window so every token vector has attended to the whole window, then cut the window into chunks and mean-pool each chunk's token vectors. The vector is chunk-sized but context-aware, ingestion cost is about the same as naive embedding, the index is the same size, and it fixes the pronoun and fragment problem in vector space. It needs the window to fit the model's context (512 here, 8k for the long-context embedders), only helps dense retrieval, and its gain depends on the model, so I measure it against naive embeddings of the same chunks."

## The new problem (cliffhanger)
Late chunking adds context in *vector space*, inside the model, for dense retrieval only. BM25 still searches the bare fragment, and the LLM that reads the retrieved chunk still sees the bare fragment. What if we wrote the missing context down, in words, and stored it with the chunk, so *both* search methods and the LLM benefit? That is contextual retrieval: [step 06](../06_contextual_retrieval/WALKTHROUGH.md).
