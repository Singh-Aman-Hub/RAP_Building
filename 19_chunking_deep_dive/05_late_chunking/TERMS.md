# Terms — Late Chunking

Folder: `19_chunking_deep_dive/05_late_chunking` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [19 README](../README.md)

**Terms in this folder:** [Late chunking](#late-chunking) · [last_hidden_state](#last_hidden_state) · [Token span of a chunk](#token-span-of-a-chunk) · [Naive chunk embedding](#naive-chunk-embedding) · [Long-context embedding model](#long-context-embedding-model) · [Window size](#window-size) · [torch.no_grad](#torchno_grad) · [NaN from an empty mean](#nan-from-an-empty-mean)

---

## Late chunking  `(concept)`
**1. Definition:** One forward pass over a long text, then mean-pool the token vectors of each chunk span (Günther et al., Jina AI, arXiv 2409.04701).
**2. In plain words:** Read everything first, then summarise each chunk.
**3. Example to quote:** > "Every chunk vector has attended to its neighbours."
```python
v = H[idx].mean(dim=0)
```
**Seen in:** main.py, late_vectors

## last_hidden_state  `(library)`
**1. Definition:** The model's output for every token: shape (batch, tokens, hidden) = (1, T, 384) for MiniLM.
**2. In plain words:** A vector for every token.
**3. Example to quote:** > "last_hidden_state: A vector for every token."
**Seen in:** main.py, late_vectors

## Token span of a chunk  `(concept)`
**1. Definition:** The token indices whose character offsets fall inside the chunk's character span; special tokens have offset (0, 0) and are excluded.
**2. In plain words:** Which token vectors belong to this chunk.
**3. Example to quote:** > "Token span of a chunk: Which token vectors belong to this chunk."
**Seen in:** main.py, late_vectors

## Naive chunk embedding  `(concept)`
**1. Definition:** Embedding each chunk alone: it cannot see its neighbours.
**2. In plain words:** The baseline.
**3. Example to quote:** > "Naive chunk embedding: The baseline."
**Seen in:** main.py, section B

## Long-context embedding model  `(concept)`
**1. Definition:** An embedder with a long input window (jina-embeddings-v3: 8,192 tokens) trained so that token vectors are useful when pooled over spans.
**2. In plain words:** The model late chunking is made for.
**3. Example to quote:** > "Plain MiniLM is not trained for it: it fixed one chunk and lowered hit@1."
**Seen in:** WALKTHROUGH, theory

## Window size  `(concept)`
**1. Definition:** How much text the model reads in one pass; here 40 (the chunk itself) to 480 tokens (BERT has 512 positions).
**2. In plain words:** The context each chunk sees.
**3. Example to quote:** > "Window size: The context each chunk sees."
**Seen in:** main.py, section C

## torch.no_grad  `(library)`
**1. Definition:** Disables gradient tracking inside the with-block: less memory, faster inference.
**2. In plain words:** We are not training.
**3. Example to quote:** > "torch.no_grad: We are not training."
```python
with torch.no_grad(): H = model(**enc).last_hidden_state[0]
```
**Seen in:** main.py, late_vectors

## NaN from an empty mean  `(error)`
**1. Definition:** Mean of zero token vectors is 0/0 = nan; happens when a span lies beyond the truncated window.
**2. In plain words:** The chunk has no tokens in the pass.
**3. Example to quote:** > "NaN from an empty mean: The chunk has no tokens in the pass."
**Seen in:** main.py, BREAK IT
