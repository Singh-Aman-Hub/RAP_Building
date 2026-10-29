# Hints: 03 RAG Retrieval

Open one hint at a time.

## Bug A: `IndexError: list index out of range`
<details><summary>Hint A1</summary>

Print `len(DOCUMENTS)`, `len(chunks)` and the value of `idx` that fails. Which list is being indexed, and how many items does it have?
</details>
<details><summary>Hint A2</summary>

`idx` is a position in the *chunk* matrix. Where else in the code is the source of a chunk available?
</details>

## Bug B: 15 chunks instead of 9, and blank texts
<details><summary>Hint B1</summary>

Print `repr(c["text"])` for every chunk. How many are empty strings?
</details>
<details><summary>Hint B2</summary>

How are paragraphs separated in the documents? What does `"a\n\nb".split("\n")` return? Try it in a Python shell.
</details>
<details><summary>Hint B3</summary>

Besides fixing the separator, what one-line guard would protect the index if a future document had stray blank lines?
</details>

## Bug C: scores above 1.0, unrelated chunks first
<details><summary>Hint C1</summary>

Read the docstring of `similarity()`. Then read its body. Does the body compute what the docstring promises? What range can that quantity take for unit vectors?
</details>
<details><summary>Hint C2</summary>

With the current body, does a *larger* number mean closer or farther? Now look at how `retrieve()` sorts and thresholds the scores.
</details>
<details><summary>Hint C3</summary>

Both vectors are already normalised (`normalize_embeddings=True`). Which simple matrix operation gives cosine similarity in that case?
</details>

## Bug D: the best chunk never shows up
<details><summary>Hint D1</summary>

Temporarily print the top 5 `(idx, score)` pairs before the loop. Compare them with what gets returned.
</details>
<details><summary>Hint D2</summary>

Read the comment next to the loop. Is the query actually stored in `chunk_vectors`? Where might that line have been copied from?
</details>

## Verification ideas
- `assert all(c["text"] for c in chunks)` and `assert len(chunks) == 9`.
- `assert -1.0 <= h["score"] <= 1.0` for every hit, and the scores are non-increasing.
- For each hit, `assert h["text"] in next(d["text"] for d in DOCUMENTS if d["source"] == h["source"])`.
