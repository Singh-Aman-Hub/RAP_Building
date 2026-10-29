# Hints: 05 Chunking and Metadata

Open one hint at a time. Try for at least 5 minutes between hints.

## Bug A: the crash
<details><summary>Hint A1</summary>

Python reports a syntax error at the line where it *noticed* the problem. The actual mistake is often on the line **before** it.
</details>
<details><summary>Hint A2</summary>

Look at how the items of the `BASE_METADATA` dictionary literal are separated.
</details>

## Bug B: chunk spacing
<details><summary>Hint B1</summary>

Print the `start` index of every window for one document. How far apart are they? How far apart *should* they be if consecutive windows share exactly `OVERLAP` words?
</details>
<details><summary>Hint B2</summary>

Draw it on paper: window 1 covers words 0–39. Which word should window 2 start at so that exactly 10 words are shared?
</details>

## Bug C: missing words at the end
<details><summary>Hint C1</summary>

`check_coverage` says some words are never covered. Which words: at the start, middle or end of a document?
</details>
<details><summary>Hint C2</summary>

Work out by hand what `range(...)` in `chunk_words` produces for a 52-word document. Does the loop ever start a window that reaches the last word? What happens if a document has fewer than 40 words?
</details>
<details><summary>Hint C3</summary>

The docstring promises that "the final window may be shorter than `chunk_size`". Slicing past the end of a Python list is safe. Ask yourself: what should the loop's stop value be?
</details>

## Bug D: every chunk has the same metadata
<details><summary>Hint D1</summary>

After `build_chunks` runs, print `BASE_METADATA`. Has it changed? Should it?
</details>
<details><summary>Hint D2</summary>

Print `id(chunk["metadata"])` for two different chunks. What does it tell you about how many dictionaries exist?
</details>
<details><summary>Hint D3</summary>

In Python, `a = b` for a dict does not copy anything. Look up the difference between assignment, a shallow copy and a deep copy, and decide which one this case needs.
</details>

## Verification ideas
- Assert that the chunk count matches the expected behaviour.
- Assert `len({id(c["metadata"]) for c in chunks}) == len(chunks)`.
- Assert that for consecutive chunks of one document, `next_start == prev_end - OVERLAP`.
