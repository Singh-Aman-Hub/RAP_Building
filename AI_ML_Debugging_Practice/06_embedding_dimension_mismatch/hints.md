# Hints: 06 Embedding Dimension Mismatch

Open one hint at a time.

## Bug A: "expected dim: 768, got 384"
<details><summary>Hint A1</summary>

Which number is the truth: the config or the model output? Print `vectors.shape` in the indexer.
</details>
<details><summary>Hint A2</summary>

The expected behaviour says the size must not be hand-typed. Where can the service get it from instead: the model, or the saved vectors?
</details>

## Bug B: "Multivector is not found in the collection"
<details><summary>Hint B1</summary>

Print the *shape* of what `self.model.encode([query], ...)` returns, then the type and nesting of what `.tolist()` produces.
</details>
<details><summary>Hint B2</summary>

Qdrant reads a list of lists as a *multivector* (several vectors for one point), which this collection wasn't configured for. The collection expects one flat list of floats.
</details>

## Bug C: weak scores; self-retrieval is ~0.35 instead of 1.0
<details><summary>Hint C1</summary>

Self-retrieval should be 1.0 only if *exactly the same function* turned the same text into a vector at index time and at query time. List every setting that influences that function, in both programs.
</details>
<details><summary>Hint C2</summary>

Compare the model names used by the indexer and by the service. Equal dimensions do not mean equal vector spaces.
</details>

## Bug D: self-retrieval improves but still isn't 1.0
<details><summary>Hint D1</summary>

The remaining difference only shows up for **long** articles. Try the self-retrieval check with the shortest article, then with the longest.
</details>
<details><summary>Hint D2</summary>

Count the tokens of an article: `len(model.tokenizer(text)["input_ids"])`. Compare that with one of the CONFIG values. What happens to the tokens beyond the limit?
</details>
<details><summary>Hint D3</summary>

Look up the model card for all-MiniLM-L6-v2 on Hugging Face. What input length was it designed for? Note that the indexer and the service also don't apply the same limit.
</details>

## Verification ideas
- Assert that the collection's configured size equals `vectors.shape[1]` before upserting.
- Self-retrieval check for **every** article, not just one: `assert score > 0.999`.
- Store the model name next to the vectors (e.g. in `meta.json`) and make the service refuse to start if it doesn't match.
