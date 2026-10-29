# Hints: 01 Qdrant Vector Search

Open one hint at a time.

## Bug A: "is not a valid UUID"
<details><summary>Hint A1</summary>

The error is about the **id** of a point, not its vector or payload. Read Qdrant's docs page *Concepts → Points* and find the section about point IDs.
</details>
<details><summary>Hint A2</summary>

Qdrant accepts exactly two kinds of id. A string is allowed only if it *is* one of those kinds. Is `"faq-0"`?
</details>
<details><summary>Hint A3</summary>

There are two reasonable fixes: one is the simplest, and one keeps the original key reproducible across re-runs (look at Python's `uuid` module). Either passes this exercise. Be ready to explain the trade-off.
</details>

## Bug B: "Vector dimension error"
<details><summary>Hint B1</summary>

Print `vectors.shape` right after `model.encode(...)`. Compare it with the collection config.
</details>
<details><summary>Hint B2</summary>

The constraints say not to hard-code a number the model can tell you. Look for a method on `SentenceTransformer` that returns the embedding size (`[m for m in dir(model) if "dimension" in m]`). Its name changed between sentence-transformers versions, so check what *your* version offers. The shape of an encoded vector is another source of truth.
</details>

## Bug C: `'tuple' object has no attribute 'score'`
<details><summary>Hint C1</summary>

Print `type(results)` and `repr(results)`. Is it a list?
</details>
<details><summary>Hint C2</summary>

Run `help(QdrantClient.query_points)` and read the **return type**. Then look at that class's fields. Iterating over a pydantic model gives you `(field_name, value)` pairs, which explains the tuple.
</details>

## Bug D: a pydantic `ValidationError` for `FieldCondition`
<details><summary>Hint D1</summary>

It only appears when a category is passed. Read the *first few lines* of the validation error: which argument is it unhappy about, and what types does it list as acceptable?
</details>
<details><summary>Hint D2</summary>

The `match` argument doesn't take a raw value. Qdrant's docs page *Concepts → Filtering → Match* shows the model class to wrap an exact keyword value in. You'll also need to import it.
</details>

## Verification ideas
- Assert that every hit returned for `category="billing"` has `payload["category"] == "billing"`.
- Change `MODEL_NAME` to `sentence-transformers/paraphrase-MiniLM-L3-v2` (also 384-d) or any other ST model. The collection should still be created with the right size, without editing any other line.
- Call `client.upsert(...)` with the same points a second time, then check `client.count(COLLECTION)`. Does it stay at 6? Now imagine the FAQ list is re-ordered before the second upsert: which id scheme survives that?
