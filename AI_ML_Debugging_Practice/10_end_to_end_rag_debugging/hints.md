# Hints: 10 End-to-End RAG Debugging

Open one hint at a time. Debug **stage by stage**: ingestion first, then retrieval, then generation.

## Bug A: `'QdrantClient' object has no attribute 'search'`
<details><summary>Hint A1</summary>

Which methods does *your* client offer? `[m for m in dir(QdrantClient) if "query" in m or "search" in m]`. Check the qdrant-client release notes for what happened to `search`.
</details>
<details><summary>Hint A2</summary>

The replacement has different **parameter names** and a different **return type**. Read its signature and return annotation with `help()` before changing the call.
</details>

## Bug B: a `KeyError` when building results
<details><summary>Hint B1</summary>

Print `hits[0].payload`. Which keys exist? Compare them with the keys the code reads, and with the keys written during ingestion.
</details>

## Bug C: everything is refused, even easy questions
<details><summary>Hint C1</summary>

Read the ingestion line again: how many chunks were ingested, and how many points does the collection hold? Why might they differ?
</details>
<details><summary>Hint C2</summary>

Scroll the collection and print each point's `id` and `doc`. Which documents survived?
</details>
<details><summary>Hint C3</summary>

In Qdrant, upserting a point with an existing id **replaces** it without complaint. Are the ids unique across the *whole* collection, or only within one document?
</details>

## Bug D: the two-part question only gets half an answer
<details><summary>Hint D1</summary>

Print `retrieved from:` for question 2. How many chunks does retrieval return? How many does the function's signature promise?
</details>
<details><summary>Hint D2</summary>

Look for a hard-coded value inside `retrieve()` that ignores its own parameter.
</details>

## Bug E: the restricted question is refused
<details><summary>Hint E1</summary>

Run the same query without the restriction. Does it find the right chunk? Then count points matching only the filter: `client.count(COLLECTION, count_filter=...)`.
</details>
<details><summary>Hint E2</summary>

The filter refers to a payload field by name. Is a field with that name ever written during ingestion? Look at what the UI calls a "source" and what the payload calls it.
</details>

## Verification ideas
- `assert client.count(COLLECTION).count == sum(len(...) for each document's paragraphs)`
- For a restricted query, `assert all(c["source"] == source for c in chunks)`.
- Think about running the upsert loop twice against the same collection (as a nightly re-index would). What *should* the point count be afterwards, and does your id scheme guarantee it?
