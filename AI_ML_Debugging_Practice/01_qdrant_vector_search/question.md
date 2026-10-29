# 01: Qdrant Vector Search

**Level:** Intermediate · **Intentional bugs:** 4 · **Dependencies:** `qdrant-client`, `sentence-transformers`

## Scenario
Live-coding round: *"Here's a prototype of our help-centre search. It embeds FAQ entries, stores them in Qdrant and lets users search, optionally within one category. It was written against an older tutorial and it doesn't run. Make it work and walk me through each change."*

The interviewer explicitly allows you to read the Qdrant documentation and use `help()` on the client.

## Expected behaviour
- `python buggy_code.py` prints `Indexed 6 FAQ entries` and then results for three questions.
- **Q1** (no category filter, top 3): the top hit is `[account] How do I reset my password?` with a clearly higher score than the other two.
- **Q2** (`category="billing"`, top 3): **only** billing entries appear, and `Why was I charged twice?` is first. There are only 2 billing entries, so 2 lines are printed.
- **Q3** (`category="shipping"`, `top_k=1`): exactly one line, `[shipping] Do you deliver outside India?`.
- Scores are cosine similarities (higher = more similar), printed with 3 decimals.
- The collection's vector size must match the embedding model, and must stay correct if `MODEL_NAME` is changed to another Sentence Transformers model.

## Observed behaviour
```
Traceback (most recent call last):
  ...
  File ".../qdrant_client/local/utils.py", line 58, in normalize_point_id
    raise ValueError(f"Point id {point_id} is not a valid UUID") from exc
ValueError: Point id faq-0 is not a valid UUID
```
Fixing that line alone is not enough. Each fix reveals the next error.

## Your task
1. Find and fix all 4 bugs. Each one is a different kind: a configuration value, a data-format rule, a request-object type, and a response-object type.
2. For each bug, say which **official doc page or `help()` output** confirmed your fix.
3. Record everything in the README bug log.

## Constraints
- Keep using an in-memory client (`QdrantClient(":memory:")`).
- Keep `query_points`. This exercise is about using the **current** query API correctly, not about switching APIs.
- Don't hard-code any number that can be read from the model.

## Interview follow-ups
- Which kinds of values can a Qdrant point id be? If your source data has string keys like `faq-0`, how would you keep a **stable** mapping so re-running the indexer doesn't create duplicates?
- What does `query_points` return, and how does that differ from what older tutorials' `search()` returned?
- Is the category filter applied before or after the nearest-neighbour search in Qdrant? Why does it matter for `top_k`?
