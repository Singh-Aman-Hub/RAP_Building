# 10: End-to-End RAG Debugging

**Level:** Intermediate+ · **Intentional bugs:** 5 · **Dependencies:** `qdrant-client`, `sentence-transformers`

## Scenario
Final-round take-home: *"This is a stripped-down version of our support assistant. It ingests three policy documents into Qdrant, retrieves relevant paragraphs, builds a grounded prompt and answers with citations. It was ported from an older codebase and it doesn't work. Get it working end to end, then tell us which bugs would have reached production unnoticed."*

`fake_llm` is a deterministic stand-in for the model and is **not** buggy. Don't edit it. Its behaviour is described in the module docstring.

## Expected behaviour
- Ingestion prints: `ingested 9 chunks; collection holds 9 points`. Every paragraph of every document is stored, and nothing is overwritten.
- Retrieval returns up to `TOP_K` (3) chunks, best first, and respects the optional source restriction.
- Expected answers:

| # | Question | Restriction | Expected answer contains | Retrieved from |
|---|---|---|---|---|
| 1 | How long does standard delivery take? | none | "Standard delivery takes 3 to 5 working days" with a citation | `shipping.md` first |
| 2 | How long does standard delivery take **and** how many days do I have to return an item? | none | **both** the 3–5 working days sentence **and** the 10 days return sentence, each cited | includes `shipping.md` **and** `returns.md` |
| 3 | Is water damage covered? | `warranty.md` only | "The warranty does not cover water damage..." | `warranty.md` only |
| 4 | Do you offer gift wrapping? | none | exactly the refusal message | `[]` |

## Observed behaviour
```
Traceback (most recent call last):
  ...
    hits = client.search(
           ^^^^^^^^^^^^^
AttributeError: 'QdrantClient' object has no attribute 'search'
```
That is the first of 5 problems: two loud ones and three silent ones. At one point along the way, **every** question, even question 1, gets the refusal message without any error. That is your cue to inspect the data, not the prompt.

## Your task
1. Make the pipeline meet every expected behaviour.
2. For each bug, record the **pipeline stage** it lives in (ingest / retrieve / generate) and whether it was loud or silent.
3. Answer the interviewer's question: which of the five would most likely have reached production unnoticed, and what monitoring or test would catch it?

## Constraints
- Keep `QdrantClient(":memory:")` and the installed qdrant-client version. Find the API that version provides. Don't downgrade the library to make old code run.
- Don't change `DOCUMENTS`, `fake_llm`, `MIN_SCORE` or the tests in `__main__`.

## Interview follow-ups
- What point-id scheme would you use so that re-ingesting an *updated* document replaces its old chunks instead of duplicating or overwriting someone else's?
- How would you notice in production that a metadata filter silently matches nothing?
- Why is `limit` / top-k a quality knob and not only a performance knob?
