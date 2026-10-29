# 02: Sentence Transformers Semantic Search

**Level:** Beginner · **Intentional bugs:** 4 · **Dependencies:** `sentence-transformers`, `numpy`

## Scenario
A teammate built the backend for a help-centre search box. The frontend team says the endpoint "crashes", and once they patched the crash locally, "the results look random". You're asked to debug `search()` before it ships. The interviewer wants you to explain **what shape and type every variable has** as you go.

## Expected behaviour
- The startup line says `9 articles exported, 7 unique articles indexed`.
- `search(query, k=3)` returns a list of **3 dicts**, ordered from most to least similar, and `json.dumps` can serialise it.
- Each dict's `title` belongs to the same article as its `snippet`. A title must never be paired with another article's text.
- `score` is a plain Python `float` (cosine similarity, higher = more similar).
- Top results:
  | Query | Top-1 title |
  |---|---|
  | why is my parcel delayed? | Delivery delays |
  | how do I get my money back? | Refund policy |
  | I can't log in | Reset your password |
- Duplicate articles appear **at most once** in the results.

## Observed behaviour
```
9 articles exported, 7 unique articles indexed

Query: why is my parcel delayed?
Traceback (most recent call last):
  ...
    return [{"title": titles[i], "snippet": bodies[i][:50], "score": scores[i]} for i in top]
                      ~~~~~~^^^
TypeError: only integer tensors of a single element can be converted to an index
```
After a quick local patch, the frontend team saw a different serialisation error. When they patched that too, titles and snippets no longer matched, and the "top" results were the least relevant articles.

## Your task
Fix all 4 bugs in `search()` and the indexing code above it. For each one, write down the **shape/type** that was wrong and how you discovered it (print, `type()`, `.shape`, docs).

## Constraints
- Keep `util.cos_sim` (or replace it with an equivalent numpy computation, but be able to explain why both are valid here).
- Keep removing duplicate articles, but make it correct.

## Interview follow-ups
- What does `model.encode` return for a single string vs a list of strings? With `convert_to_tensor=True`?
- Why does `normalize_embeddings=True` make dot product and cosine similarity the same thing?
- In Python, does `set()` preserve insertion order? Does `dict`? Why does that matter here?
