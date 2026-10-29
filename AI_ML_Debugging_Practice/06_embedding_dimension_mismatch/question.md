# 06: Embedding Dimension Mismatch

**Level:** Intermediate · **Intentional bugs:** 4 · **Dependencies:** `qdrant-client`, `sentence-transformers`, `numpy`
(First run downloads two models: ~90 MB and ~470 MB. A failed download is an environment problem, not one of the bugs.)

## Scenario
Two programs share one config: an **indexing job** that embeds help-centre articles and writes vectors to disk, and a **search service** that loads those vectors into Qdrant and embeds user queries. Different people changed the config over a few months: one "upgraded" the query model, one sped up indexing, one copied a dimension from a blog post. Now the service crashes on startup, and after a hotfix it returns weak, unreliable matches.

The interviewer: *"Find everything that makes the index and the queries disagree. I care more about the silent problems than the crash."*

## Expected behaviour
- The indexer prints `wrote 4 vectors of dim 384`, and the service prints `loaded 4 articles`.
- **Self-retrieval check:** searching with an article's *exact indexed text* returns that article with score **`1.0`**. Identical text through the identical embedding pipeline must give an identical vector.
- Top-1 results:
  | Query | Top-1 title |
  |---|---|
  | I lost my phone and can't log in | Two-factor authentication |
  | how long is the data download link valid? | Exporting your data |
  | am I charged for inactive team members? | Team plans and billing |
  | can I undo deleting my account? | Deleting your account |
- After your fix, each of those top-1 scores is noticeably higher than in the hotfixed version.
- The config can't silently drift again: the vector size is not a hand-typed number, and index and query embeddings come from one setting.

## Observed behaviour
```
[indexer] wrote 4 vectors of dim 384
Traceback (most recent call last):
  ...
ValueError: Wrong input: Vector dimension error: expected dim: 768, got 384 for vector ''
```
After the "obvious" hotfix there is a second crash. After that one, the service runs and the top-1 titles even look right on these four easy queries, but the self-retrieval check prints a score around `0.35` instead of `1.0`.

## Your task
Fix all 4 bugs. Two crash, two are silent. For each silent bug, explain **how you proved it** (the self-retrieval check is one tool, and you can build others).

## Constraints
- Keep the two-program structure. The indexer must not pass Python objects to the service; they only share files on disk and `CONFIG`.
- Keep the query model **or** the index model, whichever you can justify. Just be consistent.

## Interview follow-ups
- Two models both produce 384-dimensional vectors. Why can't you compare a vector from one with a vector from the other?
- What is `max_seq_length`, and what happens to text beyond it? How would you detect that it is happening in a real corpus?
- You must switch embedding models in production with zero downtime. Describe the steps.
