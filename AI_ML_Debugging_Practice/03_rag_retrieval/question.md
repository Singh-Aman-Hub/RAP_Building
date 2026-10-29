# 03: RAG Retrieval

**Level:** Beginner+ · **Intentional bugs:** 4 · **Dependencies:** `sentence-transformers`, `numpy`

## Scenario
An HR chatbot answers employee questions from three policy documents. Before involving a vector database, the team prototyped retrieval in plain numpy. The interviewer says: *"Employees complain the bot quotes the wrong policy, or nothing at all. The LLM part is fine. I want you to debug only the retriever."*

## Expected behaviour
- Each document is split into its paragraphs (paragraphs are separated by a blank line). The startup line is `3 documents -> 9 chunks`, and **no chunk is empty**.
- `retrieve(question, k=3)` returns **up to** 3 chunks:
  - ordered from most to least similar,
  - each with the `source` of the document **that chunk came from**,
  - each with a cosine similarity `score` between -1 and 1 (higher = more similar),
  - and only chunks with `score >= MIN_SCORE` (0.30).
- The single most similar chunk must always be included. Nothing is skipped.
- Expected top-1 results:
  | Question | Top-1 chunk | Source |
  |---|---|---|
  | How many days of paid leave can I carry forward? | "Unused paid leave can be carried over..." | `leave_policy.md` |
  | What is the hotel limit per night in Mumbai? | "Hotel stays are reimbursed up to 6,000 rupees..." | `expense_policy.md` |
  | I lost my laptop at the airport, what should I do? | "Report a lost or stolen laptop..." | `it_security.md` |
  | What is the capital of France? | *(no relevant chunks)* | n/a |

## Observed behaviour
```
3 documents -> 15 chunks

Q: How many days of paid leave can I carry forward?
Traceback (most recent call last):
  ...
    hits.append({"source": DOCUMENTS[idx]["source"], "text": chunks[idx]["text"],
                           ~~~~~~~~~^^^^^
IndexError: list index out of range
```
Fixing just the crash produces output that *looks* like results: scores above 1.0, blank chunk texts, and sources that don't match the text. Don't stop there.

## Your task
Fix the 4 bugs so that every expected behaviour holds. Note which bugs were **loud** (exceptions) and which were **silent** (wrong output with no error).

## Constraints
- Keep `similarity()` as a separate function. Its docstring describes the contract you must meet.
- Don't change `MIN_SCORE` or the documents.

## Interview follow-ups
- For unit vectors, how are Euclidean distance and cosine similarity related? If you *wanted* to use distance, what else in `retrieve()` would have to change?
- Why is it dangerous to index two different lists (documents vs chunks) with the same integer?
- What does an empty or whitespace-only chunk do to a vector index?
