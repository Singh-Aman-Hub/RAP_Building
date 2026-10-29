# 05: Chunking and Metadata

**Level:** Beginner · **Intentional bugs:** 4 · **Dependencies:** standard library only

## Scenario
You have joined a team building a customer-support RAG bot. Before anything is embedded, documents are cut into overlapping word windows, and each chunk gets metadata so the bot can say *"according to the Refund Policy..."*.

A teammate wrote the chunker and says: *"It's basically done, it just needs a quick check."* The interviewer hands you `buggy_code.py` and asks you to get it production-ready.

## Expected behaviour
With `CHUNK_SIZE = 40` and `OVERLAP = 10`:
- The script runs without errors.
- Consecutive chunks of the same document **share exactly 10 words**, so a new chunk starts every 30 words.
- `refund_policy.txt` (118 words) produces **4 chunks**, and `shipping_policy.txt` (52 words) produces **2 chunks**: **6 chunks in total**.
- **Every word is covered**: `check_coverage` prints `missing=0` for both documents. The last chunk of a document may be shorter than 40 words.
- Each chunk's metadata describes **that chunk**: its own `source`, `title`, `chunk_index` (0, 1, 2 ... within each document) and `word_range`, plus the shared `company` and `language` fields.
- `BASE_METADATA` itself stays unchanged after chunking.

## Observed behaviour
The first run stops immediately:
```
  File ".../buggy_code.py", line 43
    "language": "en",
              ^
SyntaxError: invalid syntax
```
The teammate says that once that line is patched, it "runs fine". Look closely at the printed chunk count, the metadata of each chunk, and the coverage report before you agree.

## Your task
1. Get the script running.
2. Make every expected behaviour above true.
3. For each bug, record symptom → root cause → fix → verification in the bug log from the README.

## Constraints
- Keep the function signatures of `chunk_words`, `build_chunks` and `check_coverage`.
- Don't hard-code the expected chunk counts. The chunker must work for any `CHUNK_SIZE > OVERLAP >= 0`.

## Interview follow-ups (answer after fixing)
- Why do RAG systems use overlap at all? What does it cost?
- What would go wrong downstream (in a vector DB or in citations) if two chunks shared the same metadata object?
