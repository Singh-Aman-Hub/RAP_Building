# 07: Similarity Search and Filtering

**Level:** Intermediate · **Intentional bugs:** 4 · **Dependencies:** `qdrant-client`, `sentence-transformers`

## Scenario
An internal policy search: employees type a question and pick their department from a dropdown. Results must come **only from that department** and only from **current** documents (published in or after a given year), because outdated policies have caused real HR incidents.

The wiki team delivers documents as a CSV export. After launch, every search returns nothing. The interviewer: *"There's no error, just empty lists. Debug it. And when results do come back, prove they respect both filters."*

## Expected behaviour
`python buggy_code.py` prints `ingested 8 documents` and then three result lists of `(year, department, title)` tuples:

1. `"How many weeks of parental leave do I get?"`, department `hr`, `min_year=2024`
   → HR documents from 2024 onwards only, with the **2024 Parental leave policy first**. The 2021 version must **not** appear.
2. `"When am I allowed to deploy?"`, department `engineering`, `min_year=2024`
   → Engineering documents from 2024 onwards only, with the **2025 Deployment checklist first**. The 2022 checklist and any HR or Finance document must **not** appear.
3. `"How do I claim my expenses?"`, department `finance`, `min_year=2026`
   → `[]`. No Finance document is that recent, so an empty list is the *correct* answer here.

Rules:
- A document is returned only if it matches the department **and** the year condition.
- The dropdown value may arrive in any letter case (`hr`, `HR`, `Hr`). All of them must work.
- `year` comparisons must be numeric.

## Observed behaviour
```
ingested 8 documents
[]
[]
[]
```
The third line is correct by accident. When you start fixing the filters, you will hit at least one exception before everything works.

## Your task
Fix the 4 bugs in `ingest()` and `search()`. For each silent bug, show the **probe** you used to locate it: a `scroll` of stored payloads, a `count` with only one condition, a search without the filter, and so on.

## Constraints
- The CSV string must stay as it is (it simulates an export you don't control).
- Keep Qdrant filtering *inside* the query. Don't post-filter results in Python.

## Interview follow-ups
- What's the difference between `must`, `should` and `must_not` in a Qdrant filter? How would you express "department is HR **or** Finance, **and** year ≥ 2024"?
- Why is fixing data types at **ingestion** better than at query time?
- On a Qdrant server (not in-memory), what should you create for `department` and `year` before loading millions of points, and why?
