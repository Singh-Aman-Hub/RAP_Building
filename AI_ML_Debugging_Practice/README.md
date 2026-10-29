# AI/ML Debugging Practice

Ten interview-style debugging scenarios for vector search, embeddings, PDF extraction, chunking, FastAPI inference and RAG.
Each one is a small program that **looks almost right and is not**. Your job: find every bug, explain its root cause, fix it, and prove the fix works, **without AI help and without looking up a finished solution**. There are no solutions in this folder on purpose.

This folder is separate from the study modules (`00_...` to `14_...`). It does not import anything from them, so every exercise runs on its own.

---

## 1. Setup

Python 3.10+ (the repo's `.venv` uses 3.12). From the repo root:

```bash
source .venv/bin/activate                 # or create one: uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install qdrant-client sentence-transformers pdfplumber fastapi httpx numpy
#   or: pip install qdrant-client sentence-transformers pdfplumber fastapi httpx numpy
```

| Package | Used in | Notes |
|---|---|---|
| `numpy` | most | |
| `sentence-transformers` | 02, 03, 06, 07, 10 | downloads `all-MiniLM-L6-v2` (~90 MB) the first time; 06 also downloads `paraphrase-multilingual-MiniLM-L12-v2` (~470 MB) |
| `qdrant-client` | 01, 06, 07, 10 | runs **in memory** (`QdrantClient(":memory:")`), no server or Docker needed |
| `pdfplumber` | 04 | `sample_policy.pdf` is included in the exercise folder |
| `fastapi`, `httpx` | 08 | the script uses FastAPI's `TestClient`, so no server needs to be started |

Everything runs on a laptop CPU. No API keys, no paid services, no network calls except the one-time model downloads.

**Check your versions first. Several bugs depend on API versions:**
```bash
python -c "import qdrant_client, sentence_transformers, pdfplumber, fastapi; print(qdrant_client.__version__ if hasattr(qdrant_client,'__version__') else '', sentence_transformers.__version__, pdfplumber.__version__, fastapi.__version__)"
pip show qdrant-client            # always works, even if the module has no __version__
```
These exercises were checked against **qdrant-client 1.20, sentence-transformers 6.2, pdfplumber 0.11, FastAPI 0.146, Python 3.12**. If your versions differ, an error message may differ too. Reading the error and the docs for *your* version is part of the exercise.

**Version notes (environment, not intentional bugs):**
- qdrant-client has deprecated and removed several methods across 1.x releases. Methods that appear in older tutorials may not exist in your version, or may return different types. When an exercise touches the client, check the method list and the changelog for *your* version (see section 4).
- sentence-transformers 6 renamed `get_sentence_embedding_dimension()` to `get_embedding_dimension()`. The old name still works but prints a `FutureWarning`. That warning is not one of the bugs.
- Warnings such as `Could not reach the Hub ... Using cached commit hash` or `Redirects are currently not supported in MacOs` are environment noise.

---

## 2. How to start an exercise

```bash
cd AI_ML_Debugging_Practice/01_qdrant_vector_search
cat question.md                   # 1. read the scenario and the expected behaviour
python buggy_code.py              # 2. run it and observe what actually happens
cp buggy_code.py my_fix.py        # 3. work on a copy, keep the original for comparison
# 4. open hints.md ONLY when stuck, one hint at a time
```

Suggested time box: **25–40 minutes** per exercise, as in a live interview. Talk out loud while you debug.

---

## 3. Rules for independent debugging

1. **No AI assistants, no copied solutions.** Official docs, `help()`, source code, error messages and experiments are allowed.
2. **Reproduce before you fix.** Write down the exact error or wrong output first.
3. **One bug, one change, one re-run.** Don't fix three things at once, or you won't know which change mattered.
4. **Not every bug crashes.** After the program runs cleanly, check that the *output is correct* against the expected behaviour in `question.md`. Silent bugs (wrong results, empty results, wrong order) are the ones interviewers care about most.
5. **Separate your bugs from environment problems.** `ModuleNotFoundError`, a failed model download, or a proxy error is *not* one of the intentional bugs. Fix your environment and move on. Every intentional bug is inside `buggy_code.py` (or, for 04, in how it reads the PDF).
6. Each exercise has **2–5 intentional bugs**. `question.md` tells you how many. Stop when the program meets every expected behaviour, not when it stops crashing.
7. Use hints progressively: Hint 1 points at a *place*, the later hints narrow it down. None of them give the code.

---

## 4. Researching like a professional

**Read the signature before guessing:**
```python
from qdrant_client import QdrantClient
help(QdrantClient.query_points)        # parameters, return type, docstring
import inspect
print(inspect.signature(QdrantClient.create_collection))
print(hasattr(QdrantClient, "search")) # does this method even exist in MY version?
print([m for m in dir(QdrantClient) if not m.startswith("_")])
```

**Inspect objects at runtime:**
```python
type(x), getattr(x, "shape", None), len(x)
print(repr(result))                     # pydantic models print all their fields
breakpoint()                            # drop into pdb at that line: p var, n, s, c, q
```

**Find the installed source code** (the docstrings are often better than blog posts):
```bash
python -c "import qdrant_client, os; print(os.path.dirname(qdrant_client.__file__))"
```

**Official documentation:**
| Topic | Where |
|---|---|
| Qdrant concepts (collections, points, payload, filtering, search) | https://qdrant.tech/documentation/concepts/ |
| Qdrant Python client and its changelog (look for deprecations) | https://python-client.qdrant.tech/ and https://github.com/qdrant/qdrant-client/releases |
| Sentence Transformers (`encode`, normalisation, model cards) | https://sbert.net/ and the model page on huggingface.co |
| pdfplumber (`Page.extract_text`, cropping, layout) | https://github.com/jsvine/pdfplumber |
| FastAPI (request bodies, status codes, async vs sync) | https://fastapi.tiangolo.com/tutorial/ |
| Pydantic v2 models | https://docs.pydantic.dev/latest/ |

Tip: blog posts and old tutorials often use APIs that have since been renamed or removed. If a method name from a tutorial fails, check the changelog of *your* installed version.

---

## 5. Bug log (copy this for every exercise)

```markdown
## Exercise NN: <name>          date: ____   time taken: ____ min

| # | Symptom (error / wrong output) | Root cause | Fix (in words) | How I verified it |
|---|---|---|---|---|
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |

- [ ] The program runs without errors
- [ ] Every "expected behaviour" bullet in question.md is met
- [ ] I can explain each bug in one sentence, out loud
- [ ] I found the bugs before reading hints  (hints used: __ / __)
- [ ] One thing I would add for production:
```

---

## 6. Progression: beginner to intermediate

| # | Folder | Topic | Level | Bugs | Main skills |
|---|---|---|---|---|---|
| 1 | [05_chunking_and_metadata](05_chunking_and_metadata/) | sliding-window chunking, metadata | Beginner | 4 | syntax errors, loop bounds, shared mutable objects |
| 2 | [02_sentence_transformers](02_sentence_transformers/) | semantic search with embeddings | Beginner | 4 | `encode`, sorting, list alignment |
| 3 | [04_pdf_text_extraction](04_pdf_text_extraction/) | pdfplumber page extraction | Beginner | 4 | pages, empty pages, headers/footers, line joining |
| 4 | [03_rag_retrieval](03_rag_retrieval/) | in-memory retriever with top-k | Beginner+ | 4 | similarity vs distance, slicing, caching |
| 5 | [01_qdrant_vector_search](01_qdrant_vector_search/) | Qdrant collection, upsert, search | Intermediate | 4 | point ids, vector size, filter models, response objects |
| 6 | [06_embedding_dimension_mismatch](06_embedding_dimension_mismatch/) | index/query embedding consistency | Intermediate | 4 | shapes, dims, model consistency |
| 7 | [07_similarity_search_and_filtering](07_similarity_search_and_filtering/) | Qdrant payload filters | Intermediate | 4 | `must`/`should`, types, case, payload flags |
| 8 | [08_fastapi_inference](08_fastapi_inference/) | serving an embedding model | Intermediate | 4 | Pydantic bodies, serialisation, model lifecycle, status codes |
| 9 | [09_llm_context_and_grounding](09_llm_context_and_grounding/) | prompt building, citations, refusals | Intermediate | 4 | ordering, budgets, thresholds, citation ids |
| 10 | [10_end_to_end_rag_debugging](10_end_to_end_rag_debugging/) | full ingest → retrieve → answer pipeline | Intermediate+ | 5 | everything above, in one system |

The folder numbers group exercises by topic. The table above is the recommended **order of difficulty**. Each exercise is independent, so any order works.

---

## 7. After you finish an exercise
- Re-read `question.md` and tick off every expected behaviour.
- Write one **regression check**: an `assert` that would have caught the worst bug.
- Explain the bugs in 60 seconds as if to an interviewer: symptom → cause → fix → how you'd prevent it.
