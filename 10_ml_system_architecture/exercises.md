# Module 10 — Exercises

Change **one** thing at a time; predict, run, compare. Worked answers: [solutions/](solutions/).

## Step 01 — Pipeline
- **Modify:** change `n=200` to `n=10`; record accuracy and confidence.
- **Break & read:** call `vectorizer.transform(...)` before `fit_transform` (`NotFittedError`).
- **Rebuild:** a `Pipeline` of `TfidfVectorizer` + `LogisticRegression`, train on `data.py`, predict one sentence — from memory.
- **Explain aloud:** each of the 7 stages, and one way each can go wrong.

## Step 02 — Training vs inference
- **Modify:** lower/raise the `threshold` in `predict_text`; count `uncertain` answers.
- **Break & read:** run `broken_skew.py`; explain why Part A raises no error but Part B does.
- **Modify:** make `load_artifact` warn when `sklearn_version` in metadata ≠ installed.
- **Rebuild:** `train()` and `predict_text()` sharing only a file.
- **Explain aloud:** training/serving skew with one real-life example.

## Step 03 — Serialization
- **Modify:** compare file sizes at `n=2000`; try `compress=3`.
- **Break & read:** run the printed `python -c` command and read the `AttributeError`; fix by moving `shout` into `helpers.py`.
- **Explain aloud:** why pickle is unsafe and what you'd use instead.

## Step 04 — Registry
- **Modify:** add a `staging` alias. Write `promote_if_better(registry, version)` that compares `log_loss` with production.
- **Break & read:** `reg.load("v0099")` and `reg.promote("v0003")`.
- **Rebuild:** `resolve()` supporting `latest`, aliases and pinned versions.
- **Explain aloud:** why `latest` is risky in production.

## Step 05 — Storage
- **Modify:** keep the DB in a file; add `avg_confidence_by_label()`.
- **Break & read:** misspell a column in a query (`OperationalError`).
- **Explain aloud:** parameterized queries; what you would not log (PII).

## Step 06 — Batch vs real-time
- **Modify:** chunk sizes 10 / 100 / 2000 / n=20000 — table the times.
- **Rebuild:** a generator `chunks(items, size)`.
- **Explain aloud:** when you would choose each mode and what micro-batching is.

## Step 07 — Monitoring
- **Modify:** write an alert: warn if p95 > 50 ms or error rate > 2%.
- **Modify:** feed short texts instead of long ones — what does PSI do?
- **Explain aloud:** data drift vs concept drift; what you do next.

## Step 08 — Scalability
- **Modify:** queue `maxsize` 1 / 5 / 50; explain the accepted/rejected change.
- **Calculate:** workers needed for 100 req/s at 150 ms; for 20 req/s at 2 s.
- **Run:** `load_test.py` with `clients=64`; explain p95.
- **Explain aloud:** the ladder of answers to "traffic doubled".

## Integration tasks (connect the modules)
1. Train in step 02, serve the artifact with the module-09 FastAPI app via `MODEL_PATH`, send 20 requests, and store each result with `PredictionLog` (write a tiny client script).
2. Draw (by hand) the architecture for **your** capstone from [diagram 1](diagrams/01_end_to_end_pipeline.md) and label the file/folder that implements each box.
3. Write your own answer to the design template ([05](diagrams/05_design_interview_answer_template.md)) for "an LLM summariser used by 1000 employees", and compare with [module 06](../06_llm_training_and_fast_inference/)'s latency/throughput notes.
