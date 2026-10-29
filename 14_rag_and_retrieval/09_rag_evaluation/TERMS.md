# Terms — Evaluating RAG

Folder: `14_rag_and_retrieval/09_rag_evaluation` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [14 README](../README.md)

**Terms in this folder:** [Golden set](#golden-set) · [Unanswerable questions](#unanswerable-questions) · [Hard negatives](#hard-negatives) · [hit@k](#hitk) · [recall@k](#recallk) · [precision@k](#precisionk) · [MRR](#mrr) · [nDCG@k](#ndcgk) · [Correctness](#correctness) · [Faithfulness (groundedness)](#faithfulness-groundedness) · [Answer relevance](#answer-relevance) · [Context precision and context recall](#context-precision-and-context-recall) · [Citation accuracy](#citation-accuracy) · [Refusal calibration](#refusal-calibration) · [LLM-as-judge](#llm-as-judge) · [Judge biases](#judge-biases) · [Small-sample noise](#small-sample-noise) · [Error analysis (retrieval vs generation)](#error-analysis-retrieval-vs-generation) · [Online metrics](#online-metrics) · [Regression tests in CI](#regression-tests-in-ci) · [importlib.util.spec_from_file_location](#importlibutilspec_from_file_location)

---

## Golden set  `(concept)`
**1. Definition:** A fixed, versioned collection of questions with labelled relevant documents and reference answers, used to evaluate a system repeatably.
**2. In plain words:** `data/eval_questions.json`: 29 questions, 24 with a gold chunk id and short expected answer, 5 with none. Every change is scored on the same questions.
**3. Example to quote:** > "I grow the golden set from real user questions and every bug report, and I never change it in the same commit as the system."
**Seen in:** `main.py` (`load_eval_set`)

## Unanswerable questions  `(concept)`
**1. Definition:** Evaluation questions whose answer is not in the knowledge base, labelled so that the correct behaviour is refusal.
**2. In plain words:** "Do you sell gift cards?", "Who is the founder of Chai Kitchen?". The pipeline refused 4 of 5; the founder question got a confident irrelevant answer.
**3. Example to quote:** > "Without unanswerable questions in the eval set, a system that always answers looks perfect."
**Seen in:** `main.py`, parts B and C

## Hard negatives  `(concept)`
**1. Definition:** Items that look relevant on the surface (shared words or topic) but are not, included to test whether a system can discriminate.
**2. In plain words:** The loyalty chunk for the refund-days question in step 08 is a hard negative: it says "days after delivery" and "refund window" but gives the wrong fact.
**3. Example to quote:** > "I deliberately add questions whose words match the wrong chunk; that's where rankers fail."
**Seen in:** `WALKTHROUGH.md`, theory

## hit@k  `(concept)`
**1. Definition:** For one query, 1 if at least one relevant item is in the top k results, else 0; averaged over queries.
**2. In plain words:** hit@1 for BM25 was 0.71 (17/24) and for hybrid + re-rank 0.88 (21/24).
**3. Example to quote:** > "hit@1 is the share of questions where the very first chunk is the right one."
```python
float(any(r in gold for r in ranked[:k]))
```
**Seen in:** `main.py`, `hit_at_k`

## recall@k  `(concept)`
**1. Definition:** The fraction of all relevant items that appear in the top k: |relevant ∩ top-k| / |relevant|.
**2. In plain words:** With one gold chunk per question it equals hit@k. Dense and hybrid + re-rank both reached recall@5 0.96 (23/24).
**3. Example to quote:** > "Recall@k tells me whether the answer reached the context window at all."
**Seen in:** `main.py`, `recall_at_k`

## precision@k  `(concept)`
**1. Definition:** The fraction of the top k results that are relevant: |relevant ∩ top-k| / k.
**2. In plain words:** With one gold chunk and k = 3, the best possible precision is 0.33. It matters because every irrelevant chunk in the prompt costs tokens and can distract.
**3. Example to quote:** > "Recall says the answer is in; precision says how much noise came with it."
**Seen in:** `WALKTHROUGH.md`, theory (not computed in `main.py`)

## MRR  `(concept)`
**1. Definition:** Mean reciprocal rank: the average of 1 / rank of the first relevant item, 0 when none is retrieved.
**2. In plain words:** BM25 0.79, dense 0.90, hybrid RRF 0.88, hybrid + re-rank 0.92. Ranks must start at 1: starting at 0 raised `ZeroDivisionError`.
**3. Example to quote:** > "MRR rewards putting the answer near the top; rank 2 is worth half of rank 1."
**Seen in:** `main.py`, `reciprocal_rank`

## nDCG@k  `(concept)`
**1. Definition:** Normalised discounted cumulative gain: DCG@k = Σ rel_i / log2(i + 1) over the top k, divided by the DCG of the ideal ordering.
**2. In plain words:** A hit at rank 1 is worth 1.0, at rank 2 about 0.63, at rank 3 0.5. It supports several relevant items and graded relevance. Hybrid + re-rank scored 0.93 at k = 5.
**3. Example to quote:** > "nDCG is the metric I use when there are several relevant chunks with different usefulness."
**Seen in:** `main.py`, `ndcg_at_k`

## Correctness  `(concept)`
**1. Definition:** Whether the generated answer matches the reference answer, judged by exact or fuzzy matching, a model, or a human.
**2. In plain words:** Here: does the expected string (for example "9 days") appear in the reply? 16 of 24. Crude: a correct paraphrase ("nine days") would fail.
**3. Example to quote:** > "My correctness check is a substring match, so I read the failures before blaming the model."
**Seen in:** `main.py`, part C (`is_correct`)

## Faithfulness (groundedness)  `(concept)`
**1. Definition:** The degree to which every claim in an answer is supported by the retrieved context it relies on.
**2. In plain words:** Proxy here: at least 60% of the answer's content words appear in the cited chunk. 22/22 answers were faithful, yet only 16 were correct: true sentences can miss the question.
**3. Example to quote:** > "Faithful is not the same as correct; I measure both."
**Seen in:** `main.py`, `supported`

## Answer relevance  `(concept)`
**1. Definition:** Whether the answer addresses the question that was asked, regardless of whether it is true.
**2. In plain words:** "Boiling water makes green tea bitter." is faithful but does not give the 80 degrees asked for.
**3. Example to quote:** > "Answer relevance catches faithful but off-topic answers."
**Seen in:** `main.py`, `JUDGE_PROMPT`

## Context precision and context recall  `(concept)`
**1. Definition:** RAGAS-style metrics: context precision measures whether relevant chunks are ranked high among the retrieved ones; context recall measures whether the retrieved context contains all the information in the reference answer.
**2. In plain words:** They are the generation-side cousins of precision@k and recall@k, judged on content rather than chunk ids, often by an LLM. `JUDGE_PROMPT` asks a related question as "context_relevance".
**3. Example to quote:** > "Context recall tells me if retrieval brought everything needed; context precision tells me how much junk came with it."
**Seen in:** `WALKTHROUGH.md`, theory; `main.py` (`JUDGE_PROMPT`)

## Citation accuracy  `(concept)`
**1. Definition:** Whether cited sources exist and actually support the statement they are attached to.
**2. In plain words:** The script checks existence (22/22 valid) and, through `supported`, overlap with the cited chunk. Existence alone passed the wrong distractor answer in step 08.
**3. Example to quote:** > "I check citations point to real sources and that those sources contain the claim."
**Seen in:** `main.py`, part C

## Refusal calibration  `(concept)`
**1. Definition:** Choosing the score threshold below which the system declines to answer by measuring false answers and false refusals on labelled data.
**2. In plain words:** At -6.5: 22/24 answered, 4/5 refused. At -3: all 5 refused but only 18/24 answered. Unanswerable scores reach -4.8 and answerable ones go down to -10.9, so there is no perfect threshold.
**3. Example to quote:** > "I sweep the threshold and choose the trade-off; overlapping score distributions mean some errors are unavoidable."
**Seen in:** `main.py`, part B

## LLM-as-judge  `(concept)`
**1. Definition:** Using a language model with a grading rubric to score outputs, often returning structured scores such as JSON.
**2. In plain words:** The template asks for 1-5 scores for faithfulness, answer relevance and context relevance. Cheap and scalable, but itself a model that must be validated.
**3. Example to quote:** > "Before trusting an LLM judge I measure its agreement with about 50 human labels."
**Seen in:** `main.py`, `JUDGE_PROMPT`

## Judge biases  `(concept)`
**1. Definition:** Systematic errors of LLM judges, notably position bias (favouring an option because of its place), verbosity bias (favouring longer answers) and self-preference (favouring outputs of its own model family).
**2. In plain words:** Swap the order and average, use a fixed rubric with reference answers, and use a different, stronger judge than the generator.
**3. Example to quote:** > "I randomise order in pairwise judging because judges prefer whichever answer comes first."
**Seen in:** `WALKTHROUGH.md`, theory

## Small-sample noise  `(concept)`
**1. Definition:** Random variation in a metric caused by evaluating on few examples.
**2. In plain words:** One question out of 24 is 0.04. At p ≈ 0.88 the standard error is about 0.066, so ±0.13 for a 95% interval. Dense (0.83) versus hybrid RRF (0.83) versus re-rank (0.88) is not a proven ranking.
**3. Example to quote:** > "With 24 questions I report counts like 21/24 and don't celebrate a 0.04 gain."
**Seen in:** `main.py`, part A note

## Error analysis (retrieval vs generation)  `(concept)`
**1. Definition:** Manually or automatically classifying each failure by its cause, here by whether the gold chunk was in the generator's context.
**2. In plain words:** All 6 wrong answers had the gold chunk in context, so they are generation problems; the 2 false refusals are threshold problems.
**3. Example to quote:** > "If the gold chunk was in context I fix the prompt or model; if not, I fix retrieval."
**Seen in:** `main.py`, part C (`where`)

## Online metrics  `(concept)`
**1. Definition:** Measurements collected from real usage in production, such as user feedback, resolution rate, refusal rate, latency and cost per answer.
**2. In plain words:** Thumbs up/down, "solved without a human", citation clicks. They catch question types the golden set lacks; failures get added to the golden set.
**3. Example to quote:** > "Offline evals gate releases; online metrics tell me what the offline set is missing."
**Seen in:** `WALKTHROUGH.md`, theory

## Regression tests in CI  `(concept)`
**1. Definition:** Running the evaluation suite automatically on every change and failing the build if key metrics drop below agreed thresholds.
**2. In plain words:** The silent re-rank bug in step 07 (hit@1 0.88 to 0.00) would be caught instantly.
**3. Example to quote:** > "Changing a prompt, chunk size or embedding model reruns the golden set in CI."
**Seen in:** `WALKTHROUGH.md`, theory

## importlib.util.spec_from_file_location  `(standard library)`
**1. Definition:** Creates a module spec from a file path; with `module_from_spec` and `spec.loader.exec_module` it imports a Python file that is not on the normal import path.
**2. In plain words:** `08_rag_pipeline_end_to_end` starts with a digit, so `import` cannot name it; this loads step 08's `main.py` directly and reuses `RAGPipeline`. Its `if __name__ == "__main__"` demo does not run.
**3. Example to quote:** > "I load the step 08 pipeline by file path so the evaluation tests exactly the code that ships."
```python
spec = importlib.util.spec_from_file_location("rag08", path)
rag08 = importlib.util.module_from_spec(spec); spec.loader.exec_module(rag08)
```
**Seen in:** `main.py`, top
