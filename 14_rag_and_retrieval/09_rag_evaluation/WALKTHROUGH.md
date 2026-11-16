# Step 09 — Evaluating RAG

## The story so far
[Step 08](../08_rag_pipeline_end_to_end/WALKTHROUGH.md) assembled the full pipeline and showed it failing in four ways on a handful of questions: a distractor, a false refusal, a poisoned fact, and answers that looked fine.

## The pain point
"It looks good on the five questions I tried" is how RAG systems ship broken. Change the chunk size, the threshold or the prompt and you cannot tell whether things got better or worse. Worse, a single end-to-end score hides *where* it broke: was the right chunk never retrieved, or retrieved and then ignored?

## The spark
Score the two halves separately on a fixed labelled set. Retrieval: is the gold chunk ranked high? (hit@1, recall@k, MRR, nDCG). Refusal: does the threshold separate answerable from unanswerable? Generation: is the answer correct, supported by its citations, and are unanswerable questions refused? Then read every failure one by one.

## Goal
Compute standard retrieval metrics for four retrievers, calibrate the refusal threshold, measure end-to-end correctness and faithfulness, and do error analysis that labels each failure as retrieval or generation.

## The theory (read this before the interview)
**Why separate retrieval and generation.** The answer can only be as good as the context. If the gold chunk is missing, fix retrieval (chunking, hybrid, re-ranking, query rewriting). If it is present and the answer is still wrong, fix generation (prompt, model, context order and size). One blended score cannot tell you which knob to turn.

**Building a golden set.** Questions phrased the way real users ask (paraphrases, not copies of the document wording), exact identifiers (`CK-MASALA-250`), **unanswerable** questions (here 5 of 29) to test refusal, and *hard negatives*: questions whose words match a wrong chunk (the distractor in step 08). Label the gold chunk ids and a short expected answer. Grow it from production logs and every bug report. Keep it versioned and never tune on the test portion only.

**Retrieval metrics** (ranked list, gold set G, cut-off k):
- hit@k = 1 if any gold item is in the top k, else 0; averaged over questions.
- recall@k = |G ∩ top-k| / |G|. With one gold chunk per question (as here) it equals hit@k.
- precision@k = |G ∩ top-k| / k. Matters when you pay for every chunk you put in the prompt.
- MRR = mean over questions of 1 / rank of the first relevant item (0 if absent).
- DCG@k = Σ_{i=1..k} rel_i / log2(i + 1); nDCG@k = DCG@k / IDCG@k, where IDCG is the DCG of the perfect ordering. With binary relevance a single hit at rank 1 scores 1.0, at rank 2 scores 1/log2(3) ≈ 0.63. Graded relevance often uses gain 2^rel − 1.

**Generation metrics** (RAGAS-style names, common across tools):
- *Correctness*: does the answer match the reference? Here: the expected answer string appears in the reply (crude but transparent).
- *Faithfulness / groundedness*: is every claim supported by the retrieved (cited) context? Here: ≥ 60% of the answer's content words appear in the cited chunk.
- *Answer relevance*: does the answer address the question (a faithful sentence can still be off-topic)?
- *Context relevance / context precision*: are the retrieved chunks about the question, and are the relevant ones ranked first?
- *Context recall*: does the retrieved context contain everything needed for the reference answer?
- *Citation accuracy*: do the cited sources exist and actually support the sentence they are attached to?

**Refusal calibration.** Sweep the threshold and count both errors: answered unanswerables (hallucination risk) versus refused answerables (lost value). The score distributions overlap: unanswerable best scores go up to -4.8, while answerable ones go down to -10.9. No threshold is perfect; you choose the trade-off the business prefers.

**LLM-as-judge.** A strong model grades faithfulness and relevance with a rubric and returns JSON. Pros: scales, handles paraphrase, explains itself. Known biases: *position bias* (prefers the first option in pairwise comparisons), *verbosity bias* (prefers longer answers), *self-preference* (rates outputs from its own model family higher). Mitigations: fixed rubric and 1-5 anchors, reference answers, swap order and average, use a different and stronger model than the generator, and validate against human labels (agreement on, say, ~50 examples) before trusting it.

**Small-sample noise.** With 24 questions, one question = 1/24 ≈ 0.04. A binomial standard error at p ≈ 0.88 is √(0.88 × 0.12 / 24) ≈ 0.066, so a 95% interval is roughly ±0.13. Differences of 0.04-0.08 between retrievers here are noise. Use more questions, report counts ("21/24") next to rates, and compare systems on the *same* questions (paired comparison).

**Error analysis.** Read failures, not averages. For each wrong answer ask: was the gold chunk in the context? No -> retrieval problem. Yes -> generation problem. Then group failures by cause (distractor, number missing, threshold, paraphrase).

**Online metrics and regression tests.** Offline sets do not cover everything users ask. In production, track thumbs up/down, resolution rate (no escalation to a human), refusal rate, citation click-through, and latency and cost per answer; sample conversations for human review and add the failures to the golden set. Run the offline evaluation in CI on every change to chunking, embeddings, prompts or models, with thresholds that fail the build on a regression.

## Run it
```bash
# from the repo root (~10-30 s on CPU; loads the step 08 pipeline)
.venv/bin/python 14_rag_and_retrieval/09_rag_evaluation/main.py
# with a real generator (only part C changes)
LLM_BACKEND=local .venv/bin/python 14_rag_and_retrieval/09_rag_evaluation/main.py
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| hit@k vs recall@k vs precision@k | "found in top k" | Related | hit@k: any gold chunk in the top k; recall@k: share of all gold chunks found; precision@k: share of the top k that are gold. With one gold chunk, hit@k equals recall@k |
| MRR vs nDCG@k vs MAP | rank-aware retrieval metrics | No | MRR looks only at the first correct hit (`1/rank`); nDCG discounts every hit by `1/log2(rank+1)`; MAP averages precision at each hit |
| Faithfulness vs groundedness vs correctness | "is it supported?", "is it right?" | No | Faithful = supported by the cited context; correct = matches the gold answer. A faithful answer can still be wrong |
| Retrieval error vs generation error | where did it fail | No | Retrieval: the gold chunk never reached the context; generation: it was there but the answer is wrong |
| LLM-as-judge vs human evaluation vs reference-based metric | grading methods | No | A model scores answers with a rubric; humans label; reference metrics compare to a gold string. Validate the judge against human labels |
| False refusal vs false answer | threshold errors | No | Refusing an answerable question vs answering an unanswerable one; the threshold trades one for the other |
| Golden set vs eval set vs test set | labelled questions | Same idea | The fixed set of questions with known gold chunks used to compare versions |

## Code, step by step

### 1. Docstring  (`main.py` lines 1–16)
```python
"""
GOAL:  Evaluate RAG the way you would before shipping it, on 29 labelled questions (24 answerable, 5 not):
         A) RETRIEVAL metrics for four retrievers: hit@1, recall@3, recall@5, MRR, nDCG@5
         B) REFUSAL calibration: sweep the re-rank threshold, trade false refusals against false answers
         C) GENERATION metrics on the full pipeline: correctness, faithfulness (is every answer sentence
            supported by its cited chunk?), citation validity, and an LLM-as-judge prompt template

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 14_rag_and_retrieval/09_rag_evaluation/main.py
EXPECTED OUTPUT (shortened):
    retriever            hit@1  recall@3  recall@5   MRR  nDCG@5
    BM25 ... dense ... hybrid RRF ... hybrid + re-rank ...
    threshold  answered(of 24)  refused(of 5)
    end-to-end: correct x/24 | faithful x/x | citations valid x/x | correct refusals x/5
NEEDS: torch, transformers (cached MiniLM + cross-encoder).  ~30 s.
"""
```
- **In simple words:** What is measured: retrieval, refusal and end-to-end answers on 29 labelled questions.
- **Exactly:** 24 are answerable (they have gold chunk ids) and 5 are not. Part A: four retrievers with hit@1, recall@3, recall@5, MRR, nDCG@5. Part B: sweep the refusal threshold. Part C: correctness, faithfulness, citation validity and an LLM-as-judge template.
- **Why it is here:** This is how you decide whether a change helped before shipping.

### 2. Imports and path setup  (`main.py` lines 18–28)
```python
import importlib.util
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import load_eval_set, tokenize   # noqa: E402
```
- **In simple words:** Load tools and the eval helper.
- **Exactly:** `importlib.util` loads a file by path; `math` for `log2`; `re` for citation numbers. `load_eval_set` reads the 29 questions; `tokenize` is reused for the faithfulness proxy.
- **Why it is here:** Later steps reuse code from `common`.

### 3. Load the step-08 pipeline by file path  (`main.py` lines 30–33)
```python
# Re-use the pipeline from step 08 (loaded by file path, because folder names start with digits).
spec = importlib.util.spec_from_file_location("rag08", HERE.parent / "08_rag_pipeline_end_to_end" / "main.py")
rag08 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag08)
```
- **In simple words:** Reuse the pipeline class written in step 08.
- **Exactly:** Folder names starting with digits cannot be imported normally, so `spec_from_file_location` creates a module `rag08` from `08_rag_pipeline_end_to_end/main.py` and `exec_module` runs it. The `if __name__ == "__main__"` guard in step 08 stops its demo from running.
- **Why it is here:** No duplicated code: the pipeline you evaluate is the pipeline you built.

### 4. Build the pipeline and split the questions  (`main.py` lines 35–39)
```python
rag = rag08.RAGPipeline(rag08.build_chunks())
ids = [c["chunk_id"] for c in rag.chunks]
eval_set = load_eval_set()
answerable = [x for x in eval_set if x["gold"]]
unanswerable = [x for x in eval_set if not x["gold"]]
```
- **In simple words:** Create the pipeline and separate answerable from unanswerable questions.
- **Exactly:** `rag` is a `RAGPipeline` over 36 chunks; `ids` are chunk ids in row order; `answerable` (24 items with a non-empty `gold`) and `unanswerable` (5 items with `gold == []`).
- **Why it is here:** Unanswerable questions test whether the system refuses.

### 5. Metrics: hit@k, recall@k, reciprocal rank  (`main.py` lines 42–59)
```python
# =========================================================================================================
# Metric functions. `ranked` = list of chunk ids, best first. `gold` = set of relevant chunk ids.
# =========================================================================================================
def hit_at_k(ranked, gold, k):
    return float(any(r in gold for r in ranked[:k]))


def recall_at_k(ranked, gold, k):
    """Share of ALL relevant chunks found in the top k (equals hit@k when there is one gold chunk)."""
    return len(set(ranked[:k]) & gold) / len(gold)


def reciprocal_rank(ranked, gold):
    """1/rank of the FIRST relevant chunk, 0 if none: rewards putting the answer at the very top."""
    for i, r in enumerate(ranked, 1):
        if r in gold:
            return 1 / i
    return 0.0
```
- **In simple words:** Three small functions that score one ranked list.
- **Exactly:** `ranked` = chunk ids best first; `gold` = set of relevant ids. `hit_at_k` is 1.0 if any gold in the top k. `recall_at_k = |top-k and gold| / |gold|`. `reciprocal_rank` returns `1/rank` of the first gold hit (ranks start at 1), else 0.
- **Why it is here:** Plain functions are easy to test and to explain.

### 6. Metric: nDCG@k  (`main.py` lines 62–66)
```python
def ndcg_at_k(ranked, gold, k):
    """Discounted cumulative gain: a hit at rank i is worth 1/log2(i+1); divided by the best possible."""
    dcg = sum(1 / math.log2(i + 1) for i, r in enumerate(ranked[:k], 1) if r in gold)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal
```
- **In simple words:** A score that rewards putting relevant chunks near the top, and is 1.0 when perfect.
- **Exactly:** `dcg = sum(1/log2(i+1))` over positions `i` (1-based) holding a gold chunk; `ideal` is the best possible dcg for `min(|gold|, k)` gold chunks; the ratio is in 0 to 1. With binary relevance: rank 1 gives 1.0, rank 2 gives 0.63, rank 3 gives 0.5.
- **Why it is here:** MRR cares about the first hit only; nDCG also handles several relevant chunks.

### 7. dense_only retriever  (`main.py` lines 72–74)
```python
def dense_only(q, k=20):
    s = rag.matrix @ rag.embedder.encode([q])[0]
    return [(int(j), float(s[j])) for j in np.argsort(-s)[:k]]
```
- **In simple words:** A retriever using meaning only.
- **Exactly:** Same as before: matrix product with the query vector, top 20 as `(row, score)`.
- **Why it is here:** Needed to compare against hybrid and reranked.

### 8. The four retrievers  (`main.py` lines 77–82)
```python
retrievers = {
    "BM25": lambda q: rag.bm25.search(q, 20),
    "dense": dense_only,
    "hybrid RRF": rag.retrieve,
    "hybrid + re-rank": lambda q: rag.rerank(q, rag.retrieve(q)),
}
```
- **In simple words:** BM25, dense, hybrid RRF, and hybrid plus reranking, behind one interface.
- **Exactly:** A dict name to function question to `[(row, score), ...]`.
- **Why it is here:** The same loop evaluates all of them.

### 9. Evaluate retrieval  (`main.py` lines 83–93)
```python
print(f"A) RETRIEVAL on {len(answerable)} answerable questions")
print(f"   {'retriever':<18}{'hit@1':>7}{'recall@3':>10}{'recall@5':>10}{'MRR':>7}{'nDCG@5':>8}")
for name, fn in retrievers.items():
    rows = []
    for x in answerable:
        ranked, gold = [ids[j] for j, _ in fn(x["question"])], set(x["gold"])
        rows.append((hit_at_k(ranked, gold, 1), recall_at_k(ranked, gold, 3), recall_at_k(ranked, gold, 5),
                     reciprocal_rank(ranked, gold), ndcg_at_k(ranked, gold, 5)))
    m = np.mean(rows, axis=0)
    print(f"   {name:<18}{m[0]:>7.2f}{m[1]:>10.2f}{m[2]:>10.2f}{m[3]:>7.2f}{m[4]:>8.2f}")
print("   With 24 questions one question = 0.04. Differences smaller than ~0.1 are noise, not progress.")
```
- **In simple words:** Average the five metrics over the 24 questions for each retriever.
- **Exactly:** `rows` has one 5-tuple per question; `np.mean(rows, axis=0)` averages each column. Real: BM25 0.71 / 0.88 / 0.92 / 0.79 / 0.82; dense 0.83 / 0.96 / 0.96 / 0.90 / 0.91; hybrid RRF 0.83 / 0.92 / 0.92 / 0.88 / 0.89; hybrid + re-rank 0.88 / 0.96 / 0.96 / 0.92 / 0.93. One question is 0.04, so differences under about 0.1 are noise.
- **Why it is here:** Without a table of numbers, "better" is an opinion.

### 10. Best reranker score per question  (`main.py` lines 95–98)
```python
# =========================================================================================================
# B) Refusal threshold: the best re-rank score decides "answer" or "I don't know"
# =========================================================================================================
best = {x["id"]: rag.rerank(x["question"], rag.retrieve(x["question"]))[0][1] for x in eval_set}
```
- **In simple words:** For every question, take the highest reranker score.
- **Exactly:** `best[id]` is the top logit after retrieve and rerank. This is the number the refusal guard compares with its threshold.
- **Why it is here:** Needed to calibrate the threshold.

### 11. Sweep the refusal threshold  (`main.py` lines 99–108)
```python
print(f"\nB) REFUSAL threshold on the best re-rank logit ({len(answerable)} answerable, {len(unanswerable)} unanswerable)")
print(f"   {'threshold':>9} {'answered (of 24)':>17} {'refused (of 5)':>15}  verdict")
for th in (-12, -9, -7.5, -6.5, -5, -3, 0):
    answered = sum(best[x["id"]] >= th for x in answerable)
    refused = sum(best[x["id"]] < th for x in unanswerable)
    verdict = "answers everything (hallucination risk)" if refused == 0 else \
        "refuses everything unknown, but loses real answers" if refused == len(unanswerable) and answered < 22 else ""
    print(f"   {th:>9} {answered:>17} {refused:>15}  {verdict}")
print("   unanswerable scores:", sorted(round(best[x["id"]], 1) for x in unanswerable))
print("   lowest answerable  :", sorted(round(best[x["id"]], 1) for x in answerable)[:4], "<- they overlap: no perfect threshold")
```
- **In simple words:** Try several cut-offs and count how many answerable questions still get answered and how many unanswerable ones are refused.
- **Exactly:** Real: at -12 answers 24 and refuses 0 (hallucination risk); at -6.5 answers 22 and refuses 4; at -3 answers 18 and refuses 5 (loses real answers); at 0 answers 16. Unanswerable scores are `[-11.1, -10.4, -9.3, -7.1, -4.8]`; the lowest answerable are `[-10.9, -9.5, -6.0, -4.9]`: they overlap, so there is no perfect threshold.
- **Why it is here:** Shows that the threshold is a business decision (false refusals vs false answers), set on labelled data.

### 12. supported(): a faithfulness proxy  (`main.py` lines 113–116)
```python
def supported(sentence, source_text, min_overlap=0.6):
    """Faithfulness proxy: share of the sentence's content words that also appear in the cited chunk."""
    words = set(tokenize(re.sub(r"\[\d+\]", "", sentence)))
    return bool(words) and len(words & set(tokenize(source_text))) / len(words) >= min_overlap
```
- **In simple words:** Check whether the words of a sentence appear in the chunk it cites.
- **Exactly:** Content words of the sentence (citation markers removed, stopwords dropped) must overlap the source tokens by at least `min_overlap=0.6`. A cheap proxy, not a proof; an LLM or NLI model judge is stronger.
- **Why it is here:** Catches answers that say things not in the source.

### 13. Counters for the end-to-end run  (`main.py` lines 119–121)
```python
chunk_text = {c["chunk_id"]: c["text"] for c in rag.chunks}
correct = faithful = cited_ok = answered = right_refusal = 0
failures = []
```
- **In simple words:** Set up the totals and a list for failures.
- **Exactly:** `chunk_text` maps chunk id to text; counters start at 0; `failures` collects tuples for error analysis.
- **Why it is here:** You learn more from reading failures than from averages.

### 14. End-to-end loop: handle unanswerable and refused  (`main.py` lines 122–131)
```python
for x in eval_set:
    out = rag.answer(x["question"])
    if not x["gold"]:
        right_refusal += out["refused"]
        if not out["refused"]:
            failures.append(("answered an unanswerable question", x["question"], out["answer"]))
        continue
    if out["refused"]:
        failures.append(("refused an answerable question", x["question"], out["answer"]))
        continue
```
- **In simple words:** Run the full pipeline on every question and classify refusals.
- **Exactly:** For an unanswerable question a refusal counts as correct (`right_refusal`); answering it is a failure. For an answerable question a refusal is a failure ("refused an answerable question").
- **Why it is here:** Both kinds of mistake are tracked.

### 15. End-to-end loop: correctness, citations, faithfulness  (`main.py` lines 132–142)
```python
    answered += 1
    is_correct = x["answer"].lower() in out["answer"].lower()
    correct += is_correct
    cited_ok += out["citations_valid"]
    nums = [int(n) for n in re.findall(r"\[(\d+)\]", out["answer"])]
    sources = [chunk_text[out["sources"][n - 1]] for n in nums if 1 <= n <= len(out["sources"])]
    faithful += bool(sources) and supported(out["answer"], " ".join(sources))
    if not is_correct:
        where = "gold chunk WAS in the context -> generation problem" if x["gold"][0] in out["sources"] \
            else "gold chunk NOT in the context -> retrieval problem"
        failures.append((f"wrong answer ({where})", x["question"], out["answer"]))
```
- **In simple words:** For each answered question, check correctness, citations and support.
- **Exactly:** Correct = the gold `answer` string occurs in the reply (case-insensitive). `cited_ok` counts valid citations. The numbers cited are mapped to source chunk texts; `faithful` counts replies supported by their cited text. A wrong answer is labelled "gold chunk WAS in the context -> generation problem" or "NOT in the context -> retrieval problem".
- **Why it is here:** Separates retrieval failures from generation failures, so you fix the right component.

### 16. Print the end-to-end result and failures  (`main.py` lines 144–151)
```python
print(f"\nC) END-TO-END ({rag08.os.getenv('LLM_BACKEND', 'mock simulator')})")
print(f"   answered {answered}/{len(answerable)} | correct {correct}/{len(answerable)} | faithful {faithful}/{answered} "
      f"| citations valid {cited_ok}/{answered} | correct refusals {right_refusal}/{len(unanswerable)}")
print("   failures to read one by one (error analysis beats staring at averages):")
for kind, q, a in failures:
    print(f"   - {kind}: {q!r}\n       -> {a[:110]}")
print("   Faithful is not the same as correct: a true sentence from the context can still miss the question.")
print("   Here most errors are generation errors: the word-overlap simulator picks the wrong sentence of the right chunk.")
```
- **In simple words:** Print the totals and each failure.
- **Exactly:** Real (mock simulator): answered 22/24, correct 16/24, faithful 22/22, citations valid 22/22, correct refusals 4/5. Failures include wrong sentences chosen from the right chunk (generation) and two answerable questions refused (London, pay later). "Who is the founder?" was answered instead of refused.
- **Why it is here:** Faithful 22/22 but correct 16/24: faithful is not correct.

### 17. LLM-as-judge prompt template  (`main.py` lines 153–162)
```python
JUDGE_PROMPT = """You are grading a RAG answer.
Question: {question}
Retrieved context: {context}
Answer: {answer}
Score each from 1 to 5 and reply as JSON {{"faithfulness": n, "answer_relevance": n, "context_relevance": n}}:
- faithfulness: is every claim in the answer supported by the context?
- answer_relevance: does the answer address the question?
- context_relevance: is the retrieved context about the question?"""
print("\n   LLM-as-judge template (run it with a STRONG model; validate it against ~50 human labels first):")
print("   " + JUDGE_PROMPT.splitlines()[0], "...", JUDGE_PROMPT.splitlines()[-3].strip())
```
- **In simple words:** A ready prompt asking a strong model to grade answers from 1 to 5.
- **Exactly:** Three scores: faithfulness, answer relevance, context relevance, returned as JSON (`{{` and `}}` are escaped braces for `str.format`). Must be validated against about 50 human labels before you trust it.
- **Why it is here:** Scales evaluation beyond hand checking, but a judge has its own errors.

### 18. Exercises as comments  (`main.py` lines 164–167)
```python
# TRY IT: add 5 new questions to data/eval_questions.json (including 2 unanswerable) and re-run. Do the tables move?
# TRY IT: set min_overlap in supported() to 0.9 and 0.3. How sensitive is this faithfulness proxy?
# TRY IT: re-run with rag08.RAGPipeline(..., top_k=1). Which metric changes: retrieval or end-to-end?
# BREAK IT: compute MRR with ranks starting at 0 (enumerate(ranked)) -> ZeroDivisionError for a top-1 hit.
```
- **In simple words:** Ideas to try and break.
- **Exactly:** BREAK IT: ranks from 0 in MRR cause `ZeroDivisionError` for a top-1 hit.
- **Why it is here:** Metrics bugs make you ship the wrong thing.


## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `importlib.util.spec_from_file_location(name, path)` + `exec_module` | import a `.py` by file path (folder names starting with digits cannot be imported normally) | `rag08.RAGPipeline` |
| `float(any(r in gold for r in ranked[:k]))` | 1.0 if any of the top k is gold | hit@k |
| `len(set(ranked[:k]) & gold) / len(gold)` | set intersection size / gold size | recall@k |
| `for i, r in enumerate(ranked, 1)` | ranks starting at 1 | `1 / i` is safe |
| `math.log2(i + 1)` | DCG discount | rank 1 -> 1, rank 3 -> 2 |
| `np.mean(rows, axis=0)` | column-wise average of a list of tuples | five metrics at once |
| `sum(best[x["id"]] >= th for x in answerable)` | summing booleans counts the Trues | |
| `correct += is_correct` | bool adds as 0 or 1 | |
| `"...{{...}}..."` inside `JUDGE_PROMPT` | doubled braces = literal braces in a `str.format` template | `{{"faithfulness": n}}` |
| `a[:110]` | truncate a long string for printing | |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `hit_at_k`, `recall_at_k` | this file | binary/fractional retrieval hit | `ranked`, `gold`, `k` | float |
| `reciprocal_rank` | this file | 1 / rank of first gold | – | float |
| `ndcg_at_k` | this file | position-discounted gain / ideal | `k` | float |
| `supported(sentence, source, min_overlap=0.6)` | this file | faithfulness proxy by content-word overlap | `min_overlap` | bool |
| `rag08.RAGPipeline` | step 08 | retrieve, rerank, answer | `refuse_below`, `top_k` | object |
| `load_eval_set`, `tokenize` | `common.py` | labelled questions, word tokens | – | list |
| `importlib.util` | standard library | load a module from a path | – | module |
| `math.log2` | standard library | base-2 logarithm | – | float |
| `np.mean` | numpy | averages | `axis` | ndarray |

## Execution trace (real run)
Deterministic with the simulator (timings vary; the run took ~11 s here).
```
A) RETRIEVAL on 24 answerable questions
   retriever           hit@1  recall@3  recall@5    MRR  nDCG@5
   BM25                 0.71      0.88      0.92   0.79    0.82
   dense                0.83      0.96      0.96   0.90    0.91
   hybrid RRF           0.83      0.92      0.92   0.88    0.89
   hybrid + re-rank     0.88      0.96      0.96   0.92    0.93
   With 24 questions one question = 0.04. Differences smaller than ~0.1 are noise, not progress.

B) REFUSAL threshold on the best re-rank logit (24 answerable, 5 unanswerable)
   threshold  answered (of 24)  refused (of 5)  verdict
         -12                24               0  answers everything (hallucination risk)
          -9                22               3
        -7.5                22               3
        -6.5                22               4
          -5                21               4
          -3                18               5  refuses everything unknown, but loses real answers
           0                16               5  refuses everything unknown, but loses real answers
   unanswerable scores: [-11.1, -10.4, -9.3, -7.1, -4.8]
   lowest answerable  : [-10.9, -9.5, -6.0, -4.9] <- they overlap: no perfect threshold

C) END-TO-END (mock simulator)
   answered 22/24 | correct 16/24 | faithful 22/22 | citations valid 22/22 | correct refusals 4/5
   - wrong answer (gold chunk WAS in the context -> generation problem): 'What is the money back window after the tea arrives?'
       -> The money is returned to the original payment method within 5 working days after we approve the request. [1]
   - wrong answer (gold chunk WAS in the context -> generation problem): 'How long does delivery take?'
       -> Remote pin codes in the North-East and Ladakh can take up to 9 working days. [1]
   - refused an answerable question: 'Can you send tea to my cousin in London?'
   - wrong answer (... generation problem): 'Which blend has almonds in it?'
       -> All blends are packed in a facility that also handles almonds, so traces are possible. [1]
   - wrong answer (... generation problem): 'Which teas will not keep me awake at night?'
       -> Keep tea in an airtight container away from sunlight and strong smells. [1]
   - wrong answer (... generation problem): 'What is the smallest bulk order for my cafe?'
       -> Cafes, hotels and offices can buy wholesale. [1]
   - refused an answerable question: 'Can my office pay later instead of upfront?'
   - wrong answer (... generation problem): 'What water temperature should I use for green tea?'
       -> Boiling water makes green tea bitter. [1]
   - answered an unanswerable question: 'Who is the founder of Chai Kitchen?'
       -> Refunds at Chai Kitchen are issued within 9 days of delivery if the seal on the pouch is unbroken. [1]
```
What to read from it:
- **A:** BM25 alone is weakest (hit@1 0.71 = 17/24). Dense alone (0.83, MRR 0.90) is level with or slightly ahead of hybrid RRF (0.83, MRR 0.88) on this tiny set; re-ranking is best on every column. All gaps are one to three questions, within noise.
- **B:** at the pipeline's -6.5: 22/24 answered and 4/5 refused. The two answerable questions lost are London (-10.9) and office pay-later (-9.5). The unanswerable founder question scores -4.8, *higher* than two answerable ones, so no threshold catches it without losing more real answers. At -3 all 5 are refused but only 18/24 answered.
- **C:** 24 = 16 correct + 6 wrong + 2 false refusals. All 6 wrong answers had the gold chunk in the context: generation problems (the word-overlap simulator chose the wrong sentence). Faithful is 22/22 while correct is only 16/24: every answer was a true sentence from its cited chunk, and still six missed the question. The founder question got a fluent, cited, irrelevant answer (it mentions "Chai Kitchen").
- The script then prints the first and a later line of an LLM-as-judge template (faithfulness, answer relevance, context relevance as JSON).

## Experiments (change ONE thing)
1. Add 5 new questions to `data/eval_questions.json` (two unanswerable) in a *copy* of the data, and re-run. Do the retrieval tables move by more than 0.04 per question?
2. Set `min_overlap` in `supported()` to 0.9, then 0.3. How sensitive is the faithfulness proxy? A metric that flips with its own hyper-parameter needs validating against human labels.
3. `rag08.RAGPipeline(..., top_k=1)`. Retrieval tables (A) should not change; part C may. Which questions change?
4. `LLM_BACKEND=local`: part C now measures a real model. Expect generation errors of a different kind (paraphrased answers can fail the substring "correct" check: a metric weakness, not always a model error).

## Debugging practice
**BREAK IT (confirmed in a scratch copy):** compute MRR with ranks starting at 0 (`for i, r in enumerate(ranked):`).
```
A) RETRIEVAL on 24 answerable questions
   retriever           hit@1  recall@3  recall@5    MRR  nDCG@5
Traceback (most recent call last):
  File ".../main.py", line 90, in <module>
    reciprocal_rank(ranked, gold), ndcg_at_k(ranked, gold, 5)))
  File ".../main.py", line 58, in reciprocal_rank
    return 1 / i
ZeroDivisionError: division by zero
```
It crashes as soon as any gold chunk sits at position 0 (rank 1), before even the BM25 row is printed. Lucky: an off-by-one that crashes is easy. The same off-by-one in hit@k (`ranked[:k-1]`) or in nDCG would silently produce plausible, wrong numbers. Fix: ranks are 1-based in every IR formula; write a unit test with a hand-computed case (gold at rank 2 -> RR 0.5, nDCG ≈ 0.63).

## Rebuild checklist
- [ ] golden set: realistic questions, exact codes, paraphrases, unanswerables, hard negatives; gold chunk ids + expected answer
- [ ] retrieval metrics per retriever: hit@1, recall@k, MRR, nDCG@k (1-based ranks)
- [ ] sweep the refusal threshold; print both error counts and the score distributions
- [ ] end-to-end: correctness, faithfulness, citation validity, correct refusals
- [ ] for each failure: gold in context? -> retrieval vs generation
- [ ] report counts next to rates; remember 1 question = 0.04
- [ ] an LLM judge only after checking it against human labels; run the suite in CI

## The new problem (cliffhanger)
Now we can *measure*. The measurements point at specific gaps: London is lost because the question's words ("cousin in London") do not match the document's ("international shipping"); generation picks the wrong sentence; and a real deployment needs a proper vector database, not a NumPy matrix. Next: the advanced techniques that target exactly these gaps, and the vector DB landscape: [step 10, advanced RAG and vector databases](../10_advanced_rag_and_vector_dbs/WALKTHROUGH.md).

> **Say it aloud (interview box):** I evaluate RAG in two halves on a labelled golden set that includes paraphrases, exact codes, hard negatives and unanswerable questions: retrieval with hit@k, recall@k, MRR and nDCG, and generation with correctness, faithfulness to the cited context, answer relevance and correct refusals. For every wrong answer I check whether the gold chunk was in the context, which tells me whether to fix retrieval or generation; here all six wrong answers were generation errors and the two false refusals came from an overlapping refusal threshold. I keep sample size in mind, since one question out of 24 is 0.04, validate any LLM judge against human labels for position and verbosity bias, and run the suite in CI alongside online signals like thumbs and resolution rate.

Deep dive: evaluating *chunking* strategies (offset-based relevance, context-token cost, ingestion cost, bootstrap noise) → [19/10 chunking evaluation](../../19_chunking_deep_dive/10_chunking_evaluation/WALKTHROUGH.md)
