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
