# Terms — Chunking Evaluation

Folder: `19_chunking_deep_dive/10_chunking_evaluation` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [19 README](../README.md)

**Terms in this folder:** [hit@k](#hitk) · [MRR](#mrr) · [recall@k](#recallk) · [Offset-based relevance](#offset-based-relevance) · [Context tokens sent](#context-tokens-sent) · [Context precision](#context-precision) · [Ingestion time and LLM calls](#ingestion-time-and-llm-calls) · [Bootstrap confidence interval](#bootstrap-confidence-interval) · [Slice analysis](#slice-analysis) · [Chunk attribution](#chunk-attribution)

---

## hit@k  `(concept)`
**1. Definition:** 1 if a relevant context is among the top k, else 0, averaged.
**2. In plain words:** Did the answer make the first k?
**3. Example to quote:** > "hit@k: Did the answer make the first k?"
**Seen in:** common19.evaluate

## MRR  `(concept)`
**1. Definition:** Mean of 1/rank of the first relevant context.
**2. In plain words:** Rewards putting the answer first.
**3. Example to quote:** > "MRR: Rewards putting the answer first."
**Seen in:** common19.evaluate

## recall@k  `(concept)`
**1. Definition:** Share of the answer sentences (a question may have several) covered by the top k.
**2. In plain words:** Matters for multi-evidence questions.
**3. Example to quote:** > "recall@k: Matters for multi-evidence questions."
**Seen in:** common19.evaluate

## Offset-based relevance  `(concept)`
**1. Definition:** A context is relevant if its source span covers the answer span.
**2. In plain words:** Lets different chunkers be compared.
**3. Example to quote:** > "Offset-based relevance: Lets different chunkers be compared."
**Seen in:** common19.covers

## Context tokens sent  `(concept)`
**1. Definition:** Tokens of the top-3 context texts: what the LLM prompt pays.
**2. In plain words:** The cost side of the trade-off.
**3. Example to quote:** > "Context tokens sent: The cost side of the trade-off."
**Seen in:** common19.evaluate

## Context precision  `(concept)`
**1. Definition:** Answer characters / characters sent.
**2. In plain words:** How much of the prompt is useful.
**3. Example to quote:** > "Context precision: How much of the prompt is useful."
**Seen in:** common19.char_precision

## Ingestion time and LLM calls  `(concept)`
**1. Definition:** Build + embedding time per strategy; LLM calls and tokens for LLM-based chunkers.
**2. In plain words:** What you pay up front.
**3. Example to quote:** > "Ingestion time and LLM calls: What you pay up front."
**Seen in:** main.py, main

## Bootstrap confidence interval  `(concept)`
**1. Definition:** Resample questions with replacement 1000 times; 2.5th and 97.5th percentile of the mean.
**2. In plain words:** Shows how big the noise is.
**3. Example to quote:** > "Only gaps above ~0.10 are evidence with 74 questions."
**Seen in:** main.py, bootstrap

## Slice analysis  `(concept)`
**1. Definition:** Metrics per question kind (table, code, reference) and per corpus.
**2. In plain words:** A strategy can win on average and lose where it matters.
**3. Example to quote:** > "Slice analysis: A strategy can win on average and lose where it matters."
**Seen in:** main.py, T2

## Chunk attribution  `(concept)`
**1. Definition:** Which chunk supports which part of the generated answer (citation precision/recall); needs a generation step.
**2. In plain words:** Cheap proxy: context precision.
**3. Example to quote:** > "Chunk attribution: Cheap proxy: context precision."
**Seen in:** WALKTHROUGH, theory
