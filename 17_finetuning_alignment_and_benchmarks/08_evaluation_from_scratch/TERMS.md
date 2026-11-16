# Terms — Evaluation from scratch

Folder: `17_finetuning_alignment_and_benchmarks/08_evaluation_from_scratch` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [17 README](../README.md)

**Terms in this folder:** [Perplexity](#perplexity) · [Exact match (EM)](#exact-match-em) · [Token F1](#token-f1) · [Answer normalisation](#answer-normalisation) · [Log-likelihood multiple choice](#log-likelihood-multiple-choice) · [acc vs acc_norm](#acc-vs-acc_norm) · [Letter-format MCQ (MMLU style)](#letter-format-mcq-mmlu-style) · [Generative evaluation with answer extraction](#generative-evaluation-with-answer-extraction) · [Few-shot prompting](#few-shot-prompting) · [pass@k (unbiased estimator)](#passk-unbiased-estimator) · [Bootstrap confidence interval](#bootstrap-confidence-interval) · [Wilson interval](#wilson-interval) · [Paired bootstrap](#paired-bootstrap) · [Standard error of an accuracy](#standard-error-of-an-accuracy) · [Contamination (n-gram overlap)](#contamination-n-gram-overlap) · [Harness settings that change scores](#harness-settings-that-change-scores) · [np.random.default_rng](#nprandomdefault_rng) · [Numerical edge cases (n - c < k)](#numerical-edge-cases-n---c--k)

---

## Perplexity  `(concept)`
**1. Definition:** exp of the mean per-token cross-entropy in nats.
**2. In plain words:** How surprised the model is on average.
**3. Example to quote:** > "Weighted by token count: total NLL divided by total predicted tokens."
```python
math.exp(nll / ntok)
```
**Seen in:** `main.py`, section 3

## Exact match (EM)  `(concept)`
**1. Definition:** 1 if the normalised prediction equals the normalised gold string.
**2. In plain words:** Strict credit.
**3. Example to quote:** > "EM is 1 only if the normalised prediction equals the normalised gold string."
```python
exact_match('The Paris.', 'Paris') == 1.0
```
**Seen in:** `main.py`, section 4

## Token F1  `(concept)`
**1. Definition:** Harmonic mean of token precision and recall after normalisation (multiset overlap).
**2. In plain words:** Partial credit.
**3. Example to quote:** > "'Paris, France' vs 'Paris' gives 0.67."
```python
token_f1('Paris, France', 'Paris')
```
**Seen in:** `main.py`, section 4

## Answer normalisation  `(concept)`
**1. Definition:** Lowercase, remove punctuation and articles, collapse whitespace.
**2. In plain words:** Compare meaning, not formatting.
**3. Example to quote:** > "Lowercase, strip punctuation and articles, collapse spaces: then 'The Paris.' equals 'Paris'."
```python
re.sub(r'\b(a|an|the)\b', ' ', s)
```
**Seen in:** `main.py`, `normalise`

## Log-likelihood multiple choice  `(concept)`
**1. Definition:** Score each choice by log P(choice | question) and pick the maximum.
**2. In plain words:** Which ending does the model find likeliest?
**3. Example to quote:** > "How lm-evaluation-harness scores MMLU-style and HellaSwag-style tasks."
**Seen in:** `main.py`, section 5

## acc vs acc_norm  `(concept)`
**1. Definition:** acc uses the raw summed log-prob; acc_norm divides by the length of the choice (this demo: bytes) to remove the short-answer bias.
**2. In plain words:** Fair comparison between short and long answers.
**3. Example to quote:** > "Report which one you used."
```python
lp / len(c.encode())
```
**Seen in:** `main.py`, section 5

## Letter-format MCQ (MMLU style)  `(concept)`
**1. Definition:** Show options as A/B/C/D and compare the log-probabilities of the letter tokens.
**2. In plain words:** Pick a letter instead of writing the answer.
**3. Example to quote:** > "Needs a model that knows the format; small base models are near chance."
```python
menu = ...; cont_logprob(menu, ' A')
```
**Seen in:** `main.py`, section 5

## Generative evaluation with answer extraction  `(concept)`
**1. Definition:** The model writes text; a rule extracts the answer; a checker scores it.
**2. In plain words:** Let it talk, then find the answer.
**3. Example to quote:** > "The extraction rule is part of the benchmark."
```python
raw.split('\n')[0].strip()
```
**Seen in:** `main.py`, section 8

## Few-shot prompting  `(concept)`
**1. Definition:** Prepending k worked examples to the question.
**2. In plain words:** Show it the pattern.
**3. Example to quote:** > "This demo uses 3 shots."
```python
FEWSHOT + 'Q: ...\nA:'
```
**Seen in:** `main.py`, section 8

## pass@k (unbiased estimator)  `(concept)`
**1. Definition:** 1 - C(n-c,k)/C(n,k) from n samples with c correct; 1 if n-c < k.
**2. In plain words:** Chance that at least one of k tries works.
**3. Example to quote:** > "n=10, c=3, k=5 gives 0.9167."
```python
1 - np.prod(1 - k / np.arange(n - c + 1, n + 1))
```
**Seen in:** `main.py`, `pass_at_k`

## Bootstrap confidence interval  `(concept)`
**1. Definition:** Resample items with replacement and take percentiles of the metric.
**2. In plain words:** Shake the test set and see the wobble.
**3. Example to quote:** > "I resample the 30 questions with replacement 4,000 times and report the 2.5th and 97.5th percentile of the mean."
```python
r.choice(correct, size=(4000, n)).mean(1)
```
**Seen in:** `main.py`, `bootstrap_ci`

## Wilson interval  `(concept)`
**1. Definition:** A closed-form binomial interval that behaves well at small n and extreme p.
**2. In plain words:** The better formula for small tests.
**3. Example to quote:** > "For a small n and a proportion near 0 or 1 I prefer the Wilson interval over the normal approximation."
```python
centre +- half
```
**Seen in:** `main.py`, `wilson_ci`

## Paired bootstrap  `(concept)`
**1. Definition:** Resample items once and compare two systems on the same resample.
**2. In plain words:** Compare on the same questions.
**3. Example to quote:** > "The paired bootstrap resamples the same questions for both systems, so the difference has far less noise."
```python
a[idx].mean(1) - b[idx].mean(1)
```
**Seen in:** `main.py`, `paired_bootstrap`

## Standard error of an accuracy  `(concept)`
**1. Definition:** sqrt(p(1-p)/n).
**2. In plain words:** How noisy a score is.
**3. Example to quote:** > "At n=30, p=0.5 the 95% interval is about +-0.18."
**Seen in:** `WALKTHROUGH.md` theory

## Contamination (n-gram overlap)  `(concept)`
**1. Definition:** Test items whose n-grams appear in the training text.
**2. In plain words:** The student saw the answers.
**3. Example to quote:** > "Planted items are found; unplanted ones are not."
```python
ngrams(item, 5) & corpus_grams
```
**Seen in:** `main.py`, section 10

## Harness settings that change scores  `(concept)`
**1. Definition:** Prompt format, few-shot count, answer extraction, metric name and version.
**2. In plain words:** Why 'the same benchmark' gives different numbers.
**3. Example to quote:** > "Few-shot count, prompt wording, normalisation and answer extraction each move the score; a number without its settings is not comparable."
**Seen in:** `WALKTHROUGH.md` theory

## np.random.default_rng  `(library)`
**1. Definition:** NumPy's seeded random generator object.
**2. In plain words:** Reproducible randomness.
**3. Example to quote:** > "default_rng(seed) gives a reproducible, independent random generator without touching the global state."
```python
np.random.default_rng(0)
```
**Seen in:** `main.py`, `bootstrap_ci`

## Numerical edge cases (n - c < k)  `(concept)`
**1. Definition:** Boundary conditions where a formula or product behaves wrongly.
**2. In plain words:** Test the edges.
**3. Example to quote:** > "Without the guard the pass@k product has a negative factor."
```python
if n - c < k: return 1.0
```
**Seen in:** `main.py`, `pass_at_k`
