# Terms — End-to-end report card

Folder: `17_finetuning_alignment_and_benchmarks/11_end_to_end_report_card` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [17 README](../README.md)

**Terms in this folder:** [Report card](#report-card) · [Paired difference with bootstrap CI](#paired-difference-with-bootstrap-ci) · [Verdict rule](#verdict-rule) · [Regression probe](#regression-probe) · [Catastrophic forgetting](#catastrophic-forgetting) · [Replay (data mixing)](#replay-data-mixing) · [Gentle vs aggressive recipe](#gentle-vs-aggressive-recipe) · [Generated exact match](#generated-exact-match) · [Ranked-choice accuracy](#ranked-choice-accuracy) · [Held-out evaluation](#held-out-evaluation) · [Ship / hold / roll back](#ship--hold--roll-back) · [LoRA for lower forgetting](#lora-for-lower-forgetting)

---

## Report card  `(concept)`
**1. Definition:** A table of metric, before, after, delta, confidence interval and verdict.
**2. In plain words:** The one page that justifies a decision.
**3. Example to quote:** > "A report card lists metric, before, after, delta and a confidence interval, plus the regression probes."
**Seen in:** `main.py`, `card`

## Paired difference with bootstrap CI  `(concept)`
**1. Definition:** Mean of per-item (after - before) with a percentile interval over resampled items.
**2. In plain words:** Is the gain real?
**3. Example to quote:** > "I bootstrap the per-item difference after minus before, so an interval that includes 0 means within noise."
```python
paired_bootstrap(after, before)
```
**Seen in:** `evalkit.py`

## Verdict rule  `(concept)`
**1. Definition:** Gain if the interval is above 0, regression if below, otherwise within noise.
**2. In plain words:** A decision rule on intervals.
**3. Example to quote:** > "If the interval's lower bound is above 0 it is a real gain; if its upper bound is below 0 it is a real regression; otherwise within noise."
```python
'real gain' if lo > 0 else ...
```
**Seen in:** `main.py`, `card`

## Regression probe  `(concept)`
**1. Definition:** A test of abilities you did not train, to catch collateral damage.
**2. In plain words:** Check what should not change.
**3. Example to quote:** > "General MCQ (30 items) and perplexity on general text."
**Seen in:** `main.py`, `measure`

## Catastrophic forgetting  `(concept)`
**1. Definition:** Fine-tuning on a narrow task erodes unrelated capabilities.
**2. In plain words:** Learning one trick, losing others.
**3. Example to quote:** > "The aggressive recipe won the task and lost general knowledge: catastrophic forgetting."
**Seen in:** `WALKTHROUGH.md` theory

## Replay (data mixing)  `(concept)`
**1. Definition:** Mixing general data into the fine-tuning set to preserve old skills.
**2. In plain words:** Rehearse the old lessons while learning the new one.
**3. Example to quote:** > "I mix a slice of general text into the fine-tuning data so the model keeps practising what I do not want it to forget."
**Seen in:** `WALKTHROUGH.md` theory

## Gentle vs aggressive recipe  `(concept)`
**1. Definition:** Low LR and few epochs vs high LR and many.
**2. In plain words:** Careful training vs hard training.
**3. Example to quote:** > "lr 5e-4 for 3 epochs versus lr 5e-3 for 10 epochs."
```python
lora_sft(lr=5e-3, epochs=10)
```
**Seen in:** `main.py`, section 3

## Generated exact match  `(concept)`
**1. Definition:** Greedy-decode and check that the first word equals the label.
**2. In plain words:** What the user sees.
**3. Example to quote:** > "Generated exact match checks what a user would actually see: greedy decoding must start with the right label."
```python
words[0] == r['label']
```
**Seen in:** `main.py`, `task_generated`

## Ranked-choice accuracy  `(concept)`
**1. Definition:** Compare log-probabilities of the candidate labels.
**2. In plain words:** What the model prefers.
**3. Example to quote:** > "Ranked-choice accuracy asks which of two labels the model finds more likely, without ever generating text."
```python
max(lp, key=lp.get)
```
**Seen in:** `main.py`, `task_correct`

## Held-out evaluation  `(concept)`
**1. Definition:** Test data never used in training or tuning.
**2. In plain words:** A fair exam.
**3. Example to quote:** > "Evaluating on training data makes memorisation look like skill."
**Seen in:** `main.py`, `make_sentiment_data`

## Ship / hold / roll back  `(concept)`
**1. Definition:** Decision outcomes from the report card.
**2. In plain words:** Release, wait for more evidence, or undo.
**3. Example to quote:** > "The decision follows the card: ship if the task gain is real and the regressions are within tolerance, otherwise hold or roll back."
**Seen in:** `WALKTHROUGH.md` theory

## LoRA for lower forgetting  `(concept)`
**1. Definition:** Training adapters leaves base weights unchanged, so the base can always be restored by disabling the adapter.
**2. In plain words:** An undo button.
**3. Example to quote:** > "LoRA often forgets less than full fine-tuning because the base weights stay frozen, but an aggressive learning rate can still hurt."
```python
with model.disable_adapter(): ...
```
**Seen in:** `main.py`, `lora_sft`
