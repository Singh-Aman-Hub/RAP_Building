# Step 11 — End-to-end report card

## The story so far
You now know how to measure: [step 08](../08_evaluation_from_scratch/WALKTHROUGH.md) built the metrics and the confidence intervals, [step 09](../09_llm_as_judge_and_arenas/WALKTHROUGH.md) showed how judges and arenas turn votes into rankings, and [step 10](../10_benchmark_landscape_2026_and_tools/WALKTHROUGH.md) named the benchmarks and tools. You also know how to adapt a model cheaply ([step 03](../03_peft_family/WALKTHROUGH.md)) and how to train it on answers ([step 02](../02_sft_from_scratch_with_loss_masking/WALKTHROUGH.md)). This last step puts the pieces together on one model.

## The pain point
The most common fine-tuning story is: "we fine-tuned it, the target metric went up, ship it." That sentence leaves out three things a reviewer will ask: *is the gain larger than the noise?* *what did we break?* and *how much did the extra training cost in general ability?* A fine-tune that wins the task and quietly forgets everything else is not a win.

## The spark
Treat a fine-tune as an experiment with a **report card**: for every metric, the value before, the value after, the delta, a 95% confidence interval of the *paired* delta, and a verdict computed by a rule (interval entirely above zero = real gain, entirely below = real regression, otherwise within noise). Include **regression probes** that measure what you did *not* mean to change: general knowledge and general-text perplexity.

## Goal
Take `distilgpt2`, fine-tune it with LoRA on 80 sentiment examples (completion-only loss, as in step 02), and evaluate before and after with the step-08 helpers (`evalkit.py`): task accuracy by ranked choice and by generated exact match on 40 held-out reviews, a 30-question general-knowledge MCQ, and perplexity on general text. Run a *gentle* recipe (learning rate 5e-4, 3 epochs) and an *aggressive* one (5e-3, 10 epochs) to see catastrophic forgetting appear in the regression rows.

## The theory (read this before the interview)
**Report card columns:** `metric | before | after | delta | 95% CI of delta | verdict`.

**Paired difference with a bootstrap CI.** The same items are scored by the base model and the fine-tuned model, giving two 0/1 vectors `before` and `after`. Resample item indices with replacement, and for each resample compute `mean(after) - mean(before)`; the 2.5% and 97.5% quantiles are the interval. Pairing cancels the item difficulty shared by both models, so the interval is tighter than comparing two independent intervals ([step 08](../08_evaluation_from_scratch/WALKTHROUGH.md), block 7).

**Verdict rule:** `lo > 0` real gain; `hi < 0` real regression; otherwise *within noise*. With 30 or 40 items, "within noise" is a common and honest verdict.

**What to measure besides the task:**
- *Ranked-choice task accuracy* (does the model rank the right label higher?) and *generated exact match* (does greedy decoding actually start with the right label?). They can differ: the second is what a user sees.
- *General-knowledge MCQ* (30 common-knowledge questions, scored by log-likelihood per byte) as a regression probe.
- *General-text perplexity* as a second probe: it is one number, so no paired interval is computed (a bootstrap over lines would be possible but is not done here).

**Catastrophic forgetting:** training on a narrow task shifts the weights so that earlier capabilities degrade. It grows with learning rate, number of epochs and how far the fine-tuning data is from the pretraining distribution. Mitigations: lower the learning rate, train fewer steps, use PEFT such as LoRA (the base weights stay frozen, and published comparisons report that LoRA often forgets less than full fine-tuning, though it also tends to learn less), and **replay**: mix a slice of general data into the fine-tuning set so the model keeps practising it.

**Ship / hold / roll back:** the decision follows from the card, not from a single number. Gain real and regressions within an agreed tolerance: ship. Gain real but a probe regressed: decide whether the trade is acceptable, mitigate (lower LR, replay) and re-run. Nothing real: hold. A regression found *after* shipping: roll back to the previous model version (see [10_ml_system_architecture](../../10_ml_system_architecture/README.md) for versioning and rollback).

**Caveats you should say out loud:** the test sets are tiny and hand-made; the "general ability" probe is easy trivia, so a small model near chance on it can only get worse by noise; and perplexity on a 60-line toy corpus is not a general-ability measure.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Report card | eval report, scorecard, model card results | near | One table: metric, before, after, delta, CI, verdict |
| Regression probe | regression test, capability retention check, canary eval | near | A benchmark chosen to catch what the change should not alter |
| Catastrophic forgetting | forgetting, alignment tax (related) | near | Loss of earlier abilities after narrow training; "alignment tax" is the same cost seen after preference tuning |
| Replay | data mixing, rehearsal | yes | Mix old/general data into the new training set |
| Paired bootstrap | paired resampling | near | Same-items difference, resampled |
| Ranked-choice accuracy | log-likelihood accuracy, cloze accuracy | near | Which label is more probable; no generation |
| Generated exact match | generative accuracy | near | Greedy output must start with the label |
| Verdict rule | decision rule, significance rule | near | Interval above, below or including zero |
| Held-out set | test set | near | Not used for training |

## Run it
```bash
cd 17_finetuning_alignment_and_benchmarks/11_end_to_end_report_card
python main.py
# from repo root:
HF_HUB_OFFLINE=1 .venv/bin/python 17_finetuning_alignment_and_benchmarks/11_end_to_end_report_card/main.py
# knob: SEED
```
Needs `torch`, `transformers`, `peft` and cached `distilgpt2`. About 1–2 minutes of compute on an idle laptop CPU (two LoRA trainings and three evaluations); varies by machine and load.

## Code, step by step

Lines 1–12 of `main.py` are the header docstring (GOAL / RUN / EXPECTED OUTPUT / NEEDS) already summarised above; the walk starts after it.

### 0. Imports, settings, data  (`main.py` lines 13–33)
```python
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import make_sentiment_data   # noqa: E402
from evalkit import GENERAL_MCQ, bootstrap_ci, mcq_accuracy, paired_bootstrap, perplexity   # noqa: E402

SEED = int(os.environ.get("SEED", 0))
torch.manual_seed(SEED)
tok = AutoTokenizer.from_pretrained("distilgpt2")
tok.pad_token = tok.eos_token
train, test = make_sentiment_data()
```
- **In simple words:** Imports, with the evaluation helpers coming from `evalkit.py`, the finished copy of the step-08 functions, so we do not re-implement them.
- **Exactly:** `ROOT` is the module folder (parent of this step folder); putting it on `sys.path` lets us `import common` and `import evalkit`. `evalkit` exports `GENERAL_MCQ` (the 30 questions of step 08), `mcq_accuracy` (per-item 0/1 array, log-likelihood scored per byte), `perplexity`, `bootstrap_ci` and `paired_bootstrap` (mean difference with a 95% interval). `make_sentiment_data()` gives the same 80 train and 40 test reviews as step 02 (half of the test reviews use adjectives never seen in training).
- **Why it is here:** Reusing the harness is the point: the evaluation code is a library, the fine-tune is the experiment.

### 1. Evaluation items: the task and two regression probes  (`main.py` lines 35–60)
```python
# ## 1. Evaluation items: the task (held-out reviews) and two regression probes (general MCQ, general text)
general_text = [l.strip() for l in (ROOT.parent / "06_llm_training_and_lora" / "data" / "tiny_corpus.txt").read_text().splitlines() if l.strip()]

def task_correct(model):
    """1/0 per held-out review: is log P(' positive') vs log P(' negative') ranked correctly given the SFT prompt format?"""
    from evalkit import cont_logprob
    out = []
    for r in test:
        lp = {w: cont_logprob(model, tok, r["prompt"], " " + w)[0] for w in ("positive", "negative")}
        out.append(int(max(lp, key=lp.get) == r["label"]))
    return np.array(out)

@torch.no_grad()
def task_generated(model):
    """1/0 per held-out review: does greedy decoding START with the right label word? (what a user would see)"""
    out = []
    for r in test:
        enc = tok(r["prompt"], return_tensors="pt")
        g = model.generate(**enc, max_new_tokens=3, do_sample=False, pad_token_id=tok.eos_token_id)
        words = tok.decode(g[0, enc.input_ids.shape[1]:], skip_special_tokens=True).split()
        out.append(int(bool(words) and words[0] == r["label"]))
    return np.array(out)

def measure(model):
    model.eval()
    return {"task": task_correct(model), "gen": task_generated(model), "general": mcq_accuracy(model, tok, GENERAL_MCQ), "ppl": perplexity(model, tok, general_text)}
```
- **In simple words:** Four measurements of any model: does it rank the right sentiment label higher, does it actually *write* the right label, does it still know general facts, and how surprised is it by ordinary text.
- **Exactly:** `general_text` is the corpus of module 06 (`ROOT.parent` is the repo root), used for perplexity on text unrelated to the fine-tuning task. `task_correct` returns a length-40 0/1 `numpy` array (ranked-choice: the label with the larger log-probability; `cont_logprob` returns `(sum log-prob, n_tokens)`, hence the `[0]`). `task_generated` greedy-decodes 3 tokens and compares the first word (the same check as `gen_exact_match` in step 02). `measure` returns a dict: three **per-item arrays** (`task`, `gen`, `general`), needed for paired intervals, and one **scalar** (`ppl`). Per-item arrays keep the item identity, so the same questions can be compared before and after.
- **Why it is here:** The task metric proves the gain; the two probes are the "what did we break" check. Keeping per-item vectors is what makes the paired bootstrap possible.

### 2. Fine-tune with LoRA (completion-only loss, as in step 02)  (`main.py` lines 62–87)
```python
# ## 2. Fine-tune with LoRA (completion-only loss, as in step 02)
def lora_sft(lr, epochs):
    torch.manual_seed(SEED)
    base = AutoModelForCausalLM.from_pretrained("distilgpt2")
    cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"], fan_in_fan_out=True, task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    rng = random.Random(SEED)
    for _ in range(epochs):
        model.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        for s in range(0, len(order), 8):
            rows = [train[i] for i in order[s:s + 8]]
            ids, labs = [], []
            for r in rows:
                p, c = tok(r["prompt"]).input_ids, tok(r["completion"]).input_ids + [tok.eos_token_id]
                ids.append(p + c); labs.append([-100] * len(p) + c)
            L = max(map(len, ids))
            x = torch.tensor([i + [tok.pad_token_id] * (L - len(i)) for i in ids])
            y = torch.tensor([l + [-100] * (L - len(l)) for l in labs])
            m = torch.tensor([[1] * len(i) + [0] * (L - len(i)) for i in ids])
            logits = model(input_ids=x, attention_mask=m).logits
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1), ignore_index=-100)
            opt.zero_grad(); loss.backward(); opt.step()
    return model
```
- **In simple words:** Step 02's SFT loop again (masked loss, padding, shift by one), but the model only trains a small LoRA adapter (r=8) instead of all weights.
- **Exactly:** `get_peft_model` freezes distilgpt2 and adds `A`, `B` matrices to `c_attn` (147,456 trainable parameters, see [step 03](../03_peft_family/WALKTHROUGH.md); derivation in [06 step 04](../../06_llm_training_and_lora/04_lora_math_numpy/WALKTHROUGH.md)). The optimiser receives only the parameters with `requires_grad=True`. The batch is built inline with plain lists: `ids` (prompt + completion + EOS), `labs` (the same with `-100` on the prompt), right-padded to `L` with the pad id / `-100` / 0 for ids / labels / attention mask. The loss is exactly step 02's `sft_loss`: logits `[:, :-1]` against labels `[:, 1:]`, `ignore_index=-100`. `lr` and `epochs` are the only difference between the two recipes. There is no gradient clipping here (step 02 had it).
- **Why it is here:** It is the training half of the experiment, kept short and readable. The learning rate is high for LoRA by design: LoRA tolerates much larger rates than full fine-tuning, and 5e-3 for 10 epochs is deliberately too much.

### 3. The report card  (`main.py` lines 89–99)
```python
# ## 3. The report card
def card(title, before, after):
    print(f"\n=== {title} ===")
    print(f"{'metric':<28}{'before':>8}{'after':>8}{'delta':>8}   {'95% CI of delta':<18}{'verdict'}")
    for name, key in (("task ranked-choice (n=40)", "task"), ("task generated EM (n=40)", "gen"), ("general MCQ acc (n=30)", "general")):
        d, (lo, hi) = paired_bootstrap(after[key], before[key])
        verdict = "real gain" if lo > 0 else "real regression" if hi < 0 else "within noise"
        print(f"{name:<28}{before[key].mean():>8.3f}{after[key].mean():>8.3f}{d:>+8.3f}   [{lo:+.2f}, {hi:+.2f}]{'':<5}{verdict}")
    print(f"{'general-text perplexity':<28}{before['ppl']:>8.1f}{after['ppl']:>8.1f}{after['ppl'] - before['ppl']:>+8.1f}   (lower is better; no CI: one number)")
    lo, hi = bootstrap_ci(after["gen"])
    print(f"generated exact-match after, bootstrap 95% CI: [{lo:.2f}, {hi:.2f}]")
```
- **In simple words:** Print one table row per metric: before, after, the change, how uncertain the change is, and a plain-words verdict.
- **Exactly:** For each per-item metric, `paired_bootstrap(after, before)` returns the mean difference `d` and a 95% interval `(lo, hi)` (4,000 resamples, seeded). The verdict is a nested conditional expression: `lo > 0` real gain; else `hi < 0` real regression; else within noise. Perplexity is a single number, so only the difference is printed. Last line: a bootstrap interval of the *absolute* generated exact-match after training. Reading the interval of the delta against zero, not against the other model's interval, is the whole point.
- **Why it is here:** This function turns a pile of numbers into a decision, with the uncertainty visible.

### 4. Run: before, gentle recipe, aggressive recipe  (`main.py` lines 101–108)
```python
base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
before = measure(base_model)
print(f"BEFORE: task {before['task'].mean():.3f} generated {before['gen'].mean():.3f}  general {before['general'].mean():.3f}  ppl {before['ppl']:.1f}")

gentle = lora_sft(lr=5e-4, epochs=3)
card("gentle LoRA: lr 5e-4, 3 epochs", before, measure(gentle))
aggressive = lora_sft(lr=5e-3, epochs=10)
card("aggressive LoRA: lr 5e-3, 10 epochs", before, measure(aggressive))
```
- **In simple words:** Measure the untouched model once, then train two LoRA variants (a gentle and an aggressive one) and print a report card for each against the *same* "before".
- **Exactly:** `measure(base_model)` runs all four measurements once and is reused as the baseline for both cards (so both are paired against identical per-item vectors). `lora_sft` returns a `PeftModel`; `measure` puts it in `eval()` mode. Each `lora_sft` call starts again from fresh distilgpt2 weights with the same seed, so the two recipes differ only in learning rate and epochs.
- **Why it is here:** Two recipes on one card pair is the smallest experiment that shows a trade-off.

### 5. TRY IT / BREAK IT comments  (`main.py` lines 110–111)
```python
# TRY IT: add a third recipe with r=2 or with lr=1e-4; add the "mix in general text" replay idea from the README and re-measure.
# BREAK IT: evaluate on the TRAIN reviews instead of the test reviews. Compare. That is why held-out data exists.
```
- **In simple words:** Experiments; see below.
- **Exactly:** Comments only.
- **Why it is here:** Module convention.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `from evalkit import ...` inside a function | a local import: it runs at call time (here `cont_logprob`); legal, slightly unusual |
| `"a" if c1 else "b" if c2 else "c"` | chained conditional expression |
| `max(map(len, ids))` | length of the longest list |
| `{x:>+8.3f}` | right-aligned, explicit sign, 8 wide, 3 decimals |
| `d, (lo, hi) = f(...)` | unpack a float and a 2-tuple returned together |
| `opt.zero_grad(); loss.backward(); opt.step()` | three statements on one line |
| `[p for p in model.parameters() if p.requires_grad]` | only the trainable tensors (the adapter) |

## Libraries & functions
| Function | What it does | Internally |
|---|---|---|
| `paired_bootstrap(after, before)` (`evalkit.py`) | mean difference and 95% CI on the same items | draws one `(4000, n)` matrix of item indices, computes the mean difference per resample, takes quantiles |
| `bootstrap_ci(values)` (`evalkit.py`) | 95% CI of a mean | resamples with replacement, takes the 2.5% and 97.5% quantiles |
| `mcq_accuracy(model, tok, items)` (`evalkit.py`) | 0/1 per question | scores each choice by `log P(choice | question)` divided by the choice's byte length |
| `perplexity(model, tok, texts)` (`evalkit.py`) | `exp(mean token CE)` over the texts | per-line mean losses are re-weighted by token count |
| `get_peft_model` | LoRA wrapper | freezes the base, adds `A` and `B` to `c_attn` |
| `cont_logprob(model, tok, ctx, cont)` | log-probability of a continuation | one forward pass, gather the true next-token log-probabilities |

## Execution trace (real run)
```
[transformers] `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`.
BEFORE: task 0.875 generated 0.000  general 0.667  ppl 99.7


=== gentle LoRA: lr 5e-4, 3 epochs ===
metric                        before   after   delta   95% CI of delta   verdict
task ranked-choice (n=40)      0.875   1.000  +0.125   [+0.03, +0.22]     real gain
task generated EM (n=40)       0.000   1.000  +1.000   [+1.00, +1.00]     real gain
general MCQ acc (n=30)         0.667   0.633  -0.034   [-0.10, +0.00]     within noise
general-text perplexity         99.7   230.6  +130.8   (lower is better; no CI: one number)
generated exact-match after, bootstrap 95% CI: [1.00, 1.00]


=== aggressive LoRA: lr 5e-3, 10 epochs ===
metric                        before   after   delta   95% CI of delta   verdict
task ranked-choice (n=40)      0.875   1.000  +0.125   [+0.03, +0.22]     real gain
task generated EM (n=40)       0.000   1.000  +1.000   [+1.00, +1.00]     real gain
general MCQ acc (n=30)         0.667   0.433  -0.232   [-0.40, -0.10]     real regression
general-text perplexity         99.716662076.0+16661976.3   (lower is better; no CI: one number)
generated exact-match after, bootstrap 95% CI: [1.00, 1.00]
```
(Progress bars omitted; the `transformers` `loss_type` note is real stderr.) A formatting accident in the last line of the aggressive card: the three perplexity numbers are wider than the 8-character columns and run together. Read it as before **99.7**, after **16,662,076.0**, delta **+16,661,976.3**.
- **Baseline**: ranked-choice 0.875 but *generated* exact match 0.000: the base model knows the labels but does not write them (same finding as [step 02](../02_sft_from_scratch_with_loss_masking/WALKTHROUGH.md)). General MCQ 0.667, perplexity 99.7.
- **Gentle recipe**: ranked-choice 0.875 to 1.000 (+0.125, CI [+0.03, +0.22], real gain) and generated EM 0.000 to 1.000 (CI [+1.00, +1.00]: every resample gives +1, so the interval has zero width). General MCQ 0.667 to 0.633 (-0.034, CI [-0.10, +0.00]): the verdict is "within noise" because the upper bound is exactly 0.00, not above it, but it is close to the edge. Perplexity rises from 99.7 to 230.6: the model became more than twice as "surprised" by general text. That is a cost the task metrics do not show.
- **Aggressive recipe**: the same task gains (1.000 and 1.000), but the general MCQ falls to 0.433 (-0.232, CI [-0.40, -0.10], **real regression**) and the perplexity explodes to about 16.7 million: the model now assigns almost no probability to ordinary text. This is catastrophic forgetting in numbers, achieved by raising the learning rate 10x and training 3x longer.
- **Same task win, opposite verdict**: both recipes tie on the target metric; only the regression probes tell them apart. That is why the card has more rows than the task metric.


## Experiments (change ONE thing)
| Change | What you will see | Why |
|---|---|---|
| (measured) gentle (5e-4, 3 epochs) | task +0.125 / +1.000, general MCQ within noise, perplexity 99.7 to 230.6 | small step, mild cost |
| (measured) aggressive (5e-3, 10 epochs) | same task gain, general MCQ -0.232 (real regression), perplexity about 1.7e7 | forgetting from a large, long update |
| (not run) a third recipe with `lr=1e-4` | expected a smaller perplexity rise, and possibly less task gain | less movement from the base |
| (not run) replay: mix general lines into the training set | expected to reduce the perplexity rise | the model keeps practising general text |

## Debugging practice
**BREAK IT (the file's own suggestion, real result):** evaluate the gentle model on 40 TRAIN reviews instead of the 40 test reviews. Real output of that check:
```
task ranked-choice on TEST : 1.0
task ranked-choice on TRAIN: 1.0
```
How to read it: here the test set is already at ceiling (1.000), so the train/test comparison cannot show a gap, and the lesson of this BREAK IT does not materialise numerically. Held-out data matters when the model can overfit; always report it anyway, and prefer a test set hard enough not to saturate.

**A bug I actually hit while checking this:** calling the evaluation functions on the model returned by `lora_sft` *without* `model.eval()` gave ranked-choice accuracy **0.775** instead of 1.000 on the same test reviews: `lora_sft` leaves the model in `train()` mode, so dropout was still active and the scores were noisy and lower. `measure()` calls `model.eval()` first, which is why the card is correct. Symptom: a trained model that looks worse than the numbers in the card, and different numbers on each call. Fix: `model.eval()` before every evaluation.

## Common mistakes
- Reporting only the target metric and skipping regression probes.
- Reading a delta without its interval: with `n = 40`, a change of 5 points is within noise.
- Comparing two separate intervals instead of the paired difference.
- Evaluating on training data (or on test items tuned against), then reporting the result as generalisation.
- Fine-tuning with an aggressive learning rate or too many epochs on a tiny dataset and calling the narrow gain a success.
- Treating a probe at or near chance as meaningful: a model that is already near chance on the general MCQ can only wobble.
- Forgetting that a ranked-choice score and a generated score can disagree.
- Shipping without a rollback plan.

## Rebuild checklist
1. `task_correct` (ranked choice) and `task_generated` (greedy exact match) returning per-item 0/1 arrays.
2. `measure(model)` returning those, a general-MCQ array and a perplexity scalar.
3. `lora_sft(lr, epochs)`: LoRA config, masked labels, padded batches, shifted cross-entropy.
4. `card(title, before, after)`: paired bootstrap delta, interval, verdict rule.
5. Measure the base once; run at least two recipes against that same baseline.

## Say it aloud
"I treat a fine-tune as an experiment and report a card: for each metric the before, the after, the delta, a paired bootstrap interval of the delta and a verdict by whether the interval clears zero. I include regression probes, general knowledge and general-text perplexity, because a gain on the target task can hide forgetting; a gentle LoRA recipe and an aggressive one make that trade-off visible. With forty test items I expect wide intervals, so I only call a result real when the interval excludes zero, and I keep the previous model version for rollback."

## The new problem (cliffhanger)
This closes module 17: you can adapt, align and measure a model. The next question is not about the model but the system around it. Can it be served within a latency and cost budget, how do you version and roll it back, and how do you monitor it in production? Continue with [10_ml_system_architecture](../../10_ml_system_architecture/README.md) and [09_api_development](../../09_api_development/README.md); for the serving-engine side see [21_llm_inference_engineering](../../21_llm_inference_engineering/INTERVIEW_GIST.md). Back to the [module README](../README.md).
