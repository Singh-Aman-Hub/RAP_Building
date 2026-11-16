# Step 07 — Distillation and model merging

## The story so far
[Step 06](../06_ppo_vs_grpo_rlvr/WALKTHROUGH.md) used RL to make one policy better at one skill. Two practical questions come right after training: how do you get a *small, cheap* model to inherit a big model's behaviour, and how do you combine several fine-tunes (one for sentiment, one for polite replies, one for code) without training a new model from scratch?

## The pain point
Serving a large model for every request is expensive; a small model trained from scratch on hard labels is usually worse than it needs to be. And once you have several task-specific fine-tunes (full models or [LoRA adapters](../03_peft_family/WALKTHROUGH.md)), keeping a separate model per task multiplies storage and serving cost, while retraining a joint model needs all the data again.

## The spark
Two cheap tricks, both outside the usual "minimise a loss on labels" loop:
1. **Distillation:** train the student to match the teacher's *whole probability distribution* (soft targets), which carries far more information than the single correct token.
2. **Merging:** treat each fine-tune as a vector in weight space (`fine-tuned - base`) and combine those vectors by arithmetic (average, add, interpolate on a sphere, or resolve sign conflicts as in TIES), with no training at all.

## Goal
(A) Distill `gpt2` (teacher) into `distilgpt2` (student) on a tiny corpus with a KL loss at temperature `T`, comparing hard-label training, soft targets at `T=1`, soft targets at `T=2`, and a mix. (B) Write four merging recipes (linear average, task arithmetic, SLERP, TIES) by hand and compare them on a toy where two task vectors conflict on 150 of 1,000 coordinates. (C) Train two real LoRA adapters on two different tasks and merge them with `peft` (linear and TIES), measuring the loss of each on both tasks.

## The theory (read this before the interview)
**Knowledge distillation (Hinton et al. 2015).** The teacher's output distribution over tokens at temperature `T` is `p_T = softmax(z_teacher / T)`; the student's is `q_T = softmax(z_student / T)`. The loss is
```
L = alpha * T^2 * KL( p_T || q_T )  +  (1 - alpha) * CE(student, hard label)
KL(p || q) = sum_v p_v * ( log p_v - log q_v )
```
- **Soft targets / dark knowledge:** besides "the next token is `cat`" the teacher says "`dog` is plausible, `carburettor` is not". That structure is signal the hard label throws away.
- **Temperature** flattens the distribution (`T > 1`) so the small probabilities are visible to the student. Print the entropy at `T=1` and `T=2` to see it.
- **Why `T^2`:** dividing logits by `T` shrinks the gradients of the soft term by about `1 / T^2`; multiplying by `T^2` keeps the two terms comparable when you change `T`.
- **Direction:** `KL(teacher || student)` ("forward KL") makes the student cover everything the teacher considers plausible. Variants minimise the reverse KL or sample from the student (on-policy distillation); this step uses the classic forward KL on teacher logits over a fixed corpus.
- **Logit distillation vs sequence-level distillation:** matching logits needs the teacher's weights and the same tokenizer (gpt2 and distilgpt2 share one). If you only have the teacher's *text outputs* (an API), you do sequence-level distillation: SFT on the teacher's generated answers, as in [step 02](../02_sft_from_scratch_with_loss_masking/WALKTHROUGH.md).

**Model merging.** Define the *task vector* `tau = theta_finetuned - theta_base`. Recipes:
| Recipe | Formula | Idea |
|---|---|---|
| Linear average ("model soup" of same-base fine-tunes) | `theta_base + (tau_1 + tau_2) / 2` | halve each change |
| Task arithmetic | `theta_base + lambda * (tau_1 + tau_2)` | add task vectors (lambda around 1, tuned on a validation set) |
| SLERP | spherical interpolation between two weight vectors | keeps the norm along the path instead of shrinking it like a straight average |
| TIES | trim small entries, elect a sign per coordinate, average only agreeing entries | removes interference between tasks |
(DARE, which randomly drops entries of the task vector and rescales, is a related method that is *not* implemented here.)

**Why merging can work at all:** fine-tunes of the *same pretrained base* usually stay in one basin of the loss landscape, so interpolating between them stays low-loss. It is a heuristic with no guarantee; models fine-tuned from *different* bases generally cannot be merged this way.

**Merging LoRA adapters:** adapters trained on the same base can be combined as weighted sums. In `peft`, `add_weighted_adapter(adapters, weights, adapter_name, combination_type=...)` supports types such as `linear`, `cat`, `svd`, `ties`, `dare_ties`. As implemented in `peft`, `linear` combines the A matrices and B matrices separately and requires the same rank, so the merged product contains cross terms and is not exactly the average of the two weight updates; check the current docs for the exact definition of each type before relying on it.

**Practice (stated generally, Nov 2026):** distillation is how many small open models and "mini" variants are produced; merging is widely used in the open-weights community (often with tools such as mergekit) and sometimes as an ingredient in lab pipelines. Merging results are empirical: always evaluate the merged model on every task ([step 11](../11_end_to_end_report_card/WALKTHROUGH.md) gives the report-card pattern).

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Knowledge distillation | KD, teacher-student training | yes | Student matches the teacher's distribution |
| Soft targets | soft labels, teacher probabilities | yes | The full softmax output instead of a one-hot label |
| Dark knowledge | relative probabilities of wrong classes | near | The information in the non-argmax probabilities |
| Temperature | `T` | no (here) | Not the sampling temperature of [05_llm_parameters](../../05_llm_parameters/README.md), but the same operation `softmax(z / T)` used for training |
| Logit distillation vs sequence distillation | white-box vs black-box distillation | no | Logits need the weights; sequences only need the teacher's text |
| Task vector | delta weights, fine-tune delta | yes | `theta_ft - theta_base` |
| Task arithmetic | adding/subtracting task vectors | yes | Merge by vector addition (negation removes a skill) |
| Model soup | weight averaging | near | Average of several fine-tunes of one base |
| TIES | TIES-merging | yes | Trim, elect sign, disjoint merge |
| SLERP | spherical linear interpolation | yes | Interpolate along the arc between two vectors |
| LoRA merging | adapter merging, `add_weighted_adapter` | no | Combining adapters, versus folding one adapter into the base (`merge_and_unload`) |

## Run it
```bash
cd 17_finetuning_alignment_and_benchmarks/07_distillation_and_merging
python main.py
# from repo root:
HF_HUB_OFFLINE=1 .venv/bin/python 17_finetuning_alignment_and_benchmarks/07_distillation_and_merging/main.py
# knobs: SEED, TEMP (2.0), STEPS (24)
```
It reads `06_llm_training_and_lora/data/tiny_corpus.txt`. About 1–2 minutes of compute on an idle laptop CPU (four distillation trainings of 24 steps plus two adapter trainings); varies by machine and load.

## Code, step by step

Lines 1–13 of `main.py` are the header docstring (GOAL / RUN / EXPECTED OUTPUT / NEEDS) already summarised above; the walk starts after it.

### 0. Imports and settings  (`main.py` lines 14–31)
```python
import copy
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import make_preference_data, make_sentiment_data   # noqa: E402

SEED = int(os.environ.get("SEED", 0))
TEMP = float(os.environ.get("TEMP", 2.0))
STEPS = int(os.environ.get("STEPS", 24))
torch.manual_seed(SEED)
```
- **In simple words:** Imports and three knobs: the distillation temperature, the number of training steps, and the seed.
- **Exactly:** `make_sentiment_data` and `make_preference_data` supply the two *tasks* for the adapter-merging part. `copy` and `math` are imported but not used in this file (harmless leftovers).
- **Why it is here:** `TEMP` is the main distillation hyper-parameter; `STEPS=24` keeps the demo fast.

### 1. Data: tiny corpus split into train and held-out lines  (`main.py` lines 33–42)
```python
# ## 1. Data: tiny corpus split into train and held-out lines
CORPUS = Path(__file__).resolve().parents[2] / "06_llm_training_and_lora" / "data" / "tiny_corpus.txt"
lines = [l.strip() for l in CORPUS.read_text().splitlines() if l.strip()]
train_lines, test_lines = lines[:30], lines[30:]
tok = AutoTokenizer.from_pretrained("gpt2")                # gpt2 and distilgpt2 share one tokenizer (50257 tokens)
tok.pad_token = tok.eos_token

def batch(texts):
    enc = tok(texts, return_tensors="pt", padding=True)    # input_ids, attention_mask: (B, L)
    return enc.input_ids, enc.attention_mask
```
- **In simple words:** Read the small text file from module 06, use the first 30 non-empty lines for training and the rest for testing, and define a helper that turns lines into padded tensors.
- **Exactly:** `parents[2]` from this file is the repo root, so the path points at `06_llm_training_and_lora/data/tiny_corpus.txt` (the module-06 data folder). Teacher and student can share one tokenizer because both use GPT-2's 50,257-token vocabulary; this is *required* for logit distillation, since the two distributions must be over the same tokens. `batch` returns `(ids, mask)`, both `(B, L)`; `padding=True` pads to the longest line (right padding by default for GPT-2).
- **Why it is here:** A fixed held-out split means "student quality" is a number on unseen text, as in [06 step 03](../../06_llm_training_and_lora/03_eval_perplexity_checkpoints/WALKTHROUGH.md).

### 2. Teacher and the soft-target idea  (`main.py` lines 44–53)
```python
# ## 2. Teacher and the soft-target idea
teacher = AutoModelForCausalLM.from_pretrained("gpt2").eval()
x, m = batch(test_lines)
with torch.no_grad():
    tl = teacher(x, attention_mask=m).logits[:, :-1]       # (B, L-1, vocab)
def entropy(logits, T):
    p = F.softmax(logits / T, -1)
    return -(p * torch.log(p + 1e-12)).sum(-1)             # (B, L-1)
mask = m[:, 1:].bool()
print(f"teacher mean entropy of next-token distribution: T=1 {entropy(tl, 1.0)[mask].mean():.2f} nats | T={TEMP} {entropy(tl, TEMP)[mask].mean():.2f} nats")
```
- **In simple words:** Run the teacher once on the held-out lines, keep its scores, and measure how "spread out" its predictions are at two temperatures.
- **Exactly:** `teacher(...).logits` is `(B, L, vocab)`; `[:, :-1]` drops the last position because it predicts a token beyond the text, leaving `(B, L-1, vocab)` aligned with the targets `x[:, 1:]`. `entropy(logits, T)` is `-sum p log p` of `softmax(logits / T)` (the `1e-12` avoids `log 0`), shape `(B, L-1)`. `mask = m[:, 1:].bool()` marks real (non-pad) target positions, and `[mask]` selects them before averaging. Entropy is in nats (natural log).
- **Why it is here:** It shows the effect of temperature as a number: at `T=2` the teacher's distribution is flatter (higher entropy), exposing the small probabilities the student will learn from.

### 3. The distillation loss  (`main.py` lines 55–64)
```python
# ## 3. The distillation loss
def kd_loss(student_logits, teacher_logits, labels, mask, T, alpha):
    """alpha * T^2 * KL(teacher_T || student_T) + (1 - alpha) * hard cross-entropy."""
    s = F.log_softmax(student_logits / T, -1)
    t = F.softmax(teacher_logits / T, -1)
    kl = (t * (torch.log(t + 1e-12) - s)).sum(-1)          # (B, L-1) KL per position
    kl = (kl * mask).sum() / mask.sum()
    ce = F.cross_entropy(student_logits.reshape(-1, student_logits.size(-1)), labels.reshape(-1), reduction="none")
    ce = (ce * mask.reshape(-1)).sum() / mask.sum()
    return alpha * (T ** 2) * kl + (1 - alpha) * ce       # T^2 keeps gradient magnitude comparable across temperatures
```
- **In simple words:** The student is graded on two things: how close its softened distribution is to the teacher's, and (optionally) how well it predicts the real next token.
- **Exactly:** `s` = student log-probabilities at temperature `T`; `t` = teacher probabilities at `T`. The per-position KL is `sum_v t_v (log t_v - s_v)`, shape `(B, L-1)`; the average over real positions uses the float `mask`. `F.cross_entropy(..., reduction="none")` returns one loss per token `(B*(L-1),)`, masked and averaged the same way (pads must not count). `alpha` mixes the soft term (weight `alpha * T^2`) and the hard term (`1 - alpha`). With `alpha=0` the loss is plain cross-entropy (the baseline); `alpha=1` is pure distillation.
- **Why it is here:** This one function is distillation; the four runs below differ only in `T` and `alpha`.

### 4. Evaluate a student, then four distillation runs  (`main.py` lines 66–98)
```python
# ## 4. Evaluate a student: held-out CE and KL to the teacher at T=1
@torch.no_grad()
def evaluate(student):
    student.eval()
    sl = student(x, attention_mask=m).logits[:, :-1]
    ce = F.cross_entropy(sl.reshape(-1, sl.size(-1)), x[:, 1:].reshape(-1), reduction="none")
    ce = (ce * mask.reshape(-1)).sum() / mask.sum()
    kl = (F.softmax(tl, -1) * (F.log_softmax(tl, -1) - F.log_softmax(sl, -1))).sum(-1)[mask].mean()
    return ce.item(), kl.item()

def distill(T, alpha, label):
    torch.manual_seed(SEED)
    student = AutoModelForCausalLM.from_pretrained("distilgpt2")
    opt = torch.optim.AdamW(student.parameters(), lr=2e-5)
    rng = random.Random(SEED)
    for step in range(STEPS):
        texts = rng.sample(train_lines, 6)
        xb, mb = batch(texts)
        with torch.no_grad():
            tlogits = teacher(xb, attention_mask=mb).logits[:, :-1]
        student.train()
        slogits = student(xb, attention_mask=mb).logits[:, :-1]
        loss = kd_loss(slogits, tlogits, xb[:, 1:], mb[:, 1:].float(), T, alpha)
        opt.zero_grad(); loss.backward(); opt.step()
    ce, kl = evaluate(student)
    print(f"{label:<34} held-out CE {ce:.3f}  KL(teacher||student) {kl:.3f}")

print(f"\nstudent before training: held-out CE/KL = {tuple(round(v, 3) for v in evaluate(AutoModelForCausalLM.from_pretrained('distilgpt2')))}")
print(f"teacher itself:          held-out CE = {evaluate(teacher)[0]:.3f}")
distill(1.0, 0.0, "hard labels only (alpha=0)")
distill(1.0, 1.0, "soft targets T=1 (alpha=1)")
distill(TEMP, 1.0, f"soft targets T={TEMP} (alpha=1)")
distill(TEMP, 0.5, f"mixed T={TEMP} (alpha=0.5)")
```
- **In simple words:** Define two held-out numbers (cross-entropy on the real next tokens, and KL to the teacher at `T=1`), then train four fresh students for 24 steps with different loss settings and print both numbers for each.
- **Exactly:** `evaluate` returns `(held-out cross-entropy in nats, mean KL(teacher || student) at T=1)`, both averaged over real positions. `distill` loads a fresh `distilgpt2` (same seed each time, so students start identical), samples 6 lines per step, runs the *teacher* under `no_grad` (a forward-only cost; the teacher is never updated), the student with gradients, and takes one AdamW step with learning rate 2e-5. The `;`-chained line is legal Python: `zero_grad`, `backward`, `step`. The reference lines print the untrained student and the teacher's own cross-entropy.
- **Why it is here:** It gives a controlled comparison: same data, same steps, same start; only the loss differs. Held-out CE says how well the student predicts text; KL-to-teacher says how closely it imitates the teacher.

### 5a. Merging recipes: SLERP and TIES  (`main.py` lines 100–117)
```python
# ## 5. Merging on small tensors: four recipes
def slerp(a, b, t):
    """Spherical interpolation between two flat vectors a and b."""
    an, bn = a / a.norm(), b / b.norm()
    omega = torch.acos(torch.clamp((an * bn).sum(), -1 + 1e-7, 1 - 1e-7))      # angle between them
    return (torch.sin((1 - t) * omega) * a + torch.sin(t * omega) * b) / torch.sin(omega)

def ties(taus, density=0.5):
    """TIES: trim small values per task, elect a sign per coordinate, average only agreeing entries."""
    trimmed = []
    for t in taus:
        k = max(1, int(len(t) * density))
        thr = t.abs().topk(k).values.min()                                    # keep the top-`density` by magnitude
        trimmed.append(torch.where(t.abs() >= thr, t, torch.zeros_like(t)))
    stack = torch.stack(trimmed)                                              # (n_tasks, D)
    sign = torch.sign(stack.sum(0))                                           # elected sign per coordinate (by total mass)
    agree = (torch.sign(stack) == sign) & (stack != 0)                        # entries that agree with the elected sign
    return (stack * agree).sum(0) / agree.sum(0).clamp(min=1)                 # disjoint mean over agreeing entries
```
- **In simple words:** SLERP moves along the arc between two vectors instead of the straight line. TIES keeps only each task's biggest changes, lets the tasks "vote" on the direction of each weight, and averages only the votes that agree.
- **Exactly:** *SLERP:* `omega` is the angle between the two directions (`acos` of the dot product of the unit vectors; the clamp avoids `acos` of exactly +-1); the result is `(sin((1-t) omega) a + sin(t omega) b) / sin(omega)`. With `t=0.5` it is the midpoint along the arc, and unlike a straight average its norm does not shrink for vectors with a large angle between them. *TIES:* for each task vector keep the `k = density * D` entries of largest magnitude (`topk(k).values.min()` is the threshold; ties at the threshold are kept); stack to `(n_tasks, D)`; the elected sign is `sign(sum over tasks)` (the sign with more total mass); an entry "agrees" if it is non-zero and has the elected sign; the merged value is the sum of agreeing entries divided by the *count* of agreeing entries (`clamp(min=1)` avoids 0/0). This is the "disjoint mean". The published TIES method has the same three steps (trim, elect sign, disjoint merge); details such as how the mass is computed are as written here.
- **Why it is here:** These are the two merge recipes that go beyond averaging; both are short enough to read in full.

### 5b. The conflicting-task-vectors toy  (`main.py` lines 119–135)
```python
D = 1000
base = torch.zeros(D)
tau1 = torch.randn(D) * (torch.rand(D) < 0.3)                                 # task 1 edits ~30% of the weights
tau2 = torch.randn(D) * (torch.rand(D) < 0.3)                                 # task 2 edits a different ~30%
tau2[:150] = -tau1[:150] + 0.05 * torch.randn(150)                            # ...and conflicts with task 1 on 150 coordinates
cos = lambda a, b: F.cosine_similarity(a, b, dim=0).item()
print("\nmerge toy: D=1000, tau1/tau2 conflict on the first 150 coordinates")
print(f"{'method':<28}{'cos(merged,tau1)':>18}{'cos(merged,tau2)':>18}")
for name, delta in {
    "linear average":          (tau1 + tau2) / 2,
    "task arithmetic (lam=1)": tau1 + tau2,
    "SLERP t=0.5":             slerp(tau1, tau2, 0.5),
    "TIES density 0.5":        ties([tau1, tau2], 0.5),
}.items():
    merged = base + delta
    print(f"{name:<28}{cos(merged, tau1):>18.3f}{cos(merged, tau2):>18.3f}")
print(f"{'(reference) cos(tau1,tau2)':<28}{cos(tau1, tau2):>18.3f}")
```
- **In simple words:** Invent two "fine-tunes" of a base of all zeros in a 1,000-weight world. They edit mostly different weights, but on the first 150 weights task 2 does the *opposite* of task 1. Merge them four ways and ask how much each merged vector still points in each task's direction.
- **Exactly:** `torch.randn(D) * (torch.rand(D) < 0.3)` is a random vector with about 70% of entries zeroed (a boolean mask times a float tensor). Overwriting `tau2[:150] = -tau1[:150] + noise` forces a conflict. `cos(a, b)` is cosine similarity (1 = same direction). The dict maps each method name to its merged *delta*, added to the zero base. Cosine to `tau1` and `tau2` measures how much of each task survives. Note: the toy has no model and no loss, so it measures *geometry only*, not task performance. The seed is fixed through `torch.manual_seed(SEED)` at the top, but this part's random draws also depend on how many random numbers were consumed earlier (the distillation runs), so numbers can differ if you change `STEPS` or `TEMP`.
- **Why it is here:** It makes the interference problem visible: where tasks conflict, averaging cancels both; TIES drops the minority sign.

### 6a. Two real LoRA adapters on two different tasks  (`main.py` lines 137–156)
```python
# ## 6. Merge two real LoRA adapters trained on different tasks
from peft import LoraConfig, get_peft_model       # noqa: E402
ptok = AutoTokenizer.from_pretrained("distilgpt2"); ptok.pad_token = ptok.eos_token
sent, _ = make_sentiment_data(); pref, _ = make_preference_data()
task_a = [r["prompt"] + r["completion"] for r in sent]                         # task A: sentiment words
task_b = [p["prompt"] + p["chosen"] for p in pref]                             # task B: polite replies

def lm_loss(model, texts):
    enc = ptok(texts, return_tensors="pt", padding=True)
    labels = enc.input_ids.masked_fill(enc.attention_mask == 0, -100)
    return model(**enc, labels=labels).loss

cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"], fan_in_fan_out=True, task_type="CAUSAL_LM")
pm = get_peft_model(AutoModelForCausalLM.from_pretrained("distilgpt2"), cfg, adapter_name="a")
pm.add_adapter("b", cfg)
for name, texts in (("a", task_a), ("b", task_b)):
    pm.set_adapter(name)
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=2e-3)
    for _ in range(STEPS):
        loss = lm_loss(pm, random.Random(_).sample(texts, 8)); opt.zero_grad(); loss.backward(); opt.step()
```
- **In simple words:** Build two small tasks as plain text (sentiment sentences, polite replies), attach two separate LoRA adapters to one `distilgpt2`, and train each adapter only on its own task.
- **Exactly:** Task A texts are `review prompt + label word`; task B texts are `question prompt + polite reply`. `lm_loss` is ordinary language-model loss over every real token (pads set to `-100`), here on the *whole* text, not completion-only. `get_peft_model(..., adapter_name="a")` creates an adapter called `a`; `pm.add_adapter("b", cfg)` adds a second one of identical shape; `pm.set_adapter(name)` makes one active (and trainable) while the other is not used. The optimiser only receives parameters with `requires_grad=True`. The learning rate (2e-3) is high, normal for LoRA. For the training batch `random.Random(_)` is seeded with the step number, so each adapter sees the same sequence of batches on every run (`_` is just the loop variable reused as a seed).
- **Why it is here:** LoRA is the realistic case for merging: many adapters, one base. The mechanics of one LoRA are in [06 step 05](../../06_llm_training_and_lora/05_lora_peft_finetune/WALKTHROUGH.md).

### 6b. Report and merge the adapters  (`main.py` lines 158–169)
```python
def report(tag):
    pm.eval()
    with torch.no_grad():
        print(f"{tag:<22} loss on A (sentiment) {lm_loss(pm, task_a[:30]).item():.3f}   loss on B (polite) {lm_loss(pm, task_b[:30]).item():.3f}")
with pm.disable_adapter():
    print("\nLoRA adapters (same rank 8):"); report("base, no adapter")
for name in ("a", "b"):
    pm.set_adapter(name); report(f"adapter {name} alone")
pm.add_weighted_adapter(["a", "b"], [0.5, 0.5], adapter_name="lin", combination_type="linear")
pm.set_adapter("lin"); report("linear merge 0.5/0.5")
pm.add_weighted_adapter(["a", "b"], [1.0, 1.0], adapter_name="ties", combination_type="ties", density=0.5)
pm.set_adapter("ties"); report("TIES merge density .5")
```
- **In simple words:** Measure the loss on both tasks for the plain base model, for each adapter alone, and for two merged adapters (a plain 50/50 mix and a TIES mix).
- **Exactly:** `with pm.disable_adapter():` is a context manager that temporarily bypasses all adapters, giving the base model's behaviour. `report` evaluates the first 30 texts of each task (note: these are *training* texts, so the numbers show fit, not generalisation). `add_weighted_adapter` creates a *new* adapter named `lin` or `ties` from adapters `a` and `b` with the given weights; `density=0.5` is the TIES keep-fraction. Switching with `set_adapter` and calling `report` again measures each merged adapter. The final line of the file is not code. Lower loss is better; the interesting question is whether the merge keeps both tasks' gains.
- **Why it is here:** The toy in 5b shows geometry; this is the same question on a real model: does one merged adapter serve both tasks?

### 7. TRY IT / BREAK IT comments  (`main.py` lines 171–172)
```python
# TRY IT: TEMP=1 python main.py vs TEMP=4 python main.py; change the TIES density to 0.2 in the toy.
# BREAK IT: call pm.add_weighted_adapter with adapters of DIFFERENT ranks and combination_type="linear" and read the error.
```
- **In simple words:** Experiments; see below.
- **Exactly:** Comments only.
- **Why it is here:** Module convention.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `F.softmax(z / T, -1)` | softmax over the last axis at temperature `T` |
| `torch.where(cond, x, y)` | pick `x` where `cond` else `y`, element-wise |
| `t.abs().topk(k).values.min()` | the k-th largest magnitude |
| `x * (torch.rand(D) < 0.3)` | multiply by a 0/1 mask (bool is promoted to float) |
| `a; b; c` on one line | statements separated by semicolons (legal, but less readable) |
| `with pm.disable_adapter():` | temporarily run the base model without adapters |
| `lambda a, b: ...` | a one-line anonymous function |
| `x[mask]` with a boolean mask | selects the entries where the mask is True |

## Libraries & functions
| Function | What it does | Internally |
|---|---|---|
| `F.log_softmax`, `F.softmax` | normalised log-probabilities / probabilities | `log_softmax` uses the log-sum-exp trick for stability |
| `F.cosine_similarity` | `a.b / (norm(a) norm(b))` | along the given dimension |
| `peft.get_peft_model(..., adapter_name)` | attach a named LoRA adapter | wraps targeted layers; `add_adapter` adds another set of A/B tensors under a new name |
| `PeftModel.set_adapter` | choose the active adapter | switches which A/B tensors the wrapped layers use in the forward pass |
| `PeftModel.add_weighted_adapter` | build a merged adapter from several | combines the adapter tensors according to `combination_type` and stores them under a new name |
| `PeftModel.disable_adapter()` | context manager that bypasses adapters | wrapped layers return the base-layer output only |

## Execution trace (real run)
```
teacher mean entropy of next-token distribution: T=1 4.49 nats | T=2.0 9.38 nats


student before training: held-out CE/KL = (4.818, 0.404)
teacher itself:          held-out CE = 4.440

hard labels only (alpha=0)         held-out CE 2.880  KL(teacher||student) 1.824

soft targets T=1 (alpha=1)         held-out CE 4.541  KL(teacher||student) 0.248

soft targets T=2.0 (alpha=1)       held-out CE 4.528  KL(teacher||student) 0.252

mixed T=2.0 (alpha=0.5)            held-out CE 3.045  KL(teacher||student) 0.910

merge toy: D=1000, tau1/tau2 conflict on the first 150 coordinates
method                        cos(merged,tau1)  cos(merged,tau2)
linear average                           0.664             0.655
task arithmetic (lam=1)                  0.664             0.655
SLERP t=0.5                              0.664             0.655
TIES density 0.5                         0.663             0.542
(reference) cos(tau1,tau2)              -0.130

/Users/aman/Desktop/Project_resume/RAP_interview/.venv/lib/python3.12/site-packages/peft/mapping_func.py:178: UserWarning: Adapter name 'a' should not be contained in the prefix 'lora_'. This may lead to reinitialization of the adapter weights during loading.
  warnings.warn(
[transformers] `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`.

LoRA adapters (same rank 8):
base, no adapter       loss on A (sentiment) 5.046   loss on B (polite) 4.221
adapter a alone        loss on A (sentiment) 0.874   loss on B (polite) 6.331
adapter b alone        loss on A (sentiment) 7.232   loss on B (polite) 0.850
linear merge 0.5/0.5   loss on A (sentiment) 1.698   loss on B (polite) 1.730
TIES merge density .5  loss on A (sentiment) 1.645   loss on B (polite) 1.916
```
(Progress bars omitted; the `peft` UserWarning and the `transformers` `loss_type=None` note are real stderr lines kept above.)
- **Entropy**: the teacher's next-token distribution has 4.49 nats of entropy at `T=1` and 9.38 at `T=2`: temperature really flattens it, exposing the tail.
- **Reference points**: the untrained student has held-out CE 4.818 and KL-to-teacher 0.404; the teacher's own CE is 4.440.
- **Hard labels only** gives the *lowest cross-entropy* (2.880, far below the teacher's 4.440) but the *highest KL* to the teacher (1.824, much worse than the untrained 0.404): the student fits this little corpus's style, and drifts away from the teacher. **Soft targets** (`alpha=1`) give the opposite: KL 0.248 (below the untrained 0.404) with CE 4.54, about the teacher's 4.44. The loss you choose decides what you get: imitation of the teacher versus fit to the labels. The mixed recipe lands between the two (CE 3.045, KL 0.910). With `T=1` versus `T=2` the numbers are almost the same here (KL 0.248 versus 0.252): 24 steps are too few to separate them. Only 24 steps and one seed: treat all of this as illustration.
- **Merge toy**: linear average, task arithmetic and SLERP print the same cosines (0.664 / 0.655) because cosine ignores scale and, for these vectors, the three directions coincide to three decimals; they differ in *norm*, which this table does not show. TIES keeps tau1 about as well (0.663) but is less aligned with tau2 (0.542): it elected signs on the 150 conflicting coordinates and dropped the minority sign, so one task's change survived there and the other's did not. `cos(tau1, tau2) = -0.130` is the reference: the conflict pushes the tasks apart. This is geometry only; the toy has no notion of task accuracy.
- **LoRA adapters**: base loss 5.046 on task A and 4.221 on task B. Adapter A alone: A 0.874, B 6.331 (better on A, *worse* than the base on B); adapter B alone: B 0.850, A 7.232. Linear merge 0.5/0.5: A 1.698, B 1.730; TIES: A 1.645, B 1.916. Both merges are far better than the base on both tasks, but each loses to the specialist adapter on its own task (1.7 vs 0.87): merging trades peak performance for breadth. These losses are measured on the *training* texts (the first 30 of each task), so they show fit, not generalisation.

## Experiments (change ONE thing)
| Change | What you will see | Why |
|---|---|---|
| (measured) alpha 0 vs 1 at T=1 | CE 2.880 vs 4.541; KL 1.824 vs 0.248 | hard labels fit the data, soft targets fit the teacher |
| (measured) T=1 vs T=2, alpha=1 | KL 0.248 vs 0.252, CE 4.541 vs 4.528 | no visible difference after 24 steps |
| (not run) `TEMP=4` | entropy of the teacher at T=4 would be higher still; effect on the student not measured | more flattening |
| (not run) TIES `density=0.2` in the toy | fewer entries survive trimming; expect lower cosine to both task vectors | more aggressive trimming |

## Debugging practice
**BREAK IT:** merge two adapters of different ranks with `combination_type="linear"` (the file's suggestion). Real output of the check (adapter `a` with r=8 and adapter `b` with r=4, then `add_weighted_adapter(["a","b"], [0.5,0.5], adapter_name="lin", combination_type="linear")`):
```
ValueError: All adapters must have the same r value when using combination_type linear, ties, dare_ties or dare_linear.
```

How to read it: the exception text names the constraint, and it applies to `linear`, `ties`, `dare_ties` and `dare_linear` alike (so TIES adapter merging in this file also needs equal ranks). Combining `A` and `B` tensors element-wise needs identical shapes; with different ranks look at the combination types the `peft` docs describe for that case (for example `cat` or `svd`). Note also that this file's warning `Adapter name 'a' should not be contained in the prefix 'lora_'` is real and harmless here, but a name like `a` can collide with the `lora_` prefix when saving and reloading, so use longer adapter names in real projects.

## Common mistakes
- Using a teacher and student with *different tokenizers* for logit distillation: the two softmaxes are over different tokens and the KL is meaningless. Use sequence-level distillation instead.
- Forgetting the `T^2` factor and then finding the soft term vanishes at high temperature.
- Letting padding positions contribute to the KL or the cross-entropy: always mask.
- Distilling on the *training* texts and reporting the training loss; report held-out numbers.
- Merging models fine-tuned from different bases, or with different tokenizers or architectures.
- Trusting a merge without evaluating it on *every* task and on general ability: merges are heuristics.
- Treating `lambda = 1` task arithmetic as safe: summing many task vectors overshoots; tune `lambda` on a validation set.
- Merging adapters of different ranks with `linear` (the BREAK IT).

## Rebuild checklist
1. A teacher in `eval()` under `no_grad`, a student, a shared tokenizer, padded batches with a mask.
2. `kd_loss`: `alpha * T^2 * KL(teacher_T || student_T) + (1 - alpha) * CE`, all masked.
3. `evaluate` returning held-out CE and KL to the teacher at `T=1`.
4. Four runs with the same seed varying `(T, alpha)`.
5. `slerp` and `ties` as short functions; a toy with a known conflict; cosine to each task vector.
6. Two named LoRA adapters trained on different texts; `disable_adapter`, `add_weighted_adapter` linear and TIES; a two-task report.

## Say it aloud
"Distillation trains a small student to match the teacher's softened output distribution: alpha times T squared times the KL from teacher to student at temperature T, plus an optional hard-label cross-entropy. The soft targets carry the teacher's sense of which wrong answers are nearly right. Merging treats each fine-tune as a task vector, fine-tuned minus base, and combines them with averaging, task arithmetic, SLERP or TIES, which trims small changes and resolves sign conflicts. It only works for fine-tunes of the same base, it is a heuristic, and I always evaluate the merged model on every task."

## The new problem (cliffhanger)
We have now produced, shrunk and combined models, and every step ended with some number. But how are those numbers *really* computed? Perplexity, exact match, multiple choice by log-likelihood, pass@k, confidence intervals, contamination checks: the numbers behind every leaderboard. On to [step 08](../08_evaluation_from_scratch/WALKTHROUGH.md).
