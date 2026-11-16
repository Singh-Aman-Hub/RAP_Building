# Step 07 — Regularisation

## The story so far
[Module 03 step 05](../../03_ml_foundations/05_overfitting_train_val/WALKTHROUGH.md) showed how to *detect* overfitting (train loss falls, validation loss rises) and named the cures. [Step 06](../06_optimizers_and_schedules/WALKTHROUGH.md) ended with AdamW's `weight_decay` argument and the claim that L2 and decay differ under Adam. Now we apply each cure and read the numbers.

## The pain point
A network with 17,154 parameters and 80 training points can memorise: the baseline below reaches 100 percent training accuracy and 0.817 on unseen data, with a validation loss of 1.64, worse than a coin flip's 0.69 because it is confidently wrong. Training metrics alone would have called it perfect.

## The spark
Regularisation means limiting what the model can fit so that it prefers simple explanations: penalise large weights (L2 / weight decay), randomly remove units (dropout), stop before it memorises (early stopping), show it more varied data (augmentation), forbid certainty (label smoothing). Each trades a little training fit for better generalisation.

## Goal
Train the same oversized net nine ways and compare train/validation accuracy and loss and the weight norm; implement dropout by hand and show train versus eval behaviour; see why Adam with an L2 penalty and AdamW with decay are different algorithms.

## The theory (read this before the interview)
**Generalisation gap.** `gap = train metric - validation metric`. A large gap means variance (overfitting); low train performance means bias (underfitting). Regularisation reduces variance at the cost of some bias.

| Technique | What it does | Knob | Typical use | Measured here (val acc, val loss; baseline 0.817 / 1.638) |
|---|---|---|---|---|
| L2 penalty (`wd` in `Adam`) | adds `wd * w` to the gradient (equivalent to loss `+ wd/2 * |w|^2`) | `wd` | SGD, small nets | wd=0.05: 0.850 / 0.440 (too strong: train acc falls to 0.775) |
| Decoupled weight decay (AdamW) | `w *= 1 - lr * wd` each step | `wd` | transformers | wd=0.05: 0.817 / 1.597; wd=0.5: 0.812 / 0.976 |
| Dropout | zero each unit with prob `p` in training; scale by `1/(1-p)` | `p` | MLPs, attention (small p) | p=0.3: 0.823 / 0.929 |
| Early stopping | keep the checkpoint with the best validation loss | `patience` | everywhere | stopped at epoch 78 of 500: 0.875 / 0.315 (see caveat) |
| Data augmentation | train on transformed copies | transform strength | images, audio, text (paraphrase, noise) | noise 0.15: 0.860 / 0.345 |
| Label smoothing | replace one-hot target by `(1-e)` on the true class, `e/(K-1)` on others (PyTorch spreads `e/K` over all classes) | `e` | classification, LLM pre-training in some recipes | e=0.1: 0.812 / 0.517 |
| Combination | dropout 0.2 + AdamW 0.1 + noise | | | 0.874 / 0.335 |

All val accuracies come from 2000 points, so differences under about 1.5 percentage points are noise; the *losses* are the more informative column here.

**L2 versus weight decay.** With plain SGD, adding `wd * w` to the gradient and shrinking `w` by `lr * wd` per step are identical. With Adam they are not: the L2 term passes through the division by `sqrt(v)`, so parameters with large gradients get less decay and the regularisation strength depends on the gradient history. AdamW removes that coupling. Measured: `Adam(weight_decay=0.05)` shrank the weight norm from 30.3 to 1.8 and underfit (train acc 0.775), `AdamW(weight_decay=0.05)` barely changed it (29.7) because the effective decay per step is `lr * wd = 1.5e-4`. The same number means two different things, so tune `wd` per optimizer. Convention: do not decay biases and norm gains (`param_groups`), and typical LLM `wd` is 0.1 with AdamW.

**Dropout.** In training each unit is zeroed independently with probability `p`; survivors are multiplied by `1/(1-p)` (*inverted dropout*) so the expected activation is unchanged; at evaluation dropout is the identity. It acts like training a large ensemble of thinned networks and prevents units from co-adapting. Pitfall: forgetting `model.eval()` at inference gives random, worse predictions (the module 03 `train()` / `eval()` rule). Modern LLM pre-training often uses little or no dropout because the data is huge and seen about once; fine-tuning on small data brings it (and LoRA dropout) back.

**Early stopping.** Track validation loss each epoch; keep the best weights; stop after `patience` epochs without improvement. *Caveat*: the validation set is now used to make a decision, so its score is slightly optimistic. Honest reporting needs a third set (test) that no decision used. In this file the same 2000 points select the epoch and report it; treat 0.875 as an upper-biased number.

**Data augmentation.** Cheap extra data: flips, crops, colour jitter for images (step 08), noise, time-stretch for audio, back-translation or synonym swaps for text; for LLMs, synthetic data and paraphrases. The transformation must keep the label valid (do not flip the digit 6 into a 9, do not rotate a 'b' into a 'q').

**Label smoothing.** With hard targets cross-entropy pushes the correct logit toward infinity. Smoothing caps confidence (target 0.9 / 0.1 instead of 1 / 0), often improving calibration. Measured: train loss rises (0.0996 instead of 0.039) as expected because the target is no longer reachable, and val loss falls from 1.64 to 0.52 while accuracy is unchanged (0.812): it fixed the *over-confidence*, not the errors.

**Other regularisers.** Weight-norm clipping, stochastic depth / DropPath (vision, deep transformers), mixup/cutmix, noise on inputs or gradients, smaller models, fewer epochs, more data (the best regulariser), ensembling, and `weight tying`. LoRA is itself a low-rank constraint ([module 06](../../06_llm_training_and_lora/README.md)).

**Order of attack when overfitting.** (1) More or better data and augmentation, (2) weight decay + early stopping (free), (3) dropout, (4) smaller model or lower capacity, (5) ensembling. When underfitting, do the opposite: remove regularisation, enlarge, train longer.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| weight decay | L2 regularisation, ridge penalty, Tikhonov | Same for SGD, different for Adam | Decay shrinks weights directly; L2 adds a gradient term; AdamW = real decay |
| L1 vs L2 | lasso vs ridge | Different penalties | L1 `sum |w|` gives sparse weights; L2 `sum w^2` shrinks all weights smoothly |
| dropout | Dropout1d/2d, DropPath/stochastic depth, DropConnect | Related | Dropout zeroes activations; DropConnect zeroes weights; DropPath drops a whole residual branch |
| early stopping | patience, checkpoint selection | Same idea | Stopping rule uses validation loss; it is not an optimiser setting |
| overfitting | high variance, memorisation | Same | Underfitting = high bias; both can show as poor validation score |
| validation vs test set | dev set, held-out set | Different roles | Validation guides decisions (epoch, hyperparameters); test is touched once at the end |
| data augmentation | synthetic data, jittering, back-translation | Overlapping | Augmentation transforms real samples; synthetic data may be generated from scratch |
| label smoothing | soft targets, confidence penalty | Related | Smoothing uses fixed soft targets; knowledge distillation uses a teacher's soft targets |
| generalisation gap | train-validation gap | Same | Measured on the same metric |
| regularisation | inductive bias, constraint | Broad | Anything that reduces variance, including architecture choices such as convolutions (step 08) |

## Run it
```bash
cd 15_neural_networks/07_regularization
python main.py       # about 10 seconds on CPU, torch + scikit-learn
```

## Code, step by step
Every block below is copied from `main.py` (line numbers are exact). Read the block, then the three bullets.

### 0. Docstring and imports  (`main.py` lines 1-22)
```python
"""
GOAL:  Overfit a network on purpose (80 noisy training points), then apply each regularisation technique one at a
       time and read the train / validation numbers: L2 vs decoupled weight decay, dropout (train vs eval
       mode), early stopping, data augmentation and label smoothing.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 15_neural_networks/07_regularization/main.py
EXPECTED OUTPUT (shortened):
    method                         train acc  val acc    gap  ...
    baseline (Adam)                    1.000    0.817  0.183
    L2 via Adam (wd=0.05)              0.775    0.850 -0.075
    early stopping (patience 40)       0.850    0.875 -0.025
    dropout: train mode outputs differ call to call, eval mode outputs identical
NEEDS: Python + numpy + torch + scikit-learn (make_moons). A few seconds on CPU.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons

```
- **In simple words:** Import the libraries. scikit-learn is used only to generate the toy data.
- **Exactly:** `make_moons` produces two interleaving half-circles, a standard 2-D classification set. `torch.nn.functional` supplies `cross_entropy` with its `label_smoothing` argument.
- **Why it is here:** A small, visual, reproducible dataset makes the effect of each method visible in numbers within seconds.

### 1. Data: two noisy half-moons; only 80 training points, 2000 validation points  (`main.py` lines 23-27)
```python
# [1] Data: two noisy half-moons; only 80 training points, 2000 validation points
Xall, yall = make_moons(n_samples=2080, noise=0.35, random_state=0)
Xtr, ytr = torch.tensor(Xall[:80], dtype=torch.float32), torch.tensor(yall[:80])        # (80, 2), (80,)
Xva, yva = torch.tensor(Xall[80:], dtype=torch.float32), torch.tensor(yall[80:])        # (2000, 2), (2000,)
print("train", tuple(Xtr.shape), "val", tuple(Xva.shape), "| class balance in train:", torch.bincount(ytr).tolist())
```
- **In simple words:** Make a tiny training set and a large validation set. The model sees only the 80 points; the 2000 points measure how well it generalises.
- **Exactly:** `make_moons(n_samples=2080, noise=0.35, random_state=0)` returns `X: (2080, 2)` and `y: (2080,)`. The first 80 rows are training (37 vs 43 per class, printed), the remaining 2000 validation. Noise 0.35 makes the classes overlap, so even a perfect model cannot reach 100 percent on new data.
- **Why it is here:** With little data and noisy labels, memorisation is easy and harmful: this is the setting regularisation exists for. A large validation set makes the validation score stable (about +-0.8 points).

### 2. A deliberately oversized network: 17,154 parameters for 80 examples  (`main.py` lines 30-38)
```python
# [2] A deliberately oversized network: 17,154 parameters for 80 examples
def make_net(p_drop=0.0):
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(2, 128), nn.ReLU(), nn.Dropout(p_drop),
                         nn.Linear(128, 128), nn.ReLU(), nn.Dropout(p_drop),
                         nn.Linear(128, 2))


print("parameters:", sum(p.numel() for p in make_net().parameters()))
```
- **In simple words:** Build a network that is much bigger than the problem needs: 2 -> 128 -> 128 -> 2, with optional dropout layers.
- **Exactly:** Parameter count: `2*128+128 + 128*128+128 + 128*2+2 = 17154` (printed). `nn.Dropout(p_drop)` with `p_drop=0.0` is the identity, so the same builder serves every row. The seed makes initial weights identical for every experiment.
- **Why it is here:** Controlling everything except the one thing under test is what makes the comparison valid.

### 3. One training routine; every regularisation method is an argument  (`main.py` lines 41-73)
```python
# [3] One training routine; every regularisation method is an argument
def evaluate(net, X, y):
    net.eval()                                                   # dropout off: deterministic predictions
    with torch.no_grad():
        logits = net(X)
        return F.cross_entropy(logits, y).item(), (logits.argmax(1) == y).float().mean().item()


def train(opt_name="adam", wd=0.0, p_drop=0.0, label_smoothing=0.0, augment=False, epochs=500, patience=None, lr=3e-3):
    net = make_net(p_drop)
    Opt = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[opt_name]
    opt = Opt(net.parameters(), lr=lr, weight_decay=wd)          # Adam: wd = L2 through the gradient; AdamW: decoupled
    gen = torch.Generator().manual_seed(1)
    best = (float("inf"), None, 0)                               # (val loss, state_dict copy, epoch) for early stopping
    for epoch in range(epochs):
        net.train()                                              # dropout on
        X, y = Xtr, ytr
        if augment:                                              # fresh jittered copy of the data every epoch
            X = Xtr + 0.15 * torch.randn(Xtr.shape, generator=gen)
        loss = F.cross_entropy(net(X), y, label_smoothing=label_smoothing)
        opt.zero_grad(); loss.backward(); opt.step()
        if patience is not None:
            v_loss, _ = evaluate(net, Xva, yva)                  # NOTE: early stopping uses the validation set to choose
            if v_loss < best[0]:
                best = (v_loss, {k: v.clone() for k, v in net.state_dict().items()}, epoch)
            elif epoch - best[2] >= patience:
                break
    if patience is not None:
        net.load_state_dict(best[1])                             # roll back to the best epoch
    wnorm = torch.sqrt(sum((p ** 2).sum() for n, p in net.named_parameters() if "weight" in n)).item()
    tr_loss, tr_acc = evaluate(net, Xtr, ytr)                    # report with the CLEAN training data, in eval mode
    va_loss, va_acc = evaluate(net, Xva, yva)
    return dict(tr_acc=tr_acc, va_acc=va_acc, tr_loss=tr_loss, va_loss=va_loss, wnorm=wnorm, epochs=epoch + 1)
```
- **In simple words:** One training function where each cure is a switch. After each run it reports scores on the clean training data and on the validation data with the network in 'exam mode'.
- **Exactly:** `evaluate` calls `net.eval()` (dropout off) and `torch.no_grad()`, returning `(loss, accuracy)`. `train` runs full-batch for up to 500 epochs: `net.train()`; optional input noise (`augment`) with `std 0.15`; `F.cross_entropy(..., label_smoothing=...)`; `Adam` vs `AdamW` with the same `weight_decay` argument; early stopping keeps a `state_dict` clone of the best validation loss and rolls back to it. `wnorm` is the L2 norm of all weight tensors. Training scores are computed on the clean `Xtr`, not the noisy copy.
- **Why it is here:** Reporting clean, eval-mode numbers avoids fooling yourself: a dropout network measured with dropout on or an augmented network measured on augmented data would look worse than it is.

### 4. The experiment table: same data, same network, same seed; one change per row  (`main.py` lines 76-92)
```python
# [4] The experiment table: same data, same network, same seed; one change per row
runs = {
    "baseline (Adam)":              dict(),
    "L2 via Adam (wd=0.05)":        dict(opt_name="adam", wd=0.05),
    "AdamW decoupled (wd=0.05)":    dict(opt_name="adamw", wd=0.05),
    "AdamW decoupled (wd=0.5)":     dict(opt_name="adamw", wd=0.5),
    "dropout p=0.3":                dict(p_drop=0.3),
    "early stopping (patience 40)": dict(patience=40),
    "data augmentation (noise)":    dict(augment=True),
    "label smoothing 0.1":          dict(label_smoothing=0.1),
    "dropout + AdamW + noise":      dict(p_drop=0.2, opt_name="adamw", wd=0.1, augment=True),
}
print(f"\n{'method':<30}{'train acc':>10}{'val acc':>9}{'gap':>7}{'train loss':>12}{'val loss':>10}{'|W|':>8}{'epochs':>8}")
for name, kw in runs.items():
    r = train(**kw)
    print(f"{name:<30}{r['tr_acc']:>10.3f}{r['va_acc']:>9.3f}{r['tr_acc'] - r['va_acc']:>7.3f}"
          f"{r['tr_loss']:>12.4f}{r['va_loss']:>10.4f}{r['wnorm']:>8.2f}{r['epochs']:>8}")
```
- **In simple words:** Run nine experiments and print one table.
- **Exactly:** Rows (train acc / val acc / train loss / val loss / |W| / epochs): baseline `1.000 / 0.817 / 0.0390 / 1.6376 / 30.34 / 500`; Adam with `wd=0.05` `0.775 / 0.850 / 0.5198 / 0.4403 / 1.83`; AdamW `wd=0.05` `0.988 / 0.817 / 0.0478 / 1.5969 / 29.69`; AdamW `wd=0.5` `0.975 / 0.812 / 0.0797 / 0.9759 / 25.16`; dropout 0.3 `0.913 / 0.823 / 0.1860 / 0.9289 / 25.34`; early stopping `0.850 / 0.875 / 0.3358 / 0.3149 / 10.82 / 78`; noise augmentation `0.875 / 0.860 / 0.2898 / 0.3448 / 15.06`; label smoothing `0.988 / 0.812 / 0.0996 / 0.5165 / 25.14`; combination `0.850 / 0.874 / 0.3049 / 0.3350 / 14.53`.
- **Why it is here:** The table is the deliverable of the step: every technique is a different way of moving the weight norm and train loss up a little and the validation loss down a lot.

### 5. Dropout by hand: zero random units while training and rescale the survivors, so the average stays equal  (`main.py` lines 95-117)
```python
# [5] Dropout by hand: zero random units while training and rescale the survivors, so the average stays equal
def my_dropout(x, p, training, gen):
    if not training or p == 0.0:
        return x                                                 # at inference: identity
    mask = (torch.rand(x.shape, generator=gen) > p).float()      # 1 = keep, 0 = drop, with probability p of dropping
    return x * mask / (1 - p)                                    # "inverted dropout": scale by 1/(1-p) now, so eval needs no change


g = torch.Generator().manual_seed(0)
x = torch.ones(1, 10000)
out = my_dropout(x, 0.3, True, g)
print(f"\nmy_dropout p=0.3 on 10,000 ones: fraction zeroed {(out == 0).float().mean():.3f}, survivors are {out[out > 0][0]:.4f} "
      f"(= 1/0.7), mean {out.mean():.3f} (still ~1)")

layer = nn.Dropout(0.5)
v = torch.ones(1, 8)
layer.train()
print("nn.Dropout train mode, same input twice:", layer(v).tolist()[0], layer(v).tolist()[0])
layer.eval()
print("nn.Dropout eval  mode, same input twice:", layer(v).tolist()[0], layer(v).tolist()[0])

# TRY IT: set p_drop=0.6 in the "dropout" row, or wd=2.0 in AdamW, and see underfitting (train acc falls too).
# BREAK IT: delete `net.eval()` inside `evaluate` and re-run the dropout row: the reported train and validation numbers are measured with dropout still active.
```
- **In simple words:** Write dropout yourself, then check how PyTorch's version acts in the two modes.
- **Exactly:** `torch.rand(shape) > p` is a boolean mask (keep with probability `1-p`); `x * mask / (1-p)`. On 10,000 ones with `p=0.3`: 29.5 percent zeroed, survivors `1.4286 = 1/0.7`, mean `1.007`. `nn.Dropout(0.5)`: in train mode two calls on the same input give different patterns (survivors are `2.0 = 1/0.5`); in eval mode both calls return all ones.
- **Why it is here:** Inverted dropout (scaling during training) means inference needs no change. The two modes explain the most common serving bug: a model left in train mode gives random answers.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `{'adam': torch.optim.Adam, ...}[name]` | dict used as a switch; the value is a class | `Opt(net.parameters(), ...)` |
| `{k: v.clone() for k, v in sd.items()}` | dict comprehension that copies each tensor | snapshot of the best weights |
| `float('inf')` | positive infinity as the 'worst so far' | initial best loss |
| `torch.randn(shape, generator=g)` | random numbers from a seeded generator | reproducible augmentation |
| `(out == 0).float().mean()` | fraction of zeros | 0.295 |
| `out[out > 0][0]` | boolean-mask indexing, then the first element | survivor value 1.4286 |
| `dict(tr_acc=..., va_acc=...)` | build a dict with keyword arguments | return value of `train` |
| `{**kw}` / `train(**kw)` | unpack a dict into keyword arguments | `runs` table |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `make_moons(n_samples, noise, random_state)` | scikit-learn | two-moons toy dataset | see signature | `(X, y)` |
| `F.cross_entropy(logits, y, label_smoothing=e)` | torch.nn.functional | softmax cross-entropy with optional smoothing | logits `(N,K)`, int labels | scalar |
| `nn.Dropout(p)` | torch | zero units with prob `p` in train mode, scale by `1/(1-p)` | `p` | module |
| `net.train()` / `net.eval()` | torch | switch module behaviour | - | the module |
| `net.state_dict()` / `load_state_dict` | torch | weights as a dict / restore them | - | dict / None |
| `torch.no_grad()` | torch | context: do not record gradients | - | context manager |

## Execution trace (real run)
```
train (80, 2) val (2000, 2) | class balance in train: [37, 43]
parameters: 17154

method                         train acc  val acc    gap  train loss  val loss     |W|  epochs
baseline (Adam)                    1.000    0.817  0.183      0.0390    1.6376   30.34     500
L2 via Adam (wd=0.05)              0.775    0.850 -0.075      0.5198    0.4403    1.83     500
AdamW decoupled (wd=0.05)          0.988    0.817  0.171      0.0478    1.5969   29.69     500
AdamW decoupled (wd=0.5)           0.975    0.812  0.164      0.0797    0.9759   25.16     500
dropout p=0.3                      0.913    0.823  0.089      0.1860    0.9289   25.34     500
early stopping (patience 40)       0.850    0.875 -0.025      0.3358    0.3149   10.82      78
data augmentation (noise)          0.875    0.860  0.015      0.2898    0.3448   15.06     500
label smoothing 0.1                0.988    0.812  0.175      0.0996    0.5165   25.14     500
dropout + AdamW + noise            0.850    0.874 -0.024      0.3049    0.3350   14.53     500

my_dropout p=0.3 on 10,000 ones: fraction zeroed 0.295, survivors are 1.4286 (= 1/0.7), mean 1.007 (still ~1)
nn.Dropout train mode, same input twice: [0.0, 2.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0] [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0]
nn.Dropout eval  mode, same input twice: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```
- The baseline: 100 percent train, 81.7 percent validation, validation loss 1.64: overconfident and wrong, with a weight norm of 30.
- Adam + L2 0.05 is too strong (train acc 0.775) but has the lowest weight norm (1.83); AdamW with the same number barely changes anything (norm 29.7): same `wd`, different algorithm.
- Dropout 0.3 cuts the gap from 0.183 to 0.089; early stopping and augmentation give gaps of -0.025 and 0.015 and the lowest validation losses (0.31, 0.34).
- Label smoothing leaves accuracy at 0.812 but cuts the validation loss from 1.64 to 0.52: it fixes confidence only.
- The combination (0.874) is as good as early stopping without peeking at validation.
- `my_dropout`: 29.5 percent zeroed, survivors scaled by 1/0.7, mean preserved; `nn.Dropout` differs between calls in train mode and is deterministic in eval mode.

## Experiments (change ONE thing)
1. Dropout `p=0.6`: train acc 0.887, val acc 0.826, val loss 0.556 (better loss than p=0.3's 0.929 but train accuracy is lower: capacity is now being removed).
2. AdamW `wd=2.0`: train 0.913, val 0.831, val loss 0.483, weight norm 18.0: heavy decay is now helping.
3. Adam L2: `wd=0.005`: val acc 0.844, val loss 0.366, norm 6.3 (a sweet spot); `wd=0.5`: train 0.538, val 0.498 (chance), norm 0.048: the penalty swamps the data loss and the network collapses toward zero weights.
4. Label smoothing 0.4: train loss 0.269, val loss 0.455, val acc 0.814: accuracy still unchanged.
5. Early stopping with `patience=10` instead of 40: stops at epoch 48 with val acc 0.875, val loss 0.315, essentially the same checkpoint as before (the best epoch was 38).

## Debugging practice
BREAK IT: delete `net.eval()` inside `evaluate`, then re-run the dropout row. The program does not crash. The network is still in the mode left by training, so dropout is active during evaluation. Train loss reads 0.201 instead of 0.186 and validation loss 0.902 instead of 0.929: the metrics are measured on a randomly thinned network, so they are no longer a property of the model alone (and repeated calls would give different outputs, as the `nn.Dropout` train-mode lines in the trace show). Rule: every inference path starts with `model.eval()` and `torch.no_grad()` (or `torch.inference_mode()`); add a unit test that calls the model twice on the same input and asserts equal outputs.

Related silent bug: reporting training metrics from the noisy augmented batch. Compute them on clean data, as `train` does.

## Rebuild checklist
- [ ] define generalisation gap and draw train/val curves for over- and underfitting
- [ ] write inverted dropout and explain train vs eval
- [ ] explain L2 vs AdamW decay with the measured norms
- [ ] write the early-stopping loop with `state_dict` rollback and mention the selection-bias caveat
- [ ] name three augmentations and the label-validity rule
- [ ] explain label smoothing and what it does to loss vs accuracy

## The new problem (cliffhanger)
Our regularisers treat the input as a flat vector. For images that wastes a huge prior: nearby pixels are related and a cat is a cat wherever it sits in the picture. A fully connected layer on a 28x28 image has 784 weights per unit and no idea about neighbours. Next: [step 08, convolutions](../08_cnn_convolutions/WALKTHROUGH.md), a regulariser built into the architecture.

> **Say it aloud (interview box):** Overfitting means a large train-validation gap. I fight it with more data and augmentation first, then weight decay, early stopping and dropout. Weight decay shrinks weights each step; under Adam I use AdamW because adding an L2 term to the gradient gets rescaled by the adaptive denominator. Dropout zeroes units during training and rescales survivors by 1/(1-p) so evaluation needs no change, which is why model.eval() matters. Early stopping keeps the best validation checkpoint, but then the validation score is optimistic and I need a separate test set. Label smoothing caps confidence and mostly improves calibration and loss, not accuracy.
