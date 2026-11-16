# Terms — Optimisers and learning-rate schedules

Folder: `15_neural_networks/06_optimizers_and_schedules` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [15 README](../README.md)

**Terms in this folder:** [Optimizer](#optimizer) · [Stochastic gradient descent (SGD)](#stochastic-gradient-descent-sgd) · [Momentum (heavy ball)](#momentum-heavy-ball) · [Nesterov accelerated gradient](#nesterov-accelerated-gradient) · [RMSProp](#rmsprop) · [Adam](#adam) · [AdamW (decoupled weight decay)](#adamw-decoupled-weight-decay) · [Bias correction (Adam)](#bias-correction-adam) · [Optimizer state](#optimizer-state) · [Learning-rate schedule](#learning-rate-schedule) · [Warm-up](#warm-up) · [Cosine decay (annealing)](#cosine-decay-annealing) · [Rosenbrock function](#rosenbrock-function) · [Muon](#muon) · [Schedule-free optimisation](#schedule-free-optimisation) · [Lion](#lion) · [LambdaLR](#lambdalr) · [param_groups](#paramgroups) · [UserWarning: lr_scheduler.step() before optimizer.step()](#userwarning-lrschedulerstep-before-optimizerstep)

---

## Optimizer  `(concept)`
**1. Definition:** An algorithm that turns gradients into parameter updates.
**2. In plain words:** The driver; the gradient is the map.
**3. Example to quote:** > "SGD, Adam, AdamW..."
**Seen in:** `main.py`, block 2

## Stochastic gradient descent (SGD)  `(concept)`
**1. Definition:** `p -= lr * g`, with `g` computed on a mini-batch.
**2. In plain words:** Take a step downhill using a noisy estimate of the slope.
**3. Example to quote:** > "`SGD.step`"
**Seen in:** `main.py`, block 2

## Momentum (heavy ball)  `(concept)`
**1. Definition:** Keep a velocity `v = mu v + g` and step along it.
**2. In plain words:** A rolling ball: it keeps going in a consistent direction and smooths out jitters.
**3. Example to quote:** > "mu = 0.9"
**Seen in:** `main.py`, block 2

## Nesterov accelerated gradient  `(concept)`
**1. Definition:** Momentum variant that evaluates the update with a look-ahead: `g + mu v`.
**2. In plain words:** Peek where momentum will take you before correcting.
**3. Example to quote:** > "`nesterov=True`"
**Seen in:** `main.py`, block 2

## RMSProp  `(concept)`
**1. Definition:** Divide the gradient by the square root of a running average of its square.
**2. In plain words:** Cap the speed of parameters with large gradients; boost those with small ones.
**3. Example to quote:** > "alpha = 0.99"
**Seen in:** `main.py`, block 2

## Adam  `(concept)`
**1. Definition:** Adaptive moment estimation: first moment `m`, second moment `v`, bias correction, update `lr m_hat / (sqrt(v_hat) + eps)`.
**2. In plain words:** Momentum and per-parameter step sizes together.
**3. Example to quote:** > "b1 0.9, b2 0.999, eps 1e-8"
**Seen in:** `main.py`, block 2

## AdamW (decoupled weight decay)  `(concept)`
**1. Definition:** Adam plus `p *= 1 - lr * wd` applied separately from the gradient update.
**2. In plain words:** Shrink the weights by a fixed small percentage each step, independent of gradient size.
**3. Example to quote:** > "Standard for transformers."
```python
if self.decoupled:
    p = p * (1 - self.lr * self.wd)
```
**Seen in:** `main.py`, block 2

## Bias correction (Adam)  `(concept)`
**1. Definition:** Dividing `m` and `v` by `1 - b^t` to undo their zero initialisation.
**2. In plain words:** Early averages are too small because they started at 0.
**3. Example to quote:** > "Without it the first step is 3.16 times too large."
**Seen in:** `main.py`, block 2

## Optimizer state  `(concept)`
**1. Definition:** Per-parameter tensors an optimiser keeps (`m`, `v`, step counter).
**2. In plain words:** Extra memory: two more copies of the model for Adam.
**3. Example to quote:** > "Save it in checkpoints."
**Seen in:** `main.py`, block 2

## Learning-rate schedule  `(concept)`
**1. Definition:** A function from training step to learning rate.
**2. In plain words:** Big steps early, small steps late.
**3. Example to quote:** > "`lr_at(step)`"
**Seen in:** `main.py`, block 4

## Warm-up  `(concept)`
**1. Definition:** Starting with a tiny learning rate and ramping it up over the first steps.
**2. In plain words:** Don't sprint from a cold start.
**3. Example to quote:** > "100 linear steps here."
**Seen in:** `main.py`, block 4

## Cosine decay (annealing)  `(concept)`
**1. Definition:** Lowering the lr along a half cosine from peak to a floor.
**2. In plain words:** A smooth glide path.
**3. Example to quote:** > "0.895 at step 300, 0.100 at the end."
**Seen in:** `main.py`, block 4

## Rosenbrock function  `(concept)`
**1. Definition:** `(1-x)^2 + 100 (y - x^2)^2`, minimum at (1, 1) in a narrow curved valley; a classic optimiser test.
**2. In plain words:** A banana-shaped valley.
**3. Example to quote:** > "Start (-1.5, 2): loss 12.5."
**Seen in:** `main.py`, block 1

## Muon  `(concept)`
**1. Definition:** An optimiser for 2-D hidden-layer weight matrices that orthogonalises the momentum update with Newton-Schulz iterations; other parameters use AdamW.
**2. In plain words:** A newer, matrix-aware optimiser; `torch.optim.Muon` exists in recent torch.
**3. Example to quote:** > "Verify current status before claiming anything about its benefits."
**Seen in:** `main.py`, block 6

## Schedule-free optimisation  `(concept)`
**1. Definition:** Replaces momentum with interpolation and averaging so no LR decay schedule or fixed end step is needed (Defazio et al., 2024).
**2. In plain words:** Train without deciding in advance when to stop.
**3. Example to quote:** > "`facebookresearch/schedule_free`"
**Seen in:** `WALKTHROUGH.md`, theory

## Lion  `(concept)`
**1. Definition:** An optimiser using the sign of an interpolated momentum, with a single state tensor.
**2. In plain words:** Cheaper memory than Adam; verify in the paper (Chen et al., 2023).
**3. Example to quote:** > "Not in `torch.optim` here."
**Seen in:** `main.py`, block 6

## LambdaLR  `(library)`
**1. Definition:** A scheduler that sets `lr = base_lr * fn(step)`.
**2. In plain words:** Plug in any schedule function.
**3. Example to quote:** > "`T.lr_scheduler.LambdaLR(opt, lr_at)`"
**Seen in:** `main.py`, block 4

## param_groups  `(library)`
**1. Definition:** A list of dicts inside a PyTorch optimiser, each holding parameters and their own `lr`, `weight_decay`.
**2. In plain words:** Lets different parameters have different settings (e.g. no decay on biases).
**3. Example to quote:** > "`opt.param_groups[0]['lr']`"
**Seen in:** `main.py`, block 4

## UserWarning: lr_scheduler.step() before optimizer.step()  `(error)`
**1. Definition:** The scheduler was stepped before the optimiser, skipping the first lr value.
**2. In plain words:** Wrong order of two calls.
**3. Example to quote:** > "Call `opt.step()` first."
**Seen in:** `WALKTHROUGH.md`, debugging
