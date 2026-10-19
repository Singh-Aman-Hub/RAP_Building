# Step 06 - Training parameters: learning rate, batch size, epochs, optimizers, loss

## Goal
See what each training knob does by running controlled experiments (tiny network for speed, then a real LLM fine-tune).

## Run it
```bash
.venv/bin/python 05_llm_parameters/06_training_parameters/main.py
```
(~1 minute; builds on [03_ml_foundations](../../03_ml_foundations/README.md) training-loop and overfitting steps.)

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `opts = {"sgd": lambda: ..., "adam": lambda: ...}` | dict of small anonymous functions - build only the chosen optimizer | `f = lambda x: x + 1` |
| `for i in range(0, len(X), batch_size):` | range with a step: jump by batch size | `range(0, 10, 4)` -> 0, 4, 8 |
| `-(-300 // bs)` | ceiling division trick | `-(-300 // 128)` = 3 |
| `perm = torch.randperm(n)` ; `X[perm[i:i+bs]]` | shuffle then slice indices | - |
| `tr != tr` | True only for NaN | `float("nan") != float("nan")` |
| `def train(optimizer_name="adam", lr=0.01, ...)` | many default arguments -> easy one-at-a-time experiments | `train(lr=0.1)` |
| `loss.backward(); opt.step(); opt.zero_grad()` | `;` puts statements on one line | - |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `sklearn.datasets.make_moons` | scikit-learn | synthetic 2-class 2-D data | `n_samples`, `noise`, `random_state` | X, y (numpy) |
| `nn.Sequential(layers...)` | torch | chain layers | - | model |
| `nn.Linear(in, out)` | torch | fully connected layer | sizes | layer |
| `nn.ReLU()` | torch | `max(0, x)` non-linearity | - | layer |
| `nn.CrossEntropyLoss()` | torch | softmax + negative log-likelihood on logits | - | loss function |
| `torch.optim.SGD / Adam / AdamW` | torch | optimizers | `lr`, `momentum`, `weight_decay` | optimizer |

## The knobs
| Knob | Meaning | Too low | Too high | Memory/speed link |
|---|---|---|---|---|
| **learning rate** | step size of an update | learns very slowly | overshoots, loss explodes | none |
| **batch size** | examples per update | noisy gradients, many slow updates | few updates, may generalise worse | **memory** (activations) |
| **epochs** | passes over the data | underfit | overfit (train loss falls, val loss rises) | time |
| **optimizer** | how gradients -> steps | SGD: simple, slow | Adam/AdamW: adaptive, default for LLMs | extra memory: Adam keeps 2 extra numbers per weight |
| **loss function** | what "wrong" means | - | - | cross-entropy for classification AND next-token prediction |

## Execution trace
```
1) LEARNING RATE (SGD, batch 32, 30 epochs)
   lr=0.0001   train loss=0.734  too small: barely moves
   lr=0.01     train loss=0.390  ok
   lr=0.1      train loss=0.229  good
   lr=10.0     train loss=2.859  too large: loss BLOWS UP

2) BATCH SIZE: batch=4 -> 75 updates/epoch, 0.56 s   |  batch=300 -> 1 update/epoch, 0.01 s
   final losses are all close (0.12-0.14): batch size mostly trades speed, noise and memory.

3) EPOCHS (40 training points, hidden=256)
   epoch   train loss   val loss
       1        0.235      0.280
       5        0.127      0.274
      20        0.066      0.300
      50        0.038      0.521
     100        0.029      0.684      <- train keeps falling, val gets WORSE = overfitting

4) OPTIMIZERS: sgd 0.390 | sgd+momentum 0.237 | adam 0.127 | adamw 0.131   (same lr=0.01)

5) LEARNING RATE ON distilgpt2 (8 steps):
   lr=1e-06   [6.49, 6.39, 6.27, 6.25, 6.28, 6.29, 6.21, 6.11]      crawls
   lr=5e-05   [6.49, 5.76, 5.01, 4.58, 4.18, 3.87, 3.42, 3.13]      healthy
   lr=0.005   [6.49, 6.26, 6.81, 15.54, 7.12, 6.39, 5.09, 4.69]     spikes to 15.5
```
- **Why fine-tuning LRs are tiny (1e-5 to 5e-4)**: the weights are already good; big steps damage ("catastrophically forget") pretrained knowledge.
- **Epochs and datasets:** for LLM fine-tuning 1-3 epochs is typical; more epochs on a small dataset memorises it (step 07 of module 04 showed loss 0.5 on 4 sentences).
- **Batch size on GPUs:** you pick the largest that fits in memory; if too big, use *gradient accumulation* (several small batches before one update). Covered in module 06.
- **Loss functions:** regression -> mean squared error; classification and LLM next-token prediction -> cross-entropy (see [03/06_softmax_cross_entropy](../../03_ml_foundations/06_softmax_cross_entropy/)).

## Experiments
- Add `lr=0.3` to the SGD sweep; find the largest stable value.
- Section 3 with `hidden=8`: overfitting should shrink (less capacity).
- Section 5 with `lr=5e-4` and 20 steps: smooth or not?

## Debugging practice
Delete `opt.zero_grad()` inside `train`: no traceback, but the training curve becomes worse/erratic since gradients accumulate. How would you *detect* this? Print the loss per step, compare with a version that works, and write down the hypothesis ("gradients accumulate") before changing code. Also try `batch_size=0`: `ValueError: range() arg 3 must not be zero` - a precise, readable error.

## Rebuild checklist
- [ ] Write the batch loop: shuffle, slice batch, forward, loss, backward, step, zero_grad.
- [ ] Explain epoch vs iteration vs batch.
- [ ] Draw the train/val loss curves of overfitting and mark where to stop.
- [ ] Explain why Adam uses more memory than SGD.

## Say it aloud
"Learning rate is the step size: too small and training crawls, too large and loss explodes - for LLM fine-tuning it's small, like 1e-5 to 5e-4. Batch size is how many examples per update; it affects gradient noise, speed and especially memory. Epochs are passes over the data; too many overfits, which I detect by validation loss rising while training loss falls. AdamW is the standard optimizer because it adapts the step per weight. Trade-off: every knob trades speed, memory and stability against final quality, so I change one at a time and log the result."
