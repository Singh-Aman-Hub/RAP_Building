# Terms — Training Parameters

Folder: `05_llm_parameters/06_training_parameters` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [05 LLM parameters README](../README.md)

**Terms in this folder:** [make_moons](#make_moons) · [Train / validation split](#train--validation-split) · [nn.Sequential](#nnsequential) · [nn.Linear](#nnlinear) · [nn.ReLU](#nnrelu) · [MLP (multi-layer perceptron)](#mlp-multi-layer-perceptron) · [Loss function](#loss-function) · [nn.CrossEntropyLoss](#nncrossentropyloss) · [Optimizer](#optimizer) · [SGD](#sgd) · [Momentum](#momentum) · [Adam](#adam) · [AdamW and weight decay](#adamw-and-weight-decay) · [Lambda and dict of builders](#lambda-and-dict-of-builders) · [Epoch](#epoch) · [Iteration and batch](#iteration-and-batch) · [range with a step](#range-with-a-step) · [torch.randperm](#torchrandperm) · [Learning rate](#learning-rate) · [Divergence (loss blows up)](#divergence-loss-blows-up) · [Batch size](#batch-size) · [Ceiling division trick](#ceiling-division-trick) · [Gradient accumulation](#gradient-accumulation) · [Overfitting](#overfitting) · [Model capacity (hidden size)](#model-capacity-hidden-size) · [Train/val loss history](#trainval-loss-history) · [Training step: backward, step, zero_grad](#training-step-backward-step-zero_grad) · [Semicolon statements](#semicolon-statements) · [Fine-tuning learning rates](#fine-tuning-learning-rates) · [Catastrophic forgetting](#catastrophic-forgetting) · [Optimizer memory (Adam states)](#optimizer-memory-adam-states) · [NaN check (x != x)](#nan-check-x--x) · [time.perf_counter](#timeperf_counter) · [Default arguments for experiments](#default-arguments-for-experiments) · [range() arg 3 must not be zero](#range-arg-3-must-not-be-zero)

---

## make_moons  `(third-party: scikit-learn)`
**1. Definition:** `sklearn.datasets.make_moons` generates a synthetic two-class 2-D dataset shaped like two interleaving half-circles, with optional noise.
**2. In plain words:** It is a tiny dataset that is not linearly separable, so a small neural network is needed. It trains in a blink, perfect for experiments where you must run dozens of settings.
**3. Example to quote:** > "I used `make_moons(n_samples=600, noise=0.25)` so every learning-rate experiment finishes in seconds."
```python
X, y = make_moons(n_samples=600, noise=0.25, random_state=0)
```
**Seen in:** `main.py`, data section.

## Train / validation split  `(concept)`
**1. Definition:** Dividing data into a training part, used to update weights, and a validation part, used only to measure generalisation.
**2. In plain words:** Never train on validation data. Here the first 300 points train the model and the remaining 300 check it.
**3. Example to quote:** > "I take `X[:300]` for training and `X[300:]` for validation."

**Seen in:** `main.py`.

## nn.Sequential  `(third-party: torch)`
**1. Definition:** A PyTorch container that chains layers so data flows through them in order.
**2. In plain words:** It lets you describe a network as a list of layers without writing a class.
**3. Example to quote:** > "`nn.Sequential(nn.Linear(2, 64), nn.ReLU(), ...)` is my entire model."

**Seen in:** `main.py`, `make_model`.

## nn.Linear  `(third-party: torch)`
**1. Definition:** A fully connected layer computing `x @ W + b` for given input and output sizes.
**2. In plain words:** It mixes all input features into new features using learnable weights, the same building block as the matrices in attention.
**3. Example to quote:** > "`nn.Linear(2, 64)` turns each 2-D point into 64 features."

**Seen in:** `main.py`.

## nn.ReLU  `(third-party: torch)`
**1. Definition:** An activation layer computing `max(0, x)` element-wise.
**2. In plain words:** Without a non-linearity, stacked linear layers collapse into one linear layer. ReLU adds the bend that lets the network learn curved boundaries.
**3. Example to quote:** > "ReLU between linear layers lets my network separate the moons."

**Seen in:** `main.py`.

## MLP (multi-layer perceptron)  `(concept)`
**1. Definition:** A feed-forward network made of several linear layers with non-linear activations between them.
**2. In plain words:** The simplest neural network. Here: 2 inputs, two hidden layers, 2 class scores (logits).
**3. Example to quote:** > "My MLP maps 2 inputs through two 64-unit hidden layers to 2 logits."

**Seen in:** `main.py`, `make_model`.

## Loss function  `(concept)`
**1. Definition:** A function that measures how wrong a model's outputs are, producing a number to minimise.
**2. In plain words:** Regression uses mean squared error; classification and next-token prediction use cross-entropy. Training is simply lowering this number.
**3. Example to quote:** > "I use cross-entropy because both my toy classifier and LLMs predict a class or token from logits."

**Seen in:** `main.py`.

## nn.CrossEntropyLoss  `(third-party: torch)`
**1. Definition:** PyTorch loss that applies softmax and negative log-likelihood to raw logits and a target class index.
**2. In plain words:** You pass logits, not probabilities. It punishes the model when the correct class gets low probability.
**3. Example to quote:** > "`loss_fn(model(X[idx]), y[idx])` takes raw logits and class labels directly."

**Seen in:** `main.py`, `train`.

## Optimizer  `(concept)`
**1. Definition:** The algorithm that turns gradients into weight updates.
**2. In plain words:** Different rules give different learning speeds. SGD is simple; Adam adapts the step per weight.
**3. Example to quote:** > "With the same learning rate 0.01, SGD reached loss 0.390 and Adam 0.127."

**Seen in:** `main.py`, section 4.

## SGD  `(third-party: torch)`
**1. Definition:** Stochastic gradient descent: updates each weight by minus learning rate times its gradient on a batch.
**2. In plain words:** Simple and cheap, but needs a well-chosen learning rate and can be slow.
**3. Example to quote:** > "`torch.optim.SGD(model.parameters(), lr=lr)` is the plain optimizer."

**Seen in:** `main.py`.

## Momentum  `(concept)`
**1. Definition:** An optimizer feature that keeps a running average of past gradients so updates continue in a consistent direction.
**2. In plain words:** Like a rolling ball that does not stop at every bump. It sped SGD from loss 0.390 to 0.237 here.
**3. Example to quote:** > "`momentum=0.9` made SGD converge much faster in my comparison."

**Seen in:** `main.py`, `sgd+momentum`.

## Adam  `(third-party: torch)`
**1. Definition:** An optimizer that keeps running averages of gradients and squared gradients to set an adaptive step size per weight.
**2. In plain words:** It needs less learning-rate tuning than SGD. The cost: two extra numbers stored per weight.
**3. Example to quote:** > "Adam reached train loss 0.127 versus SGD's 0.390 at the same learning rate."

**Seen in:** `main.py`.

## AdamW and weight decay  `(third-party: torch)`
**1. Definition:** AdamW is Adam with weight decay applied directly to the weights, separately from the gradient update.
**2. In plain words:** Weight decay gently shrinks weights to reduce overfitting. AdamW is the default optimizer for LLMs.
**3. Example to quote:** > "I use `AdamW(lr=..., weight_decay=0.01)` as in most LLM fine-tuning."
```python
torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
```
**Seen in:** `main.py`.

## Lambda and dict of builders  `(syntax)`
**1. Definition:** `lambda: expr` is a small anonymous function; storing lambdas in a dict builds only the chosen optimizer when called.
**2. In plain words:** `opts[optimizer_name]()` looks up a name and calls it. Without lambdas all four optimizers would be created up front.
**3. Example to quote:** > "`opts = {'sgd': lambda: torch.optim.SGD(...), ...}` lets me choose an optimizer by name."
```python
f = lambda x: x + 1
f(2)  # 3
```
**Seen in:** `main.py`, `train`.

## Epoch  `(concept)`
**1. Definition:** One complete pass over the training data.
**2. In plain words:** Too few epochs underfit; too many on a small dataset overfit. For LLM fine-tuning 1 to 3 epochs is typical.
**3. Example to quote:** > "I train for 30 epochs, shuffling the data each epoch."

**Seen in:** `main.py`.

## Iteration and batch  `(concept)`
**1. Definition:** A batch is the group of examples used for one weight update; an iteration is one such update.
**2. In plain words:** With 300 points and batch size 32, one epoch is about 10 iterations.
**3. Example to quote:** > "Each inner loop pass is one iteration: one batch, one weight update."

**Seen in:** `main.py`.

## range with a step  `(Python built-in)`
**1. Definition:** `range(start, stop, step)` produces numbers jumping by `step`.
**2. In plain words:** `range(0, len(X), batch_size)` gives the starting index of each batch. A step of 0 raises `ValueError`.
**3. Example to quote:** > "`range(0, 10, 4)` gives 0, 4, 8, which is how I walk through batches."
```python
list(range(0, 10, 4))  # [0, 4, 8]
```
**Seen in:** `main.py`, `train`.

## torch.randperm  `(third-party: torch)`
**1. Definition:** Returns a random permutation of the integers `0` to `n-1`.
**2. In plain words:** Indexing the data with it shuffles the order each epoch so batches differ every time.
**3. Example to quote:** > "`perm = torch.randperm(len(X))` shuffles my training examples."

**Seen in:** `main.py`, `train`.

## Learning rate  `(concept)`
**1. Definition:** The size of each weight update step.
**2. In plain words:** 0.0001 barely moves, 0.1 works well, 10.0 explodes in this experiment. For LLM fine-tuning use small values.
**3. Example to quote:** > "At lr=10 the loss was 2.859, worse than where it started; at 0.1 it was 0.229."

**Seen in:** `main.py`, section 1.

## Divergence (loss blows up)  `(concept)`
**1. Definition:** A training failure where the loss grows instead of shrinking, often because the step size overshoots the minimum.
**2. In plain words:** Think of stepping so far across a valley that you land higher on the other side. The fix is a smaller learning rate.
**3. Example to quote:** > "On distilgpt2, lr=5e-3 made the loss spike to 15.5."

**Seen in:** `main.py`, sections 1 and 5.

## Batch size  `(concept)`
**1. Definition:** The number of examples per weight update.
**2. In plain words:** Small batches give noisy gradients and many updates; large batches are smooth but few updates and need more memory. On GPUs it is the main memory knob.
**3. Example to quote:** > "Batch 4 gave 75 updates per epoch; batch 300 gave 1."

**Seen in:** `main.py`, section 2.

## Ceiling division trick  `(syntax)`
**1. Definition:** `-(-a // b)` computes `ceil(a / b)` using floor division on a negative number.
**2. In plain words:** It counts how many batches are needed even if the last batch is partial: 300 points at batch 128 is 3 batches.
**3. Example to quote:** > "`-(-300 // 128)` is 3 updates per epoch."
```python
-(-300 // 128)  # 3
```
**Seen in:** `main.py`, section 2.

## Gradient accumulation  `(concept)`
**1. Definition:** Running several small batches and summing their gradients before one optimizer step, to simulate a larger batch.
**2. In plain words:** When the batch you want does not fit in GPU memory, accumulate over several smaller ones. Covered in module 06.
**3. Example to quote:** > "I can't fit batch 32, so I use batch 8 with accumulation over 4 steps."

**Seen in:** `WALKTHROUGH.md`.

## Overfitting  `(concept)`
**1. Definition:** When training loss keeps falling but validation loss stops improving or rises, because the model memorises the training data.
**2. In plain words:** In the epoch table, train loss went 0.235 to 0.029 while validation went 0.280 to 0.684. Stop at the validation minimum.
**3. Example to quote:** > "Train loss kept falling but validation loss rose from 0.27 to 0.68: classic overfitting."

**Seen in:** `main.py`, section 3.

## Model capacity (hidden size)  `(concept)`
**1. Definition:** How complex a function a model can represent; increased by more parameters, such as a larger hidden size.
**2. In plain words:** A big model on tiny data overfits easily. Using `hidden=8` reduces overfitting.
**3. Example to quote:** > "With only 40 training points and hidden=256 the model memorises quickly."

**Seen in:** `main.py`, section 3.

## Train/val loss history  `(concept)`
**1. Definition:** A list of (train loss, validation loss) pairs recorded each epoch.
**2. In plain words:** Plotting or printing it shows when learning stops helping. The code stores it when `track=True`.
**3. Example to quote:** > "I read the history to find the epoch where validation loss was lowest."

**Seen in:** `main.py`, `train`.

## Training step: backward, step, zero_grad  `(concept)`
**1. Definition:** The three calls `loss.backward()`, `opt.step()`, `opt.zero_grad()` compute gradients, update weights, then clear gradients.
**2. In plain words:** Removing `zero_grad` gives no error but worse training because gradients pile up.
**3. Example to quote:** > "If I delete `zero_grad()` the loss curve gets erratic with no traceback."

**Seen in:** `main.py`.

## Semicolon statements  `(syntax)`
**1. Definition:** A `;` lets several statements share one line.
**2. In plain words:** The script writes `loss.backward(); opt.step(); opt.zero_grad()` on one line to save space. Normally each goes on its own line.
**3. Example to quote:** > "Those three statements are separate calls, only written on one line."

**Seen in:** `main.py`, section 5.

## Fine-tuning learning rates  `(concept)`
**1. Definition:** Fine-tuning usually uses low rates (about 1e-5 to 5e-4) because the pretrained weights are already good.
**2. In plain words:** On distilgpt2, 1e-6 crawled, 5e-5 was healthy, 5e-3 spiked.
**3. Example to quote:** > "I fine-tune with 5e-5: big enough to learn, small enough to keep what the model already knows."

**Seen in:** `main.py`, section 5.

## Catastrophic forgetting  `(concept)`
**1. Definition:** Loss of previously learned abilities when a model is trained too aggressively on new data.
**2. In plain words:** Big steps overwrite pretrained knowledge. Low learning rates and few epochs reduce the risk.
**3. Example to quote:** > "A large learning rate can make a fine-tuned model forget general English."

**Seen in:** `WALKTHROUGH.md`.

## Optimizer memory (Adam states)  `(concept)`
**1. Definition:** Adam stores two extra numbers per weight (running averages), so it uses more memory than SGD.
**2. In plain words:** Training a model needs weights, gradients and optimizer states, not just weights, a big reason training needs far more memory than inference.
**3. Example to quote:** > "Adam triples the per-parameter state compared with plain weights."

**Seen in:** `WALKTHROUGH.md`, knob table.

## NaN check (x != x)  `(syntax)`
**1. Definition:** `x != x` is True only when `x` is NaN, because NaN is not equal to anything, even itself.
**2. In plain words:** A way to detect exploded training without importing math.
**3. Example to quote:** > "`float('nan') != float('nan')` is True."

**Seen in:** `WALKTHROUGH.md`, syntax table.

## time.perf_counter  `(standard library)`
**1. Definition:** A high-resolution timer in seconds; subtract two readings to get elapsed time.
**2. In plain words:** Used to compare the time cost of different batch sizes.
**3. Example to quote:** > "Batch 4 took 0.56 s and batch 300 took 0.01 s."

**Seen in:** `main.py`, section 2.

## Default arguments for experiments  `(syntax)`
**1. Definition:** Giving `train` default values for each parameter so you can change one at a time.
**2. In plain words:** `train(lr=0.1)` changes only the learning rate. It makes controlled experiments trivial.
**3. Example to quote:** > "I call `train('sgd', lr=0.1)` and everything else stays at defaults."

**Seen in:** `main.py`, `def train`.

## range() arg 3 must not be zero  `(concept)`
**1. Definition:** The `ValueError` raised when a `range` step is 0, for example from batch size 0.
**2. In plain words:** A clear, readable error showing that an invalid parameter reached the loop.
**3. Example to quote:** > "`batch_size=0` raises `range() arg 3 must not be zero`, which points straight at the cause."

**Seen in:** `main.py`, BREAK IT comment.
