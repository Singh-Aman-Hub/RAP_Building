# Terms — PyTorch training loop

Folder: `03_ml_foundations/04_pytorch_training_loop` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [ML foundations README](../README.md)

**Terms in this folder:** [Training loop (PyTorch standard)](#training-loop-pytorch-standard--concept) · [torch.manual_seed()](#torchmanual_seed--third-party-torch) · [torch.randn() and unsqueeze()](#torchrandn-and-unsqueeze--third-party-torch) · [Dataset (TensorDataset)](#dataset-tensordataset--third-party-torch) · [random_split()](#random_split--third-party-torch) · [DataLoader](#dataloader--third-party-torch) · [Batch (`xb`, `yb`)](#batch-xb-yb--concept) · [nn.Module](#nnmodule--third-party-torch) · [super().__init__()](#super__init__--syntax) · [Class inheritance](#class-inheritance--syntax) · [nn.Linear](#nnlinear--third-party-torch) · [forward()](#forward--third-party-torch) · [Loss function (nn.MSELoss)](#loss-function-nnmseloss--third-party-torch) · [Optimizer (SGD / Adam)](#optimizer-sgd--adam--third-party-torch) · [model.parameters() and numel()](#modelparameters-and-numel--third-party-torch) · [optimizer.zero_grad()](#optimizerzero_grad--third-party-torch) · [optimizer.step()](#optimizerstep--third-party-torch) · [model.train() and model.eval()](#modeltrain-and-modeleval--third-party-torch) · [Validation loss](#validation-loss--concept) · [Weighted average of batch losses](#weighted-average-of-batch-losses--concept) · [Tuple unpacking in loops](#tuple-unpacking-in-loops--syntax) · [Generator expression inside sum()](#generator-expression-inside-sum--syntax) · [Overfitting check via train vs validation](#overfitting-check-via-train-vs-validation--concept) · [Universal training loop](#universal-training-loop--concept)

---

## Training loop (PyTorch standard)  `(concept)`
**1. Definition:** The standard PyTorch training loop iterates over epochs and batches, doing: forward pass, loss, `zero_grad`, `backward`, `optimizer.step`.
**2. In plain words:** This five-line skeleton is the same whether you train a line (here) or an LLM (module 06). Learning it once lets you read almost any training script.
**3. Example to quote:**
> "For each batch: predict, compute the loss, clear old gradients, backpropagate, then step the optimizer."

```python
pred = model(xb); loss = loss_fn(pred, yb)
optimizer.zero_grad(); loss.backward(); optimizer.step()
```
**Seen in:** `main.py`, training loop

## torch.manual_seed()  `(third-party: torch)`
**1. Definition:** `torch.manual_seed(n)` sets the seed of PyTorch's random number generator so random operations are repeatable.
**2. In plain words:** Weight initialisation, data shuffling and noise all use randomness. Seeding makes your results match run to run, which is vital when debugging or comparing experiments.
**3. Example to quote:**
> "I call `torch.manual_seed(0)` so my random data and initial weights are the same every run."

**Seen in:** `main.py`, top of script

## torch.randn() and unsqueeze()  `(third-party: torch)`
**1. Definition:** `torch.randn(500, 2)` creates a tensor of standard-normal random numbers; `tensor.unsqueeze(1)` inserts a new dimension of size 1 at position 1.
**2. In plain words:** `randn` fabricates 500 samples with 2 features each. `unsqueeze(1)` turns labels of shape `(500,)` into `(500, 1)`, which matches the model's output shape; mismatched shapes cause silent broadcasting bugs.
**3. Example to quote:**
> "`y.unsqueeze(1)` changes the shape from `(500,)` to `(500, 1)` so it lines up with the model's predictions."

**Seen in:** `main.py`, data section

## Dataset (TensorDataset)  `(third-party: torch)`
**1. Definition:** `TensorDataset(X, y)` wraps tensors so that indexing it returns the matching `(X[i], y[i])` pair.
**2. In plain words:** A Dataset is "how to fetch sample i". It is what a DataLoader draws from. For LLMs, the dataset returns tokenised text chunks (module 06).
**3. Example to quote:**
> "`TensorDataset(X, y)` pairs each feature row with its label."

**Seen in:** `main.py`, `dataset`

## random_split()  `(third-party: torch)`
**1. Definition:** `random_split(dataset, [400, 100])` randomly divides a dataset into non-overlapping subsets of the given lengths.
**2. In plain words:** A quick 80/20 train/validation split. The lengths must add up to the dataset size.
**3. Example to quote:**
> "`random_split(dataset, [400, 100])` gives me 400 training and 100 validation samples."

**Seen in:** `main.py`, `train_set, val_set`

## DataLoader  `(third-party: torch)`
**1. Definition:** `DataLoader(dataset, batch_size, shuffle)` iterates over a dataset, yielding mini-batches and optionally reshuffling every epoch.
**2. In plain words:** It handles batching and shuffling so the loop can simply write `for xb, yb in train_loader`. With 400 samples and batch size 32 there are 13 batches per epoch (the last one smaller).
**3. Example to quote:**
> "`DataLoader(train_set, batch_size=32, shuffle=True)` gives me shuffled batches of 32, 13 per epoch."

**Seen in:** `main.py`, `train_loader`, `val_loader`

## Batch (`xb`, `yb`)  `(concept)`
**1. Definition:** A batch is the group of samples processed together in one forward/backward pass; `xb` and `yb` are the batch's features and labels.
**2. In plain words:** Processing in batches is faster than one at a time and gives smoother gradients than a single sample. Batch size is limited by memory.
**3. Example to quote:**
> "Each loop iteration gives me `xb` with 32 rows and `yb` with the 32 matching labels."

**Seen in:** `main.py`, training loop

## nn.Module  `(third-party: torch)`
**1. Definition:** `torch.nn.Module` is the base class for all PyTorch models and layers; it tracks parameters and defines `forward`.
**2. In plain words:** Subclass it, define layers in `__init__` and the computation in `forward`. Calling `model(x)` runs `forward`. LLM models in Hugging Face are also `nn.Module`s.
**3. Example to quote:**
> "My `LinearModel` subclasses `nn.Module`, which makes PyTorch aware of its weights."

**Seen in:** `main.py`, `class LinearModel`

## super().__init__()  `(syntax)`
**1. Definition:** `super().__init__()` calls the parent class's constructor from inside the child class's `__init__`.
**2. In plain words:** For `nn.Module` it is mandatory: it sets up the machinery that registers parameters. Forgetting it gives an `AttributeError` when you assign layers.
**3. Example to quote:**
> "The first line of my `__init__` is `super().__init__()` so `nn.Module` can track the layers I add."

**Seen in:** `main.py`, `LinearModel.__init__`

## Class inheritance  `(syntax)`
**1. Definition:** `class Child(Parent):` creates a class that inherits attributes and methods from `Parent`.
**2. In plain words:** `LinearModel(nn.Module)` gets everything `nn.Module` provides (parameter tracking, `.train()`, `.eval()`, `.parameters()`) for free and adds its own layers.
**3. Example to quote:**
> "By inheriting from `nn.Module` I get `.parameters()` and `.eval()` without writing them."

**Seen in:** `main.py`, `class LinearModel(nn.Module)`

## nn.Linear  `(third-party: torch)`
**1. Definition:** `nn.Linear(in_features, out_features)` is a layer that computes `y = xWᵀ + b` with learnable weight and bias.
**2. In plain words:** It is the line from earlier, generalised to many inputs and outputs. `nn.Linear(2, 1)` has 2 weights and 1 bias, so 3 parameters in total. Neural networks stack many of these.
**3. Example to quote:**
> "`nn.Linear(2, 1)` takes two features and produces one prediction; it has 3 trainable numbers."

**Seen in:** `main.py`, `self.linear`

## forward()  `(third-party: torch)`
**1. Definition:** The `forward(self, x)` method of an `nn.Module` defines how inputs become outputs; calling `model(x)` invokes it.
**2. In plain words:** You describe the computation once and PyTorch handles gradients. Call the model like a function, not `model.forward(x)` directly, so hooks run.
**3. Example to quote:**
> "`model(xb)` runs `forward`, which here just passes the batch through the linear layer."

**Seen in:** `main.py`, `def forward`

## Loss function (nn.MSELoss)  `(third-party: torch)`
**1. Definition:** `nn.MSELoss()` is a callable that returns the mean squared error between predictions and targets.
**2. In plain words:** It scores how wrong the model is as one number, which `backward()` differentiates. For classification you would use cross-entropy instead.
**3. Example to quote:**
> "`loss = loss_fn(pred, yb)` gives a single number I can call `backward()` on."

**Seen in:** `main.py`, `loss_fn`

## Optimizer (SGD / Adam)  `(third-party: torch)`
**1. Definition:** An optimizer such as `torch.optim.SGD(params, lr)` applies parameter updates from the stored gradients on each `.step()`.
**2. In plain words:** It replaces the manual `w -= lr * grad` from before. SGD is the simplest; Adam adapts step sizes per parameter and is common for transformers. It needs `model.parameters()` to know what to update.
**3. Example to quote:**
> "I create `torch.optim.SGD(model.parameters(), lr=0.05)` and call `optimizer.step()` after `backward()`."

**Seen in:** `main.py`, `optimizer`

## model.parameters() and numel()  `(third-party: torch)`
**1. Definition:** `model.parameters()` yields all learnable tensors of the model; `p.numel()` is the number of elements in a tensor.
**2. In plain words:** Summing `numel()` over parameters counts the model's parameters: 3 here. LLMs have billions, which is why memory is a concern in later modules.
**3. Example to quote:**
> "`sum(p.numel() for p in model.parameters())` returns 3 for this model, but 124 million for GPT-2."

**Seen in:** `main.py`, parameter count

## optimizer.zero_grad()  `(third-party: torch)`
**1. Definition:** Clears the stored gradients of all parameters before computing new ones.
**2. In plain words:** Gradients accumulate by default. Skipping this call means each batch's gradient is added to the previous batches', and training becomes unstable.
**3. Example to quote:**
> "I call `optimizer.zero_grad()` before `loss.backward()` every batch so gradients do not pile up."

**Seen in:** `main.py`, training loop

## optimizer.step()  `(third-party: torch)`
**1. Definition:** Updates every parameter using its current gradient according to the optimizer's rule.
**2. In plain words:** The "take the step downhill" call. One `step()` happens per batch, so 13 steps per epoch here.
**3. Example to quote:**
> "`optimizer.step()` applies the update; with batch size 400 it would run once per epoch, with batch size 8 fifty times."

**Seen in:** `main.py`, training loop

## model.train() and model.eval()  `(third-party: torch)`
**1. Definition:** `model.train()` puts a model in training mode; `model.eval()` puts it in evaluation mode, which changes the behaviour of layers such as dropout and batch normalisation.
**2. In plain words:** This tiny model has no such layers, but you should still always switch modes: forgetting `eval()` for validation or inference gives noisy, wrong results in bigger models.
**3. Example to quote:**
> "Before validating I call `model.eval()` so dropout is off, and `model.train()` again at the start of the next epoch."

**Seen in:** `main.py`, training loop

## Validation loss  `(concept)`
**1. Definition:** The loss computed on held-out data that the model is not trained on.
**2. In plain words:** If training loss keeps falling while validation loss rises, the model is overfitting. Here both fall together (0.0089 vs 0.0112), so it generalises well.
**3. Example to quote:**
> "Epoch 10 train loss is 0.0089 and validation loss is 0.0112. They are close, so there is no overfitting."

**Seen in:** `main.py`, evaluation block

## Weighted average of batch losses  `(concept)`
**1. Definition:** Summing `loss * len(batch)` over batches and dividing by the number of samples gives the true per-sample average, even when the last batch is smaller.
**2. In plain words:** A plain average of batch losses over-weights a small final batch. Weighting by batch size fixes that.
**3. Example to quote:**
> "`sum(loss_fn(...).item() * len(xb) ...) / len(train_set)` gives the correct overall loss."

**Seen in:** `main.py`, `train_loss`, `val_loss`

## Tuple unpacking in loops  `(syntax)`
**1. Definition:** `for xb, yb in loader:` unpacks each yielded `(features, labels)` pair into two names.
**2. In plain words:** Each batch from the DataLoader is a pair. Unpacking in the `for` header avoids indexing `batch[0]` and `batch[1]`.
**3. Example to quote:**
> "`for xb, yb in train_loader` gives me the batch inputs and labels as two variables."

**Seen in:** `main.py`, training loop

## Generator expression inside sum()  `(syntax)`
**1. Definition:** `sum(expr for item in iterable)` totals values computed lazily from a generator.
**2. In plain words:** Computes the weighted loss and the parameter count without building intermediate lists.
**3. Example to quote:**
> "`sum(p.numel() for p in model.parameters())` adds up the sizes of all parameter tensors."

**Seen in:** `main.py`, loss and parameter count lines

## Overfitting check via train vs validation  `(concept)`
**1. Definition:** Comparing training and validation loss to detect whether a model is memorising the training data.
**2. In plain words:** A big gap (train low, validation high) means overfitting; step 05 explores this. A linear model with 3 parameters is too simple to overfit.
**3. Example to quote:**
> "I watch both losses. If validation loss rises while training loss falls, I stop early or simplify the model."

**Seen in:** `main.py`, epoch print

## Universal training loop  `(concept)`
**1. Definition:** The five-line sequence inside every PyTorch training step: forward (`model(x)`), loss (`loss_fn`), `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`.
**2. In plain words:** Only the model, the data and the loss change between projects. The same five lines train a 3-parameter line fit and, later, fine-tune an LLM.
**3. Example to quote:**
> "Fine-tuning an LLM is the same five-line loop as my linear model, just with a bigger model and a text loss."

**Seen in:** `main.py`, the inner `for xb, yb in train_loader` loop
