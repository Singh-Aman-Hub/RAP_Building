# Terms — Softmax and cross-entropy

Folder: `03_ml_foundations/06_softmax_cross_entropy` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [ML foundations README](../README.md)

**Terms in this folder:** [Logits](#logits--concept) · [Softmax](#softmax--concept) · [Probability distribution](#probability-distribution--concept) · [Numerical stability trick (`z - z.max()`)](#numerical-stability-trick-z---zmax--concept) · [np.exp() and np.log()](#npexp-and-nplog--third-party-numpy) · [Cross-entropy loss](#cross-entropy-loss--concept) · [F.cross_entropy (PyTorch)](#fcross_entropy-pytorch--third-party-torch) · [import torch.nn.functional as F](#import-torchnnfunctional-as-f--syntax) · [Batch dimension (`unsqueeze(0)`)](#batch-dimension-unsqueeze0--third-party-torch) · [Temperature (in softmax)](#temperature-in-softmax--concept) · [Default argument (`temperature=1.0`)](#default-argument-temperature10--syntax) · [Class index as target](#class-index-as-target--concept) · [Confident and wrong](#confident-and-wrong--concept) · [.round() and .sum() on arrays](#round-and-sum-on-arrays--third-party-numpy)

---

## Logits  `(concept)`
**1. Definition:** Logits are the raw, unnormalised scores a model outputs for each class before they are converted to probabilities.
**2. In plain words:** They can be any real number, including negative, and larger means "more likely". An LLM produces one logit per token in its vocabulary (about 50,000 for GPT-2) at each step. Temperature and sampling in module 05 work on these numbers.
**3. Example to quote:**
> "My classifier gives logits `[2.0, 1.0, 0.1]` for cat, dog and bird. Cat has the highest score, but these are not yet probabilities."

**Seen in:** `main.py`, `logits`

## Softmax  `(concept)`
**1. Definition:** Softmax converts a vector of scores into probabilities by exponentiating each score and dividing by the sum, so all outputs are positive and add to 1.
**2. In plain words:** It turns "scores" into "chances". The biggest score gets the biggest share, and exponentiation makes the gap between scores more pronounced. For logits `[2.0, 1.0, 0.1]` the result is `[0.659, 0.242, 0.099]`.
**3. Example to quote:**
> "Softmax of `[2.0, 1.0, 0.1]` gives 66% cat, 24% dog and 10% bird, and the three add up to 1."

```python
e = np.exp(z - z.max()); probs = e / e.sum()
```
**Seen in:** `main.py`, `softmax`

## Probability distribution  `(concept)`
**1. Definition:** A probability distribution over classes is a list of non-negative numbers that sum to 1, giving the likelihood of each class.
**2. In plain words:** The output of softmax. An LLM's next-token distribution is exactly this over the whole vocabulary, and generation is just picking from it.
**3. Example to quote:**
> "`probs.sum()` is 1.0, which confirms the output is a valid probability distribution."

**Seen in:** `main.py`, `probs`

## Numerical stability trick (`z - z.max()`)  `(concept)`
**1. Definition:** Subtracting the maximum score before exponentiating prevents overflow from `exp` of large numbers without changing the softmax result.
**2. In plain words:** `exp(1000)` overflows to infinity. Subtracting the max keeps the largest exponent at `exp(0) = 1`. Softmax is unchanged because the same factor cancels in numerator and denominator.
**3. Example to quote:**
> "I subtract `z.max()` first so `np.exp` never overflows, and the probabilities come out identical."

**Seen in:** `main.py`, `softmax`

## np.exp() and np.log()  `(third-party: numpy)`
**1. Definition:** `np.exp(x)` computes e to the power x element-wise; `np.log(x)` computes the natural logarithm.
**2. In plain words:** `exp` makes any score positive; `log` is its inverse and turns a probability into a loss value. Taking `log` of 0 gives `-inf`.
**3. Example to quote:**
> "`np.exp([2.0, 1.0, 0.1])` is about `[7.39, 2.72, 1.11]`, before dividing by their sum of 11.21."

**Seen in:** `main.py`, `softmax`, `loss`

## Cross-entropy loss  `(concept)`
**1. Definition:** Cross-entropy loss for one sample is `-log(p_correct)`, the negative log of the probability the model assigned to the correct class.
**2. In plain words:** If the model is confident and right the loss is near 0; if it is confident and wrong the loss is huge. If truth is cat (66%), loss is 0.417; if truth is bird (10%), loss is 2.317. It is the standard loss for classification and for LLM next-token training.
**3. Example to quote:**
> "The model gave bird only 10%, so if the answer is bird the loss is -ln(0.099) = 2.317, much larger than 0.417 for cat."

**Seen in:** `main.py`, `loss`

## F.cross_entropy (PyTorch)  `(third-party: torch)`
**1. Definition:** `torch.nn.functional.cross_entropy(logits, target)` applies log-softmax to raw logits and computes the negative log-likelihood of the target class indices.
**2. In plain words:** It expects raw logits, not probabilities, and class indices, not one-hot vectors. A classic bug is applying softmax first and softmax again inside: no error, just a wrong loss. It matches the NumPy value 0.417.
**3. Example to quote:**
> "I pass raw logits to `F.cross_entropy`. If I passed softmax output, it would softmax twice and give a wrong number silently."

**Seen in:** `main.py`, `torch_loss`

## import torch.nn.functional as F  `(syntax)`
**1. Definition:** Imports PyTorch's functional API of stateless operations (losses, activations) under the alias `F`.
**2. In plain words:** `F` is the conventional alias. Use it for operations without learnable state, such as `F.cross_entropy` or `F.softmax`.
**3. Example to quote:**
> "`import torch.nn.functional as F` lets me write `F.cross_entropy(...)`, the usual convention."

**Seen in:** `main.py`, imports

## Batch dimension (`unsqueeze(0)`)  `(third-party: torch)`
**1. Definition:** `tensor.unsqueeze(0)` adds a new leading dimension of size 1, turning shape `(3,)` into `(1, 3)`.
**2. In plain words:** PyTorch losses expect shape `(batch, classes)`. One sample is a batch of one.
**3. Example to quote:**
> "`unsqueeze(0)` turns my 3 logits into a batch of 1 sample with 3 classes, the shape cross-entropy expects."

**Seen in:** `main.py`, `torch_loss`

## Temperature (in softmax)  `(concept)`
**1. Definition:** Temperature divides the logits before softmax: low values sharpen the distribution toward the top class, high values flatten it.
**2. In plain words:** At temperature 0.5, cat jumps to 86%; at 5.0 the three classes are nearly equal (40%, 33%, 27%). This is the exact mechanism behind the temperature setting for text generation in module 05.
**3. Example to quote:**
> "At temperature 0.5 the top class gets 86%; at 5.0 it gets 40%. Lower temperature means more confident, less random output."

**Seen in:** `main.py`, `softmax(z, temperature)`

## Default argument (`temperature=1.0`)  `(syntax)`
**1. Definition:** A default argument gives a parameter a value used when the caller omits it.
**2. In plain words:** `softmax(logits)` uses temperature 1.0 (no change), while `softmax(logits, 0.5)` overrides it.
**3. Example to quote:**
> "I call `softmax(logits)` normally and `softmax(logits, t)` to explore temperature."

**Seen in:** `main.py`, `def softmax`

## Class index as target  `(concept)`
**1. Definition:** The target for cross-entropy is the integer index of the correct class, such as 0 for "cat".
**2. In plain words:** A label like "cat" must be mapped to a number first. `probs[true_class]` then picks the probability assigned to the right answer.
**3. Example to quote:**
> "Classes are `['cat', 'dog', 'bird']`, so the true class 0 means cat, and I read `probs[0]`."

**Seen in:** `main.py`, `true_class`

## Confident and wrong  `(concept)`
**1. Definition:** A prediction with high probability on the wrong class, which cross-entropy punishes very heavily because `-log(p)` grows without bound as p approaches 0.
**2. In plain words:** The loss encourages calibrated honesty: it is better to be uncertain than confidently wrong. This is why models sometimes show overconfidence issues.
**3. Example to quote:**
> "If the model gave the correct class just 1%, the loss would be about 4.6, so confident mistakes are expensive."

**Seen in:** `main.py`, cross-entropy comment

## .round() and .sum() on arrays  `(third-party: numpy)`
**1. Definition:** `arr.round(3)` rounds each element to 3 decimals; `arr.sum()` adds all elements.
**2. In plain words:** Used to print clean probabilities and to check they sum to 1.
**3. Example to quote:**
> "`probs.round(3)` prints `[0.659, 0.242, 0.099]`, and `probs.sum()` confirms 1.0."

**Seen in:** `main.py`, prints
