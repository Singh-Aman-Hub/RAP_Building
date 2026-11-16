# Step 06 — Softmax and cross-entropy

## The story so far
Step 05 taught us to trust validation numbers. But we only predicted real numbers so far. Chapter 5, Decisions & Confidence, handles picking one answer from several.

## The pain point
A classifier outputs raw scores (logits) such as `[2.0, 1.0, 0.1]`. They can be negative, do not sum to anything meaningful, and MSE is the wrong way to score them. We need probabilities and a loss that punishes confident mistakes.

## The spark
Exponentiate each score so all are positive, divide by the total so they sum to 1: that is softmax. Then take the negative log of the probability given to the correct class: that is cross-entropy. An LLM does exactly this over its whole vocabulary for every next token.

## Goal
Turn scores into probabilities and score them. Same maths as next-token prediction in an LLM.

## Run it
```bash
python main.py
.venv/bin/python 03_ml_foundations/06_softmax_cross_entropy/main.py
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Logits | raw scores, pre-softmax outputs | Yes | Unbounded real numbers, one per class; softmax turns them into probabilities |
| Softmax | normalised exponential | Yes | `exp(z_i) / sum(exp(z))`; output is positive and sums to 1 |
| Cross-entropy | log loss, negative log-likelihood (NLL) | Same for one-hot targets | `-log p(correct class)`; "NLL" in PyTorch (`nll_loss`) expects log-probabilities as input |
| `F.cross_entropy` | `log_softmax` + `nll_loss` | Yes | Takes raw logits; applying softmax first is a bug |
| Temperature | softmax temperature, T | Yes | Divides logits before softmax: T<1 sharper, T>1 flatter |
| Perplexity | exp(cross-entropy) | Related | The average "branching factor"; lower is better |
| Sigmoid vs softmax | binary vs multi-class | No | Sigmoid maps one score to a probability; softmax normalises across several classes |

## Code, step by step

### 1. Module docstring  (`main.py` lines 1–9)

```python
"""
GOAL: Understand softmax (scores -> probabilities) and cross-entropy (the classification loss).
      An LLM's next-token prediction is exactly this over a ~50,000-word vocabulary.

RUN (from this folder):  python main.py
RUN (from repo root):    .venv/bin/python 03_ml_foundations/06_softmax_cross_entropy/main.py
EXPECTED OUTPUT: probabilities summing to 1, loss values, numpy == torch agreement.
NEEDS: numpy, torch
"""
```

- **In simple words:** Says the step is softmax plus cross-entropy and that LLMs use the same maths.
- **Exactly:** Docstring only. An LLM's next-token head is a softmax over a ~50,000 entry vocabulary (50,257 for GPT-2).
- **Why it is here:** Tells you why this small step matters.

### 2. Imports  (`main.py` lines 11–13)

```python
import numpy as np
import torch
import torch.nn.functional as F
```

- **In simple words:** numpy for the by-hand version, torch and its functional API for the library version.
- **Exactly:** `torch.nn.functional` (`F`) has stateless functions such as `cross_entropy`.
- **Why it is here:** Doing it twice lets you prove the two agree.

### 3. The model's raw scores (logits)  (`main.py` lines 15–16)

```python
# A model outputs raw scores called LOGITS, one per class. Classes: ["cat", "dog", "bird"].
logits = np.array([2.0, 1.0, 0.1])                        # shape: (3,) = (n_classes,)
```

- **In simple words:** A model's last layer outputs one number per answer choice: cat, dog, bird.
- **Exactly:** `logits = np.array([2.0, 1.0, 0.1])` is shape `(3,)`, i.e. `(n_classes,)`. These are not probabilities: they can be negative or exceed 1.
- **Why it is here:** Everything below converts such scores into a probability and a loss.

### 4. Softmax with temperature and a stability trick  (`main.py` lines 19–24)

```python
def softmax(z, temperature=1.0):
    """exp each score, divide by the total -> positive numbers that sum to 1."""
    z = z / temperature                                  # temperature rescales logits (module 05 revisits this!)
    z = z - z.max()                                      # numerical-stability trick: doesn't change the result
    e = np.exp(z)
    return e / e.sum()
```

- **In simple words:** Turn scores into positive numbers that add to 1: raise e to each score, divide by the total.
- **Exactly:** `z / temperature` rescales. `z - z.max()` shifts so the largest is 0 (exp of a large number like 1000 would overflow to inf); softmax is invariant to adding a constant. `np.exp` then `e / e.sum()`. For logits `[2, 1, 0.1]` shifted `[0, -1, -1.9]` gives exp `[1, 0.368, 0.150]`, sum 1.518. Shape stays `(3,)`.
- **Why it is here:** Without the shift, large logits give `nan` in real models. Temperature is reused for sampling in module 05.

### 5. Compute and print the probabilities  (`main.py` lines 27–28)

```python
probs = softmax(logits)                                 # shape: (3,)
print("softmax:", probs.round(3), "sum =", probs.sum())
```

- **In simple words:** Apply softmax to our scores.
- **Exactly:** Real output: `softmax: [0.659 0.242 0.099] sum = 1.0`. The highest logit gets the highest probability, and order is preserved.
- **Why it is here:** Confirms the output is a valid probability distribution.

### 6. Cross-entropy loss by hand  (`main.py` lines 30–34)

```python
# Cross-entropy: -log(probability the model gave to the CORRECT class). Confident & right => ~0, confident & wrong => huge.
true_class = 0                                           # the answer is "cat"
loss = -np.log(probs[true_class])
print("loss if truth=cat :", round(float(loss), 4))
print("loss if truth=bird:", round(float(-np.log(probs[2])), 4), "(model gave bird only 10%, so loss is bigger)")
```

- **In simple words:** The loss is how surprised the model is by the right answer: low if it gave the truth a high probability.
- **Exactly:** `loss = -log(p[true_class])`. For truth=cat: -ln(0.659) = 0.417. For truth=bird: -ln(0.099) = 2.317. Real output `0.417` and `2.317`. Probability 1 gives loss 0; probability tending to 0 gives a loss tending to infinity.
- **Why it is here:** This is the quantity training minimises for classification and for LLM next-token prediction.

### 7. Same loss using PyTorch  (`main.py` lines 36–38)

```python
# Same thing in PyTorch. NOTE: F.cross_entropy takes RAW logits (it applies softmax itself) -- a classic bug is to softmax twice.
torch_loss = F.cross_entropy(torch.tensor(logits).unsqueeze(0), torch.tensor([true_class]))  # logits (1, 3), target (1,)
print("torch cross_entropy:", round(torch_loss.item(), 4), "(matches numpy)")
```

- **In simple words:** PyTorch has a built-in that does softmax and the loss together.
- **Exactly:** `torch.tensor(logits)` is float64 `(3,)`; `.unsqueeze(0)` makes `(1, 3)` = `(batch, classes)`. The target `torch.tensor([0])` is `(1,)` holding the class index (not one-hot). `F.cross_entropy` computes `log_softmax` then NLL, so it takes RAW logits. Real output `0.417`, equal to the numpy value.
- **Why it is here:** Applying softmax before `F.cross_entropy` softmaxes twice; the number is wrong but there is no error.

### 8. Effect of temperature, and TRY/BREAK IT  (`main.py` lines 40–44)

```python
# Temperature: low = sharper (more confident), high = flatter (more random)
for t in (0.5, 1.0, 5.0):
    print(f"temperature {t}: {softmax(logits, t).round(3)}")
# TRY IT: set logits = [10, 1, 0.1]. Probability of cat -> ~1 and the loss -> ~0.
# BREAK IT: pass probabilities (not logits) to F.cross_entropy and compare the number: it's wrong but raises no error.
```

- **In simple words:** Divide the scores by a small number to sharpen the distribution, by a large number to flatten it.
- **Exactly:** Real output: T=0.5 gives `[0.864 0.117 0.019]`; T=1 gives `[0.659 0.242 0.099]`; T=5 gives `[0.4 0.327 0.273]`. As T goes to 0 it becomes argmax; as T goes to infinity, uniform 1/3 each.
- **Why it is here:** This knob is exactly the LLM `temperature` parameter you meet in module 05.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| default arg `temperature=1.0` | optional parameter | `softmax(z, 0.5)` |
| `z = z - z.max()` | reassign to a shifted copy | stability trick |
| `probs[true_class]` | index with an int | |
| `.unsqueeze(0)` | add batch dimension: (3,) → (1,3) | cross_entropy wants (batch, classes) |

## Libraries & functions
| Name | Origin | Does | Notes |
|---|---|---|---|
| `np.exp`, `np.log` | numpy | exponential / natural log | |
| `F.cross_entropy(logits, target)` | torch.nn.functional | softmax + log + negative mean | takes RAW logits, class indices |

## Execution trace (real run)
Logits `[2.0, 1.0, 0.1]` → exp `[7.39, 2.72, 1.11]` → sum 11.21 → probs `[0.659 0.242 0.099]`.
```
loss if truth=cat : 0.417          (-ln 0.659)
loss if truth=bird: 2.317          (-ln 0.099)
torch cross_entropy: 0.417 (matches numpy)
temperature 0.5: [0.864 0.117 0.019]
temperature 1.0: [0.659 0.242 0.099]
temperature 5.0: [0.4   0.327 0.273]
```
Low temperature sharpens; high flattens. Module [05](../../05_llm_parameters/) uses exactly this for text generation.

## Experiments
1. Logits `[10, 1, 0.1]`: loss near 0.
2. Make all logits equal → uniform probabilities, loss = ln(3) = 1.0986.
3. Temperature 0.1 and 100.

## Debugging practice
Passing probabilities to `F.cross_entropy` produces a wrong number with no error (double softmax): for logits `[2.0, 1.0, 0.1]` and truth=cat the loss comes out `0.802` instead of `0.417`. Rule: loss functions with "cross_entropy" in PyTorch want logits.

## Rebuild checklist
- [ ] Softmax with the max-subtraction trick.
- [ ] Cross-entropy for one example.
- [ ] Explain why confident-and-wrong is punished heavily.

## The new problem (cliffhanger)
Softmax and cross-entropy give us a loss, but a single loss number does not tell us what *kind* of mistakes we make. Is 90% accuracy good? It depends on the classes. Next: [07_metrics](../07_metrics/).

## Say it aloud
> **Say it aloud (interview box):** Softmax converts raw scores into probabilities that sum to 1. Cross-entropy is the negative log of the probability given to the correct answer, so it is tiny when the model is right and confident and huge when it is confidently wrong. A language model applies exactly this over its whole vocabulary for every next token.
