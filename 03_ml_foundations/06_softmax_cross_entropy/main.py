"""
GOAL: Understand softmax (scores -> probabilities) and cross-entropy (the classification loss).
      An LLM's next-token prediction is exactly this over a ~50,000-word vocabulary.

RUN (from this folder):  python main.py
RUN (from repo root):    .venv/bin/python 03_ml_foundations/06_softmax_cross_entropy/main.py
EXPECTED OUTPUT: probabilities summing to 1, loss values, numpy == torch agreement.
NEEDS: numpy, torch
"""

import numpy as np
import torch
import torch.nn.functional as F

# A model outputs raw scores called LOGITS, one per class. Classes: ["cat", "dog", "bird"].
logits = np.array([2.0, 1.0, 0.1])                        # shape: (3,) = (n_classes,)


def softmax(z, temperature=1.0):
    """exp each score, divide by the total -> positive numbers that sum to 1."""
    z = z / temperature                                  # temperature rescales logits (module 05 revisits this!)
    z = z - z.max()                                      # numerical-stability trick: doesn't change the result
    e = np.exp(z)
    return e / e.sum()


probs = softmax(logits)                                 # shape: (3,)
print("softmax:", probs.round(3), "sum =", probs.sum())

# Cross-entropy: -log(probability the model gave to the CORRECT class). Confident & right => ~0, confident & wrong => huge.
true_class = 0                                           # the answer is "cat"
loss = -np.log(probs[true_class])
print("loss if truth=cat :", round(float(loss), 4))
print("loss if truth=bird:", round(float(-np.log(probs[2])), 4), "(model gave bird only 10%, so loss is bigger)")

# Same thing in PyTorch. NOTE: F.cross_entropy takes RAW logits (it applies softmax itself) -- a classic bug is to softmax twice.
torch_loss = F.cross_entropy(torch.tensor(logits).unsqueeze(0), torch.tensor([true_class]))  # logits (1, 3), target (1,)
print("torch cross_entropy:", round(torch_loss.item(), 4), "(matches numpy)")

# Temperature: low = sharper (more confident), high = flatter (more random)
for t in (0.5, 1.0, 5.0):
    print(f"temperature {t}: {softmax(logits, t).round(3)}")
# TRY IT: set logits = [10, 1, 0.1]. Probability of cat -> ~1 and the loss -> ~0.
# BREAK IT: pass probabilities (not logits) to F.cross_entropy and compare the number: it's wrong but raises no error.
