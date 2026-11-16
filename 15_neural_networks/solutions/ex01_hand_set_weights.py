"""
GOAL:  Exercise 1 solution. Set (not learn) the weights of neurons that compute AND, OR, NAND, then wire a
       2-layer network that computes XOR. Shows what one hyperplane can and cannot do.

RUN (from this folder):  python ex01_hand_set_weights.py
RUN (from repo root):    .venv/bin/python 15_neural_networks/solutions/ex01_hand_set_weights.py
EXPECTED OUTPUT (shortened):
    AND  [0, 0, 0, 1]
    ...
    XOR via 2 layers [0, 1, 1, 0]
NEEDS: numpy only. < 1 s on CPU.
"""
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])


def neuron(x, w, b):
    """Step-activation neuron: 1 if x.w + b > 0 else 0. x: (n, 2), w: (2,), b: scalar."""
    return (x @ np.asarray(w) + b > 0).astype(int)


# One neuron = one line w.x + b = 0. Pick the line so that only the right corners are on the positive side.
AND = neuron(X, [1, 1], -1.5)    # only (1,1) has x1+x2 > 1.5
OR = neuron(X, [1, 1], -0.5)     # anything but (0,0) has x1+x2 > 0.5
NAND = neuron(X, [-1, -1], 1.5)  # the opposite of AND
print("AND ", AND.tolist())
print("OR  ", OR.tolist())
print("NAND", NAND.tolist())

# XOR = AND(OR, NAND): the hidden layer re-maps the 4 points into a space where they ARE separable.
H = np.stack([OR, NAND], axis=1)          # (4, 2) hidden activations
XOR = neuron(H, [1, 1], -1.5)             # AND of the two hidden units
print("XOR via 2 layers", XOR.tolist())
assert XOR.tolist() == [0, 1, 1, 0]

# Why one neuron can not do it: brute force over many random lines, best accuracy is 3/4.
rng = np.random.default_rng(0)
y = np.array([0, 1, 1, 0])
best = 0
for _ in range(20000):
    w, b = rng.normal(size=2), rng.normal()
    best = max(best, (neuron(X, w, b) == y).mean())
print("best single-neuron accuracy on XOR over 20,000 random lines:", best)
assert best == 0.75
