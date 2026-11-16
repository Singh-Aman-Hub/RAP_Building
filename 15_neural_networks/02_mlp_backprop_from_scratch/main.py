"""
GOAL:  Write a 2-layer MLP in numpy: forward pass, loss, and the backward pass derived by hand with shapes.
       Prove the gradients right with a numerical gradient check and with torch autograd, then train the
       net on XOR and on a 3-class spiral that no single neuron could ever separate.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 15_neural_networks/02_mlp_backprop_from_scratch/main.py
EXPECTED OUTPUT (shortened):
    gradient check (numeric vs analytic): max relative error ~5e-9
    torch autograd vs hand backprop: max abs difference ~3e-17
    XOR predictions: [0, 1, 1, 0] target: [0, 1, 1, 0]
    spiral train accuracy: 0.993
NEEDS: Python + numpy + torch. A few seconds on CPU.
"""

import numpy as np
import torch

rng = np.random.default_rng(0)


# [1] Parameters: two weight matrices and two bias vectors
def init_params(n_in, n_hidden, n_out, rng):
    return {
        "W1": rng.normal(0, 0.5, size=(n_in, n_hidden)),   # (n_in, n_hidden)
        "b1": np.zeros(n_hidden),                           # (n_hidden,)
        "W2": rng.normal(0, 0.5, size=(n_hidden, n_out)),   # (n_hidden, n_out)
        "b2": np.zeros(n_out),                              # (n_out,)
    }


# [2] Forward pass: linear -> tanh -> linear, keeping what the backward pass will need
def forward(X, p):
    z1 = X @ p["W1"] + p["b1"]       # (N, n_hidden)  pre-activation of the hidden layer
    h = np.tanh(z1)                  # (N, n_hidden)  hidden activations
    logits = h @ p["W2"] + p["b2"]   # (N, n_out)     raw class scores
    return logits, (X, h)            # cache: inputs and hidden values are reused by backward


# [3] Loss: softmax then cross-entropy, averaged over the batch (module 03 step 06)
def softmax(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))   # subtract the row max: numerically stable
    return e / e.sum(axis=1, keepdims=True)                   # (N, n_out) probabilities per row


def loss_fn(logits, y):
    probs = softmax(logits)
    return -np.log(probs[np.arange(len(y)), y]).mean(), probs  # mean of -log p(true class)


# [4] Backward pass: the chain rule applied layer by layer, from the loss back to the inputs
def backward(y, probs, cache, p):
    X, h = cache
    N = len(y)
    dlogits = probs.copy()                      # (N, n_out)
    dlogits[np.arange(N), y] -= 1               # softmax + cross-entropy gradient is (p - onehot)
    dlogits /= N                                # because the loss is a MEAN over N samples
    grads = {}
    grads["W2"] = h.T @ dlogits                 # (n_hidden, N) @ (N, n_out) -> (n_hidden, n_out)
    grads["b2"] = dlogits.sum(axis=0)           # (n_out,)  the bias touches every sample
    dh = dlogits @ p["W2"].T                    # (N, n_out) @ (n_out, n_hidden) -> (N, n_hidden)
    dz1 = dh * (1 - h ** 2)                     # tanh'(z) = 1 - tanh(z)^2, elementwise
    grads["W1"] = X.T @ dz1                     # (n_in, N) @ (N, n_hidden) -> (n_in, n_hidden)
    grads["b1"] = dz1.sum(axis=0)               # (n_hidden,)
    return grads


# [5] Numerical gradient check: nudge each parameter by +-eps and compare with the backward pass
def total_loss(X, y, p):
    return loss_fn(forward(X, p)[0], y)[0]


X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([0, 1, 1, 0])
p_check = init_params(2, 4, 2, np.random.default_rng(1))
loss, probs = loss_fn(forward(X_xor, p_check)[0], y_xor)
analytic = backward(y_xor, probs, forward(X_xor, p_check)[1], p_check)

eps = 1e-5
worst = 0.0
for name, arr in p_check.items():
    numeric = np.zeros_like(arr)
    for idx in np.ndindex(arr.shape):                 # every single number in this parameter
        old = arr[idx]
        arr[idx] = old + eps; up = total_loss(X_xor, y_xor, p_check)
        arr[idx] = old - eps; down = total_loss(X_xor, y_xor, p_check)
        arr[idx] = old                                # restore
        numeric[idx] = (up - down) / (2 * eps)        # centered finite difference
    rel = np.abs(numeric - analytic[name]) / (np.abs(numeric) + np.abs(analytic[name]) + 1e-12)
    worst = max(worst, rel.max())
    print(f"  {name}: shape {arr.shape}, max relative error {rel.max():.2e}")
print(f"gradient check (numeric vs analytic): max relative error {worst:.2e}")


# [6] Same network in torch: autograd must give the same gradients
tp = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in p_check.items()}
t_logits = torch.tanh(torch.tensor(X_xor) @ tp["W1"] + tp["b1"]) @ tp["W2"] + tp["b2"]
t_loss = torch.nn.functional.cross_entropy(t_logits, torch.tensor(y_xor))
t_loss.backward()                                     # reverse-mode autodiff: fills .grad on every leaf
diff = max(np.abs(tp[k].grad.numpy() - analytic[k]).max() for k in tp)
print(f"torch loss {t_loss.item():.6f} vs numpy loss {loss:.6f}")
print(f"torch autograd vs hand backprop: max abs difference {diff:.2e}")


# [7] Training loop: full-batch gradient descent, update = parameter - lr * gradient
def train(X, y, n_hidden, lr, steps, rng, log_every):
    p = init_params(X.shape[1], n_hidden, int(y.max()) + 1, rng)
    for step in range(steps + 1):
        logits, cache = forward(X, p)
        loss, probs = loss_fn(logits, y)
        if step % log_every == 0:
            print(f"  step {step:>5}  loss {loss:.4f}  acc {(logits.argmax(1) == y).mean():.2f}")
        grads = backward(y, probs, cache, p)
        for k in p:
            p[k] -= lr * grads[k]                      # in-place gradient-descent update
    return p


print("XOR with a 4-unit hidden layer:")
p_xor = train(X_xor, y_xor, n_hidden=4, lr=0.5, steps=1000, rng=np.random.default_rng(0), log_every=250)
print("XOR predictions:", forward(X_xor, p_xor)[0].argmax(1).tolist(), "target:", y_xor.tolist())


# [8] A harder dataset: three interleaved spiral arms (not separable by any line)
def make_spiral(points_per_class=100, classes=3, noise=0.2, rng=rng):
    X = np.zeros((points_per_class * classes, 2))
    y = np.zeros(points_per_class * classes, dtype=int)
    for c in range(classes):
        ix = range(points_per_class * c, points_per_class * (c + 1))
        r = np.linspace(0.0, 1, points_per_class)                                  # radius grows along the arm
        t = np.linspace(c * 4, (c + 1) * 4, points_per_class) + rng.normal(0, noise, points_per_class)
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = c
    return X, y


Xs, ys = make_spiral()
print("spiral data:", Xs.shape, "labels:", np.bincount(ys))
p_sp = train(Xs, ys, n_hidden=64, lr=1.0, steps=3000, rng=np.random.default_rng(0), log_every=1000)
print(f"spiral train accuracy: {(forward(Xs, p_sp)[0].argmax(1) == ys).mean():.3f}")

# TRY IT: change n_hidden for the spiral to 2, 8, 64, 256 and note the final accuracy.
# BREAK IT: delete the `dlogits /= N` line. The gradient check then fails by a factor of N (a scale bug).
