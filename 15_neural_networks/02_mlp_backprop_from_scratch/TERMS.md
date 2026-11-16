# Terms — An MLP and backpropagation from scratch

Folder: `15_neural_networks/02_mlp_backprop_from_scratch` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [15 README](../README.md)

**Terms in this folder:** [Multi-layer perceptron (MLP, feed-forward network, dense network)](#multi-layer-perceptron-mlp-feed-forward-network-dense-network) · [Forward pass](#forward-pass) · [Backpropagation (backward pass, reverse-mode autodiff)](#backpropagation-backward-pass-reverse-mode-autodiff) · [Chain rule](#chain-rule) · [Cache (saved activations)](#cache-saved-activations) · [Gradient of softmax cross-entropy (p minus onehot)](#gradient-of-softmax-cross-entropy-p-minus-onehot) · [Shape rule for gradients](#shape-rule-for-gradients) · [Tanh activation](#tanh-activation) · [Numerical gradient check (gradcheck)](#numerical-gradient-check-gradcheck) · [Relative error](#relative-error) · [Autograd](#autograd) · [Leaf tensor and .grad](#leaf-tensor-and-grad) · [Full-batch gradient descent](#full-batch-gradient-descent) · [Symmetry breaking (why random init)](#symmetry-breaking-why-random-init) · [Hidden width (n_hidden)](#hidden-width-nhidden) · [Numerically stable softmax](#numerically-stable-softmax) · [Fancy indexing](#fancy-indexing) · [float64 for gradient checks](#float64-for-gradient-checks)

---

## Multi-layer perceptron (MLP, feed-forward network, dense network)  `(concept)`
**1. Definition:** A network of stacked fully connected layers with nonlinear activations between them: `logits = act(X W1 + b1) W2 + b2`.
**2. In plain words:** The plainest neural net: every unit sees every unit of the previous layer.
**3. Example to quote:** > "The transformer block's 'feed-forward' part is an MLP applied to each token."
**Seen in:** `main.py`, `forward`

## Forward pass  `(concept)`
**1. Definition:** Computing the network's output (and loss) from the inputs, layer by layer.
**2. In plain words:** Run the data through.
**3. Example to quote:** > "`logits, cache = forward(X, p)`"
**Seen in:** `main.py`, block 2

## Backpropagation (backward pass, reverse-mode autodiff)  `(concept)`
**1. Definition:** An algorithm that computes the gradient of a scalar loss with respect to all parameters by applying the chain rule from the output back to the inputs, reusing forward values.
**2. In plain words:** Blame flows backwards: each layer receives the error from the next and passes it on.
**3. Example to quote:** > "One backward sweep costs about as much as 2 forward passes, not one per parameter."
**Seen in:** `main.py`, `backward`

## Chain rule  `(concept)`
**1. Definition:** If `y = f(g(x))` then `dy/dx = f'(g(x)) * g'(x)`; for vectors, multiply Jacobians.
**2. In plain words:** A derivative of a pipeline is the product of the derivatives of each stage.
**3. Example to quote:** > "`dz1 = dh * (1 - h**2)`"
**Seen in:** `main.py`, block 4

## Cache (saved activations)  `(concept)`
**1. Definition:** Intermediate forward values kept so the backward pass does not have to recompute them.
**2. In plain words:** Notes taken on the way forward and used on the way back; they are why training needs far more memory than inference.
**3. Example to quote:** > "`return logits, (X, h)`"
**Seen in:** `main.py`, block 2

## Gradient of softmax cross-entropy (p minus onehot)  `(concept)`
**1. Definition:** For logits `z`, probabilities `p = softmax(z)` and true class `y`, `dL/dz_j = p_j - 1[j = y]`.
**2. In plain words:** 'What you predicted minus what was true'.
**3. Example to quote:** > "`dlogits[np.arange(N), y] -= 1`"
**Seen in:** `main.py`, block 4

## Shape rule for gradients  `(concept)`
**1. Definition:** The gradient of a parameter has the same shape as the parameter; for `Y = A B`: `dA = dY B^T`, `dB = A^T dY`.
**2. In plain words:** If you are unsure which matrix to transpose, make the shapes work.
**3. Example to quote:** > "`grads['W1'] = X.T @ dz1  # (n_in, N) @ (N, n_hidden)`"
**Seen in:** `main.py`, block 4

## Tanh activation  `(concept)`
**1. Definition:** `tanh(z) = (e^z - e^-z) / (e^z + e^-z)`, range (-1, 1), derivative `1 - tanh(z)^2`.
**2. In plain words:** A smooth S-curve centred on zero.
**3. Example to quote:** > "`h = np.tanh(z1)`"
**Seen in:** `main.py`, block 2

## Numerical gradient check (gradcheck)  `(concept)`
**1. Definition:** Comparing analytic gradients with centered finite differences `(L(w+eps) - L(w-eps)) / 2eps`.
**2. In plain words:** The slow but obviously-correct way of getting a gradient, used as a test.
**3. Example to quote:** > "Relative error below 1e-6 = correct."
**Seen in:** `main.py`, block 5

## Relative error  `(concept)`
**1. Definition:** `|a - b| / (|a| + |b| + tiny)`; compares numbers of different magnitude fairly.
**2. In plain words:** 'How far off, in proportion?'
**3. Example to quote:** > "Printed `5.51e-09` for `W1`."
**Seen in:** `main.py`, block 5

## Autograd  `(concept)`
**1. Definition:** A library that records operations on tensors in a graph and computes gradients by reverse-mode differentiation on demand.
**2. In plain words:** Backprop done by the library for any program.
**3. Example to quote:** > "`t_loss.backward()` filled `.grad` and matched ours to 3e-17."
**Seen in:** `main.py`, block 6

## Leaf tensor and .grad  `(library)`
**1. Definition:** A tensor you created with `requires_grad=True`; after `backward()` its `.grad` holds `dL/dtensor`.
**2. In plain words:** The parameters are the leaves of the graph; their gradient is what the optimizer uses.
**3. Example to quote:** > "`tp['W1'].grad`"
**Seen in:** `main.py`, block 6

## Full-batch gradient descent  `(concept)`
**1. Definition:** Update parameters using the gradient over the whole dataset each step.
**2. In plain words:** One careful step per pass.
**3. Example to quote:** > "Here `steps` = number of epochs."
**Seen in:** `main.py`, `train`

## Symmetry breaking (why random init)  `(concept)`
**1. Definition:** If all weights in a layer start equal, all units in it get equal gradients and stay equal forever; random init makes them differ.
**2. In plain words:** Identical workers given identical instructions stay identical.
**3. Example to quote:** > "Step 05 shows the zero-init failure."
**Seen in:** `main.py`, `init_params`

## Hidden width (n_hidden)  `(concept)`
**1. Definition:** The number of units in a hidden layer; a hyperparameter.
**2. In plain words:** How many bends the net can make.
**3. Example to quote:** > "Spiral accuracy: 1 unit 0.573, 8 units 0.990."
**Seen in:** `main.py`, experiments

## Numerically stable softmax  `(concept)`
**1. Definition:** Subtract the row maximum from the logits before `exp`; the result is mathematically identical but cannot overflow.
**2. In plain words:** Shifting all scores by the same amount does not change who wins.
**3. Example to quote:** > "`np.exp(logits - logits.max(axis=1, keepdims=True))`"
**Seen in:** `main.py`, block 3

## Fancy indexing  `(syntax)`
**1. Definition:** Indexing an array with integer arrays to pick elements (one per pair of indices).
**2. In plain words:** Pick the true class's probability from every row at once.
**3. Example to quote:** > "`probs[np.arange(len(y)), y]`"
**Seen in:** `main.py`, block 3

## float64 for gradient checks  `(concept)`
**1. Definition:** Double precision (about 15-16 digits) so finite-difference noise stays far below the signal.
**2. In plain words:** In float32 the rounding noise in `(up - down)` is much larger, so the check is far less sharp.
**3. Example to quote:** > "`dtype=torch.float64`"
**Seen in:** `main.py`, block 6
