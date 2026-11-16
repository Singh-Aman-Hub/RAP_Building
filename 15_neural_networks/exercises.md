# Module 15 — Exercises (neural networks)

Ten exercises, graded easy to hard. Exercises 1 to 7 use **NumPy only** (you build the mechanism), exercises 8 to 10 use **PyTorch** (you use and check the library). Rules: predict the result before you run, change one thing at a time, and write down what happened. Worked solutions, each a runnable script verified on CPU, are in [solutions/solutions.md](solutions/solutions.md). Cram sheet: [INTERVIEW_GIST.md](INTERVIEW_GIST.md). Module overview: [README.md](README.md).

Steps referred to: [01 neuron](01_neuron_and_perceptron/WALKTHROUGH.md), [02 backprop](02_mlp_backprop_from_scratch/WALKTHROUGH.md), [03 autograd](03_tiny_autograd_engine/WALKTHROUGH.md), [04 activations](04_activations_and_vanishing_gradients/WALKTHROUGH.md), [05 init and norm](05_initialization_and_normalization/WALKTHROUGH.md), [06 optimisers](06_optimizers_and_schedules/WALKTHROUGH.md), [07 regularisation](07_regularization/WALKTHROUGH.md), [08 CNN](08_cnn_convolutions/WALKTHROUGH.md), [09 RNN](09_rnn_lstm_and_why_attention/WALKTHROUGH.md), [10 precision](10_precision_and_training_stability/WALKTHROUGH.md).

## Part A — NumPy

### Exercise 1 (easy) — set the weights by hand (step 01)
Without training, choose `w` and `b` for a step-activation neuron that computes AND, OR and NAND on the four binary inputs. Then wire these into a 2-layer network that computes XOR. Finally brute-force 20,000 random lines and report the best single-neuron accuracy on XOR.
*Hint:* XOR = AND(OR, NAND). *Expected:* best accuracy 0.75.

### Exercise 2 (easy) — backprop with shapes and a gradient check (step 02)
Write forward and backward for a 1-hidden-layer tanh MLP with a mean-squared-error loss on random data. Write down the shape of every gradient before coding it. Verify all four gradients (`W1, b1, W2, b2`) with centered finite differences in float64. Which relative error counts as a pass?

### Exercise 3 (easy) — stable softmax and cross-entropy (see also [03/06](../03_ml_foundations/06_softmax_cross_entropy/WALKTHROUGH.md))
Show that the naive softmax of `[1000, 1001, 1002]` is `nan`; fix it with max-subtraction; implement cross-entropy through log-sum-exp; prove that the gradient with respect to the logits is `(p - onehot) / N` against torch.

### Exercise 4 (medium) — extend the autograd engine (step 03)
Take a micrograd-style scalar `Value` and add `sigmoid`, `log` and division. Check the gradients of `f(x, y) = log(sigmoid(x) * y + x * x) / (y + x)` against torch. `x` is used three times: what would break if you wrote `=` instead of `+=` in the backward closures?

### Exercise 5 (medium) — predict the activation variance (steps 04 and 05)
Derive `Var(y) = n_in * Var(w) * E[x^2]`. Predict the per-layer rms factor of a ReLU network for init variances `1/n`, `2/(2n)` and `2/n`; then pass random data through 30 layers of width 512 and compare prediction and measurement. Why is the He factor exactly 1 only on average?

### Exercise 6 (medium) — Adam, AdamW and bias correction (step 06; AdamW is also used in [06 step 02](../06_llm_training_and_lora/02_training_loop/WALKTHROUGH.md))
Write Adam and AdamW in numpy. (a) Show the first Adam step has size about `lr` with bias correction and about `3.16 x lr` without. (b) Match `torch.optim.Adam(weight_decay=...)` and `torch.optim.AdamW` to 1e-6 over 50 steps on an ill-scaled quadratic. (c) Explain in two sentences why the two end at different points.

### Exercise 7 (hard) — convolution as one matrix multiply (step 08)
Implement `conv2d` with im2col: unfold every patch into a column, then one GEMM. Support stride and padding, match `F.conv2d` exactly for `(1,0)`, `(2,1)`, `(1,1)`, and verify the output-size formula `floor((n + 2p - k)/s) + 1`. Count the parameters of 64 3x3 filters on RGB versus a dense layer producing the same output size.

## Part B — PyTorch

### Exercise 8 (easy) — inverted dropout and train/eval (step 07)
Write `MyDropout(nn.Module)` that scales survivors by `1/(1-p)`. Show the mean is preserved, that `eval()` makes it the identity, and that forgetting `eval()` makes one input produce different outputs.

### Exercise 9 (medium) — warm-up, cosine, clipping in a real loop (steps 06 and 10)
Train a small MLP with `AdamW`, a warm-up + cosine function plugged into `LambdaLR`, and `clip_grad_norm_` (which returns the norm *before* clipping). Log lr, gradient norm and loss. Compare with a constant lr 10x higher and no clipping and explain what you see (including why one seed proves nothing).

### Exercise 10 (hard) — numbers, loss scaling and NaN hunting (step 10)
With `torch.finfo` print range and epsilon of fp32/fp16/bf16. Show fp16 overflow (70,000), bf16 absorption (`256 + 1`), fp16 gradient underflow fixed by a loss scale, and find the first non-finite tensor in a chain (`log(0)`, `sqrt(-1)`, `exp` overflow). Why does bf16 usually need no loss scaling?

## Capstone
Train the 3-class spiral MLP from step 02 with (a) your own numpy Adam from exercise 6, (b) warm-up + cosine from exercise 9, (c) dropout from exercise 8 in numpy form. Report validation accuracy over 5 seeds with mean and standard deviation: is the difference to plain SGD larger than the seed noise?
