# Interview Gist — Neural Networks (Module 15)
### Neuron → MLP/backprop → autograd → activations → init/norm → optimisers → regularisation → CNN → RNN → numerics

One-page revision sheet. Numbers are from the runs in this module. Full module: [README.md](README.md). Practice: [exercises.md](exercises.md).

## Key formulas
- Neuron: `a = f(x.w + b)`; perceptron rule `w += lr (y - y_hat) x`, converges iff linearly separable.
- Softmax CE: `L = -log p_y`, logit gradient `(p - onehot)/N`; stable form `logsumexp(z) - z_y`.
- Backprop: `dW = input^T @ dout`, `d_input = dout @ W^T`, `dz = da * f'(z)`; gradient check relative error < 1e-6 (float64).
- Slopes: sigmoid max 0.25, tanh max 1, ReLU 1 or 0. Sigmoid x 20 layers: about 1e-13.
- Init: Xavier `2/(n_in+n_out)`; He `2/n_in` (ReLU halves the second moment); `Var(y) = n_in Var(w) E[x^2]`.
- Norms: Batch (over batch, running stats), Layer (over features), RMS `x / sqrt(mean(x^2)+eps) * g`.
- Momentum `v = mu v + g`; Adam `m = b1 m + (1-b1) g; v = b2 v + (1-b2) g^2; step = lr m_hat / (sqrt(v_hat) + eps)`, bias correction `1/(1-b^t)`; AdamW `p *= (1 - lr wd)`.
- Warm-up then cosine: `0.5 (1 + cos(pi t))` to a floor. Adam state: 2 extra fp32 tensors per parameter.
- Dropout: scale survivors by `1/(1-p)` in training, identity in eval.
- Conv output `floor((n + 2p - k)/s) + 1`; parameters `C_out (C_in k^2 + 1)`.
- LSTM: `c = f c + i g`, `h = o tanh(c)`.
- Floats: fp32 max 3.4e38; fp16 max 65,504, eps 1e-3; bf16 max 3.4e38, eps 8e-3.

## Numbers to remember
| Fact | Value |
|---|---|
| Best single neuron on XOR | 3/4 correct |
| Gradient check we got | 5.5e-9 (hand), 3e-17 (autograd) |
| 20-layer sigmoid gradient ratio | about 4e-13 |
| 50-layer sigmoid net, plain vs residual | 0.0 vs ratio 1.27 |
| Adam(L2, wd .05) vs AdamW(wd .05) weight norm | 1.8 vs 29.7 |
| 64 3x3 filters on RGB vs dense on 224x224x3 | 1,792 vs 150 million |
| RNN copy of 24 tokens through 32-d vector | 0.126 (chance) |
| First Adam step w/o bias correction | 3.16 x lr |

## Ten one-line answers
1. **Why a nonlinearity?** Stacked linear maps collapse into one linear map.
2. **What does backprop do?** Chain rule from loss to parameters with cached forward values.
3. **Why `+=` in backward?** A node used on several paths receives the sum of the gradients.
4. **Why not sigmoid?** Slope <= 0.25 and saturation: vanishing gradients.
5. **Why He init?** ReLU zeroes half the signal, so variance 2/n_in preserves the scale.
6. **Why residuals?** The gradient of `x + f(x)` contains the identity: a gradient highway.
7. **LayerNorm vs BatchNorm?** Per-sample over features (no train/eval gap) vs over the batch (running stats).
8. **Adam vs SGD?** Momentum plus per-parameter scaling plus bias correction, at the cost of 2 state tensors.
9. **Weight decay vs L2?** Same for SGD; under Adam only decoupled decay (AdamW) decays uniformly.
10. **bf16 vs fp16?** bf16 has fp32's range (no loss scaling), fp16 more precision but overflows at 65,504.

## Confusable pairs
- epoch / step / iteration · logits / probabilities · pre-activation / activation
- L2 / weight decay · BatchNorm / LayerNorm / RMSNorm · dropout / DropPath
- GELU / SiLU / SwiGLU · Xavier / He · RNN / LSTM / GRU
- cross-correlation / convolution · mixed precision / quantisation · autograd / backprop

## Already seen elsewhere (do not re-learn)
Gradient descent and the PyTorch loop: [03/02](../03_ml_foundations/02_linear_regression_numpy/WALKTHROUGH.md), [03/04](../03_ml_foundations/04_pytorch_training_loop/WALKTHROUGH.md). Softmax/CE: [03/06](../03_ml_foundations/06_softmax_cross_entropy/WALKTHROUGH.md). AdamW in use: [06/02](../06_llm_training_and_lora/02_training_loop/WALKTHROUGH.md).
