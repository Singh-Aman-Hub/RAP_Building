# Module 15 — Solutions

Every solution is a runnable script in this folder (numpy / torch, CPU, a few seconds). Run from the repo root: `.venv/bin/python 15_neural_networks/solutions/exNN_*.py`. All were executed; the numbers quoted below are from those runs (random-data results may differ slightly under other library versions). Exercises: [../exercises.md](../exercises.md).

| # | Script | Topic |
|---|---|---|
| 1 | [ex01_hand_set_weights.py](ex01_hand_set_weights.py) | hand-set AND/OR/NAND, XOR in 2 layers |
| 2 | [ex02_backprop_gradcheck.py](ex02_backprop_gradcheck.py) | backprop with shapes, finite-difference check |
| 3 | [ex03_softmax_ce.py](ex03_softmax_ce.py) | stable softmax, log-sum-exp CE, gradient |
| 4 | [ex04_autograd_ops.py](ex04_autograd_ops.py) | sigmoid / log / division in a Value engine |
| 5 | [ex05_init_variance.py](ex05_init_variance.py) | predicted vs measured variance through 30 layers |
| 6 | [ex06_adam_vs_torch.py](ex06_adam_vs_torch.py) | Adam / AdamW, bias correction, L2 vs decoupled |
| 7 | [ex07_conv_im2col.py](ex07_conv_im2col.py) | convolution as one GEMM |
| 8 | [ex08_dropout_train_eval.py](ex08_dropout_train_eval.py) | inverted dropout, train/eval |
| 9 | [ex09_warmup_cosine_clip.py](ex09_warmup_cosine_clip.py) | warm-up + cosine + clipping in a loop |
| 10 | [ex10_precision_loss_scaling.py](ex10_precision_loss_scaling.py) | number formats, loss scaling, NaN hunting |

## 1. Hand-set weights
`AND`: `w=[1,1], b=-1.5`; `OR`: `w=[1,1], b=-0.5`; `NAND`: `w=[-1,-1], b=1.5`. XOR = `AND(OR(x), NAND(x))`: the hidden layer maps the four points to `(0,0)->(0,1)`, `(0,1)->(1,1)`, `(1,0)->(1,1)`, `(1,1)->(1,0)` where an AND neuron separates them. Output of the run: `XOR via 2 layers [0, 1, 1, 0]`, best single neuron over 20,000 random lines `0.75`. Lesson: a neuron is one hyperplane; depth re-maps the space.

## 2. Backprop and gradient check
With `N` samples, `D` inputs, `H` hidden: `d_out = (out - y)/N` is `(N,1)`; `dW2 = a1^T d_out` is `(H,1)`; `da1 = d_out W2^T` is `(N,H)`; `dz1 = da1 * (1 - a1^2)`; `dW1 = X^T dz1` is `(D,H)`; bias gradients sum over the batch axis. Run output: relative errors `3.1e-09` (W1), `4.8e-10`, `8.6e-11`, `4.5e-11`. Pass criterion: below about `1e-6` in float64 (float32 cannot do better than about `1e-3`). A missing `/N` shows up as a uniform ratio error across all parameters.

## 3. Softmax and cross-entropy
Naive softmax gives `[nan nan nan]` (`exp(1000)` overflows to `inf`, `inf/inf = nan`); subtracting the max leaves the result unchanged mathematically (`softmax(z) = softmax(z - c)`) and gives `[0.09 0.2447 0.6652]`. Cross-entropy as `logsumexp(z) - z_y` matches torch to `1e-12` and never builds probabilities. The gradient `(p - onehot)/N` matches torch to `3e-17`, and every row sums to 0 (softmax outputs sum to 1, so logit gradients sum to zero). See [03/06](../../03_ml_foundations/06_softmax_cross_entropy/WALKTHROUGH.md) for the derivation.

## 4. Autograd
New local derivatives: `sigmoid' = s(1-s)`, `log' = 1/x`; division is `a * b**-1` so it reuses `mul` and `pow` (no new backward code). The test function matches torch to `1e-12` (`df/dx = 0.5446692051`, `df/dy = 0.1692806246`). With `=` instead of `+=`, `x` would keep only the gradient of the last path that wrote it; the answer would be wrong whenever a node is used more than once (here three times).

## 5. Variance through depth
`Var(y) = n_in Var(w) E[x^2]` for independent zero-mean weights. After ReLU the second moment halves, so each layer multiplies the second moment by `n_in Var(w)/2` and the rms by its square root: `0.707` for `1/n` and for `2/(2n)` (Xavier equals `1/n` for a square layer), `1.0` for `2/n`. Measured rms after 30 layers: `2.5e-05` and `3.3e-05` (predicted `3.05e-05`), He `0.70`. He is exactly 1 only in expectation: at finite width the per-layer factor fluctuates and 30 layers compound it (a random walk in log-scale), so 0.70 is within normal variation.

## 6. Adam and AdamW
(a) First step with bias correction `0.1000` (= `lr`, independent of gradient scale), without `0.3162` (= `(1-b1)/sqrt(1-b2) = 3.16` times `lr`) because `m` and `v` start at zero. (b) Differences to torch: `4.4e-16` for both. `torch.optim.Adam(weight_decay=)` is L2 (adds `wd p` to the gradient); `AdamW` multiplies the weight by `1 - lr wd`. (c) Under L2 the decay term is divided by `sqrt(v)`, so coordinates with large gradients are barely decayed; AdamW decays every weight by the same factor. On the ill-scaled quadratic the final points were `[3.047, -2.062, 0.472]` (Adam L2) vs `[2.71, -1.908, 0.476]` (AdamW). Link to the training use: [06 step 02](../../06_llm_training_and_lora/02_training_loop/WALKTHROUGH.md).

## 7. Convolution via im2col
Each output position's `C k k` patch becomes a column; weights reshape to `(F, C k k)`; one matrix product gives `(F, H_out W_out)`. Matches `F.conv2d` exactly (error 0.0) for stride/pad `(1,0)->7x7`, `(2,1)->5x5`, `(1,1)->9x9` on a 9x9 input, k=3. Parameters: 64 filters 3x3 on RGB `1,792`; a dense layer from `224x224x3` to `224x224x64` has `483,388,358,656` (weights plus bias). This GEMM view is why GPUs run convolutions fast.

## 8. Dropout
Survivors are scaled by `1/(1-p)` so the expectation is unchanged: mean `1.0022` for input mean 1, `29.8%` zeroed, survivor value `1.4286 = 1/0.7`. After `eval()` the output equals the input; without it the same input gave two different outputs. Rule: `training` flag in `nn.Module`; the function must be the identity at inference.

## 9. Warm-up, cosine and clipping
The loop logged lr rising from `0.002` to `0.03` over 30 steps and cosine-decaying to `0.003`; gradient norm fell from `0.209` to `0.002`; final loss `0.0017`. The comparison run (constant lr `0.3`, no clip) also trained fine on this easy problem (final loss `0.0003`): a schedule and clipping are insurance for harder, deeper, noisier runs, and one seed on a toy task proves nothing about stability. `clip_grad_norm_` returns the norm measured *before* clipping, which is the number to log.

## 10. Numbers and NaNs
`finfo`: fp16 max `65504` (eps `9.8e-4`), bf16 max `3.39e38` (eps `7.8e-3`), fp32 eps `1.2e-7`. `70000` becomes `inf` in fp16 and `70144` in bf16 (coarse but finite). `256 + 1` in bf16 is `256` (absorbed: the spacing at 256 is 2). `1e-8` becomes `0` in fp16; scaled by `2^14` it is stored as `1.638e-4` and recovers `9.997e-9` after unscaling in fp32. NaN hunt: `log(0) = -inf`, `log(-1) = nan`, `sqrt(-1) = nan`, `exp(100)*exp(100) = inf`; `sqrt'(0) = inf` and `inf * 0 = nan`. bf16 needs no loss scaling because its exponent range equals fp32's, so small gradients do not underflow (they lose precision instead).
