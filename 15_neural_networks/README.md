# 15 — Neural Networks (the deep dive between ML foundations and transformers)

## The story of this module
**Pain point:** [module 03](../03_ml_foundations/README.md) trained a *linear* model with gradient descent, and [module 04](../04_llm_fundamentals/README.md) hands you a transformer with attention, RMSNorm, SwiGLU, AdamW, bf16 and a warm-up schedule as if they were obvious. Between the two there is a gap: why a neuron needs a nonlinearity, what `backward()` really does, why deep nets stall, where `He init` comes from, why LLMs use RMSNorm and bf16. **Spark:** build each idea in the smallest runnable form and *measure* it: a perceptron that fails on XOR (01), a two-layer net with hand-derived backprop checked against autograd (02), your own autograd engine (03), gradients that vanish through 20 sigmoid layers (04), activation statistics through 50 layers and norms written by hand (05), six optimisers matched to `torch.optim` (06), nine regularisers on one overfitting net (07), convolutions written as loops (08), RNNs and the bottleneck that made attention necessary (09), and the number formats that decide whether a run ends in `nan` (10). **What you build:** every piece is a few dozen lines you can rewrite on a whiteboard. **Cliffhanger:** a recurrent encoder compresses a sequence into one 32-number vector and copies 24 tokens at chance level (0.126); letting the decoder look back at every input position is attention: [04_llm_fundamentals/04_attention_from_scratch](../04_llm_fundamentals/04_attention_from_scratch/WALKTHROUGH.md).

**Reading position:** after [03_ml_foundations](../03_ml_foundations/README.md) and **before** [04_llm_fundamentals](../04_llm_fundamentals/README.md) (recommended order: 00 -> 01 -> 02 -> 03 -> **15** -> 04 -> 05 -> 16 -> 21 -> 06 ...).

**Short on time?** Read [INTERVIEW_GIST.md](INTERVIEW_GIST.md). **Practice:** [exercises.md](exercises.md) (10 graded exercises, NumPy first then PyTorch) with verified answers in [solutions/solutions.md](solutions/solutions.md).

## Why this matters
Interviewers for ML-engineer roles ask the 'why' behind the defaults: *Why ReLU and not sigmoid? Why does Adam need bias correction? What is the difference between weight decay and L2? Why LayerNorm in transformers and BatchNorm in CNNs? Why bf16? How do you debug a NaN?* Each has a measured answer in this module. It is also the vocabulary for reading model cards and training logs in modules 05, 06 and 16 onward.

## Where this fits
**Builds on:** [../03_ml_foundations/](../03_ml_foundations/README.md): gradient descent (02), tensors and autograd (03), the PyTorch loop (04), overfitting (05), softmax and cross-entropy (06). This module does *not* repeat them: it uses them and goes deeper. Explicit hand-offs:
- *You already saw gradient descent and the update rule in [03/02](../03_ml_foundations/02_linear_regression_numpy/WALKTHROUGH.md); here we go deeper on momentum, per-parameter scaling and schedules ([step 06](06_optimizers_and_schedules/WALKTHROUGH.md)).*
- *You already saw the PyTorch training loop in [03/04](../03_ml_foundations/04_pytorch_training_loop/WALKTHROUGH.md); here we go deeper on what `backward()` computes ([02](02_mlp_backprop_from_scratch/WALKTHROUGH.md), [03](03_tiny_autograd_engine/WALKTHROUGH.md)) and on clipping, schedulers and autocast inside that loop ([10](10_precision_and_training_stability/WALKTHROUGH.md)).*
- *You already saw tensors and autograd in [03/03](../03_ml_foundations/03_pytorch_tensors_autograd/WALKTHROUGH.md); step 03 here rebuilds the engine behind `.backward()`.*
- *You already saw overfitting and the train/validation gap in [03/05](../03_ml_foundations/05_overfitting_train_val/WALKTHROUGH.md); here we go deeper on the remedies ([step 07](07_regularization/WALKTHROUGH.md)).*
- *You already saw softmax and cross-entropy in [03/06](../03_ml_foundations/06_softmax_cross_entropy/WALKTHROUGH.md); step 02 reuses the `(p - onehot)/N` logit gradient and checks it by finite differences.*
- *AdamW is used in the LLM training loop ([06/02](../06_llm_training_and_lora/02_training_loop/WALKTHROUGH.md)); step 06 here is where you derive why it differs from L2.*

**Leads to:**
- [../04_llm_fundamentals/](../04_llm_fundamentals/README.md): attention (step 09 here motivates it), the transformer block (residuals, RMSNorm, SwiGLU from steps 04 and 05).
- [../05_llm_parameters/](../05_llm_parameters/README.md) and [../06_llm_training_and_lora/](../06_llm_training_and_lora/README.md): optimiser memory, schedules, mixed precision, quantisation (steps 06 and 10).
- Later modules on OCR and vision (module 18, when present) reuse convolutions (step 08).

## Concept map
```
 neuron: z = x.w + b ; a = f(z)                       (01)  one hyperplane; XOR needs depth
        |
        v
 MLP: stacked layers  --forward-->  loss              (02)
        |                                    chain rule, cached activations
        v  <-------------- backward (backprop) ------- (02 by hand, 03 as an engine = autograd)
 what flows through depth?
   activations f   -> slopes decide gradient size     (04)  sigmoid vanishes, ReLU/GELU/SiLU hold, dead ReLU
   init + norm     -> variance stays ~1               (05)  Xavier/He, BatchNorm/LayerNorm/RMSNorm, residuals
 how do we walk downhill?
   optimiser       -> SGD, momentum, Adam, AdamW      (06)  + warm-up/cosine; Muon, schedule-free
 how do we generalise?
   regularisation  -> L2/decay, dropout, early stop,  (07)  augmentation, label smoothing
 other structures
   space           -> convolution, pooling            (08)
   time            -> RNN/LSTM/GRU, bottleneck        (09)  ---> attention (module 04)
 numerics          -> fp32/bf16/fp16/fp8, scaling,    (10)  clipping, NaN debugging, autocast
```

## Prerequisites
- Python and NumPy basics ([00](../00_python_foundations/README.md)); matrix shapes and broadcasting ([03/01](../03_ml_foundations/01_numpy_and_data/WALKTHROUGH.md)).
- Gradient descent and a PyTorch training loop ([03](../03_ml_foundations/README.md)): you should be able to read `loss.backward(); opt.step(); opt.zero_grad()`.
- High-school calculus: the chain rule and derivatives of `exp`, `log`, `tanh`.
- Hardware: CPU only, each step under about 2 minutes.

## Study path
| Step | Folder | You learn | Run | Time |
|---|---|---|---|---|
| 01 | [01_neuron_and_perceptron](01_neuron_and_perceptron/) | neuron, perceptron rule, linear separability, XOR; logistic regression from scratch (sigmoid, BCE, gradient `(p - y) x`, boundary, calibration, sklearn check) | `python main.py` | 90 min |
| 02 | [02_mlp_backprop_from_scratch](02_mlp_backprop_from_scratch/) | MLP, backprop by hand with shapes, gradient check, vs autograd | `python main.py` | 120 min |
| 03 | [03_tiny_autograd_engine](03_tiny_autograd_engine/) | computational graph, topological sort, reverse-mode autodiff | `python main.py` | 90 min |
| 04 | [04_activations_and_vanishing_gradients](04_activations_and_vanishing_gradients/) | activation zoo, vanishing gradients, dead ReLU, SwiGLU | `python main.py` | 75 min |
| 05 | [05_initialization_and_normalization](05_initialization_and_normalization/) | Xavier/He, BatchNorm/LayerNorm/RMSNorm by hand, residuals | `python main.py` | 90 min |
| 06 | [06_optimizers_and_schedules](06_optimizers_and_schedules/) | SGD to AdamW from scratch, warm-up + cosine, Muon/schedule-free | `python main.py` | 90 min |
| 07 | [07_regularization](07_regularization/) | L2 vs decoupled decay, dropout, early stopping, augmentation, label smoothing | `python main.py` | 75 min |
| 08 | [08_cnn_convolutions](08_cnn_convolutions/) | convolution, pooling, receptive field, CNN vs MLP | `python main.py` | 75 min |
| 09 | [09_rnn_lstm_and_why_attention](09_rnn_lstm_and_why_attention/) | RNN/LSTM/GRU, BPTT, the fixed-vector bottleneck | `python main.py` (1 to 2 min) | 90 min |
| 10 | [10_precision_and_training_stability](10_precision_and_training_stability/) | fp32/bf16/fp16/fp8, loss scaling, clipping, NaN debugging, autocast | `python main.py` | 75 min |

Plan: about 14 hours for the ten steps (see the Time column) plus [exercises.md](exercises.md). Run all from the repo root with `HF_HUB_OFFLINE=1 .venv/bin/python 15_neural_networks/<step>/main.py`. Only numpy, torch and scikit-learn are needed (step 08 and 07 use scikit-learn's bundled datasets).

## Concepts, one by one
- **Neuron:** `a = f(x.w + b)`. One neuron = one hyperplane; the perceptron rule fixes mistakes and converges iff the data are linearly separable. XOR needs a hidden layer.
- **Backprop:** chain rule from the loss backwards, reusing cached forward values. For softmax cross-entropy the logit gradient is `(p - onehot) / N`. Verify with finite differences.
- **Autograd:** record each operation with its local derivative, walk the graph in reverse topological order, *add* gradients (`+=`).
- **Activations:** sigmoid slope <= 0.25 so 20 layers give a 1e-13 gradient ratio; ReLU family holds with He init; GELU/SiLU are smooth ReLUs; SwiGLU is `silu(xW) * (xV)` with hidden size `8/3 d`.
- **Init and norm:** keep the variance constant (`1/n_in`, Xavier `2/(n_in+n_out)`, He `2/n_in`); BatchNorm normalises over the batch (train/eval differ), LayerNorm per sample, RMSNorm drops mean and bias; residuals give a gradient highway; pre-LN keeps it clean.
- **Optimisers:** momentum smooths, RMSProp rescales per parameter, Adam does both with bias correction, AdamW decouples decay; warm-up then cosine decay; Adam costs two extra fp32 tensors per parameter.
- **Regularisation:** measure the train-val gap; L2 under Adam is not weight decay; dropout needs `eval()` at inference; early stopping makes the validation score optimistic; label smoothing fixes confidence, not accuracy.
- **Convolution:** shared local kernels; output `floor((n+2p-k)/s)+1`; parameters independent of image size; limited shift robustness.
- **Recurrence:** `h_t = f(h_{t-1}, x_t)`, BPTT is a product of Jacobians; gates and additive cell state help; a fixed-size summary is a bottleneck; attention removes it.
- **Numerics:** bf16 = fp32 range, fp16 needs loss scaling; absorption, overflow, underflow; clip gradients; find NaNs with `isfinite` and anomaly mode.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| weights | parameters, coefficients, `theta`, 'the model' | Weights are one kind of parameter | Parameters also include biases, norm gains, embeddings |
| bias | intercept, offset, `b` | Yes | Unrelated to statistical bias or fairness bias |
| MLP | feed-forward network, dense network, fully connected network, FFN, linear layers | Same family | In a transformer, 'MLP' / 'FFN' is the per-token two- or three-matrix block |
| linear layer | dense, fully connected, `nn.Linear`, affine layer | Same | `y = x W^T + b`; PyTorch stores `W` as `(out, in)` |
| logits | pre-softmax scores, pre-activations of the last layer, raw scores | Same | Logits are *before* softmax; probabilities after |
| pre-activation | `z`, net input, logit (loosely) | Same | `a = f(z)` is the activation |
| backprop | reverse-mode autodiff, backward pass, autograd | Backprop is reverse-mode autodiff on a network | Backprop = algorithm; autograd = the library |
| gradient descent | SGD, steepest descent | SGD adds mini-batch noise | 'SGD' is often used for any mini-batch gradient method |
| epoch / iteration / step | pass / update / update | Step = iteration | Epoch = pass over the data = N / batch steps |
| learning rate | step size, `lr`, `eta` | Same | Schedules change it over time |
| weight decay | L2 regularisation, ridge, `wd` | Same for SGD, **different for Adam** | AdamW decouples the decay from the gradient |
| LayerNorm / RMSNorm / BatchNorm | norm layers | Different | Batch: over the batch; Layer: over features; RMS: no mean, no bias |
| residual connection | skip connection, shortcut, identity path | Same | 'Residual stream' is the transformer name for the running sum |
| SiLU | Swish | Same (beta = 1) | Swish may have a learned beta |
| SwiGLU | gated MLP, GLU variant | One member of the GLU family | GeGLU, ReGLU use other gates |
| dropout | DropConnect, DropPath, stochastic depth | Related | Dropout zeros activations, DropPath zeros whole branches |
| He init | Kaiming init | Same | Variance `2/fan_in` |
| Xavier init | Glorot init | Same | Variance `2/(fan_in+fan_out)` |
| bf16 / fp16 | bfloat16 / half | Different formats | Same width, bf16 has fp32's range, fp16 has more precision |
| mixed precision | AMP, autocast | Same idea | Not the same as quantisation |
| RNN / LSTM / GRU | recurrent nets, gated RNNs | Family | LSTM has a cell state and 3 gates, GRU 2 gates |
| attention | soft lookup, content-based addressing | Same family | Bahdanau attention (decoder over encoder) vs self-attention (sequence over itself) |
| convolution | cross-correlation, `Conv2d`, filter | Libraries do cross-correlation | Kernel is not flipped |

## Interview Q&A (short answers)
1. **Why can't one neuron learn XOR?** XOR is not linearly separable; a neuron is one hyperplane. Measured best of 20,000 random lines: 3/4. One hidden layer fixes it.
2. **What does backprop do?** Applies the chain rule from loss to parameters using cached forward values; for softmax cross-entropy the logit gradient is `(p - onehot)/N`. Shapes: a weight gradient has the weight's shape, `dW = input^T @ dout`.
3. **How do you check a gradient implementation?** Centered finite differences in float64, relative error below 1e-6 (we got 5.5e-9), or compare with autograd (3e-17).
4. **What is autograd?** A graph of operations with local derivatives; backward walks it in reverse topological order and accumulates gradients with `+=`.
5. **Why not sigmoid in deep nets?** Slope at most 0.25; 20 layers gave a gradient ratio of 4e-13 (1e-10 even with He init). ReLU/GELU/SiLU hold (ratio about 4 to 6 with He).
6. **What is a dead ReLU and how do you avoid it?** A unit negative for all inputs gets zero gradient. Lower learning rate, good init, leaky ReLU/GELU/SiLU.
7. **Derive He init.** `Var(y) = n_in Var(w) Var(x)`; ReLU halves the second moment, so `Var(w) = 2/n_in`.
8. **BatchNorm vs LayerNorm vs RMSNorm?** Axis (batch vs features), train/eval difference (BatchNorm only), RMSNorm drops mean-centring and bias; transformers use LayerNorm/RMSNorm because batches are small and variable and sequences vary.
9. **Why residual connections?** Gradient through `h + f(h)` contains the identity; 50-layer sigmoid net: 0.0 plain vs ratio 1.27 residual.
10. **Adam vs SGD?** Adam adds momentum, per-parameter scaling and bias correction; it costs 2 extra state tensors; lr acts like a max per-step change. AdamW fixes decay.
11. **L2 vs weight decay?** Same for SGD; under Adam the L2 gradient is divided by `sqrt(v)`, so they differ: `Adam(wd=0.05)` crushed our weights (norm 1.8) while `AdamW(wd=0.05)` hardly changed them (29.7).
12. **Why warm-up?** Early Adam second-moment estimates are noisy and weights random; a large lr can diverge.
13. **How does dropout work at inference?** It is the identity; training scales survivors by `1/(1-p)`; forgetting `eval()` makes outputs random.
14. **Why do CNNs use fewer parameters?** Weight sharing: 1,792 weights for 64 3x3 filters vs 150 million for a dense layer on a 224x224x3 image.
15. **Why were LSTMs invented, and why did attention replace them?** Vanishing gradients in BPTT, cured by an additive cell state; still one fixed vector and sequential computation; attention gives direct access and parallelism.
16. **bf16 vs fp16?** bf16 keeps fp32's exponent range (no loss scaling), fp16 has more mantissa but overflows at 65,504.
17. **How do you debug a NaN?** Find the first non-finite tensor, anomaly mode, check `log(0)`, `sqrt(0)`, `exp` overflow, lower lr/clip, compare fp32.
18. **What is label smoothing?** Softer targets that cap confidence; lowers the validation loss and improves calibration, accuracy roughly unchanged.

## Self-check
(Worked practice for these: [exercises.md](exercises.md), [solutions/solutions.md](solutions/solutions.md).)

- [ ] I can draw why XOR needs two layers and hand-set the weights
- [ ] I can write forward and backward for a 2-layer MLP with shapes, and a gradient check
- [ ] I can explain reverse-mode autodiff and why gradients accumulate
- [ ] I can state the max slope of sigmoid/tanh/ReLU and the consequence for deep nets
- [ ] I can derive Xavier and He variances and write BatchNorm/LayerNorm/RMSNorm
- [ ] I can write Adam and AdamW and say how they differ from L2
- [ ] I can write warm-up + cosine and explain each phase
- [ ] I can implement dropout with `1/(1-p)` and explain train vs eval
- [ ] I can compute conv output size, parameters and receptive field
- [ ] I can write LSTM gates and explain the bottleneck that led to attention
- [ ] I can give the range and precision of fp32/fp16/bf16 and a NaN checklist

## Common mistakes & gotchas
- Missing `/N` in a hand gradient (uniform error across all parameters in the gradient check).
- Forgetting `zero_grad` / not clearing `.grad` (gradients accumulate by design).
- `retain_grad()` missing on a non-leaf tensor, so `.grad` is `None`.
- Zero-initialising weights (identical units forever); using Xavier with ReLU stacks.
- `BatchNorm` in train mode with batch size 1; forgetting `model.eval()` with dropout/BatchNorm.
- Applying L2 through Adam and expecting AdamW behaviour; decaying biases and norm gains.
- Reporting the validation score that early stopping selected on (optimistic).
- Wrong `Linear` input size after convolutions (print the shape once).
- Comparing optimisers or init schemes on one seed; differences under about 1 point on 450 to 2000 test examples are noise.
- Summing in float16; softmax without max-subtraction; fp16 training without loss scaling.

## References (checked Nov 2026)
Pages verified in this session by search (titles, signatures, authors as returned):
- PyTorch `torch.optim.Muon` documentation: https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html (docs for 2.9, 2.11, 2.13 exist; this machine has torch 2.14.1 with `torch.optim.Muon`)
- Schedule-Free optimisers (code): https://github.com/facebookresearch/schedule_free
- Defazio et al., 'The Road Less Scheduled', NeurIPS 2024: https://proceedings.neurips.cc/paper_files/paper/2024/hash/136b9a13861308c8948cd308ccd02658-Abstract.html (arXiv 2405.15682 per the repository README)

Classic sources cited from memory (stable arXiv ids; open them to confirm before quoting in an interview, not re-searched here):
- Adam: https://arxiv.org/abs/1412.6980 · AdamW (Decoupled Weight Decay Regularization): https://arxiv.org/abs/1711.05101
- He et al., Delving Deep into Rectifiers (He init): https://arxiv.org/abs/1502.01852 · ResNet: https://arxiv.org/abs/1512.03385
- Batch Normalization: https://arxiv.org/abs/1502.03167 · Layer Normalization: https://arxiv.org/abs/1607.06450 · RMSNorm: https://arxiv.org/abs/1910.07467
- GLU Variants Improve Transformer (SwiGLU): https://arxiv.org/abs/2002.05202
- Seq2seq with attention (Bahdanau et al.): https://arxiv.org/abs/1409.0473 · Attention Is All You Need: https://arxiv.org/abs/1706.03762
- Lion, Symbolic Discovery of Optimization Algorithms: https://arxiv.org/abs/2302.06675 (the existence of `torch.optim.Lion` was checked locally: `False` on torch 2.14.1)
- Karpathy's micrograd (the model for step 03): https://github.com/karpathy/micrograd
- PyTorch docs: `torch.finfo`, `torch.autocast`, `torch.autograd.detect_anomaly`, `torch.nn.utils.clip_grad_norm_` at https://docs.pytorch.org/docs/stable/ (look up the current page; signatures here were exercised on torch 2.14.1)
