<!-- vocabulary draft: 15_neural_networks -->

### Perceptron (artificial neuron)
<!-- cat: ml -->
- **Technical definition:** A unit computing a = f(w.x + b), trained by the perceptron rule w += lr (y - y_hat) x; converges only if the data are linearly separable.
- **Plain English:** One weighted vote followed by a yes/no threshold.
- **Example / analogy:** A spam filter that adds up word scores and flags the email if the sum passes a line.
- **Snippet:**
```python
y_hat = int(x @ w + b > 0)
```
- **Snippet explained:** Dot product plus bias, then a step function giving 0 or 1.
- **Related terms:** [[Linear separability]], [[Activation function]], [[XOR problem]]

### Linear separability
<!-- cat: ml -->
- **Technical definition:** A dataset is linearly separable if one hyperplane w.x + b = 0 puts all examples of each class on opposite sides.
- **Plain English:** You can split the classes with one straight line (or flat plane).
- **Example / analogy:** AND and OR are separable; XOR is not.
- **Snippet:**
```python
w, b = np.array([1, 1]), -1.5   # separates AND
```
- **Snippet explained:** The line x1 + x2 = 1.5 leaves only (1,1) on the positive side.
- **Related terms:** [[Perceptron (artificial neuron)]], [[XOR problem]]

### XOR problem
<!-- cat: ml -->
- **Technical definition:** XOR (output 1 iff exactly one input is 1) is not linearly separable, so no single neuron can compute it; one hidden layer suffices.
- **Plain English:** The classic proof that depth is needed.
- **Example / analogy:** Best of 20,000 random lines classifies only 3 of the 4 XOR points.
- **Snippet:**
```python
xor = and_(or_(x), nand(x))
```
- **Snippet explained:** A hidden layer of OR and NAND makes the points separable for an AND output neuron.
- **Related terms:** [[Linear separability]], [[Activation function]]

### Hidden layer (MLP)
<!-- cat: ml -->
- **Technical definition:** A layer between input and output; an MLP (multi-layer perceptron, feed-forward network) stacks linear maps with nonlinearities, a = f(x W + b).
- **Plain English:** The middle steps where the network builds its own features.
- **Example / analogy:** Pixels to edges to shapes to digits.
- **Snippet:**
```python
h = np.tanh(X @ W1 + b1); out = h @ W2 + b2
```
- **Snippet explained:** Two layers: a tanh hidden layer then a linear output.
- **Related terms:** [[Activation function]]

### Activation function
<!-- cat: ml -->
- **Technical definition:** A nonlinear function f applied elementwise after a linear map; without it a stack of linear layers collapses to one linear layer.
- **Plain English:** The bend that lets a network draw curves.
- **Example / analogy:** ReLU max(0, z), GELU, SiLU, tanh, sigmoid.
- **Snippet:**
```python
a = np.maximum(z, 0)
```
- **Snippet explained:** ReLU zeroes negatives and passes positives unchanged.
- **Related terms:** [[ReLU and dead ReLU]], [[Sigmoid saturation]], [[GELU and SiLU (Swish)]]

### Chain rule (backpropagation through layers)
<!-- cat: ml -->
- **Technical definition:** dL/dx = dL/dy * dy/dx applied layer by layer from the loss backwards, reusing cached forward values; a weight gradient has the weight's shape, dW = input^T @ dout.
- **Plain English:** Blame flows backward, multiplied by each layer's local slope.
- **Example / analogy:** For softmax cross-entropy the logit gradient is (p - onehot)/N.
- **Snippet:**
```python
dW2 = a1.T @ dout; da1 = dout @ W2.T; dz1 = da1 * (1 - a1**2)
```
- **Snippet explained:** Three lines of backward for a tanh hidden layer.
- **Related terms:** [[Computational graph]], [[Gradient check]]

### Computational graph
<!-- cat: ml -->
- **Technical definition:** A directed acyclic graph whose nodes are operations/tensors, recorded during the forward pass so backward can traverse it in reverse topological order.
- **Plain English:** The receipt of every calculation, replayed backwards to assign blame.
- **Example / analogy:** PyTorch builds it on the fly (define-by-run) and frees it after backward().
- **Snippet:**
```python
c = a * b + a   # a has two outgoing edges
```
- **Snippet explained:** Node a is used twice so its gradient is the sum of two paths.
- **Related terms:** [[Chain rule (backpropagation through layers)]]

### Gradient check
<!-- cat: ml -->
- **Technical definition:** Compare an analytic gradient with centered finite differences (L(w+e) - L(w-e)) / 2e in float64; pass if the relative error is below about 1e-6.
- **Plain English:** A test that your backward() is right.
- **Example / analogy:** We measured 5.5e-9 for a hand-written 2-layer net.
- **Snippet:**
```python
num = (loss(w + eps) - loss(w - eps)) / (2 * eps)
```
- **Snippet explained:** Perturb one weight each way and divide by the step.
- **Related terms:** [[Chain rule (backpropagation through layers)]]

### Vanishing gradient
<!-- cat: ml -->
- **Technical definition:** Gradients shrink exponentially with depth (or time) because each layer multiplies by a factor below 1, such as sigmoid's slope of at most 0.25; early layers stop learning.
- **Plain English:** The learning signal fades before it reaches the first layers.
- **Example / analogy:** 20 sigmoid layers gave a gradient ratio of about 4e-13.
- **Snippet:**
```python
0.25 ** 20   # about 9e-13
```
- **Snippet explained:** Upper bound on the product of sigmoid slopes over 20 layers.
- **Related terms:** [[Exploding gradient]], [[Sigmoid saturation]]

### Exploding gradient
<!-- cat: ml -->
- **Technical definition:** Gradients grow exponentially with depth or time when per-layer factors exceed 1; produces huge updates and inf/NaN.
- **Plain English:** The learning signal gets louder at every layer until it blows up.
- **Example / analogy:** Recurrent nets with a recurrent matrix of spectral radius above 1.
- **Snippet:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```
- **Snippet explained:** Rescales the whole gradient so its norm is at most 1.
- **Related terms:** [[Gradient clipping]], [[Vanishing gradient]], [[BPTT (backpropagation through time)]]

### Sigmoid saturation
<!-- cat: ml -->
- **Technical definition:** For large |z| sigmoid(z) is near 0 or 1 and its slope s(1-s) is near 0, so gradient through that unit vanishes.
- **Plain English:** The curve goes flat at both ends so nothing flows back.
- **Example / analogy:** Max slope 0.25 at z=0.
- **Snippet:**
```python
s = 1 / (1 + np.exp(-z)); slope = s * (1 - s)
```
- **Snippet explained:** Slope formula; peaks at 0.25.
- **Related terms:** [[Vanishing gradient]], [[Activation function]], [[ReLU and dead ReLU]]

### ReLU and dead ReLU
<!-- cat: ml -->
- **Technical definition:** ReLU(z) = max(0, z), slope 1 for z > 0 and 0 otherwise; a dead ReLU is a unit negative for every input, so its gradient is always zero and it never recovers.
- **Plain English:** A simple on/off ramp that can get stuck off.
- **Example / analogy:** A too-high learning rate pushes a bias very negative and kills the unit; leaky ReLU keeps a small slope.
- **Snippet:**
```python
dead = ((h > 0).sum(axis=0) == 0).sum()
```
- **Snippet explained:** Counts hidden units that never fire on the batch.
- **Related terms:** [[Activation function]], [[Vanishing gradient]], [[He initialisation]]

### GELU and SiLU (Swish)
<!-- cat: ml -->
- **Technical definition:** Smooth ReLU-like functions: GELU(x) = x Phi(x), SiLU(x) = x sigmoid(x) (Swish with beta 1).
- **Plain English:** ReLU with rounded corners.
- **Example / analogy:** GELU in BERT/GPT-2; SiLU in the gate of SwiGLU in Llama.
- **Snippet:**
```python
y = torch.nn.functional.silu(x)
```
- **Snippet explained:** Equals x * torch.sigmoid(x).
- **Related terms:** [[Activation function]], [[ReLU and dead ReLU]], [[Gated MLP (GLU family)]]

### Gated MLP (GLU family)
<!-- cat: ml -->
- **Technical definition:** A feed-forward block that multiplies two projections elementwise, down(act(x W_gate) * (x W_up)); SwiGLU uses SiLU as act. Hidden size is about 8/3 d to keep parameters equal to a 4d GELU MLP.
- **Plain English:** One path decides how much of the other path passes.
- **Example / analogy:** Llama's gate_proj, up_proj, down_proj.
- **Snippet:**
```python
h = torch.nn.functional.silu(x @ Wg) * (x @ Wu)
```
- **Snippet explained:** Gate times content, then project down.
- **Related terms:** [[GELU and SiLU (Swish)]], [[Activation function]], [[Hidden layer (MLP)]]

### Xavier (Glorot) initialisation
<!-- cat: ml -->
- **Technical definition:** Weights with variance 2/(n_in + n_out), chosen to keep the variance of activations and gradients roughly constant for tanh/linear layers.
- **Plain English:** Start weights at a size that keeps signals from shrinking or blowing up.
- **Example / analogy:** torch.nn.init.xavier_uniform_.
- **Snippet:**
```python
torch.nn.init.xavier_normal_(layer.weight)
```
- **Snippet explained:** Fills the weights in place with the Glorot variance.
- **Related terms:** [[He initialisation]], [[Vanishing gradient]], [[Layer normalization]]

### He initialisation
<!-- cat: ml -->
- **Technical definition:** Weights with variance 2/n_in (Kaiming init); ReLU halves the second moment, so Var(w) = 2/n_in keeps Var(y) = Var(x).
- **Plain English:** Xavier corrected for ReLU's half-dead outputs.
- **Example / analogy:** Derivation: Var(y) = n_in Var(w) E[x^2].
- **Snippet:**
```python
torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
```
- **Snippet explained:** Normal weights with std sqrt(2/fan_in).
- **Related terms:** [[Xavier (Glorot) initialisation]], [[ReLU and dead ReLU]], [[Vanishing gradient]]

### Batch normalization
<!-- cat: ml -->
- **Technical definition:** Normalise each feature over the batch to zero mean and unit variance, then scale and shift with learned gamma and beta; uses running statistics at eval time.
- **Plain English:** Standardise each feature using the other examples in the batch.
- **Example / analogy:** Standard in CNNs; awkward with tiny or variable batches.
- **Snippet:**
```python
y = (x - x.mean(0)) / torch.sqrt(x.var(0, unbiased=False) + eps) * gamma + beta
```
- **Snippet explained:** Per-feature statistics over axis 0.
- **Related terms:** [[Layer normalization]], [[RMSNorm]], [[Train/eval mode]]

### Layer normalization
<!-- cat: ml -->
- **Technical definition:** Normalise each example over its feature axis (mean and variance per sample), then learned gain and bias; no batch dependence, same train and eval behaviour.
- **Plain English:** Standardise each example by itself.
- **Example / analogy:** Used in the original transformer and GPT-2.
- **Snippet:**
```python
y = (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + eps)
```
- **Snippet explained:** Statistics over the last axis.
- **Related terms:** [[Batch normalization]], [[RMSNorm]]

### RMSNorm
<!-- cat: ml -->
- **Technical definition:** x / sqrt(mean(x^2) + eps) * gain: LayerNorm without mean subtraction and without bias; d parameters instead of 2d.
- **Plain English:** Rescale to a fixed size; skip the centring.
- **Example / analogy:** Used in Llama, Qwen, Gemma.
- **Snippet:**
```python
y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * g
```
- **Snippet explained:** Divide by the root mean square of the features.
- **Related terms:** [[Layer normalization]], [[Batch normalization]]

### Residual connection (skip connection)
<!-- cat: ml -->
- **Technical definition:** y = x + f(x): the layer learns a correction; the gradient of y contains the identity, so signal and gradient pass through depth unattenuated. In transformers the running sum is the residual stream.
- **Plain English:** A shortcut road around each layer.
- **Example / analogy:** A 50-layer sigmoid net: gradient ratio 0.0 plain vs 1.27 with residuals.
- **Snippet:**
```python
x = x + block(norm(x))
```
- **Snippet explained:** Pre-norm residual update.
- **Related terms:** [[Vanishing gradient]], [[Layer normalization]], [[RMSNorm]]

### Momentum (SGD with momentum)
<!-- cat: ml -->
- **Technical definition:** v = mu v + g; p -= lr v. Accumulates an exponential average of gradients to smooth noise and speed through consistent directions.
- **Plain English:** A heavy ball rolling downhill.
- **Example / analogy:** mu = 0.9 averages about the last 10 gradients.
- **Snippet:**
```python
v = mu * v + g; p -= lr * v
```
- **Snippet explained:** PyTorch convention of the velocity update.
- **Related terms:** [[RMSProp]], [[Adam bias correction]], [[Learning-rate schedule (warm-up and cosine decay)]]

### RMSProp
<!-- cat: ml -->
- **Technical definition:** Divide the step by a running RMS of the gradient per parameter: v = rho v + (1-rho) g^2; p -= lr g / (sqrt(v) + eps).
- **Plain English:** Give every weight its own step size.
- **Example / analogy:** Weights with big gradients take smaller steps.
- **Snippet:**
```python
v = rho * v + (1 - rho) * g * g; p -= lr * g / (v ** 0.5 + eps)
```
- **Snippet explained:** Per-parameter scaling.
- **Related terms:** [[Momentum (SGD with momentum)]], [[Adam bias correction]], [[Learning-rate schedule (warm-up and cosine decay)]]

### Adam bias correction
<!-- cat: ml -->
- **Technical definition:** Adam's m and v start at 0 so they are biased low early; dividing by (1 - beta^t) corrects it, making the first step about lr in size.
- **Plain English:** A fix for a cold start of the running averages.
- **Example / analogy:** Without it the first step is about 3.16x lr (beta1 0.9, beta2 0.999).
- **Snippet:**
```python
mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
```
- **Snippet explained:** Corrected moments used in the update.
- **Related terms:** [[Momentum (SGD with momentum)]], [[RMSProp]], [[Decoupled weight decay (AdamW vs L2)]]

### Decoupled weight decay (AdamW vs L2)
<!-- cat: ml -->
- **Technical definition:** AdamW shrinks weights directly, p *= (1 - lr wd), outside the adaptive scaling; L2 adds wd p to the gradient, which Adam then divides by sqrt(v), so the two differ under Adam (same under plain SGD).
- **Plain English:** Weight decay done properly for Adam.
- **Example / analogy:** Adam(wd=0.05) crushed weights to norm 1.8; AdamW(wd=0.05) left them near 29.7.
- **Snippet:**
```python
opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.1)
```
- **Snippet explained:** Decoupled decay; do not decay biases and norm gains.
- **Related terms:** [[Adam bias correction]], [[L2 regularization]], [[Learning-rate schedule (warm-up and cosine decay)]]

### Learning-rate schedule (warm-up and cosine decay)
<!-- cat: ml -->
- **Technical definition:** lr(t) rises linearly for the warm-up steps, then follows 0.5 (1 + cos(pi progress)) down to a floor.
- **Plain English:** Start gently, finish gently.
- **Example / analogy:** Warm-up avoids divergence while Adam's v is noisy and weights are random.
- **Snippet:**
```python
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
```
- **Snippet explained:** lr_factor returns the multiplier at each step.
- **Related terms:** [[Adam bias correction]], [[Momentum (SGD with momentum)]], [[Gradient clipping]]

### L2 regularization
<!-- cat: ml -->
- **Technical definition:** Add (wd/2) ||w||^2 to the loss, adding wd w to the gradient, which penalises large weights.
- **Plain English:** A penalty that keeps weights small.
- **Example / analogy:** Equal to weight decay for SGD; different for Adam.
- **Snippet:**
```python
loss = ce + 0.5 * wd * sum((p ** 2).sum() for p in params)
```
- **Snippet explained:** Explicit penalty term.
- **Related terms:** [[Decoupled weight decay (AdamW vs L2)]], [[Dropout]]

### Dropout
<!-- cat: ml -->
- **Technical definition:** During training zero each activation with probability p and scale survivors by 1/(1-p) (inverted dropout); at eval it is the identity.
- **Plain English:** Randomly switch neurons off so none becomes indispensable.
- **Example / analogy:** Forgetting model.eval() makes predictions random.
- **Snippet:**
```python
y = x * (torch.rand_like(x) >= p) / (1 - p)
```
- **Snippet explained:** Mask and rescale so the expectation is unchanged.
- **Related terms:** [[Train/eval mode]], [[L2 regularization]], [[Label smoothing]]

### Label smoothing
<!-- cat: ml -->
- **Technical definition:** Replace the one-hot target by (1 - eps) on the true class and eps/(K-1) elsewhere, capping confidence.
- **Plain English:** Never be 100 percent sure.
- **Example / analogy:** Lowers validation loss and improves calibration; accuracy about unchanged.
- **Snippet:**
```python
F.cross_entropy(logits, y, label_smoothing=0.1)
```
- **Snippet explained:** Built-in smoothing option.
- **Related terms:** [[Dropout]]

### Convolution (cross-correlation)
<!-- cat: ml -->
- **Technical definition:** Slide a small kernel over the input computing local dot products with shared weights; output size floor((n + 2p - k)/s) + 1. Libraries implement cross-correlation (kernel not flipped).
- **Plain English:** One small pattern detector reused at every position.
- **Example / analogy:** 64 3x3 filters on RGB: 1,792 parameters versus 150 million for a dense layer.
- **Snippet:**
```python
y = F.conv2d(x, w, stride=1, padding=1)
```
- **Snippet explained:** Batch of images with (out, in, k, k) kernels.
- **Related terms:** [[Receptive field]]

### Receptive field
<!-- cat: ml -->
- **Technical definition:** The input region that can influence one output unit; grows with kernel size and stride across layers.
- **Plain English:** How much of the picture one neuron can see.
- **Example / analogy:** Two stacked 3x3 convs see 5x5.
- **Snippet:**
```python
rf = 1 + sum((k - 1) * jump for k, jump in layers)
```
- **Snippet explained:** Accumulate kernel growth scaled by the cumulative stride.
- **Related terms:** [[Convolution (cross-correlation)]]

### RNN (recurrent neural network)
<!-- cat: ml -->
- **Technical definition:** h_t = f(W_xh x_t + W_hh h_{t-1} + b): the same cell is applied at each step, carrying a hidden state.
- **Plain English:** A network with a memory notebook rewritten at every step.
- **Example / analogy:** Reads a sentence word by word.
- **Snippet:**
```python
h = np.tanh(x @ Wxh + h @ Whh + b)
```
- **Snippet explained:** One recurrent update.
- **Related terms:** [[BPTT (backpropagation through time)]], [[LSTM (long short-term memory)]], [[Fixed-size bottleneck]]

### BPTT (backpropagation through time)
<!-- cat: ml -->
- **Technical definition:** Backprop through the unrolled recurrence; the gradient is a product of Jacobians over T steps, so it vanishes or explodes exponentially.
- **Plain English:** Backprop where every time step is a layer.
- **Example / analogy:** Spectral radius below 1 vanishes, above 1 explodes.
- **Snippet:**
```python
loss.backward()   # through the unrolled loop
```
- **Snippet explained:** Autograd unrolls the Python loop.
- **Related terms:** [[Vanishing gradient]], [[Exploding gradient]], [[LSTM (long short-term memory)]]

### LSTM (long short-term memory)
<!-- cat: ml -->
- **Technical definition:** RNN with a cell state updated additively c_t = f c_{t-1} + i g and gates (input, forget, output) controlling flow; the additive path eases vanishing gradients. GRU merges gates into two.
- **Plain English:** An RNN with a notebook it can add to, erase from and read from.
- **Example / analogy:** Used for sequences before transformers.
- **Snippet:**
```python
c = f * c + i * g; h = o * torch.tanh(c)
```
- **Snippet explained:** Core LSTM update.
- **Related terms:** [[RNN (recurrent neural network)]], [[BPTT (backpropagation through time)]], [[Fixed-size bottleneck]]

### Fixed-size bottleneck
<!-- cat: ml -->
- **Technical definition:** An encoder-decoder RNN squeezes the whole input into one vector; accuracy falls as sequences lengthen (copying 24 tokens: 0.126, chance level). Attention lets the decoder read every position instead.
- **Plain English:** Summarising a book on a sticky note.
- **Example / analogy:** Motivation for attention in seq2seq.
- **Snippet:**
```python
ctx = enc_states[-1]   # one vector for everything
```
- **Snippet explained:** The last hidden state is all the decoder sees.
- **Related terms:** [[LSTM (long short-term memory)]], [[BPTT (backpropagation through time)]]

### Floating-point formats (fp32, fp16, bf16)
<!-- cat: ml -->
- **Technical definition:** fp32: 8 exponent + 23 mantissa bits; fp16: 5 + 10, max 65,504; bf16: 8 + 7, fp32's range with coarser precision.
- **Plain English:** Different trade-offs between range and detail.
- **Example / analogy:** bf16 needs no loss scaling; fp16 does.
- **Snippet:**
```python
torch.finfo(torch.bfloat16).max
```
- **Snippet explained:** Range limit of bf16 (about 3.4e38).
- **Related terms:** [[Loss scaling]], [[Mixed precision (autocast)]], [[NaN debugging]]

### Loss scaling
<!-- cat: ml -->
- **Technical definition:** Multiply the loss by a large factor before backward so small fp16 gradients do not underflow, then divide the gradients by it before the update; dynamic scalers adjust it.
- **Plain English:** Turn up the volume before storing quiet numbers.
- **Example / analogy:** 1e-8 becomes 0 in fp16 but survives after a 2^14 scale.
- **Snippet:**
```python
scaler = torch.amp.GradScaler()
```
- **Snippet explained:** Wraps scale, unscale, inf-check and step for fp16 training.
- **Related terms:** [[Floating-point formats (fp32, fp16, bf16)]], [[Mixed precision (autocast)]], [[Gradient clipping]]

### Mixed precision (autocast)
<!-- cat: ml -->
- **Technical definition:** Run most ops in a low-precision format while keeping master weights and sensitive ops (softmax, losses, reductions) in fp32.
- **Plain English:** Cheap numbers for the heavy lifting, careful numbers for the bookkeeping.
- **Example / analogy:** torch.autocast('cpu', dtype=torch.bfloat16).
- **Snippet:**
```python
with torch.autocast('cpu', dtype=torch.bfloat16): out = model(x)
```
- **Snippet explained:** Matmuls run in bf16 inside the block.
- **Related terms:** [[Floating-point formats (fp32, fp16, bf16)]], [[Loss scaling]], [[NaN debugging]]

### Gradient clipping
<!-- cat: ml -->
- **Technical definition:** Rescale gradients so their global norm is at most a threshold; clip_grad_norm_ returns the pre-clip norm.
- **Plain English:** A speed limit on update size.
- **Example / analogy:** max_norm=1.0 is a common default in LLM training.
- **Snippet:**
```python
gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```
- **Snippet explained:** Clips in place and returns the norm before clipping.
- **Related terms:** [[Exploding gradient]], [[Loss scaling]], [[Learning-rate schedule (warm-up and cosine decay)]]

### NaN debugging
<!-- cat: ml -->
- **Technical definition:** Locate the first non-finite tensor (torch.isfinite), turn on torch.autograd.detect_anomaly, check log(0), sqrt(0), division by zero and exp overflow, lower lr, clip, compare in fp32.
- **Plain English:** Find where the first nan appears, not where you notice it.
- **Example / analogy:** sqrt'(0) is inf; multiplied by 0 it becomes nan.
- **Snippet:**
```python
assert torch.isfinite(loss), 'loss is not finite'
```
- **Snippet explained:** Cheap guard in the training loop.
- **Related terms:** [[Gradient clipping]], [[Floating-point formats (fp32, fp16, bf16)]], [[Loss scaling]]

### Train/eval mode
<!-- cat: python -->
- **Technical definition:** model.train() and model.eval() set module.training, changing the behaviour of dropout and batch normalization (batch statistics vs running statistics).
- **Plain English:** A switch that must be set to eval before predicting.
- **Example / analogy:** Forgetting eval() gives noisy predictions.
- **Snippet:**
```python
model.eval()
with torch.no_grad(): y = model(x)
```
- **Snippet explained:** Inference setup: eval mode and no graph.
- **Related terms:** [[Dropout]], [[Batch normalization]], [[Layer normalization]]

### Logistic regression (logit model)
<!-- cat: ml -->
- **Technical definition:** A binary classifier p = sigmoid(w.x + b) fit by minimising binary cross-entropy; a single neuron with a sigmoid output. The logit w.x + b is the log-odds of class 1.
- **Plain English:** A straight-line classifier that reports a probability.
- **Example / analogy:** Spam probability from word counts: 0.97 means very likely spam; the boundary is where it equals 0.5.
- **Snippet:**
```python
p = 1 / (1 + np.exp(-(X @ w + b)))
```
- **Snippet explained:** Weighted sum plus bias, squashed into (0, 1).
- **Related terms:** [[Binary cross-entropy (log loss)]], [[Perceptron (artificial neuron)]], [[Probability calibration]]

### Binary cross-entropy (log loss)
<!-- cat: ml -->
- **Technical definition:** L = -[y ln p + (1-y) ln(1-p)] averaged over samples; the negative log-likelihood of a Bernoulli model. With a sigmoid output its gradient w.r.t. the logit is p - y.
- **Plain English:** A penalty that explodes when the model is confident and wrong.
- **Example / analogy:** Predicting p=0.01 for a true 1 costs 4.6; predicting 0.5 costs 0.69.
- **Snippet:**
```python
-np.mean(y*np.log(p) + (1-y)*np.log(1-p))
```
- **Snippet explained:** Mean negative log probability assigned to the true label.
- **Related terms:** [[Logistic regression (logit model)]], [[Probability calibration]]

### Probability calibration
<!-- cat: ml -->
- **Technical definition:** The property that predictions of probability p are correct a fraction p of the time; measured with reliability bins, ECE, Brier score or BCE; fixed post hoc by temperature or Platt scaling.
- **Plain English:** Does '80% sure' come true 8 times in 10?
- **Example / analogy:** Scaling logits by 3 keeps accuracy at 0.865 but raises ECE from 0.035 to 0.075.
- **Snippet:**
```python
over = sigmoid(3 * z)
```
- **Snippet explained:** Same decisions, bolder probabilities, worse calibration.
- **Related terms:** [[Binary cross-entropy (log loss)]], [[Logistic regression (logit model)]]

### Log-odds (logit) and odds
<!-- cat: ml -->
- **Technical definition:** Odds = p/(1-p); log-odds (logit) = ln(p/(1-p)), the inverse of the sigmoid. In logistic regression the logit is linear in the features.
- **Plain English:** How many times likelier yes is than no, on a log scale.
- **Example / analogy:** p = 0.75 means odds 3:1; a weight of 0.69 doubles the odds per unit of the feature.
- **Snippet:**
```python
odds = np.exp(z)
```
- **Snippet explained:** e^z equals p/(1-p).
- **Related terms:** [[Logistic regression (logit model)]]

### Numerical gradient check
<!-- cat: ml -->
- **Technical definition:** Comparing an analytic gradient with the central difference (L(theta+h) - L(theta-h)) / 2h; agreement near 1e-10 (h=1e-6, float64) validates a hand-derived backward pass.
- **Plain English:** Measure the slope with a ruler to confirm the algebra.
- **Example / analogy:** Our (p - y) x gradient matched to 4e-11.
- **Snippet:**
```python
(f(w + h*e) - f(w - h*e)) / (2*h)
```
- **Snippet explained:** Perturb one parameter up and down and divide the loss change by 2h.
- **Related terms:** [[Binary cross-entropy (log loss)]]
