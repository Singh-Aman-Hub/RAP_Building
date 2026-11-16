# Terms — Activation quantization and calibration

Folder: `21_llm_inference_engineering/05_activation_quant_and_calibration` · Files covered: `main.py`, `WALKTHROUGH.md`, `../quantlib.py`
Back to module: [21 README](../README.md)

**Terms in this folder:** [Weight-only quantization (W4A16, W8A16)](#weight-only-quantization-w4a16-w8a16) · [W8A8 / FP8 (weight and activation quantization)](#w8a8--fp8-weight-and-activation-quantization) · [Activation outlier channels](#activation-outlier-channels) · [Per-tensor / per-token / per-channel activation scales](#per-tensor--per-token--per-channel-activation-scales) · [Dynamic vs static quantization](#dynamic-vs-static-quantization) · [Calibration (min-max, percentile, MSE)](#calibration-min-max-percentile-mse) · [SmoothQuant](#smoothquant) · [LLM.int8() decomposition](#llmint8-decomposition) · [PTQ (post-training quantization, offline quantization)](#ptq-post-training-quantization-offline-quantization) · [QAT (quantization-aware training)](#qat-quantization-aware-training) · [Straight-through estimator (STE)](#straight-through-estimator-ste) · [Forward hook](#forward-hook)

---

## Weight-only quantization (W4A16, W8A16)  `(concept)`
**1. Definition:** Quantizing weights while activations stay 16-bit; weights are dequantized on the fly inside the matmul.
**2. In plain words:** Smaller model, same arithmetic precision.
**3. Example to quote:** > "W8A16 cost 0.02 perplexity on distilgpt2."
```python
W4 = ql.qdq(W, 4, True, 128)
```
**Seen in:** `main.py` sections 2, 7

## W8A8 / FP8 (weight and activation quantization)  `(concept)`
**1. Definition:** Both matmul operands quantized so the GEMM can use integer or FP8 tensor cores.
**2. In plain words:** The fast path, and the hard one, because of activations.
**3. Example to quote:** > "Per-token W8A8 cost 0.1 perplexity; per-tensor cost 7."
```python
act_qdq(Xe, 8, 'token') @ W8.T
```
**Seen in:** `main.py` sections 2, 7

## Activation outlier channels  `(concept)`
**1. Definition:** A few hidden dimensions whose magnitude is many times the median for every token (measured by max over tokens of |x| per channel divided by its median over channels).
**2. In plain words:** A couple of loud channels drown out the quiet ones.
**3. Example to quote:** > "The inputs of distilgpt2's linear layers reach a max/median ratio of 58.8x."
```python
np.abs(X).max(axis=0)
```
**Seen in:** `main.py` section 1

## Per-tensor / per-token / per-channel activation scales  `(concept)`
**1. Definition:** One scale for all activations, one per token row, or one per input channel; per-channel cannot be applied after an integer GEMM.
**2. In plain words:** How finely the loud values are separated from the quiet ones.
**3. Example to quote:** > "Per-tensor 23.7 dB, per-token 37.0 dB, per-channel 43.7 dB on one layer."
```python
s = np.abs(X).max(axis=1, keepdims=True) / qmax
```
**Seen in:** `main.py` section 2

## Dynamic vs static quantization  `(concept)`
**1. Definition:** Dynamic computes activation ranges at run time; static fixes them offline from a calibration set.
**2. In plain words:** Measure now versus decide in advance.
**3. Example to quote:** > "Static percentile ranges raised perplexity to 141.7, dynamic per-token stayed at 25.6."
```python
act_qdq(X, 8, 'static', amax)
```
**Seen in:** `main.py` sections 4, 7

## Calibration (min-max, percentile, MSE)  `(concept)`
**1. Definition:** Estimating activation ranges from sample data with one of three rules.
**2. In plain words:** A dress rehearsal that decides the ruler length.
**3. Example to quote:** > "MSE gave 25.8 dB, min-max 24.2, the 99.9th percentile 21.7 because it clipped the real outlier channel."
```python
static_range(Xcal, 'mse')
```
**Seen in:** `main.py` section 4

## SmoothQuant  `(concept)`
**1. Definition:** Rescaling y = XW^T into (X/s)(W*s)^T with s_j = max|X_j|^a / max|W_j|^(1-a) so activation outliers move into the weights.
**2. In plain words:** Share the difficulty between the two operands.
**3. Example to quote:** > "alpha 0.5 lifted per-tensor W8A8 from 23.5 to 31.9 dB."
```python
Xs, Ws = X / s, W * s
```
**Seen in:** `main.py` section 5

## LLM.int8() decomposition  `(concept)`
**1. Definition:** Keep input channels with magnitude above a threshold (6.0) in fp16 and quantize the rest to int8, summing both products.
**2. In plain words:** The loud channels get their own lane.
**3. Example to quote:** > "One of 768 channels was kept in fp and per-token W8A8 rose 0.8 dB."
```python
Xe[:, out_ch] @ W[:, out_ch].T + ...
```
**Seen in:** `main.py` section 3

## PTQ (post-training quantization, offline quantization)  `(concept)`
**1. Definition:** Quantizing a trained model without further training.
**2. In plain words:** Round it and ship it.
**3. Example to quote:** > "PTQ at 2 bits left the tiny MLP at 62% accuracy."
```python
ptq = MLP(BITS); ptq.load_state_dict(net.state_dict())
```
**Seen in:** `main.py` section 6

## QAT (quantization-aware training)  `(concept)`
**1. Definition:** Training with fake quantization in the forward pass so weights adapt to rounding.
**2. In plain words:** Practice with the handicap.
**3. Example to quote:** > "QAT recovered 100% accuracy at 2 bits in 200 steps."
```python
train(qat, 200, 3e-3)
```
**Seen in:** `main.py` section 6

## Straight-through estimator (STE)  `(concept)`
**1. Definition:** Using the identity as the backward pass of a non-differentiable rounding.
**2. In plain words:** Pretend rounding has slope 1 so gradients can flow.
**3. Example to quote:** > "w + (wq - w).detach() is the STE in one line."
```python
return w + (wq - w).detach()
```
**Seen in:** `main.py` section 6

## Forward hook  `(function)`
**1. Definition:** A function registered with module.register_forward_hook that runs after every forward of that module and receives its input and output.
**2. In plain words:** A tap on a pipe.
**3. Example to quote:** > "We tapped all 24 Conv1D layers to record activations."
```python
m.register_forward_hook(hook)
```
**Seen in:** `quantlib.collect_inputs`
