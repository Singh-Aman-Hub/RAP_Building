# Terms — GPTQ and AWQ from scratch

Folder: `21_llm_inference_engineering/06_gptq_awq_from_scratch` · Files covered: `main.py`, `reference_tools.py`, `WALKTHROUGH.md`
Back to module: [21 README](../README.md)

**Terms in this folder:** [RTN (round-to-nearest)](#rtn-round-to-nearest) · [GPTQ](#gptq) · [Hessian of the layer (H = 2 X^T X)](#hessian-of-the-layer-h--2-xt-x) · [Damping (percdamp)](#damping-percdamp) · [Cholesky factorization](#cholesky-factorization) · [Lazy block update](#lazy-block-update) · [AWQ (activation-aware weight quantization)](#awq-activation-aware-weight-quantization) · [Calibration set size and domain (for GPTQ/AWQ)](#calibration-set-size-and-domain-for-gptqawq) · [Group size (g128, g64)](#group-size-g128-g64) · [HQQ (half-quadratic quantization)](#hqq-half-quadratic-quantization) · [llm-compressor / GPTQModel / AutoAWQ / bitsandbytes / torchao](#llm-compressor--gptqmodel--autoawq--bitsandbytes--torchao) · [Conv1D weight layout (in, out)](#conv1d-weight-layout-in-out)

---

## RTN (round-to-nearest)  `(concept)`
**1. Definition:** Quantizing each weight independently to the nearest grid point, using no data.
**2. In plain words:** The baseline: round and hope.
**3. Example to quote:** > "At 3 bits g128 RTN perplexity was 42.2 against 25.4 in fp32."
```python
ql.qdq(W, bits, False, 128)
```
**Seen in:** `main.py` section 2

## GPTQ  `(concept)`
**1. Definition:** Layer-wise second-order PTQ: quantize columns one at a time and update the remaining columns with the inverse Hessian H = 2X^T X of the calibration inputs to minimise the layer output error.
**2. In plain words:** Round a column, then tell the others to cancel its mistake.
**3. Example to quote:** > "GPTQ reached 29.0 perplexity at 3 bits where RTN had 42.2."
```python
W[:, b1:] -= Err @ Hinv[b0:b1, b1:]
```
**Seen in:** `main.py` section 3

## Hessian of the layer (H = 2 X^T X)  `(concept)`
**1. Definition:** The second derivative of the squared output error with respect to one weight row; an in x in matrix of input-channel correlations.
**2. In plain words:** Which weight errors matter and which cancel.
**3. Example to quote:** > "With 712 calibration tokens and 768 input channels H is singular without damping."
```python
H = 2.0 * X.T @ X / X.shape[0]
```
**Seen in:** `main.py` section 3

## Damping (percdamp)  `(concept)`
**1. Definition:** Adding a fraction (1%) of the mean diagonal of H to its diagonal so it can be inverted.
**2. In plain words:** A small ridge that makes the matrix invertible.
**3. Example to quote:** > "Without it the Cholesky step fails."
```python
H += damp * torch.mean(torch.diag(H)) * torch.eye(n_in)
```
**Seen in:** `main.py` section 3

## Cholesky factorization  `(function)`
**1. Definition:** Writing a positive-definite matrix as L L^T (or U^T U); used to invert H stably.
**2. In plain words:** The numerically safe square root of a matrix.
**3. Example to quote:** > "GPTQ uses the upper Cholesky factor of H^-1."
```python
torch.linalg.cholesky(A, upper=True)
```
**Seen in:** `main.py` section 3

## Lazy block update  `(concept)`
**1. Definition:** Applying a block's accumulated error corrections to all later columns with one matrix product instead of per column.
**2. In plain words:** Batch the bookkeeping.
**3. Example to quote:** > "It cut quantization time from 271 s to 30 s with nearly the same results (perplexity 28.99 vs 29.00 at 3 bits)."
```python
W[:, b1:] -= Err @ Hinv[b0:b1, b1:]
```
**Seen in:** `main.py` section 3

## AWQ (activation-aware weight quantization)  `(concept)`
**1. Definition:** Scaling input channels by s = mean|X|^alpha before rounding (and folding 1/s into the previous op), with alpha chosen by grid search on calibration output error.
**2. In plain words:** Give the channels that carry big activations a finer grid.
**3. Example to quote:** > "AWQ gave 39.1 perplexity at 3 bits, between RTN 42.2 and GPTQ 29.0."
```python
Wq = ql.qdq(W * s, bits, False, group) / s
```
**Seen in:** `main.py` section 4

## Calibration set size and domain (for GPTQ/AWQ)  `(concept)`
**1. Definition:** The tokens whose activations estimate H or channel magnitudes; too few or off-domain gives overfit or noisy results.
**2. In plain words:** The practice questions you tune on.
**3. Example to quote:** > "We used about 700 tokens and evaluated on held-out lines."
```python
X_cal = ql.collect_inputs(model, tok, calib_text)
```
**Seen in:** `main.py` header

## Group size (g128, g64)  `(concept)`
**1. Definition:** Number of consecutive weights sharing a scale; smaller groups reduce error and increase bits per weight.
**2. In plain words:** Ruler length.
**3. Example to quote:** > "g128 with fp16 scale is 4.125 bits per weight."
```python
GROUP = 128
```
**Seen in:** `main.py` header

## HQQ (half-quadratic quantization)  `(concept)`
**1. Definition:** Calibration-free quantization that optimises scale and zero-point per group with a half-quadratic solver on a robust reconstruction loss.
**2. In plain words:** Smarter rounding without any data.
**3. Example to quote:** > "Use it when you have no calibration set and want better than RTN."
```python
(reference only)
```
**Seen in:** `WALKTHROUGH.md` theory

## llm-compressor / GPTQModel / AutoAWQ / bitsandbytes / torchao  `(concept)`
**1. Definition:** Libraries that implement these algorithms: llm-compressor makes vLLM-ready checkpoints (GPTQ, AWQ, FP8 recipes), GPTQModel the GPTQ kernels and format, AutoAWQ the AWQ format (check its project status), bitsandbytes NF4 and LLM.int8() without calibration, torchao PyTorch-native quantization.
**2. In plain words:** The off-the-shelf versions of what we wrote.
**3. Example to quote:** > "REFERENCE only: `reference_tools.py` prints recipes and never runs them."
```python
python reference_tools.py
```
**Seen in:** `reference_tools.py`

## Conv1D weight layout (in, out)  `(concept)`
**1. Definition:** GPT-2 stores linear weights as (in_features, out_features) and computes x @ W, the transpose of nn.Linear.
**2. In plain words:** The weights are stored sideways.
**3. Example to quote:** > "ql.get_w_out_in transposes it so grouping runs along the input dimension."
```python
W = mod.weight.detach().numpy().T.copy()
```
**Seen in:** `quantlib.get_w_out_in`
