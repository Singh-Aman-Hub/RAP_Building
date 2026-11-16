# Terms — Quantization math from scratch

Folder: `21_llm_inference_engineering/04_quantization_math` · Files covered: `main.py`, `WALKTHROUGH.md`, `../quantlib.py`
Back to module: [21 README](../README.md)

**Terms in this folder:** [Affine quantization (uniform, linear)](#affine-quantization-uniform-linear) · [Scale and zero-point](#scale-and-zero-point) · [Symmetric (absmax) quantization](#symmetric-absmax-quantization) · [Asymmetric (min-max) quantization](#asymmetric-min-max-quantization) · [Granularity (per-tensor, per-channel, per-group)](#granularity-per-tensor-per-channel-per-group) · [Bits per weight (bpw)](#bits-per-weight-bpw) · [SQNR (signal-to-quantization-noise ratio)](#sqnr-signal-to-quantization-noise-ratio) · [Clipping range (percentile, MSE-optimal scale)](#clipping-range-percentile-mse-optimal-scale) · [int4 packing](#int4-packing) · [FP8 (E4M3, E5M2)](#fp8-e4m3-e5m2) · [NF4 (NormalFloat4)](#nf4-normalfloat4) · [Double quantization](#double-quantization) · [MXFP4 / NVFP4 (block-scaled FP4)](#mxfp4--nvfp4-block-scaled-fp4) · [Fake quantization (QDQ)](#fake-quantization-qdq) · [np.searchsorted](#npsearchsorted) · [torch.float8_e4m3fn](#torchfloat8_e4m3fn)

---

## Affine quantization (uniform, linear)  `(concept)`
**1. Definition:** Mapping real x to an integer code with q = round(x/s) + z and back with x_hat = (q - z) * s.
**2. In plain words:** Snap every number onto an evenly spaced ruler and remember the ruler's step and where zero sits.
**3. Example to quote:** > "Every scheme in this step is the affine map; they differ in what shares (s, z) and how the range is chosen."
```python
q = np.clip(np.round(x / s) + z, 0, 255).astype(np.uint8)
```
**Seen in:** `main.py` section 1

## Scale and zero-point  `(concept)`
**1. Definition:** The scale s is the real width of one integer step; the zero-point z is the integer code that decodes to exactly 0.0.
**2. In plain words:** s is the ruler's tick spacing, z is which tick is labelled zero.
**3. Example to quote:** > "For [-0.4, 3.0] at uint8 the scale is 0.01333 and the zero-point is 30."
```python
s = (hi - lo) / 255; z = int(round(-lo / s))
```
**Seen in:** `main.py` section 1

## Symmetric (absmax) quantization  `(concept)`
**1. Definition:** Range [-m, m] with m = max|x|, zero-point 0 and signed codes in [-(2^(b-1)-1), 2^(b-1)-1].
**2. In plain words:** Zero is in the middle of the ruler; simple and fast, but wasteful for one-sided data.
**3. Example to quote:** > "Symmetric int4 on post-ReLU data lost about 6.6 dB against asymmetric."
```python
s = np.abs(v).max() / qmax
```
**Seen in:** `quantlib.qdq`, section 2

## Asymmetric (min-max) quantization  `(concept)`
**1. Definition:** Range [min, max] with a zero-point, using all 2^b codes.
**2. In plain words:** The ruler starts at the smallest value instead of being centred.
**3. Example to quote:** > "Asymmetric fits skewed activations; it needs a zero-point per group."
```python
lo = min(v.min(), 0); hi = max(v.max(), 0)
```
**Seen in:** `quantlib.qdq`, section 2

## Granularity (per-tensor, per-channel, per-group)  `(concept)`
**1. Definition:** The set of numbers that share one scale: the whole tensor, one output row, or g consecutive weights.
**2. In plain words:** How many numbers share one ruler. Fewer sharing means finer rulers but more rulers to store.
**3. Example to quote:** > "int4 on a real distilgpt2 matrix: 8.1 dB per-tensor, 17.1 per-row, 18.6 at g=128."
```python
ql.qdq(W, 4, True, 128)
```
**Seen in:** `main.py` section 3

## Bits per weight (bpw)  `(concept)`
**1. Definition:** Average storage per weight including scale and zero-point overhead: bits + (scale_bits + zero_bits) / group.
**2. In plain words:** The true bit cost once you count the rulers.
**3. Example to quote:** > "int4 with g=128 and an fp16 scale is 4.125 bits per weight, so a 7B model is about 3.6 GB."
```python
ql.bits_per_weight(4, 128)  # 4.125
```
**Seen in:** `main.py` section 8

## SQNR (signal-to-quantization-noise ratio)  `(concept)`
**1. Definition:** 10 log10(sum x^2 / sum (x - x_hat)^2) in decibels; roughly +6 dB per extra bit for uniform data.
**2. In plain words:** How loud the signal is compared with the rounding noise.
**3. Example to quote:** > "The 4-bit g=64 variants landed between 18.8 and 20.8 dB."
```python
10 * np.log10((x**2).sum() / (err**2).sum())
```
**Seen in:** `quantlib.metrics`

## Clipping range (percentile, MSE-optimal scale)  `(concept)`
**1. Definition:** Choosing the quantization range as a fraction of absmax (or a percentile) that minimises error, saturating rare outliers.
**2. In plain words:** Ignore the tallest few values so the ruler can be finer for everybody else.
**3. Example to quote:** > "On the real weight the best clip was 30% of absmax and cut MSE 9.8x."
```python
ql.mse_optimal_clip(W, 4, True, 'tensor')
```
**Seen in:** `main.py` section 4

## int4 packing  `(concept)`
**1. Definition:** Storing two 4-bit codes per uint8: low nibble for the even index, high nibble for the odd index, codes offset to 0..15.
**2. In plain words:** Two numbers share one byte.
**3. Example to quote:** > "1,000,000 codes took 500,000 bytes and unpacked exactly."
```python
packed = lo | (hi << 4)
```
**Seen in:** `quantlib.pack_int4`, section 5

## FP8 (E4M3, E5M2)  `(concept)`
**1. Definition:** 8-bit floats: E4M3 has 4 exponent and 3 mantissa bits (max 448), E5M2 has 5 and 2 (max 57344).
**2. In plain words:** A tiny float: precise near zero, coarse for large values.
**3. Example to quote:** > "E4M3 beat int8 by 11 dB on heavy-tailed data and lost by 8 dB on Gaussian data."
```python
ql.fp8_qdq(x, 'e4m3')
```
**Seen in:** `main.py` section 6

## NF4 (NormalFloat4)  `(concept)`
**1. Definition:** A 4-bit codebook of 16 quantiles of a standard normal distribution, scaled to [-1, 1], applied per block with absmax scaling.
**2. In plain words:** Put the 16 grid points where Gaussian-shaped weights actually are.
**3. Example to quote:** > "NF4 at block 64 reached 20.7 dB, as good as asymmetric int4 g64."
```python
ql.nf4_qdq(W, 64)
```
**Seen in:** `main.py` section 7

## Double quantization  `(concept)`
**1. Definition:** Quantizing the per-block scales themselves (QLoRA: 8-bit scales in blocks of 256), saving memory.
**2. In plain words:** Compress the rulers too.
**3. Example to quote:** > "It cut the overhead from 4.25 to 4.127 bits per weight with no visible error change."
```python
ql.nf4_qdq(W, 64, double_quant=True)
```
**Seen in:** `main.py` section 7

## MXFP4 / NVFP4 (block-scaled FP4)  `(concept)`
**1. Definition:** FP4 E2M1 values {0,.5,1,1.5,2,3,4,6} with a shared block scale: MXFP4 uses blocks of 32 and a power-of-two E8M0 scale, NVFP4 blocks of 16 and an FP8 E4M3 scale plus an FP32 tensor scale. Emerging, hardware-accelerated on newest GPUs.
**2. In plain words:** Tiny floats with a shared exponent per small block.
**3. Example to quote:** > "In our test NVFP4's finer scale was about 1.7 dB better than MXFP4's power-of-two scale."
```python
ql.fp4_block_qdq(W, 16, 'e4m3')
```
**Seen in:** `main.py` section 7

## Fake quantization (QDQ)  `(concept)`
**1. Definition:** Quantize then immediately de-quantize in float to measure the error without integer kernels.
**2. In plain words:** Practice rounding, keep the numbers as floats.
**3. Example to quote:** > "All numbers in this step are fake-quantized."
```python
xh = ql.qdq(x, 4)
```
**Seen in:** whole step

## np.searchsorted  `(function)`
**1. Definition:** Returns insertion indices of values into a sorted array; with midpoints between levels it finds the nearest level.
**2. In plain words:** Binary search for 'which bucket am I in'.
**3. Example to quote:** > "We snap millions of values to 16 NF4 levels without building a million-by-16 matrix."
```python
idx = np.searchsorted((lv[1:] + lv[:-1]) / 2, v)
```
**Seen in:** `quantlib._nearest`

## torch.float8_e4m3fn  `(class)`
**1. Definition:** PyTorch's native FP8 E4M3 dtype (finite, no infinities); converting a float tensor to it rounds to FP8.
**2. In plain words:** The reference to check our own rounding against.
**3. Example to quote:** > "Our simulator equals the torch conversion on 5000 values."
```python
torch.from_numpy(t).to(torch.float8_e4m3fn).float()
```
**Seen in:** `main.py` section 6
