# Terms — Quantization

Folder: `05_llm_parameters/05_quantization` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [05 LLM parameters README](../README.md)

**Terms in this folder:** [Quantization](#quantization) · [Symmetric quantization](#symmetric-quantization) · [Scale factor](#scale-factor) · [De-quantization](#de-quantization) · [Simulated (fake) quantization](#simulated-fake-quantization) · [Power operator ** and qmax](#power-operator--and-qmax) · [np.abs and .max()](#npabs-and-max) · [np.round and np.clip](#npround-and-npclip) · [.astype and integer overflow](#astype-and-integer-overflow) · [Quantization error](#quantization-error) · [Memory saving by bit width](#memory-saving-by-bit-width) · [copy.deepcopy](#copydeepcopy) · [p.ndim](#pndim) · [p.data assignment](#pdata-assignment) · [torch.from_numpy](#torchfrom_numpy) · [Integer division //](#integer-division-) · [Conditional expression](#conditional-expression) · [Outlier weights](#outlier-weights) · [Per-tensor vs group-wise quantization](#per-tensor-vs-group-wise-quantization) · [reshape(-1, group)](#reshape-1-group) · [keepdims in group scales](#keepdims-in-group-scales) · [Memory-bandwidth bound inference](#memory-bandwidth-bound-inference) · [Loss as a quality measure](#loss-as-a-quality-measure) · [GPTQ, AWQ, bitsandbytes, llama.cpp](#gptq-awq-bitsandbytes-llamacpp) · [QLoRA](#qlora) · [Unsigned vs signed integers](#unsigned-vs-signed-integers)

---

## Quantization  `(concept)`
**1. Definition:** Representing a model's numbers with fewer bits (for example int8 or int4 instead of float32), using a scale factor to map between the two.
**2. In plain words:** Weights take less space, so the model uses less memory and moves less data per token, which usually makes it faster and cheaper. The price is rounding error. int8 is nearly lossless; 4-bit needs extra tricks.
**3. Example to quote:** > "Quantizing 7B parameters from float16 (14 GB) to int4 gets roughly 3.5 GB, at some quality cost."

**Seen in:** `main.py`, docstring.

## Symmetric quantization  `(concept)`
**1. Definition:** A scheme that maps floats to signed integers using `scale = max(|w|) / (2^(bits-1) - 1)` and `q = round(w / scale)`, so zero maps to zero and the range is symmetric around it.
**2. In plain words:** Find the biggest weight, decide it maps to the biggest integer (127 for int8), and round everything else proportionally. It is the simplest recipe and the one the script implements.
**3. Example to quote:** > "For int8 the scale is max abs weight divided by 127; the weight -1.274 became -127."
```python
scale = np.abs(w).max() / 127
q = np.clip(np.round(w / scale), -127, 127)
```
**Seen in:** `main.py`, `quantize_symmetric`.

## Scale factor  `(concept)`
**1. Definition:** The float multiplier that converts between integer codes and real weights (`w_hat = q * scale`).
**2. In plain words:** It is the ruler's unit length. A large scale means coarse steps and bigger rounding errors. Each weight's error is at most scale/2.
**3. Example to quote:** > "My toy example had scale 0.01003, so errors were at most about 0.005."

**Seen in:** `main.py`.

## De-quantization  `(concept)`
**1. Definition:** Converting integer codes back to approximate real numbers by multiplying by the scale.
**2. In plain words:** Weights are stored small but turned back into real numbers when computing. The restored numbers are close to, not exactly equal to, the originals.
**3. Example to quote:** > "-0.317 was stored as -32 and restored as -0.321."

**Seen in:** `main.py`, `dequantize`.

## Simulated (fake) quantization  `(concept)`
**1. Definition:** Quantizing then immediately de-quantizing, keeping values as floats, to measure the error without special kernels.
**2. In plain words:** It does not save memory in this script; it shows how much quality you would lose. Real engines store the integers and use specialised code.
**3. Example to quote:** > "My simulation measures the quality hit on a laptop, while real engines like bitsandbytes actually store int4."

**Seen in:** `main.py`, docstring, `fake_quantize_model`.

## Power operator ** and qmax  `(syntax)`
**1. Definition:** `**` raises to a power; `qmax = 2 ** (bits - 1) - 1` is the largest positive integer for a signed `bits`-bit format.
**2. In plain words:** For 8 bits it is 127, for 4 bits it is 7. One bit is reserved for the sign.
**3. Example to quote:** > "int4 has only 7 positive levels, so each step is coarse."
```python
2 ** (8 - 1) - 1  # 127
```
**Seen in:** `main.py`, `quantize_symmetric`.

## np.abs and .max()  `(third-party: numpy)`
**1. Definition:** `np.abs` returns absolute values; `.max()` returns the largest element.
**2. In plain words:** Together they find the largest weight magnitude, which sets the scale.
**3. Example to quote:** > "`np.abs(w).max()` was 1.274 for my toy weights."

**Seen in:** `main.py`.

## np.round and np.clip  `(third-party: numpy)`
**1. Definition:** `np.round` rounds to nearest integer; `np.clip(a, lo, hi)` limits values into a range.
**2. In plain words:** Rounding creates the integer codes; clipping makes sure none fall outside the allowed range (-127 to 127), which would otherwise wrap around.
**3. Example to quote:** > "I clip after rounding so an edge value cannot overflow int8."

**Seen in:** `main.py`, `quantize_symmetric`.

## .astype and integer overflow  `(third-party: numpy)`
**1. Definition:** `.astype(dtype)` converts an array's data type; converting values outside the type's range wraps around silently.
**2. In plain words:** 200 as int8 becomes -56 with no error. That is a silent bug, caught by printing `q.min()` and `q.max()`.
**3. Example to quote:** > "Without `np.clip`, 128 would wrap to -128 and the weights would be wrong with no warning."
```python
np.array([200]).astype(np.int8)  # array([-56], dtype=int8)
```
**Seen in:** `main.py`, `quantize_symmetric`, BREAK IT.

## Quantization error  `(concept)`
**1. Definition:** The difference between an original weight and its restored value after quantization.
**2. In plain words:** Measured here as the mean absolute error: 0.000186 at int8 and 0.003382 at int4 on a random matrix, about 18 times larger at int4.
**3. Example to quote:** > "int4 gave about 18 times the error of int8 on the same matrix."

**Seen in:** `main.py`, section 2.

## Memory saving by bit width  `(concept)`
**1. Definition:** Memory is proportional to bits per weight: int8 is 4x smaller than float32, int4 is 8x smaller.
**2. In plain words:** A 768x768 float32 matrix takes 2,304 KiB; at int8 about 576 KiB; at int4 about 288 KiB. The model's matrices go from 170 MB to 42 MB at int8.
**3. Example to quote:** > "The matrix shrank from 2,304 KiB to 288 KiB at int4."

**Seen in:** `main.py`, sections 2 and 3.

## copy.deepcopy  `(standard library)`
**1. Definition:** Creates a fully independent copy of an object, including nested objects.
**2. In plain words:** The quantization functions copy the model first so the original float32 baseline stays untouched for comparison.
**3. Example to quote:** > "I `deepcopy` the model before editing weights so I can compare against the original."

**Seen in:** `main.py`, `fake_quantize_model`.

## p.ndim  `(third-party: torch)`
**1. Definition:** The number of dimensions of a tensor; 2 means a matrix.
**2. In plain words:** The script quantizes only 2-D weight matrices (linear layers) and skips embeddings and 1-D bias or normalisation vectors.
**3. Example to quote:** > "I filter with `p.ndim == 2` to pick the linear-layer matrices."

**Seen in:** `main.py`.

## p.data assignment  `(third-party: torch)`
**1. Definition:** Assigning to a parameter's `.data` replaces its values in place without tracking gradients.
**2. In plain words:** It is how the script swaps in the quantized-then-restored values.
**3. Example to quote:** > "`p.data = torch.from_numpy(...)` overwrites the layer's weights with the restored values."

**Seen in:** `main.py`.

## torch.from_numpy  `(third-party: torch)`
**1. Definition:** Wraps a numpy array as a torch tensor, sharing memory.
**2. In plain words:** It bridges the numpy quantization code and the PyTorch model.
**3. Example to quote:** > "I convert the numpy result back into a tensor with `torch.from_numpy`."

**Seen in:** `main.py`.

## Integer division //  `(syntax)`
**1. Definition:** `a // b` divides and rounds down to a whole number.
**2. In plain words:** `32 // bits` gives the size reduction factor: 4 for int8, 8 for int4.
**3. Example to quote:** > "`32 // 8` is 4, so int8 is 4x smaller."
```python
7 // 2  # 3
```
**Seen in:** `main.py`.

## Conditional expression  `(syntax)`
**1. Definition:** `A if condition else B` evaluates to A or B depending on the condition.
**2. In plain words:** A one-line if/else used to format the shrink factor for different bit widths.
**3. Example to quote:** > "`32 // bits if bits in (8, 4) else 32 / bits` handles 6- and 3-bit cases too."

**Seen in:** `main.py`, section 3.

## Outlier weights  `(concept)`
**1. Definition:** A few weights far larger than the rest, which stretch the quantization scale.
**2. In plain words:** One huge weight forces a coarse step for everyone else, so normal weights round to just a few levels. One outlier of 5.0 raised int8 error from 0.000186 to 0.009804.
**3. Example to quote:** > "A single outlier 250 times typical size wrecked per-tensor scaling."

**Seen in:** `main.py`, section 4.

## Per-tensor vs group-wise quantization  `(concept)`
**1. Definition:** Per-tensor uses one scale for a whole matrix; group-wise uses one scale per small block (such as 64 weights).
**2. In plain words:** With groups an outlier only hurts its own block. On distilgpt2, int4 loss was 9.693 per-tensor but 5.127 group-wise (baseline 4.983). The cost is one extra scale per group.
**3. Example to quote:** > "Group-wise int4 kept loss at 5.13 while per-tensor int4 damaged the model to 9.69."

**Seen in:** `main.py`, section 5.

## reshape(-1, group)  `(third-party: numpy)`
**1. Definition:** Reshapes an array into rows of `group` elements; `-1` lets numpy infer the number of rows.
**2. In plain words:** Each row becomes one quantization group with its own scale.
**3. Example to quote:** > "`reshape(-1, 64)` turns 4,096 values into 64 groups of 64."
```python
np.arange(4096).reshape(-1, 64).shape  # (64, 64)
```
**Seen in:** `main.py`, `fake_quantize_grouped`.

## keepdims in group scales  `(third-party: numpy)`
**1. Definition:** `max(axis=1, keepdims=True)` returns one value per row, kept as a column so it broadcasts.
**2. In plain words:** It gives one scale per group that you can divide each row by directly.
**3. Example to quote:** > "`np.abs(flat).max(axis=1, keepdims=True)` is one scale per group of 64."

**Seen in:** `main.py`, `fake_quantize_grouped`.

## Memory-bandwidth bound inference  `(concept)`
**1. Definition:** LLM generation is usually limited by how fast weights can be read from memory, not by arithmetic speed.
**2. In plain words:** Each token needs to read all the weights. Fewer bytes per weight means faster tokens, which is why quantized models are often faster as well as smaller.
**3. Example to quote:** > "Quantization speeds inference because there are fewer bytes to move per token."

**Seen in:** `WALKTHROUGH.md`.

## Loss as a quality measure  `(concept)`
**1. Definition:** Using the language-model loss on a fixed text to check how much quality changed after modification.
**2. In plain words:** A small increase means little damage; a big jump (4.98 to 9.69) means the model is badly hurt.
**3. Example to quote:** > "int8 loss was 4.957 versus the 4.983 baseline: essentially free."

**Seen in:** `main.py`, section 3.

## GPTQ, AWQ, bitsandbytes, llama.cpp  `(concept)`
**1. Definition:** Real-world quantization methods and libraries that store integer weights and use specialised kernels, often with group-wise scales.
**2. In plain words:** They are smarter versions of this lesson's idea. You will see how they plug in during module 06.
**3. Example to quote:** > "GPTQ and AWQ choose smarter rounding but still rely on group-wise scales."

**Seen in:** `main.py`, docstring.

## QLoRA  `(concept)`
**1. Definition:** Fine-tuning that keeps the base model in 4-bit and trains small LoRA adapters in higher precision on top.
**2. In plain words:** It lets you fine-tune a large model on a small GPU. Covered in module 06.
**3. Example to quote:** > "QLoRA combines 4-bit quantization with LoRA adapters to fit training in less memory."

**Seen in:** `WALKTHROUGH.md`.

## Unsigned vs signed integers  `(concept)`
**1. Definition:** Signed integers can hold negative values; unsigned (`uint8`) cannot and wrap negatives to large positives.
**2. In plain words:** Weights can be negative, so signed `int8` is needed. Using `uint8` would turn -1 into 255.
**3. Example to quote:** > "I use `np.int8` because negative weights would wrap with `uint8`."

**Seen in:** `main.py`, BREAK IT comment.
