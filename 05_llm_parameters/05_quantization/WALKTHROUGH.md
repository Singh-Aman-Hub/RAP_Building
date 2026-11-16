# Step 05 - Quantization

## The story so far
Chapter 15, Squeezing into VRAM. Chapter 14 gave you good output from a small model; scale it up and the first wall you hit is memory, as [step 04](../04_max_new_tokens_and_context_window/WALKTHROUGH.md) hinted.

## The pain point
Float32 spends 4 bytes on every weight. A 7B model needs 28 GB just for weights, more than almost any single GPU. Most of that precision is wasted: the model works nearly as well with far fewer bits.

## The spark
Store each weight as a small integer plus a scale factor: `int8 = round(w / scale)`. That is 4x smaller (float16 is 2x, int4 is 8x). One huge outlier weight stretches the scale and ruins the rest, so scale per small group of weights instead of per tensor.

## Goal
Shrink float32 weights to int8/int4 by hand, measure memory saved and accuracy lost, and learn why group-wise scaling matters.

## Run it
```bash
.venv/bin/python 05_llm_parameters/05_quantization/main.py
```
This script **simulates** quantization (quantize -> de-quantize, kept as floats) so the error can be measured on a laptop. Real engines store the integers and use special kernels (bitsandbytes, GPTQ, AWQ, llama.cpp); you'll see how they plug in during [module 06](../../06_llm_training_and_lora/).

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Quantisation | low-bit weights, compression (loosely), int8/int4 | Yes | Store weights with fewer bits and rescale when needed |
| Symmetric vs asymmetric quantisation | zero-point or no zero-point | No | Symmetric uses only a scale (zero maps to zero); asymmetric adds a zero-point to use the full range for skewed data |
| Per-tensor vs per-channel vs group-wise | scale granularity | Different | One scale for the whole matrix, one per output channel, or one per small group of weights (here 64) |
| PTQ vs QAT | post-training vs quantisation-aware training | No | PTQ rounds a finished model; QAT trains with simulated rounding |
| GPTQ / AWQ / GGUF (llama.cpp) / bitsandbytes (NF4) | quantisation methods and formats | Different | All store low-bit weights; they differ in how scales are chosen (calibration data, outlier handling) and in file format and kernels |
| Fake quantisation | simulated quantisation | Yes | Quantise then de-quantise back to float to measure error without special kernels |
| int8 / int4 / fp16 / bf16 / fp8 | numeric formats | Different | Integers need scales; fp16/bf16 are 16-bit floats with different range and precision |
| Outlier | massive activation / large weight | Related | One huge value stretches the scale and wastes the integer range |
| QLoRA | 4-bit base model + LoRA adapters | Not the same as plain int4 | Quantised frozen base for memory, small trainable adapters on top (module 06) |

## Code, step by step

### 1. Module docstring: the recipe  (`main.py` lines 1–22)

```python
"""
GOAL:   Understand quantization by doing it: shrink float32 weights to int8/int4, measure the
        memory saved and the accuracy lost, on a toy matrix and on real distilgpt2 weights.

RUN:    from this folder:  python main.py
        from repo root:    .venv/bin/python 05_llm_parameters/05_quantization/main.py
EXPECTED OUTPUT (short): int8 is 4x smaller with tiny error; int4 is 8x smaller with ~18x bigger
        error; distilgpt2 loss barely moves at int8 but degrades badly at int4 per-tensor (group-wise rescues it).
NEEDS:  numpy, torch, transformers (distilgpt2 cached). ~20 s.

Quantization = store each weight with FEWER BITS.  float32 uses 32 bits (4 bytes); int8 uses 8 bits.
Smaller weights -> less memory, less memory bandwidth (usually the bottleneck for LLM inference),
so faster and cheaper - at the price of some rounding error.

Symmetric quantization recipe (per tensor):
    scale  = max(|w|) / (2^(bits-1) - 1)          # e.g. max/127 for int8
    q      = round(w / scale)  (stored as integers)
    w_hat  = q * scale          (de-quantize when you need real numbers)
This file SIMULATES quantization (quantize -> de-quantize, stored as float) so you can measure the
error on any laptop. Real engines (bitsandbytes, GPTQ, AWQ, llama.cpp) store the integers and use
special kernels - see 06_llm_training_and_lora for QLoRA.
"""
```

- **In simple words:** Explains quantisation as storing each weight with fewer bits, and gives the three-line recipe.
- **Exactly:** Docstring only. Recipe: `scale = max(|w|) / (2^(bits-1) - 1)`, `q = round(w / scale)`, `w_hat = q * scale`. This file simulates it; real engines (bitsandbytes, GPTQ, AWQ, llama.cpp) store the integers and use special kernels. (A stray `k` on line 10 was removed.)
- **Why it is here:** Memory bandwidth is usually the bottleneck for LLM inference, so smaller weights mean faster and cheaper serving.

### 2. Imports  (`main.py` lines 24–27)

```python
import copy
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
```

- **In simple words:** numpy for the maths, torch and transformers for the real model, `copy` to duplicate a model.
- **Exactly:** `copy.deepcopy(model)` creates an independent copy so we never damage the baseline.
- **Why it is here:** Without a copy, quantising in place would corrupt the baseline for later comparisons.

### 3. quantize_symmetric: float numbers to small integers  (`main.py` lines 30–35)

```python
def quantize_symmetric(w, bits=8):
    """Return (integer_tensor, scale).  `w` is any numpy array of floats."""
    qmax = 2 ** (bits - 1) - 1                    # 127 for int8, 7 for int4
    scale = np.abs(w).max() / qmax                # shape: () one scale for the whole tensor
    q = np.clip(np.round(w / scale), -qmax, qmax).astype(np.int8)   # shape: same as w; int8 container even for 4-bit demo
    return q, scale
```

- **In simple words:** Find the biggest weight, decide how many steps fit on each side of zero, and round every weight to the nearest step.
- **Exactly:** `qmax = 2**(bits-1) - 1` is 127 for int8 and 7 for int4. `scale = max(|w|) / qmax` is a single scalar for the whole tensor. `np.round(w / scale)` gives the nearest integer step, `np.clip(..., -qmax, qmax)` guards the range, `.astype(np.int8)` stores it; for the 4-bit demo an int8 container is still used. Returns `(q, scale)`.
- **Why it is here:** Integers take 1 byte instead of 4. The scale is the only extra number to remember.

### 4. dequantize: back to floats  (`main.py` lines 38–39)

```python
def dequantize(q, scale):
    return q.astype(np.float32) * scale           # shape: same as q
```

- **In simple words:** Multiply the integers by the scale to get approximate original numbers.
- **Exactly:** `q.astype(np.float32) * scale`, same shape as `q`. Each restored value equals the original within half a step (`scale / 2`).
- **Why it is here:** Needed to compute with the weights and to measure the rounding error.

### 5. A toy example with five weights  (`main.py` lines 42–50)

```python
# ---------------------------------------------------------------------------
# 1. Toy example you can read
# ---------------------------------------------------------------------------
w = np.array([0.823, -0.317, 0.054, -1.274, 0.447], dtype=np.float32)   # shape: (5,)
q, s = quantize_symmetric(w, 8)
print("weights  :", w)
print("int8     :", q, f"(scale={s:.5f})")
print("restored :", np.round(dequantize(q, s), 4))
print("abs error:", np.round(np.abs(w - dequantize(q, s)), 4))
```

- **In simple words:** See the whole process on numbers small enough to read.
- **Exactly:** `w = [0.823, -0.317, 0.054, -1.274, 0.447]`. The biggest is 1.274 so `scale = 1.274/127 = 0.01003`. Real output: int8 `[82, -32, 5, -127, 45]`; restored `[0.8226, -0.321, 0.0502, -1.274, 0.4514]`; abs error `[0.0004, 0.004, 0.0038, 0.0, 0.0044]` (all below `scale/2 = 0.005`).
- **Why it is here:** Shows that error is bounded by half a step and the largest weight is exact.

### 6. Error and memory on a 768 by 768 matrix, int8 vs int4  (`main.py` lines 52–62)

```python
# ---------------------------------------------------------------------------
# 2. A bigger random matrix: error and memory at 8 vs 4 bits
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
W = rng.normal(0, 0.02, size=(768, 768)).astype(np.float32)    # shape: (d_in, d_out) = (768, 768); typical LLM weight scale
print("\nmatrix 768x768 (float32 =", W.nbytes // 1024, "KiB)")
for bits in (8, 4):
    q, s = quantize_symmetric(W, bits)
    err = np.abs(W - dequantize(q, s)).mean()
    mem = W.size * bits / 8 / 1024
    print(f"  int{bits}: ~{mem:6.0f} KiB ({32 // bits}x smaller), mean abs error = {err:.6f}")
```

- **In simple words:** Repeat on a realistic weight matrix and compare memory and mean error.
- **Exactly:** `W = rng.normal(0, 0.02, (768, 768))` float32 is 2,304 KiB. int8: `576 KiB (4x smaller)`, mean abs error `0.000186`. int4: `288 KiB (8x smaller)`, mean abs error `0.003382`, about 18x larger. `mem = W.size * bits / 8 / 1024` is the ideal packed size.
- **Why it is here:** Fewer bits save memory but increase error roughly 2x per bit removed.

### 7. Load the real model and a test text  (`main.py` lines 64–71)

```python
# ---------------------------------------------------------------------------
# 3. Quantize ALL weight matrices of real distilgpt2 and measure what happens to the loss
# ---------------------------------------------------------------------------
tok = AutoTokenizer.from_pretrained("distilgpt2")
text = ("The quick brown fox jumps over the lazy dog. Machine learning models learn patterns from data "
        "and use them to make predictions on new examples.")
batch = tok(text, return_tensors="pt")      # input_ids, attention_mask: shape (1, seq_len)
base = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
```

- **In simple words:** Take distilgpt2 and a sentence we can score.
- **Exactly:** `batch` holds `input_ids` and `attention_mask`, shape `(1, seq_len)`. `base` is the float32 model in eval mode.
- **Why it is here:** Real weights test whether the toy conclusions hold.

### 8. Loss helper  (`main.py` lines 74–76)

```python
def loss_of(m):
    with torch.no_grad():
        return m(**batch, labels=batch["input_ids"]).loss.item()
```

- **In simple words:** Score a model on our text.
- **Exactly:** `m(**batch, labels=batch["input_ids"]).loss.item()` is the next-token cross-entropy as a float. Lower is better.
- **Why it is here:** Loss gives one number that shows damage from quantisation.

### 9. Fake-quantise all linear-layer matrices  (`main.py` lines 79–86)

```python
def fake_quantize_model(model, bits):
    """Return a copy of `model` whose 2-D weight matrices were quantized then de-quantized."""
    m = copy.deepcopy(model)
    for name, p in m.named_parameters():
        if p.ndim == 2 and "wte" not in name and "wpe" not in name:    # linear-layer matrices only
            q, s = quantize_symmetric(p.detach().numpy(), bits)
            p.data = torch.from_numpy(dequantize(q, s))
    return m
```

- **In simple words:** Round every big matrix of the model to the chosen bit width and back to floats.
- **Exactly:** Loops over `named_parameters()`; the condition `p.ndim == 2` and not `wte` or `wpe` selects the linear-layer weight matrices (the embeddings are left alone, which is common practice). `p.detach().numpy()` gives the array, `p.data = torch.from_numpy(dequantize(q, s))` writes the rounded values back.
- **Why it is here:** Leaves biases and LayerNorm in float, as real methods usually do, since they are tiny and sensitive.

### 10. Measure loss at 8, 6, 4 and 3 bits  (`main.py` lines 89–94)

```python
print(f"\nfloat32 baseline loss: {loss_of(base):.3f}")
n_matrix_params = sum(p.numel() for n, p in base.named_parameters() if p.ndim == 2 and "wte" not in n and "wpe" not in n)
for bits in (8, 6, 4, 3):
    print(f"int{bits} (simulated) loss: {loss_of(fake_quantize_model(base, bits)):.3f}   "
          f"-> matrices would shrink {32 // bits if bits in (8, 4) else 32 / bits:.1f}x "
          f"({n_matrix_params * 4 / 1e6:.0f} MB -> {n_matrix_params * bits / 8 / 1e6:.0f} MB)")
```

- **In simple words:** Quantise the whole model at different precision levels and see how well it still predicts text.
- **Exactly:** Real output: baseline loss `4.983`; int8 `4.957` (no harm; the small change is noise); int6 `7.433`; int4 `9.693`; int3 `9.224`. Matrices shrink 4.0x (170 MB to 42 MB) at int8 and 8.0x (21 MB) at int4. For 4 bits and below the loss is near that of an untrained model (ln 50257 = 10.8).
- **Why it is here:** 8-bit is nearly free, but naive per-tensor 4-bit destroys the model: that is the problem group-wise methods solve.

### 11. One outlier ruins the per-tensor scale  (`main.py` lines 96–104)

```python
# ---------------------------------------------------------------------------
# 4. Outliers: ONE huge weight ruins per-tensor scaling -> why real methods quantize in small groups
# ---------------------------------------------------------------------------
W2 = W.copy()                                     # shape: (768, 768)
W2[0, 0] = 5.0                                    # a single outlier, 250x typical size
q, s = quantize_symmetric(W2, 8)
q0, s0 = quantize_symmetric(W, 8)
print(f"\nint8 mean abs error, no outlier  : {np.abs(W - dequantize(q0, s0)).mean():.6f}")
print(f"int8 mean abs error, one outlier : {np.abs(W2 - dequantize(q, s))[1:].mean():.6f} -> the scale was stretched by the outlier")
```

- **In simple words:** Plant a single huge weight and see the error of all the others grow.
- **Exactly:** `W2[0, 0] = 5.0` is 250x the typical weight (std 0.02). The scale stretches to `5.0/127 = 0.039`, so ordinary weights (about 0.02) are rounded onto very few levels. Real output: mean abs error `0.000186` without the outlier vs `0.009804` with it (excluding the outlier itself), about 53x worse.
- **Why it is here:** Real LLMs have outlier weights and activations, which is why simple per-tensor int4 fails.

### 12. Group-wise quantisation: one scale per group of 64  (`main.py` lines 106–119)

```python
# ---------------------------------------------------------------------------
# 5. Fix: GROUP-WISE quantization - one scale per small group of weights (what GPTQ/AWQ/llama.cpp do).
#    An outlier now only hurts its own group of 64 numbers.
# ---------------------------------------------------------------------------
def fake_quantize_grouped(model, bits, group=64):
    m = copy.deepcopy(model)
    for name, p in m.named_parameters():
        if p.ndim == 2 and "wte" not in name and "wpe" not in name:
            flat = p.detach().numpy().reshape(-1, group)          # shape: (n_weights / group, group); every row = one group
            qmax = 2 ** (bits - 1) - 1
            scales = np.abs(flat).max(axis=1, keepdims=True) / qmax   # shape: (n_groups, 1) one scale PER GROUP
            q = np.clip(np.round(flat / scales), -qmax, qmax)
            p.data = torch.from_numpy((q * scales).reshape(p.shape).astype(np.float32))
    return m
```

- **In simple words:** Give every little group of 64 weights its own scale so an outlier only hurts its neighbours.
- **Exactly:** `reshape(-1, group)` makes a `(n/64, 64)` matrix, each row one group. `np.abs(flat).max(axis=1, keepdims=True) / qmax` gives one scale per group: shape `(n_groups, 1)`. The rest is the same round/clip/rescale. `reshape(p.shape).astype(np.float32)` restores the matrix.
- **Why it is here:** This is the idea in GPTQ, AWQ and llama.cpp's Q4 formats; the cost is one extra scale (16 bits) per 64 weights, about 0.25 extra bits per weight.

### 13. Compare per-tensor with group-wise on the real model  (`main.py` lines 121–125)

```python
print("\nper-tensor vs group-wise (group=64) loss on distilgpt2 (baseline %.3f):" % loss_of(base))
for bits in (8, 4):
    print(f"  int{bits}: per-tensor {loss_of(fake_quantize_model(base, bits)):.3f} | "
          f"group-wise {loss_of(fake_quantize_grouped(base, bits)):.3f}")
print("  (cost of groups: one extra scale number per 64 weights = a little more memory)")
```

- **In simple words:** Same model, same bit width, two ways of choosing scales.
- **Exactly:** Real output (baseline 4.983): int8 per-tensor `4.957` vs group-wise `4.987`; int4 per-tensor `9.693` vs group-wise `5.127`. Group-wise 4-bit keeps the model usable at 8x smaller.
- **Why it is here:** The proof that granularity of scales, not bit width alone, determines quality.

### 14. Experiments and break-it comments  (`main.py` lines 127–130)

```python
# TRY IT: change `text` to something very different - does int4 hurt more or less?
# TRY IT: quantize in groups of 64 values (reshape to (-1, 64), one scale per row) and re-measure the outlier case.
# BREAK IT: remove np.clip - values can overflow the int8 range and wrap around. Print q.min()/q.max().
# BREAK IT: use dtype=np.uint8 for q: negative weights wrap to huge numbers. Why is signed int needed?
```

- **In simple words:** Try different text, dtypes and what breaks if you remove clipping.
- **Exactly:** Removing `np.clip` can let values overflow the int8 range and wrap; `uint8` cannot hold negative weights.
- **Why it is here:** Overflow bugs are silent, so check `q.min()` and `q.max()`.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `2 ** (bits - 1) - 1` | `**` = power. int8 -> `2**7 - 1` = 127 | - |
| `.astype(np.int8)` | convert dtype (values must fit: -128..127) | `np.array([200]).astype(np.int8)` wraps to `-56` |
| `copy.deepcopy(model)` | full independent copy, so the original is untouched | - |
| `p.data = tensor` | replace a parameter's values in place | - |
| `reshape(-1, 64)` | `-1` = "figure this dimension out" | 4096 values -> shape `(64, 64)` |
| `A if cond else B` | conditional expression | `32 // bits if bits in (8, 4) else 32 / bits` |
| `//` | integer (floor) division | `7 // 2` = 3 |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `np.abs(w).max()` | numpy | largest magnitude -> sets the scale | - | float |
| `np.round`, `np.clip(a, lo, hi)` | numpy | round / limit to a range | lo, hi | array |
| `copy.deepcopy` | stdlib | deep copy any object | - | copy |
| `torch.from_numpy(arr)` | torch | wrap a numpy array as a tensor | - | tensor |
| `p.ndim` | torch | number of dimensions (2 = weight matrix) | - | int |

## The recipe
```
scale = max(|w|) / 127            (int8)
q     = clip(round(w / scale), -127, 127)      <- stored: 1 byte each
w_hat = q * scale                              <- used at compute time
error = |w - w_hat|  (up to scale/2 per weight)
```

## Execution trace
```
weights  : [ 0.823 -0.317  0.054 -1.274  0.447]
int8     : [  82  -32    5 -127   45] (scale=0.01003)
restored : [ 0.8226 -0.321   0.0502 -1.274   0.4514]
abs error: [0.0004 0.004  0.0038 0.     0.0044]

matrix 768x768 (float32 = 2304 KiB)
  int8: ~576 KiB (4x smaller), mean abs error = 0.000186
  int4: ~288 KiB (8x smaller), mean abs error = 0.003382

float32 baseline loss: 4.983
int8 (simulated) loss: 4.957     -> matrices 170 MB -> 42 MB
int6 (simulated) loss: 7.433
int4 (simulated) loss: 9.693     -> 21 MB   <- model is badly damaged

int8 mean abs error, no outlier  : 0.000186
int8 mean abs error, one outlier : 0.009804 -> the scale was stretched by the outlier

per-tensor vs group-wise (group=64) loss on distilgpt2 (baseline 4.983):
  int8: per-tensor 4.957 | group-wise 4.987
  int4: per-tensor 9.693 | group-wise 5.127
```
Key lessons:
1. **int8 is nearly free** (loss 4.98 -> 4.96): 4x smaller memory, usually near-identical quality.
2. **int4 per-tensor wrecks the model** (loss 9.7) because one scale must cover the whole matrix and a few **outlier** weights stretch it so all normal weights round to a few levels.
3. **Group-wise scales fix most of it** (int4 loss 5.13): each block of 64 weights gets its own scale. Cost: one extra scale number per group (6% overhead at group 64 in the 16-bit-scale case - `solutions/ex05_group_size_curve.py`). Methods like GPTQ/AWQ are smarter versions of this.
4. LLM inference is usually **memory-bandwidth bound** - moving fewer bytes per token is what makes quantized models faster, not just smaller.
5. **QLoRA** (module 06): keep the base model in 4-bit and train small LoRA adapters in higher precision on top.

## Experiments
- Different text in `text` - is int4 more or less harmful?
- Group size 16 vs 64 vs 256 for int4; plot loss vs overhead.
- Quantize only the attention matrices, then only the MLP: which is more sensitive?

## Debugging practice
Remove `np.clip`: values near the extremes can round outside -127..127 and `astype(np.int8)` silently WRAPS (e.g. 128 -> -128). No error, wrong numbers. Print `q.min(), q.max()` - checking value ranges is a core numerical-debugging habit. Compare: a wrong dtype conversion that gives no traceback vs a shape error that does.

## Rebuild checklist
- [ ] Write `quantize_symmetric` and `dequantize`; show the error on 5 numbers.
- [ ] Compute memory for 7B parameters at 16, 8 and 4 bits.
- [ ] Explain outliers and why groups help.
- [ ] Explain the accuracy vs memory vs speed trade-off in two sentences.

## The new problem (cliffhanger)
Inference now fits. But we want to teach the model something new, and training needs gradients and optimizer state, 4 to 16 times more memory than the weights alone. First learn the knobs of training: [step 06, training parameters](../06_training_parameters/WALKTHROUGH.md).

> **Say it aloud (interview box):** Quantization stores each weight in fewer bits as a small integer times a scale factor, so int8 is 4 times smaller than float32 and int4 is 8 times smaller. A single outlier stretches a per-tensor scale and destroys accuracy, which is why int4 needs group-wise scaling. I always re-measure quality after quantizing, because the memory saving alone proves nothing.
