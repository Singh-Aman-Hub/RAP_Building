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
This script **simulates** quantization (quantize -> de-quantize, kept as floats) so the error can be measured on a laptop. Real engines store the integers and use special kernels (bitsandbytes, GPTQ, AWQ, llama.cpp); you'll see how they plug in during [module 06](../../06_llm_training_and_fast_inference/).

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
