# Terms — Numerical precision and training stability

Folder: `15_neural_networks/10_precision_and_training_stability` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [15 README](../README.md)

**Terms in this folder:** [Floating-point number (exponent and mantissa)](#floating-point-number-exponent-and-mantissa) · [float32 (fp32)](#float32-fp32) · [float16 (fp16, half)](#float16-fp16-half) · [bfloat16 (bf16)](#bfloat16-bf16) · [fp8 (e4m3, e5m2)](#fp8-e4m3-e5m2) · [Machine epsilon (eps)](#machine-epsilon-eps) · [Overflow and underflow](#overflow-and-underflow) · [Subnormal (denormal) numbers](#subnormal-denormal-numbers) · [Absorption (swamping)](#absorption-swamping) · [Loss scaling (static and dynamic)](#loss-scaling-static-and-dynamic) · [Master weights](#master-weights) · [Mixed precision (autocast)](#mixed-precision-autocast) · [Gradient clipping (global norm)](#gradient-clipping-global-norm) · [Stable softmax](#stable-softmax) · [NaN (not a number)](#nan-not-a-number) · [Anomaly detection (detect_anomaly)](#anomaly-detection-detectanomaly) · [Loss spike](#loss-spike) · [Quantisation (inference)](#quantisation-inference)

---

## Floating-point number (exponent and mantissa)  `(concept)`
**1. Definition:** A number stored as sign, exponent (range) and mantissa (precision).
**2. In plain words:** Scientific notation in binary.
**3. Example to quote:** > "fp32: 8 exponent + 23 mantissa bits."
**Seen in:** `main.py`, block 1

## float32 (fp32)  `(concept)`
**1. Definition:** 32-bit float: max 3.4e+38, eps 1.2e-07.
**2. In plain words:** The safe default.
**3. Example to quote:** > "Master weights and optimiser state."
**Seen in:** `main.py`, block 1

## float16 (fp16, half)  `(concept)`
**1. Definition:** 16-bit float with 5 exponent and 10 mantissa bits: max 65,504, smallest normal 6.1e-05.
**2. In plain words:** Precise but narrow range.
**3. Example to quote:** > "`float16(70000) = inf`"
**Seen in:** `main.py`, block 2

## bfloat16 (bf16)  `(concept)`
**1. Definition:** 16-bit float with fp32's 8 exponent bits and only 7 mantissa bits.
**2. In plain words:** Wide range, coarse precision; default for LLM training.
**3. Example to quote:** > "`bfloat16(1 + 0.001) = 1.0`"
**Seen in:** `main.py`, block 2

## fp8 (e4m3, e5m2)  `(concept)`
**1. Definition:** 8-bit floats: e4m3 (max 448, more precision) and e5m2 (max 57,344, more range).
**2. In plain words:** Tiny floats used with scaling in recent training and inference.
**3. Example to quote:** > "`torch.float8_e4m3fn`"
**Seen in:** `main.py`, block 1

## Machine epsilon (eps)  `(concept)`
**1. Definition:** The smallest relative difference the format can represent near 1: `finfo.eps`.
**2. In plain words:** The resolution of the number line.
**3. Example to quote:** > "fp16: 9.8e-04; bf16: 7.8e-03."
**Seen in:** `main.py`, block 1

## Overflow and underflow  `(concept)`
**1. Definition:** Overflow: result exceeds the max and becomes inf. Underflow: result is smaller than the smallest representable and becomes 0.
**2. In plain words:** Too big for the box / too small to see.
**3. Example to quote:** > "`float16(1e-8) = 0.0`"
**Seen in:** `main.py`, block 2

## Subnormal (denormal) numbers  `(concept)`
**1. Definition:** Numbers smaller than the smallest normal value, stored with reduced precision.
**2. In plain words:** The last, blurry rungs of the ladder before zero.
**3. Example to quote:** > "fp16 subnormals reach about 6e-08."
**Seen in:** `WALKTHROUGH.md`, theory

## Absorption (swamping)  `(concept)`
**1. Definition:** Adding a number smaller than half a spacing to a larger one leaves the larger unchanged.
**2. In plain words:** Adding a drop to a bucket and not seeing the level move.
**3. Example to quote:** > "float16 running sum stuck at 0.25."
**Seen in:** `main.py`, block 2

## Loss scaling (static and dynamic)  `(concept)`
**1. Definition:** Multiplying the loss by S before backward and dividing gradients by S afterwards to avoid fp16 underflow; dynamic scaling adapts S.
**2. In plain words:** Magnifying glass for tiny gradients.
**3. Example to quote:** > "`2**16`"
**Seen in:** `main.py`, block 3

## Master weights  `(concept)`
**1. Definition:** An fp32 copy of the parameters that the optimiser updates.
**2. In plain words:** The accurate ledger beside the quick scratch pad.
**3. Example to quote:** > "Autocast leaves `weight.dtype` as float32."
**Seen in:** `main.py`, block 6

## Mixed precision (autocast)  `(concept)`
**1. Definition:** Running selected operations in lower precision while keeping others and master weights in fp32.
**2. In plain words:** Cheap arithmetic where safe, careful arithmetic where not.
**3. Example to quote:** > "`torch.autocast('cpu', dtype=torch.bfloat16)`"
**Seen in:** `main.py`, block 6

## Gradient clipping (global norm)  `(concept)`
**1. Definition:** Scale all gradients by `min(1, max_norm / norm)` so the total gradient norm is at most `max_norm`.
**2. In plain words:** A speed limiter.
**3. Example to quote:** > "15.50 -> 1.0000"
**Seen in:** `main.py`, block 4

## Stable softmax  `(concept)`
**1. Definition:** Subtract the maximum logit before exponentiating; identical result, no overflow.
**2. In plain words:** Shift all scores down so the biggest is 0.
**3. Example to quote:** > "`F.softmax(logits.half())`"
**Seen in:** `main.py`, block 5

## NaN (not a number)  `(concept)`
**1. Definition:** A special float from undefined operations (`0/0`, `inf - inf`) that contaminates everything it touches.
**2. In plain words:** A stain that spreads.
**3. Example to quote:** > "Naive softmax in fp16 gives `[nan, 0, 0]`."
**Seen in:** `main.py`, block 5

## Anomaly detection (detect_anomaly)  `(library)`
**1. Definition:** A PyTorch debug mode that raises when a backward operation outputs NaN and prints the forward trace.
**2. In plain words:** A smoke detector for the autograd graph.
**3. Example to quote:** > "`SqrtBackward0` returned nan values."
**Seen in:** `main.py`, block 5

## Loss spike  `(concept)`
**1. Definition:** A sudden jump in training loss, often a precursor to divergence; handled by clipping, skipping batches or rewinding to a checkpoint.
**2. In plain words:** A hiccup that can turn into a crash.
**3. Example to quote:** > "Log the gradient norm to see it coming."
**Seen in:** `WALKTHROUGH.md`, theory

## Quantisation (inference)  `(concept)`
**1. Definition:** Storing weights in fewer bits (int8, int4, fp8) for cheaper inference.
**2. In plain words:** Rounding the weights to a coarser grid.
**3. Example to quote:** > "Module 06."
**Seen in:** `WALKTHROUGH.md`, theory
