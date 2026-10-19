# Module 06 — Solutions and notes

Runnable solutions: [lr_sweep.py](lr_sweep.py), [lora_rank_sweep.py](lora_rank_sweep.py). Measured results are in the step WALKTHROUGH files.

## Step 01
- `BLOCK_SIZE = 16` -> `1206 // 16 = 75` blocks.
- Traceback: `RuntimeError: shape '[25, 49]' is invalid for input of size 1200` — the element count (1200) is not divisible into 25x49. Look at the last line first; the line number above it points at `.view(...)`.
- Labels are a copy because the shift by one happens inside the model; `-100` is the "ignore index" of cross-entropy loss, used for padding.

## Step 02
- With `lr=5e-4` the mean loss per epoch was 3.23, 1.59, 0.99 vs 3.81, 2.66, 2.14 at 5e-5 and 4.71, 4.66, 4.64 at 5e-7.
- Without `zero_grad()` there is no exception: gradients from all earlier steps are summed, so the updates become inconsistent. Silent bugs are why you plot or print the loss.
- `clip_grad_norm_` returns the total norm *before* clipping, which is useful for spotting spikes.

## Step 03
- Validation loss typically stops improving by epoch 4-6 on this tiny corpus; train loss keeps going down.
- Out-of-style text has much higher perplexity because the model has specialised toward the support-ticket style.

## Step 04
- 4096x4096: full = 16,777,216; r=8 -> 65,536 (0.39%); r=64 -> 524,288 (3.1%).
```python
import numpy as np
def lora_forward(W, A, B, x, alpha, r):
    return W @ x + (alpha / r) * (B @ (A @ x))
```
- `B = 0` makes the update zero at the start; `A` random breaks symmetry so gradients can flow into `B`.

## Step 05
- r=1/8/32 -> 18,432 / 147,456 / 589,824 trainable parameters; loss after 10 epochs 2.902 / 2.145 / 1.734.
- Adding `c_proj` (768x768 per block) adds more parameters per block.
- Wrong name error: `ValueError: Target modules {'q_proj'} not found in the base model`. Fix: `print([n for n, _ in model.named_modules()])`.

## Step 06 written task: running step 05 as QLoRA
1. `pip install bitsandbytes` and use a CUDA machine.
2. Build `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)`.
3. Pass `quantization_config=` and `device_map="auto"` to `from_pretrained`.
4. Call `prepare_model_for_kbit_training(model)` before `get_peft_model`.
5. Change `target_modules` to the model's layer names; optionally use a paged 8-bit optimizer.

## Step 07
- Without the cache the cost per new token grows with the sequence length, so the ratio grows with `max_new_tokens`.
- bf16 on CPU is often not faster; GPUs have bf16/fp16 hardware.

## Step 08
- The KV cache is stored in fp16 in our table while only the weights are quantised; quantise the cache separately (int8/fp8 KV cache) to shrink it.
- Preset example for a model: read `num_hidden_layers`, `num_key_value_heads`, `hidden_size / num_attention_heads` from its `config.json`.

## Step 09
- Throughput usually flattens when the hardware is compute-bound (large batch); repeated runs differ by tens of percent on a laptop, so average several runs.

## Integration challenge
Merged models are plain transformers models, so no adapter matmuls remain; the speed-up is small for small ranks but nonzero, and deployment is simpler (no peft dependency).
