# Step 06 — QLoRA (concept + reference code, not executed)

## The story so far
[Step 05](../05_lora_peft_finetune/WALKTHROUGH.md) cut the trainable parameters to a sliver, but the frozen base model is still stored at full precision.

## The pain point
A 7B model in float16 is 14 GB of frozen weights. Add activations and the adapter's optimizer state and a 16 GB GPU is out of memory before the first step, even though almost nothing is being trained.

## The spark
Apply chapter 15 inside chapter 16: keep the frozen base in 4-bit (NF4) and train a LoRA adapter in higher precision on top. The base is only read, so quantization error does not accumulate in the weights you train. This is QLoRA.

## Goal
Understand how QLoRA lets you fine-tune a 7B model on one 16 GB GPU. The code in `qlora_reference.py` needs an NVIDIA GPU and `bitsandbytes`, so **it is not run on this Mac**. Compile-check only: `python -m py_compile qlora_reference.py`. To see it live, copy it into a free Colab GPU notebook.

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| QLoRA / LoRA / 4-bit fine-tuning | quantised LoRA | QLoRA = LoRA plus a 4-bit frozen base | Same adapters; only the storage of the frozen base weights changes. |
| NF4 / FP4 / int4 | 4-bit data types | Different | NF4 puts its 16 levels at normal-distribution quantiles; FP4 is a 4-bit float; int4 is evenly spaced. |
| Double quantisation / nested quantisation | `bnb_4bit_use_double_quant` | Yes | Quantises the per-block scale constants too, saving about 0.4 bit per parameter. |
| Compute dtype / storage dtype | `bnb_4bit_compute_dtype` | Different | Storage is 4-bit; matmuls run in bf16 after de-quantising a block. |
| bitsandbytes / bnb | the quantisation library | Yes | Provides the 4-bit/8-bit kernels and 8-bit optimizers behind `BitsAndBytesConfig`. |

## Code, step by step

### 1. Header: read-only warning and what changes vs LoRA  (`qlora_reference.py` lines 1–18)

```python
"""
REFERENCE ONLY -- NOT EXECUTED IN THIS STUDY PACK.

QLoRA = LoRA (step 05) on top of a base model whose frozen weights are stored in 4-bit.
It needs an NVIDIA GPU + the `bitsandbytes` library (Linux/Windows CUDA), so it will NOT run on a Mac.
Read it to understand the moving parts; you can run it in a free Google Colab GPU notebook.

    pip install torch transformers peft bitsandbytes accelerate datasets

What changes versus step 05 (everything else - dataset, loop, LoRA config - is identical):

    step 05 (LoRA)                          this file (QLoRA)
    ----------------------------------      -------------------------------------------------
    base weights in fp32/fp16, frozen       base weights in 4-bit NF4, frozen, de-quantised on the fly
    adapters in fp32                        adapters still in 16-bit/32-bit (these are the ONLY trained weights)
    7B model needs ~14 GB just to hold      7B model needs ~4 GB to hold weights  -> fits a 16 GB GPU
    weights in fp16
"""
```

- **In simple words:** A notice that this file is for reading, plus a table of how QLoRA differs from step 05.
- **Exactly:** Docstring only; says it needs an NVIDIA GPU and `bitsandbytes`, and that a 7B model needs about 4 GB for 4-bit weights vs about 14 GB in fp16. (bitsandbytes now also has CPU/XPU backends, but the CUDA path is the mature one.) Read, not run.
- **Why it is here:** Prevents you from running it on a Mac and getting confusing errors.

### 2. Imports and model name  (`qlora_reference.py` lines 20–24)

```python
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen2.5-0.5B"   # swap for a 7B model on a real GPU
```

- **In simple words:** Load the libraries and pick a small model to demo with.
- **Exactly:** `BitsAndBytesConfig` is in `transformers`; `prepare_model_for_kbit_training` is in `peft` (exists in peft 0.21.2, checked). `Qwen/Qwen2.5-0.5B` is the small stand-in for a 7B model.
- **Why it is here:** Without these imports there is no quantisation config and no preparation helper.

### 3. 4-bit quantisation settings  (`qlora_reference.py` lines 26–32)

```python
# 1) Tell transformers HOW to quantise while loading ------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                       # store frozen weights as 4-bit numbers
    bnb_4bit_quant_type="nf4",               # NormalFloat4: 16 levels placed to fit bell-shaped (normal) weight values
    bnb_4bit_use_double_quant=True,          # also quantise the per-block scaling constants (saves ~0.4 bit/param)
    bnb_4bit_compute_dtype=torch.bfloat16,   # the matmuls themselves are done in bf16 after de-quantising a block
)
```

- **In simple words:** Describe how to squeeze the weights into 4 bits when they are loaded.
- **Exactly:** `load_in_4bit=True`; `bnb_4bit_quant_type="nf4"`; `bnb_4bit_use_double_quant=True`; `bnb_4bit_compute_dtype=torch.bfloat16`. These four argument names match the current Transformers bitsandbytes docs (checked Nov 2026).
- **Why it is here:** Defaults would use fp4 and fp32 compute: more memory and less accurate.

### 4. Load the model quantised  (`qlora_reference.py` lines 34–39)

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,          # <- the only line that makes this "Q"
    device_map="auto",                       # let accelerate place layers on the available GPU(s)
)
```

- **In simple words:** Load the model and quantise as it loads.
- **Exactly:** `quantization_config=bnb_config` is the only change that makes this QLoRA; `device_map="auto"` lets accelerate place layers on the GPU.
- **Why it is here:** Loading in fp16 first would need the full 14 GB.

### 5. Prepare the quantised model for training  (`qlora_reference.py` lines 41–43)

```python
# 2) Prepare for training on a quantised model ----------------------------------------------------------
# casts layer norms to fp32, enables gradient checkpointing hooks, makes input embeddings require grad
model = prepare_model_for_kbit_training(model)
```

- **In simple words:** Apply a few fixes so training on a 4-bit model is stable.
- **Exactly:** `prepare_model_for_kbit_training` casts norm layers to fp32, turns on gradient checkpointing hooks and makes input embeddings require grad.
- **Why it is here:** Without it you can get zero gradients or NaN losses.

### 6. Attach LoRA adapters  (`qlora_reference.py` lines 45–52)

```python
# 3) Attach LoRA exactly like step 05 (Qwen/Llama names for the attention projections) ----------------
lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],   # for GPT-2 it was ["c_attn"]
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()
```

- **In simple words:** Add small trainable matrices to the attention layers.
- **Exactly:** `r=16`, `lora_alpha=32` (scale 2.0), `target_modules=["q_proj","k_proj","v_proj","o_proj"]` (Llama/Qwen names, vs `c_attn` in GPT-2), `print_trainable_parameters()` prints trainable vs total.
- **Why it is here:** Wrong module names raise `ValueError`; only adapters train.

### 7. Train, then save only the adapter  (`qlora_reference.py` lines 54–63)

```python
# 4) Train with the same loop as step 02, or with transformers.Trainer.
#    Recommended optimizer on GPU: "paged_adamw_8bit" (optimizer states are paged to CPU RAM if the GPU is full).
#
#      optimizer = bitsandbytes.optim.PagedAdamW8bit(trainable_params, lr=2e-4)
#
# Gradient flow: loss -> (de-quantised 4-bit weight, frozen, no grad stored) -> LoRA A/B (16-bit, updated).
# Only adapters receive gradients, so optimizer memory is tiny as well.

# 5) Save: again ONLY the adapter (few MB).  To deploy, load a 16-bit base, load adapter, merge_and_unload().
# model.save_pretrained("qlora_adapter")
```

- **In simple words:** Notes (as comments) on the optimizer, gradient flow and saving.
- **Exactly:** Recommended optimizer `paged_adamw_8bit` (in `Trainer`, `optim="paged_adamw_8bit"`) or `bitsandbytes.optim.PagedAdamW8bit`; gradients flow through de-quantised frozen weights to the 16-bit adapters; `save_pretrained("qlora_adapter")` writes a few MB.
- **Why it is here:** Explains why memory stays small and how to deploy (load a 16-bit base, load adapter, `merge_and_unload()`).

## The idea in one picture
```
LoRA (step 05)                       QLoRA
base weights fp16  (2 bytes)         base weights NF4 4-bit (0.5 bytes) --frozen
   + adapters A,B  (16-bit)             + adapters A,B  (16-bit) --trained
7B => ~14 GB weights                 7B => ~3.5-4 GB weights
```
Everything else (data, loop, LoraConfig) is the same.

## The three QLoRA ingredients
| Ingredient | Plain English |
|---|---|
| **NF4 (NormalFloat4)** | 4-bit number format whose 16 levels are placed where weights actually cluster (a bell curve), so less accuracy is lost than with evenly spaced int4 |
| **Double quantization** | the per-block scale factors are quantised too, saving ~0.4 bit/parameter |
| **Paged optimizers** | optimizer states spill to CPU RAM instead of crashing when the GPU momentarily runs out |
During the forward pass each 4-bit block is de-quantised to bf16 just-in-time, used for the matmul, then discarded.

## Syntax / API notes
| Line in the reference | Meaning |
|---|---|
| `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", ...)` | settings object read by `from_pretrained` |
| `from_pretrained(..., quantization_config=bnb_config, device_map="auto")` | load weights already quantised; `device_map` places layers on GPUs |
| `prepare_model_for_kbit_training(model)` | peft helper: fp32 norms, gradient checkpointing, input grads |
| `target_modules=["q_proj", ...]` | Llama/Qwen layer names (GPT-2 used `c_attn`) |

## Numbers you can reproduce on this Mac
```bash
python ../08_gpu_memory_calculator/main.py --skip-check
```
prints training memory for a 6.7B model: **full fine-tune 107.2 GB, LoRA 13.5 GB, QLoRA 3.5 GB** (weights + gradients + optimizer states; activations extra).

## Limitations / trade-offs
- Quantisation error: slightly lower quality than 16-bit LoRA (usually small).
- Slower per step than LoRA (de-quantisation work).
- CUDA-only in practice; not available on Apple Silicon.

## Experiments (read-only, or on a free Colab GPU)
- Change `r=8` to `r=64` in the LoRA config and predict the adapter size before running on a GPU: it grows linearly with r.
- Run `python 08_gpu_memory_calculator/main.py` and compare the full fine-tune and QLoRA training totals for the 7B preset.

## Debugging practice
Run the reference file on this Mac and read the error (confirmed): `ImportError: Using `bitsandbytes` 4-bit quantization requires bitsandbytes: `pip install -U bitsandbytes>=0.46.1``. `import bitsandbytes` itself fails with `ModuleNotFoundError: No module named 'bitsandbytes'`. The 4-bit kernels are CUDA-only, which is why this step is read-only here.

## The new problem (cliffhanger)
The model is now tuned and, with QLoRA, tuned on one cheap GPU. But it still answers one user at a time, slowly, and the KV cache we met in the generation loop eats memory. Now we serve it: [module 21 (inference engineering), starting with step 08, fast inference with Hugging Face](../../21_llm_inference_engineering/08_hf_inference_batching_kvcache/WALKTHROUGH.md).

> **Say it aloud (interview box):** QLoRA keeps the frozen base model in 4-bit NF4 and trains a LoRA adapter in higher precision on top of it. That drops the memory for a 7B fine-tune from over a hundred gigabytes to something a single 16 GB GPU can hold. The trade-off is slower steps from dequantizing weights and a small quality loss that I measure rather than assume.
