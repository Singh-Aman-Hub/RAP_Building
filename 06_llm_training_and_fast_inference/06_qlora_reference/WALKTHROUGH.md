# Step 06 — QLoRA (concept + reference code, not executed)

## The story so far
[Step 05](../05_lora_peft_finetune/WALKTHROUGH.md) cut the trainable parameters to a sliver, but the frozen base model is still stored at full precision.

## The pain point
A 7B model in float16 is 14 GB of frozen weights. Add activations and the adapter's optimizer state and a 16 GB GPU is out of memory before the first step, even though almost nothing is being trained.

## The spark
Apply chapter 15 inside chapter 16: keep the frozen base in 4-bit (NF4) and train a LoRA adapter in higher precision on top. The base is only read, so quantization error does not accumulate in the weights you train. This is QLoRA.

## Goal
Understand how QLoRA lets you fine-tune a 7B model on one 16 GB GPU. The code in `qlora_reference.py` needs an NVIDIA GPU and `bitsandbytes`, so **it is not run on this Mac**. Compile-check only: `python -m py_compile qlora_reference.py`. To see it live, copy it into a free Colab GPU notebook.

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
The model is now tuned and, with QLoRA, tuned on one cheap GPU. But it still answers one user at a time, slowly, and the KV cache we met in the generation loop eats memory. Now we serve it: [step 07, fast inference with Hugging Face](../07_hf_inference_batching_kvcache/WALKTHROUGH.md).

> **Say it aloud (interview box):** QLoRA keeps the frozen base model in 4-bit NF4 and trains a LoRA adapter in higher precision on top of it. That drops the memory for a 7B fine-tune from over a hundred gigabytes to something a single 16 GB GPU can hold. The trade-off is slower steps from dequantizing weights and a small quality loss that I measure rather than assume.
