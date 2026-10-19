# Step 06 — QLoRA (concept + reference code, not executed)

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

## Say it aloud
"QLoRA is LoRA on top of a base model stored in 4-bit NF4. Only the small 16-bit adapters get gradients, so memory drops from about 14 GB to about 4 GB of weights for a 7B model and it fits on a single consumer GPU. The price is a bit of quantisation error and slower steps, and it needs CUDA."
