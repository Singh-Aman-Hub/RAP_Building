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

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen2.5-0.5B"   # swap for a 7B model on a real GPU

# 1) Tell transformers HOW to quantise while loading ------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                       # store frozen weights as 4-bit numbers
    bnb_4bit_quant_type="nf4",               # NormalFloat4: 16 levels placed to fit bell-shaped (normal) weight values
    bnb_4bit_use_double_quant=True,          # also quantise the per-block scaling constants (saves ~0.4 bit/param)
    bnb_4bit_compute_dtype=torch.bfloat16,   # the matmuls themselves are done in bf16 after de-quantising a block
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,          # <- the only line that makes this "Q"
    device_map="auto",                       # let accelerate place layers on the available GPU(s)
)

# 2) Prepare for training on a quantised model ----------------------------------------------------------
# casts layer norms to fp32, enables gradient checkpointing hooks, makes input embeddings require grad
model = prepare_model_for_kbit_training(model)

# 3) Attach LoRA exactly like step 05 (Qwen/Llama names for the attention projections) ----------------
lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],   # for GPT-2 it was ["c_attn"]
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

# 4) Train with the same loop as step 02, or with transformers.Trainer.
#    Recommended optimizer on GPU: "paged_adamw_8bit" (optimizer states are paged to CPU RAM if the GPU is full).
#
#      optimizer = bitsandbytes.optim.PagedAdamW8bit(trainable_params, lr=2e-4)
#
# Gradient flow: loss -> (de-quantised 4-bit weight, frozen, no grad stored) -> LoRA A/B (16-bit, updated).
# Only adapters receive gradients, so optimizer memory is tiny as well.

# 5) Save: again ONLY the adapter (few MB).  To deploy, load a 16-bit base, load adapter, merge_and_unload().
# model.save_pretrained("qlora_adapter")
