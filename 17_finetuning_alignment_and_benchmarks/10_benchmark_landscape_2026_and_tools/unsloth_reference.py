"""
READ, NOT RUN.  NEEDS: pip install unsloth (NVIDIA GPU; bitsandbytes for 4-bit).  NOT verified against the Nov 2026 Unsloth docs:
this file is from the library's long-standing API shape. Check docs.unsloth.ai for current names before using.

GOAL: show how Unsloth wraps the QLoRA recipe: patched, faster kernels + 4-bit loading + LoRA + a TRL trainer.
"""
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-0.5B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,                     # QLoRA: base weights in 4-bit NF4
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16, lora_alpha=16, lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",  # memory-saving variant
    random_state=3407,
)
# then train with trl.SFTTrainer(model=model, processing_class=tokenizer, train_dataset=..., args=SFTConfig(...)) exactly as in trl_sft_reference.py
