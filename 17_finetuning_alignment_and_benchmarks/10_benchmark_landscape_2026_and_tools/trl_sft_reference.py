"""
READ, NOT RUN.  NEEDS: pip install trl peft datasets transformers   (TRL v1.x; PyPI showed trl 1.17.1 on 2026-10-30, Python >= 3.11)
A GPU is strongly recommended for anything bigger than a toy.  Argument names below were checked against the TRL v1.17.1 docs
(huggingface.co/docs/trl/sft_trainer, Nov 2026).  Re-check them for your installed version: defaults change between releases.

GOAL: the production version of step 02 (and the LoRA of step 03): SFT a small chat model with completion/assistant-only loss.
"""
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# prompt/completion in conversational form -> by default loss is computed on the completion only (completion_only_loss=None -> True)
train = Dataset.from_list([
    {"prompt":     [{"role": "user", "content": "Review: The pizza was awful.\nSentiment?"}],
     "completion": [{"role": "assistant", "content": "negative"}]},
    # ... a few thousand more rows
])

args = SFTConfig(
    output_dir="out-sft",
    learning_rate=2e-4,                  # LoRA-style learning rate; the library default is 2e-5 (full fine-tune scale)
    num_train_epochs=3,
    per_device_train_batch_size=8,
    max_length=1024,                     # was max_seq_length in older TRL; name checked in the v1.17.1 SFTConfig signature
    packing=False,                       # True packs several examples per sequence; packing_strategy="bfd" is the default strategy
    # completion_only_loss=True,         # default for prompt/completion data
    # assistant_only_loss=True,          # for `messages` data; needs a chat template with {% generation %} markers (TRL patches known ones, e.g. Qwen3)
    # eos_token="<|im_end|>",            # align EOS with the chat template for base models that ship their own template (e.g. Qwen2.5)
    # chat_template_path="HuggingFaceTB/SmolLM3-3B",   # give a base model a chat template
    gradient_checkpointing=True,         # default True in SFTConfig
    report_to="none",
)
trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",           # a string is loaded with from_pretrained (dtype defaults to float32 unless model_init_kwargs sets it)
    args=args,
    train_dataset=train,
    peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"),   # omit for full fine-tuning
)
trainer.train()
trainer.save_model("out-sft/final")
