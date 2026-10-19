"""
GOAL:  Fine-tune distilgpt2 with LoRA using the `peft` library: freeze the base model, train tiny adapters,
       save only the adapter (a few hundred KB), reload it, and merge it for fast inference.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 06_llm_training_and_fast_inference/05_lora_peft_finetune/main.py
EXPECTED OUTPUT (shortened):
    trainable params: 147,456 || all params: 82,060,032 || trainable%: 0.1797
    epoch 1 loss 4.57 ... epoch 25 loss 1.40
    base      : Ticket: How do I turn on dark mode? Reply: ...generic...
    with LoRA : Ticket: How do I turn on dark mode? Reply: Open Settings, tap Settings, then choose Dark Mode.
    adapter folder size: 0.60 MB  vs full model ~330 MB
    reloaded adapter output identical to trained adapter: True
    merged model output identical: True
NEEDS: torch, transformers, peft.  ~30-60 s.
"""

import sys
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model   # peft = Parameter-Efficient Fine-Tuning library
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import get_device, make_blocks, read_lines

MODEL_NAME = "distilgpt2"
ADAPTER_DIR = Path(__file__).resolve().parent / "adapter_out"
EPOCHS, BATCH_SIZE, LR = 25, 4, 1e-3          # LoRA tolerates (and wants) a HIGHER lr than full fine-tuning

torch.manual_seed(0)
device = get_device()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)      # keep an un-adapted copy on CPU to compare later

# ---- 1. Describe the adapter ---------------------------------------------------
config = LoraConfig(
    r=8,                          # rank of the update matrices (bigger = more capacity, more params)
    lora_alpha=16,                # scaling: the update is multiplied by lora_alpha / r  (here 2.0)
    lora_dropout=0.05,            # dropout applied inside the adapter path; regularisation
    target_modules=["c_attn"],    # WHICH layers get adapters: GPT-2's fused query/key/value projection
    fan_in_fan_out=True,          # GPT-2 stores this layer transposed (Conv1D); peft must know
    task_type="CAUSAL_LM",
)
model = get_peft_model(base, config)  # wraps the model: freezes base weights, inserts trainable A and B
model.print_trainable_parameters()    # the headline number to quote in an interview
model.to(device)

# peek at what was inserted
for name, param in model.named_parameters():
    if "h.0" in name and "lora" in name:   # first adapter matrix of transformer block 0
        print("  trainable:", name, tuple(param.shape), "requires_grad =", param.requires_grad)
        break
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"  frozen parameters: {frozen:,}")

# ---- 2. Train (same loop as step 02, but the optimizer only sees trainable params) ----
blocks = make_blocks(tokenizer, read_lines(), 48)
loader = DataLoader(TensorDataset(blocks), batch_size=BATCH_SIZE, shuffle=True)
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)  # filter: trainable only

model.train()
for epoch in range(1, EPOCHS + 1):
    total = 0.0
    for (batch,) in loader:
        batch = batch.to(device)   # shape: (BATCH_SIZE, BLOCK_SIZE)
        loss = model(input_ids=batch, labels=batch).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    print(f"epoch {epoch} loss {total / len(loader):.3f}")


def generate(m, prompt: str) -> str:
    """Greedy-generate 30 tokens from model m."""
    m.eval()
    enc = tokenizer(prompt, return_tensors="pt").to(next(m.parameters()).device)
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)


PROMPT = "Ticket: How do I turn on dark mode? Reply:"
# `with model.disable_adapter():` temporarily switches LoRA off -> you get the original behaviour.
with model.disable_adapter():
    print("\nbase      :", generate(model, PROMPT))
trained_text = generate(model, PROMPT)
print("with LoRA :", trained_text)

# ---- 3. Save ONLY the adapter ----------------------------------------------------
model.save_pretrained(ADAPTER_DIR)      # writes adapter_config.json + adapter_model.safetensors
size_mb = sum(f.stat().st_size for f in ADAPTER_DIR.iterdir()) / 1e6
print(f"\nadapter folder size: {size_mb:.2f} MB  vs full model ~330 MB")

# ---- 4. Reload: fresh base model + saved adapter ------------------------------------
fresh_base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
reloaded = PeftModel.from_pretrained(fresh_base, ADAPTER_DIR).to(device)
print("reloaded adapter output identical to trained adapter:", generate(reloaded, PROMPT) == trained_text)

# ---- 5. Merge for deployment: W + (alpha/r) B A is folded into W; no peft needed at inference ----
merged = reloaded.merge_and_unload()    # returns a plain transformers model
print("merged model output identical:", generate(merged, PROMPT) == trained_text)
print("merged model type:", type(merged).__name__)

# TRY IT: set r=2, then r=32. Compare 'trainable%' and the final loss.
# TRY IT: set target_modules=["c_attn", "c_proj", "c_fc"] to adapt more layers; what happens to trainable params?
# TRY IT: set LR = 3e-6 (typical full-fine-tune size). LoRA barely learns - why?
# BREAK IT: remove fan_in_fan_out=True and read the warning peft prints about Conv1D.
# BREAK IT: set target_modules=["q_proj"] and read the ValueError: GPT-2 has no module with that name.
