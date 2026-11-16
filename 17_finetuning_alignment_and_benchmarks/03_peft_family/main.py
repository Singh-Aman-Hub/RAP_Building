"""
GOAL:  Build every common PEFT (parameter-efficient fine-tuning) config on distilgpt2 with the installed `peft`,
       print the trainable-parameter count of each, check what the scaling factor of LoRA vs rsLoRA really is,
       and do the memory arithmetic: full fine-tune vs LoRA vs QLoRA for 0.5B / 7B / 70B models.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 17_finetuning_alignment_and_benchmarks/03_peft_family/main.py
EXPECTED OUTPUT (shortened):
    method table: LoRA r=8 147,456 trainable (0.18%), DoRA 161,280, IA3 18,432, prompt tuning 15,360 ...
    scaling: LoRA alpha/r = 2.0 vs rsLoRA alpha/sqrt(r) = 16.0
    memory table (GB): full FT 16 B/param vs LoRA vs QLoRA
NEEDS: torch, transformers, peft.  ~20 s.  (bitsandbytes / a GPU are NOT needed: QLoRA is reference-only here.)
"""
import torch
from peft import (IA3Config, LoraConfig, PrefixTuningConfig, PromptTuningConfig, TaskType, get_peft_model)
from transformers import AutoModelForCausalLM

torch.manual_seed(0)
MODEL = "distilgpt2"

# ## 1. Count parameters helper
def count(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

# ## 2. The menu of PEFT configs (all for GPT-2-style layer names)
CONFIGS = {
    "LoRA r=8 (c_attn)":        LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"], fan_in_fan_out=True, task_type=TaskType.CAUSAL_LM),
    "LoRA r=8 (attn+mlp)":      LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn", "c_proj", "c_fc"], fan_in_fan_out=True, task_type=TaskType.CAUSAL_LM),
    "LoRA r=64 (c_attn)":       LoraConfig(r=64, lora_alpha=128, target_modules=["c_attn"], fan_in_fan_out=True, task_type=TaskType.CAUSAL_LM),
    "rsLoRA r=64 (c_attn)":     LoraConfig(r=64, lora_alpha=128, use_rslora=True, target_modules=["c_attn"], fan_in_fan_out=True, task_type=TaskType.CAUSAL_LM),
    "DoRA r=8 (c_attn)":        LoraConfig(r=8, lora_alpha=16, use_dora=True, target_modules=["c_attn"], fan_in_fan_out=True, task_type=TaskType.CAUSAL_LM),
    "IA3 (c_attn, c_fc)":       IA3Config(target_modules=["c_attn", "c_fc"], feedforward_modules=["c_fc"], fan_in_fan_out=True, task_type=TaskType.CAUSAL_LM),
    "Prompt tuning (20 tok)":   PromptTuningConfig(num_virtual_tokens=20, task_type=TaskType.CAUSAL_LM),
    "Prefix tuning (20 tok)":   PrefixTuningConfig(num_virtual_tokens=20, task_type=TaskType.CAUSAL_LM),
}

# ## 3. Wrap a fresh model with each config and print the numbers
print(f"{'method':<26}{'trainable':>12}{'total':>14}{'trainable %':>13}")
peft_models = {}
for name, cfg in CONFIGS.items():
    base = AutoModelForCausalLM.from_pretrained(MODEL)
    pm = get_peft_model(base, cfg)                      # freezes the base, adds the small trainable part
    tr, tot = count(pm)
    peft_models[name] = pm
    print(f"{name:<26}{tr:>12,}{tot:>14,}{100 * tr / tot:>12.3f}%")
full = count(AutoModelForCausalLM.from_pretrained(MODEL))[0]
print(f"{'FULL fine-tune':<26}{full:>12,}{full:>14,}{100.0:>12.3f}%")

# ## 4. LoRA scaling: alpha/r versus rsLoRA alpha/sqrt(r)
for name in ("LoRA r=64 (c_attn)", "rsLoRA r=64 (c_attn)"):
    layer = peft_models[name].base_model.model.transformer.h[0].attn.c_attn
    print(f"{name:<26} scaling factor applied to B@A = {layer.scaling['default']:.3f}")

# ## 5. Which weights actually move? (names of the trainable tensors in one LoRA layer)
pm = peft_models["LoRA r=8 (c_attn)"]
print("\ntrainable tensors in layer 0:", [(n.split("h.0.")[1], tuple(p.shape)) for n, p in pm.named_parameters() if p.requires_grad and ".h.0." in n])
print("DoRA extra tensor:", [(n.split("h.0.")[1], tuple(p.shape)) for n, p in peft_models["DoRA r=8 (c_attn)"].named_parameters() if p.requires_grad and ".h.0." in n and "magnitude" in n])

# ## 6. Memory arithmetic: bytes per parameter during training (weights + grads + Adam states), activations ignored
def gb(x):
    return x / 1e9
def mem_full(n):                       # bf16 weights 2 + bf16 grads 2 + fp32 master 4 + Adam m,v 8  = 16 B/param
    return gb(16 * n)
def mem_lora(n, frac=0.01):            # frozen bf16 base (2 B) + trainable fraction at 16 B/param
    return gb(2 * n + 16 * frac * n)
def mem_qlora(n, frac=0.01):           # 4-bit base (0.5 B, ~0.56 with quant constants) + trainable fraction at 16 B/param
    return gb(0.56 * n + 16 * frac * n)
print(f"\n{'model':<8}{'full FT GB':>12}{'LoRA GB':>10}{'QLoRA GB':>10}   (1% trainable, activations NOT included)")
for label, n in (("0.5B", 0.5e9), ("7B", 7e9), ("70B", 70e9)):
    print(f"{label:<8}{mem_full(n):>12.1f}{mem_lora(n):>10.1f}{mem_qlora(n):>10.1f}")

# ## 7. Sanity: one forward/backward on a LoRA model touches only adapter weights
pm = peft_models["LoRA r=8 (c_attn)"]
x = torch.randint(0, 50257, (2, 16))                     # (batch, seq) random token ids
loss = pm(input_ids=x, labels=x).loss
loss.backward()
n_grad = sum(p.grad is not None for p in pm.parameters())
print(f"\nparameters that received a gradient: {n_grad} tensors (all of them LoRA A/B); loss={loss.item():.3f}")

# TRY IT: change r to 2 and 128 in the first config and watch the trainable count scale linearly with r.
# BREAK IT: remove fan_in_fan_out=True from a GPT-2 LoRA config; peft warns and swaps it. Read the warning.
