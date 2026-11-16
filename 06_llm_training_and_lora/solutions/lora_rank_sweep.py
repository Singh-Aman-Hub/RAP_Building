"""
SOLUTION to exercises (step 05): how do trainable parameters and final loss change with LoRA rank r?
RUN: .venv/bin/python 06_llm_training_and_fast_inference/solutions/lora_rank_sweep.py   (~1-2 min)
"""
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import get_device, make_blocks, read_lines

device = get_device()
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
blocks = make_blocks(tokenizer, read_lines(), 48)

for r in (1, 8, 32):
    torch.manual_seed(0)
    base = AutoModelForCausalLM.from_pretrained("distilgpt2")
    cfg = LoraConfig(r=r, lora_alpha=2 * r, target_modules=["c_attn"], fan_in_fan_out=True, task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg).to(device).train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    loader = DataLoader(TensorDataset(blocks), batch_size=4, shuffle=True)
    for epoch in range(10):
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            loss = model(input_ids=batch, labels=batch).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
    print(f"r={r:2d}  trainable={trainable:>9,}  loss after 10 epochs={total / len(loader):.3f}")
