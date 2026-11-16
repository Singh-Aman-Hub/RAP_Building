"""
SOLUTION to exercises (step 02): compare three learning rates on the same data, same seed, 3 epochs each.
RUN: .venv/bin/python 06_llm_training_and_fast_inference/solutions/lr_sweep.py   (~1 min)
"""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import get_device, make_blocks, read_lines

device = get_device()
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
blocks = make_blocks(tokenizer, read_lines(), 48)

for lr in (5e-7, 5e-5, 5e-4):                       # three orders of magnitude
    torch.manual_seed(0)                             # same shuffling + dropout for a fair comparison
    model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device).train()   # FRESH model each time
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(blocks), batch_size=4, shuffle=True)
    history = []
    for epoch in range(3):
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            loss = model(input_ids=batch, labels=batch).loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        history.append(round(total / len(loader), 2))
    print(f"lr={lr:g}  mean loss per epoch: {history}")
