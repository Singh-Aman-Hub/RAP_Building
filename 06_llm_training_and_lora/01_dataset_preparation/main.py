"""
GOAL:  Turn a raw text file into batches of token IDs + labels that a causal language model can train on.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 06_llm_training_and_fast_inference/01_dataset_preparation/main.py
EXPECTED OUTPUT (shortened):
    lines in file: 39
    first line -> token ids: [51, 15813, 25, ...]
    blocks tensor shape: torch.Size([N, 48])
    labels are the same IDs; the model shifts them internally by one position
    batch input_ids shape: torch.Size([4, 48])
NEEDS: torch, transformers.  First run downloads the distilgpt2 tokenizer (~1 MB).
"""

from pathlib import Path          # built-in: object-oriented file paths

import torch                      # 3rd-party: tensors (n-dimensional arrays) + training tools
from torch.utils.data import DataLoader, TensorDataset  # DataLoader = hands out mini-batches
from transformers import AutoTokenizer                  # loads the right tokenizer for a model name

# Path(__file__) = this file; .resolve() makes it absolute; .parents[1] = the module folder.
CORPUS = Path(__file__).resolve().parents[1] / "data" / "tiny_corpus.txt"
BLOCK_SIZE = 48                   # how many tokens the model sees at once during training
BATCH_SIZE = 4                    # how many blocks are processed together

# 1) READ RAW TEXT ----------------------------------------------------------
# list comprehension: build a list by looping; `if line.strip()` drops empty lines
lines = [line.strip() for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
print("lines in file:", len(lines))

# 2) TOKENIZE ---------------------------------------------------------------
# Same tokenizer family you met in module 04 (04_llm_fundamentals/01_tokenizer).
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
ids_first = tokenizer.encode(lines[0])        # text -> list of ints
print("first line -> token ids:", ids_first[:10], "...")
print("decoded back:", tokenizer.decode(ids_first[:10]))

# 3) GLUE + CHUNK -----------------------------------------------------------
# Neural nets want rectangular tensors.  Trick used in real LLM pre-training:
# join every document with an end-of-text token (EOS), then cut into equal blocks.
all_ids = []
for line in lines:
    all_ids.extend(tokenizer.encode(line) + [tokenizer.eos_token_id])
n_blocks = len(all_ids) // BLOCK_SIZE                       # // is integer division
blocks = torch.tensor(all_ids[: n_blocks * BLOCK_SIZE]).view(n_blocks, BLOCK_SIZE)   # shape: (n_blocks, BLOCK_SIZE)
print("total tokens:", len(all_ids), "| blocks tensor shape:", blocks.shape)

# 4) LABELS -----------------------------------------------------------------
# A causal LM learns "predict the NEXT token".  So input[t] should predict input[t+1].
# Hugging Face does the shift inside the model when you pass labels=input_ids,
# so labels are just a copy.  We show the shift by hand so it is not magic:
x = blocks[0]                                   # shape: (BLOCK_SIZE,)
inputs_view, targets_view = x[:-1], x[1:]       # shape: (BLOCK_SIZE - 1,) each; slices: all-but-last, all-but-first
print("\nposition | model sees | must predict")
for i in range(5):
    print(f"   {i}     | {tokenizer.decode(inputs_view[i])!r:12} | {tokenizer.decode(targets_view[i])!r}")
labels = blocks.clone()                          # shape: (n_blocks, BLOCK_SIZE); -100 would mean "ignore this position in the loss"
print("labels are the same IDs; the model shifts them internally by one position")

# 5) PADDING + ATTENTION MASK (needed when sequences have different lengths) --
# GPT-2 has no pad token, so we reuse EOS as padding and tell the model which
# positions are real (1) and which are padding (0) via attention_mask.
tokenizer.pad_token = tokenizer.eos_token
padded = tokenizer(lines[:3], padding=True, return_tensors="pt")  # input_ids, attention_mask: shape (3, longest_len); pt = PyTorch tensors
print("\npadded input_ids shape:", padded["input_ids"].shape)
shortest = padded["attention_mask"].sum(dim=1).argmin()    # index of the row with most padding
print("attention_mask of shortest line:", padded["attention_mask"][shortest].tolist())
# For padded batches you set labels to -100 where mask == 0 so padding doesn't count in the loss:
pad_labels = padded["input_ids"].masked_fill(padded["attention_mask"] == 0, -100)   # shape: (3, longest_len)
print("its labels (tail is -100 = ignored):", pad_labels[shortest].tolist()[-6:])

# 6) DATALOADER -------------------------------------------------------------
dataset = TensorDataset(blocks, labels)                       # pairs (input, label) by index
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)  # shuffle: new order every epoch
batch_inputs, batch_labels = next(iter(loader))               # shape: (BATCH_SIZE, BLOCK_SIZE) each; iter()+next() = "give me the first batch"
print("\nbatch input_ids shape:", batch_inputs.shape)          # (BATCH_SIZE, BLOCK_SIZE)

# TRY IT: set BLOCK_SIZE = 16 and watch the number of blocks grow.
# TRY IT: set shuffle=False and run twice - is the first batch identical now?
# BREAK IT: change .view(n_blocks, BLOCK_SIZE) to .view(n_blocks, BLOCK_SIZE + 1) and read the RuntimeError.
