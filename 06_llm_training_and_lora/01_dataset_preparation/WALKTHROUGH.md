# Step 01 — Dataset preparation

## The story so far
Chapter 16, The Surgical Fine-Tune. [Module 21, step 03 (memory calculator)](../../21_llm_inference_engineering/03_gpu_memory_calculator/WALKTHROUGH.md) showed that full fine-tuning is expensive. Whatever method we pick, the model still needs data in the form it understands.

## The pain point
A text file is not a training set. The model reads token ids in fixed-length tensors with labels, and padding must not count as real text. Most real-world training bugs live here: mis-shifted labels, padding inside the loss, sequences of different lengths that cannot be stacked.

## The spark
Tokenize, cut into fixed-size blocks, make labels (the same ids, shifted by the model), mark padding with -100 so the loss ignores it, and wrap everything in a DataLoader that serves batches.

## Goal
Turn a raw text file into tensors of token IDs (plus labels) that a causal language model can train on.
This is the "dataset prep and preprocessing" bullet of the syllabus, and it is where most real-world training bugs start.

## Run it
```bash
cd 06_llm_training_and_lora/01_dataset_preparation
python main.py
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Block / chunk / sequence | context window slice, training example, "packed sequence" | Close, not identical | Here a "block" is a fixed-length run of 48 token IDs cut from the glued corpus; "sequence length" is the model's maximum, "block size" is what we choose for training. |
| Labels / targets / ground truth | next-token targets, `labels=` | Yes | In causal LM the labels are the input IDs themselves (shifted by one inside the model). |
| Padding token / EOS token | `pad_token`, `eos_token`, endoftext token | Not the same job | EOS marks "text ends here"; pad fills short rows to equal length. GPT-2 has no pad token, so we reuse EOS and hide it with the attention mask. |
| Attention mask / loss mask | `attention_mask`, `-100` labels | Different | attention_mask tells attention which positions are real; `-100` in labels tells the loss to ignore a position. |
| DataLoader / Dataset | batcher / indexable store | Different | Dataset says "item i is X"; DataLoader groups items into batches and shuffles. |

## Code, step by step

### 1. Header docstring: what this script is for  (`main.py` lines 1–13)

```python
"""
GOAL:  Turn a raw text file into batches of token IDs + labels that a causal language model can train on.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 06_llm_training_and_lora/01_dataset_preparation/main.py
EXPECTED OUTPUT (shortened):
    lines in file: 39
    first line -> token ids: [51, 15813, 25, ...]
    blocks tensor shape: torch.Size([N, 48])
    labels are the same IDs; the model shifts them internally by one position
    batch input_ids shape: torch.Size([4, 48])
NEEDS: torch, transformers.  First run downloads the distilgpt2 tokenizer (~1 MB).
"""
```

- **In simple words:** Notes at the top that say what the file does and how to run it. Python ignores them when running.
- **Exactly:** A module docstring (a string literal as the first statement). It lists GOAL, RUN commands, EXPECTED OUTPUT and NEEDS. Real run: first line of output is `lines in file: 39`.
- **Why it is here:** Without it the next reader has to guess how to run the step and what a correct run looks like.

### 2. Imports  (`main.py` lines 15–19)

```python
from pathlib import Path          # built-in: object-oriented file paths

import torch                      # 3rd-party: tensors (n-dimensional arrays) + training tools
from torch.utils.data import DataLoader, TensorDataset  # DataLoader = hands out mini-batches
from transformers import AutoTokenizer                  # loads the right tokenizer for a model name
```

- **In simple words:** Bring in the tools we need: file paths, PyTorch, a batching helper and the tokenizer loader.
- **Exactly:** `Path` (pathlib) for file paths; `torch` for tensors; `DataLoader` hands out mini-batches and `TensorDataset` wraps tensors so item i is a tuple of the i-th rows; `AutoTokenizer` picks the right tokenizer class from a model name.
- **Why it is here:** Nothing runs without them; a missing import gives `NameError` at first use.

### 3. Locate corpus and set sizes  (`main.py` lines 21–24)

```python
# Path(__file__) = this file; .resolve() makes it absolute; .parents[1] = the module folder.
CORPUS = Path(__file__).resolve().parents[1] / "data" / "tiny_corpus.txt"
BLOCK_SIZE = 48                   # how many tokens the model sees at once during training
BATCH_SIZE = 4                    # how many blocks are processed together
```

- **In simple words:** Find the text file next to this module and decide how long each training chunk is.
- **Exactly:** `Path(__file__).resolve().parents[1]` goes from `.../01_dataset_preparation/main.py` up to the module folder, then `/ data / tiny_corpus.txt`. `BLOCK_SIZE = 48` tokens per training row, `BATCH_SIZE = 4` rows per step.
- **Why it is here:** Building the path from `__file__` makes the script work from any current directory.

### 4. Read the raw text into lines  (`main.py` lines 26–29)

```python
# 1) READ RAW TEXT ----------------------------------------------------------
# list comprehension: build a list by looping; `if line.strip()` drops empty lines
lines = [line.strip() for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
print("lines in file:", len(lines))
```

- **In simple words:** Open the file, split it into lines and throw away empty ones.
- **Exactly:** `read_text` returns one string, `splitlines()` returns a list of str, `strip()` removes spaces/newlines, and `if line.strip()` filters blanks. Real value: `lines in file: 39`.
- **Why it is here:** Empty lines would become empty "documents" containing only an EOS token and waste training signal.

### 5. Load the tokenizer and encode one line  (`main.py` lines 31–36)

```python
# 2) TOKENIZE ---------------------------------------------------------------
# Same tokenizer family you met in module 04 (04_llm_fundamentals/01_tokenizer).
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
ids_first = tokenizer.encode(lines[0])        # text -> list of ints
print("first line -> token ids:", ids_first[:10], "...")
print("decoded back:", tokenizer.decode(ids_first[:10]))
```

- **In simple words:** Turn the first sentence into numbers the model understands.
- **Exactly:** `AutoTokenizer.from_pretrained("distilgpt2")` loads the GPT-2 byte-pair-encoding (BPE) tokenizer, vocab size 50257. `encode` returns `list[int]`. Real: `[51, 9715, 25, 1374, 466, 314, 13259, 616, 9206, 287]`, which decodes to `Ticket: How do I reset my password in`.
- **Why it is here:** A neural net cannot read text, only integer IDs, and the IDs must come from the same tokenizer the model was trained with.

### 6. Glue all lines with EOS and cut into blocks  (`main.py` lines 38–46)

```python
# 3) GLUE + CHUNK -----------------------------------------------------------
# Neural nets want rectangular tensors.  Trick used in real LLM pre-training:
# join every document with an end-of-text token (EOS), then cut into equal blocks.
all_ids = []
for line in lines:
    all_ids.extend(tokenizer.encode(line) + [tokenizer.eos_token_id])
n_blocks = len(all_ids) // BLOCK_SIZE                       # // is integer division
blocks = torch.tensor(all_ids[: n_blocks * BLOCK_SIZE]).view(n_blocks, BLOCK_SIZE)   # shape: (n_blocks, BLOCK_SIZE)
print("total tokens:", len(all_ids), "| blocks tensor shape:", blocks.shape)
```

- **In simple words:** Stick every line together with an end-marker between them, then slice the long strip into equal pieces, like cutting a roll of ribbon into 48-token lengths.
- **Exactly:** Each line becomes `tokenizer.encode(line) + [eos_token_id]` (id 50256). `n_blocks = len(all_ids) // 48`; the tail that does not fill a block is dropped. `.view(n_blocks, 48)` reshapes the 1-D tensor without copying. Real: `total tokens: 1206 | blocks tensor shape: torch.Size([25, 48])` (25 x 48 = 1200, 6 tokens dropped).
- **Why it is here:** Neural nets need rectangular tensors; EOS stops the model learning that one ticket flows into the next one.

### 7. Show how next-token labels are shifted  (`main.py` lines 48–58)

```python
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
```

- **In simple words:** The model learns to guess the next token, so the input at position t has the token at t+1 as its answer.
- **Exactly:** `x = blocks[0]` has shape `(48,)`. `x[:-1]` (47 tokens) are inputs and `x[1:]` (47 tokens) are targets. Real table: `'T' -> 'icket'`, `'icket' -> ':'`, `':' -> ' How'`. `labels = blocks.clone()` copies the IDs because Hugging Face shifts inside the model; `-100` would mean ignore.
- **Why it is here:** Seeing the shift once removes the mystery of `labels=input_ids` used in steps 02 to 05.

### 8. Padding and attention mask for uneven lengths  (`main.py` lines 60–68)

```python
# 5) PADDING + ATTENTION MASK (needed when sequences have different lengths) --
# GPT-2 has no pad token, so we reuse EOS as padding and tell the model which
# positions are real (1) and which are padding (0) via attention_mask.
tokenizer.pad_token = tokenizer.eos_token
padded = tokenizer(lines[:3], padding=True, return_tensors="pt")  # input_ids, attention_mask: shape (3, longest_len); pt = PyTorch tensors
print("\npadded input_ids shape:", padded["input_ids"].shape)
shortest = padded["attention_mask"].sum(dim=1).argmin()    # index of the row with most padding
print("attention_mask of shortest line:", padded["attention_mask"][shortest].tolist())
# For padded batches you set labels to -100 where mask == 0 so padding doesn't count in the loss:
```

- **In simple words:** When sentences differ in length we fill the short ones with a dummy token and mark which positions are real.
- **Exactly:** GPT-2 has no pad token so `pad_token = eos_token`. `tokenizer(lines[:3], padding=True, return_tensors="pt")` gives `input_ids` of real shape `torch.Size([3, 33])` and an `attention_mask` of the same shape. The shortest row's mask is 29 ones then four zeros. `.sum(dim=1).argmin()` finds that row.
- **Why it is here:** Without a mask the model attends to padding as if it were text.

### 9. Mask padding out of the loss  (`main.py` lines 69–70)

```python
pad_labels = padded["input_ids"].masked_fill(padded["attention_mask"] == 0, -100)   # shape: (3, longest_len)
print("its labels (tail is -100 = ignored):", pad_labels[shortest].tolist()[-6:])
```

- **In simple words:** Tell the loss function to ignore the filler positions.
- **Exactly:** `masked_fill(mask == 0, -100)` replaces labels at padded positions with -100, the value `CrossEntropyLoss(ignore_index=-100)` skips. Real tail: `[6338, 13, -100, -100, -100, -100]`.
- **Why it is here:** Otherwise the model is rewarded for predicting padding, which skews the loss.

### 10. Wrap in Dataset and DataLoader, fetch one batch  (`main.py` lines 72–76)

```python
# 6) DATALOADER -------------------------------------------------------------
dataset = TensorDataset(blocks, labels)                       # pairs (input, label) by index
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)  # shuffle: new order every epoch
batch_inputs, batch_labels = next(iter(loader))               # shape: (BATCH_SIZE, BLOCK_SIZE) each; iter()+next() = "give me the first batch"
print("\nbatch input_ids shape:", batch_inputs.shape)          # (BATCH_SIZE, BLOCK_SIZE)
```

- **In simple words:** Set up a machine that hands us 4 shuffled blocks at a time.
- **Exactly:** `TensorDataset(blocks, labels)` yields `(blocks[i], labels[i])`. `DataLoader(..., batch_size=4, shuffle=True)` stacks 4 items and reshuffles each epoch. `next(iter(loader))` pulls the first batch. Real: `torch.Size([4, 48])` for both inputs and labels.
- **Why it is here:** Training on batches (not one row at a time) gives smoother gradients and uses the hardware better; shuffling avoids learning the order of the file.

### 11. TRY IT / BREAK IT comments  (`main.py` lines 78–80)

```python
# TRY IT: set BLOCK_SIZE = 16 and watch the number of blocks grow.
# TRY IT: set shuffle=False and run twice - is the first batch identical now?
# BREAK IT: change .view(n_blocks, BLOCK_SIZE) to .view(n_blocks, BLOCK_SIZE + 1) and read the RuntimeError.
```

- **In simple words:** Suggested experiments at the bottom: change a number and watch what happens, or break something on purpose.
- **Exactly:** They are comments, so Python ignores them. Example: `.view(n_blocks, BLOCK_SIZE + 1)` raises `RuntimeError: shape '[25, 49]' is invalid for input of size 1200` because 1200 values cannot fill 25 x 49.
- **Why it is here:** Changing one thing and reading the result (or the error) is the fastest way to learn what each line does.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `Path(__file__).resolve().parents[1]` | the folder one level above this file, as an absolute path | `Path("/a/b/c.py").parents[1]` -> `/a` |
| list comprehension `[f(x) for x in xs if cond]` | build a list in one line | `[s.strip() for s in ["a ", " "] if s.strip()]` -> `['a']` |
| `x[: n * k]` | slice: first `n*k` items | `[1,2,3,4,5][:4]` -> `[1,2,3,4]` |
| `x[:-1]`, `x[1:]` | all but last / all but first | `[1,2,3][1:]` -> `[2,3]` |
| `//` | integer (floor) division | `25 // 4` -> `6` |
| `for x in xs: ys.extend(...)` | `extend` appends every item of another list | `a=[1]; a.extend([2,3])` -> `[1,2,3]` |
| `iter(loader)` / `next(...)` | start an iterator / pull its first item | `next(iter([5,6]))` -> `5` |
| f-string `!r` | show the repr (quotes, `\n` visible) | `f"{'a\n'!r}"` -> `'a\\n'` |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `pathlib.Path` | built-in | file paths as objects; `.read_text()` reads a file | `encoding` | `str` |
| `AutoTokenizer.from_pretrained(name)` | transformers | downloads/loads the matching tokenizer | model name or folder | tokenizer object |
| `tokenizer.encode(text)` | transformers | text -> list of token IDs | `add_special_tokens` | `list[int]` |
| `tokenizer.decode(ids)` | transformers | IDs -> text | `skip_special_tokens` | `str` |
| `tokenizer(texts, padding=True, return_tensors="pt")` | transformers | batch encode with padding | `padding`, `truncation`, `return_tensors` | dict with `input_ids`, `attention_mask` |
| `torch.tensor(list)` | torch | list -> tensor | `dtype` | tensor |
| `tensor.view(a, b)` | torch | reshape without copying (sizes must multiply to the element count) | new shape | tensor |
| `tensor.masked_fill(mask, value)` | torch | replace entries where mask is True | mask, value | tensor |
| `TensorDataset(*tensors)` | torch | pairs tensors row by row | tensors | dataset |
| `DataLoader(ds, batch_size, shuffle)` | torch | serves mini-batches | `batch_size`, `shuffle` | iterable of batches |

## Execution trace (real run)
- **Input:** `data/tiny_corpus.txt`, 39 support lines.
- `tokenizer.encode(lines[0])` -> `[51, 9715, 25, 1374, 466, 314, 13259, 616, 9206, 287] ...` which decodes to `Ticket: How do I reset my password in`.
- Glue + chunk: 1206 tokens in total -> `1206 // 48 = 25` blocks -> tensor shape `[25, 48]` (the 6 leftover tokens are dropped).
- Label shift table: position 0 sees `'T'` and must predict `'icket'`; position 1 sees `'icket'` and must predict `':'`.
- Padding demo: 3 lines pad to length 33; the shortest line's mask ends `... 1, 1, 0, 0, 0, 0` and its labels end `[6338, 13, -100, -100, -100, -100]`.
- **Output:** one batch `input_ids` of shape `[4, 48]`.

## Experiments (change ONE thing)
1. `BLOCK_SIZE = 16` -> many more, shorter blocks (about 75). The model sees less context per example.
2. `shuffle=False` -> the first batch is identical on every run; with `True` it changes.
3. Print `blocks[0]` decoded: notice a block can start in the middle of a line. Is that a problem? (It is what real pre-training does.)

## Debugging practice
Change `.view(n_blocks, BLOCK_SIZE)` to `.view(n_blocks, BLOCK_SIZE + 1)`:
```
RuntimeError: shape '[25, 49]' is invalid for input of size 1200
```
Read it bottom-up: the last line names the error and the numbers. `1200 != 25*49`, so the shape and data disagree. Fix the shape, not the data.

## Rebuild checklist
- [ ] read file, split lines, drop blanks
- [ ] tokenize each line and append EOS
- [ ] cut to a multiple of `BLOCK_SIZE` and `.view(...)`
- [ ] wrap in `TensorDataset` + `DataLoader`

## The new problem (cliffhanger)
The data is ready: batches of ids and labels. Now we need something that learns from them. Next: [step 02, the training loop](../02_training_loop/WALKTHROUGH.md).

> **Say it aloud (interview box):** Training data for a causal language model is token ids cut into fixed-length blocks, with labels equal to the same ids shifted by one position. Padding positions get the label minus 100 so the loss ignores them, and a DataLoader serves shuffled batches. Most training bugs I have seen start in this step, so I print a batch and its labels before training.
