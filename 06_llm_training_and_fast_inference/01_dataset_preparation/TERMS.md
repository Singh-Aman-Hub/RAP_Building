# Terms — Dataset Preparation

Folder: `06_llm_training_and_fast_inference/01_dataset_preparation` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [06 README](../README.md)

**Terms in this folder:** [pathlib.Path](#pathlibpath) · [`__file__` and `.resolve().parents[1]`](#__file__-and-resolveparents1) · [List comprehension](#list-comprehension) · [Dataset preparation (preprocessing)](#dataset-preparation-preprocessing) · [AutoTokenizer.from_pretrained](#autotokenizerfrom_pretrained) · [Token ID](#token-id) · [tokenizer.encode / tokenizer.decode](#tokenizerencode--tokenizerdecode) · [EOS token (end-of-text)](#eos-token-end-of-text) · [Integer division `//`](#integer-division-) · [Slicing](#slicing) · [torch.tensor](#torchtensor) · [Tensor.view](#tensorview) · [Block size (context length in training)](#block-size-context-length-in-training) · [Causal language model](#causal-language-model) · [Labels and the one-position shift](#labels-and-the-one-position-shift) · [Ignore index (-100)](#ignore-index--100) · [Padding and pad token](#padding-and-pad-token) · [Attention mask](#attention-mask) · [tokenizer(...) batch call with return_tensors](#tokenizer-batch-call-with-return_tensors) · [Tensor.masked_fill](#tensormasked_fill) · [TensorDataset](#tensordataset) · [DataLoader](#dataloader) · [Batch size](#batch-size) · [Shuffle](#shuffle) · [iter() and next()](#iter-and-next) · [f-string with `!r`](#f-string-with-r)

---

## pathlib.Path  `(standard library)`
**1. Definition:** `pathlib.Path` is a standard-library class that represents a file-system path as an object, with methods such as `.read_text()`, `.exists()` and the `/` operator for joining parts.
**2. In plain words:** Instead of gluing strings like `"folder" + "/" + "file.txt"`, you build a `Path` and ask it to do file things. It also works on Windows and Mac without you changing slashes. Here it is used to find `data/tiny_corpus.txt` no matter where you launch the script from.
**3. Example to quote:** > "I used `Path(__file__).resolve().parents[1] / 'data' / 'tiny_corpus.txt'` so the script finds its data file even if I run it from a different directory."
```python
from pathlib import Path
text = Path("notes.txt").read_text(encoding="utf-8")
```
**Seen in:** `main.py`, `CORPUS = Path(__file__)...`

## `__file__` and `.resolve().parents[1]`  `(syntax)`
**1. Definition:** `__file__` is a built-in module variable holding the path of the running script; `.resolve()` turns a path into an absolute one, and `.parents[1]` is the directory two levels up the path chain (index 0 is the immediate parent folder).
**2. In plain words:** The script asks "where am I on disk?" and then climbs up folders to reach the module folder. This makes the code independent of your current working directory, which is a common source of "file not found" errors.
**3. Example to quote:** > "`Path('/a/b/c.py').parents[1]` is `/a`: index 0 is `/a/b`, index 1 is `/a`. In this script that climbs from the step folder to the module folder that contains `data/`."
**Seen in:** `main.py`, top constants

## List comprehension  `(syntax)`
**1. Definition:** A list comprehension `[expr for x in iterable if condition]` is Python syntax that builds a new list by evaluating `expr` for each item that passes the optional `if` filter.
**2. In plain words:** It is a one-line version of "loop, test, append". Here it strips every line of the file and throws away the empty ones. It is compact, but if it gets long, a normal loop is easier to read.
**3. Example to quote:** > "I wrote `[line.strip() for line in text.splitlines() if line.strip()]` to get a clean list of the 39 non-empty support lines."
```python
[s.strip() for s in ["a ", " "] if s.strip()]   # ['a']
```
**Seen in:** `main.py`, section 1 (READ RAW TEXT)

## Dataset preparation (preprocessing)  `(concept)`
**1. Definition:** Dataset preparation is the process of converting raw data into the exact numeric tensors and format a model expects: reading, cleaning, tokenizing, chunking, labeling and batching.
**2. In plain words:** A neural network only understands rectangular grids of numbers, not text files. Preparation is the translation step, and mistakes here (wrong shapes, wrong labels, leaked data) cause many real training bugs that don't crash but quietly give bad models.
**3. Example to quote:** > "Before training I turn 39 raw support lines into a `[25, 48]` tensor of token IDs, then wrap it in a DataLoader so the model gets batches of 4 blocks."
**Seen in:** whole `main.py`

## AutoTokenizer.from_pretrained  `(third-party: transformers)`
**1. Definition:** `AutoTokenizer.from_pretrained(name)` downloads (or loads from cache) the tokenizer files that belong to the named model and returns the correct tokenizer object for it.
**2. In plain words:** Each model was trained with its own way of cutting text into pieces, so you must use the matching tokenizer. "Auto" means you don't need to know which tokenizer class it is; the library reads the model's config and picks it. The first call downloads a small file (about 1 MB for distilgpt2).
**3. Example to quote:** > "I call `AutoTokenizer.from_pretrained('distilgpt2')`; if I used a different model's tokenizer, the token IDs would map to the wrong words."
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("distilgpt2")
```
**Seen in:** `main.py`, section 2

## Token ID  `(concept)`
**1. Definition:** A token ID is the integer index of a token (a word piece, whole word or symbol) in the tokenizer's vocabulary.
**2. In plain words:** Models can't read letters, so each piece of text is replaced by a number that points into a lookup table. The same text always maps to the same IDs for a given tokenizer. The first line in this corpus starts with IDs like `[51, 9715, 25, ...]`, which decode back to `Ticket:`.
**3. Example to quote:** > "The first support line becomes `[51, 9715, 25, 1374, ...]`, and decoding those IDs gives back 'Ticket: How do I reset my password in'."
**Seen in:** `main.py`, section 2 print statements

## tokenizer.encode / tokenizer.decode  `(third-party: transformers)`
**1. Definition:** `tokenizer.encode(text)` converts a string to a list of integer token IDs; `tokenizer.decode(ids)` converts IDs back to a string.
**2. In plain words:** These are the two directions of the translation. `encode` is what goes into the model, `decode` is how you read what comes out. `decode` can drop special tokens if you pass `skip_special_tokens=True`.
**3. Example to quote:** > "I `encode` the first line, print the first ten IDs, and `decode` them to prove the round trip works."
```python
ids = tok.encode("Hello world")
tok.decode(ids)   # 'Hello world'
```
**Seen in:** `main.py`, sections 2, 3 and 4

## EOS token (end-of-text)  `(concept)`
**1. Definition:** The EOS (end-of-sequence) token is a special token, accessed as `tokenizer.eos_token_id`, that marks where one document or sequence ends.
**2. In plain words:** When many short documents are glued into one long stream, the model needs a signal that a new, unrelated document begins. EOS is that separator. It is also what the model emits when it wants to stop generating.
**3. Example to quote:** > "I append `tokenizer.eos_token_id` after every support line so the model learns that a ticket ends there, rather than blending tickets together."
**Seen in:** `main.py`, section 3 (GLUE + CHUNK)

## Integer division `//`  `(syntax)`
**1. Definition:** The `//` operator divides two numbers and rounds the result down to a whole number.
**2. In plain words:** Use it when you need a count of complete groups. Here `1206 // 48 = 25` complete blocks, and the left-over 6 tokens are thrown away.
**3. Example to quote:** > "`n_blocks = len(all_ids) // BLOCK_SIZE` gives me only complete blocks; `25 // 4` is `6`, not `6.25`."
**Seen in:** `main.py`, section 3

## Slicing  `(syntax)`
**1. Definition:** Slicing `seq[start:stop]` returns a part of a list, string or tensor from index `start` up to but not including `stop`; omitted ends mean "from the beginning" or "to the end", and negative indexes count from the end.
**2. In plain words:** `x[:-1]` is "everything except the last item" and `x[1:]` is "everything except the first". Placing those two side by side lines each token up with the one that follows it, which is exactly what next-token prediction needs.
**3. Example to quote:** > "`inputs_view, targets_view = x[:-1], x[1:]` shows that at position 0 the model sees 'T' and must predict 'icket'."
```python
[1, 2, 3][1:]    # [2, 3]
[1, 2, 3][:-1]   # [1, 2]
```
**Seen in:** `main.py`, sections 3 and 4

## torch.tensor  `(third-party: torch)`
**1. Definition:** `torch.tensor(data)` creates a PyTorch tensor, an n-dimensional array of numbers, from a Python list; it can live on CPU or GPU and track gradients.
**2. In plain words:** A tensor is PyTorch's version of a list of numbers that the math libraries can process fast. Lists of ints become integer tensors, which is what token IDs need.
**3. Example to quote:** > "I turn the flat list of 1200 token IDs into a tensor with `torch.tensor(...)` before reshaping it."
```python
import torch
torch.tensor([5, 6, 7]).shape   # torch.Size([3])
```
**Seen in:** `main.py`, section 3

## Tensor.view  `(third-party: torch)`
**1. Definition:** `tensor.view(a, b)` returns a tensor with the same data reshaped to `(a, b)`, without copying; the product of the new sizes must equal the number of elements.
**2. In plain words:** It re-slices one long row of numbers into rows of equal length. If the numbers don't divide evenly you get a `RuntimeError` such as "shape '[25, 49]' is invalid for input of size 1200". Fix the shape, not the data.
**3. Example to quote:** > "I cut 1200 IDs into 25 rows of 48 with `.view(n_blocks, BLOCK_SIZE)`; changing to 49 columns gives a RuntimeError because 25 × 49 is not 1200."
**Seen in:** `main.py`, section 3; BREAK IT hint

## Block size (context length in training)  `(concept)`
**1. Definition:** Block size is the fixed number of tokens in each training example that the model sees at once.
**2. In plain words:** The model is trained on equally long windows of text. A bigger block gives the model more context per example but uses more memory; a smaller one gives more, shorter examples. Here it is 48.
**3. Example to quote:** > "With `BLOCK_SIZE = 48`, my 1206 tokens became 25 blocks; with 16 I'd get about 75 shorter ones."
**Seen in:** `main.py`, constants and section 3

## Causal language model  `(concept)`
**1. Definition:** A causal language model predicts each next token using only the tokens before it (a left-to-right, autoregressive model), as GPT-2 does.
**2. In plain words:** It is trained by playing "guess the next piece" over and over on text, and can never peek at the future. That same skill is what produces text when you sample one token at a time.
**3. Example to quote:** > "distilgpt2 is a causal LM: given 'Ticket:' it learns to predict ' How', and during generation it repeats that to write whole sentences."
**Seen in:** `main.py`, section 4 comments

## Labels and the one-position shift  `(concept)`
**1. Definition:** For causal-LM training the labels are the same token IDs as the inputs, and the loss compares the model's output at position *t* with the token at position *t+1*; Hugging Face performs this shift inside the model when you pass `labels=input_ids`.
**2. In plain words:** The "right answer" for each position is simply the next token in the text. You don't prepare a separate label file, because the text is its own answer key. The script shows the shift by hand so it is not magic.
**3. Example to quote:** > "I pass `labels = blocks.clone()`; the model shifts internally, so position 1 sees 'icket' and is graded on predicting ':'."
**Seen in:** `main.py`, section 4

## Ignore index (-100)  `(concept)`
**1. Definition:** In PyTorch/Hugging Face cross-entropy loss, a label value of `-100` is the default "ignore index": positions with that label contribute nothing to the loss.
**2. In plain words:** It is a way to say "don't grade me here". Padding positions are not real text, so you mark their labels `-100` so the model isn't punished or rewarded for them.
**3. Example to quote:** > "For the padded batch I set the labels to -100 wherever the mask is 0, so the tail `[... 13, -100, -100, -100, -100]` is ignored."
**Seen in:** `main.py`, section 5

## Padding and pad token  `(concept)`
**1. Definition:** Padding adds filler tokens to shorter sequences so that all sequences in a batch have equal length; the filler is the tokenizer's `pad_token`.
**2. In plain words:** A tensor must be rectangular, so short lines get extra filler at the end. GPT-2 has no pad token of its own, so the code reuses the EOS token as padding (`tokenizer.pad_token = tokenizer.eos_token`). Padding wastes compute, which is a trade-off.
**3. Example to quote:** > "My three lines were different lengths, so `padding=True` padded them to 33 tokens and I set `pad_token` to EOS because GPT-2 doesn't define one."
**Seen in:** `main.py`, section 5

## Attention mask  `(concept)`
**1. Definition:** An attention mask is a tensor of 1s and 0s, the same shape as `input_ids`, telling the model which positions are real tokens (1) and which are padding to ignore (0).
**2. In plain words:** It is a "pay attention here, skip this" sign for each token. Without it, the model would treat filler tokens as real words.
**3. Example to quote:** > "The shortest line's attention mask ends in `1, 1, 0, 0, 0, 0`, meaning its last four positions are padding."
**Seen in:** `main.py`, section 5

## tokenizer(...) batch call with return_tensors  `(third-party: transformers)`
**1. Definition:** Calling the tokenizer object directly, `tokenizer(texts, padding=True, return_tensors="pt")`, encodes a list of strings in one go and returns a dictionary-like object with `input_ids` and `attention_mask`, as PyTorch tensors when `return_tensors="pt"`.
**2. In plain words:** This is the all-in-one convenience call: it tokenizes, pads, builds the mask and gives you tensors ready for the model. `"pt"` simply means PyTorch.
**3. Example to quote:** > "`tokenizer(lines[:3], padding=True, return_tensors='pt')` gave me `input_ids` of shape `[3, 33]` plus the attention mask."
**Seen in:** `main.py`, section 5

## Tensor.masked_fill  `(third-party: torch)`
**1. Definition:** `tensor.masked_fill(mask, value)` returns a copy of the tensor in which every position where `mask` is True is replaced by `value`.
**2. In plain words:** It overwrites selected entries. Here it overwrites the IDs of padding positions with `-100` to build the label tensor.
**3. Example to quote:** > "`input_ids.masked_fill(attention_mask == 0, -100)` turns padding labels into -100 in one line."
```python
torch.tensor([1, 2, 3]).masked_fill(torch.tensor([False, True, False]), -100)  # tensor([1, -100, 3])
```
**Seen in:** `main.py`, section 5

## TensorDataset  `(third-party: torch)`
**1. Definition:** `torch.utils.data.TensorDataset(*tensors)` wraps one or more tensors of equal first dimension into a dataset where item *i* is the tuple of the *i*-th rows.
**2. In plain words:** It pairs each input row with its label row so the DataLoader can pick them together by index.
**3. Example to quote:** > "`TensorDataset(blocks, labels)` makes 25 (input, label) pairs the DataLoader can hand out."
**Seen in:** `main.py`, section 6

## DataLoader  `(third-party: torch)`
**1. Definition:** `torch.utils.data.DataLoader(dataset, batch_size, shuffle)` is an iterable that serves the dataset in mini-batches, optionally in a new random order each epoch.
**2. In plain words:** The model learns from small groups of examples rather than all data at once. The DataLoader does the grouping and mixing so your training loop can just say `for batch in loader`.
**3. Example to quote:** > "I used `DataLoader(dataset, batch_size=4, shuffle=True)` and the first batch came out with `input_ids` shape `[4, 48]`."
**Seen in:** `main.py`, section 6

## Batch size  `(concept)`
**1. Definition:** Batch size is the number of examples processed together in one forward/backward pass before the weights are updated.
**2. In plain words:** Bigger batches give smoother gradient estimates and use hardware efficiently but need more memory. Here it is 4 blocks.
**3. Example to quote:** > "`BATCH_SIZE = 4` means each optimizer step in the next step sees 4 blocks of 48 tokens."
**Seen in:** `main.py`, constants

## Shuffle  `(concept)`
**1. Definition:** Shuffling randomises the order in which training examples are served, usually reshuffled every epoch.
**2. In plain words:** If the data were always in the same order, the model might learn order-related quirks. With `shuffle=False` the first batch is the same on every run; with `True` it changes.
**3. Example to quote:** > "When I set `shuffle=False` and ran twice, the first batch was identical both times; with `True` it differs."
**Seen in:** `main.py`, section 6; TRY IT

## iter() and next()  `(Python built-in)`
**1. Definition:** `iter(x)` returns an iterator over an iterable, and `next(it)` pulls the next item from that iterator.
**2. In plain words:** Together they mean "give me just the first item". It is a quick way to peek at one batch from a DataLoader without writing a loop.
**3. Example to quote:** > "`next(iter(loader))` gave me the first batch so I could print its shape."
```python
next(iter([5, 6]))   # 5
```
**Seen in:** `main.py`, section 6

## f-string with `!r`  `(syntax)`
**1. Definition:** An f-string `f"{value!r}"` inserts `repr(value)` into the text, which shows quotes and escape characters such as `\n`.
**2. In plain words:** The `!r` flag makes invisible things visible. When decoding single tokens, it shows a leading space as `' How'` instead of an invisible blank.
**3. Example to quote:** > "I printed `{tokenizer.decode(inputs_view[i])!r}` so I could see exactly where a token has a leading space."
**Seen in:** `main.py`, section 4 print loop
