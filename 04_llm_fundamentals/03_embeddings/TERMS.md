# Terms — Embeddings

Folder: `04_llm_fundamentals/03_embeddings` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [04 LLM fundamentals README](../README.md)

**Terms in this folder:** [torch (PyTorch)](#torch-pytorch) · [torch.nn as nn](#torchnn-as-nn) · [AutoModelForCausalLM](#automodelforcausallm) · [torch.manual_seed](#torchmanual_seed) · [Embedding](#embedding) · [nn.Embedding](#nnembedding) · [.weight](#weight) · [torch.tensor](#torchtensor) · [Calling a layer like a function](#calling-a-layer-like-a-function) · [Tensor indexing with indices](#tensor-indexing-with-indices) · [torch.equal](#torchequal) · [get_input_embeddings](#get_input_embeddings) · [wte (token embedding)](#wte-token-embedding) · [Function definition with assert](#function-definition-with-assert) · [.detach()](#detach) · [Cosine similarity](#cosine-similarity) · [.item()](#item) · [Tuple unpacking in a for loop](#tuple-unpacking-in-a-for-loop) · [Position embedding (wpe)](#position-embedding-wpe) · [torch.arange](#torcharange) · [Broadcasting](#broadcasting) · [Context-free embedding](#context-free-embedding) · [IndexError (out of range in self)](#indexerror-out-of-range-in-self)

---

## torch (PyTorch)  `(third-party: torch)`
**1. Definition:** A Python deep-learning library providing tensors (n-dimensional arrays that can run on CPU/GPU), automatic differentiation and neural-network building blocks.
**2. In plain words:** It is numpy plus the ability to compute gradients automatically, which is what training needs. Almost all Hugging Face models are PyTorch models under the hood. Use it whenever you work with real models.
**3. Example to quote:** > "I used PyTorch tensors to hold the embedding table and do the lookups."
```python
import torch
```
**Seen in:** `main.py`, import.

## torch.nn as nn  `(syntax)`
**1. Definition:** `import torch.nn as nn` imports PyTorch's neural-network module and gives it the short alias `nn`.
**2. In plain words:** `nn` holds layers such as `Linear` and `Embedding`. The `as nn` part is just a nickname so you type `nn.Embedding` instead of `torch.nn.Embedding`.
**3. Example to quote:** > "`import torch.nn as nn` lets me write `nn.Embedding(5, 3)` for a small lookup layer."

**Seen in:** `main.py`, imports.

## AutoModelForCausalLM  `(third-party: transformers)`
**1. Definition:** A Hugging Face class that loads the right language model (with a next-token prediction head) for a given model name.
**2. In plain words:** "Causal" means the model predicts the next token from the previous ones, like GPT. Here we only use it to reach the embedding tables inside distilgpt2.
**3. Example to quote:** > "`AutoModelForCausalLM.from_pretrained('distilgpt2')` gave me the full model, from which I took the embedding layers."

**Seen in:** `main.py`, section 2.

## torch.manual_seed  `(third-party: torch)`
**1. Definition:** Sets the seed of PyTorch's random number generator so random initialisation is reproducible.
**2. In plain words:** `nn.Embedding` starts with random numbers. Fixing the seed makes them (and your printed output) identical on every run.
**3. Example to quote:** > "I called `torch.manual_seed(0)` so the toy table has the same values each time I run it."

**Seen in:** `main.py`, section 1.

## Embedding  `(concept)`
**1. Definition:** A learned dense vector (list of floating-point numbers) assigned to each token ID, placed so that tokens with similar usage have similar vectors.
**2. In plain words:** An embedding turns an ID, which is just a label, into a point in space where distance means something. Token 995 simply means "take row 995 of the table". Training moves the rows so related words end up close, such as Monday and Tuesday.
**3. Example to quote:** > "distilgpt2's embedding table is 50257 x 768, so each token becomes a 768-number vector."

**Seen in:** `main.py`, docstring.

## nn.Embedding  `(third-party: torch)`
**1. Definition:** `nn.Embedding(num_embeddings, embedding_dim)` is a PyTorch layer holding a trainable table with one row per ID; calling it with IDs returns the matching rows.
**2. In plain words:** It is a lookup table you can train. The first argument is how many different IDs there are (vocabulary size), the second is the vector length. An ID outside the range causes an IndexError.
**3. Example to quote:** > "I built a toy `nn.Embedding(5, 3)`: five token IDs, three numbers each."
```python
toy = nn.Embedding(num_embeddings=5, embedding_dim=3)
```
**Seen in:** `main.py`, section 1.

## .weight  `(third-party: torch)`
**1. Definition:** The attribute of a layer that stores its learnable parameters as a tensor (`Parameter`); for an embedding it is the whole table.
**2. In plain words:** If a layer is a machine, `.weight` is the dial settings that training adjusts. For `Embedding`, `weight.shape` is `(vocab, dim)`.
**3. Example to quote:** > "`toy.weight.shape` printed `torch.Size([5, 3])`, which is the table itself."

**Seen in:** `main.py`, sections 1 and 2.

## torch.tensor  `(third-party: torch)`
**1. Definition:** Creates a tensor from Python data such as lists of numbers.
**2. In plain words:** It converts a list into the data type PyTorch computes with. IDs must be integer tensors to index an embedding.
**3. Example to quote:** > "`torch.tensor([4, 0, 4])` is three token IDs; ID 4 appears twice."

**Seen in:** `main.py`, section 1.

## Calling a layer like a function  `(syntax)`
**1. Definition:** PyTorch modules define `__call__`, so `layer(x)` runs the module's `forward` method on `x`.
**2. In plain words:** Objects can behave like functions. `toy(ids)` looks up the rows. Always call `layer(x)` rather than `layer.forward(x)` so hooks and bookkeeping work.
**3. Example to quote:** > "`toy(ids)` returned rows 4, 0 and 4 of the table."

**Seen in:** `main.py`, section 1.

## Tensor indexing with indices  `(syntax)`
**1. Definition:** Indexing a tensor with another integer tensor (`t[idx]`) gathers the rows at those positions.
**2. In plain words:** It is the same operation as the embedding layer performs. That is why `toy(ids)` equals `toy.weight[ids]`.
**3. Example to quote:** > "I verified `torch.equal(toy(ids), toy.weight[ids])` is True: an embedding is just row selection."
```python
t = torch.arange(6).reshape(3, 2)
t[torch.tensor([0, 2])]  # rows 0 and 2
```
**Seen in:** `main.py`, sections 1 and 2.

## torch.equal  `(third-party: torch)`
**1. Definition:** Returns True if two tensors have the same shape and identical elements.
**2. In plain words:** A strict comparison, useful for checking that two routes give exactly the same numbers. For floating-point results with tiny differences you would use `torch.allclose`.
**3. Example to quote:** > "`torch.equal(wte(same_word)[0, 0], wte(same_word)[0, 1])` was True: the same token gets the same raw vector."

**Seen in:** `main.py`, sections 1 and 4.

## get_input_embeddings  `(third-party: transformers)`
**1. Definition:** Model method that returns the token-embedding layer (an `nn.Embedding`) of any Hugging Face model.
**2. In plain words:** Different models name this layer differently; this method finds it for you. Its weight is the vocabulary table.
**3. Example to quote:** > "`model.get_input_embeddings().weight.shape` is `(50257, 768)`."

**Seen in:** `main.py`, section 2.

## wte (token embedding)  `(concept)`
**1. Definition:** In GPT-2 "word token embedding": the table mapping each token ID to its vector (`model.transformer.wte`).
**2. In plain words:** It is the first thing every token passes through. The code names the object returned by `get_input_embeddings()` as `wte`.
**3. Example to quote:** > "`wte(text_ids)` gives a `(1, 3, 768)` tensor for a three-token sentence."

**Seen in:** `main.py`, section 2.

## Function definition with assert  `(syntax)`
**1. Definition:** `def vec(word): ...` defines a reusable function; the `assert` inside checks an assumption and raises `AssertionError` with a message if it fails.
**2. In plain words:** The function only works for words that are exactly one token. The assert turns a silent wrong answer into a readable error message you wrote for yourself.
**3. Example to quote:** > "`vec('unbelievably')` fails with my own message saying it is four tokens, which is better than a confusing downstream error."

**Seen in:** `main.py`, `def vec`.

## .detach()  `(third-party: torch)`
**1. Definition:** Returns a tensor that shares data with the original but is cut off from the autograd graph, so gradients are not tracked.
**2. In plain words:** The embedding weights are trainable parameters. For plain comparison maths we do not want PyTorch recording operations, so we detach.
**3. Example to quote:** > "I used `.detach()` so computing cosine similarity does not build a gradient graph."

**Seen in:** `main.py`, `vec`.

## Cosine similarity  `(concept)`
**1. Definition:** The cosine of the angle between two vectors: `(a·b)/(|a||b|)`; 1 means same direction, 0 unrelated (orthogonal), -1 opposite.
**2. In plain words:** It compares which way two vectors point, ignoring length. If two words are used in similar contexts their vectors point similarly, giving a high score. It is the standard way to compare embeddings, for example in search.
**3. Example to quote:** > "In distilgpt2, `Monday` and `Tuesday` scored 0.893 while `cat` and `economics` scored 0.203."
```python
torch.nn.functional.cosine_similarity(a, b, dim=0)
```
**Seen in:** `main.py`, `cosine`.

## .item()  `(third-party: torch)`
**1. Definition:** Converts a one-element tensor to a plain Python number.
**2. In plain words:** Tensor results are objects; `.item()` gives you a normal float you can print or format with `:.3f`.
**3. Example to quote:** > "I call `.item()` so the similarity prints as 0.567 instead of `tensor(0.5670)`."

**Seen in:** `main.py`, `cosine`.

## Tuple unpacking in a for loop  `(syntax)`
**1. Definition:** `for w1, w2 in pairs:` assigns each element of every tuple in the list to separate names.
**2. In plain words:** Instead of indexing `pair[0]` and `pair[1]`, you name the parts directly. The code loops through word pairs this way.
**3. Example to quote:** > "I loop over `[('cat','dog'), ('king','queen')]` and get `w1` and `w2` for each pair."
```python
for a, b in [(1, 2), (3, 4)]:
    print(a + b)
```
**Seen in:** `main.py`, section 3.

## Position embedding (wpe)  `(concept)`
**1. Definition:** A table of vectors, one per position index (0 to 1023 in GPT-2), added to token vectors to tell the model where each token is.
**2. In plain words:** Attention itself ignores order, so "dog bites man" and "man bites dog" would look identical. Adding a position vector fixes that. The same word at two positions gets different final inputs. The cost: GPT-2 cannot go beyond 1024 positions.
**3. Example to quote:** > "Two identical ' the' tokens had equal raw embeddings, but after adding position embeddings they were different."
```python
x = wte(text_ids) + wpe(torch.arange(text_ids.shape[1]))
```
**Seen in:** `main.py`, section 4.

## torch.arange  `(third-party: torch)`
**1. Definition:** Creates a 1D tensor of consecutive integers `[0, 1, ..., n-1]`.
**2. In plain words:** Used here to produce the position numbers for each token in the sentence.
**3. Example to quote:** > "`torch.arange(3)` gives `tensor([0, 1, 2])`, the positions of 'The cat sat'."

**Seen in:** `main.py`, section 4.

## Broadcasting  `(concept)`
**1. Definition:** The rule by which arrays of different but compatible shapes are expanded automatically for element-wise operations.
**2. In plain words:** Adding `(1, 3, 768)` token vectors to `(3, 768)` position vectors works because the position table is repeated across the batch dimension. Shape mismatches that cannot be broadcast raise an error.
**3. Example to quote:** > "Adding token and position embeddings relied on broadcasting a `(3,768)` tensor over the batch of one."

**Seen in:** `main.py`, section 4 comment.

## Context-free embedding  `(concept)`
**1. Definition:** An input embedding that depends only on the token ID, not on surrounding words.
**2. In plain words:** "Bank" has the same input vector in "river bank" and "bank account". Context sensitivity comes from the transformer layers after the embedding.
**3. Example to quote:** > "The embedding layer gives 'bank' one vector, and the attention layers make it contextual."

**Seen in:** `WALKTHROUGH.md`, "Say it aloud".

## IndexError (out of range in self)  `(concept)`
**1. Definition:** The error raised when an ID passed to an embedding layer is at least as large as the number of rows in the table.
**2. In plain words:** A 5-row table only has IDs 0 to 4. The same family of bug appears when a prompt is longer than the 1024-position table.
**3. Example to quote:** > "`toy(torch.tensor([7]))` raised `IndexError: index out of range in self` because the table has only five rows."

**Seen in:** `main.py`, BREAK IT comment.
