# Step 03 - Embeddings (Giving Words Geometry)

## The story so far
[Step 02](../02_vectors_and_similarity/WALKTHROUGH.md) showed what a good vector looks like: short, dense, with related words close together, but it built them once by squeezing counts with SVD on a toy corpus. [Step 01](../01_tokenizer/WALKTHROUGH.md) gave us the ids (0 to 50,256) that must now become such vectors.

## The pain point
An id is just a label. We need a vector for each of 50,257 ids, and we want the network to *learn* them while it trains, so that whatever helps predict the next token pulls related tokens together. There is a second problem hiding here: once tokens are vectors, the transformer treats them as a set. Feed it `dog bites man` and `man bites dog` and it sees the same three vectors. It is amnesiac about order.

## The spark
Make the vectors a trainable table: row `i` is the vector for token id `i` (`nn.Embedding`). Lookup is just indexing, and gradient descent nudges the rows, so `Monday` and `Tuesday` drift together. For order, add a second table indexed by position and add the two vectors: `x = wte[token] + wpe[position]`. Now the same word at two positions has two different inputs.

## Goal
See that an embedding layer is a lookup table of learned vectors, compare vectors with cosine similarity, and see how position is added.

## Run it
```bash
.venv/bin/python 04_llm_fundamentals/03_embeddings/main.py
```

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `import torch.nn as nn` | import a submodule under a short alias | `nn.Linear(2, 3)` |
| `toy(ids)` | calling an object like a function (it defines `__call__`) | `layer(x)` runs its `forward` |
| `def vec(word):` with `assert` inside | function with a sanity check | - |
| `for w1, w2 in [("cat","dog"), ...]` | loop that unpacks each tuple into two names | - |
| `wte.weight[ids]` | tensor indexing with a tensor of indices (gathers rows) | `t[torch.tensor([0, 2])]` |
| `.item()` | one-element tensor -> normal Python number | `torch.tensor(3.).item()` |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `torch` | 3rd-party | tensors + autograd | - | - |
| `nn.Embedding(num_embeddings, embedding_dim)` | torch | trainable table: id -> vector | table rows (vocab) and columns (dim) | layer |
| `layer.weight` | torch | the table itself (`Parameter`) | - | tensor |
| `model.get_input_embeddings()` | transformers | returns the token-embedding layer of any model | - | `nn.Embedding` |
| `model.transformer.wpe` | transformers (GPT-2 specific) | position embedding table (1024 positions) | - | `nn.Embedding` |
| `F.cosine_similarity(a, b, dim=0)` | torch | cos of the angle between vectors | `dim` | tensor |
| `.detach()` | torch | drop gradient tracking for plain maths | - | tensor |
| `torch.arange(n)` | torch | `[0, 1, ..., n-1]` | n | tensor |

## Execution trace
```
toy table shape: torch.Size([5, 3])
lookup of ids [4, 0, 4]:
 tensor([[-0.8567,  1.1006, -1.0712],
        [ 1.5410, -0.2934, -2.1788],
        [-0.8567,  1.1006, -1.0712]], grad_fn=<EmbeddingBackward0>)
same as indexing the table directly: True

real table shape (vocab, dim): (50257, 768)
similarity('cat', 'dog') = 0.567
similarity('king', 'queen') = 0.664
similarity('cat', 'economics') = 0.203
similarity('Monday', 'Tuesday') = 0.893
similarity('car', 'banana') = 0.264

position table shape (max positions, dim): (1024, 768)
input to first block, shape (batch, seq, dim): (1, 3, 768)
same token twice, identical raw embeddings? True
...but identical after adding position?     False

without positions the vector for ' dog' is the same in both sentences? True
  -> attention would see the same SET of vectors in both orders: it is amnesiac about order
with positions the vector for ' dog' is the same in both sentences?    False
  -> now each token carries WHERE it sits, so attention can tell who bites whom
```
- **Embedding = row lookup.** Token id 995 simply means "take row 995 of a 50257 x 768 matrix". Training changes the rows so useful similarities appear (Monday~Tuesday = 0.89).
- **Cosine similarity** compares direction, not length: 1 = same, 0 = unrelated.
- **Amnesiac check (last lines):** without positions the vector for ` dog` is identical in `dog bites man` and `man bites dog`; with `wpe` added it differs.
- **Position:** the model adds `wpe[position]` to `wte[token]`. Without it "dog bites man" and "man bites dog" would be identical bags of words. Many modern LLMs use rotary (RoPE) instead of a learned table, same purpose.
- These are the **input** embeddings; "sentence embedding" models for search (RAG) produce one vector per sentence by pooling later layers - same idea, different model.

## Experiments
- Find the most/least similar pairs you can think of from single-token words.
- Print `vec("cat")[:5]`: numbers alone mean nothing, meaning lives in relations between vectors.
- Compare `cosine(vec("the"), vec("a"))` with `cosine(vec("the"), vec("elephant"))`.

## Debugging practice
`toy(torch.tensor([7]))` -> `IndexError: index out of range in self`. The table has 5 rows (valid ids 0-4). Same family of bug appears in module 05 when a prompt is longer than 1024 (position table too short).
`vec("unbelievably")` -> your own `AssertionError` says it's 4 tokens - good assertions turn silent bugs into readable messages.

## Rebuild checklist
- [ ] Build `nn.Embedding(5, 3)` and look up ids; verify against `weight[ids]`.
- [ ] Get the real table with `get_input_embeddings()` and print its shape.
- [ ] Write `cosine(a, b)` and rank three word pairs.
- [ ] Add position embeddings to token embeddings with `torch.arange`.

## The new problem (cliffhanger)
Each token now has a learned vector and a position, but the vector is **fixed**: `bank` gets the same 768 numbers in "river bank" and "bank loan", and no token knows what its neighbours are. The input embedding is a lookup, not an understanding. We need a step where every token looks at the others and rebuilds its vector from them. That is self-attention: [Step 04 - Attention from scratch](../04_attention_from_scratch/WALKTHROUGH.md).

> **Say it aloud (interview box):** An embedding layer is a trainable lookup table of shape vocab by hidden size that maps each token id to a learned vector, so related tokens like Monday and Tuesday end up with high cosine similarity. Because attention has no sense of order, a position embedding is added to each token vector, and the same word at two positions then gets two different inputs. The limitation is that input embeddings are context-free, so bank has one vector regardless of meaning, and attention is what makes it contextual.
