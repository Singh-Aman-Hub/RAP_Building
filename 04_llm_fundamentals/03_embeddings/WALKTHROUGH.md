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

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Embedding | embedding vector, token vector, dense representation | Yes | A learned vector (here 768 numbers) for each token id |
| `nn.Embedding` | lookup table, `wte`, embedding matrix, lookup layer | Yes | A weight matrix `(vocab, dim)` where the forward pass is just "pick row i" |
| `wte` / `wpe` | word-token embedding / word-position embedding | Different tables | `wte` has one row per token id; `wpe` has one row per position (0 to 1023) |
| Positional encoding | position embedding, sinusoidal / learned / RoPE | Related, not identical | GPT-2 learns a position table; the original transformer used fixed sine waves; modern LLMs use rotary (RoPE) applied inside attention |
| Cosine similarity | cosine, angle similarity | Yes | Compares direction of two vectors, ignoring length |
| Static vs contextual embedding | Word2Vec vs BERT/GPT hidden state | No | A static embedding is one vector per token; a contextual one changes with the neighbouring words (output of attention layers) |
| `hidden_size` / `d_model` / `n_embd` / embedding dim | model width | Yes | Same number under different library names (768 for GPT-2 small) |
| `.detach()` | stop gradient, `.data`, `torch.no_grad` | Related | `detach` returns a tensor sharing data but cut from the autograd graph |

## Code, step by step

### 1. Module docstring: embedding is a table lookup  (`main.py` lines 1–17)

```python
"""
GOAL:   See that an embedding is just a row lookup in a big table, measure
        similarity between word vectors, and see how position information is added.

RUN:    from this folder:  python main.py
        from repo root:    .venv/bin/python 04_llm_fundamentals/03_embeddings/main.py
EXPECTED OUTPUT (short): a (5, 3) toy embedding; embedding table shape (50257, 768) for distilgpt2;
        cosine similarity higher for related words (cat~dog) than unrelated (cat~economics).
NEEDS:  torch + transformers. First run downloads distilgpt2 (~330 MB, cached afterwards).

Embedding = a learned vector (list of numbers) for each token ID.  Step 02 showed why we want
dense vectors where distance means something; here the NETWORK learns them (one table row per id).  Similar meanings
end up with similar vectors because training pushes them that way.
Transformers also have NO built-in sense of word order, so a "position embedding"
is added to tell the model where each token sits.  What is still missing after this step:
CONTEXT.  Each token vector is fixed; mixing in the neighbours is attention (step 04).
"""
```

- **In simple words:** Says an embedding is a row of a big learned table, and shows what is still missing (context).
- **Exactly:** Docstring only. It promises a toy `(5, 3)` table, the real `(50257, 768)` table of distilgpt2 and cosine comparisons.
- **Why it is here:** Sets up the key idea that "embedding = row lookup".

### 2. Imports  (`main.py` lines 19–21)

```python
import torch                      # PyTorch: tensors + neural-network building blocks
import torch.nn as nn             # nn = layers (Linear, Embedding, ...). 'as nn' is just a short alias
from transformers import AutoTokenizer, AutoModelForCausalLM
```

- **In simple words:** PyTorch, its layers module, and the Hugging Face loaders.
- **Exactly:** `nn.Embedding` is the lookup layer. `AutoTokenizer` and `AutoModelForCausalLM` load tokenizer and a GPT-style model with a language-modelling head.
- **Why it is here:** The real table lives inside a pre-trained model, so we need to load one.

### 3. A toy embedding table of 5 rows by 3 columns  (`main.py` lines 23–31)

```python
# ---------------------------------------------------------------------------
# 1. A TOY embedding layer: vocabulary of 5 ids, each mapped to 3 numbers.
# ---------------------------------------------------------------------------
torch.manual_seed(0)                       # reproducible random initialisation
toy = nn.Embedding(num_embeddings=5, embedding_dim=3)   # weight shape: (5, 3) = (vocab, dim); table with 5 rows x 3 columns
print("toy table shape:", toy.weight.shape)             # .weight is the learnable table
ids = torch.tensor([4, 0, 4])              # shape: (3,) = (n_ids,); three token ids (note: id 4 appears twice)
print("lookup of ids [4, 0, 4]:\n", toy(ids))   # output shape: (3, 3) = (n_ids, embedding_dim)           # row 4, row 0, row 4 -> identical rows for id 4
print("same as indexing the table directly:", torch.equal(toy(ids), toy.weight[ids]))
```

- **In simple words:** A tiny table where each of 5 token ids has 3 numbers, and looking up an id returns its row.
- **Exactly:** `torch.manual_seed(0)` fixes the random init. `nn.Embedding(num_embeddings=5, embedding_dim=3).weight` is shape `(5, 3)`, trainable. `toy(torch.tensor([4, 0, 4]))` returns `(3, 3)`: rows 4, 0, 4. Real output rows 0 and 2 are identical `[-0.8567, 1.1006, -1.0712]`. The tensor shows `grad_fn=<EmbeddingBackward0>` because the weights require grad. `torch.equal(toy(ids), toy.weight[ids])` is `True`: a lookup is plain indexing.
- **Why it is here:** Demystifies embeddings: no maths, just indexing. Training nudges the rows so that useful tokens get useful vectors.

### 4. Load the real distilgpt2 embedding table  (`main.py` lines 33–39)

```python
# ---------------------------------------------------------------------------
# 2. The REAL embedding table of distilgpt2
# ---------------------------------------------------------------------------
tok = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
wte = model.get_input_embeddings()          # weight shape: (50257, 768) = (vocab, dim); "word token embedding" layer (an nn.Embedding)
print("\nreal table shape (vocab, dim):", tuple(wte.weight.shape))
```

- **In simple words:** Fetch the actual model and grab its word-to-vector table.
- **Exactly:** `AutoModelForCausalLM.from_pretrained("distilgpt2")` loads 76 weight tensors (the log shows `76/76`). `get_input_embeddings()` returns the `nn.Embedding` whose `weight` has shape `(50257, 768)` = (vocab, dim): about 38.6 million parameters.
- **Why it is here:** In small models the embedding table is a large part of all parameters, so tied input/output embeddings are common.

### 5. Helper: get the vector of a one-token word  (`main.py` lines 41–45)

```python
def vec(word):
    """Return the embedding of a word that is exactly ONE token (we add a leading space like in real text)."""
    token_ids = tok.encode(" " + word)
    assert len(token_ids) == 1, f"{word!r} is {len(token_ids)} tokens, pick a simpler word"
    return wte.weight[token_ids[0]].detach()   # shape: (768,) = (dim,); .detach() = plain numbers, no gradient tracking
```

- **In simple words:** A function that looks up a word's vector, but only if the word is one token.
- **Exactly:** `tok.encode(" " + word)` adds the leading space to match how words appear mid-sentence; `assert len(token_ids) == 1` raises `AssertionError` with a helpful message for multi-token words. `wte.weight[id]` is shape `(768,)`; `.detach()` removes the autograd tracking.
- **Why it is here:** Multi-token words have no single vector, so the helper refuses them.

### 6. Cosine similarity between word pairs  (`main.py` lines 47–54)

```python
# ---------------------------------------------------------------------------
# 3. Cosine similarity: 1.0 = same direction, 0 = unrelated, -1 = opposite.
# ---------------------------------------------------------------------------
def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()   # .item() -> normal Python float

for w1, w2 in [("cat", "dog"), ("king", "queen"), ("cat", "economics"), ("Monday", "Tuesday"), ("car", "banana")]:
    print(f"similarity({w1!r}, {w2!r}) = {cosine(vec(w1), vec(w2)):.3f}")
```

- **In simple words:** Measure how close the vectors for several word pairs are.
- **Exactly:** `F.cosine_similarity(a, b, dim=0)` compares two `(768,)` vectors; `.item()` gives a float. Real output: cat~dog 0.567, king~queen 0.664, cat~economics 0.203, Monday~Tuesday 0.893, car~banana 0.264. Related words score higher.
- **Why it is here:** Shows that training pushed similar meanings to similar directions. Raw GPT-2 input embeddings are only a rough geometry, which is why retrieval uses purpose-built embedding models (module 14).

### 7. Load the position table  (`main.py` lines 56–60)

```python
# ---------------------------------------------------------------------------
# 4. Position embeddings: GPT-2 learns one vector per position (0..1023).
# ---------------------------------------------------------------------------
wpe = model.transformer.wpe                 # weight shape: (1024, 768) = (max_positions, dim); "word position embedding"
print("\nposition table shape (max positions, dim):", tuple(wpe.weight.shape))
```

- **In simple words:** GPT-2 also has a second table with one vector per position number.
- **Exactly:** `model.transformer.wpe.weight` has shape `(1024, 768)`: max context length 1024, same width as the token table.
- **Why it is here:** Self-attention is order-blind, so position must be injected.

### 8. Add token and position vectors  (`main.py` lines 62–66)

```python
text_ids = torch.tensor([tok.encode("The cat sat")])      # shape: (1, 3) = (batch, seq_len)
positions = torch.arange(text_ids.shape[1])                # shape: (3,) = (seq_len,), values [0, 1, 2]
# What goes into the first transformer block = token vector + position vector
x = wte(text_ids) + wpe(positions)          # (1, 3, 768) + (3, 768) -> shape: (1, 3, 768) = (batch, seq_len, dim)
print("input to first block, shape (batch, seq, dim):", tuple(x.shape))
```

- **In simple words:** The input to the first transformer layer is the token vector plus the position vector.
- **Exactly:** `text_ids` is `(1, 3)` for "The cat sat". `positions = torch.arange(3)` is `[0, 1, 2]`. `wte(text_ids)` is `(1, 3, 768)` and `wpe(positions)` is `(3, 768)`; addition broadcasts over the batch axis to `(1, 3, 768)` = (batch, seq, dim). Printed `(1, 3, 768)`.
- **Why it is here:** This tensor `x` is exactly what enters block 1 in step 05.

### 9. The same word at two positions  (`main.py` lines 68–72)

```python
# Same word at different positions gets a different final input vector:
same_word = torch.tensor([tok.encode(" the the")])      # shape: (1, 2) = (batch, seq_len)
xx = wte(same_word) + wpe(torch.arange(2))   # shape: (1, 2, 768) = (batch, seq_len, dim)
print("same token twice, identical raw embeddings?", torch.equal(wte(same_word)[0, 0], wte(same_word)[0, 1]))
print("...but identical after adding position?    ", torch.equal(xx[0, 0], xx[0, 1]))
```

- **In simple words:** Same word twice, but after adding position the two vectors differ.
- **Exactly:** `" the the"` encodes to 2 tokens. Real output: raw embeddings identical `True`; after adding position `False`.
- **Why it is here:** This is how the model can tell the first "the" from the second one.

### 10. Without positions, word order is invisible  (`main.py` lines 74–86)

```python
# The "amnesiac model" problem: attention (step 04) treats its input as a SET of vectors.
# Without positions, "dog bites man" and "man bites dog" look identical to it.
pair_a = torch.tensor([tok.encode(" dog bites man")])    # shape: (1, 3) = (batch, seq_len)
pair_b = torch.tensor([tok.encode(" man bites dog")])    # shape: (1, 3)
dog_in_a = wte(pair_a)[0, 0]                             # shape: (768,): ' dog' is first in sentence A
dog_in_b = wte(pair_b)[0, 2]                             # shape: (768,): ' dog' is last in sentence B
print("\nwithout positions the vector for ' dog' is the same in both sentences?", torch.equal(dog_in_a, dog_in_b))
print("  -> attention would see the same SET of vectors in both orders: it is amnesiac about order")
pos3 = wpe(torch.arange(3))                              # shape: (3, 768) = (seq_len, dim)
dog_a_pos = (wte(pair_a) + pos3)[0, 0]                   # shape: (768,): ' dog' + position 0
dog_b_pos = (wte(pair_b) + pos3)[0, 2]                   # shape: (768,): ' dog' + position 2
print("with positions the vector for ' dog' is the same in both sentences?   ", torch.equal(dog_a_pos, dog_b_pos))
print("  -> now each token carries WHERE it sits, so attention can tell who bites whom")
```

- **In simple words:** "dog bites man" and "man bites dog" contain the same word vectors; only positions tell them apart.
- **Exactly:** `wte(pair_a)[0, 0]` is ' dog' first in A; `wte(pair_b)[0, 2]` is ' dog' last in B. Both `(768,)`. Without positions `torch.equal(...)` is `True`; with `wpe` added it is `False`. Attention treats its input as a set, so without positions the sentences are indistinguishable.
- **Why it is here:** Explains why every transformer has some positional mechanism (learned, sinusoidal or RoPE).

### 11. Experiments and break-it comments  (`main.py` lines 88–91)

```python
# TRY IT: find word pairs with the HIGHEST and LOWEST similarity you can think of.
# TRY IT: print the first 5 numbers of vec("cat") - they are not human-readable; meaning lives in the geometry.
# BREAK IT: vec("unbelievably") -> AssertionError (multi-token word). Read the message you wrote for yourself.
# BREAK IT: toy(torch.tensor([7])) -> IndexError: index out of range in self (id 7 is not in a 5-row table).
```

- **In simple words:** Ideas to try: highest and lowest similarity pairs; the two errors you might hit.
- **Exactly:** `vec("unbelievably")` raises the `AssertionError`; `toy(torch.tensor([7]))` raises `IndexError: index out of range in self` because the table has 5 rows.
- **Why it is here:** Index-out-of-range on embeddings is a common bug when vocabulary and tokenizer mismatch.

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
