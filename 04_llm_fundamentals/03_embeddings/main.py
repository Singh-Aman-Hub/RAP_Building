"""
GOAL:   See that an embedding is just a row lookup in a big table, measure
        similarity between word vectors, and see how position information is added.

RUN:    from this folder:  python main.py
        from repo root:    .venv/bin/python 04_llm_fundamentals/03_embeddings/main.py
EXPECTED OUTPUT (short): a (5, 3) toy embedding; embedding table shape (50257, 768) for distilgpt2;
        cosine similarity higher for related words (cat~dog) than unrelated (cat~economics).
NEEDS:  torch + transformers. First run downloads distilgpt2 (~330 MB, cached afterwards).

Embedding = a learned vector (list of numbers) for each token ID.  Similar meanings
end up with similar vectors because training pushes them that way.
Transformers also have NO built-in sense of word order, so a "position embedding"
is added to tell the model where each token sits.
"""

import torch                      # PyTorch: tensors + neural-network building blocks
import torch.nn as nn             # nn = layers (Linear, Embedding, ...). 'as nn' is just a short alias
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# 1. A TOY embedding layer: vocabulary of 5 ids, each mapped to 3 numbers.
# ---------------------------------------------------------------------------
torch.manual_seed(0)                       # reproducible random initialisation
toy = nn.Embedding(num_embeddings=5, embedding_dim=3)   # table with 5 rows x 3 columns
print("toy table shape:", toy.weight.shape)             # .weight is the learnable table
ids = torch.tensor([4, 0, 4])              # three token ids (note: id 4 appears twice)
print("lookup of ids [4, 0, 4]:\n", toy(ids))           # row 4, row 0, row 4 -> identical rows for id 4
print("same as indexing the table directly:", torch.equal(toy(ids), toy.weight[ids]))

# ---------------------------------------------------------------------------
# 2. The REAL embedding table of distilgpt2
# ---------------------------------------------------------------------------
tok = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
wte = model.get_input_embeddings()          # "word token embedding" layer (an nn.Embedding)
print("\nreal table shape (vocab, dim):", tuple(wte.weight.shape))

def vec(word):
    """Return the embedding of a word that is exactly ONE token (we add a leading space like in real text)."""
    token_ids = tok.encode(" " + word)
    assert len(token_ids) == 1, f"{word!r} is {len(token_ids)} tokens, pick a simpler word"
    return wte.weight[token_ids[0]].detach()   # .detach() = plain numbers, no gradient tracking

# ---------------------------------------------------------------------------
# 3. Cosine similarity: 1.0 = same direction, 0 = unrelated, -1 = opposite.
# ---------------------------------------------------------------------------
def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()   # .item() -> normal Python float

for w1, w2 in [("cat", "dog"), ("king", "queen"), ("cat", "economics"), ("Monday", "Tuesday"), ("car", "banana")]:
    print(f"similarity({w1!r}, {w2!r}) = {cosine(vec(w1), vec(w2)):.3f}")

# ---------------------------------------------------------------------------
# 4. Position embeddings: GPT-2 learns one vector per position (0..1023).
# ---------------------------------------------------------------------------
wpe = model.transformer.wpe                 # "word position embedding"
print("\nposition table shape (max positions, dim):", tuple(wpe.weight.shape))

text_ids = torch.tensor([tok.encode("The cat sat")])      # shape (1, 3): a batch of one sentence
positions = torch.arange(text_ids.shape[1])                # tensor([0, 1, 2])
# What goes into the first transformer block = token vector + position vector
x = wte(text_ids) + wpe(positions)          # broadcasting adds (1,3,768) + (3,768)
print("input to first block, shape (batch, seq, dim):", tuple(x.shape))

# Same word at different positions gets a different final input vector:
same_word = torch.tensor([tok.encode(" the the")])
xx = wte(same_word) + wpe(torch.arange(2))
print("same token twice, identical raw embeddings?", torch.equal(wte(same_word)[0, 0], wte(same_word)[0, 1]))
print("...but identical after adding position?    ", torch.equal(xx[0, 0], xx[0, 1]))

# TRY IT: find word pairs with the HIGHEST and LOWEST similarity you can think of.
# TRY IT: print the first 5 numbers of vec("cat") - they are not human-readable; meaning lives in the geometry.
# BREAK IT: vec("unbelievably") -> AssertionError (multi-token word). Read the message you wrote for yourself.
# BREAK IT: toy(torch.tensor([7])) -> IndexError: index out of range in self (id 7 is not in a 5-row table).
