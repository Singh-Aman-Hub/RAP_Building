"""
GOAL:   Convert text -> token IDs -> text with the GPT-2 tokenizer and understand
        why words split the way they do.

RUN:    from this folder:  python main.py
        from repo root:    .venv/bin/python 04_llm_fundamentals/02_tokenizer/main.py
EXPECTED OUTPUT (short): 'Hello world' -> [15496, 995]; rare words split into several pieces;
        a leading space changes the ID.
NEEDS:  transformers (third-party). First run downloads the tokenizer files (~1 MB).

Why tokenize?  Neural networks only understand numbers.  A tokenizer chops text into
pieces ("tokens") from a fixed vocabulary and gives each piece an integer ID.
GPT-2 uses Byte-Pair Encoding (BPE): start from bytes/characters and repeatedly
merge the most frequent neighbouring pairs seen in training data.  Frequent words
become 1 token; rare words become several smaller tokens.
"""

from transformers import AutoTokenizer  # loads the right tokenizer class for a model name

# from_pretrained downloads (once) and caches the vocabulary + merge rules.
tok = AutoTokenizer.from_pretrained("distilgpt2")  # distilgpt2 shares GPT-2's tokenizer
print("vocabulary size:", tok.vocab_size)           # number of distinct tokens (50257)

# ---------------------------------------------------------------------------
# 1. Text -> IDs
# ---------------------------------------------------------------------------
text = "Hello world"
ids = tok.encode(text)          # returns a plain Python list of ints
print("\n", repr(text), "->", ids)

# convert_ids_to_tokens shows the actual string pieces. 'Ġ' means "a space came before this piece".
pieces = tok.convert_ids_to_tokens(ids)
print("pieces:", pieces)

# ---------------------------------------------------------------------------
# 2. IDs -> text (decode) must round-trip
# ---------------------------------------------------------------------------
print("decoded:", repr(tok.decode(ids)))
assert tok.decode(ids) == text  # assert = crash loudly if the claim is False

# ---------------------------------------------------------------------------
# 3. Spaces matter: "world" and " world" are DIFFERENT tokens
# ---------------------------------------------------------------------------
print("\n'world'  ->", tok.encode("world"))
print("' world' ->", tok.encode(" world"))

# ---------------------------------------------------------------------------
# 4. Rare words / typos / numbers / other languages split into many pieces
# ---------------------------------------------------------------------------
for sample in ["cat", "unbelievably", "Pneumonoultramicroscopic", "transformers", "2024-10-19", "नमस्ते"]:
    p = tok.tokenize(sample)    # tokenize = text -> string pieces (no IDs)
    print(f"{sample!r:30} -> {len(p):2d} tokens: {p}")

# ---------------------------------------------------------------------------
# 5. Special tokens: markers that are not normal text.
# ---------------------------------------------------------------------------
print("\nend-of-text token:", tok.eos_token, "id =", tok.eos_token_id)
print("pad token:", tok.pad_token, "(GPT-2 has none by default; people reuse EOS when batching)")

# ---------------------------------------------------------------------------
# 6. What the model actually receives: a batch as PyTorch tensors + attention mask
# ---------------------------------------------------------------------------
tok.pad_token = tok.eos_token          # reuse EOS as padding so we can batch unequal sentences
batch = tok(                            # calling the tokenizer object = full-featured encode
    ["Hi", "Hello there, how are you?"],
    padding=True,                       # pad shorter sentences up to the longest
    return_tensors="pt",                # "pt" = PyTorch tensors
)
print("\ninput_ids:\n", batch["input_ids"])
print("attention_mask (1 = real token, 0 = padding):\n", batch["attention_mask"])

# TRY IT: encode your own name and a Hindi sentence - how many tokens per character?
# TRY IT: compare len(tok.encode("the " * 50)) with len(tok.encode("qzxv " * 50)).
# BREAK IT: tok.decode("hello") -> TypeError-ish error; decode wants a list of ints, not a string.
# BREAK IT: AutoTokenizer.from_pretrained("distilgpt3") -> OSError; read how it tells you the repo is missing.
