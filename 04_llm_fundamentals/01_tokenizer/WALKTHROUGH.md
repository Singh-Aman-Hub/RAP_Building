# Step 01 - Tokenizer (The Bridge of Babylon)

## The story so far
[Step 00](../00_what_is_an_llm/WALKTHROUGH.md) ended with a crash: the counting toy keyed its table by whole words, so `"elephant"` (never seen) gave `ZeroDivisionError`. A real model cannot be keyed by words, and a neural network cannot read letters at all. We need a bridge from any string to a short list of integers.

## The pain point
Two obvious bridges both fail. **Character-level**: the vocabulary is tiny, but a sentence becomes very long (45 characters for a 6-word sentence), and attention cost grows with length squared (step 04). **Word-level**: sequences are short, but the vocabulary is infinite - every typo, name, and inflected form is a new entry, and any unseen word is out-of-vocabulary. In the demo a 5-word vocabulary leaves 3 of 6 words unknown.

## The spark
Build the vocabulary from pieces, not words. **Byte-pair encoding (BPE)** starts from single bytes and keeps merging the most frequent neighbouring pair seen in training text. Common words become one token, rare words are assembled from smaller known pieces, and nothing is ever unknown. GPT-2's result: 50,257 tokens, 9 tokens for the same 45-character sentence.

## Goal
Turn text into the integer IDs a model consumes (and back), and understand why words split into pieces.

## Run it
```bash
.venv/bin/python 04_llm_fundamentals/01_tokenizer/main.py
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Token | sub-word, piece, wordpiece, BPE unit | Roughly | A vocabulary item; may be a word, part of a word, or a single byte |
| BPE | byte-pair encoding, byte-level BPE | Yes | Merge the most frequent neighbouring pairs repeatedly; GPT-2 does it on bytes |
| WordPiece / SentencePiece / Unigram | other tokenisation algorithms | Different algorithms | BERT uses WordPiece, many LLMs use SentencePiece (BPE or Unigram); same goal, different training rules |
| Token ID | input id, `input_ids` | Yes | The integer index of a token in the vocabulary |
| `encode` vs `tokenize` vs `tok(...)` | text-to-ids, text-to-pieces, full call | No | `tokenize` gives strings, `encode` gives ids, calling `tok(...)` also gives `attention_mask` and tensors |
| EOS / `<|endoftext|>` | end-of-sequence token, stop token | Yes | Special token marking the end of a document; GPT-2 id 50256 |
| Padding / attention mask | pad token, mask | Related | Pad fills short rows so a batch is rectangular; the mask tells the model which entries are real |
| `Ġ` | space marker | Yes | GPT-2 byte-level notation for a leading space inside a token |
| Vocabulary size vs context length | V vs T | No | V = number of distinct tokens (50,257); T = max tokens per input (1,024 for GPT-2) |

## Code, step by step

### 1. Module docstring: why we tokenise  (`main.py` lines 1–16)

```python
"""
GOAL:   Convert text -> token IDs -> text with the GPT-2 tokenizer and understand
        why words split the way they do.

RUN:    from this folder:  python main.py
        from repo root:    .venv/bin/python 04_llm_fundamentals/01_tokenizer/main.py
EXPECTED OUTPUT (short): 'Hello world' -> [15496, 995]; rare words split into several pieces;
        a leading space changes the ID.
NEEDS:  transformers (third-party). First run downloads the tokenizer files (~1 MB).

Why tokenize?  Neural networks only understand numbers.  A tokenizer chops text into
pieces ("tokens") from a fixed vocabulary and gives each piece an integer ID.
GPT-2 uses Byte-Pair Encoding (BPE): start from bytes/characters and repeatedly
merge the most frequent neighbouring pairs seen in training data.  Frequent words
become 1 token; rare words become several smaller tokens.
"""
```

- **In simple words:** Explains that models only handle numbers, so text must be chopped into pieces with integer ids.
- **Exactly:** Docstring only. GPT-2 uses byte-level BPE: start from bytes, merge frequent pairs; frequent words become 1 token, rare words several.
- **Why it is here:** This is the front door of every LLM, and also the unit of cost, speed and context limits.

### 2. Load the tokenizer  (`main.py` lines 18–22)

```python
from transformers import AutoTokenizer  # loads the right tokenizer class for a model name

# from_pretrained downloads (once) and caches the vocabulary + merge rules.
tok = AutoTokenizer.from_pretrained("distilgpt2")  # distilgpt2 shares GPT-2's tokenizer
print("vocabulary size:", tok.vocab_size)           # number of distinct tokens (50257)
```

- **In simple words:** Download (or read from cache) the rules for chopping text the way GPT-2 does.
- **Exactly:** `AutoTokenizer.from_pretrained("distilgpt2")` picks the right tokenizer class from the model's config and loads vocabulary and merge rules (works offline from the local cache when `HF_HUB_OFFLINE=1`). `tok.vocab_size` is `50257`. DistilGPT-2 shares GPT-2's tokenizer.
- **Why it is here:** The tokenizer must match the model; a different one would map ids to different words and give nonsense.

### 3. Compare characters, words and BPE tokens  (`main.py` lines 24–35)

```python
# ---------------------------------------------------------------------------
# 0. The two bad bridges BPE replaces (the villains of this step)
#    Character-level: tiny vocabulary, but sequences get very long -> slow, and
#    attention cost grows with length squared (step 04).
#    Word-level: short sequences, but the vocabulary is endless and any new word
#    ("unbelievably", a typo, a Hindi word) is out-of-vocabulary.
# ---------------------------------------------------------------------------
sentence = "Tokenizers unbelievably handle typos like teh"
char_ids = list(sentence)                            # list[str], shape: (n_chars,)
word_ids = sentence.split()                          # list[str], shape: (n_words,)
bpe_ids = tok.encode(sentence)                       # list[int], shape: (n_bpe_tokens,)
print("chars:", len(char_ids), " words:", len(word_ids), " BPE tokens:", len(bpe_ids))
```

- **In simple words:** Count the same sentence three ways to see why subwords are the sweet spot.
- **Exactly:** `sentence = "Tokenizers unbelievably handle typos like teh"`. `list(sentence)` is 45 characters, `.split()` is 6 words, `tok.encode` is 9 BPE tokens. So BPE sequences are 5x shorter than characters but still cover any text.
- **Why it is here:** Sequence length matters because attention cost grows with length squared (step 04); character level is too long, word level has endless vocabulary.

### 4. Word-level vocabularies hit unknown words  (`main.py` lines 36–39)

```python
small_word_vocab = {"handle", "typos", "like", "the", "cat"}   # a pretend word-level vocabulary
unknown = [w for w in word_ids if w.lower() not in small_word_vocab]
print("word-level with a 5-word vocab: unknown words =", len(unknown), "of", len(word_ids))
print("BPE never has an unknown word:", tok.convert_ids_to_tokens(tok.encode(" teh")))
```

- **In simple words:** Pretend we only knew five words and see how many words in the sentence we would not recognise.
- **Exactly:** The list comprehension keeps words whose lowercase is not in the toy set: 3 of 6 are unknown (`Tokenizers`, `unbelievably`, `teh`). BPE has no unknown words: `tok.encode(" teh")` gives pieces `['Ġte', 'h']`.
- **Why it is here:** This is the out-of-vocabulary problem that BPE solves by falling back to smaller pieces, down to single bytes.

### 5. Text to ids and to pieces  (`main.py` lines 41–50)

```python
# ---------------------------------------------------------------------------
# 1. Text -> IDs
# ---------------------------------------------------------------------------
text = "Hello world"
ids = tok.encode(text)          # list[int], shape: (n_tokens,) = (2,)
print("\n", repr(text), "->", ids)

# convert_ids_to_tokens shows the actual string pieces. 'Ġ' means "a space came before this piece".
pieces = tok.convert_ids_to_tokens(ids)   # list[str], shape: (n_tokens,) = (2,)
print("pieces:", pieces)
```

- **In simple words:** Turn a sentence into numbers and show which piece each number stands for.
- **Exactly:** `tok.encode("Hello world")` returns `list[int]` `[15496, 995]`, length 2. `convert_ids_to_tokens(ids)` gives `['Hello', 'Ġworld']`. The `Ġ` marks a preceding space.
- **Why it is here:** Seeing both ids and pieces makes tokenisation concrete when you debug prompts.

### 6. Decode round trip  (`main.py` lines 52–56)

```python
# ---------------------------------------------------------------------------
# 2. IDs -> text (decode) must round-trip
# ---------------------------------------------------------------------------
print("decoded:", repr(tok.decode(ids)))
assert tok.decode(ids) == text  # assert = crash loudly if the claim is False
```

- **In simple words:** Turn the ids back into text and check we get the original.
- **Exactly:** `tok.decode(ids)` returns `'Hello world'`. `assert` raises `AssertionError` if the claim is false.
- **Why it is here:** Lossless decode is a property of byte-level BPE. A tokenizer that cannot round-trip would corrupt outputs.

### 7. Spaces change the token  (`main.py` lines 58–62)

```python
# ---------------------------------------------------------------------------
# 3. Spaces matter: "world" and " world" are DIFFERENT tokens
# ---------------------------------------------------------------------------
print("\n'world'  ->", tok.encode("world"))
print("' world' ->", tok.encode(" world"))
```

- **In simple words:** A word at the start of text and the same word after a space are different tokens.
- **Exactly:** Real output: `'world' -> [6894]` and `' world' -> [995]`. Different ids, different embeddings.
- **Why it is here:** Prompt formatting with trailing spaces or newlines can change model behaviour. It is a common source of odd results.

### 8. Rare words, dates and Hindi split into many pieces  (`main.py` lines 64–69)

```python
# ---------------------------------------------------------------------------
# 4. Rare words / typos / numbers / other languages split into many pieces
# ---------------------------------------------------------------------------
for sample in ["cat", "unbelievably", "Pneumonoultramicroscopic", "transformers", "2024-10-19", "नमस्ते"]:
    p = tok.tokenize(sample)    # list[str], shape: (n_pieces,); tokenize = text -> string pieces (no IDs)
    print(f"{sample!r:30} -> {len(p):2d} tokens: {p}")
```

- **In simple words:** Common words are one token; rare words and non-English text break into many.
- **Exactly:** `tok.tokenize(sample)` returns `list[str]`. Real counts: `cat` 1, `unbelievably` 4 (`un, bel, iev, ably`), `Pneumonoultramicroscopic` 8, `transformers` 2, `2024-10-19` 6, `नमस्ते` (6 characters) 12 tokens. The Hindi pieces look garbled because they are raw UTF-8 bytes mapped to printable symbols.
- **Why it is here:** Cost, latency and context use are priced per token, so non-English text can cost 2x or more; this motivates larger multilingual vocabularies in newer models.

### 9. Special tokens  (`main.py` lines 71–75)

```python
# ---------------------------------------------------------------------------
# 5. Special tokens: markers that are not normal text.
# ---------------------------------------------------------------------------
print("\nend-of-text token:", tok.eos_token, "id =", tok.eos_token_id)
print("pad token:", tok.pad_token, "(GPT-2 has none by default; people reuse EOS when batching)")
```

- **In simple words:** Reserved markers that are not normal text, such as "end of text".
- **Exactly:** `tok.eos_token` is `<|endoftext|>` with id `50256`. `tok.pad_token` is `None` for GPT-2, so there is no padding token by default.
- **Why it is here:** Models learn to emit EOS to stop. Without a pad token batching fails with "Asking to pad but the tokenizer does not have a padding token".

### 10. Build a padded batch of tensors with an attention mask  (`main.py` lines 77–87)

```python
# ---------------------------------------------------------------------------
# 6. What the model actually receives: a batch as PyTorch tensors + attention mask
# ---------------------------------------------------------------------------
tok.pad_token = tok.eos_token          # reuse EOS as padding so we can batch unequal sentences
batch = tok(                            # output tensors have shape (batch=2, seq_len=7); calling the tokenizer object = full-featured encode
    ["Hi", "Hello there, how are you?"],
    padding=True,                       # pad shorter sentences up to the longest
    return_tensors="pt",                # "pt" = PyTorch tensors
)
print("\ninput_ids (shape (batch, seq_len)):\n", batch["input_ids"])
print("attention_mask (1 = real token, 0 = padding):\n", batch["attention_mask"])
```

- **In simple words:** Feed two sentences of different length together: shorter one is padded, and a mask says which tokens are real.
- **Exactly:** Setting `tok.pad_token = tok.eos_token` reuses id 50256 as padding. `tok([...], padding=True, return_tensors="pt")` returns a `BatchEncoding` with `input_ids` of shape `(2, 7)`: row 0 `[17250, 50256 x6]` ("Hi" is one token), row 1 `[15496, 612, 11, 703, 389, 345, 30]`. `attention_mask` has the same shape: row 0 `[1, 0, 0, 0, 0, 0, 0]`, row 1 all ones.
- **Why it is here:** GPUs need rectangular tensors. The mask stops attention from reading padding (step 04).

### 11. Experiments and break-it comments  (`main.py` lines 89–92)

```python
# TRY IT: encode your own name and a Hindi sentence - how many tokens per character?
# TRY IT: compare len(tok.encode("the " * 50)) with len(tok.encode("qzxv " * 50)).
# BREAK IT: tok.decode("hello") -> TypeError: Can't extract `str` to `Vec`; decode wants a list of ints, not a string.
# BREAK IT: AutoTokenizer.from_pretrained("distilgpt3") -> OSError; read how it tells you the repo is missing.
```

- **In simple words:** Suggestions to try on your own name and to see two typical errors.
- **Exactly:** `tok.decode("hello")` raises `TypeError` because decode wants a list of ints; a misspelt model name raises `OSError` saying the repository was not found.
- **Why it is here:** Reading these two errors once saves time later.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `from x import Y` | import one name from a module | `from pathlib import Path` |
| `repr(text)` / `!r` in f-strings | show the string with quotes and escape characters | `repr("a\n")` -> `'a\\n'` |
| `assert cond` | crash with `AssertionError` if cond is False | `assert 1 + 1 == 3` fails |
| `for s in [..]:` | loop over a list | - |
| `f"{x!r:30}"` | f-string: `!r` = repr, `:30` = pad to width 30 | `f"{'a'!r:5}|"` -> `'a'  |` |
| `batch["input_ids"]` | dictionary lookup by key | `{"a": 1}["a"]` |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `transformers.AutoTokenizer` | 3rd-party (Hugging Face) | picks the right tokenizer class for a model name | - | class |
| `AutoTokenizer.from_pretrained(name)` | 3rd-party | downloads/caches vocabulary + merge rules | model name or local path | tokenizer |
| `tok.encode(text)` | 3rd-party | text -> list of ids | `add_special_tokens` | `list[int]` |
| `tok.decode(ids)` | 3rd-party | ids -> text | `skip_special_tokens` | `str` |
| `tok.tokenize(text)` | 3rd-party | text -> string pieces (no ids) | - | `list[str]` |
| `tok.convert_ids_to_tokens(ids)` | 3rd-party | ids -> string pieces | - | `list[str]` |
| `tok(texts, padding, return_tensors)` | 3rd-party | full pipeline: ids + attention_mask for a batch | `padding=True`, `return_tensors="pt"`, `truncation`, `max_length` | `BatchEncoding` (dict-like) |
| `tok.vocab_size`, `tok.eos_token_id` | 3rd-party | vocabulary size / end-of-text id | - | int |

## Execution trace
Real output (the Hindi pieces are byte fragments, shown as they print):
```
vocabulary size: 50257
chars: 45  words: 6  BPE tokens: 9
word-level with a 5-word vocab: unknown words = 3 of 6
BPE never has an unknown word: ['Ġte', 'h']

 'Hello world' -> [15496, 995]
pieces: ['Hello', 'Ġworld']
decoded: 'Hello world'

'world'  -> [6894]
' world' -> [995]
'cat'                          ->  1 tokens: ['cat']
'unbelievably'                 ->  4 tokens: ['un', 'bel', 'iev', 'ably']
'Pneumonoultramicroscopic'     ->  8 tokens: ['P', 'neum', 'on', 'oult', 'ram', 'icro', 'sc', 'opic']
'transformers'                 ->  2 tokens: ['transform', 'ers']
'2024-10-19'                   ->  6 tokens: ['20', '24', '-', '10', '-', '19']
'नमस्ते'                       -> 12 tokens: ['à¤', '¨', 'à¤', '®', 'à¤', '¸', 'à¥', 'į', 'à¤', '¤', 'à¥', 'ĩ']

end-of-text token: <|endoftext|> id = 50256
pad token: None (GPT-2 has none by default; people reuse EOS when batching)

input_ids (shape (batch, seq_len)):
 tensor([[17250, 50256, 50256, 50256, 50256, 50256, 50256],
        [15496,   612,    11,   703,   389,   345,    30]])
attention_mask (1 = real token, 0 = padding):
 tensor([[1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1]])
```
- **Why BPE (section 0):** 45 characters, 6 words, 9 BPE tokens for the same sentence. Characters are too long, words leave 3 of 6 unknown, BPE has no unknown words (`teh` -> `Ġte`, `h`).
- **BPE intuition:** start with single bytes; repeatedly merge the most common neighbouring pair seen in training text. Common words ("cat") end up as one token; rare ones are assembled from smaller known pieces.
- **`Ġ`** is how GPT-2 displays a leading space. `world` (6894) and ` world` (995) are different tokens - a classic source of prompt bugs.
- **Hindi costs 12 tokens for 6 characters**: GPT-2's vocabulary was built mostly on English, so other scripts fall back to raw bytes. Consequence: higher cost, shorter effective context, often worse quality.
- **Padding + attention mask:** to batch sentences of different length, short ones are padded; the mask (0 = ignore) tells attention to skip them. GPT-2 has no pad token, so we reuse `<|endoftext|>`.

## Experiments
- Tokenize your own name and a sentence in your mother tongue; compute tokens per character.
- Compare `len(tok.encode("the " * 50))` and `len(tok.encode("qzxv " * 50))`. (Repeated common word vs gibberish.)
- Switch the model name to `"gpt2"`: IDs are identical (same tokenizer). Try `"bert-base-uncased"`: different vocabulary (WordPiece), different IDs.

## Debugging practice
`tok.decode("hello")` -> ``TypeError: Can't extract `str` to `Vec` ``. Meaning: decode expects a list of ids, not a `str`. Fix: `tok.decode(tok.encode("hello"))`.
`AutoTokenizer.from_pretrained("distilgpt3")` -> `OSError: distilgpt3 is not a local folder and is not a valid model identifier...`. Read the sentence: it tells you the name is wrong (or you're offline). Check spelling, then network, then `~/.cache/huggingface`.

## Rebuild checklist
- [ ] Load a tokenizer, encode a sentence, decode it and assert equality.
- [ ] Print the pieces and explain every `Ġ`.
- [ ] Batch two sentences with padding and explain the attention mask.

## The new problem (cliffhanger)
We now have ids like `[15496, 995]`. But ids are arbitrary integers: 995 is not "closer" to 996 than to 5, and `cat` and `dog` have no relation in their numbers. A network cannot learn from numbers that carry no geometry. We need a way to turn text into vectors where distance means something. We start with the vectors you already know: [Step 02 - Vectors and similarity](../02_vectors_and_similarity/WALKTHROUGH.md).

> **Say it aloud (interview box):** A tokenizer splits text into sub-word tokens from a fixed vocabulary and maps each to an integer id, because characters make sequences too long and whole words make the vocabulary infinite. GPT-2 uses byte-pair encoding, so `Hello world` is two ids, ` world` and `world` are different tokens, and non-English text splits into many pieces that cost context and money. The ids are just labels with no geometry, so the next step is turning them into vectors.
