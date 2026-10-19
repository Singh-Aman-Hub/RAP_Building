# Terms — Tokenizer

Folder: `04_llm_fundamentals/01_tokenizer` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [04 LLM fundamentals README](../README.md)

**Terms in this folder:** [transformers](#transformers) · [Token](#token) · [Tokenizer](#tokenizer) · [Byte-Pair Encoding (BPE)](#byte-pair-encoding-bpe) · [Vocabulary](#vocabulary) · [AutoTokenizer](#autotokenizer) · [from_pretrained](#from_pretrained) · [vocab_size](#vocab_size) · [tok.encode](#tokencode) · [Token ID](#token-id) · [repr and !r](#repr-and-r) · [convert_ids_to_tokens](#convert_ids_to_tokens) · [Ġ (space marker)](#ġ-space-marker) · [tok.decode](#tokdecode) · [assert](#assert) · [tok.tokenize](#toktokenize) · [Rare words and non-English text](#rare-words-and-non-english-text) · [Special token (EOS)](#special-token-eos) · [pad_token](#pad_token) · [Calling the tokenizer (tok(...))](#calling-the-tokenizer-tok) · [padding=True](#paddingtrue) · [return_tensors="pt"](#return_tensorspt) · [attention_mask](#attention_mask) · [input_ids](#input_ids) · [Dictionary lookup](#dictionary-lookup) · [OSError (model not found)](#oserror-model-not-found) · [Character-level vs word-level tokenization](#character-level-vs-word-level-tokenization) · [Out-of-vocabulary (OOV)](#out-of-vocabulary-oov)

---

## transformers  `(third-party: Hugging Face transformers)`
**1. Definition:** An open-source Python library from Hugging Face that provides pretrained models, tokenizers and utilities for transformer architectures.
**2. In plain words:** Instead of building a language model yourself you ask this library for one by name and it downloads and runs it. This step only needs its tokenizer part. It is the standard tool for working with GPT-2, BERT and many other models.
**3. Example to quote:** > "I used the Hugging Face `transformers` library to load the distilgpt2 tokenizer in one line."
```python
from transformers import AutoTokenizer
```
**Seen in:** `main.py`, import.

## Token  `(concept)`
**1. Definition:** A piece of text (a word, part of a word, punctuation or byte sequence) from a fixed vocabulary that a language model treats as one unit.
**2. In plain words:** Models do not read letters or whole words; they read tokens. "Hello world" is two tokens, while a rare word may be four or more. Tokens are also what you pay for and what the context window limit counts.
**3. Example to quote:** > "'unbelievably' became four tokens: `un`, `bel`, `iev`, `ably`, while `cat` was one."

**Seen in:** `main.py`, section 4.

## Tokenizer  `(concept)`
**1. Definition:** A component that converts raw text into a sequence of tokens (and integer IDs) and converts IDs back to text.
**2. In plain words:** Neural networks only handle numbers, so the tokenizer is the translator between your sentence and the model's input. It uses a fixed vocabulary and fixed rules, and each model has its own, so you must use the tokenizer that matches the model.
**3. Example to quote:** > "I loaded GPT-2's tokenizer, turned 'Hello world' into `[15496, 995]`, and decoded it back to the same text."

**Seen in:** `main.py`, whole file.

## Byte-Pair Encoding (BPE)  `(concept)`
**1. Definition:** A subword tokenization algorithm that starts from bytes or characters and repeatedly merges the most frequent adjacent pair found in training text into a new vocabulary entry.
**2. In plain words:** Common sequences like "the" become single tokens because they were merged many times; rare words stay split into smaller known pieces. This gives a vocabulary of manageable size (50,257 for GPT-2) that can still spell any text, even new words, by falling back to bytes.
**3. Example to quote:** > "GPT-2 uses BPE, so 'transformers' is two pieces, `transform` and `ers`, but 'cat' is a single token."

**Seen in:** `main.py` docstring.

## Vocabulary  `(concept)`
**1. Definition:** The fixed set of tokens a tokenizer can produce, each mapped to a unique integer ID.
**2. In plain words:** Think of it as the model's dictionary. It is decided before training and never grows afterwards. If the vocabulary is mostly English, other languages are split into many small pieces.
**3. Example to quote:** > "GPT-2's vocabulary has 50,257 entries, which is also the size of the logits vector later."

**Seen in:** `main.py`, `tok.vocab_size`.

## AutoTokenizer  `(third-party: transformers)`
**1. Definition:** A Hugging Face class whose `from_pretrained` method inspects a model name and returns the correct tokenizer class for it.
**2. In plain words:** You do not need to know whether the model wants a BPE or WordPiece tokenizer; `AutoTokenizer` works that out. That keeps your code the same if you swap models.
**3. Example to quote:** > "I used `AutoTokenizer` so changing `'distilgpt2'` to `'gpt2'` needs no other code change."
```python
tok = AutoTokenizer.from_pretrained("distilgpt2")
```
**Seen in:** `main.py`, top.

## from_pretrained  `(third-party: transformers)`
**1. Definition:** A class method that downloads (once) and caches the files of a pretrained tokenizer or model from the Hugging Face Hub, or loads them from a local folder, and returns the ready object.
**2. In plain words:** The first run needs internet and downloads about 1 MB of vocabulary files; later runs use the cache in `~/.cache/huggingface`. The argument is a model name or a local path. A wrong name gives an `OSError`.
**3. Example to quote:** > "`from_pretrained('distilgpt2')` downloaded the vocabulary and merge rules on the first run and loaded them from cache after that."

**Seen in:** `main.py`, top.

## vocab_size  `(third-party: transformers)`
**1. Definition:** An attribute of a tokenizer giving the number of distinct tokens in its base vocabulary.
**2. In plain words:** It tells you how many different token IDs exist. For GPT-2 it is 50257, so IDs run from 0 to 50256.
**3. Example to quote:** > "I printed `tok.vocab_size` and got 50257."

**Seen in:** `main.py`, line after loading.

## tok.encode  `(third-party: transformers)`
**1. Definition:** Method that converts a string into a list of integer token IDs (adding special tokens if the tokenizer is configured to).
**2. In plain words:** It is the quickest way from text to numbers. It returns a plain Python list of ints, not a tensor. The result for the same text is always the same.
**3. Example to quote:** > "`tok.encode('Hello world')` returned `[15496, 995]`."
```python
ids = tok.encode("Hello world")
```
**Seen in:** `main.py`, section 1.

## Token ID  `(concept)`
**1. Definition:** The integer index of a token in the vocabulary.
**2. In plain words:** The model never sees the string; it sees the ID. IDs are arbitrary labels: 995 is not "bigger" than 15496 in any meaningful sense. IDs are used to look up the token's embedding vector.
**3. Example to quote:** > "' world' with a leading space is ID 995, but 'world' without it is ID 6894."

**Seen in:** `main.py`, section 3.

## repr and !r  `(Python built-in)`
**1. Definition:** `repr(x)` returns a string showing an object the way you would write it in code; in an f-string `{x!r}` applies `repr`.
**2. In plain words:** It shows quotes and escape characters, which makes invisible differences visible, such as a trailing space or newline. That is useful when debugging tokenizer output.
**3. Example to quote:** > "I printed `repr(tok.decode(ids))` to see that the decoded text had quotes and no hidden spaces."
```python
repr("a\n")  # "'a\\n'"
```
**Seen in:** `main.py`, sections 1, 2 and 4.

## convert_ids_to_tokens  `(third-party: transformers)`
**1. Definition:** Tokenizer method that maps a list of token IDs to their string pieces.
**2. In plain words:** It lets you see the actual sub-word pieces behind the numbers, including the special `Ġ` marker for spaces.
**3. Example to quote:** > "`convert_ids_to_tokens([15496, 995])` gave `['Hello', 'Ġworld']`."

**Seen in:** `main.py`, section 1.

## Ġ (space marker)  `(concept)`
**1. Definition:** In GPT-2's byte-level BPE, the character `Ġ` is the displayed form of a leading space attached to the following token piece.
**2. In plain words:** GPT-2 treats " world" (with space) and "world" as different tokens. The tokenizer shows the space as `Ġ`. It explains many prompt bugs where an extra or missing space changes the output.
**3. Example to quote:** > "`Ġworld` means the token is ' world' with a space before it, which has a different ID from 'world'."

**Seen in:** `main.py`, comment and output.

## tok.decode  `(third-party: transformers)`
**1. Definition:** Method that converts a list (or tensor) of token IDs back into a string.
**2. In plain words:** It is the inverse of `encode`. It expects IDs, not a string; passing `"hello"` fails. After generation it turns the model's output IDs into readable text.
**3. Example to quote:** > "Decoding `[15496, 995]` returns 'Hello world', so encoding then decoding round-trips."
```python
tok.decode([15496, 995])  # 'Hello world'
```
**Seen in:** `main.py`, section 2.

## assert  `(Python built-in)`
**1. Definition:** A statement `assert condition` that raises `AssertionError` if the condition is false.
**2. In plain words:** It is a self-check: "I claim this is true, stop loudly if not." The code asserts that decoding gives back the original text, so any tokenizer round-trip problem is caught immediately.
**3. Example to quote:** > "I wrote `assert tok.decode(ids) == text` to prove the tokenizer round-trips."

**Seen in:** `main.py`, section 2.

## tok.tokenize  `(third-party: transformers)`
**1. Definition:** Method that splits text into a list of string pieces without converting them to IDs.
**2. In plain words:** It is handy for seeing where a word is split. For 'unbelievably' it shows four pieces; counting them tells you the token cost.
**3. Example to quote:** > "`len(tok.tokenize('2024-10-19'))` is 6, so dates are expensive in tokens."

**Seen in:** `main.py`, section 4.

## Rare words and non-English text  `(concept)`
**1. Definition:** Words or scripts that are infrequent in the tokenizer's training data are split into many small pieces or raw byte fragments.
**2. In plain words:** The vocabulary was built mostly from English. Hindi "नमस्ते" (6 characters) used 12 tokens. That means higher cost, less text fitting in the context window and often lower quality for those languages.
**3. Example to quote:** > "A Hindi greeting took 12 tokens, so the same context window holds far less Hindi than English."

**Seen in:** `WALKTHROUGH.md`, execution trace.

## Special token (EOS)  `(concept)`
**1. Definition:** A reserved token that is not ordinary text, such as end-of-sequence `<|endoftext|>` (ID 50256 in GPT-2), used to mark boundaries.
**2. In plain words:** It is a marker like a full stop for the whole document. During generation, when the model emits it, generation can stop. Tokenizers expose it as `eos_token` and `eos_token_id`.
**3. Example to quote:** > "GPT-2's end-of-text token is `<|endoftext|>`, ID 50256."

**Seen in:** `main.py`, section 5.

## pad_token  `(third-party: transformers)`
**1. Definition:** The token used to fill shorter sequences to equal length when batching; GPT-2 defines none by default.
**2. In plain words:** Batches need rectangular tensors, so short sentences are padded. GPT-2 lacks a pad token, so a common workaround is `tok.pad_token = tok.eos_token`. The attention mask then tells the model to ignore the padding.
**3. Example to quote:** > "I set `tok.pad_token = tok.eos_token` so I could batch 'Hi' with a longer sentence."

**Seen in:** `main.py`, sections 5 and 6.

## Calling the tokenizer (tok(...))  `(third-party: transformers)`
**1. Definition:** Calling the tokenizer object like a function runs the full pipeline: tokenize, convert to IDs, add special tokens, optionally pad/truncate, and return a dict-like `BatchEncoding`.
**2. In plain words:** It is the "give the model exactly what it needs" call. Compared with `encode` it also builds the `attention_mask` and can return tensors for a whole batch of texts.
**3. Example to quote:** > "`tok(['Hi', 'Hello there, how are you?'], padding=True, return_tensors='pt')` gives me `input_ids` and `attention_mask` ready for the model."

**Seen in:** `main.py`, section 6.

## padding=True  `(third-party: transformers)`
**1. Definition:** Argument telling the tokenizer to pad shorter sequences in a batch to the length of the longest one.
**2. In plain words:** Without it you get lists of different lengths that cannot form one tensor. The padding adds pad tokens, which is why the attention mask is needed.
**3. Example to quote:** > "'Hi' was padded with six `50256` tokens to match the seven-token sentence."

**Seen in:** `main.py`, section 6.

## return_tensors="pt"  `(third-party: transformers)`
**1. Definition:** Argument asking the tokenizer to return PyTorch tensors; `"pt"` stands for PyTorch.
**2. In plain words:** Models expect tensors, not Python lists. This saves you converting. Other options are `"np"` for numpy and `"tf"` for TensorFlow.
**3. Example to quote:** > "I passed `return_tensors='pt'` so `input_ids` came back as a tensor I can feed to the model."

**Seen in:** `main.py`, section 6.

## attention_mask  `(concept)`
**1. Definition:** A tensor of 1s and 0s aligned with `input_ids` where 1 marks a real token and 0 marks padding to be ignored by attention.
**2. In plain words:** It tells the model which positions are real words. Without it the model would treat padding as content. In the output, 'Hi' has one 1 followed by six 0s.
**3. Example to quote:** > "The mask row `[1, 0, 0, 0, 0, 0, 0]` says only the first token of 'Hi' is real."

**Seen in:** `main.py`, section 6.

## input_ids  `(concept)`
**1. Definition:** The tensor of integer token IDs, one row per sequence, that is passed to a model.
**2. In plain words:** It is the numeric version of your text batch. Shape is `(batch_size, sequence_length)`. It is the main input to every Hugging Face language model.
**3. Example to quote:** > "My `input_ids` had two rows of length seven: the padded 'Hi' and the full sentence."

**Seen in:** `main.py`, section 6.

## Dictionary lookup  `(syntax)`
**1. Definition:** Accessing a value in a dict (or dict-like object) with square brackets and a key: `d["key"]`.
**2. In plain words:** The tokenizer's result behaves like a dictionary, so `batch["input_ids"]` pulls out one entry. A missing key raises `KeyError`.
**3. Example to quote:** > "I read `batch['attention_mask']` the same way I would read a dict."
```python
{"a": 1}["a"]  # 1
```
**Seen in:** `main.py`, section 6.

## OSError (model not found)  `(concept)`
**1. Definition:** The exception raised by `from_pretrained` when the given name is neither a local folder nor a valid Hub repository (or the network is unavailable).
**2. In plain words:** Read the message: it tells you the identifier is wrong or you are offline. Check spelling, then your connection, then the local cache.
**3. Example to quote:** > "Loading `'distilgpt3'` raised `OSError: not a valid model identifier`, so I checked the spelling."

**Seen in:** `main.py`, BREAK IT comment.

## Character-level vs word-level tokenization  `(concept)`
**1. Definition:** Two simple ways to cut text: one token per character, or one token per whitespace-separated word.
**2. In plain words:** Characters give a tiny vocabulary but very long sequences (45 characters vs 9 BPE tokens in `main.py`), which is slow because attention cost grows with length squared. Words give short sequences but need an endless vocabulary. BPE sits between the two.
**3. Example to quote:** > "Character-level makes sequences too long, word-level makes the vocabulary infinite, so GPT-2 uses BPE in between."

**Seen in:** `main.py` section 0.

## Out-of-vocabulary (OOV)  `(concept)`
**1. Definition:** A word that is not in the model's fixed vocabulary, so it has no id.
**2. In plain words:** With a word-level vocabulary every new word, typo or name breaks the pipeline (3 of 6 words were unknown in the demo). A BPE tokenizer can always fall back to smaller pieces or bytes, so nothing is ever unknown.
**3. Example to quote:** > "Sub-word tokenization removes the out-of-vocabulary problem because any text can be spelled from known pieces."

**Seen in:** `main.py` section 0.
