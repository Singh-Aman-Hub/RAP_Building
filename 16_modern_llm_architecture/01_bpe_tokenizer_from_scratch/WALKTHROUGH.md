# Step 01 — Byte-level BPE tokenizer from scratch

## The story so far
This is the first step of module 16 (reading position: after modules 03, 15, 04 and 05, before 21). [Module 04, step 01](../../04_llm_fundamentals/01_tokenizer/WALKTHROUGH.md) is the intro to tokenizers (read it first if tokens are still fuzzy); it showed what a tokenizer does (text in, integer ids out) using a ready-made one. Here we build the algorithm behind it, because every other choice in a modern LLM (embedding table size, context length in tokens, cost per request) is counted in tokens.

## The pain point
A model needs a fixed list of symbols. Words fail (a new word, a typo, a Hindi word are "unknown"). Single characters are universal but make sequences very long, and attention cost grows with the square of length. We want a vocabulary that is small enough to train, covers any text with no "unknown" symbol, and keeps common words as one piece.

## The spark
**Byte-pair encoding (BPE)**: start from the 256 possible bytes (so any text in any language can be represented), then repeatedly find the most frequent adjacent pair of symbols and glue it into a new symbol. After N merges you have a vocabulary of 256 + N tokens. Frequent words become one token, rare words fall back to smaller pieces, and nothing is ever unknown.

## Goal
Train 300 merges on a small text, write `encode` and `decode`, prove the round trip is lossless (including accents and emoji), compare the token counts with the real `gpt2` and `Qwen2.5` tokenizers, and look at special tokens, chat templates and the cost of a bigger vocabulary.

## The theory (read this before the interview)
**Training BPE** (the learning phase, done once on a large corpus):
```
ids = bytes of the corpus                       # values 0..255
repeat N times:
    count every adjacent pair (a, b) in ids
    pick the most frequent pair
    create a new token id = 256 + step; record  (a, b) -> new id
    replace every occurrence of (a, b) in ids by the new id
```
**Encoding** new text: turn it into bytes, then apply the learned merges in the order they were learned (earliest merge first). **Decoding**: look up the bytes of each id, join them, decode as UTF-8. Decoding can produce an invalid UTF-8 sequence if you cut a multi-byte character in the middle (a model may emit half a character), so use `errors="replace"`.

**Real tokenizers add a pre-tokenizer.** GPT-2 first splits text with a regular expression (words, numbers, punctuation, spaces) so merges never cross word boundaries (it avoids tokens like `dog.` and ` the` being merged in strange ways). Modern tokenizers also split digits (Qwen2.5 puts every digit in its own token; see the real run) so that arithmetic is more regular. Our toy has no pre-tokenizer: it merges across spaces and punctuation, which is why it produced `'he '` and `'s '`.

**Why byte-level.** The base alphabet is 256 bytes, never Unicode characters. Any string, including emoji and code, is representable with no `<unk>` token. A rare character costs several byte tokens instead of failing.

**Vocabulary size trade-off**
| Bigger vocabulary | Smaller vocabulary |
|---|---|
| fewer tokens per text: shorter sequences, cheaper attention, more text fits in the context window | more tokens per text, longer sequences, slower and more expensive |
| bigger embedding table (and output head if untied): `vocab * d_model` parameters | small tables |
| rare tokens get few training examples (under-trained "glitch" tokens) | every token is seen often |
| better for many languages if the merges include them | non-English text splits into many byte pieces |
Real sizes: GPT-2 50,257 · Llama 3 128,256 · Qwen2.5 151,665 real tokens (embedding table padded to 151,936) · Gemma 3 262,208 (embedding rows). The trend since 2023 is towards 100k–260k because multilingual coverage and shorter sequences pay for the bigger table, especially in large models where the table is a small fraction of all parameters. In a 0.5B model the table is 27.6% of everything (step 08).

**Other algorithms.** *WordPiece* (BERT) picks merges by likelihood and marks continuations with `##`. *Unigram LM* (SentencePiece, used by T5 and many multilingual models) starts with a huge vocabulary and prunes it, so a sentence has several possible segmentations (it can sample them, a regulariser). *SentencePiece* is a library that works on raw text and treats the space as a normal symbol `▁`, with either BPE or Unigram inside. *tiktoken* is OpenAI's fast BPE implementation (GPT-3.5/4 and `o200k` vocabularies); it is an encoder library, not a new algorithm. Hugging Face `tokenizers` is the Rust library behind `AutoTokenizer`. Byte-level BPE (GPT-2, Llama 3, Qwen, DeepSeek) is the 2026 default for LLMs.

**Special tokens.** Reserved ids that never come from the merge process: `<|endoftext|>` (end of document), and for chat models `<|im_start|>` / `<|im_end|>` (Qwen, ChatML format) or `<|begin_of_text|>` and header tokens (Llama 3). The **chat template** is a Jinja string stored in `tokenizer_config.json` that turns a list of messages into one string with these tokens. Training and inference must use the same template; a wrong template silently degrades answers. Special tokens must also be filtered from user input, otherwise a user can type `<|im_start|>system` (prompt injection at the tokenizer level).

**Token-level quirks you can be asked about.** The model never sees characters, so counting letters, reversing strings and arithmetic are harder (`strawberry` may be two tokens). Leading spaces are part of the token (`Ġthe` vs `the`). Different tokenizers give different token counts for the same text, so "context window 32k" is not comparable across models.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| BPE | byte-pair encoding, byte-level BPE | BPE is the algorithm; byte-level is the variant starting from 256 bytes | character-level BPE can still produce `<unk>`; byte-level cannot |
| Vocabulary | vocab, token list, `vocab_size` | Almost | `vocab_size` in a config is the embedding table rows, which can be larger than the number of real tokens (padding for GPU efficiency) |
| Tokenizer | encoder/decoder, `AutoTokenizer`, tiktoken, SentencePiece | No | tiktoken and SentencePiece are *libraries* that run a tokenizer; the tokenizer itself is merges + vocab + rules |
| Special tokens | control tokens, added tokens | Yes | reserved ids such as `<|im_start|>`; never produced by merges |
| Chat template | prompt format, ChatML (Qwen), Llama-3 header format | Mostly | the template is code that inserts the special tokens; ChatML is one specific format |
| WordPiece vs BPE vs Unigram | subword tokenizers | No | BPE merges by frequency, WordPiece by likelihood, Unigram prunes a big vocabulary |
| Token | subword, piece | Yes | a token is an id; a word may be several |

## Run it
```bash
cd 16_modern_llm_architecture/01_bpe_tokenizer_from_scratch
HF_HUB_OFFLINE=1 python main.py     # ~3 s on CPU; uses cached gpt2 and Qwen2.5-0.5B tokenizers
```

## Code, step by step
### 1. Imports and offline mode  (`main.py` lines 14–19)
```python
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer
```
- **In simple words:** load the helpers, and tell Hugging Face not to use the internet.
- **Exactly:** `os.environ.setdefault("HF_HUB_OFFLINE", "1")` makes `from_pretrained` read only the local cache (no network calls, so the run is offline and reproducible). `Counter` counts pairs, `Path` locates the corpus.
- **Why it is here:** without the offline flag a missing network can make loading hang or retry.
### 2. Load a small corpus and turn it into bytes  (`main.py` lines 22–25)
```python
corpus_path = Path(__file__).resolve().parent / "corpus_sample.txt"      # 20,000 characters of this repo's markdown
text = corpus_path.read_text(encoding="utf-8")
data = list(text.encode("utf-8"))                      # bytes 0..255: the base vocabulary of byte-level BPE
print(f"corpus: {len(text):,} characters = {len(data):,} bytes")
```
- **In simple words:** we take 20,000 characters of text (a sample of this repo's own markdown, saved as `corpus_sample.txt`) and break it into raw bytes, like turning a book into a long string of small numbers.
- **Exactly:** `text.encode("utf-8")` gives bytes; `list(...)` gives ints 0..255. 20,000 characters became 20,001 bytes because one character needs more than one byte in UTF-8. `Path(__file__).resolve().parent` is the step folder, so the script works from any working directory.
- **Why it is here:** BPE trains on a corpus. Starting from bytes is what makes the tokenizer able to represent any input later.
### 3. Count adjacent pairs  (`main.py` lines 29–30)
```python
def pair_counts(ids):
    return Counter(zip(ids, ids[1:]))                  # {(a, b): how often b follows a}
```
- **In simple words:** look at every neighbouring pair of symbols and count how often each pair appears.
- **Exactly:** `zip(ids, ids[1:])` yields `(ids[0], ids[1]), (ids[1], ids[2]), ...`; `Counter` maps each pair tuple to its count. Complexity O(n) per call.
- **Why it is here:** the most frequent pair is the next merge. This is the whole "learning" signal of BPE.
### 4. Replace every occurrence of a pair by one new id  (`main.py` lines 33–42)
```python
def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:   # found the pair: replace two ids with one
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
```
- **In simple words:** walk along the list; when you see the chosen pair, write the new symbol instead of the two old ones.
- **Exactly:** a manual scan with index `i`; on a match append `new_id` and skip 2, otherwise copy one element. The output is shorter by the number of matches. It is non-overlapping and left to right, so `a a a` with pair `(a, a)` becomes `[new, a]`.
- **Why it is here:** applying a merge is what shortens the sequence (compression) and creates the multi-byte tokens.
### 5. Training loop: 300 merges  (`main.py` lines 46–59)
```python
NUM_MERGES = 300                                       # final vocab = 256 bytes + 300 merges = 556
merges = {}                                            # (a, b) -> new id, in the order learned
vocab = {i: bytes([i]) for i in range(256)}            # id -> the bytes it stands for
ids = data[:]
for step in range(NUM_MERGES):
    counts = pair_counts(ids)
    pair = counts.most_common(1)[0][0]                 # most frequent adjacent pair
    new_id = 256 + step
    ids = merge(ids, pair, new_id)
    merges[pair] = new_id
    vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]    # a merged token is the concatenation of its parts
    if step < 8 or step in (99, 299):
        print(f"merge {step + 1:>3}: {pair} -> {new_id}  = {vocab[new_id]!r}  (count {counts[pair]})")
print(f"training bytes {len(data):,} -> {len(ids):,} tokens  (compression {len(data) / len(ids):.2f} bytes/token)")
```
- **In simple words:** repeat "find the most common pair, glue it" 300 times, remembering each gluing rule and what text each new token stands for.
- **Exactly:** `merges` maps `(a, b) -> new_id` in learning order (dict order is insertion order, and `encode` relies on the ids growing). `vocab[new_id] = vocab[a] + vocab[b]` concatenates bytes, so id 256 = `b'  '` (two spaces, count 591: this text is markdown with indentation), id 257 is `--`, and then English bigrams (`in`, `s `, `er`, `e `, `en`). Merge 7 is `(256, 256)` = four spaces: merges build on earlier merges. Merge 100 is `'-> '` (count 30) and merge 300 `et` (count 10). The corpus shrank from 20,001 bytes to 8,960 tokens: **2.23 bytes per token**.
- **Why it is here:** this is the training. More merges means a larger vocabulary and fewer tokens per text; later merges are rarer and help less (diminishing returns).
### 6. Encode: apply learned merges in order  (`main.py` lines 63–72)
```python
def encode(s):
    ids = list(s.encode("utf-8"))
    while len(ids) >= 2:
        counts = pair_counts(ids)
        # apply the EARLIEST-learned merge present (lowest new id): the same order as in training
        pair = min(counts, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break                                      # nothing left to merge
        ids = merge(ids, pair, merges[pair])
    return ids
```
- **In simple words:** to tokenize new text, start from its bytes and keep applying the earliest-learned rule that is present until no rule applies.
- **Exactly:** `min(counts, key=lambda p: merges.get(p, inf))` picks, among the pairs present, the one with the smallest new id (learned earliest). If that pair is not in `merges` (all are `inf`) we stop. Cost is O(n * merges applied); real implementations use a priority queue and the pre-tokenizer to be fast.
- **Why it is here:** applying merges in training order is what makes encoding deterministic and consistent with training. Applying them in another order gives different tokens.
### 7. Decode: ids back to bytes back to text  (`main.py` lines 75–76)
```python
def decode(ids):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")
```
- **In simple words:** look up the bytes of each token, stick them together, read them as UTF-8 text.
- **Exactly:** `vocab[i]` is a `bytes` object; `b"".join` concatenates; `.decode("utf-8", errors="replace")` turns invalid byte sequences into the replacement character instead of raising.
- **Why it is here:** the model generates ids; decode is how they become text. `errors="replace"` matters because a generation can stop in the middle of a multi-byte character.
### 8. Round-trip test on text, accents and emoji  (`main.py` lines 79–85)
```python
sample = "The attention layer reads the tokens."
enc = encode(sample)
print("encode:", enc)
print("pieces:", [vocab[i].decode("utf-8", "replace") for i in enc])
print("round trip ok:", decode(enc) == sample)
emoji = "naïve café 🙂"
print("unicode round trip ok:", decode(encode(emoji)) == emoji, "| bytes:", len(emoji.encode()), "| tokens:", len(encode(emoji)))
```
- **In simple words:** encode and decode and check we get the same string back, even for `naïve café 🙂`.
- **Exactly:** "The attention layer reads the tokens." (37 bytes) became 15 tokens: `T | he  | attention | ' ' | l | a | y | er  | re | ad | s  | the  | token | s | .`. The word `attention` is one token because it is very frequent in this corpus, while `layer` is split into pieces. The 17-byte accent/emoji string became 16 tokens (our merges never saw these bytes, so they stay almost raw) and still round-trips: `True`.
- **Why it is here:** lossless round trip is the correctness test of any tokenizer. Byte-level BPE guarantees it by construction.
### 9. Load the real gpt2 and Qwen tokenizers  (`main.py` lines 88–90)
```python
gpt2 = AutoTokenizer.from_pretrained("gpt2")
qwen = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
print(f"\nvocab sizes: mine {len(vocab)} | gpt2 {len(gpt2)} | qwen2.5 {len(qwen)}")
```
- **In simple words:** load two production tokenizers from the local cache (no internet).
- **Exactly:** `AutoTokenizer.from_pretrained(name)` reads `tokenizer.json` / `vocab.json` + `merges.txt` and returns a fast Rust-backed tokenizer. `len(tok)` counts real tokens: mine 556, gpt2 50,257, qwen2.5 151,665.
- **Why it is here:** to compare our toy against real ones on the same strings.
### 10. Compare token counts across tokenizers  (`main.py` lines 91–102)
```python
samples = {
    "english prose": "The attention layer reads the tokens and decides what matters.",
    "python code": "def add(a, b):\n    return a + b\n",
    "numbers": "Total: 1234567.89 rupees",
    "hindi": "मुझे चाय पसंद है",
    "emoji": "great job 🙂🙂",
}
print(f"{'text':<15}{'bytes':>6}{'mine':>6}{'gpt2':>6}{'qwen':>6}")
for name, s in samples.items():
    print(f"{name:<15}{len(s.encode()):>6}{len(encode(s)):>6}{len(gpt2.encode(s)):>6}{len(qwen.encode(s)):>6}")
print("gpt2 pieces for numbers:", gpt2.tokenize(samples["numbers"]))
print("qwen pieces for numbers:", qwen.tokenize(samples["numbers"]))
```
- **In simple words:** the same sentence costs a different number of tokens in each tokenizer; code, numbers, Hindi and emoji show the biggest gaps.
- **Exactly:** from the run: English prose 62 bytes -> mine 27, gpt2 11, qwen 11. Hindi 42 bytes -> mine 42 (no merges for these bytes), gpt2 25, qwen 15. Emoji pair 18 bytes -> mine 15, gpt2 5, qwen 4. Numbers: gpt2 groups digits (`Ġ123`, `45`, `67`) while Qwen puts every digit alone (`1`,`2`,`3`...), giving 15 tokens vs 9, on purpose: digit-by-digit tokenization makes arithmetic patterns learnable.
- **Why it is here:** it shows that token counts depend on the tokenizer and that a larger multilingual vocabulary (Qwen) compresses Hindi much better (15 vs 25 tokens).
### 11. Special tokens and the chat template  (`main.py` lines 105–108)
```python
print("\ngpt2 special:", gpt2.all_special_tokens, "| qwen special:", qwen.all_special_tokens)
msgs = [{"role": "user", "content": "Hi"}]
print("qwen chat template ->", repr(qwen.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)))
print("'<|im_start|>' is ONE id:", qwen.convert_tokens_to_ids("<|im_start|>"))
```
- **In simple words:** chat models use reserved marker tokens to say who is talking; the template turns messages into one string with those markers.
- **Exactly:** gpt2 has one special token, `<|endoftext|>`. Qwen has ChatML tokens plus vision/box markers. `apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)` returned `'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n'` (the default system message comes from the template). `convert_tokens_to_ids("<|im_start|>")` is a single id, 151644.
- **Why it is here:** an instruction-tuned model only works if prompts use the exact template it was trained with; `add_generation_prompt=True` appends the assistant header so the model continues as the assistant.
### 12. What a bigger vocabulary costs in parameters  (`main.py` lines 111–112)
```python
for name, v, d in [("gpt2", 50257, 768), ("qwen2.5-0.5b", 151936, 896)]:
    print(f"{name}: embedding table {v:,} x {d} = {v * d / 1e6:.1f}M params")
```
- **In simple words:** every token owns a row of numbers; more tokens means a bigger table.
- **Exactly:** gpt2: 50,257 x 768 = 38.6M parameters. Qwen2.5-0.5B: 151,936 x 896 = 136.1M parameters, more than a quarter of the whole 0.5B model (step 08 shows 27.6%).
- **Why it is here:** the vocabulary trade-off is not free; small models pay a large relative price, which is why tying input and output embeddings (step 07) is common there.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `s.encode("utf-8")` | string to bytes | `"é".encode()` is 2 bytes |
| `list(bytes_obj)` | bytes to list of ints 0..255 | `list(b"ab")` -> `[97, 98]` |
| `zip(ids, ids[1:])` | adjacent pairs | `[1,2,3]` -> `(1,2),(2,3)` |
| `Counter(...).most_common(1)[0][0]` | the most frequent key | pair with highest count |
| `min(d, key=lambda p: merges.get(p, inf))` | smallest by custom key | earliest-learned pair |
| `b"".join(list_of_bytes)` | concatenate bytes objects | joins token bytes |
| `f"{x!r}"` | repr in an f-string | shows `b'in'` with quotes |
| `{i: bytes([i]) for i in range(256)}` | dict comprehension | id -> its single byte |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `collections.Counter` | built-in | counts hashable items | iterable | dict-like of counts |
| `pathlib.Path.read_text` | built-in | read a file | `encoding` | `str` |
| `AutoTokenizer.from_pretrained` | transformers | load a tokenizer from the cache or hub | model name | tokenizer object |
| `tok.encode(s)` / `tok.tokenize(s)` | transformers | ids / string pieces | text | `list[int]` / `list[str]` |
| `tok.apply_chat_template` | transformers | messages to a prompt string/ids | `tokenize`, `add_generation_prompt` | `str` or ids |
| `tok.convert_tokens_to_ids` | transformers | token string to id | token | `int` |

## Execution trace (real run)
```text
corpus: 20,000 characters = 20,001 bytes
merge   1: (32, 32) -> 256  = b'  '  (count 591)
merge   2: (45, 45) -> 257  = b'--'  (count 365)
merge   3: (105, 110) -> 258  = b'in'  (count 364)
merge   4: (115, 32) -> 259  = b's '  (count 278)
merge   5: (101, 114) -> 260  = b'er'  (count 259)
merge   6: (101, 32) -> 261  = b'e '  (count 254)
merge   7: (256, 256) -> 262  = b'    '  (count 251)
merge   8: (101, 110) -> 263  = b'en'  (count 248)
merge 100: (314, 32) -> 355  = b'-> '  (count 30)
merge 300: (101, 116) -> 555  = b'et'  (count 10)
training bytes 20,001 -> 8,960 tokens  (compression 2.23 bytes/token)
encode: [84, 278, 459, 32, 108, 97, 121, 319, 281, 408, 259, 418, 438, 115, 46]
pieces: ['T', 'he ', 'attention', ' ', 'l', 'a', 'y', 'er ', 're', 'ad', 's ', 'the ', 'token', 's', '.']
round trip ok: True
unicode round trip ok: True | bytes: 17 | tokens: 16

vocab sizes: mine 556 | gpt2 50257 | qwen2.5 151665
text            bytes  mine  gpt2  qwen
english prose      62    27    11    11
python code        32    22    16    12
numbers            24    21     9    15
hindi              42    42    25    15
emoji              18    15     5     4
gpt2 pieces for numbers: ['Total', ':', 'Ġ123', '45', '67', '.', '89', 'Ġru', 'pees']
qwen pieces for numbers: ['Total', ':', 'Ġ', '1', '2', '3', '4', '5', '6', '7', '.', '8', '9', 'Ġru', 'pees']

gpt2 special: ['<|endoftext|>'] | qwen special: ['<|endoftext|>', '<|im_start|>', '<|im_end|>', '<|object_ref_start|>', '<|object_ref_end|>', '<|box_start|>', '<|box_end|>', '<|quad_start|>', '<|quad_end|>', '<|vision_start|>', '<|vision_end|>', '<|vision_pad|>', '<|image_pad|>', '<|video_pad|>']
qwen chat template -> '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n'
'<|im_start|>' is ONE id: 151644
gpt2: embedding table 50,257 x 768 = 38.6M params
qwen2.5-0.5b: embedding table 151,936 x 896 = 136.1M params
```
Reading it: the first merges are the most frequent byte pairs (two spaces, dashes, then English bigrams); compression is 2.23 bytes/token for 300 merges (a real 50k-128k vocabulary reaches roughly 4 bytes/token on English). The hand-made tokenizer is far worse than gpt2 on every row (27 vs 11 tokens on prose) because it saw only 20 KB; the *algorithm* is the same, the training data is not. Qwen's single-digit tokens are visible in `qwen pieces for numbers`.

## Experiments (change ONE thing)
Every row below was run on a temporary copy of `main.py` (the real file is untouched). Only the named line changed.

| Change | What you will see | Why |
|---|---|---|
| `NUM_MERGES = 50` (measured) | compression 1.40 bytes/token; the sample sentence splits into 23 pieces (`'T', 'he ', 'at', 't', 'en', 't', 'ion', ...`); "english prose" costs 38 tokens (was 27) | Only the most frequent bigrams exist so far, so most words are still spelled byte by byte. |
| `NUM_MERGES = 1000` (measured) | compression 3.56 bytes/token; the sentence becomes 13 pieces including whole words (`'attention'`, `'token'`); "english prose" drops to 21 tokens | More merges build longer pieces, so the same text needs fewer tokens. The 20 KB corpus is small, so these long pieces are partly memorised from the repo's own text. |
| `NUM_MERGES = 1000`: look at the hindi row (measured) | `hindi` stays at 42 tokens for 42 bytes, no matter how many merges (gpt2 25, qwen 15) | The corpus contains no Devanagari, so no merge ever involves those bytes. A vocabulary only compresses the languages it was trained on. |
| in `encode`, take the LATEST learned merge instead of the earliest (`max(..., key=merges.get(p, -1))`) (measured) | round trip still `True`, but the sample sentence needs 20 pieces (was 15 with the correct order) and "english prose" costs 34 tokens (was 27) | Merge order is part of the tokenizer. Applying merges in a different order than training still gives *valid* tokens, just worse ones. The tests that only check round trip will not catch it. |
| compare the `numbers` row, gpt2 vs qwen (measured, no code change) | gpt2 `['Ġ123', '45', '67', '.', '89']` = 9 tokens, qwen one digit per token = 15 tokens | Qwen deliberately splits numbers into single digits so arithmetic is learnable, at the cost of more tokens. |

## Debugging practice
**BREAK IT drill 1: decode forgets the vocabulary lookup.** In `decode`:

```diff
-    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")
+    return b"".join(i for i in ids).decode("utf-8", errors="replace")
```

Symptom (real run of the broken copy), raised at the first `decode` call:

```text
TypeError: sequence item 0: expected a bytes-like object, int found
```

Root cause: a token id is just an integer; the bytes it stands for live in `vocab`. `b"".join` wants bytes objects. Fix: restore `vocab[i]`.

**BREAK IT drill 2: encode applies merges in the wrong order (a silent bug).** In `encode`:

```diff
-        pair = min(counts, key=lambda p: merges.get(p, float("inf")))
+        pair = max(counts, key=lambda p: merges.get(p, -1))
```

Symptom: no error and `round trip ok: True`, `unicode round trip ok: True`. The only sign is a worse split, `pieces: ['T', 'he ', 'at', 'te', 'n', 't', 'ion', ' ', 'l', 'a', 'y', 'er ', 're', 'ad', 's ', 'th', 'e', ' token', 's', '.']` (20 pieces instead of 15) and "english prose" at 34 tokens instead of 27.

Root cause: training learned `merges` in a specific order, and later merges were built *from* earlier ones. Replaying the latest merge first uses up bytes that earlier merges needed. Fix: always pick the lowest-id (earliest) merge present. Lesson: round trip proves correctness of decode, not quality of encode; compare token counts against a reference too.

## Common mistakes
- Training the tokenizer on a different corpus than the model will read (here: English markdown, then Hindi text) and wondering why the token count explodes.
- Believing round trip equality means the tokenizer is good. Bad merge order, or no merges at all, still round-trips.
- Applying merges in the wrong order in `encode` (see drill 2); the order is part of the saved tokenizer.
- Forgetting `errors="replace"` when decoding a partial id sequence: a cut in the middle of a multi-byte character is invalid UTF-8.
- Comparing token counts across tokenizers as if they were costs in the same unit: Qwen's digit-per-token numbers look worse but are chosen on purpose.
- Changing a tokenizer after the model is trained: every id would point to a different embedding row.

## Rebuild checklist
- [ ] `pair_counts` with `Counter(zip(ids, ids[1:]))`
- [ ] `merge(ids, pair, new_id)` with non-overlapping left-to-right scan
- [ ] training loop that records `merges` and `vocab[new] = vocab[a] + vocab[b]`
- [ ] `encode` applying the earliest-learned merge first; `decode` with `errors="replace"`
- [ ] a round-trip assertion on text with accents and emoji

## The new problem (cliffhanger)
We now have tokens, and a model turns each token id into a vector. But a bag of vectors has no order: "dog bites man" and "man bites dog" would look identical to attention. The model must be told *where* each token is. GPT-2 added a learned position vector; modern models rotate the query and key. Next: [Step 02 — positional encodings](../02_positional_encodings/WALKTHROUGH.md).

> **Say it aloud (interview box):** "Byte-level BPE starts from the 256 bytes, so there is no unknown token. Training repeatedly merges the most frequent adjacent pair into a new token until the vocabulary hits the target size. A bigger vocabulary shortens sequences and helps multilingual text but enlarges the embedding table and leaves rare tokens under-trained. Chat models add special tokens and a chat template that must match training."
