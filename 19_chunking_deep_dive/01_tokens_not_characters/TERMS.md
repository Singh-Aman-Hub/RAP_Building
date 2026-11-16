# Terms — Tokens, Not Characters

Folder: `19_chunking_deep_dive/01_tokens_not_characters` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [19 README](../README.md)

**Terms in this folder:** [Token (WordPiece, subword)](#token-wordpiece-subword) · [Characters per token](#characters-per-token) · [[UNK] token](#unk-token) · [max_seq_length and silent truncation](#max_seq_length-and-silent-truncation) · [Offset mapping](#offset-mapping) · [Token window with overlap](#token-window-with-overlap) · [AssertionError](#assertionerror) · [RuntimeError: size of tensor a must match tensor b](#runtimeerror-size-of-tensor-a-must-match-tensor-b)

---

## Token (WordPiece, subword)  `(concept)`
**1. Definition:** A token is the unit a tokenizer produces: a whole word or a piece of one. MiniLM's WordPiece vocabulary has about 30,000 pieces; unknown characters become [UNK].
**2. In plain words:** The model reads pieces, not letters. 'employee' is 1 token, 'Probezeit' is several.
**3. Example to quote:** > "Limits and prices are counted in tokens, so I count with the embedder's own tokenizer."
```python
ids = tok(text, add_special_tokens=False)['input_ids']
```
**Seen in:** main.py, count_tokens

## Characters per token  `(concept)`
**1. Definition:** len(text) / number of tokens. It depends on the content and the tokenizer: 5.4 for English prose, 1.7 for JSON, 1.6 for a URL with a hash, 1.0 for Chinese in MiniLM.
**2. In plain words:** Cheap text has a high ratio. The '4 characters per token' rule of thumb is only true for English prose.
**3. Example to quote:** > "512 characters is 110 tokens of prose but 300 tokens of JSON."
**Seen in:** main.py, section B

## [UNK] token  `(concept)`
**1. Definition:** The special token for input the vocabulary cannot represent. MiniLM gave 19 [UNK] for 24 Chinese characters.
**2. In plain words:** A black box in the vector: the model cannot see that text at all. Use a multilingual model.
**3. Example to quote:** > "If a language yields many [UNK], the embedder does not support it."
**Seen in:** main.py, section B

## max_seq_length and silent truncation  `(concept)`
**1. Definition:** The most tokens the embedder reads (256 in sentence-transformers for all-MiniLM-L6-v2; the architecture has 512 positions). Longer input is cut without an error.
**2. In plain words:** Text after the limit never influences the vector. A 481-token text had 225 tokens unread.
**3. Example to quote:** > "Silent truncation is the bug you only find by measuring."
```python
batch = tok(texts, truncation=True, max_length=256, padding=True)
```
**Seen in:** main.py, section D

## Offset mapping  `(library)`
**1. Definition:** A fast tokenizer can return (char_start, char_end) for every token with return_offsets_mapping=True; special tokens get (0, 0).
**2. In plain words:** It lets you cut by tokens and still return exact slices of the original text.
**3. Example to quote:** > "offset_mapping turns token indices back into characters."
```python
tok(text, return_offsets_mapping=True)['offset_mapping']
```
**Seen in:** common19.py token_spans; main.py fixed_token_windows

## Token window with overlap  `(concept)`
**1. Definition:** Windows of `size` tokens; the next starts `size - overlap` tokens later. Needs overlap < size.
**2. In plain words:** A sliding ruler measured in tokens.
**3. Example to quote:** > "Token windows make every chunk fit the model; sentence boundaries are ignored."
```python
for i in range(0, max(len(spans) - overlap, 1), size - overlap): ...
```
**Seen in:** main.py, fixed_token_windows

## AssertionError  `(error)`
**1. Definition:** Raised by `assert cond, message` when cond is false. Used here to reject overlap >= size.
**2. In plain words:** A guard that stops a zero or negative step before it loops forever.
**3. Example to quote:** > "Validate chunk parameters at the top of the function."
```python
assert 0 <= overlap < size
```
**Seen in:** main.py, BREAK IT

## RuntimeError: size of tensor a must match tensor b  `(error)`
**1. Definition:** BERT-style models have 512 learned position embeddings; an input of 964 tokens cannot be added to them.
**2. In plain words:** Truncation normally hides this error; without it the model refuses.
**3. Example to quote:** > "Past the position table it crashes; before that it truncates silently."
**Seen in:** main.py, section D
