# Step 02 - Tokenizer

## Goal
Turn text into the integer IDs a model consumes (and back), and understand why words split into pieces.

## Run it
```bash
.venv/bin/python 04_llm_fundamentals/02_tokenizer/main.py
```

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
Real output:
```
vocabulary size: 50257

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
'नमस्ते'                       -> 12 tokens: [...byte fragments...]

end-of-text token: <|endoftext|> id = 50256
input_ids:
 tensor([[17250, 50256, 50256, 50256, 50256, 50256, 50256],
        [15496,   612,    11,   703,   389,   345,    30]])
attention_mask:
 tensor([[1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1]])
```
- **BPE intuition:** start with single bytes; repeatedly merge the most common neighbouring pair seen in training text. Common words ("cat") end up as one token; rare ones are assembled from smaller known pieces.
- **`Ġ`** is how GPT-2 displays a leading space. `world` (6894) and ` world` (995) are different tokens - a classic source of prompt bugs.
- **Hindi costs 12 tokens for 6 characters**: GPT-2's vocabulary was built mostly on English, so other scripts fall back to raw bytes. Consequence: higher cost, shorter effective context, often worse quality.
- **Padding + attention mask:** to batch sentences of different length, short ones are padded; the mask (0 = ignore) tells attention to skip them. GPT-2 has no pad token, so we reuse `<|endoftext|>`.

## Experiments
- Tokenize your own name and a sentence in your mother tongue; compute tokens per character.
- Compare `len(tok.encode("the " * 50))` and `len(tok.encode("qzxv " * 50))`. (Repeated common word vs gibberish.)
- Switch the model name to `"gpt2"`: IDs are identical (same tokenizer). Try `"bert-base-uncased"`: different vocabulary (WordPiece), different IDs.

## Debugging practice
`tok.decode("hello")` -> the error says decode expects ids/tensors, not a `str`. Fix: `tok.decode(tok.encode("hello"))`.
`AutoTokenizer.from_pretrained("distilgpt3")` -> `OSError: distilgpt3 is not a local folder and is not a valid model identifier...`. Read the sentence: it tells you the name is wrong (or you're offline). Check spelling, then network, then `~/.cache/huggingface`.

## Rebuild checklist
- [ ] Load a tokenizer, encode a sentence, decode it and assert equality.
- [ ] Print the pieces and explain every `Ġ`.
- [ ] Batch two sentences with padding and explain the attention mask.

## Say it aloud
"A tokenizer splits text into sub-word tokens from a fixed vocabulary and maps them to integer IDs, because the model only handles numbers. GPT-2 uses byte-pair encoding, which keeps common words whole and splits rare words. I can show `Hello world` becomes two IDs and decode it back. Trade-off: non-English text and numbers split into many tokens, which costs context and money, and models are sensitive to spaces and spelling."
