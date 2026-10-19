# Step 01 — Dataset preparation

## Goal
Turn a raw text file into tensors of token IDs (plus labels) that a causal language model can train on.
This is the "dataset prep and preprocessing" bullet of the syllabus, and it is where most real-world training bugs start.

## Run it
```bash
cd 06_llm_training_and_fast_inference/01_dataset_preparation
python main.py
```

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `Path(__file__).resolve().parents[1]` | the folder one level above this file, as an absolute path | `Path("/a/b/c.py").parents[1]` -> `/a` |
| list comprehension `[f(x) for x in xs if cond]` | build a list in one line | `[s.strip() for s in ["a ", " "] if s.strip()]` -> `['a']` |
| `x[: n * k]` | slice: first `n*k` items | `[1,2,3,4,5][:4]` -> `[1,2,3,4]` |
| `x[:-1]`, `x[1:]` | all but last / all but first | `[1,2,3][1:]` -> `[2,3]` |
| `//` | integer (floor) division | `25 // 4` -> `6` |
| `for x in xs: ys.extend(...)` | `extend` appends every item of another list | `a=[1]; a.extend([2,3])` -> `[1,2,3]` |
| `iter(loader)` / `next(...)` | start an iterator / pull its first item | `next(iter([5,6]))` -> `5` |
| f-string `!r` | show the repr (quotes, `\n` visible) | `f"{'a\n'!r}"` -> `'a\\n'` |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `pathlib.Path` | built-in | file paths as objects; `.read_text()` reads a file | `encoding` | `str` |
| `AutoTokenizer.from_pretrained(name)` | transformers | downloads/loads the matching tokenizer | model name or folder | tokenizer object |
| `tokenizer.encode(text)` | transformers | text -> list of token IDs | `add_special_tokens` | `list[int]` |
| `tokenizer.decode(ids)` | transformers | IDs -> text | `skip_special_tokens` | `str` |
| `tokenizer(texts, padding=True, return_tensors="pt")` | transformers | batch encode with padding | `padding`, `truncation`, `return_tensors` | dict with `input_ids`, `attention_mask` |
| `torch.tensor(list)` | torch | list -> tensor | `dtype` | tensor |
| `tensor.view(a, b)` | torch | reshape without copying (sizes must multiply to the element count) | new shape | tensor |
| `tensor.masked_fill(mask, value)` | torch | replace entries where mask is True | mask, value | tensor |
| `TensorDataset(*tensors)` | torch | pairs tensors row by row | tensors | dataset |
| `DataLoader(ds, batch_size, shuffle)` | torch | serves mini-batches | `batch_size`, `shuffle` | iterable of batches |

## Execution trace (real run)
- **Input:** `data/tiny_corpus.txt`, 39 support lines.
- `tokenizer.encode(lines[0])` -> `[51, 9715, 25, 1374, 466, 314, 13259, 616, 9206, 287] ...` which decodes to `Ticket: How do I reset my password in`.
- Glue + chunk: 1206 tokens in total -> `1206 // 48 = 25` blocks -> tensor shape `[25, 48]` (the 6 leftover tokens are dropped).
- Label shift table: position 0 sees `'T'` and must predict `'icket'`; position 1 sees `'icket'` and must predict `':'`.
- Padding demo: 3 lines pad to length 33; the shortest line's mask ends `... 1, 1, 0, 0, 0, 0` and its labels end `[6338, 13, -100, -100, -100, -100]`.
- **Output:** one batch `input_ids` of shape `[4, 48]`.

## Experiments (change ONE thing)
1. `BLOCK_SIZE = 16` -> many more, shorter blocks (about 75). The model sees less context per example.
2. `shuffle=False` -> the first batch is identical on every run; with `True` it changes.
3. Print `blocks[0]` decoded: notice a block can start in the middle of a line. Is that a problem? (It is what real pre-training does.)

## Debugging practice
Change `.view(n_blocks, BLOCK_SIZE)` to `.view(n_blocks, BLOCK_SIZE + 1)`:
```
RuntimeError: shape '[25, 49]' is invalid for input of size 1200
```
Read it bottom-up: the last line names the error and the numbers. `1200 != 25*49`, so the shape and data disagree. Fix the shape, not the data.

## Rebuild checklist
- [ ] read file, split lines, drop blanks
- [ ] tokenize each line and append EOS
- [ ] cut to a multiple of `BLOCK_SIZE` and `.view(...)`
- [ ] wrap in `TensorDataset` + `DataLoader`

## Say it aloud
"A language model trains on next-token prediction, so I tokenise the text, join documents with an end-of-text token, and cut it into fixed-size blocks. The labels are the same IDs because Hugging Face shifts them by one inside the model. When sequences differ in length I pad and use an attention mask, and set padded labels to -100 so they do not affect the loss. Trade-off: chopping at fixed blocks can split a document mid-sentence, and padding wastes compute."
