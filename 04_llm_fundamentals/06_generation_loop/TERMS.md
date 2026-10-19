# Terms — The Generation Loop

Folder: `04_llm_fundamentals/06_generation_loop` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [04 LLM fundamentals README](../README.md)

**Terms in this folder:** [Autoregressive generation](#autoregressive-generation) · [time (module)](#time-module) · [time.perf_counter](#timeperf_counter) · [Function with default argument](#function-with-default-argument) · [max_new_tokens](#max_new_tokens) · [for _ in range(n)](#for-_-in-rangen) · [argmax on last position](#argmax-on-last-position) · [torch.cat](#torchcat) · [.view()](#view) · [eos_token_id and early stop](#eos_token_id-and-early-stop) · [break](#break) · [model.generate](#modelgenerate) · [do_sample](#do_sample) · [pad_token_id](#pad_token_id) · [torch.equal](#torchequal) · [KV cache](#kv-cache) · [use_cache](#use_cache) · [Slicing off the prompt](#slicing-off-the-prompt) · [Repetition under greedy decoding](#repetition-under-greedy-decoding) · [Latency proportional to output length](#latency-proportional-to-output-length) · [Print debugging of shapes](#print-debugging-of-shapes) · [RuntimeError (tensor dimensions)](#runtimeerror-tensor-dimensions)

---

## Autoregressive generation  `(concept)`
**1. Definition:** Producing a sequence one element at a time, where each new element is predicted from all previously produced elements, including the model's own earlier outputs.
**2. In plain words:** The model writes a token, adds it to the text, and reads the longer text again to write the next one. That is how every GPT-style model creates paragraphs from one-token predictions.
**3. Example to quote:** > "My loop added ' a', then ' new', then ' way', each time feeding the longer text back in."

**Seen in:** `main.py`, docstring.

## time (module)  `(standard library)`
**1. Definition:** The standard-library module providing time-related functions such as clocks and sleeping.
**2. In plain words:** The script uses it to measure how long generation takes, so you can compare methods.
**3. Example to quote:** > "I imported `time` to compare timing with and without the KV cache."

**Seen in:** `main.py`, imports.

## time.perf_counter  `(standard library)`
**1. Definition:** Returns the value of a high-resolution performance counter in seconds; only differences between two readings are meaningful.
**2. In plain words:** Take a reading before and after, subtract, and you have elapsed time. It is better for timing code than `time.time()`.
**3. Example to quote:** > "`t0 = time.perf_counter()` ... `time.perf_counter() - t0` measured 0.81 s for cached generation."
```python
t0 = time.perf_counter()
elapsed = time.perf_counter() - t0
```
**Seen in:** `main.py`, sections 1 to 3.

## Function with default argument  `(syntax)`
**1. Definition:** In `def f(a, max_new_tokens=15):` a parameter with `=value` uses that value when the caller does not supply one.
**2. In plain words:** It makes common cases short: `my_generate(ids)` generates 15 tokens, while `my_generate(ids, max_new_tokens=50)` overrides it.
**3. Example to quote:** > "`my_generate(ids)` uses the default of 15 new tokens; I can pass a bigger number if I want."

**Seen in:** `main.py`, `def my_generate`.

## max_new_tokens  `(third-party: transformers)`
**1. Definition:** Generation argument that sets the maximum number of tokens to add after the prompt.
**2. In plain words:** It caps the length of the answer. Generation can stop sooner if the end-of-text token appears. Longer outputs cost more time because each token needs a forward pass.
**3. Example to quote:** > "I set `max_new_tokens=15`, so at most 15 tokens are added to my 3-token prompt."

**Seen in:** `main.py`, sections 1 and 2.

## for _ in range(n)  `(syntax)`
**1. Definition:** A loop that repeats `n` times; `_` is a conventional name for a loop variable you do not use.
**2. In plain words:** The generation loop only needs to run a fixed number of times, not know the counter value, so the underscore signals "ignore this".
**3. Example to quote:** > "`for _ in range(max_new_tokens)` runs one prediction per new token."

**Seen in:** `main.py`, `my_generate`.

## argmax on last position  `(concept)`
**1. Definition:** Taking the index of the highest logit at the final sequence position, `logits[0, -1].argmax()`, to pick the greedy next token.
**2. In plain words:** Only the last position predicts the new token, and argmax picks the highest-scoring one. It returns a zero-dimensional tensor, a single number.
**3. Example to quote:** > "`logits[0, -1].argmax()` gave the ID of the best next token."

**Seen in:** `main.py`, `my_generate`.

## torch.cat  `(third-party: torch)`
**1. Definition:** Concatenates a list of tensors along an existing dimension.
**2. In plain words:** It glues the new token onto the end of the sequence. `dim=1` means along the sequence (column) axis, turning shape `(1, 3)` plus `(1, 1)` into `(1, 4)`.
**3. Example to quote:** > "`torch.cat([input_ids, next_id.view(1, 1)], dim=1)` appends the predicted token to my text."
```python
torch.cat([torch.ones(1, 3), torch.ones(1, 1)], dim=1).shape  # (1, 4)
```
**Seen in:** `main.py`, `my_generate` and section 4.

## .view()  `(third-party: torch)`
**1. Definition:** Returns a tensor with the same data but a different shape, without copying.
**2. In plain words:** `next_id` is a single number (0 dimensions); `view(1, 1)` makes it a 1x1 matrix so it can be glued onto a `(1, seq)` tensor. Without it, `torch.cat` complains about mismatched dimensions.
**3. Example to quote:** > "I used `next_id.view(1, 1)` to match dimensions before `torch.cat`."

**Seen in:** `main.py`, `my_generate`.

## eos_token_id and early stop  `(concept)`
**1. Definition:** `tok.eos_token_id` is the ID of the end-of-text token (50256); generation loops stop when the model emits it.
**2. In plain words:** It is the model's way of saying "I am done". Checking for it lets the loop finish before `max_new_tokens`.
**3. Example to quote:** > "My loop breaks if `next_id.item() == tok.eos_token_id`."

**Seen in:** `main.py`, `my_generate`.

## break  `(syntax)`
**1. Definition:** Statement that exits the innermost loop immediately.
**2. In plain words:** Used here to end generation once the end-of-text token appears.
**3. Example to quote:** > "The `break` stops the loop at the end-of-text token."

**Seen in:** `main.py`, `my_generate`.

## model.generate  `(third-party: transformers)`
**1. Definition:** Hugging Face method that runs the full decoding loop (cache, sampling or greedy, stopping rules) and returns the prompt plus new token IDs.
**2. In plain words:** It is the production version of the loop you wrote by hand. Key arguments: `max_new_tokens`, `do_sample`, `temperature`, `top_k`, `top_p`, `repetition_penalty`, `use_cache`, `pad_token_id`. With greedy settings it produced exactly the same text as my loop.
**3. Example to quote:** > "`model.generate(ids, max_new_tokens=15, do_sample=False)` matched my hand-written loop token for token."
```python
out = model.generate(ids, max_new_tokens=15, do_sample=False)
```
**Seen in:** `main.py`, section 2.

## do_sample  `(third-party: transformers)`
**1. Definition:** Generation flag: `False` means greedy decoding (always the top token); `True` means sample from the probability distribution.
**2. In plain words:** With `False` you get the same output every time; with `True` it varies from run to run. Temperature, top-k and top-p only matter when it is `True`.
**3. Example to quote:** > "I used `do_sample=False` so the library result is comparable to my greedy loop."

**Seen in:** `main.py`, sections 2 and 3.

## pad_token_id  `(third-party: transformers)`
**1. Definition:** Generation argument telling the model which token ID to use for padding; GPT-2 has none, so people pass the end-of-text ID.
**2. In plain words:** It silences a warning. For a single unpadded prompt it makes no difference to the output.
**3. Example to quote:** > "I passed `pad_token_id=tok.eos_token_id` to stop the GPT-2 padding warning."

**Seen in:** `main.py`, `generate` calls.

## torch.equal  `(third-party: torch)`
**1. Definition:** Returns True if two tensors have the same shape and identical elements.
**2. In plain words:** Here it proves the hand-written loop and `generate` created exactly the same token IDs.
**3. Example to quote:** > "`torch.equal(mine, lib)` printed True."

**Seen in:** `main.py`, section 2.

## KV cache  `(concept)`
**1. Definition:** A store of the Key and Value tensors computed for earlier tokens so they are reused instead of recomputed at each generation step.
**2. In plain words:** Attention needs every earlier token's keys and values. Without a cache each step re-reads the whole text; with it each step processes only the one new token. The price is memory that grows with sequence length and batch size.
**3. Example to quote:** > "Generating 60 tokens took 1.40 s without the KV cache and 0.81 s with it."

**Seen in:** `main.py`, section 3.

## use_cache  `(third-party: transformers)`
**1. Definition:** Generation flag that enables or disables the KV cache.
**2. In plain words:** It is on by default. Turning it off is useful only to demonstrate the speed difference.
**3. Example to quote:** > "I compared `use_cache=False` with `use_cache=True` to measure the cache's effect."

**Seen in:** `main.py`, section 3.

## Slicing off the prompt  `(syntax)`
**1. Definition:** `out[0][ids.shape[1]:]` takes the items after the first N positions, dropping the prompt tokens.
**2. In plain words:** `generate` returns prompt plus new tokens. Slice from the prompt length onward when you want only the new text.
**3. Example to quote:** > "To print just the model's continuation I slice from `ids.shape[1]` onward."
```python
[10, 11, 12, 13][2:]  # [12, 13]
```
**Seen in:** `WALKTHROUGH.md`, syntax table.

## Repetition under greedy decoding  `(concept)`
**1. Definition:** The tendency of greedy decoding in small models to loop or drift into repeated text such as endless newlines.
**2. In plain words:** Always taking the top token can trap the model in a loop. That is why sampling and a repetition penalty exist (module 05).
**3. Example to quote:** > "My greedy output ended with `\n\n\n\n\n\n`, a sign of drifting into repetition."

**Seen in:** `WALKTHROUGH.md`, execution trace.

## Latency proportional to output length  `(concept)`
**1. Definition:** Because each new token needs a forward pass, generation time grows roughly linearly with the number of generated tokens.
**2. In plain words:** Asking for a 500-token answer costs roughly ten times a 50-token answer. This is why capping `max_new_tokens` matters for response time and cost.
**3. Example to quote:** > "Latency scales with output length because every token is its own forward pass."

**Seen in:** `WALKTHROUGH.md`.

## Print debugging of shapes  `(concept)`
**1. Definition:** Inserting `print(x.shape)` lines before a failing statement to see actual tensor dimensions.
**2. In plain words:** Most tensor errors are shape errors. Printing the shapes of the arguments right before the line that crashes usually reveals the mismatch.
**3. Example to quote:** > "I added `print(input_ids.shape, next_id.shape)` before `torch.cat` and saw `(1, 3)` against `()`."

**Seen in:** `WALKTHROUGH.md`, debugging section.

## RuntimeError (tensor dimensions)  `(concept)`
**1. Definition:** The PyTorch exception raised when an operation receives tensors with incompatible shapes, such as concatenating a 2D and a 0D tensor.
**2. In plain words:** The message names the shapes it cannot combine. Compare them with what you expected, then fix with `view` or `unsqueeze`.
**3. Example to quote:** > "Without `.view(1, 1)`, `torch.cat` raised `Tensors must have same number of dimensions: got 2 and 0`."

**Seen in:** `main.py`, BREAK IT comment.
