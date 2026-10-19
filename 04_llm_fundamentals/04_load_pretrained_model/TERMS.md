# Terms — Load a Pretrained Model

Folder: `04_llm_fundamentals/04_load_pretrained_model` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [04 LLM fundamentals README](../README.md)

**Terms in this folder:** [Pretrained model](#pretrained-model) · [AutoConfig](#autoconfig) · [Config vs weights](#config-vs-weights) · [n_layer](#n_layer) · [n_embd (hidden size)](#n_embd-hidden-size) · [n_head](#n_head) · [Context window (n_positions)](#context-window-n_positions) · [vocab_size (config)](#vocab_size-config) · [AutoModelForCausalLM.from_pretrained](#automodelforcausallmfrom_pretrained) · [model.eval()](#modeleval) · [Dropout](#dropout) · [Parameter](#parameter) · [model.parameters()](#modelparameters) · [numel()](#numel) · [Generator expression](#generator-expression) · [sum](#sum) · [f-string number formatting](#f-string-number-formatting) · [next](#next) · [dtype](#dtype) · [element_size()](#element_size) · [float32, float16, int8, int4](#float32-float16-int8-int4) · [Parameters x bytes = memory](#parameters-x-bytes--memory) · [named_parameters and enumerate](#named_parameters-and-enumerate) · [break](#break) · [Distilled model](#distilled-model) · [c_attn](#c_attn) · [model.half()](#modelhalf) · [OSError when loading](#oserror-when-loading)

---

## Pretrained model  `(concept)`
**1. Definition:** A model whose weights were already learned by training on a large dataset, so it can be used or fine-tuned without starting from random numbers.
**2. In plain words:** Someone else paid for the expensive training; you download the saved numbers and put them into the right architecture. Loading means reading those numbers into memory. You get two things: a config (the blueprint) and weights (the learned numbers).
**3. Example to quote:** > "I did not train distilgpt2; I loaded its pretrained weights with `from_pretrained`."

**Seen in:** `main.py`, docstring.

## AutoConfig  `(third-party: transformers)`
**1. Definition:** A Hugging Face class that loads a model's configuration (architecture sizes and settings) without loading the weights.
**2. In plain words:** The config is a small file saying how many layers, how wide, how many heads. You can inspect a model before downloading hundreds of MB of weights.
**3. Example to quote:** > "I printed `cfg.n_layer` from `AutoConfig` and saw distilgpt2 has 6 layers, with no big download."
```python
cfg = AutoConfig.from_pretrained("distilgpt2")
```
**Seen in:** `main.py`, section 1.

## Config vs weights  `(concept)`
**1. Definition:** The config describes the architecture (shapes, counts); the weights are the trained numbers that fill that architecture.
**2. In plain words:** Config is the building plan, weights are the furnished rooms. Two models can share a config but have different weights.
**3. Example to quote:** > "gpt2 and distilgpt2 share hidden size 768, but gpt2 has 12 layers and distilgpt2 has 6."

**Seen in:** `main.py`, docstring.

## n_layer  `(concept)`
**1. Definition:** GPT-2 config field: the number of stacked transformer blocks.
**2. In plain words:** More layers mean deeper processing and more parameters. distilgpt2 has 6, gpt2 has 12.
**3. Example to quote:** > "`cfg.n_layer` printed 6, so distilgpt2 stacks six blocks."

**Seen in:** `main.py`, section 1.

## n_embd (hidden size)  `(concept)`
**1. Definition:** GPT-2 config field for the size of each token vector flowing through the model (the hidden size, called `d_model` in step 01).
**2. In plain words:** It is how many numbers describe one token internally. 768 for both GPT-2 sizes.
**3. Example to quote:** > "Each token is a 768-number vector inside distilgpt2."

**Seen in:** `main.py`, section 1.

## n_head  `(concept)`
**1. Definition:** The number of attention heads: parallel attention computations in each layer, each working on a slice of the hidden vector.
**2. In plain words:** Each head can learn a different relationship (for example grammar versus meaning). distilgpt2 has 12 heads of 64 numbers each (12 x 64 = 768).
**3. Example to quote:** > "`cfg.n_head` is 12: twelve attention patterns computed in parallel per layer."

**Seen in:** `main.py`, section 1.

## Context window (n_positions)  `(concept)`
**1. Definition:** The maximum number of tokens a model can process at once; in GPT-2 it is `n_positions` = 1024.
**2. In plain words:** It is the model's working memory. Prompt plus generated text must fit in it. For GPT-2 it is fixed by the size of the position-embedding table.
**3. Example to quote:** > "distilgpt2 can see at most 1024 tokens, so a longer prompt must be truncated."

**Seen in:** `main.py`, section 1.

## vocab_size (config)  `(concept)`
**1. Definition:** The number of distinct token IDs the model understands, 50257 for GPT-2; it sets the rows of the embedding table and the width of the output logits.
**2. In plain words:** It must match the tokenizer. If the numbers differ, token IDs would point to the wrong or missing rows.
**3. Example to quote:** > "Config vocab size 50257 matches `tok.vocab_size`, and the embedding table has 50257 rows."

**Seen in:** `main.py`, section 1.

## AutoModelForCausalLM.from_pretrained  `(third-party: transformers)`
**1. Definition:** Builds the right causal language model architecture for a model name and loads its trained weights from the Hub or local cache.
**2. In plain words:** It is the "load the model" call. Useful arguments include `torch_dtype` (precision) and `device_map` (where to place it). The first run downloads about 330 MB for distilgpt2, then it is cached.
**3. Example to quote:** > "`AutoModelForCausalLM.from_pretrained('distilgpt2')` gave me a transformer with a next-token prediction head."
```python
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
```
**Seen in:** `main.py`, section 2.

## model.eval()  `(third-party: torch)`
**1. Definition:** Puts a model in evaluation mode, which disables training-only behaviour such as dropout.
**2. In plain words:** For predictions you want repeatable behaviour, not random neurons switched off. Forgetting `.eval()` causes slightly different outputs each call.
**3. Example to quote:** > "I call `model.eval()` before inference so dropout does not randomise my outputs."

**Seen in:** `main.py`, section 2.

## Dropout  `(concept)`
**1. Definition:** A regularisation technique that randomly zeroes a fraction of activations during training to reduce overfitting.
**2. In plain words:** By randomly "turning off" neurons while training, the model cannot rely on any single one. At inference we want all of them on, which is why `eval()` disables it.
**3. Example to quote:** > "Dropout is used only during training; in eval mode it does nothing."

**Seen in:** `main.py`, comment on `model.eval()`.

## Parameter  `(concept)`
**1. Definition:** A learned number in a model (an element of a weight or bias tensor) adjusted during training.
**2. In plain words:** "81.9 million parameters" means 81.9 million numbers were learned. Model size is measured in parameters and drives memory use and quality.
**3. Example to quote:** > "distilgpt2 has 81,912,576 parameters."

**Seen in:** `main.py`, section 3.

## model.parameters()  `(third-party: torch)`
**1. Definition:** Method returning an iterator over all learnable weight tensors of a module.
**2. In plain words:** It lets you loop over every weight tensor, to count them, freeze them, or give them to an optimizer. `named_parameters()` also gives each tensor's name.
**3. Example to quote:** > "I counted parameters by looping over `model.parameters()`."

**Seen in:** `main.py`, `count_params`.

## numel()  `(third-party: torch)`
**1. Definition:** Tensor method returning the total number of elements in the tensor.
**2. In plain words:** A weight of shape (768, 768) has 589,824 elements. Summing `numel()` across all tensors gives the parameter count.
**3. Example to quote:** > "`p.numel()` for a (50257, 768) embedding is about 38.6 million."

**Seen in:** `main.py`, `count_params`.

## Generator expression  `(syntax)`
**1. Definition:** An expression like `(p.numel() for p in m.parameters())` that produces values lazily, one at a time, without building a list.
**2. In plain words:** It is a compact loop that feeds values into a function such as `sum`. Without brackets inside a function call, it still works: `sum(x*x for x in [1,2,3])` is 14.
**3. Example to quote:** > "`sum(p.numel() for p in m.parameters())` counts all parameters in one line without making a list."
```python
sum(x * x for x in [1, 2, 3])  # 14
```
**Seen in:** `main.py`, `count_params`.

## sum  `(Python built-in)`
**1. Definition:** Built-in function that adds all items of an iterable and returns the total.
**2. In plain words:** Here it adds up the element counts of all weight tensors.
**3. Example to quote:** > "`sum` turned many per-tensor counts into the single 81.9 M total."

**Seen in:** `main.py`, `count_params`.

## f-string number formatting  `(syntax)`
**1. Definition:** Format specifiers inside f-strings: `{n:,}` adds thousands separators, `{x:.2f}` shows 2 decimals, `{name:18}` pads to width 18.
**2. In plain words:** They make output readable: `81,912,576` rather than `81912576`, and `0.33` rather than `0.327650304`.
**3. Example to quote:** > "I printed the memory as `{n * nbytes / 1e9:.2f} GB` to get two decimals."
```python
f"{81912576:,}"  # '81,912,576'
```
**Seen in:** `main.py`, sections 3 and 4.

## next  `(Python built-in)`
**1. Definition:** Built-in that returns the next item from an iterator.
**2. In plain words:** `next(model.parameters())` grabs just the first weight tensor so we can inspect its dtype and size.
**3. Example to quote:** > "I used `next(model.parameters())` to peek at the first weight tensor."

**Seen in:** `main.py`, section 4.

## dtype  `(third-party: torch)`
**1. Definition:** The numeric data type of a tensor's elements, such as `torch.float32`.
**2. In plain words:** It sets how many bytes each number takes and how precise it is. Models load as float32 by default.
**3. Example to quote:** > "The weights' dtype printed `torch.float32`, 4 bytes each."

**Seen in:** `main.py`, section 4.

## element_size()  `(third-party: torch)`
**1. Definition:** Tensor method returning the number of bytes of one element.
**2. In plain words:** Multiply it by the parameter count and you have the weight memory.
**3. Example to quote:** > "`first_param.element_size()` returned 4 for float32."

**Seen in:** `main.py`, section 4.

## float32, float16, int8, int4  `(concept)`
**1. Definition:** Numeric formats using 32, 16, 8 and 4 bits (4, 2, 1 and 0.5 bytes) per number.
**2. In plain words:** Fewer bits mean less memory and often faster inference but less precision. float16 halves memory; int8 and int4 are quantized formats covered in module 05.
**3. Example to quote:** > "distilgpt2 is about 0.33 GB in float32 but roughly 0.04 GB in int4."

**Seen in:** `main.py`, section 4.

## Parameters x bytes = memory  `(concept)`
**1. Definition:** Weight memory equals the number of parameters multiplied by bytes per parameter.
**2. In plain words:** A 7-billion-parameter model in float16 needs 7e9 x 2 = 14 GB just for weights; in int4 about 3.5 GB. Real use is higher because of activations, the KV cache and overhead. This is a classic interview calculation.
**3. Example to quote:** > "7B at float16 is 14 GB, so it will not fit on an 8 GB GPU without quantization."

**Seen in:** `WALKTHROUGH.md`, execution trace.

## named_parameters and enumerate  `(Python built-in)`
**1. Definition:** `model.named_parameters()` yields `(name, tensor)` pairs; `enumerate(iterable)` adds a counter, giving `(i, item)`.
**2. In plain words:** The code combines them: `for i, (name, p) in enumerate(...)` unpacks the counter and the pair, so it can print the first eight parameter names and stop.
**3. Example to quote:** > "I used `enumerate` so I could stop after eight parameters while printing names and shapes."
```python
for i, c in enumerate("ab"):
    print(i, c)
```
**Seen in:** `main.py`, section 5.

## break  `(syntax)`
**1. Definition:** Statement that exits the nearest enclosing loop immediately.
**2. In plain words:** Here it stops the printing loop after eight parameters instead of listing all of them.
**3. Example to quote:** > "I `break` when `i >= 8` so the output stays short."

**Seen in:** `main.py`, section 5.

## Distilled model  `(concept)`
**1. Definition:** A smaller "student" model trained to imitate a larger "teacher" model (knowledge distillation).
**2. In plain words:** distilgpt2 has half the layers and about two-thirds the parameters of gpt2: faster and cheaper but a bit less capable. It is why we use it for quick experiments.
**3. Example to quote:** > "distilgpt2 has 6 layers versus gpt2's 12 and runs faster at slightly lower quality."

**Seen in:** `main.py`, section 6.

## c_attn  `(concept)`
**1. Definition:** GPT-2's attention projection layer that produces Query, Key and Value together in one matrix of shape (768, 2304).
**2. In plain words:** 2304 = 3 x 768: one multiplication yields all three, which is the same Q, K, V idea as step 01 packed for speed.
**3. Example to quote:** > "`transformer.h.0.attn.c_attn.weight` is (768, 2304), the Q, K and V projections packed together."

**Seen in:** `WALKTHROUGH.md`, execution trace.

## model.half()  `(third-party: torch)`
**1. Definition:** Converts a model's floating-point parameters to float16.
**2. In plain words:** It halves weight memory. On a CPU some operations are slow or unsupported in float16, while GPUs handle it well.
**3. Example to quote:** > "After `model.half()` the dtype changed to float16 and the memory estimate halved."

**Seen in:** `main.py`, TRY IT comment.

## OSError when loading  `(concept)`
**1. Definition:** The error raised by `from_pretrained` when the model cannot be found or the Hub cannot be reached and no cached copy exists.
**2. In plain words:** Read which sentence you get: "couldn't connect" points to the network or a missing cache; "not a valid model identifier" points to a spelling mistake. Different causes need different fixes.
**3. Example to quote:** > "Offline with a model I never downloaded gave a connection `OSError`, so I checked `~/.cache/huggingface`."

**Seen in:** `main.py`, BREAK IT comments.
