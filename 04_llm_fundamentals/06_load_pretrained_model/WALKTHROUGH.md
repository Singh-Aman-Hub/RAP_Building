# Step 06 - Load a pretrained model

## The story so far
[Step 05](../05_transformer_block/WALKTHROUGH.md) built a transformer block by hand, but ours holds random numbers. Training a real stack means weeks of compute. Someone already did it, and published the result.

## The pain point
Training from scratch costs millions of dollars and a data centre. Even if you could, you would not know in advance whether the finished model fits in your GPU. Memory is the first wall you hit.

## The spark
Download the trained weights and pour them into the same architecture. Then use one line of arithmetic to predict the memory: **memory = parameters x bytes per number**. Counting parameters is also the quickest way to check that a real model has the structure we built in step 05.

## Goal
Load distilgpt2, inspect its architecture, count parameters and estimate memory in float32/float16/int8/int4.

## Run it
```bash
.venv/bin/python 04_llm_fundamentals/06_load_pretrained_model/main.py
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Parameters | weights, model size, "81.9M" | Roughly | Every learned number in the model; "weights" often includes biases and embeddings too |
| Config vs weights | architecture vs checkpoint | No | Config (small JSON) says how many layers and what width; weights (hundreds of MB) hold the learned numbers |
| `AutoModelForCausalLM` | `GPT2LMHeadModel`, causal LM head model | Auto picks the concrete class | A transformer plus a next-token prediction head |
| `from_pretrained` vs `from_config` | load trained vs random init | No | `from_pretrained` loads learned weights; `from_config` builds the same shape with random numbers |
| Checkpoint | model file, `.safetensors`, weights file | Roughly | The saved weights; safetensors is the safe, fast format that replaced pickle `.bin` |
| float32 / float16 / bfloat16 / int8 / int4 | precision, dtype, quantisation level | Different formats | 4, 2, 2, 1 and 0.5 bytes per weight; float16 and bfloat16 are both 2 bytes but with different range and precision |
| `n_embd` / `hidden_size` / `d_model` | model width | Yes | Different libraries, same number |
| `n_positions` / `max_position_embeddings` / context window / context length | maximum input length | Yes | Number of tokens the model can attend to at once (1,024 for GPT-2) |
| `model.eval()` | evaluation mode, inference mode | Not the same as `torch.no_grad()` | Switches dropout off; `no_grad` stops gradient tracking; use both at inference |

## Code, step by step

### 1. Module docstring  (`main.py` lines 1–14)

```python
"""
GOAL:   Load a pretrained language model, count its parameters, estimate its memory,
        and understand what is inside it.

RUN:    from this folder:  python main.py
        from repo root:    .venv/bin/python 04_llm_fundamentals/06_load_pretrained_model/main.py
EXPECTED OUTPUT (short): ~81.9 M parameters for distilgpt2, ~124 M for gpt2; ~0.33 GB in float32;
        the layer names of the model; embeddings vs the 6 blocks from step 05; a 7B-model memory table.
NEEDS:  torch + transformers. First run downloads distilgpt2 (~330 MB) and gpt2 (~500 MB).

"Pretrained" = someone already trained the weights on huge amounts of text.
Loading = reading those saved numbers into the right architecture.  You download
two things: the CONFIG (architecture sizes) and the WEIGHTS (the learned numbers).
"""
```

- **In simple words:** Says what "pretrained" means: someone already trained the weights; we only load them.
- **Exactly:** Docstring only. Two downloads: the config (architecture sizes) and the weights (learned numbers). Promised: 81.9M parameters for distilgpt2, about 0.33 GB in float32.
- **Why it is here:** Separating blueprint from contents explains why loading is quick and why shapes must match.

### 2. Imports  (`main.py` lines 16–17)

```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM  # Auto* picks the right class for the model name
```

- **In simple words:** Bring in PyTorch and Hugging Face's loaders.
- **Exactly:** `AutoConfig` reads the architecture file; `AutoModelForCausalLM` builds and fills the right model class from the config's `model_type`.
- **Why it is here:** `Auto*` classes mean you do not need to know the model's Python class name.

### 3. Read the config (the blueprint)  (`main.py` lines 19–27)

```python
# ---------------------------------------------------------------------------
# 1. Config: the architecture "blueprint" (tiny file, no weights)
# ---------------------------------------------------------------------------
cfg = AutoConfig.from_pretrained("distilgpt2")
print("layers (n_layer):         ", cfg.n_layer)          # how many transformer blocks stacked
print("hidden size (n_embd):     ", cfg.n_embd)           # numbers per token vector
print("attention heads (n_head): ", cfg.n_head)
print("context window (n_positions):", cfg.n_positions)   # max tokens the model can see at once
print("vocab size:               ", cfg.vocab_size)
```

- **In simple words:** Open the small file that describes the model without downloading the large weights.
- **Exactly:** Real output: `n_layer 6`, `n_embd 768`, `n_head 12`, `n_positions 1024`, `vocab_size 50257`. Head dimension is `768 / 12 = 64`. These five numbers determine every tensor shape in the model.
- **Why it is here:** Config numbers are the first thing to check when a model will not fit in memory or a context length is too short.

### 4. Load the weights and switch to eval mode  (`main.py` lines 29–33)

```python
# ---------------------------------------------------------------------------
# 2. Load the weights.  AutoModelForCausalLM = "a transformer with a next-token head".
# ---------------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
model.eval()   # eval mode: switches off dropout (random neuron dropping used only during training)
```

- **In simple words:** Build the model and fill it with the saved numbers; turn off training-only randomness.
- **Exactly:** `from_pretrained("distilgpt2")` reads `model.safetensors` from the cache (`HF_HUB_OFFLINE=1` forces the local cache) and returns a `GPT2LMHeadModel`. `model.eval()` disables dropout.
- **Why it is here:** With dropout on, inference would give different outputs on each call.

### 5. Count the parameters  (`main.py` lines 35–43)

```python
# ---------------------------------------------------------------------------
# 3. Count parameters = total number of learned numbers
# ---------------------------------------------------------------------------
def count_params(m):
    # m.parameters() yields every weight tensor; .numel() = how many numbers inside it
    return sum(p.numel() for p in m.parameters())   # generator expression inside sum()

n = count_params(model)
print(f"\nparameters: {n:,}  (= {n/1e6:.1f} million)")
```

- **In simple words:** Add up the size of every weight tensor to get the model's headline number.
- **Exactly:** `m.parameters()` yields each tensor; `p.numel()` is its element count; the generator sums them. Real output: `81,912,576` = 81.9 million.
- **Why it is here:** Parameter count drives memory, compute and cost, and is the first number to quote about any model.

### 6. Memory = parameters times bytes per number  (`main.py` lines 45–52)

```python
# ---------------------------------------------------------------------------
# 4. Memory: parameters x bytes per number.  float32 = 4 bytes, float16 = 2, int8 = 1.
# ---------------------------------------------------------------------------
first_param = next(model.parameters())
print("dtype of weights:", first_param.dtype, "-> bytes each:", first_param.element_size())
for name, nbytes in [("float32", 4), ("float16/bfloat16", 2), ("int8", 1), ("int4", 0.5)]:
    print(f"  {name:18} ~ {n * nbytes / 1e9:.2f} GB")
print("(real memory use is higher: activations, KV cache and framework overhead are extra)")
```

- **In simple words:** Convert the count into gigabytes for different number formats.
- **Exactly:** `next(model.parameters())` is the first tensor (`wte.weight`): dtype `float32`, `element_size() = 4` bytes. Real output: float32 `0.33 GB`, float16/bfloat16 `0.16 GB`, int8 `0.08 GB`, int4 `0.04 GB` (n times bytes / 1e9). Real use is higher: activations, KV cache and overhead.
- **Why it is here:** This arithmetic (params x bytes) decides which GPU a model fits on; it leads into quantisation (module 05).

### 7. Look inside: parameter names and shapes  (`main.py` lines 54–61)

```python
# ---------------------------------------------------------------------------
# 5. Look inside: parameters grouped by top-level block name
# ---------------------------------------------------------------------------
print("\nfirst 8 parameter names and shapes:")
for i, (name, p) in enumerate(model.named_parameters()):   # enumerate gives a counter i
    if i >= 8:
        break
    print(f"  {name:40} {tuple(p.shape)}")   # shapes: wte (vocab, d_model), wpe (n_positions, d_model), c_attn (d_model, 3*d_model) ...
```

- **In simple words:** Print the first 8 tensors to see how the model is organised.
- **Exactly:** `named_parameters()` yields `(name, tensor)`. Real output: `transformer.wte.weight (50257, 768)`, `transformer.wpe.weight (1024, 768)`, then block 0: `ln_1.weight/bias (768,)`, `attn.c_attn.weight (768, 2304)` (Q, K, V packed: 3 x 768), `attn.c_attn.bias (2304,)`, `attn.c_proj.weight (768, 768)`. GPT-2's `Conv1D` layers store weights as `(in, out)`.
- **Why it is here:** Knowing the names lets you freeze layers, apply LoRA to `c_attn`, or debug a mismatched checkpoint.

### 8. Where do the 81.9 million live?  (`main.py` lines 63–68)

```python
# Where do the 81.9 M live?  Compare with the block formula from step 05: 12*d^2 + 13*d per block.
emb_params = model.transformer.wte.weight.numel() + model.transformer.wpe.weight.numel()   # (50257*768) + (1024*768)
block_params = count_params(model.transformer.h[0])                                        # one block, shape-free count
print(f"\nembeddings (wte+wpe): {emb_params:,}  |  one block: {block_params:,}  |  6 blocks: {6 * block_params:,}")
print(f"formula 12*d^2+13*d with d={cfg.n_embd}: {12 * cfg.n_embd**2 + 13 * cfg.n_embd:,}  (matches one block)")
print(f"share: embeddings {emb_params / n:.0%}, blocks {6 * block_params / n:.0%}, final LayerNorm the rest")
```

- **In simple words:** Split the total into embeddings and blocks and check against the formula from step 05.
- **Exactly:** `wte + wpe = 50257*768 + 1024*768 = 39,383,808`. One block `7,087,872` which equals `12*768^2 + 13*768`. Six blocks `42,527,232`. The remainder `1,536` is the final LayerNorm (`2 x 768`). Shares: embeddings 48%, blocks 52%. The output head shares weights with `wte` (tied), so it adds no parameters.
- **Why it is here:** In small models embeddings are half of everything; in a 7B model they are a few percent. It also confirms your hand-built block matches the real one.

### 9. Memory for a 7-billion-parameter model  (`main.py` lines 70–74)

```python
# Memory arithmetic for a model you will meet in interviews: memory = params x bytes
seven_b = 7e9
print("\n7B model, weights only:")
for name, nbytes in [("float32", 4), ("float16", 2), ("int8", 1), ("int4", 0.5)]:
    print(f"  {name:8} {seven_b * nbytes / 1e9:5.1f} GB")
```

- **In simple words:** The same arithmetic for the size of model you will be asked about in interviews.
- **Exactly:** `7e9 * bytes / 1e9`: float32 `28.0 GB`, float16 `14.0 GB`, int8 `7.0 GB`, int4 `3.5 GB` (weights only).
- **Why it is here:** Explains why 7B models need a 16 GB GPU in half precision and why 4-bit quantisation lets them run on a laptop.

### 10. Compare with gpt2 using config only  (`main.py` lines 76–80)

```python
# ---------------------------------------------------------------------------
# 6. Same architecture family, bigger: compare with gpt2 (config only - no big download)
# ---------------------------------------------------------------------------
cfg2 = AutoConfig.from_pretrained("gpt2")
print(f"\ndistilgpt2 layers={cfg.n_layer}  vs  gpt2 layers={cfg2.n_layer}  (same hidden size {cfg2.n_embd})")
```

- **In simple words:** Fetch the small config file of the bigger sibling.
- **Exactly:** `AutoConfig.from_pretrained("gpt2")` has `n_layer 12` while distilgpt2 has 6; hidden size 768 for both. No weights are downloaded.
- **Why it is here:** Shows how the same architecture scales by changing a few config numbers.

### 11. Experiments and break-it comments  (`main.py` lines 82–85)

```python
# TRY IT: load "gpt2" instead of "distilgpt2" and compare parameter counts.
# TRY IT: after loading, run model.half() then re-print memory (float16 halves the size).
# BREAK IT: from_pretrained("distilgpt2", torch_dtype="banana") -> read the error about the dtype.
# BREAK IT: turn off wifi and load a model you never downloaded -> OSError; note the cache folder it mentions (~/.cache/huggingface).
```

- **In simple words:** Try gpt2 instead, convert to float16, or break the dtype name or the network.
- **Exactly:** `model.half()` halves memory. A bogus dtype string raises `AttributeError: module 'torch' has no attribute 'banana'` (real run). In transformers 5.x the keyword is `dtype`; `torch_dtype` is still accepted. A model that is not in the cache while offline raises `OSError` that names the cache folder.
- **Why it is here:** Reading these errors once makes them recognisable later.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `sum(p.numel() for p in ...)` | generator expression: lazily produces values for `sum` | `sum(x*x for x in [1,2,3])` = 14 |
| `for i, (name, p) in enumerate(...)` | counter + unpack a tuple | `for i, c in enumerate("ab")` |
| `f"{n:,}"` | thousands separators | `f"{81912576:,}"` -> `81,912,576` |
| `f"{x:.2f}"` | 2 decimals | - |
| `next(iterator)` | take the first item | `next(iter([5, 6]))` = 5 |
| `break` | leave the loop early | - |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `AutoConfig.from_pretrained(name)` | transformers | downloads only the architecture description | name | config object |
| `AutoModelForCausalLM.from_pretrained(name)` | transformers | builds the architecture + loads trained weights | `torch_dtype`, `device_map` | `nn.Module` |
| `model.eval()` | torch | inference mode (turns off dropout) | - | the model |
| `model.parameters()` / `named_parameters()` | torch | iterate over weight tensors (with names) | - | iterator |
| `p.numel()` | torch | number of elements in a tensor | - | int |
| `p.element_size()` | torch | bytes per element (4 for float32) | - | int |
| `p.dtype`, `p.shape` | torch | data type / dimensions | - | attributes |

## Execution trace
```
layers (n_layer):          6
hidden size (n_embd):      768
attention heads (n_head):  12
context window (n_positions): 1024
vocab size:                50257

parameters: 81,912,576  (= 81.9 million)
dtype of weights: torch.float32 -> bytes each: 4
  float32            ~ 0.33 GB
  float16/bfloat16   ~ 0.16 GB
  int8               ~ 0.08 GB
  int4               ~ 0.04 GB

first 8 parameter names and shapes:
  transformer.wte.weight                   (50257, 768)     <- token embeddings (step 03)
  transformer.wpe.weight                   (1024, 768)      <- position embeddings (step 03)
  transformer.h.0.ln_1.weight              (768,)
  transformer.h.0.ln_1.bias                (768,)
  transformer.h.0.attn.c_attn.weight       (768, 2304)      <- Q, K, V projections packed (3 x 768)
  transformer.h.0.attn.c_attn.bias         (2304,)
  transformer.h.0.attn.c_proj.weight       (768, 768)
  transformer.h.0.attn.c_proj.bias         (768,)

embeddings (wte+wpe): 39,383,808  |  one block: 7,087,872  |  6 blocks: 42,527,232
formula 12*d^2+13*d with d=768: 7,087,872  (matches one block)
share: embeddings 48%, blocks 52%, final LayerNorm the rest

7B model, weights only:
  float32   28.0 GB
  float16   14.0 GB
  int8       7.0 GB
  int4       3.5 GB

distilgpt2 layers=6  vs  gpt2 layers=12  (same hidden size 768)
```
- **Parameters x bytes = memory.** 81.9 M x 4 bytes = 0.33 GB. A 7B model in float16: 7e9 x 2 = 14 GB just for weights; in int4 about 3.5 GB. This arithmetic is a standard interview question and the reason quantization exists ([05_llm_parameters/05_quantization](../../05_llm_parameters/05_quantization/)).
- **`c_attn` is (768, 2304)** because 2304 = 3 x 768: Query, Key and Value projections from [step 04](../04_attention_from_scratch/WALKTHROUGH.md) in one matrix.
- **Distilled** model = a smaller student trained to imitate a bigger teacher (gpt2): half the layers, ~66% of the parameters, faster, a bit worse.
- Real memory is higher: add activations, the KV cache ([step 08](../08_generation_loop/WALKTHROUGH.md)) and framework overhead; during training also gradients and optimizer states (see [06_llm_training_and_lora](../../06_llm_training_and_lora/)).

## Experiments
- Load `"gpt2"`: expect ~124 M parameters.
- `model.half()` then `next(model.parameters()).dtype` - becomes float16. (CPU float16 is slow/unsupported for some ops; GPUs love it.)
- Embeddings are 48% of distilgpt2's parameters (printed above); one block matches step 05's `12*d^2 + 13*d` exactly. The final LayerNorm (1,536) makes up the rest.

## Debugging practice
Offline and model never downloaded -> `OSError: We couldn't connect to 'https://huggingface.co' ...`. Check the network, then whether the model is already in `~/.cache/huggingface/hub`. Misspelled name -> a different `OSError` mentioning "is not a valid model identifier". Different errors, different root causes: read the sentence, don't just rerun.

## Rebuild checklist
- [ ] Load config and model by name; print layers, hidden size, context window.
- [ ] Write `count_params(model)`.
- [ ] Compute memory for 4 dtypes by hand, then check with code.
- [ ] Explain why `.eval()` matters for inference.

## The new problem (cliffhanger)
We have a loaded model and know what is inside, but we have only looked at its weights. We have not fed it a single token. What does it actually output when you give it a sentence? A probability for each of 50,257 possible next words, and a number that says how wrong it was. Next: [07_forward_pass_logits](../07_forward_pass_logits/WALKTHROUGH.md).

> **Say it aloud (interview box):** A pretrained model is an architecture plus saved weights, which I load by name with `AutoModelForCausalLM.from_pretrained`. Memory is parameters times bytes per number: distilgpt2 has 82 million parameters, about 0.33 GB in float32, and a 7B model needs 14 GB in float16 or 3.5 GB in int4. About half of distilgpt2's parameters are embeddings and the rest are six blocks of roughly 12 times d_model squared each.
