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
- Real memory is higher: add activations, the KV cache ([step 08](../08_generation_loop/WALKTHROUGH.md)) and framework overhead; during training also gradients and optimizer states (see [06_llm_training_and_fast_inference](../../06_llm_training_and_fast_inference/)).

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
