# Step 04 - Load a pretrained model

## Goal
Load distilgpt2, inspect its architecture, count parameters and estimate memory in float32/float16/int8/int4.

## Run it
```bash
.venv/bin/python 04_llm_fundamentals/04_load_pretrained_model/main.py
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

  transformer.wte.weight                   (50257, 768)     <- token embeddings (step 03)
  transformer.wpe.weight                   (1024, 768)      <- position embeddings (step 03)
  transformer.h.0.attn.c_attn.weight       (768, 2304)      <- Q, K, V projections packed (3 x 768)
  transformer.h.0.attn.c_proj.weight       (768, 768)
distilgpt2 layers=6  vs  gpt2 layers=12
```
- **Parameters x bytes = memory.** 81.9 M x 4 bytes = 0.33 GB. A 7B model in float16: 7e9 x 2 = 14 GB just for weights; in int4 about 3.5 GB. This arithmetic is a standard interview question and the reason quantization exists (module 05 step 05).
- **`c_attn` is (768, 2304)** because 2304 = 3 x 768: Query, Key and Value projections from step 01 in one matrix.
- **Distilled** model = a smaller student trained to imitate a bigger teacher (gpt2): half the layers, ~66% of the parameters, faster, a bit worse.
- Real memory is higher: add activations, the KV cache (module 05 step 04) and framework overhead; during training also gradients and optimizer states (module 06).

## Experiments
- Load `"gpt2"`: expect ~124 M parameters.
- `model.half()` then `next(model.parameters()).dtype` - becomes float16. (CPU float16 is slow/unsupported for some ops; GPUs love it.)
- Run `solutions/ex04_param_breakdown.py`: nearly half of distilgpt2's parameters are embeddings.

## Debugging practice
Offline and model never downloaded -> `OSError: We couldn't connect to 'https://huggingface.co' ...`. Check the network, then whether the model is already in `~/.cache/huggingface/hub`. Misspelled name -> a different `OSError` mentioning "is not a valid model identifier". Different errors, different root causes: read the sentence, don't just rerun.

## Rebuild checklist
- [ ] Load config and model by name; print layers, hidden size, context window.
- [ ] Write `count_params(model)`.
- [ ] Compute memory for 4 dtypes by hand, then check with code.
- [ ] Explain why `.eval()` matters for inference.

## Say it aloud
"A pretrained model is an architecture plus saved weights. With Hugging Face I load both by name using `AutoModelForCausalLM.from_pretrained`. I can count parameters with `sum(p.numel() ...)`, and estimate memory as parameters times bytes per number: distilgpt2 has 82 million parameters, about 0.33 GB in float32. Trade-off: bigger models are better but need proportionally more memory and compute, which is why we distill and quantize."
