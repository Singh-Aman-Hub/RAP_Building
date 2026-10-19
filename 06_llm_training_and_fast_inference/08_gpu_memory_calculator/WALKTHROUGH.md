# Step 08 — GPU memory calculator

## Goal
Answer "will it fit?" with arithmetic: weights + KV cache + overhead for serving, and weights + gradients + optimizer states for training.

## Run it
```bash
cd 06_llm_training_and_fast_inference/08_gpu_memory_calculator
python main.py
python main.py --preset llama-3-8b --batch 1 --seq 8192
python main.py --preset qwen2.5-0.5b --skip-check
```

## The formulas
```
weights bytes     = parameters x bytes_per_param      (fp32 4, fp16/bf16 2, int8 1, int4 0.5)
KV cache bytes    = 2 x layers x kv_heads x head_dim x tokens x batch x bytes_per_value
                    (2 = keys AND values)
training (AdamW, mixed precision) ~ 16 bytes/param = 2 weights + 2 grads + 4 fp32 master + 4 Adam m + 4 Adam v
LoRA training     ~ frozen base (2 B/param) + 16 B x tiny trainable fraction
QLoRA training    ~ frozen base (0.5 B/param) + 16 B x tiny trainable fraction
```

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `argparse.ArgumentParser` | read `--flags` from the command line | `ap.add_argument("--seq", type=int, default=4096)` |
| `choices=PRESETS` | only dictionary keys are accepted | |
| `action="store_true"` | a flag with no value; present -> `True` | `--skip-check` |
| dict of tuples + unpacking | `p, layers, kv_heads, head_dim = PRESETS[name]` | |
| `f"{x:<10}{y:>13.1f}"` | left / right alignment and decimals | |
| `if __name__ == "__main__":` | run only when executed directly, not when imported | |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does |
|---|---|---|
| `argparse` | built-in | command-line options |
| `out.past_key_values` | transformers | the actual cache returned by a forward pass (`.layers[i].keys / .values`) |
| `tensor.numel()`, `.element_size()` | torch | count and bytes per element |

## Execution trace (real run, default: 6.7B Llama-style, seq 4096, batch 8)
| precision | weights GB | KV cache GB | total ~GB |
|---|---|---|---|
| fp32 | 26.8 | 17.2 | 48.4 |
| fp16 | 13.4 | 17.2 | 33.6 |
| int8 | 6.7 | 17.2 | 26.3 |
| int4 | 3.4 | 17.2 | 22.6 |
- Quantising weights shrinks only the weights column; the KV cache (fp16) stays 17.2 GB and dominates.
- Training a 6.7B model: full fine-tune **107.2 GB**, LoRA **13.5 GB**, QLoRA **3.5 GB**.
- `llama-3-8b --batch 1 --seq 8192`: KV cache only **1.1 GB** because grouped-query attention uses 8 KV heads instead of 32.
- Formula check on distilgpt2 (32 tokens, fp32): formula 1,179,648 bytes = measured 1,179,648 bytes.

## Experiments (change ONE thing)
1. `--batch 1` vs `--batch 32`: only the KV column scales with batch.
2. `--seq 32768 --batch 1`: long context alone can fill a 24 GB GPU.
3. Compare `llama-7b` with `llama-3-8b` at the same settings: the GQA effect.

## Debugging practice
`python main.py --preset gpt5`:
```
main.py: error: argument --preset: invalid choice: 'gpt5' (choose from llama-7b, llama-3-8b, ...)
```
argparse errors list the valid choices — read to the end of the line.

## Rebuild checklist
- [ ] write `kv_cache_bytes(...)` from memory
- [ ] explain why KV cache grows with batch and length but weights do not
- [ ] verify your formula against `past_key_values` for a small model

## Say it aloud
"Weights take parameters times bytes per parameter, so a 7B model is about 14 GB in fp16 and 3.5 GB in int4. On top of that the KV cache grows with layers, KV heads, head size, tokens and batch, and at long context or big batch it can exceed the weights. For training, AdamW needs about 16 bytes per parameter, which is why LoRA and QLoRA matter. These are estimates; real usage adds activations and fragmentation."
