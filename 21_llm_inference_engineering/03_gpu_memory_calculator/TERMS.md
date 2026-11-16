# Terms — GPU Memory Calculator

Folder: `06_llm_training_and_fast_inference/08_gpu_memory_calculator` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [06 README](../README.md)

**Terms in this folder:** [argparse](#argparse) · [add_argument, choices and action="store_true"](#add_argument-choices-and-actionstore_true) · [Dictionary of tuples and unpacking](#dictionary-of-tuples-and-unpacking) · [Bytes per value (precision table)](#bytes-per-value-precision-table) · [GB vs GiB (1e9)](#gb-vs-gib-1e9) · [Layers, KV heads and head_dim](#layers-kv-heads-and-head_dim) · [Grouped-query attention (GQA)](#grouped-query-attention-gqa) · [KV cache size formula](#kv-cache-size-formula) · [Weights memory](#weights-memory) · [Overhead (activations, CUDA context, fragmentation)](#overhead-activations-cuda-context-fragmentation) · [Training memory: 16 bytes per parameter](#training-memory-16-bytes-per-parameter) · [LoRA / QLoRA training memory](#lora--qlora-training-memory) · [Mixed precision and fp32 master weights](#mixed-precision-and-fp32-master-weights) · [past_key_values](#past_key_values) · [numel and element_size sum](#numel-and-element_size-sum) · [PagedAttention](#pagedattention) · [f-string alignment `:<10` `:>13.1f`](#f-string-alignment-10-131f) · [`if __name__ == "__main__"`](#if-__name__--__main__) · [argparse invalid choice error](#argparse-invalid-choice-error)

---

## argparse  `(standard library)`
**1. Definition:** `argparse` is the standard-library module for defining and parsing command-line options such as `--seq 4096`.
**2. In plain words:** It lets a script take settings from the terminal without editing the code, and it generates helpful `--help` and error messages automatically.
**3. Example to quote:** > "I run `python main.py --preset llama-3-8b --batch 1 --seq 8192` and `argparse` turns those flags into Python values."
```python
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--seq", type=int, default=4096)
args = ap.parse_args()
```
**Seen in:** `main.py`, bottom

## add_argument, choices and action="store_true"  `(standard library)`
**1. Definition:** `ap.add_argument(name, type=, default=, choices=, help=, action=)` declares one option; `choices` restricts accepted values and `action="store_true"` makes a flag that is `True` when present and takes no value.
**2. In plain words:** `--preset` accepts only keys of `PRESETS`; `--skip-check` is an on/off switch. Note the dash becomes an underscore: `args.skip_check`.
**3. Example to quote:** > "`--skip-check` is a store_true flag that skips loading the real model for a quick run."
**Seen in:** `main.py`, bottom

## Dictionary of tuples and unpacking  `(syntax)`
**1. Definition:** A dict maps names to tuples of settings, and `p, layers, kv_heads, head_dim = PRESETS[name]` unpacks the tuple into four variables.
**2. In plain words:** One lookup gives you all the model's shape numbers. It's tidy, but the order of values must be remembered, which is a small readability trade-off.
**3. Example to quote:** > "`PRESETS['llama-7b']` is `(6.7e9, 32, 32, 128)`: parameters, layers, KV heads and head dimension."
**Seen in:** `main.py`, `PRESETS`

## Bytes per value (precision table)  `(concept)`
**1. Definition:** The `BYTES` dictionary maps precision names to the bytes used per number: fp32 4.0, fp16 2.0, int8 1.0, int4 0.5.
**2. In plain words:** Every memory estimate in this file is "how many numbers" times one of these values.
**3. Example to quote:** > "int4 is 0.5 bytes per weight, so a 6.7B model's weights take about 3.4 GB."
**Seen in:** `main.py`

## GB vs GiB (1e9)  `(concept)`
**1. Definition:** The script defines `GB = 1e9` (decimal gigabytes), whereas GiB uses 2^30 bytes.
**2. In plain words:** The numbers are estimates and the two differ by about 7%, so don't be surprised if a GPU's reported memory differs slightly.
**3. Example to quote:** > "I use 1e9 bytes per GB, so my figures are rough; a tool reporting GiB will show slightly smaller numbers."
**Seen in:** `main.py`, `GB = 1e9`

## Layers, KV heads and head_dim  `(concept)`
**1. Definition:** `layers` is the number of transformer blocks, `kv_heads` the number of key/value attention heads, and `head_dim` the vector size per head.
**2. In plain words:** These three numbers determine how big each token's cached keys and values are.
**3. Example to quote:** > "distilgpt2 has 6 layers, 12 KV heads and a head dimension of 64."
**Seen in:** `main.py`, `PRESETS`

## Grouped-query attention (GQA)  `(concept)`
**1. Definition:** GQA shares each key/value head across several query heads, so a model has fewer KV heads than attention heads.
**2. In plain words:** Fewer KV heads means a smaller KV cache. Llama-3-8B uses 8 KV heads versus 32, a 4× smaller cache: at batch 1 and sequence 8192 its cache is about 1.1 GB.
**3. Example to quote:** > "With GQA, Llama-3-8B's KV cache is a quarter the size of a classic multi-head 7B model's at the same settings."
**Seen in:** `main.py`, `PRESETS` comments

## KV cache size formula  `(concept)`
**1. Definition:** KV cache bytes = 2 × layers × kv_heads × head_dim × tokens × batch × bytes per value, where the 2 counts keys and values.
**2. In plain words:** It grows with context length and number of simultaneous users, but not with the weights. At long context or big batch it can exceed the weights. The script checks it on distilgpt2: formula 1,179,648 bytes equals the measured 1,179,648 bytes.
**3. Example to quote:** > "For a 6.7B model at 4096 tokens and batch 8 the KV cache is 17.2 GB, more than the 13.4 GB of fp16 weights."
```python
2 * layers * kv_heads * head_dim * seq_len * batch * bytes_per_value
```
**Seen in:** `main.py`, `kv_cache_bytes`

## Weights memory  `(concept)`
**1. Definition:** Memory for the weights is parameters × bytes per parameter.
**2. In plain words:** The easy part: a 7B model is about 14 GB in fp16, 7 GB in int8 and 3.5 GB in int4.
**3. Example to quote:** > "A 6.7B model needs roughly 13.4 GB for fp16 weights and 3.4 GB for int4."
**Seen in:** `main.py`, `inference_table`

## Overhead (activations, CUDA context, fragmentation)  `(concept)`
**1. Definition:** Overhead is the extra GPU memory beyond weights and KV cache for temporary activations, the CUDA runtime context and allocator fragmentation; the script approximates it as 10%.
**2. In plain words:** Estimates are never exact; leave headroom.
**3. Example to quote:** > "I add a rough 10% overhead, so the fp16 total is about 33.6 GB, not just weights plus cache."
**Seen in:** `main.py`, `inference_table`

## Training memory: 16 bytes per parameter  `(concept)`
**1. Definition:** Full mixed-precision AdamW training needs about 16 bytes per parameter: 2 (weights) + 2 (gradients) + 4 (fp32 master copy) + 4 (Adam first moment) + 4 (Adam second moment).
**2. In plain words:** Training costs roughly eight times the fp16 weights alone. A 6.7B model needs about 107.2 GB, before activations.
**3. Example to quote:** > "Full fine-tuning is about 16 bytes per parameter, so a 6.7B model needs around 107 GB."
**Seen in:** `main.py`, `training_table`

## LoRA / QLoRA training memory  `(concept)`
**1. Definition:** LoRA training memory is the frozen base in fp16 (2 B/param) plus 16 B per parameter only for the tiny trainable fraction; QLoRA uses a 4-bit base (0.5 B/param) instead.
**2. In plain words:** That is why 107.2 GB shrinks to 13.5 GB with LoRA and 3.5 GB with QLoRA.
**3. Example to quote:** > "QLoRA brings the 6.7B training footprint from 107 GB to about 3.5 GB, so it fits a consumer GPU."
**Seen in:** `main.py`, `training_table`

## Mixed precision and fp32 master weights  `(concept)`
**1. Definition:** Mixed-precision training computes in 16-bit but keeps a 32-bit master copy of the weights for accurate updates.
**2. In plain words:** Small updates would be lost in 16-bit rounding, so a precise copy is kept. That copy is part of the 16 bytes per parameter.
**3. Example to quote:** > "The 4-byte fp32 master copy is one of the five components in the 16 bytes per parameter."
**Seen in:** `main.py`, `training_table` docstring

## past_key_values  `(third-party: transformers)`
**1. Definition:** `out.past_key_values` is the cache object returned by a forward pass with `use_cache=True`; each layer holds a keys tensor and a values tensor (`.layers[i].keys`, `.values`).
**2. In plain words:** It's the real KV cache, so you can measure it and compare with your formula.
**3. Example to quote:** > "I summed the bytes of every layer's keys and values and got 1,179,648, exactly my formula's number."
**Seen in:** `main.py`, `check_against_real_model`

## numel and element_size sum  `(third-party: torch)`
**1. Definition:** `t.numel()` counts a tensor's elements and `t.element_size()` gives bytes per element; their product is the tensor's size in bytes.
**2. In plain words:** Summing it over the cache tensors measures real memory.
**3. Example to quote:** > "`t.numel() * t.element_size()` summed over all cache tensors gave 1,179,648 bytes."
**Seen in:** `main.py`

## PagedAttention  `(concept)`
**1. Definition:** PagedAttention is the technique used by vLLM that stores the KV cache in fixed-size blocks (pages) allocated on demand instead of one large contiguous region per sequence.
**2. In plain words:** It avoids wasting memory on reserved but unused space, so more requests fit on the GPU. The script's closing note names it, along with GQA and KV-cache quantisation, as responses to the cache being huge.
**3. Example to quote:** > "Since the KV cache can dominate memory, vLLM's PagedAttention stores it in pages to cut waste."
**Seen in:** `main.py`, closing print

## f-string alignment `:<10` `:>13.1f`  `(syntax)`
**1. Definition:** In an f-string, `:<10` left-aligns in 10 characters and `:>13.1f` right-aligns a float with 1 decimal in 13 characters.
**2. In plain words:** It's how the table columns line up.
**3. Example to quote:** > "`f'{prec:<10}{weights / GB:>13.1f}'` prints neat columns."
**Seen in:** `main.py`

## `if __name__ == "__main__"`  `(syntax)`
**1. Definition:** This condition is true only when the file is run directly, not when it is imported.
**2. In plain words:** It lets the file act as both a script and a reusable module.
**3. Example to quote:** > "The argparse and printing code sits under `if __name__ == '__main__':` so importing `kv_cache_bytes` elsewhere doesn't run it."
**Seen in:** `main.py`

## argparse invalid choice error  `(concept)`
**1. Definition:** When a value is not in `choices`, argparse exits with `error: argument --preset: invalid choice: 'gpt5' (choose from ...)`.
**2. In plain words:** Read the whole line: it lists the valid options.
**3. Example to quote:** > "I ran `--preset gpt5`, read the list of valid choices at the end of the error, and used `llama-7b`."
**Seen in:** `WALKTHROUGH.md`, Debugging practice
