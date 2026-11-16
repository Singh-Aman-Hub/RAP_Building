# Terms — GGUF and llama.cpp formats

Folder: `21_llm_inference_engineering/02_gguf_and_llamacpp_formats` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [21 README](../README.md)

**Terms in this folder:** [GGUF](#gguf) · [Metadata key-value (typed)](#metadata-key-value-typed) · [Alignment (general.alignment)](#alignment-generalalignment) · [ggml dims order](#ggml-dims-order) · [Block quantization](#block-quantization) · [Q8_0](#q8_0) · [Q4_0](#q4_0) · [K-quants (Q4_K_M, Q5_K_M, Q6_K)](#k-quants-q4_k_m-q5_k_m-q6_k) · [I-quants (IQ4_XS, IQ3_XXS...)](#i-quants-iq4_xs-iq3_xxs) · [Importance matrix (imatrix)](#importance-matrix-imatrix) · [Bits per weight (bpw)](#bits-per-weight-bpw) · [convert_hf_to_gguf.py and llama-quantize](#convert_hf_to_ggufpy-and-llama-quantize) · [llama.cpp / ggml / Ollama](#llamacpp--ggml--ollama) · [struct.pack / struct.unpack_from](#structpack--structunpack_from)

---

## GGUF  `(concept)`
**1. Definition:** The single-file model format of ggml/llama.cpp: magic 'GGUF', uint32 version (3), uint64 tensor count, uint64 metadata-KV count, typed metadata key-value pairs, tensor infos (name, n_dims, dims, ggml type, offset), padding to the alignment (default 32), then the tensor data.
**2. In plain words:** One file holding the weights, the tokenizer and the settings, ready to be memory-mapped.
**3. Example to quote:** > "GGUF is self-describing: architecture, tokenizer and tensors all live in one aligned file."
```python
head = b'GGUF' + struct.pack('<I', 3) + struct.pack('<QQ', n_tensors, n_kv)
```
**Seen in:** `main.py` sections 1-3

## Metadata key-value (typed)  `(concept)`
**1. Definition:** Each metadata entry is a length-prefixed UTF-8 key, a uint32 value type (UINT8..FLOAT64, BOOL, STRING, ARRAY) and the value; arrays carry an element type and a uint64 length. Examples: general.architecture, <arch>.block_count, tokenizer.ggml.tokens.
**2. In plain words:** A labelled settings table with types.
**3. Example to quote:** > "Tokenizer vocabulary and RoPE settings are metadata inside the GGUF file."
**Seen in:** `main.py` write_gguf / Reader

## Alignment (general.alignment)  `(concept)`
**1. Definition:** Tensor data and every tensor offset are padded to a multiple of the alignment (default 32 bytes) so the data can be used straight from an mmap with aligned loads.
**2. In plain words:** Padding bytes so each tensor starts on a tidy boundary.
**3. Example to quote:** > "Offsets are relative to the aligned start of the data section."
```python
pad = (-n) % 32
```
**Seen in:** `main.py` pad_to

## ggml dims order  `(concept)`
**1. Definition:** GGUF lists tensor dimensions fastest-varying first, so a numpy shape (rows, cols) is stored as dims [cols, rows].
**2. In plain words:** The dims list is written backwards compared with numpy.
**3. Example to quote:** > "If your shapes look transposed in gguf-dump, this is why."
```python
dims = list(reversed(shape))
```
**Seen in:** `main.py` write_gguf

## Block quantization  `(concept)`
**1. Definition:** Weights are split into small blocks (32 for the legacy types); each block stores one scale (and for some types a minimum) plus low-bit integers, so an outlier only damages its own block.
**2. In plain words:** Many small rulers instead of one ruler for the whole model.
**3. Example to quote:** > "Smaller blocks track the local range better but spend more bits on scales."
**Seen in:** `main.py` quantize_q8_0

## Q8_0  `(concept)`
**1. Definition:** Block of 32 weights: one fp16 scale d = max|x|/127 and 32 int8 values q = round(x/d); 34 bytes per block = 8.5 bits per weight.
**2. In plain words:** Near-lossless 8-bit.
**3. Example to quote:** > "Q8_0 costs 8.5 bits per weight, not 8, because of the scales."
```python
d = np.abs(blocks).max(1) / 127
q = np.rint(blocks / d[:, None]).astype(np.int8)
```
**Seen in:** `main.py` quantize_q8_0

## Q4_0  `(concept)`
**1. Definition:** Legacy 4-bit type: block of 32, fp16 scale and 32 four-bit values packed two per byte = 18 bytes = 4.5 bits per weight.
**2. In plain words:** Simple 4-bit blocks, mostly superseded by k-quants.
**3. Example to quote:** > "Q4_0 is 4.5 bits per weight and noticeably noisier than Q8_0."
**Seen in:** `main.py` quantize_q4_0

## K-quants (Q4_K_M, Q5_K_M, Q6_K)  `(concept)`
**1. Definition:** llama.cpp quantization families that use super-blocks of 256 weights split into sub-blocks, with the sub-block scales themselves quantized. The _S/_M/_L suffix means a mix: sensitive tensors (e.g. some attention and feed-forward ones) get a higher-bit type.
**2. In plain words:** Smarter 4/5/6-bit that spends extra bits where the model is sensitive.
**3. Example to quote:** > "Q4_K_M is the usual default: around 4.8 bits per weight on average."
**Seen in:** WALKTHROUGH theory

## I-quants (IQ4_XS, IQ3_XXS...)  `(concept)`
**1. Definition:** llama.cpp types that use non-linear / codebook-style quantization (often with an importance matrix) to reach very low bits per weight; IQ4_XS is about 4.25 bits per weight.
**2. In plain words:** Lower-bit quantization with a lookup table, slower to compute but smaller.
**3. Example to quote:** > "I-quants trade a little speed for quality at the same size."
**Seen in:** WALKTHROUGH theory

## Importance matrix (imatrix)  `(concept)`
**1. Definition:** Optional statistics of activation magnitudes collected on sample text and given to llama-quantize so that the quantizer weights errors on important weights more.
**2. In plain words:** Tell the quantizer which weights matter most.
**3. Example to quote:** > "An imatrix is a calibration signal for low-bit GGUF files."
**Seen in:** WALKTHROUGH theory

## Bits per weight (bpw)  `(concept)`
**1. Definition:** Total file bits divided by parameter count; includes scales. File size ~ params x bpw / 8.
**2. In plain words:** The honest cost per number.
**3. Example to quote:** > "A 7B model at 4.85 bpw is about 4.1 GB."
```python
gb = 6.74e9 * 4.85 / 8 / 1e9
```
**Seen in:** `main.py` size_table

## convert_hf_to_gguf.py and llama-quantize  `(concept)`
**1. Definition:** Two llama.cpp tools: the first converts a Hugging Face checkpoint (safetensors + config + tokenizer) into an f16/f32 GGUF; the second re-quantizes a GGUF to a chosen type such as Q4_K_M.
**2. In plain words:** Convert first, then shrink.
**3. Example to quote:** > "The pipeline is HF safetensors, f16 GGUF, quantized GGUF."
```python
python convert_hf_to_gguf.py model_dir --outfile m-f16.gguf
./llama-quantize m-f16.gguf m-Q4_K_M.gguf Q4_K_M
```
**Seen in:** `main.py` footer comments (reference only)

## llama.cpp / ggml / Ollama  `(concept)`
**1. Definition:** llama.cpp is a C/C++ inference engine built on the ggml tensor library; it mmaps GGUF files and runs on CPU, Metal, CUDA and Vulkan. Ollama wraps it with a model registry and a REST API (default port 11434).
**2. In plain words:** The engine, its math library, and the friendly wrapper.
**3. Example to quote:** > "Ollama is a packaging and serving layer, llama.cpp does the math."
**Seen in:** WALKTHROUGH

## struct.pack / struct.unpack_from  `(function)`
**1. Definition:** Python stdlib functions that convert between Python numbers and fixed binary layouts using format strings: '<' little-endian, B/H/I/Q = unsigned 8/16/32/64-bit, f/d = float32/float64.
**2. In plain words:** Write and read bytes with a template.
**3. Example to quote:** > "'<Q' is an unsigned 64-bit little-endian integer."
```python
struct.pack('<QQ', 4, 9)
```
**Seen in:** `main.py` throughout
