<!-- vocabulary draft: 21_llm_inference_engineering -->

### Pickle deserialization attack
<!-- cat: python -->
- **Technical definition:** Loading a pickle executes callables named in the file (via __reduce__/REDUCE), so an untrusted pickle can run arbitrary code.
- **Plain English:** A booby-trapped recipe: reading it makes you do things.
- **Example / analogy:** Why .bin/.pt checkpoints from strangers are risky and safetensors exists.
- **Snippet:**
```python
pickle.loads(pickle.dumps(obj_with_reduce))
```
- **Snippet explained:** Builds a pickle whose __reduce__ names print; loads runs it.
- **Related terms:** [[safetensors]], [[weights_only (torch.load)]]

### weights_only (torch.load)
<!-- cat: python -->
- **Technical definition:** torch.load flag restricting unpickling to tensors and simple containers; default True since torch 2.6.
- **Plain English:** A filter that blocks code in checkpoint files.
- **Example / analogy:** torch.load(path) now refuses arbitrary objects.
- **Snippet:**
```python
torch.load('m.pt', weights_only=True)
```
- **Snippet explained:** Loads only plain tensors and containers.
- **Related terms:** [[Pickle deserialization attack]]

### Memory-mapped loading (mmap)
<!-- cat: llm -->
- **Technical definition:** Mapping a file into address space so pages load lazily on first touch and are shared via the OS page cache.
- **Plain English:** The file pretends to be RAM.
- **Example / analogy:** safe_open and llama.cpp use it to start fast.
- **Snippet:**
```python
mm=np.memmap(p,dtype=np.uint8,mode='r')
```
- **Snippet explained:** Maps the file; no bytes are read until sliced.
- **Related terms:** [[safetensors]], [[GGUF]]

### Peak RSS
<!-- cat: python -->
- **Technical definition:** Maximum resident set size of a process (resource.getrusage ru_maxrss; bytes on macOS, KB on Linux), monotonic within one process.
- **Plain English:** The most RAM a process ever held.
- **Example / analogy:** Measure each loader in a fresh process.
- **Snippet:**
```python
resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
```
- **Snippet explained:** Reads the high-water mark.
- **Related terms:** [[Memory-mapped loading (mmap)]]

### Sharded checkpoint
<!-- cat: llm -->
- **Technical definition:** A model split into model-0000X-of-0000N.safetensors plus model.safetensors.index.json with a weight_map from tensor name to shard.
- **Plain English:** A multi-volume book with an index.
- **Example / analogy:** 70B models ship as many shards.
- **Snippet:**
```python
idx['weight_map'][name]
```
- **Snippet explained:** Finds which shard file holds a tensor.
- **Related terms:** [[safetensors]]

### Meta device
<!-- cat: llm -->
- **Technical definition:** A PyTorch device with shape and dtype but no storage; used to build a model skeleton at ~zero memory.
- **Plain English:** A blueprint with no bricks.
- **Example / analogy:** Lets you size a model before loading it.
- **Snippet:**
```python
with torch.device('meta'): m=AutoModelForCausalLM.from_config(cfg)
```
- **Snippet explained:** Creates the architecture without allocating weights.
- **Related terms:** [[device_map]]

### Hugging Face cache layout
<!-- cat: llm -->
- **Technical definition:** models--org--name/{blobs, snapshots/<commit>, refs/<branch>}; snapshots are symlinks into content-addressed blobs.
- **Plain English:** Files stored once, revisions as shortcut folders.
- **Example / analogy:** Offline mode reads only from here.
- **Snippet:**
```python
HF_HUB_OFFLINE=1
```
- **Snippet explained:** Forbids network access so only the cache is used.
- **Related terms:** [[safetensors]]

### GGUF
<!-- cat: llm -->
- **Technical definition:** Single-file ggml/llama.cpp format: magic, version 3, counts, typed metadata KV, tensor infos, 32-byte alignment, tensor data.
- **Plain English:** One file with weights, tokenizer and settings.
- **Example / analogy:** What Ollama and llama.cpp load.
- **Snippet:**
```python
head=b'GGUF'+struct.pack('<IQQ',3,n_tensors,n_kv)
```
- **Snippet explained:** Packs the fixed GGUF header.
- **Related terms:** [[Block quantization]], [[K-quants (Q4_K_M, Q5_K_M, Q6_K)]]

### Block quantization
<!-- cat: llm -->
- **Technical definition:** Quantizing small blocks of weights with their own scale so outliers only affect one block.
- **Plain English:** Many small rulers.
- **Example / analogy:** Q8_0 uses blocks of 32.
- **Snippet:**
```python
blocks=x.reshape(-1,32)
```
- **Snippet explained:** Splits weights into blocks of 32.
- **Related terms:** [[Q8_0]], [[K-quants (Q4_K_M, Q5_K_M, Q6_K)]]

### Q8_0
<!-- cat: llm -->
- **Technical definition:** GGUF type: block 32, fp16 scale d=max|x|/127, int8 values; 34 bytes per block = 8.5 bits per weight.
- **Plain English:** Near-lossless 8-bit.
- **Example / analogy:** About half the size of fp16.
- **Snippet:**
```python
q=np.rint(blocks/d[:,None]).astype(np.int8)
```
- **Snippet explained:** Rounds each weight to an int8 using the block scale.
- **Related terms:** [[Block quantization]]

### Q4_0
<!-- cat: llm -->
- **Technical definition:** Legacy GGUF 4-bit type: block 32, fp16 scale, 4-bit values, 18 bytes per block = 4.5 bits per weight.
- **Plain English:** Simple 4-bit blocks.
- **Example / analogy:** Superseded by k-quants for quality.
- **Snippet:**
```python
packed=q[:,:16]|(q[:,16:]<<4)
```
- **Snippet explained:** Packs two 4-bit values per byte.
- **Related terms:** [[Block quantization]], [[K-quants (Q4_K_M, Q5_K_M, Q6_K)]]

### K-quants (Q4_K_M, Q5_K_M, Q6_K)
<!-- cat: llm -->
- **Technical definition:** llama.cpp quant families with 256-weight super-blocks and quantized sub-block scales; _M/_S suffixes are mixes giving sensitive tensors more bits.
- **Plain English:** Smarter low-bit quantization.
- **Example / analogy:** Q4_K_M, about 4.8 bits per weight, is a common default.
- **Snippet:**
```python
./llama-quantize m-f16.gguf m-Q4_K_M.gguf Q4_K_M
```
- **Snippet explained:** Reference command: re-quantize a GGUF.
- **Related terms:** [[I-quants (IQ4_XS)]], [[Bits per weight (bpw)]]

### I-quants (IQ4_XS)
<!-- cat: llm -->
- **Technical definition:** llama.cpp low-bit types using non-linear/codebook quantization (often with an importance matrix); IQ4_XS is about 4.25 bits per weight.
- **Plain English:** Lookup-table quantization.
- **Example / analogy:** Smaller than Q4_K at similar quality.
- **Snippet:**
```python
./llama-quantize --imatrix imat.dat m.gguf m-IQ4_XS.gguf IQ4_XS
```
- **Snippet explained:** Reference command using an importance matrix.
- **Related terms:** [[K-quants (Q4_K_M, Q5_K_M, Q6_K)]], [[Importance matrix (imatrix)]]

### Importance matrix (imatrix)
<!-- cat: llm -->
- **Technical definition:** Activation statistics from sample text passed to llama-quantize so important weights get lower error.
- **Plain English:** Tell the quantizer what matters.
- **Example / analogy:** Common with low-bit GGUF files.
- **Snippet:**
```python
./llama-imatrix -m m-f16.gguf -f calib.txt -o imat.dat
```
- **Snippet explained:** Reference command: collects statistics.
- **Related terms:** [[I-quants (IQ4_XS)]]

### convert_hf_to_gguf and llama-quantize
<!-- cat: llm -->
- **Technical definition:** llama.cpp tools: convert an HF checkpoint to f16 GGUF, then re-quantize to a chosen type.
- **Plain English:** Convert, then shrink.
- **Example / analogy:** Pipeline from safetensors to Q4_K_M.
- **Snippet:**
```python
python convert_hf_to_gguf.py dir --outfile m-f16.gguf
```
- **Snippet explained:** Reference command: HF to GGUF.
- **Related terms:** [[GGUF]]

### KV cache memory formula
<!-- cat: llm -->
- **Technical definition:** KV bytes = 2 x layers x kv_heads x head_dim x tokens x batch x bytes_per_value; weights bytes = params x bytes_per_param.
- **Plain English:** How many bytes the model's memory of the conversation takes.
- **Example / analogy:** Llama-3-8B fp16: 128 KiB per token.
- **Snippet:**
```python
2*32*8*128*seq*batch*2
```
- **Snippet explained:** Computes llama-3-8b KV bytes.
- **Related terms:** [[KV-cache dtype]]

### KV-cache dtype
<!-- cat: llm -->
- **Technical definition:** Storage precision of cached keys/values (fp16, fp8, int4); fp8 halves and int4 quarters the cache versus fp16.
- **Plain English:** Smaller numbers for the memory of the chat.
- **Example / analogy:** vLLM exposes --kv-cache-dtype fp8.
- **Snippet:**
```python
bytes = {'fp16':2,'fp8':1,'int4':0.5}
```
- **Snippet explained:** Bytes per cached value.
- **Related terms:** [[KV cache memory formula]]

### Affine quantization
<!-- cat: llm -->
- **Technical definition:** Mapping real values to integer codes with q = round(x/s) + z and x_hat = (q - z) * s.
- **Plain English:** Snapping numbers onto an evenly spaced ruler.
- **Example / analogy:** A uint8 ruler over [-0.4, 3.0] has step 0.0133 and zero at code 30.
- **Snippet:**
```python
q = np.clip(np.round(x / s) + z, 0, 255)
```
- **Snippet explained:** Computes integer codes from floats given scale s and zero-point z.
- **Related terms:** [[Scale and zero-point]], [[Symmetric quantization]], [[Asymmetric quantization]]

### Scale and zero-point
<!-- cat: llm -->
- **Technical definition:** The scale is the real width of one integer step; the zero-point is the integer code that decodes to 0.0.
- **Plain English:** Tick spacing and which tick is zero.
- **Example / analogy:** Asymmetric int8 stores one scale and one zero-point per group.
- **Snippet:**
```python
z = int(round(-lo / s))
```
- **Snippet explained:** Finds the code that represents exactly 0.0.
- **Related terms:** [[Affine quantization]], [[Per-group (block-wise) quantization]]

### Symmetric quantization
<!-- cat: llm -->
- **Technical definition:** Quantization with range [-max|x|, max|x|], zero-point 0 and signed integer codes.
- **Plain English:** A ruler centred on zero.
- **Example / analogy:** Weights are usually quantized symmetrically.
- **Snippet:**
```python
s = np.abs(x).max() / 127
```
- **Snippet explained:** Scale from the absolute maximum for int8.
- **Related terms:** [[Asymmetric quantization]], [[Per-channel quantization]]

### Asymmetric quantization
<!-- cat: llm -->
- **Technical definition:** Quantization using [min, max] with a zero-point, using all 2^b codes.
- **Plain English:** A ruler that starts at the smallest value.
- **Example / analogy:** Post-ReLU activations gain about one bit from it.
- **Snippet:**
```python
lo, hi = x.min(), x.max()
```
- **Snippet explained:** Range ends come from the data.
- **Related terms:** [[Symmetric quantization]], [[Scale and zero-point]]

### Per-group (block-wise) quantization
<!-- cat: llm -->
- **Technical definition:** A separate scale for each group of g consecutive weights (g = 32, 64, 128).
- **Plain English:** One ruler per small block.
- **Example / analogy:** int4 g128 with fp16 scales costs 4.125 bits per weight.
- **Snippet:**
```python
w.reshape(-1, 128)
```
- **Snippet explained:** Rows of 128 weights each get their own scale.
- **Related terms:** [[Bits per weight (bpw)]], [[Per-channel quantization]]

### Per-channel quantization
<!-- cat: llm -->
- **Technical definition:** One scale per output channel (row of a weight matrix), or per head_dim position for a K cache.
- **Plain English:** One ruler per row.
- **Example / analogy:** Standard for int8 weights.
- **Snippet:**
```python
np.abs(W).max(axis=1, keepdims=True)
```
- **Snippet explained:** Row-wise absmax.
- **Related terms:** [[Per-group (block-wise) quantization]]

### Bits per weight (bpw)
<!-- cat: llm -->
- **Technical definition:** Average storage per weight including scale and zero-point overhead.
- **Plain English:** True bit cost once rulers are counted.
- **Example / analogy:** int4 with g128 and fp16 scale is 4.125 bpw.
- **Snippet:**
```python
4 + 16 / 128
```
- **Snippet explained:** Adds the scale bits per group.
- **Related terms:** [[Per-group (block-wise) quantization]]

### SQNR
<!-- cat: llm -->
- **Technical definition:** Signal-to-quantization-noise ratio: 10 log10 of signal energy over error energy, in dB.
- **Plain English:** Loudness of signal compared with rounding noise.
- **Example / analogy:** About 6 dB per extra bit.
- **Snippet:**
```python
10 * np.log10((x**2).sum() / (e**2).sum())
```
- **Snippet explained:** Decibel ratio of energies.
- **Related terms:** [[Affine quantization]], [[Per-group (block-wise) quantization]]

### FP8 (E4M3, E5M2)
<!-- cat: llm -->
- **Technical definition:** 8-bit floating-point formats: E4M3 (max 448) favours precision, E5M2 (max 57344) favours range.
- **Plain English:** A tiny float.
- **Example / analogy:** FP8 W8A8 is the fast path on Hopper-class GPUs.
- **Snippet:**
```python
x.to(torch.float8_e4m3fn)
```
- **Snippet explained:** Rounds to the nearest FP8 value.
- **Related terms:** [[NF4]], [[MXFP4 / NVFP4]]

### NF4 (NormalFloat4)
<!-- cat: llm -->
- **Technical definition:** A 4-bit codebook of 16 normal-distribution quantiles scaled to [-1, 1], used blockwise (64) in QLoRA.
- **Plain English:** Grid points placed where Gaussian weights are.
- **Example / analogy:** bitsandbytes load_in_4bit uses NF4.
- **Snippet:**
```python
BitsAndBytesConfig(bnb_4bit_quant_type='nf4')
```
- **Snippet explained:** Selects NF4 in the config.
- **Related terms:** [[Double quantization]], [[QLoRA]]

### Double quantization
<!-- cat: llm -->
- **Technical definition:** Quantizing the per-block scale factors themselves, to save about 0.37 bits per weight in QLoRA.
- **Plain English:** Compressing the rulers.
- **Example / analogy:** 4.5 to 4.127 bits per weight.
- **Snippet:**
```python
bnb_4bit_use_double_quant=True
```
- **Snippet explained:** Turns on scale quantization.
- **Related terms:** [[NF4 (NormalFloat4)]]

### MXFP4 / NVFP4
<!-- cat: llm -->
- **Technical definition:** Block-scaled FP4 (E2M1) formats: MXFP4 uses blocks of 32 with a power-of-two scale, NVFP4 blocks of 16 with an FP8 scale; emerging hardware formats.
- **Plain English:** Tiny floats sharing an exponent per block.
- **Example / analogy:** Blackwell GPUs accelerate NVFP4.
- **Snippet:**
```python
fp4_block_qdq(W, 16, 'e4m3')
```
- **Snippet explained:** Simulates NVFP4 rounding.
- **Related terms:** [[FP8 (E4M3, E5M2)]]

### Weight-only quantization (W4A16)
<!-- cat: llm -->
- **Technical definition:** Quantizing weights while keeping activations 16-bit.
- **Plain English:** Smaller model, same arithmetic.
- **Example / analogy:** GPTQ/AWQ checkpoints are W4A16.
- **Snippet:**
```python
W4 = ql.qdq(W, 4, True, 128)
```
- **Snippet explained:** Fake-quantizes weights only.
- **Related terms:** [[W8A8]], [[GPTQ]]

### W8A8
<!-- cat: llm -->
- **Technical definition:** Both weights and activations 8-bit so the matmul uses integer or FP8 tensor cores.
- **Plain English:** The fast but harder setting.
- **Example / analogy:** Needs SmoothQuant or per-token scales for LLMs.
- **Snippet:**
```python
act_qdq(X, 8, 'token') @ W8.T
```
- **Snippet explained:** Quantizes activations per token then multiplies.
- **Related terms:** [[SmoothQuant]], [[Activation outlier channels]]

### Activation outlier channels
<!-- cat: llm -->
- **Technical definition:** Hidden dimensions with magnitudes many times the median for every token.
- **Plain English:** Loud channels drowning the quiet ones.
- **Example / analogy:** distilgpt2 layer inputs reach 59x max/median.
- **Snippet:**
```python
np.abs(X).max(0).max() / np.median(np.abs(X).max(0))
```
- **Snippet explained:** Ratio of biggest to typical channel.
- **Related terms:** [[SmoothQuant]], [[LLM.int8()]]

### SmoothQuant
<!-- cat: llm -->
- **Technical definition:** Migrating activation outliers into the weights with s_j = max|X_j|^a / max|W_j|^(1-a).
- **Plain English:** Share difficulty between operands.
- **Example / analogy:** Typical alpha is 0.5.
- **Snippet:**
```python
Xs, Ws = X / s, W * s
```
- **Snippet explained:** Exact rescaling identity.
- **Related terms:** [[W8A8]], [[Activation outlier channels]]

### LLM.int8()
<!-- cat: llm -->
- **Technical definition:** Mixed-precision decomposition: outlier channels in fp16, the rest in int8.
- **Plain English:** Outliers get their own lane.
- **Example / analogy:** bitsandbytes load_in_8bit.
- **Snippet:**
```python
BitsAndBytesConfig(load_in_8bit=True)
```
- **Snippet explained:** Enables 8-bit loading.
- **Related terms:** [[SmoothQuant]]

### PTQ vs QAT
<!-- cat: llm -->
- **Technical definition:** Post-training quantization quantizes a finished model; quantization-aware training trains with simulated quantization using a straight-through estimator.
- **Plain English:** Round and ship versus practise with a handicap.
- **Example / analogy:** QAT recovers accuracy at 2 bits.
- **Snippet:**
```python
w + (wq - w).detach()
```
- **Snippet explained:** Straight-through estimator.
- **Related terms:** [[Straight-through estimator (STE)]]

### Straight-through estimator (STE)
<!-- cat: ml -->
- **Technical definition:** Using the identity as the backward pass of a non-differentiable rounding operation.
- **Plain English:** Pretending round has slope 1.
- **Example / analogy:** Used in every QAT implementation.
- **Snippet:**
```python
return w + (wq - w).detach()
```
- **Snippet explained:** Forward value wq, gradient of w.
- **Related terms:** [[PTQ vs QAT]]

### Calibration (static vs dynamic ranges)
<!-- cat: llm -->
- **Technical definition:** Estimating activation ranges from sample data offline (static), versus computing them at run time (dynamic).
- **Plain English:** Decide in advance versus measure now.
- **Example / analogy:** min-max, percentile, MSE rules.
- **Snippet:**
```python
static_range(Xcal, 'mse')
```
- **Snippet explained:** Picks a clipping range.
- **Related terms:** [[W8A8]], [[GPTQ]]

### GPTQ
<!-- cat: llm -->
- **Technical definition:** Layer-wise PTQ that rounds columns one at a time and compensates remaining columns using the inverse Hessian of calibration inputs.
- **Plain English:** Round a column, let the others cancel its error.
- **Example / analogy:** 3-bit perplexity 29.0 vs RTN 42.2 on distilgpt2.
- **Snippet:**
```python
W[:, b1:] -= Err @ Hinv[b0:b1, b1:]
```
- **Snippet explained:** Lazy block update of later columns.
- **Related terms:** [[AWQ]], [[RTN (round-to-nearest)]]

### AWQ
<!-- cat: llm -->
- **Technical definition:** Activation-aware weight quantization: per-input-channel scale s = mean|X|^alpha chosen by search before rounding.
- **Plain English:** Finer grid for channels with big activations.
- **Example / analogy:** Robust to the calibration set.
- **Snippet:**
```python
Wq = qdq(W * s) / s
```
- **Snippet explained:** Quantize scaled weight, undo scale.
- **Related terms:** [[GPTQ]], [[SmoothQuant]]

### RTN (round-to-nearest)
<!-- cat: llm -->
- **Technical definition:** Data-free quantization rounding each weight independently.
- **Plain English:** Round and hope.
- **Example / analogy:** The baseline for GPTQ/AWQ.
- **Snippet:**
```python
ql.qdq(W, 4, False, 128)
```
- **Snippet explained:** Group-wise min-max rounding.
- **Related terms:** [[GPTQ]], [[AWQ]]

### HQQ
<!-- cat: llm -->
- **Technical definition:** Half-Quadratic Quantization, calibration-free optimisation of scale and zero-point per group.
- **Plain English:** Better-than-RTN without data.
- **Example / analogy:** Quantizes huge models in minutes.
- **Snippet:**
```python
(reference)
```
- **Snippet explained:** Not run in this course.
- **Related terms:** [[RTN (round-to-nearest)]], [[GPTQ]]

### llm-compressor
<!-- cat: llm -->
- **Technical definition:** The vLLM project's library that applies GPTQ, AWQ and FP8 recipes and saves compressed-tensors checkpoints for vLLM.
- **Plain English:** The usual way to make a vLLM-ready quantized model.
- **Example / analogy:** oneshot(model, dataset, recipe).
- **Snippet:**
```python
oneshot(model=m, dataset=d, recipe=GPTQModifier(scheme='W4A16'))
```
- **Snippet explained:** Runs calibration and writes the checkpoint.
- **Related terms:** [[GPTQ]], [[AWQ]]

### KV-cache quantization (KIVI)
<!-- cat: llm -->
- **Technical definition:** Storing K and V in fewer bits; KIVI uses per-channel scales for K and per-token for V.
- **Plain English:** Compress the cache notebook.
- **Example / analogy:** vLLM --kv-cache-dtype fp8 halves cache memory.
- **Snippet:**
```python
vllm serve m --kv-cache-dtype fp8
```
- **Snippet explained:** Stores the cache as FP8.
- **Related terms:** [[KV cache]], [[FP8 (E4M3, E5M2)]]

### Lazy block update (GPTQ)
<!-- cat: llm -->
- **Technical definition:** Applying a block's accumulated column errors to all later columns with one matrix product.
- **Plain English:** Batching the bookkeeping.
- **Example / analogy:** About 9x faster here with nearly identical results.
- **Snippet:**
```python
W[:, b1:] -= Err @ Hinv[b0:b1, b1:]
```
- **Snippet explained:** One matmul instead of one rank-1 update per column.
- **Related terms:** [[GPTQ]]

### Inference mode (no_grad)
<!-- cat: ml -->
- **Technical definition:** Running a model without gradient tracking using model.eval() and torch.no_grad().
- **Plain English:** Using the model, not teaching it, so it skips bookkeeping.
- **Example / analogy:** Driving instead of learning to drive: no instructor notes.
- **Snippet:**
```python
model.eval()
with torch.no_grad():
    out = model.generate(**enc)
```
- **Snippet explained:** eval turns off dropout; no_grad saves memory and time.
- **Related terms:** [[Training loop]], [[KV cache]]

### Batching (inference)
<!-- cat: llm -->
- **Technical definition:** Processing several prompts in one forward pass to increase hardware utilisation.
- **Plain English:** Serving many customers at the same time.
- **Example / analogy:** A bus carrying many passengers on one trip.
- **Snippet:**
```python
enc = tok(prompts, return_tensors='pt', padding=True)
out = model.generate(**enc, max_new_tokens=30)
```
- **Snippet explained:** One generate call handles all prompts in the list together.
- **Related terms:** [[Throughput]], [[Latency]], [[Continuous batching (iteration-level scheduling)]]

### GPU memory estimate
<!-- cat: llm -->
- **Technical definition:** Back-of-envelope: weights = params x bytes/param; KV cache = 2 x layers x kv_heads x head_dim x tokens x batch x bytes; training AdamW about 16 bytes/param.
- **Plain English:** A quick sum to see if the model fits on your graphics card.
- **Example / analogy:** Packing a suitcase: add the weights of everything first.
- **Snippet:**
```python
kv = 2 * layers * kv_heads * head_dim * tokens * batch * 2   # fp16
```
- **Snippet explained:** 7B, 32 layers, 32 kv heads, 128 dim, 4096 tokens, batch 8 gives ~17 GB.
- **Related terms:** [[KV cache]], [[Quantization (weights)]], [[Grouped-query attention (GQA)]]

### Prefill (prompt processing)
<!-- cat: llm -->
- **Technical definition:** One parallel forward pass over all prompt tokens that fills the KV cache and yields the first output token; compute-bound.
- **Plain English:** Reading the whole question before answering.
- **Example / analogy:** A 500-token prompt is one forward pass, not 500.
- **Snippet:**
```python
out = model(input_ids=ids, use_cache=True)
```
- **Snippet explained:** One call over the entire prompt returns logits and past_key_values.
- **Related terms:** [[Decode (token generation)]], [[TTFT (time to first token)]], [[KV cache (key-value cache, past_key_values)]]

### Decode (token generation)
<!-- cat: llm -->
- **Technical definition:** The autoregressive loop producing one token per forward pass using the KV cache; memory-bound because all weights are re-read for one token of work.
- **Plain English:** Writing the answer one word at a time.
- **Example / analogy:** Step time at batch 1 is set by weight bytes divided by bandwidth.
- **Snippet:**
```python
out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
```
- **Snippet explained:** Feeds only the newest token plus the cache.
- **Related terms:** [[Prefill (prompt processing)]], [[Memory-bound vs compute-bound]], [[ITL (inter-token latency)]]

### Arithmetic intensity
<!-- cat: ml -->
- **Technical definition:** FLOPs performed per byte moved from memory; the x-axis of the roofline.
- **Plain English:** Math done per byte fetched.
- **Example / analogy:** Decode intensity is about 2B/bytes_per_param.
- **Snippet:**
```python
intensity = 2 * batch / bytes_per_param
```
- **Snippet explained:** fp16 weights give intensity equal to the batch size.
- **Related terms:** [[Roofline model]], [[Ridge point (machine balance)]]

### Roofline model
<!-- cat: ml -->
- **Technical definition:** Attainable FLOP/s = min(peak FLOP/s, memory bandwidth x arithmetic intensity).
- **Plain English:** Two ceilings; you hit whichever is lower.
- **Example / analogy:** A100: 312 TFLOP/s and 2 TB/s.
- **Snippet:**
```python
attainable = min(peak, bw * intensity)
```
- **Snippet explained:** Compute roof and bandwidth slope in one expression.
- **Related terms:** [[Arithmetic intensity]], [[Ridge point (machine balance)]]

### Ridge point (machine balance)
<!-- cat: ml -->
- **Technical definition:** Intensity where compute and bandwidth roofs meet: peak FLOP/s divided by bandwidth (A100 about 153, H100 about 295 FLOP/byte).
- **Plain English:** The break-even FLOPs per byte.
- **Example / analogy:** Below the ridge you wait for memory.
- **Snippet:**
```python
ridge = 312e12 / 2.039e12
```
- **Snippet explained:** About 153 FLOP per byte for A100.
- **Related terms:** [[Roofline model]], [[Memory-bound vs compute-bound]]

### Memory-bound vs compute-bound
<!-- cat: ml -->
- **Technical definition:** Memory-bound: runtime set by bytes/bandwidth. Compute-bound: set by FLOPs/peak FLOP/s.
- **Plain English:** Waiting for delivery vs waiting for the calculator.
- **Example / analogy:** Decode at batch 1 is memory-bound; prefill is compute-bound.
- **Snippet:**
```python
bound = 'memory' if intensity < ridge else 'compute'
```
- **Snippet explained:** Compare intensity with the ridge.
- **Related terms:** [[Arithmetic intensity]], [[Prefill (prompt processing)]]

### TTFT (time to first token)
<!-- cat: llm -->
- **Technical definition:** Time from request start to first output token: queueing + prefill + first sampling.
- **Plain English:** How long until the answer starts.
- **Example / analogy:** Driven by prompt length, queueing and cache hits.
- **Snippet:**
```python
ttft = stamps[0] - t0
```
- **Snippet explained:** First timestamp minus start.
- **Related terms:** [[ITL (inter-token latency)]], [[E2E latency]]

### ITL (inter-token latency)
<!-- cat: llm -->
- **Technical definition:** Gap between two consecutive output tokens; report p50/p95/p99.
- **Plain English:** Smoothness of the stream.
- **Example / analogy:** One stall of 120 ms among 20 ms gaps.
- **Snippet:**
```python
itl = [b - a for a, b in zip(stamps, stamps[1:])]
```
- **Snippet explained:** Differences of neighbouring timestamps.
- **Related terms:** [[TPOT (time per output token)]], [[Percentiles p50/p95/p99]]

### TPOT (time per output token)
<!-- cat: llm -->
- **Technical definition:** (E2E - TTFT) / (N - 1): the mean ITL, excluding TTFT.
- **Plain English:** Average per-word time after the first word.
- **Example / analogy:** TPOT 45 ms equals 22 tokens/s per user.
- **Snippet:**
```python
tpot = (e2e - ttft) / (n - 1)
```
- **Snippet explained:** Decode time divided by the number of decode steps.
- **Related terms:** [[ITL (inter-token latency)]], [[E2E latency]]

### E2E latency
<!-- cat: llm -->
- **Technical definition:** Time from request start to its last token.
- **Plain English:** Total wait.
- **Example / analogy:** TTFT + (N-1) x TPOT.
- **Snippet:**
```python
e2e = stamps[-1] - t0
```
- **Snippet explained:** Last timestamp minus start.
- **Related terms:** [[TTFT (time to first token)]], [[TPOT (time per output token)]]

### Percentiles p50/p95/p99
<!-- cat: ml -->
- **Technical definition:** Values below which 50, 95, 99 percent of samples fall.
- **Plain English:** Tail behaviour that the mean hides.
- **Example / analogy:** p95 ITL 105 ms vs mean 45 ms.
- **Snippet:**
```python
np.percentile(itl, 95)
```
- **Snippet explained:** Linear-interpolated 95th percentile.
- **Related terms:** [[ITL (inter-token latency)]], [[Goodput (SLO-attaining throughput)]]

### Goodput (SLO-attaining throughput)
<!-- cat: llm -->
- **Technical definition:** Throughput counting only requests that met their latency SLO (for example TTFT and TPOT limits).
- **Plain English:** Useful throughput.
- **Example / analogy:** Overloaded server: high tokens/s, goodput near zero.
- **Snippet:**
```python
good = [r for r in reqs if r.ttft < 0.4 and r.tpot < 0.04]
```
- **Snippet explained:** Filters requests by both targets.
- **Related terms:** [[TTFT (time to first token)]], [[TPOT (time per output token)]]

### Benchmark warm-up
<!-- cat: llm -->
- **Technical definition:** Untimed runs before measuring so lazy initialisation, allocators and caches do not distort timing.
- **Plain English:** Run it once to warm the engine.
- **Example / analogy:** Skipping it makes the first row look slow.
- **Snippet:**
```python
for _ in range(2): run()
```
- **Snippet explained:** Discard two runs, then time.
- **Related terms:** [[Goodput (SLO-attaining throughput)]]

### PagedAttention
<!-- cat: llm -->
- **Technical definition:** vLLM's KV-cache manager that stores cache in fixed-size blocks allocated on demand, like virtual-memory pages.
- **Plain English:** Hand out memory in small pages instead of reserving a huge slab per user.
- **Example / analogy:** A hotel assigning rooms as guests arrive instead of reserving whole floors.
- **Snippet:**
```python
llm = LLM(model='Qwen/Qwen2.5-0.5B-Instruct', gpu_memory_utilization=0.9)
```
- **Snippet explained:** vLLM pre-allocates KV pages within 90% of GPU memory and hands them out per request.
- **Related terms:** [[KV cache]], [[Continuous batching (iteration-level scheduling)]], [[vLLM]]

### Continuous batching (iteration-level scheduling)
<!-- cat: llm -->
- **Technical definition:** The scheduler admits and retires requests at every decode iteration, so finished requests free slots immediately.
- **Plain English:** Seating guests as soon as a table frees.
- **Example / analogy:** Cut mean TTFT from 4.0 s to 0.06 s in a simulation.
- **Snippet:**
```python
while waiting or running: admit(); step(); retire()
```
- **Snippet explained:** One loop iteration is one decode step for everyone.
- **Related terms:** [[Static batching]], [[Chunked prefill]], [[PagedAttention]]

### Static batching
<!-- cat: llm -->
- **Technical definition:** A fixed batch is held until its longest request finishes.
- **Plain English:** A bus that waits for the slowest passenger.
- **Example / analogy:** Wastes slots on short requests.
- **Snippet:**
```python
steps = max(r.out for r in batch)
```
- **Snippet explained:** The batch runs as long as its longest member.
- **Related terms:** [[Continuous batching (iteration-level scheduling)]]

### Chunked prefill
<!-- cat: llm -->
- **Technical definition:** Splitting a long prompt into token-budget chunks mixed with decode tokens each iteration so decoders do not stall.
- **Plain English:** Cut the big job into slices.
- **Example / analogy:** Worst inter-token gap fell from 500 ms to 64 ms in simulation.
- **Snippet:**
```python
take = min(todo, budget - used)
```
- **Snippet explained:** Caps prompt tokens per iteration.
- **Related terms:** [[Prefill (prompt processing)]], [[Continuous batching (iteration-level scheduling)]]

### Disaggregated prefill/decode
<!-- cat: llm -->
- **Technical definition:** Running prefill and decode on separate GPU pools and transferring KV cache between them.
- **Plain English:** Prep kitchen and cooking line.
- **Example / analogy:** Tune TTFT and ITL independently.
- **Snippet:**
```python
kv = prefill_worker(prompt); decode_worker(kv)
```
- **Snippet explained:** Prefill ships the KV cache to decode.
- **Related terms:** [[Prefill (prompt processing)]], [[Decode (token generation)]]

### Automatic prefix caching
<!-- cat: llm -->
- **Technical definition:** Reusing KV blocks of identical token prefixes across requests, keyed by a chained hash of 16-token blocks (vLLM APC).
- **Plain English:** Do not redo the shared intro.
- **Example / analogy:** Shared system prompt gives 85 percent block hits.
- **Snippet:**
```python
h = hash((h, tuple(block)))
```
- **Snippet explained:** Chained so a hit means the whole prefix matches.
- **Related terms:** [[Chained block hash]], [[RadixAttention (radix tree prefix cache)]]

### Chained block hash
<!-- cat: llm -->
- **Technical definition:** Hash of (previous block hash, this block's tokens), making each key depend on the entire prefix.
- **Plain English:** A fingerprint that includes everything before it.
- **Example / analogy:** Without chaining, different contexts could share KV.
- **Snippet:**
```python
h = hash((h, tuple(tokens[j*16:(j+1)*16])))
```
- **Snippet explained:** Chains blocks via h.
- **Related terms:** [[Automatic prefix caching]]

### RadixAttention (radix tree prefix cache)
<!-- cat: llm -->
- **Technical definition:** SGLang's tree over tokens of cached prefixes with LRU eviction of leaves; matches at token granularity.
- **Plain English:** A phone-book tree of prompts.
- **Example / analogy:** Matches 500 shared tokens where block hashing matches 496.
- **Snippet:**
```python
node = node.children[token]
```
- **Snippet explained:** Walks the tree one token at a time.
- **Related terms:** [[Automatic prefix caching]], [[LRU eviction]]

### LRU eviction
<!-- cat: python -->
- **Technical definition:** Evicting the least recently used entry; OrderedDict.move_to_end plus popitem(last=False).
- **Plain English:** Throw out what you touched longest ago.
- **Example / analogy:** A cache smaller than the working set thrashes.
- **Snippet:**
```python
d.move_to_end(k); d.popitem(last=False)
```
- **Snippet explained:** Touch moves to the end; evict pops the front.
- **Related terms:** [[RadixAttention (radix tree prefix cache)]]

### Speculative decoding
<!-- cat: llm -->
- **Technical definition:** A cheap draft proposes k tokens; the target verifies them in one forward pass and keeps the longest correct prefix plus one token; output equals the target's.
- **Plain English:** A junior drafts, the senior checks a paragraph at once.
- **Example / analogy:** Lossless for greedy and exact for sampling with the rejection rule.
- **Snippet:**
```python
k = int(agree.cumprod(0).sum())
```
- **Snippet explained:** Counts the leading run of agreement.
- **Related terms:** [[Acceptance rate]], [[Medusa / EAGLE-3 / MTP draft heads]]

### Acceptance rate
<!-- cat: llm -->
- **Technical definition:** Fraction of draft tokens the target accepts; expected tokens per target pass is (1 - a^(k+1))/(1 - a).
- **Plain English:** How good the draft is.
- **Example / analogy:** a = 0.8, k = 4 gives 3.36 tokens per pass.
- **Snippet:**
```python
(1 - a**(k+1)) / (1 - a)
```
- **Snippet explained:** Truncated geometric series.
- **Related terms:** [[Speculative decoding]]

### Rejection-sampling acceptance rule
<!-- cat: ml -->
- **Technical definition:** Accept draft token x with probability min(1, p(x)/q(x)); on rejection resample from norm(max(0, p - q)); the output is distributed exactly as p.
- **Plain English:** Correcting a biased proposal exactly.
- **Example / analogy:** Empirical frequencies matched the target on a toy distribution.
- **Snippet:**
```python
if rng.random() < min(1, p[x] / q[x]): accept
```
- **Snippet explained:** Accept with probability min(1, p/q).
- **Related terms:** [[Speculative decoding]]

### Prompt lookup (n-gram speculation)
<!-- cat: llm -->
- **Technical definition:** Draft tokens copied from the continuation of a matching n-gram already in the context; needs no draft model.
- **Plain English:** Guess that the answer repeats the text above.
- **Example / analogy:** Helps summarisation and code editing.
- **Snippet:**
```python
# find last n-gram in prompt, copy what followed
```
- **Snippet explained:** No model call.
- **Related terms:** [[Speculative decoding]]

### Medusa / EAGLE-3 / MTP draft heads
<!-- cat: llm -->
- **Technical definition:** Extra prediction heads on the target (Medusa), a feature-level draft head (EAGLE-3), or multi-token-prediction heads trained with the model (MTP, DeepSeek-V3) used as speculation proposers.
- **Plain English:** Built-in guessers instead of a separate draft LM.
- **Example / analogy:** Verification is the same as for draft models.
- **Snippet:**
```python
# heads predict t+1..t+k from hidden states
```
- **Snippet explained:** Several tokens guessed per target pass.
- **Related terms:** [[Speculative decoding]], [[Acceptance rate]]

### Tensor parallelism (TP)
<!-- cat: llm -->
- **Technical definition:** Splitting each weight matrix of a layer across several GPUs so every layer is computed jointly; needs all-reduce or all-gather after matmuls.
- **Plain English:** Each GPU does a slice of every layer, then they pool results.
- **Example / analogy:** Four cooks each chop a quarter of the vegetables for every dish, then pass the bowls around.
- **Snippet:**
```python
tp = 4  # vllm serve --tensor-parallel-size 4
```
- **Snippet explained:** Declares that each layer is sharded over 4 GPUs; every layer then needs collectives.
- **Related terms:** [[Column-parallel linear]], [[Row-parallel linear]], [[All-reduce]], [[Pipeline parallelism (PP)]]

### Pipeline parallelism (PP)
<!-- cat: llm -->
- **Technical definition:** Placing consecutive groups of layers on different devices and streaming micro-batches through them; communication only at stage boundaries.
- **Plain English:** An assembly line: GPU 0 runs layers 0-15, GPU 1 runs 16-31.
- **Example / analogy:** Car factory stations.
- **Snippet:**
```python
bubble = (p-1)/(m+p-1)
```
- **Snippet explained:** Fraction of time stages idle while the pipeline fills and drains.
- **Related terms:** [[Pipeline bubble]], [[Tensor parallelism (TP)]], [[Micro-batch]]

### Pipeline bubble
<!-- cat: ml -->
- **Technical definition:** Idle time of pipeline stages during fill and drain, equal to (p-1)/(m+p-1) for p stages and m micro-batches.
- **Plain English:** The dead time at the start and end of the assembly line.
- **Example / analogy:** The first station waits for work only at start; the last station is idle until the first item arrives.
- **Snippet:**
```python
print((4-1)/(16+4-1))  # 0.158
```
- **Snippet explained:** Four stages and 16 micro-batches waste about 16% of the time.
- **Related terms:** [[Pipeline parallelism (PP)]], [[Micro-batch]]

### Micro-batch
<!-- cat: ml -->
- **Technical definition:** A small slice of a batch fed through a pipeline so that stages can work on different slices at the same time.
- **Plain English:** Chop the batch into smaller pieces so nobody waits.
- **Example / analogy:** Cutting a big cake into slices so every station has something to do.
- **Snippet:**
```python
chunks = batch.chunk(8)
```
- **Snippet explained:** Splits the batch into 8 micro-batches for pipeline scheduling.
- **Related terms:** [[Pipeline bubble]], [[Pipeline parallelism (PP)]]

### Column-parallel linear
<!-- cat: ml -->
- **Technical definition:** Splitting a weight matrix W by output columns so each device computes a slice of Y = XW from the full X; slices are concatenated.
- **Plain English:** Each GPU produces some of the output columns.
- **Example / analogy:** Each cook makes some of the dishes from the same ingredients.
- **Snippet:**
```python
Y = np.concatenate([X @ w for w in np.split(W, N, axis=1)], axis=1)
```
- **Snippet explained:** Concatenating per-device outputs equals the unsplit result.
- **Related terms:** [[Row-parallel linear]], [[All-gather]], [[Megatron MLP pattern]]

### Row-parallel linear
<!-- cat: ml -->
- **Technical definition:** Splitting W by input rows (and X by columns) so each device produces a partial sum of the full-width output; an all-reduce adds them.
- **Plain English:** Each GPU produces an incomplete full-size answer; add them up.
- **Example / analogy:** Each cook makes a part of every dish; assemble by summing.
- **Snippet:**
```python
Y = np.sum([x @ w for x, w in zip(np.split(X, N, 1), np.split(W, N, 0))], axis=0)
```
- **Snippet explained:** Summing the partial products equals X @ W.
- **Related terms:** [[Column-parallel linear]], [[All-reduce]]

### Megatron MLP pattern
<!-- cat: llm -->
- **Technical definition:** Column-parallel first MLP matrix, elementwise activation on the local slice, row-parallel second matrix: one all-reduce per MLP block.
- **Plain English:** Cut the first matrix by columns and the second by rows so GPUs only talk once.
- **Example / analogy:** The slice one step produces is exactly what the next step consumes.
- **Snippet:**
```python
out = sum(gelu(X @ a) @ b for a, b in zip(A_cols, B_rows))
```
- **Snippet explained:** Each device does its own slice of the MLP; the sum is the single all-reduce.
- **Related terms:** [[Tensor parallelism (TP)]], [[All-reduce]]

### All-reduce
<!-- cat: ml -->
- **Technical definition:** A collective that sums a tensor held in partial form on every device and leaves the full sum on every device.
- **Plain English:** Everyone contributes a piece and everyone gets the total.
- **Example / analogy:** A group adds up their bills and each person receives the grand total.
- **Snippet:**
```python
dist.all_reduce(t, op=dist.ReduceOp.SUM)
```
- **Snippet explained:** torch.distributed call that sums t across ranks in place.
- **Related terms:** [[All-gather]], [[All-to-all]]

### All-gather
<!-- cat: ml -->
- **Technical definition:** A collective where every device receives the concatenation of all devices' slices.
- **Plain English:** Everyone ends up with the whole picture.
- **Example / analogy:** Each person holds one puzzle piece; afterwards everyone holds the whole puzzle.
- **Snippet:**
```python
dist.all_gather(out_list, t)
```
- **Snippet explained:** Fills out_list with each rank's tensor on every rank.
- **Related terms:** [[All-reduce]], [[Column-parallel linear]]

### All-to-all
<!-- cat: ml -->
- **Technical definition:** A collective where each device sends a different chunk to every other device; used to dispatch tokens to experts in MoE.
- **Plain English:** Everyone swaps a different piece with everyone else.
- **Example / analogy:** Students each hand every other student a different card.
- **Snippet:**
```python
dist.all_to_all_single(out, inp)
```
- **Snippet explained:** Redistributes chunks between ranks.
- **Related terms:** [[Expert parallelism (EP)]], [[All-reduce]]

### Expert parallelism (EP)
<!-- cat: llm -->
- **Technical definition:** Placing different MoE experts on different devices; tokens are routed to the owning device (all-to-all) and results returned.
- **Plain English:** Experts live on separate GPUs; tokens travel to them.
- **Example / analogy:** Specialist doctors in different buildings; patients are sent to the right one.
- **Snippet:**
```python
top1 = router(x).argmax(-1); dev = top1 // experts_per_dev
```
- **Snippet explained:** Chooses the device holding each token's expert.
- **Related terms:** [[All-to-all]], [[Mixture of experts (MoE)]]

### Sequence parallelism / context parallelism
<!-- cat: llm -->
- **Technical definition:** Splitting the sequence dimension of activations or KV cache across devices (for example ring attention) for contexts too long for one device.
- **Plain English:** Cut a very long prompt into pieces across GPUs.
- **Example / analogy:** Several readers each hold a chapter of a very long book.
- **Snippet:**
```python
# ring attention passes K,V blocks around the ring
```
- **Snippet explained:** Each device attends to its block then receives the next K,V block from its neighbour.
- **Related terms:** [[Tensor parallelism (TP)]], [[KV cache]]

### ZeRO / FSDP
<!-- cat: ml -->
- **Technical definition:** Training memory optimisation that shards optimizer state (stage 1), gradients (2) and parameters (3) across data-parallel ranks, gathering weights just in time.
- **Plain English:** Instead of every GPU storing everything, each stores a share.
- **Example / analogy:** A library where each branch keeps a share of the books and borrows others when needed.
- **Snippet:**
```python
per_gpu = params * (2/N + 2/N + 12/N)  # ZeRO-3
```
- **Snippet explained:** Memory per GPU divides by N in the ZeRO-3 case.
- **Related terms:** [[Data parallelism (DP)]], [[Tensor parallelism (TP)]]

### Data parallelism (DP)
<!-- cat: ml -->
- **Technical definition:** Replicating the whole model on several devices and splitting the batch or the requests between them.
- **Plain English:** Many copies of the same model, each serving different inputs.
- **Example / analogy:** Opening more checkout lanes.
- **Snippet:**
```python
replicas = 4  # each serves 1/4 of requests
```
- **Snippet explained:** Scales throughput linearly with no communication when serving.
- **Related terms:** [[ZeRO / FSDP]], [[Tensor parallelism (TP)]]

### Offloading (CPU / disk)
<!-- cat: llm -->
- **Technical definition:** Keeping part of a model's weights in CPU RAM or on disk and moving them to the accelerator when needed, trading latency for capacity.
- **Plain English:** Run a model bigger than your GPU by paging layers in from slower memory.
- **Example / analogy:** Reading a book from a shelf across the room each page.
- **Snippet:**
```python
AutoModelForCausalLM.from_pretrained(n, device_map='auto', max_memory={'cpu': '30GiB'}, offload_folder='off')
```
- **Snippet explained:** Accelerate places modules by budget and spills the rest to RAM/disk.
- **Related terms:** [[device_map]], [[-ngl (n-gpu-layers)]]

### device_map
<!-- cat: python -->
- **Technical definition:** Hugging Face from_pretrained argument (and model.hf_device_map attribute) that maps each module to a device (GPU index, 'cpu' or 'disk'); 'auto' computes it from max_memory.
- **Plain English:** A dict saying which layer lives where.
- **Example / analogy:** A seating chart for layers.
- **Snippet:**
```python
print(model.hf_device_map)
```
- **Snippet explained:** Shows the module-to-device placement Accelerate chose.
- **Related terms:** [[Offloading (CPU / disk)]], [[-ngl (n-gpu-layers)]]

### -ngl (n-gpu-layers)
<!-- cat: llm -->
- **Technical definition:** llama.cpp flag that offloads the first N transformer layers to the GPU while the rest run on the CPU.
- **Plain English:** How many layers go on the graphics card.
- **Example / analogy:** Putting as many boxes as fit on the fast shelf.
- **Snippet:**
```python
llama-server -m model.gguf -ngl 99
```
- **Snippet explained:** Offloads up to 99 layers (effectively all).
- **Related terms:** [[GGUF]], [[Offloading (CPU / disk)]], [[llama.cpp]]

### llama.cpp
<!-- cat: llm -->
- **Technical definition:** C/C++ inference engine on the ggml library that runs GGUF models on CPU, Metal, CUDA, Vulkan and more, with llama-server for an OpenAI-compatible API.
- **Plain English:** The go-to engine for running quantized models locally.
- **Example / analogy:** The pocket knife of LLM inference.
- **Snippet:**
```python
llama-bench -m model.gguf -ngl 99
```
- **Snippet explained:** Measures prompt-processing and generation tokens per second.
- **Related terms:** [[GGUF]], [[Ollama]], [[-ngl (n-gpu-layers)]]

### Ollama
<!-- cat: llm -->
- **Technical definition:** A local daemon wrapping llama.cpp/ggml with a model registry and a REST API on port 11434 (plus an OpenAI-compatible /v1).
- **Plain English:** One-command local LLMs.
- **Example / analogy:** The app store plus runtime for local models.
- **Snippet:**
```python
curl localhost:11434/api/tags
```
- **Snippet explained:** Lists models already installed (does not download).
- **Related terms:** [[llama.cpp]], [[OpenAI-compatible API]]

### vLLM
<!-- cat: llm -->
- **Technical definition:** GPU LLM serving engine with PagedAttention, continuous batching, chunked prefill, automatic prefix caching and an OpenAI-compatible server.
- **Plain English:** The default open-source engine for serving many users.
- **Example / analogy:** A well-run restaurant kitchen compared with cooking at home.
- **Snippet:**
```python
vllm serve <model> --tensor-parallel-size 2
```
- **Snippet explained:** Starts an OpenAI-compatible server sharded over 2 GPUs.
- **Related terms:** [[PagedAttention]], [[Continuous batching (iteration-level scheduling)]], [[SGLang]]

### SGLang
<!-- cat: llm -->
- **Technical definition:** LLM serving engine with RadixAttention prefix cache, a CPU-overlapped scheduler, strong structured-output support and a frontend language.
- **Plain English:** Like vLLM, but especially good when prompts share prefixes.
- **Example / analogy:** A kitchen that remembers every half-prepared sauce.
- **Snippet:**
```python
python -m sglang.launch_server --model-path <m> --port 30000
```
- **Snippet explained:** Starts an OpenAI-compatible SGLang server.
- **Related terms:** [[RadixAttention (radix tree prefix cache)]], [[vLLM]]

### TensorRT-LLM
<!-- cat: llm -->
- **Technical definition:** NVIDIA's optimised LLM inference library with fused kernels, in-flight batching, paged KV cache and FP8/NVFP4 quantization; NVIDIA GPUs only.
- **Plain English:** Maximum speed on NVIDIA at the cost of flexibility.
- **Example / analogy:** A tuned race car versus a family car.
- **Snippet:**
```python
trtllm-serve <model> --port 8000
```
- **Snippet explained:** Starts an OpenAI-compatible TensorRT-LLM server.
- **Related terms:** [[vLLM]]

### OpenAI-compatible API
<!-- cat: api -->
- **Technical definition:** An HTTP interface that mimics the OpenAI /v1/chat/completions request and response format, offered by vLLM, SGLang, Ollama, llama-server and others.
- **Plain English:** Same JSON shape, so you swap only the base URL.
- **Example / analogy:** A universal power plug.
- **Snippet:**
```python
OpenAI(base_url='http://localhost:8000/v1', api_key='none')
```
- **Snippet explained:** Points the standard client at a local engine.
- **Related terms:** [[vLLM]], [[Ollama]]

### Cost per million tokens
<!-- cat: llm -->
- **Technical definition:** GPU dollars per hour divided by (tokens per second times 3600 times utilisation) times one million; priced separately for input and output tokens.
- **Plain English:** What one million tokens cost you to produce.
- **Example / analogy:** Cost per kilometre of a taxi: hourly rate over distance driven.
- **Snippet:**
```python
cost = price_h / (tps * 3600 * util) * 1e6
```
- **Snippet explained:** Converts a rental price and throughput into $/1M tokens.
- **Related terms:** [[Utilisation]], [[Break-even volume]]

### Utilisation
<!-- cat: ml -->
- **Technical definition:** The fraction of paid time an accelerator does useful work; effective cost per token is divided by it.
- **Plain English:** How often the GPU is actually busy.
- **Example / analogy:** A shop open ten hours but with customers for two.
- **Snippet:**
```python
eff_cost = cost / 0.2
```
- **Snippet explained:** At 20% busy each token costs 5x as much.
- **Related terms:** [[Cost per million tokens]], [[Break-even volume]]

### Break-even volume
<!-- cat: ml -->
- **Technical definition:** Monthly token volume above which fixed-cost self-hosting is cheaper than a pay-per-token API: fixed monthly cost divided by the API's blended price per token.
- **Plain English:** The point where buying beats renting.
- **Example / analogy:** Buying a car versus taking taxis.
- **Snippet:**
```python
be = n_gpu * price_h * 730 / blended_price_per_mtok
```
- **Snippet explained:** Gives millions of tokens per month.
- **Related terms:** [[Cost per million tokens]], [[Utilisation]]
