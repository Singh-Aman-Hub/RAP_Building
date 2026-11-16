# Interview Gist — Steps 04 to 11
### LoRA → peft → QLoRA → Fast inference → GPU memory → Throughput → vLLM → PagedAttention

This is a single-file theory sheet for revision. Each section gives:
**the problem → the idea → the maths/mechanics → numbers from your own runs → trade-offs → likely interview questions.**
The last part has a one-page cheat sheet and a rapid-fire Q&A.

---

## 0. The big picture (read this first)

The eight steps tell one story in two halves.

**Half 1: training cheaply (04–06)**
- Full fine-tuning updates every weight. With AdamW that costs about **16 bytes per parameter**, so a 7B model needs about **107 GB**.
- **LoRA (04, 05):** freeze the model and learn a small low-rank *update* instead. That leaves about 0.1–1% of parameters trainable.
- **QLoRA (06):** additionally store the frozen base in **4-bit**, so a 7B fine-tune fits on one 16 GB GPU.

**Half 2: serving fast (07–11)**
- **07:** the basic speed-ups: `no_grad`, the **KV cache**, **batching** (with **left padding**), and lower **dtype**.
- **08:** memory is arithmetic: **weights + KV cache + overhead**. The KV cache can be *bigger than the weights*.
- **09:** speed has two faces, **latency** (one user) and **throughput** (the whole system). Batching trades one for the other.
- **10:** **vLLM** adds **PagedAttention** and **continuous batching** to fix KV-cache waste and idle batch slots.
- **11:** a simulation proves it: the same 8 GB holds **8 requests contiguously and 53 paged (6.6×)**.

One sentence to hold it all together:
> "Training cost is dominated by gradients and optimizer state, so LoRA/QLoRA shrink what you train and what you store. Serving cost is dominated by the KV cache and idle hardware, so caching, batching, paging and continuous batching keep the GPU full of useful work."

---

## 1. LoRA — the maths (Step 04)

### 1.1 The problem
A single 4096×4096 projection matrix has **16.8M** parameters. Full fine-tuning:
- updates every weight,
- stores a gradient for every weight,
- stores optimizer state (Adam keeps 2 extra numbers per weight),
- saves a **full copy of the model per task**.

But the *useful change* for one task has low "intrinsic dimension". Research (Aghajanyan et al., then Hu et al. 2021, the LoRA paper) found that the weight update ΔW during fine-tuning is approximately **low-rank**.

### 1.2 The idea
Freeze the pretrained weight W₀ and write the update as a product of two thin matrices:

```
W = W₀ + ΔW = W₀ + (α / r) · B · A

A : shape (r, d_in)     "down-projection"  (squeeze input to r numbers)
B : shape (d_out, r)    "up-projection"    (expand back to output size)
r : rank, e.g. 4, 8, 16, 64       (r << d)
α : lora_alpha, a scaling constant
```

Forward pass:
```
y = W₀ x + (α/r) · B (A x)
```

### 1.3 Why so few parameters
| | Count | 2048×2048 example | 4096×4096 example |
|---|---|---|---|
| Full ΔW | d_out · d_in | 4,194,304 | 16,777,216 |
| LoRA (r=8) | r · (d_in + d_out) | **32,768 (0.78%)** | 65,536 (0.39%) |

Whole 7B model, r=8 on q, k, v, o in 32 layers: 32 × 4 × (2 × 4096 × 8) ≈ **8.4M trainable ≈ 0.12%** of 7B.

### 1.4 Initialisation (common interview question)
- **A** = small random (Gaussian / Kaiming).
- **B** = **zeros**.
- So B·A = 0 at the start, and the adapted model is *exactly* the pretrained model (your run: max diff = **0.0**). Training starts from the known-good point and moves away gradually.
- If both were zero, gradients would be zero for both (symmetry/dead start). If B were random, you would inject noise into a good model on step 0.
- Subtle point: at step 0, grad of A is zero (it is multiplied by B = 0), but grad of B is non-zero (it depends on A·x). So B moves first, and then A starts learning.

### 1.5 The α/r scaling
- The adapter output is multiplied by **α / r**.
- Purpose: when you change r, you don't have to re-tune the learning rate as much, because the update magnitude stays roughly stable.
- Common recipe: **α = 2r** (r=8, α=16 → scale 2.0), or α = r (scale 1).
- Variant to name-drop: **rsLoRA** uses α/√r, which works better at large ranks.

### 1.6 Order of multiplication
Compute `B @ (A @ x)`, **not** `(B @ A) @ x`.
- `A @ x` is tiny (r × batch).
- `(B @ A)` materialises a full d × d matrix, which is exactly the memory LoRA was meant to avoid.

### 1.7 Merging
After training: `W_merged = W₀ + (α/r) · B · A`.
- The model becomes one ordinary matrix per layer again, so there is **zero extra inference latency**.
- In your run, the merged and unmerged outputs differ only by ~1e-14 (float rounding).
- Trade-off: once merged, you can't hot-swap adapters. Keep them **unmerged** if you serve many adapters on one base (multi-LoRA serving: S-LoRA, vLLM `--enable-lora`).

### 1.8 Rank = capacity
- In your toy run the target change had rank 4.
  - **r=1** → the error plateaus above zero. The capacity is too low.
  - **r=4** → it fits (error 3.0181 → 0.0184).
  - **r=16** → fits as well, but costs 4× the adapter parameters.
- Higher r means more capacity, linearly more parameters, and more risk of overfitting. Lower *training* loss does not guarantee better generalisation.

### 1.9 Why LoRA saves memory (be precise)
LoRA does **not** shrink the forward-pass weights. The frozen W₀ is still loaded. The savings are:
1. **No gradients** for frozen weights.
2. **No optimizer state** (Adam m and v) for frozen weights. This is the big one: 12 of the 16 bytes per parameter.
3. **Tiny checkpoints:** only A and B are saved (MBs, not GBs).

Activations still need memory, because the backward pass must flow *through* the frozen layers to reach earlier adapters.

> **Say it aloud:** "LoRA freezes W₀ and learns ΔW as B·A with a small rank r, so I train under 1% of the parameters. B starts at zero, so training begins exactly at the pretrained model. After training I can merge B·A into W₀ for zero-overhead inference."

---

## 2. LoRA in practice with `peft` (Step 05)

### 2.1 The workflow
```python
config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                    target_modules=["c_attn"], fan_in_fan_out=True,
                    task_type="CAUSAL_LM")
model = get_peft_model(base, config)      # freeze base, inject A/B
model.print_trainable_parameters()        # the headline number
optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
# ... same training loop as full fine-tuning ...
model.save_pretrained(dir)                # saves ONLY adapter (0.6 MB)
reloaded = PeftModel.from_pretrained(fresh_base, dir)
merged = reloaded.merge_and_unload()      # plain model again
```

### 2.2 Key hyperparameters
| Param | Meaning | Typical |
|---|---|---|
| `r` | rank / capacity | 8–64 |
| `lora_alpha` | scale numerator (update × α/r) | r or 2r |
| `lora_dropout` | dropout on adapter path (regulariser) | 0.05–0.1 |
| `target_modules` | which layers get adapters | attention q,k,v,o; often also MLP (gate/up/down) |
| `task_type` | head type | `CAUSAL_LM` |
| `fan_in_fan_out` | layer stores weight transposed | `True` for GPT-2's `Conv1D` |

### 2.3 Your numbers (distilgpt2)
- `trainable 147,456 || all 82,060,032 || 0.18%`
- Check: 6 blocks × (768·8 + 8·2304) = 6 × 24,576 = 147,456. `c_attn` maps 768 → 2304 because **Q, K, V are fused** (3 × 768).
- Loss 4.574 → 1.404 over 25 epochs.
- **Adapter: 0.60 MB vs ~330 MB** full model.
- Reload output identical: True. Merged output identical: True. Type is back to `GPT2LMHeadModel`.

Rank sweep (10 epochs):
| r | trainable | final loss |
|---|---|---|
| 1 | 18,432 | 2.902 |
| 8 | 147,456 | 2.145 |
| 32 | 589,824 | 1.734 |

### 2.4 Things interviewers probe
- **Why a higher LR for LoRA?** (1e-3 vs ~5e-5 for full fine-tuning.) The adapters start at zero and are tiny, so small steps barely move the output. At 3e-6 LoRA hardly learns. Typical LoRA LR: 1e-4 to 1e-3.
- **Which modules to target?** The original paper adapted only q and v. Modern practice (QLoRA paper) adapts **all linear layers**, which gives better quality at a modest parameter cost.
- **Layer names differ per architecture.** GPT-2 has `c_attn`, `c_proj`, `c_fc`. Llama/Qwen have `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. The most common peft error is `ValueError: Target modules {'q_proj'} not found`. Fix it by printing `model.named_modules()`.
- **One base, many adapters:** serve one model and swap a few-MB adapter per customer or task.
- **`disable_adapter()`** gives a fair A/B comparison on the same object.

### 2.5 LoRA vs full fine-tuning: trade-offs
| | Full FT | LoRA |
|---|---|---|
| Trainable params | 100% | ~0.1–1% |
| Memory | ~16 B/param | ~2 B/param (frozen fp16) + tiny |
| Checkpoint | full model | MBs |
| Quality | ceiling | close for most tasks; can lag on big domain shifts (e.g. new language, heavy math) |
| Forgetting | higher risk | lower (base frozen) |
| Inference cost | same | same if merged |

Other PEFT methods to name-drop: prefix tuning, prompt tuning, (IA)³, adapters (Houlsby), and **DoRA**, which splits W into magnitude and direction and applies LoRA to the direction.

> **Say it aloud:** "With peft I wrap the attention layers in LoRA adapters, freeze the base, and train about 0.2% of parameters with the same loop. The result is a 0.6 MB adapter I can load on top of the base or merge into it."

---

## 3. QLoRA (Step 06)

### 3.1 The problem
LoRA removed gradient and optimizer costs, but the **frozen base still sits in fp16**: 7B × 2 bytes = **14 GB**. Add activations and a 16 GB GPU runs out.

### 3.2 The idea
**Store the frozen base in 4-bit, train LoRA adapters in 16-bit on top.** (Dettmers et al., 2023.)
- The base is read-only, so quantisation error never *accumulates* in trained weights.
- Gradients flow *through* the de-quantised base into the adapters.

```
LoRA:   base fp16 (2 B/param, frozen) + adapters 16-bit (trained)    7B → ~14 GB weights
QLoRA:  base NF4  (0.5 B/param, frozen) + adapters 16-bit (trained)  7B → ~3.5–4 GB weights
```

### 3.3 The three ingredients
1. **NF4 (NormalFloat 4-bit)**
   - Pretrained weights are roughly **normally distributed** around zero.
   - NF4 places its 16 levels at the **quantiles of a normal distribution**, so each level is used about equally often. It is "information-theoretically optimal" for normal data.
   - Plain int4 has evenly spaced levels and wastes them in the tails.
   - Quantisation is **block-wise** (blocks of 64 weights), each with its own absmax scale. Blocking limits the damage from outliers.
2. **Double quantisation**
   - The per-block scales are fp32 numbers: 32 bits / 64 weights = 0.5 bit per param of overhead.
   - Quantise *those scales* to 8-bit as well. This saves **~0.37–0.4 bit/param** (≈3 GB on a 65B model).
3. **Paged optimizers**
   - Optimizer state lives in NVIDIA **unified memory** and pages to CPU RAM during memory spikes (e.g. a long sequence) instead of throwing OOM.
   - `paged_adamw_8bit` also stores Adam state in 8-bit.

### 3.4 How compute works
- **Storage dtype ≠ compute dtype.** Weights are stored in NF4. During the forward pass each block is **de-quantised to bf16 just in time**, used in the matmul, then discarded (`bnb_4bit_compute_dtype=torch.bfloat16`).
- That is why QLoRA is **slower per step** than LoRA: there is extra de-quantisation work.
- `prepare_model_for_kbit_training` casts LayerNorms to fp32 for stability, enables gradient checkpointing, and makes input embeddings require grad so gradients flow.

### 3.5 Code skeleton
```python
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb, device_map="auto")
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32,
                       target_modules=["q_proj","k_proj","v_proj","o_proj"], task_type="CAUSAL_LM"))
```

### 3.6 Memory (6.7B, weights + grads + optimizer, activations extra)
| Method | Memory |
|---|---|
| Full fine-tune | **107.2 GB** |
| LoRA | **13.5 GB** |
| QLoRA | **3.5 GB** |

### 3.7 Trade-offs
- A small quality loss from quantisation (the paper showed it roughly matches 16-bit LoRA; you should *measure*, not assume).
- Slower steps (de-quantisation).
- In practice **CUDA-only** (`bitsandbytes`). It does not run on Apple Silicon, which is why your step was reference-only.
- **Deployment:** to merge, load a **16-bit** base, attach the adapter, and `merge_and_unload()`. Merging into a 4-bit base loses precision. Alternatively, keep it unmerged.

> **Say it aloud:** "QLoRA keeps the frozen base in 4-bit NF4 and trains 16-bit LoRA adapters on top, de-quantising each block to bf16 just in time for the matmul. That takes a 7B fine-tune from over 100 GB to a single 16 GB GPU, at the cost of slower steps and a small quality loss I'd measure."

---

## 4. Fast inference with Hugging Face (Step 07)

### 4.1 Four levers
| Lever | What it does | Your measurement |
|---|---|---|
| `model.eval()` + `torch.no_grad()` | no dropout, no autograd graph → less memory, faster | – |
| **KV cache** (`use_cache=True`) | reuse past keys and values | 100 tokens: **3.5 s vs 8.5 s** |
| **Batching** (left padding) | many prompts per forward pass | 4 prompts: **4.3 s vs 1.2 s** |
| **dtype** (bf16/fp16) | half the bytes per weight | 328 MB → 164 MB |

### 4.2 The KV cache — deeply
Autoregressive generation produces **one token per step**. Attention for the new token needs **Q of the new token** and **K and V of all previous tokens**.
- **Without a cache:** at step t you re-run the whole sequence of length t through every layer. Total work ≈ 1 + 2 + … + n = **O(n²)** token-forwards.
- **With a cache:** store each layer's K and V for past tokens. At each step compute K, V and Q **only for the new token**, append K and V to the cache, and attend. Each step processes one token.
- Why only K and V, and not Q? Past queries are never needed again. Only the current token asks the question.
- **Price:** memory that grows linearly with **sequence length × batch** (see §5). The cache trades memory for compute.

### 4.3 Prefill vs decode (important vocabulary)
- **Prefill:** process the whole prompt in one parallel pass and fill the cache. This phase is **compute-bound** (big matmuls). It determines **TTFT (time to first token)**.
- **Decode:** generate one token at a time. Each step must **read all the weights and the whole KV cache from GPU memory** to do very little math. This phase is **memory-bandwidth-bound**. It determines **TPOT / inter-token latency**.
- This explains why **batching is nearly free for decode**: the weights you load once per step serve every sequence in the batch. In your run, 16× the requests cost about the same latency.

### 4.4 Padding side — the classic bug
- Batching needs equal-length rows, so short prompts get padded.
- **Decoder-only** models (GPT, Llama) continue from the **last position**.
  - **Right padding** → the last position is a pad token, so the model continues from padding. Your output: `'Hello, my name is\n\n\n\n...'`
  - **Left padding** → the real text ends at the last position, which is correct. Your output: `'Hello, my name is J.J.K. Rowling.'`
- The attention mask tells the model to ignore pads. HF also adjusts position ids.
- This bug gives **no exception**, only a one-line warning and odd output. Debug habit: a batch result should match the batch-size-1 result.
- GPT-2 has no pad token, so set `tokenizer.pad_token = tokenizer.eos_token`.
- (Encoder-decoder models like T5 and training use right padding. Left padding is specific to batched *generation* with decoder-only models.)

### 4.5 dtype
- Bytes/param: **fp32 = 4, fp16/bf16 = 2, int8 = 1, int4 = 0.5**.
- **bf16 vs fp16:** bf16 has the same 8-bit exponent as fp32, so the same *range* and almost no overflow, but fewer mantissa bits (less precision). fp16 has more precision but a small range, so it can overflow/underflow. That is why fp16 training needs loss scaling. bf16 is the default for modern training on Ampere+ GPUs.
- On GPU, 16-bit is smaller **and** faster (tensor cores). On CPU it often isn't faster, so always measure.

### 4.6 Measurement hygiene
- **Warm up** first (lazy init, kernel compile, caches).
- `min_new_tokens = max_new_tokens` so every run does the same work.
- On GPU, **synchronise** before reading the clock (`torch.cuda.synchronize()`), because GPU calls are asynchronous.

> **Say it aloud:** "I disable gradients, keep the KV cache so each step only computes the new token's K and V, and batch prompts with left padding for decoder-only models. The cache trades memory for compute, batching trades per-user speed for throughput, and I measure each change."

---

## 5. GPU memory arithmetic (Step 08)

### 5.1 Serving formula
```
Total ≈ weights + KV cache + overhead (~10%: activations, CUDA context, fragmentation)

weights   = params × bytes_per_param
KV cache  = 2 × layers × kv_heads × head_dim × seq_len × batch × bytes_per_value
            ↑ the 2 is for K AND V
```

**Per-token KV size, Llama-7B fp16:** 2 × 32 × 32 × 128 × 2 B = **524,288 B = 0.5 MiB per token**.
That means 2048 tokens = 1 GiB per request.

### 5.2 Your table (6.7B, seq 4096, batch 8)
| precision | weights GB | KV GB | total GB |
|---|---|---|---|
| fp32 | 26.8 | 17.2 | 48.4 |
| fp16 | 13.4 | 17.2 | 33.6 |
| int8 | 6.7 | 17.2 | 26.3 |
| int4 | 3.4 | 17.2 | 22.6 |

**Key insight:** quantising the weights shrinks only the weights column. At long context or large batch, **the KV cache dominates and is bigger than the weights**.

### 5.3 Ways to shrink the KV cache
| Technique | How |
|---|---|
| **GQA (grouped-query attention)** | several query heads share one K/V head. Llama-3-8B: 32 Q heads, **8 KV heads** → 4× smaller cache (8192 tokens, batch 1 = **1.1 GB**) |
| **MQA (multi-query attention)** | the extreme: 1 KV head for all queries. Smallest cache, some quality loss |
| **KV-cache quantisation** | store K and V in fp8 or int8 |
| **PagedAttention** | don't *waste* cache memory (§8) |
| Sliding-window attention | only keep the last W tokens (e.g. Mistral) |

MHA → GQA → MQA is a spectrum from quality to memory.

### 5.4 Training formula (mixed-precision AdamW)
```
16 bytes/param = 2 (fp16 weights) + 2 (fp16 grads) + 4 (fp32 master weights) + 4 (Adam m) + 4 (Adam v)
+ activations (depend on batch × seq; reduced by gradient checkpointing)
```
- **fp32 master weights:** tiny updates (lr × grad) would round to zero in 16-bit, so a precise copy is kept for the update.
- **Adam m, v:** first moment (momentum) and second moment (per-parameter scale).
- 6.7B × 16 = **107.2 GB**. That is why full FT of 7B needs multiple A100s (or ZeRO/FSDP to shard the optimizer state).
- LoRA ≈ params × 2 + (tiny fraction × 16). QLoRA ≈ params × 0.5 + (tiny fraction × 16).

### 5.5 Sanity checks
- Your formula vs a real model: distilgpt2, 32 tokens, fp32 → **1,179,648 bytes formula = measured**.
- GB (1e9) vs GiB (2³⁰) differ by about 7%.

### 5.6 Quick mental rules
- **Inference weights:** 7B ≈ 14 GB fp16, 7 GB int8, 3.5 GB int4.
- **Full training:** about **16× params GB-ish** (7B → ~112 GB) plus activations.
- **KV, Llama-2-7B style (MHA):** ~0.5 MB/token. With GQA-8 it is ~0.125 MB/token.

> **Say it aloud:** "Serving memory is weights plus KV cache plus overhead. The KV cache is 2 × layers × KV heads × head dim × tokens × batch × bytes, and at long context it exceeds the weights. That's why GQA, KV quantisation and PagedAttention exist. Full AdamW training is about 16 bytes per parameter, so 7B needs about 107 GB."

---

## 6. Latency vs throughput (Step 09)

### 6.1 Definitions
- **Latency:** time for *one request* to finish. This is what the user feels. It splits into:
  - **TTFT** (time to first token, driven by prefill)
  - **TPOT / ITL** (time per output token, driven by decode)
  - End-to-end latency ≈ TTFT + TPOT × (n_tokens − 1)
- **Throughput:** total tokens/second across all requests. This drives **cost per token**.
- **Per-user tokens/s** = new_tokens / seconds (the inverse of latency).
- Throughput = batch × new_tokens / seconds.

### 6.2 Your run (CPU, 40 tokens)
| batch | latency s | tokens/s | per-user tok/s |
|---|---|---|---|
| 1 | 1.24 | 32.3 | 32.3 |
| 4 | 1.13 | 142.2 | 35.6 |
| 16 | 1.13 | **566.8** | 35.4 |

- 16× the requests at about the same latency gives **~17.5× throughput**. In an earlier quieter run, per-user speed fell from 70.7 to 42.8 tok/s.
- Peak RAM stayed flat (~740–805 MB). For this tiny model the weights dominate and the KV cache is negligible.

### 6.3 Why the curve looks like this
- Small batch → decode is **memory-bound**, and the hardware's compute is idle. Adding sequences is almost free.
- As batch grows, you eventually become **compute-bound** (throughput flattens) or **memory-bound by the KV cache** (you can't fit more sequences). On real GPUs the KV cache is usually the wall, which leads to vLLM.

### 6.4 The trade-off
- Bigger batch → higher throughput, lower cost per token, but each user's latency stays flat or rises.
- Production serving picks a point on this curve against an **SLA** (e.g. p95 TTFT < 500 ms).
- Always quote both numbers, and quote **percentiles (p50/p95/p99)**, not just means.

### 6.5 Benchmarking hygiene
Warm up, repeat, synchronise GPU, fix the output length, and distrust a single run. Your MPS numbers were erratic (batch 1 faster than batch 2), because small models under-use a GPU.

> **Say it aloud:** "Latency is how long one user waits; throughput is total tokens per second. Batching raises throughput because decode is memory-bound, so one weight read serves the whole batch, but per-user speed doesn't improve. Where the curve flattens tells me the bottleneck, and on real GPUs that's usually KV-cache memory."

---

## 7. vLLM (Step 10)

### 7.1 Two problems with naive serving
1. **KV-cache waste.** Each request reserves a **contiguous slab for the maximum length** (it doesn't know how long the answer will be). Most of that is never used, and the free space breaks into holes. The vLLM paper measured that existing systems used only **20–40%** of KV memory for actual tokens.
2. **Static batching.** A batch runs until its **slowest** member finishes. A 5-token answer holds its slot while a 500-token answer keeps going, and new requests wait.

### 7.2 Two fixes
| PagedAttention | Continuous (in-flight / iteration-level) batching |
|---|---|
| KV cache in fixed **blocks** (16 tokens), allocated on demand, non-contiguous, tracked by a **block table** | Scheduling happens **every decode step**: finished sequences leave, waiting ones join immediately |
| Waste is under ~4% (only the last block's tail) | The GPU never idles waiting for the slowest request |
| More concurrent requests → bigger batches → more throughput | Higher utilisation, lower queueing delay |

### 7.3 Bonus features enabled by paging
- **Prefix sharing / prefix caching:** many requests with the same system prompt can point their block tables at the **same physical blocks**, tracked by reference counts.
- **Copy-on-write:** for parallel sampling or beam search, sequences share prompt blocks and copy a block only when they diverge.
- **Preemption:** when blocks run out, evict a sequence by **swapping** its blocks to CPU or **recomputing** them later.

### 7.4 API essentials
```python
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", dtype="float16",
          gpu_memory_utilization=0.90, max_model_len=2048)
outs = llm.generate(prompts, SamplingParams(temperature=0.7, top_p=0.9, max_tokens=64))
print(outs[0].outputs[0].text)     # request 0, completion 0
```
```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000 --max-model-len 2048
# OpenAI-compatible: POST /v1/chat/completions
```
- **`gpu_memory_utilization=0.9`:** vLLM grabs 90% of the GPU up front. Whatever remains **after the weights** becomes the pool of KV blocks. Higher means more concurrency but less headroom (risk of OOM at start-up).
- **`max_model_len`:** caps prompt + output per sequence, which bounds the per-sequence cache.
- **OpenAI-compatible server:** existing clients work by changing only the base URL.

### 7.5 Transformers vs vLLM
| | HF `generate` | vLLM |
|---|---|---|
| Use | learning, experiments | production, many users |
| Batching | static, you build it | continuous, automatic |
| KV cache | contiguous | paged |
| Throughput | baseline | much higher (paper: up to ~24× vs HF, 2–4× vs prior SOTA servers) |
| Single-request latency | similar | similar |
| Platform | CPU/MPS/CUDA | Linux + CUDA mainly |

Alternatives to name-drop: **TGI** (Hugging Face), **TensorRT-LLM** (NVIDIA), **SGLang** (RadixAttention prefix cache), **llama.cpp / Ollama** (CPU/Mac, GGUF).

> **Say it aloud:** "vLLM serves many users on one GPU because PagedAttention stores the KV cache in small pages instead of one max-length reservation per request, so little memory is wasted. Continuous batching swaps finished requests out and waiting ones in at every step, so the GPU never idles on the slowest request."

---

## 8. PagedAttention and fragmentation (Step 11)

### 8.1 OS analogy (the core intuition)
| OS virtual memory | vLLM |
|---|---|
| Process | Sequence/request |
| Page (e.g. 4 KB) | KV block (16 tokens) |
| Page table | **Block table** |
| Virtual page → physical frame | logical block i → physical block j |
| Pages can sit anywhere in RAM | blocks can sit anywhere on the GPU |
| Swap to disk | Swap blocks to CPU (preemption) |

### 8.2 Two kinds of fragmentation
- **Internal fragmentation:** memory reserved *inside* an allocation but unused. A request reserves 2048 slots and uses 300, so **1748 are wasted**. In your sim: **85% of the reservation was wasted.**
- **External fragmentation:** free memory split into **holes too small** for the next request. The total free space is enough, but no single hole is. In your sim (variable-size slabs), the next request was refused in 2519 steps, with ~2113 free slots stuck in small holes.

Paging kills both:
- **External** disappears because every block is the same size, so any free block fits anyone.
- **Internal** shrinks to the **unused tail of the last block**: at most 15 slots per sequence with block 16.

### 8.3 Block table mechanics
- A free pool is a **stack of free block ids**. Allocate with `pop()`, free with `extend()`.
- Each request has a list `[phys_0, phys_1, …]`. Token t lives in block `table[t // 16]` at offset `t % 16`.
- Blocks needed = `ceil(len / 16)`, written in code as `-(-len // 16)`.
- Your demo: request A with 70 tokens, 5 blocks, table `[0, 1, 4, 3, 2]`. It is not contiguous and grabbed B's freed blocks.
- **The attention kernel must read K and V through the block table** (gather from scattered blocks). That is the actual "PagedAttention" CUDA kernel. A standard attention kernel assumes contiguous tensors.

### 8.4 Block size trade-off
- **Small blocks** → less last-block waste, but bigger tables and more indirection.
- **Large blocks** (e.g. 128) → internal fragmentation comes back.
- vLLM's default is **16**.

### 8.5 Your simulation results (8 GB KV budget = 16,384 slots = 1024 blocks; 300 requests, mean length 306)
**A) Static snapshot**
| Allocator | Requests that fit | Waste |
|---|---|---|
| Contiguous, reserve 2048 | **8** | 85% |
| Paged, block 16 | **53** | 2.4% |

That is **6.6× more requests** in the same memory.

**B) Dynamic run**
| Allocator | Mem used | Avg running | Steps |
|---|---|---|---|
| Contiguous, reserve max | 13.8% | 7.9 | 6494 |
| Contiguous, reserve prompt+max_tokens | 30.3% | 17.2 | 2961 |
| Paged | **80.3%** | **52.5** | **2268** |

Result: **~2.9× throughput** vs reserve-max.
- 690 preemptions came from a greedy toy scheduler. Real vLLM schedules more carefully, but it does also preempt.

### 8.6 Failure mode to mention
- **Block leak:** if `free()` doesn't return blocks, the pool drains, nothing is admitted, and throughput decays until the server must be restarted. Your break-it run showed this as `RuntimeError: simulation did not finish`.

### 8.7 Experiments worth remembering
- `BLOCK_SIZE=128` → more last-block waste.
- More budget → every allocator fits more, but the ratio stays similar.
- `MAX_LEN=4096` → contiguous gets much worse while paged barely changes. **Long-context models make paging more important.**

> **Say it aloud:** "Reserving the max KV length per request wastes most of the GPU because real answers are short (internal fragmentation), and variable slabs leave unusable holes (external fragmentation). PagedAttention uses fixed 16-token blocks with a per-request block table mapping logical to physical blocks, so waste is only the last block's tail. In my simulation that fit 6.6× more requests in the same 8 GB."

---

## 9. One-page cheat sheet

```
LoRA          W = W0 + (α/r)·B·A     A:(r,d_in) random,  B:(d_out,r) ZERO
              params r(d_in+d_out) vs d_in·d_out       2048², r=8 → 0.78%
              compute B@(A@x); merge W0+(α/r)BA → zero inference overhead
              saves grads + optimizer state for frozen weights (not forward weights)
              LR higher (1e-4..1e-3); α≈2r; target q,k,v,o (+MLP)

QLoRA         base NF4 4-bit (frozen) + LoRA bf16 (trained)
              NF4 = normal-quantile levels, blockwise (64) absmax
              double quant = quantise the scales (~0.4 bit/param)
              paged optimizers = spill to CPU on spikes
              store 4-bit, compute bf16 (dequant just-in-time) → slower, CUDA-only

Bytes/param   fp32 4 | fp16/bf16 2 | int8 1 | int4 0.5
Training      16 B/param = 2 w + 2 g + 4 master + 4 m + 4 v    (6.7B → 107 GB)
              LoRA 13.5 GB | QLoRA 3.5 GB (6.7B, no activations)
KV cache      2 × L × kv_heads × head_dim × seq × batch × bytes
              Llama-7B fp16 = 0.5 MiB/token → 2048 tok = 1 GiB
              GQA (8 KV heads) → 4× smaller

Inference     eval() + no_grad | KV cache (O(n²) → O(n)) | batch + LEFT pad | bf16
              prefill = compute-bound (TTFT); decode = memory-bound (TPOT)
Latency       one user (TTFT + TPOT·n)   Throughput  batch·tokens / s
vLLM          PagedAttention (blocks of 16 + block table) + continuous batching
              prefix sharing, copy-on-write, preemption (swap / recompute)
              gpu_memory_utilization, max_model_len, OpenAI-compatible server
Fragmentation internal = reserved-unused | external = holes too small
Your sim      8 GB: contiguous 8 req vs paged 53 req (6.6×), ~2.9× throughput
```

---

## 10. Rapid-fire interview Q&A

**Q: Why does LoRA work at all?**
A: Fine-tuning updates have low intrinsic rank. The pretrained model already holds the knowledge, and the task-specific change lives in a small subspace, so a rank-r product B·A can represent it.

**Q: Why is B initialised to zero and not A?**
A: Either one must be zero so ΔW = 0 at the start, and the other must be random so gradients can flow. With B = 0 and A random, B gets a non-zero gradient on step 1. If both were zero, neither would ever move.

**Q: Does LoRA reduce inference cost?**
A: No. If merged it is identical to the base. If unmerged there is a tiny extra cost. LoRA reduces *training* memory and checkpoint size.

**Q: Does LoRA reduce activation memory?**
A: Not much. Backprop still flows through the frozen layers. Gradient checkpointing is what reduces activations.

**Q: What does α do?**
A: It scales the update by α/r, keeping update magnitude stable across ranks so you don't have to re-tune the LR when you change r.

**Q: Which r should I use?**
A: Start at 8–16. Increase it if the task is far from pretraining (new domain or language) and validation loss plateaus. Watch for overfitting.

**Q: How is QLoRA different from just quantising a model?**
A: Post-training quantisation (GPTQ, AWQ) is for *inference*. QLoRA quantises the *frozen base during fine-tuning* and trains 16-bit adapters on top.

**Q: Why NF4 over int4?**
A: Weights are roughly normally distributed. NF4 puts its 16 levels at normal quantiles, so each level is used equally, which means lower error for the same bits.

**Q: Why can't you merge a QLoRA adapter into the 4-bit base directly?**
A: You'd have to re-quantise W₀ + ΔW, which adds error. Merge into a 16-bit base instead, and quantise afterwards if needed.

**Q: Why cache K and V but not Q?**
A: Each step only needs the *current* token's query. Past queries are never used again, while all past keys and values are.

**Q: What's the complexity with and without a KV cache?**
A: Without one, generating n tokens recomputes the prefix each step, so O(n²) token-forwards. With one, each step processes one new token (attention still reads all t cached tokens).

**Q: Why left padding?**
A: Decoder-only models generate from the last position. Right padding puts pads there, so the model continues from padding.

**Q: Why is batching almost free during decode?**
A: Decode is memory-bandwidth-bound: each step loads all the weights to do a tiny matmul. A batch reuses the same weight read for every sequence.

**Q: Memory needed to serve a 7B model at 4k context with batch 8?**
A: About 13.4 GB fp16 weights + 17.2 GB KV + ~10% overhead ≈ **34 GB**. With int4 weights it is still ~23 GB, because the KV cache dominates.

**Q: Memory to fully fine-tune 7B?**
A: About 16 B/param ≈ 107–112 GB plus activations. Use LoRA (~14 GB) or QLoRA (~4 GB), or shard with ZeRO/FSDP.

**Q: What is GQA and why does it matter for serving?**
A: Query heads share fewer KV heads (e.g. 32 Q, 8 KV), so the KV cache shrinks 4×. That allows more concurrent requests or longer context.

**Q: Latency vs throughput, and how does batching affect each?**
A: Latency is per-request time; throughput is system tokens/s. Batching raises throughput and leaves latency flat or slightly worse.

**Q: What are TTFT and TPOT?**
A: TTFT (time to first token) is dominated by prefill and is compute-bound. TPOT (time per output token) is dominated by decode and is memory-bound.

**Q: What problem does PagedAttention solve?**
A: KV-cache memory waste from contiguous max-length reservations: internal fragmentation (reserved but unused) and external fragmentation (unusable holes).

**Q: What is a block table?**
A: A per-sequence map from logical block index to physical GPU block, like an OS page table. The attention kernel gathers K and V through it.

**Q: Why are blocks 16 tokens and not 1 or 1024?**
A: Small blocks give low tail waste, but too small means table overhead and poor memory-access efficiency. Large blocks bring internal fragmentation back. 16 is the balance point.

**Q: What is continuous batching?**
A: Iteration-level scheduling. After every decode step, finished sequences leave and queued ones join, so short requests don't wait for long ones.

**Q: What happens when vLLM runs out of KV blocks?**
A: It preempts sequences: swaps their blocks to CPU memory or drops them and recomputes later.

**Q: How does paging enable prefix caching?**
A: Two requests with the same system prompt can map their first logical blocks to the same physical blocks, tracked with ref counts and copy-on-write on divergence.

**Q: When would you *not* use vLLM?**
A: Local experiments, CPU/Mac (use llama.cpp/Ollama), unsupported model or quant formats, or a single-user latency-critical setting where its advantage (throughput) doesn't matter.

**Q: How would you benchmark a serving setup fairly?**
A: Warm up, fix the output length, synchronise the GPU, sweep batch/concurrency, report throughput **and** p50/p95 TTFT/TPOT, and repeat runs.

---

### Suggested revision order (if short on time)
1. §0 big picture → §9 cheat sheet (10 min)
2. §1 LoRA maths + §3 QLoRA (most-asked training topics)
3. §4.2 KV cache + §5 memory formulas (most-asked serving topics)
4. §7 vLLM + §8 PagedAttention
5. §10 Q&A out loud
