# Interview Gist — Module 06 (steps 04 to 06)
### LoRA → peft → QLoRA

> The serving half of this gist (old steps 07–11: fast inference, GPU memory, throughput, vLLM, PagedAttention) moved with those steps to [module 21 — LLM inference engineering](../21_llm_inference_engineering/INTERVIEW_GIST.md).

This is a single-file theory sheet for revision. Each section gives:
**the problem → the idea → the maths/mechanics → numbers from your own runs → trade-offs → likely interview questions.**
The last part has a one-page cheat sheet and a rapid-fire Q&A.

---

## 0. The big picture (read this first)

This module tells the training half of the story; the serving half now lives in [module 21](../21_llm_inference_engineering/).

**Half 1: training cheaply (04–06)**
- Full fine-tuning updates every weight. With AdamW that costs about **16 bytes per parameter**, so a 7B model needs about **107 GB**.
- **LoRA (04, 05):** freeze the model and learn a small low-rank *update* instead. That leaves about 0.1–1% of parameters trainable.
- **QLoRA (06):** additionally store the frozen base in **4-bit**, so a 7B fine-tune fits on one 16 GB GPU.

**Half 2: serving fast** → see [21_llm_inference_engineering](../21_llm_inference_engineering/README.md) (memory calculator, KV cache, quantization, batching, PagedAttention, vLLM and other engines).

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

---

### Suggested revision order (if short on time)
1. §0 big picture → §9 cheat sheet (10 min)
2. §1 LoRA maths + §3 QLoRA (most-asked training topics)
3. §10 Q&A out loud
4. Then the serving half: [module 21 gist](../21_llm_inference_engineering/INTERVIEW_GIST.md)
