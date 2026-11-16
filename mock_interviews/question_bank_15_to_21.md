# Question Bank, Modules 15 to 21

Deep, module-specific interview questions with model answers. The cross-module "bridge" questions for modules 00 to 14 are in [question_bank.md](question_bank.md); this file goes deeper into the newer modules.

**How to use it.** Cover the answer, say yours aloud in 60-90 seconds, then compare. Each module has questions in six types:

| Type | What the interviewer is testing |
|------|---------------------------------|
| **Concept** | Can you explain it plainly and name the trade-off? |
| **Math** | Can you compute with real numbers on a whiteboard? |
| **Implementation** | Can you say what you would write, in order, with the traps? |
| **Debugging** | Symptom, then cause, then fix, then how you would have caught it earlier. |
| **Architecture** | Can you assemble the pieces into a system and defend the choices? |
| **Performance** | Do you know the bottleneck and the lever that moves it? |

Numbers marked "in our run" come from the real executions recorded in that step's WALKTHROUGH (laptop CPU; your timings will differ, the shapes of the curves will not). Hardware figures for A100/H100 are vendor spec-sheet values, not measured here.

Contents: [15](#module-15-neural-networks) | [16](#module-16-modern-llm-architecture) | [17](#module-17-fine-tuning-alignment-and-benchmarks) | [18](#module-18-ocr-and-document-parsing) | [19](#module-19-chunking) | [20](#module-20-llm-system-lld) | [21 (top priority)](#module-21-llm-inference-engineering)

---

## Module 15: Neural networks

### Concept

**15.C1. Why do we need a nonlinearity, and what does backprop actually do?**
Without a nonlinearity, stacked linear layers collapse into one linear map (`W2 (W1 x) = (W2 W1) x`), so depth buys nothing and XOR is impossible for a single neuron (best we got: 3/4 correct). Backpropagation is the chain rule applied from the loss back to every parameter, reusing values cached in the forward pass: `dW = input^T @ dout`, `d_input = dout @ W^T`, `dz = da * f'(z)`. Where a value feeds several paths, gradients add (`+=`), which is the single most common implementation bug.
Teaches: [02 MLP and backprop](../15_neural_networks/02_mlp_backprop_from_scratch/WALKTHROUGH.md), [01 neuron](../15_neural_networks/01_neuron_and_perceptron/WALKTHROUGH.md).

**15.C2. BatchNorm vs LayerNorm vs RMSNorm: what is normalised, and why do transformers avoid BatchNorm?**
BatchNorm normalises each feature over the batch and keeps running statistics for inference, so train and eval behave differently and tiny or variable batches break it. LayerNorm normalises each sample over its features (no train/eval gap, works with batch 1 and variable-length sequences). RMSNorm drops the mean-centring and bias: `x / sqrt(mean(x^2)+eps) * g`, half the parameters, same effect in practice. Sequence models use per-token normalisation because batch statistics over padded, variable-length text are meaningless.
Teaches: [05 initialisation and normalisation](../15_neural_networks/05_initialization_and_normalization/WALKTHROUGH.md).

### Math (with worked numbers)

**15.M1. Logits z = [2, 1, 0], true class 0. Give the probabilities, the loss and the logit gradient.**
`exp(z) = [7.389, 2.718, 1]`, sum 11.107, so `p = [0.665, 0.245, 0.090]`. Loss `-ln p_0 = 0.408`. Gradient wrt logits is `p - onehot = [-0.335, +0.245, +0.090]` (divided by N for a mean over N examples). It sums to zero. If you forget the `/N`, every parameter gradient is N times too big; a gradient check then shows the same relative error `(N-1)/(N+1)` everywhere (0.6 for N=4 in our run).
Teaches: [02](../15_neural_networks/02_mlp_backprop_from_scratch/WALKTHROUGH.md), [03/06 softmax and CE](../03_ml_foundations/06_softmax_cross_entropy/WALKTHROUGH.md).

**15.M2. Derive He initialisation. What happens to a 20-layer ReLU net initialised with std 0.02 and width 512?**
For `y = sum_i w_i x_i`, `Var(y) = n_in Var(w) E[x^2]`. ReLU zeroes half the inputs, so `E[x^2]` of the next layer is half the pre-activation variance. To keep the scale, `Var(w) = 2/n_in` (He); for width 512 that is std 0.0625. With std 0.02, `n_in Var(w) / 2 = 512 x 0.0004 / 2 = 0.10`, so the signal variance shrinks about 10x per layer; after 20 layers about 1e-20, the output (and the gradient) is effectively zero. Xavier uses `2/(n_in+n_out)` and suits tanh/linear.
Teaches: [05](../15_neural_networks/05_initialization_and_normalization/WALKTHROUGH.md), [04 vanishing gradients](../15_neural_networks/04_activations_and_vanishing_gradients/WALKTHROUGH.md).

**15.M3. A conv layer takes a 224x224x3 image, 64 filters of 3x3, stride 1, padding 1. Output shape, parameter count, and the dense-layer comparison? What is the output for a 7x7, stride 2, pad 3 conv?**
Output size `floor((n + 2p - k)/s) + 1 = 224` so `224x224x64`. Parameters `C_out (C_in k^2 + 1) = 64 (27 + 1) = 1,792`. A dense layer from the same input to the same number of outputs would need about 150 million weights; weight sharing and locality are the whole point. The 7x7/s2/p3 case: `floor((224+6-7)/2)+1 = 112`.
Teaches: [08 CNN](../15_neural_networks/08_cnn_convolutions/WALKTHROUGH.md).

### Implementation

**15.I1. Write the training step of a 2-layer MLP classifier in numpy, with a gradient check. What would you write, in order?**
(1) Init: He for the hidden layer, zeros for biases, shapes `(in, out)`. (2) Forward with a cache: `z1 = x@W1+b1; a1 = relu(z1); z2 = a1@W2+b2`. (3) Loss: stable softmax-CE as `logsumexp(z) - z_y`, mean over the batch. (4) Backward: `dz2 = (p - onehot)/N; dW2 = a1.T@dz2; db2 = dz2.sum(0); da1 = dz2@W2.T; dz1 = da1*(z1>0); dW1 = x.T@dz1; db1 = dz1.sum(0)`. (5) SGD update. (6) Gradient check in float64 with central differences, relative error below 1e-6 (we got 5.5e-9 by hand). Say the check out loud: it is how you prove the backward pass.
Teaches: [02](../15_neural_networks/02_mlp_backprop_from_scratch/WALKTHROUGH.md).

**15.I2. Sketch a scalar autograd engine (a `Value` class).**
Each `Value` stores `data`, `grad=0`, its parents, and a `_backward` closure. Operators create a new `Value` and a closure that does `parent.grad += local_derivative * out.grad` (note `+=`). `backward()` builds a topological order with a DFS, sets `self.grad = 1`, then calls `_backward` in reverse order. Test against the hand-written MLP gradients (we saw 3e-17 difference). Mention that PyTorch does this with tensors and a graph of `grad_fn` nodes, and that `.detach()`/`no_grad` cut the graph.
Teaches: [03 tiny autograd](../15_neural_networks/03_tiny_autograd_engine/WALKTHROUGH.md).

### Debugging (symptom, cause, fix)

**15.D1. Your hand-written gradient check prints relative error 0.6 for every parameter.**
Symptom: uniform error across W1, b1, W2, b2. Cause: something near the loss changed the scale, here a missing `dlogits /= N` (analytic gradient N times the numeric; `|4g-g|/(|4g|+|g|) = 0.6` for N=4). Fix: divide by batch size once, at the loss. Lesson: uniform error means scale near the loss; error in one parameter means a bug in that layer's derivative.
Teaches: [02 Debugging practice](../15_neural_networks/02_mlp_backprop_from_scratch/WALKTHROUGH.md).

**15.D2. A model works in your notebook but predictions depend on the batch they arrive in, and one request fails with "Expected more than 1 value per channel when training".**
Cause: BatchNorm left in `train()` mode at inference: it uses batch statistics (undefined for one sample, noisy otherwise) instead of running statistics. Fix: `model.eval()` plus `torch.no_grad()`; prefer LayerNorm/GroupNorm if you must run tiny batches; `drop_last=True` during training so a one-row last batch never occurs. Related silent bug: a reimplemented LayerNorm using `unbiased=True` differed from `nn.LayerNorm` by 9e-2 with no error; compare with `allclose` and a tolerance you chose.
Teaches: [05](../15_neural_networks/05_initialization_and_normalization/WALKTHROUGH.md).

**15.D3. Loss becomes NaN a few hundred steps into fp16 mixed-precision training.**
Cause: fp16 overflows at 65,504 (and underflows small gradients); a spike in activations or logits becomes `inf`, then NaN. Fixes: loss scaling (dynamic `GradScaler`), switch to bf16 (fp32's exponent range, so no scaling), gradient clipping, lower learning rate or longer warm-up, compute softmax/loss in fp32. Find the first bad step by logging grad norm per step.
Teaches: [10 precision and stability](../15_neural_networks/10_precision_and_training_stability/WALKTHROUGH.md).

### Architecture / system design

**15.A1. You must train a 1B-parameter transformer from scratch. Choose the optimiser, schedule, normalisation and precision, and justify each.**
AdamW (decoupled decay; under Adam plain L2 does not decay uniformly: weight norm 1.8 vs 29.7 in our run), peak LR from a small sweep with linear warm-up then cosine to a floor, gradient clipping at 1.0, pre-norm RMSNorm blocks with residuals (gradient highway: 50-layer sigmoid net 0.0 vs ratio 1.27 with residual), He/scaled init, bf16 compute with fp32 master weights and fp32 optimiser state. Log loss, grad norm and learning rate; checkpoint often; decide the stability plan (skip bad batches, restart from earlier checkpoint) before the run.
Teaches: [06 optimisers and schedules](../15_neural_networks/06_optimizers_and_schedules/WALKTHROUGH.md), [10](../15_neural_networks/10_precision_and_training_stability/WALKTHROUGH.md), [05](../15_neural_networks/05_initialization_and_normalization/WALKTHROUGH.md).

### Performance / optimisation

**15.P1. Training memory is "16 bytes per parameter". Where does that come from and how do you cut it?**
2 (bf16 weights) + 2 (grads) + 4 (fp32 master) + 4 (Adam m) + 4 (Adam v) = 16 B; a 7B model is about 112 GB before activations. Levers: shard optimiser state/grads/params (ZeRO/FSDP), gradient checkpointing (trade compute for activation memory), 8-bit optimisers, LoRA (frozen base at 2 B/param plus 16 B on a tiny trainable share), QLoRA (base at about 0.5 B/param). Adam alone is 2 extra fp32 tensors per parameter, so 7B parameters need 56 GB for m and v.
Teaches: [06](../15_neural_networks/06_optimizers_and_schedules/WALKTHROUGH.md), [21/03 memory calculator](../21_llm_inference_engineering/03_gpu_memory_calculator/WALKTHROUGH.md).

---

## Module 16: Modern LLM architecture

### Concept

**16.C1. Explain MHA, MQA, GQA and MLA in terms of the KV cache.**
All four are ways of choosing what attention stores per token. MHA: one K and V per query head (largest cache). MQA: one shared K/V head. GQA: `g` KV heads shared by groups of query heads (`g = h` is MHA, `g = 1` is MQA); Llama-3-8B uses 8 KV heads for 32 query heads, a 4x cache saving. MLA (DeepSeek): cache a low-rank latent (512) plus a small decoupled RoPE key (64) and rebuild K and V; 576 values per layer per token, about 71x smaller than MHA for V3. MLA needs a decoupled RoPE key because rotation does not commute with the up-projection that rebuilds K.
Teaches: [04 attention variants and KV cache](../16_modern_llm_architecture/04_attention_variants_and_kv_cache/WALKTHROUGH.md).

**16.C2. How does RoPE work, why on q and k but not v, and how do you extend the context window?**
RoPE rotates each (even, odd) pair of the query and key by `pos x theta_i`, `theta_i = base^(-2i/d)`. A dot product of two rotated vectors depends only on the offset `m - n` (we verified the same score, -0.960123, at offsets 0, 100 and 1000), so position is relative with no parameters. It is applied to q and k because only the q.k score should depend on position; v carries content. Extension: position interpolation (divide positions), NTK-aware (raise the base: `base x s^(d/(d-2))`, 41,829 for 4x at head_dim 64), YaRN (per-frequency ramp plus attention temperature `0.1 ln s + 1`, 1.1386 for 4x), then a short long-context fine-tune. Configured length is not usable length.
Teaches: [02 positional encodings](../16_modern_llm_architecture/02_positional_encodings/WALKTHROUGH.md).

### Math

**16.M1. KV cache for Llama-3-8B (32 layers, 8 KV heads, head_dim 128, bf16): per token, at 8k, at 128k, and compare with Llama-2-7B (MHA).**
`2 x layers x kv_heads x head_dim x bytes = 2 x 32 x 8 x 128 x 2 = 131,072 B = 128 KiB` per token. At 8,192 tokens: about 1.07 GB per sequence; at 131,072 tokens: 17.2 GB. Llama-2-7B has 32 KV heads: 512 KiB per token and 68.7 GB at 128k, 4x more. Qwen2.5-0.5B (2 KV heads, 24 layers, head_dim 64) is 12 KiB per token. At long context the cache for one sequence exceeds the weights of a small model.
Teaches: [04](../16_modern_llm_architecture/04_attention_variants_and_kv_cache/WALKTHROUGH.md), [21/03](../21_llm_inference_engineering/03_gpu_memory_calculator/WALKTHROUGH.md).

**16.M2. Why is the SwiGLU hidden size about 8d/3? Check it for d = 768.**
A classic FFN has two matrices with hidden `4d`: `2 x d x 4d = 8d^2` parameters. SwiGLU has three (gate, up, down), so to match, `3 d h = 8 d^2` gives `h = 8d/3`. For d = 768: `h = 2048`, so SwiGLU has `3 x 768 x 2048 = 4,718,592` parameters; the classic FFN has `2 x 768 x 3072 = 4,718,592` (4,722,432 with biases, which is the figure in our run). Real models round `h` to hardware-friendly sizes and some widen it (Llama-3 3.5d).
Teaches: [03 norms and FFN](../16_modern_llm_architecture/03_norms_and_ffn/WALKTHROUGH.md).

**16.M3. Do the online softmax on two chunks, `[1, 3]` then `[2, 5]`, and show the rescale.**
Chunk 1: `m = 3`, `l = e^(1-3) + e^0 = 1.1353`. Chunk 2: `m_new = 5`; rescale old sum by `e^(3-5) = 0.1353` giving `0.1536`; add `e^(2-5) + e^0 = 1.0498`; `l = 1.2034`. Direct: `e^-4 + e^-2 + e^-3 + 1 = 0.0183 + 0.1353 + 0.0498 + 1 = 1.2034`. The output accumulator is rescaled by the same factor and normalised once at the end. Result is exact (we matched naive attention to 5.6e-16).
Teaches: [05 FlashAttention and online softmax](../16_modern_llm_architecture/05_flash_attention_online_softmax/WALKTHROUGH.md).

### Implementation

**16.I1. Write the BPE training loop and the encoder. What are the traps?**
Start from the 256 byte values (no unknown token). Loop: count adjacent pairs over the corpus (weighted by word frequency if pre-tokenised), pick the most frequent (tie-break deterministically), create a new id, replace every occurrence, record the merge in order. Encode: start from bytes and apply the merges in learned order (lowest rank first). Traps: overlapping pairs in `aaa`, merging across pre-token boundaries, nondeterministic tie-breaks, forgetting special tokens. In our run 300 merges on 20 KB gave 2.23 bytes/token, 1,000 merges 3.61.
Teaches: [01 BPE from scratch](../16_modern_llm_architecture/01_bpe_tokenizer_from_scratch/WALKTHROUGH.md).

**16.I2. Implement RoPE and a KV-cached decode step. Where do the bugs hide?**
`rope(x, pos)`: build `theta = base ** (-arange(0,d,2)/d)`, angles `pos[:,None]*theta`, rotate pairs with cos/sin (check which pairing convention the weights expect: interleaved vs half-split). Rotate q and k before the key enters the cache. In decode, read `start = cache.seq_len()` once per forward (layer 0 appends to the cache before layer 1 runs, so reading it per layer shifts positions), and need no causal mask for a single query; prefill needs the causal mask. Tests: cached greedy ids equal uncached ids (logits diff 4e-6), causality, initial loss near `ln(vocab)` (5.534 vs 5.545).
Teaches: [07 build a mini Llama](../16_modern_llm_architecture/07_build_mini_llama/WALKTHROUGH.md), [02](../16_modern_llm_architecture/02_positional_encodings/WALKTHROUGH.md).

### Debugging

**16.D1. Cached generation gives different text from uncached generation.**
Likely causes in order: (1) positions for new tokens computed from the wrong offset (`start` read after layer 0 mutated the cache); (2) RoPE applied after caching, or only to new keys inconsistently; (3) a mask applied in decode that hides the cache; (4) cache not reset between prompts; (5) shared cache object mutated in place. Method: compare logits (not text) at each step, bisect by layer. Acceptance test: equal ids and logits within about 1e-5.
Teaches: [07](../16_modern_llm_architecture/07_build_mini_llama/WALKTHROUGH.md).

**16.D2. You ported Llama weights into your own model and the logits are wrong but the shapes load fine.**
Check, in order: weight name mapping (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `input_layernorm`, `post_attention_layernorm`, `embed_tokens`, `lm_head`); RoPE pairing convention and base (`rope_theta`, 1e6 for Qwen2.5); GQA head repeat order; `rms_norm_eps`; tied embeddings; transposed `Linear` weights. Validate by loading your weights into the reference and comparing logits layer by layer (ours matched HF Llama to 3.6e-7).
Teaches: [08 read a real config](../16_modern_llm_architecture/08_read_a_real_config/WALKTHROUGH.md).

### Architecture / system design

**16.A1. Design the attention and positional setup for a new 8B model that must serve 128k context cheaply.**
GQA (8 KV heads) or MLA for cache size; RoPE with a large base and YaRN-style scaling plus long-context continued training; local:global layer mix (Gemma 3: 5:1, window 1,024) or chunked layers with periodic global/NoPE layers to cut cache; keep first tokens as attention sinks if evicting; FlashAttention kernels; consider FP8 KV. State the cost: Llama-3-8B-style GQA is 17.2 GB per 128k sequence in bf16, so concurrency is memory-limited and the serving design (paging, quantised cache) matters as much as the architecture.
Teaches: [04](../16_modern_llm_architecture/04_attention_variants_and_kv_cache/WALKTHROUGH.md), [10 landscape](../16_modern_llm_architecture/10_architecture_landscape_2026/WALKTHROUGH.md).

### Performance / optimisation

**16.P1. DeepSeek-V3 is 671B parameters with 37B active. Why is it cheap per token but not cheap to host?**
Memory scales with total parameters (all experts must be resident), FLOPs per token scale with active parameters (router picks top-k plus shared experts). So per-token compute is like a 37B dense model, but you still need memory (and interconnect for expert parallelism) for 671B. Also: load imbalance (untrained router gave 4 to 1,263 tokens per expert out of 4,096 slots; capacity factor 1.25 would drop 30%), solved by auxiliary loss or aux-loss-free bias; stragglers set the step time.
Teaches: [06 mixture of experts](../16_modern_llm_architecture/06_mixture_of_experts/WALKTHROUGH.md).

**16.P2. If FlashAttention does the same FLOPs, why is it faster, and is it approximate?**
It is exact. Naive attention writes the `N x N` score and probability matrices to GPU HBM and reads them back; at 131k tokens that is 64 GiB for one head in fp16. Flash tiles Q, K, V into on-chip SRAM, uses the online softmax to avoid ever materialising S/P, so HBM traffic and memory fall from O(N^2) to O(N). Attention is memory-bound, so less traffic means less time. SDPA is just the PyTorch entry point that picks a Flash/memory-efficient/math backend; PagedAttention is a different thing (KV-cache memory management).
Teaches: [05](../16_modern_llm_architecture/05_flash_attention_online_softmax/WALKTHROUGH.md).

---

## Module 17: Fine-tuning, alignment and benchmarks

### Concept

**17.C1. SFT, DPO and GRPO: what does each optimise and when do you use which?**
SFT: cross-entropy on the answer tokens only (prompt labels are -100), teaches format and behaviour from demonstrations. DPO: from `{prompt, chosen, rejected}` pairs, `L = -log sigmoid(beta [(log pi_c - log ref_c) - (log pi_r - log ref_r)])`, no reward model and no sampling, teaches taste/safety. GRPO (RLVR when the reward is a checker): sample G completions per prompt, advantage `(r_i - mean)/std`, clipped ratio plus KL to a reference, no value network; teaches checkable skills (math, code). Order in practice: SFT first, then preference tuning or RL, then evaluate.
Teaches: [01 post-training map](../17_finetuning_alignment_and_benchmarks/01_post_training_map_and_chat_templates/WALKTHROUGH.md), [05 DPO](../17_finetuning_alignment_and_benchmarks/05_dpo_from_scratch/WALKTHROUGH.md), [06 PPO vs GRPO](../17_finetuning_alignment_and_benchmarks/06_ppo_vs_grpo_rlvr/WALKTHROUGH.md).

**17.C2. LoRA vs QLoRA vs full fine-tuning; which modules do you adapt?**
LoRA freezes `W` and trains `B@A` with rank r, output `W x + (alpha/r) B A x`; B starts at zero so training begins at the base model. QLoRA keeps the frozen base in 4-bit (NF4) and trains LoRA adapters in higher precision. Full fine-tuning updates everything (16 B/param). Target at least the attention projections; "all-linear" is the strong default. Adapters are small, swappable, and mergeable into the base for serving (no extra latency after merge). Capacity is slightly lower than full fine-tuning.
Teaches: [03 PEFT family](../17_finetuning_alignment_and_benchmarks/03_peft_family/WALKTHROUGH.md), [06/04 LoRA from scratch](../06_llm_training_and_lora/04_lora_math_numpy/WALKTHROUGH.md).

### Math

**17.M1. Compute a DPO loss. beta = 0.1; policy minus reference log-prob is +2 for the chosen answer and -3 for the rejected one.**
Margin `= 0.1 x (2 - (-3)) = 0.5`. Loss `= -ln sigmoid(0.5) = -ln 0.6225 = 0.474`. At step 0 policy = reference, margin 0, loss `ln 2 = 0.693`. If you swap the sign you reward the rejected reply with no error raised (margin +10 gave 1.313 instead of tiny loss). Higher beta keeps the policy closer to the reference.
Teaches: [05](../17_finetuning_alignment_and_benchmarks/05_dpo_from_scratch/WALKTHROUGH.md).

**17.M2. GRPO group of 4 with rewards [1, 0, 0, 1]. Advantages? What if all four are 1?**
Mean 0.5, population std 0.5, so `A = [+1, -1, -1, +1]` (sample std would give 0.87; state which one your library uses). If all rewards are equal, std is 0 (guarded by epsilon) and every advantage is 0: that group produces no gradient, so very easy or very hard prompts teach nothing. Fix with prompt filtering or curriculum, or larger G.
Teaches: [06](../17_finetuning_alignment_and_benchmarks/06_ppo_vs_grpo_rlvr/WALKTHROUGH.md).

**17.M3. 10 samples, 3 correct. Compute pass@1 and pass@5 (unbiased). Then: n = 30 items at 50 percent accuracy, how wide is the 95 percent interval?**
`pass@k = 1 - C(n-c, k)/C(n, k)`. k=1: `1 - 7/10 = 0.30`. k=5: `1 - C(7,5)/C(10,5) = 1 - 21/252 = 0.917`. Do not estimate by taking the first k samples. Interval: standard error `sqrt(0.25/30) = 0.091`, so about plus or minus 0.18; you cannot tell a 3-point gain on 30 items from noise. Use a paired bootstrap on the same items to compare two systems.
Teaches: [08 evaluation from scratch](../17_finetuning_alignment_and_benchmarks/08_evaluation_from_scratch/WALKTHROUGH.md).

**17.M4. LoRA rank 16 on a 4096x4096 projection: trainable parameters and fraction? Memory for 7B under full FT, LoRA, QLoRA?**
`r (d_in + d_out) = 16 x 8192 = 131,072` versus `16.78M` in the full matrix: 0.78 percent. 7B memory: full FT about 112 GB (16 B/param); LoRA about 14 GB frozen base plus adapter states (about 15 GB total before activations); QLoRA about 3.5 GB base plus adapters (about 5 GB). In our calculator, a 6.7B model: 107.2 / 13.5 / 3.5 GB.
Teaches: [03](../17_finetuning_alignment_and_benchmarks/03_peft_family/WALKTHROUGH.md), [21/03](../21_llm_inference_engineering/03_gpu_memory_calculator/WALKTHROUGH.md).

### Implementation

**17.I1. Write the SFT loss with completion-only masking. What exactly do you shift?**
Build `input_ids` from the chat template (prompt plus answer plus EOS), `labels = input_ids.clone()` with prompt positions set to -100. Logits at position t predict token t+1, so compare `logits[:, :-1]` with `labels[:, 1:]` using `F.cross_entropy(..., ignore_index=-100)`. Pass `attention_mask`. Append EOS so the model learns to stop. Use `add_generation_prompt=True` only at inference.
Teaches: [02 SFT with loss masking](../17_finetuning_alignment_and_benchmarks/02_sft_from_scratch_with_loss_masking/WALKTHROUGH.md).

**17.I2. Implement the DPO loss from sequence log-probs.**
Compute the summed log-prob of the completion tokens for policy and frozen reference on chosen and rejected (4 forward passes; reference under `no_grad`). `logits = beta * ((pc - rc) - (pr - rr)); loss = -F.logsigmoid(logits).mean()`. Log `rewards/chosen`, `rewards/rejected`, margin and accuracy; assert loss starts at `ln 2`. Pitfalls: summing prompt tokens, using different tokenisation for policy and reference, no SFT warm start.
Teaches: [05](../17_finetuning_alignment_and_benchmarks/05_dpo_from_scratch/WALKTHROUGH.md).

### Debugging

**17.D1. SFT training loss looks normal and falls, but the model does not learn the task.**
Cause candidates: missing the one-token shift (compare logits and labels at the same positions; no exception because shapes match), labels not masked (learning to copy the prompt), wrong chat template (silent quality bug), pad equal to EOS with EOS masked so the model never learns to stop. Fix and confirm with a held-out generation, not training loss. 80 examples can reach loss near 0 by memorising.
Teaches: [02](../17_finetuning_alignment_and_benchmarks/02_sft_from_scratch_with_loss_masking/WALKTHROUGH.md).

**17.D2. DPO accuracy is 0.87 and the margin is large, but the generated text is degenerate. Chosen log-prob fell from -34 to -82.**
Symptom of over-optimisation and likelihood displacement: the policy lowers both chosen and rejected likelihood to widen the margin. In our `LR=5e-4` run chosen log-prob fell to about -82. Fixes: lower the learning rate (5e-6 to 5e-7 for full models), raise beta, fewer epochs, SFT on chosen first or add an SFT term, watch `logps/chosen` not just accuracy.
Teaches: [05 Debugging practice](../17_finetuning_alignment_and_benchmarks/05_dpo_from_scratch/WALKTHROUGH.md).

**17.D3. GRPO reward is flat and the loss is near zero.**
Check how many groups have zero reward variance (all-correct or all-wrong give zero advantage), whether the reward function actually discriminates (verify on hand-made good/bad samples), temperature too low (identical samples), KL coefficient too strong, and that `num_generations` is large enough. Log the fraction of groups with non-zero variance.
Teaches: [06](../17_finetuning_alignment_and_benchmarks/06_ppo_vs_grpo_rlvr/WALKTHROUGH.md).

### Architecture / system design

**17.A1. Design the post-training and evaluation pipeline for a domain assistant (support tickets, 5k labelled examples, 1 GPU).**
Baseline first: prompt-only and a held-out eval set with metrics, CI, and a regression probe (general perplexity). SFT with LoRA/QLoRA on completion-only loss; optional DPO on preference pairs from logged corrections; RL only if there is a verifiable reward. Evaluate with exact-match/F1 where possible, an LLM judge with swapped-order checks and calibration to human labels, paired bootstrap for comparisons, contamination checks. Version data, config and adapters; ship with a rollback.
Teaches: [11 end-to-end report card](../17_finetuning_alignment_and_benchmarks/11_end_to_end_report_card/WALKTHROUGH.md), [09 judges and arenas](../17_finetuning_alignment_and_benchmarks/09_llm_as_judge_and_arenas/WALKTHROUGH.md).

### Performance / optimisation

**17.P1. You must fine-tune a 7B model on one 24 GB GPU. What do you do?**
QLoRA: 4-bit base (about 3.5 GB), LoRA adapters in bf16, paged 8-bit optimiser, gradient checkpointing, small micro-batch with gradient accumulation, completion-only loss, sequence packing with per-document masks, sensible max length. Full FT is about 112 GB, impossible. Measure peak memory early with a few steps.
Teaches: [03](../17_finetuning_alignment_and_benchmarks/03_peft_family/WALKTHROUGH.md), [06/06 QLoRA reference](../06_llm_training_and_lora/06_qlora_reference/WALKTHROUGH.md).

---

## Module 18: OCR and document parsing

### Concept

**18.C1. A PDF arrives. How do you decide between text extraction and OCR?**
Inspect the file: a real text layer means extract (fast, exact); image-only pages need OCR; image plus invisible text (render mode 3, someone's earlier OCR) means verify the layer against the render. Route per page, not per file. An empty string from a scanned PDF must never be indexed as "no content"; `(cid:N)` characters mean a missing ToUnicode map, so fall back to OCR for that page.
Teaches: [08 PDF internals and routing](../18_ocr_and_document_parsing/08_pdf_internals_and_routing/WALKTHROUGH.md).

**18.C2. What is CTC and why does a CRNN need it?**
Line images have no per-frame alignment labels. CTC sums the probability of every frame-level path that collapses to the target (merge repeats, then drop blanks), `P(y|x) = sum over alignments of prod p(pi_t | x_t)`. The blank symbol lets the model emit "no new character" and separate genuine repeats (`a-a` collapses to `aa`, `aa` to `a`). It lets you train on (image, transcript) pairs only. Greedy decoding takes the argmax per frame then collapses; prefix beam search can recover answers greedy loses.
Teaches: [05 CRNN and CTC](../18_ocr_and_document_parsing/05_crnn_and_ctc/WALKTHROUGH.md).

### Math

**18.M1. Reference "the cat sat", hypothesis "the cot sat". Then reference "the cat sat", hypothesis "the cat". Give CER and WER.**
First: one character substitution out of 11 characters, CER 1/11 = 0.091; one word wrong out of 3, WER 0.333. Second: one deleted word, WER 1/3; CER: deleted " sat" is 4 characters over 11 = 0.364. Both are `(S + D + I)/N_ref`, so they can exceed 1 with many insertions. In our run CER 3.4 percent was WER 15.4 percent. Always state the normalisation (case, punctuation, whitespace).
Teaches: [06 OCR evaluation](../18_ocr_and_document_parsing/06_ocr_evaluation/WALKTHROUGH.md).

**18.M2. Coordinate conversions: a box at y = 700 in a PDF with page height 842 pt; an 800 px wide render at 96 dpi; text 12 pt at 300 dpi.**
PDF origin is bottom-left, images use top-left: `y_top = 842 - 700 = 142 pt`. Pixels to points at 96 dpi: `pt = px x 0.75`, so 800 px = 600 pt. 12 pt at 300 dpi: `12/72 x 300 = 50 px` for the em height; glyphs below about 8 to 11 px tall start to break in our run (75 dpi gave 97 of 128 glyph blobs). So render scans at 200 to 300 dpi.
Teaches: [08](../18_ocr_and_document_parsing/08_pdf_internals_and_routing/WALKTHROUGH.md), [01 images as arrays](../18_ocr_and_document_parsing/01_images_as_arrays/WALKTHROUGH.md).

**18.M3. How many CTC alignments produce "ab" in 4 frames? What is the minimum number of frames for "aab"?**
In our brute-force run there were 15 valid alignments of `ab` in 4 frames and the brute-force loss 1.2996 equalled `nn.CTCLoss`. Minimum frames for a label of length L with R adjacent repeats is `L + R`: "aab" has one repeat, so 4 frames (`a`, blank, `a`, `b`). If T is shorter, the loss is infinite (zero probability); `zero_infinity=True` hides that, so check lengths.
Teaches: [05](../18_ocr_and_document_parsing/05_crnn_and_ctc/WALKTHROUGH.md).

### Implementation

**18.I1. Implement CTC greedy decoding.**
`best = logits.argmax(-1)` per frame (T,), then `collapse`: iterate, emit a symbol only if it differs from the previous frame's symbol and is not blank. Map ids to characters. Test the traps: `a a` gives `a`, `a blank a` gives `aa`. Greedy can return `''` where prefix beam search finds `a`. Check `log_probs` shape `(T, B, C)` when training with `nn.CTCLoss`, and pass `log_softmax` output, not raw logits.
Teaches: [05](../18_ocr_and_document_parsing/05_crnn_and_ctc/WALKTHROUGH.md).

**18.I2. Implement Otsu thresholding in numpy.**
Gray: `0.299R + 0.587G + 0.114B`. Histogram of 256 bins, normalise. For each t compute `omega(t)` (cumulative probability) and class means; pick `t* = argmax omega (1 - omega)(mu0 - mu1)^2`. Vectorise with cumulative sums. Know where it fails: uneven lighting (shadow page F1 0.194 for Otsu, 0.949 for a local mean threshold); dim page: fixed 128 F1 0.098 versus Otsu 0.976.
Teaches: [02 preprocessing](../18_ocr_and_document_parsing/02_preprocessing/WALKTHROUGH.md).

### Debugging

**18.D1. Overall CER is 3 percent but users say the extracted invoice is "completely wrong".**
Character metrics hide structure errors: swapped columns gave CER 0.79 with all characters present; merged table columns gave text CER 0.000 but TEDS-like 0.645. Look at reading order (XY-cut gave WER 0.014 versus naive y-sort 0.554 on the same text), table structure and field-level correctness (`230.00` vs `2300.00` hidden by fuzzy matching). Add TEDS or cell-level scoring and a gold set of your own documents.
Teaches: [06](../18_ocr_and_document_parsing/06_ocr_evaluation/WALKTHROUGH.md), [07 reading order and tables](../18_ocr_and_document_parsing/07_reading_order_and_tables/WALKTHROUGH.md).

**18.D2. Training CRNN raises `RuntimeError: input_lengths must be of size batch_size`.**
Cause: `log_probs` is `(B, T, C)` but `CTCLoss` expects `(T, B, C)`, so it reads the time axis as the batch. Fix: `permute(1, 0, 2)`. If sizes coincide there is no error and the loss is silently wrong, so assert the shape. Related: feed `log_softmax` output; `input_lengths` must be T per sample, targets lengths must be the label lengths.
Teaches: [05 Debugging practice](../18_ocr_and_document_parsing/05_crnn_and_ctc/WALKTHROUGH.md).

**18.D3. The text extracted from a two-column PDF reads across the columns.**
Cause: sorting by y (or stream order) interleaves columns. Fix: recursive XY-cut (split on the widest whitespace gutter, alternate axes) or a layout model's reading order; check that the gutter threshold is not wider than the actual gutter (25 px worked where 30 failed). Keep `reading_order` in the handoff so the chunker does not re-sort.
Teaches: [07](../18_ocr_and_document_parsing/07_reading_order_and_tables/WALKTHROUGH.md).

### Architecture / system design

**18.A1. Design a document-intelligence pipeline for 100k mixed invoices/contracts per month that feeds a RAG system.**
Ingest, per-page router (text layer / scan / hybrid), preprocessing (deskew, binarise, DPI), parser selection (hard constraints first: privacy, hardware, cost; then skills; test 50 of your own pages, do not trust leaderboards), layout and table structure, reading order, validation (field rules, confidence in a sidecar), human review queue for low confidence, handoff as typed elements with page, bbox in points (top-left origin), reading order and structured tables. Monitor CER/field accuracy on a rolling gold sample. Beware VLM OCR hallucination and omission: no confidence scores.
Teaches: [09 modern parsers 2026](../18_ocr_and_document_parsing/09_modern_document_parsers_2026/WALKTHROUGH.md), [10 parsed-document handoff](../18_ocr_and_document_parsing/10_parsed_document_handoff/WALKTHROUGH.md).

### Performance / optimisation

**18.P1. Accuracy on phone-photographed pages is poor. What do you try, in order?**
Measure by failure type first. Then: deskew (a +4 degree skew collapsed line finding to 1 line; after deskew 4), perspective correction, illumination handling (local/adaptive threshold beats Otsu under shadows), denoise with median not box blur (salt-and-pepper F1: raw 0.801, box 0.770, median 0.892), upscale small text to 200-300 dpi, augment training data (CNN 0.468 without augmentation, 0.782 with). Re-measure after each change, one at a time.
Teaches: [02 preprocessing](../18_ocr_and_document_parsing/02_preprocessing/WALKTHROUGH.md), [04 character recognition](../18_ocr_and_document_parsing/04_character_recognition/WALKTHROUGH.md).

---

## Module 19: Chunking

### Concept

**19.C1. Why measure chunks in tokens, and what is silent truncation?**
Embedding limits and LLM budgets are in tokens of that model's tokenizer; characters per token ranges from about 5.4 for English prose through 1.7 for JSON, 1.5 for Hindi to 1.0 for Chinese in MiniLM (with 19 of 24 characters `[UNK]`: wrong model). `max_seq_length` truncates silently: 481 tokens at max 256 leaves 225 tokens unread and gives the same vector as the truncated text. Beyond the position limit you get a `RuntimeError`.
Teaches: [01 tokens not characters](../19_chunking_deep_dive/01_tokens_not_characters/WALKTHROUGH.md).

**19.C2. Late chunking vs contextual retrieval vs small-to-big: what does each change?**
Semantic chunking decides where to cut; the other three change what is embedded or returned. Late chunking: run a long-context embedder over the whole window and mean-pool token vectors per chunk (no LLM, needs a model trained for it, e.g. Jina v3 with 8,192 tokens; with plain MiniLM hit@1 fell 0.69 to 0.54-0.58). Contextual retrieval: an LLM writes 1 to 2 situating sentences prepended before embedding and BM25 (Anthropic reports 35 percent fewer top-20 failures embeddings only, 49 percent with BM25, 67 percent with reranking). Small-to-big: embed small leaves, return the parent section; de-duplicate parents.
Teaches: [05 late chunking](../19_chunking_deep_dive/05_late_chunking/WALKTHROUGH.md), [06 contextual retrieval](../19_chunking_deep_dive/06_contextual_retrieval/WALKTHROUGH.md), [07 hierarchical](../19_chunking_deep_dive/07_hierarchical_parent_child/WALKTHROUGH.md).

### Math

**19.M1. A 1,000-token document, window 200, overlap 50. How many chunks and what is the storage overhead?**
Step = 150. Starts 0, 150, ..., 900: 7 windows (the last covers 900 to 1,000, shorter or shifted). Stored tokens are about 7 x 200 = 1,400, so 40 percent more than the document, more than the nominal "25 percent overlap" because the last window is mostly overlap. Overlap must be smaller than size or the step is zero or negative (the code asserts).
Teaches: [01](../19_chunking_deep_dive/01_tokens_not_characters/WALKTHROUGH.md), [02 recursive splitter](../19_chunking_deep_dive/02_recursive_splitter_rebuilt/WALKTHROUGH.md).

**19.M2. Contextual retrieval on a long document with 100 chunks: what does prompt caching save?**
Each chunk needs a call that contains the whole document. In our run the uncached input was 71,850 tokens versus 22,811 token-equivalents with the document prefix cached: 3.1x cheaper (cache reads cost a fraction of normal input, writes cost a little more). The index also grows (+67 percent stored tokens in our run). Prompt caching only works if the prefix is byte-identical: put the document first, the chunk last.
Teaches: [06](../19_chunking_deep_dive/06_contextual_retrieval/WALKTHROUGH.md), [20/07 caching](../20_llm_system_lld/07_caching/WALKTHROUGH.md).

**19.M3. Two chunkers score hit@3 of 0.91 and 0.95 on 74 questions. Is one better?**
The 95 percent interval is about `1.96 sqrt(0.9 x 0.1 / 74) = 0.07`, so the intervals overlap: not distinguishable. Compare with a paired bootstrap, on hit@1 and MRR, and report context tokens sent (bigger chunks make hit@k easier and the LLM's job harder), sliced by question kind.
Teaches: [10 chunking evaluation](../19_chunking_deep_dive/10_chunking_evaluation/WALKTHROUGH.md).

### Implementation

**19.I1. Write a recursive character/token splitter with overlap.**
`split(text, seps)`: pick the first separator that occurs; split, keeping separators (`keep_separator="start"` for headings, `"end"` for sentences); greedily merge small pieces up to `chunk_size` in tokens, carrying the last `overlap` tokens into the next chunk; recurse with finer separators on any piece still too big; last resort is a hard cut by token window. Validate `0 <= overlap < size`. Return offsets so you can cite and test that offsets reproduce the text.
Teaches: [02](../19_chunking_deep_dive/02_recursive_splitter_rebuilt/WALKTHROUGH.md).

**19.I2. Design chunk ids for idempotent re-indexing.**
`chunk_id = hash(doc_id, chunker_id, chunker_version, config_hash, text, occurrence)`, not the position. Re-chunk after editing one number: same ids for unchanged chunks (kept 18, added 1, removed 1 and only 1 text re-embedded of 19). Embedding cache keyed by text hash; build new versioned index `collection__chunker@version__confighash__embedmodel`, flip an alias, keep the old one for rollback.
Teaches: [11 chunker LLD](../19_chunking_deep_dive/11_chunker_lld/WALKTHROUGH.md).

### Debugging

**19.D1. Facts near the end of long chunks are never retrieved.**
Cause: the chunk exceeds the embedder's `max_seq_length` (256 in MiniLM), so the tail is silently dropped from the vector. Fix: count tokens with the embedder's own tokenizer, set chunk size below the limit, add a test asserting no chunk exceeds it, log truncation.
Teaches: [01](../19_chunking_deep_dive/01_tokens_not_characters/WALKTHROUGH.md).

**19.D2. Retrieved table fragments have no header; a code block is cut mid-fence.**
Cause: a structure-blind splitter cut inside a table or fenced code block (it cut 4 fences and made 2 headerless fragments on the handbook). Fix: parse blocks first, treat fenced code as one block, split tables by row groups with the header row repeated, close fences, put a breadcrumb (`Doc > H1 > H2`) in the embedded text (hit@1 0.73 to 0.78).
Teaches: [03 structure-aware markdown](../19_chunking_deep_dive/03_structure_aware_markdown/WALKTHROUGH.md).

**19.D3. After enabling overlap, each chunk ends with a heading that belongs to the next section.**
Cause: `keep_separator="end"` attaches the separator to the previous piece. Use `"start"` for heading-like separators. A related setup bug is `overlap == size`: assert or raise `ValueError` with a message.
Teaches: [02 Debugging practice](../19_chunking_deep_dive/02_recursive_splitter_rebuilt/WALKTHROUGH.md).

### Architecture / system design

**19.A1. Design chunking for a mixed corpus (handbooks, API docs, scanned PDFs) with citations, rollout and rollback.**
Layout-aware path for parsed PDFs (sort by page and reading order, drop headers/footers, merge tables across pages, bind captions, keep page and union bbox: 6 of 7 answered with a page and bbox citation versus 0 of 7 for flatten-and-window); structure-aware path for Markdown; recursive fallback at 100 to 250 tokens; breadcrumbs; small-to-big where context matters. Frozen `Chunk` objects, `Chunker` protocol and registry, versioned indexes, stream-friendly offsets, tests (determinism, offsets, max tokens, edit locality). Choose by your own gold questions.
Teaches: [09 layout-aware chunking](../19_chunking_deep_dive/09_layout_aware_chunking_from_parsed_docs/WALKTHROUGH.md), [11](../19_chunking_deep_dive/11_chunker_lld/WALKTHROUGH.md).

### Performance / optimisation

**19.P1. Semantic chunking is proposed. How do you decide whether it is worth it?**
It costs a model pass over every sentence (about 14,665 tokens, 3x the corpus with buffer 1) before the chunks are embedded again. Percentile rules always cut something even in one-topic text. In our runs it tied recursive/sentence-packing (hit@3 0.86 to 0.95 vs 0.91 to 0.92, within noise) and was worse without headings (0.73 vs 0.91). Run the A/B on your own questions with CIs and cost columns; adopt only for a measured gain.
Teaches: [04 semantic chunking](../19_chunking_deep_dive/04_semantic_chunking/WALKTHROUGH.md).

---

## Module 20: LLM system LLD

### Concept

**20.C1. Protocol vs ABC, and why constructor injection?**
A Protocol (structural typing) describes a contract without inheritance, so any object with the right methods fits and fakes are trivial; an ABC fits when you also ship shared behaviour, for example a Template Method base chunker. Runtime `isinstance` on a Protocol only checks method names. Constructor injection puts volatile parts (model, store, clock) behind seams so tests use fakes, production uses adapters, and a composition root (`build_system(config)`) wires them.
Teaches: [01 LLD foundations](../20_llm_system_lld/01_lld_foundations/WALKTHROUGH.md), [05 design patterns](../20_llm_system_lld/05_design_patterns_in_llm_apps/WALKTHROUGH.md).

**20.C2. Exact cache vs semantic cache vs provider prompt caching.**
Exact: key = hash of normalised prompt, params and scope (including index version); safe. Semantic: embedding similarity above a threshold; saves more but can return the answer to a near-miss ("not free"), so use a high threshold, same tenant and scope, never cache refusals. Provider prompt caching reuses the processed byte-identical prefix server-side (tools, system, then messages; cache writes cost a premium, reads a fraction): put stable content first and volatile content last. They compose.
Teaches: [07 caching](../20_llm_system_lld/07_caching/WALKTHROUGH.md).

### Math

**20.M1. A token bucket has capacity 10 and refills 2 tokens/s. At t=0 ten requests arrive, at t=0.5 s two more. Which pass?**
After the burst the bucket is empty. At t=0.5 s it has `2 x 0.5 = 1` token: the first of the two passes, the second is rejected (or must wait 0.5 s). Long-run rate is 2 req/s with burst up to 10. Use a monotonic clock and an injectable clock for tests. For LLM APIs run two buckets (requests per minute and tokens per minute) and check both before spending either.
Teaches: [06 resilience](../20_llm_system_lld/06_resilience/WALKTHROUGH.md), [10 LLD problems](../20_llm_system_lld/10_lld_interview_problems/WALKTHROUGH.md).

**20.M2. Three stacked layers each retry 3 times on a failing provider. How many calls hit the provider? What is full-jitter backoff for attempt n?**
Retries multiply: `3 x 3 x 3 = 27` calls from one user request, which hammers a degraded provider. Retry in one layer only, with a cap, and add a circuit breaker. Full jitter: `sleep = uniform(0, min(cap, base x 2^n))`, for base 0.5 s and n = 3: up to 4 s; honour `retry-after`. Retry only transient errors (429/5xx/timeouts) and only idempotent operations (idempotency key).
Teaches: [06](../20_llm_system_lld/06_resilience/WALKTHROUGH.md).

**20.M3. A 10,000-token system prefix is sent 100 times. Prompt caching: writes 1.25x input price, reads 0.1x. Savings?**
Uncached: 100 x 1.0 = 100 units. Cached: 1.25 + 99 x 0.1 = 11.15 units: about 89 percent cheaper on that prefix. Break-even: `1.25 + 0.1 n < n + 1` holds for n >= 1 read. Caveat: the entry expires (default 5 minutes, refreshed on hits) and any change in the prefix bytes invalidates it.
Teaches: [07](../20_llm_system_lld/07_caching/WALKTHROUGH.md).

### Implementation

**20.I1. Implement a thread-safe token-bucket rate limiter with an injectable clock.**
Fields: `capacity`, `rate`, `tokens`, `last` (from `clock.monotonic()`), a lock. `_refill`: `tokens = min(capacity, tokens + (now-last) x rate)`. `try_acquire(n)`: refill, if `tokens >= n` subtract and return True else False (and optionally return the wait time `(n - tokens)/rate`). Test with a fake clock: no sleeping, deterministic. For tokens-per-minute, reserve worst-case tokens (input plus max output) before the call and settle with actuals after.
Teaches: [06](../20_llm_system_lld/06_resilience/WALKTHROUGH.md), [10](../20_llm_system_lld/10_lld_interview_problems/WALKTHROUGH.md).

**20.I2. Make RAG ingestion idempotent and incremental.**
Chunk id = content hash; keep a manifest of document hashes and the index version (embedder plus chunker). On each run diff: new, stale (delete), unchanged (skip); embed only new chunks in batches; update the manifest after success; failures go to a dead-letter record with stage and error; same input twice gives the same store.
Teaches: [03 ingestion pipeline](../20_llm_system_lld/03_ingestion_pipeline/WALKTHROUGH.md).

### Debugging

**20.D1. The service retries a 401 "invalid API key" four times and then fails after 10 seconds.**
Cause: retry policy is `retry_on=(Exception,)`: permanent errors are treated as transient. Fix: classify errors (transient: 429, 5xx, timeouts; permanent: 400/401/403/404), fail fast on permanent, alert on auth failures.
Teaches: [06 BREAK IT](../20_llm_system_lld/06_resilience/WALKTHROUGH.md).

**20.D2. Users get an outdated answer after documents were re-indexed; another user got an answer meant for a different question.**
Stale: cache keys did not include the index version, so invalidate by version (or TTL). Wrong answer: semantic cache threshold too low or shared across tenants/scopes. Fix: version and tenant in the key, higher threshold, cache only verified (non-refusal) answers, measure false-hit rate on labelled near-miss pairs.
Teaches: [07](../20_llm_system_lld/07_caching/WALKTHROUGH.md).

**20.D3. Resilience tests are flaky and slow.**
Real `time.sleep` and `time.time` in the code. Inject a `Clock`, use a `FakeClock` that advances on `sleep`, and a scripted `FlakyGenerator`; then breaker open/half-open timing and backoff are deterministic and tests run instantly.
Teaches: [06](../20_llm_system_lld/06_resilience/WALKTHROUGH.md).

### Architecture / system design

**20.A1. Design the classes of a RAG service (LLD round).**
Frame: requirements, entities (Document, Element, Chunk, Embedding, SearchHit, Citation, Answer as frozen value objects), interfaces (Loader, Parser, Chunker, Embedder, VectorStore, Retriever, Reranker, Generator as Protocols), flows (ingestion offline; query online with hybrid retrieval, rerank, token-budget prompt builder, citation verifier, refusal policy before generation), patterns with reasons (Strategy for retrievers, Registry/Factory for plugins, Adapter for Qdrant, Decorator for cache/tracing/retry, Template Method for chunkers), failure handling, tests with fakes. Draw the class diagram first.
Teaches: [10 LLD interview problems](../20_llm_system_lld/10_lld_interview_problems/WALKTHROUGH.md), [04 query pipeline](../20_llm_system_lld/04_query_pipeline/WALKTHROUGH.md), [02 domain model](../20_llm_system_lld/02_domain_model/WALKTHROUGH.md).

**20.A2. Design an LLM gateway with fallback, streaming and cost metering.**
Unified request/response types; alias to provider list with ordered fallback; retry and circuit breaker per provider; streaming as an iterator of typed chunks (sync and async); structured output with a bounded validate-and-repair loop; tool schemas; a `CostMeter` that records input, output and cached tokens per request; budgets and guardrails; tracing with a trace id.
Teaches: [08 gateway and streaming](../20_llm_system_lld/08_llm_gateway_and_streaming/WALKTHROUGH.md), [11 versioning, guardrails, budgets](../20_llm_system_lld/11_versioning_guardrails_and_budgets/WALKTHROUGH.md).

### Performance / optimisation

**20.P1. p95 latency and cost per request both doubled after a release. How do you find out why?**
Per-stage spans (retrieve, rerank, generate), tokens and cost per request, p50/p95 per stage, cache hit rate, retry and refusal rates, join logs and traces by trace id. Likely suspects: prompt grew (token budget), cache keys changed (hit rate fell), retries amplified, a fallback model is slower or pricier. Fixes: restore prefix stability for prompt caching, stream the answer to improve perceived latency, enforce the context token budget, cap retries.
Teaches: [09 observability](../20_llm_system_lld/09_observability_and_eval_hooks/WALKTHROUGH.md), [07](../20_llm_system_lld/07_caching/WALKTHROUGH.md).

---

## Module 21: LLM inference engineering (top priority)

Teaching steps: [01 checkpoints](../21_llm_inference_engineering/01_checkpoint_formats_and_loading/WALKTHROUGH.md) | [02 GGUF](../21_llm_inference_engineering/02_gguf_and_llamacpp_formats/WALKTHROUGH.md) | [03 memory](../21_llm_inference_engineering/03_gpu_memory_calculator/WALKTHROUGH.md) | [04 quant math](../21_llm_inference_engineering/04_quantization_math/WALKTHROUGH.md) | [05 activations](../21_llm_inference_engineering/05_activation_quant_and_calibration/WALKTHROUGH.md) | [06 GPTQ/AWQ](../21_llm_inference_engineering/06_gptq_awq_from_scratch/WALKTHROUGH.md) | [07 KV quant](../21_llm_inference_engineering/07_kv_cache_quantization/WALKTHROUGH.md) | [08 HF inference](../21_llm_inference_engineering/08_hf_inference_batching_kvcache/WALKTHROUGH.md) | [09 roofline](../21_llm_inference_engineering/09_prefill_decode_roofline/WALKTHROUGH.md) | [10 benchmark](../21_llm_inference_engineering/10_latency_throughput_benchmark/WALKTHROUGH.md) | [11 paging](../21_llm_inference_engineering/11_paged_attention_fragmentation/WALKTHROUGH.md) | [12 batching and prefix](../21_llm_inference_engineering/12_continuous_batching_prefix_caching/WALKTHROUGH.md) | [13 speculative](../21_llm_inference_engineering/13_speculative_decoding/WALKTHROUGH.md) | [14 parallelism](../21_llm_inference_engineering/14_parallelism_and_offloading/) | [15 engines](../21_llm_inference_engineering/15_serving_engines_landscape/WALKTHROUGH.md) | [16 cost](../21_llm_inference_engineering/16_cost_per_token/) | [17 report](../21_llm_inference_engineering/) (steps 14, 16, 17 link a folder until their walkthroughs land). Cram sheet: [INTERVIEW_GIST.md](../21_llm_inference_engineering/INTERVIEW_GIST.md).

### Concept

**21.C1. Why is a pickle checkpoint dangerous and what is safetensors?**
Unpickling executes whatever callable the file names (`REDUCE` opcode), so loading an untrusted `.bin`/`.pt` is arbitrary code execution; in our run a 108-byte pickle printed a message during `pickle.loads`, and `torch.load` with default `weights_only` refused it (`UnpicklingError: Weights only load failed`). Safetensors is an 8-byte little-endian length, a JSON header (dtype, shape, byte offsets) and raw bytes: numbers only, memory-mappable, so you can open a huge file instantly and touch only the tensors you need. Large models are sharded as `model-0000X-of-0000N.safetensors` plus an `index.json` `weight_map`. Do not "fix" the error with `weights_only=False`; convert in a trusted environment.
Teaches: [01 checkpoint formats](../21_llm_inference_engineering/01_checkpoint_formats_and_loading/WALKTHROUGH.md).

**21.C2. What is GGUF, and what does "Q4_K_M" mean?**
GGUF is llama.cpp's single self-describing file: header (magic, version, counts), typed metadata including the tokenizer, tensor infos, then aligned (32-byte) tensor data that is memory-mapped. Q8_0 is blocks of 32 with an fp16 scale (8.5 bits per weight). K-quants use 256-weight super-blocks with sub-block scales; `_M` means a medium mix in which sensitive tensors keep more bits, so Q4_K_M is about 4.85 bits per weight, the common default; a 7B model is about 4.1 GB versus 13.5 GB in fp16. imatrix (importance matrix) weights the quantisation error by calibration activations. Path: HF safetensors, `convert_hf_to_gguf.py`, `llama-quantize`; Ollama wraps llama.cpp with a registry and REST API.
Teaches: [02 GGUF and llama.cpp](../21_llm_inference_engineering/02_gguf_and_llamacpp_formats/WALKTHROUGH.md).

**21.C3. Explain prefill vs decode and which resource limits each.**
Prefill processes the whole prompt in one forward pass of matrix-matrix products and fills the KV cache: compute-bound, sets time-to-first-token (TTFT). Decode generates one token per pass, re-reading all weights and the whole KV cache for little arithmetic: memory-bandwidth-bound, sets time-per-output-token (TPOT, also called inter-token latency, ITL). Intensity of a decode step is about `B` FLOP per byte in 16-bit (B = batch), far below a GPU's ridge point, which is why batching is nearly free in decode and why weight quantisation speeds decode but barely helps prefill. In our run on CPU: 121-token prefill 73 ms (0.6 ms/token) vs 17 ms for one decode token.
Teaches: [09 prefill/decode roofline](../21_llm_inference_engineering/09_prefill_decode_roofline/WALKTHROUGH.md), [08](../21_llm_inference_engineering/08_hf_inference_batching_kvcache/WALKTHROUGH.md).

**21.C4. What is PagedAttention and what problem does it solve?**
Naive servers reserve a contiguous KV slab for each request's maximum length: most of it is never used (internal waste) and the free space splits into unusable holes (external fragmentation); the vLLM paper measured only 20-40 percent of KV memory holding real tokens. PagedAttention splits the cache into fixed-size blocks (16 tokens) and gives each sequence a block table mapping logical to physical blocks, like virtual memory pages; the attention kernel reads K/V through the table. Waste drops to the last partial block, and blocks can be shared (prefix caching, parallel sampling) with copy-on-write. In our simulation, 8 GB of cache fit 8 requests contiguously versus 53 paged (6.6x). Do not confuse it with FlashAttention (an attention kernel) or SDPA (PyTorch's entry point).
Teaches: [11 PagedAttention and fragmentation](../21_llm_inference_engineering/11_paged_attention_fragmentation/WALKTHROUGH.md).

### Math (with worked numbers)

**21.M1. How much GPU memory do you need to serve Llama-3-8B in bf16 with 32 concurrent sequences of 4,096 tokens?**
Weights: `8.03B x 2 B = 16 GB`. KV per token: `2 x 32 x 8 x 128 x 2 = 131,072 B = 128 KiB`; per sequence at 4,096 tokens: 0.5 GiB; for 32 sequences: 16 GiB (17.2 GB). Subtotal about 33 GB; add about 10 percent for activations, CUDA context and fragmentation: about 36 GB. So it does not fit a 24 GB card and fits an 80 GB card with room for more sequences. With an fp8 cache the KV part halves (8.6 GB), with int4 weights as well (4 GB) the total is about 14 GB. In our calculator a 24 GB card with fp16 7B weights held only about 15.6k cached tokens. GB (1e9) vs GiB (2^30) differ by about 7 percent: say which.
Teaches: [03 memory calculator](../21_llm_inference_engineering/03_gpu_memory_calculator/WALKTHROUGH.md).

**21.M2. Upper-bound the batch-1 decode speed of an 8B bf16 model on an A100 and an H100, and say at what batch it becomes compute-bound.**
Each token must stream all weights: 16 GB. Speed bound `= bandwidth / bytes`: A100 (2.039 TB/s) about 127 tokens/s; H100 (3.35 TB/s) about 209 tokens/s (vendor spec values; real engines reach a fraction, and KV reads add bytes). int4 weights (about 4.1 GB with scales) roughly quadruple the bound. FLOPs per token are about `2 x params`; intensity `= 2 P B / (2 P) = B` FLOP/byte; ridge point = peak FLOPs / bandwidth: about 153 (A100, 312 TFLOP/s) and about 295 (H100, 989 TFLOP/s), so decode only becomes compute-bound near batch 150 to 300, long after the KV cache has usually filled memory. On our CPU the ridge was 20.6 FLOP/byte and decode at batch 64 was compute-bound.
Teaches: [09](../21_llm_inference_engineering/09_prefill_decode_roofline/WALKTHROUGH.md).

**21.M3. Speculative decoding: acceptance alpha = 0.8, draft length gamma = 4, draft cost ratio c = 0.1. Expected speed-up?**
Expected tokens per target pass `(1 - alpha^(gamma+1)) / (1 - alpha) = (1 - 0.328)/0.2 = 3.36`. Cost of one round `= gamma c + 1 = 1.4` target-pass equivalents. Speed-up `= 3.36 / 1.4 = 2.4x`. With a poor draft (c = 0.5) and alpha 0.62 (our measured value) it is `2.39/3.0 = 0.8x`: slower than not speculating. In our gpt2/distilgpt2 run acceptance fell 0.75, 0.62, 0.47 as gamma went 2, 4, 8, target passes dropped 3x, but wall-clock gain was only 1.11x because c = 0.65. Output equals the target alone (greedy identical; sampling uses `min(1, p/q)` and resample from `norm(max(0, p-q))`). Gain is largest at small batch.
Teaches: [13 speculative decoding](../21_llm_inference_engineering/13_speculative_decoding/WALKTHROUGH.md).

**21.M4. Compute $ per 1M output tokens for an H100 at $3.50/hour sustaining 2,500 tokens/s, then at 40 percent utilisation.**
`3.50 / (2500 x 3600) x 1e6 = $0.389` per 1M tokens at full utilisation. You pay for idle hours too: `0.389 / 0.4 = $0.97`. Output (decode) tokens cost more than input (prefill) tokens because decode is memory-bound and slow per token; APIs price output several times input and cached input at a fraction. Levers: batch (throughput up, cost down, user speed flat), quantisation, prefix-cache hit rate, utilisation. Break-even against an API is a monthly token volume; recompute with current prices (example prices only).
Teaches: [16 cost per token](../21_llm_inference_engineering/16_cost_per_token/).

**21.M5. Quantise a weight tensor to int4 with group size 128 and an fp16 scale. Bits per weight? Size of a 7B model? What about per-tensor?**
`bpw = 4 + 16/128 = 4.125`; `7e9 x 4.125 / 8 = 3.61 GB` versus 14 GB bf16. Add an fp16 zero-point (asymmetric) and it is 4.25 bpw. Quality dial, in our run on a real distilgpt2 matrix at int4: per-tensor 8.1 dB SQNR, per-row 17.1, group 128 18.6, group 32 20.2 (4.5 bpw). Affine formula `q = round(x/s) + z`, `s = (max-min)/(2^b - 1)`; symmetric wastes half its levels on one-sided (post-ReLU) data.
Teaches: [04 quantisation math](../21_llm_inference_engineering/04_quantization_math/WALKTHROUGH.md).

### Implementation (what you would write)

**21.I1. Write int4 packing and unpacking with per-group scales.**
Quantise: reshape the weights to `(n_groups, 128)`, `s = absmax/7` per group, `q = clip(round(w/s), -8, 7)`. Pack: store `q + 8` (0..15), `byte = lo | (hi << 4)` where `lo` is element 2i and `hi` element 2i+1, giving `uint8` of half the length. Unpack: `lo = b & 0xF`, `hi = b >> 4`, interleave, subtract 8, multiply by the group scale. Test `unpack(pack(q)) == q` and that dequantised error is within `s/2`. Mention the real kernels fuse dequantisation into the matmul. In our run 1M values went from 1,000,000 B to 500,000 B and the example byte for codes (-4, 7) was `0xF4`.
Teaches: [04](../21_llm_inference_engineering/04_quantization_math/WALKTHROUGH.md).

**21.I2. Write a safetensors header reader.**
Read 8 bytes as little-endian `u64` (`struct.unpack("<Q")`) = header length n; read n bytes of JSON; for each tensor entry `{dtype, shape, data_offsets: [start, end]}`, the data begin at `8 + n + start`; `np.frombuffer` (or `np.memmap`) with the dtype and reshape. Verify equal to the original. Traps: endianness, forgetting the data base offset (silently reads header bytes), `__metadata__` key in the JSON. Shows why loading is lazy: only touched pages are read (in our run eager load peak RSS 628 MB versus 131 MB reading one tensor from a 300 MB file).
Teaches: [01](../21_llm_inference_engineering/01_checkpoint_formats_and_loading/WALKTHROUGH.md).

**21.I3. Write a block-hash prefix cache.**
Split the prompt into blocks of 16 tokens; hash each full block with the previous block's hash chained in (`h_i = hash((h_{i-1}, tuple(block_i)))`) so a match implies the whole prefix matches; keep `hash -> physical block` with a reference count and an LRU of unreferenced blocks; on a request, walk blocks until the first miss, reuse those blocks, prefill only the rest, then register new full blocks. Partial last block is never cached (block hashing matched 496 of 500 shared tokens; a radix tree matches 500). Eviction under memory pressure; copy-on-write if a shared block would be written.
Teaches: [12 continuous batching and prefix caching](../21_llm_inference_engineering/12_continuous_batching_prefix_caching/WALKTHROUGH.md).

**21.I4. Write a latency/throughput benchmark for an LLM. What does the harness record?**
Warm up (2 runs), fix the output length (force no EOS), several trials, and per request timestamps for first token and every token. Derive TTFT = first token minus start; ITL = gaps between tokens (report p50/p95/p99); TPOT = `(E2E - TTFT)/(n-1)`; throughput = total tokens / wall time; per-user tokens/s; goodput = throughput of requests that meet the SLO. Report mean and spread (coefficient of variation up to 14 percent in our run), prompt length, dtype, threads/device. On GPU, synchronise before reading the clock. At serving level use an open-loop arrival process (Poisson), not a closed loop.
Teaches: [10 latency and throughput benchmark](../21_llm_inference_engineering/10_latency_throughput_benchmark/WALKTHROUGH.md).

### Debugging (symptom, cause, fix)

**21.D1. Batched generation: the short prompt in the batch produces junk (`'Hello, my name is\n\n\n\n'`), batch size 1 is fine.**
Cause: right padding with a decoder-only model: the last position is a pad token, so generation continues from padding. Only a one-line warning is printed. Fix: `tokenizer.padding_side = "left"`, set a pad token (GPT-2 has none: use EOS), pass the attention mask. Habit: a batch result must equal the batch-1 result. In our run left padding gave `'Hello, my name is J.J.K. Rowling.'`.
Teaches: [08 HF inference](../21_llm_inference_engineering/08_hf_inference_batching_kvcache/WALKTHROUGH.md).

**21.D2. The model loads fine but serving OOMs once long prompts arrive, though weights are only 16 GB on an 80 GB GPU.**
Cause: KV cache grows with `layers x kv_heads x head_dim x tokens x batch`; at 128 KiB per token (Llama-3-8B bf16) a 128k-token request needs 17.2 GB and a few of those fill the card; naive reservation of max length makes it worse (85 percent of reservation wasted in our simulation). Fix: compute the budget (step 03 calculator), cap `max_model_len` and concurrency, enable paged KV (vLLM) and `gpu_memory_utilization`, quantise the KV cache (fp8: half), use GQA models, chunk prefill, admission control.
Teaches: [03](../21_llm_inference_engineering/03_gpu_memory_calculator/WALKTHROUGH.md), [11](../21_llm_inference_engineering/11_paged_attention_fragmentation/WALKTHROUGH.md).

**21.D3. You enabled prefix caching for a shared 480-token system prompt but the hit rate is 0 percent.**
Check: (1) cache smaller than the working set: with 16 blocks and 4 prompts x 30 blocks LRU thrashed to 0 percent hits, 128 blocks gave 63 percent and 4,096 blocks 85 percent; (2) the prefix is not byte-identical (timestamp, request id or user name at the top of the system prompt, different tool order); (3) different chat template or tokenisation at the boundary; (4) block hashing is not chained (a related bug gives better-looking hit rates and wrong text). Fix: put volatile text after the stable prefix, size the cache, monitor hit rate. Sharing by mutation: if you reuse an HF cache object without `deepcopy`, the second call sees the first call's suffix (wrong logits).
Teaches: [12 BREAK IT](../21_llm_inference_engineering/12_continuous_batching_prefix_caching/WALKTHROUGH.md).

**21.D4. After W8A8 quantisation perplexity went from 25.4 to 33 (per-tensor), and a percentile-calibrated variant gave 142.**
Cause: activations have outlier channels (max/median up to 58.8 across 24 layers in distilgpt2; one channel set the per-tensor scale), so one scale wastes the int8 grid; clipping at the 99.9th percentile clips exactly those outliers that carry signal, which is catastrophic. Fixes in order of cost: per-token dynamic activation scales (25.55 in our run), SmoothQuant (divide outlier channels, multiply weight columns; 30.79 per-tensor), LLM.int8-style mixed precision for the outlier channels, or weight-only (W8A16 25.42). Evaluate with perplexity or KL on held-out text.
Teaches: [05 activation quantisation](../21_llm_inference_engineering/05_activation_quant_and_calibration/WALKTHROUGH.md).

**21.D5. Your speculative decoding output differs from the target model's greedy output.**
It must be identical in greedy mode: any difference is a bug. Check: accept rule compares the draft token to the target's argmax at the same position (off-by-one in the verify pass); the bonus token is taken from the correct position; KV caches of target and draft are rolled back to the accepted length; draft and target share a tokenizer (same vocab: True in our run); batch/padding differences. For sampling, verify with a distribution test (empirical `[0.5, 0.302, 0.149, 0.049]` against `[0.5, 0.3, 0.15, 0.05]`).
Teaches: [13](../21_llm_inference_engineering/13_speculative_decoding/WALKTHROUGH.md).

### Architecture / system design

**21.A1. Design the serving stack for a 70B chat model with a P95 TTFT under 500 ms and 200 concurrent users.**
Sizing first: weights 140 GB bf16 (about 70 GB fp8, about 35 GB int4), so 2 to 4 80 GB GPUs with tensor parallelism (Megatron split: column-parallel then row-parallel, one all-reduce per block; needs NVLink-class bandwidth) or one node of quantised weights; KV budget per token (GQA 70B: about 320 KiB bf16) decides concurrency. Engine: vLLM or SGLang for continuous batching, paged KV, chunked prefill and prefix caching; fp8 KV cache. Routing: load balancer with prefix-aware or session affinity, queue with admission control, autoscaling on queue depth and KV utilisation. Streaming via SSE. Observability: TTFT, ITL, queue time, KV utilisation, hit rate, p50/p95/p99. Consider disaggregated prefill/decode at scale, speculative decoding for latency at low batch. Validate with an open-loop load test at target QPS and report goodput against the SLO.
Teaches: [14 parallelism and offloading](../21_llm_inference_engineering/14_parallelism_and_offloading/), [15 serving engines](../21_llm_inference_engineering/15_serving_engines_landscape/WALKTHROUGH.md), [12](../21_llm_inference_engineering/12_continuous_batching_prefix_caching/WALKTHROUGH.md).

**21.A2. vLLM, SGLang, TensorRT-LLM, llama.cpp or Ollama: how do you choose?**
Choose by constraints, not fame. vLLM: general GPU serving, paged attention, continuous batching, OpenAI-compatible API, broad model support. SGLang: strong prefix reuse (RadixAttention) and structured/agentic workloads. TensorRT-LLM: NVIDIA-only, compiled engines, highest peak throughput with more build effort. llama.cpp/Ollama: CPU, laptop and edge, GGUF quantisation, simple deployment; Ollama adds a model registry and REST API. Also hosted APIs when volume is low. Say that features and rankings change quickly and that you would benchmark your own prompts and SLO on two candidates before committing.
Teaches: [15](../21_llm_inference_engineering/15_serving_engines_landscape/WALKTHROUGH.md).

**21.A3. Which quantisation do you pick for (a) a laptop assistant, (b) a GPU API, (c) a long-context service?**
(a) GGUF Q4_K_M (or Q5_K_M if quality-sensitive), about 4.8 bpw, run in llama.cpp/Ollama. (b) Weight-only int4 (GPTQ/AWQ, group 128) when decode speed and memory dominate, or FP8 (W8A8) on hardware with native FP8 for compute throughput; produce checkpoints with a calibration-based tool; evaluate against the bf16 baseline. (c) Weights plus an fp8 (or int8) KV cache, because the cache dominates memory at length; 4-bit caches (KIVI-style: K per channel, V per token) are still emerging. GPTQ in our run at 3-bit g128 reached perplexity 29.0 versus RTN 42.2 (fp32 25.4); at 4-bit the methods were within a few points, so the extra machinery matters mostly below 4 bits. Always re-check accuracy on your task.
Teaches: [06 GPTQ and AWQ](../21_llm_inference_engineering/06_gptq_awq_from_scratch/WALKTHROUGH.md), [07](../21_llm_inference_engineering/07_kv_cache_quantization/WALKTHROUGH.md), [02](../21_llm_inference_engineering/02_gguf_and_llamacpp_formats/WALKTHROUGH.md).

**21.A4. Design inference for a RAG assistant where every request carries the same 2,000-token system prompt plus 6,000 tokens of retrieved context.**
Workload is prefill-heavy (8,000 prompt tokens to maybe 300 output tokens), so TTFT and prefill throughput dominate. Enable prefix caching for the static system prompt (and few-shot examples), keep it byte-stable and first; retrieved chunks vary, so the savings are on the stable part only; chunked prefill so long prompts do not stall other users' decode; consider disaggregating prefill and decode pools; use provider prompt caching if using an API; shrink context (rerank, trim chunks) since it costs both prefill time and KV memory; cap `max_tokens`. Measure TTFT versus context length and the cache hit rate.
Teaches: [12](../21_llm_inference_engineering/12_continuous_batching_prefix_caching/WALKTHROUGH.md), [09](../21_llm_inference_engineering/09_prefill_decode_roofline/WALKTHROUGH.md), [20/07](../20_llm_system_lld/07_caching/WALKTHROUGH.md).

### Performance / optimisation

**21.P1. TTFT is fine at low load and then explodes at 4 requests per second. Why, and what do you change?**
Static batching holds a slot until the longest request finishes and new requests queue: in our simulation at 4 req/s static and dynamic batching saturated near 165 tokens/s with mean latency 36 s and p95 74 s; continuous batching (per-iteration scheduling, finished sequences leave immediately) kept 269 tokens/s, 2.1 s mean and 5.5 s p95 (mean TTFT 34 s versus 0.07 s). At 2 req/s all schedulers had the same throughput: the server was not the bottleneck; only TTFT differed (4.0 s vs 0.06 s). Fixes: continuous batching, chunked prefill, prefix caching, more replicas, admission control.
Teaches: [12](../21_llm_inference_engineering/12_continuous_batching_prefix_caching/WALKTHROUGH.md).

**21.P2. Users complain about stutter in streaming when someone submits a very long prompt.**
A long prefill occupies the iteration and delays everyone's next decode step (worst inter-token gap 500 ms in our run). Chunked prefill puts a token budget on prompt work per iteration: worst gap fell to 64 ms, paid for by a higher mean TTFT (0.17 s to 0.27 s) for those long prompts. Trade-off to state: bounded ITL versus the long request's TTFT. Alternative at scale: disaggregated prefill/decode.
Teaches: [12](../21_llm_inference_engineering/12_continuous_batching_prefix_caching/WALKTHROUGH.md).

**21.P3. How do you pick the batch size or concurrency for production?**
Measure the curve and choose the largest batch that still meets the SLO: that is where goodput peaks. In our CPU run tokens/s rose from 65 at batch 1 to 644 at batch 16 (about 10x) while per-user speed fell only from 64 to 40 tok/s and TTFT grew from 22 to 84 ms; with an SLO of TPOT at most 22 ms goodput peaked at batch 8 (300 of 375 tok/s) and fell to 129 at batch 16. On real GPUs the wall is usually KV memory before compute. Report percentiles, not means, and repeat trials (CV up to 14 percent).
Teaches: [10](../21_llm_inference_engineering/10_latency_throughput_benchmark/WALKTHROUGH.md), [09](../21_llm_inference_engineering/09_prefill_decode_roofline/WALKTHROUGH.md).

**21.P4. KV cache memory is the bottleneck. List the levers in order of effort and cite the numbers.**
(1) Do not waste it: paging (6.6x more requests in the same memory in our simulation). (2) Architecture: GQA (32 to 8 KV heads: 4x; Llama-3-8B 128 KiB/token), MLA (about 71x smaller than MHA for V3). (3) Quantise it: fp8/int8 halves (Llama-3-8B-shaped 32k context: 4.29 GB to 2.15 GB), int4 with group scales about 1.34 GB; K per channel, V per token (KIVI); int8 and FP8 were essentially free (KL about 0, 24/24 greedy tokens) while int2 was not. (4) Share it: prefix caching. (5) Bound it: sliding window or local layers, eviction keeping the attention sinks (first tokens), cap context. (6) Offload or disaggregate.
Teaches: [07 KV-cache quantisation](../21_llm_inference_engineering/07_kv_cache_quantization/WALKTHROUGH.md), [11](../21_llm_inference_engineering/11_paged_attention_fragmentation/WALKTHROUGH.md), [16/04](../16_modern_llm_architecture/04_attention_variants_and_kv_cache/WALKTHROUGH.md).

---

Next: [live_coding_drills.md](live_coding_drills.md), [debugging_drills.md](debugging_drills.md), [mock_scripts.md](mock_scripts.md), [scorecard.md](scorecard.md).
