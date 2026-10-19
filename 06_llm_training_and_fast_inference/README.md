# Module 06 — LLM Training and Fast Inference

Syllabus topic 1 (second half): dataset preparation, the training loop, LoRA/QLoRA, fast inference (Hugging Face, vLLM) and the latency / throughput / GPU-memory trade-off.

## 1. Why this matters
Interviewers ask "how would you fine-tune a model on one GPU?" and "how do you serve it to many users?". You need to *show* a training loop, explain LoRA with numbers, and reason about memory. Everything here runs on a laptop with tiny models; the reasoning is identical for 7B models.

## 2. Where this fits
**Builds on**
- [../03_ml_foundations/](../03_ml_foundations/): the forward -> loss -> backward -> step loop (`04_pytorch_training_loop`), train/validation split and overfitting (`05_overfitting_train_val`), cross-entropy (`06_softmax_cross_entropy`). Step 02 here is the same loop on a language model.
- [../04_llm_fundamentals/](../04_llm_fundamentals/): the tokenizer (`02_tokenizer`), loading a pretrained model (`04_load_pretrained_model`), logits and next-token prediction (`05_forward_pass_logits`) and the generation loop (`06_generation_loop`). The loss in step 02 is cross-entropy on exactly those logits; `07_pretrain_vs_finetune_vs_inference` is the overview that this module makes concrete.
- [../05_llm_parameters/](../05_llm_parameters/): learning rate, batch size, epochs (`06_training_parameters`), generation parameters (`02_sampling_from_scratch`, `03_hf_generate_compare`) and quantization (`05_quantization`). Here they become measured effects: the learning-rate table in step 02, bytes-per-parameter in step 08.

**Leads to**
- [../07_prompt_engineering/](../07_prompt_engineering/) and [../08_agent_harness/](../08_agent_harness/): a fine-tuned or served model is what those layers call; vLLM exposes an OpenAI-style endpoint your harness can use.
- [../09_api_development/](../09_api_development/): wrap the generation code from step 07 in a FastAPI endpoint (model loaded once at startup, `no_grad`, left padding).
- [../10_ml_system_architecture/](../10_ml_system_architecture/): checkpoints (step 03), adapters as versioned artifacts (step 05), latency/throughput targets (step 09).
- [../11_docker/](../11_docker/): why model files are mounted as volumes, why image size depends on weights (step 08 arithmetic).
- [../13_capstone_project/](../13_capstone_project/): combines a served model, an API and a UI.

## 3. Concept map
```
raw text ──tokenize──> token IDs ──chunk/pad/mask──> batches (01)
                                              │
                         ┌────────────────────┴────────────────────┐
                         v                                         v
              FULL fine-tuning (02,03)                      LoRA / QLoRA (04,05,06)
        forward→loss→backward→clip→step               freeze W, train B·A (r small)
        eval: val loss, perplexity, checkpoint        save adapter (0.6 MB), merge
        memory ≈ 16 B/param  ──────────────────────>  memory ≈ 2 B/param (LoRA), 0.5 (QLoRA)
                         │                                         │
                         └──────────────┬──────────────────────────┘
                                        v
                      INFERENCE (07,08,09,10): no_grad, KV cache, batching, dtype
                      memory = weights + KV cache; latency vs throughput
                      production: vLLM (PagedAttention + continuous batching)
```

## 4. Study path
| Step | Folder | You learn | Run | Time |
|---|---|---|---|---|
| 1 | [01_dataset_preparation](01_dataset_preparation/) | text -> tokens -> blocks -> labels -> DataLoader, padding masks, -100 | `python main.py` | 60 min |
| 2 | [02_training_loop](02_training_loop/) | the 4-beat loop, AdamW, grad clipping, loss curve, lr effect | `python main.py` | 75 min |
| 3 | [03_eval_perplexity_checkpoints](03_eval_perplexity_checkpoints/) | validation loss, perplexity, overfitting, save/reload | `python main.py` | 60 min |
| 4 | [04_lora_math_numpy](04_lora_math_numpy/) | low-rank update, B=0 init, merge, parameter counts | `python main.py` | 45 min |
| 5 | [05_lora_peft_finetune](05_lora_peft_finetune/) | real LoRA with peft, adapters, disable/merge | `python main.py` | 75 min |
| 6 | [06_qlora_reference](06_qlora_reference/) | 4-bit NF4 base + LoRA (read only) | read | 30 min |
| 7 | [07_hf_inference_batching_kvcache](07_hf_inference_batching_kvcache/) | no_grad, KV cache timing, left padding, batching, dtype | `python main.py` | 60 min |
| 8 | [08_gpu_memory_calculator](08_gpu_memory_calculator/) | weights + KV cache + training memory arithmetic | `python main.py` | 45 min |
| 9 | [09_throughput_benchmark](09_throughput_benchmark/) | measure latency vs throughput | `python main.py` | 45 min |
| 10 | [10_vllm_reference](10_vllm_reference/) | PagedAttention, continuous batching (read only) | read | 30 min |

Total ~9-10 hours including exercises. Each step folder has `main.py` (commented code) and `WALKTHROUGH.md` (syntax, libraries, real output, experiments, debugging drill). Then do [exercises.md](exercises.md).
Run from the repo root with `.venv/bin/python`, or `cd` into the step and use `python main.py`. Steps 02, 03, 05 write checkpoints/adapters to their own folder (03's is ~330 MB; delete freely).

## 5. The concepts, one by one

### Pretraining vs fine-tuning vs inference
| | Pretraining | Fine-tuning | Inference |
|---|---|---|---|
| Goal | learn language from huge raw text | adapt to a task/style with small data | produce output from a trained model |
| Data | trillions of tokens | hundreds to millions of examples | a prompt |
| Weights | all learned from random init | start from pretrained, update all (full) or few (LoRA) | frozen |
| Compute | months on thousands of GPUs | hours on 1-8 GPUs (or 1 with LoRA) | milliseconds-seconds per request |
| Memory | weights + grads + optimizer | same, or tiny with LoRA | weights + KV cache |
| In this module | (conceptual) | steps 01-06 | steps 07-10 |
Trade-off: fine-tuning changes behaviour but not knowledge reliably; for facts use prompting/retrieval (modules 07-08).

### Dataset preparation and preprocessing (step 01)
Tokenise, join documents with EOS, cut to fixed blocks, wrap in a `DataLoader`. Labels for a causal LM are the inputs themselves; the model shifts them by one inside. Padding needs an `attention_mask`, and padded label positions are set to `-100` so the loss ignores them. Limitation: fixed blocks may split a document, and bad/duplicated data is the biggest quality risk — the model learns exactly what you feed it.

### Training loop, loss, optimizer (step 02)
`forward` (get loss) -> `zero_grad` -> `backward` (gradients) -> `clip_grad_norm_` -> `optimizer.step()`. Loss is cross-entropy per token; AdamW keeps running averages of gradients to scale each update. Clipping caps the update size when a gradient spikes. Limitation: needs memory for gradients and optimizer states (about 8x the fp16 weights in total, see step 08).

### Evaluation and checkpoints (step 03)
Training loss always falls if you train long enough; **validation loss** tells you if the model generalises. Perplexity = exp(loss). `save_pretrained` stores config + weights, plus the tokenizer separately. Limitation: tiny validation sets are noisy; perplexity is not a measure of usefulness for a task.

### LoRA and QLoRA (steps 04-06)
Freeze `W`; learn `B (d_out x r)` and `A (r x d_in)`; output `Wx + (alpha/r)·B·A·x`. `B` starts at 0, so training starts at the pretrained model. A 2048x2048 layer: 4.2M vs 32,768 parameters at r=8. Adapters are tiny files, swappable per task, and can be merged. QLoRA stores the frozen base in 4-bit NF4 so a 7B model trains in ~4 GB of weights. Limitation: low rank limits what can be learned; QLoRA adds quantisation error and needs CUDA.

### Fast inference (steps 07-10)
- `model.eval()` + `torch.no_grad()`: no gradient memory.
- **KV cache**: store past keys/values so each step is O(1) new work instead of recomputing the sequence (4.8 s -> 1.5 s measured). Costs memory that grows with length x batch.
- **Batching**: more tokens per forward pass; throughput up, per-user speed down.
- **Left padding** for decoder-only models.
- **Lower precision** (bf16, int8, int4): smaller and often faster, slight quality loss.
- **vLLM**: PagedAttention + continuous batching for production serving.

### Latency, throughput and GPU memory
- Latency: time for one request. Throughput: tokens/s for the whole system. Memory: weights + KV cache + overhead.
- Rules of thumb (step 08): fp16 = 2 bytes/param (7B ~ 14 GB); int4 ~ 0.5; AdamW training ~ 16 bytes/param; KV cache = 2·layers·kv_heads·head_dim·tokens·batch·bytes.
- Levers: smaller/quantised model, shorter context, GQA, bigger batch (if latency allows), paged KV cache.

## 6. Common mistakes and gotchas
- Forgetting `optimizer.zero_grad()` (silent bug: no error, bad training).
- Forgetting `model.eval()` / `no_grad()` when evaluating or serving.
- Right padding for generation with decoder-only models.
- Not saving the tokenizer with the checkpoint.
- Wrong `target_modules` names for the architecture (`c_attn` for GPT-2, `q_proj`... for Llama/Qwen).
- Using a full fine-tuning learning rate (3e-6) for LoRA (needs ~1e-4 to 1e-3).
- Benchmarking without warm-up, with one run, or on a GPU without `synchronize()`.
- Estimating memory from weights only and forgetting the KV cache.

## 7. Self-check
- [ ] I can turn text into token blocks with labels and explain the shift by one and `-100`.
- [ ] I can write the training loop from memory and say what each of the 4 beats does.
- [ ] I can explain loss, optimizer, learning rate, batch size and gradient clipping.
- [ ] I can explain validation loss, perplexity and overfitting, and save/reload a model + tokenizer.
- [ ] I can explain LoRA with the `W + BA` picture and compute the parameter saving.
- [ ] I can explain QLoRA (NF4 base + 16-bit adapters) and why it fits on a small GPU.
- [ ] I can explain the KV cache, batching, padding side and dtype, with measured numbers.
- [ ] I can estimate GPU memory for a 7B model for serving and for training.
- [ ] I can explain PagedAttention and continuous batching and when to use vLLM.
- [ ] I can compare pretraining, fine-tuning and inference in a table.

## 8. Interview questions
1. **What is the difference between pretraining, fine-tuning and inference?** Pretraining learns general language from huge raw text; fine-tuning continues training on a small task dataset; inference only runs the frozen model. Fine-tuning shapes behaviour/style more than it injects reliable facts.
2. **How do you prepare text for causal LM training?** Tokenise, add EOS between documents, concatenate and cut into fixed blocks (or pad with an attention mask), labels = inputs (shifted inside the model), mask padding labels with -100, batch with a DataLoader. Risk: low-quality or duplicated data.
3. **Walk me through a training step.** Forward pass -> cross-entropy loss -> `zero_grad` -> `backward` -> clip gradients -> optimizer step. Mention `train()` mode and moving the batch to the device.
4. **What does gradient clipping do and why?** Rescales gradients whose norm exceeds a threshold so one noisy batch can't make a huge destructive update. Trade-off: a too-low threshold slows learning.
5. **What is perplexity?** exp(average cross-entropy): how many equally likely tokens the model is choosing among. Lower is better, but it's not the same as task quality; compare only on the same data/tokenizer.
6. **How do you detect overfitting?** Training loss keeps falling while validation loss flattens or rises. Fix: early stopping, more data, regularisation, fewer epochs.
7. **Explain LoRA.** Freeze W, add trainable low-rank `B·A` scaled by alpha/r; B initialised to zero. ~0.2% of the parameters; adapter files are MBs; merge for zero inference overhead. Limitation: rank bounds capacity.
8. **What do `r`, `lora_alpha`, `target_modules` mean?** Rank/capacity; scaling (update × alpha/r); which layers get adapters (names depend on architecture).
9. **What is QLoRA?** LoRA trained over a frozen 4-bit (NF4, double-quantised) base with paged optimizers; 7B fits in ~4 GB of weights. Cost: some quantisation error, slower, CUDA only.
10. **What is the KV cache?** Stored keys/values of past tokens so each new token only computes its own; without it every step reprocesses everything (measured 4.8 s vs 1.5 s). Cost: memory proportional to layers x kv_heads x head_dim x tokens x batch.
11. **Why left-pad for generation?** The model continues from the last position; right padding makes it continue from pad tokens.
12. **Latency vs throughput?** Latency = one request's time; throughput = tokens/s overall. Batching raises throughput and per-request latency slightly (measured x9.7 throughput for x1.6 latency at batch 16). Trade-off depends on the product.
13. **Will a 7B model fit on a 16 GB GPU?** fp16 weights ~14 GB leave little room for KV cache; int8 ~7 GB or int4 ~3.5 GB fit comfortably; training requires LoRA/QLoRA.
14. **Why is vLLM faster than `generate`?** PagedAttention (no cache fragmentation, more concurrent requests) and continuous batching (no waiting for the slowest request). Needs CUDA and a server.
15. **When would you not fine-tune?** When prompting or retrieval is enough, data is tiny/low quality, or facts change often.
