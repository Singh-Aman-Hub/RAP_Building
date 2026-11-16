# Module 06 — LLM Training and LoRA

Syllabus topic 1 (second half): dataset preparation, the training loop, evaluation and checkpoints, and parameter-efficient fine-tuning (LoRA / QLoRA).

> **Restructured (Nov 2026):** the fast-inference steps that used to be 07–11 (HF inference, GPU memory calculator, throughput benchmark, vLLM, PagedAttention) moved with their git history into [module 21 — LLM inference engineering](../21_llm_inference_engineering/README.md), which goes much deeper on loading, quantization, batching, serving engines and benchmarking. Post-training beyond LoRA (SFT masking, DPO, GRPO, evaluation) continues in [module 17](../17_finetuning_alignment_and_benchmarks/README.md).

## The story of this module
Chapters 15 (second half) to 17 of the master story. [Module 05](../05_llm_parameters/07_experiment_logger/WALKTHROUGH.md) gave you dials and a quantized model that fits.
- **Pain point:** a full fine-tune needs about 16 bytes per parameter (107 GB for a 7B model; the arithmetic is in [21/03 memory calculator](../21_llm_inference_engineering/03_gpu_memory_calculator/)) and saves a full copy of the model per task.
- **Spark:** freeze the model and learn a low-rank update, `W = W0 + B @ A`, about 1% of the parameters or less (LoRA), and keep the frozen base in 4-bit (QLoRA).
- **What you build:** dataset prep, a training loop, evaluation and checkpoints, LoRA in NumPy and in peft, and a QLoRA reference.
- **Cliffhanger:** the model is tuned. Aligning it with preferences and measuring it properly is [module 17](../17_finetuning_alignment_and_benchmarks/); serving it fast is [module 21](../21_llm_inference_engineering/).

Reading order: 01 -> 06. Recommended course order (see [ROADMAP](../ROADMAP.md)): 05 -> 16 -> 21 -> **06** -> 17 -> 07.

## 1. Why this matters
Interviewers ask "how would you fine-tune a model on one GPU?". You need to *show* a training loop, explain LoRA with numbers, and reason about memory. Everything here runs on a laptop with tiny models; the reasoning is identical for 7B models.

## 2. Where this fits
**Builds on**
- [../03_ml_foundations/](../03_ml_foundations/): the forward -> loss -> backward -> step loop (`04_pytorch_training_loop`), train/validation split and overfitting (`05_overfitting_train_val`), cross-entropy (`06_softmax_cross_entropy`). Step 02 here is the same loop on a language model.
- [../04_llm_fundamentals/](../04_llm_fundamentals/): the tokenizer (`01_tokenizer`), loading a pretrained model (`06_load_pretrained_model`), logits and next-token prediction (`07_forward_pass_logits`) and the generation loop (`08_generation_loop`). The loss in step 02 is cross-entropy on exactly those logits; `09_pretrain_vs_finetune_vs_inference` is the overview that this module makes concrete.
- [../05_llm_parameters/](../05_llm_parameters/): learning rate, batch size, epochs (`06_training_parameters`), generation parameters (`02_sampling_from_scratch`, `03_hf_generate_compare`) and quantization (`05_quantization`). Here they become measured effects: the learning-rate table in step 02, bytes-per-parameter in [21/03](../21_llm_inference_engineering/03_gpu_memory_calculator/).

**Leads to**
- [../17_finetuning_alignment_and_benchmarks/](../17_finetuning_alignment_and_benchmarks/): SFT with loss masking, the PEFT family, reward models, DPO, GRPO, evaluation and benchmarks.
- [../21_llm_inference_engineering/](../21_llm_inference_engineering/): serving the tuned model (memory, quantization, KV cache, batching, engines, benchmarks).
- [../07_prompt_engineering/](../07_prompt_engineering/) and [../08_agent_harness/](../08_agent_harness/): a fine-tuned or served model is what those layers call; vLLM exposes an OpenAI-style endpoint your harness can use.
- [../09_api_development/](../09_api_development/): wrap generation code ([21/08](../21_llm_inference_engineering/08_hf_inference_batching_kvcache/)) in a FastAPI endpoint (model loaded once at startup, `no_grad`, left padding).
- [../10_ml_system_architecture/](../10_ml_system_architecture/): checkpoints (step 03), adapters as versioned artifacts (step 05), latency/throughput targets ([21/10](../21_llm_inference_engineering/10_latency_throughput_benchmark/)).
- [../11_docker/](../11_docker/): why model files are mounted as volumes, why image size depends on weights.
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
                 serve it: module 21 (memory, quantization, KV cache,
                 batching, PagedAttention, engines, benchmarks)
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

Total ~6-7 hours including exercises. Each step folder has `main.py` (commented code) and `WALKTHROUGH.md` (syntax, libraries, real output, experiments, debugging drill). Then do [exercises.md](exercises.md).
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
| Where | (conceptual) | this module, steps 01-06 | [module 21](../21_llm_inference_engineering/) |
Trade-off: fine-tuning changes behaviour but not knowledge reliably; for facts use prompting/retrieval (modules 07-08).

### Dataset preparation and preprocessing (step 01)
Tokenise, join documents with EOS, cut to fixed blocks, wrap in a `DataLoader`. Labels for a causal LM are the inputs themselves; the model shifts them by one inside. Padding needs an `attention_mask`, and padded label positions are set to `-100` so the loss ignores them. Limitation: fixed blocks may split a document, and bad/duplicated data is the biggest quality risk — the model learns exactly what you feed it.

### Training loop, loss, optimizer (step 02)
`forward` (get loss) -> `zero_grad` -> `backward` (gradients) -> `clip_grad_norm_` -> `optimizer.step()`. Loss is cross-entropy per token; AdamW keeps running averages of gradients to scale each update. Clipping caps the update size when a gradient spikes. Limitation: needs memory for gradients and optimizer states (about 8x the fp16 weights in total, see [21/03](../21_llm_inference_engineering/03_gpu_memory_calculator/)).

### Evaluation and checkpoints (step 03)
Training loss always falls if you train long enough; **validation loss** tells you if the model generalises. Perplexity = exp(loss). `save_pretrained` stores config + weights, plus the tokenizer separately. Limitation: tiny validation sets are noisy; perplexity is not a measure of usefulness for a task.

### LoRA and QLoRA (steps 04-06)
Freeze `W`; learn `B (d_out x r)` and `A (r x d_in)`; output `Wx + (alpha/r)·B·A·x`. `B` starts at 0, so training starts at the pretrained model. A 2048x2048 layer: 4.2M vs 32,768 parameters at r=8. Adapters are tiny files, swappable per task, and can be merged. QLoRA stores the frozen base in 4-bit NF4 so a 7B model trains in ~4 GB of weights. Limitation: low rank limits what can be learned; QLoRA adds quantisation error and needs CUDA.

### Serving the tuned model
Fast inference (KV cache, batching, padding side, dtype, memory arithmetic, vLLM, PagedAttention) is now [module 21](../21_llm_inference_engineering/README.md). One number to carry from here: training with AdamW costs about 16 bytes per parameter, serving in fp16 costs 2 bytes per parameter plus the KV cache.

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Fine-tuning | SFT (supervised fine-tuning), instruction tuning, domain adaptation | Overlapping | Fine-tuning is any further training of a pretrained model; SFT trains on (prompt, answer) pairs; instruction tuning is SFT on instruction-style data. |
| LoRA | adapter, PEFT, low-rank adaptation | LoRA is one PEFT method | Trains two skinny matrices per layer instead of the whole weight matrix; `peft` is the library. |
| QLoRA | 4-bit LoRA | Yes | LoRA on a frozen 4-bit (NF4) base model. |
| Checkpoint | saved model, snapshot, weights file | Close | Weights (plus config and sometimes optimizer state) saved to disk; an adapter checkpoint holds only the LoRA matrices. |
| Perplexity | exp(loss) | Yes | A transformed validation loss; lower is better. |

## 6. Common mistakes and gotchas
- Forgetting `optimizer.zero_grad()` (silent bug: no error, bad training).
- Forgetting `model.eval()` / `no_grad()` when evaluating.
- Estimating training memory from weights only (gradients + AdamW states are ~7x more).
- Not saving the tokenizer with the checkpoint.
- Wrong `target_modules` names for the architecture (`c_attn` for GPT-2, `q_proj`... for Llama/Qwen).
- Using a full fine-tuning learning rate (3e-6) for LoRA (needs ~1e-4 to 1e-3).

## 7. Self-check
- [ ] I can turn text into token blocks with labels and explain the shift by one and `-100`.
- [ ] I can write the training loop from memory and say what each of the 4 beats does.
- [ ] I can explain loss, optimizer, learning rate, batch size and gradient clipping.
- [ ] I can explain validation loss, perplexity and overfitting, and save/reload a model + tokenizer.
- [ ] I can explain LoRA with the `W + BA` picture and compute the parameter saving.
- [ ] I can explain QLoRA (NF4 base + 16-bit adapters) and why it fits on a small GPU.
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
10. **When would you not fine-tune?** When prompting or retrieval is enough, data is tiny/low quality, or facts change often.
