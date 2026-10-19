# Step 05 — LoRA fine-tuning with `peft`

## Goal
Fine-tune distilgpt2 by training only ~0.18% of the parameters, save just the adapter, reload it and merge it.

## Run it
```bash
cd 06_llm_training_and_fast_inference/05_lora_peft_finetune
python main.py       # ~1 min; writes ./adapter_out (0.6 MB)
```

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `LoraConfig(r=8, ...)` | a settings object (a class with keyword arguments) | `cfg.r` |
| `[p for p in model.parameters() if p.requires_grad]` | filter: keep only trainable tensors | |
| `with model.disable_adapter():` | temporarily turn LoRA off inside the block | |
| `type(x).__name__` | class name as a string | `type(3).__name__` -> `'int'` |
| `sum(p.numel() for p in ...)` | generator expression: add up counts without building a list | |
| `next(m.parameters()).device` | where the model lives | |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `LoraConfig` | peft | declares the adapter | `r`, `lora_alpha`, `lora_dropout`, `target_modules`, `fan_in_fan_out`, `task_type` | config |
| `get_peft_model(model, config)` | peft | freezes base weights, inserts trainable A/B | model, config | `PeftModel` |
| `model.print_trainable_parameters()` | peft | prints trainable vs total | – | None |
| `model.save_pretrained(dir)` | peft | saves ONLY adapter weights + config | dir | None |
| `PeftModel.from_pretrained(base, dir)` | peft | attaches a saved adapter to a fresh base | base model, dir | `PeftModel` |
| `merge_and_unload()` | peft | folds `(alpha/r)·B·A` into W, removes adapter layers | – | plain transformers model |
| `disable_adapter()` | peft | context manager for A/B comparisons | – | context |

Parameter meaning: **r** = rank (capacity); **lora_alpha** = scaling numerator (update scaled by `alpha / r`, here 2.0); **target_modules** = which layers to adapt (GPT-2: `c_attn`; Llama/Qwen: `q_proj`, `k_proj`, `v_proj`, `o_proj`, ...).

## Execution trace (real run)
- `trainable params: 147,456 || all params: 82,060,032 || trainable%: 0.1797`. Check: 6 blocks x (`768*8 + 8*2304`) = 6 x 24,576 = 147,456. (`c_attn` maps 768 -> 2304 because Q, K and V are fused.)
- Frozen parameters: 81,912,576 (the original distilgpt2).
- Training 25 epochs at lr 1e-3: loss 4.574 (epoch 1) -> 2.158 (10) -> 1.606 (20) -> **1.404** (25).
- Prompt `Ticket: How do I turn on dark mode? Reply:`
  - base (adapter disabled): `I turn on dark mode? Reply: I turn on dark mode? ...` (repeats)
  - with LoRA: `Open Settings, tap Settings, then choose Dark Mode.` (learned the style, got the wording a bit muddled — expected from a tiny corpus)
- Adapter on disk: **0.60 MB** vs ~330 MB for the full model.
- Reloaded adapter output identical: `True`. After `merge_and_unload()` identical: `True`, and the type is `GPT2LMHeadModel` again.

## Experiments (change ONE thing)
Measured with `solutions/lora_rank_sweep.py` (10 epochs):
| r | trainable params | final loss |
|---|---|---|
| 1 | 18,432 | 2.902 |
| 8 | 147,456 | 2.145 |
| 32 | 589,824 | 1.734 |
1. Higher rank = more capacity = lower training loss, linear growth in parameters. Lower training loss is not automatically better generalisation.
2. `LR = 3e-6` (a full-fine-tune rate): LoRA barely moves because only tiny, zero-initialised matrices are trained; LoRA usually wants 10-100x larger rates.
3. Adding `c_proj`, `c_fc` to `target_modules` increases trainable parameters.

## Debugging practice
`target_modules=["q_proj"]`:
```
ValueError: Target modules {'q_proj'} not found in the base model ...
```
Fix by listing real layer names: `print([n for n, _ in base.named_modules()][:20])`. Layer names differ per architecture — the single most common peft error.

## Rebuild checklist
- [ ] `LoraConfig(r, lora_alpha, target_modules, task_type)`
- [ ] `get_peft_model(base, config)` + `print_trainable_parameters()`
- [ ] optimizer over trainable params only; same loop as step 02
- [ ] `save_pretrained`, reload with `PeftModel.from_pretrained`, `merge_and_unload()`

## Say it aloud
"LoRA freezes the pretrained weights and adds small low-rank matrices to chosen layers, here GPT-2's attention projection, so I train 147 thousand parameters instead of 82 million. The adapter is 0.6 megabytes, I can keep one base model and swap adapters per task, and I can merge it into the weights for deployment so inference has no overhead. Trade-off: rank and target modules limit how much it can learn, and for large behaviour changes full fine-tuning can still do better."
