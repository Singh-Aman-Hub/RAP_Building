# Terms — LoRA Fine-tuning with peft

Folder: `06_llm_training_and_fast_inference/05_lora_peft_finetune` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [06 README](../README.md)

**Terms in this folder:** [peft (Parameter-Efficient Fine-Tuning)](#peft-parameter-efficient-fine-tuning) · [LoraConfig](#loraconfig) · [r (rank)](#r-rank) · [lora_alpha](#lora_alpha) · [lora_dropout](#lora_dropout) · [target_modules](#target_modules) · [fan_in_fan_out](#fan_in_fan_out) · [task_type="CAUSAL_LM"](#task_typecausal_lm) · [get_peft_model](#get_peft_model) · [print_trainable_parameters](#print_trainable_parameters) · [requires_grad and trainable vs frozen parameters](#requires_grad-and-trainable-vs-frozen-parameters) · [named_parameters and numel](#named_parameters-and-numel) · [Optimizer over trainable parameters only](#optimizer-over-trainable-parameters-only) · [Higher learning rate for LoRA](#higher-learning-rate-for-lora) · [disable_adapter](#disable_adapter) · [Adapter (saved separately)](#adapter-saved-separately) · [PeftModel.from_pretrained](#peftmodelfrom_pretrained) · [merge_and_unload](#merge_and_unload) · [type(x).__name__](#typexname) · [Generator expression in sum()](#generator-expression-in-sum) · [ValueError: target modules not found](#valueerror-target-modules-not-found)

---

## peft (Parameter-Efficient Fine-Tuning)  `(third-party: peft)`
**1. Definition:** `peft` is a Hugging Face library implementing methods such as LoRA that fine-tune a model by training only a small number of extra or selected parameters.
**2. In plain words:** Instead of updating all 82 million weights, you add a few thousand new trainable numbers and leave the original alone. That makes fine-tuning cheaper and gives you small files to share.
**3. Example to quote:** > "I used `peft` so only 147,456 of 82 million parameters (0.18%) were trainable."
**Seen in:** `main.py`, imports

## LoraConfig  `(third-party: peft)`
**1. Definition:** `LoraConfig(...)` is a settings object that declares how the LoRA adapters are built: rank, scaling, dropout, which layers, and the task type.
**2. In plain words:** It's a recipe card; nothing changes in the model until you hand the recipe to `get_peft_model`.
**3. Example to quote:** > "My config is `LoraConfig(r=8, lora_alpha=16, target_modules=['c_attn'], task_type='CAUSAL_LM')`."
**Seen in:** `main.py`, section 1

## r (rank)  `(concept)`
**1. Definition:** `r` is the rank of the LoRA matrices: the width of the bottleneck between A and B.
**2. In plain words:** Higher r means more capacity and more trainable parameters. In the sweep, r=1 gave 18,432 parameters and loss 2.902, r=8 gave 147,456 and 2.145, r=32 gave 589,824 and 1.734. Lower training loss isn't automatically better generalisation.
**3. Example to quote:** > "Going from r=8 to r=32 quadrupled the trainable parameters and lowered the training loss from 2.145 to 1.734."
**Seen in:** `main.py` (`r=8`), `WALKTHROUGH.md`

## lora_alpha  `(concept)`
**1. Definition:** `lora_alpha` is the scaling numerator: the adapter's output is multiplied by `lora_alpha / r`.
**2. In plain words:** It sets how loud the adapter is. With alpha 16 and r 8 the multiplier is 2.0.
**3. Example to quote:** > "`lora_alpha=16` with `r=8` scales the update by 2."
**Seen in:** `main.py`

## lora_dropout  `(concept)`
**1. Definition:** `lora_dropout` is the probability of randomly zeroing activations on the adapter path during training.
**2. In plain words:** It's a mild regulariser that discourages the tiny adapter from memorising the data. 0.05 means 5%.
**3. Example to quote:** > "I used `lora_dropout=0.05` for light regularisation."
**Seen in:** `main.py`

## target_modules  `(concept)`
**1. Definition:** `target_modules` lists the names of the layers in the base model that receive LoRA adapters.
**2. In plain words:** Layer names differ per architecture: GPT-2 uses `c_attn` (fused query/key/value projection), while Llama and Qwen use `q_proj`, `k_proj`, `v_proj`, `o_proj`. Getting this wrong is the most common peft error. Listing more modules increases trainable parameters.
**3. Example to quote:** > "For distilgpt2 I target `c_attn`; `c_attn` maps 768 to 2304 because Q, K and V are fused."
**Seen in:** `main.py`

## fan_in_fan_out  `(concept)`
**1. Definition:** `fan_in_fan_out=True` tells peft that the target layer stores its weight transposed, as GPT-2's `Conv1D` layers do.
**2. In plain words:** GPT-2's layers look like linear layers but store the matrix the other way round. Without this flag, peft prints a warning, and the maths may be applied incorrectly.
**3. Example to quote:** > "I set `fan_in_fan_out=True` because GPT-2 uses Conv1D layers; removing it makes peft warn about Conv1D."
**Seen in:** `main.py`

## task_type="CAUSAL_LM"  `(concept)`
**1. Definition:** `task_type` tells peft which kind of model head to expect; `"CAUSAL_LM"` means next-token prediction.
**2. In plain words:** It lets peft wrap the model correctly for text generation.
**3. Example to quote:** > "I pass `task_type='CAUSAL_LM'` because distilgpt2 is a causal language model."
**Seen in:** `main.py`

## get_peft_model  `(third-party: peft)`
**1. Definition:** `get_peft_model(model, config)` returns a `PeftModel` that freezes the base weights and inserts trainable LoRA matrices as described by the config.
**2. In plain words:** This single call turns a normal model into a LoRA-ready one.
**3. Example to quote:** > "`model = get_peft_model(base, config)` freezes distilgpt2 and adds adapters in each of its 6 blocks."
**Seen in:** `main.py`

## print_trainable_parameters  `(third-party: peft)`
**1. Definition:** `model.print_trainable_parameters()` prints the trainable parameter count, the total, and the percentage.
**2. In plain words:** It's the headline number to quote: `trainable params: 147,456 || all params: 82,060,032 || trainable%: 0.1797`. Check: 6 blocks × (768·8 + 8·2304) = 147,456.
**3. Example to quote:** > "The printout shows 147,456 trainable parameters out of 82,060,032, which is about 0.18%."
**Seen in:** `main.py`

## requires_grad and trainable vs frozen parameters  `(third-party: torch)`
**1. Definition:** Each parameter tensor has `requires_grad`; if True, autograd computes its gradient and the optimizer can update it, if False it's frozen.
**2. In plain words:** peft sets it False for the original weights and True for the LoRA matrices. Here 81,912,576 parameters are frozen.
**3. Example to quote:** > "I printed a `lora` parameter and saw `requires_grad = True`, while the 81.9 million base parameters are frozen."
**Seen in:** `main.py`

## named_parameters and numel  `(third-party: torch)`
**1. Definition:** `model.named_parameters()` yields (name, tensor) pairs for all parameters, and `p.numel()` returns the number of elements in a tensor.
**2. In plain words:** Together they let you count and inspect parameters, e.g. `sum(p.numel() for p in model.parameters() if not p.requires_grad)`.
**3. Example to quote:** > "I summed `numel()` over frozen parameters to show 81,912,576 were untouched."
**Seen in:** `main.py`

## Optimizer over trainable parameters only  `(concept)`
**1. Definition:** Passing `[p for p in model.parameters() if p.requires_grad]` to the optimizer gives it only the adapter weights.
**2. In plain words:** The optimizer then stores state only for the few trainable numbers, which is where LoRA saves memory.
**3. Example to quote:** > "My AdamW only sees tensors with `requires_grad`, so no optimizer state is kept for the frozen 82 million."
**Seen in:** `main.py`, section 2

## Higher learning rate for LoRA  `(concept)`
**1. Definition:** LoRA training typically uses a learning rate 10–100× larger (here 1e-3) than full fine-tuning (5e-5).
**2. In plain words:** The adapters start at zero and are tiny, so small steps barely change anything. With 3e-6 LoRA hardly learns.
**3. Example to quote:** > "I used 1e-3 for LoRA; at a full-fine-tuning rate like 3e-6 the loss barely moved."
**Seen in:** `main.py`, constants

## disable_adapter  `(third-party: peft)`
**1. Definition:** `with model.disable_adapter():` is a context manager that temporarily turns the LoRA layers off so the model behaves like the original.
**2. In plain words:** It lets you do a fair before/after comparison on one model object. The base output repeated itself; with LoRA it answered "Open Settings, tap Settings, then choose Dark Mode."
**3. Example to quote:** > "Inside `disable_adapter()` the model gives the generic repeated text; outside it gives the learned support style."
**Seen in:** `main.py`

## Adapter (saved separately)  `(concept)`
**1. Definition:** The adapter is the small set of trained LoRA weights plus config (`adapter_config.json`, `adapter_model.safetensors`), saved without the base model.
**2. In plain words:** It's 0.60 MB instead of about 330 MB, so you can keep one base model and swap adapters per task.
**3. Example to quote:** > "My adapter folder is 0.60 MB versus the 330 MB full model."
**Seen in:** `main.py`, section 3

## PeftModel.from_pretrained  `(third-party: peft)`
**1. Definition:** `PeftModel.from_pretrained(base_model, adapter_dir)` attaches a saved adapter to a freshly loaded base model.
**2. In plain words:** It's how you reload an adapter later. The check that the reloaded output equals the trained output printed `True`.
**3. Example to quote:** > "I loaded a fresh distilgpt2 and attached my adapter, and the generation was identical to before saving."
**Seen in:** `main.py`, section 4

## merge_and_unload  `(third-party: peft)`
**1. Definition:** `merge_and_unload()` folds the LoRA update (alpha/r)·B·A into the base weights and returns a plain Transformers model without adapter layers.
**2. In plain words:** After merging, deployment needs no peft and has no extra latency. The type printed is `GPT2LMHeadModel` again.
**3. Example to quote:** > "I merge for deployment, and the merged model produced identical output with no peft dependency."
**Seen in:** `main.py`, section 5

## type(x).__name__  `(Python built-in)`
**1. Definition:** `type(x).__name__` returns the name of an object's class as a string.
**2. In plain words:** A quick way to see what kind of object you have after a transformation.
**3. Example to quote:** > "`type(merged).__name__` printed `GPT2LMHeadModel`, confirming it's a plain model."
```python
type(3).__name__   # 'int'
```
**Seen in:** `main.py`

## Generator expression in sum()  `(syntax)`
**1. Definition:** A generator expression `(expr for x in xs if cond)` produces values lazily, so `sum(...)` can add them without building a list.
**2. In plain words:** It's a memory-friendly way to count or total things in one line.
**3. Example to quote:** > "`sum(p.numel() for p in model.parameters() if not p.requires_grad)` counted frozen parameters."
**Seen in:** `main.py`

## ValueError: target modules not found  `(concept)`
**1. Definition:** peft raises `ValueError: Target modules {...} not found in the base model` when `target_modules` names don't exist in the architecture.
**2. In plain words:** GPT-2 has no `q_proj`. Print `[n for n, _ in base.named_modules()]` to see the real names.
**3. Example to quote:** > "I listed the model's module names and changed `q_proj` to `c_attn`."
**Seen in:** `WALKTHROUGH.md`, Debugging practice
