# Terms — The PEFT family

Folder: `17_finetuning_alignment_and_benchmarks/03_peft_family` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [17 README](../README.md)

**Terms in this folder:** [PEFT (parameter-efficient fine-tuning)](#peft-parameter-efficient-fine-tuning) · [LoRA](#lora) · [rsLoRA](#rslora) · [DoRA](#dora) · [QLoRA](#qlora) · [IA3](#ia3) · [Prompt tuning and prefix tuning](#prompt-tuning-and-prefix-tuning) · [Target modules](#target-modules) · [fan_in_fan_out](#fan_in_fan_out) · [Rank r and alpha](#rank-r-and-alpha) · [Memory arithmetic (16 bytes per parameter)](#memory-arithmetic-16-bytes-per-parameter) · [Trainable parameter count](#trainable-parameter-count) · [requires_grad and frozen weights](#requires_grad-and-frozen-weights) · [NoMatchingPeftModuleError](#nomatchingpeftmoduleerror)

---

## PEFT (parameter-efficient fine-tuning)  `(concept)`
**1. Definition:** Methods that train a small number of added or selected parameters while the base stays frozen.
**2. In plain words:** Adjust a few dials, not the whole machine.
**3. Example to quote:** > "LoRA r=8 on distilgpt2's c_attn trains 147,456 of 82,060,032 parameters (0.180%)."
```python
pm = get_peft_model(base, cfg)
```
**Seen in:** `main.py`, section 3

## LoRA  `(concept)`
**1. Definition:** Adds a low-rank update (alpha/r) * B @ A to chosen weights; only A (r x d_in) and B (d_out x r) train.
**2. In plain words:** A small side-path.
**3. Example to quote:** > "For c_attn: A is (8, 768) and B is (2304, 8)."
```python
LoraConfig(r=8, lora_alpha=16, target_modules=['c_attn'])
```
**Seen in:** `main.py`, section 2

## rsLoRA  `(concept)`
**1. Definition:** LoRA with scaling alpha/sqrt(r) instead of alpha/r.
**2. In plain words:** A scale fix for big ranks.
**3. Example to quote:** > "At r=64, alpha=128 the scale is 2.0 for LoRA and 16.0 for rsLoRA."
```python
LoraConfig(..., use_rslora=True)
```
**Seen in:** `main.py`, section 4

## DoRA  `(concept)`
**1. Definition:** Weight-decomposed LoRA: LoRA on the direction plus a trainable magnitude vector.
**2. In plain words:** LoRA with an extra volume knob per output.
**3. Example to quote:** > "Adds 2,304 values per layer: 161,280 total vs 147,456."
```python
LoraConfig(..., use_dora=True)
```
**Seen in:** `main.py`, section 2

## QLoRA  `(concept)`
**1. Definition:** LoRA adapters on a 4-bit (NF4) quantised frozen base; needs bitsandbytes and a GPU.
**2. In plain words:** Shrink the base, train the adapter in higher precision.
**3. Example to quote:** > "7B: about 5.0 GB vs 15.1 GB for LoRA at 1% trainable in the calculator."
**Seen in:** `main.py`, section 6; module 06 step 06

## IA3  `(concept)`
**1. Definition:** Learned per-channel scaling vectors on keys, values and feed-forward activations.
**2. In plain words:** Volume knobs on internal signals.
**3. Example to quote:** > "18,432 parameters (0.022%)."
```python
IA3Config(target_modules=['c_attn','c_fc'], feedforward_modules=['c_fc'])
```
**Seen in:** `main.py`, section 2

## Prompt tuning and prefix tuning  `(concept)`
**1. Definition:** Learn virtual-token embeddings at the input (prompt) or key/value prefixes in every layer (prefix).
**2. In plain words:** A secret opening phrase made of numbers.
**3. Example to quote:** > "20 virtual tokens: 15,360 (prompt) and 184,320 (prefix) parameters."
```python
PromptTuningConfig(num_virtual_tokens=20, task_type='CAUSAL_LM')
```
**Seen in:** `main.py`, section 2

## Target modules  `(concept)`
**1. Definition:** The layers that receive adapters: names in the model such as c_attn (GPT-2) or q_proj, k_proj, v_proj, o_proj (Llama/Qwen style); 'all-linear' targets every linear layer.
**2. In plain words:** Where to attach the side-paths.
**3. Example to quote:** > "Adding c_proj and c_fc to c_attn raised trainable parameters from 147,456 to 589,824."
```python
target_modules=['c_attn','c_proj','c_fc']
```
**Seen in:** `main.py`, section 2

## fan_in_fan_out  `(syntax)`
**1. Definition:** LoraConfig flag for layers that store weights as (in, out), such as GPT-2's Conv1D.
**2. In plain words:** Tells peft the weight is transposed.
**3. Example to quote:** > "Required for GPT-2 targets."
```python
fan_in_fan_out=True
```
**Seen in:** `main.py`, section 2

## Rank r and alpha  `(concept)`
**1. Definition:** r is the inner dimension of A and B; alpha scales the update by alpha/r.
**2. In plain words:** Capacity and strength.
**3. Example to quote:** > "Parameters grow linearly with r: 147,456 at r=8, 1,179,648 at r=64."
**Seen in:** `main.py`, section 3

## Memory arithmetic (16 bytes per parameter)  `(concept)`
**1. Definition:** Mixed-precision Adam training stores bf16 weights (2), bf16 gradients (2), fp32 master weights (4) and two fp32 moments (8): 16 bytes per parameter before activations.
**2. In plain words:** The bill for training every weight.
**3. Example to quote:** > "7B full fine-tune: 112.0 GB; LoRA 15.1 GB; QLoRA 5.0 GB."
```python
mem_full = lambda n: 16 * n / 1e9
```
**Seen in:** `main.py`, section 6

## Trainable parameter count  `(concept)`
**1. Definition:** Sum of numel() over tensors with requires_grad=True.
**2. In plain words:** What can learn.
**3. Example to quote:** > "LoRA r=8 on c_attn trains 147,456 parameters, about 0.18% of distilgpt2."
```python
sum(p.numel() for p in m.parameters() if p.requires_grad)
```
**Seen in:** `main.py`, `count`

## requires_grad and frozen weights  `(concept)`
**1. Definition:** Frozen tensors get no gradient and no optimizer state.
**2. In plain words:** Locked dials.
**3. Example to quote:** > "After one backward pass only 12 tensors had gradients (A and B in 6 layers)."
```python
p.grad is not None
```
**Seen in:** `main.py`, section 7

## NoMatchingPeftModuleError  `(error)`
**1. Definition:** peft could not find any module matching target_modules.
**2. In plain words:** A typo in a layer name.
**3. Example to quote:** > "Target modules {'c_atn'} not found in the base model."
```python
peft.utils.error.NoMatchingPeftModuleError
```
**Seen in:** `WALKTHROUGH.md`, debugging practice
