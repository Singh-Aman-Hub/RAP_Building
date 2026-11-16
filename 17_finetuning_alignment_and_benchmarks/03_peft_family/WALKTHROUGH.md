# Step 03 — The PEFT family (LoRA, rsLoRA, DoRA, IA3, prompt and prefix tuning, QLoRA)

## The story so far
[Step 02](../02_sft_from_scratch_with_loss_masking/WALKTHROUGH.md) fine-tuned *every* weight of distilgpt2 with AdamW. That works for 82 million parameters. The derivation of the most popular shortcut, LoRA, is already done twice in this repo: the maths with the matrices in [06 step 04](../../06_llm_training_and_lora/04_lora_math_numpy/WALKTHROUGH.md) and a real `peft` run in [06 step 05](../../06_llm_training_and_lora/05_lora_peft_finetune/WALKTHROUGH.md). **This step does not re-derive LoRA.** It zooms out: LoRA is one member of a family of parameter-efficient fine-tuning (PEFT) methods, and an interviewer expects you to name the others, say what each one trains, and do the memory arithmetic.

## The pain point
Full fine-tuning keeps weights, gradients, a master copy and two Adam moments for every parameter: about 16 bytes per parameter in mixed precision. A 7B model needs roughly 112 GB before activations, and you would store a full 14 GB copy for every task. Teams that serve many customised variants cannot afford that.

## The spark
Freeze the pretrained weights and train only a tiny add-on. The methods differ in *where* the add-on lives and *what shape* it has: a low-rank update to weight matrices (LoRA and cousins), a learned rescaling of activations (IA3), extra learned inputs (prompt tuning) or extra learned keys and values in every layer (prefix tuning). Quantise the frozen base to 4 bits as well and you get QLoRA.

## Goal
Build eight real PEFT configs on distilgpt2 with the installed `peft` library, print the trainable-parameter count of each (and verify the counts by hand), read the LoRA versus rsLoRA scaling factor off a live layer, list which tensors actually train, do the memory arithmetic for 0.5B / 7B / 70B models, and check that a LoRA backward pass only produces gradients for adapter tensors. QLoRA is covered as arithmetic and reference only: `bitsandbytes` and a GPU are not available here.

## The theory (read this before the interview)
**The common idea:** `h = W0 x + delta(x)` where `W0` is frozen and `delta` is small and trainable.

**LoRA in one paragraph (derivation: [06 step 04](../../06_llm_training_and_lora/04_lora_math_numpy/WALKTHROUGH.md)):** replace the update of a weight `W` by `W0 + (alpha / r) * B @ A` with `A: (r, in)`, `B: (out, r)`, `r` small. Trainable parameters per matrix: `r * (in + out)`. The scaling factor is `alpha / r`.

**The family:**
| Method | What is trained | Where it acts | Inference cost |
|---|---|---|---|
| LoRA | `A`, `B` per targeted weight | adds `(alpha/r) B A x` to a linear layer | none after merging into `W` |
| rsLoRA (rank-stabilised) | same as LoRA | scaling `alpha / sqrt(r)` instead of `alpha / r` | none after merging |
| DoRA (weight-decomposed) | LoRA direction update + a magnitude vector per output | `W` split into magnitude and direction; LoRA updates the direction | small extra compute |
| Adapters (bottleneck, Houlsby-style) | small down-project / up-project MLP inserted after sub-layers | adds new layers in series | extra latency (not merged away); not built in this step |
| IA3 | one learned vector per targeted activation | multiplies keys, values or FFN hidden activations element-wise | none after merging |
| Prompt tuning | `num_virtual_tokens x d_model` embeddings | learned vectors prepended to the input embeddings | longer sequence |
| Prefix tuning | learned key and value vectors per layer | prepended to the attention keys/values of every layer | longer cached context |
| QLoRA | a LoRA adapter on top of a 4-bit (NF4) quantised frozen base | as LoRA; the base is read-only and dequantised on the fly | needs a 4-bit kernel; see below |

**Details worth knowing:**
- **Target modules:** which weight matrices get an adapter. Attention-only (`c_attn` for GPT-2; `q_proj, k_proj, v_proj, o_proj` in Llama-style models) is the cheap default; adding the MLP matrices raises capacity and trainable share (here 0.18% to 0.72%).
- **rank `r` and `alpha`:** trainable parameters grow *linearly* with `r`; the ratio `alpha / r` sets the update's scale. Common practice keeps `alpha` proportional to `r` (e.g. `alpha = 2r`).
- **rsLoRA:** with `alpha / r`, raising `r` shrinks the scale, so high-rank LoRA learns more slowly than it should; dividing by `sqrt(r)` keeps learning stable as `r` grows. With `r=64, alpha=128`: classic scale 2, rsLoRA scale 16 (printed below).
- **DoRA:** decomposes the pretrained weight into a per-output-channel *magnitude* and a *direction*, trains LoRA on the direction and learns the magnitude vector directly. Cost: one extra vector per targeted matrix (`2304` numbers per `c_attn` here).
- **Prompt tuning vs prefix tuning:** both leave every weight untouched. Prompt tuning learns embeddings only at the input; prefix tuning learns key/value vectors for *each layer*, which is more expressive and has more parameters. Both lengthen the sequence the model attends over.
- **IA3:** "infused adapter by inhibiting and amplifying inner activations": learned scaling vectors on keys, values and the FFN hidden layer. The smallest method here.
- **QLoRA:** store the frozen base in 4-bit NormalFloat (NF4, with double quantization of the quantisation constants), keep the LoRA adapter in higher precision, back-propagate through the dequantised base. It exists to fit big models on one GPU. Reference code and the details: [06 step 06](../../06_llm_training_and_lora/06_qlora_reference/WALKTHROUGH.md). The same memory arithmetic for inference (weights, KV cache) is in [21 step 03](../../21_llm_inference_engineering/03_gpu_memory_calculator/WALKTHROUGH.md).

**Memory model used in section 6** (activations and the KV cache ignored):
```
full fine-tune : 16 B/param = 2 (bf16 weights) + 2 (bf16 grads) + 4 (fp32 master) + 8 (Adam m and v)
LoRA           : 2 B/param frozen base + 16 B x (trainable share, here 1%)
QLoRA          : ~0.56 B/param base (4-bit + quantisation constants) + 16 B x trainable share
7B example     : full 112 GB, LoRA ~15 GB, QLoRA ~5 GB
```
Current practice (Nov 2026, stated generally): LoRA and QLoRA are the default way to fine-tune open models on limited hardware; DoRA and rsLoRA are options in common libraries; classic adapters and IA3 are rarer; prompt and prefix tuning are mostly historical or niche for decoder LLMs.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| PEFT | parameter-efficient fine-tuning | yes | Umbrella for every method in this step; also the name of the Hugging Face library `peft` |
| LoRA | low-rank adaptation | yes | Low-rank `B@A` update on frozen weights |
| Adapter (generic) | adapter weights, LoRA adapter | near | In `peft` "adapter" often means the saved LoRA tensors; in the literature "adapter" can mean a bottleneck MLP |
| rsLoRA | rank-stabilised LoRA, `use_rslora=True` | yes | Scale `alpha / sqrt(r)` |
| DoRA | weight-decomposed LoRA, `use_dora=True` | yes | LoRA plus a learned magnitude vector |
| Soft prompt | prompt tuning, virtual tokens | near | Learned embeddings, not words you can read |
| Prefix tuning | per-layer soft prompt | no | Prefixes sit in the keys and values of every layer, not only the input |
| IA3 | scaling-vector adaptation | yes | Learned element-wise scales |
| QLoRA | LoRA on a 4-bit base | no | Same adapter; the *base* is quantised |
| Target modules | `target_modules`, adapted layers | yes | Names of the layers that receive adapters |
| Trainable share | trainable parameters %, `print_trainable_parameters()` | yes | trainable / total parameters |
| `fan_in_fan_out` | Conv1D layout flag | no | GPT-2 stores linear weights transposed as `Conv1D`; this flag tells `peft` |

## Run it
```bash
cd 17_finetuning_alignment_and_benchmarks/03_peft_family
python main.py
# from repo root:
HF_HUB_OFFLINE=1 .venv/bin/python 17_finetuning_alignment_and_benchmarks/03_peft_family/main.py
```
About 20 s of compute on an idle CPU (it loads distilgpt2 nine times); the run recorded below was on a heavily loaded machine, so wall-clock numbers vary by machine and load.

## Code, step by step

Lines 1–13 of `main.py` are the header docstring (GOAL / RUN / EXPECTED OUTPUT / NEEDS) already summarised above; the walk starts after it.

### 0. Imports and settings  (`main.py` lines 14–19)
```python
import torch
from peft import (IA3Config, LoraConfig, PrefixTuningConfig, PromptTuningConfig, TaskType, get_peft_model)
from transformers import AutoModelForCausalLM

torch.manual_seed(0)
MODEL = "distilgpt2"
```
- **In simple words:** Import one config class per method and the function that wraps a model with it.
- **Exactly:** `peft` provides a config dataclass per method (`LoraConfig` also covers rsLoRA and DoRA through flags) and `get_peft_model(model, config)`, which returns a `PeftModel` wrapper. `TaskType.CAUSAL_LM` tells it the model is a decoder LM so it builds the right wrapper and, for prompt/prefix tuning, the right attention handling.
- **Why it is here:** One consistent API means we can compare all methods by changing only the config.

### 1. Count parameters helper  (`main.py` lines 21–25)
```python
# ## 1. Count parameters helper
def count(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
```
- **In simple words:** Add up all the numbers in the model, and separately the ones that are allowed to learn.
- **Exactly:** `model.parameters()` yields every weight tensor; `p.numel()` is the number of elements; `p.requires_grad` is `True` for tensors the optimiser will update. `get_peft_model` sets `requires_grad=False` on the base weights and `True` on the new ones.
- **Why it is here:** "How many parameters do you train?" is the first question about any PEFT method.

### 2. The menu of PEFT configs  (`main.py` lines 27–37)
```python
# ## 2. The menu of PEFT configs (all for GPT-2-style layer names)
CONFIGS = {
    "LoRA r=8 (c_attn)":        LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"], fan_in_fan_out=True, task_type=TaskType.CAUSAL_LM),
    "LoRA r=8 (attn+mlp)":      LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn", "c_proj", "c_fc"], fan_in_fan_out=True, task_type=TaskType.CAUSAL_LM),
    "LoRA r=64 (c_attn)":       LoraConfig(r=64, lora_alpha=128, target_modules=["c_attn"], fan_in_fan_out=True, task_type=TaskType.CAUSAL_LM),
    "rsLoRA r=64 (c_attn)":     LoraConfig(r=64, lora_alpha=128, use_rslora=True, target_modules=["c_attn"], fan_in_fan_out=True, task_type=TaskType.CAUSAL_LM),
    "DoRA r=8 (c_attn)":        LoraConfig(r=8, lora_alpha=16, use_dora=True, target_modules=["c_attn"], fan_in_fan_out=True, task_type=TaskType.CAUSAL_LM),
    "IA3 (c_attn, c_fc)":       IA3Config(target_modules=["c_attn", "c_fc"], feedforward_modules=["c_fc"], fan_in_fan_out=True, task_type=TaskType.CAUSAL_LM),
    "Prompt tuning (20 tok)":   PromptTuningConfig(num_virtual_tokens=20, task_type=TaskType.CAUSAL_LM),
    "Prefix tuning (20 tok)":   PrefixTuningConfig(num_virtual_tokens=20, task_type=TaskType.CAUSAL_LM),
}
```
- **In simple words:** A dictionary of eight recipes. Reading the keys top to bottom is the family tour: different targets, different rank, rsLoRA, DoRA, IA3, prompt, prefix.
- **Exactly:** `r` is the rank; `lora_alpha` the scaling numerator; `target_modules` the layer *names* to wrap (GPT-2's attention projection is `c_attn`, output projection `c_proj`, MLP up-projection `c_fc`; note `c_proj` appears in both attention and MLP, so the "attn+mlp" config wraps both). `fan_in_fan_out=True` is required because GPT-2 implements these layers with `Conv1D`, whose weight is stored as `(in, out)`, the transpose of `nn.Linear`. `use_rslora=True` / `use_dora=True` switch variants of the same class. `IA3Config.feedforward_modules` marks which targets scale their *input* (the FFN) rather than their output. `num_virtual_tokens=20` is the length of the learned prefix/prompt.
- **Why it is here:** Each entry is one row of the family table; the next block prints their cost.

### 3. Wrap a fresh model with each config and print the numbers  (`main.py` lines 39–49)
```python
# ## 3. Wrap a fresh model with each config and print the numbers
print(f"{'method':<26}{'trainable':>12}{'total':>14}{'trainable %':>13}")
peft_models = {}
for name, cfg in CONFIGS.items():
    base = AutoModelForCausalLM.from_pretrained(MODEL)
    pm = get_peft_model(base, cfg)                      # freezes the base, adds the small trainable part
    tr, tot = count(pm)
    peft_models[name] = pm
    print(f"{name:<26}{tr:>12,}{tot:>14,}{100 * tr / tot:>12.3f}%")
full = count(AutoModelForCausalLM.from_pretrained(MODEL))[0]
print(f"{'FULL fine-tune':<26}{full:>12,}{full:>14,}{100.0:>12.3f}%")
```
- **In simple words:** For each recipe load a fresh model, attach the add-on, and print how much is trainable. Finish with the full fine-tune number for comparison.
- **Exactly:** A *fresh* `from_pretrained` per config because `get_peft_model` modifies the model it is given in place. `f"{tr:>12,}"` right-aligns with thousands separators. The trainable count reported here is the *add-on only*, and the total grows by exactly that amount over the plain model (82,060,032 total for LoRA r=8 equals 81,912,576 + 147,456).
- **Why it is here:** This is the answer to "how many parameters?" for each method, and the arithmetic below is checkable by hand.

**Check the numbers by hand** (distilgpt2 has 6 layers, `d_model = 768`; `c_attn` maps 768 to 2304 because it produces Q, K and V together):
- LoRA r=8 on `c_attn`: `r*(in + out) = 8*(768 + 2304) = 24,576` per layer; `x 6 layers = 147,456`. ✓.
- LoRA r=64 is 8x that: `1,179,648`. ✓ (Linear in `r`.)
- attn+mlp: `c_attn` 24,576 + attention `c_proj` `8*(768+768) = 12,288` + `c_fc` `8*(768+3072) = 30,720` + MLP `c_proj` `8*(3072+768) = 30,720` = 98,304 per layer; `x 6 = 589,824`. ✓.
- DoRA adds one magnitude vector of `2304` per layer: `161,280 - 147,456 = 13,824 = 6 x 2304`. ✓.
- IA3: `c_attn` vector `2304` + `c_fc` input-side vector `768` = `3072` per layer; `x 6 = 18,432`. ✓.
- Prompt tuning: `20 x 768 = 15,360`. ✓.
- Prefix tuning: `20 tokens x 2 (key and value) x 6 layers x 768 = 184,320`. ✓.

### 4. LoRA scaling: alpha/r versus rsLoRA alpha/sqrt(r)  (`main.py` lines 51–54)
```python
# ## 4. LoRA scaling: alpha/r versus rsLoRA alpha/sqrt(r)
for name in ("LoRA r=64 (c_attn)", "rsLoRA r=64 (c_attn)"):
    layer = peft_models[name].base_model.model.transformer.h[0].attn.c_attn
    print(f"{name:<26} scaling factor applied to B@A = {layer.scaling['default']:.3f}")
```
- **In simple words:** Open layer 0 of the two r=64 models and read the number that multiplies the low-rank update.
- **Exactly:** `peft_models[name]` is a `PeftModel`; `.base_model.model` reaches the underlying GPT-2 (`.base_model` is the `LoraModel` wrapper, `.model` the original network), then `transformer.h[0].attn.c_attn` is the first block's attention projection, now a `peft` LoRA layer. Its `scaling` dict is keyed by adapter name (`'default'`). Classic: `alpha / r = 128 / 64 = 2`. rsLoRA: `alpha / sqrt(r) = 128 / 8 = 16`.
- **Why it is here:** It turns the rsLoRA claim into a number you can inspect: at rank 64 the update is amplified 8x relative to classic LoRA.

### 5. Which weights actually move?  (`main.py` lines 56–59)
```python
# ## 5. Which weights actually move? (names of the trainable tensors in one LoRA layer)
pm = peft_models["LoRA r=8 (c_attn)"]
print("\ntrainable tensors in layer 0:", [(n.split("h.0.")[1], tuple(p.shape)) for n, p in pm.named_parameters() if p.requires_grad and ".h.0." in n])
print("DoRA extra tensor:", [(n.split("h.0.")[1], tuple(p.shape)) for n, p in peft_models["DoRA r=8 (c_attn)"].named_parameters() if p.requires_grad and ".h.0." in n and "magnitude" in n])
```
- **In simple words:** List the names and shapes of the trainable tensors in the first transformer block.
- **Exactly:** `named_parameters()` yields `(dotted name, tensor)`. Filtering by `requires_grad` and `".h.0."` keeps layer 0's trainable tensors; `n.split("h.0.")[1]` trims the long prefix. Result for LoRA: `lora_A` of shape `(8, 768)` and `lora_B` of shape `(2304, 8)`. For DoRA the extra tensor is the magnitude vector `(2304,)`.
- **Why it is here:** The shapes `(r, in)` and `(out, r)` are the LoRA derivation made concrete; the magnitude vector shows what DoRA adds.

### 6. Memory arithmetic  (`main.py` lines 61–72)
```python
# ## 6. Memory arithmetic: bytes per parameter during training (weights + grads + Adam states), activations ignored
def gb(x):
    return x / 1e9
def mem_full(n):                       # bf16 weights 2 + bf16 grads 2 + fp32 master 4 + Adam m,v 8  = 16 B/param
    return gb(16 * n)
def mem_lora(n, frac=0.01):            # frozen bf16 base (2 B) + trainable fraction at 16 B/param
    return gb(2 * n + 16 * frac * n)
def mem_qlora(n, frac=0.01):           # 4-bit base (0.5 B, ~0.56 with quant constants) + trainable fraction at 16 B/param
    return gb(0.56 * n + 16 * frac * n)
print(f"\n{'model':<8}{'full FT GB':>12}{'LoRA GB':>10}{'QLoRA GB':>10}   (1% trainable, activations NOT included)")
for label, n in (("0.5B", 0.5e9), ("7B", 7e9), ("70B", 70e9)):
    print(f"{label:<8}{mem_full(n):>12.1f}{mem_lora(n):>10.1f}{mem_qlora(n):>10.1f}")
```
- **In simple words:** Three tiny formulas for how many gigabytes training needs, applied to a 0.5B, a 7B and a 70B model.
- **Exactly:** `n` is the parameter count. Full fine-tune pays 16 bytes per parameter for the four tensors in the comment. LoRA pays 2 bytes for the frozen bf16 base plus 16 bytes only on the trainable fraction (assumed 1%). QLoRA replaces the 2 bytes with about 0.56 (4-bit weights plus per-block quantisation constants; the exact figure depends on block size). Activations and the KV cache are *not* counted, so these are lower bounds; gradient checkpointing and batch size change the real total. Treat 1% as a round assumption: the actual share depends on rank and targets (0.18% to 1.4% in the table above).
- **Why it is here:** "Can you fit it on one GPU?" is answered with exactly this arithmetic before any code runs. The inference-side version (weights plus KV cache) is [21 step 03](../../21_llm_inference_engineering/03_gpu_memory_calculator/WALKTHROUGH.md).

### 7. Sanity: one forward/backward on a LoRA model touches only adapter weights  (`main.py` lines 74–80)
```python
# ## 7. Sanity: one forward/backward on a LoRA model touches only adapter weights
pm = peft_models["LoRA r=8 (c_attn)"]
x = torch.randint(0, 50257, (2, 16))                     # (batch, seq) random token ids
loss = pm(input_ids=x, labels=x).loss
loss.backward()
n_grad = sum(p.grad is not None for p in pm.parameters())
print(f"\nparameters that received a gradient: {n_grad} tensors (all of them LoRA A/B); loss={loss.item():.3f}")
```
- **In simple words:** Run one training step on random tokens and count how many tensors got a gradient: it should be only the adapters.
- **Exactly:** `torch.randint(0, 50257, (2, 16))` is a `(2, 16)` batch of random ids (50,257 is GPT-2's vocabulary size). Passing `labels=x` makes the model compute its built-in shifted cross-entropy. After `loss.backward()` every tensor with `requires_grad=True` has a `.grad`; frozen ones keep `None`. The answer is 12: an `A` and a `B` for each of 6 layers. The loss (14.827 in the run) is large because the tokens are random, so it says nothing about quality.
- **Why it is here:** It proves the base is really frozen, which is the property that makes PEFT cheap (no optimiser state for the base) and lets you swap adapters.

### 8. TRY IT / BREAK IT comments  (`main.py` lines 82–83)
```python
# TRY IT: change r to 2 and 128 in the first config and watch the trainable count scale linearly with r.
# BREAK IT: remove fan_in_fan_out=True from a GPT-2 LoRA config; peft warns and swaps it. Read the warning.
```
- **In simple words:** Two experiments; the second is run in the Debugging section.
- **Exactly:** Comments only.
- **Why it is here:** Module convention.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `f"{tr:>12,}"` | right-align in 12 characters with thousands separators |
| `peft_models[name].base_model.model.transformer.h[0]` | unwrap the `PeftModel` to reach GPT-2's first block |
| `layer.scaling['default']` | the LoRA scale for the adapter named `default` |
| `n.split("h.0.")[1]` | keep the part of a parameter name after `h.0.` |
| `sum(p.grad is not None for p in ...)` | counts tensors that received a gradient |
| `torch.randint(lo, hi, shape)` | random integer tensor |

## Libraries & functions
| Function | What it does | Internally |
|---|---|---|
| `LoraConfig(...)` | describes a LoRA / rsLoRA / DoRA adapter | a dataclass; no model is touched until `get_peft_model` |
| `get_peft_model(model, cfg)` | wraps a model with the chosen method | sets `requires_grad=False` on base weights, replaces targeted `Linear`/`Conv1D` layers with wrapper layers holding the adapter tensors, returns a `PeftModel` |
| `IA3Config`, `PromptTuningConfig`, `PrefixTuningConfig` | configs for the other methods | IA3 wraps layers with scaling vectors; prompt tuning adds an embedding table prepended to the inputs; prefix tuning adds learned key/value tensors for every layer |
| `model.parameters()` / `named_parameters()` | iterate tensors | names are dotted module paths |
| `loss.backward()` | reverse-mode autodiff | fills `.grad` only for tensors with `requires_grad=True` |

## Execution trace (real run)
```
method                       trainable         total  trainable %
LoRA r=8 (c_attn)              147,456    82,060,032       0.180%
LoRA r=8 (attn+mlp)            589,824    82,502,400       0.715%
LoRA r=64 (c_attn)           1,179,648    83,092,224       1.420%
rsLoRA r=64 (c_attn)         1,179,648    83,092,224       1.420%
DoRA r=8 (c_attn)              161,280    82,073,856       0.197%
IA3 (c_attn, c_fc)              18,432    81,931,008       0.022%
Prompt tuning (20 tok)          15,360    81,927,936       0.019%
Prefix tuning (20 tok)         184,320    82,096,896       0.225%
FULL fine-tune              81,912,576    81,912,576     100.000%
LoRA r=64 (c_attn)         scaling factor applied to B@A = 2.000
rsLoRA r=64 (c_attn)       scaling factor applied to B@A = 16.000

trainable tensors in layer 0: [('attn.c_attn.lora_A.default.weight', (8, 768)), ('attn.c_attn.lora_B.default.weight', (2304, 8))]
DoRA extra tensor: [('attn.c_attn.lora_magnitude_vector.default.weight', (2304,))]

model     full FT GB   LoRA GB  QLoRA GB   (1% trainable, activations NOT included)
0.5B             8.0       1.1       0.4
7B             112.0      15.1       5.0
70B           1120.0     151.2      50.4

parameters that received a gradient: 12 tensors (all of them LoRA A/B); loss=14.827
```
(A one-line `transformers` note about `loss_type=None`, progress bars and the torch Redirects note are omitted: logging noise.)
- **Counts**: LoRA r=8 trains 147,456 parameters (0.18%). All the counts match the hand arithmetic above: LoRA scales linearly with rank (r=64 is exactly 8x), DoRA adds 13,824, IA3 trains only 18,432 (0.022%), prompt tuning 15,360, prefix tuning 184,320.
- **rsLoRA and LoRA have identical parameter counts** (1,179,648) and differ only in the scaling factor: 2.0 versus 16.0. It is a *training dynamics* change, not a size change.
- **Totals**: every PEFT total equals the base total plus the add-on, e.g. 81,912,576 + 147,456 = 82,060,032. Prompt tuning and prefix tuning leave the *weights* untouched and add parameters only for the virtual tokens.
- **Memory**: full fine-tuning of 7B needs 112 GB (16 B x 7e9), LoRA 15.1 GB, QLoRA 5.0 GB, before activations. The 70B row (1,120 GB vs 50 GB) is why QLoRA exists; but note the table is a floor: real runs also need activations, a KV-free but non-zero workspace and dequantisation buffers.
- **Gradients**: 12 tensors received a gradient: `lora_A` and `lora_B` in each of 6 layers. The base is frozen. The loss of 14.827 is meaningless (random tokens).

## Experiments (change ONE thing)
| Change | What you will see | Why |
|---|---|---|
| default | the table above | baseline |
| (derived from the table, not a separate run) `r` 8 -> 64 on `c_attn` | trainable 147,456 -> 1,179,648 (8x) | `r * (in + out)` per layer, linear in `r` |
| (derived) target `["c_attn","c_proj","c_fc"]` instead of `["c_attn"]` | 147,456 -> 589,824 (4x) | three more matrices per layer |
| (not run) `r=2` and `r=128` as the file suggests | 36,864 and 2,359,296 trainable (hand arithmetic: `r x 3072 x 6`) | linear in `r` |
| (not run, theory) `use_rslora=True` at `r=8, alpha=16` | scale `16 / sqrt(8) = 5.657` instead of 2 | rsLoRA changes the scale at every rank, the difference grows with `r` |

## Debugging practice
**BREAK IT:** remove `fan_in_fan_out=True` from a GPT-2 LoRA config. Real output of the check (config without the flag, then `get_peft_model`):
```
WARNING: fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True.
fan_in_fan_out now: True
trainable params: 147,456 || all params: 82,060,032 || trainable%: 0.1797
```

How to read it: GPT-2's `c_attn`, `c_proj` and `c_fc` are `Conv1D` modules, whose weight is stored as `(in, out)` instead of `nn.Linear`'s `(out, in)`. If `peft` applied the LoRA update without transposing, the shapes of `B @ A` would be wrong or, worse for square matrices, silently transposed. The library detects the `Conv1D` type, prints the warning above and corrects the flag itself, so the trainable count is still 147,456 (same as the correct config). The lesson: when targeting a layer type you have not used before, read the first warning and print `count()` to see that the trainable number is what your hand arithmetic says. The same class of bug in your own code shows up as a shape error (`mat1 and mat2 shapes cannot be multiplied`) or, for square matrices, as a model that trains but learns nothing useful.

## Common mistakes
- Quoting "LoRA trains 0.1% of the weights" as a fixed fact: the share depends on rank and on which modules are targeted (0.18% to 1.42% here).
- Forgetting that `target_modules` are *names*; a wrong name raises an error in `peft` rather than silently doing nothing, but a name that matches more layers than intended (like `c_proj`) is easy to miss.
- Using `alpha / r` intuition for rsLoRA: at rank 64 the scale is 16, not 2.
- Treating the memory table as a total: activations, KV cache, optimiser workspace and quantisation buffers come on top.
- Assuming QLoRA runs on any machine: it needs 4-bit kernels (`bitsandbytes`) and, in practice, a GPU. It is reference-only in this course.
- Merging an adapter into a quantised base without care: the merge adds a full-precision delta to quantised weights; know whether your library de-quantises first.
- Calling prompt tuning "free": it lengthens every sequence.

## Rebuild checklist
1. A `count(model)` returning `(trainable, total)`.
2. A dict of configs: LoRA (two target sets, two ranks), rsLoRA, DoRA, IA3, prompt, prefix.
3. A loop that loads a *fresh* model per config, calls `get_peft_model`, and prints the counts.
4. Read the `scaling['default']` of a LoRA and an rsLoRA layer.
5. Print the trainable tensor names and shapes of layer 0.
6. The three memory functions and a 0.5B / 7B / 70B table.
7. One forward/backward and a count of tensors with `.grad`.

## Say it aloud
"PEFT freezes the pretrained weights and trains a small add-on. LoRA adds a low-rank update to chosen weight matrices; rsLoRA changes the scale from alpha over r to alpha over the square root of r so high ranks stay stable; DoRA adds a learned magnitude vector; IA3 learns scaling vectors; prompt and prefix tuning learn virtual tokens or per-layer key-value prefixes; QLoRA is LoRA on a 4-bit quantised base. On distilgpt2, LoRA at rank 8 on the attention projection trains 147,456 parameters, 0.18%. For a 7B model full fine-tuning needs about 16 bytes per parameter, roughly 112 GB, while LoRA is about 15 GB and QLoRA about 5 GB before activations."

## The new problem (cliffhanger)
So far every step trained a model to *imitate* an answer. But "imitate" cannot say that one acceptable answer is better than another. Preference data (chosen versus rejected) can, and the classic route is to train a model that scores answers. On to [step 04](../04_reward_model_bradley_terry/WALKTHROUGH.md).
