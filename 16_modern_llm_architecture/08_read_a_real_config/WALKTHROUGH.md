# Step 08 — Read a real config: Qwen2.5-0.5B

## The story so far
[Step 07](../07_build_mini_llama/WALKTHROUGH.md) built a Llama-style decoder with our own names and a byte vocabulary. Real checkpoints use Hugging Face's names and thousands of times more parameters, but the structure is the same.

## The pain point
A model card gives you a `config.json` and a list of tensors like `model.layers.17.self_attn.k_proj.weight`. Unless you can map each field and each tensor to a concept (GQA, SwiGLU, RoPE base, tied embeddings), you cannot estimate memory, debug a shape error, convert weights or answer "why is this model 0.5B but 2 GB?".

## The spark
Every field and tensor name in a Llama-family config corresponds to one line of the class in step 07. Build the architecture from the config on the **meta device** (shapes only, no memory), list every tensor, count parameters by component, and finally prove the mapping by loading our mini-Llama's weights into Hugging Face's `LlamaForCausalLM` and checking identical logits.

## Goal
Map config fields to concepts, list tensor names and shapes, count parameters per component and verify against a formula, compute the KV cache per token, contrast with GPT-2's layout, and verify our model equals HF Llama to 4e-7.

## The theory (read this before the interview)
**Config fields (Llama/Qwen family)**
| Field | Meaning | Step |
|---|---|---|
| `hidden_size` | `d_model` | 03, 07 |
| `num_hidden_layers` | number of blocks | 07 |
| `num_attention_heads` | query heads `H` | 04 |
| `num_key_value_heads` | `G`; if missing the model is MHA | 04 |
| `head_dim` | usually `hidden_size / H`; explicit in Qwen3, gpt-oss | 04 |
| `intermediate_size` | SwiGLU hidden width | 03 |
| `hidden_act` | `silu` means SwiGLU (with gate/up/down) | 03 |
| `rms_norm_eps` | epsilon in RMSNorm | 03 |
| `rope_theta` (`rope_parameters` in transformers 5) / `rope_scaling` | RoPE base / long-context scaling (yarn, llama3, linear) | 02 |
| `max_position_embeddings` | context length the model was configured for | 02 |
| `vocab_size` | embedding rows, often padded above the real token count | 01 |
| `tie_word_embeddings` | head shares the embedding matrix | 07 |
| `sliding_window`, `use_sliding_window` | local attention | 04 |
| `num_experts`, `num_experts_per_tok`, `moe_intermediate_size` | MoE | 06 |
| `kv_lora_rank`, `q_lora_rank` | MLA (DeepSeek) | 04 |
| `torch_dtype` / `dtype` | storage precision (bf16 = 2 bytes/parameter) | |

**Tensor names (Hugging Face Llama/Qwen) and our step-07 names**
| Hugging Face name | Shape (Qwen2.5-0.5B) | Our class attribute |
|---|---|---|
| `model.embed_tokens.weight` | (151936, 896) | `embed.weight` |
| `layers.i.input_layernorm.weight` | (896,) | `blocks[i].attn_norm` |
| `layers.i.self_attn.q_proj.weight` (+bias) | (896, 896) | `attn.wq` |
| `layers.i.self_attn.k_proj.weight` (+bias) | (128, 896) | `attn.wk` |
| `layers.i.self_attn.v_proj.weight` (+bias) | (128, 896) | `attn.wv` |
| `layers.i.self_attn.o_proj.weight` | (896, 896) | `attn.wo` |
| `layers.i.post_attention_layernorm.weight` | (896,) | `blocks[i].mlp_norm` |
| `layers.i.mlp.gate_proj / up_proj / down_proj` | (4864, 896) / (4864, 896) / (896, 4864) | `mlp.gate / up / down` |
| `model.norm.weight` | (896,) | `final_norm` |
| `lm_head.weight` | (151936, 896), tied | `lm_head` |
`nn.Linear` stores `weight` as `(out_features, in_features)`, so `k_proj` is `(128, 896)`: 2 KV heads x 64 head_dim = 128 outputs. GPT-2's `Conv1D` layers store the transpose `(in, out)`, and it fuses q, k, v in one `c_attn` of width 3·768 = 2304.

**Model-card parameter counts.** "0.5B" for Qwen2.5-0.5B is the total (494M, with the embedding counted once). Qwen quotes ~0.36B *non-embedding* parameters, which matches the 357.9M computed below. For MoE models cards state total and active. Memory to load = parameters x bytes per parameter (bf16: x2): 494M x 2 bytes = 988 MB; plus the KV cache and activations at run time. **Meta device:** `torch.device("meta")` creates tensors with shapes but no storage; you can instantiate a 405B-parameter architecture on a laptop to read names and shapes (the `accelerate` library uses the same trick to load large models layer by layer).

**Why equality with HF Llama matters.** If you can load your own weights into the reference implementation and get the same logits to float precision, you have verified every design detail (rotation layout, norm epsilon, GQA grouping, tie). This is the standard way to validate a port.

**Note on cached files.** This repository's cache contains only the config and tokenizer files of Qwen2.5-0.5B, not its weights, so step 08 inspects the architecture (names, shapes, counts), which is everything the config determines. Loading real weights needs a network or a full download (`AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")`).

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| `config.json` | model config, `AutoConfig`, hyperparameters | Yes | architecture description (no weights) |
| `state_dict` | weights, checkpoint, `model.safetensors` | Mostly | name -> tensor mapping; safetensors is the file format |
| `q_proj/k_proj/v_proj/o_proj` | `wq/wk/wv/wo` (original Llama code) | Yes | same linears, different names |
| `gate_proj/up_proj/down_proj` | `w1/w3/w2` (Llama reference code) | Yes | gate = w1, up = w3, down = w2 |
| `input_layernorm` / `post_attention_layernorm` | attn_norm / ffn_norm | Yes | pre-attention and pre-MLP RMSNorm (despite the "LayerNorm" in the name) |
| `embed_tokens` | `wte`, `tok_embeddings` | Yes | embedding table |
| `lm_head` | output projection, unembedding | Yes | hidden state to logits |
| Parameters vs weights vs tensors | | Mostly | parameters = count of scalars; tensors are the arrays holding them |
| Meta device | `init_empty_weights` | Related | shapes without memory |

## Run it
```bash
cd 16_modern_llm_architecture/08_read_a_real_config
HF_HUB_OFFLINE=1 python main.py     # ~5 s; needs the cached config of Qwen/Qwen2.5-0.5B and gpt2
```

## Code, step by step
### 1. Imports, offline mode and seed  (`main.py` lines 16–25)
```python
import importlib.util
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

torch.manual_seed(0)
```
- **In simple words:** tools to import a sibling file, group counts, run offline, and load Hugging Face classes.
- **Exactly:** `importlib.util` loads step 07's `main.py` as a module without it being on `sys.path`; `defaultdict(int)` accumulates counts; `LlamaConfig/LlamaForCausalLM` are the reference implementation.
- **Why it is here:** the equality test in section 5 needs both.
### 2. Load the config  (`main.py` lines 28–28)
```python
cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
```
- **Exactly:** `AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")` reads the cached `config.json` and returns a `Qwen2Config` object.
- **In simple words:** read the model's specification sheet.
- **Why it is here:** every shape below is derived from these fields.
### 3. Field-to-concept table and printing  (`main.py` lines 29–45)
```python
meaning = {
    "hidden_size": "d_model: width of the residual stream",
    "intermediate_size": "SwiGLU hidden width (step 03)",
    "num_hidden_layers": "number of decoder blocks",
    "num_attention_heads": "query heads",
    "num_key_value_heads": "K/V heads: 2 < 14 means GQA (step 04)",
    "max_position_embeddings": "context window the RoPE tables/training cover (step 02)",
    "rope_parameters": "RoPE base (rope_theta, step 02); 10000 in the paper, 1e6 for long context (older transformers: config.rope_theta)",
    "rms_norm_eps": "epsilon inside RMSNorm (step 03)",
    "hidden_act": "silu -> SwiGLU = silu(gate) * up",
    "tie_word_embeddings": "lm_head shares the embedding matrix",
    "vocab_size": "rows of the embedding table (padded; the tokenizer has fewer real tokens, step 01)",
    "use_sliding_window": "sliding-window attention switched off here (step 04)",
    "dtype": "dtype the weights were saved in (older transformers: torch_dtype)",
}
for k, why in meaning.items():
    print(f"{k:<26}{str(getattr(cfg, k, None)):<44} {why}")
```
- **Exactly:** a dict of field name to explanation, printed with the actual values: hidden_size 896, intermediate_size 4864, 24 layers, 14 query heads, **2 KV heads**, 32,768 positions, `rope_parameters` `{'rope_theta': 1000000.0, 'rope_type': 'default'}`, `rms_norm_eps` 1e-06, `silu`, tied embeddings True, vocab 151,936, sliding window off, dtype bfloat16. (In transformers 5 the RoPE base lives in `rope_parameters`; older versions have `config.rope_theta`.) Derived: head_dim = 896/14 = 64 and 7 query heads per KV head.
- **In simple words:** each number in the config is one design decision from steps 01-07.
- **Why it is here:** this reading skill is the point of the step.
### 4. Derived values  (`main.py` lines 46–47)
```python
head_dim = cfg.hidden_size // cfg.num_attention_heads
print(f"derived: head_dim = {cfg.hidden_size}/{cfg.num_attention_heads} = {head_dim}; group size = {cfg.num_attention_heads // cfg.num_key_value_heads} query heads per KV head")
```
- **Exactly:** `head_dim = hidden_size // num_attention_heads`; the group size is 14 // 2 = 7.
- **In simple words:** values not written in the config but implied.
- **Why it is here:** these two numbers feed the KV-cache and shape formulas.
### 5. Build the architecture on the meta device  (`main.py` lines 50–52)
```python
with torch.device("meta"):
    qwen = AutoModelForCausalLM.from_config(cfg)
sd = qwen.state_dict()
```
- **Exactly:** inside `with torch.device("meta")`, `AutoModelForCausalLM.from_config(cfg)` creates all parameters as meta tensors (shape and dtype, no memory). `state_dict()` returns 291 tensors.
- **In simple words:** an empty model skeleton that costs no memory.
- **Why it is here:** lets you inspect a model of any size on a laptop.
### 6. List the tensors  (`main.py` lines 53–58)
```python
print(f"\n{type(qwen).__name__}: {len(sd)} tensors; first block (layer 0) and the non-layer tensors:")
for name, t in sd.items():
    if ".layers." not in name or ".layers.0." in name:
        print(f"  {name:<52}{tuple(t.shape)}")
print("lm_head.weight in state_dict?", "lm_head.weight" in sd, "| same storage as embed_tokens?",
      qwen.lm_head.weight is qwen.model.embed_tokens.weight)
```
- **Exactly:** prints the non-layer tensors and layer 0 only (all 24 layers repeat it): `embed_tokens (151936, 896)`; `q_proj (896, 896)` and bias `(896,)`; `k_proj (128, 896)` and `v_proj (128, 896)` (2 heads x 64) with biases `(128,)`; `o_proj (896, 896)` no bias; `gate_proj`/`up_proj` `(4864, 896)`, `down_proj` `(896, 4864)`; two norm vectors `(896,)`; `model.norm`; `lm_head (151936, 896)`. The last line shows `lm_head.weight` appears in the state dict yet shares storage with `embed_tokens` (`True`).
- **In simple words:** the full list of ingredients of the model, each with its size.
- **Why it is here:** the mapping table of the theory section, as printed by the library. Notice q, k, v have biases (Qwen2 keeps them) but `o_proj` and the MLP do not.
### 7. Count parameters by component  (`main.py` lines 61–73)
```python
groups = defaultdict(int)
for name, p in qwen.named_parameters():                                      # named_parameters de-duplicates tied weights
    if "embed_tokens" in name:
        groups["embedding (tied with lm_head)"] += p.numel()
    elif "self_attn" in name and name.endswith("bias"):
        groups["attention biases (q,k,v only)"] += p.numel()
    elif "self_attn" in name:
        groups["attention projections"] += p.numel()
    elif "mlp" in name:
        groups["MLP (SwiGLU)"] += p.numel()
    else:
        groups["RMSNorm weights"] += p.numel()
total = sum(groups.values())
```
- **In simple words:** sort every weight into a bucket (embedding, attention, MLP, norms) and add them up.
- **Exactly:** `named_parameters()` yields each *unique* parameter once, so the tied matrix is counted once under `embed_tokens`. String tests on the name decide the bucket.
- **Why it is here:** answers "where are the parameters?".
### 8. Print the breakdown  (`main.py` lines 74–78)
```python
for k, v in groups.items():
    print(f"{k:<34}{v:>14,}  {v / total:6.1%}")
print(f"{'TOTAL':<34}{total:>14,}")
non_emb = total - groups["embedding (tied with lm_head)"]
print(f"non-embedding params {non_emb:,} (the '0.36B' you see on the model card) | embedding share {groups['embedding (tied with lm_head)'] / total:.1%}")
```
- **Exactly:** embedding 136,134,656 (27.6%); attention projections 44,040,192 (8.9%); attention biases 27,648; MLP 313,786,368 (63.5%); RMSNorm weights 43,904; **total 494,032,768**; non-embedding 357,898,112. Takeaway: for a small model the embedding is over a quarter of everything, and the MLP is almost two thirds of the rest.
- **In simple words:** most of the model is the MLP; for tiny models a lot is the vocabulary table.
- **Why it is here:** tells you what to optimise (MoE/quantisation for the MLP; tie or shrink the vocabulary for small models).
### 9. Closed-form formula  (`main.py` lines 81–84)
```python
d, L, h, kv, ff, V = cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.intermediate_size, cfg.vocab_size
attn = d * d + 2 * d * (kv * head_dim) + d * d + (d + 2 * kv * head_dim)        # q, k, v, o weights + q,k,v biases
formula = V * d + L * (attn + 3 * d * ff + 2 * d) + d
print("formula  V*d + L*(attn + 3*d*ff + 2*d) + d =", f"{formula:,}", "| matches:", formula == total)
```
- **Exactly:** per layer attention = `q: d·d` + `k,v: 2·d·(kv·head_dim)` + `o: d·d` + biases `d + 2·kv·head_dim`; MLP `3·d·ff`; norms `2d`. Total `V·d + L·(attn + 3·d·ff + 2d) + d` = 494,032,768, matching the counted total exactly (`True`).
- **In simple words:** a formula that reproduces the model size from six numbers in the config.
- **Why it is here:** if your formula matches the real count you have understood every field.
### 10. KV cache per token  (`main.py` lines 85–85)
```python
print(f"KV cache: 2 * {L} layers * {kv} kv_heads * {head_dim} head_dim * 2 bytes (bf16) = {2 * L * kv * head_dim * 2} bytes/token")
```
- **Exactly:** `2 · 24 layers · 2 kv_heads · 64 · 2 bytes (bf16) = 12,288` bytes/token = 12 KB, so 32,768 tokens need 0.4 GB (the formula of step 04). With the 14 heads of MHA it would be 7x larger.
- **In simple words:** a 32k-token chat costs about 400 MB of cache.
- **Why it is here:** the number you need for capacity planning from nothing but a config.
### 11. GPT-2 contrast: build and print its tensor names  (`main.py` lines 88–93)
```python
with torch.device("meta"):
    g2 = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained("gpt2"))
print("\nGPT-2 block 0 tensors (fused qkv, biases, LayerNorm, learned positions):")
for name, t in g2.state_dict().items():
    if ".h.0." in name or "wpe" in name or "wte" in name:
        print(f"  {name:<34}{tuple(t.shape)}")
```
- **Exactly:** builds GPT-2 on the meta device and prints block 0: `wte (50257, 768)`, `wpe (1024, 768)` (learned positions), `ln_1` with weight **and bias**, fused `attn.c_attn (768, 2304)` (q, k, v concatenated) with bias, `c_proj`, `ln_2`, `mlp.c_fc (768, 3072)` and `mlp.c_proj (3072, 768)` with biases. No `rope`, no `gate_proj`, no GQA.
- **In simple words:** the 2019 design: fused attention, learned positions, biases everywhere, two-matrix MLP.
- **Why it is here:** seeing both layouts side by side makes the evolution concrete. (Note GPT-2's `Conv1D` stores weights as `(in, out)`, the transpose of `nn.Linear`.)
### 12. Import our own mini-Llama as a module  (`main.py` lines 96–98)
```python
spec = importlib.util.spec_from_file_location("mini", Path(__file__).resolve().parents[1] / "07_build_mini_llama" / "main.py")
mini = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mini)                                                   # imports the classes; training only runs under __main__
```
- **Exactly:** `spec_from_file_location` + `module_from_spec` + `exec_module` run step 07's file; its training code is inside `main()`, so only the class definitions execute.
- **In simple words:** borrow the model code from the previous folder.
- **Why it is here:** the equality test needs our `Transformer`.
### 13. Instantiate ours and the Hugging Face reference  (`main.py` lines 99–105)
```python
mc = mini.ModelConfig()
ours = mini.Transformer(mc).eval()
hf = LlamaForCausalLM(LlamaConfig(
    vocab_size=mc.vocab_size, hidden_size=mc.d_model, intermediate_size=mc.ffn_hidden, num_hidden_layers=mc.n_layers,
    num_attention_heads=mc.n_heads, num_key_value_heads=mc.n_kv_heads, max_position_embeddings=mc.max_seq,
    rms_norm_eps=mc.norm_eps, rope_theta=mc.rope_base, tie_word_embeddings=True, attention_bias=False, mlp_bias=False,
)).eval()
```
- **Exactly:** `ours` has random weights (seed 0); `hf = LlamaForCausalLM(LlamaConfig(...))` is built with the same hyperparameters: heads 8, KV heads 2, hidden 128, intermediate 352, RoPE theta 10000, tied embeddings, no attention/MLP bias.
- **In simple words:** two independent implementations of the same architecture.
- **Why it is here:** if they disagree after copying weights, one of them has a design difference.
### 14. Build the name map  (`main.py` lines 106–111)
```python
rename = {"embed.weight": "model.embed_tokens.weight", "final_norm.weight": "model.norm.weight"}
for i in range(mc.n_layers):
    for a, b in [("attn_norm", "input_layernorm"), ("mlp_norm", "post_attention_layernorm"),
                 ("attn.wq", "self_attn.q_proj"), ("attn.wk", "self_attn.k_proj"), ("attn.wv", "self_attn.v_proj"),
                 ("attn.wo", "self_attn.o_proj"), ("mlp.gate", "mlp.gate_proj"), ("mlp.up", "mlp.up_proj"), ("mlp.down", "mlp.down_proj")]:
        rename[f"blocks.{i}.{a}.weight"] = f"model.layers.{i}.{b}.weight"
```
- **Exactly:** a dict `our name -> HF name`; two fixed entries (embedding, final norm) plus nine per layer from a list of pairs (`attn.wq` -> `self_attn.q_proj`, `mlp.gate` -> `mlp.gate_proj`, ...). 2 + 9·4 = 38 entries (printed as "renamed 38 tensors").
- **In simple words:** a dictionary translating our labels to the library's labels.
- **Why it is here:** exactly the mapping table in the theory section, in code.
### 15. Copy the weights across  (`main.py` lines 112–113)
```python
our_sd = ours.state_dict()
hf.load_state_dict({rename[k]: v for k, v in our_sd.items() if k in rename}, strict=False)   # lm_head is tied, so absent
```
- **Exactly:** `load_state_dict(..., strict=False)` ignores keys that are missing on either side; `lm_head.weight` is not passed because tying makes it the embedding.
- **In simple words:** pour our weights into their model.
- **Why it is here:** after this the two models hold identical numbers.
### 16. Run both on the same input  (`main.py` lines 114–116)
```python
ids = torch.randint(0, 256, (2, 24))
with torch.no_grad():
    la, lb = ours(ids), hf(ids).logits
```
- **Exactly:** a batch of 2 sequences of 24 random byte ids; `hf(ids).logits` has shape `(2, 24, 256)`.
- **In simple words:** the same question to both.
- **Why it is here:** the test input.
### 17. Compare  (`main.py` lines 117–118)
```python
print(f"\nrenamed {len(rename)} tensors; mini-Llama vs HF LlamaForCausalLM: max |logit diff| = {(la - lb).abs().max().item():.2e}")
print("same architecture, same math:", torch.allclose(la, lb, atol=1e-4))
```
- **Exactly:** max |logit difference| = 3.58e-07 (float32 rounding), `allclose(atol=1e-4)` True. Same rotation layout, same epsilon, same GQA grouping, same tying.
- **In simple words:** two separately written implementations agree to 7 decimal places, so our design is the real Llama design.
- **Why it is here:** the proof that the toy is not a toy in structure.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `with torch.device("meta"):` | create tensors on the meta device inside the block | no memory used |
| `getattr(obj, name, default)` | attribute by string with fallback | print config fields |
| `defaultdict(int)` | dict whose missing keys start at 0 | `groups[k] += n` |
| `str.endswith("bias")` | suffix test | bias parameters |
| `{rename[k]: v for k, v in d.items() if k in rename}` | dict comprehension with filter and renamed keys | weight translation |
| `f"{x:<44}"` | left-align in 44 columns | tidy table |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `AutoConfig.from_pretrained` | transformers | load a config | name | config |
| `AutoModelForCausalLM.from_config` | transformers | build the architecture (random/meta weights) | config | model |
| `LlamaForCausalLM`, `LlamaConfig` | transformers | reference Llama implementation | config fields | model |
| `model.state_dict()` | torch | name -> tensor | | dict |
| `model.named_parameters()` | torch | unique (name, parameter) pairs | | iterator |
| `importlib.util.spec_from_file_location` | built-in | import a file by path | name, path | spec |

## Execution trace (real run)
```text
hidden_size               896                                          d_model: width of the residual stream
intermediate_size         4864                                         SwiGLU hidden width (step 03)
num_hidden_layers         24                                           number of decoder blocks
num_attention_heads       14                                           query heads
num_key_value_heads       2                                            K/V heads: 2 < 14 means GQA (step 04)
max_position_embeddings   32768                                        context window the RoPE tables/training cover (step 02)
rope_parameters           {'rope_theta': 1000000.0, 'rope_type': 'default'} RoPE base (rope_theta, step 02); 10000 in the paper, 1e6 for long context (older transformers: config.rope_theta)
rms_norm_eps              1e-06                                        epsilon inside RMSNorm (step 03)
hidden_act                silu                                         silu -> SwiGLU = silu(gate) * up
tie_word_embeddings       True                                         lm_head shares the embedding matrix
vocab_size                151936                                       rows of the embedding table (padded; the tokenizer has fewer real tokens, step 01)
use_sliding_window        False                                        sliding-window attention switched off here (step 04)
dtype                     torch.bfloat16                               dtype the weights were saved in (older transformers: torch_dtype)
derived: head_dim = 896/14 = 64; group size = 7 query heads per KV head

Qwen2ForCausalLM: 291 tensors; first block (layer 0) and the non-layer tensors:
  model.embed_tokens.weight                           (151936, 896)
  model.layers.0.self_attn.q_proj.weight              (896, 896)
  model.layers.0.self_attn.q_proj.bias                (896,)
  model.layers.0.self_attn.k_proj.weight              (128, 896)
  model.layers.0.self_attn.k_proj.bias                (128,)
  model.layers.0.self_attn.v_proj.weight              (128, 896)
  model.layers.0.self_attn.v_proj.bias                (128,)
  model.layers.0.self_attn.o_proj.weight              (896, 896)
  model.layers.0.mlp.gate_proj.weight                 (4864, 896)
  model.layers.0.mlp.up_proj.weight                   (4864, 896)
  model.layers.0.mlp.down_proj.weight                 (896, 4864)
  model.layers.0.input_layernorm.weight               (896,)
  model.layers.0.post_attention_layernorm.weight      (896,)
  model.norm.weight                                   (896,)
  lm_head.weight                                      (151936, 896)
lm_head.weight in state_dict? True | same storage as embed_tokens? True
embedding (tied with lm_head)        136,134,656   27.6%
attention projections                 44,040,192    8.9%
attention biases (q,k,v only)             27,648    0.0%
MLP (SwiGLU)                         313,786,368   63.5%
RMSNorm weights                           43,904    0.0%
TOTAL                                494,032,768
non-embedding params 357,898,112 (the '0.36B' you see on the model card) | embedding share 27.6%
formula  V*d + L*(attn + 3*d*ff + 2*d) + d = 494,032,768 | matches: True
KV cache: 2 * 24 layers * 2 kv_heads * 64 head_dim * 2 bytes (bf16) = 12288 bytes/token

GPT-2 block 0 tensors (fused qkv, biases, LayerNorm, learned positions):
  transformer.wte.weight            (50257, 768)
  transformer.wpe.weight            (1024, 768)
  transformer.h.0.ln_1.weight       (768,)
  transformer.h.0.ln_1.bias         (768,)
  transformer.h.0.attn.c_attn.weight(768, 2304)
  transformer.h.0.attn.c_attn.bias  (2304,)
  transformer.h.0.attn.c_proj.weight(768, 768)
  transformer.h.0.attn.c_proj.bias  (768,)
  transformer.h.0.ln_2.weight       (768,)
  transformer.h.0.ln_2.bias         (768,)
  transformer.h.0.mlp.c_fc.weight   (768, 3072)
  transformer.h.0.mlp.c_fc.bias     (3072,)
  transformer.h.0.mlp.c_proj.weight (3072, 768)
  transformer.h.0.mlp.c_proj.bias   (768,)

renamed 38 tensors; mini-Llama vs HF LlamaForCausalLM: max |logit diff| = 3.58e-07
same architecture, same math: True
```
Reading it: the 291 tensors are 24 x 12 (q, k, v weights and biases, o, three MLP matrices, two norms) plus embedding, final norm and head; the parameter total matches the formula to the last digit; the HF-vs-ours difference is at float32 rounding level.

## Experiments (change ONE thing)
Every row below was run on a temporary copy of `main.py` (the real file is untouched); only the named change was made. Baseline: `max |logit diff| = 3.58e-07`, `same architecture, same math: True`, total 494,032,768 parameters.

| Change | What you will see | Why |
|---|---|---|
| `rope_theta=500000.0` in the HF config (mini-Llama still uses 10000) (measured) | diff 9.07e-03, `same architecture, same math: False` | A different RoPE base means different rotation angles, so logits move. The change is small because this mini model has `head_dim` 16, but it is 4 orders of magnitude above float noise and the test catches it. |
| `rms_norm_eps=1e-2` in the HF config (mini uses 1e-5) (measured) | diff 6.89e-01, `False` | A large eps is not a rounding detail: it shifts every normalised activation in every layer. Copy `rms_norm_eps` from the checkpoint. |
| `tie_word_embeddings=False` in the HF config (measured) | diff 1.24e+00, `False` | The HF model now has its own randomly initialised `lm_head` that our weights never fill (we only copy the tied embedding). |
| leave `("mlp.up", "mlp.up_proj")` out of the rename list (measured) | `renamed 34 tensors` (was 38), diff 6.08e-01, `False`; no error | `load_state_dict(..., strict=False)` silently ignores what it did not get. See drill 1. |
| point section 1 at `AutoConfig.from_pretrained("gpt2")` (measured) | most fields print `None` (no `intermediate_size`, `rms_norm_eps`, `num_key_value_heads`...) and the script stops with `AttributeError` at the `head_dim` line | GPT-2's config uses other field names (`n_embd`, `n_layer`) and its architecture has no GQA, RMSNorm or RoPE. Config field names are model-family specific. |
| delete `with torch.device("meta"):` around the Qwen build (not run — prediction) | works, but allocates about 2 GB of fp32 weights (494M params x 4 bytes) and takes noticeably longer | The meta device holds shapes only, which is how a 70B config can be inspected on a laptop. |

## Debugging practice
**BREAK IT drill 1: forget one tensor in the rename map.** In the rename list, delete `("mlp.up", "mlp.up_proj")`:

```diff
-                 ("attn.wo", "self_attn.o_proj"), ("mlp.gate", "mlp.gate_proj"), ("mlp.up", "mlp.up_proj"), ("mlp.down", "mlp.down_proj")]:
+                 ("attn.wo", "self_attn.o_proj"), ("mlp.gate", "mlp.gate_proj"), ("mlp.down", "mlp.down_proj")]:
```

Symptom (real run): no exception, and a wrong answer.

```text
renamed 34 tensors; mini-Llama vs HF LlamaForCausalLM: max |logit diff| = 6.08e-01
same architecture, same math: False
```

Root cause: the HF model keeps its randomly initialised `up_proj` in all four layers, because `load_state_dict(strict=False)` skips missing keys without a word (we need `strict=False` only because `lm_head` is tied). Fix: restore the entry. Better fix: load with `strict=False` and print the returned `missing_keys` / `unexpected_keys`; the only missing key allowed here is `lm_head.weight`.

**BREAK IT drill 2: read a config field the architecture does not have.** Change section 1 to `AutoConfig.from_pretrained("gpt2")`.

Symptom (real run): the field table prints `None` for most rows, then

```text
AttributeError: 'GPT2Config' object has no attribute 'num_key_value_heads'
```

Root cause: the script assumes a Llama/Qwen-style config. GPT-2 calls the width `n_embd` (aliased to `hidden_size`), has no GQA, no RMSNorm, no SwiGLU. Fix: read the `config.json` first and map names per family; `getattr(cfg, name, None)` as the file does for the table is safe, but the derived-values line is not.

## Common mistakes
- Trusting `load_state_dict(strict=False)` without inspecting `missing_keys` and `unexpected_keys`. It is the standard way to load a silently wrong model.
- Comparing logits with a loose tolerance. Here float noise is about 1e-7, a wrong RoPE base is 9e-3 and a wrong eps is 0.7; use `atol` of 1e-4 or tighter.
- Confusing "total" and "non-embedding" parameters: Qwen2.5-0.5B is 494M with its 136M tied embedding and 358M without it.
- Assuming `head_dim = hidden_size / num_attention_heads` for every model. Some configs set `head_dim` explicitly.
- Forgetting the biases on q, k and v in Qwen (the parameter formula and the Llama config `attention_bias` differ from Llama).
- Loading a 0.5B model "just to look at it": the `meta` device gives names, shapes and parameter counts for free.

## Rebuild checklist
- [ ] name every config field and the step it comes from
- [ ] write the parameter-count formula from the config
- [ ] compute KV bytes/token from the config
- [ ] map `q_proj ... down_proj` to our class attributes
- [ ] explain the meta device and why `lm_head` appears in the state dict but costs no memory

## The new problem (cliffhanger)
We have a faithful decoder and we can read any Llama-family config. But real models differ from ours by a handful of switches (GQA vs MLA, dense vs MoE, local vs global layers, hybrids), and new ones appear every month. How do you place a new announcement on one map? Next: [Step 10 — The architecture landscape 2026](../10_architecture_landscape_2026/WALKTHROUGH.md). (Generation is still one token per forward pass and memory-bound; the trick that fixes that, speculative decoding, is a serving optimisation and now lives in [module 21 step 13](../../21_llm_inference_engineering/13_speculative_decoding/).)

> **Say it aloud (interview box):** "I read a Llama-family config as: hidden size, layers, query and KV heads (GQA), intermediate size (SwiGLU), RoPE theta and scaling, vocabulary and tying. From those I can compute parameters (embedding + layers x (attention + 3 d ff + norms)) and KV bytes per token. The tensor names `q_proj ... down_proj` map one-to-one to the classes of a Llama implementation, which I verified by loading my own weights into Hugging Face's Llama and matching logits."
