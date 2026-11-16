"""
GOAL:  Read a real model's config.json (Qwen2.5-0.5B) and map every field and every weight tensor (name + shape) to
       the concepts of steps 01-07; count parameters by component; then PROVE the mapping by loading our mini-Llama
       weights into Hugging Face's LlamaForCausalLM and getting identical logits.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 16_modern_llm_architecture/08_read_a_real_config/main.py
EXPECTED OUTPUT (shortened):
    hidden_size 896 ... num_key_value_heads 2 ...
    model.layers.0.self_attn.q_proj.weight (896, 896) ...
    total 494,032,768 ...
    max |logit diff| mini-Llama vs HF Llama: ~1e-7
NEEDS: torch, transformers; cached config of Qwen/Qwen2.5-0.5B (only the config, no weights are needed: we build
       the architecture on the "meta" device, which holds shapes but no data).  ~5 s.
"""
import importlib.util
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

torch.manual_seed(0)

# ---- 1. Config fields -> concepts ---------------------------------------------------------------------------------------
cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
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
head_dim = cfg.hidden_size // cfg.num_attention_heads
print(f"derived: head_dim = {cfg.hidden_size}/{cfg.num_attention_heads} = {head_dim}; group size = {cfg.num_attention_heads // cfg.num_key_value_heads} query heads per KV head")

# ---- 2. Build the architecture with no memory: the 'meta' device keeps shapes only -----------------------------------------
with torch.device("meta"):
    qwen = AutoModelForCausalLM.from_config(cfg)
sd = qwen.state_dict()
print(f"\n{type(qwen).__name__}: {len(sd)} tensors; first block (layer 0) and the non-layer tensors:")
for name, t in sd.items():
    if ".layers." not in name or ".layers.0." in name:
        print(f"  {name:<52}{tuple(t.shape)}")
print("lm_head.weight in state_dict?", "lm_head.weight" in sd, "| same storage as embed_tokens?",
      qwen.lm_head.weight is qwen.model.embed_tokens.weight)

# ---- 3. Parameter count by component ----------------------------------------------------------------------------------------------
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
for k, v in groups.items():
    print(f"{k:<34}{v:>14,}  {v / total:6.1%}")
print(f"{'TOTAL':<34}{total:>14,}")
non_emb = total - groups["embedding (tied with lm_head)"]
print(f"non-embedding params {non_emb:,} (the '0.36B' you see on the model card) | embedding share {groups['embedding (tied with lm_head)'] / total:.1%}")

# the same numbers from a formula, to prove we understand every field
d, L, h, kv, ff, V = cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.intermediate_size, cfg.vocab_size
attn = d * d + 2 * d * (kv * head_dim) + d * d + (d + 2 * kv * head_dim)        # q, k, v, o weights + q,k,v biases
formula = V * d + L * (attn + 3 * d * ff + 2 * d) + d
print("formula  V*d + L*(attn + 3*d*ff + 2*d) + d =", f"{formula:,}", "| matches:", formula == total)
print(f"KV cache: 2 * {L} layers * {kv} kv_heads * {head_dim} head_dim * 2 bytes (bf16) = {2 * L * kv * head_dim * 2} bytes/token")

# ---- 4. Contrast: GPT-2's tensor names (same ideas, older design) ---------------------------------------------------------------------
with torch.device("meta"):
    g2 = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained("gpt2"))
print("\nGPT-2 block 0 tensors (fused qkv, biases, LayerNorm, learned positions):")
for name, t in g2.state_dict().items():
    if ".h.0." in name or "wpe" in name or "wte" in name:
        print(f"  {name:<34}{tuple(t.shape)}")

# ---- 5. Proof: our mini-Llama (step 07) == Hugging Face Llama with the same weights -------------------------------------------------------------
spec = importlib.util.spec_from_file_location("mini", Path(__file__).resolve().parents[1] / "07_build_mini_llama" / "main.py")
mini = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mini)                                                   # imports the classes; training only runs under __main__
mc = mini.ModelConfig()
ours = mini.Transformer(mc).eval()
hf = LlamaForCausalLM(LlamaConfig(
    vocab_size=mc.vocab_size, hidden_size=mc.d_model, intermediate_size=mc.ffn_hidden, num_hidden_layers=mc.n_layers,
    num_attention_heads=mc.n_heads, num_key_value_heads=mc.n_kv_heads, max_position_embeddings=mc.max_seq,
    rms_norm_eps=mc.norm_eps, rope_theta=mc.rope_base, tie_word_embeddings=True, attention_bias=False, mlp_bias=False,
)).eval()
rename = {"embed.weight": "model.embed_tokens.weight", "final_norm.weight": "model.norm.weight"}
for i in range(mc.n_layers):
    for a, b in [("attn_norm", "input_layernorm"), ("mlp_norm", "post_attention_layernorm"),
                 ("attn.wq", "self_attn.q_proj"), ("attn.wk", "self_attn.k_proj"), ("attn.wv", "self_attn.v_proj"),
                 ("attn.wo", "self_attn.o_proj"), ("mlp.gate", "mlp.gate_proj"), ("mlp.up", "mlp.up_proj"), ("mlp.down", "mlp.down_proj")]:
        rename[f"blocks.{i}.{a}.weight"] = f"model.layers.{i}.{b}.weight"
our_sd = ours.state_dict()
hf.load_state_dict({rename[k]: v for k, v in our_sd.items() if k in rename}, strict=False)   # lm_head is tied, so absent
ids = torch.randint(0, 256, (2, 24))
with torch.no_grad():
    la, lb = ours(ids), hf(ids).logits
print(f"\nrenamed {len(rename)} tensors; mini-Llama vs HF LlamaForCausalLM: max |logit diff| = {(la - lb).abs().max().item():.2e}")
print("same architecture, same math:", torch.allclose(la, lb, atol=1e-4))

# TRY IT: change AutoConfig to "gpt2" or "distilgpt2" in section 1 and read the config fields that DO exist (n_embd, n_layer...).
# BREAK IT: delete `with torch.device("meta"):` for the 'qwen' build. It still works but allocates ~2 GB of fp32 weights:
#           the meta device is why a 70B config can be inspected on a laptop.
