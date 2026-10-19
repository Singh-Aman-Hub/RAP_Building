"""
GOAL:   Load a pretrained language model, count its parameters, estimate its memory,
        and understand what is inside it.

RUN:    from this folder:  python main.py
        from repo root:    .venv/bin/python 04_llm_fundamentals/06_load_pretrained_model/main.py
EXPECTED OUTPUT (short): ~81.9 M parameters for distilgpt2, ~124 M for gpt2; ~0.33 GB in float32;
        the layer names of the model; embeddings vs the 6 blocks from step 05; a 7B-model memory table.
NEEDS:  torch + transformers. First run downloads distilgpt2 (~330 MB) and gpt2 (~500 MB).

"Pretrained" = someone already trained the weights on huge amounts of text.
Loading = reading those saved numbers into the right architecture.  You download
two things: the CONFIG (architecture sizes) and the WEIGHTS (the learned numbers).
"""

import torch
from transformers import AutoConfig, AutoModelForCausalLM  # Auto* picks the right class for the model name

# ---------------------------------------------------------------------------
# 1. Config: the architecture "blueprint" (tiny file, no weights)
# ---------------------------------------------------------------------------
cfg = AutoConfig.from_pretrained("distilgpt2")
print("layers (n_layer):         ", cfg.n_layer)          # how many transformer blocks stacked
print("hidden size (n_embd):     ", cfg.n_embd)           # numbers per token vector
print("attention heads (n_head): ", cfg.n_head)
print("context window (n_positions):", cfg.n_positions)   # max tokens the model can see at once
print("vocab size:               ", cfg.vocab_size)

# ---------------------------------------------------------------------------
# 2. Load the weights.  AutoModelForCausalLM = "a transformer with a next-token head".
# ---------------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
model.eval()   # eval mode: switches off dropout (random neuron dropping used only during training)

# ---------------------------------------------------------------------------
# 3. Count parameters = total number of learned numbers
# ---------------------------------------------------------------------------
def count_params(m):
    # m.parameters() yields every weight tensor; .numel() = how many numbers inside it
    return sum(p.numel() for p in m.parameters())   # generator expression inside sum()

n = count_params(model)
print(f"\nparameters: {n:,}  (= {n/1e6:.1f} million)")

# ---------------------------------------------------------------------------
# 4. Memory: parameters x bytes per number.  float32 = 4 bytes, float16 = 2, int8 = 1.
# ---------------------------------------------------------------------------
first_param = next(model.parameters())
print("dtype of weights:", first_param.dtype, "-> bytes each:", first_param.element_size())
for name, nbytes in [("float32", 4), ("float16/bfloat16", 2), ("int8", 1), ("int4", 0.5)]:
    print(f"  {name:18} ~ {n * nbytes / 1e9:.2f} GB")
print("(real memory use is higher: activations, KV cache and framework overhead are extra)")

# ---------------------------------------------------------------------------
# 5. Look inside: parameters grouped by top-level block name
# ---------------------------------------------------------------------------
print("\nfirst 8 parameter names and shapes:")
for i, (name, p) in enumerate(model.named_parameters()):   # enumerate gives a counter i
    if i >= 8:
        break
    print(f"  {name:40} {tuple(p.shape)}")   # shapes: wte (vocab, d_model), wpe (n_positions, d_model), c_attn (d_model, 3*d_model) ...

# Where do the 81.9 M live?  Compare with the block formula from step 05: 12*d^2 + 13*d per block.
emb_params = model.transformer.wte.weight.numel() + model.transformer.wpe.weight.numel()   # (50257*768) + (1024*768)
block_params = count_params(model.transformer.h[0])                                        # one block, shape-free count
print(f"\nembeddings (wte+wpe): {emb_params:,}  |  one block: {block_params:,}  |  6 blocks: {6 * block_params:,}")
print(f"formula 12*d^2+13*d with d={cfg.n_embd}: {12 * cfg.n_embd**2 + 13 * cfg.n_embd:,}  (matches one block)")
print(f"share: embeddings {emb_params / n:.0%}, blocks {6 * block_params / n:.0%}, final LayerNorm the rest")

# Memory arithmetic for a model you will meet in interviews: memory = params x bytes
seven_b = 7e9
print("\n7B model, weights only:")
for name, nbytes in [("float32", 4), ("float16", 2), ("int8", 1), ("int4", 0.5)]:
    print(f"  {name:8} {seven_b * nbytes / 1e9:5.1f} GB")

# ---------------------------------------------------------------------------
# 6. Same architecture family, bigger: compare with gpt2 (config only - no big download)
# ---------------------------------------------------------------------------
cfg2 = AutoConfig.from_pretrained("gpt2")
print(f"\ndistilgpt2 layers={cfg.n_layer}  vs  gpt2 layers={cfg2.n_layer}  (same hidden size {cfg2.n_embd})")

# TRY IT: load "gpt2" instead of "distilgpt2" and compare parameter counts.
# TRY IT: after loading, run model.half() then re-print memory (float16 halves the size).
# BREAK IT: from_pretrained("distilgpt2", torch_dtype="banana") -> read the error about the dtype.
# BREAK IT: turn off wifi and load a model you never downloaded -> OSError; note the cache folder it mentions (~/.cache/huggingface).
