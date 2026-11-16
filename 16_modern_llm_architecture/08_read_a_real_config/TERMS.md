# Terms — Read a real config

Folder: `16_modern_llm_architecture/08_read_a_real_config` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [16 README](../README.md)

**Terms in this folder:** [config.json](#configjson) · [AutoConfig](#autoconfig) · [hidden_size / intermediate_size](#hidden_size--intermediate_size) · [num_attention_heads / num_key_value_heads](#num_attention_heads--num_key_value_heads) · [rope_theta / rope_parameters](#rope_theta--rope_parameters) · [tie_word_embeddings](#tie_word_embeddings) · [state_dict](#state_dict) · [q_proj, k_proj, v_proj, o_proj](#q_proj-k_proj-v_proj-o_proj) · [gate_proj, up_proj, down_proj](#gate_proj-up_proj-down_proj) · [input_layernorm and post_attention_layernorm](#input_layernorm-and-post_attention_layernorm) · [Meta device](#meta-device) · [Parameter count by component](#parameter-count-by-component) · [named_parameters vs state_dict](#named_parameters-vs-state_dict) · [load_state_dict(strict=False)](#load_state_dictstrictfalse) · [Non-embedding parameters](#non-embedding-parameters)

---

## config.json  `(concept)`
**1. Definition:** The file describing a model's architecture hyperparameters.
**2. In plain words:** The specification sheet.
**3. Example to quote:** > "Qwen2.5-0.5B: hidden_size 896, 24 layers, 14 heads, 2 KV heads."
**Seen in:** `main.py`, section 1

## AutoConfig  `(library)`
**1. Definition:** Hugging Face class that loads a config and returns the right config subclass.
**2. In plain words:** Read the sheet without building the model.
**3. Example to quote:** > "AutoConfig.from_pretrained('Qwen/Qwen2.5-0.5B')."
**Seen in:** `main.py`, section 1

## hidden_size / intermediate_size  `(syntax)`
**1. Definition:** d_model and the MLP hidden width.
**2. In plain words:** Width of the stream and of the MLP.
**3. Example to quote:** > "896 and 4,864."
**Seen in:** `main.py`, section 1

## num_attention_heads / num_key_value_heads  `(syntax)`
**1. Definition:** Query heads and K/V heads (GQA).
**2. In plain words:** 14 and 2: 7 query heads per KV head.
**3. Example to quote:** > "Qwen2.5-0.5B uses GQA."
**Seen in:** `main.py`, section 1

## rope_theta / rope_parameters  `(syntax)`
**1. Definition:** RoPE base frequency (stored in rope_parameters in transformers 5).
**2. In plain words:** The 10000 of step 02, here 1,000,000.
**3. Example to quote:** > "{'rope_theta': 1000000.0, 'rope_type': 'default'}"
**Seen in:** `main.py`, section 1

## tie_word_embeddings  `(syntax)`
**1. Definition:** True when lm_head shares the embedding matrix.
**2. In plain words:** One table, two jobs.
**3. Example to quote:** > "True for Qwen2.5-0.5B, false for Qwen3-8B."
**Seen in:** `main.py`, section 1

## state_dict  `(concept)`
**1. Definition:** An ordered mapping from tensor names to tensors.
**2. In plain words:** The list of all weights.
**3. Example to quote:** > "291 tensors for Qwen2.5-0.5B."
**Seen in:** `main.py`, section 2

## q_proj, k_proj, v_proj, o_proj  `(concept)`
**1. Definition:** The attention projections of a Llama-family model.
**2. In plain words:** wq, wk, wv, wo.
**3. Example to quote:** > "k_proj weight (128, 896) = 2 KV heads x 64."
**Seen in:** `main.py`, section 2

## gate_proj, up_proj, down_proj  `(concept)`
**1. Definition:** The three SwiGLU matrices.
**2. In plain words:** w1, w3, w2 in the original Llama code.
**3. Example to quote:** > "(4864, 896), (4864, 896), (896, 4864)."
**Seen in:** `main.py`, section 2

## input_layernorm and post_attention_layernorm  `(concept)`
**1. Definition:** The RMSNorm before attention and before the MLP.
**2. In plain words:** Despite the name they are RMSNorm in Llama and Qwen.
**3. Example to quote:** > "896 weights each."
**Seen in:** `main.py`, section 2

## Meta device  `(concept)`
**1. Definition:** torch.device('meta') creates tensors with shape and dtype but no storage.
**2. In plain words:** An empty skeleton.
**3. Example to quote:** > "Build a huge architecture on a laptop to read names and shapes."
**Seen in:** `main.py`, section 2

## Parameter count by component  `(concept)`
**1. Definition:** Sum of numel per group: embedding, attention, MLP, norms.
**2. In plain words:** Where the weights live.
**3. Example to quote:** > "MLP 63.5%, embedding 27.6%, attention 8.9%."
**Seen in:** `main.py`, section 3

## named_parameters vs state_dict  `(library)`
**1. Definition:** named_parameters yields unique parameters (tied weights once); state_dict lists every key including aliases.
**2. In plain words:** Why lm_head appears in one but costs nothing.
**3. Example to quote:** > "lm_head.weight in state_dict: True."
**Seen in:** `main.py`, sections 2-3

## load_state_dict(strict=False)  `(library)`
**1. Definition:** Copies matching keys and ignores missing or extra ones.
**2. In plain words:** Silent if names do not match.
**3. Example to quote:** > "A wrong rename produces a large logit difference, not an error."
**Seen in:** `main.py`, section 5

## Non-embedding parameters  `(concept)`
**1. Definition:** Total minus the embedding table.
**2. In plain words:** Quoted on model cards for small models.
**3. Example to quote:** > "357,898,112 for Qwen2.5-0.5B."
**Seen in:** `main.py`, section 3

