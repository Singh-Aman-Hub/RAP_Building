# Terms — The 2026 architecture landscape

Folder: `16_modern_llm_architecture/10_architecture_landscape_2026` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [16 README](../README.md)

**Terms in this folder:** [Open-weight model](#open-weight-model) · [Active parameters (A22B)](#active-parameters-a22b) · [Fine-grained experts](#fine-grained-experts) · [Dense layers before MoE](#dense-layers-before-moe) · [KV bytes at context length](#kv-bytes-at-context-length) · [global_layers and local_window](#global_layers-and-local_window) · [iRoPE / NoPE layers (Llama 4)](#irope--nope-layers-llama-4) · [Reasoning model](#reasoning-model) · [Test-time compute](#test-time-compute) · [GRPO](#grpo) · [State-space model (Mamba)](#state-space-model-mamba) · [Gated DeltaNet / linear attention](#gated-deltanet--linear-attention) · [Hybrid attention model](#hybrid-attention-model) · [Long-context quality vs length](#long-context-quality-vs-length) · [Native quantised checkpoints (FP8, MXFP4, FP4)](#native-quantised-checkpoints-fp8-mxfp4-fp4)

---

## Open-weight model  `(concept)`
**1. Definition:** A model whose weights can be downloaded; the licence and the openness of data and code vary.
**2. In plain words:** Free to download is not always free to use.
**3. Example to quote:** > "Qwen3.5 is Apache-2.0; DeepSeek-V4 is MIT."
**Seen in:** `WALKTHROUGH.md`

## Active parameters (A22B)  `(concept)`
**1. Definition:** Parameters used per token in an MoE model, written after A in names like Qwen3-235B-A22B.
**2. In plain words:** The compute-relevant size.
**3. Example to quote:** > "9% of the total for Qwen3-235B."
**Seen in:** `main.py`, table

## Fine-grained experts  `(concept)`
**1. Definition:** Many small experts with high top-k (DeepSeek, Qwen3) rather than few large ones (Mixtral).
**2. In plain words:** More combinations of specialists.
**3. Example to quote:** > "256 routed experts choose 8."
**Seen in:** `WALKTHROUGH.md`

## Dense layers before MoE  `(concept)`
**1. Definition:** DeepSeek-V3 keeps its first 3 layers as dense MLPs, then MoE.
**2. In plain words:** Early layers share more.
**3. Example to quote:** > "Kimi K2 keeps 1."
**Seen in:** `WALKTHROUGH.md`

## KV bytes at context length  `(concept)`
**1. Definition:** Cache per token multiplied by tokens, modified by local layers and MLA.
**2. In plain words:** The capacity planning number.
**3. Example to quote:** > "Llama 3.1 8B: 17.2 GB at 128k."
**Seen in:** `main.py`, `kv_bytes`

## global_layers and local_window  `(syntax)`
**1. Definition:** Fields saying how many layers attend to all tokens and the window of the rest.
**2. In plain words:** Gemma 3: 10 global, window 1,024.
**3. Example to quote:** > "Used in kv_bytes."
**Seen in:** `main.py`

## iRoPE / NoPE layers (Llama 4)  `(concept)`
**1. Definition:** Interleaving RoPE layers with chunked attention and global layers without positional encoding.
**2. In plain words:** Local layers know order, global layers extrapolate.
**3. Example to quote:** > "12 of 48 layers in Scout."
**Seen in:** `WALKTHROUGH.md`

## Reasoning model  `(concept)`
**1. Definition:** A model post-trained with reinforcement learning to produce long chains of thought before answering.
**2. In plain words:** Same architecture, different training.
**3. Example to quote:** > "DeepSeek-R1 on the V3 architecture."
**Seen in:** `WALKTHROUGH.md`

## Test-time compute  `(concept)`
**1. Definition:** Spending more computation at inference (more generated tokens, sampling, search) to improve answers.
**2. In plain words:** Think longer, answer better.
**3. Example to quote:** > "Cost scales with output length."
**Seen in:** `WALKTHROUGH.md`

## GRPO  `(concept)`
**1. Definition:** Group Relative Policy Optimization: a critic-free RL method that normalises rewards within a group of sampled answers.
**2. In plain words:** Compare answers to each other.
**3. Example to quote:** > "Used for DeepSeek-R1; see module 17."
**Seen in:** `WALKTHROUGH.md`

## State-space model (Mamba)  `(concept)`
**1. Definition:** A recurrent layer with a fixed-size, input-dependent state replacing attention.
**2. In plain words:** A rolling summary instead of a growing notebook.
**3. Example to quote:** > "O(1) memory per token."
**Seen in:** `WALKTHROUGH.md`

## Gated DeltaNet / linear attention  `(concept)`
**1. Definition:** A linear-attention layer with a matrix-valued state updated by a gated delta rule.
**2. In plain words:** Fixed-size memory with selective overwriting.
**3. Example to quote:** > "Qwen3.5 uses 3 per 1 full-attention layer."
**Seen in:** `WALKTHROUGH.md`

## Hybrid attention model  `(concept)`
**1. Definition:** A model mixing linear/SSM layers with a few full-attention layers.
**2. In plain words:** Cheap layers plus a few exact-recall layers.
**3. Example to quote:** > "Jamba, Qwen3.5, Ling 2.5."
**Seen in:** `WALKTHROUGH.md`

## Long-context quality vs length  `(concept)`
**1. Definition:** A larger configured window does not mean reliable use of all of it.
**2. In plain words:** Check recall with needle tests.
**3. Example to quote:** > "DeepSeek-V4 1M, Llama 4 Scout 10M configured."
**Seen in:** `WALKTHROUGH.md`

## Native quantised checkpoints (FP8, MXFP4, FP4)  `(concept)`
**1. Definition:** Weights distributed in low precision.
**2. In plain words:** Half or quarter of bf16 memory.
**3. Example to quote:** > "gpt-oss MXFP4 fits one 80 GB GPU."
**Seen in:** `WALKTHROUGH.md`

