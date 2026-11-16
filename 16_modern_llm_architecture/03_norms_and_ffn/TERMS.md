# Terms — Norms and FFN

Folder: `16_modern_llm_architecture/03_norms_and_ffn` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [16 README](../README.md)

**Terms in this folder:** [LayerNorm](#layernorm) · [RMSNorm](#rmsnorm) · [Epsilon in a norm](#epsilon-in-a-norm) · [BatchNorm (contrast)](#batchnorm-contrast) · [Pre-norm vs post-norm](#pre-norm-vs-post-norm) · [Residual stream](#residual-stream) · [Feed-forward network (FFN, MLP)](#feed-forward-network-ffn-mlp) · [GELU](#gelu) · [SiLU (swish)](#silu-swish) · [SwiGLU](#swiglu) · [GLU variants (GeGLU, ReGLU)](#glu-variants-geglu-reglu) · [Intermediate size (8/3 rule)](#intermediate-size-83-rule) · [QK-norm](#qk-norm) · [Attention entropy and logit growth](#attention-entropy-and-logit-growth)

---

## LayerNorm  `(concept)`
**1. Definition:** Normalise each token vector to zero mean and unit variance over the feature axis, then scale and shift with learned gamma and beta.
**2. In plain words:** Centre and rescale every token.
**3. Example to quote:** > "LayerNorm has 2d parameters."
**Seen in:** `main.py`, section 1

## RMSNorm  `(concept)`
**1. Definition:** Divide each token vector by sqrt(mean(x^2) + eps) and multiply by a learned gain; no mean subtraction and no bias.
**2. In plain words:** Only fix the length of the vector.
**3. Example to quote:** > "RMSNorm has d parameters and is the norm of Llama, Qwen, Gemma, DeepSeek."
**Seen in:** `main.py`, `RMSNorm`

## Epsilon in a norm  `(concept)`
**1. Definition:** A small constant added under the square root to avoid dividing by zero.
**2. In plain words:** Safety margin.
**3. Example to quote:** > "Qwen2.5 uses rms_norm_eps 1e-6; Llama 1e-5."
**Seen in:** `main.py`, `RMSNorm.__init__`

## BatchNorm (contrast)  `(concept)`
**1. Definition:** Normalises each feature across the batch dimension, using batch statistics.
**2. In plain words:** Not used in LLMs because batch statistics are unreliable for variable-length text.
**3. Example to quote:** > "LayerNorm normalises over features; BatchNorm over the batch."
**Seen in:** `WALKTHROUGH.md`

## Pre-norm vs post-norm  `(concept)`
**1. Definition:** Pre-norm: x + f(norm(x)). Post-norm: norm(x + f(x)).
**2. In plain words:** Pre-norm leaves the residual stream untouched, a clean highway for gradients.
**3. Example to quote:** > "All modern LLMs are pre-norm and add a final norm before the head."
**Seen in:** `main.py`, section 3

## Residual stream  `(concept)`
**1. Definition:** The running vector x that each sub-layer reads (after a norm) and adds an update to.
**2. In plain words:** A conveyor belt that every layer adds to.
**3. Example to quote:** > "In pre-norm models the residual stream grows with depth (RMS 3.08 after 24 layers in our test)."
**Seen in:** `main.py`, `run_stack`

## Feed-forward network (FFN, MLP)  `(concept)`
**1. Definition:** The per-token network in each block: up-project, non-linearity, down-project.
**2. In plain words:** The 'thinking' half of a block; holds about two thirds of the parameters.
**3. Example to quote:** > "FFN, MLP and feed-forward are the same thing."
**Seen in:** `main.py`, `GeluMLP`

## GELU  `(concept)`
**1. Definition:** A smooth activation x * Phi(x) used in GPT-2 and BERT.
**2. In plain words:** A soft version of ReLU.
**3. Example to quote:** > "GPT-2's MLP is Linear(d, 4d), GELU, Linear(4d, d)."
**Seen in:** `main.py`, `GeluMLP`

## SiLU (swish)  `(concept)`
**1. Definition:** Activation x * sigmoid(x).
**2. In plain words:** A smooth gate that lets positive values through.
**3. Example to quote:** > "SwiGLU uses SiLU on the gate branch."
**Seen in:** `main.py`, `SwiGLU`

## SwiGLU  `(concept)`
**1. Definition:** A gated MLP: down(silu(gate(x)) * up(x)) with three matrices and no bias.
**2. In plain words:** One path says what, the other says how much.
**3. Example to quote:** > "SwiGLU with hidden 8/3 d has the same parameters as a 4d GELU MLP (4,718,592 vs 4,722,432 at d=768)."
**Seen in:** `main.py`, `SwiGLU`

## GLU variants (GeGLU, ReGLU)  `(concept)`
**1. Definition:** Gated linear units with different activations on the gate: GELU (GeGLU, Gemma), ReLU (ReGLU), SiLU (SwiGLU).
**2. In plain words:** Same idea, different gate shape.
**3. Example to quote:** > "GeGLU is used by Gemma."
**Seen in:** `WALKTHROUGH.md`

## Intermediate size (8/3 rule)  `(concept)`
**1. Definition:** The MLP hidden width; for SwiGLU about 8d/3 so that 3 matrices cost the same as two 4d matrices.
**2. In plain words:** Llama-2-7B: 11,008 for d=4,096 (ratio 2.69).
**3. Example to quote:** > "A SwiGLU at hidden 4d would cost 1.5x the parameters."
**Seen in:** `main.py`, section 2

## QK-norm  `(concept)`
**1. Definition:** RMSNorm applied to each head's query and key before the dot product to bound attention logits.
**2. In plain words:** Stops attention scores exploding late in training.
**3. Example to quote:** > "Max logit fell from 194.8 to 2.9 in our test."
**Seen in:** `main.py`, section 4

## Attention entropy and logit growth  `(concept)`
**1. Definition:** Entropy of the softmax distribution; very large logits make it nearly one-hot (entropy near 0).
**2. In plain words:** A token that looks at exactly one other token cannot learn to look elsewhere.
**3. Example to quote:** > "Entropy 0.006 without QK-norm vs 1.473 with it."
**Seen in:** `main.py`, section 4

