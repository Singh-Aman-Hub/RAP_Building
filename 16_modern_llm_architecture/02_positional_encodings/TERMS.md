# Terms — Positional encodings

Folder: `16_modern_llm_architecture/02_positional_encodings` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [16 README](../README.md)

**Terms in this folder:** [Positional encoding](#positional-encoding) · [Sinusoidal encoding](#sinusoidal-encoding) · [Learned absolute position embedding](#learned-absolute-position-embedding) · [RoPE (rotary position embedding)](#rope-rotary-position-embedding) · [Relative position property](#relative-position-property) · [rope_theta (RoPE base)](#rope_theta-rope-base) · [Wavelength of a RoPE pair](#wavelength-of-a-rope-pair) · [ALiBi](#alibi) · [Position interpolation (PI)](#position-interpolation-pi) · [NTK-aware scaling](#ntk-aware-scaling) · [YaRN](#yarn) · [NoPE](#nope) · [torch.nn.Embedding](#torchnnembedding)

---

## Positional encoding  `(concept)`
**1. Definition:** Information added to or mixed into attention so that the model can tell token order; attention alone is permutation-invariant.
**2. In plain words:** Without it 'dog bites man' equals 'man bites dog'.
**3. Example to quote:** > "Self-attention has no sense of order unless position is injected."
**Seen in:** `main.py`, all sections

## Sinusoidal encoding  `(concept)`
**1. Definition:** A fixed table PE[pos, 2i] = sin(pos / 10000^(2i/d)), PE[pos, 2i+1] = cos(...), added to the embeddings.
**2. In plain words:** Clock hands at different speeds identify each position.
**3. Example to quote:** > "Sinusoidal encodings have no parameters."
**Seen in:** `main.py`, `sinusoidal`

## Learned absolute position embedding  `(concept)`
**1. Definition:** A trainable table nn.Embedding(max_len, d) with one vector per position (GPT-2: 1,024 x 768).
**2. In plain words:** A label per seat; there is no seat 1,025.
**3. Example to quote:** > "GPT-2 cannot read past position 1,023: the lookup raises IndexError."
**Seen in:** `main.py`, section 2

## RoPE (rotary position embedding)  `(concept)`
**1. Definition:** Rotate each (even, odd) pair of the query and key by an angle pos x theta_i; the dot product then depends only on the relative position.
**2. In plain words:** Turn each clock hand by the token's position; the angle between two hands is their distance.
**3. Example to quote:** > "RoPE rotates q and k, never v, and has no parameters."
**Seen in:** `main.py`, `apply_rope`

## Relative position property  `(concept)`
**1. Definition:** For rotated vectors, score(q at m, k at n) = f(q, k, m - n); shifting both positions by the same amount leaves the score unchanged.
**2. In plain words:** Verified: distance 3 gives -0.960123 at positions (5,2), (105,102) and (1005,1002).
**3. Example to quote:** > "RoPE turns absolute rotations into relative attention."
**Seen in:** `main.py`, section 3

## rope_theta (RoPE base)  `(concept)`
**1. Definition:** The constant 10000 in theta_i = base^(-2i/d); larger bases make slow pairs slower and support longer contexts.
**2. In plain words:** Llama 3 uses 500,000, Qwen 1,000,000.
**3. Example to quote:** > "A bigger rope_theta stretches the longest wavelength: 6.3 to 4,080,185 tokens at base 1e6."
**Seen in:** `main.py`, `rope_freqs`

## Wavelength of a RoPE pair  `(concept)`
**1. Definition:** 2 pi / theta_i, the number of tokens for one full rotation of that pair.
**2. In plain words:** Fast pairs mark nearby order; slow pairs mark long range.
**3. Example to quote:** > "9 of 32 pairs have wavelengths above the 4,096 training length."
**Seen in:** `main.py`, section 5

## ALiBi  `(concept)`
**1. Definition:** Attention with Linear Biases: subtract slope_h x distance from each score; no position vectors.
**2. In plain words:** Distant tokens are penalised linearly; each head has its own strictness.
**3. Example to quote:** > "ALiBi extrapolates to longer lengths but is rarely used in 2025-26 models."
**Seen in:** `main.py`, `alibi_bias`

## Position interpolation (PI)  `(concept)`
**1. Definition:** Context extension by dividing positions by the scale factor so they stay inside the trained range.
**2. In plain words:** Squeeze the longer ruler into the shorter one.
**3. Example to quote:** > "PI loses resolution between neighbouring tokens."
**Seen in:** `main.py`, section 5a

## NTK-aware scaling  `(concept)`
**1. Definition:** Context extension by raising the RoPE base to base x scale^(d/(d-2)), stretching slow frequencies and leaving fast ones.
**2. In plain words:** Stretch only the slow clock hands.
**3. Example to quote:** > "NTK-aware base for 4x on head_dim 64 is 41,829."
**Seen in:** `main.py`, section 5b

## YaRN  `(concept)`
**1. Definition:** Context extension with a per-frequency ramp between no change and interpolation, plus an attention-logit temperature 0.1 ln(s) + 1.
**2. In plain words:** Fast hands untouched, slow hands squeezed, middle blended.
**3. Example to quote:** > "gpt-oss and DeepSeek-V3 configs use YaRN scaling."
**Seen in:** `main.py`, section 5c

## NoPE  `(concept)`
**1. Definition:** A layer or model that uses no positional encoding; order still leaks through the causal mask.
**2. In plain words:** Some layers deliberately skip RoPE so they extrapolate better.
**3. Example to quote:** > "Llama 4 uses a NoPE layer every fourth layer."
**Seen in:** `WALKTHROUGH.md`, theory

## torch.nn.Embedding  `(library)`
**1. Definition:** A trainable lookup table mapping integer ids to vectors.
**2. In plain words:** A table with one row per id.
**3. Example to quote:** > "nn.Embedding(1024, 768) holds 786,432 numbers."
**Seen in:** `main.py`, section 2

