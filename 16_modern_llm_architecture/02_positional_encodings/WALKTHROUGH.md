# Step 02 — Positional encodings: from learned tables to RoPE

## The story so far
[Step 01](../01_bpe_tokenizer_from_scratch/WALKTHROUGH.md) turned text into token ids. A token id becomes a vector through an embedding table (see [04 / step 03](../../04_llm_fundamentals/03_embeddings/WALKTHROUGH.md)). But attention itself is permutation-blind.

## The pain point
Self-attention computes `softmax(q·k)` over a set of tokens. Shuffle the tokens and every score is the same, only reordered. "dog bites man" and "man bites dog" would be indistinguishable. The model needs *position* information. GPT-2's solution (a learned vector per position, up to 1,024) has two problems: the model has no vectors for position 1,024 or more, and the position is *added to the input once*, so the model must learn to extract relative distance from absolute labels.

## The spark
Instead of labelling tokens with their position, **rotate** each query and key vector by an angle proportional to its position (Rotary Position Embedding, RoPE, Su et al. 2021). A rotation by angle `m·θ` for the query at position m and `n·θ` for the key at position n gives a dot product that depends only on `(m - n)·θ`. Relative position falls out of the maths with no extra parameters.

## Goal
Implement sinusoidal, learned, RoPE and ALiBi; check numerically that a RoPE score is unchanged when both positions shift by 1000; then work through context extension (position interpolation, NTK-aware, YaRN) with real numbers.

## The theory (read this before the interview)
**Absolute vs relative.** Absolute schemes (sinusoidal, learned) give each position an identity and add it to the token embedding at the input. Relative schemes (RoPE, ALiBi, T5 bias) change the attention *score* so that it depends on the distance between query and key.

**Sinusoidal** (Vaswani 2017): `PE[pos, 2i] = sin(pos / 10000^(2i/d))`, `PE[pos, 2i+1] = cos(...)`. No parameters. Wavelengths grow geometrically from 2π to 10000·2π. Added to the embeddings.

**Learned absolute** (GPT-2, BERT): `nn.Embedding(max_len, d)`. For GPT-2 that is 1,024 x 768 = 786k parameters. Hard limit: position 1,024 raises an `IndexError` (see the run).

**RoPE.** Split each head vector of size `d` into `d/2` pairs. Pair `i` is a 2-D point; rotate it by angle `pos · θ_i` with `θ_i = base^(-2i/d)` (base = 10,000 originally):
```
[x1']   [cos(pos·θ)  -sin(pos·θ)] [x1]
[x2'] = [sin(pos·θ)   cos(pos·θ)] [x2]
q_m · k_n  (after rotating q by m and k by n)  =  function of (q, k, m - n) only
```
Fast pairs (small i, θ near 1) rotate quickly and encode local order; slow pairs (θ tiny) barely rotate over thousands of tokens and encode long-range structure. Properties: rotation preserves the vector norm; it is applied to **q and k only**, never to v; it is applied inside every attention layer; it has no learned parameters. The `base` is a hyperparameter: Llama 2 10,000; Llama 3 500,000; Qwen2.5 and Qwen3 1,000,000; gpt-oss 150,000. A larger base makes the slowest pairs even slower, so very long contexts remain distinguishable.

**Two layouts of the same thing.** The original paper rotates *interleaved* pairs `(x0,x1), (x2,x3), ...` (this step). Hugging Face's Llama/Qwen code rotates the *first half against the second half* (`rotate_half`, used in step 07). They are the same operation up to a fixed permutation of the dimensions, but the weights must match the layout. Converting a checkpoint between them means permuting the rows of `q_proj` and `k_proj`.

**ALiBi** (Press et al. 2021): no position vectors at all. Add `-slope_h · (i - j)` to the score of query i and key j; each head h has a fixed slope (`2^(-8h/H)`). Distant tokens are penalised linearly. It extrapolates to longer lengths better than sinusoidal/learned, but most 2024-2026 open models chose RoPE plus scaling, so ALiBi is mostly historical (BLOOM, MPT).

**Context extension.** A model trained at length L_train has never seen rotation angles beyond `L_train·θ`. Naively running at 4x length gives positions out of distribution and quality collapses. Fixes (all for RoPE):
| Method | Idea | Effect on frequencies | Needs fine-tuning? |
|---|---|---|---|
| Position interpolation (PI) | divide every position by the scale s | all frequencies divided by s; high-frequency pairs lose resolution (nearby tokens look closer) | yes, short |
| NTK-aware | raise the base: `base' = base · s^(d/(d-2))` | fast pairs nearly unchanged, slow pairs divided by about s | works somewhat without |
| YaRN | per-pair ramp: do not touch high-frequency pairs, interpolate low-frequency pairs, blend between; plus scale the attention logits by `0.1 ln(s) + 1` | best of both | little (hundreds of steps) |
| Llama 3 scaling | a similar per-frequency rule (factor 8: 8k to 128k) | piecewise | trained with it |
| NoPE layers / chunked attention | some layers (Llama 4) use no positional encoding at all | n/a | trained that way |
Real configs: gpt-oss-120b uses YaRN factor 32 over an original 4,096 context to reach 131,072; DeepSeek-V3 uses YaRN factor 40 over 4,096 to reach 163,840; Llama 3.1 uses its own `llama3` rope scaling with factor 8.0 over 8,192. Qwen3 docs describe enabling YaRN at inference for lengths beyond native context.

**Why "context length" is not free.** Even with perfect position handling, attention cost is quadratic (step 05) and the KV cache grows linearly (step 04). Position scaling only makes long inputs *legal*; quality at long range still depends on long-context training data.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| RoPE | rotary embedding, rotary position embedding, `rotary_emb` | Yes | rotates q and k by position-dependent angle |
| Positional encoding vs embedding | position encoding, position embedding | Loosely | "encoding" is often a fixed formula, "embedding" a learned table; people mix them |
| Context window | context length, `max_position_embeddings`, `n_ctx`, sequence length | Mostly | the config field is what the position scheme/training covers; actual quality can degrade earlier |
| Position interpolation vs NTK-aware vs YaRN | RoPE scaling, long-context extension, `rope_scaling` | Related | three ways to change the frequencies; YaRN also rescales the logits |
| `rope_theta` | RoPE base, base frequency, θ | Yes | the 10000 in `10000^(-2i/d)` |
| Absolute vs relative position | | No | absolute adds a label per token; relative changes the score by distance |
| NoPE | no positional encoding | Yes | layers with no position signal (causal masking still leaks order) |

## Run it
```bash
cd 16_modern_llm_architecture/02_positional_encodings
python main.py       # ~1 s, torch only
```

## Code, step by step
### 1. Imports, seed, tiny head dimension  (`main.py` lines 12–17)
```python
import math

import torch

torch.manual_seed(0)
D = 8                                                   # head dimension (tiny, so we can print things)
```
- **In simple words:** set up torch, make results repeatable, and use vectors of 8 numbers so we can print them.
- **Exactly:** `torch.manual_seed(0)` fixes the random q and k used later. `D = 8` is the head dimension (real models use 64 to 256).
- **Why it is here:** reproducible numbers in the walkthrough; a tiny size keeps the printed rows readable.
### 2. Sinusoidal table builder  (`main.py` lines 21–27)
```python
def sinusoidal(n_pos, d):
    pos = torch.arange(n_pos).float()[:, None]                         # (n_pos, 1)
    freq = 10000 ** (-torch.arange(0, d, 2).float() / d)               # (d/2,) wavelengths grow geometrically
    pe = torch.zeros(n_pos, d)
    pe[:, 0::2] = torch.sin(pos * freq)                                # even dims: sin
    pe[:, 1::2] = torch.cos(pos * freq)                                # odd dims: cos
    return pe                                                          # (n_pos, d)
```
- **In simple words:** build a table where each position gets a fingerprint of sines and cosines at different speeds, like clock hands (seconds, minutes, hours) that together identify the time.
- **Exactly:** `freq = 10000 ** (-arange(0, d, 2)/d)` has shape `(d/2,)`; `pos * freq` broadcasts `(n_pos, 1) * (d/2,)` to `(n_pos, d/2)`. Even columns get `sin`, odd columns `cos`. Output shape `(n_pos, d)`.
- **Why it is here:** the original, parameter-free way to inject position; it is the baseline for comparison.
### 3. Look at two rows  (`main.py` lines 30–31)
```python
pe = sinusoidal(16, D)
print("sinusoidal shape", tuple(pe.shape), "| row0:", pe[0].numpy().round(2), "| row1:", pe[1].numpy().round(2))
```
- **In simple words:** position 0 is all `[0, 1, 0, 1...]` (sin 0 = 0, cos 0 = 1); position 1 differs most in the first (fast) pair.
- **Exactly:** run output `row1: [0.84 0.54 0.1 1. 0.01 1. 0. 1.]`: pair 0 has rotated by 1 radian (sin 1 = 0.84), pair 3 by 0.001 radian.
- **Why it is here:** shows the "fast pairs on the left, slow pairs on the right" structure RoPE also uses.
### 4. Learned absolute table and its hard limit  (`main.py` lines 34–39)
```python
wpe = torch.nn.Embedding(1024, 768)                                    # gpt2 shape: 1024 positions
print("learned table", tuple(wpe.weight.shape), "= %.2fM params" % (wpe.weight.numel() / 1e6))
try:
    wpe(torch.tensor([1024]))                                          # position 1024 was never trained: no row
except IndexError as e:
    print("position 1024 ->", type(e).__name__)
```
- **In simple words:** GPT-2 simply stores one trainable vector per position; there is no vector for position 1024.
- **Exactly:** `nn.Embedding(1024, 768)` has weight `(1024, 768)` = 0.79M parameters. Looking up index 1024 raises `IndexError`.
- **Why it is here:** the concrete reason GPT-2 cannot read more than 1,024 tokens, and the contrast with RoPE (zero parameters, no hard limit).
### 5. RoPE frequencies  (`main.py` lines 43–44)
```python
def rope_freqs(d, base=10000.0):
    return base ** (-torch.arange(0, d, 2).float() / d)                # (d/2,) rotation speed per pair
```
- **In simple words:** one rotation speed per pair of numbers, fast to slow.
- **Exactly:** `base ** (-arange(0, d, 2) / d)`, shape `(d/2,)`, from 1.0 down to about `1/base`.
- **Why it is here:** `base` is the main RoPE hyperparameter (and what NTK scaling changes).
### 6. Rotate each (even, odd) pair by position times frequency  (`main.py` lines 47–55)
```python
def apply_rope(x, positions, base=10000.0):
    """x: (seq, d); positions: (seq,) floats (floats so we can interpolate later)."""
    ang = positions[:, None] * rope_freqs(x.shape[-1], base)[None, :]  # (seq, d/2) angle = pos * freq
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[:, 0::2], x[:, 1::2]                                    # the two halves of each pair
    out = torch.empty_like(x)
    out[:, 0::2] = x1 * cos - x2 * sin                                 # 2-D rotation matrix applied to the pair
    out[:, 1::2] = x1 * sin + x2 * cos
    return out
```
- **In simple words:** for token at position `p`, turn every pair of its numbers like a clock hand by `p * speed`.
- **Exactly:** `ang = positions[:, None] * freqs[None, :]` has shape `(seq, d/2)`. For each pair `(x1, x2)`: `x1' = x1 cos - x2 sin`, `x2' = x1 sin + x2 cos`, which is multiplication by the 2-D rotation matrix. Positions are floats so interpolation can use fractional positions.
- **Why it is here:** the core of RoPE. Applying it to q and k (not v) makes the score `q·k` depend on the relative offset.
### 7. Relative-position test and norm preservation  (`main.py` lines 58–64)
```python
q, k = torch.randn(1, D), torch.randn(1, D)                            # one query vector and one key vector
print("\nRoPE relative-position test: same q, k, only absolute positions change")
for pq, pk in [(5, 2), (105, 102), (1005, 1002), (5, 3)]:
    rq = apply_rope(q, torch.tensor([float(pq)]))
    rk = apply_rope(k, torch.tensor([float(pk)]))
    print(f"  q@{pq:<5} k@{pk:<5} distance {pq - pk}  score = {(rq @ rk.T).item():.6f}")
print("  norm of q before/after rotation:", round(q.norm().item(), 4), round(apply_rope(q, torch.tensor([77.0])).norm().item(), 4))
```
- **In simple words:** put the same two vectors at positions (5, 2), (105, 102), (1005, 1002): the score does not change because the distance is 3 each time. At distance 2 the score is different.
- **Exactly:** real run: `-0.960123`, `-0.960123`, `-0.960124` (the last differs at the 6th decimal because float32 angles at position 1005 lose precision) and `-1.101672` for distance 2. The norm of q is 3.2599 before and after rotation: rotations preserve length.
- **Why it is here:** this is the proof of the key property. It is also why RoPE works with a KV cache: cached keys keep their rotation and new queries just use their own position.
### 8. ALiBi bias matrix  (`main.py` lines 67–70)
```python
def alibi_bias(n, n_heads):
    slopes = 2 ** (-8 * torch.arange(1, n_heads + 1) / n_heads)        # geometric slopes, one per head
    dist = (torch.arange(n)[:, None] - torch.arange(n)[None, :]).clamp(min=0)   # (n, n) i - j for past tokens
    return -slopes[:, None, None] * dist                               # (heads, n, n) penalty grows with distance
```
- **In simple words:** no vectors; just subtract a penalty from the attention score that grows with distance, each head with its own strictness.
- **Exactly:** `slopes = 2^(-8h/H)` for h = 1..H; `dist[i, j] = max(i - j, 0)`; result shape `(heads, n, n)` added to scores before softmax.
- **Why it is here:** the main alternative to RoPE; shows that "position" can live entirely in the score.
### 9. Inspect ALiBi  (`main.py` lines 73–74)
```python
bias = alibi_bias(6, 4)
print("\nALiBi slopes:", (2 ** (-8 * torch.arange(1, 5) / 4)).numpy().round(4), "| head0 bias row 5:", bias[0, 5].numpy().round(3))
```
- **Exactly:** 4 heads give slopes `[0.25, 0.0625, 0.0156, 0.0039]`. Head 0 (steepest) at the last query: `[-1.25 -1. -0.75 -0.5 -0.25 -0.]`, a linear penalty of 0.25 per token of distance.
- **In simple words:** the first head cares mostly about the last few tokens; the last head barely penalises distance.
- **Why it is here:** concrete picture of "different heads see different ranges".
### 10. Wavelengths: which RoPE pairs never finished a turn in training  (`main.py` lines 77–83)
```python
TRAIN_LEN, TARGET_LEN = 4096, 16384
scale = TARGET_LEN / TRAIN_LEN                                         # 4x longer
freqs = rope_freqs(64)                                                 # a realistic head_dim of 64
wavelen = 2 * math.pi / freqs                                          # tokens needed for one full turn
print(f"\nhead_dim 64: wavelengths from {wavelen[0]:.1f} to {wavelen[-1]:,.0f} tokens; train length {TRAIN_LEN}")
print("pairs whose wavelength exceeds the train length (never completed a turn in training):",
      int((wavelen > TRAIN_LEN).sum()), "of", len(wavelen))
```
- **In simple words:** each pair is a clock hand; a slow hand may not have completed one lap within the training length, so the model never saw its whole cycle.
- **Exactly:** with head_dim 64 the wavelengths `2π/θ` run from 6.3 tokens to 47,117 tokens. 9 of the 32 pairs have a wavelength above the 4,096 training length. Stretching to 16,384 puts those pairs at angles never seen in training.
- **Why it is here:** this is the exact reason naive context extension fails and what scaling methods repair.
### 11. Position interpolation  (`main.py` lines 85–86)
```python
# 5a. Position interpolation (PI): squeeze positions, pos' = pos / scale. Everything stays inside the trained range.
print("PI: position 16000 behaves like", 16000 / scale)
```
- **In simple words:** squeeze the new, longer ruler into the old one: position 16,000 is treated as 4,000.
- **Exactly:** `pos / scale` with scale 4. All angles stay in the trained range. The cost: tokens that were 1 apart are now 0.25 apart, so the fast pairs lose the ability to tell neighbours apart.
- **Why it is here:** the simplest extension method (Chen et al. 2023); usually needs a short fine-tune.
### 12. NTK-aware base change  (`main.py` lines 88–93)
```python
# 5b. NTK-aware: enlarge the base so low frequencies stretch a lot and high frequencies barely change
head_dim = 64
ntk_base = 10000.0 * scale ** (head_dim / (head_dim - 2))
f_new = rope_freqs(head_dim, ntk_base)
print(f"NTK-aware base: 10000 -> {ntk_base:,.0f}")
print(f"  freq ratio new/old for the fastest pair {f_new[0] / freqs[0]:.3f}, for the slowest pair {f_new[-1] / freqs[-1]:.3f} (PI would give {1 / scale:.3f} for all)")
```
- **In simple words:** instead of squeezing everything, stretch only the slow hands and leave the fast hands.
- **Exactly:** `base' = 10000 · 4^(64/62) = 41,829`. The fastest pair's frequency ratio new/old is 1.000, the slowest 0.250, while PI divides all pairs by 4 (0.250).
- **Why it is here:** keeps local resolution and extends global range; the idea behind many "just change `rope_theta`" tricks.
### 13. YaRN: a per-pair ramp plus a logit temperature  (`main.py` lines 95–105)
```python
# 5c. YaRN idea: per-pair blend. High-frequency pairs untouched, low-frequency pairs interpolated, ramp between.
def yarn_freqs(d, scale, train_len, alpha=1.0, beta=32.0):
    f = rope_freqs(d)
    rotations = train_len / (2 * math.pi / f)                         # full turns inside the training window
    ramp = ((rotations - alpha) / (beta - alpha)).clamp(0, 1)         # 0 = interpolate fully, 1 = leave alone
    return f / scale * (1 - ramp) + f * ramp, ramp


f_yarn, ramp = yarn_freqs(64, scale, TRAIN_LEN)
print("YaRN ramp (1 = untouched):", ramp[::4].numpy().round(2))
print("YaRN attention temperature factor 0.1*ln(scale)+1 =", round(0.1 * math.log(scale) + 1, 4))
```
- **In simple words:** decide per clock hand: fast hands untouched, slow hands squeezed, medium hands blended; and sharpen attention slightly to compensate for the squeezed positions.
- **Exactly:** `rotations = train_len / wavelength` is how many full turns a pair makes inside the training window. `ramp` is 0 below `alpha=1` turn (interpolate fully) and 1 above `beta=32` turns (untouched). The printed ramp for every 4th pair: `[1, 1, 1, 0.63, 0.18, 0.03, 0, 0]`. The factor `0.1·ln(4) + 1 = 1.1386` scales the logits (YaRN's temperature).
- **Why it is here:** this is what gpt-oss and DeepSeek-V3 use in their configs (`rope_scaling.type = yarn`).
### 14. Summary table  (`main.py` lines 108–113)
```python
print("\nmethod        params       extrapolates?  where used")
for row in [("sinusoidal", "0", "poorly", "original Transformer"),
            ("learned abs", "n_pos*d", "no (hard limit)", "GPT-2, BERT"),
            ("RoPE", "0", "needs scaling", "Llama, Qwen, Mistral, Gemma, DeepSeek"),
            ("ALiBi", "0", "decent", "BLOOM, MPT")]:
    print(f"{row[0]:<14}{row[1]:<13}{row[2]:<17}{row[3]}")
```
- **In simple words:** a recap of the four schemes and who uses them.
- **Exactly:** a tuple loop; the `extrapolates?` column is the commonly reported behaviour, not something this script measures.
- **Why it is here:** an interview-ready one-glance comparison.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `x[:, 0::2]` | every second column starting at 0 (even indices) | `[a b c d]` -> `[a c]` |
| `pos[:, None]` | add an axis so it broadcasts against a row | `(n,)` -> `(n, 1)` |
| `torch.empty_like(x)` | uninitialised tensor with x's shape, filled right after | |
| `.clamp(min=0)` / `.clamp(0, 1)` | limit values to a range | the YaRN ramp |
| `(a > b).sum()` | count True values | pairs above the train length |
| `try: ... except IndexError` | catch the error from an out-of-range lookup | position 1024 |
| `f"{x:,.0f}"` | thousands separator, no decimals | `47,117` |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `torch.arange` | torch | integer range | start, stop, step | tensor |
| `torch.sin / cos` | torch | elementwise trig | tensor | tensor |
| `torch.nn.Embedding(n, d)` | torch | trainable lookup table | rows, width | module |
| `math.pi, math.log` | built-in | constants / log | | float |
| `tensor.item()` | torch | 0-d tensor to Python float | | float |

## Execution trace (real run)
```text
sinusoidal shape (16, 8) | row0: [0. 1. 0. 1. 0. 1. 0. 1.] | row1: [0.84 0.54 0.1  1.   0.01 1.   0.   1.  ]
learned table (1024, 768) = 0.79M params
position 1024 -> IndexError

RoPE relative-position test: same q, k, only absolute positions change
  q@5     k@2     distance 3  score = -0.960123
  q@105   k@102   distance 3  score = -0.960123
  q@1005  k@1002  distance 3  score = -0.960124
  q@5     k@3     distance 2  score = -1.101672
  norm of q before/after rotation: 3.2599 3.2599

ALiBi slopes: [0.25   0.0625 0.0156 0.0039] | head0 bias row 5: [-1.25 -1.   -0.75 -0.5  -0.25 -0.  ]

head_dim 64: wavelengths from 6.3 to 47,117 tokens; train length 4096
pairs whose wavelength exceeds the train length (never completed a turn in training): 9 of 32
PI: position 16000 behaves like 4000.0
NTK-aware base: 10000 -> 41,829
  freq ratio new/old for the fastest pair 1.000, for the slowest pair 0.250 (PI would give 0.250 for all)
YaRN ramp (1 = untouched): [1.   1.   1.   0.63 0.18 0.03 0.   0.  ]
YaRN attention temperature factor 0.1*ln(scale)+1 = 1.1386

method        params       extrapolates?  where used
sinusoidal    0            poorly           original Transformer
learned abs   n_pos*d      no (hard limit)  GPT-2, BERT
RoPE          0            needs scaling    Llama, Qwen, Mistral, Gemma, DeepSeek
ALiBi         0            decent           BLOOM, MPT
```
Reading it: the RoPE scores are equal to 5 decimals for distance 3 at three different absolute offsets and different for distance 2. That is the whole claim of "relative position". The YaRN ramp starts at 1 (fastest pairs untouched) and falls to 0 (slowest pairs fully interpolated).

## Experiments (change ONE thing)
Every row below was run on a temporary copy of `main.py` (the real file is untouched); only the named line changed.

| Change | What you will see | Why |
|---|---|---|
| `base=10000.0` -> `1_000_000.0` in `rope_freqs` and `apply_rope` (measured) | longest wavelength grows from 47,117 to 4,080,185 tokens; pairs that never finish a turn within 4096 tokens go from 9 to 16 of 32; the relative-position test still passes (-0.996479 at three offsets) | A bigger base slows the low-frequency pairs, so far-apart positions stay distinguishable. This is why Llama 3 (base 500,000) and long-context models raise it. Relativity is untouched. |
| `TARGET_LEN = 8192` instead of 16384 (measured) | `PI: position 16000 behaves like 8000.0`; NTK base 10000 -> 20,452; slowest-pair ratio 0.500; YaRN temperature factor 1.0693 (was 1.1386) | Scale is now 2x, so every scaling method stretches by half as much. A milder extension distorts the trained behaviour less. |
| YaRN `beta=32.0` -> `8.0` (measured) | ramp `[1. 1. 1. 1. 0.79 0.15 0. 0.]` (was `[1. 1. 1. 0.63 0.18 0.03 0. 0.]`) | The ramp reaches 1 once a pair completes `beta` turns inside the training window. A smaller beta (8 instead of 32) means more pairs qualify as "fast enough to leave alone", so the untouched region grows by one more sampled pair. The ramp is the knob that decides how many pairs YaRN trusts as they are. |
| `D = 7` (measured) | `RuntimeError` inside `sinusoidal` (see drill below) | Pairs need an even number of dimensions: sin goes to even columns, cos to odd columns. |
| rotate only `q`, leave `k` unrotated (measured) | the four scores become `7.88, 5.49, 3.67, 7.88`: distance 3 no longer gives equal scores, and it now depends on the absolute position of q | RoPE only works because *both* sides are rotated: the angle difference `(pq - pk) * freq` is what survives in the dot product. |

## Debugging practice
**BREAK IT drill 1: rotate only the query.** In the test loop:

```diff
-    rk = apply_rope(k, torch.tensor([float(pk)]))
+    rk = k
```

Symptom (real run), no exception, just numbers that should have matched:

```text
  q@5     k@2     distance 3  score = 7.882545
  q@105   k@102   distance 3  score = 5.491624
  q@1005  k@1002  distance 3  score = 3.666344
  q@5     k@3     distance 2  score = 7.882545
```

Root cause: with only q rotated, the score depends on `pq` alone (note that q@5 gives the same score for k@2 and k@3). The relative-distance property needs both rotations so that the angles subtract. Fix: apply RoPE to q and to k (never to v).

**BREAK IT drill 2: a sign error in the rotation.** In `apply_rope`:

```diff
-    out[:, 1::2] = x1 * sin + x2 * cos
+    out[:, 1::2] = x1 * sin - x2 * cos
```

Symptom (real run): scores jump around, `1.838104, -3.605584, 1.219970` for the same distance 3, and the norm line shows `3.2599 2.6848` instead of `3.2599 3.2599`.

Root cause: the two lines together must form a 2-D rotation matrix `[[cos, -sin], [sin, cos]]`. With the wrong sign it is a reflection mixed with scaling in places, so lengths change and relative position is lost. Fix: restore `+`. Cheap guard: always assert that the norm is preserved after rotation.

## Common mistakes
- Rotating `v` as well as `q` and `k`. Only the score needs relative position; v carries content.
- Using an odd head dimension: RoPE and sinusoidal both need `d` even (pairs of dimensions).
- Applying a context-extension trick at inference only and expecting it to work without testing: PI, NTK and YaRN change the trained frequencies, so quality at short lengths can drop (YaRN's temperature factor and a short fine-tune exist for that reason).
- Mixing the two RoPE layouts. This file rotates interleaved (even, odd) pairs; Hugging Face Llama rotates the first half against the second half (`rotate_half`). The maths is equivalent but the weights are laid out for one layout only.
- Forgetting that learned absolute embeddings have a hard limit (`IndexError` at position 1024 for gpt2) while RoPE has no table, only a quality limit.
- Confusing "no parameters" with "free": RoPE adds no weights but it costs a rotation of q and k at every layer.

## Rebuild checklist
- [ ] `rope_freqs(d, base)` returning `base ** (-arange(0, d, 2) / d)`
- [ ] `apply_rope(x, positions)` with the 2-D rotation of even/odd pairs
- [ ] a test that `score(q@m, k@n)` depends only on `m - n`
- [ ] PI as `pos / scale`; NTK as `base * scale ** (d/(d-2))`
- [ ] say which of q, k, v is rotated (q and k)

## The new problem (cliffhanger)
Attention now knows order. But the vectors flowing through 32 to 100+ layers need to stay well-scaled, and the MLP inside each block is where most parameters live. GPT-2's LayerNorm and GELU MLP were replaced by RMSNorm and SwiGLU in every modern model. Next: [Step 03 — norms and FFN](../03_norms_and_ffn/WALKTHROUGH.md).

> **Say it aloud (interview box):** "RoPE rotates each pair of query and key dimensions by an angle proportional to the token position, so the dot product depends only on relative distance, with no parameters. To extend context you must change the frequencies: position interpolation divides positions, NTK-aware raises the base, and YaRN does a per-frequency blend plus an attention temperature. Learned absolute positions simply have no vector beyond the trained length."
