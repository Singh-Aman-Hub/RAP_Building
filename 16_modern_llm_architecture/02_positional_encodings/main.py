"""
GOAL:  Implement sinusoidal, learned-absolute, RoPE and ALiBi position schemes; verify numerically that RoPE
       scores depend only on RELATIVE distance; show context extension (position interpolation, NTK-aware, YaRN idea).

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 16_modern_llm_architecture/02_positional_encodings/main.py
EXPECTED OUTPUT (shortened):
    RoPE: score(q@5, k@2) = ... score(q@105, k@102) = ...  (equal)
    ...
NEEDS: torch only.  ~1 s.
"""
import math

import torch

torch.manual_seed(0)
D = 8                                                   # head dimension (tiny, so we can print things)


# ---- 1. Sinusoidal absolute encoding (Vaswani 2017): added to the token embedding ------------------------
def sinusoidal(n_pos, d):
    pos = torch.arange(n_pos).float()[:, None]                         # (n_pos, 1)
    freq = 10000 ** (-torch.arange(0, d, 2).float() / d)               # (d/2,) wavelengths grow geometrically
    pe = torch.zeros(n_pos, d)
    pe[:, 0::2] = torch.sin(pos * freq)                                # even dims: sin
    pe[:, 1::2] = torch.cos(pos * freq)                                # odd dims: cos
    return pe                                                          # (n_pos, d)


pe = sinusoidal(16, D)
print("sinusoidal shape", tuple(pe.shape), "| row0:", pe[0].numpy().round(2), "| row1:", pe[1].numpy().round(2))

# ---- 2. Learned absolute (GPT-2): a table nn.Embedding(max_len, d); positions past max_len do not exist ---
wpe = torch.nn.Embedding(1024, 768)                                    # gpt2 shape: 1024 positions
print("learned table", tuple(wpe.weight.shape), "= %.2fM params" % (wpe.weight.numel() / 1e6))
try:
    wpe(torch.tensor([1024]))                                          # position 1024 was never trained: no row
except IndexError as e:
    print("position 1024 ->", type(e).__name__)


# ---- 3. RoPE: rotate each (even, odd) pair of q and k by an angle proportional to the position ------------
def rope_freqs(d, base=10000.0):
    return base ** (-torch.arange(0, d, 2).float() / d)                # (d/2,) rotation speed per pair


def apply_rope(x, positions, base=10000.0):
    """x: (seq, d); positions: (seq,) floats (floats so we can interpolate later)."""
    ang = positions[:, None] * rope_freqs(x.shape[-1], base)[None, :]  # (seq, d/2) angle = pos * freq
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[:, 0::2], x[:, 1::2]                                    # the two halves of each pair
    out = torch.empty_like(x)
    out[:, 0::2] = x1 * cos - x2 * sin                                 # 2-D rotation matrix applied to the pair
    out[:, 1::2] = x1 * sin + x2 * cos
    return out


q, k = torch.randn(1, D), torch.randn(1, D)                            # one query vector and one key vector
print("\nRoPE relative-position test: same q, k, only absolute positions change")
for pq, pk in [(5, 2), (105, 102), (1005, 1002), (5, 3)]:
    rq = apply_rope(q, torch.tensor([float(pq)]))
    rk = apply_rope(k, torch.tensor([float(pk)]))
    print(f"  q@{pq:<5} k@{pk:<5} distance {pq - pk}  score = {(rq @ rk.T).item():.6f}")
print("  norm of q before/after rotation:", round(q.norm().item(), 4), round(apply_rope(q, torch.tensor([77.0])).norm().item(), 4))

# ---- 4. ALiBi: no vectors at all, subtract slope * distance from the attention score -----------------------
def alibi_bias(n, n_heads):
    slopes = 2 ** (-8 * torch.arange(1, n_heads + 1) / n_heads)        # geometric slopes, one per head
    dist = (torch.arange(n)[:, None] - torch.arange(n)[None, :]).clamp(min=0)   # (n, n) i - j for past tokens
    return -slopes[:, None, None] * dist                               # (heads, n, n) penalty grows with distance


bias = alibi_bias(6, 4)
print("\nALiBi slopes:", (2 ** (-8 * torch.arange(1, 5) / 4)).numpy().round(4), "| head0 bias row 5:", bias[0, 5].numpy().round(3))

# ---- 5. Context extension: what happens at positions never seen in training ---------------------------------
TRAIN_LEN, TARGET_LEN = 4096, 16384
scale = TARGET_LEN / TRAIN_LEN                                         # 4x longer
freqs = rope_freqs(64)                                                 # a realistic head_dim of 64
wavelen = 2 * math.pi / freqs                                          # tokens needed for one full turn
print(f"\nhead_dim 64: wavelengths from {wavelen[0]:.1f} to {wavelen[-1]:,.0f} tokens; train length {TRAIN_LEN}")
print("pairs whose wavelength exceeds the train length (never completed a turn in training):",
      int((wavelen > TRAIN_LEN).sum()), "of", len(wavelen))

# 5a. Position interpolation (PI): squeeze positions, pos' = pos / scale. Everything stays inside the trained range.
print("PI: position 16000 behaves like", 16000 / scale)

# 5b. NTK-aware: enlarge the base so low frequencies stretch a lot and high frequencies barely change
head_dim = 64
ntk_base = 10000.0 * scale ** (head_dim / (head_dim - 2))
f_new = rope_freqs(head_dim, ntk_base)
print(f"NTK-aware base: 10000 -> {ntk_base:,.0f}")
print(f"  freq ratio new/old for the fastest pair {f_new[0] / freqs[0]:.3f}, for the slowest pair {f_new[-1] / freqs[-1]:.3f} (PI would give {1 / scale:.3f} for all)")

# 5c. YaRN idea: per-pair blend. High-frequency pairs untouched, low-frequency pairs interpolated, ramp between.
def yarn_freqs(d, scale, train_len, alpha=1.0, beta=32.0):
    f = rope_freqs(d)
    rotations = train_len / (2 * math.pi / f)                         # full turns inside the training window
    ramp = ((rotations - alpha) / (beta - alpha)).clamp(0, 1)         # 0 = interpolate fully, 1 = leave alone
    return f / scale * (1 - ramp) + f * ramp, ramp


f_yarn, ramp = yarn_freqs(64, scale, TRAIN_LEN)
print("YaRN ramp (1 = untouched):", ramp[::4].numpy().round(2))
print("YaRN attention temperature factor 0.1*ln(scale)+1 =", round(0.1 * math.log(scale) + 1, 4))

# ---- 6. Which positions can the 4 methods "see"? A summary table ---------------------------------------------------
print("\nmethod        params       extrapolates?  where used")
for row in [("sinusoidal", "0", "poorly", "original Transformer"),
            ("learned abs", "n_pos*d", "no (hard limit)", "GPT-2, BERT"),
            ("RoPE", "0", "needs scaling", "Llama, Qwen, Mistral, Gemma, DeepSeek"),
            ("ALiBi", "0", "decent", "BLOOM, MPT")]:
    print(f"{row[0]:<14}{row[1]:<13}{row[2]:<17}{row[3]}")

# TRY IT: change base=10000 to 1_000_000 in rope_freqs and print the longest wavelength (Llama 3 uses 500000).
# BREAK IT: set D = 7 (odd head dimension). sinusoidal() fails first: the odd columns (3) cannot hold the 4 cos values -> RuntimeError.
