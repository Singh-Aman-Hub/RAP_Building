"""Exercise Q3 solution: pack signed 4-bit codes into bytes, also for an ODD number of values (pad one nibble), and verify."""
import numpy as np

def pack(q):
    q = np.asarray(q, dtype=np.int16).ravel()
    n = len(q)
    if n % 2:
        q = np.append(q, 0)                                # pad so that the pairs line up
    u = (q + 8).astype(np.uint8)                           # codes -8..7 -> 0..15
    return (u[0::2] | (u[1::2] << 4)).astype(np.uint8), n

def unpack(p, n):
    lo, hi = (p & 15).astype(np.int16) - 8, (p >> 4).astype(np.int16) - 8
    out = np.empty(len(p) * 2, dtype=np.int16)
    out[0::2], out[1::2] = lo, hi
    return out[:n]                                         # drop the padding

rng = np.random.default_rng(0)
for n in (1, 2, 7, 1000, 1001):
    q = rng.integers(-8, 8, n)
    p, m = pack(q)
    assert np.array_equal(unpack(p, m), q)
    print(f"n={n:5d} -> {len(p):4d} bytes, round trip ok")
print("OK")
