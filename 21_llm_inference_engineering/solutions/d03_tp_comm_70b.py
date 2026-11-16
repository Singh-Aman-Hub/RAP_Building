"""Exercise D3: TP communication per token for a 70B-class model (80 layers, hidden 8192, fp16) at TP=8, and link time."""


def tp_bytes(layers, hidden, n, bytes_per_value=2):
    return layers * 2 * hidden * bytes_per_value * 2 * (n - 1) / n     # 2 all-reduces/layer, ring factor 2(N-1)/N


b = tp_bytes(80, 8192, 8)
print(f"D3: {b / 1e6:.2f} MB per token per GPU in {80 * 2} all-reduces of {8192 * 2 / 1e3:.0f} KB")
assert abs(b - 80 * 2 * 8192 * 2 * 2 * 7 / 8) < 1e-6
for name, bw in (("NVLink 450 GB/s", 450e9), ("100GbE 12 GB/s", 12e9)):
    print(f"    bandwidth-time over {name}: {b / bw * 1e6:.0f} us (+160 x per-collective latency)")
assert b / 12e9 > 30 * b / 450e9                                       # ethernet is >30x slower on bandwidth alone
