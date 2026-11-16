"""Exercise D6: smallest ZeRO stage that trains a 13B model (mixed precision Adam, 16 B/param, activations ignored)
on 16 GPUs with 40 GB each (weights + grads + optimizer must fit in ~32 GB, leaving 8 GB for activations)."""
P, N, BUDGET = 13e9, 16, 32e9


def per_gpu(stage):
    w, g, o = 2 * P, 2 * P, 12 * P
    if stage >= 1:
        o /= N
    if stage >= 2:
        g /= N
    if stage >= 3:
        w /= N
    return w + g + o


for s in range(4):
    print(f"  ZeRO-{s}: {per_gpu(s) / 1e9:6.1f} GB per GPU {'fits' if per_gpu(s) <= BUDGET else 'does NOT fit'}")
best = next(s for s in range(4) if per_gpu(s) <= BUDGET)
print(f"D6 ok: smallest stage that fits = ZeRO-{best}")
assert best == 3 and per_gpu(2) > BUDGET
