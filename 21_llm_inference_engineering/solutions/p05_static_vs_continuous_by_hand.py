"""Exercise P5 solution: 4 requests, 2 slots, output lengths 2, 8, 2, 2 (all arrive at t=0; 1 decode step = 1 unit, prefill ignored)."""
OUT = [2, 8, 2, 2]
SLOTS = 2


def static(out):
    t, finish = 0, [0] * len(out)
    for i in range(0, len(out), SLOTS):
        batch = range(i, min(i + SLOTS, len(out)))
        steps = max(out[j] for j in batch)                      # slots held until the longest finishes
        for j in batch:
            finish[j] = t + steps                               # static batching returns the whole batch together, at the END
        t += steps
    return finish, t


def continuous(out):
    waiting, running, remaining, t, finish = list(range(len(out))), [], list(out), 0, [0] * len(out)
    while waiting or running:
        while waiting and len(running) < SLOTS:
            running.append(waiting.pop(0))
        t += 1
        for j in list(running):
            remaining[j] -= 1
            if remaining[j] == 0:
                finish[j] = t
                running.remove(j)                               # slot freed immediately
    return finish, t


fs, ts = static(OUT)
fc, tc = continuous(OUT)
print("static    finish times", fs, "total", ts)
print("continuous finish times", fc, "total", tc)
assert ts == 8 + 2 and tc == 8                                  # static: 8 (longest of batch 1) + 2; continuous: slot 2 serves the short ones
assert sum(fc) / 4 < sum(fs) / 4 or tc < ts
print("OK: continuous finishes everything in", tc, "units vs", ts, "(static waits for the 8-token request before starting request 3)")
