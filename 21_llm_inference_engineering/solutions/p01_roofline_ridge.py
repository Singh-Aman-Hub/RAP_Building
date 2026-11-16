"""Exercise P1 solution: ridge points and the batch size where decode stops being memory-bound, per weight dtype."""
GPUS = {"A100 80GB": (312e12, 2.039e12), "H100 SXM": (989e12, 3.35e12)}   # dense bf16 FLOP/s, bytes/s (spec sheets)
BYTES = {"fp16": 2.0, "int8": 1.0, "int4": 0.5}
for name, (flops, bw) in GPUS.items():
    ridge = flops / bw
    print(f"{name}: ridge = {ridge:.0f} FLOP/byte")
    for dt, b in BYTES.items():
        # decode intensity (weights only) = 2*P*B / (P*b) = 2*B/b  ->  B* = ridge * b / 2
        b_star = ridge * b / 2
        print(f"   {dt}: decode compute-bound only above batch ~{b_star:.0f}")
assert round(312e12 / 2.039e12) == 153
assert abs(GPUS["H100 SXM"][0] / GPUS["H100 SXM"][1] * 2.0 / 2 - 295.2) < 1       # fp16: B* = ridge
print("OK: fp16 B* equals the ridge; int4 weights make the crossover 4x smaller (fewer bytes per FLOP of work).")
