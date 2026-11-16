"""Exercise Q4 solution: the MSE-optimal clip factor is < 1 for heavy-tailed data at 4 bits, and closer to 1 at 8 bits."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import quantlib as ql

rng = np.random.default_rng(0)
data = {"gaussian": rng.normal(0, 1, (256, 256)), "student-t (df=3)": rng.standard_t(3, (256, 256))}
res = {}
for name, x in data.items():
    for bits in (8, 4):
        c, best, at1 = ql.mse_optimal_clip(x.astype(np.float32), bits, True, "tensor")
        res[(name, bits)] = c
        print(f"{name:<18} int{bits}: best clip {c:.2f}  MSE {best:.2e}  vs absmax {at1:.2e}  ({at1 / best:.1f}x)")
assert res[("student-t (df=3)", 4)] < 1.0
assert res[("student-t (df=3)", 4)] <= res[("student-t (df=3)", 8)] + 1e-9   # fewer bits -> clip harder
print("OK")
