"""
GOAL:  Exercise 5 solution. Predict then measure the activation standard deviation through 30 ReLU layers
       for init variances 1/n (LeCun), 2/(n_in+n_out) (Xavier) and 2/n (He).
       Derivation: Var(y) = n_in Var(w) E[x^2]; ReLU halves the second moment => He needs Var(w) = 2/n_in.

RUN (from this folder):  python ex05_init_variance.py
RUN (from repo root):    .venv/bin/python 15_neural_networks/solutions/ex05_init_variance.py
EXPECTED OUTPUT (shortened):
    init      predicted per-layer factor   measured std after 30 layers
    ...
NEEDS: numpy only. < 2 s.
"""
import numpy as np

rng = np.random.default_rng(0)
n, depth, batch = 512, 30, 256
inits = {"LeCun 1/n": 1 / n, "Xavier 2/(2n)": 2 / (n + n), "He 2/n": 2 / n}

print(f"{'init':<16}{'predicted rms factor/layer':>28}{'measured rms after 30 layers':>32}")
for name, var in inits.items():
    x = rng.normal(size=(batch, n))
    for _ in range(depth):
        W = rng.normal(size=(n, n)) * np.sqrt(var)
        x = np.maximum(x @ W, 0)                  # ReLU
    # second moment per layer scales by n * var / 2 (ReLU halves it); rms by the square root of that
    pred_factor = np.sqrt(n * var / 2)
    print(f"{name:<16}{pred_factor:>28.3f}{np.sqrt((x ** 2).mean()):>32.3e}   (predicted {pred_factor ** depth:.2e})")
print("(He ends at 0.70 rather than 1.0: finite width makes the per-layer factor fluctuate and 30 layers compound the noise.)")
print("He keeps the signal order 1; the others shrink by about 0.71 per layer (0.71^30 = 2e-5): vanishing forward signal.")
