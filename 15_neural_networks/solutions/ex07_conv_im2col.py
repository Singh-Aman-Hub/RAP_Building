"""
GOAL:  Exercise 7 solution. Convolution as a single matrix multiplication (im2col) in numpy, matched to
       torch conv2d; output-size formula; parameter counts conv vs dense.

RUN (from this folder):  python ex07_conv_im2col.py
RUN (from repo root):    .venv/bin/python 15_neural_networks/solutions/ex07_conv_im2col.py
EXPECTED OUTPUT (shortened):
    max |im2col - torch| = ...
    output size ...
NEEDS: numpy, torch (CPU). < 1 s.
"""
import numpy as np
import torch
import torch.nn.functional as F


def out_size(n, k, s=1, p=0):
    return (n + 2 * p - k) // s + 1


def conv2d_im2col(x, w, stride=1, pad=0):
    """x: (C, H, W), w: (F, C, k, k) -> (F, H_out, W_out). Cross-correlation (kernel NOT flipped), like torch."""
    C, H, W = x.shape
    Fo, _, k, _ = w.shape
    x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    Ho, Wo = out_size(H, k, stride, pad), out_size(W, k, stride, pad)
    cols = np.empty((C * k * k, Ho * Wo))               # every output position becomes one column
    idx = 0
    for i in range(Ho):
        for j in range(Wo):
            patch = x[:, i * stride:i * stride + k, j * stride:j * stride + k]
            cols[:, idx] = patch.ravel()
            idx += 1
    return (w.reshape(Fo, -1) @ cols).reshape(Fo, Ho, Wo)   # one GEMM does all positions and filters


rng = np.random.default_rng(0)
x = rng.normal(size=(3, 9, 9))
w = rng.normal(size=(4, 3, 3, 3))
for stride, pad in [(1, 0), (2, 1), (1, 1)]:
    mine = conv2d_im2col(x, w, stride, pad)
    ref = F.conv2d(torch.tensor(x)[None], torch.tensor(w), stride=stride, padding=pad)[0].numpy()
    err = np.abs(mine - ref).max()
    print(f"stride {stride} pad {pad}: output {mine.shape}, formula {out_size(9, 3, stride, pad)}, max |im2col - torch| = {err:.2e}")
    assert err < 1e-12

conv_params = 64 * (3 * 3 * 3) + 64
dense_params = (224 * 224 * 3) * (224 * 224 * 64) + 224 * 224 * 64
print(f"64 filters 3x3 on RGB: {conv_params:,} parameters; a dense layer mapping 224x224x3 to 224x224x64: {dense_params:,}")
