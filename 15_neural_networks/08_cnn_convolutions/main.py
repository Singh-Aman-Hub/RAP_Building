"""
GOAL:  Understand convolution by writing it yourself (numpy), check it against torch, learn the output-size and
       receptive-field formulas, then train a small CNN and an MLP on sklearn's 8x8 digits and compare accuracy,
       parameter count and robustness to a 1-pixel shift.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 15_neural_networks/08_cnn_convolutions/main.py
EXPECTED OUTPUT (shortened):
    my conv2d vs torch F.conv2d: max abs difference ~1e-7 (several stride/padding settings)
    receptive field of conv3-pool2-conv3-pool2: ...
    MLP params ... CNN params ...; test accuracy ...; accuracy after shifting images by 1 pixel ...
NEEDS: Python + numpy + torch + scikit-learn (load_digits, bundled with sklearn, no download). ~20 seconds on CPU.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# [1] Data: 1797 handwritten digits, each an 8x8 grayscale image with values 0..16
digits = load_digits()
images = digits.images.astype(np.float32) / 16.0                 # (1797, 8, 8) scaled to 0..1
labels = digits.target                                           # (1797,) ints 0..9
Xtr, Xte, ytr, yte = train_test_split(images, labels, test_size=0.25, random_state=0, stratify=labels)
print("train", Xtr.shape, "test", Xte.shape, "| one digit is a", labels[0], "image of shape", images[0].shape)


# [2] Convolution by hand: slide a small kernel over the image; each output = sum of (patch * kernel)
def conv2d_naive(img, kernel, stride=1, pad=0):
    """img: (H, W); kernel: (kh, kw). Like torch, this is cross-correlation (the kernel is NOT flipped)."""
    img = np.pad(img, pad)                                       # zero-padding on all four sides
    kh, kw = kernel.shape
    out_h = (img.shape[0] - kh) // stride + 1                    # output size formula
    out_w = (img.shape[1] - kw) // stride + 1
    out = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            patch = img[i * stride:i * stride + kh, j * stride:j * stride + kw]   # (kh, kw) window
            out[i, j] = (patch * kernel).sum()                   # one dot product per output position
    return out


vertical_edge = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)    # Sobel: responds to vertical edges
digit = images[0]                                                # a 0
for stride, pad in [(1, 0), (1, 1), (2, 0), (2, 1)]:
    mine = conv2d_naive(digit, vertical_edge, stride, pad)
    ref = F.conv2d(torch.tensor(digit)[None, None], torch.tensor(vertical_edge)[None, None], stride=stride, padding=pad)[0, 0].numpy()
    print(f"stride {stride} pad {pad}: output {mine.shape} (formula floor((8 + 2*{pad} - 3)/{stride}) + 1 = {(8 + 2 * pad - 3) // stride + 1}), "
          f"max |mine - torch| = {np.abs(mine - ref).max():.1e}")
print("edge map of digit 0 (stride 1, pad 1), rounded:\n", np.round(conv2d_naive(digit, vertical_edge, 1, 1), 1))


# [3] Pooling by hand: keep the maximum of each non-overlapping 2x2 block (halves the resolution)
def maxpool2x2(x):                                               # x: (H, W) with even H and W
    H, W = x.shape
    return x.reshape(H // 2, 2, W // 2, 2).max(axis=(1, 3))      # group into 2x2 blocks, take the max of each


pooled = maxpool2x2(conv2d_naive(digit, vertical_edge, 1, 1))
ref_pool = F.max_pool2d(torch.tensor(conv2d_naive(digit, vertical_edge, 1, 1))[None, None], 2)[0, 0].numpy()
print("pooled shape", pooled.shape, "| matches torch max_pool2d:", np.allclose(pooled, ref_pool))


# [4] Receptive field: which input pixels can influence one unit of layer L?
def receptive_field(layers):                                     # layers: list of (kernel, stride)
    rf, jump = 1, 1                                              # jump = distance between neighbouring units, in input pixels
    for k, s in layers:
        rf += (k - 1) * jump                                     # each layer widens the view by (k-1) * current jump
        jump *= s
    return rf


print("receptive field: conv3 ->", receptive_field([(3, 1)]),
      "| conv3-pool2-conv3 ->", receptive_field([(3, 1), (2, 2), (3, 1)]),
      "| conv3-pool2-conv3-pool2 ->", receptive_field([(3, 1), (2, 2), (3, 1), (2, 2)]), "pixels (our images are 8 wide)")


# [5] Two models: a fully connected net and a small CNN with a similar parameter budget
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
    def forward(self, x):                                        # x: (batch, 1, 8, 8)
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)   # (batch,1,8,8)  -> (batch,8,8,8); weights 8*1*3*3 + 8 bias
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # (batch,8,4,4)  -> (batch,16,4,4); weights 16*8*3*3 + 16 bias
        self.fc = nn.Linear(16 * 2 * 2, 10)                      # after two 2x2 pools: (batch,16,2,2) -> 64 -> 10 logits
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)               # (batch, 8, 4, 4)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)               # (batch, 16, 2, 2)
        return self.fc(x.flatten(1))                             # (batch, 10)


count = lambda m: sum(p.numel() for p in m.parameters())
print(f"\nMLP parameters: {count(MLP())} | CNN parameters: {count(CNN())}")


# [6] Train both with the standard loop (module 03) and evaluate on the clean test set and on 1-pixel-shifted images
def to_tensor(a):
    return torch.tensor(a)[:, None]                              # (N, 8, 8) -> (N, 1, 8, 8): add the channel axis


def train_eval(model, epochs=30, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    X, y = to_tensor(Xtr), torch.tensor(ytr)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):                           # mini-batches of 64
            idx = perm[i:i + 64]
            loss = F.cross_entropy(model(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        acc = lambda imgs: (model(to_tensor(imgs)).argmax(1) == torch.tensor(yte)).float().mean().item()
        shifted = np.roll(Xte, 1, axis=2)                        # move every image 1 pixel to the right (wraps around)
        return acc(Xte), acc(shifted), (model(X).argmax(1) == y).float().mean().item()


for name, make in (("MLP", MLP), ("CNN", CNN)):
    torch.manual_seed(0)
    test_acc, shift_acc, train_acc = train_eval(make())
    print(f"{name}: train acc {train_acc:.3f} | test acc {test_acc:.3f} | test acc with images shifted 1 pixel {shift_acc:.3f}")


# [7] Why convolutions scale: parameter count for one layer on a 224x224 RGB image
dense = 224 * 224 * 3 * 1000                                     # fully connected to 1000 units
conv = 64 * 3 * 3 * 3 + 64                                       # 64 filters of 3x3x3 (+ bias), shared across the whole image
print(f"\ndense layer 224x224x3 -> 1000 units: {dense:,} weights | conv layer with 64 filters 3x3: {conv:,} weights")

# TRY IT: change kernel_size to 5 (and padding to 2) in conv1 and compare parameter count and accuracy.
# BREAK IT: change self.fc to nn.Linear(16 * 4 * 4, 10) and read the shape error at the first forward pass.
