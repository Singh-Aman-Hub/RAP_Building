# Step 08 — Convolutional networks

## The story so far
[Step 07](../07_regularization/WALKTHROUGH.md) regularised a flat-input MLP by changing the loss, the optimiser and the data. The strongest regulariser is a good *architectural prior*: build what you know about the data into the network. For images, what we know is that neighbouring pixels belong together and that a pattern means the same wherever it appears.

## The pain point
A fully connected layer treats a 224x224 RGB image as 150,528 unrelated numbers: one layer to 1000 units needs 150,528,000 weights, it must learn 'an edge in the top left' separately from 'an edge in the bottom right', and shifting the picture by a pixel looks like a new input (our MLP drops from 97.3 to 48.0 percent when digits are shifted by one pixel).

## The spark
Reuse a small filter everywhere. A 3x3 kernel (9 weights) slides over the image and computes the same dot product at every position: local connectivity plus weight sharing. Stack layers and pool: early filters find edges, later ones combine them into strokes and parts. The same 1,792 weights that detect 64 patterns at one position detect them at all positions.

## Goal
Implement 2-D convolution and max-pooling in numpy and match torch exactly; use the output-size and receptive-field formulas; train an MLP (6,570 params) and a CNN (1,898 params) on 8x8 digits and compare test accuracy (0.973 vs 0.976) and robustness to a 1-pixel shift (0.480 vs 0.633).

## The theory (read this before the interview)
**Convolution as used in deep learning (technically cross-correlation).** For an input `x` (H x W) and a kernel `k` (kh x kw):
```
out[i, j] = sum_{a, b} x[i*s + a, j*s + b] * k[a, b]        (+ bias)
```
Mathematicians flip the kernel; deep-learning libraries do not (the network learns the flipped kernel anyway). A multi-channel layer has weights `(C_out, C_in, kh, kw)`: every output channel (filter) sums over all input channels.

**Output size.** `out = floor((n + 2*pad - k) / stride) + 1`. Checked in the run: `n=8, k=3`: pad 0 stride 1 gives 6; pad 1 stride 1 gives 8 ('same'); stride 2 gives 3 and 4. 'Same' padding for odd `k` is `pad = (k-1)/2`.

**Parameters.** `C_out * C_in * kh * kw + C_out` (bias). Independent of image size! That is weight sharing. Our CNN: `conv1 = 8*1*9 + 8 = 80`, `conv2 = 16*8*9 + 16 = 1168`, `fc = 64*10 + 10 = 650`: total 1,898. The MLP: `64*64+64 + 64*32+32 + 32*10+10 = 6,570`.

**Why it works (inductive biases).**
- *Locality*: a unit looks at a small neighbourhood.
- *Translation equivariance*: shift the input and the feature map shifts with it (`conv(shift(x)) = shift(conv(x))`, away from borders).
- *Translation invariance* (approximately) comes from pooling and the final classifier.
- *Hierarchy*: depth composes edges into textures into parts into objects.

**Pooling.** Max or average over small windows (2x2, stride 2) halves resolution, adds small shift tolerance and enlarges the receptive field. Modern nets often use strided convolution instead, and global average pooling before the classifier (no large `Flatten -> Linear` needed).

**Receptive field.** The input region that can influence one unit. Update rule per layer: `rf += (k - 1) * jump; jump *= stride`. Computed: conv3 sees 3 pixels, conv3-pool2-conv3 sees 8, conv3-pool2-conv3-pool2 sees 10. In our 8-pixel-wide images the second conv layer already covers the whole digit, which is why this tiny CNN works.

**1x1 convolutions** mix channels per pixel (a per-position `Linear`); **depthwise-separable convolutions** split spatial and channel mixing to save parameters (MobileNet); **dilated convolutions** enlarge the receptive field without more weights; **transposed convolutions** upsample.

**Architectures to know.** LeNet (1998, digits), AlexNet (2012, GPU, ReLU, dropout), VGG (stacks of 3x3), ResNet (2015, residual connections, [step 05](../05_initialization_and_normalization/WALKTHROUGH.md)), EfficientNet / ConvNeXt (modernised convnets). **Vision Transformers** (ViT) cut the image into patches and apply attention ([module 04](../../04_llm_fundamentals/README.md)); they have fewer built-in priors and need more data or pre-training, while convnets remain strong, efficient and common for small data, edge devices and many OCR/detection pipelines.

**Relevance to OCR and documents (module 18).** Classic OCR text detectors and recognisers (CRNN: convolutional features plus a recurrent or attention decoder) are built from the blocks here; the digits task is the 'hello world' of that pipeline. Real documents also need resizing, normalisation and augmentation (step 07).

**Practical points.** Input tensors are `(batch, channels, height, width)` in PyTorch (NCHW). `nn.Conv2d` expects the channel axis even for grayscale. Use BatchNorm after convs (step 05) in deeper nets. Augment with small shifts, rotations, crops: the CNN's built-in shift tolerance is partial and the measured shift-robustness gap here is only 15 points on a tiny image.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| convolution | cross-correlation, filter, `Conv2d`, 'conv layer' | Deep-learning 'convolution' is cross-correlation | Maths convolution flips the kernel; libraries do not; the learned kernel absorbs the flip |
| kernel | filter, mask, weights of a conv layer | Same | 'Filter' sometimes means the whole stack across input channels, 'kernel' the 2-D slice |
| feature map | activation map, output channel | Same | One feature map per filter: shape `(H_out, W_out)` |
| stride | step size of the sliding window | - | Stride 2 halves the output; not the same as pooling but has a similar effect |
| padding | 'same', 'valid', zero-padding | 'same' keeps size (stride 1); 'valid' means no padding | PyTorch `padding='same'` or an integer |
| pooling | max pool, average pool, subsampling, downsampling | Related | Pooling has no weights; strided convs learn how to downsample |
| receptive field | field of view, context window (of a conv unit) | Same concept | Not the same as an LLM 'context window', although the idea is analogous |
| weight sharing | parameter tying, translation equivariance | Cause and effect | Sharing is the mechanism; equivariance is the consequence |
| NCHW vs NHWC | channels-first vs channels-last | Layouts | PyTorch default NCHW; TensorFlow default NHWC; channels-last can be faster on some hardware |
| CNN vs ViT | convnet vs vision transformer | Different architectures | CNN: local filters with shared weights; ViT: patches with global attention |

## Run it
```bash
cd 15_neural_networks/08_cnn_convolutions
python main.py       # about 20 seconds on CPU (slower on a busy machine), torch + scikit-learn
```

## Code, step by step
Every block below is copied from `main.py` (line numbers are exact). Read the block, then the three bullets.

### 0. Docstring and imports  (`main.py` lines 1-21)
```python
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

```
- **In simple words:** Import numpy, torch and scikit-learn. The digits dataset ships inside scikit-learn, so no download is needed.
- **Exactly:** `load_digits` and `train_test_split` come from `sklearn`; `F` holds `conv2d`, `max_pool2d`, `relu`, `cross_entropy`.
- **Why it is here:** A real image dataset small enough to train in seconds on a CPU.

### 1. Data: 1797 handwritten digits, each an 8x8 grayscale image with values 0..16  (`main.py` lines 22-27)
```python
# [1] Data: 1797 handwritten digits, each an 8x8 grayscale image with values 0..16
digits = load_digits()
images = digits.images.astype(np.float32) / 16.0                 # (1797, 8, 8) scaled to 0..1
labels = digits.target                                           # (1797,) ints 0..9
Xtr, Xte, ytr, yte = train_test_split(images, labels, test_size=0.25, random_state=0, stratify=labels)
print("train", Xtr.shape, "test", Xte.shape, "| one digit is a", labels[0], "image of shape", images[0].shape)
```
- **In simple words:** Load 1,797 tiny handwritten digits and split them into a training set and a test set the model never sees.
- **Exactly:** `digits.images` has shape `(1797, 8, 8)` with integer values 0..16; dividing by 16 gives floats in 0..1. `train_test_split(..., test_size=0.25, stratify=labels)` gives train `(1347, 8, 8)` and test `(450, 8, 8)` with the same class proportions.
- **Why it is here:** Scaling inputs to a small range keeps activations sane (step 05). Stratifying keeps all ten classes represented in the test set.

### 2. Convolution by hand: slide a small kernel over the image; each output = sum of (patch * kernel)  (`main.py` lines 30-52)
```python
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
```
- **In simple words:** Do the sliding-window yourself: cut out a 3x3 patch, multiply by the kernel, add up, write one output pixel, move on.
- **Exactly:** `np.pad(img, pad)` adds `pad` zeros on every side. The output size is `(n - k)//stride + 1` after padding. The kernel `[[1,0,-1],[2,0,-2],[1,0,-1]]` is the Sobel filter: it outputs a large positive number where the image gets darker to the right and negative where brighter. For four stride/padding combinations the shapes are `(6,6)`, `(8,8)`, `(3,3)`, `(4,4)` and the maximum difference to `F.conv2d` is `0.0e+00`. The printed edge map is an `(8, 8)` array whose strong negative entries (about -3) sit on the left edge of the zero's ring and positive entries (about +3) on its right side.
- **Why it is here:** Convolution is only a loop of dot products; writing it removes the mystery and shows that kernel size, stride and padding fully determine the output shape.

### 3. Pooling by hand: keep the maximum of each non-overlapping 2x2 block (halves the resolution)  (`main.py` lines 55-63)
```python
# [3] Pooling by hand: keep the maximum of each non-overlapping 2x2 block (halves the resolution)
def maxpool2x2(x):                                               # x: (H, W) with even H and W
    H, W = x.shape
    return x.reshape(H // 2, 2, W // 2, 2).max(axis=(1, 3))      # group into 2x2 blocks, take the max of each


pooled = maxpool2x2(conv2d_naive(digit, vertical_edge, 1, 1))
ref_pool = F.max_pool2d(torch.tensor(conv2d_naive(digit, vertical_edge, 1, 1))[None, None], 2)[0, 0].numpy()
print("pooled shape", pooled.shape, "| matches torch max_pool2d:", np.allclose(pooled, ref_pool))
```
- **In simple words:** Shrink the picture by keeping only the strongest response in every 2x2 square.
- **Exactly:** `x.reshape(H//2, 2, W//2, 2)` splits each axis into (block, position-in-block); `.max(axis=(1, 3))` takes the maximum over the two within-block axes. The `(8, 8)` edge map becomes `(4, 4)` and equals `F.max_pool2d` (`True`).
- **Why it is here:** Pooling reduces computation, adds a little shift tolerance (a feature that moves one pixel inside its block gives the same pooled value) and grows the receptive field.

### 4. Receptive field: which input pixels can influence one unit of layer L?  (`main.py` lines 66-77)
```python
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
```
- **In simple words:** Work out how much of the original picture each unit can 'see'.
- **Exactly:** Each layer adds `(k-1) * jump` to the field and multiplies `jump` (distance between neighbouring units in input pixels) by its stride. Printed: `3`, `8`, `10`.
- **Why it is here:** If the receptive field is smaller than the object, the network cannot recognise it by one unit: the central design check for any CNN (and for OCR text-line models).

### 5. Two models: a fully connected net and a small CNN with a similar parameter budget  (`main.py` lines 80-102)
```python
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
```
- **In simple words:** Define two classifiers: an ordinary fully connected network and a CNN with fewer than a third of the parameters.
- **Exactly:** `MLP`: `Flatten` turns `(batch,1,8,8)` into `(batch,64)` and then 64 -> 64 -> 32 -> 10, 6,570 parameters. `CNN`: `conv1 (1->8, 3x3, pad 1)` keeps 8x8, pooling gives 4x4, `conv2 (8->16)` and pooling give `(16, 2, 2)`, flattening gives 64 numbers into `Linear(64, 10)`. 80 + 1,168 + 650 = 1,898 parameters, as printed.
- **Why it is here:** A fair comparison: the CNN is smaller, so any advantage comes from the architectural prior, not from capacity.

### 6. Train both with the standard loop (module 03) and evaluate on the clean test set and on 1-pixel-shifted images  (`main.py` lines 105-131)
```python
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
```
- **In simple words:** Train both for 30 epochs with the standard loop, then test on clean images and on the same images moved one pixel to the right.
- **Exactly:** Add a channel axis (`[:, None]`) to get `(N, 1, 8, 8)`. Mini-batches of 64 from a random permutation, Adam with lr 3e-3. `np.roll(Xte, 1, axis=2)` shifts columns right by 1 with wrap-around. Results: MLP train 0.996, test 0.973, shifted 0.480; CNN train 0.993, test 0.976, shifted 0.633.
- **Why it is here:** On clean data both are excellent (this is an easy dataset). The shift test exposes the difference in what they have learned: the MLP memorised pixel positions, the CNN's shared filters transfer partially.

### 7. Why convolutions scale: parameter count for one layer on a 224x224 RGB image  (`main.py` lines 134-140)
```python
# [7] Why convolutions scale: parameter count for one layer on a 224x224 RGB image
dense = 224 * 224 * 3 * 1000                                     # fully connected to 1000 units
conv = 64 * 3 * 3 * 3 + 64                                       # 64 filters of 3x3x3 (+ bias), shared across the whole image
print(f"\ndense layer 224x224x3 -> 1000 units: {dense:,} weights | conv layer with 64 filters 3x3: {conv:,} weights")

# TRY IT: change kernel_size to 5 (and padding to 2) in conv1 and compare parameter count and accuracy.
# BREAK IT: change self.fc to nn.Linear(16 * 4 * 4, 10) and read the shape error at the first forward pass.
```
- **In simple words:** Compare how many weights you would need for a real-size photo.
- **Exactly:** Dense: `224*224*3*1000 = 150,528,000` weights for one layer. Conv: 64 filters of `3x3x3` plus 64 biases = `1,792` weights, and it also produces a full feature map at every position.
- **Why it is here:** This single comparison is the whole argument for convolutions at scale: parameters independent of image size and a prior that matches images.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `np.pad(img, pad)` | pad an array with zeros on all sides | `(8,8)`, pad 1 -> `(10,10)` |
| `img[i*s : i*s+kh, j*s : j*s+kw]` | 2-D slice: a patch | window for output position (i, j) |
| `(patch * kernel).sum()` | elementwise product then sum = dot product of flattened arrays | - |
| `x.reshape(H//2, 2, W//2, 2).max(axis=(1, 3))` | view as blocks, reduce over two axes | 2x2 max pooling |
| `tensor[None, None]` | add two leading axes of size 1 | `(8,8) -> (1,1,8,8)` |
| `a[:, None]` | add an axis at position 1 | `(N,8,8) -> (N,1,8,8)` |
| `x.flatten(1)` | flatten all dims from dim 1 on, keep the batch | `(B,16,2,2) -> (B,64)` |
| `class CNN(nn.Module)` with `super().__init__()` | define a model; `forward` describes the computation | - |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `F.conv2d(x, w, stride, padding)` | torch.nn.functional | 2-D cross-correlation; `x (N,Cin,H,W)`, `w (Cout,Cin,kh,kw)` | `stride`, `padding` | `(N,Cout,H',W')` |
| `nn.Conv2d(cin, cout, kernel_size, padding)` | torch | learnable convolution layer | `stride`, `bias`, `groups` | module |
| `F.max_pool2d(x, 2)` | torch.nn.functional | 2x2 max pooling, stride 2 by default | `kernel_size`, `stride` | tensor |
| `nn.Flatten` | torch | flatten all dims except batch | `start_dim=1` | module |
| `load_digits()` | scikit-learn | bundled 8x8 digits dataset | `return_X_y` | Bunch with `.images`, `.target` |
| `train_test_split` | scikit-learn | random split with optional stratification | `test_size`, `stratify`, `random_state` | 4 arrays |
| `np.roll(a, shift, axis)` | numpy | shift with wrap-around | - | array |

## Execution trace (real run)
```
train (1347, 8, 8) test (450, 8, 8) | one digit is a 0 image of shape (8, 8)
stride 1 pad 0: output (6, 6) (formula floor((8 + 2*0 - 3)/1) + 1 = 6), max |mine - torch| = 0.0e+00
stride 1 pad 1: output (8, 8) (formula floor((8 + 2*1 - 3)/1) + 1 = 8), max |mine - torch| = 0.0e+00
stride 2 pad 0: output (3, 3) (formula floor((8 + 2*0 - 3)/2) + 1 = 3), max |mine - torch| = 0.0e+00
stride 2 pad 1: output (4, 4) (formula floor((8 + 2*1 - 3)/2) + 1 = 4), max |mine - torch| = 0.0e+00
edge map of digit 0 (stride 1, pad 1), rounded:
 [[ 0.  -1.4 -2.6 -0.3  1.5  1.4  1.1  0.3]
 [-0.2 -2.9 -2.6  1.1  0.2  0.7  2.6  1.1]
 [-0.6 -3.4 -0.6  2.8 -1.6 -1.2  2.8  1.8]
 [-1.  -2.9  0.9  2.9 -2.1 -2.   2.2  2. ]
 [-1.1 -2.4  1.1  2.4 -2.4 -1.9  2.4  1.9]
 [-0.9 -2.8  0.6  2.  -2.5 -0.6  2.8  1.4]
 [-0.5 -2.8 -0.9  0.9 -0.8  1.5  2.2  0.4]
 [-0.1 -1.6 -1.8 -0.2  1.2  1.9  0.8  0. ]]
pooled shape (4, 4) | matches torch max_pool2d: True
receptive field: conv3 -> 3 | conv3-pool2-conv3 -> 8 | conv3-pool2-conv3-pool2 -> 10 pixels (our images are 8 wide)

MLP parameters: 6570 | CNN parameters: 1898
MLP: train acc 0.996 | test acc 0.973 | test acc with images shifted 1 pixel 0.480
CNN: train acc 0.993 | test acc 0.976 | test acc with images shifted 1 pixel 0.633

dense layer 224x224x3 -> 1000 units: 150,528,000 weights | conv layer with 64 filters 3x3: 1,792 weights
```
- Shapes `(6,6) (8,8) (3,3) (4,4)` agree with the formula and with torch (difference exactly 0.0).
- The edge map has negative values on the left side of the digit's ring and positive on the right: that is a vertical-edge detector at work.
- `pooled shape (4, 4)`: matches torch.
- Receptive fields 3, 8, 10.
- Parameters: MLP 6,570, CNN 1,898.
- Accuracy: both about 97.5 percent; shifting by one pixel drops the MLP to 0.480 and the CNN to 0.633. The CNN is not shift-proof: pooling gives limited tolerance, so augmentation is still needed.
- 150,528,000 versus 1,792 weights.

## Experiments (change ONE thing)
1. Kernel 5 with padding 2 in `conv1`: 2,026 parameters, test accuracy 0.971, shifted 0.633. More parameters, no improvement on this tiny task.
2. Other seeds (test acc / shifted acc): seed 1 MLP 0.971 / 0.438, CNN 0.971 / 0.558; seed 2 MLP 0.964 / 0.456, CNN 0.978 / 0.649. Clean accuracy differences are within noise (about 1 point) but the CNN is consistently 12 to 19 points better on shifted images.
3. Exercise: change the shift to 2 pixels (`np.roll(Xte, 2, axis=2)`) to see how fast each model degrades (not run here).

## Debugging practice
BREAK IT: change `self.fc = nn.Linear(16 * 2 * 2, 10)` to `nn.Linear(16 * 4 * 4, 10)`:
```
  File ".../torch/nn/modules/linear.py", line 134, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5x64 and 256x10)
```
Read it: `mat1` is the input to `fc` (batch 5, 64 features); `mat2` is the weight, `256 -> 10`. The conv stack produced `16*2*2 = 64` features, not `16*4*4 = 256`. Trace the shapes by hand: `8 -> pool -> 4 -> pool -> 2`. A quick way to find the number without arithmetic: run one dummy batch through the convolutional part and print `x.shape` before the Linear.

## Rebuild checklist
- [ ] write `conv2d_naive` with stride and padding and state the output-size formula
- [ ] count conv-layer parameters `Cout*Cin*kh*kw + Cout`
- [ ] write max-pool with `reshape` and `max`
- [ ] compute a receptive field for a stack
- [ ] build a small CNN and track tensor shapes layer by layer
- [ ] explain locality, weight sharing, equivariance and where invariance comes from

## The new problem (cliffhanger)
Convolutions exploit space; they treat each image independently. Text, audio and sensor streams are *sequences* where order and memory matter, and a card number or a sentence can be any length. How do we make a network that reads one element at a time and remembers? Next: [step 09, RNNs, LSTMs and why attention was invented](../09_rnn_lstm_and_why_attention/WALKTHROUGH.md).

> **Say it aloud (interview box):** A convolution slides a small shared kernel over the input, so each output is a local dot product. That gives locality, weight sharing and translation equivariance, and the parameter count depends on the kernel and channels, not the image size: 1,792 weights for 64 3x3 filters on an RGB image versus 150 million for a dense layer. Output size is floor((n + 2p - k)/s) + 1. Pooling and stride shrink resolution and grow the receptive field. On 8x8 digits the CNN matched the MLP with a third of the parameters and was more robust to a one-pixel shift, but it still needs augmentation.
