# Terms — Convolutional networks

Folder: `15_neural_networks/08_cnn_convolutions` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [15 README](../README.md)

**Terms in this folder:** [Convolution (cross-correlation)](#convolution-cross-correlation) · [Kernel (filter)](#kernel-filter) · [Feature map](#feature-map) · [Stride](#stride) · [Padding (zero padding, 'same')](#padding-zero-padding-same) · [Output size formula](#output-size-formula) · [Sobel filter (edge detector)](#sobel-filter-edge-detector) · [Max pooling](#max-pooling) · [Receptive field](#receptive-field) · [Weight sharing](#weight-sharing) · [Translation equivariance and invariance](#translation-equivariance-and-invariance) · [Channels and NCHW layout](#channels-and-nchw-layout) · [Flatten](#flatten) · [Inductive bias](#inductive-bias) · [1x1 convolution](#1x1-convolution) · [Vision Transformer (ViT)](#vision-transformer-vit) · [RuntimeError: mat1 and mat2 shapes cannot be multiplied](#runtimeerror-mat1-and-mat2-shapes-cannot-be-multiplied) · [Stratified split](#stratified-split)

---

## Convolution (cross-correlation)  `(concept)`
**1. Definition:** Sliding a small kernel over an input and taking the dot product with each patch.
**2. In plain words:** A magnifying template that checks 'does this local pattern appear here?' at every position.
**3. Example to quote:** > "`conv2d_naive`"
**Seen in:** `main.py`, block 2

## Kernel (filter)  `(concept)`
**1. Definition:** The small weight tensor of a convolution, `(kh, kw)` per input channel.
**2. In plain words:** The pattern it looks for; learned from data.
**3. Example to quote:** > "Sobel `[[1,0,-1],[2,0,-2],[1,0,-1]]` detects vertical edges."
**Seen in:** `main.py`, block 2

## Feature map  `(concept)`
**1. Definition:** The output of one filter across all positions.
**2. In plain words:** A picture of where the pattern was found.
**3. Example to quote:** > "`(8, 8)` edge map."
**Seen in:** `main.py`, block 2

## Stride  `(concept)`
**1. Definition:** Number of positions the window moves between outputs.
**2. In plain words:** Step length of the scan.
**3. Example to quote:** > "Stride 2 turns 8 into 3 or 4."
**Seen in:** `main.py`, block 2

## Padding (zero padding, 'same')  `(concept)`
**1. Definition:** Adding border values (zeros) so the kernel can be centred on edge pixels and the size can be preserved.
**2. In plain words:** A frame around the picture.
**3. Example to quote:** > "`pad=1` keeps 8x8."
**Seen in:** `main.py`, block 2

## Output size formula  `(concept)`
**1. Definition:** `floor((n + 2*pad - k)/stride) + 1`.
**2. In plain words:** Count how many window positions fit.
**3. Example to quote:** > "`(8 + 2 - 3)//1 + 1 = 8`."
**Seen in:** `main.py`, block 2

## Sobel filter (edge detector)  `(concept)`
**1. Definition:** A fixed 3x3 kernel that approximates the horizontal image gradient.
**2. In plain words:** Highlights where brightness changes left to right.
**3. Example to quote:** > "Learned first-layer filters in CNNs often resemble such edge detectors."
**Seen in:** `main.py`, block 2

## Max pooling  `(concept)`
**1. Definition:** Replace each window by its maximum; reduces resolution without weights.
**2. In plain words:** Keep the strongest response of each neighbourhood.
**3. Example to quote:** > "`maxpool2x2`"
**Seen in:** `main.py`, block 3

## Receptive field  `(concept)`
**1. Definition:** The set of input pixels that can influence a given unit; grows with depth, kernel size and stride.
**2. In plain words:** How much of the picture a unit can see.
**3. Example to quote:** > "10 pixels after conv-pool-conv-pool."
**Seen in:** `main.py`, block 4

## Weight sharing  `(concept)`
**1. Definition:** The same kernel weights are used at every spatial position.
**2. In plain words:** One expert reused everywhere instead of hiring one per location.
**3. Example to quote:** > "1,792 weights instead of 150 million."
**Seen in:** `main.py`, block 7

## Translation equivariance and invariance  `(concept)`
**1. Definition:** Equivariance: shifting the input shifts the feature map. Invariance: the final output does not change under a shift (only approximate in CNNs).
**2. In plain words:** Find the same thing wherever it is; and then answer 'cat' wherever it was.
**3. Example to quote:** > "Shifted-digit accuracy 0.633 (CNN) vs 0.480 (MLP)."
**Seen in:** `main.py`, block 6

## Channels and NCHW layout  `(concept)`
**1. Definition:** The axes of an image batch: batch, channels, height, width.
**2. In plain words:** Grayscale has 1 channel, RGB 3, feature maps many.
**3. Example to quote:** > "`(N, 1, 8, 8)`"
**Seen in:** `main.py`, block 6

## Flatten  `(library)`
**1. Definition:** Reshape `(batch, C, H, W)` to `(batch, C*H*W)` before a Linear layer.
**2. In plain words:** Unroll the picture into a list.
**3. Example to quote:** > "`x.flatten(1)`"
**Seen in:** `main.py`, block 5

## Inductive bias  `(concept)`
**1. Definition:** Assumptions built into an architecture (locality and sharing for CNNs).
**2. In plain words:** What the model believes before seeing data.
**3. Example to quote:** > "CNN vs MLP on shifted digits."
**Seen in:** `WALKTHROUGH.md`, theory

## 1x1 convolution  `(concept)`
**1. Definition:** A convolution with kernel size 1: mixes channels at each pixel.
**2. In plain words:** A Linear layer applied at every position.
**3. Example to quote:** > "Used to change channel count."
**Seen in:** `WALKTHROUGH.md`, theory

## Vision Transformer (ViT)  `(concept)`
**1. Definition:** A transformer applied to image patches treated as tokens.
**2. In plain words:** Attention instead of local filters.
**3. Example to quote:** > "Less built-in bias, needs more data."
**Seen in:** `WALKTHROUGH.md`, theory

## RuntimeError: mat1 and mat2 shapes cannot be multiplied  `(error)`
**1. Definition:** A Linear layer received a different number of features than its weight expects.
**2. In plain words:** The flattened conv output has a different size than you assumed.
**3. Example to quote:** > "`(5x64 and 256x10)`"
**Seen in:** `WALKTHROUGH.md`, debugging

## Stratified split  `(library)`
**1. Definition:** A train/test split that preserves the class proportions.
**2. In plain words:** Every class appears in the test set in the right share.
**3. Example to quote:** > "`stratify=labels`"
**Seen in:** `main.py`, block 1
