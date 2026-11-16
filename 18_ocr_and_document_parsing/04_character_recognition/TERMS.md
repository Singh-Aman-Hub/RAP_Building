# Terms — Character recognition

Folder: `18_ocr_and_document_parsing/04_character_recognition` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [18 README](../README.md)

**Terms in this folder:** [Template matching](#template-matching) · [1-nearest-neighbour (1-NN)](#1-nearest-neighbour-1-nn) · [Euclidean distance (cdist)](#euclidean-distance-cdist) · [Glyph frame and baseline anchor](#glyph-frame-and-baseline-anchor) · [CNN (convolutional neural network)](#cnn-convolutional-neural-network) · [Convolution, padding and pooling](#convolution-padding-and-pooling) · [Data augmentation](#data-augmentation) · [Affine grid and grid_sample](#affine-grid-and-grid_sample) · [Cross-entropy loss](#cross-entropy-loss) · [Held-out font (domain shift)](#held-out-font-domain-shift) · [Confusion pairs (0/O, 1/l/I, rn/m)](#confusion-pairs-0o-1li-rnm) · [ICR, HTR, OMR](#icr-htr-omr) · [RuntimeError: expected input to have 1 channels, but got 5 channels](#runtimeerror-expected-input-to-have-1-channels-but-got-5-channels)

---

## Template matching  `(concept)`
**1. Definition:** Classifying an image by finding the stored template with the smallest pixel distance (or highest correlation).
**2. In plain words:** Hold the letter up against a deck of reference cards. Perfect for one font, poor for another: 0.565 on a new font, 0.363 once rotated and noisy.
**3. Example to quote:** > "Template matching compares pixels, so a 2 px shift or a new font breaks it."
**Seen in:** `main.py`, `nn_predict`

## 1-nearest-neighbour (1-NN)  `(concept)`
**1. Definition:** A classifier that returns the label of the single closest training example under a distance.
**2. In plain words:** "Who does this look most like?" with no learning step.
**3. Example to quote:** > "Giving 1-NN 558 templates instead of 62 barely helped (0.573), so coverage matters less than learned invariance."
```python
d = torch.cdist(q.flatten(1), t.flatten(1))
label = tlabels[d.argmin(dim=1)]
```
**Seen in:** `main.py`, section 3

## Euclidean distance (cdist)  `(syntax)`
**1. Definition:** sqrt(sum (a_i - b_i)^2); `torch.cdist(a, b)` returns all pairwise distances, shape (n, m).
**2. In plain words:** How far apart two pictures are, treating pixels as coordinates.
**3. Example to quote:** > "In the bundled font l and I are 1.92 apart: the same picture for practical purposes."
**Seen in:** `main.py`, `nn_predict`, `dist`

## Glyph frame and baseline anchor  `(concept)`
**1. Definition:** A fixed-size canvas on which every character is drawn with its baseline on the same row (`anchor="ls"`), keeping the height differences between letters.
**2. In plain words:** All letters stand on the same line, so a tall O and a short o stay different.
**3. Example to quote:** > "Anchoring by the top instead dropped held-out accuracy from 0.782 to 0.081."
**Seen in:** `main.py`, `glyph`

## CNN (convolutional neural network)  `(concept)`
**1. Definition:** A network built from convolutions (shared local filters) and pooling, ending in linear layers.
**2. In plain words:** Small stroke detectors slid over the image, then a vote.
**3. Example to quote:** > "105,662 parameters were enough to reach 0.782 on an unseen font."
**Seen in:** `main.py`, `GlyphCNN`

## Convolution, padding and pooling  `(syntax)`
**1. Definition:** `Conv2d(in, out, 3, padding=1)` applies out filters of size 3x3 and keeps height/width; `MaxPool2d(2)` halves them by keeping the max of each 2x2.
**2. In plain words:** Detect strokes, then keep a coarser map of where they were.
**3. Example to quote:** > "Shape trace: (B,1,32,32) -> (B,8,16,16) -> (B,16,8,8) -> flatten 1024 -> 96 -> 62."
**Seen in:** `main.py`, `GlyphCNN`

## Data augmentation  `(concept)`
**1. Definition:** Randomly transforming training samples (rotation, scale, shear, shift, noise) every step so the model learns invariances.
**2. In plain words:** The same letter, seen slightly crooked, bigger, noisier, every time. Accuracy on warped unseen-font glyphs rose from 0.468 to 0.782.
**3. Example to quote:** > "Too much augmentation hurts: strength 2.0 gave 0.677."
**Seen in:** `main.py`, `augment`

## Affine grid and grid_sample  `(syntax)`
**1. Definition:** `F.affine_grid(theta, size)` creates sampling coordinates from a per-image 2x3 matrix; `F.grid_sample(x, grid)` resamples the images at those coordinates.
**2. In plain words:** A batch-wide "rotate, scale and shift these pictures" in two calls.
**3. Example to quote:** > "theta maps OUTPUT coordinates to input coordinates, so its effect is the inverse transform."
**Seen in:** `main.py`, `augment`

## Cross-entropy loss  `(concept)`
**1. Definition:** -log softmax(logits)[true class], averaged over the batch; `F.cross_entropy` takes raw logits and integer labels.
**2. In plain words:** Penalty that grows as the model gives the true class a lower probability.
**3. Example to quote:** > "Do not apply softmax before F.cross_entropy; it does it internally."
**Seen in:** `main.py`, `train`

## Held-out font (domain shift)  `(concept)`
**1. Definition:** A test set drawn from a distribution the training set does not contain, here a font never used in training.
**2. In plain words:** Tests whether the model learned letters or memorised fonts.
**3. Example to quote:** > "I always evaluate on a font (or scanner, or language) that training did not see."
**Seen in:** `main.py`, `TEST_FONT`

## Confusion pairs (0/O, 1/l/I, rn/m)  `(concept)`
**1. Definition:** Character pairs whose images are nearly identical, so shape alone cannot separate them.
**2. In plain words:** l vs I distance 1.92; rn vs m distance 4.19 (closer than m vs n at 10.93).
**3. Example to quote:** > "Production OCR resolves these with a language prior or field type, not with a better CNN."
**Seen in:** `main.py`, section 6

## ICR, HTR, OMR  `(concept)`
**1. Definition:** ICR: recognition of hand-printed characters. HTR: handwritten text recognition on lines. OMR: optical mark recognition (checkboxes, bubbles).
**2. In plain words:** OCR for print, ICR/HTR for handwriting, OMR for ticks.
**3. Example to quote:** > "Modern VLM OCR covers print and handwriting in one model; HTR still has special benchmarks."
**Seen in:** `WALKTHROUGH.md`, same-thing table

## RuntimeError: expected input to have 1 channels, but got 5 channels  `(error)`
**1. Definition:** Conv2d received `(5, 32, 32)`, which PyTorch reads as an unbatched (C, H, W) image with 5 channels.
**2. In plain words:** The batch axis was dropped or the channel axis is missing.
**3. Example to quote:** > "I add the channel axis with `x[:, None]` and print shapes at the model boundary."
**Seen in:** BREAK IT in `WALKTHROUGH.md`
