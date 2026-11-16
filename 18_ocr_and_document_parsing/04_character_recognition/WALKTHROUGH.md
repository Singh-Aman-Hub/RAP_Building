# Step 04 — Character recognition: templates, then a small CNN

## The story so far
[Step 03](../03_layout_segmentation/WALKTHROUGH.md) cut a clean page into lines, words and one box per character. We now have thousands of small images and no idea what is in them.

## The pain point
A character looks different in every font, size, rotation and noise level. A program that compares each box with one stored picture of each letter works on its own font and falls apart on another. And some letters are almost the same shape: `0` and `O`, `1`, `l` and `I`, `rn` and `m`.

## The spark
First try the simplest thing: nearest-neighbour matching against rendered templates. Then let a small convolutional network learn what "an A" looks like across fonts, and make it robust by showing it randomly warped, noisy copies of the training glyphs (augmentation).

## Goal
Render 62 classes (digits, upper and lower case letters) in several fonts, measure 1-nearest-neighbour template matching on an unseen font, train a small CNN with and without augmentation, list the confusion pairs, and measure how close the dangerous pairs really are in pixel space.

## The theory (read this before the interview)
**Template matching.** Store one image per class. For a query glyph, compute the distance to every template (Euclidean, or normalised cross-correlation) and return the label of the nearest one. This is 1-nearest-neighbour (1-NN) in pixel space.
```
d(q, t) = sqrt( sum over pixels (q_i - t_i)^2 )        label = argmin over templates
```
It is exact for the font it was built for, and brittle for everything else: a shift by 2 px or a 5 degree rotation moves strokes onto different pixels, so the distance explodes.

**CNN classifier.** Convolutions look for local strokes (edges, curves), pooling makes the position less exact, and a linear layer turns the features into 62 scores. Trained with cross-entropy: `loss = -log softmax(logits)[true class]`. A CNN generalises to new fonts only to the extent that the training data covers the variation.

**Augmentation.** Each training batch is randomly warped (rotation, scale, shear, shift) and noised on the fly. The network never sees the same image twice, so it cannot memorise pixels and has to learn shape. Too strong augmentation hurts (the glyph stops looking like itself).

**Baseline alignment matters.** The classifier input here is a fixed frame with all glyphs on a common baseline. That keeps height cues that distinguish `o`/`O`, `c`/`C`, `s`/`S`, `l`/`I`. If you crop each glyph to its tight bounding box and rescale, `o` and `O` become the same picture.

**Confusion pairs.** Some pairs are inseparable by shape alone:
| Pair | Why | What resolves it |
|---|---|---|
| `0` vs `O` | same oval, differ by width or a slash/dot in some fonts | context: digits among digits (field type, language model, dictionary) |
| `1` vs `l` vs `I` | a vertical stroke, with or without a flag and serifs | context, and the font (in the bundled font `l` and `I` are 1.9 apart, in pixel distance, nearly the same picture) |
| `rn` vs `m` | two glyphs side by side look like one wider glyph | width, spacing, dictionary (`modern` vs `rnodern`) |
| `5` vs `S`, `8` vs `B`, `u` vs `U`, `o` vs `O`, `x` vs `X` | shape-alike, size is the only cue | baseline frame, language model |
Production OCR therefore always adds a **language prior** (dictionary, n-gram, or a learned decoder) to pick between visually ambiguous characters. CTC with a beam search (step 05) and VLMs do this implicitly.

**Modern (2026) practice.** Single-character classifiers are history: engines recognise whole lines (CRNN + CTC in Tesseract 4/5 and PaddleOCR, attention / transformer decoders in TrOCR and PARSeq) and VLMs read whole pages. The classifier is still the right way to learn the basics: features, augmentation, confusion matrices.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Template matching | nearest-neighbour matching, correlation matching, "pattern matching OCR" | near | Compares pixels directly; no learning |
| Character recogniser | glyph classifier, OCR classifier | yes | Maps one image to one label |
| ICR (intelligent character recognition) | handwriting-aware OCR, "HTR" for text lines | near | Machine print = OCR; handwritten characters = ICR; handwritten text lines = HTR |
| OMR | optical mark recognition | no | Detects ticked boxes and bubbles, not text |
| Data augmentation | synthetic variation, domain randomisation | near | Augmentation warps real samples; synthetic data generates new ones from fonts |
| Held-out font | out-of-distribution test, "domain shift" | near | A test set that differs from training on purpose |
| Confusion matrix | error matrix | yes | Rows = true, columns = predicted |

## Run it
```bash
cd 18_ocr_and_document_parsing/04_character_recognition
python main.py       # ~40 s of CPU time (2 minutes on a busy machine); writes sample_glyph_strip.png
```
Fonts: the run trains on the bundled Pillow font plus Arial and Courier New, and tests on Verdana. These are macOS system fonts (`/System/Library/Fonts/Supplemental/`). If they are missing, the script falls back to the bundled font only, and the numbers will differ from the ones shown here.

## Code, step by step

### 1. Imports, seeds, the 62 classes and the glyph frame  (`main.py` lines 16–37)
```python
import string
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from docimg import get_font, available_system_fonts, save_png   # noqa: E402

HERE = Path(__file__).parent
torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(2)                                # small convs: more threads only add overhead
CLASSES = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)     # 62 classes
C2I = {c: i for i, c in enumerate(CLASSES)}
CANVAS, BASE = 32, 24                                   # glyph frame: 32x32 pixels, baseline at row 24
```
- **In simple words:** Load tools, fix the random seeds, and define the 62 things we want to recognise.
- **Exactly:** `CLASSES` is `0-9`, `A-Z`, `a-z`; `C2I` maps each to an index 0 to 61. `CANVAS = 32` and `BASE = 24` define a 32 x 32 frame whose baseline is row 24. `torch.set_num_threads(2)` is deliberate: for such small convolutions more threads add overhead.
- **Why it is here:** A fixed frame with a fixed baseline is what keeps `o` and `O` different.

### 2. Choose training and held-out fonts  (`main.py` lines 39–43)
```python
sysf = available_system_fonts()
TRAIN_FONTS = [None] + sysf[:2]                         # None = Pillow's bundled font; plus Arial and Courier New if present
TEST_FONT = sysf[2] if len(sysf) > 2 else None          # Verdana if present: never seen in training
print("fonts: train", [Path(f).name if f else "bundled" for f in TRAIN_FONTS],
      " held-out", Path(TEST_FONT).name if TEST_FONT else "bundled (no system fonts found)")
```
- **In simple words:** Pick which fonts the model may see, and one it may not.
- **Exactly:** `available_system_fonts()` returns the macOS fonts that exist. Training: bundled font, Arial, Courier New. Held-out: Verdana. `None` as a font path means the bundled font.
- **Why it is here:** Testing on a font from training would measure memory, not recognition.

### 3. Render one glyph into the frame  (`main.py` lines 46–55)
```python
# ---- 1. Render one glyph into a fixed frame, baseline-aligned (so 'o' vs 'O' and 'l' vs 'I' keep their height) ----
def glyph(ch, font_path, size):
    im = Image.new("L", (CANVAS, CANVAS), 0)            # black background, white ink -> ink = 1.0
    ImageDraw.Draw(im).text((8, BASE), ch, fill=255, font=get_font(size, font_path), anchor="ls")
    a = np.asarray(im, dtype=np.float32) / 255.0
    cols = np.flatnonzero(a.sum(axis=0) > 0)
    if cols.size:
        shift = CANVAS // 2 - (cols[0] + cols[-1]) // 2  # centre the ink horizontally (not vertically)
        a = np.roll(a, shift, axis=1)
    return a                                            # (32, 32)
```
- **In simple words:** Draw a letter in white on black, sitting on the baseline, and slide it so it is centred left to right.
- **Exactly:** `anchor="ls"` means "left, baseline" so `text((8, 24), ch)` puts the baseline at row 24 whatever the letter is. `cols[0]` and `cols[-1]` are the first and last ink columns; `np.roll` moves the glyph so their midpoint is column 16. Result `float32 (32, 32)` in 0..1.
- **Why it is here:** Without a common baseline, heights are lost; without horizontal centring, every small shift would count as a difference.

### 4. Build the train and test sets  (`main.py` lines 58–72)
```python
def make_set(fonts, sizes):
    X, y = [], []
    for f in fonts:
        for s in sizes:
            for ch in CLASSES:
                X.append(glyph(ch, f, s))
                y.append(C2I[ch])
    return torch.tensor(np.stack(X))[:, None], torch.tensor(y)   # (N, 1, 32, 32), (N,)


Xtr, ytr = make_set(TRAIN_FONTS, [18, 22, 26])
Xte, yte = make_set([TEST_FONT], [20, 24])
print("train glyphs", tuple(Xtr.shape), " test glyphs", tuple(Xte.shape))
save_png((np.concatenate([glyph(c, TEST_FONT, 24) for c in "0OIl1rnmB8S5"], axis=1) * 255).astype(np.uint8),
         HERE / "sample_glyph_strip.png")
```
- **In simple words:** Draw every class in every font at several sizes and stack them into arrays.
- **Exactly:** Train: 3 fonts x 3 sizes (18, 22, 26) x 62 = `(558, 1, 32, 32)`. Test: Verdana x sizes (20, 24) x 62 = `(124, 1, 32, 32)`; the leading 1 is the channel axis PyTorch expects. A strip of confusable glyphs is saved to `sample_glyph_strip.png`.
- **Why it is here:** Everything is generated from fonts, so labels are free and the experiment is reproducible.

### 5. Random warps and noise on a batch  (`main.py` lines 75–87)
```python
# ---- 2. Augmentation on a batch: random rotation, scale, shear, shift, blur-free noise ------------------------
def augment(x, gen, strength=1.0):
    n = x.shape[0]
    ang = (torch.rand(n, generator=gen) * 2 - 1) * 8 * strength * np.pi / 180        # +-8 degrees
    sc = 1 + (torch.rand(n, generator=gen) * 2 - 1) * 0.12 * strength                 # +-12 % scale
    sh = (torch.rand(n, generator=gen) * 2 - 1) * 0.15 * strength                     # shear
    tx, ty = [(torch.rand(n, generator=gen) * 2 - 1) * 0.08 * strength for _ in range(2)]   # +-2.5 px shift
    theta = torch.zeros(n, 2, 3)                                                       # affine matrix per sample
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = sc * torch.cos(ang), -sc * torch.sin(ang) + sh, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sc * torch.sin(ang), sc * torch.cos(ang), ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)                         # (n, 32, 32, 2) sampling positions
    out = F.grid_sample(x, grid, align_corners=False)                                  # warp the images
    return (out + torch.randn(out.shape, generator=gen) * 0.15 * strength).clamp(0, 1)  # + gaussian noise
```
- **In simple words:** Slightly rotate, resize, slant and shift every picture in the batch and add grain.
- **Exactly:** `theta` is an affine matrix `(n, 2, 3)` per image: rotation by +/-8 degrees, scale +/-12%, shear, shift +/-2.5 px. `F.affine_grid` builds sampling positions `(n, 32, 32, 2)` and `F.grid_sample` warps all images in one call, bilinearly. `torch.randn(..., generator=gen) * 0.15` adds Gaussian noise; `clamp(0, 1)` keeps valid pixel values. `strength` scales every effect.
- **Why it is here:** This is the entire robustness mechanism: no new data, only varied views of the same data.

### 6. 1-nearest-neighbour matcher  (`main.py` lines 90–95)
```python
# ---- 3. Baseline: 1-nearest-neighbour template matching ---------------------------------------------------------
def nn_predict(queries, templates, tlabels):
    q = queries.flatten(1)                              # (n, 1024)
    t = templates.flatten(1)                            # (m, 1024)
    d = torch.cdist(q, t)                               # (n, m) Euclidean distances
    return tlabels[d.argmin(dim=1)]
```
- **In simple words:** For every query picture, find the template that looks most alike pixel by pixel.
- **Exactly:** `flatten(1)` turns `(n, 1, 32, 32)` into `(n, 1024)`. `torch.cdist` returns the `(n, m)` Euclidean distance matrix; `argmin(dim=1)` picks the nearest template per query, and `tlabels[...]` converts it to a class index.
- **Why it is here:** It is the baseline any learned model has to beat, and a useful sanity check.

### 7. Score the matcher on an unseen font  (`main.py` lines 98–105)
```python
gen = torch.Generator().manual_seed(123)
Xte_aug = augment(Xte, gen)                             # test glyphs: new font, rotated, noisy
templ_clean_default, tl_clean = make_set([None], [22])  # templates: ONE clean glyph per class, bundled font only
acc_clean = (nn_predict(Xte, templ_clean_default, tl_clean) == yte).float().mean().item()
acc_noisy = (nn_predict(Xte_aug, templ_clean_default, tl_clean) == yte).float().mean().item()
acc_multi = (nn_predict(Xte_aug, Xtr, ytr) == yte).float().mean().item()
print(f"1-NN, 1 template/class: held-out font clean {acc_clean:.3f}, held-out font + rotation + noise {acc_noisy:.3f}")
print(f"1-NN, all {len(Xtr)} training glyphs as templates:  held-out font + rotation + noise {acc_multi:.3f}")
```
- **In simple words:** Test the template matcher on Verdana, once clean and once rotated and noisy.
- **Exactly:** One clean bundled-font template per class gets 0.565 on clean Verdana and 0.363 once the test glyphs are warped and noisy. Using all 558 training glyphs (3 fonts, 3 sizes) as templates reaches 0.573 on the warped set.
- **Why it is here:** It shows how fragile pixel comparison is, and that more templates help far less than learning does.

### 8. The CNN  (`main.py` lines 108–119)
```python
# ---- 4. A small CNN ------------------------------------------------------------------------------------------------
class GlyphCNN(nn.Module):
    def __init__(self, n_classes=62):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),       # (B,8,16,16)
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),      # (B,16,8,8)
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(16 * 8 * 8, 96), nn.ReLU(), nn.Linear(96, n_classes))

    def forward(self, x):                               # x: (B, 1, 32, 32)
        return self.head(self.conv(x))                  # logits (B, 62)
```
- **In simple words:** Two small convolution blocks that find strokes, then two layers that decide which of the 62 letters it is.
- **Exactly:** Input `(B, 1, 32, 32)`; `Conv2d(1, 8, 3, padding=1)` -> `(B, 8, 32, 32)`, `MaxPool2d(2)` -> `(B, 8, 16, 16)`; `Conv2d(8, 16, 3)` and pool -> `(B, 16, 8, 8)`; `Flatten` -> 1024; `Linear(1024, 96)`; `Linear(96, 62)` gives logits `(B, 62)`. In total 105,662 parameters.
- **Why it is here:** Convolutions share weights across positions, so a stroke detector learned once works everywhere in the frame.

### 9. The training loop  (`main.py` lines 122–136)
```python
def train(use_aug, steps=300, batch=64, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = GlyphCNN()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for step in range(steps):
        idx = torch.randint(0, len(Xtr), (batch,), generator=g)
        xb, yb = Xtr[idx], ytr[idx]
        if use_aug:
            xb = augment(xb, g)                         # a fresh random warp every step: the net never sees the same image twice
        loss = F.cross_entropy(model(xb), yb)           # 62-way classification
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model
```
- **In simple words:** Repeat: take a random batch, optionally warp it, measure the error, nudge the weights.
- **Exactly:** 300 steps of 64 samples, Adam with learning rate 2e-3. `torch.randint(0, len(Xtr), (64,), generator=g)` samples indices with replacement. `F.cross_entropy(logits, labels)` expects raw logits `(64, 62)` and integer labels `(64,)`. `zero_grad` clears old gradients, `backward` computes new ones, `step` updates.
- **Why it is here:** The only difference between the two models is the one `if use_aug` line.

### 10. Evaluation helper  (`main.py` lines 139–142)
```python
@torch.no_grad()
def evaluate(model, x, y):
    pred = model(x).argmax(dim=1)
    return (pred == y).float().mean().item(), pred
```
- **In simple words:** Run the model on test images and return accuracy and predictions.
- **Exactly:** `@torch.no_grad()` turns off gradient tracking (saves memory and time). `argmax(dim=1)` gives the predicted class per image.
- **Why it is here:** Evaluation must not change the model or build a graph.

### 11. Train both models and compare  (`main.py` lines 145–153)
```python
t0 = time.time()
cnn_plain = train(use_aug=False)
cnn_aug = train(use_aug=True)
a_plain, _ = evaluate(cnn_plain, Xte_aug, yte)
a_aug, pred = evaluate(cnn_aug, Xte_aug, yte)
a_aug_clean, _ = evaluate(cnn_aug, Xte, yte)
print(f"CNN trained WITHOUT augmentation: held-out font + rotation + noise {a_plain:.3f}")
print(f"CNN trained WITH augmentation:    held-out font + rotation + noise {a_aug:.3f}  (clean held-out {a_aug_clean:.3f})"
      f"   [{time.time() - t0:.0f}s for both trainings, {sum(p.numel() for p in cnn_aug.parameters()):,} parameters]")
```
- **In simple words:** Train once without augmentation and once with, then test both on the warped, noisy, unseen-font glyphs.
- **Exactly:** Without augmentation: 0.468. With: 0.782 (clean held-out: 0.895). 105,662 parameters. Same data, same steps, same seed, only the warping differs. Compare with 1-NN at 0.573.
- **Why it is here:** It is the cleanest demonstration that augmentation, not model size, buys robustness here.

### 12. Which characters are confused?  (`main.py` lines 155–158)
```python
# ---- 5. Which characters get confused? --------------------------------------------------------------------------------
pairs = Counter((CLASSES[a], CLASSES[b]) for a, b in zip(yte.tolist(), pred.tolist()) if a != b)
print("most confused (true -> predicted):", [(f"{a}->{b}", n) for (a, b), n in pairs.most_common(8)])
print("wrong predictions:", sum(pairs.values()), "of", len(yte), "| by true class:", dict(Counter(a for (a, b), n in pairs.items() for _ in range(n)).most_common(6)))
```
- **In simple words:** Count the mistakes by (true, predicted) pairs.
- **Exactly:** `Counter` over `(true, predicted)` for wrong predictions. Top pairs include `1->l`, `I->l`, `M->H`, `O->Q` and `i->l`: the vertical-stroke family appears repeatedly. 27 of 124 test glyphs are wrong.
- **Why it is here:** Error analysis tells you whether you need more data, a better model, or a language prior.

### 13. How alike are the dangerous pairs?  (`main.py` lines 161–169)
```python
# ---- 6. Why some pairs are hard: distance between rendered glyphs and between strings ----------------------------------------
def dist(a, b):
    return float(np.linalg.norm(a - b))


G = lambda ch: glyph(ch, None, 22)                      # bundled font, size 22
print("pixel distance in the bundled font (smaller = more alike):")
for a, b in [("0", "O"), ("l", "I"), ("1", "l"), ("o", "O"), ("5", "S"), ("m", "n"), ("A", "B")]:
    print(f"   {a} vs {b}: {dist(G(a), G(b)):.2f}")
```
- **In simple words:** Measure the pixel distance between glyph pairs in one font. Smaller means harder.
- **Exactly:** `dist` is the Euclidean norm of the difference. In the bundled font: `l` vs `I` 1.92 (nearly identical), `5` vs `S` 4.89, `0` vs `O` 6.72, `1` vs `l` 7.41, `o` vs `O` 7.56, `m` vs `n` 10.68, `A` vs `B` 8.07.
- **Why it is here:** It puts numbers on "these look the same", and shows `l`/`I` cannot be separated by shape in this font.

### 14. `rn` versus `m`  (`main.py` lines 172–184)
```python
def string_img(s, width=48):
    """Render a string at size 22 and squeeze it into a fixed 32 x width frame so 'rn' and 'm' are comparable."""
    im = Image.new("L", (80, CANVAS), 0)
    ImageDraw.Draw(im).text((4, BASE), s, fill=255, font=get_font(22), anchor="ls")
    a = np.asarray(im, dtype=np.float32) / 255.0
    cols = np.flatnonzero(a.sum(axis=0) > 0)
    return np.asarray(Image.fromarray((a[:, cols[0]:cols[-1] + 1] * 255).astype(np.uint8)).resize((width, CANVAS)), dtype=np.float32) / 255.0


print("pixel distance after squeezing the strings to the same width: rn vs m %.2f,  m vs n %.2f,  rn vs nn %.2f"
      % (dist(string_img("rn"), string_img("m")), dist(string_img("m"), string_img("n")), dist(string_img("rn"), string_img("nn"))))
print("rn rendered width = %d px, m rendered width = %d px (the width is the clue a segmenter has)"
      % (get_font(22).getlength("rn"), get_font(22).getlength("m")))
```
- **In simple words:** Draw "rn" and "m", squeeze both to the same width, and see how alike they are.
- **Exactly:** `string_img` renders the string, crops to the ink columns and resizes to `(32, 48)`. Distance rn vs m = 4.19, far less than m vs n (10.93) or rn vs nn (12.05). The unsqueezed widths are 22 px against 20 px: width is the only clue left to a segmenter, and it is within 2 px.
- **Why it is here:** It explains why a dictionary or language model has to arbitrate `modern` vs `rnodern`.

### 15. TRY IT and BREAK IT notes  (`main.py` lines 186–188)
```python
# TRY IT:   raise the augmentation strength to 2.0 in train(); or train for 1500 steps and see whether held-out accuracy still moves.
# BREAK IT: remove `anchor="ls"` from `glyph` (use the default top-left anchor): letters no longer share a baseline,
#           and o/O, c/C, s/S become much harder because height is no longer a reliable cue.
```
- **In simple words:** Experiments for you.
- **Exactly:** Comments only.
- **Why it is here:** Practice.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `x[:, None]` | add a channel axis | `(N, 32, 32) -> (N, 1, 32, 32)` |
| `torch.tensor(np.stack(list))` | list of arrays to one tensor | `(558, 32, 32)` |
| `@torch.no_grad()` | decorator: no autograd inside the function | `evaluate` |
| `a, b = x, y` / `theta[:, 0, 0], ... = ...` | tuple assignment, assigns several slices at once | `augment` |
| `[expr for _ in range(2)]` | `_` marks an unused loop variable | `tx, ty` |
| `Counter(pairs).most_common(8)` | multiset counting | confusion pairs |
| `f"{a}->{b}"` | f-string | pair labels |
| `lambda ch: glyph(ch, None, 22)` | one-line function | `G` |

## Libraries & functions
| Name | Library | What it does | Important args | Returns |
|---|---|---|---|---|
| `ImageDraw.text(xy, s, anchor="ls")` | Pillow | draw text with the baseline at y | `anchor` | None |
| `font.getlength(s)` | Pillow | advance width of a string in px | | float |
| `torch.cdist(a, b)` | torch | all pairwise Euclidean distances | `(n,d),(m,d)` | `(n, m)` |
| `F.affine_grid`, `F.grid_sample` | torch | warp images with per-sample matrices | `align_corners=False` | grid, images |
| `nn.Conv2d(in, out, k, padding)` | torch | 2-D convolution | `padding=1` keeps size for k=3 | tensor |
| `nn.MaxPool2d(2)` | torch | halve height and width | | tensor |
| `F.cross_entropy(logits, y)` | torch | softmax + negative log-likelihood | logits `(B, C)`, ints `(B,)` | scalar |
| `torch.optim.Adam(params, lr)` | torch | optimiser | `lr` | optimiser |

## Execution trace (real run)
```
fonts: train ['bundled', 'Arial.ttf', 'Courier New.ttf']  held-out Verdana.ttf
train glyphs (558, 1, 32, 32)  test glyphs (124, 1, 32, 32)
1-NN, 1 template/class: held-out font clean 0.565, held-out font + rotation + noise 0.363
1-NN, all 558 training glyphs as templates:  held-out font + rotation + noise 0.573
CNN trained WITHOUT augmentation: held-out font + rotation + noise 0.468
CNN trained WITH augmentation:    held-out font + rotation + noise 0.782  (clean held-out 0.895)   [13s for both trainings, 105,662 parameters]
most confused (true -> predicted): [('1->l', 2), ('I->l', 2), ('M->H', 2), ('O->Q', 2), ('i->l', 2), ('5->S', 1), ('8->s', 1), ('9->6', 1)]
wrong predictions: 27 of 124 | by true class: {'1': 2, '8': 2, 'I': 2, 'J': 2, 'M': 2, 'O': 2}
pixel distance in the bundled font (smaller = more alike):
   0 vs O: 6.72
   l vs I: 1.92
   1 vs l: 7.41
   o vs O: 7.56
   5 vs S: 4.89
   m vs n: 10.68
   A vs B: 8.07
pixel distance after squeezing the strings to the same width: rn vs m 4.19,  m vs n 10.93,  rn vs nn 12.05
rn rendered width = 22 px, m rendered width = 20 px (the width is the clue a segmenter has)
```
- **fonts / sets:** 558 training glyphs, 124 test glyphs from a font the models never saw.
- **1-NN:** 0.565 on clean Verdana with one template per class, falling to 0.363 when the test glyphs are rotated and noisy. 558 templates: 0.573.
- **CNN:** 0.468 without augmentation, 0.782 with; clean held-out 0.895.
- **Confusions:** vertical strokes (`1`, `I`, `i`, `l`) and size-only pairs (`O`/`Q`, `M`/`H`).
- **Distances:** `l` vs `I` 1.92 is the smallest by far. `rn` vs `m` at 4.19 is closer than `m` vs `n`.

## Experiments (change ONE thing)
1. Augmentation strength 1.0 -> 2.0 in `train` (`augment(xb, g, 2.0)`): held-out accuracy falls from 0.782 to 0.677 (clean held-out 0.726). Too strong a warp makes the training glyphs unlike real text.
2. Remove the baseline anchor (`anchor="la"`, align by the top of the letters instead): with augmentation 0.081, without 0.024, and the pair distances collapse (`0` vs `O` 0.54, `m` vs `n` 0.00). Letters no longer share a baseline, so height cues disappear and glyphs of different heights overlap in the frame.
3. Compare 1-NN with all 558 templates (0.573) to the CNN without augmentation (0.468): memorising more pixels is not better than learning, and a CNN trained on unwarped data can be worse than 1-NN on warped test data.

## Debugging practice
BREAK IT: pass images without the channel axis, e.g. `model(Xte[:, 0])` of shape `(124, 32, 32)`:
```
  File ".../torch/nn/modules/conv.py", line 560, in _conv_forward
    return F.conv2d(
RuntimeError: Given groups=1, weight of size [8, 1, 3, 3], expected input[1, 5, 32, 32] to have 1 channels, but got 5 channels instead
```
Read it: PyTorch treats a 3-D input as an unbatched `(C, H, W)` image, so `(5, 32, 32)` is "5 channels, no batch". The convolution wants 1 channel. The fix is `x[:, None]` to create `(B, 1, H, W)`. Always print `.shape` at the model boundary.

## Rebuild checklist
- [ ] render a glyph on a baseline in a fixed frame
- [ ] 1-NN with `torch.cdist` and `argmin`
- [ ] random affine augmentation with `affine_grid` and `grid_sample`
- [ ] a 2-conv-block CNN and a cross-entropy training loop
- [ ] evaluate on a held-out font, not on the training fonts
- [ ] list the top confusion pairs and say what resolves each one

## The new problem (cliffhanger)
This only works if step 03 cut perfectly: touching or broken letters, connected scripts and handwriting break the cut, and even a perfect classifier cannot read `rn` as `m` or `m` as `rn`. We need a recogniser that reads the whole word image and decides the letter boundaries itself. Next: [step 05, CRNN and CTC](../05_crnn_and_ctc/WALKTHROUGH.md).

> **Say it aloud (interview box):** Template matching compares pixels and breaks as soon as the font, shift or noise changes. A small CNN trained with augmentation, such as random rotation, scale, shift and noise, generalised much better to an unseen font in my test (0.78 against 0.47 without augmentation). The remaining errors are pairs like `l`, `I`, `1` and `rn`/`m` that shape cannot separate, so real OCR adds a language prior, and modern systems read whole lines instead of cutting characters.
