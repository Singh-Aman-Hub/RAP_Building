"""
GOAL:  Recognise single characters two ways: (a) nearest-template matching and (b) a small CNN trained on
       synthetic glyphs with augmentation. Test on a font neither has seen, and study the confusion pairs
       (0/O, 1/l/I, rn/m).

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 18_ocr_and_document_parsing/04_character_recognition/main.py
EXPECTED OUTPUT (shortened):
    fonts: train [...]  held-out [...]
    1-NN templates on held-out font + noise:  acc ...
    CNN trained WITHOUT augmentation: ...     CNN WITH augmentation: ...
    most confused pairs: ...
    pixel distance rn vs m ...
NEEDS: numpy, torch, Pillow.  About 20-60 seconds on a laptop CPU (slower if the machine is busy).
"""
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

sysf = available_system_fonts()
TRAIN_FONTS = [None] + sysf[:2]                         # None = Pillow's bundled font; plus Arial and Courier New if present
TEST_FONT = sysf[2] if len(sysf) > 2 else None          # Verdana if present: never seen in training
print("fonts: train", [Path(f).name if f else "bundled" for f in TRAIN_FONTS],
      " held-out", Path(TEST_FONT).name if TEST_FONT else "bundled (no system fonts found)")


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


# ---- 3. Baseline: 1-nearest-neighbour template matching ---------------------------------------------------------
def nn_predict(queries, templates, tlabels):
    q = queries.flatten(1)                              # (n, 1024)
    t = templates.flatten(1)                            # (m, 1024)
    d = torch.cdist(q, t)                               # (n, m) Euclidean distances
    return tlabels[d.argmin(dim=1)]


gen = torch.Generator().manual_seed(123)
Xte_aug = augment(Xte, gen)                             # test glyphs: new font, rotated, noisy
templ_clean_default, tl_clean = make_set([None], [22])  # templates: ONE clean glyph per class, bundled font only
acc_clean = (nn_predict(Xte, templ_clean_default, tl_clean) == yte).float().mean().item()
acc_noisy = (nn_predict(Xte_aug, templ_clean_default, tl_clean) == yte).float().mean().item()
acc_multi = (nn_predict(Xte_aug, Xtr, ytr) == yte).float().mean().item()
print(f"1-NN, 1 template/class: held-out font clean {acc_clean:.3f}, held-out font + rotation + noise {acc_noisy:.3f}")
print(f"1-NN, all {len(Xtr)} training glyphs as templates:  held-out font + rotation + noise {acc_multi:.3f}")


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


@torch.no_grad()
def evaluate(model, x, y):
    pred = model(x).argmax(dim=1)
    return (pred == y).float().mean().item(), pred


t0 = time.time()
cnn_plain = train(use_aug=False)
cnn_aug = train(use_aug=True)
a_plain, _ = evaluate(cnn_plain, Xte_aug, yte)
a_aug, pred = evaluate(cnn_aug, Xte_aug, yte)
a_aug_clean, _ = evaluate(cnn_aug, Xte, yte)
print(f"CNN trained WITHOUT augmentation: held-out font + rotation + noise {a_plain:.3f}")
print(f"CNN trained WITH augmentation:    held-out font + rotation + noise {a_aug:.3f}  (clean held-out {a_aug_clean:.3f})"
      f"   [{time.time() - t0:.0f}s for both trainings, {sum(p.numel() for p in cnn_aug.parameters()):,} parameters]")

# ---- 5. Which characters get confused? --------------------------------------------------------------------------------
pairs = Counter((CLASSES[a], CLASSES[b]) for a, b in zip(yte.tolist(), pred.tolist()) if a != b)
print("most confused (true -> predicted):", [(f"{a}->{b}", n) for (a, b), n in pairs.most_common(8)])
print("wrong predictions:", sum(pairs.values()), "of", len(yte), "| by true class:", dict(Counter(a for (a, b), n in pairs.items() for _ in range(n)).most_common(6)))


# ---- 6. Why some pairs are hard: distance between rendered glyphs and between strings ----------------------------------------
def dist(a, b):
    return float(np.linalg.norm(a - b))


G = lambda ch: glyph(ch, None, 22)                      # bundled font, size 22
print("pixel distance in the bundled font (smaller = more alike):")
for a, b in [("0", "O"), ("l", "I"), ("1", "l"), ("o", "O"), ("5", "S"), ("m", "n"), ("A", "B")]:
    print(f"   {a} vs {b}: {dist(G(a), G(b)):.2f}")


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

# TRY IT:   raise the augmentation strength to 2.0 in train(); or train for 1500 steps and see whether held-out accuracy still moves.
# BREAK IT: remove `anchor="ls"` from `glyph` (use the default top-left anchor): letters no longer share a baseline,
#           and o/O, c/C, s/S become much harder because height is no longer a reliable cue.
