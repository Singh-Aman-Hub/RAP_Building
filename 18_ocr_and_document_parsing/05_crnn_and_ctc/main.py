"""
GOAL:  Read whole word images WITHOUT cutting characters: a CRNN (conv + BiGRU) trained with CTC loss.
       First CTC is built by hand (collapsing, a brute-force loss check, greedy and prefix-beam decoding),
       then a tiny CRNN is trained on synthetic words and tested on an unseen font.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 18_ocr_and_document_parsing/05_crnn_and_ctc/main.py
EXPECTED OUTPUT (shortened):
    collapse examples ...   brute-force CTC loss == nn.CTCLoss: ...
    toy: greedy decodes '' (p=0.36) but prefix beam finds 'a' (p=0.64)
    step 100 loss ...  ... held-out font: greedy exact-match ... CER ...   beam ...
    sample predictions: ...
NEEDS: numpy, torch, Pillow.  About 1-2 minutes on a laptop CPU (CPU time ~40 s; slower on a busy machine).
"""
import itertools
import string
import sys
import time
from collections import defaultdict
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
torch.set_num_threads(2)
rng = np.random.default_rng(0)

# ---- 1. The alphabet: index 0 is the CTC "blank", it is not a character ---------------------------------------
ALPHABET = string.ascii_lowercase + string.digits           # 36 characters
BLANK = 0
IDX = {c: i + 1 for i, c in enumerate(ALPHABET)}            # 'a' -> 1 ... '9' -> 36
N_CLASSES = len(ALPHABET) + 1                               # 37 outputs per time step


def collapse(path):
    """CTC rule: 1) merge repeated symbols  2) delete blanks.   path: list of class ids -> list of class ids."""
    out, prev = [], None
    for p in path:
        if p != prev and p != BLANK:
            out.append(p)
        prev = p
    return out


show = lambda ids: "".join(ALPHABET[i - 1] if i else "-" for i in ids)
for p in [[1, 1, 0, 2, 2], [1, 0, 1], [1, 1, 1], [0, 1, 0, 1, 0]]:
    print(f"collapse({show(p)!r:>9}) = {show(collapse(p))!r}")

# ---- 2. CTC loss by brute force on a tiny problem: T=4 steps, classes {-, a, b}, label "ab" -------------------------
T, C = 4, 3
toy = torch.softmax(torch.randn(T, C), dim=1)                # per-step probabilities, rows sum to 1
target = [1, 2]                                              # "ab"
total = 0.0
paths = [p for p in itertools.product(range(C), repeat=T) if collapse(list(p)) == target]
for p in paths:                                              # every alignment that collapses to "ab"
    total += float(np.prod([toy[t, p[t]] for t in range(T)]))
loss_brute = -np.log(total)                                  # CTC loss = -log( sum of probabilities of all valid paths )
loss_torch = nn.CTCLoss(blank=0, reduction="sum")(toy.log().unsqueeze(1), torch.tensor([target]),
                                                  torch.tensor([T]), torch.tensor([2])).item()
print(f"valid alignments of 'ab' in {T} steps: {len(paths)};  brute-force loss {loss_brute:.4f}  vs nn.CTCLoss {loss_torch:.4f}")


# ---- 3. Greedy decoding and prefix beam search, both from scratch --------------------------------------------------
def greedy_decode(probs):
    """probs (T, C) -> best class per step, then collapse."""
    return collapse(probs.argmax(axis=1).tolist())


def prefix_beam_search(probs, beam=5, prune=1e-3):
    """Keep the `beam` best PREFIXES (collapsed strings). For each prefix track two probabilities:
    p_b = prefix ends in blank,  p_nb = prefix ends in its last character."""
    beams = {(): (1.0, 0.0)}
    for t in range(probs.shape[0]):
        nxt = defaultdict(lambda: [0.0, 0.0])
        for prefix, (pb, pnb) in beams.items():
            for c in np.flatnonzero(probs[t] > prune):       # ignore classes with negligible probability
                p = probs[t, c]
                if c == BLANK:                               # a blank: prefix unchanged, now ends in blank
                    nxt[prefix][0] += p * (pb + pnb)
                    continue
                ext = prefix + (int(c),)
                if prefix and prefix[-1] == c:               # same char again
                    nxt[ext][1] += p * pb                    #   new copy only if a blank separated them
                    nxt[prefix][1] += p * pnb                #   otherwise it is the same character continuing
                else:
                    nxt[ext][1] += p * (pb + pnb)
        beams = dict(sorted(nxt.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:beam])
    best = max(beams.items(), key=lambda kv: kv[1][0] + kv[1][1])
    return list(best[0]), best[1][0] + best[1][1]


two_steps = np.array([[0.6, 0.4], [0.6, 0.4]])               # classes: blank, 'a'; two time steps
g = greedy_decode(two_steps)
b, pb_ = prefix_beam_search(two_steps)
print(f"toy: greedy decodes {show(g)!r} (path prob 0.6*0.6=0.36) but prefix beam finds {show(b)!r} "
      f"(sum over paths aa, a-, -a = {pb_:.2f})")

# ---- 4. Synthetic word images: 32 x 96 px, random word, random font size and offset -----------------------------------
sysf = available_system_fonts()
TRAIN_FONTS = [None] + sysf[:2]                               # bundled font, Arial, Courier New
TEST_FONT = sysf[2] if len(sysf) > 2 else None                # Verdana: never seen in training
H, W = 32, 96


def render_word(word, font_path, size, dx):
    im = Image.new("L", (W, H), 255)
    ImageDraw.Draw(im).text((dx, 24), word, fill=0, font=get_font(size, font_path), anchor="ls")
    return np.asarray(im, dtype=np.float32) / 255.0


def make_words(n, fonts, seed):
    r = np.random.default_rng(seed)
    X, words = [], []
    for _ in range(n):
        w = "".join(r.choice(list(ALPHABET), size=int(r.integers(2, 6))))     # 2 to 5 characters
        f = fonts[int(r.integers(len(fonts)))]
        X.append(render_word(w, f, int(r.integers(18, 25)), int(r.integers(3, 9))))
        words.append(w)
    return torch.tensor(np.stack(X))[:, None], words           # (n, 1, 32, 96), list[str]


Xtr, wtr = make_words(3000, TRAIN_FONTS, seed=1)
Xte, wte = make_words(300, [TEST_FONT], seed=2)
save_png((np.concatenate([Xte[i, 0].numpy() for i in range(6)], axis=1) * 255).astype(np.uint8), HERE / "sample_words.png")
print("train", tuple(Xtr.shape), "test", tuple(Xte.shape), "e.g. test words:", wte[:6])


# ---- 5. The CRNN: conv features -> BiGRU over the width axis -> per-step class scores ----------------------------------------
class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),    # (B, 16, 16, 48)
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),   # (B, 32, 8, 24)
        )
        self.rnn = nn.GRU(32 * 8, 64, bidirectional=True, batch_first=True)   # input = 256 features per column
        self.fc = nn.Linear(2 * 64, N_CLASSES)

    def forward(self, x):                                  # x: (B, 1, 32, 96)
        f = self.cnn(x)                                    # (B, 32, 8, 24): height 8, width 24 = 24 time steps
        B, C, Hh, Tt = f.shape
        f = f.permute(0, 3, 1, 2).reshape(B, Tt, C * Hh)   # (B, 24, 256): one feature vector per column
        out, _ = self.rnn(f)                               # (B, 24, 128): both directions concatenated
        return self.fc(out)                                # (B, 24, 37) logits per time step


def augment(x, gen):
    """Light augmentation: small rotation, shift, noise (same idea as step 04)."""
    n = x.shape[0]
    ang = (torch.rand(n, generator=gen) * 2 - 1) * 3 * np.pi / 180
    tx = (torch.rand(n, generator=gen) * 2 - 1) * 0.04
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = torch.cos(ang), -torch.sin(ang) * H / W, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = torch.sin(ang) * W / H, torch.cos(ang), 0
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    out = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
    return (out + torch.randn(out.shape, generator=gen) * 0.08).clamp(0, 1)


# ---- 6. Training with nn.CTCLoss ------------------------------------------------------------------------------------------------
model = CRNN()
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)            # zero_infinity: ignore impossible alignments instead of inf
g = torch.Generator().manual_seed(0)
STEPS, BATCH = 800, 48
t0 = time.time()
for step in range(1, STEPS + 1):
    idx = torch.randint(0, len(Xtr), (BATCH,), generator=g)
    xb = augment(Xtr[idx], g)
    words = [wtr[i] for i in idx.tolist()]
    targets = torch.tensor([IDX[c] for w in words for c in w])                 # all labels concatenated: (sum of lengths,)
    tgt_len = torch.tensor([len(w) for w in words])                            # (B,)
    log_probs = model(xb).log_softmax(2).permute(1, 0, 2)                      # (T, B, C): CTCLoss wants time first
    in_len = torch.full((BATCH,), log_probs.shape[0], dtype=torch.long)        # every image has T = 24 steps
    loss = ctc(log_probs, targets, in_len, tgt_len)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 100 == 0 or step == 1:
        print(f"step {step:4d}  CTC loss {loss.item():.3f}")
print(f"trained in {time.time() - t0:.0f}s, {sum(p.numel() for p in model.parameters()):,} parameters")


# ---- 7. Evaluate on the unseen font: greedy vs prefix beam search --------------------------------------------------------------
def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


model.eval()
with torch.no_grad():
    P = model(Xte).softmax(2).numpy()                      # (300, 24, 37) probabilities
for name, dec in [("greedy", lambda p: greedy_decode(p)), ("prefix beam (5)", lambda p: prefix_beam_search(p, 5)[0])]:
    preds = ["".join(ALPHABET[i - 1] for i in dec(p)) for p in P]
    exact = np.mean([p == w for p, w in zip(preds, wte)])
    cer = sum(edit_distance(p, w) for p, w in zip(preds, wte)) / sum(len(w) for w in wte)
    print(f"held-out font, {name:<16} exact-match {exact:.3f}   CER {cer:.3f}")
    if name == "greedy":
        greedy_preds = preds
print("sample (truth -> greedy):", [(w, p) for w, p in list(zip(wte, greedy_preds))[:8]])
print("frames of the first test word (argmax per step):", show(P[0].argmax(axis=1).tolist()), "->", show(collapse(P[0].argmax(axis=1).tolist())), "| truth:", wte[0])

# TRY IT:   train for 900 steps (or 200) and watch CER; replace GRU with LSTM; widen the image to 128 so T = 32.
# BREAK IT: pass log_probs of shape (B, T, C) (skip the permute): 'RuntimeError: input_lengths must be of size batch_size'.
#           Or feed raw logits without log_softmax (the loss becomes meaningless), or use an image so narrow that T < len(label).
