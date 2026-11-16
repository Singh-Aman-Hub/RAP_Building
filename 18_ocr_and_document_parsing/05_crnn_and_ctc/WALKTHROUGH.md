# Step 05 — CRNN and CTC: reading a word without cutting letters

## The story so far
[Step 04](../04_character_recognition/WALKTHROUGH.md) classified single characters, but only after step 03 had cut them out perfectly. The cut fails on touching letters (`rn`), broken letters and connected scripts, and the classifier cannot tell `l` from `I` without context.

## The pain point
Where does one letter end and the next begin? In a word image of width 96 px with 3 letters, nobody labelled "the first letter is columns 5 to 25". Labelling every character box in a million scans is too expensive, and the whole idea of cutting is fragile.

## The spark
Do not cut. Slide along the word image and let the network output a letter (or "nothing") for every vertical slice. Many slices will show the same letter, and many will show nothing. A simple rule, **CTC** (Connectionist Temporal Classification, Graves et al. 2006), turns that long output into the word and lets the loss be computed from only the final text, summing over all possible alignments.

## Goal
Implement CTC collapsing, verify CTC loss against a brute-force sum over all paths, implement greedy and prefix-beam decoding from scratch, then train a tiny CRNN (convolutions + bidirectional GRU + `nn.CTCLoss`) on synthetic words and test it on an unseen font.

## The theory (read this before the interview)
**The model: CRNN.** Shi et al. 2015. A CNN turns the image (height 32, width 96) into a feature map; each **column** becomes one feature vector; a bidirectional recurrent layer (LSTM or GRU) lets each column see its left and right context; a linear layer gives scores over `C = num_characters + 1` classes per column. The `+1` is the **blank**. Output shape `(T, C)`, here `T = 24` time steps for a 96 px wide image.
```
image (1, 32, 96) -> conv/pool -> (32, 8, 24) -> columns (24, 256) -> BiGRU (24, 128) -> linear (24, 37)
```

**CTC.** The network emits one class per time step. A **path** is such a sequence, for example `- j j d - - k o - 3`. The mapping `B` from path to text has two rules:
```
1. merge consecutive repeated symbols      "aab-b"  -> "ab-b"
2. delete the blanks                       "ab-b"   -> "abb"
```
The blank is what lets the network spell a double letter (`a - a` -> `aa`) and what lets it say "nothing here" between letters. Many paths give the same text. CTC defines
```
P(text | image) = sum of P(path | image) over all paths pi with B(pi) = text
P(path | image) = product over t of  p_t(pi_t)          (the steps are treated as independent given the image)
loss = - log P(text | image)
```
That sum has an exponential number of terms, but the forward-backward dynamic programme (like an HMM) computes it in `O(T * len(label))`. `nn.CTCLoss` implements it. In this step we check it against the brute-force sum: for `T = 4` and the label `ab` there are 15 valid paths, and both calculations give 1.2996.

**Requirements.** `T >= len(label)`, and `T >= len(label) + number of repeated adjacent letters` (a repeat needs a blank between). Otherwise no valid path exists, the loss is infinite, and `zero_infinity=True` replaces it with 0.

**Decoding.**
| Method | How | Cost | Weakness |
|---|---|---|---|
| Greedy (best path) | argmax per step, then collapse | `O(T C)` | It finds the best *path*, not the best *text*: many small paths of "a" can beat one big path of "blank" |
| Prefix beam search | keep the `k` best collapsed prefixes, each with `p_blank` and `p_non-blank`, summing all paths per prefix | `O(T k C)` | needs a beam width; can add a language model score |
| Beam + language model / lexicon | add an n-gram or dictionary score to prefixes | more | needs an LM; pays off on noisy text |

Toy proof in `main.py`: two steps with `P(blank)=0.6`, `P(a)=0.4` each. Greedy says blank, blank, so `""` with path probability 0.36. But `P("a") = P(aa) + P(a-) + P(-a) = 0.16 + 0.24 + 0.24 = 0.64`. Beam search finds `a`.

**Training quirks.** CTC has an early plateau: the loss drops to about 4 and the network outputs only blanks (every step is `blank`, which is the safest guess) before it "discovers" the letters. In this step's run the loss stayed near 4.1 for 200 steps and fell below 1 by step 600. More steps, BatchNorm and a sensible learning rate shorten it. The log-probabilities must be `(T, B, C)` after `log_softmax`.

**What came after CRNN.** Same line-recognition idea, stronger models:
- **Attention encoder-decoder** (autoregressive): the model generates characters one at a time, attending to the image.
- **TrOCR** (Microsoft, 2021): ViT image encoder + text transformer decoder, pre-trained and fine-tuned on printed and handwritten lines.
- **PARSeq** (2022): permuted autoregressive sequence model for scene text; iterative refinement.
- **SVTR** (PaddleOCR PP-OCRv3/v4/v5 recognisers) keeps CTC with a transformer-style backbone; Tesseract 4/5 uses an LSTM line recogniser with CTC.
- **VLMs** (step 09) read an entire page as a sequence and decode text with a language model, with no CTC. The price is hallucination risk (it can output plausible text that is not on the page).

**Why CTC is still alive.** It is fast (one forward pass, no autoregression), it needs no character boxes, it is monotonic (output order = image order, so text cannot be reordered or invented as easily as in a generative decoder), and the per-frame probabilities give a natural confidence.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| CRNN | conv + recurrent + CTC, "Shi et al." recogniser | yes | CNN features, BiLSTM or BiGRU, CTC loss |
| CTC | Connectionist Temporal Classification, "blank-based alignment-free loss" | yes | Sums over every alignment of text to frames |
| Blank token | epsilon, `<b>`, `-` | yes | The "no character here" symbol; not a space |
| Greedy decoding | best-path decoding, argmax decoding | yes | One path, no search |
| Beam search (CTC) | prefix beam search | near | Merges paths with the same collapsed prefix; ordinary beam search does not |
| Line recognition | text recognition, "recogniser" (vs "detector") | near | Reads an already cropped text line; detection finds the boxes |
| Seq2seq OCR | attention OCR, TrOCR, PARSeq | near | Generates characters one at a time instead of aligning frames |
| Whole-page VLM OCR | OCR-free, end-to-end document parsing | no | Skips detection and line cropping; see step 09 |

## Run it
```bash
cd 18_ocr_and_document_parsing/05_crnn_and_ctc
python main.py       # ~1-2 minutes: CPU time about 80 s; slower on a busy machine
```

## Code, step by step

### 1. Imports, seeds, threads  (`main.py` lines 15–34)
```python
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
```
- **In simple words:** Load tools and fix randomness.
- **Exactly:** `itertools` enumerates all alignments in the brute-force check; `defaultdict` is used by the beam search; `torch.set_num_threads(2)` limits CPU threads; `docimg.get_font` and `available_system_fonts` provide fonts.
- **Why it is here:** Seeds make the training curve and accuracies reproducible.

### 2. The alphabet and the blank  (`main.py` lines 36–40)
```python
# ---- 1. The alphabet: index 0 is the CTC "blank", it is not a character ---------------------------------------
ALPHABET = string.ascii_lowercase + string.digits           # 36 characters
BLANK = 0
IDX = {c: i + 1 for i, c in enumerate(ALPHABET)}            # 'a' -> 1 ... '9' -> 36
N_CLASSES = len(ALPHABET) + 1                               # 37 outputs per time step
```
- **In simple words:** 36 letters and digits, plus one special "nothing" symbol at index 0.
- **Exactly:** `IDX = {'a': 1, ..., '9': 36}`; `N_CLASSES = 37`. Index 0 is reserved for the blank, which is the default `nn.CTCLoss(blank=0)`.
- **Why it is here:** Mixing up the blank index with a real character is the classic CTC bug.

### 3. Collapsing a path into text  (`main.py` lines 43–55)
```python
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
```
- **In simple words:** Squash repeats, then drop the "nothing" symbols.
- **Exactly:** The loop appends a symbol only if it differs from the previous symbol and is not the blank; `prev` is updated for every step. `show` prints `-` for the blank. Output: `aa-bb` -> `ab`; `a-a` -> `aa`; `aaa` -> `a`; `-a-a-` -> `aa`.
- **Why it is here:** This one function defines CTC. `a-a` -> `aa` versus `aaa` -> `a` is exactly why the blank exists.

### 4. CTC loss by brute force, checked against PyTorch  (`main.py` lines 57–68)
```python
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
```
- **In simple words:** List every possible way to fill 4 time steps that spells "ab", add up their probabilities, and compare with the library.
- **Exactly:** `toy` is `(4, 3)` random probabilities. `itertools.product(range(3), repeat=4)` yields 81 paths; 15 collapse to `[1, 2]`. `total` is the sum of products of per-step probabilities, `-log(total)` = 1.2996. `nn.CTCLoss(reduction="sum")` takes `log_probs (T, N, C) = (4, 1, 3)`, targets `[1, 2]`, input length 4, target length 2, and returns the same 1.2996.
- **Why it is here:** It proves that the CTC loss is nothing more than "-log of the sum over valid alignments".

### 5. Greedy decoding  (`main.py` lines 72–74)
```python
def greedy_decode(probs):
    """probs (T, C) -> best class per step, then collapse."""
    return collapse(probs.argmax(axis=1).tolist())
```
- **In simple words:** At each step take the most likely symbol, then collapse.
- **Exactly:** `probs.argmax(axis=1)` is `(T,)`; `collapse` turns it into the label.
- **Why it is here:** It is the fast default; one line.

### 6. Prefix beam search from scratch  (`main.py` lines 77–97)
```python
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
```
- **In simple words:** Keep several candidate spellings alive. For each, remember how likely it is when its last frame was "nothing" and when it was a letter, so repeated letters are handled correctly.
- **Exactly:** `beams` maps a prefix tuple to `(p_blank, p_nonblank)`. For each time step and each class with probability above `prune`: a blank extends nothing and moves mass to `p_blank`; a new character extends the prefix; an equal-to-last character either extends the prefix (only from a path that ended in a blank, `p * pb`) or stays the same prefix (from a path that ended in the letter, `p * pnb`). After each step only the `beam` highest-scoring prefixes survive (`sorted(...)[:beam]`). The function returns the best prefix and its total probability.
- **Why it is here:** It sums over all paths that share a collapsed prefix, which greedy search cannot do.

### 7. A toy where greedy is wrong  (`main.py` lines 100–104)
```python
two_steps = np.array([[0.6, 0.4], [0.6, 0.4]])               # classes: blank, 'a'; two time steps
g = greedy_decode(two_steps)
b, pb_ = prefix_beam_search(two_steps)
print(f"toy: greedy decodes {show(g)!r} (path prob 0.6*0.6=0.36) but prefix beam finds {show(b)!r} "
      f"(sum over paths aa, a-, -a = {pb_:.2f})")
```
- **In simple words:** Two time steps, each 60% blank and 40% "a". Greedy says nothing was written. Beam search says "a".
- **Exactly:** Greedy: blank, blank, probability 0.36 for `""`. Beam search sums the paths `aa`, `a-`, `-a`: 0.16 + 0.24 + 0.24 = 0.64 for `"a"`.
- **Why it is here:** It is the standard interview example of why best path is not best text.

### 8. Fonts and image size  (`main.py` lines 106–110)
```python
# ---- 4. Synthetic word images: 32 x 96 px, random word, random font size and offset -----------------------------------
sysf = available_system_fonts()
TRAIN_FONTS = [None] + sysf[:2]                               # bundled font, Arial, Courier New
TEST_FONT = sysf[2] if len(sysf) > 2 else None                # Verdana: never seen in training
H, W = 32, 96
```
- **In simple words:** Train on three fonts, test on a fourth. Every image is 32 pixels high and 96 wide.
- **Exactly:** The same font split as step 04 (bundled, Arial, Courier New for training; Verdana held out). `H, W = 32, 96`.
- **Why it is here:** The width fixes `T`: two 2x2 poolings give 96 / 4 = 24 time steps.

### 9. Render one word  (`main.py` lines 113–116)
```python
def render_word(word, font_path, size, dx):
    im = Image.new("L", (W, H), 255)
    ImageDraw.Draw(im).text((dx, 24), word, fill=0, font=get_font(size, font_path), anchor="ls")
    return np.asarray(im, dtype=np.float32) / 255.0
```
- **In simple words:** Draw a word in black on white on the baseline at row 24, starting a few pixels from the left.
- **Exactly:** `anchor="ls"` places the baseline; returns `float32 (32, 96)` in 0..1.
- **Why it is here:** Synthetic words give unlimited free, perfectly labelled data.

### 10. Build 3,000 training words and 300 test words  (`main.py` lines 119–133)
```python
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
```
- **In simple words:** Pick random 2 to 5 character words, a random font and size, and render them.
- **Exactly:** `r.integers(2, 6)` gives lengths 2 to 5. Train: `(3000, 1, 32, 96)`. Test: `(300, 1, 32, 96)` in Verdana. Seeds 1 and 2 keep the sets separate. `sample_words.png` shows six test images, for example `jdko3`, `v30`, `bujhx`.
- **Why it is here:** Real handwritten or scanned data is not needed to learn the mechanics.

### 11. The CRNN model  (`main.py` lines 137–152)
```python
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
```
- **In simple words:** A small convolution stack describes the image, then a two-way GRU reads the description left to right and right to left, then a linear layer scores 37 classes per column.
- **Exactly:** After two conv + BatchNorm + ReLU + `MaxPool2d(2)` blocks the feature map is `(B, 32, 8, 24)`. `permute(0, 3, 1, 2)` gives `(B, 24, 32, 8)` and `reshape` to `(B, 24, 256)`: one 256-vector per column. `GRU(256, 64, bidirectional=True, batch_first=True)` returns `(B, 24, 128)`. `Linear(128, 37)` returns `(B, 24, 37)` logits. 133,317 parameters in total.
- **Why it is here:** Width turns into time. That is the whole trick that lets a sequence model read an image without boxes.

### 12. Light augmentation  (`main.py` lines 155–165)
```python
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
```
- **In simple words:** Tilt, shift and add grain to each batch of word images, like step 04.
- **Exactly:** Rotation of +/-3 degrees, horizontal shift of +/-4%, Gaussian noise 0.08. The `H / W` factors in `theta` correct for the image not being square. `padding_mode="border"` repeats the white border instead of adding black corners.
- **Why it is here:** Without it the model overfits the 3,000 training images.

### 13. Model, optimiser, loss  (`main.py` lines 169–173)
```python
model = CRNN()
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)            # zero_infinity: ignore impossible alignments instead of inf
g = torch.Generator().manual_seed(0)
STEPS, BATCH = 800, 48
```
- **In simple words:** Create the network, the optimiser and the CTC loss.
- **Exactly:** `nn.CTCLoss(blank=0, zero_infinity=True)`. `STEPS, BATCH = 800, 48`. Adam, lr 2e-3.
- **Why it is here:** `zero_infinity` stops one impossible sample (label longer than the frames) from turning the loss into `inf` and poisoning the weights.

### 14. The training loop  (`main.py` lines 175–189)
```python
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
```
- **In simple words:** For 800 steps: take a batch, warp it, compute the loss from the true words, update the weights.
- **Exactly:** `targets` is all labels of the batch concatenated into one 1-D tensor (the layout `CTCLoss` accepts), `tgt_len` has the length of each label `(48,)`. `model(xb)` is `(48, 24, 37)`; `log_softmax(2)` normalises over classes; `permute(1, 0, 2)` makes it `(24, 48, 37)` = `(T, B, C)`. `in_len` is 24 for every sample. The printed loss falls from 23.2 (step 1) to 4.1 (step 100) and stays there until step 300 (the blank-only plateau), then falls to 0.48 at step 800.
- **Why it is here:** This is the entire training recipe: images and text in, no alignment needed.

### 15. Edit distance for CER  (`main.py` lines 193–200)
```python
def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
```
- **In simple words:** Count the minimum number of single-letter edits that turn one string into another.
- **Exactly:** Standard dynamic programme with one row of memory: substitution, insertion, deletion each cost 1. Step 06 does this properly.
- **Why it is here:** Character error rate needs it.

### 16. Evaluate greedy and beam on the unseen font  (`main.py` lines 203–212)
```python
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
```
- **In simple words:** Predict the 300 held-out words two ways and score them.
- **Exactly:** `model.eval()` fixes BatchNorm statistics. `P` is `(300, 24, 37)` probabilities. Exact match is the fraction of words predicted perfectly; CER = total edit distance / total reference characters. Greedy: exact match 0.673, CER 0.122. Prefix beam (5): 0.687, 0.117.
- **Why it is here:** Two metrics: a word-level one (strict) and a character-level one (graded).

### 17. Show predictions and the raw frames  (`main.py` lines 213–214)
```python
print("sample (truth -> greedy):", [(w, p) for w, p in list(zip(wte, greedy_preds))[:8]])
print("frames of the first test word (argmax per step):", show(P[0].argmax(axis=1).tolist()), "->", show(collapse(P[0].argmax(axis=1).tolist())), "| truth:", wte[0])
```
- **In simple words:** Print a few predictions and the per-column guesses for the first word.
- **Exactly:** The first test word `jdko3` has frames `-jjd---k-o---3----------`: repeated letters (`jj`) and blanks collapse to `jdko3`.
- **Why it is here:** You can see CTC working: letters appear where the glyphs are, blanks everywhere else.

### 18. TRY IT and BREAK IT notes  (`main.py` lines 216–218)
```python
# TRY IT:   train for 900 steps (or 200) and watch CER; replace GRU with LSTM; widen the image to 128 so T = 32.
# BREAK IT: pass log_probs of shape (B, T, C) (skip the permute): 'RuntimeError: input_lengths must be of size batch_size'.
#           Or feed raw logits without log_softmax (the loss becomes meaningless), or use an image so narrow that T < len(label).
```
- **In simple words:** Experiments for you.
- **Exactly:** Comments only.
- **Why it is here:** Practice.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `itertools.product(range(C), repeat=T)` | all `C^T` sequences | 81 paths for C=3, T=4 |
| `defaultdict(lambda: [0.0, 0.0])` | dict with a default value per new key | beam probabilities |
| `prefix + (c,)` | extend a tuple | `(1, 2) + (3,)` |
| `f"{x!r:>9}"` | repr, right-aligned in 9 columns | collapse table |
| `x.permute(1, 0, 2)` | reorder axes | `(B, T, C) -> (T, B, C)` |
| `torch.full((B,), T, dtype=torch.long)` | tensor filled with a value | input lengths |
| `sorted(d.items(), key=lambda kv: -score)[:k]` | top-k by score | beam pruning |

## Libraries & functions
| Name | Library | What it does | Important args | Returns |
|---|---|---|---|---|
| `nn.CTCLoss(blank, reduction, zero_infinity)` | torch | CTC loss | `log_probs (T,N,C)`, `targets`, `input_lengths (N,)`, `target_lengths (N,)` | scalar |
| `nn.GRU(in, hidden, bidirectional, batch_first)` | torch | recurrent layer | `batch_first=True` gives `(B, T, F)` | `(out, h_n)` |
| `nn.BatchNorm2d(c)` | torch | normalise per channel; needs `eval()` at test time | | tensor |
| `Tensor.log_softmax(dim)` | torch | log probabilities | `dim=2` | tensor |
| `np.flatnonzero(cond)` | numpy | indices where true | | array |
| `Tensor.reshape`, `.permute` | torch | change shape / axis order | | tensor |

## Execution trace (real run)
```
collapse(  'aa-bb') = 'ab'
collapse(    'a-a') = 'aa'
collapse(    'aaa') = 'a'
collapse(  '-a-a-') = 'aa'
valid alignments of 'ab' in 4 steps: 15;  brute-force loss 1.2996  vs nn.CTCLoss 1.2996
toy: greedy decodes '' (path prob 0.6*0.6=0.36) but prefix beam finds 'a' (sum over paths aa, a-, -a = 0.64)
train (3000, 1, 32, 96) test (300, 1, 32, 96) e.g. test words: ['jdko3', 'v30', 'bujhx', 'f0p', 'phw78', 'obg']
step    1  CTC loss 23.236
step  100  CTC loss 4.131
step  200  CTC loss 3.955
step  300  CTC loss 3.583
step  400  CTC loss 2.637
step  500  CTC loss 1.605
step  600  CTC loss 1.014
step  700  CTC loss 0.632
step  800  CTC loss 0.481
trained in 55s, 133,317 parameters
held-out font, greedy           exact-match 0.673   CER 0.122
held-out font, prefix beam (5)  exact-match 0.687   CER 0.117
sample (truth -> greedy): [('jdko3', 'jdko3'), ('v30', 'v0'), ('bujhx', 'bujhx'), ('f0p', 'f0p'), ('phw78', 'phww78'), ('obg', 'obg'), ('sy65', 'sy65'), ('67i', '67i')]
frames of the first test word (argmax per step): -jjd---k-o---3---------- -> jdko3 | truth: jdko3
```
- **collapse lines:** the four rules in action.
- **15 alignments:** brute force and PyTorch agree to four decimals.
- **toy:** greedy `''` (0.36), beam `a` (0.64).
- **loss:** 23.2 -> 4.1 plateau (blank-only) -> below 1 -> 0.48. The plateau is normal.
- **held-out font:** greedy 0.673 exact, CER 0.122; beam search is slightly better on both because it sums paths. Errors are doubled letters (`phww78`), dropped letters (`v0`) and look-alikes, as step 04 predicted.

## Experiments (change ONE thing)
1. `STEPS = 400` instead of 800: the run ends on the plateau. Loss 2.6 at step 400, greedy exact-match 0.003 and CER 0.955 (almost every prediction is empty), beam search 0.020 and 0.836. A model that has not left the blank-only phase looks completely broken, but it is only under-trained.
2. `STEPS = 1400`: loss 0.175, greedy exact-match 0.827 and CER 0.065, beam 0.833 and 0.063. The trend from 400 to 800 to 1400 steps is exact-match 0.003, 0.673, 0.827. Training took 315 s on a busy machine, which is why `main.py` stops at 800.
3. Not run here, for you: replace `nn.GRU` with `nn.LSTM` (return value `(out, (h, c))`, so `out, _ = ...` still works); widen the image to 128 so T = 32.

## Debugging practice
BREAK IT: skip the permute, so `log_probs` is `(B, T, C)` instead of `(T, B, C)`:
```
  File ".../torch/nn/functional.py", line 3165, in ctc_loss
    return torch.ctc_loss(
RuntimeError: input_lengths must be of size batch_size
```
Read it: `CTCLoss` takes the **second** axis as the batch, so it thinks the batch is 24 (the time steps), but `in_len` has 4 (or 48) entries. If the sizes happened to coincide there would be no error and the loss would be silently wrong. Always check `log_probs.shape == (T, B, C)`. Another silent failure: passing logits instead of `log_softmax` output.

## Rebuild checklist
- [ ] `collapse(path)` with the repeat rule and the blank rule
- [ ] explain why `a-a` and `aa` differ, and why the blank is needed
- [ ] brute-force CTC for a tiny case; say what `nn.CTCLoss` computes
- [ ] greedy decoding; prefix beam search with `p_blank` / `p_non-blank`
- [ ] CRNN shapes: image -> columns -> `(B, T, C)` -> `(T, B, C)` for the loss
- [ ] name the CTC length constraint and what `zero_infinity` does

## The new problem (cliffhanger)
The CRNN has CER 0.12 and exact-match 0.67. Is that good? A CER of 0.12 on a page of 2,000 characters means 240 wrong characters, and word error rate is higher still. We have not defined these metrics properly, and a scanned invoice also needs field-level and table-level metrics. Next: [step 06, OCR evaluation](../06_ocr_evaluation/WALKTHROUGH.md).

> **Say it aloud (interview box):** A CRNN turns the image width into time: a CNN makes one feature vector per column, a BiLSTM or BiGRU adds context, and a linear layer outputs a probability over characters plus a blank at each step. CTC sums over all alignments that collapse to the target text, where collapsing merges repeats and deletes blanks, so I need no character boxes. Greedy decoding takes the best path and beam search sums paths per prefix. Newer recognisers such as TrOCR and PARSeq use attention decoders, and VLM OCR generates the whole page, which brings hallucination risk.
