"""
GOAL:  Overfit a network on purpose (80 noisy training points), then apply each regularisation technique one at a
       time and read the train / validation numbers: L2 vs decoupled weight decay, dropout (train vs eval
       mode), early stopping, data augmentation and label smoothing.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 15_neural_networks/07_regularization/main.py
EXPECTED OUTPUT (shortened):
    method                         train acc  val acc    gap  ...
    baseline (Adam)                    1.000    0.817  0.183
    L2 via Adam (wd=0.05)              0.775    0.850 -0.075
    early stopping (patience 40)       0.850    0.875 -0.025
    dropout: train mode outputs differ call to call, eval mode outputs identical
NEEDS: Python + numpy + torch + scikit-learn (make_moons). A few seconds on CPU.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons

# [1] Data: two noisy half-moons; only 80 training points, 2000 validation points
Xall, yall = make_moons(n_samples=2080, noise=0.35, random_state=0)
Xtr, ytr = torch.tensor(Xall[:80], dtype=torch.float32), torch.tensor(yall[:80])        # (80, 2), (80,)
Xva, yva = torch.tensor(Xall[80:], dtype=torch.float32), torch.tensor(yall[80:])        # (2000, 2), (2000,)
print("train", tuple(Xtr.shape), "val", tuple(Xva.shape), "| class balance in train:", torch.bincount(ytr).tolist())


# [2] A deliberately oversized network: 17,154 parameters for 80 examples
def make_net(p_drop=0.0):
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(2, 128), nn.ReLU(), nn.Dropout(p_drop),
                         nn.Linear(128, 128), nn.ReLU(), nn.Dropout(p_drop),
                         nn.Linear(128, 2))


print("parameters:", sum(p.numel() for p in make_net().parameters()))


# [3] One training routine; every regularisation method is an argument
def evaluate(net, X, y):
    net.eval()                                                   # dropout off: deterministic predictions
    with torch.no_grad():
        logits = net(X)
        return F.cross_entropy(logits, y).item(), (logits.argmax(1) == y).float().mean().item()


def train(opt_name="adam", wd=0.0, p_drop=0.0, label_smoothing=0.0, augment=False, epochs=500, patience=None, lr=3e-3):
    net = make_net(p_drop)
    Opt = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[opt_name]
    opt = Opt(net.parameters(), lr=lr, weight_decay=wd)          # Adam: wd = L2 through the gradient; AdamW: decoupled
    gen = torch.Generator().manual_seed(1)
    best = (float("inf"), None, 0)                               # (val loss, state_dict copy, epoch) for early stopping
    for epoch in range(epochs):
        net.train()                                              # dropout on
        X, y = Xtr, ytr
        if augment:                                              # fresh jittered copy of the data every epoch
            X = Xtr + 0.15 * torch.randn(Xtr.shape, generator=gen)
        loss = F.cross_entropy(net(X), y, label_smoothing=label_smoothing)
        opt.zero_grad(); loss.backward(); opt.step()
        if patience is not None:
            v_loss, _ = evaluate(net, Xva, yva)                  # NOTE: early stopping uses the validation set to choose
            if v_loss < best[0]:
                best = (v_loss, {k: v.clone() for k, v in net.state_dict().items()}, epoch)
            elif epoch - best[2] >= patience:
                break
    if patience is not None:
        net.load_state_dict(best[1])                             # roll back to the best epoch
    wnorm = torch.sqrt(sum((p ** 2).sum() for n, p in net.named_parameters() if "weight" in n)).item()
    tr_loss, tr_acc = evaluate(net, Xtr, ytr)                    # report with the CLEAN training data, in eval mode
    va_loss, va_acc = evaluate(net, Xva, yva)
    return dict(tr_acc=tr_acc, va_acc=va_acc, tr_loss=tr_loss, va_loss=va_loss, wnorm=wnorm, epochs=epoch + 1)


# [4] The experiment table: same data, same network, same seed; one change per row
runs = {
    "baseline (Adam)":              dict(),
    "L2 via Adam (wd=0.05)":        dict(opt_name="adam", wd=0.05),
    "AdamW decoupled (wd=0.05)":    dict(opt_name="adamw", wd=0.05),
    "AdamW decoupled (wd=0.5)":     dict(opt_name="adamw", wd=0.5),
    "dropout p=0.3":                dict(p_drop=0.3),
    "early stopping (patience 40)": dict(patience=40),
    "data augmentation (noise)":    dict(augment=True),
    "label smoothing 0.1":          dict(label_smoothing=0.1),
    "dropout + AdamW + noise":      dict(p_drop=0.2, opt_name="adamw", wd=0.1, augment=True),
}
print(f"\n{'method':<30}{'train acc':>10}{'val acc':>9}{'gap':>7}{'train loss':>12}{'val loss':>10}{'|W|':>8}{'epochs':>8}")
for name, kw in runs.items():
    r = train(**kw)
    print(f"{name:<30}{r['tr_acc']:>10.3f}{r['va_acc']:>9.3f}{r['tr_acc'] - r['va_acc']:>7.3f}"
          f"{r['tr_loss']:>12.4f}{r['va_loss']:>10.4f}{r['wnorm']:>8.2f}{r['epochs']:>8}")


# [5] Dropout by hand: zero random units while training and rescale the survivors, so the average stays equal
def my_dropout(x, p, training, gen):
    if not training or p == 0.0:
        return x                                                 # at inference: identity
    mask = (torch.rand(x.shape, generator=gen) > p).float()      # 1 = keep, 0 = drop, with probability p of dropping
    return x * mask / (1 - p)                                    # "inverted dropout": scale by 1/(1-p) now, so eval needs no change


g = torch.Generator().manual_seed(0)
x = torch.ones(1, 10000)
out = my_dropout(x, 0.3, True, g)
print(f"\nmy_dropout p=0.3 on 10,000 ones: fraction zeroed {(out == 0).float().mean():.3f}, survivors are {out[out > 0][0]:.4f} "
      f"(= 1/0.7), mean {out.mean():.3f} (still ~1)")

layer = nn.Dropout(0.5)
v = torch.ones(1, 8)
layer.train()
print("nn.Dropout train mode, same input twice:", layer(v).tolist()[0], layer(v).tolist()[0])
layer.eval()
print("nn.Dropout eval  mode, same input twice:", layer(v).tolist()[0], layer(v).tolist()[0])

# TRY IT: set p_drop=0.6 in the "dropout" row, or wd=2.0 in AdamW, and see underfitting (train acc falls too).
# BREAK IT: delete `net.eval()` inside `evaluate` and re-run the dropout row: the reported train and validation numbers are measured with dropout still active.
