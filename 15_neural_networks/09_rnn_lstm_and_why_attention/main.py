"""
GOAL:  Understand recurrent networks: a vanilla RNN written by hand, the vanishing/exploding gradient over TIME
       (BPTT), LSTM gates written by hand and checked against torch, a toy memory task where RNN / LSTM / GRU
       differ, and the fixed-size-vector bottleneck that motivated attention.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 15_neural_networks/09_rnn_lstm_and_why_attention/main.py
EXPECTED OUTPUT (shortened):
    BPTT gradient norm at t=1 for a loss at t=50: rho 0.5 -> 0, rho 1.0 -> ~1e-10, rho 3.0 -> ~3e+05
    my LSTM cell vs nn.LSTMCell: max abs difference ~6e-08
    memory task (first digit after T steps; chance 0.200): all models 1.000 at T=5; at T=50 only the vanilla RNN still 1.000
    copy through a 32-number vector: per-token accuracy 0.994 (L=4), 0.590 (L=12), 0.126 (L=24 = chance)
NEEDS: Python + torch. About 1 to 2 minutes on a laptop CPU (the slowest step of this module).
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# [1] A vanilla RNN written by hand: h_t = tanh(W_xh x_t + W_hh h_{t-1} + b); the SAME weights at every time step
def rnn_forward(xs, W_xh, W_hh, b, h0):
    """xs: (T, input_dim). Returns the list of hidden states h_1..h_T, each (hidden,)."""
    h, hs = h0, []
    for x in xs:                                                 # a Python loop over time: the recurrence
        h = torch.tanh(W_xh @ x + W_hh @ h + b)                  # (hidden,)
        hs.append(h)
    return hs


# [2] Backpropagation through time: how much does a loss at the LAST step care about step t?
def gradient_vs_time(rho, T=50, hidden=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(hidden, hidden, generator=g)
    W = W * rho / torch.linalg.eigvals(W).abs().max()            # rescale so the spectral radius (largest |eigenvalue|) is rho
    W.requires_grad_()                                           # make the graph exist so hidden states are non-leaf tensors
    W_xh = torch.randn(hidden, 4, generator=g) * 0.5
    xs = torch.randn(T, 4, generator=g)
    h0 = torch.zeros(hidden)
    hs = rnn_forward(xs, W_xh, W, torch.zeros(hidden), h0)
    for h in hs:
        h.retain_grad()                                          # keep dLoss/dh_t for every t
    hs[-1].pow(2).sum().backward()                               # loss depends on the final state only
    return [h.grad.norm().item() for h in hs]


print("norm of dLoss/dh_t for a loss at step 50 (BPTT), for recurrent weight matrices of different spectral radius rho:")
print(f"{'rho':>5}{'t=50':>12}{'t=40':>12}{'t=25':>12}{'t=10':>12}{'t=1':>12}")
for rho in (0.5, 0.9, 1.0, 1.5, 3.0):
    gr = gradient_vs_time(rho)
    print(f"{rho:>5}{gr[49]:>12.2e}{gr[39]:>12.2e}{gr[24]:>12.2e}{gr[9]:>12.2e}{gr[0]:>12.2e}")


# [3] LSTM by hand: three gates and a cell state; the cell is updated by ADDITION (a gradient highway)
def lstm_cell(x, h, c, W_ih, W_hh, b_ih, b_hh):
    gates = W_ih @ x + b_ih + W_hh @ h + b_hh                    # (4*hidden,) all four pre-activations in one matmul
    i, f, g, o = gates.chunk(4)                                  # torch's order: input, forget, cell candidate, output
    i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)   # gates in (0, 1): how much to write / keep / expose
    g = torch.tanh(g)                                            # candidate new content in (-1, 1)
    c_new = f * c + i * g                                        # forget some old memory, add some new
    h_new = o * torch.tanh(c_new)                                # expose a filtered view of the memory
    return h_new, c_new


cell = nn.LSTMCell(4, 8)
x, h, c = torch.randn(4), torch.randn(8), torch.randn(8)
mine = lstm_cell(x, h, c, cell.weight_ih, cell.weight_hh, cell.bias_ih, cell.bias_hh)
ref = cell(x[None], (h[None], c[None]))
print(f"\nmy LSTM cell vs nn.LSTMCell: max abs difference h {(mine[0] - ref[0][0]).abs().max().item():.1e}, c {(mine[1] - ref[1][0]).abs().max().item():.1e}")
print("parameters: RNN", sum(p.numel() for p in nn.RNN(4, 8).parameters()), "| GRU", sum(p.numel() for p in nn.GRU(4, 8).parameters()),
      "| LSTM", sum(p.numel() for p in nn.LSTM(4, 8).parameters()), "(hidden 8, input 4: ratio 1 : 3 : 4)")


# [4] A toy memory task: the label is the FIRST digit; everything after it is noise. Can the net remember?
def make_batch(batch, T, vocab=5, gen=None):
    seq = torch.randint(0, vocab, (batch, T), generator=gen)    # (batch, T) random digits
    return F.one_hot(seq, vocab).float(), seq[:, 0]               # inputs (batch, T, vocab), label = first digit (batch,)


class Recaller(nn.Module):
    def __init__(self, kind, vocab=5, hidden=32):
        super().__init__()
        self.rnn = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM, "LSTM+fb": nn.LSTM}[kind](vocab, hidden, batch_first=True)
        if kind == "LSTM+fb":                                    # "forget-gate bias 1": start with the memory switched ON
            for name, p in self.rnn.named_parameters():
                if name.startswith("bias"):
                    with torch.no_grad():
                        p[hidden:2 * hidden].fill_(1.0)          # gate order is (input, forget, cell, output): slice 1 = forget
        self.out = nn.Linear(hidden, vocab)
    def forward(self, x):                                        # x: (batch, T, vocab)
        states, _ = self.rnn(x)                                  # (batch, T, hidden): hidden state at every step
        return self.out(states[:, -1])                           # classify from the LAST state only


def memory_accuracy(kind, T, steps=400):
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1)
    model = Recaller(kind)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(steps):
        x, y = make_batch(64, T, gen=gen)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)        # standard for RNNs: tame exploding gradients
        opt.step()
    x, y = make_batch(1000, T, gen=torch.Generator().manual_seed(99))
    with torch.no_grad():
        return (model(x).argmax(1) == y).float().mean().item()


print("\nmemory task, accuracy of recalling the first digit after T steps (chance = 0.200):")
print(f"{'T':>4}{'RNN':>8}{'GRU':>8}{'LSTM':>8}{'LSTM+fb':>9}")
t0 = time.time()
for T in (5, 20, 50):
    print(f"{T:>4}" + "".join(f"{memory_accuracy(k, T):>{9 if k == 'LSTM+fb' else 8}.3f}" for k in ("RNN", "GRU", "LSTM", "LSTM+fb")))
print(f"(took {time.time() - t0:.0f} s)")


# [5] The bottleneck: an encoder squeezes a whole sequence into ONE fixed vector, a decoder must rebuild it
class Seq2SeqCopy(nn.Module):
    def __init__(self, vocab=8, hidden=32):
        super().__init__()
        self.vocab = vocab
        self.emb = nn.Embedding(vocab + 1, hidden)               # +1: a start-of-sequence token for the decoder
        self.enc = nn.GRU(hidden, hidden, batch_first=True)
        self.dec = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab)
    def forward(self, seq):                                      # seq: (batch, L) ints
        _, summary = self.enc(self.emb(seq))                     # summary: (1, batch, hidden) = THE bottleneck: 32 numbers
        start = torch.full_like(seq[:, :1], self.vocab)
        dec_in = torch.cat([start, seq[:, :-1]], dim=1)          # teacher forcing: feed the previous true token
        states, _ = self.dec(self.emb(dec_in), summary)          # the decoder sees ONLY the summary as its memory of the input
        return self.out(states)                                  # (batch, L, vocab)


def copy_accuracy(L, steps=600):
    torch.manual_seed(0)
    model = Seq2SeqCopy()
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    gen = torch.Generator().manual_seed(2)
    for _ in range(steps):
        seq = torch.randint(0, 8, (64, L), generator=gen)
        loss = F.cross_entropy(model(seq).reshape(-1, 8), seq.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    seq = torch.randint(0, 8, (500, L), generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        return (model(seq).argmax(-1) == seq).float().mean().item()


print("\ncopy a sequence through a 32-number summary vector (chance = 0.125):")
for L in (4, 12, 24):
    print(f"  length {L:>2}: per-token accuracy {copy_accuracy(L):.3f}   (information: {L * 3} bits to store in 32 floats)")

# TRY IT: raise hidden to 128 in Seq2SeqCopy and see how much longer a sequence it can copy.
# BREAK IT: remove `nn.utils.clip_grad_norm_` in memory_accuracy and run the RNN at T=50 several times; watch for loss spikes or nan.
