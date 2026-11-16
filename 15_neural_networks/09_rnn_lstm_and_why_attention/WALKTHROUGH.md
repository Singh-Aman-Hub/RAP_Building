# Step 09 — RNNs, LSTMs and why attention was invented

## The story so far
[Step 08](../08_cnn_convolutions/WALKTHROUGH.md) handled grids with shared local filters. Text, audio and time series are sequences of variable length where order matters. [Step 04](../04_activations_and_vanishing_gradients/WALKTHROUGH.md) showed that a product of many factors below 1 vanishes with depth; a recurrent net is a network that is as deep as the sequence is long.

## The pain point
A feed-forward net needs a fixed input size and has no memory of what came before. 'The animal did not cross the street because it was too tired': to resolve `it` the model must carry information across words. Reading one token at a time and updating a hidden state solves the first problem; the second problem is that the gradient from a loss at step 50 to an input at step 1 passes through 49 multiplications by the same matrix.

## The spark
Reuse one cell at every time step: `h_t = f(h_{t-1}, x_t)` (the RNN). Protect the memory with gates that decide what to write, keep and show (LSTM, GRU) so information and gradient can travel a long way through an additive cell state. Yet the final hidden state is still one fixed-size vector, so long inputs must be squeezed through a bottleneck, and the sequential loop cannot be parallelised. Attention (next module) lets the decoder look back at *every* input position directly.

## Goal
Write a vanilla RNN and an LSTM cell by hand (LSTM matches `nn.LSTMCell` to 6e-08); measure the BPTT gradient over 50 steps for five recurrent-matrix scales; run a first-digit memory task for RNN, GRU, LSTM and LSTM with forget bias 1; and show a 32-number summary vector failing to copy sequences longer than about a dozen tokens.

## The theory (read this before the interview)
**Recurrent network.** One set of weights `(W_xh, W_hh, b)` applied at every step:
```
h_t = tanh(W_xh x_t + W_hh h_{t-1} + b)        h_0 = 0
y_t = W_hy h_t                                  (or only use h_T for a sequence label)
```
Shapes for hidden size `H` and input size `D`: `W_xh (H, D)`, `W_hh (H, H)`, `h_t (H,)`. Parameter count does not depend on the sequence length.

**Backpropagation through time (BPTT).** Unroll the loop into a deep network with *shared* weights, then apply ordinary backprop ([step 03](../03_tiny_autograd_engine/WALKTHROUGH.md): the shared weights receive the *sum* of their gradients from every step). The gradient of a loss at step `T` with respect to `h_t` is a product of `T - t` Jacobians:
```
dL/dh_t = dL/dh_T * prod_{k=t+1..T} diag(tanh'(z_k)) W_hh        (each factor: W_hh times an activation slope <= 1)
```
Its size is governed by the largest singular value (roughly the spectral radius for typical matrices) of `W_hh` times the slope. Below 1: *vanishing* (no signal from far past: long-range dependencies cannot be learned). Above 1: *exploding* (unstable updates). Measured (hidden 32, `T = 50`, gradient norm at `t = 1` relative to the last step's 7.5 to 10): rho 0.5: `0.0` (below float32 range); rho 0.9: `8e-13`; rho 1.0: `1e-10` (tanh slopes below 1 still shrink it); rho 1.5: `4e-04`; rho 3.0: `2.6e+05`. Remedies: **gradient clipping** (rescale if the global norm exceeds a threshold, step 10) for exploding; **gated cells and good initialisation** for vanishing; truncated BPTT (backprop only the last `k` steps) to save memory.

**LSTM (Hochreiter and Schmidhuber, 1997; forget gate added 1999).** Two states: hidden `h` and cell `c`.
```
i = sigmoid(W_i [x, h] + b_i)      input gate: how much new content to write
f = sigmoid(W_f [x, h] + b_f)      forget gate: how much old memory to keep
g = tanh(   W_g [x, h] + b_g)      candidate content
o = sigmoid(W_o [x, h] + b_o)      output gate: how much memory to expose
c_new = f * c + i * g               ADDITIVE update: the cell state highway
h_new = o * tanh(c_new)
```
`dc_new/dc = f`: if the forget gate is near 1 the gradient along the cell state is multiplied by about 1 per step, so it neither vanishes nor explodes. Common trick: initialise the forget-gate bias to 1 so the cell starts by remembering. PyTorch packs the four gates in one matrix in the order **input, forget, cell, output**. **GRU** (Cho et al., 2014) merges cell and hidden state and uses two gates (update, reset): three weight sets instead of four (parameters RNN : GRU : LSTM = 1 : 3 : 4, as printed).

**Measured memory task** (label = the first of `T` random digits, chance 0.200; hidden 32, Adam 1e-2, 400 steps, gradient clipping 1.0):

| T | RNN | GRU | LSTM | LSTM, forget bias 1 |
|---|---|---|---|---|
| 5 | 1.000 | 1.000 | 1.000 | 1.000 |
| 20 | 1.000 | 0.402 | 0.192 | 1.000 |
| 50 | 1.000 | 0.198 | 0.217 | 0.202 |

Read this honestly: the textbook story (vanilla RNNs fail at long range, gated ones succeed) is not what this *particular short run* shows. The vanilla RNN solved even T = 50 here (with clipping and a one-hot signal that survives through tanh), while default-initialised gated cells had not found the 'keep the memory' solution within 400 steps, and the forget-bias-1 LSTM found it at T = 20 but not at T = 50 in the same budget. A separate run at T = 100 (not part of `main.py`) had all four models at chance (0.17 to 0.19). Takeaways: initialisation and training budget matter as much as architecture; the BPTT table is the property of the *untrained* network; and the truly reliable fix for long range is not gating but a different mechanism.

**The bottleneck.** An encoder-decoder RNN (seq2seq, Sutskever et al. 2014) compresses the whole input into the last hidden state and the decoder starts from it. Measured on copying with a 32-number summary: length 4 gives 0.994 per-token accuracy, length 12 gives 0.590, length 24 gives 0.126 (chance for 8 symbols is 0.125). Information-wise 24 tokens of 3 bits is 72 bits, and the vector can hold some bits per float but training cannot reliably pack them. Bahdanau, Cho and Bengio (2014) fixed it by letting the decoder compute a weighted sum over *all* encoder states at each step: **attention**. Vaswani et al. (2017, 'Attention is All You Need') then removed the recurrence entirely.

**Why attention won over recurrence.** (1) Direct paths: any position reaches any other in one step, so gradient paths are short. (2) Parallel training: all positions in a sequence are processed at once instead of one at a time. (3) Scales with data and compute. Cost: attention is `O(T^2)` in sequence length and keeps a KV cache at inference ([module 06](../../06_llm_training_and_lora/README.md)). In 2025-2026 there is renewed interest in linear-recurrence / state-space layers (for example Mamba-style models and hybrids) because their inference cost is `O(1)` per token; check current model cards before claiming any of them replaces attention.

**Next stop:** [module 04 step 04 (attention from scratch)](../../04_llm_fundamentals/04_attention_from_scratch/WALKTHROUGH.md).

**Where RNNs still appear.** Small on-device models, streaming speech and time-series models, CRNN text recognisers in OCR (module 18: CNN features, then a bidirectional LSTM, then CTC decoding), and as the conceptual ancestor of the KV-cache idea.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| RNN | recurrent neural network, Elman network, vanilla RNN | Same | 'RNN' can also mean the whole family (LSTM, GRU); say 'vanilla RNN' for the tanh cell |
| LSTM | long short-term memory | - | Four gate/candidate transforms, cell state `c` and hidden `h` |
| GRU | gated recurrent unit | A simpler cousin of the LSTM | Update and reset gates, no separate cell state; 3 weight sets vs 4 |
| BPTT | backpropagation through time, unrolled backprop | Same | Truncated BPTT limits how far back the gradient flows |
| hidden state | memory, `h_t`, context vector (in seq2seq) | Same | The last hidden state of an encoder is also called the 'context' or 'thought vector' |
| cell state | `c_t`, memory cell | LSTM only | Updated by addition; `h` is a gated read-out of it |
| forget gate | keep gate (`f` is how much to keep, not to forget) | Naming quirk | `f = 1` means remember everything |
| seq2seq | encoder-decoder, sequence-to-sequence | Same | In transformers 'encoder-decoder' also exists but uses attention |
| teacher forcing | feeding ground-truth previous tokens to the decoder in training | Same | At inference the model feeds back its own predictions, a train/test mismatch ('exposure bias') |
| attention (Bahdanau / Luong) | alignment, soft lookup, content-based addressing | Same family as transformer self-attention | Original: decoder attends to encoder states; self-attention: a sequence attends to itself |
| bidirectional RNN | BiLSTM | - | Two RNNs, one per direction, concatenated; not usable for left-to-right generation |
| state-space model (SSM) | Mamba-style layer, linear recurrence | Related to RNNs | Linear (parallelisable) recurrences with structured state matrices; verify claims per model |

## Run it
```bash
cd 15_neural_networks/09_rnn_lstm_and_why_attention
python main.py       # about 1 to 2 minutes on CPU (slowest step), torch only
```

## Code, step by step
Every block below is copied from `main.py` (line numbers are exact). Read the block, then the three bullets.

### 0. Docstring and imports  (`main.py` lines 1-24)
```python
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

```
- **In simple words:** Import torch and friends; `time` to print how long the slowest experiment takes.
- **Exactly:** Everything else is plain torch. `torch.manual_seed(0)` fixes the global generator; later functions use their own seeded `Generator` for data so every model sees the same sequences.
- **Why it is here:** A shared data stream is what makes the table fair across RNN, GRU and LSTM.

### 1. A vanilla RNN written by hand: h_t = tanh(W_xh x_t + W_hh h_{t-1} + b); the SAME weights at every time step  (`main.py` lines 25-32)
```python
# [1] A vanilla RNN written by hand: h_t = tanh(W_xh x_t + W_hh h_{t-1} + b); the SAME weights at every time step
def rnn_forward(xs, W_xh, W_hh, b, h0):
    """xs: (T, input_dim). Returns the list of hidden states h_1..h_T, each (hidden,)."""
    h, hs = h0, []
    for x in xs:                                                 # a Python loop over time: the recurrence
        h = torch.tanh(W_xh @ x + W_hh @ h + b)                  # (hidden,)
        hs.append(h)
    return hs
```
- **In simple words:** A vanilla RNN is a loop: mix the new input with the previous memory, squash, repeat. The same weights are reused at every step.
- **Exactly:** `W_xh @ x + W_hh @ h + b` has shape `(hidden,)` because `W_xh (H, D) @ x (D,)` and `W_hh (H, H) @ h (H,)` each give `(H,)`. `torch.tanh` bounds the state to (-1, 1). The function returns the list of every hidden state so we can inspect gradients at each step.
- **Why it is here:** This loop *is* the architecture; seeing it as a plain Python `for` shows why sequences must be processed one after another.

### 2. Backpropagation through time: how much does a loss at the LAST step care about step t?  (`main.py` lines 35-55)
```python
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
```
- **In simple words:** Ask: if the final output is wrong, how much blame goes to each earlier step? Do this for recurrent matrices of different strength.
- **Exactly:** `torch.linalg.eigvals(W).abs().max()` is the spectral radius; dividing `W` by it and multiplying by `rho` sets it exactly. `requires_grad_()` on `W` makes the hidden states part of a graph, and `h.retain_grad()` keeps `dLoss/dh_t` for every step. The loss is `h_T^2` summed. Table of gradient norms at `t = 50 / 40 / 25 / 10 / 1`: rho 0.5: `7.50e+00 / 8.52e-05 / 6.01e-13 / 1.78e-20 / 0.00e+00`; 0.9: `8.09 / 1.82e-02 / 9.49e-07 / 2.08e-10 / 7.99e-13`; 1.0: `8.28 / 4.33e-02 / 1.22e-05 / 1.18e-08 / 1.35e-10`; 1.5: `9.16 / 0.735 / 2.46e-02 / 2.85e-03 / 4.29e-04`; 3.0: `10.1 / 40.2 / 1.29e+03 / 2.95e+04 / 2.57e+05`.
- **Why it is here:** It makes 'vanishing and exploding gradients through time' concrete: even at rho = 1 the tanh slopes shrink the signal by 1e-10 over 50 steps; only rho well above 1 preserves it, and then it explodes in other directions.

### 3. LSTM by hand: three gates and a cell state; the cell is updated by ADDITION (a gradient highway)  (`main.py` lines 58-75)
```python
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
```
- **In simple words:** An LSTM cell: three valves (write, forget, show) controlling a memory belt that is updated by adding and scaling, not by repeated squashing.
- **Exactly:** `W_ih @ x + W_hh @ h + biases` has shape `(4*H,)`; `.chunk(4)` splits it into `i, f, g, o` (PyTorch order). Sigmoids for the three gates, `tanh` for the candidate. `c_new = f * c + i * g` is elementwise. The maximum absolute difference to `nn.LSTMCell` is `6.0e-08` for both `h` and `c`. Parameter counts with input 4, hidden 8: RNN 112, GRU 336, LSTM 448 (ratio 1 : 3 : 4).
- **Why it is here:** Writing the gates out removes the mystery of the diagram; and `c_new = f * c + ...` is the line that explains the LSTM's gradient highway.

### 4. A toy memory task: the label is the FIRST digit; everything after it is noise. Can the net remember?  (`main.py` lines 78-120)
```python
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
```
- **In simple words:** A memory test: show the net 5, 20 or 50 random digits and then ask what the first one was. Everything after it is noise to ignore.
- **Exactly:** `make_batch` returns one-hot inputs `(batch, T, 5)` and the first digit as label `(batch,)`. `Recaller` wraps `nn.RNN` / `nn.GRU` / `nn.LSTM` (`batch_first=True` gives `(batch, T, hidden)` outputs) and classifies from `states[:, -1]`. The `LSTM+fb` variant sets the forget-gate slice of every bias (indices `hidden:2*hidden`) to 1. Each model trains 400 Adam steps (lr 1e-2, batch 64) with `clip_grad_norm_(..., 1.0)`; test on 1000 fresh sequences. Results: T=5 all 1.000; T=20: RNN 1.000, GRU 0.402, LSTM 0.192, LSTM+fb 1.000; T=50: RNN 1.000, GRU 0.198, LSTM 0.217, LSTM+fb 0.202.
- **Why it is here:** A controlled test of memory length. The surprise (vanilla RNN good, default LSTM poor in a short budget) is a real lesson about initialisation and budget; do not hide it.

### 5. The bottleneck: an encoder squeezes a whole sequence into ONE fixed vector, a decoder must rebuild it  (`main.py` lines 123-159)
```python
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
```
- **In simple words:** Make the model rebuild a whole sequence from a single short summary and watch it break as the sequence grows.
- **Exactly:** `Seq2SeqCopy`: an embedding `(vocab+1, 32)`, a GRU encoder whose final hidden state `summary (1, batch, 32)` is the *only* thing handed to a GRU decoder, which gets the previous true token at every step (teacher forcing, shifted right with a start token). Output `(batch, L, 8)`. Per-token accuracy after 600 steps: L=4 `0.994`, L=12 `0.590`, L=24 `0.126` (chance 0.125).
- **Why it is here:** The bottleneck experiment in numbers. It is exactly the weakness that the 2014 attention mechanism removed by letting the decoder read all encoder states.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `for x in xs:` | iterate over the first axis of a tensor (time steps) | `xs.shape == (50, 4)` |
| `gates.chunk(4)` | split a tensor into 4 equal pieces along dim 0 | `(32,) -> four (8,)` |
| `i, f, g, o = ...` | unpack a tuple of tensors | the four LSTM parts |
| `states[:, -1]` | all batch items, last time step | `(batch, T, H) -> (batch, H)` |
| `{'RNN': nn.RNN, ...}[kind](vocab, hidden, batch_first=True)` | choose a class from a dict and instantiate it | - |
| `F.one_hot(seq, vocab).float()` | integer ids to one-hot vectors | `(B,T) -> (B,T,V)` |
| `with torch.no_grad(): p[a:b].fill_(1.0)` | edit parameter values in place without recording gradients | forget-bias init |
| `torch.cat([a, b], dim=1)` | concatenate along an axis | start token + shifted sequence |
| `reshape(-1, 8)` | flatten leading dims; `-1` is inferred | `(B,L,8) -> (B*L,8)` |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `nn.RNN / nn.GRU / nn.LSTM(input, hidden, batch_first)` | torch | multi-step recurrent layers; return `(outputs, final_state)` | `num_layers`, `bidirectional`, `dropout` | tuple |
| `nn.LSTMCell` | torch | a single LSTM step | `input_size`, `hidden_size` | module returning `(h, c)` |
| `nn.Embedding(n, d)` | torch | lookup table from ids to vectors | - | module |
| `nn.utils.clip_grad_norm_(params, max_norm)` | torch | rescale gradients if their global norm exceeds `max_norm` | `max_norm` | total norm before clipping |
| `torch.linalg.eigvals` | torch | eigenvalues (complex) of a square matrix | - | complex tensor |
| `torch.Generator().manual_seed(n)` | torch | independent seeded RNG | - | generator |

## Execution trace (real run)
```
norm of dLoss/dh_t for a loss at step 50 (BPTT), for recurrent weight matrices of different spectral radius rho:
  rho        t=50        t=40        t=25        t=10         t=1
  0.5    7.50e+00    8.52e-05    6.01e-13    1.78e-20    0.00e+00
  0.9    8.09e+00    1.82e-02    9.49e-07    2.08e-10    7.99e-13
  1.0    8.28e+00    4.33e-02    1.22e-05    1.18e-08    1.35e-10
  1.5    9.16e+00    7.35e-01    2.46e-02    2.85e-03    4.29e-04
  3.0    1.01e+01    4.02e+01    1.29e+03    2.95e+04    2.57e+05

my LSTM cell vs nn.LSTMCell: max abs difference h 6.0e-08, c 6.0e-08
parameters: RNN 112 | GRU 336 | LSTM 448 (hidden 8, input 4: ratio 1 : 3 : 4)

memory task, accuracy of recalling the first digit after T steps (chance = 0.200):
   T     RNN     GRU    LSTM  LSTM+fb
   5   1.000   1.000   1.000    1.000
  20   1.000   0.402   0.192    1.000
  50   1.000   0.198   0.217    0.202
(took 73 s)

copy a sequence through a 32-number summary vector (chance = 0.125):
  length  4: per-token accuracy 0.994   (information: 12 bits to store in 32 floats)
  length 12: per-token accuracy 0.590   (information: 36 bits to store in 32 floats)
  length 24: per-token accuracy 0.126   (information: 72 bits to store in 32 floats)
```
- BPTT table: with rho = 0.5 the gradient to step 1 is exactly 0.0 in float32; rho = 1.0 still shows 1.35e-10; only rho >= 1.5 keeps it above 1e-4 and rho = 3 explodes to 2.6e+05.
- `my LSTM cell` equals `nn.LSTMCell` to 6e-08; parameter ratio 112 : 336 : 448.
- Memory table: see theory; the vanilla RNN stays at 1.000 up to T = 50, default LSTM/GRU fail to find the solution in 400 steps, forget-bias-1 LSTM succeeds at T = 20.
- Copy through a 32-number vector: 0.994, 0.590, 0.126 per-token accuracy for L = 4, 12, 24: the bottleneck collapses to chance exactly where the sequence carries more information than the summary can hold.

## Experiments (change ONE thing)
1. T = 100 (edit the loop to `(5, 20, 50, 100)`): RNN 0.168, GRU 0.181, LSTM 0.191, LSTM+fb 0.188, all at chance within 400 steps (measured in a separate run of the same code).
2. LSTM at T = 20 with a lower lr (3e-3, 400 steps, default bias): 0.185, chance; with forget bias 1 and lr 3e-3 for 1000 steps: 1.000. Forget-gate initialisation and optimisation budget decide whether the gating solution is found.
3. The BPTT table with `T = 100`: edit `T=50` in `gradient_vs_time` and the indices; the vanishing columns grow by many more orders of magnitude (not run here).

## Debugging practice
BREAK IT 1: in `gradient_vs_time` delete `W.requires_grad_()`:
```
  File ".../main.py", line 46, in gradient_vs_time
    h.retain_grad()                                          # keep dLoss/dh_t for every t
RuntimeError: can't retain_grad on Tensor that has requires_grad=False
```
Read it: with no tensor requiring gradients, the hidden states are plain tensors with no graph, so there is nothing to retain. Make at least one weight a leaf with `requires_grad=True` so every later result becomes a non-leaf node.

BREAK IT 2 (silent): remove `clip_grad_norm_` from `memory_accuracy` and train the vanilla RNN several times: occasional gradient spikes make loss jump, and on bad seeds you can get `nan`. Clipping bounds the update size but does not fix vanishing gradients: it only addresses the exploding side. (Not run here: seed-dependent.)

## Rebuild checklist
- [ ] write the RNN recurrence and give shapes
- [ ] explain BPTT as backprop on the unrolled graph with shared weights
- [ ] derive why the gradient is a product of Jacobians and what spectral radius does
- [ ] write the LSTM gate equations from memory and say why `c_new = f*c + i*g` helps
- [ ] state the RNN : GRU : LSTM parameter ratio
- [ ] explain the seq2seq bottleneck and how attention removes it

## The new problem (cliffhanger)
A fixed-size summary fails at 24 tokens, and a loop over time cannot use a GPU's parallelism. Both problems vanish if every position can look directly at every other position and all positions are computed at once. That is attention: queries, keys and values. Next, in module 04: [attention from scratch](../../04_llm_fundamentals/04_attention_from_scratch/WALKTHROUGH.md). Before that, [step 10](../10_precision_and_training_stability/WALKTHROUGH.md) covers how to keep such training runs numerically alive.

> **Say it aloud (interview box):** An RNN reuses one cell across time steps, carrying a hidden state. Training unrolls it (BPTT), so the gradient is a product of per-step Jacobians: below spectral radius 1 it vanishes, above it explodes. Clipping handles the exploding side. LSTMs add a cell state updated additively, c = f*c + i*g, with gates, so gradients can flow through it; GRUs are a cheaper two-gate variant. But the encoder-decoder squeezes the whole input into one fixed vector (copy accuracy fell from 0.99 to chance as length grew from 4 to 24) and the recurrence is sequential. Attention lets the decoder read every encoder state directly and, as self-attention in transformers, removes the recurrence.
