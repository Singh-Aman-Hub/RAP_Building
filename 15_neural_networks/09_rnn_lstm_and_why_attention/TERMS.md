# Terms — RNNs, LSTMs and why attention was invented

Folder: `15_neural_networks/09_rnn_lstm_and_why_attention` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [15 README](../README.md)

**Terms in this folder:** [Recurrent neural network (RNN)](#recurrent-neural-network-rnn) · [Hidden state](#hidden-state) · [Backpropagation through time (BPTT)](#backpropagation-through-time-bptt) · [Spectral radius](#spectral-radius) · [Vanishing and exploding gradients through time](#vanishing-and-exploding-gradients-through-time) · [Gradient clipping](#gradient-clipping) · [LSTM (long short-term memory)](#lstm-long-short-term-memory) · [Gates (input, forget, output)](#gates-input-forget-output) · [Cell state](#cell-state) · [Forget-gate bias initialisation](#forget-gate-bias-initialisation) · [GRU (gated recurrent unit)](#gru-gated-recurrent-unit) · [Sequence-to-sequence (encoder-decoder)](#sequence-to-sequence-encoder-decoder) · [Information bottleneck (fixed-size context vector)](#information-bottleneck-fixed-size-context-vector) · [Teacher forcing](#teacher-forcing) · [Attention (preview)](#attention-preview) · [Truncated BPTT](#truncated-bptt) · [batch_first](#batchfirst) · [One-hot encoding](#one-hot-encoding) · [RuntimeError: can't retain_grad on Tensor that has requires_grad=False](#runtimeerror-cant-retaingrad-on-tensor-that-has-requiresgradfalse)

---

## Recurrent neural network (RNN)  `(concept)`
**1. Definition:** A network that applies the same cell to each element of a sequence, passing a hidden state: `h_t = tanh(W_xh x_t + W_hh h_{t-1} + b)`.
**2. In plain words:** Reading a sentence word by word while keeping notes.
**3. Example to quote:** > "`rnn_forward`"
**Seen in:** `main.py`, block 1

## Hidden state  `(concept)`
**1. Definition:** The vector carried from step to step; the network's memory of the past.
**2. In plain words:** The notes.
**3. Example to quote:** > "`(hidden,)` per step."
**Seen in:** `main.py`, block 1

## Backpropagation through time (BPTT)  `(concept)`
**1. Definition:** Backprop on the unrolled recurrent graph; weight gradients are summed over all time steps.
**2. In plain words:** Treat the loop as a very deep network sharing weights.
**3. Example to quote:** > "`hs[-1].pow(2).sum().backward()`"
**Seen in:** `main.py`, block 2

## Spectral radius  `(concept)`
**1. Definition:** The largest absolute eigenvalue of a square matrix; controls growth of repeated multiplication.
**2. In plain words:** Whether repeated application shrinks or grows vectors.
**3. Example to quote:** > "Set to 0.5 ... 3.0."
**Seen in:** `main.py`, block 2

## Vanishing and exploding gradients through time  `(concept)`
**1. Definition:** The gradient to early steps shrinks or grows exponentially with the distance.
**2. In plain words:** A whisper passed along 50 people versus a shout.
**3. Example to quote:** > "rho 0.5: 0.0; rho 3: 2.6e+05."
**Seen in:** `main.py`, block 2

## Gradient clipping  `(concept)`
**1. Definition:** Rescaling the gradient so its global norm is at most a threshold.
**2. In plain words:** A speed limit on updates.
**3. Example to quote:** > "`clip_grad_norm_(params, 1.0)`"
**Seen in:** `main.py`, block 4

## LSTM (long short-term memory)  `(concept)`
**1. Definition:** A recurrent cell with input, forget and output gates and an additively updated cell state.
**2. In plain words:** A notebook with write, erase and read controls.
**3. Example to quote:** > "`c_new = f * c + i * g`"
**Seen in:** `main.py`, block 3

## Gates (input, forget, output)  `(concept)`
**1. Definition:** Sigmoid outputs in (0, 1) that multiply other signals to let a fraction pass.
**2. In plain words:** Dimmer switches.
**3. Example to quote:** > "`i, f, o = torch.sigmoid(...)`"
**Seen in:** `main.py`, block 3

## Cell state  `(concept)`
**1. Definition:** The LSTM's internal memory, updated by scaling and adding rather than being squashed at every step.
**2. In plain words:** A conveyor belt carrying information.
**3. Example to quote:** > "`dc_new/dc = f`"
**Seen in:** `main.py`, block 3

## Forget-gate bias initialisation  `(concept)`
**1. Definition:** Setting the forget-gate bias to about 1 so the cell initially keeps its memory.
**2. In plain words:** Start with the notebook open.
**3. Example to quote:** > "T = 20: 0.192 -> 1.000."
**Seen in:** `main.py`, block 4

## GRU (gated recurrent unit)  `(concept)`
**1. Definition:** A two-gate (update, reset) recurrent cell without a separate cell state.
**2. In plain words:** A lighter LSTM.
**3. Example to quote:** > "336 vs 448 parameters at hidden 8."
**Seen in:** `main.py`, block 3

## Sequence-to-sequence (encoder-decoder)  `(concept)`
**1. Definition:** An encoder reads the input into a state; a decoder generates the output from it.
**2. In plain words:** Read the whole message, then write the reply.
**3. Example to quote:** > "`Seq2SeqCopy`"
**Seen in:** `main.py`, block 5

## Information bottleneck (fixed-size context vector)  `(concept)`
**1. Definition:** All input information must pass through one vector of fixed size.
**2. In plain words:** Summarising a book in a tweet.
**3. Example to quote:** > "Copy accuracy 0.994 -> 0.126."
**Seen in:** `main.py`, block 5

## Teacher forcing  `(concept)`
**1. Definition:** Feeding the true previous token to the decoder during training.
**2. In plain words:** Training wheels on the decoder.
**3. Example to quote:** > "`dec_in = cat([start, seq[:, :-1]])`"
**Seen in:** `main.py`, block 5

## Attention (preview)  `(concept)`
**1. Definition:** A learned weighted average over all input positions' vectors, with weights computed from content.
**2. In plain words:** Looking back at the page instead of relying on memory.
**3. Example to quote:** > "Module 04 step 04."
**Seen in:** `WALKTHROUGH.md`, theory

## Truncated BPTT  `(concept)`
**1. Definition:** Backpropagating only through the last k steps to save memory and limit vanishing.
**2. In plain words:** Remember only the recent past when learning.
**3. Example to quote:** > "Used for very long sequences."
**Seen in:** `WALKTHROUGH.md`, theory

## batch_first  `(library)`
**1. Definition:** A flag that makes recurrent layers use `(batch, time, features)` instead of `(time, batch, features)`.
**2. In plain words:** Which axis is which.
**3. Example to quote:** > "`nn.LSTM(..., batch_first=True)`"
**Seen in:** `main.py`, block 4

## One-hot encoding  `(concept)`
**1. Definition:** Representing an integer id as a vector with a single 1.
**2. In plain words:** A light switch panel with one switch on.
**3. Example to quote:** > "`F.one_hot(seq, 5)`"
**Seen in:** `main.py`, block 4

## RuntimeError: can't retain_grad on Tensor that has requires_grad=False  `(error)`
**1. Definition:** You asked to keep a gradient for a tensor that is not part of an autograd graph.
**2. In plain words:** Nothing upstream requires gradients.
**3. Example to quote:** > "Fix: make a weight `requires_grad`."
**Seen in:** `WALKTHROUGH.md`, debugging
