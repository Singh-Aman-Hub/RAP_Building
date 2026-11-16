# The Story of ML → Neural Networks → Deep Learning → LLMs

**The whole story in one sentence:** every step happened because the previous idea hit a wall — and each new idea kept the old machinery (*model → loss → gradient → update*) and only changed *what the model is made of* or *how much data and compute it gets*.

Think of it like transport: walking (rules) → bicycle (ML) → car (neural nets) → aeroplane (deep learning) → rocket (LLMs). You never throw away "moving forward"; you just find a better engine when the old one can't go further.

> Dates are approximate and names are the usual credits; real history is messier. Each chapter ends with **"The wall"** (why it stopped working) and **"Terms"** (what to remember). Repo links show where you practise it.

---

## Chapter 0 — Before ML: people write the rules (until the 1950s)

**The story.** Early programs were *rules written by humans*: `if temperature > 38 then fever`. A programmer thinks, then types the rule. This works great for tax calculators and bank balances.

**The wall.** Try writing rules to recognise a cat in a photo, or to tell spam from real mail. There are millions of exceptions. No human can write all of them.

**The new idea:** *don't write the rules — show examples and let the computer find the rules.*

**Terms:** *rule-based system*, *explicit programming*.

---

## Chapter 1 — Machine Learning is born: learn from examples (1950s–1990s)

**The story.** In 1959 Arthur Samuel (checkers program that improved by playing itself) popularised the phrase **"machine learning"**: programs that improve with experience without being explicitly programmed.

How it works, in the simplest form (this is your **Phase 1, Lesson 1**):

| Piece | Exam-score example |
|---|---|
| **Data** (samples) | 5 students |
| **Feature** `x` (input) | hours studied |
| **Label** `y` (answer) | exam score |
| **Model** (a formula with knobs) | `prediction = weight × x + bias` |
| **Parameters** (the knobs) | `weight`, `bias` |
| **Loss** (how wrong?) | mean squared error |
| **Learning** | turn the knobs until loss is small |

How the knobs get turned: **gradient descent** — feel the slope of the loss and step downhill (Lessons 2–3). Swap the straight line for a sigmoid and you can say yes/no: **logistic regression** (Lesson 4). Later came **decision trees, SVMs, random forests** — the workhorses of the 1990s–2000s for spreadsheet-style (tabular) data (Lesson 7).

Alongside, people learned the hard lessons: a model can *memorise* (**overfitting**), so you test on unseen data; and you must avoid **data leakage** (Lessons 5–6).

**The wall.** Classical ML works on *features a human designs*. For house prices, "size, rooms, age" are easy. For a photo, what is the "feature"? Pixel number 4,017? Experts spent years hand-crafting features (edges, corners, word counts). Quality was capped by human cleverness. And models like a straight line can only draw straight boundaries.

**The new idea:** *let the model learn the features too.* Stack many simple learners so that early ones find simple patterns and later ones combine them.

**Terms:** *supervised / unsupervised / reinforcement learning, regression, classification, loss, gradient descent, overfitting, generalisation, train/validation/test, baseline.*

---

## Chapter 2 — The Neural Network: a stack of small learners (1943 → 1986)

**The roots.** 1943: McCulloch & Pitts model a brain cell as a tiny switch. 1957: Frank Rosenblatt builds the **perceptron** — literally "weighted sum + threshold", i.e. *your Lesson 1 formula* plus a decision. It could learn to separate two groups.

**The first wall (1969).** Minsky & Papert showed a single perceptron cannot learn even **XOR** (true when exactly one input is on) — because one straight line can't separate it. Funding dried up: the first **"AI winter"**.

**The fix (in theory, long before it worked):** put layers between input and output. Each layer does a *linear step* (`W·x + b`, the same as your line) followed by a *non-linear* **activation** (sigmoid, tanh, later **ReLU**). Without the non-linearity, ten layers collapse into one line; with it, the network can bend boundaries into any shape.

**The second wall:** how do you train a network with thousands of knobs spread across layers? You need the slope of the loss for *every* knob, including those buried deep inside.

**The breakthrough — backpropagation (popularised 1986, Rumelhart, Hinton & Williams).** It is just the **chain rule** from calculus (your Phase 2) applied backwards through the layers, reusing the work. Think of a factory line with a defective product: you trace the blame backwards station by station, telling each station how much it contributed. Then ordinary **gradient descent** updates every knob. Same loop as Chapter 1 — more knobs.

**What stayed the same:** loss, gradient, update. **What changed:** the model went from `wx + b` to *layers of (linear + activation)* — "learnable non-linear representations". Better optimisers followed later: SGD → Momentum → RMSProp → **Adam / AdamW**.

**The wall.** In the 1990s neural nets worked in places (LeCun's digit reader for cheques, 1989) but were hard to train, slow, and needed lots of data and compute that did not exist. Meanwhile SVMs and random forests were easier and often won. Neural nets went quiet again.

**Terms:** *perceptron, weights, bias, activation (sigmoid/tanh/ReLU), layer, forward pass, backpropagation, chain rule, epoch, mini-batch, learning rate, initialisation, dropout.* → **Phase 3**.

---

## Chapter 3 — Deep Learning: neural nets + big data + GPUs (≈2006–2012)

**The story.** Three things quietly changed:
1. **Data** exploded (the internet, ImageNet with ~1M labelled images).
2. **Compute**: video-game **GPUs** turned out to be perfect at the giant matrix multiplications nets need.
3. **Tricks** that made *deep* (many-layer) networks trainable: ReLU activations, better initialisation, dropout, normalisation.

In 2012 **AlexNet** (a deep convolutional network on two GPUs) crushed the image-recognition competition by a huge margin. The world noticed. "Deep" simply means *many layers*, and depth lets the model build features in steps: pixels → edges → shapes → eyes → faces. Features are **learned, not hand-made** — that was the wall of Chapter 1 broken.

**Why images needed something special — the CNN.** A photo has 1,000×1,000 pixels; connecting every pixel to every neuron needs a billion weights. But a cat's ear looks the same wherever it sits. A **convolution** slides one small filter (a **kernel**) across the whole image, *reusing the same weights everywhere* (**weight sharing**). Fewer parameters, and "find an edge anywhere" for free. Pooling shrinks the map; stacked layers see larger and larger regions (**receptive field**).

**What stayed the same:** loss, backprop, gradient descent. **What changed:** *architecture fitted to the data's shape* (grids → convolution) plus scale.

**The wall.** CNNs assume *grid-like, local* structure. Language is different: it is a **sequence**, order matters, and meaning depends on words far away ("The trophy didn't fit in the suitcase because **it** was too big" — what is "it"?).

**Terms:** *deep network, GPU, convolution, kernel, stride, padding, pooling, feature map, receptive field, transfer learning.* → **Phase 4**.

---

## Chapter 4 — Sequences: RNNs and LSTMs (1980s → 2010s)

**The story.** To read text one word at a time, give the network a **memory**: a **hidden state** that is updated at every step and carries what it has read so far. This is the **RNN** (recurrent neural network). Its job in language: **next-token prediction** — "after 'the cat sat on the', predict 'mat'". (Your `mini_model.py` predicts the next character; the RNN adds memory to it.) Words are first turned into numbers: **one-hot vectors**, then learned **embeddings** (a word becomes a point in space where similar words are near each other — e.g. **word2vec**, 2013).

**The wall — forgetting.** Training an RNN means backpropagating *through time*, multiplying many numbers together. Multiply many numbers smaller than 1 → the signal shrinks to nothing (**vanishing gradient**); bigger than 1 → it explodes. The network forgets the start of a long sentence — like the game of "telephone" where the message degrades as it is passed along.

**Fix 1 — LSTM / GRU (1997, Hochreiter & Schmidhuber):** add **gates** (open/close valves) and a **cell state** (a conveyor belt that carries information unchanged unless a gate acts). Much better memory.

**The wall again.** Even LSTMs: (a) still pass everything through one narrow hidden state; (b) must read **one word at a time**, so they can't use a GPU's parallel power — training on huge text is painfully slow; (c) far-apart words are still many steps apart.

**Terms:** *sequence, hidden state, unrolling, BPTT, vanishing/exploding gradients, LSTM gates, cell state, GRU, embedding, language modelling.* → **Phase 5**.

---

## Chapter 5 — Turning text into numbers (always needed) 

Networks only eat numbers. So text goes through **tokenisation**: split into pieces (*tokens* — words, characters, or **subwords** like "un" + "believ" + "able") and map each to an ID. Characters never hit "unknown word" but make sequences long; whole words make huge vocabularies and break on new words; **subword (BPE)** is the compromise modern LLMs use. Then each ID becomes an **embedding** vector. Old methods (**one-hot, TF-IDF**) count words but give "bank" the same vector in "river bank" and "bank loan" and miss similarity between "good" and "great"; learned embeddings capture similarity, and later **contextual** representations capture *meaning in context*. (Your `Tokenization.py` and `project_a_lm/tokenizer.py` are first steps.) → **Phase 6**.

---

## Chapter 6 — Attention and the Transformer (2014 → 2017)

**The idea of attention (2014, Bahdanau et al.).** Instead of cramming a whole sentence into one memory, let the model **look back at every word and decide which matter right now** — like re-reading a paragraph with a highlighter when answering a question, rather than answering from vague memory.

**The 2017 leap — "Attention Is All You Need" (Google).** Throw away recurrence completely. Build the model from attention alone: the **Transformer**.

How attention works (Phase 7): every word makes three vectors — a **Query** ("what am I looking for?"), a **Key** ("what do I contain?") and a **Value** ("what I hand over if chosen"). Compare each Query with every Key (dot product → **softmax** → weights that sum to 1), then take a weighted mix of the Values. So "it" can attend directly to "trophy" no matter how far away — **one step, not fifty**.

Extras that make it work: **multi-head** (several viewpoints in parallel), **positional encoding** (attention alone ignores order, so add position information), **residual connections** and **layer normalisation** (stable deep stacks), a **feed-forward** sublayer, and for generating text a **causal mask** (a word may only look at the *past*, never the answer it must predict).

**Why it beat RNNs:**
| | RNN / LSTM | Transformer |
|---|---|---|
| Distance between two words | grows with distance | 1 step |
| Parallel training on GPU | no (word by word) | yes (all positions at once) |
| Cost with long text | grows linearly | grows with length² (a real trade-off!) |
| Memory | one squeezed state | looks at everything |

**The wall.** Not a failure — a new question: it trains in parallel, so *how big can we go?*

**Terms:** *attention, query/key/value, scaled dot-product, multi-head, causal mask, positional encoding, encoder vs decoder, residual, LayerNorm.* → **Phase 7**.

---

## Chapter 7 — From Transformer to LLM: scale and one simple game (2018 → 2022)

**The story.** Two ideas fused:

1. **A beautifully simple training game: predict the next token.** Take enormous amounts of text; hide the next token; the loss is **cross-entropy** (the same idea as Lesson 4's logistic loss, over thousands of possible tokens). No human labels are needed — the text labels itself! (**self-supervised** learning.) To predict well, the model is forced to learn grammar, facts, style, even some reasoning patterns.
2. **Decoder-only Transformers (GPT, 2018 →)** stack Transformer blocks with a causal mask: tokens → embeddings (+ positions) → N blocks → probabilities for the next token. Generation is **autoregressive**: predict a token, append it, predict again.

Then came **scale**. GPT-2 (2019), GPT-3 (2020, 175 billion parameters). Researchers found **scaling laws**: more parameters + more data + more compute improves loss in a smooth, predictable way. Surprising abilities (few-shot learning from examples in the prompt) appeared as models grew. That is why the word **"Large"** is in **Large Language Model**.

**Raw next-token predictors are not assistants.** They continue text; they don't *follow instructions*. So after **pretraining** came **fine-tuning**: instruction tuning and learning from human preferences (**RLHF**, e.g. InstructGPT → ChatGPT, 2022). That is the step that made them usable as chat assistants.

**What stayed the same as Chapter 1?** Everything essential: a model with parameters, a loss (next-token cross-entropy), gradients by backprop, an optimiser (AdamW), and evaluation on held-out data (validation loss). Only the model (a big Transformer) and the scale changed. **Phase 8** has you build a tiny one yourself — it shows the mechanism, not the abilities of a giant.

**Terms:** *token, next-token prediction, pretraining, fine-tuning, autoregressive generation, temperature / top-k / top-p, scaling laws, RLHF.* → **Phases 8–9**.

---

## Chapter 8 — Making LLMs practical (≈2022 → now)

New walls, new engineering:

| Wall | Fix |
|---|---|
| Generating is slow (recomputes attention over the past each token) | **KV cache**: store each token's Keys/Values once |
| Models too big for memory | **Quantization**: store weights in fewer bits (e.g. 16 → 4) |
| Training too big for one batch/GPU | gradient accumulation, mixed precision, checkpoints |
| Confident-sounding falsehoods (**hallucination**) | better data, retrieval of real documents, verification, honest evaluation |
| An LLM only produces text | **Agents**: wrap the model in a *harness* — it can call **tools** (search, code, databases), with schemas, validation, stop conditions and safety checks (**Phase 10**) |
| A notebook model helps nobody | serve it with an **API** (FastAPI), package with **Docker**, add a UI (**Phases 11–12**) |

---

## The whole thread in one table

| Era | Model is… | Learns… | Needed because the previous… | Same loop? |
|---|---|---|---|---|
| Rules | `if / else` | nothing | — | no learning |
| Classical ML | `w·x + b`, trees, SVM | parameters | rules couldn't cover reality | loss → gradient → update |
| Neural net | layers of (linear + activation) | parameters *and* internal features | straight lines & hand-made features were limiting | + backprop |
| Deep / CNN | many layers, shared filters | visual features automatically | shallow nets/hand features hit a ceiling; data + GPUs arrived | same |
| RNN / LSTM | network with memory | patterns over time | CNNs/MLPs ignore order | + backprop through time |
| Transformer | attention blocks | relations between any two tokens | RNNs forgot, and were slow & sequential | same |
| LLM | giant Transformer | language by predicting the next token | Transformers could be scaled and trained on all text | same loop at huge scale |
| Agent / product | LLM + tools + API | — | text alone doesn't do tasks | engineering around the model |

## Why this repo is ordered the way it is
`Phase 1` (loss, gradient descent) → `Phase 2` (math behind it: derivatives, softmax, entropy) → `Phase 3` (stack layers, backprop) → `Phase 4` (CNN: weight sharing in space) → `Phase 5` (RNN/LSTM: weight sharing in time) → `Phase 6` (tokens & embeddings) → `Phase 7` (attention) → `Phase 8–9` (a language model; training & inference) → `Phases 10–12` (agents, APIs, deployment) → `Phase 13` (interviews).

In this repo the same story runs through the modules in the recommended reading order: see the [root README](../../README.md) and the [ROADMAP](../../ROADMAP.md).

## Retell it yourself (best interview practice)
Explain out loud, no notes, one sentence per arrow:
**rules → ML → neural nets → deep learning/CNN → RNN/LSTM → attention/Transformer → LLM → agents.**
For each arrow answer: *What was the wall? What was the new idea? What stayed the same?*

*Honesty note:* years and attributions above are simplified for storytelling — verify specifics (e.g. exact dates, who deserves credit for backpropagation) before quoting them in an interview.