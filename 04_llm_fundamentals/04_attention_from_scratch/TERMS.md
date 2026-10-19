# Terms — Attention From Scratch

Folder: `04_llm_fundamentals/04_attention_from_scratch` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [04 LLM fundamentals README](../README.md)

**Terms in this folder:** [numpy](#numpy) · [np.random.default_rng](#nprandomdefault_rng) · [Random seed](#random-seed) · [rng.normal](#rngnormal) · [Array shape](#array-shape) · [Token vector](#token-vector) · [d_model](#d_model) · [Weight matrix](#weight-matrix) · [Query, Key, Value](#query-key-value) · [Matrix multiplication (@)](#matrix-multiplication-) · [Transpose (.T)](#transpose-t) · [Attention scores](#attention-scores) · [Scaling by sqrt(d_head)](#scaling-by-sqrtd_head) · [softmax (function)](#softmax-function) · [keepdims](#keepdims) · [axis=-1](#axis-1) · [Attention weights](#attention-weights) · [Self-attention](#self-attention) · [Causal mask](#causal-mask) · [np.triu](#nptriu) · [np.where](#npwhere) · [-infinity masking](#-infinity-masking) · [Transformer block](#transformer-block) · [Quadratic cost of attention](#quadratic-cost-of-attention) · [Docstring](#docstring) · [list](#list) · [Softmax saturation](#softmax-saturation)

---

## numpy  `(third-party: numpy)`
**1. Definition:** A Python library that provides the `ndarray`, a fast fixed-type n-dimensional array, together with vectorised mathematical functions.
**2. In plain words:** Python lists are flexible but slow for maths. numpy stores numbers in compact blocks and applies an operation to every element at once. Almost all ML libraries (PyTorch included) copy its ideas, so learning numpy first makes tensors feel familiar. Limitation: every element in an array has one type, unlike a list.
**3. Example to quote:** > "I used numpy to build attention by hand, because `x @ W_q` multiplies a whole 4x8 matrix in one line instead of looping."
```python
import numpy as np
x = np.ones((4, 8))   # 4 rows, 8 columns
```
**Seen in:** `main.py`, first import.

## np.random.default_rng  `(third-party: numpy)`
**1. Definition:** Creates a `Generator` object, numpy's modern random number generator, optionally initialised with a seed.
**2. In plain words:** It is the object you ask for random numbers. Passing `seed=0` makes it produce the same "random" sequence every run, so your printed output matches the walkthrough. The older `np.random.seed` style works on a hidden global generator; `default_rng` keeps the state in an object you hold.
**3. Example to quote:** > "I created `rng = np.random.default_rng(seed=0)` so my random weights are the same every run and I can compare outputs."
```python
rng = np.random.default_rng(seed=0)
```
**Seen in:** `main.py`, top of the file.

## Random seed  `(concept)`
**1. Definition:** A starting number given to a pseudo-random generator so that it produces a reproducible sequence of values.
**2. In plain words:** Computers do not produce true randomness; they run a formula. If you start the formula from the same seed you get exactly the same numbers. That is how experiments become repeatable: when something changes in the output you know it was your code, not luck.
**3. Example to quote:** > "With seed 0 the attention weight table is identical on every run; without it each run would give different numbers and I could not debug."

**Seen in:** `main.py`, `seed=0`.

## rng.normal  `(third-party: numpy)`
**1. Definition:** `Generator.normal(loc=0.0, scale=1.0, size=None)` draws samples from a normal (bell-curve) distribution and returns them as an array of shape `size`.
**2. In plain words:** It fills a matrix with random numbers that cluster around zero. Real models start with small random weights like this before training adjusts them. The `size` argument is a shape tuple such as `(4, 8)`.
**3. Example to quote:** > "I made the token vectors with `rng.normal(size=(4, 8))`, which gives four tokens with eight random numbers each."
```python
x = rng.normal(size=(4, 8))
print(x.shape)  # (4, 8)
```
**Seen in:** `main.py`, creating `x`, `W_q`, `W_k`, `W_v`.

## Array shape  `(concept)`
**1. Definition:** The tuple of dimension sizes of an array, available as `.shape`, for example `(4, 8)` meaning 4 rows and 8 columns.
**2. In plain words:** Most beginner bugs in ML are shape mismatches, so you print shapes constantly. A shape tells you what each axis means: here rows are tokens and columns are features per token. Always ask "what does each number in this shape stand for?"
**3. Example to quote:** > "Input is `(4, 8)` and the attention output is also `(4, 8)`, which is why transformer layers can be stacked."
```python
print(x.shape)  # (4, 8)
```
**Seen in:** `main.py`, every `print(... .shape)`.

## Token vector  `(concept)`
**1. Definition:** A list of numbers (a row of the input matrix) that represents one token in a form the network can compute with.
**2. In plain words:** A model cannot read the word "cat"; it needs numbers. Each token becomes a row of numbers, 8 of them here (thousands in real models). Where these numbers really come from is embeddings, covered in step 03.
**3. Example to quote:** > "My sentence 'The cat sat down' is a 4x8 matrix: four tokens, each an 8-number vector."

**Seen in:** `main.py`, section 1.

## d_model  `(concept)`
**1. Definition:** The width of the vectors flowing through a transformer, that is, the number of features used to represent each token (8 in this demo, 768 in GPT-2).
**2. In plain words:** It is the size of each token's "description". A bigger `d_model` lets the model store more information per token but costs more memory and compute. It stays constant from layer to layer so layers can be stacked.
**3. Example to quote:** > "In my demo `d_model = 8`; GPT-2 small uses 768, so every token is a 768-number vector."

**Seen in:** `main.py`, `seq_len, d_model = 4, 8`.

## Weight matrix  `(concept)`
**1. Definition:** A matrix of learnable numbers that a layer multiplies its input by; training adjusts these numbers to reduce the loss.
**2. In plain words:** Weights are the "knowledge" of a model. Here `W_q`, `W_k`, `W_v` are random, so the attention pattern means nothing yet, but the mechanics are the same as in GPT. Training is simply nudging these matrices.
**3. Example to quote:** > "The weights are random in my demo, so attention is meaningless; training would adjust `W_q`, `W_k`, `W_v` until useful patterns appear."

**Seen in:** `main.py`, section 2.

## Query, Key, Value  `(concept)`
**1. Definition:** Three linear projections of each token vector: the Query (what the token is looking for), the Key (what the token offers to be matched against) and the Value (the content passed on if the token is attended to).
**2. In plain words:** Think of a library search. The Query is your search phrase, each book's Key is its label, and the Value is the book's content. You compare your Query with every Key, then read mostly from the best matches. Making three separate projections lets a token search differently from how it advertises itself.
**3. Example to quote:** > "I compute `Q = x @ W_q`, `K = x @ W_k`, `V = x @ W_v`; the Query of 'sat' is compared with the Key of every other token."
```python
Q = x @ W_q
K = x @ W_k
V = x @ W_v
```
**Seen in:** `main.py`, section 2.

## Matrix multiplication (@)  `(syntax)`
**1. Definition:** The `@` operator multiplies two arrays as matrices; for `A @ B` the last dimension of `A` must equal the first dimension of `B`.
**2. In plain words:** Each output cell is a row of A multiplied element by element with a column of B and summed. It is how a layer mixes all input features into new features. If you see a "mismatch in its core dimension" error, the inner sizes disagree.
**3. Example to quote:** > "`(4,8) @ (8,8)` gives `(4,8)`; if I change `W_k` to `(7,8)` I get a ValueError because 8 does not equal 7."
```python
np.ones((2, 3)) @ np.ones((3, 4))  # shape (2, 4)
```
**Seen in:** `main.py`, `x @ W_q`, `Q @ K.T`, `weights @ V`.

## Transpose (.T)  `(third-party: numpy)`
**1. Definition:** The `.T` attribute returns the array with its axes reversed; for a 2D array rows become columns.
**2. In plain words:** To compare every query with every key you need the key matrix flipped so the multiplication lines up: `(4,8) @ (8,4)`. Transposing does not change the numbers, only how the axes are arranged.
**3. Example to quote:** > "`K.T` turns the `(4,8)` key matrix into `(8,4)`, so `Q @ K.T` produces a 4x4 table of scores."
```python
np.ones((2, 3)).T.shape  # (3, 2)
```
**Seen in:** `main.py`, section 3.

## Attention scores  `(concept)`
**1. Definition:** The matrix `Q @ K.T` (scaled), where entry (i, j) measures how relevant token j is to token i.
**2. In plain words:** It is a table of raw "how interesting is each token to me" numbers. Large positive means pay attention, negative means ignore. They are not yet probabilities; softmax converts them in the next step.
**3. Example to quote:** > "My scores matrix is 4x4: every token compared against every token, row i being what token i is looking at."

**Seen in:** `main.py`, section 3.

## Scaling by sqrt(d_head)  `(concept)`
**1. Definition:** Dividing the dot-product scores by the square root of the key dimension to keep their magnitude roughly constant regardless of vector size.
**2. In plain words:** The dot product of long vectors grows with the vector length. Huge scores make softmax extremely spiky (almost one-hot) and gradients tiny. Dividing by `sqrt(d_head)` keeps numbers moderate so training stays stable.
**3. Example to quote:** > "I divide by `np.sqrt(d_head)` so scores do not blow up as vectors get wider; without it softmax saturates."
```python
scores = Q @ K.T / np.sqrt(d_head)
```
**Seen in:** `main.py`, section 3.

## softmax (function)  `(concept)`
**1. Definition:** A function that maps a vector of real numbers to probabilities that are non-negative and sum to 1: `softmax(a)_i = exp(a_i) / sum_j exp(a_j)`.
**2. In plain words:** It turns arbitrary scores into shares of 100%. Bigger scores get disproportionately bigger shares. The code subtracts the maximum first, which does not change the result but stops `exp` from overflowing on large numbers.
**3. Example to quote:** > "After softmax each row of my attention table sums to 1.0, so each token splits its attention across tokens."
```python
a = a - a.max(axis=-1, keepdims=True)
e = np.exp(a); e / e.sum(axis=-1, keepdims=True)
```
**Seen in:** `main.py`, `def softmax`.

## keepdims  `(third-party: numpy)`
**1. Definition:** A reduction argument (`max`, `sum`, ...) that keeps the reduced axis with size 1 instead of removing it.
**2. In plain words:** After taking a row-wise max you get shape `(4,)`, which cannot be broadcast against `(4,4)` correctly. With `keepdims=True` the shape is `(4,1)`, and numpy subtracts each row's max from that row. It avoids a common shape bug.
**3. Example to quote:** > "I use `keepdims=True` so `a.max(axis=-1)` has shape `(4,1)` and broadcasts across each row."
```python
x.max(axis=-1, keepdims=True).shape  # (4, 1)
```
**Seen in:** `main.py`, inside `softmax`.

## axis=-1  `(third-party: numpy)`
**1. Definition:** Selecting the last axis of an array for a reduction; for a 2D array it means "across columns, one result per row".
**2. In plain words:** Negative axis numbers count from the end, so `-1` always means the final dimension no matter how many dimensions there are. For attention, the last axis is "the tokens I could look at", so softmax and sums over it give per-row results.
**3. Example to quote:** > "`weights.sum(axis=1)` printed `[1. 1. 1. 1.]`, confirming every row is a probability distribution."

**Seen in:** `main.py`, `softmax` and the row-sum print.

## Attention weights  `(concept)`
**1. Definition:** The output of softmax over the scores: a matrix whose row i gives the fraction of attention token i pays to each token.
**2. In plain words:** These are the final "how much do I listen to each token" percentages. Multiplying them with the Value vectors produces a blended output per token. In a trained model these patterns capture grammar and meaning; here they are random.
**3. Example to quote:** > "Row 1 of my table is `[1, 0, 0, 0]` in the causal case: 'The' can only look at itself."

**Seen in:** `main.py`, `weights = softmax(scores)`.

## Self-attention  `(concept)`
**1. Definition:** An operation in which each position in a sequence computes a weighted combination of the value vectors of all (allowed) positions in the same sequence, with weights from query-key similarity.
**2. In plain words:** Each token looks at the other tokens of its own sentence and mixes in useful information. "It" can pull in meaning from "cat", for example. The output has the same shape as the input, so the operation can be repeated layer after layer.
**3. Example to quote:** > "Self-attention lets every token look at the others: I score every pair, softmax into weights, then take a weighted average of the Values."
```python
out = weights @ V
```
**Seen in:** `main.py`, section 4.

## Causal mask  `(concept)`
**1. Definition:** A mask that prevents position i from attending to positions after i, so each token only uses itself and earlier tokens.
**2. In plain words:** A GPT-style model is trained to predict the next token. If it could see the answer in the future tokens it would cheat. The mask hides the future, so the upper triangle of the weight table becomes zero.
**3. Example to quote:** > "With the causal mask, row 2 of my table is `[0.17, 0.36, 0.47, 0]`: 'sat' sees 'The', 'cat' and itself but not 'down'."

**Seen in:** `main.py`, section 5.

## np.triu  `(third-party: numpy)`
**1. Definition:** `np.triu(m, k)` returns a copy of `m` with elements below the k-th diagonal set to zero, keeping the upper triangle.
**2. In plain words:** Using a matrix of ones with `k=1` marks everything strictly above the main diagonal, which are exactly the "future" positions. Cast to boolean, it becomes the mask.
**3. Example to quote:** > "`np.triu(np.ones((4,4), dtype=bool), k=1)` gives True for every future position."
```python
mask = np.triu(np.ones((4, 4), dtype=bool), k=1)
```
**Seen in:** `main.py`, section 5.

## np.where  `(third-party: numpy)`
**1. Definition:** `np.where(cond, a, b)` returns an array taking elements from `a` where `cond` is True and from `b` elsewhere.
**2. In plain words:** It is an element-wise if/else without a loop. The code uses it to replace scores at masked positions with `-inf` and keep the rest.
**3. Example to quote:** > "I used `np.where(mask, -np.inf, scores)` to blank out the future scores in one line."
```python
np.where(np.array([1, -1]) > 0, 1, 0)  # array([1, 0])
```
**Seen in:** `main.py`, `masked_scores`.

## -infinity masking  `(concept)`
**1. Definition:** Setting scores to negative infinity before softmax so their exponentials equal zero and they receive exactly zero probability.
**2. In plain words:** You cannot just delete entries from a matrix, so you push them to the lowest possible value. `exp(-inf)` is 0, so softmax gives those positions a weight of exactly 0 and the remaining weights still sum to 1.
**3. Example to quote:** > "I apply the mask before softmax because masking afterwards would leave rows that no longer sum to 1."

**Seen in:** `main.py`, section 5.

## Transformer block  `(concept)`
**1. Definition:** The repeating unit of a transformer: a multi-head self-attention sub-layer followed by a small feed-forward network (MLP), each with residual connections and normalisation.
**2. In plain words:** One block lets tokens exchange information (attention) and then process it individually (MLP). Stacking blocks (6 in distilgpt2, 12 in gpt2) gives deeper understanding. Several "heads" run attention in parallel, each learning a different pattern.
**3. Example to quote:** > "distilgpt2 is six of these blocks stacked; my demo is just the attention half of a single head."

**Seen in:** `WALKTHROUGH.md`, execution trace.

## Quadratic cost of attention  `(concept)`
**1. Definition:** Because the score matrix is `seq_len x seq_len`, attention compute and memory grow proportionally to the square of the sequence length.
**2. In plain words:** Doubling the context makes the table four times bigger. That is why long prompts are slow and expensive, and why tricks such as KV caching and efficient attention exist.
**3. Example to quote:** > "At 4 tokens the table has 16 cells; at 4,000 tokens it has 16 million, which is why context length is a cost trade-off."

**Seen in:** `WALKTHROUGH.md`, "Say it aloud".

## Docstring  `(syntax)`
**1. Definition:** A string literal placed as the first statement of a module, function or class that documents it and is stored in `__doc__`.
**2. In plain words:** It is the official documentation of a function, written in triple quotes. Tools like `help()` display it. The `softmax` function has one explaining what it returns.
**3. Example to quote:** > "I wrote a docstring on `softmax` so `help(softmax)` explains that it returns probabilities summing to 1."
```python
def softmax(a):
    """Turn numbers into probabilities."""
```
**Seen in:** `main.py`, `def softmax`.

## list  `(Python built-in)`
**1. Definition:** An ordered, mutable sequence of Python objects written with square brackets.
**2. In plain words:** The code uses a list of strings to hold the four tokens. Lists can hold mixed items and grow; `len(tokens)` gives the count. They are the everyday container in Python.
**3. Example to quote:** > "`tokens = ['The', 'cat', 'sat', 'down']` is a list; `len(tokens)` is 4."
```python
tokens = ["The", "cat", "sat", "down"]
```
**Seen in:** `main.py`, section 1.

## Softmax saturation  `(concept)`
**1. Definition:** When the inputs to softmax are very large in magnitude, the output becomes almost one-hot and its gradient is almost zero.
**2. In plain words:** If the scores are huge, one token gets weight 1.00 and the rest 0.00, and training cannot nudge the weights. Dot products of d-dimensional vectors have variance about d, so dividing by sqrt(d) keeps scores near variance 1. `main.py` prints this: at d = 1024 the unscaled variance is about 1051 and the max weight is 1.00; scaled, 1.03 and 0.29.
**3. Example to quote:** > "We divide by the square root of d_head so the score variance stays about 1 and softmax does not saturate."

**Seen in:** `main.py` section 6.
