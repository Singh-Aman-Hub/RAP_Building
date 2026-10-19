# Terms — The Transformer Block

Folder: `04_llm_fundamentals/05_transformer_block` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [04 LLM fundamentals README](../README.md)

**Terms in this folder:** [Transformer block](#transformer-block) · [Residual connection](#residual-connection) · [LayerNorm](#layernorm) · [Pre-norm](#pre-norm) · [Feed-forward network (MLP)](#feed-forward-network-mlp) · [GELU](#gelu) · [Multi-head attention](#multi-head-attention) · [head_dim](#head_dim) · [Causal mask](#causal-mask) · [12 x d_model^2](#12-x-d_model2) · [Vanishing gradient](#vanishing-gradient) · [nn.Module](#nnmodule) · [view and transpose](#view-and-transpose)

---

## Transformer block  `(concept)`
**1. Definition:** One repeated unit of a GPT model: attention and a feed-forward network, each with LayerNorm and a residual connection.
**2. In plain words:** Attention lets tokens read each other; the MLP lets each token think about what it read. The block keeps the shape unchanged, so you stack it 6, 12 or 96 times.
**3. Example to quote:** > "distilgpt2 is six identical blocks, each 7,087,872 parameters."
**Seen in:** `main.py`, section 4.

## Residual connection  `(concept)`
**1. Definition:** Adding a layer's input to its output: `x = x + layer(x)`.
**2. In plain words:** The layer only learns a correction. The original signal, and the gradient on the way back, always has a direct path. Without it, the gradient at depth 24 was 23x smaller in our run.
**3. Example to quote:** > "Removing residuals cut the input gradient from 0.334 to 0.0143 at depth 24."
```python
x = x + self.attn(self.ln1(x))
```
**Seen in:** `main.py`, `Block.forward`.

## LayerNorm  `(concept)`
**1. Definition:** Normalising each token vector to mean 0 and standard deviation 1 across its features, then applying a learned scale (gamma) and shift (beta).
**2. In plain words:** It stops numbers from drifting larger or smaller layer after layer. It works per token, so it does not depend on batch size.
**3. Example to quote:** > "My hand-written LayerNorm matched `nn.LayerNorm` to 1e-5."
**Seen in:** `main.py`, `MyLayerNorm`.

## Pre-norm  `(concept)`
**1. Definition:** Placing LayerNorm before each sub-layer (`x + f(LN(x))`) instead of after.
**2. In plain words:** GPT-2 and most modern models use this layout because it trains more stably in deep stacks.
**3. Example to quote:** > "GPT-2 is pre-norm: `ln_1` runs before attention, `ln_2` before the MLP."
**Seen in:** `main.py`, `Block`.

## Feed-forward network (MLP)  `(concept)`
**1. Definition:** Two linear layers with a non-linearity between: expand from d_model to 4 x d_model, apply GELU, shrink back.
**2. In plain words:** It acts on each token separately. It is where much of the model's stored knowledge is thought to live, and it holds two thirds of each block's parameters.
**3. Example to quote:** > "The MLP is 8 d^2 parameters; attention is 4 d^2."
**Seen in:** `main.py`, `MLP`.

## GELU  `(third-party: torch)`
**1. Definition:** A smooth activation function, roughly `x` for large positive inputs and 0 for large negative ones, with a soft curve in between.
**2. In plain words:** A gentler version of ReLU. Without a non-linearity, two linear layers would collapse into one.
**3. Example to quote:** > "GPT-2 uses GELU in its feed-forward layer."
**Seen in:** `main.py`, `MLP.forward`.

## Multi-head attention  `(concept)`
**1. Definition:** Running several attention operations in parallel on slices of the vector, then joining the results and mixing them with an output projection.
**2. In plain words:** Each head can focus on a different relationship (for example the previous word, or a matching noun). Heads split d_model; they do not add parameters.
**3. Example to quote:** > "With 4 heads and d_model 64, each head works on 16 numbers."
**Seen in:** `main.py`, `CausalSelfAttention`.

## head_dim  `(concept)`
**1. Definition:** `d_model // n_heads`, the vector size seen by one attention head.
**2. In plain words:** It must divide evenly, which is why `d_model=64, n_heads=5` raises an error.
**3. Example to quote:** > "distilgpt2: 768 / 12 = 64 numbers per head."
**Seen in:** `main.py`, section 9.

## Causal mask  `(concept)`
**1. Definition:** A triangular mask that sets scores for future positions to minus infinity before softmax.
**2. In plain words:** A token may read itself and earlier tokens only. After softmax the future gets weight exactly 0.
**3. Example to quote:** > "I verified the mask: changing the last token did not change earlier outputs."
```python
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
```
**Seen in:** `main.py`, `CausalSelfAttention.forward`.

## 12 x d_model^2  `(concept)`
**1. Definition:** The approximate parameter count of one block: 4 d^2 (attention) + 8 d^2 (MLP). Biases and LayerNorm add 13 d more.
**2. In plain words:** A quick estimate: total parameters of the blocks ~ 12 x layers x d^2. For d=768 one block is 7,077,888 + 9,984 = 7,087,872.
**3. Example to quote:** > "A block is about 12 d squared; for d=768 that is 7 million."
**Seen in:** `main.py`, section 6.

## Vanishing gradient  `(concept)`
**1. Definition:** Gradients becoming tiny as they are multiplied through many layers, so early layers barely learn.
**2. In plain words:** The loss signal fades on its way back. Residual connections give it a shortcut.
**3. Example to quote:** > "Without residuals the gradient norm at the input fell to 1.4e-2 at depth 24."
**Seen in:** `main.py`, experiment A.

## nn.Module  `(third-party: torch)`
**1. Definition:** The PyTorch base class for layers and models; it tracks parameters and calls `forward` when you call the object.
**2. In plain words:** Define the parts in `__init__`, the computation in `forward`, and `.parameters()` finds every weight for you.
**3. Example to quote:** > "`count_params(block)` works because `nn.Module` registers all sub-layer weights."
**Seen in:** `main.py`, every class.

## view and transpose  `(third-party: torch)`
**1. Definition:** `view` reshapes a tensor without copying; `transpose(a, b)` swaps two axes.
**2. In plain words:** Split heads with `view` then `transpose`; join them with `transpose` then `view`. The order matters: a wrong order gives the right shape and wrong numbers.
**3. Example to quote:** > "`(B,T,C)` becomes `(B,H,T,D)` with `view` then `transpose(1, 2)`."
**Seen in:** `main.py`, `CausalSelfAttention.forward`.
