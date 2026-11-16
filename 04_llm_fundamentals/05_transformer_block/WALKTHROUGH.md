# Step 05 - The transformer block (Assembling the Beast)

## The story so far
[Step 04](../04_attention_from_scratch/WALKTHROUGH.md) gave each token a way to read the other tokens. But it ended with a warning: that is ONE mixing step. One layer cannot build deep understanding, nothing keeps the signal stable if we stack many, and no step lets a token "think" about what it just read. This step wraps attention into a block that can be stacked.

## The pain point
Stack plain attention layers and three things go wrong. (1) Each layer only blends vectors; there is no per-token computation. (2) Gradients must travel back through every layer, and they shrink along the way, so early layers barely learn. (3) Number scales drift from layer to layer, so training becomes unstable.

## The spark
Add three small helpers around attention:
- **Residual connection**: `x = x + layer(x)`. The input is added back, so the signal and the gradient always have a short path.
- **LayerNorm**: re-centre and re-scale every token vector before each sub-layer, so scales stay tame.
- **Feed-forward MLP**: expand each token vector 4x, apply GELU, shrink back. This is the per-token thinking step.

Put together: `x = x + attn(LN(x))` then `x = x + mlp(LN(x))`. That is one GPT block. Shape in = shape out `(batch, seq_len, d_model)`, so you can stack as many as you like.

## Goal
Build one block by hand in torch, stack it, check the shape stays constant, check the parameter count against `12 * d_model^2`, break it on purpose, and compare with a real distilgpt2 block.

## Run it
```bash
.venv/bin/python 04_llm_fundamentals/05_transformer_block/main.py
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Transformer block | transformer layer, decoder layer, GPT block | Yes | One unit of attention + MLP with residuals and norms, stacked N times |
| LayerNorm / RMSNorm / BatchNorm | normalisation layers | Different | LayerNorm normalises over the feature axis per token (subtract mean, divide by std); RMSNorm skips the mean subtraction (cheaper, used in Llama-family models); BatchNorm normalises over the batch and is rare in LLMs |
| Residual connection | skip connection, shortcut, `x + f(x)` | Yes | Adds the layer input back to its output so gradients have a direct path |
| Pre-norm vs post-norm | norm before vs after the sublayer | No | GPT-2 and modern LLMs are pre-norm (`x + f(LN(x))`); the original 2017 transformer was post-norm |
| MLP | feed-forward network (FFN), position-wise FFN, `c_fc/c_proj` | Yes | Two linear layers with a nonlinearity, expand 4x then shrink; applied to each token independently |
| GELU / ReLU / SwiGLU | activation functions | Different | GELU is smooth ReLU used in GPT-2; SwiGLU (gated) is used in Llama-style models |
| Multi-head attention | MHA, n_heads | Yes | Several attention computations on slices of the width; MQA and GQA share K/V between heads to shrink the KV cache |
| `n_embd` / `hidden_size` / `d_model` / C | model width | Yes | Same number under different names |
| Decoder-only vs encoder-decoder | GPT vs T5/BERT-like | No | Decoder-only uses a causal mask and only predicts the next token; modern chat LLMs are decoder-only |

## Code, step by step

### 1. Module docstring and notation  (`main.py` lines 1–16)

```python
"""
GOAL:   Build ONE GPT-style transformer block by hand (multi-head causal attention + residual +
        LayerNorm + feed-forward MLP), stack it N times, and watch the shape stay constant.

RUN:    from this folder:  python main.py
        from repo root:    .venv/bin/python 04_llm_fundamentals/05_transformer_block/main.py
EXPECTED OUTPUT (short): shapes (2, 10, 64) after every block; one block has 12*d_model^2 weights
        plus a few biases; removing the residual makes the gradient vanish; removing LayerNorm
        makes activations grow with depth; d_model not divisible by n_heads raises an error;
        a real distilgpt2 block has exactly the same parameter structure.
NEEDS:  torch (+ transformers and the cached distilgpt2 for the last section only).

One attention layer is a single mixing step.  A block wraps it with the pieces that make
DEPTH possible:  residual (keep the signal), LayerNorm (keep the scale), MLP (think per token).
Notation: B = batch, T = seq_len, C = d_model, H = n_heads, D = C // H = head_dim.
"""
```

- **In simple words:** Says what we build (one GPT block) and defines the letters B, T, C, H, D used in shape comments.
- **Exactly:** B = batch, T = sequence length, C = d_model, H = n_heads, D = C // H = head dim. Promised outputs: shapes `(2, 10, 64)` after every block, `12*d^2` weights per block, experiments for residual and LayerNorm, and a comparison with real distilgpt2.
- **Why it is here:** A shape vocabulary lets every comment in the file be read as an exact tensor shape.

### 2. Imports and a fixed seed  (`main.py` lines 18–22)

```python
import math
import torch
import torch.nn as nn

torch.manual_seed(0)
```

- **In simple words:** Load maths, PyTorch and the layers module, and make random weights repeatable.
- **Exactly:** `torch.manual_seed(0)` fixes the initial weights and the random input, so the printed numbers below are reproducible.
- **Why it is here:** Experiments compare printed numbers, so they must be repeatable.

### 3. LayerNorm by hand: parameters  (`main.py` lines 24–32)

```python
# ---------------------------------------------------------------------------
# 1. LayerNorm by hand: normalise each token vector to mean 0 / std 1, then learn a scale+shift
# ---------------------------------------------------------------------------
class MyLayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))    # shape: (C,)  learned scale
        self.beta = nn.Parameter(torch.zeros(d_model))    # shape: (C,)  learned shift
        self.eps = eps
```

- **In simple words:** A LayerNorm layer has two learnable lists of numbers: a scale and a shift, one of each per feature.
- **Exactly:** `gamma` is `nn.Parameter(torch.ones(C))` and `beta` is `torch.zeros(C)`, both `(C,)`. `nn.Parameter` registers them as trainable. `eps = 1e-5` is a small constant to avoid dividing by 0.
- **Why it is here:** Starting at gamma=1 and beta=0 makes the layer initially a pure normaliser; training can then rescale it.

### 4. LayerNorm by hand: the forward pass  (`main.py` lines 34–38)

```python
    def forward(self, x):                                  # x shape: (B, T, C)
        mean = x.mean(dim=-1, keepdim=True)                # shape: (B, T, 1)
        var = x.var(dim=-1, unbiased=False, keepdim=True)  # shape: (B, T, 1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)    # shape: (B, T, C)
        return self.gamma * x_hat + self.beta              # shape: (B, T, C)
```

- **In simple words:** For each token, subtract the average of its numbers and divide by their spread, then rescale.
- **Exactly:** `x.mean(dim=-1, keepdim=True)` is `(B, T, 1)` (mean over the C features of each token). `x.var(dim=-1, unbiased=False, ...)` is the population variance `(B, T, 1)`. `x_hat = (x - mean) / sqrt(var + eps)` is `(B, T, C)` with mean 0 and std 1 per token; `gamma * x_hat + beta` rescales. Verified later: `MyLayerNorm == nn.LayerNorm: True`.
- **Why it is here:** Keeps activations at a stable scale as depth grows (experiment B). Normalising per token (not per batch) works for any batch size and sequence length.

### 5. Causal self-attention: layer setup  (`main.py` lines 40–50)

```python
# ---------------------------------------------------------------------------
# 2. Multi-head causal self-attention (the idea from ../04_attention_from_scratch, batched + split in heads)
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads                  # D
        self.qkv = nn.Linear(d_model, 3 * d_model)          # nn.Linear weight shape: (3C, C) = (out, in); Q, K, V packed
        self.proj = nn.Linear(d_model, d_model)             # weight shape: (C, C); mixes the heads back
```

- **In simple words:** Prepare one combined layer that creates Q, K and V and another that mixes the heads together.
- **Exactly:** `assert d_model % n_heads == 0` guards the head split. `head_dim = C // H` = 16 for C=64, H=4. `nn.Linear(C, 3C)` has weight `(3C, C)` in PyTorch layout (GPT-2 stores `(C, 3C)`) and produces Q, K and V at once; `proj` is `Linear(C, C)`.
- **Why it is here:** One big matmul for Q, K, V is faster than three small ones. The assert gives a readable error rather than a reshape crash.

### 6. Compute Q, K, V in one go  (`main.py` lines 52–54)

```python
    def forward(self, x):                                   # x shape: (B, T, C)
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)              # each shape: (B, T, C)
```

- **In simple words:** Apply the big layer and cut the result into three equal parts.
- **Exactly:** `x` is `(B, T, C) = (2, 10, 64)`. `self.qkv(x)` is `(2, 10, 192)`; `.split(C, dim=-1)` returns three `(2, 10, 64)` tensors.
- **Why it is here:** Packs the three projections of step 04 into a single multiplication.

### 7. Split into heads  (`main.py` lines 55–58)

```python
        # split C into H heads of size D, then move heads before time so each head is independent
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)   # shape: (B, H, T, D)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)   # shape: (B, H, T, D)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)   # shape: (B, H, T, D)
```

- **In simple words:** Cut each 64-number vector into 4 chunks of 16 so 4 independent attentions can run in parallel.
- **Exactly:** `.view(B, T, H, D)` reshapes `(2,10,64) -> (2,10,4,16)` without copying; `.transpose(1, 2)` swaps axes to `(B, H, T, D) = (2, 4, 10, 16)` so heads act like extra batch entries.
- **Why it is here:** Different heads can learn different relationships (for example syntax vs coreference). Forgetting `transpose` silently scrambles the output (BREAK IT at the end).

### 8. Scores, causal mask and softmax per head  (`main.py` lines 59–62)

```python
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)     # shape: (B, H, T, T)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)  # shape: (T, T) True = future
        scores = scores.masked_fill(mask, float("-inf"))                # shape: (B, H, T, T) no peeking ahead
        weights = torch.softmax(scores, dim=-1)                         # shape: (B, H, T, T) rows sum to 1
```

- **In simple words:** Same as step 04: compare queries with keys, hide the future, turn each row into probabilities.
- **Exactly:** `k.transpose(-2, -1)` is `(B,H,D,T)`; `q @ k^T` is `(B,H,T,T) = (2,4,10,10)`, divided by `sqrt(D) = 4`. `torch.triu(ones(T,T,bool), diagonal=1)` is True above the diagonal; `masked_fill(mask, -inf)` removes the future; `softmax(dim=-1)` makes each row sum to 1.
- **Why it is here:** The mask lets all T positions be trained at once without peeking ahead; scaling prevents saturated softmax.

### 9. Weighted sum of values and glue the heads back  (`main.py` lines 63–65)

```python
        out = weights @ v                                               # shape: (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, C)            # shape: (B, T, C) glue heads together
        return self.proj(out)                                           # shape: (B, T, C)
```

- **In simple words:** Blend values by attention weights, then lay the 4 head outputs side by side and mix them.
- **Exactly:** `weights @ v` is `(B,H,T,T) @ (B,H,T,D) -> (B,H,T,D)`. `transpose(1, 2).contiguous().view(B, T, C)` returns to `(2, 10, 64)` (`contiguous` is needed because a transposed tensor is not laid out for `view`). `proj` is the output matrix `W_O`.
- **Why it is here:** The output has the same shape as the input, so blocks can be stacked.

### 10. The feed-forward MLP  (`main.py` lines 67–78)

```python
# ---------------------------------------------------------------------------
# 3. Feed-forward MLP: expand 4x, GELU, shrink back.  Applied to each token separately.
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model, 4 * d_model)           # nn.Linear weight shape: (4C, C) = (out, in)
        self.proj = nn.Linear(4 * d_model, d_model)         # nn.Linear weight shape: (C, 4C) = (out, in)

    def forward(self, x):                                   # x shape: (B, T, C)
        h = nn.functional.gelu(self.fc(x))                  # shape: (B, T, 4C)
        return self.proj(h)                                 # shape: (B, T, C)
```

- **In simple words:** A small two-layer network applied to each token alone: widen to 4 times, apply a smooth nonlinearity, narrow back.
- **Exactly:** `fc` is `Linear(C, 4C)` (64 -> 256), `proj` is `Linear(4C, C)`. `gelu(fc(x))` is `(B, T, 4C) = (2, 10, 256)`. Weights: `C*4C + 4C*C = 8C^2`. Attention is the "mixing between tokens" step; the MLP is "thinking within each token" and holds much of the model's stored knowledge.
- **Why it is here:** Without a nonlinearity the two linear layers would collapse into one and the block could not compute anything new.

### 11. The block: pre-norm with residuals  (`main.py` lines 80–98)

```python
# ---------------------------------------------------------------------------
# 4. The block: pre-norm GPT layout.  x = x + attn(LN(x));  x = x + mlp(LN(x))
#    use_residual / use_ln switches exist ONLY so experiments can break the block.
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d_model, n_heads, use_residual=True, use_ln=True):
        super().__init__()
        self.ln1 = MyLayerNorm(d_model) if use_ln else nn.Identity()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = MyLayerNorm(d_model) if use_ln else nn.Identity()
        self.mlp = MLP(d_model)
        self.use_residual = use_residual

    def forward(self, x):                                   # x shape: (B, T, C)
        a = self.attn(self.ln1(x))                          # shape: (B, T, C)
        x = x + a if self.use_residual else a               # shape: (B, T, C) residual = add the input back
        m = self.mlp(self.ln2(x))                           # shape: (B, T, C)
        x = x + m if self.use_residual else m               # shape: (B, T, C)
        return x
```

- **In simple words:** Assemble the pieces: normalise, attend, add the input back; normalise, think, add back again.
- **Exactly:** Computes `x = x + attn(ln1(x))` then `x = x + mlp(ln2(x))`; every intermediate is `(B, T, C)`. `use_residual` and `use_ln` exist only so the experiments can switch them off. `nn.Identity()` is a layer that returns its input.
- **Why it is here:** The residual path lets gradients and information flow around each sublayer; pre-norm keeps the residual stream clean. This is how GPT-2 and most modern LLMs are laid out.

### 12. Count parameters helper  (`main.py` lines 100–101)

```python
def count_params(module):
    return sum(p.numel() for p in module.parameters())
```

- **In simple words:** Add up every number in a module.
- **Exactly:** `module.parameters()` yields every weight tensor; `.numel()` is the element count; the generator sums them.
- **Why it is here:** Used to compare with `12*d^2` and with the real model.

### 13. Fake token vectors  (`main.py` lines 103–104)

```python
B, T, C, H = 2, 10, 64, 4                                   # batch, seq_len, d_model, n_heads
x = torch.randn(B, T, C)                                    # shape: (B, T, C) fake token vectors
```

- **In simple words:** Random input standing in for embedded tokens.
- **Exactly:** `B, T, C, H = 2, 10, 64, 4`. `x = torch.randn(2, 10, 64)`.
- **Why it is here:** Small sizes make the experiments quick on a laptop.

### 14. One block keeps the shape  (`main.py` lines 106–109)

```python
print("== 1. one block ==")
print("input shape :", tuple(x.shape))
block = Block(C, H)
print("output shape:", tuple(block(x).shape), " <- identical: that is what lets us stack blocks")
```

- **In simple words:** Pass the tensor through one block and compare shapes.
- **Exactly:** Real output: input `(2, 10, 64)`, output `(2, 10, 64)`.
- **Why it is here:** Identical shapes are what lets us stack blocks.

### 15. Check our LayerNorm against PyTorch's  (`main.py` lines 111–112)

```python
# sanity check: our LayerNorm equals torch's
print("MyLayerNorm == nn.LayerNorm:", torch.allclose(MyLayerNorm(C)(x), nn.LayerNorm(C)(x), atol=1e-5))
```

- **In simple words:** Make sure our hand-written version gives the same answer as the library.
- **Exactly:** `torch.allclose(..., atol=1e-5)` returns `True` (real output). Both start with gamma=1, beta=0.
- **Why it is here:** Comparing to the library is the cheapest correctness test.

### 16. Check causality  (`main.py` lines 114–117)

```python
# causal check: changing the LAST token must not change earlier outputs
x2 = x.clone(); x2[:, -1, :] += 5.0                         # shape: (B, T, C)
same_before = torch.allclose(block(x)[:, :-1], block(x2)[:, :-1], atol=1e-5)
print("earlier tokens unaffected by a change in the last token (causal):", same_before)
```

- **In simple words:** Change only the last token and verify that earlier outputs do not move.
- **Exactly:** `x2 = x.clone(); x2[:, -1, :] += 5.0` modifies the last token. `block(x)[:, :-1]` vs `block(x2)[:, :-1]` are equal within tolerance: `True`.
- **Why it is here:** Proves the mask works; a leak would make training cheat.

### 17. Stack six blocks  (`main.py` lines 119–127)

```python
# ---------------------------------------------------------------------------
# 5. Stack N blocks: shape stays (B, T, C) all the way
# ---------------------------------------------------------------------------
print("\n== 2. stack of 6 blocks ==")
stack = nn.ModuleList([Block(C, H) for _ in range(6)])
h = x                                                       # shape: (B, T, C)
for i, blk in enumerate(stack):
    h = blk(h)                                              # shape: (B, T, C)
    print(f"after block {i + 1}: shape {tuple(h.shape)}")
```

- **In simple words:** Run the output of one block into the next, six times.
- **Exactly:** `nn.ModuleList` registers sub-modules so their parameters are tracked. Real output: `after block i: shape (2, 10, 64)` for i = 1..6.
- **Why it is here:** Depth is the main source of power; shape constancy makes it possible. distilgpt2 has 6 blocks, GPT-2 small 12.

### 18. Count parameters per block: 12 d squared plus a few more  (`main.py` lines 129–139)

```python
# ---------------------------------------------------------------------------
# 6. Parameter count: 12 * d_model^2 weights (+ 13 * d_model biases / LayerNorm params)
#    attention: qkv 3C^2 + proj C^2 = 4C^2 ; MLP: C*4C + 4C*C = 8C^2 ; total 12C^2
# ---------------------------------------------------------------------------
print("\n== 3. parameter count of one block ==")
n_block = count_params(block)
formula = 12 * C * C
extra = 13 * C   # biases: qkv 3C + proj C + fc 4C + proj2 C = 9C ; two LayerNorms 2 x 2C = 4C
print(f"d_model={C}: counted {n_block:,} | 12*d^2 = {formula:,} | 13*d (biases+LN) = {extra:,} | sum = {formula + extra:,}")
C_big = 768
print(f"d_model={C_big}: 12*d^2 = {12 * C_big**2:,}  (+ 13*d = {13 * C_big:,}  ->  {12 * C_big**2 + 13 * C_big:,})")
```

- **In simple words:** Predict the number of weights in one block with a formula and check it.
- **Exactly:** Attention `3C^2 + C^2 = 4C^2`, MLP `C*4C + 4C*C = 8C^2`, total `12C^2`. Biases `3C + C + 4C + C = 9C` plus two LayerNorms `2 * 2C = 4C` gives `13C`. For C=64: counted `49,984` = `49,152 + 832`. For C=768 the formula gives `7,087,872`.
- **Why it is here:** A fast way to estimate model size in interviews: params is about `12 * layers * d_model^2` plus embeddings.

### 19. Experiment A: remove the residual  (`main.py` lines 141–154)

```python
# ---------------------------------------------------------------------------
# 7. Experiment A: remove the residual -> the gradient reaching the input dies with depth
# ---------------------------------------------------------------------------
def grad_at_input(depth, use_residual, use_ln=True):
    torch.manual_seed(1)
    blocks = nn.Sequential(*[Block(C, H, use_residual, use_ln) for _ in range(depth)])
    inp = torch.randn(B, T, C, requires_grad=True)          # shape: (B, T, C)
    out = blocks(inp)                                       # shape: (B, T, C)
    out.pow(2).mean().backward()                            # a dummy loss; gradient flows back through every block
    return inp.grad.norm().item()                           # scalar: size of the gradient at the very first input

print("\n== 4. experiment A: gradient norm at the input (depth 24) ==")
print(f"with residual   : {grad_at_input(24, True):.3e}")
print(f"without residual: {grad_at_input(24, False):.3e}   <- signal from the loss barely reaches the early layers")
```

- **In simple words:** Stack 24 blocks with and without the shortcut and measure how much learning signal reaches the first layer.
- **Exactly:** `grad_at_input` builds a depth-24 stack, runs backward on a dummy loss `out.pow(2).mean()` and returns the gradient norm at the input `(B, T, C)`. Real output: with residual `3.342e-01`, without `1.426e-02`, about 23 times smaller.
- **Why it is here:** Without residuals gradients shrink with depth (vanishing gradients), so early layers barely learn. This is why every deep network uses skip connections.

### 20. Experiment B: remove LayerNorm  (`main.py` lines 156–173)

```python
# ---------------------------------------------------------------------------
# 8. Experiment B: remove LayerNorm -> activation scale drifts with depth
# ---------------------------------------------------------------------------
def std_per_depth(use_ln, depth=24):
    torch.manual_seed(2)
    h = torch.randn(B, T, C)                                # shape: (B, T, C)
    stds = []
    with torch.no_grad():
        for _ in range(depth):
            h = Block(C, H, True, use_ln)(h)                # shape: (B, T, C)  new random block each time
            stds.append(h.std().item())
    return stds

print("\n== 5. experiment B: std of activations after depth 1, 6, 12, 24 ==")
for use_ln in (True, False):
    s = std_per_depth(use_ln)
    label = "with LayerNorm   " if use_ln else "without LayerNorm"
    print(label, [f"{s[d - 1]:.2f}" for d in (1, 6, 12, 24)])
```

- **In simple words:** Pass data through 24 random blocks and track the spread of the activations.
- **Exactly:** `std_per_depth` uses a new random block at each depth under `no_grad`. Real output (depths 1, 6, 12, 24): with LayerNorm `1.03, 1.24, 1.50, 1.85`; without `1.03, 1.34, 2.07, 4.24`. The residual stream still grows with depth but much faster without LayerNorm.
- **Why it is here:** Unchecked scale growth makes training unstable (overflow, huge logits). LayerNorm keeps each sublayer's input at a fixed scale.

### 21. Break it: d_model not divisible by n_heads  (`main.py` lines 175–186)

```python
# ---------------------------------------------------------------------------
# 9. Break it: d_model not divisible by n_heads
# ---------------------------------------------------------------------------
print("\n== 6. break it ==")
try:
    CausalSelfAttention(d_model=64, n_heads=5)
except AssertionError as e:
    print("AssertionError:", e)
try:
    nn.MultiheadAttention(embed_dim=64, num_heads=5)
except AssertionError as e:
    print("torch's own layer says:", e)
```

- **In simple words:** Ask for 5 heads on a width of 64 and see the error.
- **Exactly:** `64 % 5 != 0` so our assert raises `AssertionError: d_model=64 must be divisible by n_heads=5`; PyTorch's own `nn.MultiheadAttention` raises `embed_dim must be divisible by num_heads`.
- **Why it is here:** A common config mistake; the head dimension must be an integer.

### 22. Compare with a real distilgpt2 block  (`main.py` lines 188–198)

```python
# ---------------------------------------------------------------------------
# 10. Compare with a REAL block: distilgpt2 transformer.h[0]
# ---------------------------------------------------------------------------
print("\n== 7. real distilgpt2 block vs ours ==")
from transformers import AutoModelForCausalLM
real = AutoModelForCausalLM.from_pretrained("distilgpt2").eval().transformer.h[0]
for name, p in real.named_parameters():
    print(f"  {name:22} {tuple(p.shape)}")                 # GPT-2 stores Linear weights as (in, out)
n_real = count_params(real)
print(f"real block params: {n_real:,} | ours at d_model=768: {12 * 768**2 + 13 * 768:,} | equal: {n_real == 12 * 768**2 + 13 * 768}")
print("real block param names: ln_1, attn.c_attn (=our qkv), attn.c_proj, ln_2, mlp.c_fc, mlp.c_proj -> same 6 pieces")
```

- **In simple words:** Load the real model's first block and check it has the same pieces and the same number of parameters.
- **Exactly:** `transformer.h[0]` has 12 parameter tensors: `ln_1`, `attn.c_attn.weight (768, 2304)`, `attn.c_proj (768, 768)`, `ln_2`, `mlp.c_fc (768, 3072)`, `mlp.c_proj (3072, 768)` and biases. GPT-2 stores Linear weights as `(in, out)`, transposed relative to `nn.Linear`. Real count `7,087,872`, equal to `12*768^2 + 13*768`: `True`.
- **Why it is here:** Shows that what you built is not a toy; the architecture is exactly GPT-2's block.

### 23. Experiments and break-it comments  (`main.py` lines 200–203)

```python
# TRY IT: change n_heads to 8 or 1 - the shape and the parameter count do not change (heads only split C).
# TRY IT: depth 48 in experiment A - the no-residual gradient shrinks further.
# BREAK IT: remove `.transpose(1, 2)` before `.view(B, T, C)` in CausalSelfAttention -> shapes still fit, output is silently scrambled.
# BREAK IT: set n_heads=5 -> AssertionError with a readable message (see section 6).
```

- **In simple words:** Try changing head count or depth, and breaking the transpose.
- **Exactly:** Heads only split C, so n_heads 1, 4 or 8 give the same shapes and parameter count. Removing `.transpose(1, 2)` before the final `view` still runs but scrambles values.
- **Why it is here:** Silent shape-compatible bugs are the hardest to find, so test with the causal check.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `class Block(nn.Module):` | a reusable layer; `__init__` builds parts, `forward` runs them | `block(x)` calls `forward` |
| `super().__init__()` | run the parent class set-up first | required in every `nn.Module` |
| `x.view(B, T, H, D)` | reshape without copying | `(2,10,64)` -> `(2,10,4,16)` |
| `x.transpose(1, 2)` | swap two axes | `(B,T,H,D)` -> `(B,H,T,D)` |
| `q, k, v = t.split(C, dim=-1)` | cut a tensor into equal pieces and unpack | `(B,T,3C)` -> three `(B,T,C)` |
| `x = x + a if flag else a` | conditional expression | residual on/off |
| `assert cond, "message"` | crash with a message if `cond` is False | used for `d_model % n_heads` |
| `try: ... except AssertionError as e:` | catch an error and print it | shows the real message |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `nn.Linear(i, o)` | torch | `x @ W + b` | in, out features | layer |
| `nn.Parameter(t)` | torch | marks a tensor as learnable | tensor | parameter |
| `nn.ModuleList` / `nn.Sequential` | torch | hold / chain layers | list of layers | container |
| `torch.triu(m, diagonal=1)` | torch | keep the part above the diagonal (the future) | `diagonal` | tensor |
| `t.masked_fill(mask, -inf)` | torch | overwrite entries where mask is True | mask, value | tensor |
| `nn.functional.gelu` | torch | smooth activation used in GPT-2 | - | tensor |
| `loss.backward()` | torch | fill `.grad` of every tensor that needs it | - | None |
| `nn.LayerNorm(C)` | torch | reference LayerNorm to check ours | features | layer |

## Execution trace
Real output from a run (loading messages trimmed):
```
== 1. one block ==
input shape : (2, 10, 64)
output shape: (2, 10, 64)  <- identical: that is what lets us stack blocks
MyLayerNorm == nn.LayerNorm: True
earlier tokens unaffected by a change in the last token (causal): True

== 2. stack of 6 blocks ==
after block 1: shape (2, 10, 64)
...
after block 6: shape (2, 10, 64)

== 3. parameter count of one block ==
d_model=64: counted 49,984 | 12*d^2 = 49,152 | 13*d (biases+LN) = 832 | sum = 49,984
d_model=768: 12*d^2 = 7,077,888  (+ 13*d = 9,984  ->  7,087,872)

== 4. experiment A: gradient norm at the input (depth 24) ==
with residual   : 3.342e-01
without residual: 1.426e-02   <- signal from the loss barely reaches the early layers

== 5. experiment B: std of activations after depth 1, 6, 12, 24 ==
with LayerNorm    ['1.03', '1.24', '1.50', '1.85']
without LayerNorm ['1.03', '1.34', '2.07', '4.24']

== 6. break it ==
AssertionError: d_model=64 must be divisible by n_heads=5
torch's own layer says: embed_dim must be divisible by num_heads

== 7. real distilgpt2 block vs ours ==
  ln_1.weight            (768,)
  attn.c_attn.weight     (768, 2304)
  attn.c_proj.weight     (768, 768)
  ln_2.weight            (768,)
  mlp.c_fc.weight        (768, 3072)
  mlp.c_proj.weight      (3072, 768)
  (biases omitted here)
real block params: 7,087,872 | ours at d_model=768: 7,087,872 | equal: True
```
- **Where `12 d^2` comes from:** attention has Q, K, V (3 d^2) and the output projection (d^2) = 4 d^2. The MLP has d x 4d and 4d x d = 8 d^2. Total 12 d^2. The extra `13 d` is biases and the two LayerNorms. Two thirds of every block is the MLP.
- **A real block matches ours exactly:** 7,087,872 parameters. `c_attn` is our packed `qkv`, `c_fc` is our MLP expand layer. GPT-2 stores linear weights as `(in, out)`.
- **Total check:** 6 blocks x 7,087,872 = 42.5 M of distilgpt2's 81.9 M parameters. The rest is embeddings (step 06 shows this).
- **Causal check:** changing the last token leaves all earlier outputs untouched, so the mask works.

## Experiments
- Experiment A (printed above): the gradient reaching the input at depth 24 is about 23x smaller without residuals (0.0143 vs 0.334). Try depth 48.
- Experiment B (printed above): with LayerNorm the std grows slowly (1.03 to 1.85); without it, it more than doubles that growth (4.24 at depth 24). The effect is modest here because the weights are small at this depth; deeper or larger models drift much more. Do not over-claim: this is a small demonstration.
- Change `n_heads` to 1, 2, 8: shapes and parameter count stay the same. Heads only split `d_model`.

## Debugging practice
Use `n_heads=5` with `d_model=64`. The code stops with `AssertionError: d_model=64 must be divisible by n_heads=5` (torch's own layer says `embed_dim must be divisible by num_heads`). Reason: `view(B, T, H, D)` needs `C = H x D` exactly. A quieter bug (not tested here): drop the `transpose(1, 2)` before the final `view(B, T, C)`. Shapes still fit, no error, but the heads are scrambled. Always check the meaning of the axes, not only the shape.

## Rebuild checklist
- [ ] Write LayerNorm: subtract mean, divide by std, scale and shift.
- [ ] Write multi-head causal attention with shape comments.
- [ ] Write the pre-norm block with two residual additions.
- [ ] Stack N blocks and confirm the shape is unchanged.
- [ ] Derive `12 d^2` from memory.

## The new problem (cliffhanger)
We now have a recipe, but our block holds random numbers: it knows nothing. Training a real stack from scratch costs weeks and a fortune. Someone has already paid that cost for distilgpt2. Can we load their trained blocks, confirm they have exactly this structure, count what is inside, and work out how much memory it needs? Next: [06_load_pretrained_model](../06_load_pretrained_model/WALKTHROUGH.md).

> **Say it aloud (interview box):** A transformer block is pre-norm multi-head causal attention followed by a 4x-expanding GELU feed-forward layer, each wrapped in a residual connection, so the shape stays (batch, seq_len, d_model) and blocks can be stacked. Residuals keep gradients alive, LayerNorm keeps activation scale stable, and the MLP does per-token thinking. One block has about 12 times d_model squared parameters: 4 d^2 in attention and 8 d^2 in the MLP.
