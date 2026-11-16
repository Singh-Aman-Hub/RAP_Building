# Step 03 — Norms and FFN: RMSNorm, pre-norm, SwiGLU, QK-norm

## The story so far
[Step 02](../02_positional_encodings/WALKTHROUGH.md) gave attention a sense of order. A decoder block has two sub-layers, attention and an MLP (the block of [04 / step 05](../../04_llm_fundamentals/05_transformer_block/WALKTHROUGH.md)). Around each sits a normalisation layer and a residual connection. GPT-2 and Llama differ in all three choices.

## The pain point
Deep networks are unstable: activations grow or shrink layer by layer, gradients vanish or explode, and attention logits can become so large that softmax turns one-hot. At the same time the MLP holds about two thirds of the parameters, so a small gain in its design is a large gain overall.

## The spark
Four small changes became universal: (1) normalise **before** each sub-layer (pre-norm) so the residual stream stays an unbroken sum; (2) use **RMSNorm**, which only rescales and is cheaper than LayerNorm; (3) replace the GELU MLP with a gated **SwiGLU** MLP, with hidden width cut to 8/3·d to keep parameters equal; (4) in many 2025 models normalise q and k (**QK-norm**) to bound attention logits.

## Goal
Implement RMSNorm and verify it against `torch.nn.RMSNorm`; implement GELU-MLP and SwiGLU and count parameters; compare post-norm and pre-norm stacks; show what QK-norm does to attention logits.

## The theory (read this before the interview)
**LayerNorm** `y = (x - mean) / sqrt(var + eps) * gamma + beta` subtracts the mean and divides by the standard deviation over the feature axis, with learned scale and shift (2d parameters). **RMSNorm** (Zhang & Sennrich 2019) drops the mean and the shift: `y = x / sqrt(mean(x^2) + eps) * gamma` (d parameters). The claim is that re-centering is not what makes LayerNorm work; re-scaling is. It saves compute and memory traffic and is used by Llama, Qwen, Gemma, DeepSeek, gpt-oss. **BatchNorm** is different in kind: it normalises over the *batch* axis, depends on batch statistics and is not used in LLMs (variable-length sequences, small per-device batches).

**Post-norm vs pre-norm.**
```
post-norm (original Transformer, BERT):   x = norm(x + sublayer(x))
pre-norm  (GPT-2 onwards, all modern):    x = x + sublayer(norm(x))
```
In pre-norm the residual path `x` is never normalised: it is an identity path from the input to the output, so gradients reach early layers directly, and training works without careful warm-up at large depth. The cost: the stream's magnitude grows with depth, so a **final norm** before the output head is required (`model.norm` in Llama). Post-norm can reach slightly better quality when it trains, but is fragile in deep models. Some 2025 models (Gemma 3, OLMo 2) add a norm *after* the sub-layer as well ("sandwich" or post-sublayer norm) to tame magnitude growth. Variants to name: **DeepNorm** (rescaled post-norm for 1,000 layers), **QK-norm** (below).

**FFN / MLP.** Original: `down(act(up(x)))` with `up: d -> 4d`, `down: 4d -> d`, GELU. **GLU variants** (Shazeer 2020) multiply two projections: `down(act(gate(x)) * up(x))`. With `act = SiLU` (x·sigmoid(x), also called swish) it is **SwiGLU**; with GELU it is GeGLU (Gemma). The gate lets the layer decide per hidden unit how much to pass. It has three weight matrices instead of two, so the hidden size is reduced to keep the budget: `2 · d · 4d = 8d^2` equals `3 · d · h` when `h = 8d/3`. Real models round or deviate: Llama-2-7B hidden 11,008 (ratio 2.69), Llama-3-8B 14,336 (3.5, a deliberately wider MLP), Qwen2.5-0.5B 4,864 (5.4). Modern models also drop all bias terms in the MLP and norms (Qwen2.5 keeps biases only on q, k, v).

**QK-norm.** Apply RMSNorm to each head's q and k (before RoPE) so `q·k` is bounded by the norm weights. It prevents attention-logit growth, a known cause of loss spikes at large scale. Used in Qwen3, Gemma 3, OLMo 2 (and Llama 4 per its config `use_qk_norm`).

**Where does the compute go?** Per layer at width d: attention ≈ `4d^2` params (with GQA less), MLP ≈ `8d^2` (or `3dh`). MLP ≈ 2/3 of non-embedding parameters (Qwen2.5-0.5B: 63.5%, step 08). This is why MoE (step 06) replaces the MLP, not the attention.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| FFN | MLP, feed-forward block, position-wise feed-forward, `mlp`, `intermediate` | Yes | the two- or three-matrix network applied to each token independently |
| `intermediate_size` | hidden size of the MLP, `ffn_hidden`, `d_ff`, `n_inner` | Yes | the width of the middle layer (the 4d or 8d/3) |
| SwiGLU | gated MLP, GLU variant, `gate_proj/up_proj/down_proj` | SwiGLU = GLU with SiLU | GeGLU uses GELU; plain MLP has no gate |
| SiLU | swish, `F.silu` | Yes | `x · sigmoid(x)` |
| LayerNorm vs RMSNorm vs BatchNorm | `nn.LayerNorm`, `RMSNorm`, `nn.BatchNorm1d` | No | LayerNorm centres+scales over features; RMSNorm scales only; BatchNorm uses batch statistics |
| Pre-norm vs post-norm | pre-LN, post-LN | No | norm inside the residual branch (pre) or after the addition (post) |
| Residual stream | skip connection, shortcut, residual connection | Mostly | "stream" is the way of thinking about `x` that every layer reads and adds to |
| `d_model` | `hidden_size`, `n_embd`, `dim` | Yes | width of the residual stream |
| QK-norm | `q_norm`/`k_norm`, query-key normalisation | Yes | RMSNorm on q and k per head |

## Run it
```bash
cd 16_modern_llm_architecture/03_norms_and_ffn
python main.py       # ~3 s, torch only
```

## Code, step by step
### 1. Imports, seed, width of GPT-2 small  (`main.py` lines 12–17)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
D = 768
```
- **In simple words:** set up torch and use the same width as GPT-2 small (768) so numbers are comparable with real models.
- **Exactly:** `nn` for layers, `F` for stateless functions (`gelu`, `silu`). `D = 768` is `d_model`.
- **Why it is here:** parameter counts later match real GPT-2 blocks.
### 2. RMSNorm module  (`main.py` lines 21–29)
```python
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))                      # learned gain only: no bias, no mean
        self.eps = eps

    def forward(self, x):                                              # x: (batch, seq, d)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()   # 1 / sqrt(mean(x^2) + eps)  (batch, seq, 1)
        return x * rms * self.weight
```
- **In simple words:** measure how big the numbers in a vector are (root mean square) and divide by it, so every token vector has length about 1, then let a learned weight rescale each feature.
- **Exactly:** `x.pow(2).mean(dim=-1, keepdim=True)` is the mean of squares per token, shape `(batch, seq, 1)`; `.add(eps).rsqrt()` is `1/sqrt(. + eps)`; the result multiplies `x` of shape `(batch, seq, d)` (broadcast) and then `weight` of shape `(d,)`. Only `d` parameters, no bias, no mean subtraction.
- **Why it is here:** this exact class (with eps from the config) is what Llama, Qwen and gpt-oss use. `keepdim=True` is essential for broadcasting.
### 3. Compare LayerNorm and RMSNorm outputs  (`main.py` lines 32–40)
```python
x = torch.randn(2, 5, D) * 3 + 1.5                                     # deliberately NOT centred, not unit scale
ln, rms = nn.LayerNorm(D, eps=1e-6), RMSNorm(D)
y_ln, y_rms = ln(x), rms(x)
print("input mean/std      :", round(x.mean().item(), 3), round(x.std().item(), 3))
print("LayerNorm mean/std  :", round(y_ln.mean().item(), 4), round(y_ln.std().item(), 4), "(centres AND rescales)")
print("RMSNorm  mean/std   :", round(y_rms.mean().item(), 4), round(y_rms.std().item(), 4), "(rescales only; mean stays)")
print("RMS of RMSNorm out  :", round(y_rms.pow(2).mean(-1).sqrt().mean().item(), 4))
print("RMSNorm matches torch.nn.RMSNorm:", torch.allclose(y_rms, nn.RMSNorm(D, eps=1e-6)(x), atol=1e-6))
print("params: LayerNorm", sum(p.numel() for p in ln.parameters()), "| RMSNorm", sum(p.numel() for p in rms.parameters()))
```
- **In simple words:** feed a badly scaled input (mean 1.5, spread 3) into both norms and see what each fixes.
- **Exactly:** input mean 1.481, std 3.013. LayerNorm output: mean -0.0, std 1.0001 (centred and rescaled). RMSNorm output: mean 0.4408, std 0.8977 (the mean is kept) but root-mean-square exactly 1.0. Our class equals `torch.nn.RMSNorm(D, eps=1e-6)` (`allclose` True). Parameters: LayerNorm 1,536 (gamma + beta), RMSNorm 768.
- **Why it is here:** shows that RMSNorm fixes the scale but not the mean, and that it matches the library implementation, so you can trust it.
### 4. GPT-2 style MLP  (`main.py` lines 44–50)
```python
class GeluMLP(nn.Module):                                              # GPT-2 style: d -> 4d -> d
    def __init__(self, d):
        super().__init__()
        self.up, self.down = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))
```
- **In simple words:** expand to 4 times the width, apply a smooth non-linearity, shrink back.
- **Exactly:** `up: Linear(768, 3072)`, `down: Linear(3072, 768)`, both with bias. Shape `(batch, seq, 768) -> (batch, seq, 3072) -> (batch, seq, 768)`.
- **Why it is here:** baseline to compare SwiGLU against.
### 5. SwiGLU MLP  (`main.py` lines 53–61)
```python
class SwiGLU(nn.Module):                                               # Llama style: down( silu(gate(x)) * up(x) )
    def __init__(self, d, hidden):
        super().__init__()
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):                                              # (batch, seq, d) -> (batch, seq, d)
        return self.down(F.silu(self.gate(x)) * self.up(x))            # the gate decides, elementwise, what passes
```
- **In simple words:** two parallel projections: one produces the "content", the other a "gate" that scales each content feature, then project back.
- **Exactly:** `down(silu(gate(x)) * up(x))`; `*` is elementwise. Three bias-free linears: `gate`, `up`: `d -> h`; `down`: `h -> d`.
- **Why it is here:** the MLP of every current open model; the multiplicative gate gives better quality per parameter than a plain GELU MLP at equal size (Shazeer 2020).
### 6. Parameter matching and real ratios  (`main.py` lines 64–73)
```python
count = lambda m: sum(p.numel() for p in m.parameters())
gelu = GeluMLP(D)
hidden = int(8 * D / 3)                                                # 2048: three matrices of d*h ~ two of d*4d
swi = SwiGLU(D, hidden)
print(f"\nGELU MLP : hidden {4 * D}, 2 matrices + biases = {count(gelu):,} params")
print(f"SwiGLU   : hidden {hidden}, 3 matrices, no biases = {count(swi):,} params  (ratio {count(swi) / count(gelu):.4f})")
naive = SwiGLU(D, 4 * D)
print(f"SwiGLU with hidden 4d would cost {count(naive):,} params ({count(naive) / count(gelu):.2f}x) -> that is why hidden is cut to 8/3 d")
for name, d, h in [("Llama-2-7B", 4096, 11008), ("Llama-3-8B", 4096, 14336), ("Qwen2.5-0.5B", 896, 4864)]:
    print(f"  {name}: d={d}, intermediate={h}, ratio {h / d:.3f}  (8/3 = 2.667)")
```
- **In simple words:** to keep the model the same size, a three-matrix MLP needs a narrower hidden layer: 8/3 of d instead of 4 times d.
- **Exactly:** GELU MLP with hidden 3,072: 4,722,432 params (incl. 3,840 bias). SwiGLU with hidden `int(8*768/3) = 2048`: 4,718,592 (ratio 0.9992). SwiGLU at hidden 4d would cost 7,077,888 (1.50x). Real ratios from configs: Llama-2-7B 2.688, Llama-3-8B 3.5, Qwen2.5-0.5B 5.429.
- **Why it is here:** this is the reasoning behind odd-looking `intermediate_size` values, a classic interview question. Newer models deliberately use wider ratios, trading depth for width.
### 7. A stand-in sub-layer  (`main.py` lines 76–82)
```python
class Sub(nn.Module):                                                  # stand-in for attention or MLP
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        return self.lin(x)
```
- **In simple words:** a plain linear layer standing in for "attention or MLP" so we can study the wiring only.
- **Exactly:** `Linear(d, d)` with default init.
- **Why it is here:** isolates the effect of where the norm sits from everything else.
### 8. Post-norm vs pre-norm stack, forward direction  (`main.py` lines 85–94)
```python
def run_stack(kind, depth=24, d=64):
    torch.manual_seed(1)
    layers = nn.ModuleList(Sub(d) for _ in range(depth))
    norms = nn.ModuleList(RMSNorm(d) for _ in range(depth))
    h = torch.randn(1, 4, d)
    start = h.clone()
    for sub, nm in zip(layers, norms):
        h = nm(h + sub(h)) if kind == "post" else h + sub(nm(h))     # post: norm(x + f(x)); pre: x + f(norm(x))
    # does the original input still travel, unchanged, along the residual stream?
    return h.pow(2).mean().sqrt().item(), F.cosine_similarity(h.flatten(), start.flatten(), dim=0).item()
```
- **In simple words:** build 24 layers both ways and look at the output.
- **Exactly:** `h = nm(h + sub(h))` is post-norm; `h = h + sub(nm(h))` is pre-norm. The function returns the RMS of the output and its cosine similarity with the original input. Same seed gives identical weights for both kinds.
- **Why it is here:** shows what the stream looks like: post-norm resets the scale every layer; pre-norm accumulates.
### 9. Print the forward comparison  (`main.py` lines 97–100)
```python
for kind in ("post", "pre"):
    rms_out, cos = run_stack(kind)
    print(f"{kind}-norm, 24 layers: output RMS {rms_out:.2f}, cosine(out, input) {cos:.3f}")
print("pre-norm: the residual stream is an unbroken sum, so gradients reach early layers directly (identity path).")
```
- **Exactly:** post-norm, 24 layers: output RMS 1.00, cosine with input 0.106. Pre-norm: output RMS 3.08 (grows with depth, hence the mandatory final norm), cosine 0.460: the original input is still strongly present in the stream because of the identity path.
- **In simple words:** in pre-norm the original information travels along a highway; in post-norm it is squeezed through a toll booth at every layer.
- **Why it is here:** the identity path is the reason deep pre-norm models train stably.
### 10. Gradient at the first and last layer  (`main.py` lines 103–111)
```python
def grad_first(kind, depth=48, d=64):
    torch.manual_seed(1)
    layers = nn.ModuleList(Sub(d) for _ in range(depth))
    norms = nn.ModuleList(RMSNorm(d) for _ in range(depth))
    h = torch.randn(1, 4, d)
    for sub, nm in zip(layers, norms):
        h = nm(h + sub(h)) if kind == "post" else h + sub(nm(h))
    h.pow(2).sum().backward()
    return layers[0].lin.weight.grad.norm().item(), layers[-1].lin.weight.grad.norm().item()
```
- **In simple words:** run a 48-layer stack, ask for gradients, and compare how large they are at layer 1 and layer 48.
- **Exactly:** loss `h.pow(2).sum()`, `.backward()`, then the norm of the weight gradient in the first and last `Linear`.
- **Why it is here:** the ratio first/last shows how well the learning signal reaches the bottom of the network.
### 11. Print gradient ratios  (`main.py` lines 114–116)
```python
for kind in ("post", "pre"):
    g0, gN = grad_first(kind)
    print(f"{kind}-norm, 48 layers: grad norm first layer {g0:.2e} | last layer {gN:.2e} | ratio first/last {g0 / gN:.3f}")
```
- **Exactly:** post-norm: first layer 7.92e-05, last 2.25e-04, ratio 0.352 (early layers get a third of the signal of the last). Pre-norm: 3.89e+03 vs 1.16e+03, ratio 3.365. The absolute values are not comparable (the two stacks have different output scales, hence different losses); **only the ratio is meaningful**. At random initialisation the effect is moderate: the real-world symptom is that post-norm needs a long warm-up and diverges at depth, while pre-norm trains.
- **In simple words:** pre-norm lets the early layers hear the teacher as loudly as the late layers.
- **Why it is here:** an honest, measured version of the usual textbook claim (the claim is about training stability, not about this single number).
### 12. Attention logits function  (`main.py` lines 120–121)
```python
def attn_logits(q, k):
    return (q @ k.transpose(-1, -2)) / q.shape[-1] ** 0.5               # (heads, seq, seq)
```
- **In simple words:** the score between every query and key, scaled by the square root of the head size.
- **Exactly:** `(q @ k^T) / sqrt(hd)`, shape `(heads, seq, seq)`.
- **Why it is here:** QK-norm acts on exactly this quantity.
### 13. QK-norm test  (`main.py` lines 124–127)
```python
q, k = torch.randn(4, 6, 64) * 8, torch.randn(4, 6, 64) * 8             # big activations, as in late training
qn, kn = RMSNorm(64), RMSNorm(64)
print(f"\nmax |logit| without QK-norm {attn_logits(q, k).abs().max():.1f} | with QK-norm {attn_logits(qn(q), kn(k)).abs().max():.1f}")
print("softmax of the large logits is almost one-hot (entropy):",
```
- **In simple words:** give attention huge q and k (as happens late in training), then normalise them and watch the scores shrink.
- **Exactly:** without QK-norm the biggest logit is 194.8; with RMSNorm applied to q and k it is 2.9.
- **Why it is here:** logits that large make softmax a hard one-hot (gradient nearly zero), a cause of training spikes.
### 14. Entropy of the attention distribution  (`main.py` lines 128–129)
```python
      round(-(F.softmax(attn_logits(q, k), -1) * F.log_softmax(attn_logits(q, k), -1)).sum(-1).mean().item(), 3),
      "vs with QK-norm:", round(-(F.softmax(attn_logits(qn(q), kn(k)), -1) * F.log_softmax(attn_logits(qn(q), kn(k)), -1)).sum(-1).mean().item(), 3))
```
- **Exactly:** entropy 0.006 nats without QK-norm (almost one-hot) vs 1.473 with it (spread over several tokens; uniform over 6 would be 1.79).
- **In simple words:** without it each token looks at exactly one other token and cannot learn to look elsewhere.
- **Why it is here:** quantifies "attention collapse".
### 15. Bias terms  (`main.py` lines 132–132)
```python
print("bias params in a GELU MLP:", 4 * D + D, "of", count(gelu), "| SwiGLU has none")
```
- **Exactly:** the GELU MLP has 4·768 + 768 = 3,840 bias parameters out of 4,722,432 (0.08%). Biases are dropped in modern models: they add memory traffic and no measurable quality.
- **In simple words:** tiny gain, so removed.
- **Why it is here:** explains the `bias=False` you see everywhere in Llama-style code.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `nn.Parameter(torch.ones(d))` | a tensor the optimiser trains | RMSNorm gain |
| `x.pow(2).mean(dim=-1, keepdim=True)` | mean of squares along the last axis, keep the axis | `(2,5,768) -> (2,5,1)` |
| `torch.rsqrt(t)` | `1/sqrt(t)` | the normaliser |
| `lambda m: ...` | one-line function | parameter counter |
| `h = a if cond else b` | conditional expression | choose post- or pre-norm |
| `nn.ModuleList(gen)` | list of modules registered with the parent | stack of layers |
| `loss.backward()` | compute gradients | `weight.grad` |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `nn.LayerNorm(d, eps)` | torch | centre + scale + affine | width, eps | module |
| `nn.RMSNorm(d, eps)` | torch (2.4+) | RMS scale + gain | width, eps | module |
| `F.gelu`, `F.silu` | torch | activations | tensor | tensor |
| `F.softmax / log_softmax` | torch | probabilities / log-probabilities | `dim` | tensor |
| `p.numel()` | torch | element count | | int |
| `torch.allclose(a, b, atol)` | torch | equality within tolerance | | bool |

## Execution trace (real run)
```text
input mean/std      : 1.481 3.013
LayerNorm mean/std  : -0.0 1.0001 (centres AND rescales)
RMSNorm  mean/std   : 0.4408 0.8977 (rescales only; mean stays)
RMS of RMSNorm out  : 1.0
RMSNorm matches torch.nn.RMSNorm: True
params: LayerNorm 1536 | RMSNorm 768

GELU MLP : hidden 3072, 2 matrices + biases = 4,722,432 params
SwiGLU   : hidden 2048, 3 matrices, no biases = 4,718,592 params  (ratio 0.9992)
SwiGLU with hidden 4d would cost 7,077,888 params (1.50x) -> that is why hidden is cut to 8/3 d
  Llama-2-7B: d=4096, intermediate=11008, ratio 2.688  (8/3 = 2.667)
  Llama-3-8B: d=4096, intermediate=14336, ratio 3.500  (8/3 = 2.667)
  Qwen2.5-0.5B: d=896, intermediate=4864, ratio 5.429  (8/3 = 2.667)
post-norm, 24 layers: output RMS 1.00, cosine(out, input) 0.106
pre-norm, 24 layers: output RMS 3.08, cosine(out, input) 0.460
pre-norm: the residual stream is an unbroken sum, so gradients reach early layers directly (identity path).
post-norm, 48 layers: grad norm first layer 7.92e-05 | last layer 2.25e-04 | ratio first/last 0.352
pre-norm, 48 layers: grad norm first layer 3.89e+03 | last layer 1.16e+03 | ratio first/last 3.365

max |logit| without QK-norm 194.8 | with QK-norm 2.9
softmax of the large logits is almost one-hot (entropy): 0.006 vs with QK-norm: 1.473
bias params in a GELU MLP: 3840 of 4722432 | SwiGLU has none
```
Reading it: LayerNorm has mean 0; RMSNorm keeps mean 0.44 but RMS 1. SwiGLU at 8/3·d is within 0.1% of the GELU MLP's size. Pre-norm's output RMS grows (3.08) while post-norm stays at 1.00.

## Experiments (change ONE thing)
Every row below was run on a temporary copy of `main.py` (the real file is untouched); only the named line changed.

| Change | What you will see | Why |
|---|---|---|
| `hidden = 4 * D` instead of `int(8 * D / 3)` (measured) | SwiGLU params 7,077,888 (ratio 1.4988 vs the GELU MLP; was 4,718,592, ratio 0.9992) | SwiGLU has three matrices instead of two, so keeping hidden at 4d costs 1.5x. Cutting hidden to 8/3 d is what makes the comparison fair. |
| `eps=1e-2` in `RMSNorm` (measured) | `RMS of RMSNorm out` 0.9996 (was 1.0) and `RMSNorm matches torch.nn.RMSNorm: False` | A large eps inside the square root shrinks every output slightly, and it no longer equals the reference built with `eps=1e-6`. eps is for safety against divide-by-zero; keep it tiny and identical to the checkpoint's value. |
| `x.var(...)` instead of `x.pow(2).mean(...)` in the RMS (measured) | `RMS of RMSNorm out` 1.1143 and `matches torch.nn.RMSNorm: False` | Variance subtracts the mean first; RMS does not. The input here is deliberately off-centre (mean 1.5), so the two statistics differ. That subtraction is exactly what RMSNorm drops. |
| shallower stacks: `depth=6` for the 24-layer run, `depth=12` for the 48-layer run (measured) | post-norm first/last gradient ratio 0.758 (was 0.352 at 48 layers); pre-norm output RMS 1.74 (was 3.08 at 24 layers); pre-norm ratio 2.149 (was 3.365) | Post-norm gradients shrink further from the last layer as depth grows (each norm sits on the gradient path); pre-norm keeps an identity path, and the residual stream grows with depth as every block adds to it. |
| activations `* 1` instead of `* 8` in the QK-norm demo (measured) | max logit 3.0 without QK-norm vs 2.9 with; entropy 1.472 vs 1.473 | QK-norm only matters when q and k get large. With 8x activations the max logit was 194.8 and the softmax almost one-hot (entropy 0.006). Large activations square into the logits. |

## Debugging practice
**BREAK IT drill 1: drop `keepdim=True`.** In `RMSNorm.forward`:

```diff
-        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
+        rms = x.pow(2).mean(dim=-1).add(self.eps).rsqrt()
```

Symptom (real run), at the first forward call:

```text
RuntimeError: The size of tensor a (768) must match the size of tensor b (5) at non-singleton dimension 2
```

Root cause: without `keepdim`, `rms` has shape `(batch, seq) = (2, 5)`. Multiplying `(2, 5, 768) * (2, 5)` aligns the trailing dimensions, 768 against 5. Fix: `keepdim=True` so `rms` is `(2, 5, 1)` and broadcasts across the feature axis. (If batch happened to equal d, it would silently broadcast wrongly instead of raising.)

**BREAK IT drill 2: use variance instead of mean-square (a silent bug).**

```diff
-        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
+        rms = x.var(dim=-1, keepdim=True).add(self.eps).rsqrt()
```

Symptom (real run): nothing crashes, but `RMS of RMSNorm out : 1.1143` (should be 1.0) and `RMSNorm matches torch.nn.RMSNorm: False`.

Root cause: `var` centres the data first, so it ignores the mean of 1.5 that RMS keeps. The scale factor is therefore wrong for off-centre inputs. Fix: `x.pow(2).mean(...)`. The built-in `torch.nn.RMSNorm` comparison line in the file exists precisely to catch this kind of bug.

## Common mistakes
- Describing RMSNorm as "LayerNorm without the mean" and then using `var` (which still subtracts the mean) in the code.
- Forgetting `keepdim=True` and hitting a broadcasting error, or worse, a silent mis-broadcast when two dimensions happen to match.
- Keeping SwiGLU hidden at 4d and then comparing quality with a GELU MLP: it is a 1.5x bigger layer, not the same budget.
- Mixing up post-norm `norm(x + f(x))` and pre-norm `x + f(norm(x))`: only pre-norm leaves an unnormalised identity path for gradients.
- Changing `eps` when loading a checkpoint. Use the value in its `config.json` (`rms_norm_eps`).
- Applying QK-norm (a norm per head on q and k) to v, or applying it after the dot product; it must sit before `q @ k^T`.

## Rebuild checklist
- [ ] RMSNorm with `keepdim=True`, `rsqrt`, learned weight, matches `nn.RMSNorm`
- [ ] SwiGLU: `down(silu(gate(x)) * up(x))`, hidden = 8/3·d, no bias
- [ ] pre-norm block: `x = x + f(norm(x))`, final norm before the head
- [ ] QK-norm: RMSNorm on the head dimension of q and k
- [ ] explain why hidden is 8/3·d and why pre-norm trains deeper

## The new problem (cliffhanger)
The MLP is now efficient, but the attention sub-layer still has a memory problem at inference: the keys and values of every past token must be kept (the KV cache), and with 32 heads each caching K and V it dominates GPU memory for long contexts. Next: [Step 04 — attention variants and the KV cache](../04_attention_variants_and_kv_cache/WALKTHROUGH.md).

> **Say it aloud (interview box):** "Modern blocks are pre-norm with RMSNorm: normalise inside the residual branch so the residual stream is an identity highway, and skip the mean and bias because rescaling is what matters. The MLP is SwiGLU, `down(silu(gate x) * up x)`, with hidden size about 8/3 of d so it has the same parameters as a 4d GELU MLP. Newer models add QK-norm to keep attention logits bounded."
