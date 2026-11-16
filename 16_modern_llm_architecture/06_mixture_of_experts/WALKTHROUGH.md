# Step 06 — Mixture of Experts (MoE)

## The story so far
[Step 05](../05_flash_attention_online_softmax/WALKTHROUGH.md) fixed the memory cost of attention. The remaining big cost is the MLP: with SwiGLU ([step 03](../03_norms_and_ffn/WALKTHROUGH.md)) it holds about two thirds of a dense model's parameters (63.5% in Qwen2.5-0.5B, step 08), and every token pays for all of them.

## The pain point
Knowledge scales with parameters; cost per token scales with the parameters *used*. A dense 400B model is slow and expensive to run on every token, yet most tokens (a comma, a Python keyword, a French word) do not need all of it.

## The spark
Replace the one MLP with `E` expert MLPs and a small **router** that picks the top-`k` experts for each token. Total parameters grow by `E`, compute per token grows only by `k` (plus a tiny router). DeepSeek-V3 has 671B parameters and uses 37B per token.

## Goal
Build an MoE layer (router, top-k, experts, shared expert), count total vs active parameters, watch an untrained router overload a few experts, then balance them two ways: the classical auxiliary loss and DeepSeek-V3's auxiliary-loss-free bias.

## The theory (read this before the interview)
**The layer.** For a token vector `x`: `logits = W_r x` (one score per expert), `p = softmax(logits)`, pick the `k` largest, renormalise their weights to sum to 1, and output `sum_i w_i · Expert_i(x)` (+ shared experts). Experts are ordinary SwiGLU MLPs. Attention is **not** made sparse: only the MLP is.

**Total vs active parameters.** *Total* = every weight (memory you must hold: all experts). *Active* = weights touched per token (compute and memory bandwidth per token). Reported as e.g. "Qwen3-235B-A22B" (A = active). Memory is set by total, FLOPs per token by active. Examples from public model cards: Mixtral 8x7B 46.7B total / 12.9B active (top-2 of 8); DeepSeek-V3 671B / 37B (top-8 of 256 routed + 1 shared); Qwen3-235B-A22B 235B / 22B (top-8 of 128); gpt-oss-120b 116.8B / 5.1B (top-4 of 128); Kimi K2 1T / 32B (top-8 of 384 + 1 shared).

**Fine-grained and shared experts.** DeepSeekMoE splits each expert into many small ones (more combinations: choosing 8 of 256 gives astronomically more specialisations than 2 of 8) and keeps **shared experts** that every token uses to hold common knowledge, so routed experts can specialise. Llama 4 Scout: 16 experts, top-1 plus a shared expert path (per its config the shared dense MLP is always on).

**Load balancing: the central problem.** Left alone the router collapses: a few experts get most tokens, they train faster, so the router prefers them more ("rich get richer"); others never learn. Consequences: wasted capacity, and on many GPUs (expert parallelism: experts live on different devices) the busiest expert sets the speed.
1. **Auxiliary loss** (Switch Transformer, GShard): `L_aux = E · sum_e f_e · P_e`, where `f_e` is the fraction of token slots sent to expert e (not differentiable) and `P_e` the mean router probability for e (differentiable). It equals 1 when both are uniform and is larger otherwise; add `alpha · L_aux` (alpha ~ 0.01) to the training loss. Downside: its gradient interferes with the language-modelling objective.
2. **Auxiliary-loss-free balancing** (DeepSeek-V3): keep a per-expert **bias** `b_e` added to the router scores *only for choosing the top-k*; the weights that scale expert outputs still use the unbiased probabilities. After each step, lower `b_e` for overloaded experts and raise it for underloaded ones by a fixed step `gamma`. No extra loss term and no gradient interference; the model reports better quality than aux-loss balancing at the same balance. (DeepSeek-V3 also keeps a tiny sequence-level aux loss.)
3. **Capacity factor** (GShard/Switch): each expert accepts at most `C = capacity_factor · tokens · k / E` tokens per batch; the rest are dropped (skip the layer via the residual) or re-routed. Training with expert parallelism often uses it; many recent inference systems are dropless.

**Routing variants.** Softmax-then-top-k (Mixtral) vs top-k-then-softmax; sigmoid scoring with normalisation (DeepSeek-V3); group-limited routing (choose a few device groups first to cut communication); expert-choice routing (experts pick tokens). Router in fp32 is common for stability.

**Serving MoE.** All experts must be in memory (e.g. DeepSeek-V3 671B ≈ 1.3 TB in bf16, hence FP8/FP4 weights, gpt-oss in MXFP4 fitting one 80 GB GPU). At batch size 1 you load a few experts per layer (fast); at large batch nearly all experts are touched, so decoding is memory-bound on total weights. Expert parallelism puts experts on different GPUs with all-to-all communication.

**Upcycling.** Initialise experts as copies of a dense model's MLP and continue training (Qwen-MoE, Mixtral-style recipes), cheaper than training from scratch.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| MoE | mixture of experts, sparse model, sparsely-gated MoE, "experts" | Yes | many MLPs plus a router, few active per token |
| Active parameters | activated params, "A22B" | Yes | weights used per token; compute is proportional to this |
| Total parameters | full size, "235B" | Yes | all weights incl. unused experts; memory is proportional to this |
| Dense model | non-MoE | No | every token uses every parameter |
| Router | gate, gating network, `gate`, `router` | Yes | linear layer scoring experts |
| Top-k gating | top-k routing | Yes | choose the k highest-scoring experts |
| Shared expert | always-on expert | Yes | used by all tokens, not routed |
| Aux loss vs aux-loss-free | load-balancing loss vs bias balancing | No | extra loss term with gradients vs non-gradient bias update |
| Capacity factor | expert capacity | Yes | max tokens per expert; overflow is dropped |
| Expert (in MoE) vs "expert" (human) | | No | an expert is an MLP, not a topic specialist; specialisation is partial and token-level |

## Run it
```bash
cd 16_modern_llm_architecture/06_mixture_of_experts
python main.py       # ~10 s on CPU, torch only
```

## Code, step by step
### 1. Imports and seed  (`main.py` lines 13–17)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
```
- **In simple words:** torch plus a fixed seed so the "random" router and tokens are the same every run.
- **Exactly:** `F` provides `silu`.
- **Why it is here:** utilisation numbers below are reproducible.
### 2. One expert is a SwiGLU MLP  (`main.py` lines 20–26)
```python
class Expert(nn.Module):                                               # an ordinary SwiGLU MLP (see step 03)
    def __init__(self, d, h):
        super().__init__()
        self.gate, self.up, self.down = nn.Linear(d, h, bias=False), nn.Linear(d, h, bias=False), nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
```
- **In simple words:** each expert is a small ordinary MLP, exactly the one from step 03.
- **Exactly:** `gate`, `up: d -> h`, `down: h -> d`, no biases; with d = 64, h = 128 it has 3·64·128 = 24,576 parameters.
- **Why it is here:** experts are not a new kind of layer.
### 3. MoE constructor  (`main.py` lines 29–36)
```python
class MoE(nn.Module):
    def __init__(self, d=64, h=128, n_experts=8, top_k=2, n_shared=1):
        super().__init__()
        self.k, self.n = top_k, n_experts
        self.router = nn.Linear(d, n_experts, bias=False)              # one score per expert, per token
        self.experts = nn.ModuleList(Expert(d, h) for _ in range(n_experts))
        self.shared = nn.ModuleList(Expert(d, h) for _ in range(n_shared))   # always-on experts (DeepSeek, Qwen-MoE)
        self.register_buffer("bias", torch.zeros(n_experts))           # aux-loss-free balancing: used for SELECTION only
```
- **In simple words:** a router that scores experts, a list of experts, a list of always-on shared experts, and a balancing bias that is not trained by gradient.
- **Exactly:** `router: Linear(d, n_experts, bias=False)` (512 parameters for d=64, 8 experts); `nn.ModuleList` registers all experts; `register_buffer("bias", zeros(n_experts))` stores a tensor that moves with the module and is saved in the state dict but is not a Parameter, so no optimiser touches it.
- **Why it is here:** the bias must not receive gradients: that is the "auxiliary-loss-free" idea.
### 4. Routing: scores, top-k, weights  (`main.py` lines 38–43)
```python
    def forward(self, x):                                              # x: (tokens, d)
        logits = self.router(x)                                        # (tokens, n_experts)
        probs = logits.softmax(-1)                                     # router probabilities
        top_idx = (probs + self.bias).topk(self.k, dim=-1).indices     # (tokens, k) choose with the bias...
        top_w = probs.gather(-1, top_idx)                              # ...but weight with the RAW probabilities
        top_w = top_w / top_w.sum(-1, keepdim=True)                    # renormalise the k weights to sum to 1
```
- **In simple words:** score every expert for every token, pick the best two, and decide how much to trust each.
- **Exactly:** `logits (tokens, 8)` -> `probs` (softmax). `top_idx = (probs + self.bias).topk(k).indices` has shape `(tokens, 2)`: **selection uses the biased score**. `top_w = probs.gather(-1, top_idx)` takes the **unbiased** probabilities of the chosen experts, then divides by their sum so the two weights add to 1.
- **Why it is here:** separating "who is picked" (bias allowed) from "how much weight" (clean probabilities) is the DeepSeek-V3 trick.
### 5. Dispatch tokens to experts  (`main.py` lines 44–48)
```python
        out = torch.zeros_like(x)
        for e in range(self.n):                                        # loop over experts (real systems use grouped GEMMs)
            tok, slot = (top_idx == e).nonzero(as_tuple=True)          # which tokens picked expert e, in which slot
            if len(tok):
                out[tok] += top_w[tok, slot, None] * self.experts[e](x[tok])
```
- **In simple words:** for each expert, collect the tokens that chose it, run only those through it, and add the results back, scaled by the router weight.
- **Exactly:** `(top_idx == e).nonzero(as_tuple=True)` returns two index vectors: `tok` (which tokens) and `slot` (position 0 or 1 in their top-k). `x[tok]` is `(n_e, d)`; `top_w[tok, slot, None]` is `(n_e, 1)`. `out[tok] +=` accumulates; a token appears in two experts' loops. (Production kernels sort tokens by expert and use grouped matrix multiplies instead of a Python loop.)
- **Why it is here:** this loop is the sparsity: an expert never sees tokens that did not choose it.
### 6. Shared experts and bookkeeping  (`main.py` lines 49–52)
```python
        for s in self.shared:
            out = out + s(x)                                           # every token goes through shared experts
        self.last = dict(probs=probs, top_idx=top_idx)
        return out
```
- **Exactly:** every token passes through each shared expert and its output is added without a weight. `self.last` keeps the probabilities and chosen indices for diagnostics.
- **In simple words:** a common-knowledge expert that everyone consults.
- **Why it is here:** lets routed experts specialise (DeepSeekMoE).
### 7. Tokens per expert  (`main.py` lines 54–55)
```python
    def counts(self):                                                  # how many tokens each expert received
        return torch.bincount(self.last["top_idx"].flatten(), minlength=self.n)
```
- **Exactly:** `torch.bincount(top_idx.flatten(), minlength=n)` counts how many (token, slot) pairs each expert received; total = tokens x k.
- **In simple words:** the load of each expert.
- **Why it is here:** the quantity we want to balance.
### 8. Switch-style auxiliary loss  (`main.py` lines 58–63)
```python
def balance_loss(probs, top_idx, n):
    """Switch/GShard aux loss: n * sum_e f_e * P_e. f_e = fraction of routed slots to e, P_e = mean router prob of e.
    Minimum (=1) when both are uniform."""
    f = torch.bincount(top_idx.flatten(), minlength=n).float() / top_idx.numel()
    P = probs.mean(0)
    return n * (f * P).sum()
```
- **In simple words:** a penalty that is smallest when every expert gets the same share of tokens and the router gives them the same probability.
- **Exactly:** `f` (fraction of routed slots per expert, counted, no gradient) and `P = probs.mean(0)` (mean router probability per expert, differentiable). `n · sum(f·P)` equals 1.0 when both are uniform and grows with imbalance. The gradient flows only through `P`.
- **Why it is here:** the standard way to train a balanced router before DeepSeek's alternative.
### 9. Total vs active parameters  (`main.py` lines 67–74)
```python
moe = MoE()
count = lambda m: sum(p.numel() for p in m.parameters())
per_expert = count(moe.experts[0])
total = count(moe)
active = count(moe.router) + moe.k * per_expert + count(moe.shared)
dense_equiv = per_expert
print(f"one expert {per_expert:,} | router {count(moe.router):,} | total layer {total:,}")
print(f"active per token (router + {moe.k} routed + 1 shared) = {active:,}  ->  {active / total:.1%} of the total")
```
- **Exactly:** one expert 24,576 params, router 512, whole layer 221,696 (8 routed + 1 shared experts + router). Active per token = router + 2 routed + 1 shared = 74,240, i.e. 33.5% of the layer.
- **In simple words:** the layer is 3x bigger than the compute it spends per token.
- **Why it is here:** the "total vs active" distinction in numbers.
### 10. A clustered token batch  (`main.py` lines 77–80)
```python
torch.manual_seed(1)
centers = torch.randn(3, 64) * 2                                       # 3 "topics"
x = centers[torch.randint(0, 3, (2048,))] + 0.5 * torch.randn(2048, 64)    # (tokens, d)
y = moe(x)
```
- **In simple words:** real text is not uniform; tokens cluster. We make 2,048 token vectors around three centres and send them through an untrained router.
- **Exactly:** `centers (3, 64)`, each token = random centre + noise, so `x (2048, 64)`; `y` is `(2048, 64)`.
- **Why it is here:** a uniform random batch would hide the imbalance that clustered data causes.
### 11. Utilisation before training the router  (`main.py` lines 81–83)
```python
print("\noutput", tuple(y.shape), "| tokens x k =", 2048 * 2, "routed slots")
print("utilisation, untrained router      :", moe.counts().tolist())
print(f"aux balance loss (1.0 is perfect)  : {balance_loss(moe.last['probs'], moe.last['top_idx'], 8).item():.3f}")
```
- **Exactly:** 2,048 tokens x top-2 = 4,096 routed slots. Utilisation per expert: `[73, 161, 1047, 650, 1263, 4, 59, 839]` (perfect balance would be 512 each). Expert 5 gets 4 slots, expert 4 gets 1,263. Balance loss 1.505 (1.0 is perfect).
- **In simple words:** two experts do half the work and one is nearly unused.
- **Why it is here:** shows why balancing is needed even with a fresh router.
### 12. Capacity and dropped tokens  (`main.py` lines 86–88)
```python
cap = int(1.25 * 2048 * moe.k / moe.n)                                 # capacity factor 1.25
dropped = (moe.counts() - cap).clamp(min=0).sum().item()
print(f"capacity per expert (factor 1.25) = {cap}; slots that would be DROPPED = {dropped} ({dropped / (2048 * 2):.1%})")
```
- **Exactly:** capacity = 1.25 · 2048 · 2 / 8 = 640 slots per expert. Overflow (slots beyond 640) totals 1,239, i.e. 30.2% of all slots would be dropped.
- **In simple words:** a size limit per expert makes imbalance cost real tokens.
- **Why it is here:** the practical penalty of imbalance in capacity-limited systems.
### 13. Copy the router for method A  (`main.py` lines 91–94)
```python
torch.manual_seed(2)
A = MoE()
A.load_state_dict(moe.state_dict())
opt = torch.optim.Adam(A.router.parameters(), lr=0.01)                 # only the router, to isolate the effect
```
- **Exactly:** a second `MoE`, `load_state_dict(moe.state_dict())` gives identical weights; only the router is optimised so the comparison is clean.
- **In simple words:** two identical starting points, two different balancing methods.
- **Why it is here:** fair experiment design.
### 14. Method A: train the router on the aux loss  (`main.py` lines 95–100)
```python
for step in range(200):
    A(x)
    loss = balance_loss(A.last["probs"], A.last["top_idx"], 8)
    opt.zero_grad()
    loss.backward()
    opt.step()
```
- **Exactly:** 200 Adam steps (lr 0.01) on `balance_loss`: forward, loss, `zero_grad`, `backward`, `step`.
- **In simple words:** the router learns by gradient descent to spread tokens.
- **Why it is here:** the classical approach.
### 15. Method A result  (`main.py` lines 101–102)
```python
A(x)
print("\nA) aux-loss trained router, utilisation:", A.counts().tolist(), f"| loss {balance_loss(A.last['probs'], A.last['top_idx'], 8).item():.3f}")
```
- **Exactly:** utilisation `[528, 499, 512, 522, 510, 504, 516, 505]`, loss 1.000. Max/min load ratio 1.06x. (With lr 0.05 the same loop collapsed to dead experts `[650, 1, 0, 683, ...]`: aux-loss training is sensitive to the learning rate, see Experiments.)
- **In simple words:** nearly perfect balance, but it needed gradient training of the router and a tuned learning rate.
- **Why it is here:** the baseline for the bias method.
### 16. Method B: bias update, no extra loss  (`main.py` lines 105–111)
```python
B = MoE()
B.load_state_dict(moe.state_dict())
gamma = 0.01                                                           # bias update speed
for step in range(300):
    B(x)
    c = B.counts().float()
    B.bias -= gamma * torch.sign(c - c.mean())                         # overloaded -> bias down; underloaded -> bias up
```
- **Exactly:** `gamma = 0.01`; each step count the load `c`, then `bias -= gamma · sign(c - mean(c))`: experts above average lose 0.01, below average gain 0.01. No gradient, no loss; 300 steps over the same batch.
- **In simple words:** a thermostat: if an expert is too busy, make it slightly less attractive.
- **Why it is here:** this is DeepSeek-V3's auxiliary-loss-free balancing.
### 17. Method B result  (`main.py` lines 112–115)
```python
B(x)
print("B) bias-balanced (no extra loss), utilisation:", B.counts().tolist(), "| bias:", B.bias.numpy().round(2))
print("   router weights untouched by balancing:", torch.equal(B.router.weight, moe.router.weight))
print(f"   max/min load: untrained {moe.counts().max() / max(moe.counts().min(), 1):.1f}x | A {A.counts().max() / max(A.counts().min(), 1):.2f}x | B {B.counts().max() / B.counts().min():.2f}x")
```
- **Exactly:** utilisation `[552, 551, 480, 489, 467, 584, 522, 451]`, max/min 1.29x; bias `[0.14, 0.13, -0.36, -0.16, ...]` (expert 2, the overloaded one, got the most negative bias). `torch.equal(B.router.weight, moe.router.weight)` is True: **the router weights were never changed**.
- **In simple words:** almost as balanced as method A without touching the router's learning.
- **Why it is here:** shows the mechanism works through selection only. (Production uses a running load estimate per step, not the same batch repeatedly.)
### 18. Real total vs active numbers  (`main.py` lines 118–122)
```python
print("\nmodel                 total params   active/token   experts (routed + shared), top-k")
for row in [("Mixtral 8x7B", "46.7B", "12.9B", "8 + 0, top-2"),
            ("DeepSeek-V3", "671B", "37B", "256 + 1, top-8"),
            ("Qwen3-30B-A3B", "30.5B", "3.3B", "128 + 0, top-8")]:
    print(f"{row[0]:<22}{row[1]:<15}{row[2]:<15}{row[3]}")
```
- **Exactly:** a hard-coded table from public model cards: Mixtral 8x7B 46.7B/12.9B; DeepSeek-V3 671B/37B; Qwen3-30B-A3B 30.5B/3.3B (128 experts, top-8). Printed only, not computed.
- **In simple words:** real models for scale.
- **Why it is here:** connects the toy to what you will be asked about.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `t.topk(k, dim=-1).indices` | indices of the k largest values | `(tokens, k)` |
| `p.gather(-1, idx)` | pick values at indices along an axis | weights of the chosen experts |
| `(a == e).nonzero(as_tuple=True)` | row and column indices where True | token and slot |
| `out[tok] += v` | indexed in-place add | scatter-add of expert outputs |
| `torch.bincount(x, minlength=n)` | count occurrences | tokens per expert |
| `self.register_buffer(name, t)` | non-trainable state in the module | balancing bias |
| `t.clamp(min=0)` | floor at zero | overflow counts |
| `torch.sign(x)` | -1, 0, +1 | direction of the bias update |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `nn.ModuleList` | torch | holds sub-modules | iterable | module |
| `F.silu` | torch | x·sigmoid(x) | tensor | tensor |
| `torch.optim.Adam` | torch | optimiser | params, lr | optimiser |
| `module.load_state_dict` | torch | copy weights | dict | result |
| `torch.equal` | torch | exact equality | two tensors | bool |

## Execution trace (real run)
```text
one expert 24,576 | router 512 | total layer 221,696
active per token (router + 2 routed + 1 shared) = 74,240  ->  33.5% of the total

output (2048, 64) | tokens x k = 4096 routed slots
utilisation, untrained router      : [73, 161, 1047, 650, 1263, 4, 59, 839]
aux balance loss (1.0 is perfect)  : 1.505
capacity per expert (factor 1.25) = 640; slots that would be DROPPED = 1239 (30.2%)

A) aux-loss trained router, utilisation: [528, 499, 512, 522, 510, 504, 516, 505] | loss 1.000
B) bias-balanced (no extra loss), utilisation: [552, 551, 480, 489, 467, 584, 522, 451] | bias: [ 0.14  0.13 -0.36 -0.16 -0.14  0.18  0.15  0.08]
   router weights untouched by balancing: True
   max/min load: untrained 315.8x | A 1.06x | B 1.29x

model                 total params   active/token   experts (routed + shared), top-k
Mixtral 8x7B          46.7B          12.9B          8 + 0, top-2
DeepSeek-V3           671B           37B            256 + 1, top-8
Qwen3-30B-A3B         30.5B          3.3B           128 + 0, top-8
```
Reading it: the untrained router sends 1,263 slots to expert 4 and 4 to expert 5; 30% of slots would be dropped at capacity factor 1.25. The aux loss reaches loss 1.000 (perfect); the bias method reaches 1.29x max/min without ever updating the router.

## Experiments (change ONE thing)
Every row below was run on a temporary copy of `main.py` (the real file is untouched); only the named setting changed. Baseline for comparison: untrained utilisation `[73, 161, 1047, 650, 1263, 4, 59, 839]`, 30.2% of slots dropped, aux-loss router max/min 1.06x, bias-balanced max/min 1.29x, active 33.5% of the layer.

| Change | What you will see | Why |
|---|---|---|
| `top_k=1, n_shared=0` (measured) | active per token 12.7% of the layer (was 33.5%); untrained router sends everything to 4 experts, `[0, 0, 712, 650, 682, 0, 0, 4]`; bias balancing ends at max/min 2.47x after the same 300 steps (was 1.29x) | Fewer chosen experts means less compute per token but a coarser, more extreme routing; with k=1 each expert's load moves in bigger jumps, so the same bias step needs more time to even out. |
| `n_shared=0` only (measured) | active per token 25.2% of the layer (was 33.5%) and total layer 197,120 params (was 221,696) | The shared expert is one more full expert (24,576 params) that every token pays for. |
| `gamma = 0.1` (measured) | bias-balanced loads `[1206, 1, 286, 373, 149, 873, 1180, 28]`, max/min 1206x (was `[552, 551, 480, ...]`, 1.29x) | Too large a bias step overshoots: experts flip between overloaded and starved every update instead of settling. The bias is a thermostat, and a big step makes it oscillate. |
| `gamma = 0.0001` (measured) | max/min 36.0x after 300 steps, bias values only about 0.03 | Too small a step simply has not moved the bias far enough yet: the total drift is `steps * gamma`, here at most 0.03. |
| `top_k=9` with 8 experts (measured) | `RuntimeError: selected index k out of range` | You cannot pick more distinct experts than exist. |

## Debugging practice
**BREAK IT drill 1: ask for more experts than exist.** In `MoE.__init__` defaults:

```diff
-    def __init__(self, d=64, h=128, n_experts=8, top_k=2, n_shared=1):
+    def __init__(self, d=64, h=128, n_experts=8, top_k=9, n_shared=1):
```

Symptom (real run), raised at the first forward pass:

```text
    top_idx = (probs + self.bias).topk(self.k, dim=-1).indices
RuntimeError: selected index k out of range
```

Root cause: `topk(9)` over a dimension of size 8. Fix: `top_k <= n_experts` (real models use 2 of 8, 8 of 128, 8 of 256). Add `assert top_k <= n_experts` to fail with a readable message.

**BREAK IT drill 2: flip the sign of the bias update.** In the Method B loop:

```diff
-    B.bias -= gamma * torch.sign(c - c.mean())
+    B.bias += gamma * torch.sign(c - c.mean())
```

Symptom (real run), no exception, but balancing makes things worse:

```text
B) bias-balanced (no extra loss), utilisation: [0, 0, 1107, 650, 1378, 0, 0, 961] | bias: [-3. -3.  3.  3.  3. -3. -3.  3.]
   max/min load: untrained 315.8x | A 1.06x | B infx
```

Root cause: the rule is "overloaded experts get a lower bias so fewer tokens select them". With `+=` the busiest experts are boosted and the idle ones penalised, a runaway feedback loop: the bias grows to exactly `+-300 * 0.01 = 3` and four experts receive nothing. Fix: restore `-=`. Cheap guard: print the max/min load every few steps; it should fall, not rise.

## Common mistakes
- Mixing up total and active parameters. Memory has to hold all experts (total); per-token compute only touches the active ones.
- Using the biased scores to weight the expert outputs. In DeepSeek-V3 style balancing the bias decides *selection* only; the output weights come from the raw router probabilities (see `top_w = probs.gather(...)`).
- Forgetting to renormalise the k selected weights so they sum to 1.
- Looking only at the average load. The aux loss and capacity matter because the maximum load decides how long the slowest GPU takes, and capacity overflow drops tokens.
- Setting the balancing step too high (oscillation, drill 2 variants) or too low (nothing moves in time).
- Assuming an untrained router is balanced: here it left an expert with 4 of 4096 slots and another with 1263.

## Rebuild checklist
- [ ] router: `Linear(d, E)`, softmax, `topk`, renormalise the k weights
- [ ] dispatch loop over experts with `nonzero` and indexed add
- [ ] shared experts added for all tokens
- [ ] aux loss `E * sum(f * P)` and why `f` has no gradient
- [ ] bias balancing: select with `p + b`, weight with `p`, update `b` by `-gamma * sign(load - mean)`
- [ ] total vs active parameters, and what each costs (memory vs FLOPs)

## The new problem (cliffhanger)
We now have every part of a 2026 decoder: BPE tokens, RoPE, RMSNorm, SwiGLU or MoE, GQA/MLA, Flash kernels. Reading about them separately is not the same as seeing them work together. Next we assemble them into one small, trainable, cache-aware Llama-style model: [Step 07 — build a mini-Llama](../07_build_mini_llama/WALKTHROUGH.md).

> **Say it aloud (interview box):** "MoE replaces the MLP by many expert MLPs and a router that picks the top-k per token, so total parameters (memory) grow but active parameters (compute) stay small. The router must be load-balanced, either with a Switch-style auxiliary loss or, as in DeepSeek-V3, with a per-expert bias used only for selection and nudged up or down according to load. Shared experts hold common knowledge."
