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
