# Step 14 — Parallelism and offloading: when one GPU is not enough

## The story so far
[Step 13](../13_speculative_decoding/WALKTHROUGH.md) made one model faster by guessing tokens ahead. Steps 08 to 12 squeezed one GPU: KV cache, roofline, batching, paged memory, prefix caching. [Step 03](../03_gpu_memory_calculator/WALKTHROUGH.md) gave the formula that says whether weights plus KV cache fit. This step is about the case where the answer is "no".

## The pain point
A 70B model in fp16 is about 140 GB of weights. An 80 GB GPU cannot hold it, and even a model that fits leaves no room for the KV cache of many users. You can quantize (steps 04 to 07), but at some size you must either split the model across several devices (parallelism) or park part of it in slower memory (offloading). Both sound easy and both hide a trap: every split creates communication, and communication is slow compared with arithmetic.

## The spark
Matrix multiplication is *linear algebra that can be cut into pieces*. Cut a weight matrix by columns and each device computes a slice of the output; cut it by rows and each device computes a partial sum. Put a column cut then a row cut back to back (the Megatron pattern) and a whole MLP block needs only one all-reduce. Cut the *layers* instead and devices need to exchange only one activation vector per boundary. NumPy can prove all of this exactly, with no GPU.

## Goal
Prove tensor parallelism equals the unsplit layer, estimate communication per token, simulate the pipeline bubble and MoE all-to-all, compare data parallel vs ZeRO/FSDP, and run one real offload (`device_map` + `max_memory`) on distilgpt2. Then decide: TP, PP, EP, or offload?

## The theory (read this before the interview)

**Five ways to split work** (the names you will hear):

```
Data parallel (DP)      every GPU holds the WHOLE model, different requests/batches      -> scales throughput, no memory saving
Tensor parallel (TP)    every layer's matrices are cut across GPUs                       -> scales memory + per-token speed, needs fast links
Pipeline parallel (PP)  consecutive layers on different GPUs (stage 0: layers 0-15 ...)  -> scales memory, little traffic, has a bubble
Expert parallel (EP)    MoE experts live on different GPUs; tokens travel to them        -> all-to-all traffic, load imbalance
Sequence/context par.   the SEQUENCE is cut across GPUs (ring attention)                 -> for contexts whose KV cache alone is too big
```

**Collectives** (communication primitives; NCCL implements them on NVIDIA, RCCL on AMD):

```
all-gather      each device has a slice -> every device gets the concatenation of all slices
all-reduce      each device has a partial tensor -> every device gets the SUM (= reduce-scatter + all-gather)
all-to-all      device i sends a different chunk to every device j (a transpose of who-holds-what)
point-to-point  send/recv between two neighbours (pipeline stages)
ring all-reduce moves 2(N-1)/N x S bytes per device for a tensor of S bytes
```

**Column-parallel linear.** `Y = X W`, `W` is `(d_in, d_out)`. Split `W = [W1 | W2 | ... | WN]` by columns. Device `i` computes `Y_i = X W_i` from the full `X`; `Y = [Y_1 | ... | Y_N]`. No communication to compute; concatenation (all-gather) only if the next op needs all columns.

**Row-parallel linear.** Split `W` by rows and `X` by columns: `X W = sum_i X_i W_i`. Each device holds only its slice `X_i` and produces a *partial sum of full width*; an all-reduce adds them.

**Megatron MLP.** `MLP(X) = GELU(X A) B`. Split `A` by columns, `B` by rows. GELU is elementwise, so device `i` can apply it to its own slice. The slice it produced is exactly the input slice the row-parallel `B_i` wants. So the block needs **one all-reduce** (at the end). Splitting `A` by rows instead would need a sum *before* GELU, because `GELU(a + b) != GELU(a) + GELU(b)`.

**Attention.** Heads are independent, so give each device `n_heads / N` heads: column-split `W_Q, W_K, W_V` (each device gets whole heads), run attention locally, then row-split `W_O` and all-reduce. Each device also stores the KV cache of only its own heads, which is why TP divides KV memory (see [03](../03_gpu_memory_calculator/WALKTHROUGH.md)). With GQA, `kv_heads` must be at least `N` (or be replicated).

**Communication per token.** A layer has 2 all-reduces in the forward pass (after attention output and after MLP down-projection). Each moves `hidden x bytes` per token, ring-scaled by `2(N-1)/N`. For a llama-3-8b-shaped model (32 layers, hidden 4096, fp16): `64 x 8 KB x 2(N-1)/N` per token per GPU, about 0.9 MB at `N=8`. The *size* is small; the problem is **64 small, latency-bound collectives per token**, which is only tolerable over NVLink inside a node.

**Pipeline parallelism and the bubble.** `p` stages, a batch cut into `m` micro-batches. A stage cannot start micro-batch `j` before the previous stage finished it, so at the start and end some stages idle:

```
bubble fraction = (p - 1) / (m + p - 1)        e.g. p=4: m=1 -> 75%, m=4 -> 43%, m=16 -> 16%, m=64 -> 4.5%
```

For decode there is a second effect: one sequence's next token depends on the previous token, so PP only keeps stages busy when many requests are in flight.

**Data parallel vs ZeRO/FSDP (training side; see [module 06](../../06_llm_training_and_lora/README.md)).** DDP replicates weights, gradients and optimizer state on every GPU: 16 bytes/param with mixed-precision Adam. ZeRO-1 shards the optimizer state, ZeRO-2 also gradients, ZeRO-3 / FSDP (fully sharded data parallel) also the weights, gathering each layer just in time. Same arithmetic as DP, memory divided by N, extra all-gather traffic.

**Expert parallelism (MoE; see [16/06](../../16_modern_llm_architecture/06_mixture_of_experts/WALKTHROUGH.md)).** A router sends each token to its top-k experts. If experts sit on different GPUs, tokens are shipped there (all-to-all #1), processed, and shipped back (all-to-all #2). If the router sends most tokens to a few experts those GPUs become stragglers (load imbalance).

**Offloading (a different trade: memory for latency).** Keep part of the model in CPU RAM or on disk and move it to the GPU when needed.
- Hugging Face Accelerate: `from_pretrained(..., device_map="auto", max_memory={0: "10GiB", "cpu": "30GiB"})` fills GPU 0 first, then CPU RAM, then disk (`offload_folder`).
- llama.cpp: `-ngl N` puts N layers on the GPU; the rest are computed on CPU.
- Every token must stream the offloaded weights over PCIe (~tens of GB/s) or from disk (a few GB/s), so decode can become 10 to 100 times slower. It is a way to run, not a way to serve fast.

**TP vs PP: the rule of thumb.**

```
Within a node (NVLink/NVSwitch, hundreds of GB/s)   -> tensor parallel (TP = 2, 4, 8)
Across nodes (InfiniBand/Ethernet, ~10-50 GB/s)     -> pipeline parallel (PP) between nodes, TP inside each node
MoE models                                          -> expert parallel on top
Throughput only, model fits on one GPU              -> just replicate (data parallel): zero communication
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| TP (tensor parallelism) | intra-layer parallelism, model parallel (Megatron-LM), `--tensor-parallel-size`, `--tp` | Yes | Each matrix is cut across GPUs; every layer communicates. |
| PP (pipeline parallelism) | inter-layer parallelism, `--pipeline-parallel-size` | Yes | Whole layers on different GPUs; communication only between stages. |
| DP (data parallelism) | replicas, DDP, data-parallel serving | Yes | Full copy per GPU. |
| EP (expert parallelism) | MoE sharding | Yes | Experts spread across GPUs, tokens move by all-to-all. |
| SP / CP (sequence / context parallelism) | ring attention, Ulysses | Related, not identical | Cut the sequence dimension instead of the weight dimension. |
| ZeRO-1/2/3 | DeepSpeed ZeRO, FSDP (PyTorch) | Close | Shard optimizer / gradients / weights over DP ranks (training). |
| all-reduce | NCCL AllReduce, ring all-reduce | Yes | Sum across devices, result on all. |
| all-gather | concat of shards | Yes | Everyone receives everyone's slice. |
| all-to-all | AllToAll, token shuffle | Yes | Transpose of who holds which chunk (MoE dispatch). |
| Offloading | CPU offload, disk offload, `device_map`, `-ngl`, "layer splitting" | Mostly | Trade latency for capacity by paging weights from slower memory. |
| Bubble | pipeline bubble, pipeline fill/drain | Yes | Idle time of stages while the pipeline fills and drains. |

## Run it

```
cd 21_llm_inference_engineering/14_parallelism_and_offloading
python main.py                                  # sections A-H are pure NumPy; I loads distilgpt2 (use the global lock)
HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/14_parallelism_and_offloading/main.py   # from repo root
```

About 10 seconds on CPU. If `accelerate` fails or is missing, section I prints a skip message and everything else still runs.

## Code, step by step

### 1. Header, imports and the simulated "devices"  (`main.py` lines 1–29)

```python
"""
GOAL:  Prove with NumPy that tensor parallelism is exact (column split = all-gather, row split = all-reduce, Megatron
       MLP = one all-reduce per block, attention = split by heads), simulate pipeline bubbles and MoE all-to-all,
       estimate communication per token, and try a REAL CPU/disk offload of distilgpt2 layers with device_map.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/14_parallelism_and_offloading/main.py
EXPECTED OUTPUT (shortened):
    [A] column-parallel (N=4)  max|diff| = ~1e-15  -> all-gather of output slices equals the full layer
    [B] row-parallel    (N=4)  max|diff| = ~1e-15  -> all-reduce (sum) of partial outputs equals the full layer
    [C] Megatron MLP  (N=4)    max|diff| = ~1e-15  -> ONE all-reduce for the whole MLP block
    [D] attention sharded by heads (N=4) max|diff| = ~1e-16
    [E] comm per token ... [F] pipeline bubble table ... [G] MoE all-to-all ... [H] ZeRO table
    [I] REAL offload: distilgpt2 with max_memory -> some layers on 'disk' / 'cpu'; logits match the plain model
NEEDS: numpy; torch + transformers + accelerate for [I] (distilgpt2 cached; skipped if it fails). ~10 s on CPU.
       Section [I] loads a causal LM: run through the global lock.
"""
import os
import tempfile

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np

rng = np.random.default_rng(0)
N = 4                                   # number of simulated devices (GPUs).  Each "device" is just a slice of an array.


def gelu(x):                            # tanh approximation used by GPT-2 / Megatron
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
```

- **In simple words:** Each of the `N=4` "devices" is just a slice of a NumPy array. No GPUs are involved, because the *mathematics* of splitting does not depend on hardware.
- **Exactly:** `rng` is a seeded NumPy random generator so every run gives the same matrices. Section I needs `torch`, `transformers` and `accelerate` and imports them lazily inside a function.
- **Why it is here:** If the proof works in float64 NumPy, any later mismatch on real GPUs is a precision or communication problem, not a math problem.

### 2. Column-parallel linear = all-gather  (`main.py` lines 32–42)

```python
# ---- [A] Column-parallel linear: split the WEIGHT's output columns; outputs are CONCATENATED (all-gather) ------------
# Y = X @ W   with X (tokens, d_in), W (d_in, d_out).  Device i holds W[:, i-th block of columns] and the FULL X.
# Each device computes a slice of Y with no communication; gathering the slices is a concatenation.
d_in, d_out, T = 16, 32, 5
X = rng.standard_normal((T, d_in))
W = rng.standard_normal((d_in, d_out))
Y_full = X @ W
col_shards = np.split(W, N, axis=1)                         # N arrays of shape (d_in, d_out/N)
Y_parts = [X @ w for w in col_shards]                       # runs on device i, independent of the others
Y_col = np.concatenate(Y_parts, axis=1)                     # "all-gather": every device ends up with all columns
print(f"[A] column-parallel (N={N})  max|diff| = {np.abs(Y_col - Y_full).max():.1e}  -> all-gather of output slices == full layer")
```

- **In simple words:** Cut the weight into 4 vertical strips; each device multiplies the same input by its strip; gluing the four output pieces side by side gives the original answer.
- **Exactly:** `np.split(W, N, axis=1)` returns 4 views of shape `(16, 8)`. `np.concatenate(..., axis=1)` is the all-gather. The difference printed is `0.0`: column splitting does not even change the floating-point order of any sum.
- **Why it is here:** It is the half of tensor parallelism with no communication needed to *compute*.

### 3. Row-parallel linear = all-reduce  (`main.py` lines 44–51)

```python
# ---- [B] Row-parallel linear: split the WEIGHT's input rows (and X's columns); outputs are SUMMED (all-reduce) --------
# X @ W = sum_i X[:, block i] @ W[block i, :].  Device i needs only its slice of X (which a previous column-parallel
# layer produced for free) and holds W[block i, :].  Each makes a PARTIAL sum of the full-width output.
row_shards = np.split(W, N, axis=0)                         # (d_in/N, d_out)
x_shards = np.split(X, N, axis=1)                           # (T, d_in/N)
partials = [x @ w for x, w in zip(x_shards, row_shards)]    # each is (T, d_out): a partial sum, NOT a final answer
Y_row = np.sum(partials, axis=0)                            # "all-reduce(sum)": the one communication step
print(f"[B] row-parallel    (N={N})  max|diff| = {np.abs(Y_row - Y_full).max():.1e}  -> all-reduce (sum) of partial outputs == full layer")
```

- **In simple words:** Cut the weight into horizontal strips and the input into matching column pieces; each device makes a full-size but incomplete answer; adding the four gives the true answer.
- **Exactly:** `x @ w` has shape `(T, d_out)`, the same shape as the final output, so the all-reduce moves `T x d_out` numbers. The ~1e-15 difference is float rounding from adding in a different order.
- **Why it is here:** This is where the communication cost of TP comes from.

### 4. Megatron MLP: one all-reduce for the whole block  (`main.py` lines 53–66)

```python
# ---- [C] Megatron MLP: column-parallel, elementwise GELU, row-parallel -> ONE all-reduce -----------------------------
# MLP(X) = GELU(X @ A) @ B.  Split A by columns: GELU is elementwise so each device applies it to ITS slice alone.
# Split B by rows: the slice each device already holds is exactly the input slice the row-parallel layer wants.
# No communication between the two matmuls; only the final partial sums need an all-reduce.
d, h = 16, 64                                               # hidden size and MLP inner size (usually 4*d)
A, B = rng.standard_normal((d, h)), rng.standard_normal((h, d))
Xm = rng.standard_normal((T, d))
mlp_full = gelu(Xm @ A) @ B
dev_out = [gelu(Xm @ a) @ b for a, b in zip(np.split(A, N, axis=1), np.split(B, N, axis=0))]   # per-device partial result
mlp_tp = np.sum(dev_out, axis=0)                            # the single all-reduce
print(f"[C] Megatron MLP    (N={N})  max|diff| = {np.abs(mlp_tp - mlp_full).max():.1e}  -> ONE all-reduce for the whole MLP block")
# If we had split A by ROWS instead, GELU(partial sum) != sum(GELU(partial)): we would need an all-reduce BEFORE GELU too.
wrong = np.sum([gelu(x @ a) for x, a in zip(np.split(Xm, N, axis=1), np.split(A, N, axis=0))], axis=0)
print(f"    (row-split first layer, no sync before GELU would be WRONG: max|diff| = {np.abs(wrong @ B - mlp_full).max():.2f})")
```

- **In simple words:** Do the first matrix by columns, apply GELU on each device's own slice, do the second matrix by rows, and add up once at the end. The `wrong` line shows what breaks if you cut the first matrix the other way without syncing before GELU.
- **Exactly:** `gelu` is elementwise, so `gelu(concat(parts)) == concat(gelu(parts))`; that is the property that removes the middle communication. The `wrong` computation applies GELU to partial sums, which is not the same function, hence a large error.
- **Why it is here:** It is the central design trick of Megatron-LM, and a favourite interview question ("why column then row?").

### 5. Attention sharded by heads  (`main.py` lines 68–94)

```python
# ---- [D] Attention: shard by HEADS (heads are independent), output projection is row-parallel ------------------------
n_heads, dh = 8, 4
dm = n_heads * dh
Wq, Wk, Wv, Wo = (rng.standard_normal((dm, dm)) for _ in range(4))
Xa = rng.standard_normal((T, dm))
causal = np.tril(np.ones((T, T), bool))


def head_attn(x, wq, wk, wv):
    """One group of heads: x (T, dm); wq/wk/wv (dm, g*dh). Returns (T, g*dh). Heads never talk to each other."""
    g = wq.shape[1] // dh
    q, k, v = (x @ w for w in (wq, wk, wv))
    outs = []
    for j in range(g):
        s = slice(j * dh, (j + 1) * dh)
        sc = q[:, s] @ k[:, s].T / np.sqrt(dh)
        sc = np.where(causal, sc, -1e30)
        p = np.exp(sc - sc.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        outs.append(p @ v[:, s])
    return np.concatenate(outs, 1)


attn_full = head_attn(Xa, Wq, Wk, Wv) @ Wo
shards = zip(np.split(Wq, N, 1), np.split(Wk, N, 1), np.split(Wv, N, 1), np.split(Wo, N, 0))
attn_tp = np.sum([head_attn(Xa, q, k, v) @ o for q, k, v, o in shards], axis=0)      # column-split QKV, row-split O
print(f"[D] attention sharded by heads (N={N}) max|diff| = {np.abs(attn_tp - attn_full).max():.1e}  (each device keeps only ITS heads' KV cache)")
```

- **In simple words:** Each device owns whole attention heads. Heads never read each other, so no talking is needed until the output projection.
- **Exactly:** `head_attn` runs causal softmax attention for a group of heads. `np.split(Wq, N, 1)` gives each device `n_heads/N = 2` heads (columns `2*dh` wide). `Wo` is split by rows, so the partial outputs are summed. Each device would keep the KV cache of only its own heads.
- **Why it is here:** It explains why TP divides KV-cache memory and why `kv_heads` must be at least the TP degree.

### 6. Communication volume per token  (`main.py` lines 96–113)

```python
# ---- [E] Communication volume per token (why TP needs NVLink) ---------------------------------------------------
# A ring all-reduce of a tensor of S bytes over N devices moves 2*(N-1)/N * S bytes per device.
# A transformer layer has 2 all-reduces in the forward pass (after attention-out and after MLP-down), each of size
# hidden * bytes_per_value per token.
def tp_comm_bytes_per_token(layers, hidden, n, bytes_per_value=2):
    per_allreduce = hidden * bytes_per_value * 2 * (n - 1) / n
    return layers * 2 * per_allreduce


llama8b = dict(layers=32, hidden=4096)
print("[E] TP communication per generated token (llama-3-8b shape, fp16 activations):")
for n in (2, 4, 8):
    b = tp_comm_bytes_per_token(llama8b["layers"], llama8b["hidden"], n)
    print(f"      TP={n}:  {b / 1e6:.2f} MB per token per GPU  | 64 all-reduces of {llama8b['hidden'] * 2 / 1e3:.0f} KB each (latency-bound!)")
print("      PP sends only ONE activation vector per stage boundary: hidden*2 bytes =", llama8b["hidden"] * 2, "bytes per token")
for name, bw in (("NVLink ~450 GB/s", 450e9), ("PCIe5 ~50 GB/s", 50e9), ("100GbE ~12 GB/s", 12e9)):
    t = tp_comm_bytes_per_token(32, 4096, 8) / bw
    print(f"      TP=8 bandwidth-time per token over {name:<17}: {t * 1e6:8.1f} us (+ 64 x link latency)")
```

- **In simple words:** How many bytes must each GPU send per generated token, and how long would that take on different links?
- **Exactly:** ring all-reduce factor `2(N-1)/N`; two all-reduces per layer; `hidden * 2` bytes is one fp16 activation vector. Bandwidth figures are rounded examples of NVLink, PCIe 5 and 100 GbE, not measurements of your machine.
- **Why it is here:** Volume is small (under 1 MB) but there are 64 sequential collectives per token, so *latency* and link speed decide whether TP is worth it. That is why TP stays inside a node.

### 7. Pipeline parallelism and the bubble  (`main.py` lines 115–133)

```python
# ---- [F] Pipeline parallelism: micro-batches and the bubble ---------------------------------------------------------
# p stages, m micro-batches, each (stage, micro-batch) takes 1 time unit. Stage s can start micro-batch j at time s+j.
def pipeline_schedule(p, m):
    total = m + p - 1
    busy = np.zeros((p, total), int)
    for s in range(p):
        for j in range(m):
            busy[s, s + j] = 1
    return busy


print("[F] pipeline parallelism: bubble fraction (p-1)/(m+p-1)")
for p in (4,):
    for m in (1, 4, 16, 64):
        busy = pipeline_schedule(p, m)
        sim = 1 - busy.sum() / busy.size                     # idle cells / all cells
        print(f"      p={p} m={m:<3} simulated idle = {sim:.3f} | formula = {(p - 1) / (m + p - 1):.3f}")
for row in pipeline_schedule(4, 4):                          # draw the p=4, m=4 timeline
    print("      " + "".join("#" if c else "." for c in row))
```

- **In simple words:** Draw a timetable: rows are stages, columns are time ticks, `#` is busy. The dots are idle time (the bubble). More micro-batches shrink the dots.
- **Exactly:** stage `s` handles micro-batch `j` at tick `s + j`, so the timetable has `m + p - 1` columns. `busy.sum()/busy.size` counts busy cells; one minus that is the measured bubble, which equals `(p-1)/(m+p-1)`.
- **Why it is here:** It turns a formula into something you can see, and shows why PP needs many micro-batches.

### 8. Expert parallelism and all-to-all  (`main.py` lines 135–151)

```python
# ---- [G] Expert parallelism (MoE): each device owns some experts; tokens travel by ALL-TO-ALL ----------------------------
n_exp, T2, dm2 = 8, 24, 8                                    # 8 experts, 4 devices -> 2 experts per device
Xt = rng.standard_normal((T2, dm2))
Wr = rng.standard_normal((dm2, n_exp))
experts = rng.standard_normal((n_exp, dm2, dm2))
top1 = (Xt @ Wr).argmax(1)                                   # router: top-1 expert per token (real MoE uses top-2..8)
moe_full = np.stack([Xt[t] @ experts[top1[t]] for t in range(T2)])
dev_of = top1 // (n_exp // N)                                # which device owns the chosen expert
src_dev = np.arange(T2) // (T2 // N)                         # tokens start spread over the devices (data parallel)
sent = int((dev_of != src_dev).sum())                        # tokens that must cross the network (all-to-all #1)
moe_ep = np.zeros_like(Xt)
for dev in range(N):                                         # each device processes the tokens routed to ITS experts
    idx = np.where(dev_of == dev)[0]
    for t in idx:
        moe_ep[t] = Xt[t] @ experts[top1[t]]                 # (results travel back: all-to-all #2)
print(f"[G] expert parallelism: max|diff| = {np.abs(moe_ep - moe_full).max():.1e} | tokens crossing devices: {sent}/{T2} (each way) "
      f"| per-device load = {np.bincount(dev_of, minlength=N).tolist()}  <- imbalance = stragglers")
```

- **In simple words:** A router picks one expert per token; experts are spread over 4 devices; most tokens must travel to another device and come back.
- **Exactly:** `dev_of` is the device that owns the chosen expert; `src_dev` is where the token started; `sent` counts tokens crossing devices in each direction (one all-to-all each way). `bincount` shows how many tokens each device must process: an uneven count (like `[9, 1, 13, 1]` in the real run below) means the busiest device sets the step time.
- **Why it is here:** It is the MoE-specific cost (link [16/06](../../16_modern_llm_architecture/06_mixture_of_experts/WALKTHROUGH.md)); real routers add load-balancing losses and capacity limits to fight this imbalance.

### 9. Data parallel vs ZeRO / FSDP (training)  (`main.py` lines 153–161)

```python
# ---- [H] Data parallelism vs ZeRO / FSDP (training side; see module 06 and 16/06 for MoE) ---------------------------
# Mixed-precision Adam: 2 B weights + 2 B grads + 12 B optimizer state (fp32 master + m + v) = 16 B/param.
params, n = 7e9, 8
print(f"[H] training memory per GPU, 7B params, {n} GPUs (weights+grads+Adam, activations excluded):")
for name, w, g, o in (("DDP (replicate all)", 1, 1, 1), ("ZeRO-1 (shard optimizer)", 1, 1, 1 / n),
                      ("ZeRO-2 (+ gradients)", 1, 1 / n, 1 / n), ("ZeRO-3 / FSDP (+ weights)", 1 / n, 1 / n, 1 / n)):
    gb = params * (2 * w + 2 * g + 12 * o) / 1e9
    print(f"      {name:<28}{gb:7.1f} GB")
print("      sequence/context parallelism (concept): split the SEQUENCE across devices (ring attention) when one request's KV cache is too big.")
```

- **In simple words:** With 7B parameters and Adam, replicating everything needs 112 GB per GPU. Sharding the optimizer state, then gradients, then weights brings it down to 14 GB across 8 GPUs.
- **Exactly:** 16 bytes per parameter = 2 (fp16 weights) + 2 (fp16 grads) + 12 (fp32 master copy + Adam `m` + `v`); the weights/grads/optimizer fractions are divided by `n` as each ZeRO stage shards another piece. Activations are not counted.
- **Why it is here:** It is the training-side answer to the same "does not fit" problem; inference rarely needs it.

### 10. A real offload: `device_map` and `max_memory`  (`main.py` lines 163–197)

```python
# ---- [I] REAL offloading demo: device_map + max_memory with accelerate on distilgpt2 ------------------------------------
# accelerate walks the layers in order and puts as many as fit into the memory budget you give per device, then spills
# to the next place: "cpu" (RAM) and finally "disk" (weights saved as .dat and loaded layer-by-layer at forward time).
# On a GPU box: max_memory={0: "10GiB", "cpu": "30GiB"}; here we pretend our only 'device' is a tiny CPU budget.
def offload_demo():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    ids = tok("Parallelism splits one model across many chips", return_tensors="pt").input_ids
    ref = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
    with torch.no_grad():
        ref_logits = ref(ids).logits
    del ref
    with tempfile.TemporaryDirectory() as off_dir:
        m = AutoModelForCausalLM.from_pretrained(
            "distilgpt2", device_map="auto",
            max_memory={"cpu": "220MiB"},                       # 'device' budget smaller than the 330 MB model, so the last layers spill to disk
            offload_folder=off_dir,
        ).eval()
        dm = m.hf_device_map
        where = {}
        for k, v in dm.items():
            where.setdefault(str(v), []).append(k)
        for dev, mods in where.items():
            print(f"      {dev:<5}: {len(mods)} module(s), e.g. {mods[:3]}")
        with torch.no_grad():
            out = m(ids).logits
        print(f"      logits max|diff| vs plain model = {(out - ref_logits).abs().max().item():.2e}  (offloading changes WHERE, not WHAT)")


print("[I] REAL offload: distilgpt2, device_map='auto', max_memory={'cpu': '220MiB'}, offload_folder=tmp")
try:
    offload_demo()
except Exception as e:                                       # accelerate/device_map problems must not break the lesson
    print(f"      skipped ({type(e).__name__}: {str(e)[:120]}) -- see WALKTHROUGH for the API")
```

- **In simple words:** Tell Hugging Face "you may only use 220 MiB of CPU RAM". The model is about 330 MB, so the early layers stay in RAM and the later ones are stored on disk and read back when needed. The output is still identical.
- **Exactly:** `device_map="auto"` makes Accelerate compute a placement of every top-level module; `max_memory` caps each device; whatever does not fit goes to `"disk"` with weights written to `offload_folder`. Accelerate installs hooks that move a module's weights to the compute device just before its forward and release them after. `m.hf_device_map` is the dict `module name -> device`. On a GPU machine you would write `max_memory={0: "10GiB", "cpu": "30GiB"}`.
- **Why it is here:** It shows the offloading API working for real and proves it changes where weights live, not what the model computes. It also shows the cost: every forward re-reads disk-resident layers.

### 11. Try it / break it  (`main.py` lines 199–203)

```python
# TRY IT: change N to 2 or 8 in [A]-[D] (heads and widths must divide N); the diff stays ~1e-15.
# TRY IT: in [F] print p=8, m=8 and compare the bubble to p=4, m=8: why do deeper pipelines need more micro-batches?
# TRY IT: in [I] set max_memory={"cpu": "300MiB"} and watch fewer modules land on disk.
# BREAK IT: in [B] sum only N-1 of the partials (forget one device): the error is large, as with a failed all-reduce.
# BREAK IT: in [C] use gelu AFTER summing partials of a row-split first layer: it is wrong, which is why Megatron splits A by columns.
```

- **In simple words:** Ideas for experiments and deliberate bugs.
- **Exactly:** comments only.
- **Why it is here:** Habit: change one thing and predict the result first.

## Syntax explained

| Syntax | Meaning |
|---|---|
| `np.split(W, N, axis=1)` | cut an array into N equal pieces along an axis (a list of views) |
| `np.concatenate(parts, axis=1)` | glue arrays back together along an axis (the all-gather) |
| `np.sum(list_of_arrays, axis=0)` | element-wise sum of equal-shape arrays (the all-reduce) |
| `zip(a, b)` | iterate two lists in lock-step: "device i's shard of A and of B" |
| `np.tril(np.ones((T, T), bool))` | lower-triangular mask: position t may look only at positions <= t |
| `busy[s, s + j] = 1` | fill the pipeline timetable cell for stage s, micro-batch j |
| `with tempfile.TemporaryDirectory() as d:` | a scratch folder that is deleted on exit (holds the offloaded weights) |
| `try: ... except Exception` | keep the lesson running if accelerate is missing or fails |

## Libraries & functions

| Function | What it does | Internally |
|---|---|---|
| `AutoModelForCausalLM.from_pretrained(name, device_map="auto", max_memory=..., offload_folder=...)` | Load a model with a computed device placement | Builds the model on the "meta" device (no memory), asks Accelerate `infer_auto_device_map` for a placement, loads each weight to its device, saves disk-bound weights as memory-mapped files |
| `model.hf_device_map` | dict of module name to device | The placement Accelerate decided |
| Accelerate hooks (`AlignDevicesHook`) | Move weights in and out per module | Before forward: load the offloaded weights to the execution device; after: release them |
| NCCL all-reduce (not run here) | the real collective | Ring or tree algorithm over NVLink/IB; PyTorch calls it as `torch.distributed.all_reduce` |

## Execution trace (real run)

```
[A] column-parallel (N=4)  max|diff| = 0.0e+00  -> all-gather of output slices == full layer
[B] row-parallel    (N=4)  max|diff| = 3.6e-15  -> all-reduce (sum) of partial outputs == full layer
[C] Megatron MLP    (N=4)  max|diff| = 2.1e-14  -> ONE all-reduce for the whole MLP block
    (row-split first layer, no sync before GELU would be WRONG: max|diff| = 33.31)
[D] attention sharded by heads (N=4) max|diff| = 1.4e-14  (each device keeps only ITS heads' KV cache)
[E] TP communication per generated token (llama-3-8b shape, fp16 activations):
      TP=2:  0.52 MB per token per GPU  | 64 all-reduces of 8 KB each (latency-bound!)
      TP=4:  0.79 MB per token per GPU  | 64 all-reduces of 8 KB each (latency-bound!)
      TP=8:  0.92 MB per token per GPU  | 64 all-reduces of 8 KB each (latency-bound!)
      PP sends only ONE activation vector per stage boundary: hidden*2 bytes = 8192 bytes per token
      TP=8 bandwidth-time per token over NVLink ~450 GB/s :      2.0 us (+ 64 x link latency)
      TP=8 bandwidth-time per token over PCIe5 ~50 GB/s   :     18.4 us (+ 64 x link latency)
      TP=8 bandwidth-time per token over 100GbE ~12 GB/s  :     76.5 us (+ 64 x link latency)
[F] pipeline parallelism: bubble fraction (p-1)/(m+p-1)
      p=4 m=1   simulated idle = 0.750 | formula = 0.750
      p=4 m=4   simulated idle = 0.429 | formula = 0.429
      p=4 m=16  simulated idle = 0.158 | formula = 0.158
      p=4 m=64  simulated idle = 0.045 | formula = 0.045
      ####...
      .####..
      ..####.
      ...####
[G] expert parallelism: max|diff| = 0.0e+00 | tokens crossing devices: 19/24 (each way) | per-device load = [9, 1, 13, 1]  <- imbalance = stragglers
[H] training memory per GPU, 7B params, 8 GPUs (weights+grads+Adam, activations excluded):
      DDP (replicate all)           112.0 GB
      ZeRO-1 (shard optimizer)       38.5 GB
      ZeRO-2 (+ gradients)           26.2 GB
      ZeRO-3 / FSDP (+ weights)      14.0 GB
      sequence/context parallelism (concept): split the SEQUENCE across devices (ring attention) when one request's KV cache is too big.
[I] REAL offload: distilgpt2, device_map='auto', max_memory={'cpu': '220MiB'}, offload_folder=tmp


Some parameters are on the meta device because they were offloaded to the cpu and disk.
      cpu  : 5 module(s), e.g. ['transformer.wte', 'lm_head', 'transformer.wpe']
      disk : 6 module(s), e.g. ['transformer.h.1', 'transformer.h.2', 'transformer.h.3']
      logits max|diff| vs plain model = 0.00e+00  (offloading changes WHERE, not WHAT)
```

- Column splitting is exact (`0.0` difference); row splitting and the MLP differ at the 1e-14 level, which is only floating-point re-ordering of sums: the split computes the same function.
- The "wrong" Megatron variant is off by tens (not 1e-14): the choice of column-then-row is a correctness requirement, not a style.
- TP=8 communication is under 1 MB per token, yet over 100 GbE it is about 40 times slower than NVLink, and the 64 collectives each add their own latency: that is why TP is kept inside a node.
- The simulated pipeline idle fraction matches `(p-1)/(m+p-1)` to three decimals: with 4 stages you need about 16 micro-batches to get below 16% bubble.
- The expert load `[9, 1, 13, 1]` is a real (random) imbalance, which stalls the busiest device; and offloading placed 5 modules on CPU and 6 on disk with logits unchanged.

## Experiments (change ONE thing)

| Change | What you will see | Why |
|---|---|---|
| `N = 2` or `8` (heads and widths divide by N) | differences stay at ~1e-14 | the math holds for any split |
| `N = 3` | an error from `np.split` | 16 and 32 are not divisible by 3: shards must be equal |
| `pipeline_schedule(8, 8)` vs `(4, 8)` | larger bubble for deeper pipeline | bubble = (p-1)/(m+p-1) |
| `max_memory={"cpu": "300MiB"}` | fewer modules on disk | more room in the faster tier |
| `n_exp = 64`, `T2 = 24` | many experts get zero tokens | too few tokens per expert: a real decode batch problem |

## Debugging practice

**BREAK IT:** in section B sum all but one partial: `Y_row = np.sum(partials[:-1], axis=0)`.
- **Symptom:** `max|diff|` becomes order 1 to 10, not 1e-15. A real all-reduce that loses one rank's contribution gives silently wrong outputs.
- **Fix:** restore the sum over all `N` partials. In real code, assert that every rank participates (NCCL timeouts and hangs are the common failure instead).

**BREAK IT:** `N = 3`.
- **Symptom:** `ValueError: array split does not result in an equal division`.
- **Fix:** choose N that divides heads, hidden and MLP width (and `kv_heads`), as real TP degrees must.

## Common mistakes

- Thinking TP and DP are alternatives for the same thing: DP scales throughput, TP scales memory and per-token speed.
- Using TP across nodes over Ethernet: the 64 collectives per token dominate.
- Forgetting that PP has a bubble, and that a single decoding stream cannot fill a pipeline.
- Treating offloading as a performance feature: it is a capacity feature; speed falls to PCIe or disk bandwidth.
- Quoting "ZeRO" for inference: it is a training memory technique (inference sharding is TP/PP).
- Forgetting that GQA with `kv_heads < tp_size` forces KV-head replication, which wastes memory.

## Rebuild checklist

1. Make `X`, `W`; compute `Y_full = X @ W`.
2. Column split: `np.split(W, N, 1)`, per-shard matmul, `np.concatenate` on axis 1; compare.
3. Row split: split `W` on axis 0 and `X` on axis 1, per-shard matmul, `np.sum`; compare.
4. Megatron MLP: column-split `A`, row-split `B`, GELU per shard, one sum; compare; show the row-split-first variant is wrong.
5. Head-sharded attention: split `Wq/Wk/Wv` by columns and `Wo` by rows.
6. Write `tp_comm_bytes_per_token` with the ring factor and `pipeline_schedule(p, m)` with `busy[s, s+j]`.
7. Load a model with `device_map="auto"` and `max_memory`, print `hf_device_map`.

## Say it aloud

- "Tensor parallelism cuts each weight matrix: column-parallel outputs are concatenated, row-parallel outputs are summed with an all-reduce, and Megatron puts them back to back so an MLP block needs a single all-reduce."
- "TP is exact but chatty: two all-reduces per layer per token, so I keep it inside an NVLink node and use pipeline parallelism between nodes."
- "Pipeline bubble is (p-1)/(m+p-1): you need many micro-batches, or many concurrent requests when decoding."
- "Offloading with `device_map` or llama.cpp `-ngl` lets a model run when it does not fit, at the price of PCIe or disk bandwidth per token."
- "If the model fits on one GPU I prefer replicas: no communication and linear throughput scaling."

## The new problem (cliffhanger)

Now you know how to put a model on one GPU, or split it over several, and how every piece (quantization, KV cache, batching, paging, speculation, parallelism) is a different lever. But the levers live in different tools. Which engine pulls which levers, and which should you pick? On to [step 15 — serving engines landscape](../15_serving_engines_landscape/WALKTHROUGH.md).
