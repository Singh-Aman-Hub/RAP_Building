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

# ---- [B] Row-parallel linear: split the WEIGHT's input rows (and X's columns); outputs are SUMMED (all-reduce) --------
# X @ W = sum_i X[:, block i] @ W[block i, :].  Device i needs only its slice of X (which a previous column-parallel
# layer produced for free) and holds W[block i, :].  Each makes a PARTIAL sum of the full-width output.
row_shards = np.split(W, N, axis=0)                         # (d_in/N, d_out)
x_shards = np.split(X, N, axis=1)                           # (T, d_in/N)
partials = [x @ w for x, w in zip(x_shards, row_shards)]    # each is (T, d_out): a partial sum, NOT a final answer
Y_row = np.sum(partials, axis=0)                            # "all-reduce(sum)": the one communication step
print(f"[B] row-parallel    (N={N})  max|diff| = {np.abs(Y_row - Y_full).max():.1e}  -> all-reduce (sum) of partial outputs == full layer")

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

# ---- [H] Data parallelism vs ZeRO / FSDP (training side; see module 06 and 16/06 for MoE) ---------------------------
# Mixed-precision Adam: 2 B weights + 2 B grads + 12 B optimizer state (fp32 master + m + v) = 16 B/param.
params, n = 7e9, 8
print(f"[H] training memory per GPU, 7B params, {n} GPUs (weights+grads+Adam, activations excluded):")
for name, w, g, o in (("DDP (replicate all)", 1, 1, 1), ("ZeRO-1 (shard optimizer)", 1, 1, 1 / n),
                      ("ZeRO-2 (+ gradients)", 1, 1 / n, 1 / n), ("ZeRO-3 / FSDP (+ weights)", 1 / n, 1 / n, 1 / n)):
    gb = params * (2 * w + 2 * g + 12 * o) / 1e9
    print(f"      {name:<28}{gb:7.1f} GB")
print("      sequence/context parallelism (concept): split the SEQUENCE across devices (ring attention) when one request's KV cache is too big.")

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

# TRY IT: change N to 2 or 8 in [A]-[D] (heads and widths must divide N); the diff stays ~1e-15.
# TRY IT: in [F] print p=8, m=8 and compare the bubble to p=4, m=8: why do deeper pipelines need more micro-batches?
# TRY IT: in [I] set max_memory={"cpu": "300MiB"} and watch fewer modules land on disk.
# BREAK IT: in [B] sum only N-1 of the partials (forget one device): the error is large, as with a failed all-reduce.
# BREAK IT: in [C] use gelu AFTER summing partials of a row-split first layer: it is wrong, which is why Megatron splits A by columns.
