# Terms — Parallelism and offloading

Folder: `21_llm_inference_engineering/14_parallelism_and_offloading` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [21 README](../README.md)

**Terms in this folder:** [Tensor parallelism (TP)](#tensor-parallelism-tp) · [Column-parallel linear](#column-parallel-linear) · [Row-parallel linear](#row-parallel-linear) · [Megatron MLP pattern](#megatron-mlp-pattern) · [Attention head sharding](#attention-head-sharding) · [All-reduce / all-gather / all-to-all](#all-reduce--all-gather--all-to-all) · [Communication volume per token](#communication-volume-per-token) · [Pipeline parallelism (PP)](#pipeline-parallelism-pp) · [Pipeline bubble](#pipeline-bubble) · [Expert parallelism (EP)](#expert-parallelism-ep) · [Sequence / context parallelism](#sequence--context-parallelism) · [Data parallelism, ZeRO and FSDP](#data-parallelism-zero-and-fsdp) · [device_map and max_memory](#device_map-and-max_memory) · [Offloading](#offloading) · [-ngl (n-gpu-layers)](#-ngl-n-gpu-layers)

---

## Tensor parallelism (TP)  `(concept)`
**1. Definition:** Splitting each layer's weight matrices across devices so all devices compute every layer together, using all-gather/all-reduce after the matmuls.
**2. In plain words:** Every GPU does a slice of every layer. Exact, but chatty.
**3. Example to quote:** > "Tensor parallelism cuts each matrix across GPUs, so I keep it inside an NVLink node."
**Seen in:** `main.py` sections A-E

## Column-parallel linear  `(concept)`
**1. Definition:** Split W by output columns; each device computes its slice of Y = XW from the full X; outputs are concatenated (all-gather).
**2. In plain words:** Each GPU makes some of the output columns, no communication needed to compute.
**3. Example to quote:** > "Column-parallel outputs are concatenated, so no sum is needed."
**Seen in:** `main.py` section A

## Row-parallel linear  `(concept)`
**1. Definition:** Split W by input rows and X by columns; each device makes a full-width partial sum; an all-reduce adds them.
**2. In plain words:** Each GPU makes an incomplete full-size answer; add them.
**3. Example to quote:** > "Row-parallel needs an all-reduce to sum partial outputs."
**Seen in:** `main.py` section B

## Megatron MLP pattern  `(concept)`
**1. Definition:** Column-parallel first matrix, elementwise GELU on the local slice, row-parallel second matrix, giving one all-reduce per MLP block.
**2. In plain words:** Column then row, so the GPUs talk once instead of twice.
**3. Example to quote:** > "Because GELU is elementwise, column-then-row needs only one all-reduce."
**Seen in:** `main.py` section C

## Attention head sharding  `(concept)`
**1. Definition:** Giving each device whole attention heads (column-split Q, K, V; row-split output projection); each device caches K,V of only its heads.
**2. In plain words:** Heads are independent, so each GPU owns some heads.
**3. Example to quote:** > "Heads are independent, so TP shards them and divides the KV cache too."
**Seen in:** `main.py` section D

## All-reduce / all-gather / all-to-all  `(concept)`
**1. Definition:** Collectives: all-reduce sums partial tensors and returns the sum to all; all-gather concatenates slices on all; all-to-all sends different chunks to each peer.
**2. In plain words:** The three communication patterns behind parallelism.
**3. Example to quote:** > "TP uses all-reduce, MoE uses all-to-all."
**Seen in:** `main.py` sections A-G

## Communication volume per token  `(concept)`
**1. Definition:** Per-GPU bytes per token for TP = layers x 2 all-reduces x hidden x bytes x 2(N-1)/N.
**2. In plain words:** Small in size, but many sequential collectives per token.
**3. Example to quote:** > "It is under a megabyte per token but 64 latency-bound collectives, so TP needs NVLink."
**Seen in:** `main.py` section E

## Pipeline parallelism (PP)  `(concept)`
**1. Definition:** Placing consecutive layers on different devices; micro-batches stream through; only activations cross stage boundaries.
**2. In plain words:** An assembly line of GPUs.
**3. Example to quote:** > "PP sends one activation vector per boundary, so it suits slower inter-node links."
**Seen in:** `main.py` section F

## Pipeline bubble  `(concept)`
**1. Definition:** Idle fraction (p-1)/(m+p-1) of p stages with m micro-batches, caused by pipeline fill and drain.
**2. In plain words:** Dead time at the start and end.
**3. Example to quote:** > "With 4 stages I need about 16 micro-batches to get the bubble under 16%."
**Seen in:** `main.py` section F

## Expert parallelism (EP)  `(concept)`
**1. Definition:** Placing MoE experts on different devices and shipping tokens to them with two all-to-alls (dispatch and combine).
**2. In plain words:** Tokens travel to the GPU that owns their expert.
**3. Example to quote:** > "Expert parallelism trades memory for all-to-all traffic and suffers from load imbalance."
**Seen in:** `main.py` section G

## Sequence / context parallelism  `(concept)`
**1. Definition:** Splitting the sequence dimension across devices (for example ring attention) when one request's KV cache is too large.
**2. In plain words:** Cut a very long prompt across GPUs.
**3. Example to quote:** > "Context parallelism is for million-token contexts, not for ordinary serving."
**Seen in:** `WALKTHROUGH.md`, theory

## Data parallelism, ZeRO and FSDP  `(concept)`
**1. Definition:** Data parallel replicates the model; ZeRO-1/2/3 and FSDP shard optimizer state, gradients and weights across data-parallel ranks (training).
**2. In plain words:** Replicate for throughput; shard to save training memory.
**3. Example to quote:** > "ZeRO-3 divides the 16 bytes per parameter of training state by N."
**Seen in:** `main.py` section H

## device_map and max_memory  `(function)`
**1. Definition:** `from_pretrained(device_map="auto", max_memory=..., offload_folder=...)` makes Accelerate place modules across GPU, CPU and disk within per-device budgets; `model.hf_device_map` shows the result.
**2. In plain words:** A seating plan for layers given memory limits.
**3. Example to quote:** > "With max_memory capped, the last layers landed on disk and the logits were unchanged."
**Seen in:** `main.py` section I

## Offloading  `(concept)`
**1. Definition:** Keeping weights in CPU RAM or on disk and moving them to the accelerator per layer as needed.
**2. In plain words:** Capacity for speed: a way to run a model, not to serve it fast.
**3. Example to quote:** > "Offloading lets a model run that does not fit, but each token pays PCIe or disk bandwidth."
**Seen in:** `main.py` section I

## -ngl (n-gpu-layers)  `(syntax)`
**1. Definition:** llama.cpp option that puts N layers on the GPU and leaves the rest on CPU.
**2. In plain words:** The llama.cpp version of layer offloading.
**3. Example to quote:** > "With -ngl 20 the first 20 layers run on the GPU."
**Seen in:** `../15_serving_engines_landscape/llamacpp_reference.md`
