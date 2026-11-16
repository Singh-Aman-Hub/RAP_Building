# Terms — Prefill vs decode and the roofline

Folder: `21_llm_inference_engineering/09_prefill_decode_roofline` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [21 README](../README.md)

**Terms in this folder:** [Prefill (prompt processing, context encoding)](#prefill-prompt-processing-context-encoding) · [Decode (token generation, autoregressive generation)](#decode-token-generation-autoregressive-generation) · [FLOPs per token](#flops-per-token) · [Arithmetic intensity (FLOP per byte)](#arithmetic-intensity-flop-per-byte) · [Roofline model](#roofline-model) · [Ridge point (machine balance)](#ridge-point-machine-balance) · [Compute-bound vs memory-bound](#compute-bound-vs-memory-bound) · [Why batching is almost free in decode](#why-batching-is-almost-free-in-decode) · [Chunked prefill (split-fuse)](#chunked-prefill-split-fuse) · [Disaggregated prefill/decode (PD split)](#disaggregated-prefilldecode-pd-split) · [Matmul GFLOP/s measurement](#matmul-gflops-measurement) · [Copy bandwidth measurement](#copy-bandwidth-measurement) · [Best of N timing and warm-up](#best-of-n-timing-and-warm-up) · [lambda for timing](#lambda-for-timing)

---

## Prefill (prompt processing, context encoding)  `(concept)`
**1. Definition:** One parallel forward pass over all prompt tokens that produces the first output token and fills the KV cache.
**2. In plain words:** Reading the whole question at once before answering.
**3. Example to quote:** > "Prefill is compute-bound and sets TTFT."
**Seen in:** `main.py`, `prefill_sweep`

## Decode (token generation, autoregressive generation)  `(concept)`
**1. Definition:** The loop that produces one token per forward pass, feeding the previous token and the KV cache.
**2. In plain words:** Writing the answer one word at a time, re-reading your notes each time.
**3. Example to quote:** > "Decode is memory-bound: it streams all weights for one token of math."
**Seen in:** `main.py`, `decode_sweep`

## FLOPs per token  `(concept)`
**1. Definition:** About 2 x number of parameters: each weight does one multiply and one add per token.
**2. In plain words:** Two operations per weight.
**3. Example to quote:** > "A 7B model needs about 14 GFLOP per token."
**Seen in:** `main.py`, `analyse`

## Arithmetic intensity (FLOP per byte)  `(concept)`
**1. Definition:** FLOPs performed divided by bytes moved from memory.
**2. In plain words:** How much math you do for each byte you fetch.
**3. Example to quote:** > "Decode intensity is about 2B over bytes per parameter."
**Seen in:** `main.py`, `analyse`

## Roofline model  `(concept)`
**1. Definition:** Attainable FLOP/s = min(peak FLOP/s, bandwidth x intensity).
**2. In plain words:** Two ceilings: a flat one (math) and a slanted one (memory).
**3. Example to quote:** > "The roofline tells me which roof a workload hits."
**Seen in:** `main.py`, `roofline_gflops`

## Ridge point (machine balance)  `(concept)`
**1. Definition:** Intensity at which the compute and bandwidth roofs meet: peak FLOP/s divided by bandwidth.
**2. In plain words:** The break-even number of FLOPs per byte.
**3. Example to quote:** > "Ridge is about 21 here, about 153 on A100 and 295 on H100."
**Seen in:** `main.py`, `RIDGE`

## Compute-bound vs memory-bound  `(concept)`
**1. Definition:** Compute-bound: time set by peak FLOP/s. Memory-bound: time set by bytes / bandwidth.
**2. In plain words:** Waiting for the calculator vs waiting for the delivery truck.
**3. Example to quote:** > "Decode at batch 1 is memory-bound; prefill is compute-bound."
**Seen in:** `main.py`, sweeps

## Why batching is almost free in decode  `(concept)`
**1. Definition:** Weights are streamed once per step regardless of batch size, and spare compute absorbs the extra tokens, so step time is flat until the ridge.
**2. In plain words:** One truck delivery serves 16 customers.
**3. Example to quote:** > "Step time grew 1.6x from batch 1 to 16 while tokens/s rose 10x."
**Seen in:** `main.py`, `decode_sweep`

## Chunked prefill (split-fuse)  `(concept)`
**1. Definition:** Splitting a long prompt into token-budget chunks and mixing them with decode tokens in each scheduler iteration.
**2. In plain words:** Do not let one long prompt block the road for everyone.
**3. Example to quote:** > "Chunking caps the decode stall at about the chunk time."
**Seen in:** `main.py`, section 6; step 12

## Disaggregated prefill/decode (PD split)  `(concept)`
**1. Definition:** Running prefill and decode on separate GPU pools with KV-cache transfer between them.
**2. In plain words:** A kitchen with a prep team and a cooking team.
**3. Example to quote:** > "Disaggregation tunes TTFT and ITL independently at the cost of KV transfer."
**Seen in:** `WALKTHROUGH.md`, theory

## Matmul GFLOP/s measurement  `(function)`
**1. Definition:** Time a large matrix product and divide `2*n^3` by seconds.
**2. In plain words:** How fast can this machine do math?
**3. Example to quote:** > "`measure_gflops` gives about 940 GFLOP/s here."
**Seen in:** `main.py`, `measure_gflops`

## Copy bandwidth measurement  `(function)`
**1. Definition:** Time a copy of a tensor larger than cache; bytes moved = read + write.
**2. In plain words:** How fast can this machine move bytes?
**3. Example to quote:** > "`measure_gbps` gives about 45 GB/s here."
**Seen in:** `main.py`, `measure_gbps`

## Best of N timing and warm-up  `(concept)`
**1. Definition:** Run once untimed, then take the minimum of several timed runs.
**2. In plain words:** Ignore the first slow attempt.
**3. Example to quote:** > "Without the warm-up call the first row looks 5-10x too slow."
**Seen in:** `main.py`, `best_of`

## lambda for timing  `(syntax)`
**1. Definition:** An anonymous one-expression function so a timer can call any operation.
**2. In plain words:** A tiny wrapper function.
**3. Example to quote:** > "`_time(lambda: a @ b)`."
**Seen in:** `main.py`, `measure_gflops`
