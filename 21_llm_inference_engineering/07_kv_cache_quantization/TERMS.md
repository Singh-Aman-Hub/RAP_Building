# Terms — KV-cache quantization

Folder: `21_llm_inference_engineering/07_kv_cache_quantization` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [21 README](../README.md)

**Terms in this folder:** [KV-cache quantization](#kv-cache-quantization) · [Per-token vs per-channel quantization of K and V](#per-token-vs-per-channel-quantization-of-k-and-v) · [KIVI](#kivi) · [FP8 KV cache (--kv-cache-dtype fp8)](#fp8-kv-cache---kv-cache-dtype-fp8) · [NVFP4 KV cache](#nvfp4-kv-cache) · [KL divergence of next-token logits](#kl-divergence-of-next-token-logits) · [Greedy continuation agreement](#greedy-continuation-agreement) · [DynamicCache](#dynamiccache) · [Per-channel SQNR](#per-channel-sqnr) · [kv_cache_bytes (imported from step 03)](#kv_cache_bytes-imported-from-step-03)

---

## KV-cache quantization  `(concept)`
**1. Definition:** Storing cached keys and values in fewer bits (int8, FP8, int4, int2) to shrink memory per token.
**2. In plain words:** Compress the notebook of past tokens.
**3. Example to quote:** > "int8 and FP8 gave almost no KL change on distilgpt2."
```python
kv_qdq(k, 'channel', 4)
```
**Seen in:** `main.py` section 3

## Per-token vs per-channel quantization of K and V  `(concept)`
**1. Definition:** Per-token: a scale per token vector; per-channel: a scale per head_dim position over tokens.
**2. In plain words:** Which numbers share a ruler in the cache.
**3. Example to quote:** > "K per-channel gave +4 dB SQNR over per-token at int4."
```python
a.transpose(0, 2, 1)
```
**Seen in:** `main.py` section 3

## KIVI  `(concept)`
**1. Definition:** Tuning-free asymmetric 2-bit KV quantization with K per-channel, V per-token and a small full-precision recent window.
**2. In plain words:** Match the grouping to how K and V behave.
**3. Example to quote:** > "The int4 and int2 (KIVI) rows use K per-channel, V per-token."
```python
(see rows in the trace)
```
**Seen in:** `main.py` section 4

## FP8 KV cache (--kv-cache-dtype fp8)  `(concept)`
**1. Definition:** vLLM option that stores K,V as FP8 E4M3 with scales, halving cache memory against fp16.
**2. In plain words:** One byte per cached number.
**3. Example to quote:** > "FP8 simulated in our table had KL about 0.0001."
```python
vllm serve <model> --kv-cache-dtype fp8
```
**Seen in:** `WALKTHROUGH.md` theory

## NVFP4 KV cache  `(concept)`
**1. Definition:** Emerging 4-bit block-scaled FP4 cache (16-value blocks, FP8 scales) reported in recent vLLM releases; check current status.
**2. In plain words:** A 4-bit cache on the newest hardware.
**3. Example to quote:** > "Treat as emerging, validate accuracy yourself."
```python
(none)
```
**Seen in:** `WALKTHROUGH.md` theory

## KL divergence of next-token logits  `(concept)`
**1. Definition:** KL(p_fp || p_q) = sum p (log p - log q) between baseline and perturbed next-token distributions.
**2. In plain words:** How different the two answers' probabilities are.
**3. Example to quote:** > "int2 per-token KL was 0.164 nats, KIVI-style 0.097."
```python
F.kl_div(lq, lp, log_target=True, reduction='sum')
```
**Seen in:** `main.py` section 4

## Greedy continuation agreement  `(concept)`
**1. Definition:** Fraction of generated token positions identical to the baseline continuation.
**2. In plain words:** Do the two models say the same sentence.
**3. Example to quote:** > "A brittle metric: one flip derails the rest."
```python
sum(a == b for a, b in zip(toks, base_toks))
```
**Seen in:** `main.py` section 4

## DynamicCache  `(class)`
**1. Definition:** Transformers' growable KV cache object; `.update(k, v, layer_idx)` appends and returns the concatenated tensors.
**2. In plain words:** The notebook that grows each step.
**3. Example to quote:** > "We rebuilt one from de-quantized tensors."
```python
c = DynamicCache(); c.update(k, v, 0)
```
**Seen in:** `main.py` section 1

## Per-channel SQNR  `(concept)`
**1. Definition:** Computing SQNR for each channel separately and averaging, so outlier channels do not dominate the metric.
**2. In plain words:** Judge every channel, not just the loudest.
**3. Example to quote:** > "K/chan was 17.9 dB for int4 per-token vs 22.5 dB per-channel."
```python
(x**2).sum(1) / (e**2).sum(1)
```
**Seen in:** `main.py` chan_sqnr

## kv_cache_bytes (imported from step 03)  `(function)`
**1. Definition:** The canonical memory formula 2 x layers x kv_heads x head_dim x tokens x batch x bytes, imported from ../03_gpu_memory_calculator/main.py.
**2. In plain words:** One source of truth.
**3. Example to quote:** > "It matched the measured 4,571,136 bytes exactly."
```python
gm.kv_cache_bytes(6, 12, 64, 124, 1, 4)
```
**Seen in:** `main.py` section 5
