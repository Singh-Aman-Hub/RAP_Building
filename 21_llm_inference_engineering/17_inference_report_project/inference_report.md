# Inference report: distilgpt2 on CPU

Machine-dependent numbers; 4 torch threads; prompt 32 tokens, 16 new tokens, median of 3 trials; cost uses an EXAMPLE price of $0.1/hour (edit PRICE_PER_HOUR).

| dtype | quant | batch | kv | load_s | ttft_ms | itl_ms | tok_s | rss_mb | ppl | cost_per_mtok |
|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | none | 1 | on | 3.149 | 22.02 | 12.48 | 75.95 | 896 | 42.53 | 0.3657 |
| fp32 | none | 1 | off | 3.061 | 21.66 | 26.48 | 37.7 | 896 | 42.53 | 0.7367 |
| fp32 | none | 4 | on | 3.147 | 44.94 | 17.68 | 206.3 | 924.3 | 42.53 | 0.1347 |
| fp32 | none | 4 | off | 3.056 | 45.35 | 57.69 | 71.64 | 1412 | 42.53 | 0.3877 |
| fp32 | int8 | 1 | on | 3.242 | 20.64 | 11.66 | 81.59 | 1103 | 42.62 | 0.3405 |
| fp32 | int8 | 1 | off | 3.297 | 20.53 | 24.84 | 40.72 | 1104 | 42.62 | 0.6822 |
| fp32 | int8 | 4 | on | 3.236 | 42.28 | 16.78 | 217.8 | 1140 | 42.62 | 0.1275 |
| fp32 | int8 | 4 | off | 3.292 | 41.64 | 51.74 | 74.96 | 1606 | 42.62 | 0.3706 |
| bf16 | none | 1 | on | 3.175 | 907.1 | 31.63 | 11.57 | 1007 | 42.84 | 2.401 |

## Findings

- **KV cache:** at batch 1 the cache gives 2.0x the tokens/s of re-running the whole sequence each step.
- **Batching:** batch 4 gives 2.7x system throughput at 1.4x the per-step latency.
- **int8 weight-only (fake):** perplexity 42.62 vs 42.53 for fp32, so quality is preserved; speed and RSS are NOT improved here because the weights stay float (a real int8 kernel or GPU memory-bound decode would gain).
- **bf16 on CPU:** perplexity 42.84; speed vs fp32 depends on whether this CPU has native bf16 math (often slower on laptops).

## Recommendation

- **Throughput / cost pick (quality within 2% of fp32 perplexity 42.53):** fp32 / weights none / batch 4 / KV cache on -> 206 tok/s, $0.135 per 1M output tokens (example price).
- **Interactive / latency pick:** fp32 / weights none / batch 1 / KV cache on -> TTFT 22 ms, ITL 12.5 ms.
- **Memory add-on:** int8 weights cost only 0.2% perplexity here and would shrink weights 4x vs fp32 (2x vs fp16) with a real int8 kernel: worth it when memory, not compute, is the limit.
- Always keep the KV cache on; use the largest batch the latency target allows; quantize only after checking perplexity.

## Caveats

- Tiny model on CPU: decode is cheap here, so batching gains are smaller than on a GPU where decode is memory-bound.
- Fake quantization measures quality only. Real speedups need kernels (GPTQ/AWQ/FP8 in vLLM, GGUF in llama.cpp).
- Perplexity is measured on a few hundred tokens: differences below ~1% are noise.
