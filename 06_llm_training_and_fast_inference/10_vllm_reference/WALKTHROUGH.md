# Step 10 — vLLM (concept + reference code, not executed)

## Goal
Explain why vLLM serves many users faster than `transformers.generate`, and read its API. `vllm_reference.py` needs Linux + NVIDIA GPU; here it is compile-checked only.

## Problem it solves
Step 07/09 showed batching helps. But naive batching has two memory/scheduling problems:
1. **KV cache waste:** each request reserves a contiguous slab for its maximum possible length; most of it is never used (fragmentation), so fewer requests fit.
2. **Static batches wait for the slowest request:** a request that stops after 5 tokens still occupies its slot until the 500-token request finishes.

## The two fixes
```
PagedAttention                          Continuous batching
KV cache split into fixed pages         every decoding step, finished requests leave
(like OS virtual memory) allocated      and waiting requests join immediately
on demand, non-contiguous               -> GPU never idles waiting for the slowest one
-> ~no wasted cache, more requests
   fit, shared prefixes can be reused
```

## Using it (see the file)
- `LLM(model=..., dtype=..., gpu_memory_utilization=0.9, max_model_len=2048)` loads the model and pre-allocates KV pages.
- `SamplingParams(temperature, top_p, max_tokens)` = the sampling knobs from module 05.
- `llm.generate(list_of_prompts, params)` batches internally; you just pass many prompts.
- `vllm serve <model>` starts an OpenAI-compatible HTTP server: module 08's agent harness or module 09's API can call it via plain HTTP.

## Syntax notes
| Syntax | Meaning |
|---|---|
| `outputs[0].outputs[0].text` | the first request's first completion text (a request can return several) |
| shell `vllm serve ...` | run a command, not Python |
| `curl -d '{...}' -H "Content-Type: application/json"` | call an HTTP API from the terminal (module 09) |

## Compare
| | Transformers `generate` | vLLM |
|---|---|---|
| Best for | learning, experiments, small scale | production serving, many users |
| Batching | you build the batch | continuous, automatic |
| KV cache | contiguous, wasteful | paged |
| Platform | CPU / MPS / CUDA | Linux + CUDA (mainly) |

## Limitations / trade-offs
- Needs a GPU server, adds operational complexity.
- Great at throughput; single-request latency is similar to Transformers.
- Not every model/quantisation format is supported immediately.

## Say it aloud
"vLLM is an inference engine. PagedAttention stores the KV cache in small pages like virtual memory, so little GPU memory is wasted and more concurrent requests fit. Continuous batching swaps finished requests out and new ones in at every step so the GPU stays busy. Result: much higher throughput than plain Transformers for many users, exposed through an OpenAI-compatible server. Trade-off: it needs a CUDA GPU and more ops work, so for local experiments I still use Transformers."
