# Step 10 — vLLM (concept + reference code, not executed)

## The story so far
[Step 09](../09_throughput_benchmark/WALKTHROUGH.md) showed throughput climbing with batch size until memory ran out, and [step 08](../08_gpu_memory_calculator/WALKTHROUGH.md) showed how large the KV cache is.

## The pain point
Naive batching has two problems. Each request reserves a contiguous KV-cache slab for its maximum possible length, so most of the memory is reserved and unused, and the free space splits into unusable holes. And a static batch waits for its slowest member while finished requests keep a slot.

## The spark
Borrow from the operating system. PagedAttention keeps the KV cache in small fixed pages with a block table, allocated on demand, so almost nothing is wasted. Continuous batching lets finished requests leave and waiting ones join at every step.

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

**See it:** [step 11](../11_paged_attention_fragmentation/WALKTHROUGH.md) simulates both allocators (contiguous reservation versus paged blocks with a block table) and prints how many requests fit in the same memory.

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

## Experiments (read-only, or on a rented GPU)
- Lower `max_model_len` from 2048 to 512 and predict whether more or fewer requests fit; then run [step 11](../11_paged_attention_fragmentation/WALKTHROUGH.md) with `MAX_LEN = 512` and check your prediction.
- Raise `gpu_memory_utilization` from 0.90 to 0.95: more KV pages, less headroom for spikes.

## Debugging practice
On this machine `import vllm` fails with `ModuleNotFoundError: No module named 'vllm'` (confirmed). vLLM needs Linux and an NVIDIA GPU. The corresponding real-world error is a CUDA out-of-memory at start-up when `gpu_memory_utilization` is too high for the card.

## The new problem (cliffhanger)
That is a claim in words, and you should not take memory arguments on faith. Build both allocators and watch the same GPU budget fit 8 requests or 53: [step 11, PagedAttention fragmentation](../11_paged_attention_fragmentation/WALKTHROUGH.md).

> **Say it aloud (interview box):** vLLM serves many users on one GPU because PagedAttention stores the KV cache in small pages instead of one large reservation per request, so little memory is wasted. Continuous batching swaps finished requests out and waiting requests in at every step so the GPU never idles on the slowest one. The cost is a Linux and CUDA setup, so I use plain Transformers for local experiments.
