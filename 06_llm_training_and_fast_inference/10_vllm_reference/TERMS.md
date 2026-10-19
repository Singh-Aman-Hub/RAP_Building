# Terms — vLLM (Reference)

Folder: `06_llm_training_and_fast_inference/10_vllm_reference` · Files covered: `vllm_reference.py`, `WALKTHROUGH.md` (reference only, not executed; needs Linux and an NVIDIA GPU)
Back to module: [06 README](../README.md)

**Terms in this folder:** [vLLM](#vllm) · [PagedAttention](#pagedattention) · [KV cache waste (fragmentation)](#kv-cache-waste-fragmentation) · [Continuous batching](#continuous-batching) · [Static batching](#static-batching) · [LLM (vllm class)](#llm-vllm-class) · [SamplingParams](#samplingparams) · [gpu_memory_utilization](#gpu_memory_utilization) · [max_model_len](#max_model_len) · [dtype="float16" and quantization="awq"](#dtypefloat16-and-quantizationawq) · [llm.generate (batch of prompts)](#llmgenerate-batch-of-prompts) · [outputs[0].outputs[0].text](#outputs00text) · [vllm serve](#vllm-serve) · [OpenAI-compatible API](#openai-compatible-api) · [curl with -H and -d](#curl-with--h-and--d) · [Transformers generate vs vLLM](#transformers-generate-vs-vllm)

---

## vLLM  `(third-party: vllm)`
**1. Definition:** vLLM is a high-throughput LLM inference and serving engine that runs on Linux with NVIDIA GPUs, built around PagedAttention and continuous batching.
**2. In plain words:** It's what you use when many users hit one model at once. Its advantage is throughput, claimed in the file at 2 to 20 times higher than plain Transformers on the same GPU; single-request latency is about the same. It adds operational complexity and a GPU requirement, so for local experiments Transformers is still fine.
**3. Example to quote:** > "I'd serve many concurrent users with vLLM for throughput, but experiment locally with Transformers because I don't need a GPU server for that."
**Seen in:** `vllm_reference.py`, docstring

## PagedAttention  `(concept)`
**1. Definition:** PagedAttention stores the KV cache in small fixed-size blocks (pages) allocated on demand and not necessarily contiguous, like virtual-memory pages in an operating system.
**2. In plain words:** Instead of reserving a huge slab per request for its maximum possible length, memory is handed out in small pieces as needed. Almost no cache memory is wasted, so more requests fit on the GPU, and shared prefixes can be reused.
**3. Example to quote:** > "PagedAttention treats the KV cache like virtual memory pages, so a request only uses the memory it actually needs."
**Seen in:** `vllm_reference.py`; `WALKTHROUGH.md`

## KV cache waste (fragmentation)  `(concept)`
**1. Definition:** Naive serving reserves a contiguous region for each request sized to its maximum length, so unused reserved space and gaps (fragmentation) waste memory.
**2. In plain words:** Most requests are shorter than the maximum, so a big share of the reservation is never used and fewer requests fit.
**3. Example to quote:** > "Reserving the full 2048-token cache for every request wastes memory when most answers are short."
**Seen in:** `WALKTHROUGH.md`, "Problem it solves"

## Continuous batching  `(concept)`
**1. Definition:** Continuous batching lets finished requests leave the batch and waiting requests join at every decoding step.
**2. In plain words:** The GPU never sits idle waiting for the slowest request. A request that stops after 5 tokens frees its slot immediately.
**3. Example to quote:** > "With continuous batching, a 5-token answer doesn't hold its slot until a 500-token answer finishes."
**Seen in:** `vllm_reference.py`

## Static batching  `(concept)`
**1. Definition:** Static batching groups requests into a fixed batch that runs until all of its members have finished.
**2. In plain words:** It's what you get when you call `generate` on a list yourself: short requests wait for the longest one, leaving capacity unused.
**3. Example to quote:** > "In static batching, one long request holds back the whole group, which continuous batching fixes."
**Seen in:** `WALKTHROUGH.md`

## LLM (vllm class)  `(third-party: vllm)`
**1. Definition:** `vllm.LLM(model=..., ...)` loads a model into the vLLM engine and pre-allocates KV-cache pages.
**2. In plain words:** It's the entry point for offline batch inference, like `AutoModelForCausalLM` but with the serving engine built in.
**3. Example to quote:** > "`LLM(model='Qwen/Qwen2.5-0.5B-Instruct', dtype='float16')` loads the model and prepares memory for the cache."
**Seen in:** `vllm_reference.py`, section A

## SamplingParams  `(third-party: vllm)`
**1. Definition:** `SamplingParams(temperature, top_p, max_tokens)` bundles the generation settings; `max_tokens` is vLLM's name for `max_new_tokens`.
**2. In plain words:** These are the same sampling knobs from module 05, just packaged in an object.
**3. Example to quote:** > "I set `temperature=0.7, top_p=0.9, max_tokens=64`, the same knobs as in the Transformers `generate` call."
**Seen in:** `vllm_reference.py`

## gpu_memory_utilization  `(concept)`
**1. Definition:** `gpu_memory_utilization=0.90` is the fraction of GPU memory vLLM may take for weights plus KV-cache blocks.
**2. In plain words:** vLLM grabs that memory up front; what remains after the weights becomes the pool of cache pages, so a higher fraction allows more concurrent requests.
**3. Example to quote:** > "At 0.90, vLLM pre-allocates 90% of the GPU and uses what's left after the weights as KV-cache pages."
**Seen in:** `vllm_reference.py`

## max_model_len  `(concept)`
**1. Definition:** `max_model_len` is the longest prompt-plus-output length the engine will accept.
**2. In plain words:** It bounds the KV cache per sequence, so a lower value lets more requests fit.
**3. Example to quote:** > "I set `max_model_len=2048` so no single sequence can claim more than 2048 tokens of cache."
**Seen in:** `vllm_reference.py`

## dtype="float16" and quantization="awq"  `(concept)`
**1. Definition:** `dtype` sets the weight precision; `quantization="awq"` loads an AWQ-quantised checkpoint with smaller weights.
**2. In plain words:** Lower precision gives smaller weights and more room for KV cache, with a small quality trade-off (see module 05).
**3. Example to quote:** > "I load in float16, and with a quantised checkpoint I'd pass `quantization='awq'` to free memory for cache."
**Seen in:** `vllm_reference.py`

## llm.generate (batch of prompts)  `(third-party: vllm)`
**1. Definition:** `llm.generate(prompts, params)` takes a list of prompts and returns one result object per prompt, batching continuously inside the engine.
**2. In plain words:** You pass many prompts in one call and vLLM schedules them for you.
**3. Example to quote:** > "One `llm.generate(prompts, params)` call handles three prompts without me building a batch."
**Seen in:** `vllm_reference.py`

## outputs[0].outputs[0].text  `(syntax)`
**1. Definition:** The first `outputs[0]` is the first request's result object; its `.outputs[0].text` is its first completion's generated text, since a request can return several completions.
**2. In plain words:** Two levels of indexing: which prompt, then which answer.
**3. Example to quote:** > "`out.outputs[0].text` is the first generated answer for that prompt."
**Seen in:** `vllm_reference.py`; `WALKTHROUGH.md`

## vllm serve  `(command)`
**1. Definition:** `vllm serve <model> --port 8000` starts an HTTP server exposing the model, run in the shell and not from Python.
**2. In plain words:** It turns the model into a network service that other programs, such as the module 09 API or the module 08 agent, call over HTTP.
**3. Example to quote:** > "I start `vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000` and then call it like any hosted API."
```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000 --max-model-len 2048
```
**Seen in:** `vllm_reference.py`, section B

## OpenAI-compatible API  `(concept)`
**1. Definition:** The server implements the same HTTP request and response format as OpenAI's chat completions endpoint (`/v1/chat/completions`).
**2. In plain words:** Code written for hosted APIs can point at your own server by changing only the base URL, so the module 08 agent harness works with it unchanged.
**3. Example to quote:** > "Because it's OpenAI-compatible, my agent only needs a different base URL to use the local vLLM server."
**Seen in:** `vllm_reference.py`

## curl with -H and -d  `(command)`
**1. Definition:** `curl` is a command-line HTTP client; `-H` adds a header (here `Content-Type: application/json`) and `-d` sends the request body.
**2. In plain words:** It's the quick way to call a web API from the terminal, which you'll use a lot in module 09.
**3. Example to quote:** > "I test the server with `curl ... -H 'Content-Type: application/json' -d '{...}'` before writing any client code."
```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"...","messages":[{"role":"user","content":"Hello"}]}'
```
**Seen in:** `vllm_reference.py`, section B

## Transformers generate vs vLLM  `(concept)`
**1. Definition:** Transformers `generate` is simple and flexible, running one batch you build; vLLM is a production engine with paged cache and continuous batching.
**2. In plain words:** Use Transformers for learning and experiments, vLLM for serving many users. Limitation: vLLM needs a CUDA GPU server, and not every model or quantisation format is supported immediately.
**3. Example to quote:** > "For a demo on my laptop I use Transformers; for production traffic I'd choose vLLM and accept the extra operations work."
**Seen in:** `vllm_reference.py`, section C; `WALKTHROUGH.md`
