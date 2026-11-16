# Step 15 — Serving engines landscape: which tool pulls which levers

> This step moved here from module 06 (old step 10, `vllm_reference`) in the Nov 2026 restructure and was extended from "vLLM only" to the whole landscape.

## The story so far
[Step 14](../14_parallelism_and_offloading/WALKTHROUGH.md) showed how to split a model over devices or spill it to RAM and disk. Before that, steps 04 to 13 gave you the individual levers: quantization ([04](../04_quantization_math/WALKTHROUGH.md)), KV cache ([08](../08_hf_inference_batching_kvcache/WALKTHROUGH.md)), paged memory ([11](../11_paged_attention_fragmentation/WALKTHROUGH.md)), batching and prefix caching (step 12), speculation (step 13). Nobody pulls these levers by hand: an *engine* does.

## The pain point
There are a dozen engines and every blog post says a different one is fastest. Ollama, llama.cpp, vLLM, SGLang, TensorRT-LLM, TGI, MLX, `transformers`: if you cannot say what each does *inside* (scheduler, KV management, kernels, quantization formats, hardware) you cannot choose, and you cannot answer the interview question "why is vLLM faster than `generate()`?" or "when would you not use vLLM?".

## The spark
Compare engines on the same five axes: **scheduler** (how requests are batched), **KV-cache management** (contiguous, paged, radix tree), **kernels** (hand-written, FlashAttention/FlashInfer, NVIDIA fused), **quantization formats** it can load, **hardware** it runs on. Add API compatibility (almost everyone offers an OpenAI-compatible endpoint) and the decision becomes a small, explainable rule set, which you can write as code and test.

## Goal
Know the internals of each engine well enough to pick one for a stated scenario, run a decision helper (`main.py`, pure Python), read the vLLM, SGLang and Ollama reference scripts and the llama.cpp / TensorRT-LLM notes, and know which of them are only references here.

## The theory (read this before the interview)

**What a serving engine adds over `model.generate()`.**
1. A **scheduler** that continuously batches many users' requests (step 12) and respects a token budget per step, with chunked prefill and preemption when memory runs out.
2. A **KV-cache manager**: paged blocks (step 11), prefix reuse (step 12), optionally quantized KV (step 07).
3. **Optimized kernels**: FlashAttention-style fused attention (16/05), CUDA graphs to remove launch overhead, fused MoE kernels, quantized GEMMs.
4. **Model loading and quantization support** (steps 01 to 06).
5. **A server**: HTTP API (OpenAI-compatible), streaming, metrics, parallelism (step 14).

**The landscape (snapshot of architecture, Nov 2026; versions move fast, check each project's release page).**

| Engine | What it is | Scheduler | KV management | Kernels | Quant formats | Hardware | OpenAI-compatible API | Use it when |
|---|---|---|---|---|---|---|---|---|
| **HF Transformers** | reference library | none by default (one static batch); newer releases add experimental continuous batching and `transformers serve` | DynamicCache / StaticCache | PyTorch + SDPA, optional FlashAttention-2 | bitsandbytes, GPTQ, AWQ, FP8, torchao, GGUF (dequantised) | CPU, CUDA, MPS, ROCm, XPU | via `transformers serve` (emerging) | prototyping, research, fine-tuning, new architectures |
| **llama.cpp** | C/C++ on ggml | `llama-server` slots, continuous batching | one KV buffer shared by slots, no paging; quantizable K/V | hand-written ggml (AVX/NEON, Metal, CUDA, HIP, Vulkan) | GGUF: k-quants, i-quants, Q8_0, F16 | CPU, Apple, CUDA, ROCm, Vulkan | yes (`llama-server`) | laptops, edge, CPU/GPU split with `-ngl`, few users |
| **Ollama** | daemon around llama.cpp/ggml + registry | backend's; keeps model loaded for `keep_alive`; limited parallelism | backend's | ggml | GGUF | CPU, Apple, CUDA, ROCm | yes (`:11434`, `/v1`) | easiest local developer experience |
| **vLLM** | GPU serving engine (V1 architecture) | continuous batching, chunked prefill, token budget, preemption | PagedAttention + automatic prefix caching, fp8 KV | FlashAttention/FlashInfer, CUDA graphs, fused MoE | FP8, AWQ, GPTQ, compressed-tensors (llm-compressor), bitsandbytes, NVFP4/MXFP4 (emerging) | NVIDIA, AMD, TPU, limited CPU, plugins | yes | default for many-user GPU serving |
| **SGLang** | GPU engine + structured-generation frontend | continuous batching, CPU-overlapped ("zero-overhead") scheduler, PD disaggregation | paged + **RadixAttention** (radix tree of prefixes) | FlashInfer, CUDA graphs | FP8, AWQ, GPTQ, compressed-tensors | NVIDIA, AMD | yes | heavy prefix sharing (agents, RAG templates), JSON/grammar outputs |
| **TensorRT-LLM** | NVIDIA's optimized stack | in-flight batching, chunked context, disaggregated serving | paged, KV reuse, fp8/int8 KV | NVIDIA-tuned fused kernels | FP8, NVFP4, INT4-AWQ, SmoothQuant | NVIDIA only | yes (`trtllm-serve`) | max tokens/s per NVIDIA GPU, big stable deployments |
| **TGI** | Hugging Face server (Rust + Python) | continuous batching | paged | FlashAttention | bitsandbytes, GPTQ, AWQ, FP8 | NVIDIA, ROCm, others | yes | **maintenance mode: existing deployments only; HF recommends vLLM/SGLang** |
| **MLX / LM Studio** | Apple array framework / desktop app | single-user oriented | MLX cache | Metal, unified memory | MLX 4/8-bit, GGUF (LM Studio) | Apple silicon | yes (local server) | Mac laptops, GUI users |

**Engine internals in one paragraph each.**
- *llama.cpp*: models are GGUF files ([step 02](../02_gguf_and_llamacpp_formats/WALKTHROUGH.md)); a graph of ggml ops runs on the chosen backend; `-ngl N` puts N layers on the GPU, the rest on CPU; `mmap` makes loading near instant. See `llamacpp_reference.md`.
- *Ollama*: a Go daemon that downloads GGUF layers from a registry, launches a llama.cpp (or its own) runner process, and exposes REST. Convenience over control. See `ollama_reference.py`.
- *vLLM*: the engine core loops: the **scheduler** picks which running and waiting sequences go in the next step within a token budget; the **KV cache manager** allocates fixed-size blocks and hashes full blocks for prefix reuse; the **model runner** builds a flat batch of tokens, runs attention kernels reading K and V through the block table, samples. `vllm_reference.py`.
- *SGLang*: same skeleton but the cache index is a radix tree; the CPU scheduler runs ahead while the GPU executes (overlap), and a grammar backend masks logits for structured output. `sglang_reference.py`.
- *TensorRT-LLM*: replaces generic kernels with NVIDIA's fused ones and best-in-class FP8/FP4 paths; trades flexibility for peak speed. `trtllm_reference.md`.

**Which levers does each pull?**

| Lever | transformers | llama.cpp / Ollama | vLLM | SGLang | TensorRT-LLM |
|---|---|---|---|---|---|
| Continuous batching | experimental | yes (slots) | yes | yes | yes |
| Paged KV cache | no | no | **yes** | yes | yes |
| Prefix caching | manual | per-slot prompt cache | automatic (block hash) | **RadixAttention** | KV reuse |
| Quantized weights | many | GGUF (best CPU story) | many (FP8/AWQ/GPTQ) | many | FP8/NVFP4/AWQ |
| Quantized KV | partial | yes | fp8 (+ emerging fp4) | fp8 | fp8/int8 |
| Speculative decoding | assisted generation | supported | yes (draft, n-gram, EAGLE, MTP) | yes | yes |
| Tensor/pipeline parallel | via accelerate | layer split | TP/PP/EP | TP/EP | TP/PP/EP |
| Structured output | via libraries | GBNF grammars | yes (xgrammar etc.) | **strong** | yes |

**Decision rules (also encoded in `main.py`).**
1. New architecture on day one? `transformers`.
2. Mac or CPU or a small GPU with few users? llama.cpp family (Ollama for convenience), MLX on a Mac.
3. Many users on GPUs? vLLM by default; SGLang if prefixes are shared or structured output is central.
4. NVIDIA-only fleet and every token per dollar matters? TensorRT-LLM (or vLLM with FP8) after benchmarking.
5. Do not start new work on TGI.

**Benchmark your own traffic.** Throughput leadership flips with model, batch, prompt length and version. Use the same load generator against each engine (`vllm bench serve`, GenAI-Perf, guidellm, or the loop from [step 10](../10_latency_throughput_benchmark/WALKTHROUGH.md)) and compare TTFT, ITL and goodput, then price it in [step 16](../16_cost_per_token/WALKTHROUGH.md).

**Fundamentals vs current practice vs emerging vs outdated.**

| Class | Items |
|---|---|
| Fundamentals | continuous batching, KV cache, memory-bound decode, quantization, OpenAI-compatible API as the lingua franca |
| Current practice (2026) | vLLM/SGLang on GPUs, llama.cpp/Ollama locally, FP8 on Hopper, prefix caching, chunked prefill |
| Emerging | disaggregated prefill/decode, NVFP4/MXFP4 KV and weights, `transformers serve`, MTP/EAGLE-3 speculation |
| Outdated / maintenance | TGI for new projects, static batching servers, hand-rolled FastAPI around `generate` for production traffic |

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Continuous batching | in-flight batching (TensorRT-LLM), iteration-level scheduling, dynamic batching (loosely) | Mostly | Finished sequences leave and new ones join at every decode step. |
| PagedAttention | paged KV cache, block table | Yes | KV stored in fixed-size blocks mapped by a table. |
| RadixAttention | radix-tree prefix cache (SGLang) | Different mechanism, same goal as vLLM's block-hash prefix caching | Shared prefixes are found by walking a token tree, at any length, not only whole blocks. |
| Prefix caching | automatic prefix caching (APC), KV reuse, prompt cache | Yes | Reuse KV of an identical prompt prefix. |
| Chunked prefill | chunked context (TensorRT-LLM), split-fuse | Yes | Long prompts are processed in slices mixed with decode steps. |
| OpenAI-compatible server | `/v1/chat/completions` endpoint, "drop-in" API | Yes | Same JSON shape as the OpenAI API; change only `base_url`. |
| `-ngl` | `--n-gpu-layers`, `device_map`, layer offloading | Same idea | How many layers run on GPU. |
| Engine / runtime / server | inference stack | Loosely | "Engine" runs the model; "server" exposes the API; most projects are both. |

## Run it

```
cd 21_llm_inference_engineering/15_serving_engines_landscape
python main.py                                                   # comparison + 6 scenarios (stdlib only, ~0.1 s)
python main.py --hardware nvidia-datacenter --users 50 --structured    # ask your own question
python ollama_reference.py                                       # prints SKIP if no Ollama server is running
HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/15_serving_engines_landscape/main.py    # from repo root
```

`vllm_reference.py`, `sglang_reference.py`, `llamacpp_reference.md`, `trtllm_reference.md` are **references: not run in this course** (they need CUDA or binaries that are not installed). Only the first two are Python files, and they are compile-checked (`python -m py_compile`), not executed.

## Code, step by step

### 1. The knowledge base: one row per engine  (`main.py` lines 19–112, after the header and the `import argparse` line)

```python
# ---- [S1] The knowledge base: one row per engine --------------------------------------------------------------------
# Every field is a plain string/list so the table is easy to print and to edit when the landscape changes.
ENGINES = {
    "transformers": dict(
        kind="library (reference implementation)",
        scheduler="none by default: you call generate() on one static batch; newer releases add experimental continuous batching + `transformers serve`",
        kv="DynamicCache (grows by concatenation), StaticCache (pre-allocated, torch.compile friendly); no paging by default",
        kernels="PyTorch ops + SDPA; optional FlashAttention-2 / hub kernels",
        quant=["bitsandbytes int8/nf4", "GPTQ", "AWQ", "FP8", "torchao", "GGUF (dequantised on load)"],
        hardware=["cpu", "cuda", "mps", "rocm", "xpu"],
        openai_api="via `transformers serve` (emerging); otherwise wrap it yourself",
        best="research, prototyping, fine-tuning, any new architecture on day one",
        caveat="low throughput under concurrency; wasteful KV memory"),
    "llama.cpp": dict(
        kind="C/C++ engine on the ggml tensor library; `llama-cli`, `llama-server`",
        scheduler="llama-server: N parallel 'slots', continuous batching, prompt cache per slot",
        kv="one pre-allocated KV buffer divided among slots (contiguous cells, no paging); K/V can be quantised (--cache-type-k q8_0)",
        kernels="hand-written ggml kernels: AVX2/AVX-512/NEON on CPU, Metal, CUDA, HIP, Vulkan, SYCL",
        quant=["GGUF k-quants (Q4_K_M ...)", "i-quants (IQ4_XS ...)", "Q8_0", "F16/BF16"],
        hardware=["cpu", "apple-silicon", "cuda", "rocm", "vulkan"],
        openai_api="yes: llama-server exposes /v1/chat/completions",
        best="laptops, edge, CPU+GPU split (-ngl N layers on GPU), single or few users, GGUF models",
        caveat="not built for hundreds of concurrent users; fewer batching/scheduling features than vLLM"),
    "ollama": dict(
        kind="app/daemon wrapping llama.cpp/ggml (plus its own engine for some models) + model registry",
        scheduler="inherits the backend; keeps a loaded model in memory for keep_alive (default minutes), then unloads; limited parallel requests",
        kv="backend's (ggml) KV buffer; context size set by num_ctx",
        kernels="ggml (same as llama.cpp)",
        quant=["GGUF (default tags are usually 4-bit k-quants)"],
        hardware=["cpu", "apple-silicon", "cuda", "rocm"],
        openai_api="yes: REST on :11434 (/api/chat, /api/generate) and an OpenAI-compatible /v1",
        best="the easiest local developer experience: `ollama run`, Modelfile, one-command model pulls",
        caveat="convenience over control; not for high-concurrency production"),
    "vllm": dict(
        kind="Python/CUDA serving engine (V1 architecture), `vllm serve`, `LLM` class",
        scheduler="continuous (iteration-level) batching with chunked prefill; tokens-per-step budget; preemption when KV is full",
        kv="PagedAttention: fixed-size blocks + block table; automatic prefix caching by block hashing; fp8 KV cache option",
        kernels="FlashAttention / FlashInfer backends, CUDA graphs, torch.compile, fused MoE kernels",
        quant=["FP8", "AWQ", "GPTQ", "compressed-tensors (llm-compressor)", "bitsandbytes", "NVFP4/MXFP4 (emerging)", "GGUF (experimental)"],
        hardware=["nvidia", "rocm", "tpu", "cpu (limited)", "other accelerators via plugins"],
        openai_api="yes: OpenAI-compatible server (chat, completions, embeddings)",
        best="general-purpose production serving with many concurrent users on GPUs",
        caveat="Linux + accelerator; heavy install; fast-moving releases"),
    "sglang": dict(
        kind="Python/CUDA serving engine + structured-generation frontend",
        scheduler="continuous batching with a CPU-overlapped ('zero-overhead') scheduler; prefill/decode disaggregation support",
        kv="paged KV + RadixAttention: a radix tree over token prefixes so any shared prefix (system prompt, few-shot, chat history, agent loops) is reused",
        kernels="FlashInfer-based attention, CUDA graphs, fused MoE",
        quant=["FP8", "AWQ", "GPTQ", "compressed-tensors", "FP4 (emerging)"],
        hardware=["nvidia", "rocm"],
        openai_api="yes: OpenAI-compatible server + native /generate",
        best="workloads with heavy prefix sharing (agents, RAG templates, multi-turn), structured outputs (JSON/regex grammars)",
        caveat="smaller ecosystem than vLLM; GPU only"),
    "tensorrt-llm": dict(
        kind="NVIDIA library (PyTorch-based workflow is now default; older compiled-engine workflow exists), `trtllm-serve`",
        scheduler="in-flight batching, chunked context, disaggregated serving (with NVIDIA Dynamo / Triton)",
        kv="paged KV cache, KV reuse, fp8 / int8 KV",
        kernels="NVIDIA-tuned fused attention/GEMM kernels, CUTLASS, plugins; best FP8/NVFP4 support on Hopper/Blackwell",
        quant=["FP8", "NVFP4", "INT4-AWQ", "INT8 SmoothQuant", "GPTQ"],
        hardware=["nvidia only"],
        openai_api="yes: trtllm-serve (OpenAI-compatible); Triton backend for production fleets",
        best="maximum tokens/s per NVIDIA GPU when you can afford engineering effort and stay on NVIDIA",
        caveat="NVIDIA lock-in; build/compile complexity; slower to support brand-new architectures"),
    "tgi": dict(
        kind="Hugging Face Text Generation Inference (Rust router + Python/CUDA)",
        scheduler="continuous batching, token streaming",
        kv="paged attention (flash-decoding style)",
        kernels="FlashAttention, custom CUDA",
        quant=["bitsandbytes", "GPTQ", "AWQ", "FP8"],
        hardware=["nvidia", "rocm", "gaudi", "others"],
        openai_api="yes: Messages API",
        best="existing deployments only: TGI is in MAINTENANCE MODE; Hugging Face recommends vLLM or SGLang as backends",
        caveat="do not start new projects on it"),
    "mlx/lm-studio": dict(
        kind="MLX = Apple's array framework (mlx-lm runs LLMs); LM Studio = desktop GUI app using llama.cpp and MLX backends",
        scheduler="single-user oriented (MLX); LM Studio runs a local server",
        kv="MLX KV cache (optionally quantised / rotating); llama.cpp's in LM Studio",
        kernels="Metal kernels, unified memory (no CPU<->GPU copy)",
        quant=["MLX 4/8-bit group quant", "GGUF (via llama.cpp backend)"],
        hardware=["apple-silicon"],
        openai_api="mlx_lm.server and LM Studio both offer a local OpenAI-compatible server",
        best="Mac laptops and desktops, fast local experiments, GUI users",
        caveat="Apple-only, not a multi-tenant server"),
}


def show_table():
    print("ENGINE COMPARISON (internals)")
    for name, e in ENGINES.items():
        print(f"\n== {name} == {e['kind']}")
        for key in ("scheduler", "kv", "kernels", "openai_api", "best", "caveat"):
            print(f"  {key:<10}: {e[key]}")
        print(f"  {'quant':<10}: {', '.join(e['quant'])}")
        print(f"  {'hardware':<10}: {', '.join(e['hardware'])}")
```

- **In simple words:** A Python dictionary with one entry per engine, holding the table above as data, so it can be printed and queried.
- **Exactly:** `ENGINES` is `dict[str, dict]`. Each inner dict has string fields (`kind`, `scheduler`, `kv`, `kernels`, `openai_api`, `best`, `caveat`) and two lists (`quant`, `hardware`). Nothing is fetched or measured: it is a hand-maintained snapshot.
- **Why it is here:** Writing the comparison as data forces you to fill every cell for every engine, which is the actual study work. `show_table()` below prints it.

The helper `show_table()` loops over the dict and prints each field (same file, same section).

### 2. The decision helper  (`main.py` lines 115–154)

```python
# ---- [S2] The decision helper -----------------------------------------------------------------------------------
def recommend(hardware: str, users: int, shared_prefix: bool = False, structured: bool = False,
              gguf: bool = False, max_perf_nvidia: bool = False, new_architecture: bool = False):
    """Return (engine, reason). Rules are ordered; the first match wins. This is a heuristic, not a law:
    always benchmark your OWN model and traffic (step 10 and 17)."""
    if new_architecture:
        return "transformers", "a brand-new architecture usually lands in Transformers first; serve it there until an engine adds support"
    gpu_server = hardware in ("nvidia-datacenter", "amd-datacenter", "nvidia-consumer")
    if hardware == "apple-silicon":
        if users <= 2:
            return "mlx/lm-studio" if not gguf else "ollama", "unified-memory Mac, one or two users: MLX is the Mac-native choice, Ollama/llama.cpp if you have GGUF"
        return "llama.cpp", "Mac with several users: llama-server slots + continuous batching"
    if hardware == "cpu" or (hardware == "nvidia-consumer" and users <= 2):
        return ("ollama" if users <= 2 else "llama.cpp"), "CPU or small GPU, few users: GGUF + llama.cpp family (-ngl splits layers CPU/GPU)"
    if gpu_server:
        if max_perf_nvidia and hardware == "nvidia-datacenter":
            return "tensorrt-llm", "NVIDIA-only, throughput-per-dollar matters more than flexibility: TensorRT-LLM FP8/NVFP4"
        if shared_prefix or structured:
            return "sglang", "RadixAttention reuses shared prefixes; strong grammar-constrained decoding"
        return "vllm", "default general-purpose GPU server: PagedAttention + continuous batching + prefix caching"
    return "transformers", "unknown setup: start with the reference implementation and measure"


def run_scenarios():
    cases = [
        ("laptop CPU, 1 user, GGUF model", dict(hardware="cpu", users=1, gguf=True), "ollama"),
        ("MacBook, 1 user, MLX model", dict(hardware="apple-silicon", users=1), "mlx/lm-studio"),
        ("1x24GB consumer GPU, 20 users", dict(hardware="nvidia-consumer", users=20), "vllm"),
        ("8xH100 node, 500 users, shared system prompt", dict(hardware="nvidia-datacenter", users=500, shared_prefix=True), "sglang"),
        ("NVIDIA fleet, squeeze every token/$", dict(hardware="nvidia-datacenter", users=2000, max_perf_nvidia=True), "tensorrt-llm"),
        ("day-one release of a new architecture", dict(hardware="nvidia-datacenter", users=1, new_architecture=True), "transformers"),
    ]
    print("\nSCENARIOS")
    ok = 0
    for label, kw, expect in cases:
        eng, why = recommend(**kw)
        ok += eng == expect
        print(f"  {label:<48} -> {eng:<14} ({why})")
    print(f"\nCHECKS passed: {ok}/{len(cases)}")
    assert ok == len(cases)
```

- **In simple words:** `recommend()` is a list of if-rules, tried in order, that returns an engine and a reason. `run_scenarios()` checks it on six cases and asserts the expected answers.
- **Exactly:** keyword arguments describe the situation (`hardware`, `users`, flags). The first matching rule returns `(engine, reason)`. `assert ok == len(cases)` makes the script fail loudly if someone edits a rule and breaks a scenario, a tiny regression test.
- **Why it is here:** Rules you can run are rules you can argue about. The order matters: a CPU case must be tested before the "GPU server" case.

### 3. Command-line entry  (`main.py` lines 157–172)

```python
# ---- [S3] Command line: ask your own question --------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hardware", choices=["cpu", "apple-silicon", "nvidia-consumer", "nvidia-datacenter", "amd-datacenter"])
    ap.add_argument("--users", type=int, default=1, help="expected concurrent users")
    ap.add_argument("--shared-prefix", action="store_true")
    ap.add_argument("--structured", action="store_true")
    ap.add_argument("--gguf", action="store_true")
    ap.add_argument("--max-perf-nvidia", action="store_true")
    a = ap.parse_args()
    if a.hardware:                                            # custom question: just answer it
        eng, why = recommend(a.hardware, a.users, a.shared_prefix, a.structured, a.gguf, a.max_perf_nvidia)
        print(f"-> {eng}: {why}")
    else:
        show_table()
        run_scenarios()
```

- **In simple words:** With no arguments print the table and the scenarios; with `--hardware ...` answer one question.
- **Exactly:** `argparse` builds the options; `choices=[...]` rejects unknown hardware with an error; `store_true` flags default to `False`. The `if __name__ == "__main__":` guard means importing the file does not run it.
- **Why it is here:** Lets you play with the rules in 10 seconds.

### 4. Reference: the vLLM header  (`vllm_reference.py` lines 1–19)

```python
"""
REFERENCE ONLY -- NOT EXECUTED IN THIS STUDY PACK.

vLLM is a high-throughput LLM inference engine.  It runs on Linux + NVIDIA GPU (CUDA); there is no
Mac/CPU path worth using, so read this file, then run it on a rented GPU or Colab if you want to see it live.

    pip install vllm
    # Version note (Nov 2026, medium confidence from third-party release summaries): vLLM was around v0.33
    # (CUDA-13-only wheels). The V1 engine is the default. Flag names below are stable concepts; if one errors,
    # run `vllm serve --help` for your installed version.

Two ideas make vLLM fast (see README.md in this module for the pictures):

1. PagedAttention  - KV cache is stored in small fixed-size blocks (like OS virtual-memory pages), so
                     no GPU memory is wasted on "reserved but unused" cache space.  More concurrent
                     requests fit in the same GPU memory.
2. Continuous batching - as soon as ONE request in the batch finishes, a waiting request takes its slot
                     immediately.  Naive static batching waits for the slowest request in the batch.
"""
```

- **In simple words:** A note that this file is read-only, plus the two ideas that make vLLM fast.
- **Exactly:** needs Linux and an NVIDIA GPU (other accelerators via plugins). The file carries a version note: around v0.33 in Nov 2026 per third-party release summaries (medium confidence); run `vllm serve --help` for your installed version.
- **Why it is here:** The ideas first, the API second.

### 5. Reference: offline batch inference  (`vllm_reference.py` lines 21–43)

```python
# ---------- A) Offline batch inference (python API) --------------------------------------------------------
from vllm import LLM, SamplingParams

# SamplingParams = the same knobs you learned in module 05, just bundled in an object.
params = SamplingParams(
    temperature=0.7,     # module 05: randomness
    top_p=0.9,           # module 05: nucleus sampling
    max_tokens=64,       # vLLM's name for max_new_tokens
)

llm = LLM(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    dtype="float16",                 # weights precision; quantised checkpoints also work (quantization="awq")
    gpu_memory_utilization=0.90,     # fraction of GPU memory vLLM may grab: weights + KV-cache blocks
    max_model_len=2048,              # longest prompt+output it will accept (bounds KV cache per sequence)
    enable_prefix_caching=True,      # reuse KV blocks of shared prompt prefixes (hash of each block); step 12
    kv_cache_dtype="fp8",            # fp8 (e4m3) KV cache: ~half the cache memory; step 07. "auto" = model dtype
)

prompts = ["Explain what a KV cache is.", "Write a haiku about GPUs.", "What is LoRA?"]
outputs = llm.generate(prompts, params)      # ONE call: vLLM batches all prompts for you, continuously
for out in outputs:
    print(out.prompt, "->", out.outputs[0].text)
```

- **In simple words:** Build sampling settings, create an `LLM`, and call `generate` with a list of prompts: vLLM batches them for you.
- **Exactly:** `SamplingParams` bundles `temperature`, `top_p`, `max_tokens` (vLLM's name for `max_new_tokens`). `LLM(...)` loads the weights, then **profiles memory and pre-allocates KV-cache blocks** up to `gpu_memory_utilization` of the GPU. `enable_prefix_caching` turns on block-hash prefix reuse; `kv_cache_dtype="fp8"` stores K and V in fp8 (e4m3). `generate` returns a list of `RequestOutput`; text is at `out.outputs[0].text`.
- **Why it is here:** It is the API you would show in an interview.

### 6. Reference: serving and flags  (`vllm_reference.py` lines 45–62)

```python
# ---------- B) Serve an OpenAI-compatible HTTP API (shell, not python) ------------------------------------------
#   vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000 --max-model-len 2048
#
# Then call it like any hosted API (this is how module 09/10 services would talk to a model server):
#   curl http://localhost:8000/v1/chat/completions \
#        -H "Content-Type: application/json" \
#        -d '{"model": "Qwen/Qwen2.5-0.5B-Instruct",
#             "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 32}'
#
# Because the endpoint is OpenAI-compatible, your module 08 agent harness can use it with only a base-URL change.

# ---------- B2) Useful server flags (shell) -----------------------------------------------------------------------------
#   --tensor-parallel-size 4        split every layer over 4 GPUs (step 14)        --pipeline-parallel-size 2   across nodes
#   --max-num-seqs 256              max concurrent sequences in one step           --max-num-batched-tokens 8192  per-step token budget
#   --quantization fp8              or a pre-quantised AWQ/GPTQ/compressed-tensors checkpoint (steps 04-06)
#   --speculative-config '{"method": "ngram", ...}'   speculative decoding (step 13; the config schema changes between versions)
# Benchmark it with the tool shipped with vLLM:  vllm bench serve ...  (TTFT / ITL / throughput, see step 10)

```

- **In simple words:** One shell command starts an OpenAI-compatible server; the flags are the levers of steps 04, 07, 12, 13 and 14.
- **Exactly:** `--tensor-parallel-size` shards each layer ([step 14](../14_parallelism_and_offloading/WALKTHROUGH.md)); `--max-num-seqs` and `--max-num-batched-tokens` set the scheduler budget; `--quantization fp8` or a pre-quantized AWQ/GPTQ/compressed-tensors checkpoint sets weight precision. Flag names evolve: read `--help`.
- **Why it is here:** A request to `/v1/chat/completions` looks identical for vLLM, SGLang, Ollama, llama-server and TensorRT-LLM, so your agent code ([module 08](../../08_agent_harness/README.md) and [09](../../09_api_development/README.md)) only changes `base_url`.

### 7. Reference: transformers vs vLLM  (`vllm_reference.py` lines 63–65)

```python
# ---------- C) Transformers vs vLLM in one line each ----------------------------------------------------------------
# transformers.generate : simple, flexible, great for learning/experiments, one batch at a time, memory-hungry cache.
# vLLM                  : production serving, many users at once, 2-20x higher throughput on the same GPU.
```

- **In simple words:** The one-line summary.
- **Exactly:** the "2-20x" figure is a rule-of-thumb range for many-concurrent-user throughput, workload dependent; single-request latency is similar.
- **Why it is here:** To anchor the contrast without overclaiming.

### 8. Other references  (`ollama_reference.py`, `sglang_reference.py`, `llamacpp_reference.md`, `trtllm_reference.md`)

- `ollama_reference.py` (45 lines, **guarded and run**): tries `GET http://localhost:11434/api/tags` with a 1.5 s timeout. If the server is not up it prints a `SKIP` line plus a read-only summary of the REST API (`/api/chat`, `/api/generate`, `/v1/chat/completions`, and the `eval_count / eval_duration` fields that give decode tokens/s). If a server is running it lists installed models and, only with `OLLAMA_RUN=1`, sends one tiny chat request to the first installed model. It **never pulls a model**.
- `sglang_reference.py` (44 lines, **not run**): RadixAttention and the zero-overhead scheduler explained in the docstring; launch command; OpenAI client with `response_format={"type": "json_object"}` (grammar-constrained decoding); the frontend `@sgl.function` where multiple calls share a cached system prompt.
- `llamacpp_reference.md` (**not run**): tools (`llama-cli`, `llama-server`, `llama-quantize`, `llama-bench`, `llama-perplexity`), the key flags (`-ngl`, `-c`, `-t`, `-np`, `--cache-type-k`), and what each means internally.
- `trtllm_reference.md` (**not run**): TensorRT-LLM scheduler, KV, kernels and quantization story and when (not) to choose it.

## Syntax explained

| Syntax | Meaning |
|---|---|
| `dict(kind=..., quant=[...])` | build a dictionary with keyword arguments |
| `ENGINES.items()` | iterate (name, row) pairs |
| `f"{key:<10}: {value}"` | left-align a field in a 10-character column |
| `def recommend(hardware: str, users: int, shared_prefix: bool = False, ...)` | type hints plus default values (module 00 step 03) |
| `return a, b` | a tuple: unpack with `eng, why = recommend(...)` |
| `ok += eng == expect` | add a bool to an int (True counts as 1) |
| `assert cond` | crash with `AssertionError` if the condition is false |
| `argparse` `choices=[...]` / `action="store_true"` | restrict values / boolean flag |
| `urllib.request.urlopen(url, timeout=1.5)` | stdlib HTTP GET, no extra package |
| `try: ... except Exception: ... else: ...` | `else` runs only when no exception happened |

## Libraries & functions

| Function | What it does | Internally |
|---|---|---|
| `argparse.ArgumentParser` | parse command-line flags | reads `sys.argv`, validates against declared options |
| `urllib.request.urlopen` | HTTP request | opens a TCP socket; raises `URLError` if refused |
| `vllm.LLM(...)` (reference) | offline engine | loads weights, profiles memory, allocates KV blocks, starts scheduler |
| `LLM.generate(prompts, params)` (reference) | batch generation | adds all requests to the scheduler, steps until all finish |
| `vllm serve` (reference) | server | same engine behind an async OpenAI-compatible HTTP frontend |
| `sgl.function` (reference) | SGLang program | records the calls so shared prefixes hit the radix cache |

## Execution trace (real run)

`python main.py` (the table is cut with `...`; the scenarios are complete):

```
ENGINE COMPARISON (internals)

== transformers == library (reference implementation)
  scheduler : none by default: you call generate() on one static batch; newer releases add experimental continuous batching + `transformers serve`
  kv        : DynamicCache (grows by concatenation), StaticCache (pre-allocated, torch.compile friendly); no paging by default
  kernels   : PyTorch ops + SDPA; optional FlashAttention-2 / hub kernels
  openai_api: via `transformers serve` (emerging); otherwise wrap it yourself
  best      : research, prototyping, fine-tuning, any new architecture on day one
  caveat    : low throughput under concurrency; wasteful KV memory
  quant     : bitsandbytes int8/nf4, GPTQ, AWQ, FP8, torchao, GGUF (dequantised on load)
  hardware  : cpu, cuda, mps, rocm, xpu

== llama.cpp == C/C++ engine on the ggml tensor library; `llama-cli`, `llama-server`
  scheduler : llama-server: N parallel 'slots', continuous batching, prompt cache per slot
  kv        : one pre-allocated KV buffer divided among slots (contiguous cells, no paging); K/V can be quantised (--cache-type-k q8_0)
  kernels   : hand-written ggml kernels: AVX2/AVX-512/NEON on CPU, Metal, CUDA, HIP, Vulkan, SYCL
  openai_api: yes: llama-server exposes /v1/chat/completions
  best      : laptops, edge, CPU+GPU split (-ngl N layers on GPU), single or few users, GGUF models
  caveat    : not built for hundreds of concurrent users; fewer batching/scheduling features than vLLM
  quant     : GGUF k-quants (Q4_K_M ...), i-quants (IQ4_XS ...), Q8_0, F16/BF16
  hardware  : cpu, apple-silicon, cuda, rocm, vulkan

...
SCENARIOS
  laptop CPU, 1 user, GGUF model                   -> ollama         (CPU or small GPU, few users: GGUF + llama.cpp family (-ngl splits layers CPU/GPU))
  MacBook, 1 user, MLX model                       -> mlx/lm-studio  (unified-memory Mac, one or two users: MLX is the Mac-native choice, Ollama/llama.cpp if you have GGUF)
  1x24GB consumer GPU, 20 users                    -> vllm           (default general-purpose GPU server: PagedAttention + continuous batching + prefix caching)
  8xH100 node, 500 users, shared system prompt     -> sglang         (RadixAttention reuses shared prefixes; strong grammar-constrained decoding)
  NVIDIA fleet, squeeze every token/$              -> tensorrt-llm   (NVIDIA-only, throughput-per-dollar matters more than flexibility: TensorRT-LLM FP8/NVFP4)
  day-one release of a new architecture            -> transformers   (a brand-new architecture usually lands in Transformers first; serve it there until an engine adds support)

CHECKS passed: 6/6
```

Asking your own question (`--hardware nvidia-datacenter --users 50 --structured`, then the same without `--structured`):

```
-> sglang: RadixAttention reuses shared prefixes; strong grammar-constrained decoding
-> vllm: default general-purpose GPU server: PagedAttention + continuous batching + prefix caching
```

- All six scenarios pass: the rules reproduce the decision logic in the theory section.
- The same hardware and user count flips from vLLM to SGLang once structured output is required: the decision depends on workload shape, not only on hardware.
- `python ollama_reference.py` on this laptop prints `SKIP: no Ollama server on http://localhost:11434 (URLError)` (the `ollama` binary is installed but no server was running; the script never pulls anything).

## Experiments (change ONE thing)

| Change | What you will see | Why |
|---|---|---|
| `--users 1` with `--hardware nvidia-datacenter` | vLLM, or SGLang with `--structured` | the rules do not special-case one user on a datacenter GPU; for one user a simpler engine would also be fine |
| Add `--max-perf-nvidia` | TensorRT-LLM | rule 4 outranks the generic GPU rule |
| `--hardware apple-silicon --users 10` | llama.cpp | several users on a Mac need slots + batching, not MLX single-user |
| Delete the `new_architecture` rule | the last scenario now returns vLLM | shows why rule order and rule 1 matter |
| `OLLAMA_RUN=1 python ollama_reference.py` | with a server and an installed model, one reply and tokens/s | the same decode-rate arithmetic as step 10 |

## Debugging practice

**BREAK IT:** `python main.py --hardware tpu`.
- **Symptom:** `error: argument --hardware: invalid choice: 'tpu' (choose from cpu, apple-silicon, ...)`.
- **Fix:** argparse is protecting the rules; add `tpu` to `choices` *and* a rule, or you would fall through to the default.

**BREAK IT:** move the `gpu_server` check above the CPU rule in `recommend`.
- **Symptom:** nothing crashes, but a scenario returns the wrong engine and `assert ok == len(cases)` fails.
- **Fix:** keep specific rules before general ones; that assert is why the scenarios exist.

**Real-world error:** calling a vLLM server with the wrong `model` string gives HTTP 404 `The model ... does not exist`. Fix: use the exact name passed to `vllm serve` (see `GET /v1/models`).

## Common mistakes

- Picking an engine from a leaderboard instead of benchmarking your model, prompt lengths and concurrency.
- Using Ollama or `generate()` for hundreds of concurrent users.
- Using vLLM on a laptop: it needs an accelerator and Linux for the supported paths.
- Expecting the same quant format everywhere: GGUF is the llama.cpp world, AWQ/GPTQ/FP8 the GPU-engine world.
- Starting a new project on TGI: it is in maintenance mode.
- Forgetting that the OpenAI-compatible API is a *shape*, not a guarantee: sampling parameters, tool calling, and structured-output options vary by engine.

## Rebuild checklist

1. Write the five comparison axes: scheduler, KV management, kernels, quant formats, hardware (plus API).
2. Fill them for transformers, llama.cpp, Ollama, vLLM, SGLang, TensorRT-LLM; note TGI is maintenance mode.
3. Write `recommend(hardware, users, ...)` with ordered rules.
4. Write six scenarios with the expected answer and an `assert`.
5. Add an `argparse` front end with `choices`.
6. From memory, write the vLLM `LLM(...)` call and the `vllm serve` curl request.

## Say it aloud

- "A serving engine adds a scheduler, a KV-cache manager, optimized kernels, quantization support and an API on top of the model; vLLM's edge is PagedAttention plus continuous batching, SGLang's is RadixAttention prefix reuse and structured output."
- "llama.cpp and Ollama are the local, GGUF, CPU/Metal/CUDA world; vLLM, SGLang and TensorRT-LLM are the GPU server world; TGI is in maintenance mode."
- "I would pick vLLM by default for many users, SGLang if prefixes are shared, TensorRT-LLM if I am NVIDIA-only and optimizing cost, and I would benchmark all three on my traffic first."
- "All of them speak an OpenAI-compatible API, so switching engines is mostly a `base_url` change."

## The new problem (cliffhanger)

You can now choose an engine, but your manager asks the question that decides the project: *what will it cost per million tokens, and is it cheaper than just calling an API?* That needs throughput, GPU price, utilisation and a break-even. On to [step 16 — cost per token](../16_cost_per_token/WALKTHROUGH.md).
