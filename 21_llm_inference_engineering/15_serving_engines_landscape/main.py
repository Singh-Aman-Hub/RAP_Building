"""
GOAL:  Compare the LLM serving engines (HF Transformers, llama.cpp, Ollama, vLLM, SGLang, TensorRT-LLM, TGI, MLX/LM Studio)
       on their internals, and turn the comparison into a small decision helper: "given my hardware and workload, which engine?".

RUN (from this step folder):  python main.py
                              python main.py --hardware nvidia-datacenter --users 200 --shared-prefix --structured
RUN (from repo root):         .venv/bin/python 21_llm_inference_engineering/15_serving_engines_landscape/main.py
EXPECTED OUTPUT (shortened):
    ENGINE COMPARISON (internals) ...
    SCENARIOS
      laptop, 1 user, GGUF model            -> ollama  (...)
      8xH100 node, 500 users, shared system prompt -> sglang / vllm ...
    CHECKS passed: 6/6
NEEDS: Python stdlib only (no downloads, no GPU). ~0.1 s.
Note: engine facts are a snapshot of the architecture (Nov 2026); version numbers move fast, so check each project's release page.
"""
import argparse

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

# TRY IT: ask `--hardware nvidia-datacenter --users 50 --structured` then drop --structured. What changes and why?
# TRY IT: add a new engine row (e.g. an engine you read about) and a rule for it; keep the scenarios passing.
# BREAK IT: reorder the rules so `gpu_server` is tested before `hardware == "cpu"`: which scenario breaks?
# BREAK IT: pass --hardware tpu and read the argparse error (choices guard bad input).
