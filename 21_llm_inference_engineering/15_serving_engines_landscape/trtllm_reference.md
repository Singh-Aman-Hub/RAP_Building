# TensorRT-LLM (reference, not run: NVIDIA GPU required)

**What it is.** NVIDIA's inference library for squeezing the most tokens/s out of NVIDIA GPUs. Historically you *compiled* a model into a TensorRT engine for one GPU type and one set of shapes; as of late 2026 (medium confidence) the PyTorch-based workflow is the default and the compiled-engine path is the older route. Releases: a stable 1.2.x line in spring 2026, 1.4 release candidates in autumn 2026. Check NVIDIA's release notes for the current one.

**Internals (what to say in an interview).**
- *Scheduler:* in-flight batching (= continuous batching), chunked context (= chunked prefill), optional disaggregated prefill/decode serving.
- *KV cache:* paged blocks, KV reuse across requests (prefix caching), fp8/int8 KV cache.
- *Kernels:* hand-tuned fused attention and GEMM kernels (CUTLASS-based), custom plugins, CUDA graphs; best-in-class FP8 on Hopper and NVFP4 on Blackwell.
- *Quantization:* FP8, NVFP4, INT4-AWQ, INT8 SmoothQuant, GPTQ; produced with NVIDIA's Model Optimizer / `modelopt` toolkit.
- *Parallelism:* TP, PP, EP (for MoE) and combinations; NVLink-aware.
- *Serving:* `trtllm-serve` gives an OpenAI-compatible endpoint; Triton Inference Server or NVIDIA Dynamo handle fleets, routing and KV transfer.

**Sketch (shell, from the docs' style; flags change by version).**
```bash
pip install tensorrt-llm                       # needs matching CUDA/driver; NVIDIA container images are the easy route
trtllm-serve Qwen/Qwen2.5-7B-Instruct --port 8000 --tp_size 1
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 32}'
```

**When to choose it.** NVIDIA-only fleet, stable model, large volume, and an engineering team that can tune it: the tokens-per-dollar gain matters (step 16). **When not to:** you need AMD/Mac/CPU, day-one support for a new architecture, or minimal ops effort (vLLM/SGLang are easier).
