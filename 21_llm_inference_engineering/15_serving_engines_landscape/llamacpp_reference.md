# llama.cpp (reference: needs the llama.cpp binaries, not installed in this repo's venv)

**What it is.** A C/C++ inference engine built on the **ggml** tensor library. Models are **GGUF** files (step 02). Runs on CPU (AVX2/AVX-512/NEON), Apple Metal, CUDA, HIP, Vulkan and SYCL. Releases are build-numbered (`bNNNN`). Ollama and LM Studio both embed it.

**Tools you will hear about**
| Tool | Purpose |
|---|---|
| `llama-cli` | interactive / one-shot generation in the terminal |
| `llama-server` | HTTP server with an OpenAI-compatible `/v1/chat/completions`, parallel slots and continuous batching |
| `llama-quantize` | convert an F16 GGUF to Q4_K_M, Q5_K_M, Q8_0, IQ4_XS ... |
| `llama-bench` | measures prompt-processing (pp) and token-generation (tg) tokens/s |
| `llama-perplexity` | perplexity on a text file: how you check a quantization (step 04/17) |
| `convert_hf_to_gguf.py` | turn an HF safetensors checkpoint into GGUF |

**Key flags**
| Flag | Meaning |
|---|---|
| `-m model.gguf` | the model file |
| `-ngl N` (`--n-gpu-layers`) | put N transformer layers on the GPU, the rest stay on CPU RAM: this is *layer offloading*, the same idea as `device_map`/`max_memory` in step 14. Layers cost the same each, so choose N so that weights + KV cache fit VRAM |
| `-c N` (`--ctx-size`) | context length; sets KV cache size (step 03 formula) |
| `-t N` | CPU threads (decode is memory-bound: more threads help little past the memory-bandwidth limit) |
| `-np N` | parallel slots in `llama-server` (the KV buffer is divided between them) |
| `--cache-type-k q8_0 --cache-type-v q8_0` | quantised KV cache (step 07) |
| `-fa` | flash attention kernel |
| `--mmap` / `--no-mmap` | memory-map the file (fast start, pages loaded on demand) or read fully |

**Example (shell)**
```bash
llama-quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M
llama-server -m model-Q4_K_M.gguf -c 4096 -ngl 99 -np 4 --port 8080
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 32}'
llama-bench -m model-Q4_K_M.gguf -ngl 99        # pp512 and tg128 in tokens/s
```

**Internals to remember.** KV cache is one pre-allocated buffer of cells shared by slots (no PagedAttention-style block table); `-ngl` splits layers between devices; k-quants use super-blocks of 256 weights with per-block scales (step 02); prompt processing (prefill) is compute-bound and token generation (decode) memory-bound (step 09).
