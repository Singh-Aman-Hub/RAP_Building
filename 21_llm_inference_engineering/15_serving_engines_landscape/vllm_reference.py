"""
REFERENCE ONLY -- NOT EXECUTED IN THIS STUDY PACK.

vLLM is a high-throughput LLM inference engine.  It runs on Linux + NVIDIA GPU (CUDA); there is no
Mac/CPU path worth using, so read this file, then run it on a rented GPU or Colab if you want to see it live.

    pip install vllm

Two ideas make vLLM fast (see README.md in this module for the pictures):

1. PagedAttention  - KV cache is stored in small fixed-size blocks (like OS virtual-memory pages), so
                     no GPU memory is wasted on "reserved but unused" cache space.  More concurrent
                     requests fit in the same GPU memory.
2. Continuous batching - as soon as ONE request in the batch finishes, a waiting request takes its slot
                     immediately.  Naive static batching waits for the slowest request in the batch.
"""

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
)

prompts = ["Explain what a KV cache is.", "Write a haiku about GPUs.", "What is LoRA?"]
outputs = llm.generate(prompts, params)      # ONE call: vLLM batches all prompts for you, continuously
for out in outputs:
    print(out.prompt, "->", out.outputs[0].text)

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

# ---------- C) Transformers vs vLLM in one line each ----------------------------------------------------------------
# transformers.generate : simple, flexible, great for learning/experiments, one batch at a time, memory-hungry cache.
# vLLM                  : production serving, many users at once, 2-20x higher throughput on the same GPU.
