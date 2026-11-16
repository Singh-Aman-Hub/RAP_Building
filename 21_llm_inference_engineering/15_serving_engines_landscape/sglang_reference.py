"""
REFERENCE ONLY -- NOT EXECUTED IN THIS STUDY PACK (needs Linux + NVIDIA/AMD GPU and `pip install sglang`).

SGLang = serving engine + a small frontend language for structured / multi-call LLM programs.  Version note (Nov 2026):
around v0.5.x (medium confidence); flags move quickly, check `python -m sglang.launch_server --help`.

Its two signature ideas
  1. RadixAttention: all cached KV blocks live in a RADIX TREE keyed by token prefixes. A new request walks the tree,
     reuses the longest matching prefix and computes only the rest. LRU eviction removes leaf nodes. Great for chat
     history, few-shot prompts, RAG templates and agent loops where many requests share a long prefix (step 12).
  2. Zero-overhead scheduler: CPU-side batching/scheduling work overlaps with GPU compute, so the GPU is not idle between steps.
"""

# ---------- A) Launch an OpenAI-compatible server (shell) ---------------------------------------------------------
#   python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --port 30000 --tp 1
#   (--tp 4 = tensor parallel over 4 GPUs, step 14;  --kv-cache-dtype fp8_e4m3 for an fp8 KV cache, step 07)
#
# Call it with the OpenAI client by only changing the base URL (same trick as vLLM / Ollama):
from openai import OpenAI                                    # needs `pip install openai`

client = OpenAI(base_url="http://localhost:30000/v1", api_key="none")
resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Return a JSON object with keys name and age."}],
    max_tokens=64,
    # Structured output: the engine masks logits with a grammar compiler so ONLY valid JSON tokens can be sampled.
    response_format={"type": "json_object"},
)
print(resp.choices[0].message.content)

# ---------- B) The native frontend language (multi-call programs share prefixes automatically) ------------------------
import sglang as sgl


@sgl.function
def qa(s, question):
    s += sgl.system("You are a terse assistant.")            # this shared prefix is cached ONCE in the radix tree
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=48))


sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))
states = qa.run_batch([{"question": q} for q in ["What is KV cache?", "What is TP?"]])   # both reuse the system-prompt KV
print([s["answer"] for s in states])
