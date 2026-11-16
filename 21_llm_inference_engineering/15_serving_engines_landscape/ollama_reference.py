"""
REFERENCE (guarded) -- talks to a LOCAL Ollama server only if one is already running. Never pulls or downloads a model.

Ollama = a daemon that wraps llama.cpp/ggml, keeps a model registry, and serves a REST API on http://localhost:11434.
Install separately from ollama.com; `ollama pull <model>` (NOT done by this script) downloads a GGUF model.

RUN:  python ollama_reference.py                (lists installed models if the server is up, else prints a skip message)
      OLLAMA_RUN=1 python ollama_reference.py   (also sends ONE tiny chat request to the first installed model)
NEEDS: Python stdlib only.  ~0.1 s.
"""
import json
import os
import urllib.request

BASE = "http://localhost:11434"


def get(path, timeout=1.5):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.load(r)


try:
    tags = get("/api/tags")                                   # GET /api/tags -> models already on disk (does not pull)
except Exception as e:                                        # connection refused, timeout, etc.
    print(f"SKIP: no Ollama server on {BASE} ({type(e).__name__}). Start it with `ollama serve` to try this file.")
    print("""
What the API looks like (read-only summary):
  POST /api/chat      {"model": "<name>", "messages": [{"role": "user", "content": "hi"}], "stream": false}
  POST /api/generate  {"model": "<name>", "prompt": "hi", "options": {"num_ctx": 2048, "temperature": 0.7}}
  POST /v1/chat/completions   <- OpenAI-compatible: point the openai SDK at base_url="http://localhost:11434/v1"
  Reply fields worth knowing: eval_count (output tokens), eval_duration (ns), prompt_eval_count, prompt_eval_duration,
  load_duration  -> tokens/s = eval_count / eval_duration * 1e9   (same TTFT/decode split as step 10).
  keep_alive controls how long the model stays in RAM after the last request.""")
else:
    models = [m["name"] for m in tags.get("models", [])]
    print("Ollama is up. Installed models:", models or "none (this script will not pull any)")
    if models and os.environ.get("OLLAMA_RUN") == "1":
        body = json.dumps({"model": models[0], "stream": False, "options": {"num_predict": 16},
                           "messages": [{"role": "user", "content": "Say hi in five words."}]}).encode()
        req = urllib.request.Request(BASE + "/api/chat", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.load(r)
        tps = out["eval_count"] / out["eval_duration"] * 1e9
        print(out["message"]["content"], f"| {out['eval_count']} tokens, {tps:.1f} tok/s decode")
