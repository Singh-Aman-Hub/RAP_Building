# Cross-Module Question Bank

Answer aloud first, then compare. Each answer names the modules it connects.

**1. Why does temperature 0 give the same answer every time?**
The model outputs logits; softmax turns them into probabilities. Temperature divides logits before softmax; as it approaches 0 the top token dominates, so decoding becomes argmax (greedy) and is deterministic. Trade-off: repetitive, no diversity. (03 softmax, 04 logits, 05 sampling)

**2. Your API runs out of memory with 4 Gunicorn workers. Why, and what do you do?**
Each worker is a separate process that loads its own copy of the model. Memory = workers x model size. Fix: fewer workers, load once with `--preload` (copy-on-write helps only partly), quantize, or serve the model in one dedicated process/service. (04 params x bytes, 05 quantization, 09 workers)

**3. WSGI vs ASGI?**
WSGI is the synchronous Python web interface (Flask + Gunicorn); one request per worker at a time. ASGI supports async and websockets (FastAPI + Uvicorn); one worker can interleave many I/O-bound requests. CPU-bound model inference still blocks the event loop unless run in a thread/process pool. (09)

**4. Your agent loops forever. What do you check?**
Max-step limit, a clear stop condition, whether tool results are fed back, repeated identical tool calls; read the JSONL trace. (08, 01)

**5. The LLM returned invalid JSON and the endpoint crashed.**
Constrain the prompt and schema, extract + validate (Pydantic), retry with the error message, then fail gracefully: 502/422 style response, not a 500 traceback. (07, 08, 09)

**6. Container starts then exits immediately.**
`docker logs <id>`; check CMD path/syntax, missing dependency, app binding to 127.0.0.1 instead of 0.0.0.0, process running in background so PID 1 ends. (11, 01)

**7. Why is generation slower for long outputs?**
Attention cost grows with sequence length; the KV cache avoids recomputation but grows in memory per token. (04, 06, 02 Big-O)

**8. How would you serve 100 requests/s?**
Measure first. Batch requests, cache repeated inputs, quantize, scale replicas behind a load balancer, async I/O, queue for bursts, set timeouts. (06, 09, 10, 11)

**9. Fine-tune vs prompt vs LoRA?**
Prompt first (cheapest, reversible). Fine-tune when behavior/style must change and you have data. LoRA trains small low-rank matrices: far less memory, swappable adapters; slightly less capacity than full fine-tuning. (06, 07)

**10. Pickle/joblib model files: any concern?**
Unpickling can execute arbitrary code: only load trusted files; prefer safetensors/ONNX for shared weights; pin library versions. (03, 10)

**11. Learning rate too high vs too low?**
Too high: loss oscillates/diverges. Too low: slow, may stall. Batch size trades gradient noise against memory and speed. (03, 05)

**12. Prompt injection: what and how to mitigate?**
Untrusted text carrying instructions that override yours. Separate data from instructions, least-privilege tools, confirmation for side effects, validate outputs, never treat model output as trusted code. (07, 08)

**13. Explain the code you wrote with AI.**
Walk input -> processing -> output, name each library call and its arguments, state one limitation. Use the WALKTHROUGH "Say it aloud" blocks. (all)

**14. Environment vs code bug: how do you tell?**
`which python`, `pip list`, `python -c "import x; print(x.__file__)"`, run a minimal repro in a fresh venv. (01)

**15. Batch vs real-time inference?**
Batch: scheduled, high throughput, latency irrelevant. Real-time: per-request low latency, needs always-on serving. Pick by freshness need and cost. (10)
