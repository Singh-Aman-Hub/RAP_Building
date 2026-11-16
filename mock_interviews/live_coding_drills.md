# Live Coding Drills (timer on, no AI, no notes)

| # | Task | Time | Module to review if stuck |
|---|------|-----:|---------------------------|
| 1 | Word-frequency counter from a string; print top 3 | 10 min | 00, 02 |
| 2 | Two Sum (brute force, then dict); state Big-O | 15 min | 02 |
| 3 | Valid Anagram and Group Anagrams | 15 min | 02 |
| 4 | Longest substring without repeats (sliding window) | 20 min | 02 |
| 5 | Subarray sum via prefix sums | 15 min | 02 |
| 6 | Class `Config` that loads JSON, with defaults and a clear error if the file is missing | 15 min | 00 |
| 7 | Gradient descent for y = 3x + 2 in numpy | 20 min | 03 |
| 8 | Tokenize text with a HF tokenizer, print ids, decode back | 10 min | 04 |
| 9 | Implement temperature + top-k sampling on a logits array | 20 min | 05 |
| 10 | Agent loop: calculator tool, max 5 steps, validate args, log each call | 30 min | 08 |
| 11 | FastAPI `/health` and POST `/predict` with Pydantic validation and 422 on bad input | 20 min | 09 |
| 12 | Dockerfile + `docker run -p` for that API | 15 min | 11 |
| 13 | Streamlit page: text box, button, show prediction, error state | 15 min | 12 |

Review: did you state edge cases (empty input, duplicates) before coding? Did you dry-run one example?

---

## Drills for modules 15 to 21 (timer on, no AI, no notes)

State the edge cases, write a tiny test first or right after, and dry-run one example. "Stuck?" points to the step that already contains a worked version; try before you look.

| # | Task | Time | Stuck? Review |
|---|------|-----:|---------------|
| 14 | Backprop for a 2-layer MLP classifier in numpy (He init, ReLU, softmax-CE) plus a central-difference gradient check; relative error under 1e-6 in float64 | 30 min | [15/02](../15_neural_networks/02_mlp_backprop_from_scratch/WALKTHROUGH.md) |
| 15 | BPE: train N merges on a string (bytes, most frequent pair, deterministic ties), then `encode` and `decode`; assert round trip | 25 min | [16/01](../16_modern_llm_architecture/01_bpe_tokenizer_from_scratch/WALKTHROUGH.md) |
| 16 | Online softmax over chunks of a vector (running max `m`, running sum `l`, rescale by `exp(m_old - m_new)`); assert equal to `softmax` to 1e-12 including large values like 1000 | 15 min | [16/05](../16_modern_llm_architecture/05_flash_attention_online_softmax/WALKTHROUGH.md) |
| 17 | LoRA linear layer: frozen `W`, trainable `A (r,in)` and `B (out,r)` with `B=0`, scale `alpha/r`; forward; count trainable parameters; `merge()` and assert merged output equals unmerged | 20 min | [06/04](../06_llm_training_and_lora/04_lora_math_numpy/WALKTHROUGH.md), [17/03](../17_finetuning_alignment_and_benchmarks/03_peft_family/WALKTHROUGH.md) |
| 18 | int4 quantiser with group size 128: scales, `q` in [-8, 7], pack two codes per byte, unpack, dequantise; report max error and bits per weight | 20 min | [21/04](../21_llm_inference_engineering/04_quantization_math/WALKTHROUGH.md) |
| 19 | `kv_cache_bytes(layers, kv_heads, head_dim, seq, batch, bytes)` and `max_tokens(gpu_gb, weights_gb)`; check Llama-3-8B gives 131,072 B per token | 10 min | [21/03](../21_llm_inference_engineering/03_gpu_memory_calculator/WALKTHROUGH.md) |
| 20 | Block-hash prefix cache: 16-token blocks, chained hash, `match(prompt) -> n_cached_tokens`, `insert`, LRU eviction with capacity; show a prompt sharing 500 tokens matches 496 | 30 min | [21/12](../21_llm_inference_engineering/12_continuous_batching_prefix_caching/WALKTHROUGH.md) |
| 21 | Recursive chunker: separators coarse to fine, merge to `chunk_size` tokens with `overlap`, recurse on oversized pieces, return character offsets; validate `0 <= overlap < size` | 30 min | [19/02](../19_chunking_deep_dive/02_recursive_splitter_rebuilt/WALKTHROUGH.md) |
| 22 | CTC greedy decode of a `(T, C)` log-prob matrix (blank id 0): argmax, merge repeats, drop blanks; test `a a` -> `a` and `a blank a` -> `aa` | 15 min | [18/05](../18_ocr_and_document_parsing/05_crnn_and_ctc/WALKTHROUGH.md) |
| 23 | Token-bucket rate limiter with an injectable clock: `try_acquire(n)`, capacity, refill rate, thread-safe; test with a fake clock (burst, refill, rejection) | 25 min | [20/06](../20_llm_system_lld/06_resilience/WALKTHROUGH.md) |

Review for 14 to 23: did you test against a trusted reference (autograd, `torch.softmax`, the real tokenizer, `nn.CTCLoss`)? Did you handle the empty and the single-element input? Could you state the complexity and where the memory goes?
