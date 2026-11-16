# Tech Radar 2026 — what tools and versions are used today

*Checked 2026-11-05 by web search plus this repo's `.venv`. Read the honesty rules first.*

## Honesty rules (read this)
- **"Seen in .venv"** = the exact version installed in this repo's `.venv` (read from `site-packages/*.dist-info`). This is the most reliable number here, because it is real and recent.
- **"Web (date)"** = a version a search result reported, with the date that result gave. Many are from blogs or mirrors, not official pages, and are marked *(3rd-party)*. Search results were often 2-5 months old, so newer releases very likely exist.
- **"see PyPI/releases"** = I could not confirm a number. Do not quote a version in an interview from this file; quote the *name and role*, and say "I'd check PyPI".
- Model-family rows are names only plus official links. No specs are claimed.
- Interview tip: versions change monthly. Tools, roles, and trade-offs are what you are tested on.

Related: [CONFUSABLES.md](CONFUSABLES.md) (look-alike terms), [README.md](README.md), [ROADMAP.md](ROADMAP.md), [TERMS_INDEX.md](TERMS_INDEX.md).

Column key: **In .venv** = installed here (version) or "-" if not installed.

---

## 1. Python and tooling
**Changed since 2024:** Python 3.13 and 3.14 shipped (3.14: template strings, free-threaded build officially supported, subinterpreters in stdlib). uv replaced pip/venv/pip-tools/pyenv for many teams. Ruff replaced flake8+black+isort for many projects.

| Tool | What it is for | Latest version seen (date) | Status / note | In .venv | Source |
|---|---|---|---|---|---|
| Python | Language runtime | 3.14.6 (10 Jun 2026, python.org doc index); newer bugfix likely | 3.14 is the current stable line; repo uses 3.12 | 3.12 | [python.org downloads](https://www.python.org/downloads/) |
| uv | Fast package + venv + Python manager (Astral) | 0.11.16 (21 May 2026, 3rd-party PR) | Dominant new default; newer versions exist | - | [PyPI uv](https://pypi.org/project/uv/) |
| Ruff | Linter + formatter (Astral) | 0.15.11 (16 Apr 2026, Red Hat tracker) | Replaces flake8/black/isort; newer exist | - | [PyPI ruff](https://pypi.org/project/ruff/) |
| pytest | Testing | see PyPI/releases | Standard | 9.1.1 | [PyPI pytest](https://pypi.org/project/pytest/) |
| NumPy | Arrays | see PyPI/releases | 2.x line | 2.5.4 | [PyPI numpy](https://pypi.org/project/numpy/) |
| pandas | Dataframes | see PyPI/releases | 3.x line (seen in venv) | 3.0.6 | [PyPI pandas](https://pypi.org/project/pandas/) |
| scikit-learn | Classic ML | see PyPI/releases | Standard | 1.9.1 | [PyPI scikit-learn](https://pypi.org/project/scikit-learn/) |
| Pydantic | Validation (v2) | see PyPI/releases | v2 is the only line you should use | 2.14.1 | [PyPI pydantic](https://pypi.org/project/pydantic/) |

## 2. Deep learning frameworks
**Changed since 2024:** PyTorch is the default for research and LLMs; `torch.compile` matured; TensorFlow/Keras is a niche for LLM work; Transformers v5 made PyTorch the primary backend and sunset TF/Flax support. JAX stays strong for TPU and large-scale training.

| Tool | What it is for | Latest version seen (date) | Status / note | In .venv | Source |
|---|---|---|---|---|---|
| PyTorch | Tensors, autograd, training | 2.14.1 installed here (Nov 2026); web search could not confirm newer | Default framework | 2.14.1 | [pytorch.org](https://pytorch.org/get-started/locally/) |
| JAX | Autodiff + XLA, TPU-friendly | see PyPI/releases | Used by Google, Anthropic-style large training stacks, research | - | [PyPI jax](https://pypi.org/project/jax/) |
| safetensors | Safe weight file format | see PyPI/releases | Replaces pickle `.bin` | 0.8.0 | [PyPI safetensors](https://pypi.org/project/safetensors/) |

## 3. Hugging Face stack
**Changed since 2024:** Transformers v5 (announced 1 Dec 2025): simpler model definitions, PyTorch-first, `transformers serve` OpenAI-compatible server, TF/Flax dropped. `huggingface_hub` moved to 2.x (seen in venv). TRL reached 1.x.

| Tool | What it is for | Latest version seen (date) | Status / note | In .venv | Source |
|---|---|---|---|---|---|
| transformers | Model definitions, pipelines | 5.20.1 installed here; v5 launched Dec 2025 | v5 line is current | 5.20.1 | [HF blog v5](https://huggingface.co/blog/transformers-v5) |
| tokenizers | Fast BPE/WordPiece in Rust | see PyPI/releases | Used by transformers | 0.23.3 | [PyPI tokenizers](https://pypi.org/project/tokenizers/) |
| datasets | Data loading, Arrow-backed | see PyPI/releases | Standard | 5.1.0 | [PyPI datasets](https://pypi.org/project/datasets/) |
| peft | LoRA, QLoRA, adapters | see PyPI/releases | Standard for adapters | 0.21.2 | [PyPI peft](https://pypi.org/project/peft/) |
| accelerate | Multi-device launch | see PyPI/releases | Standard | 1.15.0 | [PyPI accelerate](https://pypi.org/project/accelerate/) |
| sentence-transformers | Bi-encoders, cross-encoders, rerankers | see PyPI/releases | Standard embedding library | 6.2.0 | [PyPI sentence-transformers](https://pypi.org/project/sentence-transformers/) |
| huggingface_hub | Hub client | see PyPI/releases | 2.x line | 2.2.0 | [PyPI huggingface-hub](https://pypi.org/project/huggingface-hub/) |
| TRL | SFT, DPO, GRPO, reward trainers | 1.9.2 (28 Jul 2026, 3rd-party table); another source said 1.8.0 | 1.x line | - | [PyPI trl](https://pypi.org/project/trl/) |

## 4. Fine-tuning tools
**Changed since 2024:** GRPO / RLVR-style training entered the mainstream toolkits; Unsloth became the go-to for single-GPU QLoRA; Axolotl and LLaMA-Factory are config-driven; torchtune lost momentum.

| Tool | What it is for | Latest version seen (date) | Status / note | In .venv | Source |
|---|---|---|---|---|---|
| TRL | HF-native SFT/DPO/GRPO trainers | 1.9.2 (Jul 2026, 3rd-party) | Active | - | [github.com/huggingface/trl](https://github.com/huggingface/trl) |
| Unsloth | Fast, low-VRAM LoRA/QLoRA/RL | 2026.8.5 (Aug 2026, 3rd-party) | Very active, several releases per week | - | [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth) |
| Axolotl | YAML-config fine-tuning, multi-GPU | 0.18.0 (17 Jul 2026, 3rd-party) | Active | - | [github.com/axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl) |
| LLaMA-Factory | GUI + CLI fine-tuning of many models | 0.9.5 (30 May 2026, 3rd-party) | About 2 releases/year | - | [github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) |
| torchtune | PyTorch-native post-training library | see PyPI/releases | **Unclear/likely winding down**: no official notice found; one PyTorch-org PR said development wound down in 2025. Verify before adopting | - | [github.com/pytorch/torchtune](https://github.com/pytorch/torchtune) |

## 5. Inference and serving
**Changed since 2024:** vLLM (PagedAttention, continuous batching) is the open default; SGLang (RadixAttention) is its main rival; **TGI is in maintenance mode** (official banner; HF points to vLLM/SGLang for new work); llama.cpp/Ollama dominate local; disaggregated prefill/decode and speculative decoding became standard features.

| Tool | What it is for | Latest version seen (date) | Status / note | In .venv | Source |
|---|---|---|---|---|---|
| vLLM | High-throughput GPU serving | v0.26.x seen on a release tracker (undated); 0.24.0 announced (undated); Intel notes 0.21.0 = 26 Jun 2026 (3rd-party) | Open default; version number uncertain | - | [github.com/vllm-project/vllm/releases](https://github.com/vllm-project/vllm/releases) |
| SGLang | Serving with RadixAttention, structured output | 0.5.22 on PyPI (16 Oct 2026, 3rd-party) | Active | - | [github.com/sgl-project/sglang](https://github.com/sgl-project/sglang) |
| llama.cpp / GGUF | CPU/Metal/consumer-GPU inference, GGUF files | build-numbered `bNNNN` (b9733 on 20 Jun 2026 seen); no semver | Daily builds; use latest build | - | [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |
| Ollama | Easy local model runner (wraps llama.cpp-style engine) | v0.40.10 (23 Oct 2026, 3rd-party, conflicting) | Active | - | [ollama.com/download](https://ollama.com/download) |
| TensorRT-LLM | NVIDIA-optimized serving | see PyPI/releases | NVIDIA-only, fastest on NVIDIA when tuned | - | [github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) |
| TGI | HF Text Generation Inference | see releases | **Maintenance mode** (official README caution) | - | [github.com/huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference) |
| transformers serve | OpenAI-compatible dev server in transformers v5 | ships with transformers 5.x | Convenience, not for high load | 5.20.1 | [HF blog v5](https://huggingface.co/blog/transformers-v5) |

## 6. Quantization
**Changed since 2024:** FP8 is the near-lossless default on Hopper/Ada/Blackwell; 4-bit floating formats (NVFP4, MXFP4) need Blackwell-class hardware; AWQ (with Marlin kernels) is a common INT4 datacenter default; GGUF K-quants for local.

| Tool / format | What it is for | Latest version seen (date) | Status / note | In .venv | Source |
|---|---|---|---|---|---|
| FP8 | 8-bit float weights/activations | n/a (format) | Near-lossless, widely supported; vLLM can convert at load time | - | [vLLM quantization docs](https://docs.vllm.ai/en/latest/features/quantization/) |
| NVFP4 / MXFP4 | 4-bit float, block-scaled | n/a (formats) | Need Blackwell FP4 tensor cores; one benchmark found INT4-AWQ faster on 3 of 4 models | - | [spheron guide (3rd-party)](https://www.spheron.network/blog/nvfp4-vs-mxfp4-gpu-cloud-4bit-quantization-guide/) |
| AWQ | Activation-aware 4-bit PTQ | see PyPI/releases | Common INT4 default | - | [github.com/mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq) |
| GPTQ | Hessian-based 4-bit PTQ | see PyPI/releases | Supports LoRA on GPTQ-Int4 in vLLM | - | [github.com/IST-DASLab/gptq](https://github.com/IST-DASLab/gptq) |
| bitsandbytes | 8-bit/4-bit (NF4) for QLoRA | see PyPI/releases (status not confirmed by search) | Standard for QLoRA training | - | [PyPI bitsandbytes](https://pypi.org/project/bitsandbytes/) |
| GGUF | llama.cpp file format with K-quants | n/a | Local default | - | [llama.cpp](https://github.com/ggml-org/llama.cpp) |

## 7. Evaluation tools
**Changed since 2024:** LLM-as-judge and agent evals became normal; Inspect (UK AISI) became a serious framework; tool-version claims below are weakly confirmed.

| Tool | What it is for | Latest version seen (date) | Status / note | In .venv | Source |
|---|---|---|---|---|---|
| lm-evaluation-harness | Run standard benchmarks on models (EleutherAI) | 0.4.13 (31 Aug 2026, 3rd-party) | Standard | - | [github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) |
| lighteval | HF eval toolkit | 0.13.0 (24 Nov 2025, 3rd-party; may be stale) | Active | - | [github.com/huggingface/lighteval](https://github.com/huggingface/lighteval) |
| Inspect AI | Eval framework with agents/sandboxes (UK AISI) | see PyPI/releases | Rising | - | [inspect.aisi.org.uk](https://inspect.aisi.org.uk/) |
| OpenAI evals / simple-evals | OpenAI's eval registry and small reference evals | see repo | Reference code | - | [github.com/openai/simple-evals](https://github.com/openai/simple-evals) |
| RAGAS | RAG metrics (faithfulness, context precision/recall) | see PyPI/releases | Standard RAG eval | - | [docs.ragas.io](https://docs.ragas.io/) |
| DeepEval | pytest-style LLM app evals | see PyPI/releases | Active | - | [github.com/confident-ai/deepeval](https://github.com/confident-ai/deepeval) |

### Benchmarks in 2026 (3rd-party summaries, directional)
| Benchmark | Measures | Status in 2026 | Source |
|---|---|---|---|
| MMLU, GSM8K, HumanEval, HellaSwag, ARC-Challenge | Knowledge, grade-school math, simple code, commonsense | **Saturated** and contaminated; floor checks only | [futureagi](https://futureagi.com/blog/llm-benchmarking-state-2026/) |
| MMLU-Pro, GPQA Diamond | Harder knowledge / graduate science | Used, GPQA Diamond near top of model cards | [atlan](https://atlan.com/know/llm-benchmarks-explained/) |
| SWE-bench Verified | Real GitHub issue fixing | Still widely cited but **contested**: one report says OpenAI stopped reporting it in Feb 2026 over contamination | [buildmvpfast](https://www.buildmvpfast.com/blog/ai-benchmark-saturation-2026) |
| Humanity's Last Exam (HLE), FrontierMath, ARC-AGI-2, AIME | Hard reasoning | Headline "hard" benchmarks in model cards | [atlan](https://atlan.com/know/llm-benchmarks-explained/) |
| LMArena (Chatbot Arena) | Human pairwise preference Elo | Still the preference reference | [lmarena.ai](https://lmarena.ai/) |

## 8. Embeddings and rerankers
**Changed since 2024:** LLM-based embedders (Qwen3-Embedding, Gemini Embedding) overtook BERT-style ones; Matryoshka dims and multilingual/code coverage are normal; leaderboards shift weekly and are gameable, so test on your own data.

| Model / tool | What it is for | Status / note (search snapshots, conflicting) | In .venv | Source |
|---|---|---|---|---|
| MTEB leaderboard | Embedding benchmark | Snapshots disagree: NV-Embed-v2, Gemini Embedding 001 (English), Qwen3-Embedding-8B (multilingual) named as leaders. Check live board | - | [HF MTEB](https://huggingface.co/spaces/mteb/leaderboard) |
| Qwen3-Embedding (open) | Open multilingual embeddings + rerankers | Cited as multilingual/code leader | - | [huggingface.co/Qwen](https://huggingface.co/Qwen) |
| BGE (BGE-M3, bge-reranker) | Open dense+sparse+multi-vector, rerankers | Frequently cited open model | - | [huggingface.co/BAAI](https://huggingface.co/BAAI) |
| Gemini Embedding | Hosted embeddings | Cited as top English | - | [ai.google.dev](https://ai.google.dev/gemini-api/docs/embeddings) |
| Voyage (voyage-4 family), Cohere Embed/Rerank, OpenAI text-embedding-3 | Hosted embeddings / rerankers | Commercial options | - | [voyageai.com](https://www.voyageai.com/) |
| sentence-transformers | Library to run bi-/cross-encoders | 6.2.0 here | 6.2.0 | [sbert.net](https://www.sbert.net/) |

## 9. Vector databases
**Changed since 2024:** vendors added strong quantization (Milvus RaBitQ 1-bit, Qdrant TurboQuant per one 2026 article), hybrid search with built-in sparse + BM25, pgvector gained sparse vectors per one source; "just use pgvector" is a respected default below very large scale.

| Tool | What it is for | Version seen (3rd-party, unverified) | Status / note | In .venv | Source |
|---|---|---|---|---|---|
| Qdrant | Rust vector DB, filtering, hybrid | server 1.18 mentioned in a 2026 article | Active; client installed | qdrant_client 1.20.0 | [qdrant.tech](https://qdrant.tech/documentation/) |
| Milvus | Distributed vector DB | 2.6 line cited | Active | - | [milvus.io](https://milvus.io/docs) |
| Weaviate | Vector DB with hybrid BM25+vector | see releases (a source said 1.20; treat as unverified) | Active | - | [weaviate.io](https://weaviate.io/developers/weaviate) |
| pgvector | Postgres extension | 0.9 cited (Apr 2026, unverified) | Default for Postgres shops | - | [github.com/pgvector/pgvector](https://github.com/pgvector/pgvector) |
| Chroma | Simple embedded vector DB | see PyPI/releases | Popular for prototypes | - | [trychroma.com](https://www.trychroma.com/) |
| LanceDB | Embedded DB on Lance columnar format | see PyPI/releases | Rising | - | [lancedb.com](https://lancedb.com/) |
| FAISS | Similarity-search library, not a DB | see releases | Still the reference library | - | [github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss) |

## 10. RAG and orchestration frameworks
**Changed since 2024:** LangChain 1.0 (Oct 2025) with LangGraph as the agent runtime; LangChain "chains" are legacy; DSPy treats prompts as optimizable programs.

| Tool | What it is for | Latest version seen (date) | Status / note | In .venv | Source |
|---|---|---|---|---|---|
| LangChain | LLM app building blocks | 1.3.18 (Sep 2026, 3rd-party); other blog said 1.3.2 | 1.x stable line | - | [PyPI langchain](https://pypi.org/project/langchain/) |
| LangGraph | Stateful agent graphs | 1.2.2 (3rd-party blog) | Recommended agent runtime in LangChain ecosystem | - | [PyPI langgraph](https://pypi.org/project/langgraph/) |
| LlamaIndex | Data/RAG framework | 0.14.22 (May 2026, 3rd-party) | Active | - | [PyPI llama-index](https://pypi.org/project/llama-index/) |
| Haystack | Pipelines for RAG (deepset) | 2.29.0 (May 2026, 3rd-party) | Active | - | [haystack.deepset.ai](https://haystack.deepset.ai/) |
| DSPy | Programmatic prompt/pipeline optimization | see PyPI/releases | Active | - | [dspy.ai](https://dspy.ai/) |

## 11. Chunking libraries
**Changed since 2024:** Dedicated light libraries (Chonkie) and structure-aware chunkers inside document converters (Docling) became common; late chunking and LLM-based chunkers exist but are costlier. See module 19.

| Tool | What it is for | Version | Status / note | In .venv | Source |
|---|---|---|---|---|---|
| Chonkie | Light chunking lib: Token, Recursive, Semantic, SDPM, Late, Code, Table, Slumber chunkers | see PyPI/releases | Active | - | [docs.chonkie.ai](https://docs.chonkie.ai/oss/chunkers/overview) |
| semchunk | Fast semantic-aware splitting | see PyPI/releases | Small, fast | - | [PyPI semchunk](https://pypi.org/project/semchunk/) |
| Docling chunkers | Hierarchical + hybrid chunking from parsed structure (IBM) | see PyPI/releases | Good after Docling parsing | - | [docling-project.github.io](https://docling-project.github.io/docling/) |
| Unstructured | Parsing + structure-aware chunking (by title) | see PyPI/releases | Established | - | [docs.unstructured.io](https://docs.unstructured.io/) |
| LangChain / LlamaIndex splitters | Recursive character/token splitters | with framework | Baseline everyone compares to | - | [LangChain text splitters](https://python.langchain.com/docs/concepts/text_splitters/) |

## 12. OCR and document parsing
**Changed since 2024:** small (about 1-3B) document vision-language models now lead open benchmarks over classic pipelines; Tesseract remains the CPU baseline. Scores below are 3rd-party OmniDocBench snapshots (version of benchmark varies); verify on your own documents. See module 18.

| Tool | What it is for | Snapshot claim (3rd-party) | Status / note | In .venv | Source |
|---|---|---|---|---|---|
| Tesseract (+OCRmyPDF) | Classic CPU OCR | n/a | Baseline for CPU-only | - | [github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract) |
| PaddleOCR / PaddleOCR-VL | OCR toolkit; 0.9B document VLM | PaddleOCR-VL ranked first on OmniDocBench v1.6 (96.34) in an Aug 2026 Roboflow guide | Active | - | [github.com/PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) |
| MinerU | PDF to Markdown/JSON, VLM backend | MinerU2.5-Pro 1.2B at 95.75 (same guide) | Weights have usage thresholds per one guide | - | [github.com/opendatalab/MinerU](https://github.com/opendatalab/MinerU) |
| GLM-OCR | 0.9B document OCR model | 95.22 (same guide) | Permissive license per one guide | - | [huggingface.co/zai-org](https://huggingface.co/zai-org) |
| Docling | Document conversion to structured form (IBM) | n/a | MIT-licensed, feeds chunkers | - | [docling-project.github.io](https://docling-project.github.io/docling/) |
| olmOCR | AllenAI VLM OCR for PDFs | 7B, ~85.7 (one table) | Large, accurate on detail | - | [github.com/allenai/olmocr](https://github.com/allenai/olmocr) |
| DeepSeek-OCR (and OCR 2) | Optical-compression VLM OCR | ~90-91 (conflicting sources) | Fast | - | [github.com/deepseek-ai/DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) |
| dots.ocr | 3B multilingual layout+OCR VLM | 90.77 (one table) | Active | - | [github.com/rednote-hilab/dots.ocr](https://github.com/rednote-hilab/dots.ocr) |
| Marker | PDF to Markdown (Datalab) | no figure found | Weights have revenue thresholds per one guide | - | [github.com/datalab-to/marker](https://github.com/datalab-to/marker) |
| Mistral OCR | Hosted OCR API | n/a | Commercial | - | [docs.mistral.ai](https://docs.mistral.ai/capabilities/document_ai/) |
| Cloud OCR (AWS Textract, Google Document AI, Azure Document Intelligence) | Managed IDP | n/a | Commercial, compliance friendly | - | [AWS Textract](https://aws.amazon.com/textract/) |
| pdfplumber / pypdfium2 / pdfminer.six | Text-layer PDF extraction (no OCR) | n/a | Use first when a PDF has a text layer | pdfplumber 0.11.10, pypdfium2 5.14.0, pdfminer_six 20260107 | [PyPI pdfplumber](https://pypi.org/project/pdfplumber/) |

## 13. Agents and protocols
**Changed since 2024:** MCP was donated (Dec 2025) to the Agentic AI Foundation under the Linux Foundation (founded by Anthropic, Block, OpenAI); per 3rd-party posts the spec revision dated **2026-07-28** has a stateless core, and one source says MCP hit v1.0 in April 2026. Vendor SDKs (OpenAI Agents SDK, Claude Agent SDK) joined LangGraph and CrewAI.

| Tool | What it is for | Latest version seen (date) | Status / note | In .venv | Source |
|---|---|---|---|---|---|
| MCP | Open protocol for tools/resources/prompts across apps | spec 2026-07-28 (3rd-party reports; verify) | Industry standard | - | [modelcontextprotocol.io](https://modelcontextprotocol.io/) |
| Claude Agent SDK | Anthropic's agent loop + tools SDK | see PyPI/releases (not found by search) | Active | anthropic 1.15.0 (the plain API client, not the Agent SDK) | [docs.claude.com](https://docs.claude.com/) |
| OpenAI Agents SDK | Agents, handoffs, guardrails, tracing | Python 0.15.1 (2 May 2026, 3rd-party) | Active | - | [openai.github.io/openai-agents-python](https://openai.github.io/openai-agents-python/) |
| LangGraph | Graph-based agent runtime | 1.2.2 (3rd-party) | Active | - | [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/) |
| CrewAI | Role-based multi-agent | 1.15.10 (31 Jul 2026, 3rd-party, unverified) | Active, 1.x | - | [github.com/crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) |

## 14. APIs and web frameworks
**Changed since 2024:** FastAPI is still 0.x (no 1.0 seen); Pydantic v2 is the baseline; Starlette reached 1.x (seen in venv); uvicorn/gunicorn pair unchanged.

| Tool | What it is for | Latest version seen (date) | Status / note | In .venv | Source |
|---|---|---|---|---|---|
| FastAPI | Async API framework with typed validation | 0.146.0 installed here; web: 0.139.1 (16 Jul 2026) | Default for ML APIs | 0.146.0 | [fastapi.tiangolo.com](https://fastapi.tiangolo.com/) |
| Starlette | ASGI toolkit under FastAPI | see PyPI | 1.x | 1.9.0 | [starlette.io](https://www.starlette.io/) |
| Flask | Sync WSGI microframework | see PyPI/releases | Stable, simple | 3.1.3 | [flask.palletsprojects.com](https://flask.palletsprojects.com/) |
| Uvicorn | ASGI server | see PyPI/releases | Standard | 0.54.0 | [uvicorn.org](https://www.uvicorn.org/) |
| Gunicorn | WSGI/process manager | see PyPI/releases | Standard | 26.2.0 | [gunicorn.org](https://gunicorn.org/) |
| Pydantic v2 | Data validation | see PyPI/releases | v2 only | 2.14.1 | [docs.pydantic.dev](https://docs.pydantic.dev/) |
| httpx | HTTP client | see PyPI/releases | Standard | 0.28.1 | [python-httpx.org](https://www.python-httpx.org/) |

## 15. UI
| Tool | What it is for | Latest version seen (date) | Status / note | In .venv | Source |
|---|---|---|---|---|---|
| Streamlit | Script-to-app dashboards | 1.66.0 installed here | Standard | 1.66.0 | [streamlit.io](https://streamlit.io/) |
| Gradio | ML demo UIs | 6.30.0 installed here (Gradio 6 line) | Standard for HF Spaces | 6.30.0 | [gradio.app](https://www.gradio.app/) |

**Changed since 2024:** Gradio moved to a 6.x major line; Streamlit continued 1.x.

## 16. Docker and containers
**Changed since 2024:** Docker Compose v2 (`docker compose`) only; BuildKit default; NVIDIA Container Toolkit for GPUs; Docker Model Runner and `docker init` appeared. I did not run a search for container versions, so no numbers are given.

| Tool | What it is for | Version | Note | In .venv | Source |
|---|---|---|---|---|---|
| Docker Engine / Desktop | Build and run containers | see releases | Standard | - | [docs.docker.com](https://docs.docker.com/) |
| Docker Compose | Multi-container apps | see releases | `docker compose` (v2) | - | [docs.docker.com/compose](https://docs.docker.com/compose/) |
| NVIDIA Container Toolkit | GPU access in containers | see releases | Needed for GPU containers | - | [docs.nvidia.com](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) |
| Podman | Daemonless Docker alternative | see releases | Alternative | - | [podman.io](https://podman.io/) |

## 17. Observability
**Changed since 2024:** OpenTelemetry `gen_ai.*` semantic conventions exist but were still marked experimental/in development per a Feb 2026 source (semconv v1.40.0); Langfuse is reported to be owned by ClickHouse since Jan 2026 (3rd-party).

| Tool | What it is for | Status / note (3rd-party) | In .venv | Source |
|---|---|---|---|---|
| OpenTelemetry GenAI semconv | Vendor-neutral LLM trace attributes | Experimental per Feb 2026 source | opentelemetry_api 1.45.1 | [opentelemetry.io](https://opentelemetry.io/docs/specs/semconv/gen-ai/) |
| Langfuse | OSS (MIT) LLM tracing, evals, prompts | OTLP HTTP endpoint; ClickHouse-owned (reported) | - | [langfuse.com](https://langfuse.com/) |
| LangSmith | Commercial tracing/evals (LangChain) | Deepest LangChain integration | - | [smith.langchain.com](https://smith.langchain.com/) |
| Arize Phoenix | Tracing/evals on OpenInference | Elastic License 2.0 (source-available) | - | [phoenix.arize.com](https://phoenix.arize.com/) |

## 18. Frontier and open model families (names only, late 2026)
Search results were blogs and aggregators with conflicting dates. This list gives **families to recognise**; open the official page for real specs. Anything marked (reported) is unverified.

| Family | Maker | Open weights? | Official page |
|---|---|---|---|
| GPT-5.x (GPT-5.5 reported Apr 2026), gpt-oss | OpenAI | gpt-oss yes, GPT-5.x no | [openai.com](https://openai.com/) |
| Claude (Opus 4.7, Sonnet 4.6 reported; Mythos Preview restricted, reported) | Anthropic | No | [anthropic.com](https://www.anthropic.com/) |
| Gemini 3.x (3.1 Pro, 3.5 Flash reported) | Google | No | [deepmind.google](https://deepmind.google/) |
| Gemma (Gemma 4 reported Apr 2026) | Google | Yes | [ai.google.dev/gemma](https://ai.google.dev/gemma) |
| Llama 4 family; Muse Spark (reported proprietary) | Meta | Llama yes; Muse Spark no (reported) | [ai.meta.com](https://ai.meta.com/) |
| DeepSeek V4 (V4-Pro, V4-Flash reported Apr 2026, MIT) | DeepSeek | Yes | [deepseek.com](https://www.deepseek.com/) |
| Qwen 3.x | Alibaba | Mostly yes | [qwen.ai](https://qwen.ai/) |
| GLM-5 (reported) | Zhipu / Z.ai | Yes | [z.ai](https://z.ai/) |
| Kimi K2.x (K2.6 reported) | Moonshot | Yes | [moonshot.ai](https://www.moonshot.ai/) |
| Mistral family | Mistral AI | Mixed | [mistral.ai](https://mistral.ai/) |
| Inkling (reported Jul 2026, Apache 2.0) | Thinking Machines | Yes (reported) | [thinkingmachines.ai](https://thinkingmachines.ai/) |

---

## Could not verify (summary)
Official latest versions of: PyTorch (only the installed 2.14.1 is real), vLLM, uv, ruff, TRL, Unsloth, Axolotl, LLaMA-Factory, LangChain/LangGraph, FastAPI, Ollama, llama.cpp, CrewAI, lm-eval-harness; torchtune's official status; Claude Agent SDK, bitsandbytes, Inspect, RAGAS, DeepEval, Chonkie, Docker versions; MTEB current ranking; all model-family release dates. Web search results were stale or third-party; always confirm on the linked PyPI/GitHub page.
