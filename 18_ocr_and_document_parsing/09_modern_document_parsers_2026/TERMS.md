# Terms — Modern document parsers (2026)

Folder: `18_ocr_and_document_parsing/09_modern_document_parsers_2026` · Files covered: `main.py`, `reference_engines.py`, `WALKTHROUGH.md`
Back to module: [18 README](../README.md)

**Terms in this folder:** [Classic OCR engine](#classic-ocr-engine) · [Pipeline toolkit (modular parser)](#pipeline-toolkit-modular-parser) · [End-to-end VLM OCR](#end-to-end-vlm-ocr) · [Cloud document AI](#cloud-document-ai) · [Vision tokens](#vision-tokens) · [OCR hallucination (omission, fabrication, loops)](#ocr-hallucination-omission-fabrication-loops) · [OmniDocBench](#omnidocbench) · [olmOCR-Bench](#olmocr-bench) · [Weight memory arithmetic](#weight-memory-arithmetic) · [Hard filter vs soft weight](#hard-filter-vs-soft-weight) · [Gold set (your own 50 pages)](#gold-set-your-own-50-pages) · [Guarded optional import](#guarded-optional-import) · [OpenAI-compatible endpoint (served VLM)](#openai-compatible-endpoint-served-vlm)

---

## Classic OCR engine  `(concept)`
**1. Definition:** A text-detection model plus a text-recognition model that outputs lines, boxes and confidence: Tesseract, PaddleOCR (PP-OCRv5), EasyOCR.
**2. In plain words:** Reads text, nothing more: no tables or reading order.
**3. Example to quote:** > "On a CPU laptop with scans I start with PaddleOCR or Tesseract."
**Seen in:** `main.py`, `TOOLS`

## Pipeline toolkit (modular parser)  `(concept)`
**1. Definition:** A toolkit chaining layout detection, reading order, table, formula and OCR models into Markdown/JSON: Docling, MinerU, marker, PP-StructureV3.
**2. In plain words:** Several specialised models glued together.
**3. Example to quote:** > "Pipelines give structure, but errors compound between stages."
**Seen in:** `main.py`, `TOOLS`

## End-to-end VLM OCR  `(concept)`
**1. Definition:** One vision-language model reads a page image and generates Markdown/HTML/LaTeX: PaddleOCR-VL, olmOCR-2, DeepSeek-OCR, dots.ocr, GLM-OCR, GOT-OCR2.0.
**2. In plain words:** A model that writes the page instead of copying it.
**3. Example to quote:** > "VLM OCR gives layout in one pass but can hallucinate or skip text."
**Seen in:** `main.py`, `TOOLS`

## Cloud document AI  `(concept)`
**1. Definition:** Managed APIs for OCR, tables, forms and extractors: Amazon Textract, Azure Document Intelligence, Google Document AI.
**2. In plain words:** Pay per page, no infrastructure, data leaves your network.
**3. Example to quote:** > "Cloud is a hard-filter question first: may the documents leave?"
**Seen in:** `main.py`, `TOOLS`

## Vision tokens  `(concept)`
**1. Definition:** The patch embeddings a vision encoder passes to the language decoder; their number drives memory and cost.
**2. In plain words:** How many pieces the page is chopped into.
**3. Example to quote:** > "DeepSeek-OCR studies optical compression: far fewer vision tokens than text tokens."
**Seen in:** Theory section

## OCR hallucination (omission, fabrication, loops)  `(error)`
**1. Definition:** A generating model outputs text not on the page, drops text that is, or repeats a phrase.
**2. In plain words:** Fluent but wrong.
**3. Example to quote:** > "I cross-check numbers with a classic engine and use field-level checks."
**Seen in:** Theory section

## OmniDocBench  `(concept)`
**1. Definition:** A document parsing benchmark (CVPR 2025, OpenDataLab): text edit distance, formula CDM, table TEDS, reading-order edit distance.
**2. In plain words:** A page-level scoreboard with versions.
**3. Example to quote:** > "Scores across versions are not comparable: verify version, date and runner."
**Seen in:** `main.py`, `BENCHMARKS`

## olmOCR-Bench  `(concept)`
**1. Definition:** Ai2's pass/fail unit-test benchmark for PDF-to-text.
**2. In plain words:** Does the phrase appear, are the lines in order?
**3. Example to quote:** > "Strict about exact text; verify the current release."
**Seen in:** `main.py`, `BENCHMARKS`

## Weight memory arithmetic  `(function)`
**1. Definition:** params x bytes per parameter: bf16 = 2 bytes, 4-bit about 0.5; weights only, no KV cache or activations.
**2. In plain words:** Can the model even be loaded?
**3. Example to quote:** > "A 3B model is 5.6 GiB in bf16: it does not fit 8 GB with the OS, but fits in 4-bit."
```python
weight_gb = params_b * 1e9 * bytes_per_param / 2**30
```
**Seen in:** `main.py`, `weight_gb`, `fits_report`

## Hard filter vs soft weight  `(concept)`
**1. Definition:** A constraint that removes candidates (privacy, no GPU) versus a preference that adds score.
**2. In plain words:** A weight can be outvoted; a filter cannot.
**3. Example to quote:** > "Privacy is a filter, not a weight."
```python
if need.get("onprem_required") and not t.onprem: continue
```
**Seen in:** `main.py`, `recommend`

## Gold set (your own 50 pages)  `(concept)`
**1. Definition:** A small, hand-checked evaluation set drawn from your real documents.
**2. In plain words:** The only benchmark that predicts your production quality.
**3. Example to quote:** > "I sample 50 pages, score CER, field accuracy and TEDS, and read the worst 10."
**Seen in:** `main.py`, `BENCHMARKS`

## Guarded optional import  `(syntax)`
**1. Definition:** Check availability with importlib.util.find_spec and skip with a message; downloads additionally need an explicit environment flag.
**2. In plain words:** Never install or download behind the user's back.
**3. Example to quote:** > "Reference scripts run only with the package installed and ALLOW_MODEL_DOWNLOAD=1."
```python
importlib.util.find_spec("docling") is not None
```
**Seen in:** `reference_engines.py`, `have`

## OpenAI-compatible endpoint (served VLM)  `(api)`
**1. Definition:** A server (vLLM, SGLang, Ollama and others) exposing /v1/chat/completions so any client can send an image and a prompt.
**2. In plain words:** A standard doorway to a model you host.
**3. Example to quote:** > "I serve the OCR VLM with vLLM and call it with a base64 image."
```python
{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
```
**Seen in:** `reference_engines.py`, `vlm_ocr_openai_compatible`
