# Step 09 — Modern document parsers (2026): which tool for which document?

## The story so far
[Step 08](../08_pdf_internals_and_routing/WALKTHROUGH.md) built the router: native pages use the text layer, scans go to OCR. It sent the hard pages to our template OCR from steps 03-05, a teaching engine that reads one bundled font. Real documents need something much stronger, and in 2026 there are many candidates.

## The pain point
"Which OCR should we use?" has no single answer, and the answer ages in months. A table-heavy scientific PDF, a stack of faded multilingual invoices, a million-page archive and a customer-facing form-filling flow need different tools, different hardware and different risk profiles (a VLM can invent text; a cloud API sends your documents away). Choosing from a leaderboard screenshot, or installing the trendiest model on a laptop that cannot hold it, wastes weeks.

## The spark
Do not memorise names; learn the **four families** and what separates them: pipeline toolkits (several models glued together), classic engines (detect + recognise text, nothing more), end-to-end vision-language models (one model reads the page and writes Markdown) and managed cloud APIs. Then decide with explicit requirements: document type, hardware, privacy, volume, structure needed. This step encodes that reasoning as data plus a transparent scoring helper you can argue with, and checks the arithmetic of what fits in 8 GB.

## Goal
Describe 15 tools as data, generate the comparison table, compute weight memory for the VLMs (what fits an 8 GB laptop), run a rule-based recommender on six scenarios, and list guarded reference calls (never auto-run, never auto-download). Numbers about quality are deliberately absent: learn how to check them, not what they were last month.

## The theory (read this before the interview)
**Epistemic status of this step.** Fundamentals (the four families, why VLM OCR hallucinates, what benchmarks measure) are stable. The *names and rankings* are current industry practice or emerging as of Nov 2026 and **will move**: the
versions below come from our research notes, everything marked "verify" must be checked at the source before you rely on it. Benchmark scores are intentionally not quoted.

**The four families.**
| Family | Idea | Examples (2026) | Strength | Weakness |
|---|---|---|---|---|
| Classic OCR engine | text **detection** model + text **recognition** model (CRNN-like, step 05), outputs lines + boxes + confidence | Tesseract (LSTM, 100+ languages, CPU), PaddleOCR 3.x with PP-OCRv5 (Apache-2.0; 3.7.0 in our notes), EasyOCR (CRAFT + CRNN) | small, fast on CPU, per-line confidence, deterministic, no hallucination | no tables, formulas or reading order of its own; needs preprocessing (steps 01-02) |
| Pipeline toolkit | layout detector + reading order + table model + OCR + formula model, assembled into Markdown/JSON | Docling (IBM; MIT; 2.147.0 in our notes), MinerU (4.1.7), marker (marker-pdf 2.1.0), PaddleOCR PP-StructureV3 | structure out of the box, uses the text layer when present, mix-and-match components | many models to install, GPU preferred for the heavy ones, error compounding between stages |
| End-to-end VLM OCR | one vision-language model (vision encoder + LLM decoder) reads a page image and **writes** Markdown/HTML/LaTeX | PaddleOCR-VL (0.9B in the first release; PaddleOCR-VL-1.6 reported top on OmniDocBench v1.6 by a third-party report: verify), olmOCR-2 (8B), DeepSeek-OCR (3B), dots.ocr (3B), GLM-OCR (0.9B), GOT-OCR2.0 (2024, ~0.58B) | layout, order, tables, formulas in one pass; handles messy pages and sometimes handwriting; fine-tunable | needs a GPU for real volume; can **hallucinate, skip or loop**; no calibrated confidence; token cost per dense page |
| Cloud API | managed OCR + tables + forms + prebuilt extractors | Amazon Textract, Azure Document Intelligence, Google Document AI | no infra, key-value/form extraction and custom extractors, SLAs | per-page cost, data leaves your network (privacy and residency), vendor lock-in, black box |

**How the VLM family works, in one paragraph.** A vision encoder (a ViT, like [module 15's](../../15_neural_networks/README.md) CNN idea replaced by patches and attention) turns the page into a few hundred to a few thousand "vision tokens"; a language-model decoder (hundreds of millions to 8B
parameters) then generates the page as text autoregressively, exactly like [module 04](../../04_llm_fundamentals/README.md). Designs differ: PaddleOCR-VL-style systems first run a layout detector and let the VLM read each cropped element (shorter outputs, easier to attach boxes);
olmOCR-style systems read the whole page and are trained for fidelity with unit-test-style rewards; DeepSeek-OCR studies **optical context compression**: a page needs far fewer vision tokens than the text tokens it contains, which matters for long-context cost. The consequence of "generate, don't copy": no per-character confidence, possible repetition loops, dropped
headers/footers, "corrected" typos (bad for legal and financial text). Mitigations: constrained decoding or short element-level crops, a second pass with a classic engine to cross-check numbers, and field-level checks (step 06).

**Hardware reality (arithmetic, not benchmarks).** Weights alone need `params x bytes per parameter`: bf16 is 2 bytes, 4-bit about 0.5. On an 8 GB laptop with about 3 GiB kept for the OS and Python: a 0.9B model is 1.7 GiB in bf16 (fits), a 3B model is 5.6 GiB (does not fit in bf16, fits in 4-bit at 1.4 GiB), an 8B model is 14.9 GiB in bf16 (needs 4-bit at 3.7 GiB or a GPU). That ignores the **vision tokens' KV cache** and activations, which grow with page size; and CPU generation of a page of Markdown (thousands of tokens) is minutes per page, not seconds. So: classic engines and Docling's CPU path for laptops; VLMs on a GPU server (vLLM/SGLang, module 21); quantised small VLMs only for low volume. Quantisation
and serving: [module 21](../../21_llm_inference_engineering/).

**Decision flow.**
```
born-digital PDF page?  --yes-->  text layer (step 08) + layout/tables from Docling / MinerU / marker
        | no (scan / image)
data may leave the company?  --yes, small volume, forms/invoices-->  cloud API (Textract / Azure DI / Document AI)
        | no
GPU available and volume high / layout complex?  --yes-->  VLM OCR (PaddleOCR-VL, olmOCR-2, dots.ocr, ...) or MinerU/marker on GPU
        | no
CPU only:  PaddleOCR (best default), Tesseract (huge language coverage, tune psm), Docling with an OCR backend for structure
always:  sample 50 of YOUR pages, score CER/WER + field accuracy + table TEDS (step 06), read the worst 10
```

**Licences (state only what you are sure of; verify the rest).** Tesseract, PaddleOCR and EasyOCR are Apache-2.0; Docling is MIT. MinerU's licence has changed over time (it started AGPL-based): verify the current one. marker's code and its model weights carry different terms (check the weights licence before
commercial use). Several VLM OCR checkpoints are Apache-2.0 or MIT but some carry custom terms: always read the model card. Cloud services are commercial APIs with their own data-processing terms.

**Evaluation (and why leaderboard numbers need care).** OmniDocBench (CVPR 2025, OpenDataLab) scores full-page parsing: text edit distance, formula CDM, table TEDS, reading-order edit distance, over many document types. olmOCR-Bench (Ai2) uses pass/fail unit tests. Both have **versions**; scores across versions are not comparable; many numbers in vendor blogs are
self-reported; a model can be tuned to a benchmark's document mix. Treat the claim "PaddleOCR-VL-1.6 is top on OmniDocBench v1.6" as a lead to **verify** (leaderboard, date, who ran it), not a fact to quote. The only number that predicts your production quality is a score on your own gold set.

**Cost and operations.** Cloud: pay per page (check current pricing; it differs per feature: plain OCR vs tables vs forms). Self-hosted VLM: GPU-hours divided by pages per hour; batch offline jobs fill GPUs best. Classic engines: CPU cores. Always add: retries, timeouts, per-page routing (step 08), a cache keyed by file hash, and a review queue for low-confidence pages.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| OCR engine | text recogniser, OCR library | near | Detect + recognise text; no document structure |
| Document parsing | document conversion, PDF-to-Markdown, document AI, layout parsing | near | OCR plus layout, tables, formulas, reading order, output as Markdown/JSON |
| Pipeline (modular) parser | multi-stage, cascade, toolkit | yes | Several specialised models; errors compound between stages |
| End-to-end VLM OCR | OCR-free parsing, OCR-VLM, page-to-Markdown model | near | One vision-language model generates the document text |
| Vision tokens | image tokens, visual tokens, patch embeddings | near | What the VLM's encoder hands the decoder; count drives cost |
| Hallucination (in OCR) | fabrication, omission, repetition loop | near | Output text that is not on the page, or missing text that is |
| OmniDocBench / olmOCR-Bench | document parsing benchmarks | no | Different metrics (edit distance/TEDS/CDM vs unit tests) and versions |
| TEDS | tree-edit-distance-based similarity | yes | Table structure + content similarity ([step 06](../06_ocr_evaluation/WALKTHROUGH.md)) |
| Document AI (cloud) | intelligent document processing (IDP), form recogniser | near | Managed extraction services with prebuilt processors |

## Run it
```bash
# from this folder
python main.py                                     # table + fit check + six scenarios; stdlib only
SHOW_NOTES=1 python main.py                        # also print the one-line notes per tool
python reference_engines.py                        # optional, guarded: lists which engines are importable; nothing downloads
# from the repo root
.venv/bin/python 18_ocr_and_document_parsing/09_modern_document_parsers_2026/main.py
```
Under a second on a CPU laptop. `reference_engines.py` (labelled REFERENCE) runs nothing heavy unless you set `ALLOW_MODEL_DOWNLOAD=1` AND the package is installed; nothing is installed in this course, so every engine prints a skip line.

## Code, step by step

### 1. Imports  (`main.py` lines 15–16)
```python
import os
from dataclasses import dataclass
```
- **In simple words:** Only `os` (for one env knob) and `dataclasses`.
- **Exactly:** No numpy, no torch: this step must run anywhere, instantly.
- **Why it is here:** The landscape is data and a few rules; heavy imports would be dishonest here.

### 2. The Tool record  (`main.py` lines 25–39)
```python
@dataclass
class Tool:
    name: str
    family: str
    params_b: float | None
    cpu_ok: bool            # usable on a laptop CPU for small jobs (not necessarily fast)
    onprem: bool            # can run without sending documents to a third party
    licence: str
    scans: int
    born_digital: int
    tables: int
    formulas: int
    multilingual: int
    forms_kv: int           # key-value / form field extraction
    note: str
```
- **In simple words:** A dataclass with the facts we compare: family, parameters, CPU usability, on-prem, licence, six 0-2 skill scores and a note.
- **Exactly:** `@dataclass` generates `__init__`, `__repr__` and `__eq__` from the annotations. `float | None` is a union type (Python 3.10+). The 0-2 scores are **course judgement** from public descriptions, not measurements (the comment says so).
- **Why it is here:** Putting the landscape in typed data makes it editable and testable: your own team's scores are one edit away.

### 3. The 15 tools  (`main.py` lines 42–73)
```python
TOOLS = [
    Tool("Tesseract", "engine", None, True, True, "Apache-2.0", 1, 0, 0, 0, 2, 0,
         "LSTM line recogniser + classic layout analysis; weak on tables, formulas, columns, photos"),
    Tool("PaddleOCR 3.x (PP-OCRv5)", "engine", None, True, True, "Apache-2.0", 2, 0, 0, 0, 2, 0,
         "Text detection + recognition models, small and fast; CPU works, GPU faster (3.7.0 as of Nov 2026)"),
    Tool("EasyOCR", "engine", None, True, True, "Apache-2.0", 1, 0, 0, 0, 2, 0,
         "CRAFT detector + CRNN recogniser in PyTorch; easy API, slower than PaddleOCR on CPU, little maintenance news"),
    Tool("Docling", "toolkit", None, True, True, "MIT", 1, 2, 2, 1, 1, 0,
         "IBM layout model + TableFormer + optional OCR; uses the PDF text layer when present; DoclingDocument -> Markdown/JSON (2.147.0)"),
    Tool("MinerU", "toolkit", None, False, True, "verify (changed over time)", 2, 2, 2, 2, 2, 0,
         "Pipeline backend + VLM backend, strong on formulas and Chinese/English papers; wants a GPU (4.1.7)"),
    Tool("marker", "toolkit", None, False, True, "verify (code and weights licensed differently)", 1, 2, 2, 2, 1, 0,
         "Surya layout/OCR models + optional LLM post-processing; fast on GPU, slow on CPU (marker-pdf 2.1.0)"),
    Tool("PaddleOCR-VL", "vlm", 0.9, False, True, "verify (Apache-2.0 reported)", 2, 2, 2, 2, 2, 0,
         "Layout model + ~1B VLM element reader; PaddleOCR-VL-1.6 reported top on OmniDocBench v1.6 (third-party report: verify)"),
    Tool("olmOCR-2", "vlm", 8.0, False, True, "verify", 2, 1, 2, 2, 1, 0,
         "Ai2 fine-tune of a Qwen-VL model with RL on unit-test rewards; built for bulk PDF -> text on GPU clusters (8B per the notes)"),
    Tool("DeepSeek-OCR", "vlm", 3.0, False, True, "verify", 2, 1, 1, 1, 1, 0,
         "Optical context compression: a vision encoder shrinks a page into few vision tokens for a mixture-of-experts decoder (3B total per the notes)"),
    Tool("dots.ocr", "vlm", 3.0, False, True, "verify", 2, 1, 2, 2, 2, 0,
         "One ~3B VLM for layout detection + recognition, multilingual"),
    Tool("GLM-OCR", "vlm", 0.9, False, True, "verify", 2, 1, 1, 1, 1, 0,
         "Smallest of the VLM group (0.9B per the notes): the best candidate for a modest GPU or an optimised CPU build"),
    Tool("GOT-OCR2.0", "vlm", 0.58, False, True, "verify (Apache-2.0 reported)", 2, 1, 1, 2, 1, 0,
         "Early (2024) unified OCR transformer: text, formulas, sheet music, charts; older than the rest of the group"),
    Tool("AWS Textract", "cloud", None, True, False, "commercial API", 2, 1, 2, 0, 1, 2,
         "Managed OCR + tables + forms + queries; pay per page; documents leave your network"),
    Tool("Azure Document Intelligence", "cloud", None, True, False, "commercial API", 2, 1, 2, 1, 1, 2,
         "Managed layout + prebuilt invoice/receipt/ID models + custom models; pay per page"),
    Tool("Google Document AI", "cloud", None, True, False, "commercial API", 2, 1, 2, 1, 2, 2,
         "Managed OCR + processors for invoices, forms, custom extractors; pay per page"),
]
```
- **In simple words:** One `Tool(...)` per entry: three engines, three toolkits, six VLM OCR models, three cloud APIs.
- **Exactly:** Positional arguments follow the dataclass field order. `licence` is stated only where we are sure; everything else says `verify`. Sizes marked in billions of parameters come from the research notes of this course (olmOCR-2 8B, DeepSeek-OCR 3B, dots.ocr 3B, GLM-OCR 0.9B; PaddleOCR-VL 0.9B in its first release; GOT-OCR2.0 about 0.58B).
- **Why it is here:** These rows are the part that goes stale first: re-check them before using the table in a decision.

### 4. Benchmarks, with their caveats  (`main.py` lines 75–82)
```python
BENCHMARKS = [
    ("OmniDocBench (CVPR 2025, OpenDataLab)", "page-level parsing: text edit distance, formula CDM, table TEDS, reading-order edit distance",
     "numbers depend on the version (v1.5, v1.6 ...) and on who ran the model: verify the leaderboard before quoting"),
    ("olmOCR-Bench (Ai2)", "pass/fail unit tests on PDF pages: a phrase must be present, two lines must be in this order, a table cell must match",
     "tests are strict about exact text; verify the current release and its test count before quoting"),
    ("Your own 50 pages", "CER/WER + field accuracy + table TEDS on documents like YOURS (step 06)",
     "the only number that predicts your production quality"),
]
```
- **In simple words:** Three evaluation sources, each with what it measures and a warning.
- **Exactly:** A list of 3-tuples; printed later. No scores are stored on purpose.
- **Why it is here:** A benchmark number without version, date and runner is not evidence.

### 5. markdown_table: comparison table generator  (`main.py` lines 86–97)
```python
def size_str(t):
    return f"{t.params_b:g}B params" if t.params_b else "small models" if t.family in ("engine", "toolkit") else "service"


def markdown_table(tools):
    rows = ["| Tool | Family | Size | Runs on a CPU laptop? | On-prem? | Licence | Scans | Tables | Formulas |",
            "|---|---|---|---|---|---|---|---|---|"]
    stars = lambda n: "+" * n if n else "-"
    for t in tools:
        rows.append(f"| {t.name} | {t.family} | {size_str(t)} | {'yes' if t.cpu_ok else 'slow / GPU advised'} | {'yes' if t.onprem else 'no (cloud)'} "
                    f"| {t.licence} | {stars(t.scans)} | {stars(t.tables)} | {stars(t.formulas)} |")
    return "\n".join(rows)
```
- **In simple words:** Render selected fields of every tool as a Markdown table.
- **Exactly:** `stars` is a lambda turning 0-2 into `-`, `+`, `++`. f-strings build each row; `"\n".join` assembles the table. Output can be pasted into a README.
- **Why it is here:** A generated table cannot drift from the data.

### 6. fits_report: what fits in 8 GB (weights only)  (`main.py` lines 101–113)
```python
def weight_gb(params_b, bytes_per_param):
    return params_b * 1e9 * bytes_per_param / 2**30        # GiB


def fits_report(ram_gb=8, reserve_gb=3):
    """A laptop with `ram_gb` RAM keeps ~`reserve_gb` for the OS, Python and activations. KV cache / image tokens add more."""
    out = []
    for t in TOOLS:
        if t.params_b:
            bf16, int4 = weight_gb(t.params_b, 2), weight_gb(t.params_b, 0.5)
            out.append(f"{t.name:<14} {t.params_b:>5g}B  bf16 {bf16:5.1f} GiB  4-bit {int4:5.1f} GiB  -> bf16 {'fits' if bf16 <= ram_gb - reserve_gb else 'NO'}, "
                       f"4-bit {'fits' if int4 <= ram_gb - reserve_gb else 'NO'} (budget {ram_gb - reserve_gb} GiB)")
    return out
```
- **In simple words:** For each single-model entry compute `params x bytes` in GiB for bf16 and 4-bit and compare with the usable RAM.
- **Exactly:** `weight_gb(params_b, bytes) = params_b * 1e9 * bytes / 2**30`. The budget is `ram_gb - reserve_gb` (8 - 3). It ignores KV cache, vision-token activations and framework overhead, so "fits" is a necessary, not a sufficient, condition.
- **Why it is here:** It turns "can I run this locally?" into arithmetic you can do in an interview.

### 7. recommend: transparent rule-based scoring  (`main.py` lines 117–138)
```python
def recommend(need, top=3):
    """need = dict(doc='scanned'|'born_digital'|'mixed', gpu=bool, onprem_required=bool, tables=bool, formulas=bool, multilingual=bool,
    forms=bool, pages_per_day=int). Hard filters first, then a weighted sum of the 0..2 skill scores. Returns [(score, tool, reasons)]."""
    ranked = []
    for t in TOOLS:
        if need.get("onprem_required") and not t.onprem:
            continue                                                    # hard filter: data may not leave
        if not need.get("gpu") and not t.cpu_ok:
            continue                                                    # hard filter: no GPU -> only CPU-usable tools
        w = dict(scans=1.0 if need["doc"] in ("scanned", "mixed") else 0.2, born_digital=1.0 if need["doc"] in ("born_digital", "mixed") else 0.2,
                 tables=1.5 if need.get("tables") else 0.2, formulas=1.5 if need.get("formulas") else 0.0,
                 multilingual=1.0 if need.get("multilingual") else 0.0, forms_kv=1.5 if need.get("forms") else 0.0)
        score = sum(w[k] * getattr(t, k) for k in w)
        if t.family == "vlm" and need.get("pages_per_day", 0) > 100_000 and not need.get("gpu"):
            score -= 5
        if t.family == "cloud" and need.get("pages_per_day", 0) > 200_000:
            score -= 2                                                  # per-page pricing starts to hurt at scale
        if t.family == "engine" and need["doc"] == "born_digital":
            score -= 1                                                  # needless: use the text layer (step 08)
        why = [k for k in w if w[k] >= 1 and getattr(t, k) == 2]
        ranked.append((round(score, 1), t, why))
    return sorted(ranked, key=lambda r: -r[0])[:top]
```
- **In simple words:** Apply hard filters (data must stay on-prem; no GPU means CPU-usable tools only), weight each skill by the stated need, add small penalties, and return the top three.
- **Exactly:** `w` maps skills to weights (tables 1.5 if needed, formulas 1.5, forms 1.5, scans/born-digital 1.0 or 0.2 depending on the document). `score = sum(w[k] * getattr(t, k))`. Penalties: a VLM at over 100k pages/day without a GPU, cloud at very high volume, and a plain OCR engine on born-digital documents (use the text layer). `why` lists the skills where the tool scores 2 and the weight is at least 1. Sorted by score descending.
- **Why it is here:** A scoring helper is not an oracle: the point is that the weights ARE your requirements, written down so a colleague can disagree with a specific number.

### 8. Six scenarios  (`main.py` lines 141–148)
```python
SCENARIOS = [
    ("8 GB laptop, scanned English invoices, data must stay local", dict(doc="scanned", gpu=False, onprem_required=True, forms=False)),
    ("Born-digital research papers with tables and formulas, CPU only", dict(doc="born_digital", gpu=False, onprem_required=True, tables=True, formulas=True)),
    ("Mixed-language scans, one 24 GB GPU, 1M pages", dict(doc="scanned", gpu=True, onprem_required=True, tables=True, multilingual=True, pages_per_day=1_000_000)),
    ("Mixed bundle (some native, some scans), tables, GPU available, on-prem", dict(doc="mixed", gpu=True, onprem_required=True, tables=True, formulas=True)),
    ("Small team, invoices and forms with key-value extraction, cloud allowed", dict(doc="scanned", gpu=False, onprem_required=False, forms=True, tables=True)),
    ("CPU only, scanned multilingual pages, no tables needed", dict(doc="scanned", gpu=False, onprem_required=True, multilingual=True)),
]
```
- **In simple words:** Each scenario is a title and a dict of needs.
- **Exactly:** Keys: `doc`, `gpu`, `onprem_required`, `tables`, `formulas`, `multilingual`, `forms`, `pages_per_day`.
- **Why it is here:** The cases cover laptop-scan, CPU-papers, GPU-archive, mixed bundle, cloud-forms, CPU-multilingual.

### 9. Print everything  (`main.py` lines 151–167)
```python
if __name__ == "__main__":
    print(markdown_table(TOOLS))
    print("\nBenchmarks (always check the version and who ran it):")
    for n, what, caveat in BENCHMARKS:
        print(f"  - {n}: {what}\n      caveat: {caveat}")

    print("\nDoes it fit? weights only, 8 GB machine with 3 GiB kept for the OS and Python:")
    print("\n".join("   " + l for l in fits_report()))

    for i, (title, need) in enumerate(SCENARIOS, 1):
        print(f"\nscenario {i}: {title}")
        for rank, (score, t, why) in enumerate(recommend(need), 1):
            print(f"   {rank}. {t.name:<28} score {score:>4}  ({t.family}; strong at: {', '.join(why) or 'nothing flagged'})")

    if os.environ.get("SHOW_NOTES"):
        for t in TOOLS:
            print(f"{t.name}: {t.note}")
```
- **In simple words:** Table, benchmark caveats, fit report, then the top three tools per scenario; optionally the notes.
- **Exactly:** Guarded by `if __name__ == "__main__":` so another script can `import main` and reuse `TOOLS` and `recommend`.
- **Why it is here:** The output is the artifact you would put in a design doc.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `@dataclass class T: x: int` | auto-generated constructor and repr from annotations |
| `float | None` | union type: a float or None |
| `stars = lambda n: ...` | small anonymous function stored in a variable |
| `getattr(t, k)` | attribute lookup by name held in a string |
| `sorted(xs, key=lambda r: -r[0])[:top]` | sort by descending score and keep the top N |
| `f"{x:<28}"` / `f"{x:>5g}"` | left-align in 28 characters / right-align a float compactly |
| `if __name__ == "__main__":` | run only when executed directly, not when imported |

## Libraries & functions
| Function | What it does | Internally |
|---|---|---|
| `dataclasses.dataclass` | record class generator | adds methods from `__annotations__` at class-creation time |
| `os.environ.get` | read an environment variable | dict-like view of the process environment |
| `str.join`, f-strings | assemble the table | pure string operations |
| `reference_engines.tesseract` (REFERENCE) | `pytesseract.image_to_string` | writes the image to a temp file, runs the `tesseract` binary, returns its stdout |
| `reference_engines.paddleocr_v3` (REFERENCE) | `PaddleOCR(...).predict(img)` | text detection model, optional orientation models, recognition model; results carry `rec_texts` and boxes (unverified here) |
| `reference_engines.docling_convert` (REFERENCE) | `DocumentConverter().convert(src).document.export_to_markdown()` | layout model + table model (+ OCR on pages without text) building a `DoclingDocument` |
| `reference_engines.vlm_ocr_openai_compatible` (REFERENCE) | POST a base64 page image to `/chat/completions` of a server you run | the server's VLM encodes the image into vision tokens and generates Markdown |

## Execution trace (real run)
```
| Tool | Family | Size | Runs on a CPU laptop? | On-prem? | Licence | Scans | Tables | Formulas |
|---|---|---|---|---|---|---|---|---|
| Tesseract | engine | small models | yes | yes | Apache-2.0 | + | - | - |
| PaddleOCR 3.x (PP-OCRv5) | engine | small models | yes | yes | Apache-2.0 | ++ | - | - |
| EasyOCR | engine | small models | yes | yes | Apache-2.0 | + | - | - |
| Docling | toolkit | small models | yes | yes | MIT | + | ++ | + |
| MinerU | toolkit | small models | slow / GPU advised | yes | verify (changed over time) | ++ | ++ | ++ |
| marker | toolkit | small models | slow / GPU advised | yes | verify (code and weights licensed differently) | + | ++ | ++ |
| PaddleOCR-VL | vlm | 0.9B params | slow / GPU advised | yes | verify (Apache-2.0 reported) | ++ | ++ | ++ |
| olmOCR-2 | vlm | 8B params | slow / GPU advised | yes | verify | ++ | ++ | ++ |
| DeepSeek-OCR | vlm | 3B params | slow / GPU advised | yes | verify | ++ | + | + |
| dots.ocr | vlm | 3B params | slow / GPU advised | yes | verify | ++ | ++ | ++ |
| GLM-OCR | vlm | 0.9B params | slow / GPU advised | yes | verify | ++ | + | + |
| GOT-OCR2.0 | vlm | 0.58B params | slow / GPU advised | yes | verify (Apache-2.0 reported) | ++ | + | ++ |
| AWS Textract | cloud | service | yes | no (cloud) | commercial API | ++ | ++ | - |
| Azure Document Intelligence | cloud | service | yes | no (cloud) | commercial API | ++ | ++ | + |
| Google Document AI | cloud | service | yes | no (cloud) | commercial API | ++ | ++ | + |

Benchmarks (always check the version and who ran it):
  - OmniDocBench (CVPR 2025, OpenDataLab): page-level parsing: text edit distance, formula CDM, table TEDS, reading-order edit distance
      caveat: numbers depend on the version (v1.5, v1.6 ...) and on who ran the model: verify the leaderboard before quoting
  - olmOCR-Bench (Ai2): pass/fail unit tests on PDF pages: a phrase must be present, two lines must be in this order, a table cell must match
      caveat: tests are strict about exact text; verify the current release and its test count before quoting
  - Your own 50 pages: CER/WER + field accuracy + table TEDS on documents like YOURS (step 06)
      caveat: the only number that predicts your production quality

Does it fit? weights only, 8 GB machine with 3 GiB kept for the OS and Python:
   PaddleOCR-VL     0.9B  bf16   1.7 GiB  4-bit   0.4 GiB  -> bf16 fits, 4-bit fits (budget 5 GiB)
   olmOCR-2           8B  bf16  14.9 GiB  4-bit   3.7 GiB  -> bf16 NO, 4-bit fits (budget 5 GiB)
   DeepSeek-OCR       3B  bf16   5.6 GiB  4-bit   1.4 GiB  -> bf16 NO, 4-bit fits (budget 5 GiB)
   dots.ocr           3B  bf16   5.6 GiB  4-bit   1.4 GiB  -> bf16 NO, 4-bit fits (budget 5 GiB)
   GLM-OCR          0.9B  bf16   1.7 GiB  4-bit   0.4 GiB  -> bf16 fits, 4-bit fits (budget 5 GiB)
   GOT-OCR2.0      0.58B  bf16   1.1 GiB  4-bit   0.3 GiB  -> bf16 fits, 4-bit fits (budget 5 GiB)

scenario 1: 8 GB laptop, scanned English invoices, data must stay local
   1. PaddleOCR 3.x (PP-OCRv5)     score  2.0  (engine; strong at: scans)
   2. Docling                      score  1.8  (toolkit; strong at: nothing flagged)
   3. Tesseract                    score  1.0  (engine; strong at: nothing flagged)

scenario 2: Born-digital research papers with tables and formulas, CPU only
   1. Docling                      score  6.7  (toolkit; strong at: born_digital, tables)
   2. PaddleOCR 3.x (PP-OCRv5)     score -0.6  (engine; strong at: nothing flagged)
   3. Tesseract                    score -0.8  (engine; strong at: nothing flagged)

scenario 3: Mixed-language scans, one 24 GB GPU, 1M pages
   1. MinerU                       score  7.4  (toolkit; strong at: scans, tables, multilingual)
   2. PaddleOCR-VL                 score  7.4  (vlm; strong at: scans, tables, multilingual)
   3. dots.ocr                     score  7.2  (vlm; strong at: scans, tables, multilingual)

scenario 4: Mixed bundle (some native, some scans), tables, GPU available, on-prem
   1. MinerU                       score 10.0  (toolkit; strong at: scans, born_digital, tables, formulas)
   2. PaddleOCR-VL                 score 10.0  (vlm; strong at: scans, born_digital, tables, formulas)
   3. marker                       score  9.0  (toolkit; strong at: born_digital, tables, formulas)

scenario 5: Small team, invoices and forms with key-value extraction, cloud allowed
   1. AWS Textract                 score  8.2  (cloud; strong at: scans, tables, forms_kv)
   2. Azure Document Intelligence  score  8.2  (cloud; strong at: scans, tables, forms_kv)
   3. Google Document AI           score  8.2  (cloud; strong at: scans, tables, forms_kv)

scenario 6: CPU only, scanned multilingual pages, no tables needed
   1. PaddleOCR 3.x (PP-OCRv5)     score  4.0  (engine; strong at: scans, multilingual)
   2. Tesseract                    score  3.0  (engine; strong at: multilingual)
   3. EasyOCR                      score  3.0  (engine; strong at: multilingual)

--- python reference_engines.py (every engine skips: nothing installed, nothing downloaded) ---
sample image: <repo>/18_ocr_and_document_parsing/01_images_as_arrays/sample_gray.png   ALLOW_MODEL_DOWNLOAD=0
- tesseract                  '[skip] tesseract binary and/or pytesseract not installed'
- paddleocr_v3               '[skip] paddleocr not installed'
- ppstructure_v3             '[skip] paddleocr not installed'
- easyocr_reader             '[skip] easyocr not installed'
- docling_convert            '[skip] docling not installed'
- vlm_ocr_openai_compatible  '[skip] set OCR_VLM_BASE_URL (e.g. http://localhost:8000/v1) and OCR_VLM_MODEL to use a served VLM OCR model'
- cloud APIs                 [manual] Textract / Azure Document Intelligence / Google Document AI need paid accounts and credentials: never put keys in code; use environment variables and send only documents you are allowed to send.
```
- The table is generated from data: only Tesseract, PaddleOCR, EasyOCR, Docling and the three cloud APIs are marked as comfortable on a CPU laptop; every VLM says "slow / GPU advised".
- The fit report is pure arithmetic: a 0.9B model is 1.7 GiB in bf16 and fits; 3B (5.6 GiB) and 8B (14.9 GiB) do not fit in bf16 within the 5 GiB budget, but do in 4-bit (1.4 and 3.7 GiB). It says nothing about speed or the vision-token cache.
- Scenario 1 (8 GB laptop, scans, on-prem) returns PaddleOCR first and Docling second; scenario 2 (born-digital papers with tables and formulas, CPU) returns Docling far ahead because engines are penalised for born-digital input. Scenario 5 (cloud allowed, forms) returns the three cloud APIs tied: **a tie is the helper being honest** that pricing, region and existing cloud contracts decide.
- Scenarios 3 and 4 return MinerU and PaddleOCR-VL tied at the top: on the same scores the helper cannot separate them; your gold set can.
- `reference_engines.py` prints skip lines for all engines: nothing is installed and nothing was downloaded. The calls in it are written from documentation and are **not verified in this repo**.

## Experiments (change ONE thing)
| Change | What you will see | Why |
|---|---|---|
| Run `fits_report(ram_gb=16)` | 3B models fit in bf16 (5.6 GiB <= 13 GiB budget), 8B still needs 4-bit | Budget is RAM minus 3 GiB |
| Remove the `onprem_required` hard filter in scenario 1 | The three cloud APIs take all top-3 slots (score 2.6 vs PaddleOCR 2.0) | They score high on scans and tables; privacy was the only thing excluding them |
| Set `forms=True` for scenario 3 | Top three unchanged (MinerU, PaddleOCR-VL, dots.ocr) | Forms/key-value scores are 0 for everything self-hosted in our table, so the weight adds nothing: a limitation of the data, not a fact about the tools |
| Add your own `Tool` with scans=2, tables=2 | It lands in the top 3 for scenarios 3-4 | The score is only as honest as the 0-2 inputs |
| Change every weight to 1 | Ranking becomes uninformative | The weights encode requirements |
| `SHOW_NOTES=1 python main.py` | One-line description per tool | The notes are the part to verify first |

## Debugging practice
**BREAK IT:** delete the `onprem_required` filter in `recommend` and run scenario 1 ("data must stay local").

**Symptom:** a cloud API (AWS Textract / Azure Document Intelligence / Google Document AI) appears among the top three for a requirement that explicitly forbids sending documents away. Nothing crashes: the recommender is wrong because a hard constraint was turned into a soft preference.

**Fix:** hard constraints (privacy, residency, no GPU) must be **filters**, not weights. A weight can always be outvoted by enough good skill scores; a filter cannot. In design reviews, list the hard constraints first and only then rank what survives.

## Common mistakes
- Picking by leaderboard rank alone (version, date, runner and document mix all matter) instead of a gold set from your own documents.
- Using a VLM OCR model on contracts, invoices or medical text without a numeric cross-check: fluent hallucination is the failure mode.
- Installing a 3B-8B VLM on an 8 GB laptop and concluding "the model is slow" when it was swapping: do the memory arithmetic first.
- Sending sensitive documents to a cloud API without checking the data-processing terms, region and retention.
- Running a heavy parser on born-digital pages that had a perfectly good text layer ([step 08](../08_pdf_internals_and_routing/WALKTHROUGH.md)).
- Quoting licences from memory. Read the repository and model card at the time you ship.

## Rebuild checklist
1. Define a `Tool` dataclass with family, parameters, CPU/on-prem flags, licence and 0-2 skill scores.
2. List the tools you actually consider (engines, toolkits, VLMs, cloud) and mark every uncertain fact `verify`.
3. Write `markdown_table`, `weight_gb` and `fits_report`.
4. Write `recommend`: hard filters, weighted sum, penalties, top-N with reasons.
5. Write 4-6 scenarios from your own life and read the results critically: where does the helper tie, where is it wrong?
6. Add a guarded reference script per engine with an explicit opt-in for downloads.

## Say it aloud
- "There are four families: classic engines that detect and recognise text, pipeline toolkits that add layout, tables and reading order, end-to-end VLMs that write the page as Markdown, and cloud APIs."
- "VLM OCR gives structure in one pass but generates text, so it can hallucinate or skip content and has no calibrated confidence; I cross-check numbers and sample-check outputs."
- "On an 8 GB CPU laptop I use PaddleOCR or Tesseract, or Docling for structure; a 3B VLM needs 5.6 GiB in bf16 for weights alone, so it needs 4-bit or a GPU."
- "I treat hard constraints like data residency as filters and only rank what survives; leaderboard numbers are leads to verify, because versions and runners differ."
- "The only evaluation that matters is 50 of my own pages scored with CER, field accuracy and table TEDS.

## The new problem (cliffhanger)
We can now choose a parser and route pages to it. The last problem is the border: the parser's output is a pile of tool-specific objects, but the chunker, the retriever and the citation UI all need one agreed, validated shape. [Step 10](../10_parsed_document_handoff/WALKTHROUGH.md) defines that contract and proves module 19's chunker can read it.
