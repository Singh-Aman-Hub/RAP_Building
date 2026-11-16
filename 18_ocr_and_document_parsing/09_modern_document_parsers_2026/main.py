"""
GOAL:  Map the 2026 document-parsing landscape (pipeline toolkits, classic OCR engines, end-to-end VLM OCR models, cloud APIs) as DATA,
       generate a comparison table, check which models even fit in 8 GB of RAM (arithmetic, not a benchmark), and run a transparent
       rule-based "which tool for my situation?" helper on six scenarios. No heavy imports, nothing downloaded.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 18_ocr_and_document_parsing/09_modern_document_parsers_2026/main.py
EXPECTED OUTPUT (shortened):
    | Tool | Family | Size | Runs on a CPU laptop? | ...        (markdown comparison table)
    fits in 8 GB? (weights only): tesseract yes ... olmOCR-2 (8B) no ...
    scenario 1: ... -> 1. ... 2. ... 3. ...                      (six scenarios)
NEEDS: Python standard library only.  < 1 s.
Everything quantitative about quality is deliberately NOT here: benchmark numbers change monthly. See the WALKTHROUGH and verify at the source.
"""
import os
from dataclasses import dataclass

# ---- 1. The landscape as data ------------------------------------------------------------------------------------------------
# family: "toolkit" (pipeline of several models), "engine" (text detection + recognition), "vlm" (one vision-language model reads the
# page and writes Markdown/HTML/JSON), "cloud" (a paid API). params_b = parameters in billions when it is ONE model, else None.
# Skill scores 0 (poor / not offered) .. 2 (strong) are COURSE JUDGEMENT from public descriptions, not measurements: verify on your own pages.
# licence: only stated when the author is confident; "verify" means check the repository / model card before commercial use.


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

BENCHMARKS = [
    ("OmniDocBench (CVPR 2025, OpenDataLab)", "page-level parsing: text edit distance, formula CDM, table TEDS, reading-order edit distance",
     "numbers depend on the version (v1.5, v1.6 ...) and on who ran the model: verify the leaderboard before quoting"),
    ("olmOCR-Bench (Ai2)", "pass/fail unit tests on PDF pages: a phrase must be present, two lines must be in this order, a table cell must match",
     "tests are strict about exact text; verify the current release and its test count before quoting"),
    ("Your own 50 pages", "CER/WER + field accuracy + table TEDS on documents like YOURS (step 06)",
     "the only number that predicts your production quality"),
]


# ---- 2. Comparison table generator ---------------------------------------------------------------------------------------------
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


# ---- 3. Does the model even fit? Weights only: params x bytes-per-parameter ------------------------------------------------------
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


# ---- 4. A transparent decision helper ----------------------------------------------------------------------------------------------
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


SCENARIOS = [
    ("8 GB laptop, scanned English invoices, data must stay local", dict(doc="scanned", gpu=False, onprem_required=True, forms=False)),
    ("Born-digital research papers with tables and formulas, CPU only", dict(doc="born_digital", gpu=False, onprem_required=True, tables=True, formulas=True)),
    ("Mixed-language scans, one 24 GB GPU, 1M pages", dict(doc="scanned", gpu=True, onprem_required=True, tables=True, multilingual=True, pages_per_day=1_000_000)),
    ("Mixed bundle (some native, some scans), tables, GPU available, on-prem", dict(doc="mixed", gpu=True, onprem_required=True, tables=True, formulas=True)),
    ("Small team, invoices and forms with key-value extraction, cloud allowed", dict(doc="scanned", gpu=False, onprem_required=False, forms=True, tables=True)),
    ("CPU only, scanned multilingual pages, no tables needed", dict(doc="scanned", gpu=False, onprem_required=True, multilingual=True)),
]


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

# TRY IT:   add a Tool with your own scores and see where it lands; set SHOW_NOTES=1 to print the one-line notes; change the 8 GB budget in
#           fits_report(ram_gb=16) to see which VLMs become possible.
# BREAK IT: remove the onprem_required filter and watch cloud APIs jump into a "data must stay local" scenario; set every weight to 1 and
#           notice the ranking becomes meaningless: the weights ARE the requirements.
