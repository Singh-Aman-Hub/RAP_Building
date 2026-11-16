"""
REFERENCE -- not run in this course (needs optional packages and, for most of them, model downloads or a GPU).
Every function is GUARDED: a missing package prints a skip message, and anything that would download model weights additionally needs
the explicit opt-in  ALLOW_MODEL_DOWNLOAD=1.  Nothing here is auto-installed or auto-downloaded, and the API calls were written from the
public documentation: they are NOT verified in this repo. Check each project's current docs before relying on them.

RUN (from this step folder):  python reference_engines.py            # prints which engines are importable, runs nothing heavy
                              ALLOW_MODEL_DOWNLOAD=1 python reference_engines.py ../01_images_as_arrays/sample_gray.png
"""
import base64
import importlib.util
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

ALLOW = os.environ.get("ALLOW_MODEL_DOWNLOAD") == "1"
SAMPLE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parents[1] / "01_images_as_arrays" / "sample_gray.png"


def have(mod):
    return importlib.util.find_spec(mod) is not None


def tesseract(path):
    """Classic engine. Needs the `tesseract` BINARY on PATH plus the pytesseract wrapper; models (traineddata) ship with the binary."""
    if not (have("pytesseract") and shutil.which("tesseract")):
        return "[skip] tesseract binary and/or pytesseract not installed"
    import pytesseract
    from PIL import Image
    return pytesseract.image_to_string(Image.open(path))            # internally: runs the binary, reads its stdout. image_to_data gives boxes


def paddleocr_v3(path):
    """PaddleOCR 3.x: det + rec models. First use downloads weights, hence the opt-in."""
    if not have("paddleocr"):
        return "[skip] paddleocr not installed"
    if not ALLOW:
        return "[skip] would download PP-OCR weights: set ALLOW_MODEL_DOWNLOAD=1 to allow"
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
    return [r["rec_texts"] for r in ocr.predict(str(path))]         # one result object per image; dict-like with rec_texts / rec_scores / rec_boxes


def ppstructure_v3(path):
    """Layout + table + formula pipeline from PaddleOCR 3.x. Heavy: several models."""
    if not have("paddleocr"):
        return "[skip] paddleocr not installed"
    if not ALLOW:
        return "[skip] would download several PP-StructureV3 models: set ALLOW_MODEL_DOWNLOAD=1 to allow"
    from paddleocr import PPStructureV3
    out = PPStructureV3().predict(input=str(path))
    return [r.markdown["markdown_texts"] if hasattr(r, "markdown") else str(r) for r in out]


def easyocr_reader(path):
    if not have("easyocr"):
        return "[skip] easyocr not installed"
    if not ALLOW:
        return "[skip] would download detector + recogniser weights: set ALLOW_MODEL_DOWNLOAD=1 to allow"
    import easyocr
    return easyocr.Reader(["en"], gpu=False).readtext(str(path))    # list of (box_corners, text, confidence)


def docling_convert(path):
    """Docling: layout model + TableFormer (+ OCR when a page has no text layer). Output is a DoclingDocument."""
    if not have("docling"):
        return "[skip] docling not installed"
    if not ALLOW:
        return "[skip] would download Docling layout/table models: set ALLOW_MODEL_DOWNLOAD=1 to allow"
    from docling.document_converter import DocumentConverter
    return DocumentConverter().convert(str(path)).document.export_to_markdown()


def vlm_ocr_openai_compatible(path):
    """Any VLM OCR model served behind an OpenAI-compatible endpoint (vLLM, SGLang, Ollama, a hosted gateway ...).
    You run the server yourself; this script only needs the URL and model name. Standard library only."""
    base, model = os.environ.get("OCR_VLM_BASE_URL"), os.environ.get("OCR_VLM_MODEL")
    if not (base and model):
        return "[skip] set OCR_VLM_BASE_URL (e.g. http://localhost:8000/v1) and OCR_VLM_MODEL to use a served VLM OCR model"
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    body = {"model": model, "max_tokens": 2048, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Convert this page to Markdown. Keep tables as Markdown tables. Output only the content."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]}
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def cloud_note():
    return ("[manual] Textract / Azure Document Intelligence / Google Document AI need paid accounts and credentials: "
            "never put keys in code; use environment variables and send only documents you are allowed to send.")


if __name__ == "__main__":
    print(f"sample image: {SAMPLE}   ALLOW_MODEL_DOWNLOAD={'1' if ALLOW else '0'}")
    for fn in (tesseract, paddleocr_v3, ppstructure_v3, easyocr_reader, docling_convert, vlm_ocr_openai_compatible):
        try:
            res = fn(SAMPLE)
        except Exception as e:                                       # a reference script must never crash the course
            res = f"[error] {type(e).__name__}: {e}"
        print(f"- {fn.__name__:<26} {str(res)[:110]!r}")
    print("- cloud APIs                 " + cloud_note())
