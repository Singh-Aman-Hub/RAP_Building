"""
Exercise 04: extracting clean, citable page text from a PDF with pdfplumber.

First stage of a document-QA pipeline: turn `sample_policy.pdf` (4 pages, included in this folder) into one
record per page with clean body text, so later stages can chunk it and cite "page N" in answers.
Pages without a text layer (scans) are reported so they can be sent to OCR instead.

Run:   python buggy_code.py
Needs: pip install pdfplumber
"""

import json
from pathlib import Path

import pdfplumber

PDF_PATH = Path(__file__).resolve().parent / "sample_policy.pdf"

HEADER = "ACME RETAIL - EMPLOYEE HANDBOOK 2025"     # printed at the top of every page
FOOTER = "Confidential - Page 1 of 4"               # printed at the bottom of every page


def clean_page(text):
    """Remove the running header/footer and merge the remaining lines into one paragraph."""
    lines = [line.strip() for line in text.split("\n")]
    body = [line for line in lines if line != HEADER and line != FOOTER]
    return "".join(body)


def extract_pages(path):
    """Return (records, pages_needing_ocr). Page numbers are 1-based, as printed in the document."""
    records, needs_ocr = [], []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text is None:
                needs_ocr.append(i)
                continue
            body = clean_page(text)
            records.append({"page": i, "text": body, "n_words": len(body.split())})
    return records, needs_ocr


def find(records, phrase):
    """Pages whose text contains `phrase` (case-insensitive)."""
    return [r["page"] for r in records if phrase.lower() in r["text"].lower()]


if __name__ == "__main__":
    records, needs_ocr = extract_pages(PDF_PATH)
    for r in records:
        print(json.dumps(r))
    print(f"\npages needing OCR: {needs_ocr}")
    for phrase in ["reimbursed up to 6,000", "per week with approval", "employee handbook", "confidential"]:
        print(f"find({phrase!r:<26}) -> {find(records, phrase)}")
