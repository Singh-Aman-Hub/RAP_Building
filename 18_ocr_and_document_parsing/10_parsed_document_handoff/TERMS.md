# Terms — The parsed-document handoff

Folder: `18_ocr_and_document_parsing/10_parsed_document_handoff` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [18 README](../README.md)

**Terms in this folder:** [Parsed document (layout elements)](#parsed-document-layout-elements) · [Element type (DocLayNet-style labels)](#element-type-doclaynet-style-labels) · [Bounding box in PDF points](#bounding-box-in-pdf-points) · [reading_order](#reading_order) · [Provenance (page + bbox)](#provenance-page--bbox) · [Data contract at a pipeline boundary](#data-contract-at-a-pipeline-boundary) · [pydantic BaseModel and model_validate](#pydantic-basemodel-and-model_validate) · [extra="forbid"](#extraforbid) · [field_validator and model_validator](#field_validator-and-model_validator) · [ValidationError and error location path](#validationerror-and-error-location-path) · [Sidecar metadata file](#sidecar-metadata-file) · [Table header + rows payload](#table-header--rows-payload) · [Importing a module from a file path](#importing-a-module-from-a-file-path) · [Heading versus section_header (vocabulary mismatch)](#heading-versus-section_header-vocabulary-mismatch)

---

## Parsed document (layout elements)  `(concept)`
**1. Definition:** A document as a list of typed elements with page, bounding box and reading order instead of a string.
**2. In plain words:** The page as labelled boxes.
**3. Example to quote:** > "The OCR to chunking handoff is typed elements, not text."
**Seen in:** `main.py`, `ParsedDocument`

## Element type (DocLayNet-style labels)  `(concept)`
**1. Definition:** page_header, page_footer, title, section_header, text, list_item, caption, table, figure, footnote.
**2. In plain words:** What each box is.
**3. Example to quote:** > "A Literal type makes 'heading' fail at the border."
```python
ElementType = Literal["title", "section_header", ...]
```
**Seen in:** `main.py`, `ElementType`

## Bounding box in PDF points  `(concept)`
**1. Definition:** [x0, y0, x1, y1] with origin top-left, units 1/72 inch; pixels at 96 dpi times 0.75.
**2. In plain words:** Where on the page, in the unit a viewer uses.
**3. Example to quote:** > "Mixing pixels and points shifts every citation by a constant factor."
```python
round(v * PX_TO_PT, 1)
```
**Seen in:** `main.py`, `to_parsed_document`

## reading_order  `(concept)`
**1. Definition:** An integer sort key per page; the consumer sorts by (page, reading_order).
**2. In plain words:** The order to read the blocks.
**3. Example to quote:** > "Footers get 99 so they sort last."
**Seen in:** `main.py`, `to_parsed_document`

## Provenance (page + bbox)  `(concept)`
**1. Definition:** Source location carried on every element so an answer can cite and highlight it.
**2. In plain words:** Where did this text come from?
**3. Example to quote:** > "Dropping page and bbox at the handoff makes answers unverifiable."
**Seen in:** `main.py`, `Element`

## Data contract at a pipeline boundary  `(concept)`
**1. Definition:** The agreed, validated shape of data passed between stages.
**2. In plain words:** A schema both sides sign.
**3. Example to quote:** > "I validate where data crosses a boundary, not deep inside the consumer."
**Seen in:** `main.py`, `ParsedDocument`

## pydantic BaseModel and model_validate  `(class)`
**1. Definition:** A class whose annotations define a schema; model_validate(dict) parses, coerces and raises ValidationError.
**2. In plain words:** Typed JSON with error messages.
**3. Example to quote:** > "Same model accepts our file and module 19's sample, so it is a shared contract."
```python
ParsedDocument.model_validate(data)
```
**Seen in:** `main.py`

## extra="forbid"  `(syntax)`
**1. Definition:** ConfigDict option that rejects unknown keys.
**2. In plain words:** No surprise fields.
**3. Example to quote:** > "An author key on an element is an error."
```python
model_config = ConfigDict(extra="forbid")
```
**Seen in:** `main.py`, `Element`

## field_validator and model_validator  `(syntax)`
**1. Definition:** Hooks that check one field or the whole built model and raise ValueError for violations.
**2. In plain words:** Custom rules like level only for section_header.
**3. Example to quote:** > "Cross-field rules live in the model, not in the consumer."
```python
@model_validator(mode="after")
```
**Seen in:** `main.py`, `Element`, `TablePayload`

## ValidationError and error location path  `(error)`
**1. Definition:** pydantic's aggregated error; errors() lists loc, msg and type.
**2. In plain words:** Tells you exactly which element and field broke.
**3. Example to quote:** > "elements.4.table: rows [0] do not have 3 cells."
```python
err.errors()[0]["loc"]
```
**Seen in:** `main.py`, negative tests

## Sidecar metadata file  `(concept)`
**1. Definition:** A companion file next to the data holding fields the strict schema forbids (OCR confidence).
**2. In plain words:** Extra notes that must not break the contract.
**3. Example to quote:** > "Confidence travels in parsed_document.ocr_meta.json keyed by element id."
**Seen in:** `main.py`, `meta`

## Table header + rows payload  `(concept)`
**1. Definition:** table = {header: [...], rows: [[...]]} with rectangular rows.
**2. In plain words:** A table the chunker can split and repeat the header for.
**3. Example to quote:** > "Never flatten tables to text at the handoff."
**Seen in:** `main.py`, `TablePayload`

## Importing a module from a file path  `(function)`
**1. Definition:** importlib.util.spec_from_file_location + module_from_spec + exec_module; register in sys.modules for dataclasses.
**2. In plain words:** Import a script whose folder name starts with a digit.
**3. Example to quote:** > "I called module 19's own load and LayoutChunker on our file to prove the handoff."
```python
spec = importlib.util.spec_from_file_location(name, path)
```
**Seen in:** `main.py`

## Heading versus section_header (vocabulary mismatch)  `(error)`
**1. Definition:** Producer and consumer use different names for the same concept; fails at the border or, worse, much later.
**2. In plain words:** Two teams, two words.
**3. Example to quote:** > "pydantic names the bad value and the allowed set."
**Seen in:** `main.py`; BREAK IT
