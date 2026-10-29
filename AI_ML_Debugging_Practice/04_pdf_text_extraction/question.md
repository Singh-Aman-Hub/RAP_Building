# 04: PDF Text Extraction (pdfplumber)

**Level:** Beginner · **Intentional bugs:** 4 · **Dependencies:** `pdfplumber` · **Data:** `sample_policy.pdf` (included, 4 pages)

## Scenario
You're joining a document-QA project. Stage 1 turns PDFs into one clean record per page, which later stages chunk, embed and cite as *"(Employee Handbook, page 2)"*. Users report that the bot cites the wrong page numbers, quotes "Confidential - Page 2 of 4" as if it were policy text, and fails to find phrases that are clearly in the document. The interviewer hands you the extractor and the sample PDF.

Open `sample_policy.pdf` in any PDF viewer first, so you know what the "truth" looks like.

## Expected behaviour
`python buggy_code.py` should produce:
- **One record per page that has text.** The PDF has 4 pages, and page 3 has **no text layer** (it simulates a scanned page). So there are exactly **3 records**, for pages **1, 2 and 4**.
- `"page"` uses the **same numbering a human sees** in the viewer and in the footer (1-based).
- Pages with no extractable text are **not** turned into records. They are listed in `pages needing OCR: [3]`.
- `"text"` contains only the body. The running header (`ACME RETAIL - EMPLOYEE HANDBOOK 2025`) and the footer (`Confidential - Page N of 4`) are removed from **every** page.
- Words that were on different lines in the PDF stay separate words: `...two days per week with approval...`, never `weekwith`.
- `n_words` matches the cleaned text.
- The `find(...)` checks print:
  ```
  find('reimbursed up to 6,000'  ) -> [2]
  find('per week with approval'  ) -> [1]
  find('employee handbook'       ) -> []
  find('confidential'            ) -> []
  ```

## Observed behaviour
```
{"page": 0, "text": "1. Working HoursOffice hours are 9:30 am to 6:00 pm, Monday to Friday.Employees may work from home up to two days per weekwith approval from their manager.", "n_words": 27}
{"page": 1, "text": "2. Travel and ExpensesTravel expenses must be ... elsewhere.Confidential - Page 2 of 4", "n_words": 37}
{"page": 2, "text": "", "n_words": 0}
{"page": 3, "text": "3. LeaveFull-time employees ... Sick leave is 12 days per year.Confidential - Page 4 of 4", "n_words": 34}

pages needing OCR: []
find('reimbursed up to 6,000'  ) -> [1]
find('per week with approval'  ) -> []
find('employee handbook'       ) -> []
find('confidential'            ) -> [1, 3]
```
No exception is raised. Every bug in this exercise is silent.

## Your task
Fix the 4 bugs in `clean_page` and `extract_pages` (and their constants). Your fixes must still work for a **longer** handbook, say 40 pages, without editing the code.

## Constraints
- Use pdfplumber, and don't add OCR. Just report the pages that need it.
- Do not modify `sample_policy.pdf`.

## Interview follow-ups
- What does `page.extract_text()` return for a page with no text layer in *your* pdfplumber version? Where did you confirm it?
- Removing headers by exact string match is fragile. Name a more robust approach, e.g. using the position of text on the page.
- Why do page numbers matter so much in a RAG system?
