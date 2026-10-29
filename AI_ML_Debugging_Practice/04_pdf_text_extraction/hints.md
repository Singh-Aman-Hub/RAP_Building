# Hints: 04 PDF Text Extraction

Open one hint at a time.

## First move
<details><summary>Hint 0</summary>

Before touching `clean_page`, print `repr(page.extract_text())` and `page.page_number` for every page. Compare them with what you see in a PDF viewer. `repr` shows newlines and empty strings that `print` hides.
</details>

## Bug A: page numbers are off
<details><summary>Hint A1</summary>

Compare the `"page"` values with the footer text "Page N of 4" on the same page.
</details>
<details><summary>Hint A2</summary>

`enumerate` has an optional argument. pdfplumber's `Page` objects also carry their own page number attribute (`dir(page)`).
</details>

## Bug B: the blank page becomes a record, and the OCR list is empty
<details><summary>Hint B1</summary>

Look at the `repr` of page 3's extracted text. Is it `None`?
</details>
<details><summary>Hint B2</summary>

What should count as "no extractable text"? Consider `None`, `""`, and a string of only spaces/newlines. Write a condition that treats all three the same way.
</details>

## Bug C: the footer survives on some pages
<details><summary>Hint C1</summary>

On which pages is the footer removed, and on which is it kept? What differs between those footers?
</details>
<details><summary>Hint C2</summary>

`FOOTER` is one fixed string, but the real footer changes on every page. Think about matching a *pattern* (prefix, or a regular expression) instead of an exact string. Remember the 40-page requirement.
</details>

## Bug D: words glued together ("weekwith", "HoursOffice")
<details><summary>Hint D1</summary>

Find where lines are merged back into a paragraph. What character is placed between two lines?
</details>
<details><summary>Hint D2</summary>

Count words in the cleaned page 1 by hand and compare with `n_words`. Which number is right?
</details>

## Verification ideas
- `assert [r["page"] for r in records] == [1, 2, 4] and needs_ocr == [3]`
- `assert not any("Confidential" in r["text"] or "HANDBOOK" in r["text"] for r in records)`
- `assert "weekwith" not in records[0]["text"]`
