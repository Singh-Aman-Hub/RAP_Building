# Hints: 09 LLM Context and Grounding

Open one hint at a time.

## Start here
<details><summary>Hint 0</summary>

Add a temporary `print(prompt)` inside `answer_question` (after the prompt is built). Most of these bugs become visible the moment you *read the prompt the model actually receives*.
</details>

## Bug A: answerable questions are refused, unanswerable ones are not
<details><summary>Hint A1</summary>

The scores are cosine **similarities**. Is a high score good news or bad news?
</details>
<details><summary>Hint A2</summary>

Read the refusal condition out loud as an English sentence. Does it say what the docstring and `question.md` say?
</details>

## Bug B: the best chunk is missing from the prompt
<details><summary>Hint B1</summary>

Print the order of chunk ids *after* sorting in `select_context`. Which chunk is considered first?
</details>
<details><summary>Hint B2</summary>

The loop stops at the first chunk that doesn't fit the budget. So the order decides *which* chunks survive. Check what `sorted()` does by default.
</details>

## Bug C: the citation points at the wrong document
<details><summary>Hint C1</summary>

Compare the label the model sees in the prompt with the index `parse_citations` uses to look the chunk up. Try it by hand for label `[0]`.
</details>
<details><summary>Hint C2</summary>

The system prompt shows the model an example citation format. Do the labels in SOURCES follow the same convention? Python lists and humans count differently. Both sides have to agree on one convention.
</details>
<details><summary>Hint C3</summary>

What does `some_list[-1]` return? Why does that make this bug silent instead of an `IndexError`?
</details>

## Bug D: an archived policy is used
<details><summary>Hint D1</summary>

There *is* a line that removes archived chunks. Follow the variable it creates. Is it used anywhere?
</details>
<details><summary>Hint D2</summary>

A linter would flag this one (look up "assigned but never used", e.g. ruff/pyflakes `F841`). Try running one.
</details>

## Verification ideas
- Assert that no chunk with `status == "archived"` is ever in the returned context.
- For every citation, assert the quoted sentence actually appears in the cited source's text.
- Assert the refusal path does not call `fake_llm` at all.
