# Module 07 — Exercises

Rule: change ONE thing at a time, predict the output first, then run. Answers/worked code: [solutions/](solutions/).

## Step 01 — roles & instructions
- **Modify:** add a 4th label `mixed` to `SYSTEM_C` and a review that deserves it. Does the output format stay clean?
- **Break it:** send a `role` of `"robot"` with `LLM_BACKEND=anthropic`; read the 400 error's field path.
- **Rebuild:** close the file and write a role+task+constraints system prompt for a *language detector*.
- **Explain aloud:** why does the system/user split help?

## Step 02 — few-shot
- **Modify:** add an example with a *neutral* sarcasm-free sentence; reorder examples. Run with `LLM_BACKEND=local` and compare.
- **Break it:** end `few_shot_turns` with an assistant message; observe.
- **Rebuild:** build few-shot chat turns from a list of `(text, label)` tuples without looking.
- **Explain aloud:** name two risks of few-shot examples.

## Step 03 — templates
- **Modify:** add `{tone}` and an optional `{max_words}` slot with a default.
- **Break it:** run `build_prompt("   ", labels)`, read the traceback line by line.
- **Rebuild:** render the same JSON-containing prompt with `format` (doubled braces) and `Template`.
- **Explain aloud:** when do you prefer `Template`?

## Step 04 — structured JSON
- **Read:** print `str(ValidationError)` for `{"label": "happy", "confidence": 7}` and name the two failing fields.
- **Modify:** add `reason: str` to `Sentiment`; make hand validator and prompt match (see `solutions/ex04_reason_field.py`).
- **Break it:** `json.loads("{'a': 1}")` — explain the error position.
- **Rebuild:** write `validate_by_hand` for `{"name": str, "age": int>=0}`.
- **Explain aloud:** why include the validation error in the retry message?

## Step 05 — decomposition
- **Modify:** add a stage 4 that validates the draft is < 40 words.
- **Break it:** delete the `STAGE 2` rule; add try/except so the pipeline reports *which stage* failed.
- **Rebuild:** write a vote function `majority(answers)` that returns `None` on a tie.
- **Explain aloud:** when is one prompt better than a pipeline?

## Step 06 — injection
- **Modify:** add three new attacks (role-play, "system override", base64-looking). Which pass on the real backend?
- **Break it:** remove `safe_label` and see what flows downstream.
- **Rebuild:** write the defended prompt + output allow-list from memory.
- **Explain aloud:** direct vs indirect injection with an example of each.

## Step 07 — prompt testing
- **Modify:** add 3 test cases (negation, mixed sentiment, other language) and create `v4` that changes exactly one thing (see `solutions/ex07_add_cases.py`).
- **Break it:** make `parse_label` stricter (exact match only) and see how many replies become `invalid`.
- **Rebuild:** write `run_version` + the per-category table from memory.
- **Explain aloud:** what does the mock prove vs not prove? Why temperature 0?

## Step 08 — minimal RAG
- **Modify:** add a gift-card document to `DOCS` and re-ask "Do you sell gift cards?". Then try `k=1` and `k=6` and compare the prompts.
- **Break it:** run the money-back question with `min_score=0.0`, read which chunk is injected, then fix it with a threshold. Also embed the query with a different vectorizer and read the `matmul` shape error.
- **Rebuild:** write `retrieve` (cosine with NumPy, top-k with `argsort`) from memory.
- **Explain aloud:** why does a retrieval miss produce a confident wrong answer?

## Stretch
Run step 07 with `LLM_BACKEND=local`. Write 5 lines: which version won, did the ranking match the simulation, and why might it differ?
