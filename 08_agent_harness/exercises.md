# Module 08 — Exercises

Change ONE thing at a time; predict, then run. Worked answers: [solutions/](solutions/).

## Step 01 — LLM call
- **Modify:** add a `/reset` command that clears `history`; show what the bot forgets.
- **Break it:** drop the line that appends the assistant reply; explain the change in request size.
- **Rebuild:** write `chat()` from memory (history list, system prompt, append both sides).
- **Explain aloud:** why does the model "remember" nothing between calls?

## Step 02 — agent loop
- **Modify:** add a `power` operation and teach the system prompt about it; ask `2 ** 3`.
- **Break it:** ask `5 / 0`; read the traceback bottom-up and say which later step fixes it.
- **Rebuild:** the loop with `max_steps` and two stop conditions.
- **Explain aloud:** who executes the tool — the model or your code?

## Step 03 — registry & validation
- **Modify:** add `word_count(text)` to `REGISTRY` only; ask the validator to reject text longer than 1000 chars (new `max_len` rule).
- **Break it:** remove the `bool` check; call with `a=True`.
- **Rebuild:** `validate_call` for a tool with one required `str` arg and one optional `int` arg.
- **Explain aloud:** hand validation vs pydantic: pros/cons.

## Step 04 — history & context
- **Modify:** also truncate any single message longer than 300 characters (`"...[truncated]"`).
- **Break it:** remove the "drop leading assistant" loop; find which API rule could reject the result.
- **Rebuild:** `truncate` that always pins the first message.
- **Explain aloud:** truncate vs summarise trade-offs.

## Step 05 — memory
- **Modify:** add a `forget(key)` tool and a cap of 20 keys.
- **Break it:** delete `data/` and remove the `mkdir` line; read the `FileNotFoundError`.
- **Rebuild:** `load_memory` / `save_memory` / `remember` / `recall`.
- **Explain aloud:** what could go wrong if memory stores text copied from a web page?

## Step 06 — errors & retries
- **Modify:** only retry `TimeoutError` (not `ConnectionError`); add a total deadline of 2 seconds.
- **Break it:** make a permanent error (`ValueError`) happen inside `call_with_retry`; verify it is NOT retried.
- **Rebuild:** `call_with_retry` with backoff + jitter.
- **Explain aloud:** why jitter? why not retry a 400?

## Step 07 — safety
- **Modify:** add a `list_files` tool (read-only) limited to the sandbox; add a max write size of 1000 chars.
- **Break it:** remove `is_relative_to` temporarily and run `../../.env`; then fix it.
- **Rebuild:** `safe_path` and `guarded_execute`.
- **Explain aloud:** why is a sandbox folder not a security boundary?

## Step 08 — trace & full agent
- **Modify:** add the tool duration to `show_trace.py`'s summary (slowest tool call) and a `--run-id` filter.
- **Break it:** make `log()` crash on a non-JSON value (e.g. a `Path`); fix with `default=str` in `json.dumps`.
- **Rebuild:** a minimal `Agent` class with `run()` and `log()`.
- **Explain aloud:** list the statuses a run can end with and what triggers each.

## Step 09 — native tool calling
- **Modify:** add a second tool block to the simulated response and return **both** results in one user message.
- **Explain aloud:** three differences between the text protocol and native tool calling.

## Broken drills (do these without AI)
1. `broken_no_max_steps.py` — fix with a bounded loop and repetition detection.
2. `broken_unvalidated_args.py` — fix with validation; return the problem as an observation.
3. `broken_bad_json.py` — fix with `extract_json` + try/except + one repair attempt.
4. `broken_stale_history.py` — find the root cause; write one sentence on how you found it.
5. `broken_path_escape.py` — fix with resolve + `is_relative_to`; test `/etc/passwd` too.

For each: write **symptom → hypothesis → experiment → root cause → fix → how verified** (6 lines).
