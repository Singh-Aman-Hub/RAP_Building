# Step 04 — Conversation history and context-window limits

## Goal
Stop history from growing past the model's context window by estimating tokens, truncating and summarising.

## Run it
```bash
cd 08_agent_harness/04_history_and_context && ../../.venv/bin/python main.py
```
Real output:
```
original: 39 messages, ~1226 tokens
truncated: 10 messages, ~288 tokens (lost the middle 29 messages)
summarised: 5 messages, ~166 tokens

summary message: [Summary of earlier conversation] User plans a Goa trip in December with a 30000 rupee budget; we gave booking tips and discussed local transport options.

prepared for the next call: ~ 166 tokens
```

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `messages[0]`, `messages[1:]`, `messages[-4:]`, `messages[:-4]` | index / slice from the start or end | `[1,2,3,4][-2:]` -> `[3, 4]` |
| `first, rest = messages[0], messages[1:]` | assign two values at once | `a, b = 1, 2` |
| `reversed(list)` | iterate newest -> oldest | `list(reversed([1,2]))` |
| `list.insert(0, x)` | put at the front | `a=[2]; a.insert(0,1)` |
| `while kept and kept[0]["role"] == "assistant":` | loop while the list is non-empty and a condition holds | – |
| `"str " * 6` | repeat a string | `"ab" * 2` -> `abab` |

## Libraries & functions
| Name | Kind | Does | Returns |
|---|---|---|---|
| `estimate_tokens` | our helper | characters // 4 as a rough token count (real tokenizers: module [04](../../04_llm_fundamentals/)) | int |
| `sum(generator)` | built-in | add up lengths | int |
| `str.join` | built-in | join lines of a transcript | str |
| `generate` | shared | model call used to summarise | str |

## Execution trace
39 messages (~1226 tokens) vs budget 300. Truncate: keep message 0 (the goal and budget) + as many newest messages as fit -> 10 messages. Summarise: ask the model to compress everything except the last 4 messages -> 1 summary message + 4 = 5 messages, ~166 tokens. `prepare_for_call` decides which to use based on the estimate.

## Experiments
- `BUDGET = 150` / `1000`: how many messages survive?
- Remove the pinned first message: the Goa budget is forgotten.
- Truncate tool observations separately (e.g. keep 2000 characters).

## Debugging practice
If the agent "forgets" a fact, print the exact `messages` list sent on that call and check whether the fact is still in it. Most "the model forgot" bugs are "we trimmed it".

## Rebuild checklist
- [ ] `estimate_tokens`. [ ] `truncate` keeping first + newest. [ ] `summarise_and_trim`. [ ] A check before each call.

## Say it aloud
"A model can only read a fixed context window, and cost and latency grow with it. When the history approaches the budget I truncate (cheap, but forgets — so I pin the first message) or summarise older turns with an extra model call (keeps meaning, can distort details). I check the size before every call."
