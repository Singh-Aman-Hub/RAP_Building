# Step 04 — Conversation history and context-window limits

## The story so far
In [step 03](../03_tool_registry_and_validation/) calls are checked, so the agent can run long. Every step appends two messages.

## The pain point
The whole list is resent on every call. It gets slower and more expensive, and past the context window the call fails or the oldest facts are silently cut.

## The spark
Budget the history: estimate tokens, always keep the first user message, keep the newest messages, and summarise or drop the middle.

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

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Context window | context length, max context, token limit | Same | The max tokens (prompt + reply) a model can read in one call. |
| Truncation | trimming, sliding window, windowing | Near-same | Drop old messages; a sliding window keeps only the newest N. |
| Summarisation | compaction, compression, condensing | Near-same | Replace old turns with a model-written summary (costs an extra call). |
| Token estimate | token count, tokenisation | Not same | chars/4 is a guess; a real tokenizer gives the exact count. |
| History | transcript, message list, short-term memory | Same | The list resent each call. |

## Code, step by step

### 1. Imports and path setup  (`main.py` lines 8–13)

```python
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
from llm_client import generate, MockLLM
```

- **In simple words:** Make the shared LLM helper importable.
- **Exactly:** `generate` calls a backend; `MockLLM` is the scripted brain used for summaries.
- **Why it is here:** Summarisation needs a model call.

### 2. Rough token counter  (`main.py` lines 15–17)

```python
def estimate_tokens(messages) -> int:
    """Rough rule of thumb for English: 1 token ~ 4 characters. Good enough to decide WHEN to trim."""
    return sum(len(m["content"]) for m in messages) // 4
```

- **In simple words:** Guess size by counting characters and dividing by 4.
- **Exactly:** `sum(len(content)) // 4` (integer division). English averages about 4 characters per token. Real run: 39 messages -> ~1226 tokens.
- **Why it is here:** You need a cheap number to decide when to trim; exact counting needs the model's tokenizer.

### 3. Build a long fake chat  (`main.py` lines 19–24)

```python
# ---- a long fake conversation ------------------------------------------------------------
history = [{"role": "user", "content": "Hi! I am planning a trip to Goa in December. Budget is 30000 rupees."}]
for i in range(1, 20):
    history.append({"role": "assistant", "content": f"Tip {i}: " + "book early and compare prices. " * 6})
    history.append({"role": "user", "content": f"Thanks, and question {i}: what about local transport option {i}?"})
print(f"original: {len(history)} messages, ~{estimate_tokens(history)} tokens")
```

- **In simple words:** Create a pretend conversation that is too long.
- **Exactly:** One opening user message, then 19 pairs of assistant tip + user question = 1 + 38 = 39 messages (the module docstring says 40; the real count is 39). Printed: `original: 39 messages, ~1226 tokens`.
- **Why it is here:** Gives the trimming strategies something to compress; the first message holds the goal (Goa, 30000 rupees).

### 4. Set the token budget  (`main.py` lines 26–26)

```python
BUDGET = 300   # pretend the context window only allows this many tokens for history (real ones: thousands)
```

- **In simple words:** Pretend the window only allows 300 tokens.
- **Exactly:** `BUDGET = 300` (tiny so the effect is visible; real windows are thousands to millions).
- **Why it is here:** Without a limit there is nothing to trim to.

### 5. Truncate: keep the first message and the newest ones  (`main.py` lines 31–42)

```python
def truncate(messages, budget):
    first, rest = messages[0], messages[1:]
    kept = []
    # walk backwards (newest first) and stop when the budget is full
    for message in reversed(rest):
        if estimate_tokens([first] + [message] + kept) > budget:
            break
        kept.insert(0, message)                  # insert at the FRONT to restore chronological order
    # Many APIs expect the history to start with a user turn: drop a leading assistant message if we cut mid-pair
    while kept and kept[0]["role"] == "assistant":
        kept.pop(0)
    return [first] + kept
```

- **In simple words:** Walk backwards from the newest message, keep what fits, and always keep message 0.
- **Exactly:** `first, rest = messages[0], messages[1:]`. For each message in `reversed(rest)`, stop when `estimate_tokens([first]+[message]+kept) > budget`; `kept.insert(0, ...)` restores order. Then a `while` pops leading assistant messages so the history starts with a user turn after `first`. Returns `[first] + kept`.
- **Why it is here:** Pinning the first message keeps the goal; dropping a leading assistant turn avoids API errors about role order.

### 6. Run truncation  (`main.py` lines 44–46)

```python
short = truncate(history, BUDGET)
print(f"truncated: {len(short)} messages, ~{estimate_tokens(short)} tokens "
      f"(lost the middle {len(history) - len(short)} messages)")
```

- **In simple words:** See how much was cut.
- **Exactly:** Real: `truncated: 10 messages, ~288 tokens (lost the middle 29 messages)`, under the 300 budget.
- **Why it is here:** Confirms the budget is respected.

### 7. A mock summariser  (`main.py` lines 51–52)

```python
summary_mock = MockLLM(rules=[(r"summar", "User plans a Goa trip in December with a 30000 rupee budget; "
                                           "we gave booking tips and discussed local transport options.")])
```

- **In simple words:** A fake model that answers any "summar..." request with a fixed summary.
- **Exactly:** `MockLLM(rules=[(r"summar", "...")])` matches the regex "summar" in the request.
- **Why it is here:** Keeps the demo offline; a real model would write the summary.

### 8. Summarise old turns, keep the recent ones  (`main.py` lines 54–61)

```python
def summarise_and_trim(messages, keep_last=4):
    old, recent = messages[:-keep_last], messages[-keep_last:]
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in old)
    summary = generate([{"role": "user", "content": "Summarise this conversation in 2 sentences, "
                                                    "keeping facts and decisions:\n" + transcript}],
                       mock=summary_mock)
    # the summary becomes ONE message at the start; recent turns follow unchanged
    return [{"role": "user", "content": f"[Summary of earlier conversation] {summary}"}] + recent
```

- **In simple words:** Squash the old part into one message and keep the last four turns word for word.
- **Exactly:** `messages[:-keep_last]` is old, `messages[-keep_last:]` is recent. `transcript` joins `role: content` lines. `generate(...)` returns the summary text. The result is one `user` message prefixed `[Summary of earlier conversation]` plus 4 recent = 5 messages.
- **Why it is here:** Keeps facts from the whole chat at a fraction of the size, at the cost of one extra model call.

### 9. Run summarisation  (`main.py` lines 63–65)

```python
compact = summarise_and_trim(history)
print(f"summarised: {len(compact)} messages, ~{estimate_tokens(compact)} tokens")
print("\nsummary message:", compact[0]["content"])
```

- **In simple words:** Compare sizes and print the summary message.
- **Exactly:** Real: `summarised: 5 messages, ~166 tokens`, followed by the summary text.
- **Why it is here:** Shows summarising is smaller than truncating here (166 vs 288 tokens).

### 10. Decide whether to compress  (`main.py` lines 70–73)

```python
def prepare_for_call(messages, budget=BUDGET):
    if estimate_tokens(messages) <= budget:
        return messages                          # fits: send as-is
    return summarise_and_trim(messages)          # too big: compress
```

- **In simple words:** Only compress when the history is too big.
- **Exactly:** If `estimate_tokens(messages) <= budget` return unchanged, else `summarise_and_trim`.
- **Why it is here:** Always compressing wastes calls and loses detail.

### 11. Prepare history before the next call  (`main.py` lines 75–75)

```python
print("\nprepared for the next call: ~", estimate_tokens(prepare_for_call(history)), "tokens")
```

- **In simple words:** Run the check on the full history.
- **Exactly:** Real: `prepared for the next call: ~ 166 tokens` (the 1226-token history exceeded 300).
- **Why it is here:** This is where the helper plugs into an agent loop.

### 12. Trade-off notes and experiments  (`main.py` lines 77–82)

```python
# Trade-offs: truncation is free but forgets (the Goa budget survives only because we pinned message 0);
# summarisation costs an extra model call and can lose or distort details; both change what the model "knows".
# Tool results can be HUGE (a whole file) - truncate observations too (e.g. first 2000 chars + '...[truncated]').
# TRY IT: set BUDGET = 150 and 1000; watch how many messages survive.
# TRY IT: pin the first message in truncate() ... then remove that pin; what does the model forget?
# BREAK IT: remove the `while kept ... == "assistant"` loop; some real APIs reject a history whose first turn is the assistant.
```

- **In simple words:** Comments about costs and things to try.
- **Exactly:** Comments only. Observations (tool results) can also be huge and should be truncated.
- **Why it is here:** Reminds you of limits of both strategies.

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

## The new problem (cliffhanger)
Trimming forgets old turns for good, and every run starts from a blank list. The agent cannot remember that your name is Asha tomorrow. Next: [05_memory](../05_memory/).

> **Say it aloud (interview box):** The whole message list is resent on every call, so I estimate its token count and trim it to a budget. I keep the first user message and the newest turns, and summarise or drop the middle. Trimming is lossy, so anything that must survive goes into memory instead.
