# Terms — Step 04: Conversation history and context-window limits

Folder: `08_agent_harness/04_history_and_context` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [08 Agent harness README](../README.md)

**Terms in this folder:** [Context window](#context-window) · [Token](#token) · [Token estimate (characters divided by 4)](#token-estimate-characters-divided-by-4) · [Floor division (//)](#floor-division) · [sum with a generator expression](#sum-with-a-generator-expression) · [Truncation](#truncation) · [Pinning the first message](#pinning-the-first-message) · [Summarisation](#summarisation) · [Slicing ([0], [1:], [:-4], [-4:])](#slicing-0-1--4--4) · [Multiple assignment (tuple unpacking)](#multiple-assignment-tuple-unpacking) · [reversed](#reversed) · [list.insert(0, x)](#listinsert0-x) · [list.pop(0)](#listpop0) · [while loop with a compound condition](#while-loop-with-a-compound-condition) · [Alternating roles requirement](#alternating-roles-requirement) · [String repetition ("text" * 6)](#string-repetition-text--6) · [str.join (transcript)](#strjoin-transcript) · [Summary message](#summary-message) · [MockLLM with a rule (summary mock)](#mockllm-with-a-rule-summary-mock) · [prepare_for_call](#prepare_for_call) · [Truncate tool observations](#truncate-tool-observations) · [Trade-offs of truncation vs summarisation](#trade-offs-of-truncation-vs-summarisation) · [Expressions inside f-string braces](#expressions-inside-f-string-braces)

---

## Context window
*Type: concept*

**1. Definition:** The maximum number of tokens (input plus output) a model can consider in one request.
**2. In plain words:** The model can only "read" this much at once. A long chat eventually exceeds it, so older turns must be dropped or compressed. Bigger windows cost more and are slower.
**3. Example to quote:**
> "I pretend the window allows only `BUDGET = 300` tokens for history, so the 39-message chat has to be shortened."

**Seen in:** `main.py`, `BUDGET`.

## Token
*Type: concept*

**1. Definition:** The unit of text a model reads and writes, usually a word piece; roughly four characters of English on average.
**2. In plain words:** Models count tokens, not words or characters, for limits and billing. Module 04 shows a real tokenizer.
**3. Example to quote:**
> "The original chat was about 1226 tokens by my estimate, far over a budget of 300."

**Seen in:** `main.py`, `estimate_tokens`.

## Token estimate (characters divided by 4)
*Type: project function / heuristic*

**1. Definition:** `estimate_tokens(messages)` returns the total characters of all message contents integer-divided by 4, a rule of thumb for English text.
**2. In plain words:** It is not exact, but it is good enough to decide when to trim. For real counts use the model's tokenizer.
**3. Example to quote:**
> "`sum(len(m['content']) for m in messages) // 4` told me the chat was about 1226 tokens."

**Seen in:** `main.py`, `estimate_tokens`.

## Floor division (`//`)
*Type: Python syntax*

**1. Definition:** The `//` operator divides and rounds down to a whole number.
**2. In plain words:** Token counts are whole numbers, so dividing by 4 with `//` drops any remainder.
**3. Example to quote:**
> "`9 // 4` is 2, so a 9-character message counts as two tokens."

```python
9 // 4   # 2
```

**Seen in:** `main.py`, `estimate_tokens`.

## sum with a generator expression
*Type: Python built-in / syntax*

**1. Definition:** `sum(len(m["content"]) for m in messages)` adds up values produced one at a time by a generator expression.
**2. In plain words:** It totals the lengths of all messages without building a temporary list.
**3. Example to quote:**
> "It adds the character count of every message in the history."

**Seen in:** `main.py`, `estimate_tokens`.

## Truncation
*Type: concept*

**1. Definition:** Keeping only the first message and the newest messages that fit the budget, discarding the middle.
**2. In plain words:** It is free and fast but forgets. Pinning the first message keeps the original goal (here the Goa budget) alive.
**3. Example to quote:**
> "Truncation cut 39 messages to 10 and lost the middle 29, but kept message 0 with the 30000 rupee budget."

**Seen in:** `main.py`, `truncate`.

## Pinning the first message
*Type: concept*

**1. Definition:** Always keeping the opening message (which often states the goal or key facts) when trimming history.
**2. In plain words:** Without it, the model forgets what the conversation was about once the start scrolls out of the window.
**3. Example to quote:**
> "I remove the pin in the experiment and the Goa budget is gone from the history."

**Seen in:** `main.py`, `truncate`; TRY IT comment.

## Summarisation
*Type: concept*

**1. Definition:** Asking the model to compress older turns into a short summary message and keeping recent turns verbatim.
**2. In plain words:** It keeps the meaning of old turns in far fewer tokens, but costs an extra model call and can lose or distort details.
**3. Example to quote:**
> "The summary turned 35 old messages into one sentence, giving 5 messages and about 166 tokens."

**Seen in:** `main.py`, `summarise_and_trim`.

## Slicing (`[0]`, `[1:]`, `[:-4]`, `[-4:]`)
*Type: Python syntax*

**1. Definition:** `list[a:b]` returns a sub-list; negative indices count from the end, so `[-4:]` is the last four items and `[:-4]` is everything except them.
**2. In plain words:** It splits the history into "old" and "recent" in one line each.
**3. Example to quote:**
> "`old, recent = messages[:-keep_last], messages[-keep_last:]` separates the part to summarise from the four newest messages."

```python
[1, 2, 3, 4][-2:]   # [3, 4]
```

**Seen in:** `main.py`, `truncate`, `summarise_and_trim`.

## Multiple assignment (tuple unpacking)
*Type: Python syntax*

**1. Definition:** Assigning several names in one statement, such as `first, rest = messages[0], messages[1:]`.
**2. In plain words:** It names both halves of the split at once.
**3. Example to quote:**
> "`first, rest = messages[0], messages[1:]` gives me the pinned message and everything after it."

**Seen in:** `main.py`, `truncate`.

## reversed
*Type: Python built-in*

**1. Definition:** `reversed(sequence)` returns an iterator over the items from last to first without modifying the original.
**2. In plain words:** Walking from the newest message backwards lets you stop when the budget is full, keeping the freshest turns.
**3. Example to quote:**
> "`for message in reversed(rest)` visits the newest message first."

**Seen in:** `main.py`, `truncate`.

## list.insert(0, x)
*Type: Python built-in*

**1. Definition:** Inserts an item at the given position; index 0 puts it at the front.
**2. In plain words:** Because the loop goes newest to oldest, each kept message is inserted at the front to restore chronological order.
**3. Example to quote:**
> "`kept.insert(0, message)` rebuilds the order oldest to newest."

```python
a = [2]; a.insert(0, 1)   # [1, 2]
```

**Seen in:** `main.py`, `truncate`.

## list.pop(0)
*Type: Python built-in*

**1. Definition:** Removes and returns the item at index 0.
**2. In plain words:** After cutting mid-pair, the first kept message may be from the assistant. Removing it keeps the history starting with a user turn, which some APIs require.
**3. Example to quote:**
> "`while kept and kept[0]['role'] == 'assistant': kept.pop(0)` fixes a history that starts with an assistant message."

**Seen in:** `main.py`, `truncate`.

## while loop with a compound condition
*Type: Python syntax*

**1. Definition:** `while cond1 and cond2:` repeats while both conditions hold; `and` stops evaluating if the first is false.
**2. In plain words:** `kept and ...` first checks the list is not empty so `kept[0]` cannot raise `IndexError`.
**3. Example to quote:**
> "The loop checks that `kept` is non-empty before reading `kept[0]`."

**Seen in:** `main.py`, `truncate`.

## Alternating roles requirement
*Type: concept*

**1. Definition:** Many chat APIs expect the message list to start with a user turn (and sometimes to alternate user and assistant turns).
**2. In plain words:** Cutting history at a random point can leave an assistant message first, which some APIs reject, so the code drops it.
**3. Example to quote:**
> "If I remove the `while` loop, a history that starts with the assistant may be rejected by a real API."

**Seen in:** `main.py`, comment in `truncate`; BREAK IT comment.

## String repetition (`"text" * 6`)
*Type: Python syntax*

**1. Definition:** Multiplying a string by an integer repeats it that many times.
**2. In plain words:** It is used to make long fake assistant messages quickly.
**3. Example to quote:**
> "`'book early and compare prices. ' * 6` builds a long tip message."

```python
"ab" * 2   # 'abab'
```

**Seen in:** `main.py`, fake conversation.

## str.join (transcript)
*Type: Python built-in*

**1. Definition:** `"\n".join(lines)` concatenates strings with a newline between them.
**2. In plain words:** It turns the list of old messages into a single transcript text to send to the summariser.
**3. Example to quote:**
> "Each old message becomes a line like `user: Hi! I am planning a trip...` and they are joined with newlines."

**Seen in:** `main.py`, `summarise_and_trim`.

## Summary message
*Type: concept*

**1. Definition:** A single message placed at the start of the trimmed history that contains the model-written summary, here prefixed `[Summary of earlier conversation]`.
**2. In plain words:** It replaces many old messages with one, keeping key facts like the destination and budget.
**3. Example to quote:**
> "The history now begins with the summary message and ends with the four newest turns."

**Seen in:** `main.py`, `summarise_and_trim`.

## MockLLM with a rule (summary mock)
*Type: project helper*

**1. Definition:** `MockLLM(rules=[(r"summar", "...")])` returns a fixed summary whenever the prompt contains "summar".
**2. In plain words:** It stands in for the real model so the summarising code can be tested offline.
**3. Example to quote:**
> "The mock returns a canned two-sentence summary, so the demo is deterministic."

**Seen in:** `main.py`, `summary_mock`.

## prepare_for_call
*Type: project function*

**1. Definition:** A function run before each model call that returns the messages unchanged if they fit the budget, otherwise a summarised version.
**2. In plain words:** Checking before every call means the history can never silently overflow the window.
**3. Example to quote:**
> "`prepare_for_call(history)` returned the compact version at about 166 tokens because the full one was far over budget."

**Seen in:** `main.py`, `prepare_for_call`.

## Truncate tool observations
*Type: concept*

**1. Definition:** Cutting very long tool results (for example the contents of a big file) to a maximum length, such as the first 2000 characters plus a truncation marker.
**2. In plain words:** One huge observation can fill the whole window on its own, so trim it as well as the chat.
**3. Example to quote:**
> "I keep the first 2000 characters of a file result and add `...[truncated]`."

**Seen in:** `main.py`, trade-offs comment.

## Trade-offs of truncation vs summarisation
*Type: concept*

**1. Definition:** Truncation is free but forgets; summarisation keeps meaning but costs an extra call and may distort details; both change what the model knows.
**2. In plain words:** Pick based on how much accuracy matters versus cost. If a fact disappears, check whether you trimmed it.
**3. Example to quote:**
> "When the agent forgot the budget, I printed the exact messages sent and saw I had trimmed it."

**Seen in:** `main.py`, trade-offs comment; `WALKTHROUGH.md`, debugging practice.

## Expressions inside f-string braces
*Type: Python syntax*

**1. Definition:** f-strings can contain expressions such as `{len(history) - len(short)}` and call functions, evaluated when the line runs.
**2. In plain words:** The summary line reports the number of lost messages by computing it inline.
**3. Example to quote:**
> "`(lost the middle {len(history) - len(short)} messages)` printed 29."

**Seen in:** `main.py`, print after `truncate`.
