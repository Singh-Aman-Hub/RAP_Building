# Step 01 — Calling an LLM and the message list

## The story so far
In [07_prompt_engineering](../../07_prompt_engineering/) the model answers from your facts and follows validated formats, but it only *talks*. Giving it hands starts with the plainest piece: how one call to a model actually works.

## The pain point
A model has no memory between calls. If you send only the newest question, "and what about its price?" has no "it". Many people think the model remembers; it does not.

## The spark
The memory is a Python list of messages that you resend, in full, on every call. Wrap the call in one function (`generate`) so the backend (fake, local, hosted) can be swapped without touching the rest.

## Goal
Call a model through one function and see that "conversation memory" is simply a Python list you resend on every call.

## Run it
```bash
cd 08_agent_harness/01_llm_api_call && ../../.venv/bin/python main.py
```
Real output:
```
user: My name is Priya
  (request had 1 messages)
bot : Nice to meet you!
user: What is my name?
  (request had 3 messages)
bot : You told me earlier: Priya

The whole 'memory' is this list:
   {'role': 'user', 'content': 'My name is Priya'}
   {'role': 'assistant', 'content': 'Nice to meet you!'}
   {'role': 'user', 'content': 'What is my name?'}
   {'role': 'assistant', 'content': 'You told me earlier: Priya'}

Without history -> I don't know your name yet.
Caught: Unknown LLM backend 'not-a-backend'. Use mock, local or anthropic.
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Message list | conversation, chat history, `messages`, context | Same thing in practice | The list you resend every call; "context" is that list as the model sees it (plus system prompt). |
| System prompt | system message, instructions, preamble | Same idea, different API slot | Anthropic takes `system=` as a separate argument; OpenAI-style APIs put it as a `role: system` message. |
| Mock backend | fake LLM, stub, test double | Yes | A scripted function that returns canned replies so the code runs offline. |
| Temperature | sampling temperature, randomness | Not same as top_p | Scales randomness of token choice; 0 = most predictable. |
| max_tokens | max_new_tokens, output cap | Near-same | Cap on reply length only, not on the prompt. |

## Code, step by step

### 1. Imports and path setup so shared helpers can be found  (`main.py` lines 9–16)

```python
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))     # <repo>/shared  -> llm_client
sys.path.insert(0, str(HERE.parent))                    # <repo>/08_agent_harness -> fake_llm
from llm_client import generate, MockLLM
from fake_llm import make_fake_agent_llm
```

- **In simple words:** We tell Python where the repo's `shared/` and module folders live so we can reuse the LLM helper and the fake brain.
- **Exactly:** `HERE` is the step folder; `HERE.parents[1]` is the repo root, so `<repo>/shared` goes on `sys.path` and `llm_client` (which gives `generate` and `MockLLM`) can be imported. `HERE.parent` is `08_agent_harness`, which holds `fake_llm.py`. `make_fake_agent_llm` is imported but unused in this step.
- **Why it is here:** Without the `sys.path.insert` calls the imports fail with ModuleNotFoundError because the helpers are not in this folder.

### 2. Import the regex module  (`main.py` lines 18–18)

```python
import re
```

- **In simple words:** Regular expressions let the mock find the name inside a sentence.
- **Exactly:** `re.search(pattern, text, flags)` returns a match object or `None`.
- **Why it is here:** Needed by `recall_name` below.

### 3. A mock brain that finds the name in past messages  (`main.py` lines 20–26)

```python
def recall_name(messages, system):
    """Mock 'brain': look back through ALL messages for 'my name is X' (this is what a real model does too)."""
    for m in messages:
        found = re.search(r"my name is (\w+)", m["content"], re.I)   # re.I = ignore case
        if found:
            return "You told me earlier: " + found.group(1)
    return "I don't know your name yet."
```

- **In simple words:** This pretend model looks through the whole conversation for "my name is X" and repeats X.
- **Exactly:** Signature `(messages, system)` is what `MockLLM` calls for a rule. It loops over every dict in `messages`, searches `content` with `my name is (\w+)` (case-insensitive via `re.I`), and returns `"You told me earlier: " + group(1)` -> `You told me earlier: Priya` in the real run. If no message matches it returns `I don't know your name yet.`.
- **Why it is here:** It shows that "remembering" works only because the earlier message is in the list; with no history the fallback string appears (see block 9).

### 4. Rules: map patterns to replies  (`main.py` lines 28–32)

```python
# A tiny conversational mock so this step reads like a chat (rules = regex -> reply).
chat_mock = MockLLM(rules=[
    (r"what.*my name", recall_name),         # checked first: a question
    (r"my name is (\w+)", "Nice to meet you!"),
])
```

- **In simple words:** A tiny table that says "if the user text looks like this, answer like that".
- **Exactly:** `MockLLM(rules=[(regex, reply_or_callable), ...])`; rules are checked in order. The question rule `what.*my name` comes first so "What is my name?" does not fall into the statement rule. A string reply is returned as-is; a callable is called with `(messages, system)`.
- **Why it is here:** Order matters: if the second rule were first, "My name is Priya" and the question could be confused.

### 5. The history list is the memory  (`main.py` lines 34–34)

```python
history = []            # THE memory of the chat: a list of {"role", "content"} dicts
```

- **In simple words:** A plain Python list that stores everything said so far.
- **Exactly:** Type `list[dict]`, each dict `{"role": "user"|"assistant", "content": str}`. Starts empty; it is module-level so it survives between `chat()` calls.
- **Why it is here:** The model has no memory of its own; this list is the only memory.

### 6. The chat function: append, call, append, report  (`main.py` lines 36–46)

```python
def chat(user_text: str) -> str:
    """Send one user message plus ALL previous messages; store both sides in `history`."""
    history.append({"role": "user", "content": user_text})
    reply = generate(history,                       # the full conversation so far
                     system="You are a friendly assistant.",   # rules, sent separately each call
                     max_tokens=200,                # hard cap on the reply length
                     temperature=0.0,               # 0 = most predictable
                     mock=chat_mock)
    history.append({"role": "assistant", "content": reply})
    print(f"  (request had {len(history) - 1} messages)")
    return reply
```

- **In simple words:** Add the user's words to history, send all of history to the model, save the answer too, and print how big the request was.
- **Exactly:** `generate(history, system=..., max_tokens=200, temperature=0.0, mock=chat_mock)` returns a `str`. The printed count `len(history) - 1` runs after the assistant reply is appended, so it equals the number of messages that were sent: 1 on the first turn, 3 on the second (real run).
- **Why it is here:** Without the second `append` the assistant's side vanishes and the next request lacks it; without sending `history` the model forgets.

### 7. Run two chat turns  (`main.py` lines 48–51)

```python
print("user: My name is Priya")
print("bot :", chat("My name is Priya"))
print("user: What is my name?")
print("bot :", chat("What is my name?"))
```

- **In simple words:** Say your name, then ask for it back.
- **Exactly:** Real output: `bot : Nice to meet you!` then `bot : You told me earlier: Priya`. The second reply works because the request held 3 messages including "My name is Priya".
- **Why it is here:** Demonstrates that growth of the message list is what gives memory.

### 8. Print the whole memory  (`main.py` lines 53–55)

```python
print("\nThe whole 'memory' is this list:")
for message in history:
    print("  ", message)
```

- **In simple words:** Show the list that makes up the conversation.
- **Exactly:** Prints 4 dicts: user, assistant, user, assistant, in order.
- **Why it is here:** Makes it visible that nothing hidden is stored.

### 9. Ask again without history  (`main.py` lines 57–60)

```python
# KEY IDEA: the model itself remembers NOTHING between calls. If you forget to resend `history`,
# it has no idea who Priya is. Test that: ----------------------------------------------------
fresh = generate([{"role": "user", "content": "What is my name?"}], mock=chat_mock)
print("\nWithout history ->", fresh)
```

- **In simple words:** A new, empty conversation does not know the name.
- **Exactly:** A one-element list is sent, so `recall_name` finds nothing: `Without history -> I don't know your name yet.`
- **Why it is here:** Proves the model remembers nothing between calls.

### 10. Catch an error on purpose  (`main.py` lines 62–66)

```python
# Error handling basics: network/API calls fail. Never let one failure produce a bare traceback in prod.
try:
    generate(history, backend="not-a-backend")      # BREAK IT on purpose: unknown backend
except ValueError as err:
    print("Caught:", err)
```

- **In simple words:** Asking for a backend that does not exist raises an error and we handle it politely.
- **Exactly:** `generate(..., backend="not-a-backend")` raises `ValueError`; printed: `Caught: Unknown LLM backend 'not-a-backend'. Use mock, local or anthropic.`
- **Why it is here:** Real API calls fail (network, rate limits); production code must catch errors instead of dying with a traceback.

### 11. Experiment comments  (`main.py` lines 68–70)

```python
# TRY IT: change temperature to 0.9 with LLM_BACKEND=anthropic - does the reply wording vary?
# TRY IT: print len(str(history)) after each turn - history (and cost!) only ever grows (see step 04).
# BREAK IT: remove `history.append({"role": "assistant", ...})` and ask the second question.
```

- **In simple words:** Suggestions for you to try, not executable code.
- **Exactly:** Comments only; nothing runs.
- **Why it is here:** Gives practice tasks: raising temperature, measuring history size, removing the assistant append.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `history = []` at module level + `history.append(...)` inside a function | a list that lives outside the function and is mutated by it (no `global` needed for mutation) | `a = []; def f(): a.append(1)` |
| `def chat(user_text: str) -> str:` | function with type hints | – |
| `re.search(r"...", text, re.I)` / `found.group(1)` | regex search, ignore case, get the first captured part | `re.search(r"(\d+)", "a12").group(1)` -> `'12'` |
| `lambda messages, system: ...` / `def recall_name(...)` | a callable the mock uses to compute a reply | – |
| `try: ... except ValueError as err:` | catch an error and use its message | – |

## Libraries & functions
| Name | Kind | Does | Important args | Returns |
|---|---|---|---|---|
| `generate` | shared helper | one call to the selected backend | `messages`, `system`, `max_tokens`, `temperature`, `backend`, `mock` | reply `str` |
| `max_tokens` | model parameter | upper bound on reply length (module [05](../../05_llm_parameters/)) | int | – |
| `MockLLM` | shared helper | fake model: `rules=[(regex, reply)]` or `script=[...]` | – | – |
| `re` | standard library | regular expressions | – | – |

## Execution trace
`chat("My name is Priya")`: append user message -> `generate(history)` with 1 message -> reply appended -> history has 2. Second call: history (3 messages) is sent; the mock's `recall_name` scans every message and finds "my name is Priya". A fresh list (no history) knows nothing.

## Experiments
- Remove the line that appends the assistant reply and re-ask: the second request lacks the bot's own turns (and a real model may get confused about who said what).
- Print `len(str(history))` each turn: the request grows, so cost and latency grow with it (step 04).

## Debugging practice
`ValueError: Unknown LLM backend 'not-a-backend'` is raised on purpose. Read the traceback bottom-up: last line = what, lines above = where. Typical real failures here: `401` bad API key, `429` rate limit, timeouts (step 06).

## Rebuild checklist
- [ ] `history` list, a `chat()` function, system prompt passed separately, assistant reply stored.

## The new problem (cliffhanger)
One call answers one question. But a task like "calculate 12 + 8, then multiply by 3" needs the model to act, look at the result and decide again. We need a loop. Next: [02_agent_loop](../02_agent_loop/).

> **Say it aloud (interview box):** An LLM call is stateless, so conversation memory is just a list of role-and-content messages that I resend on every call. I hide the backend behind one function so I can swap a fake, a local and a hosted model. Every call costs tokens for the whole list, which is why history later has to be managed.
