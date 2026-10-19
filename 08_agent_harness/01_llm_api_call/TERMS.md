# Terms — Step 01: Calling an LLM and the message list

Folder: `08_agent_harness/01_llm_api_call` · Files covered: `main.py`, `WALKTHROUGH.md` (uses `../../shared/llm_client.py` and `../fake_llm.py`)
Back to module: [08 Agent harness README](../README.md)

**Terms in this folder:** [LLM API call](#llm-api-call) · [Stateless model (memory is a list)](#stateless-model-memory-is-a-list) · [messages list](#messages-list) · [generate](#generate) · [System prompt (passed separately)](#system-prompt-passed-separately) · [max_tokens](#max_tokens) · [temperature](#temperature) · [Mock model (MockLLM)](#mock-model-mockllm) · [Regular expression (re.search, group, re.I)](#regular-expression-research-group-rei) · [Callable reply (function as mock rule)](#callable-reply-function-as-mock-rule) · [Module-level list mutated in a function](#module-level-list-mutated-in-a-function) · [Type hints on a function](#type-hints-on-a-function) · [sys.path.insert and HERE.parents](#syspathinsert-and-hereparents) · [try / except ValueError](#try--except-valueerror) · [Traceback (bottom-up reading)](#traceback-bottom-up-reading) · [Environment variable for a real backend](#environment-variable-for-a-real-backend) · [Fake agent LLM (fake_llm.make_fake_agent_llm)](#fake-agent-llm-fake_llmmake_fake_agent_llm) · [History growth (cost and context window)](#history-growth-cost-and-context-window)

---

## LLM API call
*Type: concept*

**1. Definition:** A request from your program to a language model service (or local model) that sends messages and receives the model's next message.
**2. In plain words:** Your code does not "talk" to the model continuously; it makes one request, gets one reply and the model forgets everything afterwards. Everything the model should know must be sent again in the next request.
**3. Example to quote:**
> "Each call to `generate` is a separate request. The model only knew Priya's name because I resent the full history."

**Seen in:** `main.py`, `generate(history, ...)`.

## Stateless model (memory is a list)
*Type: concept*

**1. Definition:** A model that keeps no information between calls, so conversation "memory" is a list of messages that the caller stores and resends.
**2. In plain words:** If you forget to send the history, the model has no idea what was said. That makes history size your responsibility: longer history means more tokens, cost and latency, and eventually the context window limit.
**3. Example to quote:**
> "Without history it answered 'I don't know your name yet.' With the four-message history it said 'You told me earlier: Priya.'"

**Seen in:** `main.py`, `history` and the `fresh` call.

## messages list
*Type: concept*

**1. Definition:** A list of dictionaries with `role` and `content` keys in conversation order, the standard chat input format.
**2. In plain words:** Each turn is a small dict. User turns and assistant turns alternate. You append to the list and resend it.
**3. Example to quote:**
> "`{'role': 'user', 'content': 'My name is Priya'}` is the first entry of my history."

**Seen in:** `main.py`, `history.append(...)`.

## generate
*Type: project function*

**1. Definition:** `shared/llm_client.py::generate(messages, system, max_tokens, temperature, backend, mock)` sends the messages to the chosen backend and returns the reply as text.
**2. In plain words:** One function hides the difference between the offline fake, a local model and a hosted model. Choose with the `backend` argument or the `LLM_BACKEND` environment variable.
**3. Example to quote:**
> "I call `generate(history, system='You are a friendly assistant.', max_tokens=200, temperature=0.0, mock=chat_mock)` and get the next reply as a string."

**Seen in:** `main.py`, `chat()`.

## System prompt (passed separately)
*Type: concept*

**1. Definition:** Standing instructions passed through a dedicated `system` argument rather than as a message in the list.
**2. In plain words:** Rules such as "You are a friendly assistant" are sent with every call but are not part of the stored history, so they do not pile up as turns.
**3. Example to quote:**
> "`system='You are a friendly assistant.'` is passed each call but never appended to `history`."

**Seen in:** `main.py`, `chat()`.

## max_tokens
*Type: model parameter*

**1. Definition:** A parameter capping the number of tokens the model may generate in its reply.
**2. In plain words:** It protects against runaway long replies and runaway cost. If the cap is too low the reply is cut off mid-sentence. See module 05 for details.
**3. Example to quote:**
> "I set `max_tokens=200` so no single reply can exceed 200 tokens."

**Seen in:** `main.py`, `generate(... max_tokens=200 ...)`.

## temperature
*Type: model parameter*

**1. Definition:** A sampling parameter controlling randomness; 0 picks the most probable token, higher values vary more.
**2. In plain words:** For predictable agent behaviour and reproducible tests you want 0. Higher values make wording vary between runs.
**3. Example to quote:**
> "I keep `temperature=0.0` so the same question gets the same wording, which makes debugging easier."

**Seen in:** `main.py`, `generate(... temperature=0.0 ...)`; TRY IT comment.

## Mock model (`MockLLM`)
*Type: project helper*

**1. Definition:** `MockLLM` from `shared/llm_client.py`, a fake model driven by `rules=[(regex, reply)]` or `script=[...]`; it records every call in `.calls`.
**2. In plain words:** It lets you study the code offline and deterministically. A real model will behave less neatly, which is why real code must be defensive.
**3. Example to quote:**
> "`chat_mock` has two rules: a question rule and a statement rule. It is a stand-in so I can practise without an API key."

**Seen in:** `main.py`, `chat_mock = MockLLM(rules=[...])`.

## Regular expression (`re.search`, `group`, `re.I`)
*Type: Python standard library*

**1. Definition:** `re.search(pattern, text, flags)` finds the first match of a pattern; `match.group(1)` returns the first parenthesised part; `re.I` ignores letter case.
**2. In plain words:** The fake model uses a regex to find "my name is X" anywhere in the conversation. If nothing matches, `search` returns `None`, so check before calling `.group`.
**3. Example to quote:**
> "`re.search(r'my name is (\\w+)', text, re.I)` captured `Priya` from 'My name is Priya'."

```python
import re
re.search(r"(\d+)", "a12").group(1)   # '12'
```

**Seen in:** `main.py`, `recall_name`.

## Callable reply (function as mock rule)
*Type: Python syntax / concept*

**1. Definition:** In `MockLLM` rules the reply can be a function `(messages, system) -> str` instead of a fixed string, and it is called to compute the answer.
**2. In plain words:** The fake needs to inspect the whole conversation to "remember" a name, so it uses a function. Functions in Python can be passed around like any value.
**3. Example to quote:**
> "`recall_name` scans every message in the list, just as a real model attends to the whole conversation."

**Seen in:** `main.py`, `recall_name` used in `chat_mock`.

## Module-level list mutated in a function
*Type: Python syntax*

**1. Definition:** A list defined outside a function can be changed inside the function with methods like `.append` without a `global` statement, because the name is not reassigned.
**2. In plain words:** `history` lives at the top of the file and `chat()` appends to it. You would need `global` only if you wrote `history = []` inside the function.
**3. Example to quote:**
> "`chat()` appends to the shared `history` list; because I never reassign the name, no `global` is needed."

**Seen in:** `main.py`, `history` and `chat`.

## Type hints on a function
*Type: Python syntax*

**1. Definition:** Annotations such as `user_text: str` and `-> str` that document the expected types; Python does not enforce them.
**2. In plain words:** They tell readers and editors what the function takes and returns.
**3. Example to quote:**
> "`def chat(user_text: str) -> str` says it takes the user's text and returns the bot's reply."

**Seen in:** `main.py`, `chat`.

## sys.path.insert and HERE.parents
*Type: Python standard library*

**1. Definition:** `sys.path` lists folders Python searches for imports; inserting a folder lets you import modules from it. `HERE.parents[1]` is the folder two levels above the script.
**2. In plain words:** The shared client and the fake model live in other folders, so the script adds those folders to the search path before importing.
**3. Example to quote:**
> "I insert `<repo>/shared` into `sys.path` so `from llm_client import generate` works from any folder."

**Seen in:** `main.py`, import setup.

## try / except ValueError
*Type: Python syntax*

**1. Definition:** A `try` block runs code; if a `ValueError` is raised, the `except ValueError as err` block runs with the exception in `err`.
**2. In plain words:** API calls fail in real life (bad key, no network, bad argument). Catching the error lets you print a clear message or retry instead of showing a bare traceback.
**3. Example to quote:**
> "`generate(history, backend='not-a-backend')` raises `ValueError`, and I print 'Caught: Unknown LLM backend...'."

**Seen in:** `main.py`, bottom section.

## Traceback (bottom-up reading)
*Type: concept*

**1. Definition:** The printed record of the call chain leading to an unhandled exception; the last line states the error type and message.
**2. In plain words:** Read from the bottom: the last line tells you what went wrong, the lines above tell you where. Typical API failures are 401 (bad key), 429 (rate limit) and timeouts.
**3. Example to quote:**
> "The last line said `Unknown LLM backend 'not-a-backend'`, so I knew the argument was wrong before reading anything else."

**Seen in:** `WALKTHROUGH.md`, debugging practice.

## Environment variable for a real backend
*Type: concept / command*

**1. Definition:** Settings like `LLM_BACKEND=anthropic` and `ANTHROPIC_API_KEY=...` placed in the shell environment to choose and authorise the real model.
**2. In plain words:** The same script runs with the fake model by default and with a real model when you set the variable, with no code change. Never hard-code a key in the file.
**3. Example to quote:**
> "`LLM_BACKEND=anthropic ANTHROPIC_API_KEY=... python main.py` runs the same chat against a hosted model."

```bash
LLM_BACKEND=anthropic python main.py
```

**Seen in:** `main.py` docstring (`NEEDS`).

## Fake agent LLM (`fake_llm.make_fake_agent_llm`)
*Type: project helper*

**1. Definition:** A `MockLLM` with one catch-all rule that routes to `fake_reply`, which answers in the module's JSON tool protocol. It is imported here but used mainly from step 02.
**2. In plain words:** It is the scripted "brain" used by every step in this module so the harness can be studied deterministically.
**3. Example to quote:**
> "I import `make_fake_agent_llm` for later steps; in this step I use a simple chat mock instead."

**Seen in:** `main.py`, `from fake_llm import make_fake_agent_llm`.

## History growth (cost and context window)
*Type: concept*

**1. Definition:** Because the whole history is resent each call, request size, token cost and latency grow every turn until the context window limit is reached.
**2. In plain words:** The first request had 1 message and the second had 3. In a long chat this becomes expensive and eventually fails, which is why step 04 trims history.
**3. Example to quote:**
> "The log said 'request had 1 messages' then 'request had 3 messages'. History only grows."

**Seen in:** `main.py`, print in `chat()`; TRY IT comment.
