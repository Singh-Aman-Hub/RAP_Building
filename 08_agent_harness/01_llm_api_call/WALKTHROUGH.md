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
