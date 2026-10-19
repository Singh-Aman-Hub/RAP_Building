# Step 01 — Calling an LLM and the message list

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

## Say it aloud
"An LLM call is stateless: I send the system prompt plus the full list of previous messages and get the next message back. Memory is that list. That makes history size my problem — longer history means more tokens, cost and latency, and eventually the context window limit."
