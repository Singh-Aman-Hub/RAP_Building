# Terms — Shared LLM Client

Folder: `shared` · Files covered: `llm_client.py`
Back to repo: [README](../README.md)

**Terms in this folder:** [Backend (mock / local / anthropic)](#backend-mock--local--anthropic) · [LLM_BACKEND environment variable and os.getenv](#llm_backend-environment-variable-and-osgetenv) · [Message format (role / content)](#message-format-role--content) · [System prompt (separate argument)](#system-prompt-separate-argument) · [`from __future__ import annotations`](#from-__future__-import-annotations) · [Type hints (`list[str] | None`, Callable)](#type-hints-liststr--none-callable) · [MockLLM](#mockllm) · [Scripted mode (script=[...])](#scripted-mode-script) · [Rule mode (rules=[(regex, reply)])](#rule-mode-rulesregex-reply) · [re.search and re.IGNORECASE](#research-and-reignorecase) · [list.pop(0)](#listpop0) · [next() with a default and reversed()](#next-with-a-default-and-reversed) · [Recording calls (self.calls)](#recording-calls-selfcalls) · [Lazy import](#lazy-import) · [Module-level cache with `global`](#module-level-cache-with-global) · [transformers pipeline("text-generation")](#transformers-pipelinetext-generation) · [do_sample and temperature](#do_sample-and-temperature) · [Chat template output indexing](#chat-template-output-indexing) · [anthropic.Anthropic client and messages.create](#anthropicanthropic-client-and-messagescreate) · [Response content blocks](#response-content-blocks) · [max_tokens](#max_tokens) · [generate() dispatcher](#generate-dispatcher) · [extract_json](#extract_json) · [Greedy regex with re.DOTALL](#greedy-regex-with-redotall) · [json.loads and JSONDecodeError](#jsonloads-and-jsondecodeerror) · [`if __name__ == "__main__"` self-test](#if-__name__--__main__-self-test)

---

## Backend (mock / local / anthropic)  `(concept)`
**1. Definition:** A backend is the concrete system that answers an LLM request; this file offers three behind one `generate()` function: `mock` (scripted fake), `local` (small Hugging Face model) and `anthropic` (hosted API).
**2. In plain words:** Your agent and prompt code always call the same function, and only an environment variable decides what answers. That lets you test the harness offline for free and swap in a real model later. The mock proves your plumbing works, not that a prompt is good for a real model.
**3. Example to quote:** > "My examples run on the mock backend by default; to try a real model I only set `LLM_BACKEND=local` or `anthropic`, with no code change."
**Seen in:** `llm_client.py`, docstring and `generate()`

## LLM_BACKEND environment variable and os.getenv  `(standard library)`
**1. Definition:** An environment variable is a named setting provided by the shell to a process; `os.getenv(name, default)` reads it and returns the default if it is not set.
**2. In plain words:** It lets you change behaviour without editing code: `LLM_BACKEND=local python script.py`. The same idea appears later in APIs and Docker (`ENV`, `-e`).
**3. Example to quote:** > "`os.getenv('LLM_BACKEND', 'mock')` returns 'mock' unless I launch the script with `LLM_BACKEND=anthropic`."
```python
import os
os.getenv("LLM_BACKEND", "mock")
```
**Seen in:** `llm_client.py`, `generate()`; also `LOCAL_MODEL`, `ANTHROPIC_MODEL`

## Message format (role / content)  `(concept)`
**1. Definition:** Chat APIs take a list of dictionaries such as `{"role": "user", "content": "Hello"}`, where role is `user` or `assistant` (and `system` for instructions) and the list is the conversation so far.
**2. In plain words:** The model is stateless; you resend the whole conversation each time. This list shape is nearly universal across providers.
**3. Example to quote:** > "I pass `[{'role': 'user', 'content': 'What is 2+2?'}]` and get back a plain string."
**Seen in:** `llm_client.py`, docstring

## System prompt (separate argument)  `(concept)`
**1. Definition:** A system prompt is the instruction block that sets the model's role and rules; in this client it is a separate `system=` argument (Anthropic style) that the local backend inserts as a first `{"role": "system"}` message.
**2. In plain words:** Different libraries want it in different places, so the client hides that difference.
**3. Example to quote:** > "I give `system='You are a strict classifier.'` and the client places it correctly for whichever backend is active."
**Seen in:** `llm_client.py`, `_generate_local` and `_generate_anthropic`

## `from __future__ import annotations`  `(syntax)`
**1. Definition:** This import makes Python treat type annotations as strings evaluated lazily, enabling newer syntax like `list[str] | None` on older Python versions.
**2. In plain words:** It's purely about type hints; it doesn't change runtime behaviour of your logic.
**3. Example to quote:** > "I put `from __future__ import annotations` at the top so my modern type hints don't break on older Python."
**Seen in:** `llm_client.py`, line after the docstring

## Type hints (`list[str] | None`, Callable)  `(syntax)`
**1. Definition:** Type hints annotate expected types, e.g. `script: list[str] | None = None` ("a list of strings or None") and `Callable` ("something you can call like a function"); Python does not enforce them.
**2. In plain words:** They document the API for humans and tools. The `| None` marks an optional argument.
**3. Example to quote:** > "`rules: list[tuple[str, str | Callable]] | None` says each rule is a pair of a regex and either a reply string or a function."
**Seen in:** `llm_client.py`, `MockLLM.__init__`

## MockLLM  `(concept)`
**1. Definition:** `MockLLM` is a fake language model class whose replies come from a script or regex rules, so it is instant, offline and deterministic.
**2. In plain words:** It lets you test an agent loop or a prompt-evaluation harness with exact, repeatable "model" behaviour. If nothing matches it returns a polite fallback string.
**3. Example to quote:** > "To test my agent loop I script the mock's replies, so I decide the model's next move and can reproduce a bug exactly."
```python
demo = MockLLM(rules=[(r"hello", "Hi there!")])
```
**Seen in:** `llm_client.py`, `MockLLM`

## Scripted mode (script=[...])  `(concept)`
**1. Definition:** In scripted mode the mock returns the items of a list one by one, in order, regardless of the input.
**2. In plain words:** Ideal for agent demos: first reply is a tool call, second is the final answer.
**3. Example to quote:** > "`MockLLM(script=['{\"tool\": ...}', '{\"final\": ...}'])` plays a two-step agent run."
**Seen in:** `llm_client.py`, `MockLLM.generate`

## Rule mode (rules=[(regex, reply)])  `(concept)`
**1. Definition:** In rule mode the mock tests each `(regex, reply)` pair against the last user message and returns the reply of the first match; a reply can be a string or a function receiving `(messages, system)`.
**2. In plain words:** It's a tiny pattern-matching fake: "if the user mentions 2+2, answer 4".
**3. Example to quote:** > "My rules map `2\\s*\\+\\s*2` to `'4'`, so the mock answers arithmetic questions deterministically."
**Seen in:** `llm_client.py`

## re.search and re.IGNORECASE  `(standard library)`
**1. Definition:** `re.search(pattern, text, flags)` scans a string for the first match of a regular expression and returns a match object or `None`; `re.IGNORECASE` makes matching case-insensitive.
**2. In plain words:** It finds a pattern anywhere in the text, unlike `re.match` which only checks the start. Case-insensitivity means "Hello" and "hello" both match.
**3. Example to quote:** > "`re.search(r'hello', 'Say HELLO', flags=re.IGNORECASE)` matches even though the case differs."
**Seen in:** `llm_client.py`, `MockLLM.generate`

## list.pop(0)  `(Python built-in)`
**1. Definition:** `list.pop(0)` removes and returns the first item of a list.
**2. In plain words:** The script uses it to consume replies one at a time. It mutates the list, which is why the constructor copies it with `list(script)`.
**3. Example to quote:** > "Each call does `self.script.pop(0)`, so the third call would run out of replies unless I add more."
**Seen in:** `llm_client.py`, `MockLLM.generate`

## next() with a default and reversed()  `(Python built-in)`
**1. Definition:** `next(iterable, default)` returns the next item or the default if empty; `reversed(seq)` iterates a sequence backwards; combined with a generator expression they find the last matching item.
**2. In plain words:** `next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")` means "find the most recent user message, or empty text if none".
**3. Example to quote:** > "I look backwards through the conversation to find the last user message, and fall back to an empty string."
**Seen in:** `llm_client.py`, `MockLLM.generate`

## Recording calls (self.calls)  `(concept)`
**1. Definition:** The mock appends every request (messages and system prompt) to `self.calls`.
**2. In plain words:** After a test run you can inspect exactly what your code sent to the model, which makes debugging and assertions easy.
**3. Example to quote:** > "After the run I print `mock.calls` to check the agent resent the tool result in the second request."
**Seen in:** `llm_client.py`, `MockLLM.__init__`

## Lazy import  `(concept)`
**1. Definition:** A lazy import places `import x` inside a function so the library loads only when that code path is used.
**2. In plain words:** `transformers`, `torch` and `anthropic` are heavy or optional. Importing them lazily means the mock backend works even if they aren't installed and starts instantly.
**3. Example to quote:** > "I import `anthropic` inside `_generate_anthropic` so people without an API key or the package can still run the mock."
**Seen in:** `llm_client.py`, `_generate_local`, `_generate_anthropic`

## Module-level cache with `global`  `(syntax)`
**1. Definition:** A module-level variable such as `_LOCAL_PIPE = None` holds a value for the lifetime of the process, and the `global` statement lets a function assign to it.
**2. In plain words:** Loading a model takes seconds, so the first call builds the pipeline and stores it; later calls reuse it. The leading underscore signals "internal".
**3. Example to quote:** > "`_LOCAL_PIPE` caches the loaded model, so only the first local call pays the loading cost."
**Seen in:** `llm_client.py`, `_generate_local`

## transformers pipeline("text-generation")  `(third-party: transformers)`
**1. Definition:** `pipeline("text-generation", model=...)` builds a ready-to-use text-generation object that handles tokenizing, running the model and decoding in one call.
**2. In plain words:** It's the high-level shortcut compared with the hand-written loop in module 04. Here it loads `Qwen/Qwen2.5-0.5B-Instruct` by default (change with `LOCAL_MODEL`), about 1 GB on first use.
**3. Example to quote:** > "For the local backend I just call `pipeline('text-generation', model=model_id)` and pass the chat messages."
**Seen in:** `llm_client.py`, `_generate_local`

## do_sample and temperature  `(concept)`
**1. Definition:** `do_sample=False` means greedy decoding (always the most likely token); `do_sample=True` with a `temperature` samples randomly. The client sets `do_sample = temperature > 0` and passes `temperature=None` when not sampling.
**2. In plain words:** Temperature 0 gives repeatable answers, which is what you want for tests. Passing a temperature while sampling is off would make the library complain.
**3. Example to quote:** > "I treat temperature 0 as greedy by setting `do_sample = temperature > 0`, so my tests are deterministic."
**Seen in:** `llm_client.py`, `_generate_local`

## Chat template output indexing  `(concept)`
**1. Definition:** When you give the pipeline a list of chat messages, it returns the whole conversation including the new reply, so the answer is `out[0]["generated_text"][-1]["content"]`.
**2. In plain words:** `[-1]` takes the last message in the list, which is the model's new reply, and `["content"]` is its text.
**3. Example to quote:** > "The pipeline returns the full chat, so I take the last item, `[-1]`, as the assistant's reply."
**Seen in:** `llm_client.py`, `_generate_local`

## anthropic.Anthropic client and messages.create  `(third-party: anthropic)`
**1. Definition:** `anthropic.Anthropic()` creates an API client that reads `ANTHROPIC_API_KEY` from the environment; `client.messages.create(model, max_tokens, temperature, system, messages)` sends a conversation and returns the response.
**2. In plain words:** Never hard-code the API key in source files; keep it in the environment. The model name comes from `ANTHROPIC_MODEL` with a default in the file.
**3. Example to quote:** > "I create `anthropic.Anthropic()` with no key in code, since it reads `ANTHROPIC_API_KEY` from my environment."
**Seen in:** `llm_client.py`, `_generate_anthropic`

## Response content blocks  `(concept)`
**1. Definition:** An Anthropic response's `content` is a list of blocks, each with a `type`; text replies are blocks with `type == "text"` and a `.text` attribute.
**2. In plain words:** A reply can contain several blocks (text, tool calls), so the code joins only the text ones.
**3. Example to quote:** > "I join `block.text` for every block with `type == 'text'` to get the reply string."
```python
"".join(b.text for b in resp.content if b.type == "text")
```
**Seen in:** `llm_client.py`, `_generate_anthropic`

## max_tokens  `(concept)`
**1. Definition:** `max_tokens` caps how many tokens the model may generate in its reply.
**2. In plain words:** It limits cost and length; if set too low, the answer is cut off mid-sentence. It is the same idea as `max_new_tokens` in module 05.
**3. Example to quote:** > "I pass `max_tokens=300` so a reply can't run on forever, but a JSON answer could be truncated if I set it too low."
**Seen in:** `llm_client.py`, all backends

## generate() dispatcher  `(concept)`
**1. Definition:** The public `generate(messages, system, max_tokens, temperature, backend, mock)` function picks the backend by name and returns the reply text, raising `ValueError` for an unknown name.
**2. In plain words:** One function, three engines. Raising a clear error for a typo like `LLM_BACKEND=mok` is better than silently doing the wrong thing.
**3. Example to quote:** > "Calling `generate(messages, backend='mok')` raises `ValueError: Unknown LLM backend 'mok'`, which tells me exactly what's wrong."
**Seen in:** `llm_client.py`, `generate()`

## extract_json  `(concept)`
**1. Definition:** `extract_json(text)` finds the first `{...}` span in a model reply and parses it into a dictionary, raising `ValueError` if none exists.
**2. In plain words:** Models often wrap JSON in chatter like "Sure! Here you go: {...} Hope that helps." This function pulls out just the JSON. Callers should catch the error and retry.
**3. Example to quote:** > "`extract_json('Sure! {\"label\": \"positive\"} Hope that helps.')` returns `{'label': 'positive'}`."
**Seen in:** `llm_client.py`

## Greedy regex with re.DOTALL  `(standard library)`
**1. Definition:** The pattern `\{.*\}` matches from the first `{` to the last `}`; the `re.DOTALL` flag lets `.` also match newlines, so multi-line JSON works. "Greedy" means it takes the longest possible match.
**2. In plain words:** It handles JSON spread over several lines, but if the reply contains two separate JSON objects or stray braces, the match may be wrong, which is a limitation.
**3. Example to quote:** > "I use `re.DOTALL` so a pretty-printed multi-line JSON reply is still captured."
**Seen in:** `llm_client.py`, `extract_json`

## json.loads and JSONDecodeError  `(standard library)`
**1. Definition:** `json.loads(text)` parses a JSON string into Python objects and raises `json.JSONDecodeError`, a subclass of `ValueError`, if the text is invalid.
**2. In plain words:** Because `JSONDecodeError` is a `ValueError`, one `except ValueError` catches both "no JSON found" and "bad JSON".
**3. Example to quote:** > "A trailing comma makes `json.loads` raise `JSONDecodeError`; I catch `ValueError` and retry the model call."
**Seen in:** `llm_client.py`, `extract_json`

## `if __name__ == "__main__"` self-test  `(syntax)`
**1. Definition:** The block at the bottom runs only when the file is executed directly (`python shared/llm_client.py`), not when it is imported.
**2. In plain words:** It doubles as a quick smoke test: it should print `Hi there!`, `4` and `{'label': 'positive', 'score': 0.9}`.
**3. Example to quote:** > "I run `python shared/llm_client.py` as a sanity check; importing it from another module doesn't trigger it."
**Seen in:** `llm_client.py`, bottom
