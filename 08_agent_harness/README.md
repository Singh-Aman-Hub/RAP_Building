# 08 — Agent Harness Design Without Frameworks (Syllabus Topic 3)

## Why this matters
"Build an agent without LangChain" is a favourite interview task because it exposes whether you understand what frameworks hide: a **loop**, a **tool dispatcher**, a **message list**, and a lot of **defensive code**. If you can write this in ~100 lines, explain each part, and say where it breaks, you pass the topic.

## Where this fits
**Builds on**
- [00_python_foundations](../00_python_foundations/) — dicts of functions, `**kwargs`, exceptions, classes, file I/O and `pathlib`.
- [01_debugging_and_environment](../01_debugging_and_environment/) — reading tracebacks; the `broken_examples/` here are drills in that skill.
- [04_llm_fundamentals](../04_llm_fundamentals/) and [05_llm_parameters](../05_llm_parameters/) — tokens and the **context window** explain why history must be trimmed (step 04); `max_tokens` and `temperature` are the knobs on each call (step 01).
- [07_prompt_engineering](../07_prompt_engineering/) — the tool-call protocol is a **structured-output prompt** (07/04: JSON + validate + retry); the safety layer reuses **prompt-injection** thinking (07/06).

**Leads to**
- [09_api_development](../09_api_development/) — wrap `Agent.run()` in a FastAPI endpoint; the registry's argument checks become Pydantic request models.
- [10_ml_system_architecture](../10_ml_system_architecture/) — the JSONL trace is your logging/monitoring layer; memory becomes a database; retries/timeouts are reliability patterns.
- [11_docker](../11_docker/) — a container is the *real* sandbox that step 07 says a folder is not.
- [13_capstone_project](../13_capstone_project/) — the capstone service reuses this harness.

## Concept map
```
                       +----------------- context management (04) <----- token limits (module 04/05)
                       |                         ^
 user question --> messages list (01) --> LLM call (01) --retry/backoff (06)--> reply text
                       ^                                                          |
                       |                                                  parse JSON / repair (06, 07-prompting/04)
                       |                                                          |
          Observation: result <---- execute <---- validate (03) <---- {"tool","args"} or {"final"}  (02)
                       ^               |                |
                       |        permissions/sandbox/confirm (07)
                       |               |
                       |        tools: calculator, memory (05), files
                       |
                  every step --> JSONL trace (08) --> debugging, monitoring (module 10)
                  stop conditions: "final" | max_steps | repeated failure (02, 06)
```

## Study path
| # | Folder | You learn | Run (from the step folder) | Time |
|---|---|---|---|---|
| 01 | [01_llm_api_call](01_llm_api_call/) | API call shape, message list = memory | `python main.py` | 45 min |
| 02 | [02_agent_loop](02_agent_loop/) | the loop, tool call protocol, stop conditions | `python main.py` | 90 min |
| 03 | [03_tool_registry_and_validation](03_tool_registry_and_validation/) | registry, dispatch, validation, pydantic | `python main.py` | 90 min |
| 04 | [04_history_and_context](04_history_and_context/) | token estimate, truncate, summarise | `python main.py` | 60 min |
| 05 | [05_memory](05_memory/) | short- vs long-term memory | `python main.py` | 45 min |
| 06 | [06_errors_and_retries](06_errors_and_retries/) | backoff, JSON repair, errors as data | `python main.py` | 75 min |
| 07 | [07_safety_and_permissions](07_safety_and_permissions/) | least privilege, sandbox, confirmation | `python main.py` | 75 min |
| 08 | [08_trace_and_full_agent](08_trace_and_full_agent/) | JSONL trace + the combined `agent.py` | `python main.py && python show_trace.py` | 90 min |
| 09 | [09_native_tool_calling](09_native_tool_calling/) | native `tool_use` API (reference) | `python main.py` | 45 min |
| — | [broken_examples](broken_examples/) | 5 debugging drills | `python broken_*.py` | 60 min |

Each step has `main.py` + `WALKTHROUGH.md`. Exercises: [exercises.md](exercises.md). Solutions: [solutions/](solutions/).

### What runs by default
Everything uses [`fake_llm.py`](fake_llm.py), a **regex-driven stand-in for a model** (it answers `{"tool":...}`/`{"final":...}` for arithmetic, memory and file requests). It is deterministic and offline, so you study the **harness**. The harness code does not change for a real model: set `LLM_BACKEND=anthropic` (and `ANTHROPIC_API_KEY`) and the same loops talk to a real model — which will make mistakes the fake never makes (bad JSON, invented tools, wrong types). That is exactly why steps 03, 06 and 07 exist. `09_native_tool_calling/native_tool_use_reference.py` is **reference code, not executed** by default.

## Concepts, one by one

### LLM API integration
A chat API takes `messages` (+ optional `system`) and returns the next message. It is **stateless**: you resend history every time. Important parameters: `max_tokens`, `temperature`, `model`. *Failure modes:* timeouts, rate limits (429), invalid requests (400), auth (401). *Trade-off:* longer history = more cost and latency.

### Agent execution loop
`while not done: ask model -> parse action -> run tool -> append result`. The model never executes anything; your code does. *Limitation:* models can loop, hallucinate tools, or stop early — so you need limits and checks.

### Tool / function calling, registration and dispatch
A tool = Python function + name + description + parameter schema. A **registry** (dict) maps names to functions so dispatch is `REGISTRY[name]["fn"](**args)`. Generate the prompt's tool list *from* the registry so there is one source of truth. Two ways to ask for a call: our JSON text protocol (works everywhere) or the provider's native tool-calling (more reliable, provider-specific).

### Argument validation
Treat model output like user input: check tool name, required/unknown arguments, types (watch `bool` vs `int`), allowed values. Return failures as observations so the model can retry. Pydantic expresses the same rules declaratively (lax coercion by default; use strict mode if you want exact types).

### Context and history management; context-window limits and summarisation
History grows each step (tool outputs can be huge). Before each call estimate tokens; if over budget, **truncate** (keep the system prompt/first message + newest turns) or **summarise** older turns. Truncate large tool outputs too. *Trade-off:* truncation forgets; summarisation costs a call and can distort facts.

### Memory management
Short-term = current `messages`. Long-term = persisted store accessed by tools (file/DB/vector store). Cap size, avoid secrets, beware memory poisoning, and note exact-key lookup vs semantic search.

### Stop conditions and maximum steps
Stop when the model returns `final`, when `max_steps` is reached, on repeated identical tool calls, on unrecoverable errors or a time/cost budget. Always return a clear status (`final`, `max_steps`, `llm_unavailable`, ...).

### Error handling and retries
Retry transient errors (timeouts, 429, 5xx) with **exponential backoff + jitter** and a cap; do not retry permanent errors (400/401/404). Ask the model to repair invalid JSON a bounded number of times. Convert tool exceptions into observations.

### Tool safety and permissions
Least privilege (expose only needed tools), read-only default, sandboxed resolved paths, allow-listed file types, size caps, confirmation for destructive actions, no `eval`/`exec`/raw shell, logging. *A folder sandbox is not a security boundary against malicious code* — containers/VMs are (module 11).

### Tracing / logging
One JSON object per event (`start`, `llm_reply`, `tool_call`, `end`) with run id, step, args, result, ok, duration. Debug by reading the trace for the first failing event.

## Common mistakes & gotchas
- No `max_steps` -> runaway cost ([`broken_no_max_steps.py`](broken_examples/broken_no_max_steps.py)).
- Forgetting to append the assistant turn / observation ([`broken_stale_history.py`](broken_examples/broken_stale_history.py)).
- Calling tools with unvalidated arguments ([`broken_unvalidated_args.py`](broken_examples/broken_unvalidated_args.py)).
- `json.loads` directly on model text ([`broken_bad_json.py`](broken_examples/broken_bad_json.py)).
- `SANDBOX / user_path` with `..` or absolute paths ([`broken_path_escape.py`](broken_examples/broken_path_escape.py)).
- Using `eval` in a calculator tool.
- Retrying every error, including permanent ones.
- Testing only with a deterministic fake brain and assuming a real model behaves the same.

## Self-check
- [ ] I can write the agent loop from memory with a step limit.
- [ ] I can explain the JSON tool protocol and the native tool-calling alternative.
- [ ] I can register tools in a dict and dispatch with `**args`.
- [ ] I can validate tool arguments by hand and with pydantic.
- [ ] I can manage history size by truncation and by summarisation.
- [ ] I can explain short-term vs long-term memory and its risks.
- [ ] I can list stop conditions and implement retries with backoff and jitter.
- [ ] I can implement path sandboxing, read-only mode and a confirmation hook.
- [ ] I can write a JSONL trace and use it to find a failing step.
- [ ] I solved all 5 broken drills without AI.

## Interview questions
1. **What is an agent?** A loop where an LLM chooses actions (tool calls) that code executes, feeding results back until it answers or a limit is hit.
2. **How does the model "call" a tool?** It emits structured output (JSON text or a native `tool_use` block); my code validates and runs the function, then returns the result as a message.
3. **How do you stop infinite loops?** `max_steps`, repetition detection, time/cost budgets, clear terminal statuses.
4. **How do you handle a model inventing a tool or wrong args?** Validate against the registry; return an error observation listing available tools; cap repair attempts.
5. **How do you keep within the context window?** Estimate tokens each call; truncate (pin first message) or summarise; truncate tool outputs.
6. **Short-term vs long-term memory?** Message list vs persisted store accessed via tools; discuss size caps, secrets, poisoning, retrieval.
7. **What do you retry and how?** Transient errors only, exponential backoff + jitter, capped; never retry 4xx bugs.
8. **How do you make file tools safe?** Resolve path and require it inside the sandbox, allow-list suffixes, read-only default, size caps, confirmation, containers for real isolation.
9. **Why not `eval` in a calculator tool?** It executes arbitrary code supplied (indirectly) by the model/user.
10. **Text-protocol vs native function calling?** Portability/control vs reliability/ids/parallel calls; validation needed in both.
11. **How would you debug an agent that gives wrong answers?** Read the trace; find the first wrong event; reproduce minimally; fix that layer.
12. **How would you test an agent harness without a real model?** Scripted/fake brains with injected failures (timeouts, bad JSON) — tests the harness, not model quality.
13. **Prompt injection in an agent?** Tool outputs and files are untrusted input; least-privilege tools, confirmation, output validation, logs.
