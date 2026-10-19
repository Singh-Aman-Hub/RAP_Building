# Step 02 — The agent loop

## The story so far
In [step 01](../01_llm_api_call/) you can call a model and keep the conversation in a list, but a single call cannot use a calculator or read a file.

## The pain point
The model can write "20 * 3" but it cannot compute reliably, and it cannot touch the world. Doing it once by hand does not scale to tasks that need several actions in a row.

## The spark
Wrap the model in a loop: Prompt -> LLM call -> parse tool call -> execute tool -> return observation -> iterate until final answer. The model only *chooses* an action as JSON text; your code runs it and feeds the result back. A step limit guarantees the loop stops.

## Goal
Build the smallest agent: model decides -> harness runs tool -> result goes back -> repeat until `final` or `max_steps`.

## Run it
```bash
cd 08_agent_harness/02_agent_loop && ../../.venv/bin/python main.py
```
Real output:
```
Q: Calculate 12 + 8 then * 3
  step 1: model said {"tool": "calculator", "args": {"a": 12, "op": "+", "b": 8}}
          tool calculator({'a': 12, 'op': '+', 'b': 8}) -> 20
  step 2: model said {"tool": "calculator", "args": {"a": 20, "op": "*", "b": 3}}
          tool calculator({'a': 20, 'op': '*', 'b': 3}) -> 60
  step 3: model said {"final": "Result: 60"}
A: Result: 60

Q: Tell me a joke
  step 1: model said {"final": "I don't know how to help with that."}
A: I don't know how to help with that.
```
The first question, `What is 6 x 7?`, takes two steps (`calculator(6, '*', 7)` -> 42, then `final`); it was left out of the excerpt above.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `for step in range(1, max_steps + 1):` | bounded loop; guarantees termination | `range(1, 4)` -> 1, 2, 3 |
| `TOOLS[tool_name](**args)` | look up a function in a dict, call it, unpack a dict into keyword arguments | `f(**{"a": 1, "b": 2})` == `f(a=1, b=2)` |
| `"final" in decision` | does this dict contain this key? | `"a" in {"a": 1}` |
| `{"+": lambda x, y: x + y}` | dict of operator -> tiny anonymous function | `ops["+"](1, 2)` -> 3 |
| `return` inside a loop | exits the function immediately (stop condition 1) | – |
| triple-quoted string | multi-line text for the system prompt | – |

## Libraries & functions
| Name | Kind | Does | Returns |
|---|---|---|---|
| `extract_json` | shared | find and parse the first JSON object in text | dict |
| `json` | standard library | (used inside `extract_json`) | – |
| `calculator(a, op, b)` | our tool | arithmetic via a dict, **no `eval`** | str |
| `fake_llm.make_fake_agent_llm` | our helper | scripted brain for offline study | `MockLLM` |

## Execution trace
The explicit loop (also written as a comment above `run_agent` in `main.py`):
```
Prompt -> LLM call -> parse tool call -> execute tool -> return observation -> iterate until final answer
```
`messages = [user question]` -> loop: (1) `generate` returns text; (2) `extract_json` -> dict; (3) append assistant text; (4) `final`? return; else (5) call tool, append `Observation: <result>` as a user message; back to (1). Two stop conditions: model says `final`, or the loop runs out of steps.

## Experiments
- `max_steps=1` on `6 x 7` -> `STOPPED: reached max_steps...` (step budget protects you from runaway loops).
- Ask `5 / 0` -> uncaught `ZeroDivisionError` kills the agent (fixed in steps 03 and 06).

## Debugging practice
Run [`broken_examples/broken_no_max_steps.py`](../broken_examples/broken_no_max_steps.py) and [`broken_stale_history.py`](../broken_examples/broken_stale_history.py) — see exercises.

## Rebuild checklist
- [ ] A tool function and a `TOOLS` dict. [ ] A system prompt stating the protocol. [ ] A bounded loop with append-observe-repeat and a final return.

## The new problem (cliffhanger)
The loop trusts everything: it calls whatever tool name and arguments the model invents. A bad operator raises `KeyError` and `5 / 0` kills the agent. We need to describe tools precisely and check every call. Next: [03_tool_registry_and_validation](../03_tool_registry_and_validation/).

> **Say it aloud (interview box):** An agent is a loop around a model: prompt, LLM call, parse the tool call, execute the tool, append the observation, repeat until a final answer. The model only chooses actions as structured text, and my code executes them, which is where validation and safety belong. I always bound the loop with a maximum number of steps.
