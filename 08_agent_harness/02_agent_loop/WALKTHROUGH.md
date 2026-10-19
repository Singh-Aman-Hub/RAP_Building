# Step 02 — The agent loop

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
(The first question, `6 x 7`, produces 42 in two steps.)

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
`messages = [user question]` -> loop: (1) `generate` returns text; (2) `extract_json` -> dict; (3) append assistant text; (4) `final`? return; else (5) call tool, append `Observation: <result>` as a user message; back to (1). Two stop conditions: model says `final`, or the loop runs out of steps.

## Experiments
- `max_steps=1` on `6 x 7` -> `STOPPED: reached max_steps...` (step budget protects you from runaway loops).
- Ask `5 / 0` -> uncaught `ZeroDivisionError` kills the agent (fixed in steps 03 and 06).

## Debugging practice
Run [`broken_examples/broken_no_max_steps.py`](../broken_examples/broken_no_max_steps.py) and [`broken_stale_history.py`](../broken_examples/broken_stale_history.py) — see exercises.

## Rebuild checklist
- [ ] A tool function and a `TOOLS` dict. [ ] A system prompt stating the protocol. [ ] A bounded loop with append-observe-repeat and a final return.

## Say it aloud
"An agent is a loop around a model: the model chooses an action as structured text, my code executes it, and the result is appended to the history for the next turn. I always bound the loop with a maximum number of steps and a clear stop condition. The model never executes anything itself — my code does, which is where validation and safety belong."
