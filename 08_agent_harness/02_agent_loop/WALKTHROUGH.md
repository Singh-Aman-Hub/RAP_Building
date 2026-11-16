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

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Agent loop | ReAct loop, tool loop, run loop, orchestration loop | Same idea | Repeat: model decides, harness runs a tool, result goes back; stop on final answer. |
| Tool | function, action, skill (loosely), function call | Mostly same | An ordinary Python function the model may ask us to run. |
| Observation | tool result, tool output, `tool_result` | Same | The text we feed back to the model after running a tool. |
| Registry | tool table, tool map, dispatch dict | Same | A dict from tool name to function. |
| Harness | scaffold, runtime, agent framework | Near-same | The non-model code (loop, tools, safety) around the model. |

## Code, step by step

### 1. Imports and path setup  (`main.py` lines 8–16)

```python
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
sys.path.insert(0, str(HERE.parent))
from llm_client import generate, extract_json
from fake_llm import make_fake_agent_llm
```

- **In simple words:** Load the JSON helper and path tools, then make the shared helper modules importable.
- **Exactly:** `generate` calls an LLM backend; `extract_json` pulls a JSON object out of reply text; `make_fake_agent_llm` returns a scripted `MockLLM`.
- **Why it is here:** The loop needs these helpers; without `sys.path` edits imports fail.

### 2. Create the model (mock)  (`main.py` lines 18–18)

```python
LLM = make_fake_agent_llm()      # swap for a real backend by setting LLM_BACKEND (the mock is ignored then)
```

- **In simple words:** Choose the pretend brain.
- **Exactly:** `LLM` is a `MockLLM` with one catch-all rule calling `fake_reply`. With `LLM_BACKEND=anthropic` the mock is ignored.
- **Why it is here:** Makes the run deterministic and offline.

### 3. The calculator tool  (`main.py` lines 20–27)

```python
# ---------------------------------------------------------------------------
# 1. A TOOL = an ordinary Python function. It takes simple arguments and returns a string.
# ---------------------------------------------------------------------------
def calculator(a: float, op: str, b: float) -> str:
    # A dict of operator -> tiny function. We NEVER use eval(): eval would run arbitrary code.
    operations = {"+": lambda x, y: x + y, "-": lambda x, y: x - y,
                  "*": lambda x, y: x * y, "/": lambda x, y: x / y}
    return str(operations[op](a, b))     # may raise KeyError (bad op) or ZeroDivisionError
```

- **In simple words:** A normal function that does one arithmetic operation.
- **Exactly:** `calculator(a: float, op: str, b: float) -> str`. `operations` maps `"+","-","*","/"` to lambdas; `operations[op](a, b)` is converted with `str`. Real run: `calculator(6,'*',7)` -> `'42'`. Bad `op` raises `KeyError`, `b=0` with `/` raises `ZeroDivisionError`.
- **Why it is here:** `eval()` would let a model-written string run arbitrary code; the dict only allows four fixed operations. It returns a string because observations are text.

### 4. Tool registry  (`main.py` lines 29–29)

```python
TOOLS = {"calculator": calculator}       # name the model uses -> the function to call (the "registry")
```

- **In simple words:** A phone book of tools: name to function.
- **Exactly:** `TOOLS = {"calculator": calculator}`; `TOOLS[name](**args)` is how the loop dispatches.
- **Why it is here:** Lets the loop run any tool by name without if/else chains.

### 5. The protocol prompt  (`main.py` lines 31–37)

```python
# ---------------------------------------------------------------------------
# 2. The PROTOCOL is explained to the model in the system prompt.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an assistant that can use tools. Reply with ONLY one JSON object:
  {"tool": "calculator", "args": {"a": <number>, "op": "+|-|*|/", "b": <number>}}   to use the calculator
  {"final": "<answer>"}                                                              when you are done
Tool results are sent back to you as messages starting with 'Observation:'."""
```

- **In simple words:** Tell the model exactly what format to answer in.
- **Exactly:** A triple-quoted `str`: only one JSON object per reply, either `{"tool":..., "args":...}` or `{"final":...}`, and results come back as messages starting with `Observation:`.
- **Why it is here:** The harness parses text; if the model does not follow the format, parsing fails.

### 6. Comment diagram of the six moves  (`main.py` lines 39–45)

```python
# ---------------------------------------------------------------------------
# 3. THE LOOP  (the whole idea of an agent, in six moves)
#    Prompt -> LLM call -> parse tool call -> execute tool -> return observation -> iterate until final answer
#      (1) build the prompt (system protocol + messages so far)      (4) execute the tool in OUR code
#      (2) call the LLM                                              (5) append "Observation: ..." to messages
#      (3) parse the reply into a tool call or a final answer        (6) go back to (1), or stop on "final"/max_steps
# ---------------------------------------------------------------------------
```

- **In simple words:** A map of the loop: prompt, call, parse, execute, observe, repeat.
- **Exactly:** Comments only.
- **Why it is here:** Reminds you what each line of `run_agent` is for.

### 7. Start the agent with the question  (`main.py` lines 46–47)

```python
def run_agent(question: str, max_steps: int = 5) -> str:
    messages = [{"role": "user", "content": question}]
```

- **In simple words:** The conversation starts with just the user's question.
- **Exactly:** `messages = [{"role":"user","content":question}]`; `max_steps` defaults to 5.
- **Why it is here:** A fresh list per call avoids leaking old questions between runs.

### 8. Loop, call the model, parse, record  (`main.py` lines 48–52)

```python
    for step in range(1, max_steps + 1):                       # bounded loop = guaranteed to stop
        reply = generate(messages, system=SYSTEM_PROMPT, mock=LLM)   # (1)+(2) prompt -> LLM call
        print(f"  step {step}: model said {reply}")
        decision = extract_json(reply)                         # (3) parse: text -> dict
        messages.append({"role": "assistant", "content": reply})   # keep the model's own turn in history
```

- **In simple words:** Ask the model what to do next and note down what it said.
- **Exactly:** `range(1, max_steps+1)` gives at most 5 iterations. `generate(...)` returns a `str` such as `{"tool": "calculator", "args": {"a": 6, "op": "*", "b": 7}}`. `extract_json` turns it into a `dict`. The assistant turn is appended to history.
- **Why it is here:** The bounded `for` guarantees termination; keeping the assistant turn lets the model see its own earlier decision.

### 9. Stop when the model says final  (`main.py` lines 54–55)

```python
        if "final" in decision:                                # STOP CONDITION 1: model is done
            return decision["final"]
```

- **In simple words:** If the model gives an answer, hand it back.
- **Exactly:** Step 2 of "6 x 7" returns `{"final": "Result: 42"}` so the function returns `"Result: 42"`.
- **Why it is here:** Without a stop condition the loop would only end at max_steps.

### 10. Run the tool and feed the result back  (`main.py` lines 57–60)

```python
        tool_name, args = decision["tool"], decision["args"]
        result = TOOLS[tool_name](**args)                      # (4) execute. **args: unpack dict into keyword arguments
        print(f"          tool {tool_name}({args}) -> {result}")
        messages.append({"role": "user", "content": f"Observation: {result}"})   # (5) observation back; (6) loop
```

- **In simple words:** Do what the model asked, then tell it the answer as a new message.
- **Exactly:** `decision["tool"]`, `decision["args"]`; `TOOLS[tool_name](**args)` unpacks the dict into keywords. Real output: `tool calculator({'a': 6, 'op': '*', 'b': 7}) -> 42`. The observation goes in as a `user` message `Observation: 42`.
- **Why it is here:** This is the feedback that makes it an agent rather than a one-shot call. Note: no validation yet, so a bad tool name raises `KeyError` (step 03).

### 11. Give up after max_steps  (`main.py` lines 61–61)

```python
    return "STOPPED: reached max_steps without a final answer"  # STOP CONDITION 2: step budget used up
```

- **In simple words:** If the model never finishes, return a failure string.
- **Exactly:** Runs only when the loop ends without `return`; e.g. `max_steps=1` on "6 x 7" gives this message.
- **Why it is here:** Prevents infinite loops and runaway cost.

### 12. Run three questions  (`main.py` lines 63–65)

```python
for question in ["What is 6 x 7?", "Calculate 12 + 8 then * 3", "Tell me a joke"]:
    print(f"\nQ: {question}")
    print("A:", run_agent(question))
```

- **In simple words:** Try an easy one, a two-step one, and one the model cannot handle.
- **Exactly:** Real results: `Result: 42` (2 steps), `Result: 60` (3 steps: 12+8=20, then 20*3=60), and `I don't know how to help with that.` (1 step, no tool).
- **Why it is here:** Shows the three shapes a run can take: one tool, chained tools, no tool.

### 13. Experiment comments  (`main.py` lines 67–71)

```python
# TRY IT: run_agent("What is 6 x 7?", max_steps=1) -> hits the step limit before the final answer.
# TRY IT: add a "power" operation to `operations` and ask "What is 2 ** 3"? (also update the system prompt!)
# BREAK IT: ask "What is 5 / 0?"  -> uncaught ZeroDivisionError kills the whole agent. Step 03/06 fix this.
# BREAK IT: ask "What is 5 ^ 2?" -> the fake model never produces '^', but a REAL model might invent
#           a bad op or a tool name that is not in TOOLS -> KeyError. Step 03 adds validation.
```

- **In simple words:** Things to try and break.
- **Exactly:** Comments only; `5 / 0` raises an uncaught `ZeroDivisionError`.
- **Why it is here:** Motivates validation and error handling in steps 03 and 06.

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
