# Terms — Step 02: The agent loop

Folder: `08_agent_harness/02_agent_loop` · Files covered: `main.py`, `WALKTHROUGH.md` (helper `../fake_llm.py`)
Back to module: [08 Agent harness README](../README.md)

**Terms in this folder:** [Agent](#agent) · [Agent loop](#agent-loop) · [Tool](#tool) · [Tool registry](#tool-registry) · [JSON text protocol](#json-text-protocol) · [Observation](#observation) · [Stop condition](#stop-condition) · [max_steps (bounded loop)](#max_steps-bounded-loop) · [range](#range) · [extract_json](#extract_json) · [Dict membership test ("final" in decision)](#dict-membership-test-final-in-decision) · [Keyword argument unpacking (**args)](#keyword-argument-unpacking-args) · [Dict of lambdas (operator table)](#dict-of-lambdas-operator-table) · [Why never eval](#why-never-eval) · [Exceptions from tools (KeyError, ZeroDivisionError)](#exceptions-from-tools-keyerror-zerodivisionerror) · [System prompt (protocol description)](#system-prompt-protocol-description) · [Triple-quoted string](#triple-quoted-string) · [Type hints on tool functions](#type-hints-on-tool-functions) · [Fake brain (fake_llm.py, fake_reply)](#fake-brain-fake_llmpy-fake_reply) · [Regular expression with named groups and number pattern](#regular-expression-with-named-groups-and-number-pattern) · [json.dumps](#jsondumps) · [Two-step question ("12 + 8 then * 3")](#two-step-question-12--8-then--3)

---

## Agent
*Type: concept*

**1. Definition:** A program that repeatedly asks a model what to do next, executes the chosen action (usually a tool call) in ordinary code, and feeds the result back until a stop condition is met.
**2. In plain words:** The model never runs anything itself; it only writes text that asks for an action. Your code decides whether and how to run it, which is where validation and safety belong.
**3. Example to quote:**
> "For '12 + 8 then * 3' the model asked for the calculator twice, my code ran it each time, and the third reply was the final answer 60."

**Seen in:** `main.py`, `run_agent`.

## Agent loop
*Type: concept*

**1. Definition:** The cycle: send messages to the model, parse its decision, run the requested tool, append the observation, repeat.
**2. In plain words:** It is just a `for` loop with an if inside. The skill is in the details: bounding it, parsing safely and recording what happened.
**3. Example to quote:**
> "Each pass I call `generate`, parse the JSON, either return the final answer or call a tool and append `Observation:` to the messages."

**Seen in:** `main.py`, the `for step in range(...)` loop.

## Tool
*Type: concept*

**1. Definition:** An ordinary function the agent is allowed to call, taking simple arguments and returning a result (here a string).
**2. In plain words:** Tools are how a text-only model affects the world: do maths, read a file, look up data. Each tool should do one thing and be safe to call with any arguments you validate.
**3. Example to quote:**
> "`calculator(a, op, b)` is my only tool; it returns the result as a string so it can go straight into an Observation message."

**Seen in:** `main.py`, `def calculator`.

## Tool registry
*Type: concept*

**1. Definition:** A dictionary mapping the tool names the model may use to the Python functions that implement them (`TOOLS = {"calculator": calculator}`).
**2. In plain words:** The registry is both a lookup table and an allow-list: a name not in it cannot be executed. A name the model invents causes a `KeyError`, which step 03 handles with validation.
**3. Example to quote:**
> "`TOOLS[tool_name](**args)` looks up the function by name; if the model invents a tool, the lookup fails instead of running something unknown."

**Seen in:** `main.py`, `TOOLS`.

## JSON text protocol
*Type: concept*

**1. Definition:** An agreed text format the model must reply in, here `{"tool": ..., "args": {...}}` to request a tool or `{"final": ...}` to finish.
**2. In plain words:** You describe the format in the system prompt and parse the reply. It works with any model that can write JSON, without relying on a provider's special tool-calling feature (step 09 compares the two).
**3. Example to quote:**
> "The system prompt says reply with only one JSON object: either a tool call or a final answer."

**Seen in:** `main.py`, `SYSTEM_PROMPT`.

## Observation
*Type: concept*

**1. Definition:** The result of a tool call, sent back to the model as a message beginning with `Observation:` so it can decide the next step.
**2. In plain words:** It is how the model "sees" what happened. Without it the model would be guessing at the result.
**3. Example to quote:**
> "After the calculator returned 20, I appended `Observation: 20` as a user message, and the next reply asked to multiply by 3."

**Seen in:** `main.py`, `messages.append({"role": "user", "content": f"Observation: {result}"})`.

## Stop condition
*Type: concept*

**1. Definition:** A rule that ends the loop: either the model returns a `final` answer, or the step budget is exhausted.
**2. In plain words:** Without a stop condition a confused model can loop forever and burn money. Two exits are better than one.
**3. Example to quote:**
> "Stop condition one is `'final' in decision`; stop condition two is the loop running out of `max_steps`."

**Seen in:** `main.py`, comments `STOP CONDITION 1` and `2`.

## max_steps (bounded loop)
*Type: concept / parameter*

**1. Definition:** An upper limit on loop iterations, implemented as `for step in range(1, max_steps + 1)`, which guarantees the loop terminates.
**2. In plain words:** It is a safety net. With `max_steps=1` the question 6 x 7 returns `STOPPED: reached max_steps...` because the model needed two steps.
**3. Example to quote:**
> "I set `max_steps=5` by default; a runaway agent stops after five model calls instead of looping forever."

**Seen in:** `main.py`, `run_agent` signature and TRY IT comment.

## range
*Type: Python built-in*

**1. Definition:** `range(start, stop)` produces integers from `start` up to but not including `stop`.
**2. In plain words:** `range(1, max_steps + 1)` counts steps from 1 to `max_steps`, so the printed step numbers start at 1 like a human would count.
**3. Example to quote:**
> "`range(1, 4)` gives 1, 2, 3, so a limit of three steps means three iterations."

**Seen in:** `main.py`, `run_agent`.

## extract_json
*Type: project function*

**1. Definition:** `shared/llm_client.py::extract_json(text)` finds the first JSON object in the model's reply and parses it into a dict; raises `ValueError` if none is found.
**2. In plain words:** Models often wrap JSON in chatter. This pulls the object out. If it raises, the model broke the protocol and your loop needs to handle that (step 06).
**3. Example to quote:**
> "`decision = extract_json(reply)` turns the model's text into a dict I can test with `'final' in decision`."

**Seen in:** `main.py`, `run_agent`.

## Dict membership test (`"final" in decision`)
*Type: Python syntax*

**1. Definition:** `key in dict` returns True if the dictionary has that key.
**2. In plain words:** It is the quick way to see which kind of decision the model made: a final answer or a tool request.
**3. Example to quote:**
> "`if 'final' in decision:` returns the answer and ends the loop."

```python
"a" in {"a": 1}   # True
```

**Seen in:** `main.py`, `run_agent`.

## Keyword argument unpacking (`**args`)
*Type: Python syntax*

**1. Definition:** `func(**d)` calls `func` passing each key-value pair of the dict `d` as a keyword argument.
**2. In plain words:** The model's `args` dict `{"a": 12, "op": "+", "b": 8}` becomes `calculator(a=12, op="+", b=8)`. Danger: if the model includes a wrong or extra key you get a `TypeError`, so validate first (step 03).
**3. Example to quote:**
> "`TOOLS[tool_name](**args)` turns the dict from the model into named arguments."

```python
f(**{"a": 1, "b": 2})   # same as f(a=1, b=2)
```

**Seen in:** `main.py`, `result = TOOLS[tool_name](**args)`.

## Dict of lambdas (operator table)
*Type: Python syntax*

**1. Definition:** A dictionary mapping operator strings to `lambda` functions, e.g. `{"+": lambda x, y: x + y}`; calling `operations[op](a, b)` performs the operation.
**2. In plain words:** It replaces a long if/elif chain and, importantly, replaces `eval`. A `lambda` is a tiny anonymous function.
**3. Example to quote:**
> "`operations['+'](12, 8)` returns 20. An unknown operator raises `KeyError` instead of executing arbitrary text."

**Seen in:** `main.py`, `calculator`.

## Why never `eval`
*Type: concept (security)*

**1. Definition:** `eval(text)` executes a string as Python code, so using it on model output lets anyone who influences the model run arbitrary commands.
**2. In plain words:** A calculator built on `eval` could be tricked into deleting files. A dict of allowed operations can only do the four things you listed.
**3. Example to quote:**
> "I use a dict of operations instead of `eval` so the model can only add, subtract, multiply and divide."

**Seen in:** `main.py`, comment in `calculator`.

## Exceptions from tools (KeyError, ZeroDivisionError)
*Type: Python built-in exceptions*

**1. Definition:** `KeyError` is raised for a missing dict key (a bad operator or tool name) and `ZeroDivisionError` for dividing by zero.
**2. In plain words:** In this step they are not caught, so `5 / 0` kills the whole agent. Steps 03 and 06 add validation and error handling so one bad call does not end the run.
**3. Example to quote:**
> "Asking `5 / 0` crashes with `ZeroDivisionError`, which shows why tool calls must be wrapped in error handling."

**Seen in:** `main.py` BREAK IT comments.

## System prompt (protocol description)
*Type: concept*

**1. Definition:** The standing instruction that tells the model what reply format to use and what the tools are.
**2. In plain words:** The model only follows the protocol because you describe it. If you add a new operation to the code, you must also add it to this text.
**3. Example to quote:**
> "My system prompt lists the calculator's argument names and says results come back as messages starting with 'Observation:'."

**Seen in:** `main.py`, `SYSTEM_PROMPT`.

## Triple-quoted string
*Type: Python syntax*

**1. Definition:** A string literal written with `"""` that can span several lines.
**2. In plain words:** It is convenient for long instructions such as the system prompt.
**3. Example to quote:**
> "`SYSTEM_PROMPT` is a triple-quoted string describing the two reply formats."

**Seen in:** `main.py`, `SYSTEM_PROMPT`.

## Type hints on tool functions
*Type: Python syntax*

**1. Definition:** Annotations like `a: float, op: str, b: float -> str` documenting the expected types of a function.
**2. In plain words:** They show what arguments a tool expects, and step 03 reuses that idea to validate what the model sends.
**3. Example to quote:**
> "`def calculator(a: float, op: str, b: float) -> str` says two numbers and an operator go in and a string comes out."

**Seen in:** `main.py`, `calculator`, `run_agent`.

## Fake brain (`fake_llm.py`, `fake_reply`)
*Type: project helper*

**1. Definition:** A scripted stand-in model that uses regular expressions on the latest question to answer in the JSON protocol, returned as a `MockLLM` by `make_fake_agent_llm()`.
**2. In plain words:** It is not intelligent; it lets you study the loop deterministically. A real model makes mistakes this one never makes, which is why the harness must be defensive.
**3. Example to quote:**
> "`LLM = make_fake_agent_llm()` supplies the model; set `LLM_BACKEND` to switch to a real one and the loop stays the same."

**Seen in:** `main.py`, `LLM = make_fake_agent_llm()`; `../fake_llm.py`.

## Regular expression with named groups and number pattern
*Type: Python standard library*

**1. Definition:** The fake model builds patterns like `NUM = r"(-?\d+(?:\.\d+)?)"` (an int or decimal) and uses `(?P<name>...)` named groups, read as `m["name"]`.
**2. In plain words:** It finds numbers and operators inside the question, such as `12 + 8 then * 3`. `float.is_integer()` is then used so `5.0` is shown as `5`.
**3. Example to quote:**
> "The pattern captured `12`, `+`, `8` from the question and the fake model turned them into a calculator call."

**Seen in:** `../fake_llm.py`, `NUM`, `fake_reply`, `_num`.

## json.dumps
*Type: Python standard library*

**1. Definition:** `json.dumps(obj)` converts a Python object to a JSON string.
**2. In plain words:** The fake model uses it to write replies exactly in the protocol format: `{"tool": ..., "args": {...}}`.
**3. Example to quote:**
> "`json.dumps({'final': 'Result: 60'})` produces the string the loop then parses with `extract_json`."

**Seen in:** `../fake_llm.py`, `fake_reply`.

## Two-step question ("12 + 8 then * 3")
*Type: concept*

**1. Definition:** A request that needs two tool calls where the second uses the result of the first.
**2. In plain words:** This is why an agent is a loop rather than one call: the model must see the first Observation before it can compute the second step.
**3. Example to quote:**
> "The answer 60 took three model calls: add, multiply, then final."

**Seen in:** `main.py`, second question; `WALKTHROUGH.md` real output.
