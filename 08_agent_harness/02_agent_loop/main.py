"""
GOAL   : Build the smallest possible agent loop: ask the model -> if it asks for a tool, run it ->
         feed the result back -> repeat until it says "final" (or we hit max_steps).
RUN    : cd 08_agent_harness/02_agent_loop && python main.py
EXPECTED OUTPUT: for each question, the trace of model decisions, tool calls and a final answer.
NEEDS  : nothing (mock = fake_llm.py). Same code works with LLM_BACKEND=anthropic.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
sys.path.insert(0, str(HERE.parent))
from llm_client import generate, extract_json
from fake_llm import make_fake_agent_llm

LLM = make_fake_agent_llm()      # swap for a real backend by setting LLM_BACKEND (the mock is ignored then)

# ---------------------------------------------------------------------------
# 1. A TOOL = an ordinary Python function. It takes simple arguments and returns a string.
# ---------------------------------------------------------------------------
def calculator(a: float, op: str, b: float) -> str:
    # A dict of operator -> tiny function. We NEVER use eval(): eval would run arbitrary code.
    operations = {"+": lambda x, y: x + y, "-": lambda x, y: x - y,
                  "*": lambda x, y: x * y, "/": lambda x, y: x / y}
    return str(operations[op](a, b))     # may raise KeyError (bad op) or ZeroDivisionError

TOOLS = {"calculator": calculator}       # name the model uses -> the function to call (the "registry")

# ---------------------------------------------------------------------------
# 2. The PROTOCOL is explained to the model in the system prompt.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an assistant that can use tools. Reply with ONLY one JSON object:
  {"tool": "calculator", "args": {"a": <number>, "op": "+|-|*|/", "b": <number>}}   to use the calculator
  {"final": "<answer>"}                                                              when you are done
Tool results are sent back to you as messages starting with 'Observation:'."""

# ---------------------------------------------------------------------------
# 3. THE LOOP
# ---------------------------------------------------------------------------
def run_agent(question: str, max_steps: int = 5) -> str:
    messages = [{"role": "user", "content": question}]
    for step in range(1, max_steps + 1):                       # bounded loop = guaranteed to stop
        reply = generate(messages, system=SYSTEM_PROMPT, mock=LLM)
        print(f"  step {step}: model said {reply}")
        decision = extract_json(reply)                         # text -> dict
        messages.append({"role": "assistant", "content": reply})   # keep the model's own turn in history

        if "final" in decision:                                # STOP CONDITION 1: model is done
            return decision["final"]

        tool_name, args = decision["tool"], decision["args"]
        result = TOOLS[tool_name](**args)                      # **args: unpack dict into keyword arguments
        print(f"          tool {tool_name}({args}) -> {result}")
        messages.append({"role": "user", "content": f"Observation: {result}"})   # feed result back
    return "STOPPED: reached max_steps without a final answer"  # STOP CONDITION 2: step budget used up

for question in ["What is 6 x 7?", "Calculate 12 + 8 then * 3", "Tell me a joke"]:
    print(f"\nQ: {question}")
    print("A:", run_agent(question))

# TRY IT: run_agent("What is 6 x 7?", max_steps=1) -> hits the step limit before the final answer.
# TRY IT: add a "power" operation to `operations` and ask "What is 2 ** 3"? (also update the system prompt!)
# BREAK IT: ask "What is 5 / 0?"  -> uncaught ZeroDivisionError kills the whole agent. Step 03/06 fix this.
# BREAK IT: ask "What is 5 ^ 2?" -> the fake model never produces '^', but a REAL model might invent
#           a bad op or a tool name that is not in TOOLS -> KeyError. Step 03 adds validation.
