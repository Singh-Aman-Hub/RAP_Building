"""
native_tool_use_reference.py -- REFERENCE ONLY. Not executed by the study pack (needs an API key + network).

GOAL : the same agent loop as step 02, but using the provider's NATIVE tool-calling API (Anthropic SDK).
RUN  : ANTHROPIC_API_KEY=sk-... python native_tool_use_reference.py     (only if you want to try it)
Compile-check without running:  python -m py_compile native_tool_use_reference.py
"""
import os
import sys
from pathlib import Path

import anthropic   # 3rd-party SDK: pip install anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_schema import to_native_schema, REGISTRY, execute   # converter + our validated dispatch

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")

def run_native_agent(question: str, max_steps: int = 5) -> str:
    client = anthropic.Anthropic()                          # reads ANTHROPIC_API_KEY
    tools = [to_native_schema(name, spec) for name, spec in REGISTRY.items()]
    messages = [{"role": "user", "content": question}]

    for _ in range(max_steps):                              # same bounded loop as step 02
        response = client.messages.create(model=MODEL, max_tokens=500, tools=tools, messages=messages)

        if response.stop_reason != "tool_use":              # model produced a normal answer -> done
            return "".join(b.text for b in response.content if b.type == "text")

        # IMPORTANT: put the assistant's whole turn (text + tool_use blocks) back into the history
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":                    # block.name, block.input (dict), block.id
                result = execute(block.name, block.input)   # our validation + permissions + error handling
                tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                     "content": result, "is_error": result.startswith("ERROR")})
        messages.append({"role": "user", "content": tool_results})   # ALL results in ONE user message

    return "STOPPED: max_steps"

if __name__ == "__main__":
    print(run_native_agent("What is 6 x 7?"))
