"""
GOAL   : Compare OUR JSON-text protocol with the NATIVE tool-calling format of hosted APIs, and show
         that the same registry/validation/dispatch code serves both.
RUN    : cd 08_agent_harness/09_native_tool_calling && python main.py
EXPECTED OUTPUT: the two request formats side by side, a simulated native 'tool_use' response being
         dispatched through the same validator, and the 'tool_result' message to send back.
NEEDS  : nothing (no network). The real SDK version is native_tool_use_reference.py (NOT executed here).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from native_schema import REGISTRY, validate_call, execute, to_native_schema    # reuses step 03 by path

# --- 1. How OUR protocol tells the model about tools: plain text in the system prompt -----------
print("TEXT PROTOCOL  (step 02/03): the tool list is words in the system prompt; the model answers with JSON text")
print('  system: Reply with ONLY {"tool": "calculator", "args": {...}} or {"final": ...}')
print('  model : {"tool": "calculator", "args": {"a": 6, "op": "*", "b": 7}}\n')

# --- 2. How a NATIVE API is told: a structured `tools` list with JSON Schema (see native_schema.py) -----
print("NATIVE FORMAT: the API receives `tools=[...]` with JSON Schema")
print(json.dumps(to_native_schema("calculator", REGISTRY["calculator"]), indent=2), "\n")

# --- 3. A native response is STRUCTURED: no JSON-in-text parsing needed ---------------------------------
simulated_native_response = {
    "stop_reason": "tool_use",                                   # the model stopped because it wants a tool
    "content": [
        {"type": "text", "text": "I'll multiply those."},
        {"type": "tool_use", "id": "toolu_01ABC", "name": "calculator", "input": {"a": 6, "op": "*", "b": 7}},
    ],
}
for block in simulated_native_response["content"]:
    if block["type"] == "tool_use":
        problems = validate_call(block["name"], block["input"])  # STILL validate: schema != guarantee
        result = execute(block["name"], block["input"])
        tool_result_message = {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": block["id"],  # must echo the id so the model can match call <-> result
             "content": result, "is_error": result.startswith("ERROR")},
        ]}
        print("validation problems:", problems)
        print("message to send back:", json.dumps(tool_result_message))

print("""
DIFFERENCES to remember
  text protocol : works with ANY model, you parse + repair JSON yourself, you design the protocol.
  native calling: model is trained for it (fewer format errors), structured blocks, ids for parallel calls,
                  provider-specific field names (Anthropic tool_use/tool_result vs OpenAI tool_calls).
  both          : the model only REQUESTS a call - YOUR code runs it, so validation, permissions, retries,
                  step limits and tracing (steps 03-08) are needed either way.
""")

# TRY IT: change simulated 'input' to {"a": "six", ...} and read the validator's complaint.
# TRY IT: open native_tool_use_reference.py and map each line to a step in 02-08.
