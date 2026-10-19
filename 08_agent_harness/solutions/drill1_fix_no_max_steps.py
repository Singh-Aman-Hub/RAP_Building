"""Fix for broken_no_max_steps.py: bounded loop + repetition detection + clear status.
RUN: python drill1_fix_no_max_steps.py   (from this folder)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "broken_examples"))
from _common import calculator, extract_json, MockLLM, generate

always_same_tool = MockLLM(rules=[(r"", '{"tool": "calculator", "args": {"a": 1, "op": "+", "b": 1}}')])

def run(question, max_steps=8, max_repeats=3):
    messages, seen = [{"role": "user", "content": question}], []
    for step in range(1, max_steps + 1):                       # FIX 1: bounded loop
        decision = extract_json(generate(messages, mock=always_same_tool))
        if "final" in decision:
            return "final", decision["final"]
        call = (decision["tool"], tuple(sorted(decision["args"].items())))   # hashable fingerprint of the call
        seen.append(call)
        if seen.count(call) >= max_repeats:                    # FIX 2: repetition detection
            return "stopped_repeating", f"same call {call[0]} repeated {max_repeats} times at step {step}"
        messages.append({"role": "user", "content": f"Observation: {calculator(**decision['args'])}"})
    return "max_steps", "step limit reached"                   # FIX 3: explicit terminal status

print(run("What is 1 + 1?"))
