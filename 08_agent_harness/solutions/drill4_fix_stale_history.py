"""Fix for broken_stale_history.py. ROOT CAUSE: the assistant turn and the Observation were never appended,
so every iteration sent the SAME messages and the model gave the SAME reply.
HOW FOUND: print(messages) at the top of the loop -> it never changed between steps.
RUN: python drill4_fix_stale_history.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "broken_examples"))
from _common import calculator, extract_json, generate, make_fake_agent_llm

llm = make_fake_agent_llm()
messages = [{"role": "user", "content": "What is 6 x 7?"}]
for step in range(1, 5):
    reply = generate(messages, mock=llm)
    decision = extract_json(reply)
    if "final" in decision:
        print("final answer:", decision["final"])
        break
    messages.append({"role": "assistant", "content": reply})                                   # FIX a
    messages.append({"role": "user", "content": f"Observation: {calculator(**decision['args'])}"})  # FIX b
