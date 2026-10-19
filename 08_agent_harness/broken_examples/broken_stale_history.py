"""
DRILL 4 -- DOES NOT CRASH, BUT IS WRONG (the hardest kind of bug).   Bug: tool results never added to history.
RUN: python broken_stale_history.py    Find why the agent calls the same tool every step until max_steps.

Hint: print `messages` at the top of every loop iteration and compare with what the model needs to see.
(An agent's only memory is the message list - module 08 step 01.)
"""
from _common import calculator, extract_json, generate, make_fake_agent_llm

llm = make_fake_agent_llm()
messages = [{"role": "user", "content": "What is 6 x 7?"}]
for step in range(1, 5):
    reply = generate(messages, mock=llm)
    decision = extract_json(reply)
    print(f"step {step}: {reply}")
    if "final" in decision:
        print("final answer:", decision["final"])
        break
    result = calculator(**decision["args"])
    # BUG: the assistant turn and the Observation are never appended to `messages`!
else:
    print("STOPPED: max steps reached and no final answer - why?")
