"""
GOAL   : Call an LLM through one function and understand the MESSAGE LIST that every chat API uses;
         the conversation "memory" is just a Python list you resend each time.
RUN    : cd 08_agent_harness/01_llm_api_call && python main.py
         (repo root: .venv/bin/python 08_agent_harness/01_llm_api_call/main.py)
EXPECTED OUTPUT: three turns of chat; the request grows each turn (1 -> 3 -> 5 messages).
NEEDS  : nothing (mock). Real model: LLM_BACKEND=anthropic ANTHROPIC_API_KEY=... python main.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))     # <repo>/shared  -> llm_client
sys.path.insert(0, str(HERE.parent))                    # <repo>/08_agent_harness -> fake_llm
from llm_client import generate, MockLLM
from fake_llm import make_fake_agent_llm

import re

def recall_name(messages, system):
    """Mock 'brain': look back through ALL messages for 'my name is X' (this is what a real model does too)."""
    for m in messages:
        found = re.search(r"my name is (\w+)", m["content"], re.I)   # re.I = ignore case
        if found:
            return "You told me earlier: " + found.group(1)
    return "I don't know your name yet."

# A tiny conversational mock so this step reads like a chat (rules = regex -> reply).
chat_mock = MockLLM(rules=[
    (r"what.*my name", recall_name),         # checked first: a question
    (r"my name is (\w+)", "Nice to meet you!"),
])

history = []            # THE memory of the chat: a list of {"role", "content"} dicts

def chat(user_text: str) -> str:
    """Send one user message plus ALL previous messages; store both sides in `history`."""
    history.append({"role": "user", "content": user_text})
    reply = generate(history,                       # the full conversation so far
                     system="You are a friendly assistant.",   # rules, sent separately each call
                     max_tokens=200,                # hard cap on the reply length
                     temperature=0.0,               # 0 = most predictable
                     mock=chat_mock)
    history.append({"role": "assistant", "content": reply})
    print(f"  (request had {len(history) - 1} messages)")
    return reply

print("user: My name is Priya")
print("bot :", chat("My name is Priya"))
print("user: What is my name?")
print("bot :", chat("What is my name?"))

print("\nThe whole 'memory' is this list:")
for message in history:
    print("  ", message)

# KEY IDEA: the model itself remembers NOTHING between calls. If you forget to resend `history`,
# it has no idea who Priya is. Test that: ----------------------------------------------------
fresh = generate([{"role": "user", "content": "What is my name?"}], mock=chat_mock)
print("\nWithout history ->", fresh)

# Error handling basics: network/API calls fail. Never let one failure produce a bare traceback in prod.
try:
    generate(history, backend="not-a-backend")      # BREAK IT on purpose: unknown backend
except ValueError as err:
    print("Caught:", err)

# TRY IT: change temperature to 0.9 with LLM_BACKEND=anthropic - does the reply wording vary?
# TRY IT: print len(str(history)) after each turn - history (and cost!) only ever grows (see step 04).
# BREAK IT: remove `history.append({"role": "assistant", ...})` and ask the second question.
