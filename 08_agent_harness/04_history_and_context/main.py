"""
GOAL   : Keep the conversation inside the model's CONTEXT WINDOW: estimate tokens, trim old turns,
         and summarise what is dropped.
RUN    : cd 08_agent_harness/04_history_and_context && python main.py
EXPECTED OUTPUT: token counts before/after two strategies (truncate, summarise) on a 40-message chat.
NEEDS  : nothing (token estimate = characters / 4; a real tokenizer is in module 04_llm_fundamentals).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
from llm_client import generate, MockLLM

def estimate_tokens(messages) -> int:
    """Rough rule of thumb for English: 1 token ~ 4 characters. Good enough to decide WHEN to trim."""
    return sum(len(m["content"]) for m in messages) // 4

# ---- a long fake conversation ------------------------------------------------------------
history = [{"role": "user", "content": "Hi! I am planning a trip to Goa in December. Budget is 30000 rupees."}]
for i in range(1, 20):
    history.append({"role": "assistant", "content": f"Tip {i}: " + "book early and compare prices. " * 6})
    history.append({"role": "user", "content": f"Thanks, and question {i}: what about local transport option {i}?"})
print(f"original: {len(history)} messages, ~{estimate_tokens(history)} tokens")

BUDGET = 300   # pretend the context window only allows this many tokens for history (real ones: thousands)

# ---------------------------------------------------------------------------
# Strategy 1: TRUNCATE - always keep the FIRST message (it often holds the goal) + the newest messages.
# ---------------------------------------------------------------------------
def truncate(messages, budget):
    first, rest = messages[0], messages[1:]
    kept = []
    # walk backwards (newest first) and stop when the budget is full
    for message in reversed(rest):
        if estimate_tokens([first] + [message] + kept) > budget:
            break
        kept.insert(0, message)                  # insert at the FRONT to restore chronological order
    # Many APIs expect the history to start with a user turn: drop a leading assistant message if we cut mid-pair
    while kept and kept[0]["role"] == "assistant":
        kept.pop(0)
    return [first] + kept

short = truncate(history, BUDGET)
print(f"truncated: {len(short)} messages, ~{estimate_tokens(short)} tokens "
      f"(lost the middle {len(history) - len(short)} messages)")

# ---------------------------------------------------------------------------
# Strategy 2: SUMMARISE - ask the model to compress the old part, keep recent turns verbatim.
# ---------------------------------------------------------------------------
summary_mock = MockLLM(rules=[(r"summar", "User plans a Goa trip in December with a 30000 rupee budget; "
                                           "we gave booking tips and discussed local transport options.")])

def summarise_and_trim(messages, keep_last=4):
    old, recent = messages[:-keep_last], messages[-keep_last:]
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in old)
    summary = generate([{"role": "user", "content": "Summarise this conversation in 2 sentences, "
                                                    "keeping facts and decisions:\n" + transcript}],
                       mock=summary_mock)
    # the summary becomes ONE message at the start; recent turns follow unchanged
    return [{"role": "user", "content": f"[Summary of earlier conversation] {summary}"}] + recent

compact = summarise_and_trim(history)
print(f"summarised: {len(compact)} messages, ~{estimate_tokens(compact)} tokens")
print("\nsummary message:", compact[0]["content"])

# ---------------------------------------------------------------------------
# When to apply? Check BEFORE every model call.
# ---------------------------------------------------------------------------
def prepare_for_call(messages, budget=BUDGET):
    if estimate_tokens(messages) <= budget:
        return messages                          # fits: send as-is
    return summarise_and_trim(messages)          # too big: compress

print("\nprepared for the next call: ~", estimate_tokens(prepare_for_call(history)), "tokens")

# Trade-offs: truncation is free but forgets (the Goa budget survives only because we pinned message 0);
# summarisation costs an extra model call and can lose or distort details; both change what the model "knows".
# Tool results can be HUGE (a whole file) - truncate observations too (e.g. first 2000 chars + '...[truncated]').
# TRY IT: set BUDGET = 150 and 1000; watch how many messages survive.
# TRY IT: pin the first message in truncate() ... then remove that pin; what does the model forget?
# BREAK IT: remove the `while kept ... == "assistant"` loop; some real APIs reject a history whose first turn is the assistant.
