"""
GOAL   : Read the JSONL trace and print a debugging summary (what ran, what failed, how long).
RUN    : python show_trace.py            (after python main.py)
This is how you debug an agent WITHOUT asking an AI: read the evidence, find the first failing event.
"""
import json
from collections import Counter
from pathlib import Path

TRACE = Path(__file__).resolve().parent / "traces" / "agent_trace.jsonl"

# one line = one JSON object; a list comprehension parses every non-empty line
events = [json.loads(line) for line in TRACE.read_text().splitlines() if line.strip()]
print(f"{len(events)} events, {len({e['run_id'] for e in events})} runs\n")

print("event counts:", dict(Counter(e["event"] for e in events)))

print("\nrun_id   status       steps  question")
questions = {e["run_id"]: e["question"] for e in events if e["event"] == "start"}
for e in events:
    if e["event"] == "end":
        print(f"{e['run_id']} {e['status']:12s} {e['steps']:<5d}  {questions[e['run_id']]}")

print("\nFailed tool calls (the first place to look when an agent misbehaves):")
for e in events:
    if e["event"] == "tool_call" and not e["ok"]:
        print(f"  [{e['run_id']}] step {e['step']}: {e['tool']}({e['args']}) -> {e['result']}")

# Replay one run in order: this is the full story of what the agent did.
first_run = events[0]["run_id"]
print(f"\nReplay of run {first_run}:")
for e in (x for x in events if x["run_id"] == first_run):    # generator expression
    detail = {k: v for k, v in e.items() if k not in ("ts", "run_id", "event")}
    print(f"  {e['event']:10s} {detail}")
