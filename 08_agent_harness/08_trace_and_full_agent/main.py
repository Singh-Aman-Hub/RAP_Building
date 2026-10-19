"""
GOAL   : Run the full agent on several tasks and leave a JSONL trace you can inspect afterwards.
RUN    : cd 08_agent_harness/08_trace_and_full_agent && python main.py && python show_trace.py
EXPECTED OUTPUT: one result dict per task; traces/agent_trace.jsonl is written, then summarised.
NEEDS  : nothing (mock = fake_llm.py). Real model: LLM_BACKEND=anthropic python main.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                              # so `import agent` works from any cwd
from agent import Agent, MEMORY_FILE, SANDBOX
from fake_llm import make_fake_agent_llm

TRACE = HERE / "traces" / "agent_trace.jsonl"
TRACE.parent.mkdir(exist_ok=True)
TRACE.unlink(missing_ok=True)                              # fresh trace for a repeatable demo
MEMORY_FILE.unlink(missing_ok=True)
(SANDBOX / "out.txt").unlink(missing_ok=True)

llm = make_fake_agent_llm()
agent = Agent(mock=llm, trace_file=TRACE, mode="read_write")

TASKS = [
    "What is 6 x 7?",
    "Calculate 12 + 8 then * 3",                           # two tool calls in one run
    "Remember that my name is Priya",                      # writes long-term memory
    "What is my name?",                                    # reads it back
    "read file notes.txt",
    "read file ../../.env",                                # blocked by the sandbox
    "What is 5 / 0?",                                      # tool error becomes an observation
    "delete file notes.txt",                               # needs confirmation -> denied by default
    "Tell me a joke",                                      # model has no tool for this
]
for task in TASKS:
    result = agent.run(task)
    print(f"{result['status']:10s} steps={result['steps']}  {task!r:36s} -> {result['answer']}")

# max_steps demonstration: 1 step is not enough for a two-tool question
tiny = Agent(mock=llm, trace_file=TRACE, max_steps=1)
print("\nmax_steps=1:", tiny.run("Calculate 12 + 8 then * 3"))
print(f"\nTrace written to {TRACE.relative_to(HERE.parents[1])}  -> now run: python show_trace.py")

# TRY IT: change mode to "read_only" and re-run: which tasks change?   TRY IT: pass confirm=lambda n, a: True.
# TRY IT: open traces/agent_trace.jsonl in an editor - one JSON object per line.
# BREAK IT: Agent(tools=["calculator"]) then ask "read file notes.txt" -> 'unknown tool' (least privilege).
