"""Exercise: add a new tool (word_count) to the full agent WITHOUT editing agent.py - just extend REGISTRY.
RUN: python ex_add_tool_to_full_agent.py"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
agent_path = HERE.parent / "08_trace_and_full_agent" / "agent.py"
sys.path.insert(0, str(agent_path.parent))
spec = importlib.util.spec_from_file_location("full_agent", agent_path)
full_agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full_agent)

def word_count(text):
    return str(len(text.split()))

full_agent.REGISTRY["word_count"] = dict(fn=word_count, params={"text": str}, read_only=True,
                                         confirm=False, allowed={})
agent = full_agent.Agent()          # tools=None -> all registry entries, built AFTER we extended it
print(agent.system_prompt())
print(agent.execute("word_count", {"text": "one two three"}))
print(agent.execute("word_count", {"text": 3}))        # wrong type -> validation message
