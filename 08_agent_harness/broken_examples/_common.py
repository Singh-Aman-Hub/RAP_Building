"""Shared helpers for the broken_* drills (so each drill stays short and focused on ONE bug)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
sys.path.insert(0, str(HERE.parent))
from llm_client import generate, extract_json, MockLLM   # noqa: F401
from fake_llm import make_fake_agent_llm, fake_reply      # noqa: F401


def calculator(a, op, b):
    operations = {"+": lambda x, y: x + y, "-": lambda x, y: x - y, "*": lambda x, y: x * y, "/": lambda x, y: x / y}
    return str(operations[op](a, b))
