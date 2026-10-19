"""
fake_llm.py -- a tiny SCRIPTED stand-in for an LLM, used by every step in module 08.

It is NOT intelligent. It looks at the last user question with regular expressions and answers
in the JSON text protocol our agent harness expects:

    {"tool": "<name>", "args": {...}}      <- "please run this tool and tell me the result"
    {"final": "<answer for the user>"}     <- "I am done"

Observations (tool results) come back to it as a user message starting with "Observation:".

Why fake it? So you can study the HARNESS (loop, validation, memory, retries, safety, tracing)
deterministically and offline.  The harness code is identical for a real model: switch with
LLM_BACKEND=anthropic (or local) and the same loop runs - only the "brain" changes, and a real brain
will make mistakes this fake one never makes (that is exactly why the harness must be defensive).
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from llm_client import MockLLM  # noqa: E402

NUM = r"(-?\d+(?:\.\d+)?)"           # regex for an int or decimal number


def _num(text):
    """'5' -> 5, '2.5' -> 2.5  (ints stay ints so output looks natural)."""
    value = float(text)
    return int(value) if value.is_integer() else value


def _question_and_obs_count(messages):
    """Return (latest user question, how many Observation messages followed it)."""
    n_obs = 0
    for m in reversed(messages):
        if m["role"] != "user":
            continue                                   # skip assistant turns
        if m["content"].startswith("Observation:"):
            n_obs += 1
        else:
            return m["content"], n_obs
    return "", n_obs


def fake_reply(messages, system=None):
    question, n_obs = _question_and_obs_count(messages)
    last = messages[-1]["content"]

    # ---- the previous turn was a tool result: decide what to do next ----------------------
    if last.startswith("Observation:"):
        obs = last[len("Observation:"):].strip()
        two_step = re.search(NUM + r"\s*([+*/-])\s*" + NUM + r"\s+then\s+([+*/-])\s*" + NUM, question)
        if obs.startswith("ERROR"):
            return json.dumps({"final": f"Sorry, the tool failed: {obs}"})
        if two_step and n_obs == 1:                    # first result in hand -> run the second operation
            return json.dumps({"tool": "calculator",
                               "args": {"a": _num(obs), "op": two_step.group(4), "b": _num(two_step.group(5))}})
        return json.dumps({"final": f"Result: {obs}"})

    # ---- a fresh question: route to a tool ------------------------------------------------
    m = re.search(NUM + r"\s*([+*/-])\s*" + NUM + r"\s+then\s+", question)
    if m is None:
        m = re.search(NUM + r"\s*([+*/x-])\s*" + NUM, question)
    if m:
        op = "*" if m.group(2) == "x" else m.group(2)
        return json.dumps({"tool": "calculator", "args": {"a": _num(m.group(1)), "op": op, "b": _num(m.group(3))}})

    m = re.search(r"remember (?:that )?(?P<k>[\w ]+?) is (?P<v>.+)", question, re.I)
    if m:
        return json.dumps({"tool": "remember", "args": {"key": re.sub(r"^my ", "", m["k"].strip().lower()), "value": m["v"].strip(" .")}})
    m = re.search(r"what(?:'s| is) my (?P<k>[\w ]+?)\??$", question.strip(), re.I)
    if m:
        return json.dumps({"tool": "recall", "args": {"key": m["k"].strip().lower()}})
    m = re.search(r"read (?:the )?file (?P<p>\S+)", question, re.I)
    if m:
        return json.dumps({"tool": "read_file", "args": {"path": m["p"]}})
    m = re.search(r"write '(?P<c>.*)' to (?P<p>\S+)", question, re.I)
    if m:
        return json.dumps({"tool": "write_file", "args": {"path": m["p"], "content": m["c"]}})
    m = re.search(r"delete (?:the )?file (?P<p>\S+)", question, re.I)
    if m:
        return json.dumps({"tool": "delete_file", "args": {"path": m["p"]}})
    return json.dumps({"final": "I don't know how to help with that."})


def make_fake_agent_llm():
    """A MockLLM whose one catch-all rule runs fake_reply. Use: generate(msgs, system=..., mock=make_fake_agent_llm())."""
    return MockLLM(rules=[(r"", fake_reply)])


if __name__ == "__main__":
    from llm_client import generate
    llm = make_fake_agent_llm()
    print(generate([{"role": "user", "content": "What is 6 x 7?"}], mock=llm))
