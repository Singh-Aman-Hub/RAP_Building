"""
GOAL   : Split one hard request into several small prompts (task decomposition) and compare
         with other 'think before answering' tricks: reasoning field, self-consistency voting.
RUN    : cd 07_prompt_engineering/05_decomposition && python main.py
EXPECTED OUTPUT: a 3-stage pipeline result for a messy support ticket, a reasoning-field JSON
         reply, and a majority vote over 5 sampled answers.
NEEDS  : nothing. The mock replies are SCRIPTED rules (simulation), not a real model.
"""
import sys
from collections import Counter     # Counter: dict subclass that counts items (used again in module 02)
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from llm_client import generate, extract_json, MockLLM

ticket = ("My order #4521 arrived late and the box was crushed. Support was friendly though. "
          "I want a refund for the shipping cost.")

# A fake model that answers whichever STAGE instruction it receives (matched by keyword).
stage_mock = MockLLM(rules=[
    (r"STAGE 1", '{"issues": ["late delivery", "damaged box", "refund request"], "good": ["friendly support"]}'),
    (r"STAGE 2", '{"priority": "high", "reason": "refund requested and goods damaged"}'),
    (r"STAGE 3", "Apologise for the late, damaged delivery, thank them for the kind words about support, "
                 "and approve the shipping refund."),
])

# ---------------------------------------------------------------------------
# Decomposition: each stage has ONE job, a small prompt, and an output we can check.
# The output of stage N is data fed into the prompt of stage N+1.
# ---------------------------------------------------------------------------
def ask(stage_instruction: str, payload: str) -> str:
    return generate([{"role": "user", "content": f"{stage_instruction}\n\n{payload}"}], mock=stage_mock)

stage1 = extract_json(ask("STAGE 1: list the issues and the positives as JSON.", ticket))
print("stage 1 ->", stage1)                                  # inspect intermediate results!
stage2 = extract_json(ask("STAGE 2: give a priority (low/medium/high) as JSON with a reason.",
                          f"Issues: {stage1['issues']}"))
print("stage 2 ->", stage2)
stage3 = ask("STAGE 3: write a one-sentence plan for the reply email.",
             f"Issues: {stage1['issues']}\nPriority: {stage2['priority']}\nPositives: {stage1['good']}")
print("stage 3 ->", stage3)
# Why decompose? Easier to debug (you can see which stage failed), each stage can have its own
# validation and even its own cheaper model. Cost: more calls = more latency and money.

# ---------------------------------------------------------------------------
# Alternative 1 to "think step by step": a structured REASONING FIELD placed BEFORE the answer.
# The reasoning is machine-readable, so you can log it, but the answer is still a clean field.
# ---------------------------------------------------------------------------
reasoning_mock = MockLLM(script=['{"reasoning": "Words like slow and crushed are negative; '
                                 'friendly support is positive but minor.", "label": "negative"}'])
reply = generate([{"role": "user", "content":
                   'Return JSON {"reasoning": "...", "label": "..."} - reasoning FIRST, then label.\n'
                   f"Text: {ticket}"}], mock=reasoning_mock)
data = extract_json(reply)
print("\nreasoning field ->", data["reasoning"], "| label:", data["label"])

# ---------------------------------------------------------------------------
# Alternative 2: SELF-CONSISTENCY. Ask the same question several times with temperature > 0,
# then take the majority vote. Costs N calls, reduces one-off random mistakes.
# ---------------------------------------------------------------------------
votes_mock = MockLLM(script=["negative", "negative", "neutral", "negative", "positive"])
answers = [generate([{"role": "user", "content": f"Label this: {ticket}"}],
                    temperature=0.8, mock=votes_mock) for _ in range(5)]   # list comprehension x5
winner, count = Counter(answers).most_common(1)[0]   # most_common(1) -> [("negative", 3)]
print("\nsampled answers:", answers, "-> majority:", winner, f"({count}/5)")

# TRY IT: change the third script entry so only 2 of 5 say 'negative' - is the vote still clear?
# TRY IT: make STAGE 2 return invalid JSON and add validation + retry (reuse step 04).
# BREAK IT: remove the 'STAGE 1' regex rule -> the mock falls back to its default sentence and
#           extract_json raises ValueError: your pipeline needs error handling between stages.
