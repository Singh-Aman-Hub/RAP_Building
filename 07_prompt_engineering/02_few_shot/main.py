"""
GOAL   : Show few-shot prompting: give worked examples (as text, or as fake chat turns)
         so the model copies the pattern - especially for tricky cases like sarcasm.
RUN    : cd 07_prompt_engineering/02_few_shot && python main.py
EXPECTED OUTPUT: zero-shot gets the sarcastic review wrong ('positive'),
         few-shot gets it right ('negative').  (SIMULATION on the mock backend.)
NEEDS  : nothing (mock). Real results: LLM_BACKEND=local / anthropic.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim_llm import make_sim
from llm_client import generate

SIM = make_sim()

SYSTEM = (
    "You are a sentiment classifier. Allowed labels: positive, negative, neutral.\n"
    "Reply with the label only, nothing else."
)
tricky = "Oh great, another delay. Just what I needed."   # sarcasm: looks positive, means negative

# ---- Zero-shot: no examples -------------------------------------------------
zero_shot = [{"role": "user", "content": f"Text: {tricky}"}]

# ---- Few-shot style 1: examples inside ONE user message ---------------------
EXAMPLES = [
    ("I love this phone, great battery.", "positive"),
    ("Oh great, the app crashed again.", "negative"),      # <- teaches sarcasm
    ("The parcel arrived on Tuesday.", "neutral"),
]
# list comprehension: builds one string per example, "\n\n".join glues them together
shots_text = "\n\n".join(f"Text: {t}\nLabel: {lbl}" for t, lbl in EXAMPLES)
few_shot_one_message = [{"role": "user", "content": f"{shots_text}\n\nText: {tricky}\nLabel:"}]

# ---- Few-shot style 2: examples as alternating user/assistant turns ---------
# The model sees a "conversation" in which it already answered correctly before.
few_shot_turns = []
for text, label in EXAMPLES:
    few_shot_turns.append({"role": "user", "content": f"Text: {text}"})
    few_shot_turns.append({"role": "assistant", "content": label})
few_shot_turns.append({"role": "user", "content": f"Text: {tricky}"})

for name, msgs in [("zero-shot", zero_shot),
                   ("few-shot (one message)", few_shot_one_message),
                   ("few-shot (chat turns)", few_shot_turns)]:
    print(f"{name:25s} -> {generate(msgs, system=SYSTEM, mock=SIM)!r}")

# Trade-off: every example costs tokens on EVERY request (money + latency + context window),
# and badly chosen / unbalanced examples bias the model toward one label.
# TRY IT: remove the sarcasm example from EXAMPLES. (The SIMULATION still passes because it only counts
#         examples; a REAL model (LLM_BACKEND=local/anthropic) is where the sarcasm example matters.)
# TRY IT: make all three examples 'positive' and think about what bias that creates.
# BREAK IT: end few_shot_turns with an assistant message -> some APIs reject it
#           (conversation must end with a user turn).
