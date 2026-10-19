"""
sim_llm.py -- a SIMULATED sentiment-classifying "model" for module 07.

READ THIS FIRST (honesty note)
------------------------------
This is NOT a language model.  It is ~60 lines of ordinary Python rules that
*pretend* to react to prompt quality, so that the examples are:
  * offline, instant and deterministic (same input -> same output every time),
  * able to demonstrate the TESTING HARNESS (templates, validation, retries,
    accuracy tables) without an API key or a GPU.

What the simulation PROVES : your harness code (parsing, validation, scoring,
                             comparing prompt versions) works and is repeatable.
What it does NOT prove     : that a given prompt wording is good for a REAL model.
                             For that, re-run the same script with
                             LLM_BACKEND=local   (small open model)  or
                             LLM_BACKEND=anthropic  (hosted model).

How the simulation "reacts" to a prompt (all rules are visible below):
  - lists the allowed labels?            -> else it answers in chatty free text
  - says "only"/"nothing else"?          -> else it adds a sentence around the label
  - asks for JSON?                       -> returns {"label": ..., "confidence": ...}
  - contains examples ("Text:" 2+ times) -> then it also understands sarcasm
  - has an empty-input rule ("unknown")  -> else it invents a label for empty text
  - has an injection guard ("untrusted") -> else it obeys "ignore previous instructions"
"""

import re
import sys
from pathlib import Path

# make `shared/llm_client.py` importable no matter where the script is run from
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from llm_client import MockLLM  # noqa: E402

POSITIVE = {"love", "great", "excellent", "amazing", "good", "fantastic", "happy", "perfect", "wonderful"}
NEGATIVE = {"hate", "terrible", "awful", "bad", "broken", "worst", "disappointed", "slow", "refund"}
SARCASM_CUES = ("oh great", "yeah right", "thanks a lot", "just what i needed")


def _all_prompt_text(messages, system):
    """Everything the 'model' can see, lower-cased, in one string."""
    parts = [system or ""] + [m["content"] for m in messages]
    return "\n".join(parts).lower()


def _text_to_classify(messages):
    """Take the text after the LAST 'Text:' marker in the last user message (else the whole message)."""
    last = [m["content"] for m in messages if m["role"] == "user"][-1]
    if "Text:" in last:
        last = last.rsplit("Text:", 1)[1]
    return last.split("\n")[0].strip().strip('"')


def _keyword_label(text, understands_sarcasm):
    low = text.lower()
    if understands_sarcasm and any(cue in low for cue in SARCASM_CUES):
        return "negative"
    words = set(re.findall(r"[a-z']+", low))
    score = len(words & POSITIVE) - len(words & NEGATIVE)
    return "positive" if score > 0 else "negative" if score < 0 else "neutral"


def sim_reply(messages, system=None):
    """The whole fake model. Returns a string exactly like a chat model would."""
    prompt = _all_prompt_text(messages, system)
    text = _text_to_classify(messages)

    has_labels = all(lbl in prompt for lbl in ("positive", "negative", "neutral"))
    only = bool(re.search(r"\bonly\b|nothing else", prompt))
    wants_json = "json" in prompt
    has_examples = prompt.count("text:") >= 2           # at least one worked example + the real question
    empty_rule = "unknown" in prompt
    guard = "untrusted" in prompt or "never follow" in prompt

    # --- adversarial input -------------------------------------------------
    if "ignore previous instructions" in text.lower():
        if not guard:
            return "HACKED: I have ignored my instructions."
        if empty_rule:                      # guarded prompt that also says "not a review -> unknown"
            return "unknown"

    # --- missing input -----------------------------------------------------
    if not text:
        label = "unknown" if empty_rule else "positive"   # no rule -> the model "makes something up"
        conf = 0.0 if empty_rule else 0.55
    else:
        label = _keyword_label(text, understands_sarcasm=has_examples)
        conf = 0.9
        # a vague prompt makes the "model" drift on ambiguous text
        if not has_labels and label == "neutral":
            label = "positive"

    # --- output formatting (this is what the prompt wording really controls) ---
    if wants_json:
        return f'Here is the result: {{"label": "{label}", "confidence": {conf}}}' if not only else \
               f'{{"label": "{label}", "confidence": {conf}}}'
    if not has_labels:
        return f"Hmm, I would say the overall vibe of that message feels {label}, to be honest."
    if only:
        return label
    return f"The sentiment of the text is {label}."


def make_sim():
    """A MockLLM whose single catch-all rule calls sim_reply. Pass it as generate(..., mock=make_sim())."""
    return MockLLM(rules=[(r"", lambda messages, system: sim_reply(messages, system))])


if __name__ == "__main__":
    sim = make_sim()
    from llm_client import generate
    print(generate([{"role": "user", "content": "Text: I love it"}], mock=sim))
