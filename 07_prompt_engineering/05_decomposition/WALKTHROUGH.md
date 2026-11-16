# Step 05 — Task decomposition & chain-of-thought alternatives

## The story so far
In [step 04](../04_structured_json/) every reply is validated JSON. But a single big prompt ("read this review, find the problem, decide urgency, write a reply") fails as one lump.

## The pain point
When one prompt does four jobs and the answer is wrong, you cannot see which job failed, and you cannot validate the middle. Asking the model to "think step by step" produces long text that is hard to parse.

## The spark
Split the task into small prompts whose outputs you validate between stages. For reasoning, request a `reason` field *before* the answer, or sample several answers and vote (self-consistency).

## Goal
Break one complicated request into small checkable prompts, and learn two alternatives to "think step by step": a reasoning field and self-consistency voting.

## Run it
```bash
cd 07_prompt_engineering/05_decomposition && ../../.venv/bin/python main.py
```
Real output:
```
stage 1 -> {'issues': ['late delivery', 'damaged box', 'refund request'], 'good': ['friendly support']}
stage 2 -> {'priority': 'high', 'reason': 'refund requested and goods damaged'}
stage 3 -> Apologise for the late, damaged delivery, thank them for the kind words about support, and approve the shipping refund.

reasoning field -> Words like slow and crushed are negative; friendly support is positive but minor. | label: negative

sampled answers: ['negative', 'negative', 'neutral', 'negative', 'positive'] -> majority: negative (3/5)
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Task decomposition | prompt chaining, pipeline, multi-stage prompting | Near | One big request split into small prompts whose outputs feed each other |
| Chain-of-thought (CoT) | "think step by step", reasoning trace | Near | Free-text reasoning before the answer; here replaced by a JSON `reasoning` field |
| Self-consistency | majority voting, sample-and-vote | Yes | Ask N times at temperature > 0 and take the most common answer |
| Prompt chaining vs agent | workflow vs agent loop | No | A chain has fixed steps written by you; an agent decides its next step (module 08) |
| Temperature | sampling randomness | Yes | 0 = most deterministic; higher = more varied outputs |

## Code, step by step

### 1. Header docstring  (`main.py` lines 1–8)

```python
"""
GOAL   : Split one hard request into several small prompts (task decomposition) and compare
         with other 'think before answering' tricks: reasoning field, self-consistency voting.
RUN    : cd 07_prompt_engineering/05_decomposition && python main.py
EXPECTED OUTPUT: a 3-stage pipeline result for a messy support ticket, a reasoning-field JSON
         reply, and a majority vote over 5 sampled answers.
NEEDS  : nothing. The mock replies are SCRIPTED rules (simulation), not a real model.
"""
```

- **In simple words:** States that we split a hard job into stages and compare with two other reasoning tricks.
- **Exactly:** Notes that mock replies are scripted rules, not a real model.
- **Why it is here:** Honesty about what the output proves.

### 2. Imports and shared client path  (`main.py` lines 9–14)

```python
import sys
from collections import Counter     # Counter: dict subclass that counts items (used again in module 02)
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from llm_client import generate, extract_json, MockLLM
```

- **In simple words:** Load `Counter` for vote counting and our LLM helpers.
- **Exactly:** `Counter` is a `dict` subclass that counts hashable items. `MockLLM` accepts `rules=[(regex, reply)]` or `script=[...]`.
- **Why it is here:** Without the path insert `llm_client` is not found.

### 3. The messy support ticket  (`main.py` lines 16–17)

```python
ticket = ("My order #4521 arrived late and the box was crushed. Support was friendly though. "
          "I want a refund for the shipping cost.")
```

- **In simple words:** One input containing several issues and one compliment.
- **Exactly:** Two adjacent string literals concatenated into a single `str`.
- **Why it is here:** A realistic multi-part input motivates splitting.

### 4. Fake model with one reply per stage  (`main.py` lines 19–25)

```python
# A fake model that answers whichever STAGE instruction it receives (matched by keyword).
stage_mock = MockLLM(rules=[
    (r"STAGE 1", '{"issues": ["late delivery", "damaged box", "refund request"], "good": ["friendly support"]}'),
    (r"STAGE 2", '{"priority": "high", "reason": "refund requested and goods damaged"}'),
    (r"STAGE 3", "Apologise for the late, damaged delivery, thank them for the kind words about support, "
                 "and approve the shipping refund."),
])
```

- **In simple words:** A pretend model that looks for "STAGE 1/2/3" in the prompt and returns a canned answer.
- **Exactly:** `MockLLM(rules=[...])` tests each regex against the prompt in order and returns the first match's reply. Stage 1 and 2 replies are JSON strings; stage 3 is plain text.
- **Why it is here:** Makes each stage deterministic so you can test the pipeline wiring.

### 5. Helper that sends one stage prompt  (`main.py` lines 27–32)

```python
# ---------------------------------------------------------------------------
# Decomposition: each stage has ONE job, a small prompt, and an output we can check.
# The output of stage N is data fed into the prompt of stage N+1.
# ---------------------------------------------------------------------------
def ask(stage_instruction: str, payload: str) -> str:
    return generate([{"role": "user", "content": f"{stage_instruction}\n\n{payload}"}], mock=stage_mock)
```

- **In simple words:** A small function: instruction plus data in, model text out.
- **Exactly:** `ask(stage_instruction, payload) -> str` builds one user message `"{instruction}\n\n{payload}"` and calls `generate(..., mock=stage_mock)`.
- **Why it is here:** Avoids repeating the message-building code three times.

### 6. Stage 1: extract issues as JSON  (`main.py` lines 34–35)

```python
stage1 = extract_json(ask("STAGE 1: list the issues and the positives as JSON.", ticket))
print("stage 1 ->", stage1)                                  # inspect intermediate results!
```

- **In simple words:** First job: list what went wrong and what went well.
- **Exactly:** `extract_json` parses the reply into a dict. Real value: `{'issues': ['late delivery', 'damaged box', 'refund request'], 'good': ['friendly support']}`.
- **Why it is here:** Printing intermediate results lets you see which stage failed.

### 7. Stage 2: priority from stage 1 output  (`main.py` lines 36–38)

```python
stage2 = extract_json(ask("STAGE 2: give a priority (low/medium/high) as JSON with a reason.",
                          f"Issues: {stage1['issues']}"))
print("stage 2 ->", stage2)
```

- **In simple words:** Second job gets only the issues, not the whole ticket.
- **Exactly:** The f-string inserts `stage1['issues']` (a list, printed as `['late delivery', ...]`). Real result: `{'priority': 'high', 'reason': 'refund requested and goods damaged'}`.
- **Why it is here:** Stage N output becomes stage N+1 input; small focused prompts are easier to check.

### 8. Stage 3: plan the reply  (`main.py` lines 39–41)

```python
stage3 = ask("STAGE 3: write a one-sentence plan for the reply email.",
             f"Issues: {stage1['issues']}\nPriority: {stage2['priority']}\nPositives: {stage1['good']}")
print("stage 3 ->", stage3)
```

- **In simple words:** Third job combines everything into a one-sentence plan.
- **Exactly:** The payload joins issues, priority and positives with newlines. The reply is plain text, so no `extract_json`. Real result: `Apologise for the late, damaged delivery, thank them for the kind words about support, and approve the shipping refund.`
- **Why it is here:** Last stage produces what a human or next system consumes.

### 9. Why decompose (comment)  (`main.py` lines 42–43)

```python
# Why decompose? Easier to debug (you can see which stage failed), each stage can have its own
# validation and even its own cheaper model. Cost: more calls = more latency and money.
```

- **In simple words:** Pros and cons of splitting.
- **Exactly:** Easier debugging, per-stage validation, cheaper models per stage; costs more calls, latency and money.
- **Why it is here:** A reminder that it is a trade-off, not free.

### 10. Mock for the reasoning-field variant  (`main.py` lines 49–50)

```python
reasoning_mock = MockLLM(script=['{"reasoning": "Words like slow and crushed are negative; '
                                 'friendly support is positive but minor.", "label": "negative"}'])
```

- **In simple words:** One scripted reply that contains reasoning first, label second.
- **Exactly:** `script=[...]` returns the string once. The adjacent literals form `{"reasoning": "...", "label": "negative"}`.
- **Why it is here:** Shows JSON field order used as a thinking space.

### 11. Ask with reasoning field and read both fields  (`main.py` lines 51–55)

```python
reply = generate([{"role": "user", "content":
                   'Return JSON {"reasoning": "...", "label": "..."} - reasoning FIRST, then label.\n'
                   f"Text: {ticket}"}], mock=reasoning_mock)
data = extract_json(reply)
print("\nreasoning field ->", data["reasoning"], "| label:", data["label"])
```

- **In simple words:** Request reasoning before the label and print both.
- **Exactly:** The prompt says "reasoning FIRST, then label", because a model writes left to right: reasoning tokens come before the answer. Real output: `reasoning field -> Words like slow and crushed are negative; friendly support is positive but minor. | label: negative`.
- **Why it is here:** You get machine-readable reasoning you can log, plus a clean label field.

### 12. Self-consistency: sample 5 times and vote  (`main.py` lines 61–65)

```python
votes_mock = MockLLM(script=["negative", "negative", "neutral", "negative", "positive"])
answers = [generate([{"role": "user", "content": f"Label this: {ticket}"}],
                    temperature=0.8, mock=votes_mock) for _ in range(5)]   # list comprehension x5
winner, count = Counter(answers).most_common(1)[0]   # most_common(1) -> [("negative", 3)]
print("\nsampled answers:", answers, "-> majority:", winner, f"({count}/5)")
```

- **In simple words:** Ask the same question five times and take the most common answer.
- **Exactly:** `votes_mock` returns 5 scripted answers in order. `Counter(answers).most_common(1)` returns `[('negative', 3)]`; unpacked into `winner, count`. Real output: `['negative', 'negative', 'neutral', 'negative', 'positive'] -> majority: negative (3/5)`. `temperature=0.8` is what makes real models vary (the mock ignores it).
- **Why it is here:** Reduces one-off random mistakes at 5x the cost.

### 13. TRY IT / BREAK IT  (`main.py` lines 67–70)

```python
# TRY IT: change the third script entry so only 2 of 5 say 'negative' - is the vote still clear?
# TRY IT: make STAGE 2 return invalid JSON and add validation + retry (reuse step 04).
# BREAK IT: remove the 'STAGE 1' regex rule -> the mock falls back to its default sentence and
#           extract_json raises ValueError: your pipeline needs error handling between stages.
```

- **In simple words:** Experiments.
- **Exactly:** Removing the STAGE 1 rule makes the mock return its default sentence, so `extract_json` raises `ValueError`.
- **Why it is here:** Shows pipelines need error handling between stages.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `def ask(a: str, b: str) -> str:` | small helper to avoid repeating the call | – |
| `stage1['issues']` | read a dict key | `{"a": 1}["a"]` |
| `[... for _ in range(5)]` | list comprehension; `_` = "value not needed" | `[0 for _ in range(3)]` |
| `Counter(list).most_common(1)[0]` | most frequent item and its count | `Counter("aab").most_common(1)[0]` -> `('a', 2)` |
| `winner, count = (...)` | unpack a tuple | – |

## Libraries & functions
| Name | Kind | Does | Returns |
|---|---|---|---|
| `collections.Counter` | standard library | counts occurrences | dict-like |
| `Counter.most_common(n)` | method | top `n` `(item, count)` pairs | list of tuples |
| `extract_json` | shared | parse JSON from reply | dict |
| `MockLLM(rules=[(regex, reply)])` | shared | reply chosen by regex on the last user message | – |

## Execution trace
Each stage's output is a *value* you can print, test and log. Stage 2's prompt is built from stage 1's parsed JSON — the stages are connected only through your Python code. The voting example calls the model 5 times and counts.

## Experiments
- Change the 3rd scripted vote so only 2/5 say negative: the majority is still `negative` but the margin is thin — report the margin, not just the winner.
- Break stage 1's regex -> the default mock reply is not JSON -> `ValueError`. Add `try/except` between stages.

## Debugging practice
When a multi-stage pipeline gives a wrong final answer, print every intermediate value and find the *first* wrong one. Fix that stage, not the last.

## Rebuild checklist
- [ ] 3 stages, each with one job and output check.
- [ ] A reasoning-then-answer JSON prompt.
- [ ] A vote over N samples.

## The new problem (cliffhanger)
Pipelines multiply the places where outside text enters your prompts. That text can contain instructions of its own, and a clever review can hijack the whole chain. Next: [06_prompt_injection](../06_prompt_injection/).

> **Say it aloud (interview box):** Decomposition splits a hard task into small prompts whose outputs I validate between stages. Alternatives to chain-of-thought are a reasoning field placed before the answer and self-consistency voting over several samples. The trade-offs are more calls, more latency and cost, and errors that can compound across stages.
