# Terms — Step 05: Task decomposition and chain-of-thought alternatives

Folder: `07_prompt_engineering/05_decomposition` · Files covered: `main.py`, `WALKTHROUGH.md` (uses `../../shared/llm_client.py`)
Back to module: [07 Prompt engineering README](../README.md)

**Terms in this folder:** [Task decomposition](#task-decomposition) · [Pipeline (multi-stage prompt chain)](#pipeline-multi-stage-prompt-chain) · [Intermediate result](#intermediate-result) · [Chain-of-thought (alternatives)](#chain-of-thought-alternatives) · [Reasoning field](#reasoning-field) · [Self-consistency](#self-consistency) · [Majority vote margin](#majority-vote-margin) · [Temperature above zero (sampling)](#temperature-above-zero-sampling) · [collections.Counter](#collectionscounter) · [Counter.most_common](#countermost_common) · [MockLLM with rules](#mockllm-with-rules) · [MockLLM with script](#mockllm-with-script) · [extract_json](#extract_json) · [Helper function (ask)](#helper-function-ask) · [List comprehension (with _)](#list-comprehension-with-_) · [Tuple unpacking (winner, count)](#tuple-unpacking-winner-count) · [Dictionary key access](#dictionary-key-access) · [Error handling between stages](#error-handling-between-stages)

---

## Task decomposition
*Type: concept*

**1. Definition:** Splitting one complex request into several smaller prompts, each with a single job, where the output of one stage becomes input of the next.
**2. In plain words:** Small steps are easier to get right and to debug. If the final answer is wrong you can print each stage and find the first wrong one. The cost is extra model calls, so more latency and money, and mistakes can carry from one stage to the next.
**3. Example to quote:**
> "For the support ticket I made three stages: list issues, assign a priority, then plan the reply. When the plan was off I printed stage 1 and saw the issue list was wrong."

**Seen in:** `main.py`, `stage1`, `stage2`, `stage3`.

## Pipeline (multi-stage prompt chain)
*Type: concept*

**1. Definition:** A sequence of processing steps where each step's output is passed as data to the next step.
**2. In plain words:** The stages talk to each other only through your Python variables, so you can validate, log or replace any stage independently.
**3. Example to quote:**
> "Stage 2's prompt is built from `stage1['issues']`, so a validated stage-1 result is the only thing it sees."

**Seen in:** `main.py`, the three-stage block.

## Intermediate result
*Type: concept*

**1. Definition:** The output of an earlier stage in a pipeline, kept as a value you can inspect before the next stage uses it.
**2. In plain words:** Printing intermediate results is the main debugging tool for chained prompts. Fix the first wrong stage, not the last.
**3. Example to quote:**
> "I print `stage 1 ->` right after it runs, so I can see the parsed issues before stage 2 builds on them."

**Seen in:** `main.py`, the `print("stage 1 ->", ...)` lines.

## Chain-of-thought (alternatives)
*Type: concept*

**1. Definition:** Prompting a model to produce intermediate reasoning before its answer; here two alternatives are shown, a structured reasoning field and self-consistency voting.
**2. In plain words:** Thinking before answering often improves accuracy, but free-form "think step by step" text is messy to parse. The alternatives keep the answer machine-readable.
**3. Example to quote:**
> "Instead of asking for free reasoning I ask for JSON with `reasoning` first and `label` second, so I can log the reasoning but parse the label."

**Seen in:** `main.py`, comments for alternatives 1 and 2.

## Reasoning field
*Type: concept*

**1. Definition:** A JSON key (e.g. `"reasoning"`) placed before the answer key in a requested output so the model writes its reasoning first, in a parseable place.
**2. In plain words:** You get the benefit of reasoning and a clean `label` field. The reasoning is also a log you can read when the answer looks odd. The model's stated reasoning may not be a faithful account of how it truly decided.
**3. Example to quote:**
> "The reply was `{\"reasoning\": \"slow and crushed are negative...\", \"label\": \"negative\"}`; my code reads only `data['label']`."

**Seen in:** `main.py`, `reasoning_mock`, `data["reasoning"]`.

## Self-consistency
*Type: concept*

**1. Definition:** Asking the same question several times with temperature above zero and taking the most common answer (majority vote).
**2. In plain words:** One sample may be a random slip; the majority is usually steadier. It costs N model calls, so use it only when accuracy matters more than cost.
**3. Example to quote:**
> "I sampled five answers and got negative three times, neutral once, positive once, so the majority is negative with 3 of 5."

**Seen in:** `main.py`, `answers` and `Counter(...)`.

## Majority vote margin
*Type: concept*

**1. Definition:** The difference in votes between the winning answer and the others, a measure of how decisive the vote was.
**2. In plain words:** A 3 to 2 win is much weaker evidence than 5 to 0. Report the margin, not just the winner.
**3. Example to quote:**
> "If only two of five say negative, the winner is still negative but the result is thin, so I would flag it as uncertain."

**Seen in:** `WALKTHROUGH.md`, experiments.

## Temperature above zero (sampling)
*Type: concept*

**1. Definition:** Setting `temperature` greater than 0 makes the model sample tokens randomly in proportion to their probability, so repeated calls can differ.
**2. In plain words:** Voting needs variety, otherwise five identical answers add no information. That is why `temperature=0.8` is used here while classification elsewhere uses 0.
**3. Example to quote:**
> "Self-consistency only works with temperature above zero; at zero every sample would be identical."

**Seen in:** `main.py`, `temperature=0.8`.

## collections.Counter
*Type: Python standard library*

**1. Definition:** A dictionary subclass in `collections` that counts hashable items and maps each item to its count.
**2. In plain words:** Pass it a list and it tells you how many times each element appears. It is the quickest way to count votes or word frequencies.
**3. Example to quote:**
> "`Counter(answers)` turns my five votes into counts per label."

```python
from collections import Counter
Counter(["a", "a", "b"])   # Counter({'a': 2, 'b': 1})
```

**Seen in:** `main.py`, import and `Counter(answers)`.

## Counter.most_common
*Type: Python standard library*

**1. Definition:** A `Counter` method returning a list of the `n` most frequent `(item, count)` pairs, highest first.
**2. In plain words:** Asking for one gives you the winner and its count; `[0]` takes the first tuple out of the list.
**3. Example to quote:**
> "`Counter(answers).most_common(1)[0]` returned `('negative', 3)`."

**Seen in:** `main.py`, `winner, count = ...`.

## MockLLM with rules
*Type: project helper*

**1. Definition:** `MockLLM(rules=[(regex, reply), ...])` returns the reply of the first regex that matches the last user message.
**2. In plain words:** Here each stage's instruction contains `STAGE 1`, `STAGE 2` or `STAGE 3`, and the mock answers according to that keyword. It is a stand-in so the pipeline logic can be tested offline.
**3. Example to quote:**
> "My fake model recognises the text `STAGE 2` in the prompt and returns the priority JSON."

**Seen in:** `main.py`, `stage_mock`.

## MockLLM with script
*Type: project helper*

**1. Definition:** `MockLLM(script=[...])` returns canned replies one by one in order, regardless of the prompt.
**2. In plain words:** Perfect for the voting demo, where you want five known answers in a known order.
**3. Example to quote:**
> "`votes_mock` returns negative, negative, neutral, negative, positive in that order, so the vote is reproducible."

**Seen in:** `main.py`, `votes_mock`, `reasoning_mock`.

## extract_json
*Type: project function*

**1. Definition:** `shared/llm_client.py::extract_json(text)` pulls the first JSON object out of a reply and parses it; raises `ValueError` if none is found.
**2. In plain words:** It turns the stage's text into a dict. If a stage does not return JSON it raises, which is why pipelines need error handling between stages.
**3. Example to quote:**
> "When I removed the STAGE 1 rule, `extract_json` raised `ValueError` because the fallback sentence has no braces."

**Seen in:** `main.py`, stage 1 and 2 parsing; BREAK IT comment.

## Helper function (`ask`)
*Type: Python syntax / pattern*

**1. Definition:** A small function that wraps repeated code, here building a one-message conversation from an instruction and a payload and calling `generate`.
**2. In plain words:** It removes copy-paste and keeps each stage call to one line.
**3. Example to quote:**
> "`ask('STAGE 1: ...', ticket)` hides the message-building so the pipeline reads like three short calls."

**Seen in:** `main.py`, `def ask(...)`.

## List comprehension (with `_`)
*Type: Python syntax*

**1. Definition:** `[expr for item in iterable]` builds a list; the name `_` by convention marks a loop variable you do not use.
**2. In plain words:** It runs the expression five times and collects the results, here five model calls. The underscore tells readers the counter itself does not matter.
**3. Example to quote:**
> "`[generate(...) for _ in range(5)]` calls the model five times and gives me a list of answers."

**Seen in:** `main.py`, `answers = [...]`.

## Tuple unpacking (`winner, count`)
*Type: Python syntax*

**1. Definition:** Assigning the items of a tuple to several variables at once.
**2. In plain words:** `most_common(1)[0]` is the tuple `('negative', 3)`; unpacking names its two parts so the print line reads clearly.
**3. Example to quote:**
> "`winner, count = Counter(answers).most_common(1)[0]` gives me the winning label and its vote count."

**Seen in:** `main.py`, voting section.

## Dictionary key access
*Type: Python syntax*

**1. Definition:** `d["key"]` returns the value stored under that key and raises `KeyError` if absent.
**2. In plain words:** Parsed JSON becomes a dict, so stage 2's prompt reads `stage1['issues']`. If the model forgot that key you get a `KeyError`, another reason to validate between stages.
**3. Example to quote:**
> "I feed `stage1['issues']` into stage 2, which only works because stage 1 returned JSON with an `issues` key."

**Seen in:** `main.py`, stage 2 and 3 prompts.

## Error handling between stages
*Type: concept*

**1. Definition:** Wrapping each stage's parsing and validation so a bad intermediate output is caught, retried or reported instead of crashing the pipeline or feeding garbage onward.
**2. In plain words:** Chained prompts multiply errors: a malformed stage 1 breaks everything after it. Add `try/except` and validation (as in step 04) at each boundary.
**3. Example to quote:**
> "If stage 1 returns prose, `extract_json` raises; I catch it, retry that stage, and only then continue."

**Seen in:** `main.py` BREAK IT comment; `WALKTHROUGH.md` experiments.
