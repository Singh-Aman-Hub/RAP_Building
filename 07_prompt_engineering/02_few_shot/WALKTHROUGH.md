# Step 02 — Few-shot prompting

## The story so far
In [step 01](../01_roles_and_instructions/) a clear role, task and constraints made the reply the right *shape*. But shape is not judgement: the instructions never told the model what to do with sarcasm like "Oh great, another delay".

## The pain point
Describing a tricky rule in words is long and still ambiguous. The model reads "great" and says positive, because the instruction did not contain the exception.

## The spark
Show, do not tell. Add two or three worked input -> output examples to the prompt. The model continues the pattern, including the edge cases the examples demonstrate.

## Goal
Give the model worked examples so it imitates the pattern (format *and* tricky decisions such as sarcasm).

## Run it
```bash
cd 07_prompt_engineering/02_few_shot && ../../.venv/bin/python main.py
```
Real output (simulation):
```
zero-shot                 -> 'positive'
few-shot (one message)    -> 'negative'
few-shot (chat turns)     -> 'negative'
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Few-shot prompting | in-context learning (ICL), example-based prompting | Near | Worked examples inside the prompt; no weights change |
| Few-shot vs fine-tuning | training on examples | No | Few-shot costs tokens on every request and forgets after the call; fine-tuning changes the weights once |
| Shot | example, demonstration, exemplar | Yes | One input-output pair in the prompt |
| Zero-shot / one-shot / few-shot | 0 / 1 / N examples | Yes | Just the count of examples |
| Chat turns vs one message | multi-turn few-shot vs inline few-shot | No | Same information, different packaging: fake user/assistant turns or a single text block |

## Code, step by step

### 1. Header docstring  (`main.py` lines 1–8)

```python
"""
GOAL   : Show few-shot prompting: give worked examples (as text, or as fake chat turns)
         so the model copies the pattern - especially for tricky cases like sarcasm.
RUN    : cd 07_prompt_engineering/02_few_shot && python main.py
EXPECTED OUTPUT: zero-shot gets the sarcastic review wrong ('positive'),
         few-shot gets it right ('negative').  (SIMULATION on the mock backend.)
NEEDS  : nothing (mock). Real results: LLM_BACKEND=local / anthropic.
"""
```

- **In simple words:** Says the step shows examples making the model copy a pattern.
- **Exactly:** Documents expected output: zero-shot `'positive'` (wrong), few-shot `'negative'` (right), on the simulated backend.
- **Why it is here:** Tells the reader the sarcasm case is the point.

### 2. Imports, path setup, simulator  (`main.py` lines 9–16)

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim_llm import make_sim
from llm_client import generate

SIM = make_sim()
```

- **In simple words:** Same setup as step 01: make our helpers importable and create the fake model.
- **Exactly:** `sys.path.insert(0, parent)` exposes `sim_llm`; `SIM = make_sim()` is the rule-based `MockLLM`.
- **Why it is here:** Without the path line the imports fail from this folder.

### 3. System prompt and the sarcastic test text  (`main.py` lines 18–22)

```python
SYSTEM = (
    "You are a sentiment classifier. Allowed labels: positive, negative, neutral.\n"
    "Reply with the label only, nothing else."
)
tricky = "Oh great, another delay. Just what I needed."   # sarcasm: looks positive, means negative
```

- **In simple words:** The rules plus a sentence that sounds happy but means unhappy.
- **Exactly:** `SYSTEM` is a two-line string (labels, label-only). `tricky` is `"Oh great, another delay. Just what I needed."` containing sarcasm cues ("oh great", "just what i needed") that the simulator only recognises when examples exist.
- **Why it is here:** A sarcasm case is a known place where zero-shot fails.

### 4. Zero-shot prompt  (`main.py` lines 24–25)

```python
# ---- Zero-shot: no examples -------------------------------------------------
zero_shot = [{"role": "user", "content": f"Text: {tricky}"}]
```

- **In simple words:** Ask with no examples.
- **Exactly:** One user message `"Text: Oh great, ..."`. Real result: `'positive'`, because the simulator counts the word "great" as positive.
- **Why it is here:** The baseline to beat.

### 5. Few-shot inside one user message  (`main.py` lines 27–35)

```python
# ---- Few-shot style 1: examples inside ONE user message ---------------------
EXAMPLES = [
    ("I love this phone, great battery.", "positive"),
    ("Oh great, the app crashed again.", "negative"),      # <- teaches sarcasm
    ("The parcel arrived on Tuesday.", "neutral"),
]
# list comprehension: builds one string per example, "\n\n".join glues them together
shots_text = "\n\n".join(f"Text: {t}\nLabel: {lbl}" for t, lbl in EXAMPLES)
few_shot_one_message = [{"role": "user", "content": f"{shots_text}\n\nText: {tricky}\nLabel:"}]
```

- **In simple words:** Paste three solved examples above the real question.
- **Exactly:** `EXAMPLES` is a list of `(text, label)` tuples. `shots_text` joins `"Text: ...\nLabel: ..."` blocks with blank lines. The final message ends with `Text: <tricky>\nLabel:` so the model continues after `Label:`. Real result: `'negative'`.
- **Why it is here:** Teaches the pattern (including that sarcasm is negative) at the cost of extra tokens each call.

### 6. Few-shot as alternating user/assistant turns  (`main.py` lines 37–43)

```python
# ---- Few-shot style 2: examples as alternating user/assistant turns ---------
# The model sees a "conversation" in which it already answered correctly before.
few_shot_turns = []
for text, label in EXAMPLES:
    few_shot_turns.append({"role": "user", "content": f"Text: {text}"})
    few_shot_turns.append({"role": "assistant", "content": label})
few_shot_turns.append({"role": "user", "content": f"Text: {tricky}"})
```

- **In simple words:** Pretend the model already answered earlier questions correctly, then ask the new one.
- **Exactly:** The loop appends `{"role":"user"}` and `{"role":"assistant"}` pairs, so `few_shot_turns` has 3x2+1 = 7 messages, ending with a user turn. Real result: `'negative'`.
- **Why it is here:** Chat models are trained on conversation shape, so past assistant turns are a strong format cue.

### 7. Compare the three prompts  (`main.py` lines 45–48)

```python
for name, msgs in [("zero-shot", zero_shot),
                   ("few-shot (one message)", few_shot_one_message),
                   ("few-shot (chat turns)", few_shot_turns)]:
    print(f"{name:25s} -> {generate(msgs, system=SYSTEM, mock=SIM)!r}")
```

- **In simple words:** Run each version and print what comes back.
- **Exactly:** `generate(msgs, system=SYSTEM, mock=SIM)` for each. Output: zero-shot `'positive'`, one message `'negative'`, chat turns `'negative'`.
- **Why it is here:** One loop = same conditions for each variant.

### 8. Trade-off note, TRY IT, BREAK IT  (`main.py` lines 50–56)

```python
# Trade-off: every example costs tokens on EVERY request (money + latency + context window),
# and badly chosen / unbalanced examples bias the model toward one label.
# TRY IT: remove the sarcasm example from EXAMPLES. (The SIMULATION still passes because it only counts
#         examples; a REAL model (LLM_BACKEND=local/anthropic) is where the sarcasm example matters.)
# TRY IT: make all three examples 'positive' and think about what bias that creates.
# BREAK IT: end few_shot_turns with an assistant message -> some APIs reject it
#           (conversation must end with a user turn).
```

- **In simple words:** Reminder that examples cost money and can bias the model.
- **Exactly:** Comments. The simulator only counts "Text:" markers (2 or more), so removing the sarcasm example still passes there; a real model would not.
- **Why it is here:** Warns against trusting a simulation for wording quality.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| list of tuples `[("text", "label"), ...]` | an ordered collection of pairs | `pairs = [("a", 1)]` |
| generator inside `join` | `"\n\n".join(f"..." for t, lbl in EXAMPLES)` builds many strings and glues them | `", ".join(str(i) for i in range(3))` -> `0, 1, 2` |
| `for text, label in EXAMPLES:` | tuple unpacking in a loop | `for k, v in [(1, 2)]:` |
| `list.append(x)` | add one item at the end | `a = []; a.append(1)` |
| `msgs += [...]` | extend a list in place | `a += [2, 3]` |

## Libraries & functions
| Name | Kind | Does | Args | Returns |
|---|---|---|---|---|
| `str.join` | built-in | glue strings with a separator | iterable of str | str |
| `list.append` | built-in | append one item | item | None |
| `generate` | shared helper | call the model | `messages`, `system`, `mock` | str |

## Execution trace
1. **Input:** the sarcastic text `"Oh great, another delay. Just what I needed."`
2. **Processing:** zero-shot has one `Text:` marker so the simulated model has no examples and reads "great" as positive. The two few-shot versions contain 3+ `Text:` markers (the examples), which the simulation treats as "has examples" and unlocks sarcasm handling. *(A real model learns this from the example `Oh great, the app crashed again. -> negative`.)*
3. **Output:** `positive` (wrong) vs `negative` (right).

## Experiments
- Delete the sarcasm example from `EXAMPLES` -> few-shot still has 2 examples in the simulation, so the sim still "gets it"; note this is a **limit of the simulation** — with a real backend the sarcasm example is what matters. Try `LLM_BACKEND=local`.
- Make all three examples `positive` -> real models skew toward `positive` (label bias).
- Shuffle example order: real models are sensitive to order (the last example has the strongest pull).

## Debugging practice
Conversation ending with an `assistant` message: some APIs reply `400` ("last message must be user") and others continue the assistant text. Always end few-shot turns with the real user question.

## Rebuild checklist
- [ ] Build 3 balanced examples (one per label).
- [ ] Format them as text once, as chat turns once.
- [ ] Append the real question last.

## The new problem (cliffhanger)
Examples work, but they are now a block of text you copy-paste and edit by hand. Change one word and you cannot tell which version was running. We need prompts as reusable, versioned objects. Next: [03_prompt_templates](../03_prompt_templates/).

> **Say it aloud (interview box):** Few-shot prompting puts a few worked examples in the prompt so the model imitates both the format and the tricky decisions. The examples can be text in one message or as alternating user and assistant chat turns. The costs are extra tokens on every call and bias, because unbalanced or reordered examples push the model toward certain labels.
