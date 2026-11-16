# Step 01 — Roles, tasks, instructions (zero-shot) and system vs user

## The story so far
In [module 05](../../05_llm_parameters/) you learned to control the model (temperature, top-k) and in [module 06](../../06_llm_training_and_lora/) you shrank, tuned and sped it up. But a raw model only *completes text*: give it a sentence and it continues it, whether or not that is what you wanted. This is the start of Act IV: how do we make the model do a job?

## The pain point
Send a vague prompt like "What do you think of this review?" and you get a chatty paragraph. Your program wanted one word. Free-form replies cannot be compared, counted or fed to the next line of code.

## The spark
Brief the model like a new intern: say who it is (role), what to do (task) and what the answer must look like (constraints). Put the stable rules in a `system` message and the changing input in the `user` message.

## Goal
See that the *wording* of a prompt (role, task, constraints) controls the **shape** of the reply, and learn which text belongs in the `system` prompt vs the `user` message.

## Run it
```bash
cd 07_prompt_engineering/01_roles_and_instructions
../../.venv/bin/python main.py
```
Real output (mock backend = simulation):
```
A vague                                       -> 'Hmm, I would say the overall vibe of that message feels negative, to be honest.'
B task only                                   -> 'The sentiment of the text is negative.'
C role+task+constraints (system/user split)   -> 'negative'
```
Only prompt C returns something your code can use directly (`reply == "negative"`).

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| System prompt | system message, developer message (OpenAI), `system=` parameter (Anthropic) | Mostly | Stable rules sent with every request; Anthropic takes it as a separate argument, OpenAI as a message with a special role |
| Zero-shot prompting | direct prompting, no-examples prompting | Yes | The prompt contains the task but zero worked examples |
| Role prompt | persona, "act as" prompt | Yes | One sentence telling the model who it is; it steers tone and vocabulary, not capability |
| Instruction vs constraint | task vs output format rule | No | Instruction says what to do ("classify"); constraint says how the answer must look ("label only") |
| Prompt | input, query, context | Near | Everything sent to the model, not only the user's question |

## Code, step by step

### 1. Header docstring: goal, run command, expected output  (`main.py` lines 1–10)

```python
"""
GOAL   : See how role / task / instruction wording and the system-vs-user split change
         what comes back from the model (zero-shot prompting = no examples given).
RUN    : cd 07_prompt_engineering/01_roles_and_instructions && python main.py
         (from repo root: .venv/bin/python 07_prompt_engineering/01_roles_and_instructions/main.py)
EXPECTED OUTPUT (mock backend = SIMULATION, see sim_llm.py):
         three replies: a chatty one for the vague prompt, a sentence for the half-specified
         prompt, and a bare label for the precise prompt.
NEEDS  : nothing (mock backend). For real results: LLM_BACKEND=local or anthropic.
"""
```

- **In simple words:** A note at the top that says what this script is for and how to run it. Python ignores it.
- **Exactly:** A module docstring (a string literal as the first statement). It documents that the output uses the mock backend, i.e. the rule-based simulator in `sim_llm.py`, not a real LLM.
- **Why it is here:** Without it a reader does not know that the replies are simulated.

### 2. Imports and path setup for sim_llm and llm_client  (`main.py` lines 11–17)

```python
import sys
from pathlib import Path

# Make sim_llm.py (one folder up) and shared/llm_client.py importable from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim_llm import make_sim                      # our simulated model (module-local helper)
from llm_client import generate                   # the shared text-in/text-out function
```

- **In simple words:** Pull in the tools we need. The `sys.path` line tells Python where to look for our helper files.
- **Exactly:** `sys.path.insert(0, <parent folder of this step>)` puts `07_prompt_engineering/` first on the import path, so `sim_llm` is found. `from llm_client import generate` works because `sim_llm.py` itself already inserted `<repo>/shared` on `sys.path` when it was imported (line order matters: `sim_llm` is imported first). `generate(messages, system=None, mock=...)` is text in, text out.
- **Why it is here:** Without the path insert, `python main.py` from this folder fails with `ModuleNotFoundError: No module named 'sim_llm'`.

### 3. Create the simulator and the review text  (`main.py` lines 19–21)

```python
SIM = make_sim()  # used only when LLM_BACKEND=mock; ignored by real backends

review = "The delivery was slow and the box was broken."
```

- **In simple words:** Build the fake model once and define the sentence we want classified.
- **Exactly:** `SIM = make_sim()` returns a `MockLLM` whose reply function is the rule set in `sim_llm.py` (deterministic). It is used only when `LLM_BACKEND=mock` (the default). `review` is a plain `str`.
- **Why it is here:** A fixed input and a deterministic model make runs comparable; you change only the prompt.

### 4. Prompt A: vague, no role, labels or format  (`main.py` lines 23–24)

```python
# ---- Prompt A: vague. No role, no labels, no output format. -------------------
vague = [{"role": "user", "content": f"What do you think about this? Text: {review}"}]
```

- **In simple words:** The lazy way to ask: "what do you think?". No hint about what kind of answer you want.
- **Exactly:** `vague` is a list with one message dict `{"role": "user", "content": "What do you think about this? Text: The delivery was slow and the box was broken."}`. Real output: `'Hmm, I would say the overall vibe of that message feels negative, to be honest.'` The simulator sees no label list, so it answers in chatty free text.
- **Why it is here:** It is the baseline that shows why wording matters.

### 5. Prompt B: task and labels, no output constraint  (`main.py` lines 26–30)

```python
# ---- Prompt B: task + labels, but no output constraint. ------------------------
task_only = [{
    "role": "user",
    "content": f"Classify the sentiment as positive, negative or neutral.\nText: {review}",
}]
```

- **In simple words:** Now we say what to do and which answers are allowed, but not how to format the answer.
- **Exactly:** Adds the instruction and the closed label set. Real output: `'The sentiment of the text is negative.'` The label is right but wrapped in a sentence, which is awkward for code that expects exactly `negative`.
- **Why it is here:** Shows that a correct answer in the wrong shape still breaks a program.

### 6. Prompt C: role, task and constraints split system/user  (`main.py` lines 32–40)

```python
# ---- Prompt C: ROLE + TASK + INSTRUCTIONS(constraints) split across system/user ---
# system  = stable rules that apply to every request (who you are, what format to use)
# user    = the changing input for THIS request
SYSTEM_C = (
    "You are a sentiment classifier for customer reviews.\n"          # role
    "Allowed labels: positive, negative, neutral.\n"                   # closed set
    "Reply with the label only, in lower case, nothing else."          # output constraint
)
precise = [{"role": "user", "content": f"Text: {review}"}]
```

- **In simple words:** Stable rules go in the system message, the changing text goes in the user message, like a job description versus today's ticket.
- **Exactly:** `SYSTEM_C` is three lines joined by implicit string concatenation: role, allowed labels, output constraint ("label only, in lower case, nothing else"). `precise` holds only `"Text: ..."` as the user message. The system text is later passed through `system=`, not inside `messages`.
- **Why it is here:** Keeping rules out of the per-request text makes them reusable and lets you cache and version them.

### 7. Loop over the three prompts and print replies  (`main.py` lines 42–49)

```python
for name, messages, system in [
    ("A vague", vague, None),
    ("B task only", task_only, None),
    ("C role+task+constraints (system/user split)", precise, SYSTEM_C),
]:
    # temperature=0.0 -> least random output, best for classification & testing
    reply = generate(messages, system=system, temperature=0.0, mock=SIM)
    print(f"{name:45s} -> {reply!r}")
```

- **In simple words:** Run all three prompts the same way and print the answers side by side.
- **Exactly:** Each tuple is `(name, messages, system)`. `generate(..., temperature=0.0, mock=SIM)` returns a `str`. `f"{name:45s}"` pads the name to 45 characters; `{reply!r}` prints the repr with quotes so you see trailing spaces or newlines. Real output for C: `'negative'`, a bare label.
- **Why it is here:** Using one loop guarantees every prompt is called identically; only the prompt differs.

### 8. TRY IT / BREAK IT suggestions  (`main.py` lines 51–54)

```python
# TRY IT: delete the sentence "Reply with the label only..." from SYSTEM_C and re-run.
# TRY IT: change `review` to "I love it" and watch all three prompts change.
# BREAK IT: write messages=[{"role": "robot", "content": "hi"}] and run with
#           LLM_BACKEND=anthropic -> the API rejects unknown roles (a 400 error).
```

- **In simple words:** Experiments for you to run after reading.
- **Exactly:** Comments only. The BREAK IT needs `LLM_BACKEND=anthropic`; the mock does not check roles.
- **Why it is here:** Practice changing one thing and observing the effect.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `sys.path.insert(0, ...)` | add a folder to the places Python searches for imports | `sys.path.insert(0, "/tmp/x")` |
| `Path(__file__).resolve().parents[1]` | absolute path of the folder one level above this file | `Path("/a/b/c.py").parents[1]` -> `/a` |
| `f"...{review}..."` | f-string: puts a variable's value inside text | `f"hi {name}"` |
| `[{"role": ..., "content": ...}]` | list of dicts = the chat message format | `msgs = [{"role": "user", "content": "hi"}]` |
| `for name, messages, system in [(...), (...)]` | loop that unpacks each tuple into 3 names | `for a, b in [(1, 2), (3, 4)]:` |
| `{name:45s}` | format spec: pad the string to width 45 | `f"{'ab':5s}|"` -> `ab   |` |
| `{reply!r}` | show the `repr` (with quotes) — reveals hidden spaces/newlines | `f"{'a'!r}"` -> `'a'` |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `generate` (`shared/llm_client.py`) | our code | sends messages to the chosen backend | `messages`, `system`, `temperature`, `mock` | reply `str` |
| `make_sim` (`sim_llm.py`) | our code | builds the fake rule-based model | – | `MockLLM` |
| `temperature` | model parameter | 0 = most deterministic, higher = more random (module [05](../../05_llm_parameters/)) | float | – |
| `system` | API field | stable rules/role for the whole conversation | str or None | – |

## Execution trace (input -> processing -> output)
1. **Input:** the same review `"The delivery was slow and the box was broken."` placed in three different prompts.
2. **Processing:** `generate` forwards `messages` (+ `system`) to the backend. The simulated model counts keywords (`slow`, `broken` -> negative) and *then* formats the reply according to what the prompt asked for: no labels listed -> chatty; labels listed -> a sentence; "label only" -> bare label.
3. **Output:** three strings. The content (negative) is identical; the *format* differs. Format is what breaks downstream code.

## Experiments (change ONE thing)
- Remove `Reply with the label only...` from `SYSTEM_C` -> C degrades to the sentence form.
- Change `review` to `"I love it"` -> all three flip to positive; only the wrapper text differs.
- Move the allowed-labels line from system into the user message -> same result in the simulation; with a real model, test whether it matters (it often matters little; consistency matters more).

## Debugging practice
`messages = [{"role": "robot", "content": "hi"}]` with a real API returns HTTP 400 `invalid_request_error`. Read the message: it names the field (`messages.0.role`). Rule: **read the error's field path first**.

## Rebuild checklist
- [ ] Write a system prompt with role, closed label set, output constraint.
- [ ] Build `messages` with one user dict.
- [ ] Loop over 3 prompts and print aligned results.

## The new problem (cliffhanger)
One precise instruction works for easy cases. For tricky ones (sarcasm) the model still misreads, and words alone cannot show it what you mean. What if you *showed* it solved examples? Next: [02_few_shot](../02_few_shot/).

> **Say it aloud (interview box):** Zero-shot means I give instructions but no examples. I put stable rules (role, allowed labels, output format) in the system prompt and the changing input in the user message, at temperature 0 for repeatable results. The trade-off is that instructions can still be misread, so I validate the output instead of trusting it.
