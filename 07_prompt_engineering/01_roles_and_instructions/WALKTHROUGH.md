# Step 01 — Roles, tasks, instructions (zero-shot) and system vs user

## The story so far
In [module 05](../../05_llm_parameters/) you learned to control the model (temperature, top-k) and in [module 06](../../06_llm_training_and_fast_inference/) you shrank, tuned and sped it up. But a raw model only *completes text*: give it a sentence and it continues it, whether or not that is what you wanted. This is the start of Act IV: how do we make the model do a job?

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
