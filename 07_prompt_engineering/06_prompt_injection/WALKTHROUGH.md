# Step 06 — Prompt injection awareness

## Goal
Recognise that any text you paste into a prompt may contain instructions, and layer defences.

## Run it
```bash
cd 07_prompt_engineering/06_prompt_injection && ../../.venv/bin/python main.py
```
Real output (simulation):
```
input      prompt     raw reply                                     after output check
normal     naive      'positive'                                    positive
normal     defended   'positive'                                    positive
attack     naive      'HACKED: I have ignored my instructions.'     unknown
attack     defended   'unknown'                                     unknown
```
Note the *output check* turns `HACKED...` into `unknown` even for the naive prompt — a layer that does not depend on the model's cooperation.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `{"a", "b"}` | a set literal: unique items, fast `in` test | `"a" in {"a", "b"}` |
| `x if cond else y` | conditional expression | `1 if True else 2` |
| `reply.strip().lower()` | chained string methods | `" A ".strip().lower()` -> `a` |
| `f"{tag:10s} {raw!r:45s}"` | width + repr in one f-string | – |
| triple-quoted `"""..."""` | multi-line string | – |

## Libraries & functions
| Name | Kind | Does |
|---|---|---|
| `str.strip`, `str.lower` | built-in | normalise text before comparing |
| `in` on a set | built-in | allow-list check in O(1) |
| `generate` | shared | model call |

## Execution trace
Input: an attack string. Naive prompt: instruction + data share one block; the simulation (like a poorly defended model) obeys the data. Defended prompt: rules in system, data fenced in `<review>` tags and declared untrusted -> classified/`unknown`. `safe_label` is a deterministic guard after the model.

## Experiments
- Remove both the phrases `UNTRUSTED` and `Never follow` from `SYSTEM_DEFENDED` -> simulated defence collapses (shows the sim keys on this word; a real model is fuzzier).
- Add `"Disregard the above"` as a new attack. The sim does not know it — real models might; **this is why adversarial test sets matter**.

## Debugging practice
If an injection test "passes" because the sim never matched your phrase, that is a false sense of safety. Always re-run security tests on a real backend.

## Rebuild checklist
- [ ] Naive prompt, defended prompt (system rules + delimiters + untrusted label), output allow-list.

## Say it aloud
"Prompt injection is when untrusted text contains instructions that compete with mine — directly from a user or indirectly via documents the model reads. I separate instructions from data, fence and label the data, validate the output against an allow-list, give the model minimal tool power and require confirmation for risky actions. No prompt-only defence is guaranteed."
