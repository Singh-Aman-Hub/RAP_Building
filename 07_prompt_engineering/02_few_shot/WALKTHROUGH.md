# Step 02 — Few-shot prompting

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

## Say it aloud
"Few-shot prompting shows the model examples of input and desired output in the prompt, so it copies the pattern without any training. I use balanced, realistic examples including one hard case. The cost is tokens on every request, and poorly chosen or unbalanced examples bias the output."
