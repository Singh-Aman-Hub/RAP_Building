# Step 05 — Task decomposition & chain-of-thought alternatives

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

## Say it aloud
"Decomposition splits a task into small prompts whose outputs I can validate. Instead of free-form chain-of-thought I can request a reasoning field before the answer, or sample several answers and vote. The trade-off is more calls — more latency and cost — and errors can compound between stages."
