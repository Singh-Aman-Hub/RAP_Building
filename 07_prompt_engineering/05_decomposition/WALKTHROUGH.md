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
