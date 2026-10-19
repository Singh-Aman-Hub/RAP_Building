# Step 07 — Prompt testing & evaluation

## The story so far
In [step 06](../06_prompt_injection/) you defended a prompt, but you judged it by a couple of hand-picked attacks.

## The pain point
Two or three examples prove nothing. Change a prompt and you cannot say whether it got better, or whether you quietly broke the sarcasm case it handled yesterday.

## The spark
Treat prompts as code: a fixed test set (normal, ambiguous, missing, adversarial), temperature 0, a parser, a per-category accuracy table, and saved results per version.

## Goal
Compare prompt versions with a small, fixed, categorised test set and a parser, and save results so changes can be compared over time.

## Run it
```bash
cd 07_prompt_engineering/07_prompt_testing && ../../.venv/bin/python main.py
LLM_BACKEND=local ../../.venv/bin/python main.py        # real small model
```
Real output (mock backend = simulation):
```
backend = mock  (13 test cases)

version                  normal    ambiguous      missing  adversarial   OVERALL
v1_vague                    4/4          0/5          0/2          0/2      31%
v2_labels_format            4/4          2/5          2/2          0/2      62%
v3_full                     4/4          4/5          2/2          2/2      92%

Best version: v3_full. Its failures:
  [ambiguous] text='It was not bad at all.' expected=positive got=negative reply='negative'

Saved results_mock.json
```

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| list of 3-tuples `TESTS` | rows of data: (category, text, expected) | – |
| dict of tuples `PROMPTS` | name -> (system, user template) | – |
| `{n: f(n) for n in names}` | dict comprehension | `{x: x*2 for x in [1, 2]}` |
| `defaultdict(list)` | dict that creates `[]` for a new key | `d = defaultdict(list); d["a"].append(1)` |
| `max(d, key=lambda n: ...)` | choose the item with the biggest computed score; `lambda` = tiny anonymous function | `max([1,5,3], key=lambda x: -x)` -> 1 |
| `sum(list_of_bools)` | True counts as 1 | `sum([True, False, True])` -> 2 |
| `f"{x:>9.0%}"` | right-align, show as percent | `f"{0.5:.0%}"` -> `50%` |
| `Path.write_text(json.dumps(..., indent=2))` | save pretty JSON to file | – |

## Libraries & functions
| Name | Kind | Does | Returns |
|---|---|---|---|
| `re.findall(pattern, text)` | standard library | all matches of a regex | list[str] |
| `os.getenv("X", default)` | standard library | read an environment variable | str |
| `collections.defaultdict` | standard library | dict with default values | – |
| `json.dumps(obj, indent=2)` | standard library | Python -> JSON text | str |
| `datetime.now().isoformat()` | standard library | timestamp string | str |

## Execution trace
12-13 tests × 3 prompts = ~39 model calls at temperature 0. Each reply goes through `parse_label` (find allowed words; none or conflicting -> `invalid`) then compared with `expected`. Rows are grouped per category and printed. Output: the table + failing cases of the best version + a JSON file.

**What the mock proves:** the harness (fixed set, parser, grouping, saved results, repeatability). **What it does not:** that v3 is better for a real model — the numbers come from rules *I* wrote. Run with `LLM_BACKEND=local` or `anthropic` for evidence.

## Experiments
- Add 3 tests of your own (negation, mixed sentiment, other language).
- Create `v4` by changing exactly one thing from v3 and diff the JSON results.
- Real backend: run twice with `temperature=0.9` -> tables differ; that is why temperature 0 is used for testing.

## Debugging practice
`parse_label` returning `invalid` for a reply that "looks fine" -> print `repr(reply)`; a trailing period or extra words are the usual cause. Then decide: loosen the parser, or tighten the prompt.

## Rebuild checklist
- [ ] Test set with 4 categories; 2+ prompt versions; parser; per-category table; JSON save.

## The new problem (cliffhanger)
Your prompts are now tested and safe, and the model follows instructions well. But the tests only check *format and behaviour*. They cannot add a fact the model never saw: ask about your own refund policy and it confidently invents one. Next: [08_minimal_rag](../08_minimal_rag/).

> **Say it aloud (interview box):** I test a prompt with a fixed, categorised test set (normal, ambiguous, missing-input, adversarial) at temperature 0 and compare versions in an accuracy table. I change one thing per version and save the results so regressions are visible. A tiny or simulated test set only checks my harness, so I re-run on a real model before trusting the numbers.
