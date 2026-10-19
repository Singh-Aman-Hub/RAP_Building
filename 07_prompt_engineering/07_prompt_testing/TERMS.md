# Terms — Step 07: Prompt testing and evaluation

Folder: `07_prompt_engineering/07_prompt_testing` · Files covered: `main.py`, `WALKTHROUGH.md`, `results_mock.json` (helper `../sim_llm.py`)
Back to module: [07 Prompt engineering README](../README.md)

**Terms in this folder:** [Prompt testing (evaluation)](#prompt-testing-evaluation) · [Test set](#test-set) · [Test categories (normal, ambiguous, missing, adversarial)](#test-categories-normal-ambiguous-missing-adversarial) · [Negation (known weak spot)](#negation-known-weak-spot) · [Prompt version](#prompt-version) · [Output parser (parse_label)](#output-parser-parse_label) · [Regular expression (re.findall)](#regular-expression-refindall) · [List comprehension with filter](#list-comprehension-with-filter) · [Runner function (run_version)](#runner-function-run_version) · [Dictionary comprehension](#dictionary-comprehension) · [Temperature 0 for testing](#temperature-0-for-testing) · [collections.defaultdict](#collectionsdefaultdict) · [Accuracy](#accuracy) · [Summing booleans](#summing-booleans) · [max with key and lambda](#max-with-key-and-lambda) · [Format spec (alignment and percent)](#format-spec-alignment-and-percent) · [os.getenv](#osgetenv) · [pathlib.Path (write_text, / operator)](#pathlibpath-write_text--operator) · [json.dumps](#jsondumps) · [datetime.now().isoformat](#datetimenowisoformat) · [Mock proves the harness, not the prompt](#mock-proves-the-harness-not-the-prompt) · [JSON results file (results_mock.json)](#json-results-file-results_mockjson)

---

## Prompt testing (evaluation)
*Type: concept*

**1. Definition:** Measuring how well a prompt performs by running it against a fixed set of inputs with known expected outputs and scoring the results.
**2. In plain words:** Instead of trying a prompt on two examples and saying "looks good", you treat it like code with tests. You change one thing, rerun the same set, and see if the score moved. A small set is noisy and only measures what you thought to include.
**3. Example to quote:**
> "I have 13 test cases and three prompt versions, and one command prints an accuracy table, so I can show v3 beats v1 on this set rather than argue about it."

**Seen in:** `main.py`, whole file.

## Test set
*Type: concept*

**1. Definition:** A fixed collection of test cases, each a category, an input and an expected output, kept stable so results are comparable between runs.
**2. In plain words:** The test set is your ruler. If you keep changing it, you cannot tell whether a prompt improved. Keep it small, versioned and diverse.
**3. Example to quote:**
> "`TESTS` is a list of (category, text, expected) rows, for example ('missing', '', 'unknown')."

**Seen in:** `main.py`, `TESTS`.

## Test categories (normal, ambiguous, missing, adversarial)
*Type: concept*

**1. Definition:** Groups of test cases that probe different risks: typical inputs, unclear or tricky inputs, absent inputs and hostile inputs.
**2. In plain words:** An overall percentage hides weak spots. Per-category scores show that, for instance, v1 gets normal cases right but fails every adversarial one.
**3. Example to quote:**
> "v2 scored 4/4 on normal but 0/2 on adversarial, so the problem is injection, not basic classification."

**Seen in:** `main.py`, `categories` and the report table.

## Negation (known weak spot)
*Type: concept*

**1. Definition:** Language that reverses meaning with words like "not", e.g. "It was not bad at all", which means positive despite containing the word "bad".
**2. In plain words:** Keyword-based logic and sometimes models misread it. The best prompt still fails this case, a reminder that tests reveal limits.
**3. Example to quote:**
> "'It was not bad at all' expected positive but got negative, the one failure of my best prompt."

**Seen in:** `main.py`, ambiguous tests; `WALKTHROUGH.md` failures.

## Prompt version
*Type: concept*

**1. Definition:** A named revision of a prompt, here stored as an entry in the `PROMPTS` dict (`v1_vague`, `v2_labels_format`, `v3_full`) with its system prompt and user template.
**2. In plain words:** Change exactly one thing per version so you know what caused any improvement. Keeping versions lets you compare and roll back.
**3. Example to quote:**
> "v2 adds labels and a format rule to v1; v3 adds an empty-input rule, an untrusted-data warning and one example."

**Seen in:** `main.py`, `PROMPTS`.

## Output parser (`parse_label`)
*Type: project function*

**1. Definition:** A function that converts a free-text reply into one allowed label or the value `invalid`, by finding allowed words in it.
**2. In plain words:** Real models add words around the label. The parser extracts the label and treats none, or conflicting labels, as invalid, so scoring is fair and strict.
**3. Example to quote:**
> "'The sentiment of the text is negative.' parses to `negative`, while a reply mentioning both positive and negative parses to `invalid`."

**Seen in:** `main.py`, `parse_label`.

## Regular expression (`re.findall`)
*Type: Python standard library*

**1. Definition:** `re.findall(pattern, text)` returns all non-overlapping matches of a regular expression pattern as a list of strings.
**2. In plain words:** The pattern `[a-z]+` means "runs of lower-case letters", so it splits a reply into words and ignores punctuation. Regexes are compact but easy to get wrong, so test them.
**3. Example to quote:**
> "`re.findall(r\"[a-z]+\", reply.lower())` turns 'The sentiment is negative.' into a list of words."

```python
import re
re.findall(r"[a-z]+", "hi, you!")   # ['hi', 'you']
```

**Seen in:** `main.py`, `parse_label`.

## List comprehension with filter
*Type: Python syntax*

**1. Definition:** `[w for w in words if w in ALLOWED]` builds a list of only those items that satisfy the condition.
**2. In plain words:** It reads as "keep each word that is an allowed label". It filters and builds in one line.
**3. Example to quote:**
> "`found = [w for w in words if w in ALLOWED]` keeps only label words from the reply."

**Seen in:** `main.py`, `parse_label`.

## Runner function (`run_version`)
*Type: project function*

**1. Definition:** A function that runs one prompt version over every test case, calls the model, parses each reply and records a result row.
**2. In plain words:** It keeps the loop in one place, so adding a new prompt version needs no new code. Each row stores the input, expected value, raw reply, parsed answer and whether it was correct.
**3. Example to quote:**
> "`run_version('v3_full')` returns 13 rows, each with a `correct` flag I can sum into accuracy."

**Seen in:** `main.py`, `run_version`.

## Dictionary comprehension
*Type: Python syntax*

**1. Definition:** `{key: value for item in iterable}` builds a dict by evaluating an expression for each item.
**2. In plain words:** Here it runs every prompt version and stores the rows under the version's name, in one line.
**3. Example to quote:**
> "`all_results = {name: run_version(name) for name in PROMPTS}` runs all three versions and keys each result by its name."

```python
{x: x * 2 for x in [1, 2]}   # {1: 2, 2: 4}
```

**Seen in:** `main.py`, `all_results`.

## Temperature 0 for testing
*Type: concept*

**1. Definition:** Using `temperature=0.0` so the model picks the most likely token each time, making outputs (nearly) repeatable.
**2. In plain words:** If outputs vary between runs, you cannot tell whether a score change came from your prompt or from randomness. With real backends at higher temperature, two runs give different tables.
**3. Example to quote:**
> "I test at temperature zero so a change in the table means my prompt changed, not luck."

**Seen in:** `main.py`, `run_version`.

## collections.defaultdict
*Type: Python standard library*

**1. Definition:** A dictionary subclass that creates a default value (here an empty list) automatically when a missing key is accessed.
**2. In plain words:** It saves the "if key not in dict" check when grouping, such as collecting correct/incorrect flags per category.
**3. Example to quote:**
> "`per_cat = defaultdict(list)` lets me do `per_cat[r['category']].append(r['correct'])` without initialising each key."

```python
from collections import defaultdict
d = defaultdict(list); d["a"].append(1)
```

**Seen in:** `main.py`, report section.

## Accuracy
*Type: concept (metric)*

**1. Definition:** The fraction of test cases answered correctly, correct divided by total.
**2. In plain words:** The simplest score. Here it is printed per category and overall. With only 13 cases each case moves it by about 8 percentage points, so small differences are not reliable.
**3. Example to quote:**
> "v3's overall accuracy is 92%, which means 12 of 13 cases passed."

**Seen in:** `main.py`, `overall`.

## Summing booleans
*Type: Python syntax*

**1. Definition:** In Python `True` equals 1 and `False` equals 0, so `sum()` of a list of booleans counts the true values.
**2. In plain words:** It is a neat way to count correct answers.
**3. Example to quote:**
> "`sum(r['correct'] for r in rows)` counts how many rows were right."

```python
sum([True, False, True])   # 2
```

**Seen in:** `main.py`, `overall` and `best`.

## max with key and lambda
*Type: Python built-in / syntax*

**1. Definition:** `max(iterable, key=func)` returns the item for which `func(item)` is largest; `lambda n: ...` defines a tiny unnamed function inline.
**2. In plain words:** It picks the best prompt version by its score without writing a separate named function.
**3. Example to quote:**
> "`max(all_results, key=lambda n: sum(...))` returns the name of the version with the most correct answers."

```python
max([1, 5, 3], key=lambda x: -x)   # 1
```

**Seen in:** `main.py`, `best = max(...)`.

## Format spec (alignment and percent)
*Type: Python syntax*

**1. Definition:** Inside f-strings, `:>13s` right-aligns in width 13, `.rjust(13)` does the same on a string, and `:>9.0%` formats a fraction as a right-aligned percentage with no decimals.
**2. In plain words:** It keeps the report table columns tidy and turns 0.92 into `92%`.
**3. Example to quote:**
> "`f\"{overall:>9.0%}\"` prints 0.923 as ` 92%` in the last column."

```python
f"{0.5:.0%}"   # '50%'
```

**Seen in:** `main.py`, report printing.

## os.getenv
*Type: Python standard library*

**1. Definition:** `os.getenv(name, default)` returns the value of an environment variable or the default if it is not set.
**2. In plain words:** It reads `LLM_BACKEND` so the script reports which backend produced the results, with `mock` as the fallback.
**3. Example to quote:**
> "`os.getenv('LLM_BACKEND', 'mock')` tells me whether this run used the simulation or a real model."

**Seen in:** `main.py`, `BACKEND`.

## pathlib.Path (`write_text`, `/` operator)
*Type: Python standard library*

**1. Definition:** `Path` objects represent file paths; `Path / "name"` joins paths and `write_text(text)` writes a string to the file.
**2. In plain words:** It saves the results file next to the script, whatever your current folder is. `HERE = Path(__file__).resolve().parent` is the script's folder.
**3. Example to quote:**
> "`out = HERE / f'results_{BACKEND}.json'` then `out.write_text(...)` saves the results beside the script."

**Seen in:** `main.py`, `HERE`, `out`.

## json.dumps
*Type: Python standard library*

**1. Definition:** `json.dumps(obj, indent=2)` converts a Python object to a JSON string, pretty-printed with the given indent.
**2. In plain words:** It writes the results in a format you can open, diff after the next prompt change, or load later.
**3. Example to quote:**
> "I dump `all_results` with `indent=2` into `results_mock.json`, so I can compare two runs line by line."

**Seen in:** `main.py`, save section; `results_mock.json`.

## datetime.now().isoformat
*Type: Python standard library*

**1. Definition:** `datetime.now()` returns the current date-time; `.isoformat(timespec="seconds")` formats it as `YYYY-MM-DDTHH:MM:SS`.
**2. In plain words:** A timestamp in the results file tells you when the run happened and keeps files comparable.
**3. Example to quote:**
> "The JSON starts with `\"time\": \"2026-10-19T11:14:00\"` so I know which run it was."

**Seen in:** `main.py`, save section.

## Mock proves the harness, not the prompt
*Type: concept*

**1. Definition:** On the mock backend the model behaviour is written by the lesson's author, so the scores demonstrate the testing code works but are not evidence about a real model.
**2. In plain words:** The harness (fixed set, parser, scoring, saved results) is real and reusable. The numbers are not. Re-run with `LLM_BACKEND=local` or `anthropic` for real evidence.
**3. Example to quote:**
> "The table shows my harness works; to claim v3 is better I must run it against a real model."

**Seen in:** `main.py`, closing comments; `WALKTHROUGH.md`.

## JSON results file (`results_mock.json`)
*Type: data file*

**1. Definition:** A saved record of every test row per prompt version, with a timestamp and backend name, written by the script.
**2. In plain words:** After you change a prompt you can diff it against the earlier file to see exactly which cases changed.
**3. Example to quote:**
> "Each row has `category`, `text`, `expected`, `reply`, `got` and `correct`, so I can search for every failing case."

**Seen in:** `results_mock.json`.
