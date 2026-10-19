# Terms — Broken examples (debugging drills)

Folder: `08_agent_harness/broken_examples` · Files covered: `_common.py`, `broken_no_max_steps.py`, `broken_unvalidated_args.py`, `broken_bad_json.py`, `broken_stale_history.py`, `broken_path_escape.py`
Back to module: [08 Agent harness README](../README.md)

**Terms in this folder:** [Debugging drill](#debugging-drill) · [Runaway loop](#runaway-loop) · [while True](#while-true) · [Safety net (drill-only guard)](#safety-net-drill-only-guard) · [RuntimeError](#runtimeerror) · [Repetition detection](#repetition-detection) · [MockLLM with a catch-all rule](#mockllm-with-a-catch-all-rule) · [Unvalidated tool arguments](#unvalidated-tool-arguments) · [TypeError](#typeerror) · [Traceback reading order (last line first)](#traceback-reading-order-last-line-first) · [Keyword argument unpacking (**decision["args"])](#keyword-argument-unpacking-decisionargs) · [extract_json](#extract_json) · [json.loads on raw model text](#jsonloads-on-raw-model-text) · [JSONDecodeError](#jsondecodeerror) · [Repair loop with a cap](#repair-loop-with-a-cap) · [Stale history (missing append)](#stale-history-missing-append) · [for ... else](#for--else) · [Print debugging](#print-debugging) · [Path traversal](#path-traversal) · [Path joining with /](#path-joining-with) · [Path.read_text and splitlines slicing](#pathread_text-and-splitlines-slicing) · [Path(__file__).resolve().parents](#path__file__resolveparents) · [Shared helper module (_common.py)](#shared-helper-module-_commonpy) · [noqa comment](#noqa-comment) · [calculator tool (operator dict)](#calculator-tool-operator-dict) · [sys.path.insert](#syspathinsert) · [Fake agent LLM (make_fake_agent_llm, fake_reply)](#fake-agent-llm-make_fake_agent_llm-fake_reply) · [Silent failure vs crash](#silent-failure-vs-crash)

---

## Debugging drill
*Type: concept*

**1. Definition:** A small piece of code with a deliberate bug used to practise finding and fixing problems without asking an AI.
**2. In plain words:** Three of these crash, two do not. The ones that run without an error but give a wrong or unsafe result are harder, because there is no traceback to follow. Read the evidence, find the first wrong thing and write down the root cause.
**3. Example to quote:**
> "`broken_stale_history.py` never crashes, but the agent calls the same tool until the step limit; printing `messages` at the top of each loop showed the observation was never appended."

**Seen in:** all `broken_*.py` docstrings.

## Runaway loop
*Type: concept*

**1. Definition:** A loop with no effective stop condition, such as `while True` waiting for a `final` message the model never sends.
**2. In plain words:** A real agent in this state would call the model forever and burn money. The drill adds a safety net after 25 iterations so your laptop survives, but production code needs a real `max_steps` limit.
**3. Example to quote:**
> "The loop raised `RuntimeError: runaway loop detected after 26 iterations`; the fix is `for step in range(max_steps)` and a clear 'STOPPED' message."

**Seen in:** `broken_no_max_steps.py`.

## while True
*Type: Python syntax*

**1. Definition:** A loop whose condition is always true, ended only by `break`, `return` or an exception.
**2. In plain words:** Fine when a break is guaranteed, dangerous when it depends on an external system like a model behaving well.
**3. Example to quote:**
> "The only exit is `if 'final' in decision: break`, and the mock never sends one."

**Seen in:** `broken_no_max_steps.py`.

## Safety net (drill-only guard)
*Type: concept*

**1. Definition:** A temporary guard such as `if iterations > 25: raise RuntimeError(...)` that stops a runaway loop during practice.
**2. In plain words:** It is not a design. Real limits belong in the harness as `max_steps` and repetition detection.
**3. Example to quote:**
> "The safety net let me see the bug; the real fix is a bounded `for` loop."

**Seen in:** `broken_no_max_steps.py`.

## RuntimeError
*Type: Python built-in exception*

**1. Definition:** A general-purpose exception for errors that fit no specific category, raised deliberately here with an explanatory message.
**2. In plain words:** A descriptive message such as 'the harness has no max_steps' turns a mystery hang into a clear diagnosis.
**3. Example to quote:**
> "`raise RuntimeError(f'runaway loop detected after {iterations} iterations')` stops the drill with a clear reason."

**Seen in:** `broken_no_max_steps.py`.

## Repetition detection
*Type: concept*

**1. Definition:** Stopping early when the same tool call with the same arguments repeats several times.
**2. In plain words:** It catches a stuck agent faster than waiting for the step limit and lets you report why it stopped.
**3. Example to quote:**
> "If `('calculator', {'a': 1, 'op': '+', 'b': 1})` appears three times in a row, I stop and say the agent is repeating itself."

**Seen in:** `broken_no_max_steps.py`, FIX HINTS.

## MockLLM with a catch-all rule
*Type: project helper*

**1. Definition:** `MockLLM(rules=[(r"", reply)])` has an empty regex that matches every message, so it always returns the same reply.
**2. In plain words:** It simulates a model that keeps requesting the same tool.
**3. Example to quote:**
> "`always_same_tool` returns the calculator request for 1 + 1 every time, so the loop never reaches `final`."

**Seen in:** `broken_no_max_steps.py`.

## Unvalidated tool arguments
*Type: concept*

**1. Definition:** Calling a tool with whatever arguments the model produced, without checking names, types or allowed values.
**2. In plain words:** The model sent `"a": "6"` (a string). Python then fails inside the tool with `TypeError: can only concatenate str (not "int") to str`. With `*` it would not crash and would silently return `'6666666'`, which is worse.
**3. Example to quote:**
> "`'6' + 7` raises a TypeError, but `'6' * 7` quietly gives `'6666666'`; validation before the call prevents both."

**Seen in:** `broken_unvalidated_args.py`.

## TypeError
*Type: Python built-in exception*

**1. Definition:** The exception raised when an operation is applied to a value of the wrong type, such as adding a string and an integer.
**2. In plain words:** Read the last line of the traceback first (what happened), then go upward to see where it was called from. The bug is often earlier than the line that crashed.
**3. Example to quote:**
> "The last line says 'can only concatenate str (not \"int\") to str', so I look for where a string entered the calculation."

**Seen in:** `broken_unvalidated_args.py`.

## Traceback reading order (last line first)
*Type: concept*

**1. Definition:** The method of reading Python errors from the bottom: the final line names the exception and message, the lines above show the call chain that led to it.
**2. In plain words:** The bottom tells you what, the frames above tell you where. Start there before changing anything.
**3. Example to quote:**
> "I read the last line, then walked up through `calculator`, then the `print` call in the script."

**Seen in:** `broken_unvalidated_args.py` docstring.

## Keyword argument unpacking (`**decision["args"]`)
*Type: Python syntax*

**1. Definition:** `calculator(**d)` passes each item of `d` as a named argument.
**2. In plain words:** It makes the model's dict into function arguments with no checks, which is exactly why validation must come first.
**3. Example to quote:**
> "`calculator(**{'a': '6', 'op': '+', 'b': 7})` becomes `calculator(a='6', op='+', b=7)`."

**Seen in:** `broken_unvalidated_args.py`, `broken_no_max_steps.py`, `broken_stale_history.py`.

## extract_json
*Type: project function*

**1. Definition:** `extract_json(text)` finds the first `{...}` in a reply and parses it as JSON.
**2. In plain words:** In the unvalidated-args drill it parses the model's output fine; the bug is what happens after. It is also the fix for the bad-JSON drill.
**3. Example to quote:**
> "`extract_json(llm_output)` returned a dict with `tool` and `args`; the failure came later inside the calculator."

**Seen in:** `broken_unvalidated_args.py`, `_common.py`.

## json.loads on raw model text
*Type: Python standard library*

**1. Definition:** Parsing a model reply directly with `json.loads` and no handling raises `json.JSONDecodeError` when the text has chatter, single quotes or missing quotes.
**2. In plain words:** Models often write `Sure! Here is my action: {'tool': ...}`. JSON requires double quotes and nothing before the opening brace. The error message gives a line and column pointing at the first bad character.
**3. Example to quote:**
> "`json.loads(\"Sure! Here is my action: {...}\")` fails at line 1 column 1 because 'S' is not a valid start; the fix is `extract_json` inside `try/except` with a repair request."

**Seen in:** `broken_bad_json.py`.

## JSONDecodeError
*Type: Python standard library (json)*

**1. Definition:** The exception raised by `json.loads` for invalid JSON; it includes `lineno`, `colno` and a message such as 'Expecting value'.
**2. In plain words:** Use the line and column to locate the first bad character, then decide whether the problem is the model output or your parsing.
**3. Example to quote:**
> "'Expecting value: line 1 column 1 (char 0)' told me the text did not start with JSON at all."

**Seen in:** `broken_bad_json.py`.

## Repair loop with a cap
*Type: concept*

**1. Definition:** On a parse failure, send the error back to the model and ask again, up to a fixed number of repairs (step 06).
**2. In plain words:** It gives the model a chance to fix itself without allowing an endless loop.
**3. Example to quote:**
> "After two failed repairs the agent ends with a clear failure instead of retrying forever."

**Seen in:** `broken_bad_json.py`, docstring.

## Stale history (missing append)
*Type: concept*

**1. Definition:** A bug where the assistant turn and the tool Observation are never appended to `messages`, so every model call sees the same unchanged conversation.
**2. In plain words:** The model's only memory is the message list. With no new information it makes the same decision every step until the limit. There is no crash, so you must inspect the data.
**3. Example to quote:**
> "I printed `messages` at the top of each iteration and it never grew, which is why the agent repeated the same tool call."

**Seen in:** `broken_stale_history.py`.

## for ... else
*Type: Python syntax*

**1. Definition:** An `else` block attached to a `for` loop that runs only if the loop finished without hitting `break`.
**2. In plain words:** Here it prints 'STOPPED: max steps reached' when the loop used all its steps without the model giving a final answer.
**3. Example to quote:**
> "The `else:` after the loop runs only because the loop never reached `break`."

```python
for i in range(3):
    pass
else:
    print("no break")
```

**Seen in:** `broken_stale_history.py`.

## Print debugging
*Type: concept*

**1. Definition:** Inserting `print` statements to inspect values while the program runs.
**2. In plain words:** For wrong-but-not-crashing bugs, printing the key data (here `messages`) at the point of suspicion is the quickest way to see what the model actually received.
**3. Example to quote:**
> "`print(messages)` at the top of the loop showed one message every time."

**Seen in:** `broken_stale_history.py`, hint.

## Path traversal
*Type: concept (security)*

**1. Definition:** Using `..` segments or absolute paths in a user- or model-supplied path to reach files outside an intended folder.
**2. In plain words:** `SANDBOX / "../fake_llm.py"` points outside the sandbox. The drill reads a file from the parent folder successfully, which is the bug. It does not crash, so it is easy to miss.
**3. Example to quote:**
> "The drill printed the first lines of `fake_llm.py`, a file outside `sandbox_files`, because the tool trusted the path."

**Seen in:** `broken_path_escape.py`.

## Path joining with `/`
*Type: Python standard library (pathlib)*

**1. Definition:** `Path / "name"` builds a new path; `..` segments are kept as written until the path is resolved.
**2. In plain words:** Joining alone does not enforce any boundary. You must call `.resolve()` and check the result lies inside the allowed folder (step 07's `safe_path`).
**3. Example to quote:**
> "`(SANDBOX / path).read_text()` happily follows the `..` and reads the file outside."

**Seen in:** `broken_path_escape.py`.

## Path.read_text and splitlines slicing
*Type: Python standard library / built-in*

**1. Definition:** `read_text()` returns a file's full text; `.splitlines()` splits it into lines; `[:3]` keeps the first three.
**2. In plain words:** The drill prints only the first three lines of the escaped file as evidence.
**3. Example to quote:**
> "`read_file(attack_path).splitlines()[:3]` shows the start of the outside file."

**Seen in:** `broken_path_escape.py`.

## Path(__file__).resolve().parents
*Type: Python standard library*

**1. Definition:** Computes absolute locations relative to the current script, with `parents[1]` meaning the folder two levels above the script's file.
**2. In plain words:** It lets the drill find `sandbox_files` and the shared helpers regardless of the working directory.
**3. Example to quote:**
> "`Path(__file__).resolve().parents[1] / 'sandbox_files'` is the sandbox folder."

**Seen in:** `broken_path_escape.py`, `_common.py`.

## Shared helper module (`_common.py`)
*Type: project helper*

**1. Definition:** A small file that sets up import paths and defines `calculator`, so each drill stays short and focused on one bug.
**2. In plain words:** The leading underscore marks it as internal. The `# noqa: F401` comments tell linters that the imports are intentional even though they look unused here.
**3. Example to quote:**
> "`from _common import calculator, extract_json, MockLLM, generate` pulls in everything a drill needs."

**Seen in:** `_common.py`.

## noqa comment
*Type: Python tooling*

**1. Definition:** `# noqa: F401` tells a linter to ignore its "imported but unused" warning on that line.
**2. In plain words:** The names are re-exported for other files to import, so the warning is a false alarm.
**3. Example to quote:**
> "Without `# noqa`, my linter would flag the imports in `_common.py` as unused."

**Seen in:** `_common.py`.

## calculator tool (operator dict)
*Type: project function*

**1. Definition:** `calculator(a, op, b)` looks up the operator in a dict of lambdas and applies it, returning the result as a string.
**2. In plain words:** It is the same tool as in earlier steps. Without validation it accepts any value types, which the drills exploit.
**3. Example to quote:**
> "`calculator(a=1, op='+', b=1)` returns `'2'`."

**Seen in:** `_common.py`.

## sys.path.insert
*Type: Python standard library*

**1. Definition:** Adds a folder to Python's import search path so modules there can be imported.
**2. In plain words:** `_common.py` adds the shared and module folders so the drills can import `llm_client` and `fake_llm` from anywhere.
**3. Example to quote:**
> "`sys.path.insert(0, str(HERE.parents[1] / 'shared'))` makes `import llm_client` work."

**Seen in:** `_common.py`.

## Fake agent LLM (`make_fake_agent_llm`, `fake_reply`)
*Type: project helper*

**1. Definition:** Scripted stand-ins for a model that answer in the JSON tool protocol.
**2. In plain words:** The stale-history drill uses the fake model, which behaves correctly only when it can see the Observation; without it, it repeats the tool request.
**3. Example to quote:**
> "Because the Observation is never appended, the fake model asks for the calculator again every step."

**Seen in:** `broken_stale_history.py`, `_common.py`.

## Silent failure vs crash
*Type: concept*

**1. Definition:** A crash gives a traceback; a silent failure produces a wrong or unsafe result without any error.
**2. In plain words:** Silent failures such as the path escape and the stale history are more dangerous because nothing tells you something is wrong. Tests and traces are how you catch them.
**3. Example to quote:**
> "The path escape never raised an exception, so only a test that tried `../` would have caught it."

**Seen in:** `broken_stale_history.py`, `broken_path_escape.py`.
