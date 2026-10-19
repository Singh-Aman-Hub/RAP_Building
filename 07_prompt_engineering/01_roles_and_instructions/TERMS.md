# Terms — Step 01: Roles, tasks and instructions (zero-shot)

Folder: `07_prompt_engineering/01_roles_and_instructions` · Files covered: `main.py`, `WALKTHROUGH.md` (and the helper `../sim_llm.py` it imports)
Back to module: [07 Prompt engineering README](../README.md)

**Terms in this folder:** [Prompt](#prompt) · [Zero-shot prompting](#zero-shot-prompting) · [Role (system / user / assistant)](#role-system--user--assistant) · [System prompt](#system-prompt) · [User message](#user-message) · [messages list (chat message format)](#messages-list-chat-message-format) · [Task](#task) · [Output constraint](#output-constraint) · [Closed label set](#closed-label-set) · [Sentiment classification](#sentiment-classification) · [Temperature](#temperature) · [generate (shared helper)](#generate-shared-helper) · [Mock / simulated model (make_sim, sim_llm.py)](#mock--simulated-model-make_sim-sim_llmpy) · [LLM_BACKEND](#llm_backend) · [sys.path.insert](#syspathinsert) · [Path(__file__).resolve().parents[1]](#path__file__resolveparents1) · [f-string](#f-string) · [for loop with tuple unpacking](#for-loop-with-tuple-unpacking) · [Format spec ({name:45s}) and !r](#format-spec-name45s-and-r) · [HTTP 400 (invalid request)](#http-400-invalid-request)

---

## Prompt
*Type: concept*

**1. Definition:** The input text (plus any structured messages) sent to a language model that tells it what to do; the model's reply is generated as a continuation conditioned on it.
**2. In plain words:** A prompt is simply everything you hand the model before it starts answering. The model has no memory of your intent other than what is written there, so vague prompts give vague answers. In this step the same review is placed in three prompts of increasing precision, and the reply format changes each time.
**3. Example to quote:**
> "The model only knows what is in the prompt, so I put the task, the allowed labels and the output format in it. With the vague prompt I got a chatty sentence; with the precise one I got the single word `negative`."

**Seen in:** `main.py`, the `vague`, `task_only` and `precise` variables.

## Zero-shot prompting
*Type: concept*

**1. Definition:** Asking a model to perform a task using instructions only, with no worked examples of input and output included in the prompt.
**2. In plain words:** "Zero" refers to the number of examples you give. You rely on what the model already learned in training plus your written instructions. It is the cheapest style because every request stays short, but it can misjudge unusual cases (step 02 shows sarcasm failing zero-shot).
**3. Example to quote:**
> "Prompt C is zero-shot: it has a role, labels and a format rule but not a single example. It costs few tokens, but if the model misreads sarcasm I would move to few-shot."

**Seen in:** `main.py` docstring ("zero-shot prompting = no examples given").

## Role (system / user / assistant)
*Type: concept*

**1. Definition:** A field on each chat message (`"role"`) that tells the model who is speaking: `system` for standing rules, `user` for the human's input, `assistant` for earlier model replies.
**2. In plain words:** Chat models are trained on conversations where each line has a speaker label. Using the labels correctly lets the model treat your rules differently from the changing input. APIs reject role names they do not know, such as `"robot"`.
**3. Example to quote:**
> "Every message dict has a role. I use `user` for the review text, and the rules go in `system`. If I typed `\"role\": \"robot\"` the API would return a 400 error."

```python
{"role": "user", "content": "Text: I love it"}
```

**Seen in:** `main.py`, every message dict; BREAK IT comment.

## System prompt
*Type: concept*

**1. Definition:** A message (or separate API field) that sets persistent instructions for the whole conversation, such as the model's role, allowed outputs and format.
**2. In plain words:** Think of it as the job description you give a new employee before they see any customer. It stays the same for every request, so you write it once and reuse it. Keep changing data out of it; that belongs in the user message.
**3. Example to quote:**
> "My `SYSTEM_C` says 'You are a sentiment classifier... Reply with the label only.' It is identical for every review, so each request only carries the new text."

**Seen in:** `main.py`, the `SYSTEM_C` constant.

## User message
*Type: concept*

**1. Definition:** A chat message with role `user` carrying the input specific to this request.
**2. In plain words:** This is the part that changes each call, like the actual customer review. Separating it from the system prompt keeps the stable rules clean and makes testing easier because you swap only this part.
**3. Example to quote:**
> "In prompt C the user message is just `Text: The delivery was slow and the box was broken.` Everything else lives in the system prompt."

**Seen in:** `main.py`, the `precise` list.

## messages list (chat message format)
*Type: concept*

**1. Definition:** A Python list of dictionaries, each with `role` and `content` keys, in conversation order; the standard input shape for chat-completion APIs.
**2. In plain words:** Chat APIs do not take one big string; they take a list of turns. The order matters because the model reads it top to bottom. You build the list in your code and pass it to the helper function.
**3. Example to quote:**
> "I build `messages = [{\"role\": \"user\", \"content\": f\"Text: {review}\"}]` and pass it to `generate`. For a longer chat I would append assistant and user dicts to the same list."

**Seen in:** `main.py`, `vague`, `task_only`, `precise`.

## Task
*Type: concept*

**1. Definition:** The specific job you ask the model to do (for example "classify the sentiment"), stated as the core instruction of a prompt.
**2. In plain words:** The task is the verb of the prompt. Prompt B states it ("Classify the sentiment as positive, negative or neutral") but nothing else, so the model does the job but formats the answer however it likes.
**3. Example to quote:**
> "A prompt with a task but no output constraint still works, but I got 'The sentiment of the text is negative.' instead of a clean label."

**Seen in:** `main.py`, `task_only`.

## Output constraint
*Type: concept*

**1. Definition:** An instruction that limits the form of the model's reply, such as "label only, lower case, nothing else".
**2. In plain words:** Models love to add polite extra words. A constraint tells them to stop, which matters because your code has to parse the reply. Constraints reduce but do not eliminate surprises, so you still validate.
**3. Example to quote:**
> "Adding 'Reply with the label only, in lower case, nothing else' turned the sentence into the bare word `negative`, which I can compare with `==`."

**Seen in:** `main.py`, last line of `SYSTEM_C`.

## Closed label set
*Type: concept*

**1. Definition:** A fixed, explicitly listed collection of allowed answers (here `positive`, `negative`, `neutral`) the model must choose from.
**2. In plain words:** Instead of letting the model invent categories like "mixed feelings", you give it a menu. This makes results comparable and easy to count or test.
**3. Example to quote:**
> "My system prompt says 'Allowed labels: positive, negative, neutral', so downstream code can use a simple `if label == 'negative'`."

**Seen in:** `main.py`, `SYSTEM_C`.

## Sentiment classification
*Type: concept*

**1. Definition:** The task of assigning a piece of text an emotional polarity label such as positive, negative or neutral.
**2. In plain words:** It is a common first NLP task: "Is this review happy or unhappy?" It is used here only as a small, easy-to-check example for prompt wording.
**3. Example to quote:**
> "'The delivery was slow and the box was broken' should be classified negative; I use that review to compare three prompts."

**Seen in:** `main.py`, the `review` variable.

## Temperature
*Type: concept (model parameter)*

**1. Definition:** A sampling parameter that rescales the model's output probabilities before choosing a token; 0 means always pick the most likely token, higher values add randomness.
**2. In plain words:** Low temperature makes the model predictable, high temperature makes it creative. For classification and for tests you want the same answer every time, so you use 0.0. The details are in module 05.
**3. Example to quote:**
> "I pass `temperature=0.0` for classification so my tests are repeatable; if I wanted varied marketing copy I would raise it."

```python
reply = generate(messages, system=system, temperature=0.0, mock=SIM)
```

**Seen in:** `main.py`, the `generate(...)` call.

## generate (shared helper)
*Type: project function*

**1. Definition:** `shared/llm_client.py::generate(messages, system, max_tokens, temperature, backend, mock)`, a function that sends chat messages to the selected backend (mock, local or Anthropic) and returns the reply text.
**2. In plain words:** It hides which model you use behind one function, so the same script runs offline with a fake model or against a real one by changing an environment variable. It takes messages in and gives a string out.
**3. Example to quote:**
> "I call `generate(messages, system=SYSTEM_C, temperature=0.0, mock=SIM)` and get back a string. Switching to a real model only needs `LLM_BACKEND=anthropic`."

**Seen in:** `main.py`, the loop at the bottom.

## Mock / simulated model (`make_sim`, `sim_llm.py`)
*Type: project helper*

**1. Definition:** A rule-based Python stand-in for a language model (`make_sim()` returns a `MockLLM` whose rules call `sim_reply`) that reacts deterministically to prompt wording; it is not a real LLM.
**2. In plain words:** It lets you practise prompt testing without an API key or GPU. It proves your test code works, but it does not prove a prompt is good for a real model, so important results should be repeated with `LLM_BACKEND=local` or `anthropic`.
**3. Example to quote:**
> "The default backend here is a simulation, so I treat the output as a demonstration of the harness, not evidence that my wording is best for a real model."

**Seen in:** `main.py`, `SIM = make_sim()`.

## LLM_BACKEND
*Type: environment variable*

**1. Definition:** An environment variable read by `shared/llm_client.py` to choose the backend: `mock` (default), `local` or `anthropic`.
**2. In plain words:** It is a switch outside your code. Setting it changes which model answers, without editing the script. If it is not set, the offline mock is used.
**3. Example to quote:**
> "I run `LLM_BACKEND=local python main.py` to try the same prompts on a small open model."

```bash
LLM_BACKEND=anthropic python main.py
```

**Seen in:** `main.py` docstring (`NEEDS` line).

## sys.path.insert
*Type: Python standard library*

**1. Definition:** `sys.path` is the list of folders Python searches when importing; `list.insert(0, folder)` puts a folder first so modules inside it can be imported.
**2. In plain words:** The helper files live in a different folder than the script. Adding that folder to the search list lets `import sim_llm` work no matter where you start the script from.
**3. Example to quote:**
> "The script adds the parent folder to `sys.path` so it can import `sim_llm`, even if I run it from the repo root."

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

**Seen in:** `main.py`, import setup.

## Path(__file__).resolve().parents[1]
*Type: Python standard library (pathlib)*

**1. Definition:** `__file__` is the path of the current script; `Path(...).resolve()` makes it absolute; `.parents[1]` is the folder two levels up the chain counting from the file's directory as index 0.
**2. In plain words:** It computes "the folder above the folder this file is in" without hard-coding your username or location. That keeps the script working on any computer and from any current working directory.
**3. Example to quote:**
> "`Path(__file__).resolve().parents[1]` gives me `07_prompt_engineering`, where `sim_llm.py` lives, regardless of where I launched Python."

**Seen in:** `main.py`, import setup.

## f-string
*Type: Python syntax*

**1. Definition:** A string literal prefixed with `f` whose `{expression}` parts are evaluated and inserted at runtime.
**2. In plain words:** It is the easiest way to put variables inside text. Here the review is dropped into each prompt. Remember the values are inserted immediately when the line runs.
**3. Example to quote:**
> "I use `f\"Text: {review}\"` so each prompt contains the current review."

```python
name = "Asha"; print(f"hi {name}")
```

**Seen in:** `main.py`, prompt strings and the final `print`.

## for loop with tuple unpacking
*Type: Python syntax*

**1. Definition:** A `for` loop whose target is several names, e.g. `for name, messages, system in [...]`, which unpacks each tuple element into separate variables.
**2. In plain words:** The list holds triples, and the loop splits each triple into three named variables. It lets you run the same code on several prompt variants without repeating it.
**3. Example to quote:**
> "I loop over three tuples, each holding a label, a messages list and a system prompt, and call `generate` for each."

**Seen in:** `main.py`, the final `for` loop.

## Format spec (`{name:45s}`) and `!r`
*Type: Python syntax*

**1. Definition:** Inside an f-string, `:45s` pads a string to width 45, and `!r` formats the value with `repr()` so quotes and escape characters are visible.
**2. In plain words:** The padding lines the output up in neat columns; `!r` shows exactly what the reply contains, including trailing spaces or newlines that would otherwise be invisible when debugging a model reply.
**3. Example to quote:**
> "I print the reply with `!r` so I can see if the model added a stray newline around the label."

```python
print(f"{'ab':5s}|{'a'!r}")   # ab   |'a'
```

**Seen in:** `main.py`, the `print` line.

## HTTP 400 (invalid request)
*Type: concept (error)*

**1. Definition:** An HTTP status code meaning the server rejected the request as malformed; model APIs return it for things like an unknown message role.
**2. In plain words:** 4xx errors mean "your request is wrong", not "the server broke". The error body usually names the offending field, such as `messages.0.role`, so read that first.
**3. Example to quote:**
> "When I used the role `robot`, the API answered 400 and pointed at `messages.0.role`; the fix is to use `user`, `assistant` or a system field."

**Seen in:** `main.py` BREAK IT comment; `WALKTHROUGH.md` debugging practice.
