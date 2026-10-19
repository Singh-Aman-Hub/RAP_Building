# Terms — Step 02: Few-shot prompting

Folder: `07_prompt_engineering/02_few_shot` · Files covered: `main.py`, `WALKTHROUGH.md` (helper `../sim_llm.py`)
Back to module: [07 Prompt engineering README](../README.md)

**Terms in this folder:** [Few-shot prompting](#few-shot-prompting) · [Zero-shot (comparison baseline)](#zero-shot-comparison-baseline) · [Sarcasm](#sarcasm) · [Worked example (shot)](#worked-example-shot) · [Label bias](#label-bias) · [Examples in one message vs chat turns](#examples-in-one-message-vs-chat-turns) · [List of tuples](#list-of-tuples) · [str.join](#strjoin) · [Generator expression inside join](#generator-expression-inside-join) · [Tuple unpacking in a for loop](#tuple-unpacking-in-a-for-loop) · [list.append](#listappend) · [Assistant role message](#assistant-role-message) · [Token cost of examples](#token-cost-of-examples) · [Simulation limit](#simulation-limit) · [generate with mock](#generate-with-mock)

---

## Few-shot prompting
*Type: concept*

**1. Definition:** Including a small number of worked input-output examples inside the prompt so the model imitates the demonstrated pattern without any weight updates.
**2. In plain words:** Instead of only explaining the task, you show it. The model sees "this text gets this label" a few times and copies the style and decisions. It is quick to set up, but every example is paid for in tokens on every request.
**3. Example to quote:**
> "I added three examples, one per label, including a sarcastic one. The zero-shot prompt called 'Oh great, another delay' positive, but with examples the answer became negative."

**Seen in:** `main.py`, `EXAMPLES` and the two few-shot variants.

## Zero-shot (comparison baseline)
*Type: concept*

**1. Definition:** A prompt that gives instructions but no examples; used here as the baseline against which few-shot is compared.
**2. In plain words:** Always test the simple version first. If zero-shot is good enough, you save tokens and complexity; if it fails on hard cases, add examples.
**3. Example to quote:**
> "Zero-shot returned 'positive' for the sarcastic review, which is wrong; that failure is my evidence that I need examples."

**Seen in:** `main.py`, the `zero_shot` list.

## Sarcasm
*Type: concept*

**1. Definition:** Language whose literal wording says the opposite of its intended meaning, e.g. "Oh great, another delay."
**2. In plain words:** Word-counting methods see "great" and think positive. Real understanding needs context. Sarcasm is a classic hard case used to show why examples help.
**3. Example to quote:**
> "'Just what I needed' after a delay is sarcasm; the surface words are positive but the label should be negative."

**Seen in:** `main.py`, the `tricky` variable.

## Worked example (shot)
*Type: concept*

**1. Definition:** One input paired with its correct output, placed in the prompt as a demonstration; each such pair is called a "shot".
**2. In plain words:** A shot is a mini lesson. Good shots are realistic, correct and cover the tricky cases; bad shots teach the wrong thing.
**3. Example to quote:**
> "`('Oh great, the app crashed again.', 'negative')` is my sarcasm shot; it teaches the model that polite words can still mean negative."

**Seen in:** `main.py`, the `EXAMPLES` list.

## Label bias
*Type: concept*

**1. Definition:** A tendency of a model to favour a label because it is over-represented, or appears last or most often, among the examples in the prompt.
**2. In plain words:** If all your examples say "positive", the model starts leaning positive on everything. Balance the labels and check the order.
**3. Example to quote:**
> "If I made all three examples positive, I'd expect the model to over-predict positive, so I keep one example per label."

**Seen in:** `main.py` TRY IT comment; `WALKTHROUGH.md` experiments.

## Examples in one message vs chat turns
*Type: concept*

**1. Definition:** Two ways to supply shots: writing them as text inside a single user message, or as alternating `user`/`assistant` messages that look like earlier conversation turns.
**2. In plain words:** The second style shows the model a conversation in which it "already answered correctly", which many chat models follow closely. The conversation must still end with the real user question.
**3. Example to quote:**
> "I tried both styles and both gave 'negative'. With chat turns I append a user dict and an assistant dict per example, then end with the real question."

**Seen in:** `main.py`, `few_shot_one_message` and `few_shot_turns`.

## List of tuples
*Type: Python syntax*

**1. Definition:** A list whose items are tuples, here `[("text", "label"), ...]`, an ordered collection of fixed-size pairs.
**2. In plain words:** Each tuple keeps a text and its label together, and the list keeps their order. You can loop over it and unpack both parts at once.
**3. Example to quote:**
> "`EXAMPLES` is a list of three (text, label) tuples, so adding an example is one extra line."

```python
pairs = [("good", "positive"), ("bad", "negative")]
```

**Seen in:** `main.py`, `EXAMPLES`.

## str.join
*Type: Python built-in*

**1. Definition:** `separator.join(iterable)` concatenates the strings from an iterable with the separator between them and returns one string.
**2. In plain words:** It glues many pieces into one block. Here it turns three formatted examples into one text block with blank lines between them.
**3. Example to quote:**
> "`\"\\n\\n\".join(...)` joins the three example strings with a blank line between each."

```python
", ".join(str(i) for i in range(3))   # '0, 1, 2'
```

**Seen in:** `main.py`, `shots_text`.

## Generator expression inside join
*Type: Python syntax*

**1. Definition:** An expression like `f"..." for t, lbl in EXAMPLES` without square brackets that produces items one at a time; passed to `join` it builds the final string without making a list first.
**2. In plain words:** It is a compact loop written inside a function call. It reads "for each example, make this string". The file's comment calls it a comprehension, which is the same idea.
**3. Example to quote:**
> "Instead of a loop and append, I wrote the formatting loop inside `join`, so building the example block is a single expression."

**Seen in:** `main.py`, `shots_text = "\n\n".join(f"..." for t, lbl in EXAMPLES)`.

## Tuple unpacking in a for loop
*Type: Python syntax*

**1. Definition:** Assigning the elements of each tuple to separate names in a loop header, e.g. `for text, label in EXAMPLES:`.
**2. In plain words:** Rather than indexing `pair[0]` and `pair[1]`, you name the two parts. It makes the code self-explanatory.
**3. Example to quote:**
> "`for text, label in EXAMPLES` gives me both pieces to append as a user turn and an assistant turn."

**Seen in:** `main.py`, building `few_shot_turns`.

## list.append
*Type: Python built-in*

**1. Definition:** A list method that adds one item to the end of the list in place and returns `None`.
**2. In plain words:** It is how you grow the conversation turn by turn. Because it changes the list in place, you do not reassign the result.
**3. Example to quote:**
> "I append a user dict and an assistant dict for each example, then one final user dict with the real text."

```python
turns = []; turns.append({"role": "user", "content": "hi"})
```

**Seen in:** `main.py`, `few_shot_turns.append(...)`.

## Assistant role message
*Type: concept*

**1. Definition:** A chat message with role `assistant` representing something the model said earlier in the conversation.
**2. In plain words:** In few-shot chat turns you write these yourself to show the model its own ideal answers. Do not end the list with one: many APIs expect the last message to be from the user.
**3. Example to quote:**
> "My assistant turns contain only the label, like `negative`, so the model learns to reply with just the label."

**Seen in:** `main.py`, `few_shot_turns`; BREAK IT comment.

## Token cost of examples
*Type: concept (trade-off)*

**1. Definition:** Every example in a prompt adds tokens that are billed, processed and counted against the context window on every request.
**2. In plain words:** Few-shot is not free. Ten long examples sent with each of a million requests become real money and latency. Use as few good examples as achieve the accuracy you need.
**3. Example to quote:**
> "Few-shot improves accuracy but adds tokens to every call, so I measure whether each extra example is worth it."

**Seen in:** `main.py`, trade-off comment near the end.

## Simulation limit
*Type: concept*

**1. Definition:** The mock model in `sim_llm.py` treats a prompt with two or more `Text:` markers as "has examples" and unlocks sarcasm handling; it counts markers rather than learning from the examples.
**2. In plain words:** That means deleting the sarcasm example does not change the simulated result, but it would change a real model's answer. This is why you must re-run on a real backend to judge example quality.
**3. Example to quote:**
> "The simulation only counts examples, so I rely on `LLM_BACKEND=local` to see whether the sarcasm shot really matters."

**Seen in:** `main.py` TRY IT comment; `WALKTHROUGH.md` experiments.

## generate with mock
*Type: project function*

**1. Definition:** `generate(messages, system=..., mock=SIM)` from `shared/llm_client.py`; the `mock` argument supplies the simulated model used when the backend is `mock`.
**2. In plain words:** The same call works for all backends. The `mock` argument is ignored by real backends, so you can switch without editing code.
**3. Example to quote:**
> "I loop over three prompt variants and call `generate(msgs, system=SYSTEM, mock=SIM)` to compare their replies."

**Seen in:** `main.py`, the final loop.
