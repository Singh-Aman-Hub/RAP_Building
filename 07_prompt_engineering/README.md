# 07 — Prompt Engineering (Syllabus Topic 4)

## The story of this module
**Pain point:** modules 05 and 06 gave you a model that is controlled, shrunk, tuned and fast, yet a raw model only *completes text*. Ask vaguely and you get rambling; ask for data and you get prose your code cannot read.
**Spark:** instruct it. Roles and examples shape the answer, templates make prompts reusable, a Pydantic schema makes the output enforceable, decomposition and injection defences make it robust, and a test set makes it measurable.
**What you build:** a tested sentiment prompt that returns validated JSON (steps 01-07), then a 20-line vector search that pastes private facts into the prompt (step 08, minimal RAG).
**Cliffhanger:** the model now follows instructions and answers from your documents, but it still only *talks*. It cannot run a calculator, read a file or call a service. Next module: [08_agent_harness](../08_agent_harness/).

## Why this matters
Interviewers use prompts as the "API" of an LLM. They want to see that you can (1) write a precise prompt, (2) make the output **machine-readable and validated**, (3) know the failure modes (ambiguity, injection), and (4) **test** prompts instead of guessing. Everything in module [08_agent_harness](../08_agent_harness/) depends on this: an agent's tool calls are just structured-output prompts.

## Where this fits
**Builds on**
- [04_llm_fundamentals](../04_llm_fundamentals/) — a prompt is turned into tokens ([01_tokenizer](../04_llm_fundamentals/01_tokenizer/)); the model predicts the next token. That is why *wording and examples change behaviour* and why prompts cost money (tokens).
- [05_llm_parameters](../05_llm_parameters/) — `temperature`, `max_new_tokens` and the context window are the knobs you set on every call here (we use temperature 0 for tests; max tokens limits the reply; the context window limits how many few-shot examples fit).
- [00_python_foundations](../00_python_foundations/) — f-strings, dicts, try/except and comprehensions are used in every step.

**Leads to**
- [08_agent_harness](../08_agent_harness/) — the next module: gives the model hands. It reuses the JSON-output + validate + retry pattern (step 04) for tool calls, and the injection defences (step 06) for tool safety.
- [09_api_development](../09_api_development/) — the same Pydantic model you write in step 04 becomes the request/response schema of a FastAPI endpoint.
- [10_ml_system_architecture](../10_ml_system_architecture/) — prompt versions are "model versions": log and monitor them.
- [13_capstone_project](../13_capstone_project/) — the capstone's LLM step uses a tested prompt.

## Concept map
```
   instruction design (role, task, constraints)  <-- 01
          |  system vs user split
          v
   examples? ---------- few-shot (02)
          |
          v
   prompt TEMPLATE (03) -- versions v1,v2,v3 ----------+
          |                                            |
          v                                            v
   output format: JSON + schema (04) --> validate --> retry     TEST SET (07):
          |                                              normal/ambiguous/missing/adversarial
          v                                            ^
   hard tasks: decompose / reasoning field / vote (05)  |
          |                                            |
          v                                            |
   untrusted input? injection defences (06) ------------+ (adversarial cases feed the tests)
```

## Study path
| # | Folder | You learn | Run | Time |
|---|---|---|---|---|
| 01 | [01_roles_and_instructions](01_roles_and_instructions/) | role/task/constraints, zero-shot, system vs user | `python main.py` | 45 min |
| 02 | [02_few_shot](02_few_shot/) | examples in text vs chat turns, biases | `python main.py` | 45 min |
| 03 | [03_prompt_templates](03_prompt_templates/) | f-string / `format` / `Template`, brace pitfalls | `python main.py` | 45 min |
| 04 | [04_structured_json](04_structured_json/) | JSON extraction, hand validator, Pydantic schema as the enforcer, real `ValidationError`, retry with the error | `python main.py` | 90 min |
| 05 | [05_decomposition](05_decomposition/) | multi-stage prompts, reasoning field, voting | `python main.py` | 60 min |
| 06 | [06_prompt_injection](06_prompt_injection/) | direct/indirect injection, layered defence | `python main.py` | 60 min |
| 07 | [07_prompt_testing](07_prompt_testing/) | test set, versions, accuracy table | `python main.py` | 90 min |
| 08 | [08_minimal_rag](08_minimal_rag/) | why frozen models hallucinate; NumPy vector search; inject top-k into the prompt; retrieval failure modes | `python main.py` | 60 min |

Each step has `main.py` + `WALKTHROUGH.md`. Do the exercises in [exercises.md](exercises.md) before moving on. Run commands from inside the step folder with the repo venv (`../../.venv/bin/python main.py`) or activate it first (`source .venv/bin/activate`).

### Important: what the mock backend is (and is not)
By default `LLM_BACKEND=mock`, and these examples use [`sim_llm.py`](sim_llm.py): a **rule-based simulation** that reacts to prompt features (labels listed? "only"? JSON? examples?). It lets you run everything offline and get identical output each time.
- **It proves:** your *harness* works (templates, parser, validators, retries, scoring, saved results).
- **It does not prove:** that a prompt is good for a real model. For that, run the same script with `LLM_BACKEND=local` (Qwen2.5-0.5B-Instruct, ~1 GB first download, small but real) or `LLM_BACKEND=anthropic` (set `ANTHROPIC_API_KEY`). Expect *different* tables — that difference is the lesson.

## Concepts, one by one

### Role, task and instruction design
A good prompt has: **role** (who the model is), **task** (what to do), **input** (the data, clearly separated), **constraints** (allowed values, length, tone) and **output format**. Be specific and closed ("one of positive/negative/neutral") rather than open ("what do you think?"). *Analogy:* briefing a new intern — vague brief, vague work. *Limitation:* longer prompts cost more tokens and can contradict themselves.

### Zero-shot vs few-shot
Zero-shot = instructions only. Few-shot = instructions + a few worked examples. Examples are the strongest way to show format and edge cases (sarcasm). *Trade-offs:* tokens on every call, label bias if examples are unbalanced, sensitivity to order.

### System vs user prompts
`system` = stable rules/role/format for the whole conversation; `user` = the changing input. Keeping them apart keeps your rules in one place and makes injection defences easier. Some APIs take system separately (Anthropic), others as a first message with role `system` (OpenAI-style, local chat templates). *Limitation:* system prompts are strong hints, not security boundaries.

### Structured output (JSON), constraints and schemas
Free text is hard to parse. Ask for JSON with named fields and allowed values, then **validate**: parse -> check types/ranges/allowed values -> retry or fail. Pydantic lets you declare the schema as a class; a hand-written validator makes clear what is being checked. *Limitation:* valid shape does not mean true content.

### Prompt templates
A template is a constant string with slots plus a function that fills it. Use `f"..."` for one-offs, `str.format` for reusable constants, `string.Template` when the text has lots of `{}` (JSON). Name versions `PROMPT_V1`, `PROMPT_V2`.

### Chain-of-thought alternatives and decomposition
For harder tasks: (a) **decompose** into stages whose outputs you validate, (b) request a **reasoning field before the answer**, (c) **self-consistency**: sample N answers and vote. *Trade-offs:* more calls = more latency/cost; errors compound across stages.

### Prompt injection awareness
Untrusted text can carry instructions ("ignore previous instructions..."), directly from users or indirectly via web pages/PDFs. Defences are layered: system rules, delimiters + "this is data", minimal tool permissions, output allow-lists, human confirmation, logging. *No prompt is a guarantee.*

### Prompt testing and evaluation
Prompts are code: a fixed test set (normal, ambiguous, missing-input, adversarial), temperature 0, a parser, per-category accuracy, results saved. Change one thing per version. *Limitation:* tiny test sets are noisy and only test what you thought of.

### Retrieval-augmented generation (minimal RAG)
A frozen model cannot know private or new facts, and it guesses instead of saying so. RAG embeds your documents, finds the top-k most similar to the question (cosine similarity), and pastes them into the prompt as context. *Limitation:* if retrieval misses or returns an irrelevant chunk, the model answers confidently from junk, so use a similarity threshold and allow "I don't know".

## Common mistakes & gotchas
- Trusting the reply format without parsing/validating.
- `str.format` with JSON braces -> `KeyError` (double the braces or use `Template`).
- Mixing instructions and untrusted data in one blob.
- Changing three things between prompt versions and not knowing which helped.
- Judging prompts by 2-3 hand-picked examples.
- Believing a passing mock test = a good real prompt.
- Pasting retrieved text into a prompt without a score threshold (a miss becomes a confident wrong answer).

## Self-check
- [ ] I can write a role + task + constraints prompt and split it into system/user.
- [ ] I can explain zero-shot vs few-shot and one risk of few-shot.
- [ ] I can render a template three ways and avoid the brace pitfall.
- [ ] I can get JSON, validate it by hand and with pydantic, and retry with feedback.
- [ ] I can explain decomposition, reasoning fields and self-consistency.
- [ ] I can demonstrate a prompt injection and list 3+ defences.
- [ ] I can build a test set with 4 categories and compare prompt versions.
- [ ] I can explain why a frozen model hallucinates and build a top-k retrieval + prompt-injection of context.
- [ ] I can say what the mock backend proves and what it does not.

## Interview questions
1. **What makes a prompt good?** Clear role/task, closed output format, constraints, delimited input, examples for hard cases. *Trade-off:* length = cost/latency.
2. **Zero-shot vs few-shot?** No examples vs a few worked examples. Few-shot improves format/edge cases but costs tokens and can bias labels.
3. **System vs user prompt?** System = persistent rules/role; user = this request's input. Not a security boundary.
4. **How do you get reliable JSON?** Specify the schema in the prompt, parse, validate (pydantic), retry with the error message, cap attempts, fail loudly. Some APIs also offer constrained/structured-output modes.
5. **Why temperature 0 for classification/testing?** Least randomness -> repeatable results. (Still not perfectly deterministic on all hosted models.)
6. **What is prompt injection and how do you mitigate it?** Untrusted text overriding instructions. Layers: separate/fence data, minimal permissions, output validation, human confirmation, adversarial tests.
7. **Chain-of-thought alternatives?** Decomposition, reasoning field, self-consistency voting. Cost: more tokens/calls.
8. **How do you evaluate a prompt?** Versioned prompt + fixed categorised test set + parser + metric per category; change one variable at a time; re-run on regression.
9. **A prompt works on 3 examples but fails in production — why?** Tiny, non-representative sample; no ambiguous/adversarial/missing cases; format not validated.
10. **`str.format` raises `KeyError: '"label"'` — why?** Unescaped JSON braces; double them or use `string.Template`.
11. **When is decomposition worse than one prompt?** Simple tasks: extra latency/cost and compounding errors.
12. **What can a mock/simulated LLM test and not test?** It tests harness code (parsing, validation, scoring) but not real model behaviour.
13. **Why do frozen models hallucinate about private facts, and how does RAG help?** Weights stop at training time and the model guesses the most typical answer. RAG retrieves the relevant documents by embedding similarity and puts them in the prompt. *Failure modes:* retrieval miss, irrelevant chunk; mitigate with a threshold and "I don't know".
14. **How do you enforce a JSON schema on model output?** Declare a Pydantic model, validate every reply, retry with the validation error (bounded), then fail loudly.
