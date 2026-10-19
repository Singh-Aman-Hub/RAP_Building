# Step 03 - Compare with Hugging Face `generate()`

## The story so far
[Step 02](../02_sampling_from_scratch/WALKTHROUGH.md) gave you a hand-written sampler. It looks plausible, but plausible is not proof, and in production you will call `model.generate()` rather than your own loop.

## The pain point
A home-made sampler can hide bugs: a wrong order of operations or an off-by-one in top-p still produces fluent-looking text. You also need to know that `generate()` is not magic before you trust it.

## The spark
Feed the same logits through your functions and through the library's `LogitsWarper` classes and compare the tensors. If they are equal, `generate()` is just your maths. Then change one argument at a time in `generate()` and watch the output change.

## Goal
Prove your sampler matches the library's, then drive `model.generate()` from `config.json` and compare outputs changing ONE parameter at a time.

## Run it
```bash
.venv/bin/python 05_llm_parameters/03_hf_generate_compare/main.py
```

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `sys.path.insert(0, folder)` | tell Python where else to look for imports | - |
| `from config_loader import ...` | import from a sibling file that is on `sys.path` | - |
| `# noqa: E402` | tells linters "imports not at top is intentional" | - |
| `kwargs.update(overrides)` | merge dicts in place; later keys win | - |
| `def run(label, **overrides)` | `**` collects extra keyword args into a dict | `run("x", temperature=0.2)` |
| `out[0][ids.shape[1]:]` | slice off the prompt tokens | - |
| `torch.allclose(a, b, atol=1e-6)` | tensors equal within tolerance | float maths is never bit-exact |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `TemperatureLogitsWarper(T)`, `TopKLogitsWarper(k)`, `TopPLogitsWarper(p)` | transformers | the library's own versions of steps 02's functions | the parameter | callable `(input_ids, scores) -> scores` |
| `model.generate(ids, **kwargs)` | transformers | complete decoding loop | see below | ids tensor |
| `torch.manual_seed(n)` | torch | fixes random draws | n | - |

`generate()` arguments used here: `do_sample` (False=greedy), `temperature`, `top_k` (0 disables), `top_p` (1.0 disables), `max_new_tokens`, `repetition_penalty`, `num_beams` (beam search, optional), `pad_token_id` (needed because GPT-2 has no pad token).

## Execution trace
```
our distribution == HF distribution: True

[config.json as-is]        ' by following some of the most popular tutorials you can find here.<|endoftext|>'
[greedy]                   ' by reading the book.\nI'm not sure if I can get a good grasp of ...'
[temperature 0.2]          ' by reading the book.\nI have been working on a lot of books ...'
[temperature 1.5]          ' only examples Given pace ------- crank70\nM、Cal Warehouse ...'
[top_k=5 only]             ' by learning a lot of the basics.\nThis blog post will give you ...'
[max_new_tokens=8]         ' by following some of the most popular tutorials'

repetition_penalty=1.0: ' pizza. I like pizza. I like pizza. I like pizza. ...'
repetition_penalty=1.3: ' it too, but that's not what we do here in the city of Chicago."...'
```
- `True` = the maths you wrote in step 02 equals the library's: nothing in `generate()` is magic.
- With the same **seed**, `config.json as-is` and `max_new_tokens=8` start with identical text: the random stream is the same, so changing only the length leaves the beginning unchanged. Fixing the seed isolates the effect of one parameter - the basis of fair experiments (step 07).
- **Warning you may see:** "generation flags are not valid and may be ignored: ['temperature', 'top_p']" appears for the greedy run because `do_sample=False` makes sampling parameters meaningless. Read warnings; they are free debugging hints.
- **Repetition loop** under greedy (`pizza. I like pizza...`) and how the penalty breaks it - at the price of drifting off-topic ("Chicago"). Too high a penalty makes the model avoid legitimate repeats.

## Experiments
- Change the seed in `run()`: which settings are stable (low T) and which change a lot (high T)?
- Add `num_beams=4, do_sample=False`: beam search explores several candidate sequences; slower, often more "generic".
- Set `top_p=0.5` only (T=1, top_k=0).

## Debugging practice
`temperature=0.0` with `do_sample=True` raises `ValueError: `temperature` (=0.0) has to be a strictly positive float...`. A good library error tells you the rule and often the fix - read all of it, then check the docs (`help(model.generate)` or the Hugging Face "text generation" docs page) for the parameter table.

## Rebuild checklist
- [ ] Call `generate` with sampling parameters loaded from a dict.
- [ ] Explain what `pad_token_id` is for.
- [ ] Explain why the seed must be fixed when comparing settings.
- [ ] Say which `generate` arguments correspond to your step 02 functions.

## The new problem (cliffhanger)
Sampling is understood and verified. But every request also has a length limit that cuts the answer, and an invisible ceiling on prompt plus answer. Next: [step 04, max_new_tokens and the context window](../04_max_new_tokens_and_context_window/WALKTHROUGH.md).

> **Say it aloud (interview box):** I checked my temperature, top-k and top-p functions against the library's logits warpers and the tensors matched, so generate has no hidden magic. With the same seed, changing one argument at a time shows exactly which setting causes which change. That discipline of one change, fixed seed, compare outputs is how I tune generation in practice.
