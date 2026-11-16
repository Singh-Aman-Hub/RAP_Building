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

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| `model.generate()` | decoding, inference call, `pipeline("text-generation")` | `pipeline` wraps `generate` | `generate` is the core Hugging Face decoding loop; `pipeline` adds tokenising and decoding around it |
| Logits warper / processor | `LogitsProcessor`, `LogitsWarper` | Mostly | A small callable that edits logits before sampling; HF's classes for temperature, top-k, top-p, repetition penalty. Newer versions merge "warpers" into "processors" |
| `GenerationConfig` | generation defaults, `generation_config.json` | Related | The object where defaults like temperature live; kwargs to `generate` override it |
| Seed | `torch.manual_seed`, random state | Yes | Fixes the random stream so only the parameter being tested changes |
| `do_sample` | sampling switch | Yes | `False` is greedy (temperature, top-k, top-p ignored); `True` samples |
| `max_new_tokens` vs `max_length` | output cap vs total cap | No | New tokens only vs prompt plus new tokens |
| EOS stop | `eos_token_id`, `<|endoftext|>` | Yes | Model can end generation before `max_new_tokens` |
| Beam search | `num_beams` | No | Deterministic multi-path search; not sampling |

## Code, step by step

### 1. Module docstring  (`main.py` lines 1–13)

```python
"""
GOAL:   Prove your from-scratch sampler matches Hugging Face, then use model.generate()
        with the settings from config.json and compare outputs one parameter at a time.

RUN:    from this folder:  python main.py
        from repo root:    .venv/bin/python 05_llm_parameters/03_hf_generate_compare/main.py
EXPECTED OUTPUT (short): "our distribution == HF distribution: True" then ~7 generated
        continuations of the same prompt under different settings.
NEEDS:  torch + transformers (distilgpt2 cached). Imports step 01 (config) and step 02 (sampling).

Builds on: step 02 (your functions) and 01 (config.json).  Leads to: step 07 (logs these runs to a table)
and module 09 (an API endpoint whose request body carries exactly these parameters).
"""
```

- **In simple words:** Says we verify our from-scratch sampler against Hugging Face, then use `generate()` with the config file.
- **Exactly:** Docstring only. Expected: `our distribution == HF distribution: True`, then several continuations of the same prompt.
- **Why it is here:** A re-implementation is only trusted once it matches the library.

### 2. Imports including Hugging Face's own logits processors  (`main.py` lines 15–21)

```python
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import (   # HF's own versions of what we wrote
    TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper)
```

- **In simple words:** Bring in the tokenizer, the model and the library's own temperature, top-k and top-p classes.
- **Exactly:** `TemperatureLogitsWarper`, `TopKLogitsWarper` and `TopPLogitsWarper` are what `generate()` builds internally from your kwargs.
- **Why it is here:** Comparing with them shows our maths is the same.

### 3. Reuse steps 01 and 02  (`main.py` lines 23–27)

```python
here = Path(__file__).parent
sys.path.insert(0, str(here.parent / "01_config_file"))
sys.path.insert(0, str(here.parent / "02_sampling_from_scratch"))
from config_loader import load_config, generation_kwargs   # noqa: E402  (step 01)
import sampling as S                               # noqa: E402
```

- **In simple words:** Make the config loader and the sampler importable from their folders.
- **Exactly:** `here.parent / "01_config_file"` is inserted into `sys.path` (and so is the 02 folder). `# noqa: E402` silences a linter warning about imports after code.
- **Why it is here:** Reusing code avoids copies that drift apart; the folder-specific loader name avoids the `main.py` collision.

### 4. Load config, model and the prompt  (`main.py` lines 29–33)

```python
cfg = load_config()
tok = AutoTokenizer.from_pretrained(cfg["model_name"])
model = AutoModelForCausalLM.from_pretrained(cfg["model_name"]).eval()
prompt = "The best way to learn programming is"
ids = tok(prompt, return_tensors="pt")["input_ids"]   # shape: (1, prompt_len)
```

- **In simple words:** Read settings from config.json, then load the model named there and encode the prompt.
- **Exactly:** `cfg["model_name"]` is `distilgpt2`. `ids` is `(1, 7)` for "The best way to learn programming is".
- **Why it is here:** The model name lives in config, so switching models is a one-line edit.

### 5. Check our maths equals Hugging Face's  (`main.py` lines 35–46)

```python
# ---------------------------------------------------------------------------
# 1. Is our math the same as the library's?  Compare the final probability distributions.
# ---------------------------------------------------------------------------
with torch.no_grad():
    logits = model(ids).logits[:, -1, :]                # keep batch dim (1, vocab): HF warpers expect 2-D

ours = S.top_p_filter(S.top_k_filter(S.apply_temperature(logits[0], 0.8), 50), 0.9)
hf = logits.clone()
for warper in (TemperatureLogitsWarper(0.8), TopKLogitsWarper(50), TopPLogitsWarper(0.9)):
    hf = warper(ids, hf)                                # each warper edits the scores like our functions
same = torch.allclose(torch.softmax(ours, -1), torch.softmax(hf[0], -1), atol=1e-6)
print("our distribution == HF distribution:", same)
```

- **In simple words:** Apply temperature, top-k and top-p our way and the library's way, and compare the final probabilities.
- **Exactly:** `logits = model(ids).logits[:, -1, :]` keeps the batch axis: `(1, 50257)` because HF's warpers expect 2-D scores. Ours: `S.top_p_filter(S.top_k_filter(S.apply_temperature(logits[0], 0.8), 50), 0.9)`. HF's: three warpers applied in order, each called as `warper(ids, scores)`. `torch.allclose(softmax(ours), softmax(hf[0]), atol=1e-6)` is `True` (real output).
- **Why it is here:** This is the evidence that what you wrote in step 02 is exactly what `generate` does.

### 6. A helper that runs generate with fixed seed  (`main.py` lines 48–59)

```python
# ---------------------------------------------------------------------------
# 2. generate() with the config.json settings
# ---------------------------------------------------------------------------
def run(label, **overrides): 
    """Generate once with config defaults + overrides. Seed is fixed so ONLY the parameter changes."""
    kwargs = generation_kwargs(cfg)
    kwargs.update(overrides)                            # dict.update: later keys win
    torch.manual_seed(7)                                # same random stream for every run
    with torch.no_grad():
        out = model.generate(ids, pad_token_id=tok.eos_token_id, **kwargs)   # shape: (1, prompt_len + new_tokens)
    text = tok.decode(out[0][ids.shape[1]:])            # slice off the prompt: keep only NEW tokens
    print(f"\n[{label}]\n  {text!r}")
```

- **In simple words:** Generate once with config values, allowing a few overrides, so only one setting changes between runs.
- **Exactly:** `generation_kwargs(cfg)` returns a dict; `kwargs.update(overrides)` lets later keys win. `torch.manual_seed(7)` gives the same random stream every call. `model.generate(ids, pad_token_id=..., **kwargs)` returns `(1, prompt_len + new_tokens)`. `out[0][ids.shape[1]:]` slices off the prompt so only new tokens are decoded.
- **Why it is here:** Controlled experiments: if the seed changed too, you could not tell the effect of a parameter from luck.

### 7. Six runs, one parameter at a time  (`main.py` lines 61–66)

```python
run("config.json as-is")
run("greedy (do_sample=False)", do_sample=False)
run("temperature 0.2", temperature=0.2)
run("temperature 1.5", temperature=1.5, top_k=0, top_p=1.0)    # top_k=0 / top_p=1.0 switch the filters OFF
run("top_k=5 only", temperature=1.0, top_k=5, top_p=1.0)
run("max_new_tokens=8", max_new_tokens=8)
```

- **In simple words:** Run the same prompt under different settings and compare the text.
- **Exactly:** Real output: config as-is gives `' by following some of the most popular tutorials you can find here.<|endoftext|>'` (it ended early with the EOS token). Greedy and temperature 0.2 both start `' by reading the book.'`. temperature 1.5 with filters off produces gibberish (`' only examples Given pace -----...'`). top_k=5 is coherent but different. max_new_tokens=8 gives exactly 8 tokens. `top_k=0` and `top_p=1.0` switch the filters off. In transformers 5.x, `do_sample=False` also prints a notice that `temperature` and `top_p` are not valid and may be ignored.
- **Why it is here:** Shows each knob's effect and the greedy rule that sampling parameters are ignored when `do_sample=False`.

### 8. Why repetition_penalty exists  (`main.py` lines 68–75)

```python
# ---------------------------------------------------------------------------
# 3. Why repetition_penalty exists: greedy decoding falls into loops
# ---------------------------------------------------------------------------
loop_prompt = tok("I like pizza. I like pizza. I like", return_tensors="pt")["input_ids"]
for rp in (1.0, 1.3):
    out = model.generate(loop_prompt, max_new_tokens=25, do_sample=False,
                         repetition_penalty=rp, pad_token_id=tok.eos_token_id)
    print(f"\nrepetition_penalty={rp}: {tok.decode(out[0][loop_prompt.shape[1]:])!r}")
```

- **In simple words:** Greedy decoding on a repetitive prompt loops forever; the penalty breaks the loop.
- **Exactly:** With penalty 1.0 the model repeats `' pizza. I like pizza. I like pizza...'`. With 1.3 it escapes: `' it too, but that's not what we do here in the city of Chicago."...'`.
- **Why it is here:** Loops are a known failure of greedy and low-temperature decoding.

### 9. Experiments and break-it comments  (`main.py` lines 77–80)

```python
# TRY IT: change the seed in run() and see which settings are stable (low T) vs wildly different (high T).
# TRY IT: set num_beams=4, do_sample=False - beam search is another decoding strategy (explore several paths).
# BREAK IT: temperature=0.0 with do_sample=True -> HF raises ValueError ("temperature ... strictly positive"). Read it.
# BREAK IT: pass top_p=1.5 -> a ValueError/warning; compare with the message YOU wrote in step 01.
```

- **In simple words:** Ideas: change the seed, try beam search, use illegal values.
- **Exactly:** `temperature=0.0` with `do_sample=True` raises a `ValueError` from Hugging Face; `top_p=1.5` is rejected too. Compare with your own message in step 01.
- **Why it is here:** Libraries validate parameters much like your `load_config` does.

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
