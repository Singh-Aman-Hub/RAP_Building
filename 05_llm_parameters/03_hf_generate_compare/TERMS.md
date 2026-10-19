# Terms — Compare With Hugging Face generate()

Folder: `05_llm_parameters/03_hf_generate_compare` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [05 LLM parameters README](../README.md)

**Terms in this folder:** [Importing from sibling folders](#importing-from-sibling-folders) · [# noqa: E402](#-noqa-e402) · [LogitsWarper classes](#logitswarper-classes) · [torch.allclose](#torchallclose) · [Keeping the batch dimension](#keeping-the-batch-dimension) · [model.generate](#modelgenerate) · [Config-driven generation](#config-driven-generation) · [def run(label, **overrides)](#def-runlabel-overrides) · [dict.update](#dictupdate) · [Fixed seed (torch.manual_seed)](#fixed-seed-torchmanual_seed) · [Slicing off the prompt](#slicing-off-the-prompt) · [pad_token_id](#pad_token_id) · [Disabling filters (top_k=0, top_p=1.0)](#disabling-filters-top_k0-top_p10) · [Greedy repetition loop](#greedy-repetition-loop) · [repetition_penalty](#repetition_penalty) · [Beam search (num_beams)](#beam-search-num_beams) · [Generation-flag warnings](#generation-flag-warnings) · [Fair experiment (change one parameter)](#fair-experiment-change-one-parameter) · [ValueError for invalid temperature](#valueerror-for-invalid-temperature) · [help() and documentation](#help-and-documentation)

---

## Importing from sibling folders  `(syntax)`
**1. Definition:** Adding sibling directories to `sys.path` so Python can import helper modules from them.
**2. In plain words:** This step reuses `config_loader` from step 01 and `sampling` from step 02. Rather than copy files, it points Python at those folders with `sys.path.insert(0, ...)`.
**3. Example to quote:** > "I insert `01_config_file` and `02_sampling_from_scratch` into `sys.path`, so I reuse my own functions instead of duplicating them."
```python
sys.path.insert(0, str(here.parent / "02_sampling_from_scratch"))
```
**Seen in:** `main.py`, top.

## # noqa: E402  `(syntax)`
**1. Definition:** A comment that tells linters to ignore rule E402 ("module-level import not at top of file") on that line.
**2. In plain words:** The imports come after the `sys.path` change on purpose; the comment says this is intentional so tools do not warn.
**3. Example to quote:** > "`# noqa: E402` marks that these imports are late deliberately, after I set up the path."

**Seen in:** `main.py`, imports.

## LogitsWarper classes  `(third-party: transformers)`
**1. Definition:** Hugging Face classes `TemperatureLogitsWarper`, `TopKLogitsWarper` and `TopPLogitsWarper` that modify the logits just as temperature, top-k and top-p do inside `generate`.
**2. In plain words:** They are the library's versions of the functions written in step 02. Each is called with `(input_ids, scores)` and returns the edited scores. Comparing them with your own proves there is no hidden magic in `generate`.
**3. Example to quote:** > "I applied `TemperatureLogitsWarper(0.8)`, `TopKLogitsWarper(50)` and `TopPLogitsWarper(0.9)` and compared the result with my own filters."

**Seen in:** `main.py`, section 1.

## torch.allclose  `(third-party: torch)`
**1. Definition:** Returns True if two tensors are equal within a tolerance (`atol`, `rtol`).
**2. In plain words:** Floating-point arithmetic is never bit-exact between two implementations, so you compare with a small tolerance (`atol=1e-6`), not with `==`.
**3. Example to quote:** > "`torch.allclose(..., atol=1e-6)` returned True, so my sampler equals the library's."

**Seen in:** `main.py`, section 1.

## Keeping the batch dimension  `(concept)`
**1. Definition:** Slicing logits as `[:, -1, :]` keeps shape `(1, vocab)` instead of `(vocab,)`.
**2. In plain words:** Hugging Face warpers expect 2-D scores (batch, vocab), while your own functions use 1-D logits. Remember to index `[0]` when moving between them.
**3. Example to quote:** > "I keep the batch dimension for the HF warpers and pass `logits[0]` to my own functions."

**Seen in:** `main.py`, section 1.

## model.generate  `(third-party: transformers)`
**1. Definition:** The Hugging Face method that runs the complete decoding loop with sampling, caching and stopping.
**2. In plain words:** Arguments used here: `do_sample`, `temperature`, `top_k`, `top_p`, `max_new_tokens`, `repetition_penalty`, `num_beams`, `pad_token_id`. It returns a tensor of IDs containing the prompt plus new tokens.
**3. Example to quote:** > "`model.generate(ids, pad_token_id=tok.eos_token_id, **kwargs)` produced the continuation with my config's settings."

**Seen in:** `main.py`, sections 2 and 3.

## Config-driven generation  `(concept)`
**1. Definition:** Building the `generate` arguments from `config.json` instead of typing them in code.
**2. In plain words:** One file holds the defaults; each experiment overrides one or two values. This connects step 01 to the API in module 09, whose request body carries these same parameters.
**3. Example to quote:** > "`generation_kwargs(cfg)` gave me the defaults and I overrode only `temperature` for each run."

**Seen in:** `main.py`, section 2.

## def run(label, **overrides)  `(syntax)`
**1. Definition:** A function whose `**overrides` parameter collects any extra keyword arguments into a dictionary.
**2. In plain words:** `run("temperature 0.2", temperature=0.2)` makes `overrides == {"temperature": 0.2}`, which is merged into the config defaults.
**3. Example to quote:** > "`run('top_k=5 only', top_k=5, top_p=1.0)` changes just those settings."

**Seen in:** `main.py`, section 2.

## dict.update  `(Python built-in)`
**1. Definition:** Merges another dict into a dict in place; for duplicate keys, the later value wins.
**2. In plain words:** Start with config defaults, then `update(overrides)` replaces only the keys you want to change.
**3. Example to quote:** > "`kwargs.update(overrides)` swaps in my new temperature and leaves the rest."
```python
d = {"t": 1}; d.update({"t": 2})  # {'t': 2}
```
**Seen in:** `main.py`, `run`.

## Fixed seed (torch.manual_seed)  `(third-party: torch)`
**1. Definition:** Setting PyTorch's random seed before each run so the random stream is identical.
**2. In plain words:** With the same seed, changing only one parameter lets you see that parameter's effect. Without it, differences could be luck.
**3. Example to quote:** > "I call `torch.manual_seed(7)` before every run so only the parameter changes."

**Seen in:** `main.py`, `run`.

## Slicing off the prompt  `(syntax)`
**1. Definition:** `out[0][ids.shape[1]:]` keeps only the tokens after the prompt length.
**2. In plain words:** `generate` returns prompt plus continuation. Slicing leaves just the new text for display.
**3. Example to quote:** > "I decode `out[0][ids.shape[1]:]` to print only what the model added."

**Seen in:** `main.py`, `run`.

## pad_token_id  `(third-party: transformers)`
**1. Definition:** Argument telling `generate` which ID to use for padding; GPT-2 has none so the end-of-text ID is passed.
**2. In plain words:** It avoids a warning and has no effect on a single unpadded prompt.
**3. Example to quote:** > "I pass `pad_token_id=tok.eos_token_id` to silence the GPT-2 padding warning."

**Seen in:** `main.py`.

## Disabling filters (top_k=0, top_p=1.0)  `(concept)`
**1. Definition:** Setting `top_k=0` and `top_p=1.0` turns those filters off in `generate`.
**2. In plain words:** The config has top-k 50 and top-p 0.95 on by default. To test temperature alone you switch the filters off.
**3. Example to quote:** > "For the temperature-1.5 run I set `top_k=0, top_p=1.0` so only temperature changes the distribution."

**Seen in:** `main.py`, section 2.

## Greedy repetition loop  `(concept)`
**1. Definition:** Greedy decoding repeating the same phrase endlessly, such as "I like pizza. I like pizza..."
**2. In plain words:** If the most likely next tokens recreate the previous text, the model gets trapped. The repetition penalty breaks the loop.
**3. Example to quote:** > "At penalty 1.0 the model repeated 'pizza. I like' over and over."

**Seen in:** `main.py`, section 3.

## repetition_penalty  `(third-party: transformers)`
**1. Definition:** `generate` argument above 1 that reduces the probability of already generated tokens.
**2. In plain words:** At 1.3 the loop broke, at the cost of drifting off-topic ("Chicago"). Too high makes the model avoid legitimate repeats.
**3. Example to quote:** > "`repetition_penalty=1.3` stopped the pizza loop but changed the topic."

**Seen in:** `main.py`, section 3.

## Beam search (num_beams)  `(concept)`
**1. Definition:** A decoding strategy keeping several candidate sequences at once and choosing the best overall sequence.
**2. In plain words:** It explores more options than greedy; it is slower and often gives safer, more generic text. Enabled with `num_beams=4, do_sample=False`.
**3. Example to quote:** > "I could try `num_beams=4` to compare beam search with sampling."

**Seen in:** `main.py`, TRY IT comment.

## Generation-flag warnings  `(concept)`
**1. Definition:** Library warnings such as "generation flags are not valid and may be ignored: ['temperature', 'top_p']".
**2. In plain words:** With `do_sample=False`, sampling parameters are ignored, and the library tells you. Read warnings; they are free debugging hints.
**3. Example to quote:** > "The greedy run warned that temperature and top_p are ignored, because sampling is off."

**Seen in:** `WALKTHROUGH.md`.

## Fair experiment (change one parameter)  `(concept)`
**1. Definition:** Comparing settings by changing a single variable while holding everything else, including the random seed, fixed.
**2. In plain words:** With the same seed, the config run and the `max_new_tokens=8` run start with identical text; only the length differs. That isolates cause and effect.
**3. Example to quote:** > "I fix the seed and vary one setting per run, so differences are due to that setting."

**Seen in:** `WALKTHROUGH.md`.

## ValueError for invalid temperature  `(concept)`
**1. Definition:** The error Hugging Face raises when sampling is on and `temperature` is not strictly positive.
**2. In plain words:** The message states the rule. Use `do_sample=False` for deterministic output instead of temperature 0.
**3. Example to quote:** > "`temperature=0.0` with `do_sample=True` raised 'has to be a strictly positive float'."

**Seen in:** `main.py`, BREAK IT comment.

## help() and documentation  `(Python built-in)`
**1. Definition:** `help(obj)` prints an object's docstring and signature.
**2. In plain words:** Use `help(model.generate)` or the Hugging Face docs to read the parameter table instead of guessing. Reading docs yourself is a core debugging skill.
**3. Example to quote:** > "I checked `help(model.generate)` for what `repetition_penalty` expects."

**Seen in:** `WALKTHROUGH.md`, debugging section.
