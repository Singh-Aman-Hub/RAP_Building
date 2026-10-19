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

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import (   # HF's own versions of what we wrote
    TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper)

here = Path(__file__).parent
sys.path.insert(0, str(here.parent / "01_config_file"))
sys.path.insert(0, str(here.parent / "02_sampling_from_scratch"))
from config_loader import load_config, generation_kwargs   # noqa: E402  (step 01)
import sampling as S                               # noqa: E402

cfg = load_config()
tok = AutoTokenizer.from_pretrained(cfg["model_name"])
model = AutoModelForCausalLM.from_pretrained(cfg["model_name"]).eval()
prompt = "The best way to learn programming is"
ids = tok(prompt, return_tensors="pt")["input_ids"]   # shape: (1, prompt_len)

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

run("config.json as-is")
run("greedy (do_sample=False)", do_sample=False)
run("temperature 0.2", temperature=0.2)
run("temperature 1.5", temperature=1.5, top_k=0, top_p=1.0)    # top_k=0 / top_p=1.0 switch the filters OFF
run("top_k=5 only", temperature=1.0, top_k=5, top_p=1.0)
run("max_new_tokens=8", max_new_tokens=8)

# ---------------------------------------------------------------------------
# 3. Why repetition_penalty exists: greedy decoding falls into loops
# ---------------------------------------------------------------------------
loop_prompt = tok("I like pizza. I like pizza. I like", return_tensors="pt")["input_ids"]
for rp in (1.0, 1.3):
    out = model.generate(loop_prompt, max_new_tokens=25, do_sample=False,
                         repetition_penalty=rp, pad_token_id=tok.eos_token_id)
    print(f"\nrepetition_penalty={rp}: {tok.decode(out[0][loop_prompt.shape[1]:])!r}")

# TRY IT: change the seed in run() and see which settings are stable (low T) vs wildly different (high T).
# TRY IT: set num_beams=4, do_sample=False - beam search is another decoding strategy (explore several paths).
# BREAK IT: temperature=0.0 with do_sample=True -> HF raises ValueError ("temperature ... strictly positive"). Read it.
# BREAK IT: pass top_p=1.5 -> a ValueError/warning; compare with the message YOU wrote in step 01.
