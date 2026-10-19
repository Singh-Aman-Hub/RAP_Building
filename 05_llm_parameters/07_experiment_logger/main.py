"""
GOAL:   Run several generation settings, measure output length / latency / memory, add a simple
        quality signal, and save everything to a CSV + Markdown table you can compare.

RUN:    from this folder:  python main.py
        from repo root:    .venv/bin/python 05_llm_parameters/07_experiment_logger/main.py
EXPECTED OUTPUT (short): a Markdown table with one row per experiment, also saved to
        results/experiments.csv and results/experiments.md; gpt2 is slower/bigger than distilgpt2.
NEEDS:  torch, transformers.  First run downloads gpt2 (~500 MB) in addition to distilgpt2.  ~1-2 min.

Why log?  "It felt better" is not evidence.  Changing ONE parameter at a time and recording the
numbers is how engineers tune an LLM service (and how you answer "what happens if you raise
temperature / use a bigger model?" in an interview with data instead of guesses).

Uses: step 01 (config_loader: defaults), step 02/03 knobs, step 04 (token counts), step 05 idea (memory).
"""

import csv
import resource                       # standard library (Unix): process memory usage
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

here = Path(__file__).parent
sys.path.insert(0, str(here.parent / "01_config_file"))
from config_loader import load_config, generation_kwargs   # noqa: E402

PROMPT = "The future of artificial intelligence is"
REPEATS = 3                           # time each setting several times and take the median (timings are noisy)
cfg = load_config()
BASE = generation_kwargs(cfg)         # defaults come from config.json; each experiment overrides ONE thing

# Each experiment = (name, model_name, overrides to BASE).  Change one thing per row!
EXPERIMENTS = [
    ("baseline (config.json)",   "distilgpt2", {}),
    ("greedy",                   "distilgpt2", {"do_sample": False}),
    ("temperature 0.3",          "distilgpt2", {"temperature": 0.3}),
    ("temperature 1.5",          "distilgpt2", {"temperature": 1.5, "top_k": 0, "top_p": 1.0}),
    ("top_k 5",                  "distilgpt2", {"top_k": 5, "top_p": 1.0}),
    ("repetition_penalty 1.0",   "distilgpt2", {"repetition_penalty": 1.0, "do_sample": False}),
    ("max_new_tokens 10",        "distilgpt2", {"max_new_tokens": 10}),
    ("max_new_tokens 100",       "distilgpt2", {"max_new_tokens": 100}),
    ("bigger model: gpt2",       "gpt2",       {}),
]


def rss_mb():
    """Peak memory of this process in MB. macOS reports bytes, Linux reports kilobytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def distinct_ratio(token_ids):
    """Quality proxy: unique tokens / total tokens.  Near 1.0 = varied; low = repetitive loops."""
    return len(set(token_ids)) / max(len(token_ids), 1)


def quality_hint(ratio, n_new, max_new):
    """Very rough automatic note. YOUR reading of the actual text (column my_notes) matters more."""
    if ratio < 0.5:
        return "repetitive"
    if n_new < max_new:
        return "stopped early (EOS)"
    return "varied"


loaded = {}                           # cache models so each is loaded once: {name: (tokenizer, model, MB)}
rows = []
for name, model_name, overrides in EXPERIMENTS:
    if model_name not in loaded:
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).eval()
        param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
        with torch.no_grad():                           # warm-up run: first call is always slower (lazy init, caches)
            model.generate(tok("warm up", return_tensors="pt")["input_ids"], max_new_tokens=5,
                           do_sample=False, pad_token_id=tok.eos_token_id)
        loaded[model_name] = (tok, model, param_mb)
    tok, model, param_mb = loaded[model_name]

    kwargs = {**BASE, **overrides}    # {**a, **b}: merge dicts, b wins. A dict "unpacking" idiom.
    if not kwargs["do_sample"]:       # greedy ignores sampling params; drop them to avoid HF warnings
        for k in ("temperature", "top_k", "top_p"):
            kwargs.pop(k, None)
    ids = tok(PROMPT, return_tensors="pt")["input_ids"]   # shape: (1, prompt_len)

    times, last_out = [], None
    for r in range(REPEATS):
        torch.manual_seed(r)
        t0 = time.perf_counter()
        with torch.no_grad():
            last_out = model.generate(ids, pad_token_id=tok.eos_token_id, **kwargs)
        times.append(time.perf_counter() - t0)

    new_ids = last_out[0][ids.shape[1]:].tolist()      # only the generated tokens
    latency = statistics.median(times)
    ratio = distinct_ratio(new_ids)
    rows.append({
        "experiment": name,
        "model": model_name,
        "new_tokens": len(new_ids),
        "latency_s": round(latency, 2),
        "tokens_per_s": round(len(new_ids) / latency, 1),
        "weights_MB": round(param_mb),
        "peak_rss_MB": round(rss_mb()),
        "distinct_ratio": round(ratio, 2),
        "auto_note": quality_hint(ratio, len(new_ids), kwargs["max_new_tokens"]),
        "my_notes": "",                                # <- fill this in by READING the sample below
        "sample": tok.decode(new_ids[:25]).replace("\n", " "),
    })
    print(f"done: {name}")

# ---------------------------------------------------------------------------
# Save CSV (opens in Excel/Numbers) and a Markdown table (readable in the editor)
# ---------------------------------------------------------------------------
out_dir = here / "results"
out_dir.mkdir(exist_ok=True)                           # exist_ok: no error if the folder already exists
with open(out_dir / "experiments.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

cols = ["experiment", "new_tokens", "latency_s", "tokens_per_s", "weights_MB", "peak_rss_MB", "distinct_ratio", "auto_note"]
md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
(out_dir / "experiments.md").write_text("\n".join(md) + "\n\n## Samples\n\n" +
                                        "\n".join(f"- **{r['experiment']}**: {r['sample']}" for r in rows) + "\n")
print("\n" + "\n".join(md))
print(f"\nsaved to {out_dir}/experiments.csv and experiments.md")
print("peak_rss_MB only ever grows (it is a high-water mark), so read differences, not absolutes.")

# TRY IT: add an experiment ("top_p 0.5", "distilgpt2", {"top_p": 0.5}) and re-run.
# TRY IT: add a second prompt and average over both - why is one prompt a weak test? (see module 07 testing)
# BREAK IT: use REPEATS = 0 -> statistics.StatisticsError / empty list. Where is the traceback pointing?
# BREAK IT: add an experiment with {"temperature": -1} - HF refuses; compare with config_loader's validation.
