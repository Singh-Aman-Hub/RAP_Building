"""
GOAL:  A reference you can query: what each major benchmark measures, how it is scored, and its status as of Nov 2026, with an
       honest evidence tag per row (V = verified in a Nov 2026 web check, K = stable fact from the benchmark's paper/design,
       S = only secondary sources, check the leaderboard). NO scores are quoted: leaderboards move weekly. The script also shows
       which eval / training libraries are installed in this venv and prints the exact commands for the ones that are not.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 17_finetuning_alignment_and_benchmarks/10_benchmark_landscape_2026_and_tools/main.py
EXPECTED OUTPUT (shortened):
    landscape table (16 rows grouped by what they test), counts per status, installed-library check, CLI one-liners
NEEDS: Python standard library only.  Instant.
"""
import importlib.util

# ## 1. The landscape as data: (name, measures, scoring, status Nov 2026, evidence)
BENCH = [
    # --- knowledge / reasoning (multiple choice, scored by log-likelihood or letter) ---
    ("MMLU",            "57-subject school-to-professional knowledge, 4-choice MCQ", "accuracy",                       "saturated for frontier models; still a sanity check for small models", "S"),
    ("MMLU-Pro",        "harder MMLU, 10 choices, more reasoning",                    "accuracy",                       "frontier models cluster near the top; low discrimination at the high end", "S"),
    ("GPQA Diamond",    "198 graduate-level science MCQs written to be search-proof", "accuracy",                       "near saturation; label noise limits the ceiling; ties within CI", "K+S"),
    ("HLE",             "Humanity's Last Exam: 2,500 expert questions (Nature, Jan 2026)", "accuracy (exact-match / judge)", "headroom remains; score depends on tools and subset, check leaderboard", "V"),
    # --- math ---
    ("AIME 2024/25/26", "competition math, 30 problems per year, integer answers 0-999", "pass@1 or avg over samples",   "treated as solved by frontier reasoning models; 30 items = huge variance", "S"),
    ("MATH-500",        "500-problem subset of MATH, free-form answers",              "exact match after normalisation", "widely reported as saturated for reasoning models", "S"),
    # --- code and agents ---
    ("LiveCodeBench",   "fresh competitive-programming problems with release dates",  "pass@1 on a date window",        "still used because new problems limit contamination", "K+S"),
    ("SWE-bench Verified", "500 human-validated real GitHub issues, patch must pass tests", "% resolved",              "OpenAI stopped reporting it (Feb 2026): contamination + flawed tests", "V"),
    ("SWE-bench Pro",   "longer-horizon, harder repo tasks (Scale AI)",               "% resolved",                     "recommended successor; a May 2026 audit claims grader errors (unconfirmed)", "V+S"),
    ("Terminal-Bench",  "agent solves tasks in a real shell (2.0: 89 tasks, ICLR 2026)", "% tasks passed",             "active; newer versions may exist, check tbench.ai", "V"),
    ("tau-bench / tau2-bench", "tool-agent-user dialogues (retail, airline, telecom ...) with policies", "pass^k (all k trials succeed)", "active; pass^k exposes unreliability that pass@1 hides", "V"),
    ("BFCL V4",         "function calling incl. agentic web search, memory, format sensitivity", "weighted accuracy",   "active; an Epoch AI review (Sep 2026) rates it 'Flawed' on sampled items", "V"),
    # --- instruction following, preference, long context, embeddings ---
    ("IFEval",          "verifiable instructions ('under 100 words', 'include X')",   "strict/loose prompt-level accuracy", "status not checked in Nov 2026; rule-based so cheap and reproducible", "K"),
    ("Arena (ex-LMArena)", "crowd pairwise votes on chat answers",                   "Bradley-Terry rating + CI",       "rebranded 'Arena' Jan 2026; style/length bias is the known weakness", "V+K"),
    ("RULER",           "synthetic long-context tasks at configurable lengths",       "accuracy per length",            "shows effective context < advertised context", "K"),
    ("MTEB / MMTEB",    "text embeddings across retrieval, STS, classification, clustering", "mean of task metrics",    "the standard embedding leaderboard (see module 14); check current version", "K"),
]
print(f"{'benchmark':<24}{'ev':<5}{'scoring':<40}status Nov 2026")
for name, measures, scoring, status, ev in BENCH:
    print(f"{name:<24}{ev:<5}{scoring:<40}{status}")
    print(f"{'':<24}      measures: {measures}")

# ## 2. How many rows rest on which evidence
from collections import Counter
print("\nevidence tags:", dict(Counter(b[4] for b in BENCH)), "  (V verified, K stable design fact, S secondary only)")

# ## 3. What is installed here, and the command to use when it is not
TOOLS = {
    "torch": "pip install torch", "transformers": "pip install transformers", "peft": "pip install peft",
    "trl": "pip install trl            # v1.x (PyPI showed 1.17.1 on 2026-10-30)",
    "lm_eval": "pip install lm_eval[hf]    # 0.4.13 on PyPI (2026-08-31)",
    "lighteval": "pip install lighteval", "inspect_ai": "pip install inspect-ai",
    "bitsandbytes": "pip install bitsandbytes  # CUDA/ROCm; needed for QLoRA", "unsloth": "pip install unsloth",
    "axolotl": "pip install axolotl", "llamafactory": "pip install llamafactory",
}
print("\nlibrary check:")
for mod, cmd in TOOLS.items():
    ok = importlib.util.find_spec(mod) is not None
    print(f"  {mod:<14}{'installed' if ok else 'missing  ':<10}{'' if ok else cmd}")

# ## 4. One-liners (see the *_reference.py files for full code; none of these run here)
ONE_LINERS = {
    "lm-evaluation-harness": "lm_eval --model hf --model_args pretrained=gpt2 --tasks hellaswag,arc_easy --batch_size 8 --limit 100",
    "list harness tasks":    "lm-eval ls tasks",
    "Axolotl":               "axolotl train config.yml                       # YAML holds base_model, datasets, adapter: lora ...",
    "LLaMA-Factory":         "llamafactory-cli train examples/train_lora/<your_sft>.yaml",
    "TRL SFT script":        "python trl_sft_reference.py",
}
print("\none-liners:")
for k, v in ONE_LINERS.items():
    print(f"  {k:<22}{v}")

# ## 5. A rule of thumb you can code: is a gap between two scores believable?
import math
def gap_is_real(acc_a, acc_b, n):
    """Two-proportion z-test on independent samples of size n (an approximation; use a paired test when items are shared)."""
    p = (acc_a + acc_b) / 2
    se = math.sqrt(2 * p * (1 - p) / n)
    return abs(acc_a - acc_b) / se
for n in (30, 500, 5000):
    print(f"n={n:<5} gap 0.60 vs 0.65 -> z = {gap_is_real(0.60, 0.65, n):.2f}  (|z| > 1.96 needed)")

# TRY IT: add your own benchmark row. BREAK IT: quote a score from memory next to a row and ask where it came from (evidence tag!).
