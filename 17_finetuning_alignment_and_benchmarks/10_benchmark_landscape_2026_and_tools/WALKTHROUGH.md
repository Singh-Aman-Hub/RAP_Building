# Step 10 — The benchmark landscape (Nov 2026) and the tools

## The story so far
[Step 08](../08_evaluation_from_scratch/WALKTHROUGH.md) showed how scores are computed and [step 09](../09_llm_as_judge_and_arenas/WALKTHROUGH.md) how human and judge votes become rankings. In an interview you are also expected to *name* benchmarks, say what each measures, and know which ones still discriminate between models.

## The pain point
Benchmarks age. MMLU was hard in 2021 and is a sanity check now. Public test sets leak into training data. A headline number without the benchmark's name, version, prompt format and sample size means nothing, and numbers quoted from memory are usually stale.

## The spark
Keep a small, honest reference: for each benchmark, what it measures, how it is scored, its status, and **how well that status was verified**. Never quote a score you cannot source; send people to the live leaderboard instead.

## Goal
Print the landscape as a table with an evidence tag per row, check which evaluation and training libraries exist in this venv (and give the install command for the rest), list the one-line commands for the main tools, and finish with a z-test helper for "is this gap believable?".

## The theory (read this before the interview)
**Why benchmarks stop being useful:**
| Failure | Meaning | Example (Nov 2026) |
|---|---|---|
| Saturation | top models are near the maximum, differences sit inside noise | AIME, MATH-500, MMLU, near-saturated GPQA Diamond (status from secondary sources: check) |
| Contamination | test items (or near copies) are in training data | OpenAI announced on 2026-02-23 that it stopped reporting SWE-bench Verified because of contamination and flawed tests, and recommended SWE-bench Pro |
| Flawed items | wrong gold labels, tests that reject valid answers | the same OpenAI post audited hard Verified problems and found a large share with flawed tests; an Epoch AI review (Sep 2026) rated BFCL V4 "Flawed" on a 50-item sample |
| Small n | 30 items cannot separate 5 points | AIME has 30 problems per year |
| Prompt sensitivity | harness, few-shot count, answer extraction change the score | compare only runs with the same harness and settings |

**What each family measures** (see the table below for the full list): knowledge and reasoning by multiple choice (MMLU, MMLU-Pro, GPQA Diamond, HLE); competition maths (AIME, MATH-500); code (LiveCodeBench for fresh algorithmic problems, SWE-bench for repository issues); agents (Terminal-Bench for shell tasks, tau-bench / tau2-bench for tool-agent-user conversations scored with **pass^k**, BFCL for function calling); instruction following (IFEval, rule-checked); human preference (Arena); long context (RULER); embeddings (MTEB).

**pass^k vs pass@k:** pass@k = at least one of k tries succeeds (optimistic, fits "generate and verify"). pass^k = all k tries succeed (strict, fits customer-facing agents). tau-bench introduced pass^k to expose unreliability.

**Evaluation tooling (what to name in an interview):**
| Tool | Role | One-liner |
|---|---|---|
| lm-evaluation-harness (EleutherAI) | the standard open harness: log-likelihood and generative tasks | `lm_eval --model hf --model_args pretrained=gpt2 --tasks hellaswag --batch_size 8` (PyPI showed 0.4.13 on 2026-08-31) |
| lighteval (Hugging Face) | alternative harness | check `lighteval --help` |
| Inspect (`inspect_ai`, UK AISI) | agentic and model-graded evals | `inspect eval <task> --model <provider/model>` |
| TRL (Hugging Face) | training: SFT, DPO, GRPO trainers | v1.x; PyPI showed 1.17.1 on 2026-10-30 |
| Unsloth | faster, lower-memory LoRA/QLoRA training | API shape in `unsloth_reference.py` (not re-verified) |
| Axolotl | YAML-driven fine-tuning | `axolotl train config.yml` |
| LLaMA-Factory | CLI/web UI fine-tuning | `llamafactory-cli train <config.yaml>` |

**Evidence tags used in the table:** `V` = confirmed in a Nov 2026 web check; `K` = a stable design fact of the benchmark (from its paper); `S` = only secondary sources, so check the leaderboard; combinations mean part of the row is of each kind. **No score is quoted anywhere in this module on purpose.**

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Benchmark | eval set, test set, leaderboard task | near | A benchmark is a public eval with a fixed protocol |
| Saturation vs contamination | ceiling vs leakage | no | Saturation: too easy now; contamination: seen during training |
| pass@k vs pass^k | at-least-one vs all-of-k | no | Optimistic vs strict reliability |
| SWE-bench Verified vs Pro | human-validated subset vs harder successor | no | Verified is 500 validated issues; Pro is the longer-horizon successor OpenAI recommended |
| Harness vs benchmark | lm-eval-harness vs MMLU | no | The harness is the runner; MMLU is a task it can run |
| LMArena vs Arena | Chatbot Arena | same project | Rebranded to "Arena" in January 2026 |
| TRL vs Axolotl vs LLaMA-Factory vs Unsloth | training frameworks | no | A library of trainers vs YAML/CLI wrappers vs a speed-optimised stack |

## Run it
```bash
cd 17_finetuning_alignment_and_benchmarks/10_benchmark_landscape_2026_and_tools
python main.py
# read (do not run) these:
#   lm_eval_harness_reference.py  trl_sft_reference.py  trl_dpo_reference.py  trl_grpo_reference.py  unsloth_reference.py
```

## Code, step by step

Lines 1–12 of `main.py` are the header docstring (GOAL / RUN / EXPECTED OUTPUT / NEEDS) already summarised above; the walk starts after it.

### 0. Imports, settings and seeds  (`main.py` lines 13–13)
```python
import importlib.util
```
- **In simple words:** Load the one library this script needs. It uses only Python's standard library.
- **Exactly:** `importlib.util` is the standard module that can ask "is this package importable?" without importing it. There are no seeds because there is no randomness.
- **Why it is here:** A reference table should run anywhere, including a laptop with nothing installed.

### 1. The landscape as data: (name, measures, scoring, status Nov 2026, evidence)  (`main.py` lines 15–41)
```python
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
```
- **In simple words:** The landscape is just data: one row per benchmark. Five fields tell you what it is and how much to trust my note about it.
- **Exactly:** `BENCH` is a list of 16 tuples `(name, measures, scoring, status, evidence)`. The `evidence` tag is one of `V`, `K`, `S` or a combination. The loop prints name, tag, scoring and status on one line and the `measures` text on the next. No scores appear.
- **Why it is here:** Keeping it as data means you can sort, filter, count or extend it, and the tag forces an honest answer to "how do you know?".

### 2. How many rows rest on which evidence  (`main.py` lines 43–45)
```python
# ## 2. How many rows rest on which evidence
from collections import Counter
print("\nevidence tags:", dict(Counter(b[4] for b in BENCH)), "  (V verified, K stable design fact, S secondary only)")
```
- **In simple words:** Count how many rows rest on which kind of evidence.
- **Exactly:** `Counter(b[4] for b in BENCH)` counts the fifth field. Output: `{'S': 4, 'K+S': 2, 'V': 5, 'V+S': 1, 'K': 3, 'V+K': 1}`.
- **Why it is here:** It turns "I did some checking" into a number.

### 3. What is installed here, and the command to use when it is not  (`main.py` lines 47–59)
```python
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
```
- **In simple words:** Check which tools are installed here and print the install command for the ones that are not.
- **Exactly:** `TOOLS` maps a module name to its pip command. `find_spec(mod) is not None` is True when importable. Installed: `torch`, `transformers`, `peft`. Missing: `trl`, `lm_eval`, `lighteval`, `inspect_ai`, `bitsandbytes`, `unsloth`, `axolotl`, `llamafactory`. The comments on the `trl` and `lm_eval` commands record the PyPI versions seen in Nov 2026.
- **Why it is here:** It explains why this module has `*_reference.py` files instead of runnable TRL code.

### 4. One-liners (see the *_reference.py files for full code; none of these run here)  (`main.py` lines 61–71)
```python
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
```
- **In simple words:** Print the one-line command for each main tool.
- **Exactly:** `ONE_LINERS` is a dict from tool name to the shell command. They are printed, not executed.
- **Why it is here:** In an interview you are often asked "how would you run it?"; having the exact command in your head is useful.

### 5. A rule of thumb you can code: is a gap between two scores believable?  (`main.py` lines 73–83)
```python
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
```
- **In simple words:** A tiny statistical rule of thumb as code: is a gap between two accuracies bigger than noise?
- **Exactly:** `gap_is_real(acc_a, acc_b, n)` returns `|a-b| / se` with `p = (a+b)/2` and `se = sqrt(2*p*(1-p)/n)`: a two-proportion z statistic for independent samples (use a paired test when both systems are scored on the same items, as in step 08). For the same 0.60-vs-0.65 gap it prints z = 0.40, 1.63, 5.16 for n = 30, 500, 5000. The last two lines of the file are the TRY IT and BREAK IT comments.
- **Why it is here:** It makes the point of the whole module in three lines: a score needs a sample size.


## Reference files in this folder (read, not run)
| File | Needs | What it shows | Checked against |
|---|---|---|---|
| [lm_eval_harness_reference.py](lm_eval_harness_reference.py) | `lm_eval[hf]` | CLI and `simple_evaluate(model=HFLM(...), tasks=[...], limit=...)`, the keys `acc,none`, `acc_norm,none` | PyPI README, python-API docs (Nov 2026) |
| [trl_sft_reference.py](trl_sft_reference.py) | `trl`, `peft`, GPU | `SFTConfig` (`max_length`, `packing`, `completion_only_loss`, `assistant_only_loss`, `eos_token`, `chat_template_path`) and `SFTTrainer(peft_config=...)` | TRL v1.17.1 docs |
| [trl_dpo_reference.py](trl_dpo_reference.py) | `trl`, `peft`, GPU | `DPOConfig(beta=0.1, loss_type=["sigmoid"], max_length, precompute_ref_log_probs)` and `DPOTrainer` | TRL v1.17.1 docs |
| [trl_grpo_reference.py](trl_grpo_reference.py) | `trl`, GPU, optional `trl[vllm]` | `GRPOConfig(num_generations, beta, loss_type, scale_rewards, epsilon, max_completion_length)`, reward functions `f(completions, **kwargs) -> list[float]`, extra dataset columns as keyword arguments | TRL v1.17.1 docs |
| [unsloth_reference.py](unsloth_reference.py) | `unsloth`, NVIDIA GPU | `FastLanguageModel.from_pretrained(..., load_in_4bit=True)` and `get_peft_model` | not re-verified (check docs.unsloth.ai) |

## Syntax explained
| Syntax | Meaning |
|---|---|
| `BENCH = [(name, measures, scoring, status, ev), ...]` | a list of 5-tuples; each row unpacked in the `for` loop |
| `f"{name:<24}{ev:<5}"` | left-align in a field of 24 and 5 characters |
| `importlib.util.find_spec(mod)` | returns a spec if the module can be imported, else `None`; no import side effects |
| `Counter(b[4] for b in BENCH)` | counts how many rows carry each evidence tag |
| `abs(a - b) / se` | a z statistic: how many standard errors apart the two accuracies are |

## Libraries & functions
| Library / function | What it does here |
|---|---|
| `importlib.util.find_spec` | tells you which of the libraries in the tools table are installed |
| `collections.Counter` | tally of evidence tags |
| `math.sqrt` | standard error of the difference between two proportions |

## Execution trace (real run)
```
benchmark               ev   scoring                                 status Nov 2026
MMLU                    S    accuracy                                saturated for frontier models; still a sanity check for small models
                              measures: 57-subject school-to-professional knowledge, 4-choice MCQ
MMLU-Pro                S    accuracy                                frontier models cluster near the top; low discrimination at the high end
                              measures: harder MMLU, 10 choices, more reasoning
GPQA Diamond            K+S  accuracy                                near saturation; label noise limits the ceiling; ties within CI
                              measures: 198 graduate-level science MCQs written to be search-proof
HLE                     V    accuracy (exact-match / judge)          headroom remains; score depends on tools and subset, check leaderboard
                              measures: Humanity's Last Exam: 2,500 expert questions (Nature, Jan 2026)
AIME 2024/25/26         S    pass@1 or avg over samples              treated as solved by frontier reasoning models; 30 items = huge variance
                              measures: competition math, 30 problems per year, integer answers 0-999
MATH-500                S    exact match after normalisation         widely reported as saturated for reasoning models
                              measures: 500-problem subset of MATH, free-form answers
LiveCodeBench           K+S  pass@1 on a date window                 still used because new problems limit contamination
                              measures: fresh competitive-programming problems with release dates
SWE-bench Verified      V    % resolved                              OpenAI stopped reporting it (Feb 2026): contamination + flawed tests
                              measures: 500 human-validated real GitHub issues, patch must pass tests
SWE-bench Pro           V+S  % resolved                              recommended successor; a May 2026 audit claims grader errors (unconfirmed)
                              measures: longer-horizon, harder repo tasks (Scale AI)
Terminal-Bench          V    % tasks passed                          active; newer versions may exist, check tbench.ai
                              measures: agent solves tasks in a real shell (2.0: 89 tasks, ICLR 2026)
tau-bench / tau2-bench  V    pass^k (all k trials succeed)           active; pass^k exposes unreliability that pass@1 hides
                              measures: tool-agent-user dialogues (retail, airline, telecom ...) with policies
BFCL V4                 V    weighted accuracy                       active; an Epoch AI review (Sep 2026) rates it 'Flawed' on sampled items
                              measures: function calling incl. agentic web search, memory, format sensitivity
IFEval                  K    strict/loose prompt-level accuracy      status not checked in Nov 2026; rule-based so cheap and reproducible
                              measures: verifiable instructions ('under 100 words', 'include X')
Arena (ex-LMArena)      V+K  Bradley-Terry rating + CI               rebranded 'Arena' Jan 2026; style/length bias is the known weakness
                              measures: crowd pairwise votes on chat answers
RULER                   K    accuracy per length                     shows effective context < advertised context
                              measures: synthetic long-context tasks at configurable lengths
MTEB / MMTEB            K    mean of task metrics                    the standard embedding leaderboard (see module 14); check current version
                              measures: text embeddings across retrieval, STS, classification, clustering

evidence tags: {'S': 4, 'K+S': 2, 'V': 5, 'V+S': 1, 'K': 3, 'V+K': 1}   (V verified, K stable design fact, S secondary only)

library check:
  torch         installed 
  transformers  installed 
  peft          installed 
  trl           missing   pip install trl            # v1.x (PyPI showed 1.17.1 on 2026-10-30)
  lm_eval       missing   pip install lm_eval[hf]    # 0.4.13 on PyPI (2026-08-31)
  lighteval     missing   pip install lighteval
  inspect_ai    missing   pip install inspect-ai
  bitsandbytes  missing   pip install bitsandbytes  # CUDA/ROCm; needed for QLoRA
  unsloth       missing   pip install unsloth
  axolotl       missing   pip install axolotl
  llamafactory  missing   pip install llamafactory

one-liners:
  lm-evaluation-harness lm_eval --model hf --model_args pretrained=gpt2 --tasks hellaswag,arc_easy --batch_size 8 --limit 100
  list harness tasks    lm-eval ls tasks
  Axolotl               axolotl train config.yml                       # YAML holds base_model, datasets, adapter: lora ...
  LLaMA-Factory         llamafactory-cli train examples/train_lora/<your_sft>.yaml
  TRL SFT script        python trl_sft_reference.py
n=30    gap 0.60 vs 0.65 -> z = 0.40  (|z| > 1.96 needed)
n=500   gap 0.60 vs 0.65 -> z = 1.63  (|z| > 1.96 needed)
n=5000  gap 0.60 vs 0.65 -> z = 5.16  (|z| > 1.96 needed)
```
- The table lists 16 benchmarks; each row has the evidence tag next to it. Nine of the 16 rows rest at least partly on secondary sources (`S`, `K+S`) or only on design facts (`K`), so the table itself tells you where to double check.
- `evidence tags` count line: 5 rows `V`, 1 `V+S`, 1 `V+K`, 3 `K`, 2 `K+S`, 4 `S`.
- Library check: in this venv `torch`, `transformers` and `peft` are installed; `trl`, `lm_eval`, `bitsandbytes`, `unsloth`, `axolotl`, `llamafactory`, `lighteval`, `inspect_ai` are not, so the matching examples are `*_reference.py` files.
- z-test: a 5-point gap (0.60 vs 0.65) has z = 0.40 at n = 30, 1.63 at n = 500 and 5.16 at n = 5000. Only the last exceeds 1.96. The same gap is noise, borderline, then decisive, depending only on the number of items.

## Experiments (change ONE thing)
| Change | Real result | Lesson |
|---|---|---|
| `n` in the z-test loop: 30 / 500 / 5000 | z = 0.40 / 1.63 / 5.16 | sample size decides whether a gap is a result |
| add a row with a 4-field tuple | `ValueError` (see below) | rows must carry all five fields |
| remove a library from `TOOLS` | the `library check` list shrinks by one line | the check is data-driven |

## Debugging practice
**BREAK IT:** delete the evidence tag from one row (the file shown in the traceback is the temporary broken copy of `main.py`):
```
measures: agent solves tasks in a real shell (2.0: 89 tasks, ICLR 2026)
tau-bench / tau2-bench  V    pass^k (all k trials succeed)           active; pass^k exposes unreliability that pass@1 hides
                              measures: tool-agent-user dialogues (retail, airline, telecom ...) with policies
BFCL V4                 V    weighted accuracy                       active; an Epoch AI review (Sep 2026) rates it 'Flawed' on sampled items
                              measures: function calling incl. agentic web search, memory, format sensitivity
IFEval                  K    strict/loose prompt-level accuracy      status not checked in Nov 2026; rule-based so cheap and reproducible
                              measures: verifiable instructions ('under 100 words', 'include X')
Traceback (most recent call last):
  File "/Users/aman/Desktop/Project_resume/RAP_interview/17_finetuning_alignment_and_benchmarks/10_benchmark_landscape_2026_and_tools/_brk.py", line 39, in <module>
    for name, measures, scoring, status, ev in BENCH:
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: not enough values to unpack (expected 5, got 4)
```
How to read it: the failing line is the `for name, measures, scoring, status, ev in BENCH:` unpack; the message says the tuple has 4 values but 5 names were requested. Fix: restore the tag. The deeper point: a row without an evidence tag is exactly the kind of claim that should not be in an interview answer.

## Rebuild checklist
- [ ] Say what MMLU, GPQA Diamond, HLE, AIME, LiveCodeBench, SWE-bench, Terminal-Bench, tau-bench, BFCL, IFEval, Arena, RULER and MTEB measure, in one line each
- [ ] Say which are saturated and which are contaminated, and how you know
- [ ] Write the `lm_eval` command for two tasks with a limit
- [ ] Name the TRL trainers and the three config knobs that matter for each
- [ ] Explain pass^k

## The new problem (cliffhanger)
You know the measuring tools and the landscape. Time to use them on your own model: fine-tune it, evaluate before and after with confidence intervals, and check what you broke: [step 11](../11_end_to_end_report_card/WALKTHROUGH.md).

> **Say it aloud (interview box):** "I never quote benchmark scores from memory. I say what the benchmark measures, whether it is saturated or contaminated, and I report my own numbers with the harness, prompt and a confidence interval. For fine-tuning I use TRL, for evaluation lm-evaluation-harness."

