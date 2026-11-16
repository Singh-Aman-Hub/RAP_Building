# Terms — The benchmark landscape and tools

Folder: `17_finetuning_alignment_and_benchmarks/10_benchmark_landscape_2026_and_tools` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [17 README](../README.md)

**Terms in this folder:** [Benchmark saturation](#benchmark-saturation) · [Benchmark contamination](#benchmark-contamination) · [Evidence tags (V, K, S)](#evidence-tags-v-k-s) · [MMLU and MMLU-Pro](#mmlu-and-mmlu-pro) · [GPQA Diamond](#gpqa-diamond) · [Humanity's Last Exam (HLE)](#humanitys-last-exam-hle) · [AIME and MATH-500](#aime-and-math-500) · [LiveCodeBench](#livecodebench) · [SWE-bench Verified and Pro](#swe-bench-verified-and-pro) · [Terminal-Bench](#terminal-bench) · [tau-bench and pass^k](#tau-bench-and-passk) · [BFCL (Berkeley Function Calling Leaderboard)](#bfcl-berkeley-function-calling-leaderboard) · [IFEval](#ifeval) · [RULER](#ruler) · [MTEB / MMTEB](#mteb--mmteb) · [lm-evaluation-harness](#lm-evaluation-harness) · [lighteval and Inspect](#lighteval-and-inspect) · [TRL (SFTTrainer, DPOTrainer, GRPOTrainer)](#trl-sfttrainer-dpotrainer-grpotrainer) · [Unsloth, Axolotl, LLaMA-Factory](#unsloth-axolotl-llama-factory) · [importlib.util.find_spec](#importlibutilfind_spec) · [Two-proportion z-test](#two-proportion-z-test) · [ValueError: not enough values to unpack](#valueerror-not-enough-values-to-unpack)

---

## Benchmark saturation  `(concept)`
**1. Definition:** Top models are near the maximum score so differences fall inside noise.
**2. In plain words:** The test is too easy to rank leaders.
**3. Example to quote:** > "AIME and MATH-500 are widely reported as saturated; check live leaderboards."
**Seen in:** `main.py`, table

## Benchmark contamination  `(concept)`
**1. Definition:** Test items appear in training data, inflating scores.
**2. In plain words:** The student saw the exam.
**3. Example to quote:** > "OpenAI stopped reporting SWE-bench Verified (2026-02-23) citing contamination and flawed tests."
**Seen in:** `WALKTHROUGH.md` theory

## Evidence tags (V, K, S)  `(concept)`
**1. Definition:** This module's labels: V verified in a Nov 2026 web check, K stable design fact, S secondary source only.
**2. In plain words:** Where did I learn this?
**3. Example to quote:** > "No score is quoted without a source."
```python
('MMLU', ..., 'S')
```
**Seen in:** `main.py`, `BENCH`

## MMLU and MMLU-Pro  `(concept)`
**1. Definition:** 57-subject multiple-choice knowledge test, and a harder 10-choice version.
**2. In plain words:** General knowledge exam.
**3. Example to quote:** > "Frontier models cluster near the top; useful mostly for small models."
**Seen in:** `main.py`, table

## GPQA Diamond  `(concept)`
**1. Definition:** 198 graduate-level science multiple-choice questions designed to resist search.
**2. In plain words:** Expert science questions.
**3. Example to quote:** > "Near saturation per secondary sources; label noise limits the ceiling."
**Seen in:** `main.py`, table

## Humanity's Last Exam (HLE)  `(concept)`
**1. Definition:** 2,500 expert-written hard questions (CAIS and Scale AI; Nature, Jan 2026).
**2. In plain words:** A deliberately very hard exam.
**3. Example to quote:** > "Scores depend on tools and subset; check the leaderboard."
**Seen in:** `main.py`, table

## AIME and MATH-500  `(concept)`
**1. Definition:** AIME: 30 competition problems per year with integer answers; MATH-500: a 500-problem subset of MATH.
**2. In plain words:** Competition math.
**3. Example to quote:** > "30 items means high variance."
**Seen in:** `main.py`, table

## LiveCodeBench  `(concept)`
**1. Definition:** Competitive-programming problems with release dates so you can test on problems newer than the model's training data.
**2. In plain words:** A code exam that keeps getting new questions.
**3. Example to quote:** > "Date windows limit contamination."
**Seen in:** `main.py`, table

## SWE-bench Verified and Pro  `(concept)`
**1. Definition:** Real GitHub issues the model must fix so hidden tests pass; Verified has 500 human-validated tasks; Pro is a harder successor.
**2. In plain words:** Fix a real bug.
**3. Example to quote:** > "OpenAI recommended Pro after retiring Verified."
**Seen in:** `main.py`, table

## Terminal-Bench  `(concept)`
**1. Definition:** Agent tasks in a real command-line environment; 2.0 has 89 tasks (arXiv 2601.11868, ICLR 2026).
**2. In plain words:** Can the agent drive a shell?
**3. Example to quote:** > "Newer versions may exist: check tbench.ai."
**Seen in:** `main.py`, table

## tau-bench and pass^k  `(concept)`
**1. Definition:** Tool-agent-user conversations under domain policies (retail, airline, telecom ...); pass^k requires all k trials to succeed.
**2. In plain words:** Reliability for customer-facing agents.
**3. Example to quote:** > "pass^k drops quickly with k when the agent is inconsistent."
**Seen in:** `main.py`, table

## BFCL (Berkeley Function Calling Leaderboard)  `(concept)`
**1. Definition:** Function-calling benchmark; V4 adds agentic web search, memory and format sensitivity.
**2. In plain words:** Does the model call tools correctly?
**3. Example to quote:** > "An Epoch AI review (Sep 2026) flagged label problems in sampled items."
**Seen in:** `main.py`, table

## IFEval  `(concept)`
**1. Definition:** Prompts with programmatically checkable instructions such as word limits.
**2. In plain words:** Instruction following you can check with code.
**3. Example to quote:** > "Cheap and reproducible because no judge is needed."
**Seen in:** `main.py`, table

## RULER  `(concept)`
**1. Definition:** Synthetic long-context tasks at configurable lengths.
**2. In plain words:** Does the long context actually work?
**3. Example to quote:** > "Reveals that effective context is shorter than advertised."
**Seen in:** `main.py`, table

## MTEB / MMTEB  `(concept)`
**1. Definition:** Massive text embedding benchmark across retrieval, STS, classification and clustering.
**2. In plain words:** The embedding leaderboard.
**3. Example to quote:** > "Used for embedding model choice (module 14)."
**Seen in:** `main.py`, table

## lm-evaluation-harness  `(library)`
**1. Definition:** EleutherAI's open evaluation framework (CLI `lm_eval` and `lm_eval.simple_evaluate`).
**2. In plain words:** The standard benchmark runner.
**3. Example to quote:** > "PyPI showed 0.4.13 on 2026-08-31."
```python
lm_eval --model hf --model_args pretrained=gpt2 --tasks hellaswag
```
**Seen in:** `lm_eval_harness_reference.py`

## lighteval and Inspect  `(library)`
**1. Definition:** Alternative eval frameworks from Hugging Face and the UK AI Security Institute.
**2. In plain words:** Other benchmark runners.
**3. Example to quote:** > "Check their CLI help for current syntax."
**Seen in:** `lm_eval_harness_reference.py`

## TRL (SFTTrainer, DPOTrainer, GRPOTrainer)  `(library)`
**1. Definition:** Hugging Face's post-training library; v1.17.1 on PyPI on 2026-10-30.
**2. In plain words:** The reference training toolkit.
**3. Example to quote:** > "Argument names change between versions: check the docs for your version."
```python
from trl import SFTTrainer, DPOTrainer, GRPOTrainer
```
**Seen in:** `trl_*_reference.py`

## Unsloth, Axolotl, LLaMA-Factory  `(library)`
**1. Definition:** Fine-tuning stacks: Unsloth (memory/speed optimised), Axolotl (YAML), LLaMA-Factory (CLI/UI).
**2. In plain words:** Convenience wrappers around the same ideas.
**3. Example to quote:** > "`axolotl train config.yml` and `llamafactory-cli train <yaml>`."
**Seen in:** `main.py`, one-liners

## importlib.util.find_spec  `(syntax)`
**1. Definition:** Returns a module spec if the module can be imported, otherwise None.
**2. In plain words:** Is this package installed?
**3. Example to quote:** > "find_spec('trl') returns None when the package is missing, so I can skip gracefully instead of crashing on import."
```python
importlib.util.find_spec('trl') is not None
```
**Seen in:** `main.py`, section 3

## Two-proportion z-test  `(concept)`
**1. Definition:** z = |a-b| / sqrt(2*p*(1-p)/n) for independent samples.
**2. In plain words:** Is the gap bigger than noise?
**3. Example to quote:** > "0.60 vs 0.65 gives z=0.40 at n=30 and 5.16 at n=5000."
**Seen in:** `main.py`, section 5

## ValueError: not enough values to unpack  `(error)`
**1. Definition:** Tuple unpacking with more names than elements.
**2. In plain words:** A row is missing a field.
**3. Example to quote:** > "expected 5, got 4."
```python
for a, b, c, d, e in rows:
```
**Seen in:** `WALKTHROUGH.md`, debugging practice
