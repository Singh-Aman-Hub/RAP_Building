"""
READ, NOT RUN.  NEEDS: pip install "lm_eval[hf]"  (PyPI showed lm-eval 0.4.13 on 2026-08-31; check PyPI for the current one).
No GPU is strictly required for gpt2, but downloading datasets for real tasks needs network access.

GOAL: show the two ways to run EleutherAI's lm-evaluation-harness: the CLI and the Python API, and where the scoring we built by hand
      in step 08 lives (log-likelihood multiple choice -> acc / acc_norm).
"""

# ---------------------------------------------------------------- CLI (verified in the PyPI README)
CLI = """
pip install "lm_eval[hf]"
lm-eval ls tasks                                   # list available tasks
lm_eval --model hf \\
    --model_args pretrained=EleutherAI/gpt-j-6B \\
    --tasks hellaswag \\
    --device cuda:0 \\
    --batch_size 8
# useful flags (check `lm_eval --help` for your version): --num_fewshot 5  --limit 100  --output_path results/  --log_samples
"""

# ---------------------------------------------------------------- Python API
def run():
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained="gpt2", batch_size=16)                 # wraps an HF causal LM once, reuse it for many tasks
    results = lm_eval.simple_evaluate(                          # the "most common entry point" in the docs
        model=lm,                                               # or model="hf", model_args="pretrained=gpt2,dtype=float32"
        tasks=["arc_easy", "hellaswag"],
        num_fewshot=0,
        limit=100,                                              # first 100 items per task: a smoke test, NOT a benchmark result
    )
    for task, metrics in results["results"].items():
        # typical keys: "acc,none", "acc_stderr,none", "acc_norm,none"  (acc_norm = length-normalised, see step 08)
        print(task, metrics)

# ---------------------------------------------------------------- what the harness does for a multiple-choice task
# 1. builds the prompt (few-shot examples + question)
# 2. for each choice computes log P(choice | prompt) with the model   -> `acc`
# 3. divides by choice length in bytes (or tokens for some tasks)     -> `acc_norm`
# 4. reports accuracy and a standard error (bootstrap or analytic)
# Generative tasks (gsm8k, ...) use `generate_until` with stop sequences and a filter that extracts the answer.

# ---------------------------------------------------------------- other evaluation frameworks (mention, check docs before use)
OTHERS = """
lighteval  (Hugging Face)   : CLI + python, task strings such as "suite|task|num_fewshot"; check `lighteval --help` for the current syntax
inspect_ai (UK AISI)        : `inspect eval <task> --model <provider/model>`; strong for agentic / tool-use and model-graded evals
"""
if __name__ == "__main__":
    print(CLI)
    print(OTHERS)
