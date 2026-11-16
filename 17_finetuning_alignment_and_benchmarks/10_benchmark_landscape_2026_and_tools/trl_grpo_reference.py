"""
READ, NOT RUN.  NEEDS: pip install trl datasets (+ "trl[vllm]" for use_vllm=True) and a GPU.  Checked against the TRL v1.17.1 GRPO docs
(huggingface.co/docs/trl/grpo_trainer, Nov 2026): defaults num_generations=8, beta=0.0, loss_type="dapo", scale_rewards="group",
epsilon=0.2, max_completion_length=512.  Re-check for your installed version.

GOAL: the production version of step 06: GRPO with a verifiable reward function (RLVR).
"""
import re
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

# Dataset needs a "prompt" column. Every OTHER column (here: answer) is passed to reward functions as a keyword argument.
train = Dataset.from_list([
    {"prompt": [{"role": "user", "content": "What is 17 + 25? Reply with the final number in \\boxed{}."}], "answer": "42"},
    # ...
])

def correctness_reward(completions, answer, **kwargs):
    """Return one float per completion. completions[i] is a list of messages for conversational data, so read [0]["content"]."""
    out = []
    for comp, gold in zip(completions, answer):
        m = re.search(r"\\boxed\{([^}]*)\}", comp[0]["content"])
        out.append(1.0 if m and m.group(1).strip() == gold else 0.0)       # verifiable: code, not a reward model
    return out

def format_reward(completions, **kwargs):
    return [0.2 if "\\boxed{" in c[0]["content"] else 0.0 for c in completions]   # small shaping reward; multiple funcs are summed

args = GRPOConfig(
    output_dir="out-grpo",
    num_generations=8,              # group size G (the effective batch size must be divisible by it)
    beta=0.0,                       # KL coefficient; 0.0 means the reference model is not even loaded; DeepSeek-R1 used 0.001
    loss_type="dapo",               # docs list "dapo" (default), "dr_grpo", "sapo"
    scale_rewards="group",          # divide advantages by the per-group std ("batch" and False are the alternatives)
    epsilon=0.2,                    # PPO clip range
    max_completion_length=512,
    use_vllm=False,                 # True: much faster sampling; needs trl[vllm]
    learning_rate=1e-6,
    report_to="none",
)
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    args=args,
    reward_funcs=[correctness_reward, format_reward],
    train_dataset=train,
)
trainer.train()
