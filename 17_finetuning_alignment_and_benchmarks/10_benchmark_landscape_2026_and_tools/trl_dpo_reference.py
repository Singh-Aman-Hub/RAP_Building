"""
READ, NOT RUN.  NEEDS: pip install trl peft datasets   (TRL v1.x; checked against the v1.17.1 DPO docs, Nov 2026).  GPU recommended.

GOAL: the production version of step 05.  TRL keeps the frozen reference model for you: with ref_model=None it uses the
      initial policy; with a PEFT adapter it uses the base model with the adapter disabled (so no second copy in memory).
"""
from datasets import Dataset
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

# preference format: explicit prompt + chosen + rejected (standard text or conversational lists of messages)
train = Dataset.from_list([
    {"prompt": [{"role": "user", "content": "How do I reset my password?"}],
     "chosen": [{"role": "assistant", "content": "Sure, open Settings > Account > Reset password."}],
     "rejected": [{"role": "assistant", "content": "Figure it out yourself."}]},
])

args = DPOConfig(
    output_dir="out-dpo",
    beta=0.1,                    # default 0.1: higher beta = stay closer to the reference
    loss_type=["sigmoid"],       # default; others from the docs: "hinge", "ipo", "bco_pair", "sppo_hard", "apo_zero", "sigmoid_norm" (SimPO-style length norm) ...
    learning_rate=5e-6,          # default is 1e-6; the docs suggest about 1e-5 when training adapters
    max_length=1024,             # truncation of prompt+completion
    precompute_ref_log_probs=False,   # True saves memory: computes reference log-probs once before training
    num_train_epochs=1,
    per_device_train_batch_size=4,
    report_to="none",
)
trainer = DPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",     # start from the SFT model, not the raw base
    args=args,
    train_dataset=train,
    peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"),
)
trainer.train()
# logged metrics to watch (names from the docs): rewards/chosen, rewards/rejected, rewards/margins, rewards/accuracies, logps/chosen
