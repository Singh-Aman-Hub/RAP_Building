# Module 17 — Worked answers

Runnable scripts are in this folder (`.venv/bin/python 17_finetuning_alignment_and_benchmarks/solutions/<file>.py`). Numbers below come from real runs on a CPU.

## Step 01
- **Three-turn mask:** build the full string with the template, build the prompt prefix up to and including the LAST `<|im_start|>assistant\n`, and mask everything before it. Earlier assistant turns are also usually masked unless you use `assistant_only_loss` (which trains ALL assistant turns).
- **Break it:** with `add_generation_prompt=True` on a finished conversation the string ends with an empty open assistant header after the real answer. A model trained on that learns to start another answer after answering.

## Step 02
- **Packing (`s02_packing.py`):** 80 examples, 1,093 real tokens pack into 18 blocks of 64 (94.9% full); 17 of 18 blocks mix documents, so you need document-aware attention masks.
- **Why EOS:** otherwise the model never sees "the answer ends here" and keeps generating.

## Step 03
- **Memory (`s03_memory_calc.py`):** 7B full fine-tune 112 GB, LoRA about 15 GB, QLoRA about 5 GB before activations. On a 24 GB GPU only LoRA/QLoRA are possible, QLoRA leaves room for activations.
- **Shapes:** for a 768 -> 2304 layer at r=8: A is (8, 768), B is (2304, 8) (verified in step 03 output); trainable parameters grow linearly with r.

## Step 04
- **`s04_bt_check.py`:** equal rewards give `ln 2`; adding the same constant to both rewards changes nothing, so only differences matter; the wrong order costs far more than the right order saves.

## Step 05
- **`s05_dpo_loss_check.py`:** raising chosen by 5 nats and lowering rejected by 5 nats give the same loss (0.4741 at beta 0.1); a policy that lowers BOTH replies but lowers the rejected one more still reduces the loss (0.4032): that is likelihood displacement, so log `logp(chosen)` as well as the margin. Higher beta sharpens the loss: margin 5 gives 0.6685, 0.4741, 0.0067 for beta 0.01, 0.1, 1.0.

## Step 06
- **`s06_group_advantage.py`:** "all wrong" and "all right" groups have advantage 0 (no signal). "one right": +1.5 for the winner with std division, +0.75 without. Std division rescales per prompt (TRL exposes this choice as `scale_rewards`, and length normalisation through `loss_type`).

## Step 07
- **`s07_slerp_check.py`:** for two random high-dimensional unit vectors the angle is near 90 degrees, linear interpolation shrinks the norm to about 0.70, SLERP keeps 1.0. That is why SLERP is preferred when weights should keep their scale.

## Step 08
- **`s08_ci_coverage.py`:** at n=20, p=0.9 the percentile bootstrap covers only 84% of the time (target 95%) while Wilson covers 96%; at n=30 and n=200 both are close to 95%. Use Wilson (or exact) for small n and extreme accuracy.
- **`s08_pass_at_k_naive_vs_unbiased.py`:** the formula gives 0.9167, equal to the exact value; one random subset per problem returns 0 or 1 (std 0.27), which averages out only over many problems.

## Step 09
- **`s09_judge_swap.py`:** with bias 3 the judge says "A" 6 of 6 times and is consistent after the swap 0 of 3 times; with bias 0 or 1 it is consistent 3 of 3. Position bias is only visible if you swap.

## Steps 10 and 11
- **Step 10:** a score without a source should carry no tag; the evidence tags (V, K, S) exist to force that question.
- **Step 11:** if the CI of the delta on the target task excludes 0 and no regression probe has a CI below 0, ship. If general ability regresses beyond your tolerance, try a smaller learning rate, fewer epochs, a lower rank, or replay of general data.
