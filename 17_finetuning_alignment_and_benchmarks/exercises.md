# Module 17 — Exercises

Rules: change ONE thing at a time, predict the result before running, write down what happened. Worked answers are in [solutions/solutions.md](solutions/solutions.md) with runnable code in [solutions/](solutions/) (marked **[sol]**). Do not look until you have tried. Most `main.py` files read knobs from environment variables, for example `LR=1e-4 python main.py`.

## Step 01 — post-training map and chat templates
- **Modify:** add a system message to the conversation and print the template again. Which tokens changed, and how many?
- **Modify:** build a three-turn conversation (user, assistant, user) and mark the SFT loss mask so ONLY the assistant turn is trained. **[sol]** in solutions.md.
- **Break it:** apply the template with `add_generation_prompt=True` to a FULL conversation and read the end of the string. Why would training on this be wrong?
- **Rebuild:** write the four dataset formats from memory and convert a `messages` row into a prompt/completion row.
- **Explain aloud:** base vs instruct vs reasoning model, in three sentences.

## Step 02 — SFT with loss masking
- **Modify:** `EPOCHS=1`, then `EPOCHS=6`. Record generated exact-match and held-out loss. Where does training loss keep falling but held-out stops improving?
- **Modify:** `LR=5e-4`. What happens, and why does a full fine-tune want a small LR?
- **Break it:** use `finetune(False)` only (train on the prompt too). Compare held-out loss and generated EM with completion-only.
- **Rebuild:** write `collate` from memory with right padding and -100 labels. **[sol]** `solutions/s02_packing.py` for the packing version.
- **Explain aloud:** why append EOS to the completion?

## Step 03 — PEFT family
- **Modify:** change `r` from 8 to 2 and to 128 in the first config. Plot trainable parameters against r (a line: why?).
- **Modify:** run the memory calculator for a 13B model with `frac=0.02`. **[sol]** `solutions/s03_memory_calc.py`.
- **Break it:** remove `fan_in_fan_out=True` for GPT-2. What does peft say, and why does GPT-2's `Conv1D` need it?
- **Rebuild:** write down, without looking, the shapes of A and B for a 768 -> 2304 layer at r=8.
- **Explain aloud:** LoRA vs QLoRA vs full fine-tune for a 7B model on one 24 GB GPU.

## Step 04 — reward model
- **Modify:** `EPOCHS=1` and `LR=2e-4`. Which is better, and why can a high LR hurt a pretrained backbone?
- **Modify:** `SHUFFLE_LABELS` is already in the control section; change the flip probability from 0.5 to 0.2. How does held-out accuracy move?
- **Break it:** always read the hidden state at position -1 with right padding. What are you scoring now?
- **Rebuild:** write the Bradley-Terry loss and verify the three facts in **[sol]** `solutions/s04_bt_check.py`.
- **Explain aloud:** why is only the reward *difference* meaningful?

## Step 05 — DPO
- **Modify:** `BETA=0.5`, then `BETA=0.02`. Compare margin, accuracy and `logp(chosen)`.
- **Modify:** `LOSS=ipo`, then `LOSS=simpo`. Which keeps `logp(chosen)` highest?
- **Break it:** `LR=5e-4`. Read what happens to `logp(chosen)`. This is likelihood displacement. **[sol]** `solutions/s05_dpo_loss_check.py` shows it with numbers.
- **Rebuild:** write the DPO loss from memory given four log-probs.
- **Explain aloud:** why does the reference model exist, and which variants remove it?

## Step 06 — PPO vs GRPO
- **Modify:** `G=4`, then `G=32`. Look at "groups w/ signal". Why does a small group waste samples?
- **Modify:** `SCALE_STD=0`. **[sol]** `solutions/s06_group_advantage.py` shows the advantage numbers.
- **Break it:** `BETA=0 EPS=100 LR=3e-2`. Compare the KL column.
- **Rebuild:** write the clipped objective for one token from memory.
- **Explain aloud:** why does GRPO not need a value network, and what replaces it?

## Step 07 — distillation and merging
- **Modify:** `TEMP=1` vs `TEMP=4`. Which student has the lowest KL to the teacher?
- **Modify:** change TIES `density` from 0.5 to 0.2 in the toy. **[sol]** `solutions/s07_slerp_check.py` explains why SLERP differs from linear in high dimensions.
- **Break it:** merge two LoRA adapters of different ranks with `combination_type="linear"` and read the error.
- **Rebuild:** write `kd_loss` from memory.
- **Explain aloud:** why multiply the KL by T squared?

## Step 08 — evaluation
- **Modify:** `MODEL=distilgpt2`. Which scoring method changes the most?
- **Modify:** run **[sol]** `solutions/s08_ci_coverage.py`. When does the percentile bootstrap undercover?
- **Break it:** change `np.arange(n - c + 1, n + 1)` to `np.arange(n - c, n + 1)` in `pass_at_k` and compare with the Monte-Carlo line. Which check would have caught it without the simulation?
- **Rebuild:** write `bootstrap_ci` and `token_f1` from memory. **[sol]** `solutions/s08_pass_at_k_naive_vs_unbiased.py`.
- **Explain aloud:** why `acc_norm` exists.

## Step 09 — judges and arenas
- **Modify:** `POSITION_BIAS=0`, then `POSITION_BIAS=1`. **[sol]** `solutions/s09_judge_swap.py` measures bias for any judge function.
- **Modify:** change the number of simulated votes from 1500 to 200. What happens to the BT bootstrap intervals?
- **Break it:** `POSITION_BIAS=5`. Why does naive accuracy collapse to the share of good-first cases?
- **Rebuild:** write the Elo update from memory.
- **Explain aloud:** why does the Arena fit Bradley-Terry instead of online Elo?

## Step 10 — benchmark landscape
- **Modify:** add a benchmark you use at work to the table with an evidence tag.
- **Modify:** in the z-test helper try n=100 for a 0.60 vs 0.65 gap.
- **Break it:** quote a leaderboard score without a source. Which tag should it carry?
- **Rebuild:** write the `lm_eval` CLI line for `arc_easy` and `hellaswag` with a 100-item limit from memory.
- **Explain aloud:** saturation vs contamination, with one 2026 example of each.

## Step 11 — report card
- **Modify:** add a third recipe (`r=2`, `lr=1e-4`) and add it as a third card.
- **Modify:** mix 30 lines of general text into the SFT data (replay) and re-measure perplexity.
- **Break it:** evaluate on the training reviews. What does the report card claim now?
- **Rebuild:** write the paired bootstrap from memory.
- **Explain aloud:** what would make you ship, hold, or roll back this model?
