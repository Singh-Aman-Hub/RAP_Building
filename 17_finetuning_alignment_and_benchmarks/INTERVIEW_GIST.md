# Interview Gist — Fine-tuning, Alignment and Benchmarks (Module 17)
### Base model -> SFT -> preference tuning -> RL -> evaluation -> report card

One page of theory for revision, then a quick-fire Q&A. Detailed treatment: [README.md](README.md).

## 0. The pipeline in one breath
```
base (next-token CE)  --SFT (CE on answers only)-->  instruct
                      --DPO / reward model (pairs)-->  aligned
                      --GRPO / PPO (reward or verifier, KL leash)-->  reasoning
                      --evaluate: metric + CI + regression + contamination-->  ship or iterate
```
> "Post-training is a sequence of small objectives. SFT teaches the format, preference tuning teaches taste, RL with a verifier teaches checkable skills, and evaluation is an experiment, not a vibe."

## 1. Chat templates and data formats
- A chat template turns messages into one string with special tokens. Qwen2.5 (ChatML style): `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`. A default system prompt is inserted if you give none.
- `add_generation_prompt=True` ends with the open assistant header: use it at inference, not in training data.
- Formats: prompt/completion; messages; preference `{prompt, chosen, rejected}`; verifiable `{prompt, ground_truth}`.
- Using the wrong template for a model is a silent quality bug.

## 2. SFT
- Loss = mean over completion tokens of `-log p`. Labels = -100 on the prompt (`ignore_index`). Append EOS so the model learns to stop.
- Padding wastes positions; packing fills fixed-length blocks (needs per-document attention masks).
- LR: about 1e-5 to 2e-5 full fine-tune, about 1e-4 to 2e-4 for LoRA.
- Evaluate on held-out data: SFT on 80 examples can drive training loss to ~0 by memorising.

## 3. PEFT and memory
- LoRA: `W + (alpha/r) * B@A`, train A and B only. rsLoRA scales by `alpha/sqrt(r)`. DoRA adds a magnitude vector. IA3 rescales activations. Prompt/prefix tuning train virtual tokens.
- Target modules: attention projections at minimum (`q,k,v,o`); "all-linear" is the common strong default.
- Memory per parameter, mixed-precision Adam: 16 bytes full fine-tune; LoRA = 2 bytes frozen base + 16 bytes x trainable share; QLoRA = about 0.5-0.56 bytes base (4-bit) + the same adapter cost. A 7B model: ~112 GB full, ~15 GB LoRA, ~5 GB QLoRA (plus activations). Inference-side memory: [21 step 03](../21_llm_inference_engineering/03_gpu_memory_calculator/WALKTHROUGH.md).

## 4. Reward model and Bradley-Terry
- `P(chosen > rejected) = sigmoid(r_c - r_r)`; loss = `-log sigmoid(r_c - r_r)`; starts at `ln 2 = 0.693`.
- Scalar head on the last real token's hidden state. Only differences matter. Test: held-out pair accuracy; control: shuffled labels must NOT reproduce the real accuracy (expect roughly chance; on 30 test pairs the recorded control scored 0.13-0.33, below 0.5, because the random swap was not perfectly balanced).

## 5. DPO
- `L = -log sigmoid(beta * [(log pi_c - log ref_c) - (log pi_r - log ref_r)])`; implicit reward `r = beta * log(pi/ref)`.
- Step 0: policy = reference, margin 0, loss `ln 2`. Watch: `rewards/chosen`, `rewards/rejected`, `margins`, `accuracies`, `logps/chosen` (if it falls a lot, you are over-training: likelihood displacement).
- Variants: IPO (squared loss to margin 1/(2 beta)), KTO (unpaired good/bad), ORPO (SFT + odds ratio, no reference), SimPO (length-normalised log-prob, target margin, no reference).
- Start from the SFT model. beta higher = stay closer to the reference.

## 6. PPO vs GRPO vs RLVR
- PPO (RLHF): policy, frozen reference, reward model, value network; GAE advantages; clipped ratio `min(r*A, clip(r,1-eps,1+eps)*A)`; KL penalty.
- GRPO: sample G completions per prompt; `A_i = (r_i - mean(r)) / std(r)`; same clipped objective; KL to reference; no value network. If every sample in a group gets the same reward the advantage is 0 and the group teaches nothing.
- RLVR: reward = code (answer checker, unit tests). RLAIF: reward = an AI judge. RLHF: reward model from human preferences.
- TRL GRPO (v1.17.1 docs): `num_generations=8`, `beta=0.0` default (no reference loaded), `loss_type="dapo"` default (`dr_grpo`, `sapo` also listed), `scale_rewards="group"`, `epsilon=0.2`.

## 7. Distillation and merging
- KD: `alpha*T^2*KL(softmax(t/T) || softmax(s/T)) + (1-alpha)*CE`. Temperature exposes the teacher's "dark knowledge" about wrong-but-plausible tokens.
- Merging: linear average; task arithmetic `base + lambda*sum(tau_i)`; SLERP on the sphere; TIES = trim small deltas, elect a sign per coordinate, disjoint mean. Merging fine-tunes of the same base only.

## 8. Evaluation mechanics
- Perplexity `= exp(mean CE)`. EM and token-F1 after normalising (lowercase, strip punctuation and articles).
- MCQ by log-likelihood: score `log P(choice | question)`. Raw sum prefers short choices; `acc_norm` divides by the length of the choice text (this module uses bytes; check the harness task for characters vs bytes). MMLU-style letter format scores " A/B/C/D".
- pass@k unbiased: `1 - C(n-c,k)/C(n,k)`. Do not estimate by taking the first k samples.
- CI: bootstrap items. n=30 gives a CI about +-0.18 wide at p=0.5. Compare systems with a paired bootstrap on the same items.
- Contamination: n-gram overlap of test items with training text; for closed models, use fresh/dated sets.

## 9. Judges and arenas
- Pointwise (rubric, JSON score) vs pairwise (A or B). Known biases: position, verbosity, self-preference. Fix: swap order, count only consistent verdicts, calibrate against human labels.
- Elo: online, order-dependent. Bradley-Terry: MLE over all votes, bootstrap CIs.

## 10. Landscape (Nov 2026; verify live)
- Saturated for frontier: MMLU, MMLU-Pro (near ceiling), GPQA Diamond (near), AIME, MATH-500.
- Contaminated / retired: SWE-bench Verified (OpenAI stopped reporting, Feb 2026); SWE-bench Pro recommended.
- Active: HLE, Terminal-Bench, tau2-bench (pass^k), BFCL V4, LiveCodeBench, Arena, RULER, MTEB.
- Tools: lm-evaluation-harness (`lm_eval --model hf --tasks ...`), lighteval, inspect_ai; training: TRL, Unsloth, Axolotl, LLaMA-Factory.

## 11. Cheat sheet
| Need | Use |
|---|---|
| Teach a format | SFT, completion-only loss |
| Cheap adaptation | LoRA / QLoRA |
| Teach tone / safety from pairs | DPO (or reward model + RL) |
| Teach math / code with a checker | GRPO with a verifiable reward |
| Compare two models | paired bootstrap + CI |
| Did we break general ability? | regression probe + general perplexity |

## 12. Quick-fire Q&A
1. **What do you set labels to for prompt tokens?** -100.
2. **Loss at the start of DPO?** `ln 2`, because policy = reference.
3. **What does beta do in DPO?** Strength of the KL anchor: larger = closer to reference.
4. **Why is GRPO cheaper than PPO?** No value network.
5. **What is `scale_rewards="group"`?** Divide advantages by the group std.
6. **Why can't you trust accuracy on 30 items?** CI too wide.
7. **What does acc_norm fix?** Length bias in raw log-likelihood.
8. **How many bytes per parameter for full fine-tune with Adam?** About 16 (mixed precision).
9. **Which merging method resolves sign conflicts?** TIES.
10. **Name a retired benchmark and why.** SWE-bench Verified: contamination and flawed tests (OpenAI, Feb 2026).
11. **Position bias fix?** Swap order, require agreement.
12. **Elo vs BT?** Elo is order-dependent online; BT is a likelihood fit.
