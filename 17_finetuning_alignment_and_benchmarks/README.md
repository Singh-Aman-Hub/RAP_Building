# Module 17 — Fine-tuning, Alignment and Benchmarks

The full post-training journey as practised in Nov 2026: from a pretrained base model to a model that follows instructions, matches preferences, reasons with reinforcement learning, and is then measured honestly. Every step runs on a laptop CPU with tiny seeded models and hand-written data, and every step also names the production tool (TRL, lm-evaluation-harness, Unsloth, Axolotl, LLaMA-Factory) that does the same thing at scale.

## The story of this module
[06_llm_training_and_lora](../06_llm_training_and_lora/README.md) taught you the training loop, perplexity, LoRA math and one `peft` LoRA run. This module starts where that stops.
- **Pain point:** a pretrained base model only continues text. It does not follow instructions, it does not know what a "good" answer is, and you cannot tell whether a change you made helped. "We fine-tuned it and it feels better" is not an engineering result.
- **Spark:** post-training is a pipeline of small, well-defined objectives. SFT teaches the format (cross-entropy on good answers). Preference tuning teaches taste (a loss on chosen vs rejected). Reinforcement learning with a verifier teaches skills you can check (the reward is code). Evaluation closes the loop: a number with a confidence interval, plus a check for what you broke.
- **What you build:** a chat-template inspector, an SFT loop with completion-only loss, every PEFT config on one model, a Bradley-Terry reward model, DPO from scratch with a frozen reference, GRPO from scratch on a verifiable task, distillation and model merging, a benchmark-scoring harness (perplexity, EM/F1, log-likelihood MCQ, pass@k, bootstrap CIs, contamination), an LLM-judge with position-bias test, Elo vs Bradley-Terry arena ranking, and a before/after report card.
- **Cliffhanger:** the report card tells you whether the model is better. Whether it is *worth serving* is a systems question: [10_ml_system_architecture](../10_ml_system_architecture/) and [09_api_development](../09_api_development/).

**Short on time?** Read [INTERVIEW_GIST.md](INTERVIEW_GIST.md) first.

## 1. Why this matters
Interviews for ML and LLM roles now ask: "How would you fine-tune a model for our task, and how would you know it worked?" Strong answers separate the stages (SFT, preference tuning, RL), say which loss each uses, give the memory arithmetic of LoRA vs full fine-tuning, and treat evaluation as an experiment: held-out data, a confidence interval, a regression check, a contamination check. Weak answers quote leaderboard numbers.

## 2. Where this fits
**Builds on**
- [06_llm_training_and_lora](../06_llm_training_and_lora/README.md): [dataset preparation](../06_llm_training_and_lora/01_dataset_preparation/), [the training loop](../06_llm_training_and_lora/02_training_loop/), [perplexity and checkpoints](../06_llm_training_and_lora/03_eval_perplexity_checkpoints/), [LoRA math](../06_llm_training_and_lora/04_lora_math_numpy/), [peft LoRA](../06_llm_training_and_lora/05_lora_peft_finetune/), [QLoRA reference](../06_llm_training_and_lora/06_qlora_reference/). This module links to those instead of repeating them.
- [04_llm_fundamentals](../04_llm_fundamentals/): tokenizers, logits, softmax, cross-entropy.
- [05_llm_parameters](../05_llm_parameters/): temperature and sampling (used when GRPO samples groups and when pass@k needs diverse samples).
- [07_prompt_engineering](../07_prompt_engineering/) and [shared/llm_client.py](../shared/llm_client.py): the mock/local/Anthropic backend used for the judge in step 09.
- [14_rag_and_retrieval](../14_rag_and_retrieval/): fine-tuning changes behaviour, RAG adds knowledge; its [evaluation step](../14_rag_and_retrieval/09_rag_evaluation/) used hit@k and LLM-as-judge for retrieval.

**Leads to**
- [21_llm_inference_engineering](../21_llm_inference_engineering/INTERVIEW_GIST.md): serving what you trained (quantisation, batching, KV cache, serving engines).
- [10_ml_system_architecture](../10_ml_system_architecture/): versioning models, shadow evaluation, rollback.
- [13_capstone_project](../13_capstone_project/): a fine-tuned or evaluated component.

## 3. Concept map
```
 pretraining (next-token CE on web text)            -> base model: continues text
   mid-training (curated / synthetic / long context) -> stronger base
        |
        v   01 chat template: <|im_start|>user ... <|im_end|> turns, special tokens, which tokens carry loss
   SFT  (02)  loss = CE on completion tokens only (labels = -100 on prompt)        -> instruct model
        |       full fine-tune or PEFT (03): LoRA / QLoRA / DoRA / rsLoRA / IA3 / prompt & prefix tuning
        v
   preference tuning                                  data: (prompt, chosen, rejected)
      reward model (04): Bradley-Terry  -log sigmoid(r_chosen - r_rejected)
      DPO (05): -log sigmoid(beta * [(log pi/ref)_chosen - (log pi/ref)_rejected])   no reward model, no sampling
        |
        v
   RL  (06)   RLHF: PPO = policy + reward model + value net + reference + KL
              RLVR: GRPO = group of samples, advantage = (r - mean)/std, verifier is code, no value net
        |
        v   07 distillation (KL to a teacher) and merging (linear, task arithmetic, SLERP, TIES)
   evaluation: 08 metrics from scratch -> 09 judges and arenas -> 10 benchmark landscape and tools
        |
        v
   11 report card: metric, before, after, delta, 95% CI, regression probes (forgetting)
```

## 4. Study path
| Step | Folder | You learn | Run | Time |
|---|---|---|---|---|
| 1 | [01_post_training_map_and_chat_templates](01_post_training_map_and_chat_templates/) | the stages, base vs instruct vs reasoning models, Qwen2.5 chat template tokens, 4 dataset formats | `python main.py` | 45 min |
| 2 | [02_sft_from_scratch_with_loss_masking](02_sft_from_scratch_with_loss_masking/) | SFT loop, labels = -100, padding vs packing, before/after | `python main.py` | 60 min |
| 3 | [03_peft_family](03_peft_family/) | the PEFT family (LoRA is derived in [06 step 04](../06_llm_training_and_lora/04_lora_math_numpy/)): rsLoRA, DoRA, IA3, prompt/prefix tuning, QLoRA, target modules, memory arithmetic | `python main.py` | 50 min |
| 4 | [04_reward_model_bradley_terry](04_reward_model_bradley_terry/) | scalar-head reward model, Bradley-Terry loss, held-out pair accuracy, shuffled-label control | `python main.py` | 50 min |
| 5 | [05_dpo_from_scratch](05_dpo_from_scratch/) | DPO loss with frozen reference, implicit rewards, margins, IPO/KTO/ORPO/SimPO | `python main.py` | 75 min |
| 6 | [06_ppo_vs_grpo_rlvr](06_ppo_vs_grpo_rlvr/) | PPO's parts, GAE, GRPO from scratch on a verifiable task, KL, clipping | `python main.py` | 75 min |
| 7 | [07_distillation_and_merging](07_distillation_and_merging/) | KD with temperature, linear / task-arithmetic / SLERP / TIES merging, LoRA adapter merging | `python main.py` | 60 min |
| 8 | [08_evaluation_from_scratch](08_evaluation_from_scratch/) | perplexity (theory in [06 step 03](../06_llm_training_and_lora/03_eval_perplexity_checkpoints/)), EM/F1, log-likelihood MCQ (acc vs acc_norm), pass@k, bootstrap CIs, contamination | `python main.py` | 90 min |
| 9 | [09_llm_as_judge_and_arenas](09_llm_as_judge_and_arenas/) | pointwise vs pairwise judging, position bias, Elo vs Bradley-Terry | `python main.py` | 60 min |
| 10 | [10_benchmark_landscape_2026_and_tools](10_benchmark_landscape_2026_and_tools/) | what each benchmark measures and its Nov 2026 status; lm-eval-harness, TRL, Unsloth, Axolotl, LLaMA-Factory | `python main.py` | 45 min |
| 11 | [11_end_to_end_report_card](11_end_to_end_report_card/) | LoRA SFT, before/after report card with CIs, catastrophic forgetting | `python main.py` | 60 min |

Total about 10 hours with exercises. Each step has `main.py` (commented, runnable), `WALKTHROUGH.md` (story, theory, a line-by-line "Code, step by step" section, real output, experiments, a debugging drill) and `TERMS.md`. Then do [exercises.md](exercises.md); worked answers are in [solutions/](solutions/).

Run from the repo root with `HF_HUB_OFFLINE=1 .venv/bin/python 17_finetuning_alignment_and_benchmarks/<step>/main.py`. Cached models used: `distilgpt2`, `gpt2`, and the **tokenizer only** of `Qwen/Qwen2.5-0.5B` (its weights are not cached, so nothing here runs Qwen itself). Step 10 also ships read-only reference files: `lm_eval_harness_reference.py`, `trl_sft_reference.py`, `trl_dpo_reference.py`, `trl_grpo_reference.py`, `unsloth_reference.py`. They need libraries or a GPU that this venv does not have (no `trl`, no `bitsandbytes`).

**Shared files:** [common.py](common.py) holds the seeded datasets (`make_sentiment_data`, `make_preference_data`) and `seq_logprob`. [evalkit.py](evalkit.py) holds the finished step-08 metrics for reuse in step 11.

## 5. The concepts, one by one

### The post-training pipeline (step 01)
| Stage | Data | Loss / signal | Result |
|---|---|---|---|
| Pretraining | trillions of web/code tokens | next-token cross-entropy | base model (continues text) |
| Mid-training | curated, synthetic, long-context | same loss, different mix | stronger base |
| SFT | prompt + ideal answer | CE on answer tokens | instruct model |
| Preference tuning | (prompt, chosen, rejected) | DPO-style or reward-model loss | aligned tone, helpfulness, safety |
| RL | prompts + reward (model or verifier) | policy gradient with KL leash | reasoning / agentic skills |
| Evaluation | held-out tasks, judges, humans | metrics with CIs | ship / iterate decision |

Base models are for continuation and further training; instruct (chat) models have seen the chat template and SFT; reasoning models are further trained with RL on verifiable tasks and emit long chains of thought before the answer. The chat template is part of the model: using the wrong one is a silent quality bug.

### SFT and loss masking (step 02)
Loss = mean over *completion* tokens of `-log p(token | previous tokens)`. Prompt tokens get label `-100`, which `cross_entropy(ignore_index=-100)` skips. Why: you want to learn the answer, not to memorise the question template, and prompt tokens otherwise dilute the loss. Packing concatenates short examples into full-length blocks to avoid padding waste, at the cost of needing document-aware attention masking. TRL: `completion_only_loss` for prompt/completion data, `assistant_only_loss` for messages data.

### PEFT (step 03; LoRA basics are in [06 step 04](../06_llm_training_and_lora/04_lora_math_numpy/) and [06 step 05](../06_llm_training_and_lora/05_lora_peft_finetune/))
| Method | What is trained | Typical trainable share | Note |
|---|---|---|---|
| LoRA | low-rank `B@A` added to chosen weights, scaled `alpha/r` | 0.1-2% | the default |
| rsLoRA | same, scaled `alpha/sqrt(r)` | same | keeps learning stable at high rank |
| DoRA | LoRA on the direction + a trained magnitude vector | slightly above LoRA | `use_dora=True` |
| QLoRA | LoRA on a 4-bit-quantised frozen base ([06 step 06](../06_llm_training_and_lora/06_qlora_reference/)) | same | needs bitsandbytes + GPU |
| IA3 | learned vectors that rescale activations | ~0.02% | tiny |
| Prompt / prefix tuning | virtual tokens / per-layer key-value prefix | 0.02-0.2% | no weight change |
Full fine-tuning with Adam in mixed precision costs about 16 bytes per parameter (weights, gradients, fp32 master copy, two Adam moments) before activations. LoRA keeps the base at 2 bytes and pays 16 bytes only for the adapter. The inference-side memory (weights plus KV cache) is worked out in [21 step 03](../21_llm_inference_engineering/03_gpu_memory_calculator/).

### Reward models and Bradley-Terry (step 04)
A reward model maps (prompt, answer) to a scalar. Bradley-Terry says `P(chosen beats rejected) = sigmoid(r_chosen - r_rejected)`; the loss is its negative log. Only differences matter, so the scale is arbitrary. Always evaluate on held-out pairs and run a shuffled-label control.

### DPO and friends (step 05)
`L = -log sigmoid( beta * [ log(pi(y_w|x)/ref(y_w|x)) - log(pi(y_l|x)/ref(y_l|x)) ] )`. The policy is secretly the reward model: the implicit reward is `beta * log(pi/ref)`. The reference is frozen and anchors the policy; `beta` controls how far it may drift. Variants differ in the loss shape (IPO), the data (KTO: unpaired thumbs), or the reference (ORPO and SimPO need none).

### PPO vs GRPO and RLVR (step 06)
RLHF with PPO needs four models (policy, frozen reference, reward model, value network) and uses GAE advantages, a clipped ratio and a KL penalty. GRPO drops the value network: sample G answers per prompt, advantage = (reward - group mean) / group std, same clipped ratio, KL to a reference. RLVR means the reward comes from a verifier (a unit test, an answer checker), not a learned model, so it cannot be reward-hacked the way a learned reward can (it can still be gamed if the checker is weak). TRL's GRPO exposes loss variants through `loss_type` (docs list `dapo` as default, `dr_grpo`, `sapo`) and `scale_rewards`.

### Distillation and merging (step 07)
Distillation trains a student on the teacher's softened distribution: `T^2 * KL(softmax(t/T) || softmax(s/T))`. Merging combines weights without training: linear average, task arithmetic (`base + lambda * sum(task vectors)`), SLERP (interpolate on a sphere), TIES (trim, elect sign, disjoint mean) to reduce interference.

### Evaluation from scratch (step 08)
Perplexity = `exp(mean CE)`. EM and token-F1 after normalisation. Multiple choice by log-likelihood: score each choice's log-probability given the question; raw sum favours short choices, so harnesses also report a length-normalised variant (`acc_norm`). pass@k uses an unbiased estimator `1 - C(n-c,k)/C(n,k)`. Always give a CI: with n=30 items a 0.05 gap is noise. Check contamination by n-gram overlap.

### Judges and arenas (step 09)
LLM judges have position bias, verbosity bias and self-preference. Swap the order and count a win only if it survives. Arena-style leaderboards turn pairwise votes into ratings: online Elo depends on vote order, a Bradley-Terry maximum-likelihood fit does not and supports bootstrap CIs.

### Benchmarks and tools (step 10)
Many classic benchmarks are saturated for frontier models; contamination retired SWE-bench Verified from OpenAI's reporting (Feb 2026). The table in step 10 tags every row with how well its status was verified. No scores are quoted: use the live leaderboards.

### The report card (step 11)
Metric, before, after, delta, CI, verdict, plus regression probes. A gain on the target task with a loss on general ability is a trade-off to report, not a win.

## 6. Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Fine-tuning | SFT, instruction tuning, supervised fine-tuning | SFT is a kind of fine-tuning | "Fine-tuning" is any further training; SFT = cross-entropy on (prompt, answer); instruction tuning = SFT on instruction data |
| Alignment | preference tuning, RLHF, post-training | overlapping | Alignment is the goal; RLHF/DPO are methods; "post-training" is everything after pretraining |
| DPO | offline preference optimisation, direct alignment algorithms | DPO is one of them | IPO, KTO, ORPO, SimPO are siblings with different losses |
| Reward model | RM, preference model, scoring model | usually yes | "Preference model" stresses it is trained from comparisons |
| RLHF / RLAIF / RLVR | human / AI feedback / verifiable rewards | no | Same RL machinery; the reward comes from humans, an AI judge, or code |
| PEFT | adapters, LoRA | LoRA is one PEFT method | PEFT = train a small number of parameters |
| Evals | benchmarks, test sets | near | A benchmark is a published eval with a fixed protocol |
| pass@1 vs accuracy | single-sample success | same when n=1 | pass@1 is the expected accuracy of one sample; accuracy usually means greedy or one fixed answer |
| Perplexity vs cross-entropy loss | exp(loss) | related | ppl = exp(mean CE in nats) |
| LLM-as-judge | auto-eval, model-graded eval | yes | A model scores answers instead of a human |
| Chat template | prompt template, message format | near | The exact string with special tokens the model was trained on |
| GRPO vs PPO | critic-free RL vs actor-critic RL | no | GRPO replaces the value network with the group mean |

## 7. Interview Q&A
**Q: What is the difference between SFT and preference tuning?** SFT maximises the likelihood of a good answer. Preference tuning uses pairs and pushes up the better answer relative to the worse one, teaching what *not* to say.
**Q: Why mask the prompt in SFT?** The model should learn to produce answers, not reproduce the prompt template, and prompt tokens would dilute the gradient. Set labels to -100.
**Q: Write the DPO loss.** `-log sigmoid(beta * ((log pi_c - log ref_c) - (log pi_r - log ref_r)))`. At step 0 policy equals reference so the loss is `ln 2`.
**Q: Why does DPO need a reference model?** It defines the implicit reward `beta*log(pi/ref)` and acts as the KL anchor. Without it (SimPO, ORPO) other regularisers are used.
**Q: PPO vs GRPO?** PPO needs a value network and GAE; GRPO uses the group-mean reward as baseline, so no critic, lower memory, ideal for verifiable rewards.
**Q: When would you pick LoRA over full fine-tuning?** Limited memory, many task variants (one adapter each), lower forgetting risk; full fine-tuning when you need large capability changes and have the compute.
**Q: How do you know a fine-tune helped?** Held-out task metric with a bootstrap CI, a paired comparison, regression probes on general ability, a contamination check.
**Q: Why is MCQ scoring by log-likelihood length-sensitive?** Longer choices have lower total log-probability, so raw `acc` favours short answers; `acc_norm` divides by length.
**Q: What is position bias in LLM judges and how do you fix it?** The judge prefers a slot. Run both orders and only count consistent verdicts.
**Q: Elo vs Bradley-Terry?** Elo updates online and depends on vote order; Bradley-Terry fits all votes at once by maximum likelihood, order-independent, and gives CIs by bootstrap.

## 8. Self-check
- Can you write the SFT loss with the -100 mask, the Bradley-Terry loss and the DPO loss from memory?
- Can you compute the memory of full fine-tuning vs LoRA vs QLoRA for a 7B model on paper?
- Can you explain why GRPO group-normalised advantages are zero when all samples in a group get the same reward?
- Can you say why a 5-point gap on a 30-item benchmark is not a result?
- Can you name two reasons a benchmark loses value (saturation, contamination) and one example from 2026?

## References (checked Nov 2026)
- TRL documentation, SFT / DPO / GRPO trainers (v1.17.1): https://huggingface.co/docs/trl/sft_trainer , https://huggingface.co/docs/trl/dpo_trainer , https://huggingface.co/docs/trl/grpo_trainer
- TRL on PyPI (1.17.1, released 2026-10-30, Python >= 3.11): https://pypi.org/project/trl/
- lm-evaluation-harness on PyPI (0.4.13, 2026-08-31) and Python API docs: https://pypi.org/project/lm-eval/ , https://lm-evaluation-harness.readthedocs.io/running_evals/python_api/
- DPO paper: https://arxiv.org/abs/2305.18290 ; IPO: https://arxiv.org/abs/2310.12036 ; SimPO: https://arxiv.org/abs/2405.14734
- OpenAI, "Why we no longer evaluate SWE-bench Verified" (2026-02-23): https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/
- Terminal-Bench paper (arXiv 2601.11868): https://arxiv.org/abs/2601.11868
- BFCL V4 (Gorilla blog and PyPI `bfcl-eval`): https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html , https://pypi.org/project/bfcl-eval/
- tau2-bench: https://github.com/sierra-research/tau2-bench
- Humanity's Last Exam (CAIS and Scale AI): https://lastexam.ai
- Evaluating LLMs trained on code (pass@k estimator), Chen et al. 2021: https://arxiv.org/abs/2107.03374
- Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (position bias, verbosity bias): https://arxiv.org/abs/2306.05685
- LoRA https://arxiv.org/abs/2106.09685 ; QLoRA https://arxiv.org/abs/2305.14314 ; DoRA https://arxiv.org/abs/2402.09353 ; rsLoRA https://arxiv.org/abs/2312.03732 ; IA3 https://arxiv.org/abs/2205.05638
- TIES-merging https://arxiv.org/abs/2306.01708 ; Task arithmetic https://arxiv.org/abs/2212.04089
- GRPO / DeepSeekMath https://arxiv.org/abs/2402.03300 ; PPO https://arxiv.org/abs/1707.06347
- Not re-verified in Nov 2026 (check before quoting): Unsloth API names (https://docs.unsloth.ai), Axolotl (`axolotl train config.yml`), LLaMA-Factory (`llamafactory-cli train <yaml>`), lighteval and inspect_ai CLI syntax.
