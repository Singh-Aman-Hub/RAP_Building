<!-- vocabulary draft: 17_finetuning_alignment_and_benchmarks -->

### Post-training
<!-- cat: llm -->
- **Technical definition:** All training after pretraining: mid-training, SFT, preference tuning, reinforcement learning, and distillation.
- **Plain English:** The finishing school that turns a text-continuing base model into an assistant.
- **Example / analogy:** Pretraining reads the library; post-training teaches manners and skills.
- **Snippet:**
```python
print(['pretrain','SFT','preference','RL','eval'])
```
- **Snippet explained:** Lists the stages in order; each uses a different loss or reward.
- **Related terms:** [[Supervised fine-tuning (SFT, instruction tuning)]], [[Preference tuning]], [[RLVR (reinforcement learning with verifiable rewards)]]

### Chat template
<!-- cat: llm -->
- **Technical definition:** A Jinja template stored with the tokenizer that converts a list of role/content messages into the single string (with special tokens) the model was trained on.
- **Plain English:** The exact envelope format the model expects for each message.
- **Example / analogy:** Qwen2.5 wraps each turn as <|im_start|>role\n...<|im_end|>.
- **Snippet:**
```python
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
```
- **Snippet explained:** Builds the prompt string and ends with the open assistant header so the model starts answering.
- **Related terms:** [[Special tokens]], [[Generation prompt]], [[Supervised fine-tuning (SFT, instruction tuning)]]

### Generation prompt
<!-- cat: llm -->
- **Technical definition:** The trailing assistant header added by add_generation_prompt=True so generation starts at the assistant's reply.
- **Plain English:** The 'your turn' marker.
- **Example / analogy:** Used at inference; not included in training conversations that already contain the answer.
- **Snippet:**
```python
tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
```
- **Snippet explained:** Without it the model may continue the user turn.
- **Related terms:** [[Chat template]]

### Base model vs instruct model
<!-- cat: llm -->
- **Technical definition:** A base model is only pretrained and continues text; an instruct model has additionally been SFT-ed (and often preference-tuned) on chat-formatted data.
- **Plain English:** Base = raw autocomplete; instruct = trained assistant.
- **Example / analogy:** Ask gpt2 a question and it continues the sentence instead of answering.
- **Snippet:**
```python
tok.apply_chat_template(...)  # only meaningful for instruct models
```
- **Snippet explained:** Chat templates assume the model saw them in training.
- **Related terms:** [[Chat template]], [[Post-training]]

### Supervised fine-tuning (SFT, instruction tuning)
<!-- cat: llm -->
- **Technical definition:** Further training with next-token cross-entropy on (prompt, ideal answer) pairs, usually computing loss only on the answer tokens.
- **Plain English:** Showing the model worked examples of good answers.
- **Example / analogy:** Instruction tuning is SFT where the data are instructions and responses.
- **Snippet:**
```python
labels = [-100]*len(prompt_ids) + completion_ids
```
- **Snippet explained:** Prompt positions are ignored by the loss; only the completion is learned.
- **Related terms:** [[Loss masking (completion-only loss)]], [[Cross-entropy loss]]

### Loss masking (completion-only loss)
<!-- cat: llm -->
- **Technical definition:** Setting labels to -100 on tokens that should not contribute to the loss (the prompt and padding), so cross_entropy(ignore_index=-100) skips them.
- **Plain English:** Grade only the answer, not the question.
- **Example / analogy:** Prompt tokens would otherwise be trained too, diluting the gradient.
- **Snippet:**
```python
F.cross_entropy(logits.view(-1,V), labels.view(-1), ignore_index=-100)
```
- **Snippet explained:** -100 is PyTorch's default ignore index.
- **Related terms:** [[Supervised fine-tuning (SFT, instruction tuning)]]

### PEFT (parameter-efficient fine-tuning)
<!-- cat: llm -->
- **Technical definition:** Methods that train a small number of added or selected parameters while the base model stays frozen.
- **Plain English:** Adjusting a few dials instead of rebuilding the machine.
- **Example / analogy:** LoRA, DoRA, IA3, prompt and prefix tuning are PEFT methods.
- **Snippet:**
```python
pm = get_peft_model(base, LoraConfig(r=8, target_modules=['c_attn']))
```
- **Snippet explained:** Freezes the base and injects trainable adapters.
- **Related terms:** [[LoRA]], [[QLoRA]], [[DoRA]]

### DoRA
<!-- cat: llm -->
- **Technical definition:** Weight-decomposed LoRA: splits a weight into magnitude and direction, applies LoRA to the direction and trains the magnitude vector.
- **Plain English:** LoRA plus a learnable per-output scale.
- **Example / analogy:** Adds a small magnitude vector per adapted layer.
- **Snippet:**
```python
LoraConfig(r=8, use_dora=True, target_modules=['c_attn'])
```
- **Snippet explained:** Switches on DoRA in peft.
- **Related terms:** [[PEFT (parameter-efficient fine-tuning)]]

### rsLoRA
<!-- cat: llm -->
- **Technical definition:** Rank-stabilised LoRA: scales the adapter by alpha/sqrt(r) instead of alpha/r so learning does not collapse at high rank.
- **Plain English:** A scaling fix so bigger ranks actually help.
- **Example / analogy:** With r=64, alpha=128: scale 2.0 for LoRA, 16.0 for rsLoRA.
- **Snippet:**
```python
LoraConfig(r=64, lora_alpha=128, use_rslora=True)
```
- **Snippet explained:** Turns on the sqrt(r) scaling.
- **Related terms:** [[PEFT (parameter-efficient fine-tuning)]]

### IA3
<!-- cat: llm -->
- **Technical definition:** Infused Adapter by Inhibiting and Amplifying Inner Activations: learns per-channel scaling vectors for keys, values and feed-forward activations.
- **Plain English:** Volume knobs on internal signals.
- **Example / analogy:** Far fewer parameters than LoRA.
- **Snippet:**
```python
IA3Config(target_modules=['c_attn','c_fc'], feedforward_modules=['c_fc'])
```
- **Snippet explained:** Declares which layers get a learned rescale.
- **Related terms:** [[PEFT (parameter-efficient fine-tuning)]]

### Prompt tuning and prefix tuning
<!-- cat: llm -->
- **Technical definition:** Learning continuous virtual-token embeddings (prompt tuning) or per-layer key/value prefixes (prefix tuning) while the model weights stay frozen.
- **Plain English:** Teaching the model a secret opening phrase made of numbers.
- **Example / analogy:** 20 virtual tokens on distilgpt2 is 15,360 trainable parameters.
- **Snippet:**
```python
PromptTuningConfig(num_virtual_tokens=20, task_type='CAUSAL_LM')
```
- **Snippet explained:** Adds 20 trainable embedding rows prepended to every input.
- **Related terms:** [[PEFT (parameter-efficient fine-tuning)]]

### Memory arithmetic (bytes per parameter)
<!-- cat: ml -->
- **Technical definition:** Training memory per parameter: about 16 bytes for mixed-precision Adam full fine-tuning (weights 2, grads 2, fp32 master 4, two moments 8), before activations.
- **Plain English:** Count what must be stored for every number.
- **Example / analogy:** 7B full fine-tune is about 112 GB; LoRA keeps the base at 2 bytes per parameter.
- **Snippet:**
```python
print(16 * 7e9 / 1e9)
```
- **Snippet explained:** Prints 112.0 (GB).
- **Related terms:** [[LoRA]], [[QLoRA]]

### Preference tuning
<!-- cat: llm -->
- **Technical definition:** Training a model from comparisons (chosen vs rejected answers) rather than single ideal answers.
- **Plain English:** Learning taste by seeing which of two answers people liked.
- **Example / analogy:** DPO and reward-model RLHF both start from preference pairs.
- **Snippet:**
```python
{'prompt':..., 'chosen':..., 'rejected':...}
```
- **Snippet explained:** One preference example.
- **Related terms:** [[DPO (direct preference optimization)]], [[Reward model]]

### Reward model
<!-- cat: llm -->
- **Technical definition:** A model that maps (prompt, answer) to a scalar score, trained from human or AI preference pairs.
- **Plain English:** An automatic judge that outputs a number.
- **Example / analogy:** Also called preference model or scoring model.
- **Snippet:**
```python
self.head = nn.Linear(768, 1)
```
- **Snippet explained:** Scalar head on the last token's hidden state.
- **Related terms:** [[Bradley-Terry model]], [[RLHF (reinforcement learning from human feedback)]]

### Bradley-Terry model
<!-- cat: ml -->
- **Technical definition:** A model of pairwise comparisons where P(i beats j) = sigmoid(s_i - s_j) for latent scores s.
- **Plain English:** Turns 'A beat B' votes into numeric strengths.
- **Example / analogy:** Used for reward-model training and for Arena rankings.
- **Snippet:**
```python
loss = -F.logsigmoid(r_chosen - r_rejected).mean()
```
- **Snippet explained:** Negative log-likelihood of the observed preference.
- **Related terms:** [[Reward model]], [[Elo rating]]

### DPO (direct preference optimization)
<!-- cat: llm -->
- **Technical definition:** Preference loss -log sigmoid(beta*[(log pi/ref)_chosen - (log pi/ref)_rejected]) trained directly on the policy with a frozen reference; no explicit reward model or sampling.
- **Plain English:** Skip the reward model: the policy itself is the reward model.
- **Example / analogy:** Implicit reward = beta * log(pi/ref).
- **Snippet:**
```python
-F.logsigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r)))
```
- **Snippet explained:** pi_c etc. are summed token log-probs of the chosen/rejected replies.
- **Related terms:** [[Implicit reward]], [[Reference model]], [[KL divergence]]

### Implicit reward
<!-- cat: llm -->
- **Technical definition:** In DPO, r(x,y) = beta * log(pi(y|x)/ref(y|x)); its chosen-minus-rejected gap is the reward margin.
- **Plain English:** A reward you read off the policy instead of a separate model.
- **Example / analogy:** Logged as rewards/chosen, rewards/rejected, rewards/margins.
- **Snippet:**
```python
beta * (pi_c - ref_c)
```
- **Snippet explained:** Reward of the chosen reply.
- **Related terms:** [[DPO (direct preference optimization)]]

### Reference model
<!-- cat: llm -->
- **Technical definition:** A frozen copy of the starting policy used to compute log-ratios and to anchor the KL penalty in DPO and RLHF.
- **Plain English:** The 'you must not drift too far from this' anchor.
- **Example / analogy:** SimPO and ORPO remove it.
- **Snippet:**
```python
reference = copy.deepcopy(policy).eval()
```
- **Snippet explained:** Frozen copy of the policy at step 0.
- **Related terms:** [[DPO (direct preference optimization)]], [[KL divergence]]

### IPO, KTO, ORPO, SimPO
<!-- cat: llm -->
- **Technical definition:** DPO relatives: IPO squares the margin toward 1/(2*beta); KTO uses unpaired good/bad labels; ORPO adds an odds-ratio penalty to SFT with no reference; SimPO uses length-normalised log-probs and a target margin with no reference.
- **Plain English:** Different recipes for the same preference data.
- **Example / analogy:** TRL exposes several via DPOConfig(loss_type=...).
- **Snippet:**
```python
DPOConfig(loss_type=['ipo'])
```
- **Snippet explained:** Selects the IPO loss.
- **Related terms:** [[DPO (direct preference optimization)]]

### RLHF (reinforcement learning from human feedback)
<!-- cat: llm -->
- **Technical definition:** Training a policy with RL (typically PPO) against a reward model learned from human preferences, with a KL penalty to a reference model.
- **Plain English:** Practice answering and get graded by a learned judge.
- **Example / analogy:** Four models in memory: policy, reference, reward, value.
- **Snippet:**
```python
# see 06_ppo_vs_grpo_rlvr
```
- **Snippet explained:** Conceptual pipeline.
- **Related terms:** [[PPO (proximal policy optimization)]], [[Reward model]]

### RLAIF
<!-- cat: llm -->
- **Technical definition:** RL from AI feedback: the reward or preference labels come from an AI judge instead of humans.
- **Plain English:** Same pipeline, machine graders.
- **Example / analogy:** Cheaper but inherits the judge's biases.
- **Snippet:**
```python
# judge(prompt, a, b) -> 'A' or 'B'
```
- **Snippet explained:** Pairwise AI label.
- **Related terms:** [[LLM-as-judge]]

### RLVR (reinforcement learning with verifiable rewards)
<!-- cat: llm -->
- **Technical definition:** RL where the reward is computed by code that checks the answer (math answer match, unit tests), so no learned reward model is needed.
- **Plain English:** Practice problems with an answer key.
- **Example / analogy:** Reward 1.0 if the final number equals the gold answer.
- **Snippet:**
```python
reward = 1.0 if s == str(a + b) else 0.0
```
- **Snippet explained:** A verifier for a+b.
- **Related terms:** [[GRPO (group relative policy optimization)]]

### PPO (proximal policy optimization)
<!-- cat: ml -->
- **Technical definition:** Policy-gradient algorithm that maximises a clipped surrogate objective min(r*A, clip(r,1-eps,1+eps)*A) with advantages from a value function (GAE).
- **Plain English:** Take improvement steps but refuse steps that change the policy too much.
- **Example / analogy:** r is new/old probability ratio.
- **Snippet:**
```python
torch.min(ratio*adv, torch.clamp(ratio,1-eps,1+eps)*adv)
```
- **Snippet explained:** The clipped objective per token.
- **Related terms:** [[GAE (generalized advantage estimation)]], [[KL divergence]]

### GAE (generalized advantage estimation)
<!-- cat: ml -->
- **Technical definition:** An advantage estimate A_t = sum (gamma*lambda)^l * delta_{t+l} with TD errors delta = r + gamma*V(s') - V(s).
- **Plain English:** A smoothed estimate of how much better an action was than expected.
- **Example / analogy:** Needs a learned value network.
- **Snippet:**
```python
delta = r[t] + gamma*v[t+1] - v[t]
```
- **Snippet explained:** One TD error.
- **Related terms:** [[PPO (proximal policy optimization)]]

### GRPO (group relative policy optimization)
<!-- cat: llm -->
- **Technical definition:** PPO-style optimisation without a value network: sample G completions per prompt and use A_i = (r_i - mean(r))/std(r) within the group as the advantage.
- **Plain English:** Grade on a curve within each batch of answers to the same question.
- **Example / analogy:** Zero-variance groups give zero advantage.
- **Snippet:**
```python
adv = (r - r.mean(1,keepdim=True)) / (r.std(1,keepdim=True)+1e-4)
```
- **Snippet explained:** Group-normalised advantages.
- **Related terms:** [[RLVR (reinforcement learning with verifiable rewards)]], [[PPO (proximal policy optimization)]]

### KL divergence
<!-- cat: ml -->
- **Technical definition:** KL(p||q) = sum p log(p/q); in RL fine-tuning, a penalty keeping the policy near the reference.
- **Plain English:** A measure of how far one distribution has drifted from another.
- **Example / analogy:** GRPO often uses the k3 estimator exp(d) - d - 1 with d = log ref - log pi.
- **Snippet:**
```python
kl = torch.exp(d) - d - 1
```
- **Snippet explained:** Non-negative per-sample estimate.
- **Related terms:** [[Reference model]], [[GRPO (group relative policy optimization)]]

### Knowledge distillation
<!-- cat: llm -->
- **Technical definition:** Training a student to match a teacher's output distribution, usually T^2 * KL(softmax(t/T) || softmax(s/T)) plus optional hard-label CE.
- **Plain English:** The student copies the teacher's confidence, not just its final answers.
- **Example / analogy:** Higher temperature reveals which wrong tokens the teacher finds plausible.
- **Snippet:**
```python
F.kl_div(F.log_softmax(s/T,-1), F.softmax(t/T,-1), reduction='batchmean')*T*T
```
- **Snippet explained:** Temperature-scaled KL.
- **Related terms:** [[KL divergence]]

### Model merging (linear, task arithmetic, SLERP, TIES)
<!-- cat: llm -->
- **Technical definition:** Combining fine-tuned weights without training: averaging, adding task vectors (theta_ft - theta_base), spherical interpolation, or TIES (trim, elect sign, disjoint mean).
- **Plain English:** Mixing recipes instead of cooking again.
- **Example / analogy:** Only sensible when models share a base.
- **Snippet:**
```python
merged = base + lam * (tau1 + tau2)
```
- **Snippet explained:** Task arithmetic.
- **Related terms:** [[PEFT (parameter-efficient fine-tuning)]]

### Perplexity
<!-- cat: ml -->
- **Technical definition:** exp of the mean per-token cross-entropy (in nats) of a language model on held-out text.
- **Plain English:** How surprised the model is on average; lower is better.
- **Example / analogy:** Cross-entropy 3.0 nats means perplexity about 20.
- **Snippet:**
```python
math.exp(nll / ntok)
```
- **Snippet explained:** Converts mean loss to perplexity.
- **Related terms:** [[Cross-entropy loss]]

### Exact match and token F1
<!-- cat: ml -->
- **Technical definition:** EM: normalised prediction equals the gold string; F1: harmonic mean of token precision and recall after normalisation.
- **Plain English:** Strict vs partial credit for short answers.
- **Example / analogy:** 'Paris, France' vs 'Paris': EM 0, F1 0.67.
- **Snippet:**
```python
token_f1('Paris, France','Paris')
```
- **Snippet explained:** Partial overlap score.
- **Related terms:** [[Answer extraction]]

### Log-likelihood multiple choice (acc, acc_norm)
<!-- cat: llm -->
- **Technical definition:** Score each answer choice by log P(choice | question) and pick the highest; acc_norm divides by choice length (bytes) to remove length bias.
- **Plain English:** Ask the model which ending it finds most likely.
- **Example / analogy:** lm-evaluation-harness reports both.
- **Snippet:**
```python
scores = [lp / len(c.encode()) for lp, c in zip(lps, choices)]
```
- **Snippet explained:** Byte-normalised score.
- **Related terms:** [[Perplexity]]

### pass@k
<!-- cat: ml -->
- **Technical definition:** Probability that at least one of k samples is correct; estimated unbiasedly from n samples with c correct as 1 - C(n-c,k)/C(n,k).
- **Plain English:** The chance a k-try attempt succeeds.
- **Example / analogy:** n=10, c=3, k=5 gives 0.9167.
- **Snippet:**
```python
1 - np.prod(1 - k/np.arange(n-c+1, n+1))
```
- **Snippet explained:** Stable product form.
- **Related terms:** [[Bootstrap confidence interval]]

### Bootstrap confidence interval
<!-- cat: ml -->
- **Technical definition:** Resample evaluation items with replacement many times and take percentiles of the metric to get an interval.
- **Plain English:** Shake the test set to see how much the score wobbles.
- **Example / analogy:** 30 items at 50% give an interval about 0.35 wide.
- **Snippet:**
```python
np.quantile(means, [0.025, 0.975])
```
- **Snippet explained:** The percentile interval.
- **Related terms:** [[pass@k]]

### Benchmark contamination
<!-- cat: llm -->
- **Technical definition:** Test items (or close paraphrases) appearing in training data, inflating scores.
- **Plain English:** The student saw the exam beforehand.
- **Example / analogy:** Detected with n-gram overlap against training text.
- **Snippet:**
```python
grams(item) & grams(train_corpus)
```
- **Snippet explained:** Overlap of n-grams.
- **Related terms:** [[Log-likelihood multiple choice (acc, acc_norm)]]

### LLM-as-judge
<!-- cat: llm -->
- **Technical definition:** Using a language model to score answers (pointwise with a rubric) or to pick the better of two (pairwise).
- **Plain English:** A model grading homework.
- **Example / analogy:** Biases: position, verbosity, self-preference.
- **Snippet:**
```python
json = extract_json(generate(messages))
```
- **Snippet explained:** Parses the judge's JSON verdict.
- **Related terms:** [[Position bias]], [[Bradley-Terry model]]

### Position bias
<!-- cat: llm -->
- **Technical definition:** A judge's tendency to prefer an answer because of where it appears (usually first); detected by swapping the order.
- **Plain English:** Always picking the left-hand option.
- **Example / analogy:** Count a win only if it holds in both orders.
- **Snippet:**
```python
r2 = judge(q, second, first)
```
- **Snippet explained:** Swapped call.
- **Related terms:** [[LLM-as-judge]]

### Elo rating
<!-- cat: ml -->
- **Technical definition:** Online rating update r_i += K*(score - expected) with expected = 1/(1+10^((r_j-r_i)/400)); depends on the order of games.
- **Plain English:** Chess-style rating after each match.
- **Example / analogy:** Bradley-Terry maximum likelihood is the order-independent alternative.
- **Snippet:**
```python
e = 1/(1+10**((rj-ri)/400))
```
- **Snippet explained:** Expected score of i.
- **Related terms:** [[Bradley-Terry model]]

### Catastrophic forgetting
<!-- cat: ml -->
- **Technical definition:** Loss of previously learned abilities after fine-tuning on a narrow task.
- **Plain English:** Learning a new trick and forgetting old ones.
- **Example / analogy:** Measured by regression probes (general MCQ, general-text perplexity) before and after.
- **Snippet:**
```python
perplexity(model, tok, general_text)
```
- **Snippet explained:** Probe of general language ability.
- **Related terms:** [[Perplexity]], [[PEFT (parameter-efficient fine-tuning)]]

### Answer extraction
<!-- cat: llm -->
- **Technical definition:** The rule that turns free-form model output into the string that is scored (first line, last number, a boxed expression, a regex group).
- **Plain English:** Deciding which part of the model's rambling counts as its answer.
- **Example / analogy:** A teacher who only reads the final underlined line of an essay.
- **Snippet:**
```python
answer = raw_text.split("\n")[0].strip()
```
- **Snippet explained:** Keeps only the first line of the generation and strips spaces before exact-match or F1 scoring.
- **Related terms:** [[Exact match and token F1]], [[Log-likelihood multiple choice (acc, acc_norm)]]

### Special tokens
<!-- cat: llm -->
- **Technical definition:** Reserved vocabulary entries (such as <|im_start|>, <|im_end|> or <|endoftext|>) that mark structure like turn boundaries or end of text; each is a single token id.
- **Plain English:** Punctuation marks that only the model reads.
- **Example / analogy:** Stage directions in a script: not spoken, but they tell everyone who speaks when.
- **Snippet:**
```python
tok.convert_tokens_to_ids("<|im_start|>")
```
- **Snippet explained:** Looks up the single id of a special token in the Qwen2.5 tokenizer (151644).
- **Related terms:** [[Chat template]], [[Generation prompt]]
