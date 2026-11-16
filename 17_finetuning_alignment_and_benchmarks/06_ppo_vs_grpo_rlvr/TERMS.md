# Terms — PPO vs GRPO and RLVR

Folder: `17_finetuning_alignment_and_benchmarks/06_ppo_vs_grpo_rlvr` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [17 README](../README.md)

**Terms in this folder:** [RLHF with PPO](#rlhf-with-ppo) · [GRPO (group relative policy optimisation)](#grpo-group-relative-policy-optimisation) · [RLVR (verifiable rewards)](#rlvr-verifiable-rewards) · [Verifier](#verifier) · [Group size G](#group-size-g) · [Advantage](#advantage) · [GAE (generalized advantage estimation)](#gae-generalized-advantage-estimation) · [Value network (critic)](#value-network-critic) · [Clipped ratio objective](#clipped-ratio-objective) · [Importance ratio](#importance-ratio) · [KL penalty and the k3 estimator](#kl-penalty-and-the-k3-estimator) · [Reference model (frozen)](#reference-model-frozen) · [Reward scaling (scale_rewards, std division)](#reward-scaling-scale_rewards-std-division) · [Loss normalisation (token-level vs sequence-level)](#loss-normalisation-token-level-vs-sequence-level) · [Groups with signal](#groups-with-signal) · [Memorisation vs generalisation in RL](#memorisation-vs-generalisation-in-rl) · [Boolean broadcasting error `size of tensor a must match size of tensor b`](#boolean-broadcasting-error-size-of-tensor-a-must-match-size-of-tensor-b) · [torch.distributions.Categorical](#torchdistributionscategorical) · [torch.cumsum](#torchcumsum) · [Gradient clipping (clip_grad_norm_)](#gradient-clipping-clip_grad_norm_)

---

## RLHF with PPO  `(concept)`
**1. Definition:** Reinforcement learning from human feedback optimises a policy against a learned reward model using PPO, which needs a policy, a frozen reference, a reward model and a value network.
**2. In plain words:** Four models in memory: the one you train, the anchor, the grader, and a guesser of expected reward.
**3. Example to quote:** > "PPO-based RLHF is powerful but heavy: it needs a value network and a reward model, and the learned reward can be exploited."
**Seen in:** `WALKTHROUGH.md` theory

## GRPO (group relative policy optimisation)  `(concept)`
**1. Definition:** A critic-free PPO variant: sample G completions per prompt and use A_i = (r_i - mean(r)) / (std(r) + eps) within the group as the advantage of every token of completion i.
**2. In plain words:** Grade on a curve among the answers to the same question. No value network needed.
**3. Example to quote:** > "In the toy run only 12% of groups had any success at step 0, because a group with identical rewards has zero advantage."
```python
adv = r - r.mean(1, keepdim=True)
adv = adv / (r.std(1, keepdim=True) + 1e-4)
```
**Seen in:** `main.py`, `grpo_step`

## RLVR (verifiable rewards)  `(concept)`
**1. Definition:** Reinforcement learning where the reward is computed by a program that checks the answer, not by a learned model.
**2. In plain words:** An answer key instead of a human grader.
**3. Example to quote:** > "The reward is 1.0 only if the number equals a + b and the answer terminates with EOS."
```python
def reward_fn(a, b, toks): ...
```
**Seen in:** `main.py`, `reward_fn`

## Verifier  `(concept)`
**1. Definition:** The function that returns the reward in RLVR, such as an answer matcher or a unit-test runner.
**2. In plain words:** The checker. Only as strict as you write it.
**3. Example to quote:** > "A verifier that ignores the end-of-answer token would reward answers that never stop."
**Seen in:** `main.py`, section 3

## Group size G  `(concept)`
**1. Definition:** The number of completions sampled per prompt in GRPO (TRL: num_generations, default 8 in the v1.17.1 docs).
**2. In plain words:** How many attempts each question gets.
**3. Example to quote:** > "With G=4 the final reward was 0.47; with G=16 it was 0.97 in the same 200 steps."
```python
G = int(os.environ.get('G', 16))
```
**Seen in:** `main.py`, settings

## Advantage  `(concept)`
**1. Definition:** How much better than a baseline an action or sequence was; in GRPO, the group-normalised reward.
**2. In plain words:** Reward relative to what was expected.
**3. Example to quote:** > "Zero advantage means that sample contributes no gradient."
**Seen in:** `main.py`, `grpo_step`

## GAE (generalized advantage estimation)  `(concept)`
**1. Definition:** An advantage estimator A_t = sum_l (gamma*lambda)^l delta_{t+l} with delta_t = r_t + gamma V(s_{t+1}) - V(s_t); needs a value function.
**2. In plain words:** Spread a delayed reward back over earlier steps using a learned guess.
**3. Example to quote:** > "With rewards [0,0,0,1] and values [0.2,0.3,0.5,0.7] it gives [0.728, 0.661, 0.485, 0.3]."
```python
def gae(rewards, values, gamma=1.0, lam=0.95): ...
```
**Seen in:** `main.py`, section 7

## Value network (critic)  `(concept)`
**1. Definition:** A network that predicts expected return from a state; PPO uses it as a baseline.
**2. In plain words:** A second model whose job is to guess how well things are going.
**3. Example to quote:** > "GRPO removes it and uses the group mean instead."
**Seen in:** `WALKTHROUGH.md` theory

## Clipped ratio objective  `(concept)`
**1. Definition:** PPO's surrogate loss -min(ratio*A, clip(ratio, 1-eps, 1+eps)*A) which stops the policy moving too far in one update.
**2. In plain words:** A safety rail on each step.
**3. Example to quote:** > "With eps=100 clipping never triggers and the clip fraction stayed 0.000 in the experiment."
```python
pg = -torch.min(ratio * adv, torch.clamp(ratio, 1 - EPS, 1 + EPS) * adv)
```
**Seen in:** `main.py`, `grpo_step`

## Importance ratio  `(concept)`
**1. Definition:** pi_new(a|s) / pi_old(a|s), the factor correcting for samples drawn from the old policy.
**2. In plain words:** How much more (or less) likely the new policy finds the same token.
**3. Example to quote:** > "It equals 1.0 on the first inner pass, which is why MU>1 is needed for clipping to matter."
```python
ratio = torch.exp(lp - old_lp)
```
**Seen in:** `main.py`, `grpo_step`

## KL penalty and the k3 estimator  `(concept)`
**1. Definition:** A penalty beta*KL(pi || pi_ref); the k3 per-sample estimator is exp(d) - d - 1 with d = log pi_ref - log pi.
**2. In plain words:** A leash to the reference model.
**3. Example to quote:** > "BETA=0 removes the leash and KL reached 8.63 in the experiment."
```python
kl = torch.exp(d) - d - 1
```
**Seen in:** `main.py`, `grpo_step`

## Reference model (frozen)  `(concept)`
**1. Definition:** A copy of the starting policy whose log-probs anchor the KL term.
**2. In plain words:** The 'where we started' photo.
**3. Example to quote:** > "TRL's GRPO default beta=0.0 does not even load it."
```python
reference = copy.deepcopy(policy).eval()
```
**Seen in:** `main.py`, section 4

## Reward scaling (scale_rewards, std division)  `(concept)`
**1. Definition:** Whether advantages are divided by the group std (TRL: scale_rewards='group' default, 'batch', or False).
**2. In plain words:** A choice about how much to boost rare successes.
**3. Example to quote:** > "Without std division the toy run started slower (0.078 vs 0.172 reward at step 20)."
```python
SCALE_STD=0 python main.py
```
**Seen in:** `main.py`, `SCALE_STD`

## Loss normalisation (token-level vs sequence-level)  `(concept)`
**1. Definition:** Averaging the per-token loss over all valid tokens in a batch versus averaging per sequence first; TRL's loss_type options include dapo (default), dr_grpo, sapo in the v1.17.1 docs.
**2. In plain words:** Do long answers count more or the same?
**3. Example to quote:** > "This step divides by the number of valid tokens in the whole batch."
```python
loss = per_tok.sum() / mask.sum()
```
**Seen in:** `main.py`, `grpo_step`

## Groups with signal  `(concept)`
**1. Definition:** The fraction of groups in a batch whose rewards are not all equal.
**2. In plain words:** How many questions taught us anything this step.
**3. Example to quote:** > "It fell to 0.00 at step 100 once most groups were all-correct."
```python
nonzero = (r.std(1) > 0).float().mean().item()
```
**Seen in:** `main.py`, `grpo_step`

## Memorisation vs generalisation in RL  `(concept)`
**1. Definition:** RL improves reward on the prompts it sees; it may not transfer to unseen prompts.
**2. In plain words:** Practising 20 questions does not teach the rule.
**3. Example to quote:** > "Greedy accuracy was 0.90 on training prompts and 0.00 on 5 held-out prompts."
```python
accuracy(held_out)
```
**Seen in:** `main.py`, section 9

## Boolean broadcasting error `size of tensor a must match size of tensor b`  `(error)`
**1. Definition:** Raised when two tensors cannot be broadcast because trailing dimensions differ.
**2. In plain words:** A shape mismatch: (128, 3) vs (128,).
**3. Example to quote:** > "The advantage must be (N, 1) to broadcast over the three token positions."
```python
RuntimeError: The size of tensor a (3) must match the size of tensor b (128) at non-singleton dimension 1
```
**Seen in:** `WALKTHROUGH.md`, debugging practice

## torch.distributions.Categorical  `(library)`
**1. Definition:** A categorical distribution built from logits; .sample() draws tokens.
**2. In plain words:** Roll a weighted die for each row.
**3. Example to quote:** > "Used to sample the next token on-policy."
```python
Categorical(logits=logits).sample()
```
**Seen in:** `main.py`, `sample`

## torch.cumsum  `(library)`
**1. Definition:** Cumulative sum along a dimension.
**2. In plain words:** Running total.
**3. Example to quote:** > "cumsum(is_eos) - is_eos counts EOS tokens strictly before each position."
```python
before = torch.cumsum(is_eos, dim=1) - is_eos
```
**Seen in:** `main.py`, `valid_mask`

## Gradient clipping (clip_grad_norm_)  `(library)`
**1. Definition:** Rescales all gradients so their global L2 norm is at most max_norm.
**2. In plain words:** A cap on step size from outliers.
**3. Example to quote:** > "Used with max_norm 1.0 after backward."
```python
torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
```
**Seen in:** `main.py`, `grpo_step`
