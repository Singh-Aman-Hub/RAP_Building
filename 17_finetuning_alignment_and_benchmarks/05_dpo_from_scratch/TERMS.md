# Terms — DPO from scratch

Folder: `17_finetuning_alignment_and_benchmarks/05_dpo_from_scratch` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [17 README](../README.md)

**Terms in this folder:** [DPO (direct preference optimization)](#dpo-direct-preference-optimization) · [Reference model](#reference-model) · [Implicit reward](#implicit-reward) · [Reward margin and accuracy](#reward-margin-and-accuracy) · [beta (KL strength)](#beta-kl-strength) · [Sequence log-probability](#sequence-log-probability) · [Likelihood displacement](#likelihood-displacement) · [IPO](#ipo) · [SimPO](#simpo) · [KTO](#kto) · [ORPO](#orpo) · [Offline vs online preference optimisation](#offline-vs-online-preference-optimisation) · [Gradient accumulation](#gradient-accumulation) · [torch.set_grad_enabled](#torchset_grad_enabled)

---

## DPO (direct preference optimization)  `(concept)`
**1. Definition:** Trains the policy directly on preference pairs with -log sigmoid(beta * [(log pi/ref)_chosen - (log pi/ref)_rejected]); no reward model and no sampling.
**2. In plain words:** Skip the reward model: the policy is the reward model.
**3. Example to quote:** > "At step 0 the loss is ln 2 = 0.693 because policy equals reference."
```python
-F.logsigmoid(beta * logits)
```
**Seen in:** `main.py`, `preference_loss`

## Reference model  `(concept)`
**1. Definition:** A frozen copy of the starting policy used for log-ratios and as the KL anchor.
**2. In plain words:** The 'do not drift too far' anchor.
**3. Example to quote:** > "SimPO and ORPO remove it."
```python
reference = copy.deepcopy(policy).eval()
```
**Seen in:** `main.py`, section 1

## Implicit reward  `(concept)`
**1. Definition:** r = beta * log(pi(y|x) / ref(y|x)).
**2. In plain words:** A reward read off the policy.
**3. Example to quote:** > "Logged as rewards/chosen and rewards/rejected in TRL."
```python
beta * (pi_c - ref_c)
```
**Seen in:** `main.py`, `metrics`

## Reward margin and accuracy  `(concept)`
**1. Definition:** Margin: chosen minus rejected implicit reward; accuracy: share of pairs with positive margin.
**2. In plain words:** How well the policy separates the pair.
**3. Example to quote:** > "I watch the implicit reward margin and the fraction of pairs where the chosen reward beats the rejected one."
**Seen in:** `main.py`, `metrics`

## beta (KL strength)  `(concept)`
**1. Definition:** Scales the log-ratios; higher beta keeps the policy closer to the reference.
**2. In plain words:** How tight the leash is.
**3. Example to quote:** > "TRL default 0.1."
```python
BETA = 0.1
```
**Seen in:** `main.py`, settings

## Sequence log-probability  `(concept)`
**1. Definition:** Sum over completion tokens of log p(token | prefix).
**2. In plain words:** How likely the model finds the whole reply.
**3. Example to quote:** > "Computed by seq_logprob in common.py."
```python
tok_lp[:, Lp-1:].sum()
```
**Seen in:** `common.py`

## Likelihood displacement  `(concept)`
**1. Definition:** Both chosen and rejected log-probs fall while the margin grows.
**2. In plain words:** The model gets less sure of everything.
**3. Example to quote:** > "Watch logp(chosen); a high learning rate makes it fall sharply."
**Seen in:** `main.py`, BREAK IT

## IPO  `(concept)`
**1. Definition:** Squared loss (margin - 1/(2*beta))^2 on the log-ratio margin; bounded target.
**2. In plain words:** DPO with a ceiling.
**3. Example to quote:** > "TRL: loss_type='ipo'."
```python
(logits - 1/(2*beta))**2
```
**Seen in:** `main.py`, `preference_loss`

## SimPO  `(concept)`
**1. Definition:** Reference-free: length-normalised average log-prob as reward with a target margin gamma.
**2. In plain words:** DPO without the reference model.
**3. Example to quote:** > "TRL's closest option: loss_type='sigmoid_norm' (length-normalised)."
**Seen in:** `main.py`, `preference_loss`

## KTO  `(concept)`
**1. Definition:** Uses unpaired good/bad labels and a prospect-theory-shaped objective.
**2. In plain words:** Thumbs up / thumbs down instead of pairs.
**3. Example to quote:** > "KTO needs only a thumbs-up or thumbs-down per example, not paired replies."
**Seen in:** `main.py`, variants table

## ORPO  `(concept)`
**1. Definition:** SFT loss on the chosen answer plus an odds-ratio penalty on the rejected one; no reference model.
**2. In plain words:** One stage instead of two.
**3. Example to quote:** > "ORPO folds the preference signal into the SFT loss with an odds-ratio penalty, so there is no reference model and one training stage."
**Seen in:** `main.py`, variants table

## Offline vs online preference optimisation  `(concept)`
**1. Definition:** Offline: fixed dataset (DPO). Online: new samples are drawn and judged during training (RL).
**2. In plain words:** Static pairs vs fresh attempts.
**3. Example to quote:** > "DPO is offline: it learns from a fixed set of pairs and never samples from the current policy."
**Seen in:** `WALKTHROUGH.md` theory

## Gradient accumulation  `(concept)`
**1. Definition:** Summing gradients over several mini-batches before one optimiser step.
**2. In plain words:** A bigger batch on small memory.
**3. Example to quote:** > "This step accumulates 4 pairs per update."
```python
loss = ... / 4; loss.backward()
```
**Seen in:** `main.py`, section 6

## torch.set_grad_enabled  `(syntax)`
**1. Definition:** Context that toggles autograd for a block.
**2. In plain words:** Gradients on or off.
**3. Example to quote:** > "seq_logprob(grad=False) is used for evaluation and the reference."
```python
with torch.set_grad_enabled(grad):
```
**Seen in:** `common.py`
