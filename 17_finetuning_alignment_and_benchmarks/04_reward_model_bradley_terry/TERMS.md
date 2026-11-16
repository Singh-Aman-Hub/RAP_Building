# Terms — Reward model with the Bradley-Terry loss

Folder: `17_finetuning_alignment_and_benchmarks/04_reward_model_bradley_terry` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [17 README](../README.md)

**Terms in this folder:** [Reward model](#reward-model) · [Preference pair](#preference-pair) · [Bradley-Terry model and loss](#bradley-terry-model-and-loss) · [Scalar head and last-token pooling](#scalar-head-and-last-token-pooling) · [Pairwise accuracy](#pairwise-accuracy) · [Reward margin](#reward-margin) · [Shuffled-label control](#shuffled-label-control) · [Held-out generalisation split](#held-out-generalisation-split) · [Right padding vs left padding](#right-padding-vs-left-padding) · [AutoModel (no LM head)](#automodel-no-lm-head) · [Reward hacking](#reward-hacking) · [RLHF reward pipeline](#rlhf-reward-pipeline) · [torch.nn.init.normal_](#torchnninitnormal_)

---

## Reward model  `(concept)`
**1. Definition:** A model mapping (prompt, answer) to one scalar score, trained from preference pairs.
**2. In plain words:** An automatic grader.
**3. Example to quote:** > "Here: distilgpt2 backbone plus nn.Linear(768, 1) on the last real token."
```python
self.head = nn.Linear(768, 1)
```
**Seen in:** `main.py`, `RewardModel`

## Preference pair  `(concept)`
**1. Definition:** A prompt with a chosen (better) and a rejected (worse) answer.
**2. In plain words:** Which of these two is better?
**3. Example to quote:** > "A preference pair is a prompt with one chosen and one rejected reply; the label says which is better, not by how much."
```python
{'prompt':..., 'chosen':..., 'rejected':...}
```
**Seen in:** `common.py`, `make_preference_data`

## Bradley-Terry model and loss  `(concept)`
**1. Definition:** P(chosen beats rejected) = sigmoid(r_c - r_r); loss = -log of that probability.
**2. In plain words:** Turn comparisons into scores.
**3. Example to quote:** > "Equal rewards give ln 2 = 0.693."
```python
-F.logsigmoid(r_chosen - r_rejected).mean()
```
**Seen in:** `main.py`, `bt_loss`

## Scalar head and last-token pooling  `(concept)`
**1. Definition:** A linear layer reading the hidden state of the last real token to produce one number.
**2. In plain words:** Summarise the whole text at its last word.
**3. Example to quote:** > "With right padding the last real index is attention_mask.sum(1) - 1."
```python
h[torch.arange(B), last]
```
**Seen in:** `main.py`, `forward`

## Pairwise accuracy  `(concept)`
**1. Definition:** Fraction of held-out pairs where the chosen answer gets the higher reward.
**2. In plain words:** Did we rank the pair correctly?
**3. Example to quote:** > "Chance is 0.5."
```python
(rc > rr).float().mean()
```
**Seen in:** `main.py`, `evaluate`

## Reward margin  `(concept)`
**1. Definition:** Mean of r_chosen - r_rejected.
**2. In plain words:** How far apart the scores are.
**3. Example to quote:** > "Only differences matter; adding a constant to every reward changes nothing."
**Seen in:** `main.py`, `evaluate`

## Shuffled-label control  `(concept)`
**1. Definition:** Train the same recipe with chosen/rejected randomly swapped; it must not reach the real run's accuracy (expect roughly chance, 0.5, give or take noise; recorded 0.13-0.33 with an unbalanced swap).
**2. In plain words:** Proof the model learned the labels and not a quirk.
**3. Example to quote:** > "With chosen and rejected randomly swapped the same recipe scored 0.13 to 0.33 instead of 1.00, so the real run's accuracy is learning, not a bug."
**Seen in:** `main.py`, section 7

## Held-out generalisation split  `(concept)`
**1. Definition:** Half of the test pairs use reply phrases never seen in training.
**2. In plain words:** Not just memorising phrases.
**3. Example to quote:** > "Half the test pairs use reply phrases never seen in training, so accuracy there measures generalisation."
**Seen in:** `common.py`

## Right padding vs left padding  `(concept)`
**1. Definition:** Pads at the end (right) or the start (left) of a sequence; reward models usually read the last real token, so the index must account for it.
**2. In plain words:** Where the blanks go.
**3. Example to quote:** > "With right padding the last real token sits at attention_mask.sum() - 1, not at index -1."
```python
tok.padding_side = 'right'
```
**Seen in:** `main.py`, section 1

## AutoModel (no LM head)  `(library)`
**1. Definition:** Loads the transformer body and returns hidden states instead of vocabulary logits.
**2. In plain words:** The engine without the output layer.
**3. Example to quote:** > "last_hidden_state shape is (B, L, 768)."
```python
AutoModel.from_pretrained('distilgpt2')
```
**Seen in:** `main.py`, `RewardModel`

## Reward hacking  `(concept)`
**1. Definition:** A policy exploits flaws of a learned reward to score high without being good.
**2. In plain words:** Gaming the grader.
**3. Example to quote:** > "One reason RLVR prefers verifiers where possible."
**Seen in:** `WALKTHROUGH.md` theory

## RLHF reward pipeline  `(concept)`
**1. Definition:** Collect comparisons, train a reward model, then optimise a policy against it with a KL penalty.
**2. In plain words:** The classic recipe.
**3. Example to quote:** > "Collect preferences, train a Bradley-Terry reward model, then optimise the policy against it with a KL leash to the reference."
**Seen in:** `WALKTHROUGH.md` theory

## torch.nn.init.normal_  `(library)`
**1. Definition:** In-place random initialisation with a given std.
**2. In plain words:** Start the head near zero.
**3. Example to quote:** > "The head starts with std 0.02 so initial rewards are close to equal."
```python
nn.init.normal_(self.head.weight, std=0.02)
```
**Seen in:** `main.py`, `RewardModel`
