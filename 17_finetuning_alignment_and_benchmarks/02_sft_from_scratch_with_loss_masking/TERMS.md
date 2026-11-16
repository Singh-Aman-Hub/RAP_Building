# Terms — SFT from scratch with loss masking

Folder: `17_finetuning_alignment_and_benchmarks/02_sft_from_scratch_with_loss_masking` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [17 README](../README.md)

**Terms in this folder:** [Supervised fine-tuning (SFT, instruction tuning)](#supervised-fine-tuning-sft-instruction-tuning) · [Completion-only loss (loss masking)](#completion-only-loss-loss-masking) · [Label shift](#label-shift) · [ignore_index](#ignore_index) · [Padding, attention mask](#padding-attention-mask) · [Packing](#packing) · [EOS appended to the completion](#eos-appended-to-the-completion) · [Gradient clipping](#gradient-clipping) · [AdamW](#adamw) · [Learning rate for full fine-tuning](#learning-rate-for-full-fine-tuning) · [Ranked-choice accuracy vs generated exact match](#ranked-choice-accuracy-vs-generated-exact-match) · [Held-out and unseen-adjective test](#held-out-and-unseen-adjective-test) · [Seeds and determinism](#seeds-and-determinism)

---

## Supervised fine-tuning (SFT, instruction tuning)  `(concept)`
**1. Definition:** Further training with next-token cross-entropy on (prompt, ideal answer) pairs; instruction tuning is SFT on instruction data.
**2. In plain words:** Show worked examples and imitate them.
**3. Example to quote:** > "Generated exact-match went from 0.000 to 1.000 after three epochs on 80 examples."
**Seen in:** `main.py`, `finetune`

## Completion-only loss (loss masking)  `(concept)`
**1. Definition:** Computing the loss only on completion tokens by setting prompt labels to -100.
**2. In plain words:** Learn the answer, not the question.
**3. Example to quote:** > "TRL does this by default for prompt/completion data (completion_only_loss)."
```python
labels = ([-100] * len(p) if mask_prompt else p) + c
```
**Seen in:** `main.py`, `encode`

## Label shift  `(concept)`
**1. Definition:** Position t predicts token t+1, so logits[:, :-1] are compared with labels[:, 1:].
**2. In plain words:** Predict the next word.
**3. Example to quote:** > "Position t predicts token t+1, so I drop the last logit and the first label before the cross-entropy."
```python
F.cross_entropy(logits[:, :-1].reshape(-1, V), y[:, 1:].reshape(-1), ignore_index=-100)
```
**Seen in:** `main.py`, `sft_loss`

## ignore_index  `(syntax)`
**1. Definition:** cross_entropy argument; targets equal to it contribute zero loss and no gradient.
**2. In plain words:** The 'skip' value.
**3. Example to quote:** > "-100 is PyTorch's convention."
```python
F.cross_entropy(..., ignore_index=-100)
```
**Seen in:** `main.py`, `sft_loss`

## Padding, attention mask  `(concept)`
**1. Definition:** Short sequences are filled with a pad id up to the batch length; the attention mask marks real tokens with 1.
**2. In plain words:** Blank space and a note saying where the text ends.
**3. Example to quote:** > "Padded labels are also -100 so padding is never learned."
```python
m[b, :len(i)] = 1
```
**Seen in:** `main.py`, `collate`

## Packing  `(concept)`
**1. Definition:** Concatenating several examples into fixed-length blocks to avoid padding waste.
**2. In plain words:** Fill the suitcase.
**3. Example to quote:** > "Padding wastes about 28% of positions here; packing about 0% (see output)."
**Seen in:** `main.py`, section 9

## EOS appended to the completion  `(concept)`
**1. Definition:** A final end-of-sequence token in the training target teaches the model to stop.
**2. In plain words:** Learn where to stop.
**3. Example to quote:** > "Without it the model keeps generating."
```python
c = tok(row['completion']).input_ids + [tok.eos_token_id]
```
**Seen in:** `main.py`, `encode`

## Gradient clipping  `(concept)`
**1. Definition:** Scale gradients so their global norm is at most a threshold.
**2. In plain words:** A safety cap on step size.
**3. Example to quote:** > "I clip the gradient norm to 1.0 so one rare huge gradient cannot wreck the pretrained weights."
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```
**Seen in:** `main.py`, `finetune`

## AdamW  `(library)`
**1. Definition:** Adam with decoupled weight decay.
**2. In plain words:** The default optimiser for transformers.
**3. Example to quote:** > "AdamW is Adam with decoupled weight decay; I set weight decay to 0 here because the run is tiny."
```python
torch.optim.AdamW(model.parameters(), lr=5e-5)
```
**Seen in:** `main.py`, `finetune`

## Learning rate for full fine-tuning  `(concept)`
**1. Definition:** Typically 1e-5 to 5e-5 for full fine-tuning; adapters use higher rates.
**2. In plain words:** A small step because every weight moves.
**3. Example to quote:** > "Full fine-tuning wants a small learning rate, around 5e-5 here, because every weight moves."
```python
LR = 5e-5
```
**Seen in:** `main.py`, settings

## Ranked-choice accuracy vs generated exact match  `(concept)`
**1. Definition:** Ranked-choice compares log P of the candidate answers; generated EM checks what greedy decoding actually writes.
**2. In plain words:** What the model prefers vs what it says.
**3. Example to quote:** > "Before SFT ranked-choice was 0.875 but generated EM was 0.000."
**Seen in:** `main.py`, section 5-7

## Held-out and unseen-adjective test  `(concept)`
**1. Definition:** Test reviews that never appeared in training, half using adjectives never seen in training.
**2. In plain words:** A fair exam.
**3. Example to quote:** > "Half the test reviews use adjectives never seen in training, so a good score cannot be memorisation."
**Seen in:** `main.py`, `split_acc`

## Seeds and determinism  `(concept)`
**1. Definition:** Fixing torch and python seeds makes shuffles and initialisations repeatable.
**2. In plain words:** Same dice every run.
**3. Example to quote:** > "I seed torch and the shuffler, so a rerun gives the same batches and the same numbers on the same machine."
```python
torch.manual_seed(SEED)
```
**Seen in:** `main.py`, settings
