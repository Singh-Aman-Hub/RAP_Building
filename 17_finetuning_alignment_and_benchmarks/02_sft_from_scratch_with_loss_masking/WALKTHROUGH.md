# Step 02 — SFT from scratch with loss masking

## The story so far
[Step 01](../01_post_training_map_and_chat_templates/WALKTHROUGH.md) showed that a base model only continues text, that a chat template wraps turns in special tokens, and that only the *assistant* tokens should carry the loss (everything else labelled `-100`). It drew the map but trained nothing. The training loop itself you already know from [06 step 02](../../06_llm_training_and_lora/02_training_loop/WALKTHROUGH.md); here we point that loop at the SFT objective.

## The pain point
"Fine-tune on my data" hides four decisions that quietly decide whether it works: which tokens get a loss, how a batch of different-length examples is padded, whether the model learns to *stop*, and how you measure the result. Get the mask wrong and the model wastes capacity re-learning the question template. Skip the end-of-sequence token and it never stops talking. Measure only with a likelihood probe and you can think a model is good when a user would still see nonsense.

## The spark
SFT (supervised fine-tuning) is ordinary next-token cross-entropy with one twist: set the label to `-100` for every token you do not want to be graded on. `F.cross_entropy(..., ignore_index=-100)` then averages the loss over completion tokens only. Add an EOS token to every completion so the model learns to stop, and evaluate two ways (ranked choice and what is actually generated).

## Goal
Hand-write the whole SFT recipe on `distilgpt2` and 80 tiny seeded sentiment examples: encode `(prompt, completion)` into `input_ids` and `labels`, right-pad a batch, compute the shifted masked loss, train with AdamW, and compare *before* and *after* on 40 held-out reviews, half of which use adjectives never seen in training. Then compare completion-only loss with full-sequence loss, show a generated answer before and after, and do the token arithmetic of padding versus packing.

## The theory (read this before the interview)
**SFT loss (completion-only):**
```
sequence:   [ p1 p2 ... pP | c1 c2 ... cC EOS ]        prompt tokens | completion tokens + EOS
labels:     [ -100 ... -100 | c1 c2 ... cC EOS ]        -100 = "do not grade this position"
loss = - (1 / #graded) * sum over graded t of  log p( token_t | tokens before t )
```
Because the model at position `t` predicts token `t+1`, the logits are shifted: `logits[:, :-1]` are compared with `labels[:, 1:]`.

**Why mask the prompt?** You want the model to *produce answers*, not to model the question template. Prompt tokens are the same in every example (`Review: ...\nSentiment:`), so training on them adds easy, repetitive gradient that dilutes the signal and, with small data, can distort the model. TRL exposes this as `completion_only_loss` (prompt/completion data) and `assistant_only_loss` (messages data); see [step 10](../10_benchmark_landscape_2026_and_tools/WALKTHROUGH.md) for the TRL reference files.

**EOS in the labels:** the model only learns to stop if the stop token is a training target. Here the completion is `" positive"` followed by `<|endoftext|>` (token 50256).

**Padding vs packing:** a batch needs equal lengths. *Right-padding* fills short sequences with a pad token and masks those positions (`labels = -100`, `attention_mask = 0`). *Packing* concatenates many short examples into fixed-size blocks to waste (almost) nothing, at the price of needing an attention mask that stops examples from attending to each other (or accepting a small leak).

**Two kinds of evaluation:** *ranked choice* asks "does the model assign higher log-probability to the right label than the wrong one?" (no generation). *Generated exact match* asks "does greedy decoding actually start with the right word?" (what a user sees). They can disagree a lot: a base model often ranks right and still generates something else.

**Full fine-tuning hyper-parameters:** small learning rate (about 1e-5 to 5e-5 for full fine-tuning of a small LM), AdamW, a few epochs, gradient-norm clipping at 1.0.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| SFT | supervised fine-tuning, instruction tuning, behaviour cloning | near | Cross-entropy on target answers; the names stress the data or the idea |
| Loss mask | label masking, `-100` labels, completion-only loss, `assistant_only_loss` | yes | Positions skipped by cross-entropy |
| `ignore_index` | padding_idx (for embeddings, different thing) | no | `ignore_index` skips *loss* positions; `padding_idx` freezes an *embedding* row |
| Padding | right padding, batch padding | near | Fill with pad tokens; here they are masked in loss and attention |
| Packing | sequence packing, concatenation | near | Concatenate examples to fill blocks; needs document-aware attention to be exact |
| Teacher forcing | next-token training | yes | The model sees the true previous tokens, not its own samples |
| Held-out set | test set, unseen data | near | Data never used for training; here also holds unseen adjectives |
| Exact match vs ranked choice | generative vs log-likelihood eval | no | One generates, one scores candidates |

## Run it
```bash
cd 17_finetuning_alignment_and_benchmarks/02_sft_from_scratch_with_loss_masking
python main.py
# from repo root:
HF_HUB_OFFLINE=1 .venv/bin/python 17_finetuning_alignment_and_benchmarks/02_sft_from_scratch_with_loss_masking/main.py
# knobs: SEED, EPOCHS (3), LR (5e-5), BATCH (8)
```
The script trains distilgpt2 twice (about 82 million parameters each). On an idle laptop CPU the one-epoch run took about 55 seconds wall time, so the default 3 epochs is a couple of minutes; runtime varies by machine, and it is much slower when other heavy jobs share the CPU and memory (the run recorded below was on a heavily loaded machine).

## Code, step by step

Lines 1–15 of `main.py` are the header docstring (GOAL / RUN / EXPECTED OUTPUT / NEEDS) already summarised above; the walk starts after it.

### 0. Imports, settings and seeds  (`main.py` lines 16–34)
```python
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import POS_HELD, NEG_HELD, make_sentiment_data, seq_logprob   # noqa: E402

SEED = int(os.environ.get("SEED", 0))
EPOCHS = int(os.environ.get("EPOCHS", 3))
LR = float(os.environ.get("LR", 5e-5))
BATCH = int(os.environ.get("BATCH", 8))
MODEL = "distilgpt2"
torch.manual_seed(SEED)
random.seed(SEED)
```
- **In simple words:** Load the libraries, import the shared data helpers from the module folder, and read the knobs.
- **Exactly:** `sys.path.insert(0, parent folder)` makes `common.py` (in the module folder, one level up) importable. `POS_HELD` and `NEG_HELD` are the adjectives never used in training; `make_sentiment_data()` returns `(train, test)` lists of dicts with `prompt`, `completion`, `label`; `seq_logprob` returns the sum of log-probabilities of a completion given a prompt. Defaults: 3 epochs, learning rate 5e-5, batch size 8. `torch.manual_seed` and `random.seed` make shuffling and any random init repeatable.
- **Why it is here:** One place to change the experiment; and the held-out adjective lists are what makes the test a generalisation test and not a memory test.

### 1. Load tokenizer and the data  (`main.py` lines 36–40)
```python
# ## 1. Load tokenizer and the data
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token                      # GPT-2 has no pad token; reuse EOS (masked out of the loss below)
train, test = make_sentiment_data()
print(f"train={len(train)} test={len(test)}  example: {train[0]['prompt']!r} -> {train[0]['completion']!r}")
```
- **In simple words:** Get the tokenizer, give it a padding symbol, and build the 80 training and 40 test examples.
- **Exactly:** GPT-2's tokenizer has no pad token, so `tok.pad_token = tok.eos_token` reuses id 50256. That is safe *only because* pad positions are masked from the loss and the attention (blocks 3 and 4); otherwise the model would be taught to predict EOS everywhere. The test set is 20 reviews with seen adjectives in new sentences and 20 with unseen adjectives. No duplicates and no train/test leakage by construction (see `common.py`).
- **Why it is here:** The data format `Review: ...\nSentiment:` followed by `" positive"` is the simplest prompt/completion SFT dataset (format 1 of [step 01](../01_post_training_map_and_chat_templates/WALKTHROUGH.md)).

### 2. Turn one example into input_ids and labels  (`main.py` lines 42–52)
```python
# ## 2. Turn one example into input_ids and labels
def encode(row, mask_prompt=True):
    p = tok(row["prompt"]).input_ids                          # prompt token ids
    c = tok(row["completion"]).input_ids + [tok.eos_token_id]  # completion ids + EOS so the model learns to STOP
    ids = p + c                                               # one sequence, (Lp + Lc,)
    labels = ([-100] * len(p) if mask_prompt else p) + c      # -100 = ignored by cross_entropy
    return ids, labels

ids, labels = encode(train[0])
print("\ninput_ids:", ids)
print("labels   :", labels)
```
- **In simple words:** Glue the prompt tokens and the answer tokens into one sequence, and write a second list that says which positions are graded.
- **Exactly:** `tok(text).input_ids` is a `list[int]` (for GPT-2 no BOS is added). `ids` has length `Lp + Lc`. With `mask_prompt=True` the first `Lp` labels are `-100`; with `False` they equal the prompt ids (that is the "full-sequence loss" variant). The completion labels equal the completion ids including EOS. For the first example the real output is 11 prompt tokens, then `4633` (` negative`) and `50256` (EOS), so `labels` is eleven `-100`s followed by `4633, 50256`.
- **Why it is here:** This function *is* SFT's data side. `mask_prompt` is the one flag that separates completion-only from full-sequence training.

### 3. Pad a batch (right padding) and build the attention mask  (`main.py` lines 54–63)
```python
# ## 3. Pad a batch (right padding) and build the attention mask
def collate(rows, mask_prompt=True):
    enc = [encode(r, mask_prompt) for r in rows]
    L = max(len(i) for i, _ in enc)                           # longest sequence in this batch
    x = torch.full((len(enc), L), tok.pad_token_id)           # (B, L) filled with pad
    y = torch.full((len(enc), L), -100)                       # (B, L) pad positions also ignored in the loss
    m = torch.zeros((len(enc), L), dtype=torch.long)          # (B, L) 1 = real token
    for b, (i, l) in enumerate(enc):
        x[b, :len(i)], y[b, :len(l)], m[b, :len(i)] = torch.tensor(i), torch.tensor(l), 1
    return x, y, m
```
- **In simple words:** Make every sequence in the batch the same length by filling the end with padding, and record which positions are real.
- **Exactly:** `x` is `(B, L)` initialised with the pad id; `y` is `(B, L)` initialised with `-100`; `m` is `(B, L)` int tensor of zeros. For each row we copy the real ids/labels into the front and set `m = 1` there. Returns `(input_ids, labels, attention_mask)`. `torch.full(size, value)` creates a tensor filled with `value`.
- **Why it is here:** Pad positions must be invisible twice: to the loss (`-100`) and to attention (`m = 0`). Right padding is correct for training a causal LM; left padding is for batched *generation*.

### 4. The loss: shift by one, cross-entropy, ignore -100  (`main.py` lines 65–69)
```python
# ## 4. The loss: shift by one, cross-entropy, ignore -100
def sft_loss(model, x, y, m):
    logits = model(input_ids=x, attention_mask=m).logits      # (B, L, vocab)
    # position t predicts token t+1, so drop the last logit and the first label
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1), ignore_index=-100)
```
- **In simple words:** Ask the model for scores over the vocabulary at every position, line up "prediction at t" with "answer at t+1", and average the error over graded positions only.
- **Exactly:** The model returns `.logits` of shape `(B, L, 50257)`. `logits[:, :-1]` drops the last position (it predicts a token we do not have); `y[:, 1:]` drops the first label (nothing predicts it). Both are flattened to `(B*(L-1), vocab)` and `(B*(L-1),)`. `F.cross_entropy` applies log-softmax internally, picks the log-probability of the target, negates, and with `ignore_index=-100` averages over the non-ignored entries only (reduction `mean` divides by the *count of graded tokens*, not by `B*(L-1)`).
- **Why it is here:** This 3-line function is the whole objective. The alignment of the shift is the classic off-by-one bug; the label of the first completion token sits at index `Lp`, so it is predicted from the logit at `Lp - 1`, the last prompt position.

### 5. Evaluation: ranked choice, generated exact match, seen vs unseen adjectives  (`main.py` lines 71–98)
```python
# ## 5. Evaluation: accuracy by comparing two candidate completions, plus held-out loss
@torch.no_grad()
def evaluate(model, rows):
    model.eval()
    correct = 0
    for r in rows:
        lp = {w: seq_logprob(model, tok, r["prompt"], " " + w).item() for w in ("positive", "negative")}
        correct += max(lp, key=lp.get) == r["label"]
    x, y, m = collate(rows)
    return correct / len(rows), sft_loss(model, x, y, m).item()

@torch.no_grad()
def gen_exact_match(model, rows):
    """Generative check: greedy-decode 3 tokens and require the first word to equal the label (this is what a user would see)."""
    model.eval()
    hits = 0
    for r in rows:
        enc = tok(r["prompt"], return_tensors="pt")
        out = model.generate(**enc, max_new_tokens=3, do_sample=False, pad_token_id=tok.eos_token_id)
        words = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).split()
        hits += bool(words) and words[0] == r["label"]
    return hits / len(rows)

def split_acc(model):
    held = POS_HELD + NEG_HELD                                # adjectives the model never saw in training
    unseen = [r for r in test if any(a in r["prompt"] for a in held)]
    seen = [r for r in test if r not in unseen]
    return evaluate(model, seen)[0], evaluate(model, unseen)[0]
```
- **In simple words:** Three ways to grade a model: which label is more probable, whether greedy decoding starts with the right word, and the same accuracy split by "seen" and "unseen" adjectives.
- **Exactly:** `@torch.no_grad()` turns off autograd (no graph, less memory). `evaluate` scores both candidate completions with `seq_logprob` (a dict of two floats) and takes `max(lp, key=lp.get)`, the label with the larger log-probability; it also returns the completion-only loss on the whole set. `gen_exact_match` runs `model.generate` with `do_sample=False` (greedy decoding), slices off the prompt tokens (`out[0, L:]`), decodes with `skip_special_tokens=True` (drops EOS), splits on whitespace and compares the first word to the label. `split_acc` partitions `test` by whether the prompt contains one of the four held-out adjectives and returns `(seen_acc, unseen_acc)`.
- **Why it is here:** Ranked choice says what the model *knows*; generated exact match says what a user *sees*; the seen/unseen split separates learning a rule from memorising words. Perplexity-style held-out loss in the sense of [06 step 03](../../06_llm_training_and_lora/03_eval_perplexity_checkpoints/WALKTHROUGH.md) is the third number.

### 6. The training loop (AdamW, shuffled mini-batches)  (`main.py` lines 100–120)
```python
# ## 6. The training loop (AdamW, shuffled mini-batches)
def finetune(mask_prompt):
    torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    rng = random.Random(SEED)
    for ep in range(EPOCHS):
        model.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        tot = 0.0
        for s in range(0, len(order), BATCH):
            x, y, m = collate([train[i] for i in order[s:s + BATCH]], mask_prompt)
            loss = sft_loss(model, x, y, m)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # keeps rare big gradients from wrecking the weights
            opt.step()
            tot += loss.item()
        print(f"  epoch {ep + 1}  mean train loss {tot / (len(order) / BATCH):.3f}")
    return model
```
- **In simple words:** Load a fresh copy of the model, then for each epoch shuffle the 80 examples, cut them into batches of 8, and take one optimiser step per batch.
- **Exactly:** A fresh `from_pretrained` per call means the two training variants start from identical weights (same seed). `AdamW(..., weight_decay=0.0)` keeps momentum and variance estimates per parameter (two extra tensors per weight; that is why full fine-tuning costs about 16 bytes per parameter in mixed precision, see [step 03](../03_peft_family/WALKTHROUGH.md)). The standard five-line step: `zero_grad`, `backward`, `clip_grad_norm_` (rescale all gradients so their global L2 norm is at most 1.0), `step`. `model.train()` turns dropout on. 80 examples with batch 8 gives 10 steps per epoch, 30 in total. The printed epoch loss divides by `len(order) / BATCH = 10`.
- **Why it is here:** This is the loop of [06 step 02](../../06_llm_training_and_lora/02_training_loop/WALKTHROUGH.md) with a masked loss; nothing SFT-specific is hidden in a trainer.

### 7. Before / after  (`main.py` lines 122–136)
```python
# ## 7. Before / after
base = AutoModelForCausalLM.from_pretrained(MODEL)
acc0, loss0 = evaluate(base, test)
s0, u0 = split_acc(base)
print(f"\nBEFORE  ranked-choice acc={acc0:.3f} (seen-adj {s0:.3f}, unseen-adj {u0:.3f})  generated exact-match={gen_exact_match(base, test):.3f}  held-out loss={loss0:.3f}")
print("training with completion-only loss (prompt labels = -100):")
m_c = finetune(True)
acc1, loss1 = evaluate(m_c, test)
s1, u1 = split_acc(m_c)
print(f"AFTER   ranked-choice acc={acc1:.3f} (seen-adj {s1:.3f}, unseen-adj {u1:.3f})  generated exact-match={gen_exact_match(m_c, test):.3f}  held-out loss={loss1:.3f}")
print("training with full-sequence loss (prompt tokens also trained):")
m_f = finetune(False)
acc2, _ = evaluate(m_f, test)
s2, u2 = split_acc(m_f)
print(f"AFTER   ranked-choice acc={acc2:.3f} (seen-adj {s2:.3f}, unseen-adj {u2:.3f})  generated exact-match={gen_exact_match(m_f, test):.3f}   <- full-sequence variant")
```
- **In simple words:** Grade the untouched model, train with the mask and grade again, then train a second copy *without* the mask and grade that one too.
- **Exactly:** `base` is evaluated with the same three probes as the trained models, so the three result lines are directly comparable. `finetune(True)` and `finetune(False)` each return a trained model object. The full-sequence variant is not compared on loss because its training loss includes the prompt tokens, a different quantity.
- **Why it is here:** A number without a "before" is not a result, and the mask-versus-no-mask comparison is the experiment that justifies the technique.

### 8. A generated answer before and after  (`main.py` lines 138–144)
```python
# ## 8. A generated answer before and after
def generate(model, prompt):
    enc = tok(prompt, return_tensors="pt")
    out = model.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc.input_ids.shape[1]:])
q = test[-1]["prompt"]
print(f"\nprompt {q!r}\n  base     -> {generate(base, q)!r}\n  SFT      -> {generate(m_c, q)!r}")
```
- **In simple words:** Show one concrete example so you can see the behaviour change, not just the numbers.
- **Exactly:** Greedy decode up to 4 new tokens; `decode` without `skip_special_tokens` so the EOS the model learned to emit is visible as `<|endoftext|>`.
- **Why it is here:** It proves the model learned to answer *and to stop*: the SFT output ends with the EOS token.

### 9. Padding vs packing: how many positions are wasted?  (`main.py` lines 146–153)
```python
# ## 9. Padding vs packing: how many positions are wasted?
lens = [len(encode(r)[0]) for r in train]                     # real token counts per example
pad_total = sum(max(lens[i:i + BATCH]) * len(lens[i:i + BATCH]) for i in range(0, len(lens), BATCH))   # padded batch area
real = sum(lens)
BLOCK = 64
packed_blocks = -(-real // BLOCK)                             # ceil: greedy concatenation into fixed 64-token blocks
print(f"\nreal tokens={real}  padded positions={pad_total} (waste {1 - real / pad_total:.1%})  "
      f"packed into {packed_blocks} blocks of {BLOCK} (waste {1 - real / (packed_blocks * BLOCK):.1%})")
```
- **In simple words:** Count how many array cells hold real tokens versus padding, for the batches we actually used and for a packed layout.
- **Exactly:** `lens` is the length of each encoded example. For each consecutive group of `BATCH` examples the padded area is `max length * group size`; the sum is `pad_total`. Packing needs only `ceil(real / 64)` blocks of 64; `-(-a // b)` is the integer-ceiling trick. Waste is `1 - real / area`. This is pure arithmetic: no model is involved, and it assumes examples are in the original order and that a block may cut an example in two.
- **Why it is here:** Padding waste is wasted compute. With uniform tiny reviews it is small; with real data (lengths from 20 to 2,000 tokens) padding can waste a third or more of every batch, which is why trainers offer packing or length-grouped batching.

### 10. TRY IT / BREAK IT comments  (`main.py` lines 155–156)
```python
# TRY IT: EPOCHS=1 python main.py   (how much does one epoch buy?)   LR=5e-4 python main.py   (too hot?)
# BREAK IT: delete the `labels[...] = -100` mask by calling finetune(False) only, then look at what the prompt tokens cost.
```
- **In simple words:** Suggested experiments; the full-sequence run in section 7 already is the BREAK IT.
- **Exactly:** Comments only.
- **Why it is here:** Convention of the module.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `[-100] * len(p) + c` | list of `len(p)` ignored labels followed by the real completion ids |
| `torch.full((B, L), v)` | tensor of shape `(B, L)` filled with `v` |
| `x[b, :n] = torch.tensor(i)` | write a list into the first `n` positions of row `b` |
| `logits[:, :-1]` / `y[:, 1:]` | drop the last logit / the first label: the one-token shift |
| `.reshape(-1, vocab)` | flatten batch and time into one axis of predictions |
| `max(d, key=d.get)` | the dictionary key with the largest value |
| `-(-a // b)` | ceiling division |
| `hits += bool(words) and words[0] == label` | adds 1 for a hit (`True`), 0 otherwise; `bool(words)` guards an empty answer |
| `@torch.no_grad()` | decorator: run the function without building an autograd graph |

## Libraries & functions
| Function | What it does | Internally |
|---|---|---|
| `F.cross_entropy(logits, targets, ignore_index=-100)` | mean negative log-likelihood of the targets | log-softmax over the vocabulary, gather target log-probabilities, drop ignored rows, average |
| `torch.optim.AdamW` | optimiser with decoupled weight decay | keeps first and second moment estimates per parameter |
| `clip_grad_norm_(params, 1.0)` | caps the global gradient norm | computes the total L2 norm; multiplies every gradient by `1.0 / norm` if norm exceeds 1.0 |
| `model.generate(..., do_sample=False)` | greedy decoding | loops forward pass, argmax of last position, append token, uses cached keys/values |
| `AutoModelForCausalLM.from_pretrained` | load weights with an LM head | builds the architecture from `config.json`, loads `safetensors` |
| `seq_logprob` (`common.py`) | sum of log-probabilities of the completion tokens | concatenates prompt and completion ids, one forward pass, log-softmax, gather the true next-token log-probabilities, sum over completion positions |

## Execution trace (real run)
```
train=80 test=40  example: 'Review: The book was disgusting.\nSentiment:' -> ' negative'

input_ids: [14832, 25, 383, 1492, 373, 23374, 13, 198, 31837, 3681, 25, 4633, 50256]
labels   : [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 4633, 50256]

BEFORE  ranked-choice acc=0.875 (seen-adj 0.850, unseen-adj 0.900)  generated exact-match=0.000  held-out loss=5.653
training with completion-only loss (prompt labels = -100):
  epoch 1  mean train loss 1.485
  epoch 2  mean train loss 0.045
  epoch 3  mean train loss 0.000
AFTER   ranked-choice acc=1.000 (seen-adj 1.000, unseen-adj 1.000)  generated exact-match=1.000  held-out loss=0.000
training with full-sequence loss (prompt tokens also trained):
  epoch 1  mean train loss 2.984
  epoch 2  mean train loss 1.286
  epoch 3  mean train loss 0.809
AFTER   ranked-choice acc=0.975 (seen-adj 1.000, unseen-adj 0.950)  generated exact-match=0.975   <- full-sequence variant

prompt 'Review: The soup was excellent.\nSentiment:'
  base     -> ' 5.5 stars'
  SFT      -> ' positive<|endoftext|>'

real tokens=1093  padded positions=1192 (waste 8.3%)  packed into 18 blocks of 64 (waste 5.1%)
```
(Progress bars and a torch "Redirects" note on stderr are omitted.)
- **Before**: ranked-choice accuracy is already 0.875 (the model finds " negative" or " positive" likelier than the other in most cases) but generated exact match is **0.000**: the base model starts its answer with something else (` 5.5 stars`). Knowing the right label is not the same as saying it.
- The held-out loss before training is 5.653 nats per graded token: it is large, plausibly because a base model gives very little probability to the EOS token right after the label (the loss also counts that token); the run does not isolate the cause.
- **After SFT with the mask**: ranked-choice 1.000 on seen *and* unseen adjectives, generated exact match 1.000, loss about 0 (printed 0.000). The unseen-adjective accuracy shows the model generalises the rule (sentiment of an adjective) rather than memorising the six training words per class. The caveat: a test set of 40 reviews is tiny, and one more example changes accuracy by 2.5 points; a confidence interval would be wide ([step 08](../08_evaluation_from_scratch/WALKTHROUGH.md) builds that tooling).
- **Full-sequence loss** reaches 0.975 / 0.975, close but not equal, and its training loss is higher (0.809 at epoch 3 vs 0.000) because it also has to predict the repetitive prompt tokens. Differences of one example (0.975 vs 1.000) on 40 items are within noise, so do not read this as proof that masking always wins; the principled argument is about what you spend gradient on.
- **The sample**: base writes ` 5.5 stars`; the SFT model writes ` positive<|endoftext|>`, i.e. the label and then the stop token.
- **Padding vs packing**: 1,093 real tokens sit in 1,192 padded positions (8.3% waste) versus 18 blocks of 64 = 1,152 positions (5.1% waste). The packed waste is only the last partial block. Here the examples are short and similar in length; with realistic length spread the gap is far larger.

## Experiments (change ONE thing)
| Change | What you will see | Why |
|---|---|---|
| default | the trace above | baseline |
| (measured) `EPOCHS=1` | completion-only: ranked-choice 1.000 (seen 1.000, unseen 1.000), generated EM 1.000, held-out loss 0.108. Full-sequence: ranked-choice 0.525, generated EM 0.525 | one epoch (10 steps) is enough with the mask; without it the optimiser spends the same steps on predicting the repetitive prompt and has barely learned the label yet (this is the clearest evidence for masking in this step) |
| (not run) `BATCH=2` | more optimiser steps per epoch (40) and typically faster early loss decrease, noisier curve | step count = examples / batch size |
| (not run) mask the label too (train only on the prompt) | generated exact match stays near 0.000 | the label is the only thing that teaches the answer |

## Debugging practice
**BREAK IT:** drop the shift and compare `logits` with `y` at the same positions, i.e. replace the last line of `sft_loss` with `F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)`. There is no exception: both tensors have the same number of rows. The symptom is that the loss looks plausible but the model learns to *copy the current token* (predict `t` from `t`), the accuracy probes barely move or generation becomes degenerate. The way to catch it is a unit check: take one example, print the position of the first non-`-100` label (`Lp`) and confirm it is compared with the logit at index `Lp - 1`. A cheap sanity rule: a correct next-token loss on a *pretrained* model should start near the model's natural cross-entropy (a few nats), not near 0.

**Second drill:** forget `attention_mask=m` in `sft_loss`. GPT-2 then attends to pad tokens (which equal EOS here). For right-padded training the effect is small because pads come after the real tokens and the causal mask hides them from earlier positions; the bug bites when you later evaluate with left padding. Know why: in a causal model, padding at the *end* is invisible by construction; padding at the *start* is not.

## Common mistakes
- Training on the prompt tokens and then wondering why a tiny dataset overfits the template.
- Forgetting the EOS in the completion: the fine-tuned model never stops.
- Using the pad token id as a *label* without masking it (the model learns to predict padding/EOS everywhere).
- Mixing up the shift: logits `[:, :-1]` with labels `[:, 1:]`.
- Evaluating only by log-likelihood ranking and shipping a model whose generated output is still wrong.
- Testing on adjectives or templates that also appear in training and calling it generalisation.
- Using a large learning rate for full fine-tuning of a small model and overwriting what it knew.

## Rebuild checklist
1. `encode(row, mask_prompt)` returning `(ids, labels)` with `-100` on the prompt and an EOS after the completion.
2. `collate` with right padding, a labels tensor initialised to `-100`, and an attention mask.
3. `sft_loss` with the one-token shift and `ignore_index=-100`.
4. `finetune`: fresh model, `AdamW`, shuffle, batch, `zero_grad / backward / clip_grad_norm_ / step`.
5. Two evaluations: ranked choice by `seq_logprob` and greedy-generated exact match.
6. A seen-vs-unseen split of the test set and a before/after print.
7. The padding versus packing token arithmetic.

## Say it aloud
"SFT is next-token cross-entropy where I set the labels of the prompt and the padding to -100, so the loss averages over the answer tokens only, and I append an EOS to the answer so the model learns to stop. The logits at position t are compared with the label at t+1. On my toy task the base model already ranked the right label highly but generated exact match was zero; after SFT it was perfect, including on adjectives it had never seen, though with 40 test items I would want a confidence interval before claiming more. Padding wastes compute and packing avoids it, but packed examples need document-aware attention masks."

## The new problem (cliffhanger)
Full fine-tuning updated every weight of an 82-million-parameter model, and kept an optimiser state for each. That is fine here; for a 7B model it needs well over a hundred gigabytes. How do you adapt a model by training only a tiny fraction of the parameters, and what are all the ways to do it (LoRA, DoRA, adapters, IA3, prompt and prefix tuning, QLoRA)? On to [step 03](../03_peft_family/WALKTHROUGH.md).
