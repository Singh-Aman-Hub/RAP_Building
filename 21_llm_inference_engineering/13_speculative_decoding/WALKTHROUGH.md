# Step 13 — Speculative decoding

## The story so far
[Step 12](../12_continuous_batching_prefix_caching/WALKTHROUGH.md) made the *scheduler* efficient: continuous batching keeps every slot busy and prefix caching skips repeated prompt work. But for a single request, generation is still one token per forward pass, and [step 09](../09_prefill_decode_roofline/WALKTHROUGH.md) showed why that is wasteful: at batch 1 each decode step streams all the weights to do almost no arithmetic. The GPU's compute units sit idle.

## The pain point
Decode is memory-bound, so the latency of one token is fixed by weight bytes divided by bandwidth, no matter how clever the kernels. A 70B fp16 model on one GPU stream needs about 140 GB per token. You cannot make one sequential step faster, and you cannot batch away a single user's latency (batching raises throughput, not one user's speed, as [step 10](../10_latency_throughput_benchmark/WALKTHROUGH.md) measured).

## The spark
Verifying is cheaper than generating. A forward pass scoring 5 tokens in parallel costs nearly the same as scoring 1 (same weight read, spare compute from step 09). So let a cheap **draft** guess the next gamma tokens, then let the big **target** model check all gamma at once in a single pass. Keep the longest correct prefix plus one token from the target. Every accepted token is a step the target did not have to take sequentially, and the output is *exactly* what the target alone would produce.

## Goal
Implement greedy speculative decoding with distilgpt2 as draft and gpt2 as target; prove the output is identical to target-only greedy; measure acceptance rate and tokens per target pass; check the wall-clock honestly; and verify the general sampling acceptance rule on a toy distribution.

## The theory (read this before the interview)

**The loop (one round).**
```
context ids ──► draft proposes x1..xk, one token at a time (cheap, sequential)
            ──► TARGET scores [context + x1..xk] in ONE forward pass -> its own pick at k+1 positions
            ──► accept x1..xj while they equal the target's choice (j <= k)
            ──► append x1..xj + ONE target token (the correction at the first mismatch, or a bonus if all k matched)
```
Every round yields between 1 and k+1 tokens for one target pass; at worst you lose nothing but the draft's work.

**Expected tokens per target pass.** If each draft token is accepted independently with probability alpha (the *acceptance rate*), the number of tokens produced per round is a truncated geometric series:

`E[tokens per target pass] = 1 + alpha + alpha^2 + ... + alpha^k = (1 - alpha^(k+1)) / (1 - alpha)`

(The leading 1 is the guaranteed target token.) Limits: alpha -> 0 gives 1 (plain decoding); alpha -> 1 gives k+1.

**Speedup.** Let c = cost of one draft step divided by cost of one target step. A round costs `k*c + 1` target-step units, so

`speedup = (1 - alpha^(k+1)) / ((1 - alpha) * (k*c + 1))`   (Leviathan et al. 2023; the formula in `main.py`)

Table for c = 0.05 (a small draft next to a big target), computed from the formula:

| alpha \ k | k=2 | k=4 | k=8 |
|---|---|---|---|
| 0.5 | 1.59x | 1.61x | 1.43x |
| 0.7 | 1.99x | 2.31x | 2.28x |
| 0.8 | 2.22x | 2.80x | 3.09x |
| 0.9 | 2.46x | 3.41x | 4.38x |

Two lessons: (1) there is an optimal k and past it you only add draft cost; (2) acceptance rate matters more than draft length. With c = 0.65, as here (distilgpt2 is not much cheaper than gpt2), speedup is below 1.

**When is output exactly the same?** Greedy: yes, because we only accept tokens equal to the target's argmax. Sampling: use the rejection rule: accept draft token x with probability `min(1, p(x)/q(x))` (p = target, q = draft distribution); on rejection resample from the residual `norm(max(0, p - q))`. This provably samples from p exactly (part 4 of `main.py` checks it empirically).

**Why it works physically.** From step 09: decode intensity at batch 1 is about 1 FLOP/byte, far below the ridge, so there is idle compute. Verifying k+1 tokens in one pass raises the intensity roughly (k+1)-fold at almost no extra time. Speculation turns spare compute into lower latency. It helps most at small batch (latency-bound, memory-bound); at large batch the GPU is already busy and the extra verified tokens are no longer free, so the gain shrinks.

**Variants of where the guesses come from (as of Nov 2026, all current practice):**

| Method | What proposes tokens | Needs | Notes |
|---|---|---|---|
| Draft model | a small separate LM (same tokenizer) | a good small model, extra memory | classic; this step |
| n-gram / prompt lookup | copy the continuation of the last matching n-gram found *in the prompt* | nothing (no model) | great for summarisation, code editing, RAG, where outputs copy the input |
| Medusa | several extra decoding heads on the target that predict tokens t+2, t+3, ... | fine-tune the heads | tree of candidates verified together |
| EAGLE / EAGLE-3 | a light draft head that predicts the target's next hidden features (EAGLE-3 uses features from several layers) | train a small head per target | among the highest reported acceptance rates; supported in the mainstream engines |
| MTP (multi-token prediction) heads | extra prediction modules trained *with* the model (e.g. in DeepSeek-V3), reused for speculation | built in at pre-training | acceptance is high because the head was trained jointly |

All of them keep the same verify step; they differ in how cheaply and accurately the k guesses are produced. The engines that expose these (vLLM, SGLang, TensorRT-LLM) are surveyed in [step 15](../15_serving_engines_landscape/WALKTHROUGH.md); none of them runs on this laptop, so this step implements the draft-model version by hand.

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Speculative decoding (speculative sampling, assisted generation) | HF `assistant_model` | Yes | A cheap model proposes, the target verifies; lossless. |
| Draft model (assistant model, speculator, proposer) | small model | Yes | Cheap model that guesses tokens. |
| Target model (verifier, main model, large model) | | Yes | The model whose output distribution you must preserve. |
| gamma / k (draft length, num speculative tokens) | `num_speculative_tokens` | Yes | Tokens proposed per round. |
| Acceptance rate (alpha, accept ratio) | | Near | Fraction of proposed tokens kept; sometimes counted per position. |
| Tokens per target pass (mean accepted length, tokens per step) | | Related | `(1-alpha^(k+1))/(1-alpha)` under independent acceptance. |
| Prompt lookup decoding (n-gram speculation, lookahead from prompt) | `prompt_lookup_num_tokens` in HF | Yes | Guesses come from copying text already in the context. |
| Lookahead decoding | Jacobi decoding | Different | Uses the target alone, iterating on guesses; no draft model. |
| EAGLE-3 / Medusa heads / MTP heads | draft heads | Related | Extra small heads on the target instead of a separate model. |

## Run it
```bash
cd 21_llm_inference_engineering/13_speculative_decoding
python main.py
# from repo root (loads gpt2 + distilgpt2: use the heavy-run lock if others are running):
HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/13_speculative_decoding/main.py
```
About 1 minute on CPU; no download (gpt2 and distilgpt2 are cached).

## Code, step by step

### 1. Header docstring  (`main.py` lines 1–13)

```python
"""
GOAL:  Speculative decoding with a small DRAFT model (distilgpt2) and a big TARGET model (gpt2, same tokenizer).
       Implement the greedy version (output provably identical to the target alone), measure acceptance rate and
       the number of target passes saved, and verify the general rejection-sampling rule on a toy distribution.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/13_speculative_decoding/main.py
EXPECTED OUTPUT (shortened):
    identical to target-only greedy: True
    gamma  acceptance  tokens/target pass  ...
    empirical distribution of speculative sampling == target: ...
NEEDS: torch, numpy, transformers; cached gpt2 + distilgpt2.  ~1 min on CPU.
"""
```

- **In simple words:** What is built and expected.
- **Exactly:** Docstring in the required format.
- **Why it is here:** Sets expectations.

### 2. Imports, models, shared tokenizer  (`main.py` lines 14–28)

```python
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
torch.set_num_threads(2)
tok = AutoTokenizer.from_pretrained("gpt2")                                       # distilgpt2 shares this tokenizer
target = AutoModelForCausalLM.from_pretrained("gpt2").eval()
draft = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
n_t, n_d = (sum(p.numel() for p in m.parameters()) / 1e6 for m in (target, draft))
print(f"target gpt2 {n_t:.0f}M params | draft distilgpt2 {n_d:.0f}M params | same vocab: {target.config.vocab_size == draft.config.vocab_size}")
```

- **In simple words:** Load gpt2 (target) and distilgpt2 (draft) and print sizes.
- **Exactly:** Both use the same 50,257-token vocabulary, so token ids mean the same thing to both: a hard requirement of draft-model speculation. `set_num_threads(2)` and `manual_seed(0)` keep the run repeatable; `.eval()` switches off dropout.
- **Why it is here:** The two roles of the technique.

### 3. Plain greedy decoding (the reference)  (`main.py` lines 31–37)

```python
# ---- 1. Plain greedy decoding with the target (the reference answer) --------------------------------------------------------
@torch.no_grad()
def greedy(model, ids, n_new):
    for _ in range(n_new):
        nxt = model(ids).logits[0, -1].argmax()                                   # (vocab,) -> scalar
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
    return ids
```

- **In simple words:** The slow baseline: one forward pass per new token.
- **Exactly:** `model(ids).logits` has shape `(1, L, vocab)`; `[0, -1]` takes the last position; `argmax()` picks the top token (a scalar); `torch.cat` appends it. No KV cache here, to keep the algorithm visible.
- **Why it is here:** The speculative output must equal this.

### 4. Speculative decoding, greedy version  (`main.py` lines 40–64)

```python
# ---- 2. Speculative decoding, greedy version -----------------------------------------------------------------------------------
@torch.no_grad()
def speculative_greedy(ids, n_new, gamma):
    stats = dict(target_passes=0, proposed=0, accepted=0)
    goal = ids.shape[1] + n_new
    while ids.shape[1] < goal:
        # (a) the cheap draft model proposes `gamma` tokens, one by one
        prop = ids
        for _ in range(gamma):
            nxt = draft(prop).logits[0, -1].argmax()
            prop = torch.cat([prop, nxt.view(1, 1)], dim=1)
        # (b) ONE target pass scores the prefix plus all gamma proposals at once (a normal forward pass is parallel over positions)
        logits = target(prop).logits[0]                                           # (len(prop), vocab)
        stats["target_passes"] += 1
        n0 = ids.shape[1]
        target_choice = logits[n0 - 1:].argmax(-1)                                # (gamma + 1,): target's own greedy token after each prefix
        proposed = prop[0, n0:]                                                   # (gamma,)
        # (c) accept the longest prefix of proposals that equals the target's own choice
        agree = (proposed == target_choice[:gamma]).int()
        k = int(agree.cumprod(0).sum())                                           # number of leading agreements
        stats["proposed"] += gamma
        stats["accepted"] += k
        # (d) append the k accepted tokens + ONE token from the target (the correction, or a bonus if all were accepted)
        ids = torch.cat([ids, proposed[:k].view(1, -1), target_choice[k].view(1, 1)], dim=1)
    return ids[:, :goal], stats
```

- **In simple words:** One round: draft proposes gamma tokens, target checks all at once, accept the matching prefix, add one target token.
- **Exactly:** (a) gamma sequential draft steps build `prop`. (b) ONE target pass over `prop` returns logits `(len(prop), vocab)`; `target_choice = logits[n0-1:].argmax(-1)` has `gamma+1` entries: the target's own choice after the context and after each proposal. (c) `agree.cumprod(0).sum()` is the length of the leading run of agreement. (d) append `proposed[:k]` and `target_choice[k]`, the correction (first mismatch) or the bonus (all matched).
- **Why it is here:** The core algorithm; the three counters feed the metrics.

### 5. Run all gammas and prompts, check identical output  (`main.py` lines 67–85)

```python
prompts = ["The capital of France is", "In machine learning, a neural network is", "def fibonacci(n):\n    if n <", "Once upon a time, there was a little"]
N_NEW = 32
print(f"\n{'gamma':>5}{'acceptance':>12}{'tokens/target pass':>20}{'target passes':>15}{'vs plain':>10}  identical")
results = {}
all_same = True
refs = {p: greedy(target, tok(p, return_tensors="pt").input_ids, N_NEW) for p in prompts}   # reference outputs computed ONCE
for gamma in (2, 4, 8):
    acc = prop = passes = 0
    same = True
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids
        ref = refs[p]
        out, st = speculative_greedy(ids, N_NEW, gamma)
        same &= torch.equal(out, ref)                                            # the whole point: identical output
        acc, prop, passes = acc + st["accepted"], prop + st["proposed"], passes + st["target_passes"]
    results[gamma] = (acc / prop, passes)
    all_same &= same
    print(f"{gamma:>5}{acc / prop:>12.2f}{len(prompts) * N_NEW / passes:>20.2f}{passes:>15}{passes / (len(prompts) * N_NEW):>10.2f}  {same}")
print("identical to target-only greedy for every gamma and prompt:", all_same)
```

- **In simple words:** Sweep gamma in 2, 4, 8 on four prompts, compare with the reference, print acceptance and tokens per target pass.
- **Exactly:** `refs` (reference outputs) are computed once; `torch.equal` demands the exact same ids; `acc / prop` is acceptance; `prompts*N_NEW / passes` is tokens per target pass.
- **Why it is here:** The correctness check plus the headline numbers.

### 6. Wall-clock honesty check  (`main.py` lines 87–94)

```python
# ---- 3. Wall-clock honesty check: speedup needs the draft to be MUCH cheaper than the target -------------------------------------
ids = tok(prompts[0], return_tensors="pt").input_ids
t0 = time.time(); greedy(target, ids, N_NEW); t_plain = time.time() - t0
t0 = time.time(); greedy(draft, ids, N_NEW); t_draft = time.time() - t0
c = t_draft / t_plain                                                             # cost of one draft step relative to one target step
t0 = time.time(); speculative_greedy(ids, N_NEW, 4); t_spec = time.time() - t0
print(f"\nmeasured: target-only {t_plain:.2f}s | draft-only {t_draft:.2f}s (c = {c:.2f}) | speculative gamma=4 {t_spec:.2f}s -> speedup {t_plain / t_spec:.2f}x")
print("(no KV cache here, and distilgpt2 is only 2x cheaper than gpt2: the wall-clock gain is small or negative)")
```

- **In simple words:** Time target-only, draft-only and speculative.
- **Exactly:** `c = t_draft / t_plain` is the cost ratio that the speedup formula needs; speculative time is measured for gamma=4 on the first prompt.
- **Why it is here:** Fewer target passes is not the same as less time.

### 7. The speedup formula  (`main.py` lines 97–104)

```python
def expected_speedup(alpha, gamma, c):
    """Leviathan et al.: tokens per round (1 - a^(g+1)) / (1 - a), cost per round g*c + 1 target-pass units."""
    return (1 - alpha ** (gamma + 1)) / (1 - alpha) / (gamma * c + 1)


alpha = results[4][0]
print(f"formula with measured acceptance {alpha:.2f}: c=0.5 -> {expected_speedup(alpha, 4, 0.5):.2f}x | "
      f"c=0.05 (e.g. 70B target, 3B draft... ) -> {expected_speedup(alpha, 4, 0.05):.2f}x | c=0.01 -> {expected_speedup(alpha, 4, 0.01):.2f}x")
```

- **In simple words:** Closed-form speedup from acceptance, gamma and cost ratio.
- **Exactly:** tokens per round `(1-a^(g+1))/(1-a)` divided by round cost `g*c + 1`; evaluated with the measured alpha for c = 0.5, 0.05, 0.01.
- **Why it is here:** Shows what a realistic draft (20 to 100x cheaper) would give.

### 8. The sampling rule on a toy distribution  (`main.py` lines 106–122)

```python
# ---- 4. The general (sampling) rule: accept draft token x with prob min(1, p(x)/q(x)); on rejection resample from norm(max(0, p - q)) ------
rng = np.random.default_rng(0)
p = np.array([0.5, 0.3, 0.15, 0.05])                                              # target distribution over 4 tokens
q = np.array([0.25, 0.25, 0.25, 0.25])                                            # draft distribution (quite wrong)
samples = np.zeros(4)
accepted = 0
TRIALS = 50_000
for _ in range(TRIALS):
    x = rng.choice(4, p=q)                                                        # draft samples a token
    if rng.random() < min(1.0, p[x] / q[x]):                                      # accept with prob min(1, p/q)
        samples[x] += 1
        accepted += 1
    else:
        resid = np.maximum(p - q, 0)                                              # residual distribution
        samples[rng.choice(4, p=resid / resid.sum())] += 1                        # resample from it
print("\ntarget p                :", p)
print("empirical speculative    :", (samples / TRIALS).round(3), f"(acceptance {accepted / TRIALS:.3f}; theory = sum min(p,q) = {np.minimum(p, q).sum():.3f})")
```

- **In simple words:** Verify that accept-with-min(1, p/q) plus residual resampling reproduces the target distribution.
- **Exactly:** `p` target, `q` uniform draft. Accept x with prob `min(1, p[x]/q[x])`; otherwise sample from `max(p-q, 0)` normalised. 50,000 trials; empirical frequencies match `p`; acceptance equals `sum(min(p, q))`.
- **Why it is here:** This is why speculative sampling is exact, not approximate.

### 9. TRY IT / BREAK IT  (`main.py` lines 124–126)

```python
# TRY IT: change prompts to something unpredictable (random words) and watch acceptance fall; change gamma to 16.
# BREAK IT: replace `target_choice[k]` in the append with `proposed[k]` (index error when k == gamma) or remove cumprod:
#           the output stops matching target-only greedy ("identical" turns False), i.e. you accepted tokens after a mismatch.
```

- **In simple words:** Experiments in comments.
- **Exactly:** Unpredictable prompts lower acceptance; breaking the prefix logic makes the identity check fail.
- **Why it is here:** Practice in diagnosing a lossless-ness bug.

## Syntax explained

| Syntax | Meaning | Tiny example |
|---|---|---|
| `@torch.no_grad()` as a decorator | the whole function runs without building a gradient graph | saves memory and time in inference |
| `logits[n0 - 1:]` | slice from the last context position to the end | the prediction made at position n0-1 is the first proposed token's check |
| `.argmax(-1)` | index of the largest value along the last axis | greedy token choice |
| `(a == b).int().cumprod(0).sum()` | count of leading 1s: the accepted prefix | `[1,1,0,1]` becomes `[1,1,0,0]` then 2 |
| `torch.cat([a, b.view(1, -1)], dim=1)` | append tokens along the sequence axis | `ids` shape `(1, L)` becomes `(1, L+k)` |
| `rng.choice(4, p=q)` | sample an index with given probabilities | draft samples a token |

## Libraries & functions

| Name | What it does | Internally |
|---|---|---|
| `AutoModelForCausalLM.from_pretrained` | load a GPT-2-family model | builds the architecture and fills the weights from the cached safetensors |
| `model(ids).logits` | one forward pass | returns `(batch, seq, vocab)` scores for every position in parallel; this parallelism is what verification exploits |
| `torch.manual_seed`, `torch.set_num_threads` | reproducibility, CPU threads | fix RNG state and the intra-op thread pool |
| `np.random.default_rng(0)` | seeded random generator | NumPy's PCG64 |
| `torch.equal(a, b)` | exact tensor equality | used to prove the outputs are identical |

## Execution trace (real run)

Real output (CPU, M1, 2 threads; timing lines vary by machine):

```
target gpt2 124M params | draft distilgpt2 82M params | same vocab: True

gamma  acceptance  tokens/target pass  target passes  vs plain  identical
    2        0.75                2.42             53      0.41  True
    4        0.62                3.28             39      0.30  True
    8        0.47                4.27             30      0.23  True
identical to target-only greedy for every gamma and prompt: True

measured: target-only 1.33s | draft-only 0.87s (c = 0.65) | speculative gamma=4 1.20s -> speedup 1.11x
(no KV cache here, and distilgpt2 is only 2x cheaper than gpt2: the wall-clock gain is small or negative)
formula with measured acceptance 0.62: c=0.5 -> 0.79x | c=0.05 (e.g. 70B target, 3B draft... ) -> 1.98x | c=0.01 -> 2.28x

target p                : [0.5  0.3  0.15 0.05]
empirical speculative    : [0.5   0.302 0.149 0.049] (acceptance 0.703; theory = sum min(p,q) = 0.700)
```

Reading the numbers:
- **Identical output** for every gamma and prompt: speculative greedy is lossless. This is the correctness test to run on any implementation.
- **Acceptance rate falls as gamma grows** (0.75, 0.62, 0.47): later guesses in a round are conditioned on earlier guesses that may be wrong. **Tokens per target pass rises** (2.4, 3.3, 4.3) with diminishing returns in cost: the target does 53, 39, 30 passes instead of 128 sequential ones.
- **Wall-clock gain is small (1.11x) on one prompt** even though target passes drop by 3x, because the draft costs 65 percent of a target step (c = 0.65). The formula with the measured alpha agrees in spirit: c = 0.5 gives 0.79x, c = 0.05 gives 1.98x. (The measured 1.11x is for the first prompt only, which is an easy, predictable one, while alpha is averaged over four prompts; and this code has no KV cache so the baseline is not at its best.) The lesson: **the draft must be an order of magnitude cheaper than the target**.
- **Sampling rule:** empirical distribution `[0.5, 0.302, 0.149, 0.049]` matches the target `[0.5, 0.3, 0.15, 0.05]` even though the draft was uniform, and the acceptance 0.703 matches theory `sum(min(p, q)) = 0.7`.

## Experiments (change ONE thing)

| Change | What you will see | Why |
|---|---|---|
| prompts of random words | acceptance drops sharply | the draft cannot predict unpredictable text |
| `gamma` 16 | tokens per pass up a little, acceptance down, draft work up | diminishing returns, then net loss |
| swap draft for the target itself | acceptance 1.0 and tokens per pass gamma+1 | proves the verifier logic |
| `N_NEW = 64` | same acceptance, more passes | length does not change per-token acceptance |
| uniform `q` replaced by `q = p` in part 4 | acceptance 1.0, residual never used | a perfect draft |

## Debugging practice
**BREAK IT:** remove `.cumprod(0)` in `speculative_greedy` so that `k = int(agree.sum())`. Symptom: the final line `identical to target-only greedy ... : False`. Cause: tokens after the first mismatch can coincide with the target by luck but are conditioned on a wrong earlier token, so they must not be accepted. Fix: the cumulative product turns the 0/1 vector into "1 only while no mismatch yet", so the sum counts the leading run.

**BREAK IT 2:** append `proposed[k]` instead of `target_choice[k]`. Symptom: `IndexError` when all gamma were accepted (k == gamma), and wrong text otherwise (you keep a rejected token).

## Common mistakes
- Believing speculative decoding changes the output distribution. With the correct acceptance rule it does not.
- Using a draft model with a different tokenizer (token ids no longer line up).
- Forgetting the bonus token: when all k are accepted the target pass also yields one extra token for free.
- Benchmarking at large batch and concluding "it does not help": the gain is largest at small batch where decode is memory-bound.
- Ignoring the draft's own cost c in the speedup formula.
- Measuring acceptance on easy prompts and extrapolating to all traffic.

## Rebuild checklist
1. Write `greedy(model, ids, n)`: argmax of the last position, append, repeat.
2. Write `speculative_greedy`: draft k tokens; one target forward on `prop`; `target_choice = logits[n0-1:].argmax(-1)`.
3. Accepted count `k = (proposed == target_choice[:gamma]).cumprod(0).sum()`.
4. Append `proposed[:k]` plus `target_choice[k]`; count target passes.
5. Assert `torch.equal(speculative, greedy)`.
6. Implement `expected_speedup(alpha, gamma, c)`.
7. Verify the min(1, p/q) plus residual rule on a 4-token toy distribution.

## Say it aloud
- "Speculative decoding uses a cheap draft model to guess k tokens and the target model to verify all of them in a single forward pass, keeping the longest matching prefix plus one target token, so the output is identical to the target alone."
- "It works because decode is memory-bound: verifying k tokens costs about the same as generating one, so idle compute becomes lower latency."
- "Expected tokens per target pass is (1 - alpha^(k+1)) / (1 - alpha), and speedup divides that by k*c + 1, where c is the draft-to-target cost ratio, so you need both high acceptance and a very cheap draft."
- "Variants differ in where guesses come from: a draft model, n-gram or prompt lookup for input-copying tasks, Medusa heads, EAGLE-3 feature-level drafts, or multi-token-prediction heads trained with the model, as in DeepSeek-V3."
- "The gain is biggest at small batch and shrinks as the GPU gets busy."

## The new problem (cliffhanger)
Everything so far fits one GPU, or at least one machine. A 70B model in fp16 needs 140 GB, and the KV cache for long contexts adds more. How do you split a model across several GPUs, and what do you do when it does not fit at all? [Step 14, parallelism and offloading](../14_parallelism_and_offloading/WALKTHROUGH.md).
