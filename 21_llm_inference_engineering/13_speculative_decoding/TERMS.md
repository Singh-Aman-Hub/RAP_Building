# Terms — Speculative decoding

Folder: `21_llm_inference_engineering/13_speculative_decoding` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [21 README](../README.md)

**Terms in this folder:** [Speculative decoding](#speculative-decoding) · [Draft model and target model](#draft-model-and-target-model) · [gamma (draft length)](#gamma-draft-length) · [Acceptance rate](#acceptance-rate) · [Tokens per target pass](#tokens-per-target-pass) · [Parallel verification](#parallel-verification) · [Rejection sampling acceptance rule](#rejection-sampling-acceptance-rule) · [Residual distribution](#residual-distribution) · [Expected speedup formula](#expected-speedup-formula) · [Cumulative product for the accepted prefix](#cumulative-product-for-the-accepted-prefix) · [Medusa](#medusa) · [EAGLE](#eagle) · [Multi-token prediction (MTP)](#multi-token-prediction-mtp) · [torch.no_grad decorator](#torchno_grad-decorator)  · [Prompt lookup (n-gram speculation)](#prompt-lookup-n-gram-speculation) · [Expected tokens per target pass formula](#expected-tokens-per-target-pass-formula) · [EAGLE-3 and MTP heads](#eagle-3-and-mtp-heads)

---

## Speculative decoding  `(concept)`
**1. Definition:** A small draft model proposes several tokens; the large target model verifies them in one forward pass and keeps the correct prefix, so the output matches the target.
**2. In plain words:** A junior drafts, the senior checks a paragraph at once.
**3. Example to quote:** > "Output is identical to the target alone for greedy decoding."
**Seen in:** `main.py`, `speculative_greedy`

## Draft model and target model  `(concept)`
**1. Definition:** The cheap proposer and the expensive verifier; they must share a tokenizer.
**2. In plain words:** distilgpt2 drafts, gpt2 verifies.
**3. Example to quote:** > "82M vs 124M parameters here."
**Seen in:** `main.py`, setup

## gamma (draft length)  `(concept)`
**1. Definition:** Number of tokens the draft proposes per round.
**2. In plain words:** How far ahead to guess.
**3. Example to quote:** > "We test 2, 4, 8."
**Seen in:** `main.py`, `speculative_greedy`

## Acceptance rate  `(concept)`
**1. Definition:** Fraction of proposed tokens the target agrees with.
**2. In plain words:** How good the junior is.
**3. Example to quote:** > "See the real run table."
**Seen in:** `main.py`, results

## Tokens per target pass  `(concept)`
**1. Definition:** Average tokens produced per expensive forward pass.
**2. In plain words:** The effective speedup in passes.
**3. Example to quote:** > "Greater than 1 means fewer target passes than plain decoding."
**Seen in:** `main.py`, table

## Parallel verification  `(concept)`
**1. Definition:** One forward pass over prefix + proposals gives the target's prediction at every position.
**2. In plain words:** Checking five answers costs one pass.
**3. Example to quote:** > "target(prop).logits[0] has len(prop) rows."
**Seen in:** `main.py`, `speculative_greedy`

## Rejection sampling acceptance rule  `(concept)`
**1. Definition:** Accept draft token x with probability min(1, p(x)/q(x)); on rejection sample from normalise(max(0, p - q)). The output distribution equals the target's.
**2. In plain words:** A fair correction that keeps the target's statistics.
**3. Example to quote:** > "Empirical distribution matched p."
**Seen in:** `main.py`, section 4

## Residual distribution  `(concept)`
**1. Definition:** normalise(max(0, p - q)): where the target has more mass than the draft.
**2. In plain words:** What the draft under-proposed.
**3. Example to quote:** > "Used after a rejection."
**Seen in:** `main.py`, section 4

## Expected speedup formula  `(concept)`
**1. Definition:** (1 - a^(g+1)) / ((1 - a)(g c + 1)) for acceptance a, draft length g and draft/target cost ratio c.
**2. In plain words:** Gain needs a cheap draft and high acceptance.
**3. Example to quote:** > "With c = 0.5 the gain is small; with c = 0.05 it is large."
**Seen in:** `main.py`, `expected_speedup`

## Cumulative product for the accepted prefix  `(syntax)`
**1. Definition:** agree.cumprod(0).sum() counts leading ones.
**2. In plain words:** Stop counting at the first disagreement.
**3. Example to quote:** > "k = number of accepted tokens."
**Seen in:** `main.py`, step (c)

## Medusa  `(concept)`
**1. Definition:** Extra decoding heads on the target model that predict several future tokens, verified with a tree attention.
**2. In plain words:** No separate draft model.
**3. Example to quote:** > "A draft-free variant."
**Seen in:** `WALKTHROUGH.md`

## EAGLE  `(concept)`
**1. Definition:** A lightweight draft head working on the target's hidden features to propose tokens with high acceptance.
**2. In plain words:** A smarter small drafter.
**3. Example to quote:** > "Used in serving engines."
**Seen in:** `WALKTHROUGH.md`

## Multi-token prediction (MTP)  `(concept)`
**1. Definition:** Training objective with extra heads predicting tokens further ahead; the heads can serve as drafters.
**2. In plain words:** DeepSeek-V3 and Step 3.5 Flash use it.
**3. Example to quote:** > "MTP heads double as speculative drafters."
**Seen in:** `WALKTHROUGH.md`

## torch.no_grad decorator  `(syntax)`
**1. Definition:** Disables gradient tracking for a function.
**2. In plain words:** Inference only.
**3. Example to quote:** > "@torch.no_grad() on greedy."
**Seen in:** `main.py`

## Prompt lookup (n-gram speculation)  `(concept)`
**1. Definition:** Draft tokens are copied from the continuation of the most recent matching n-gram already in the context; no draft model.
**2. In plain words:** Guess that the answer repeats the text above.
**3. Example to quote:** > "Great for summarisation, code editing and RAG where outputs copy the input."
**Seen in:** `WALKTHROUGH.md`, variants table

## Expected tokens per target pass formula  `(concept)`
**1. Definition:** `(1 - alpha^(k+1)) / (1 - alpha)` for acceptance alpha and draft length k.
**2. In plain words:** The accepted-prefix average plus the one guaranteed target token.
**3. Example to quote:** > "With alpha 0.8 and k 4 you get 3.36 tokens per target pass."
**Seen in:** `WALKTHROUGH.md`, theory

## EAGLE-3 and MTP heads  `(concept)`
**1. Definition:** Draft heads that predict from the target's hidden features (EAGLE-3) or heads trained jointly with the model (MTP, as in DeepSeek-V3), instead of a separate draft LM.
**2. In plain words:** A tiny add-on predictor sitting on the big model.
**3. Example to quote:** > "MTP heads reach high acceptance because they were trained with the model."
**Seen in:** `WALKTHROUGH.md`, variants table
