# Terms — Sampling From Scratch

Folder: `05_llm_parameters/02_sampling_from_scratch` · Files covered: `main.py`, `sampling.py`, `WALKTHROUGH.md`
Back to module: [05 LLM parameters README](../README.md)

**Terms in this folder:** [Sampling vs greedy decoding](#sampling-vs-greedy-decoding) · [Decoding pipeline order](#decoding-pipeline-order) · [Logits (1-D tensor)](#logits-1-d-tensor) · [Repetition penalty](#repetition-penalty) · [set()](#set) · [tensor.clone()](#tensorclone) · [Temperature](#temperature) · [Top-k filter](#top-k-filter) · [torch.topk](#torchtopk) · [torch.where](#torchwhere) · [float("-inf") filtering](#float-inf-filtering) · [Top-p (nucleus) filter](#top-p-nucleus-filter) · [torch.sort](#torchsort) · [torch.cumsum](#torchcumsum) · [masked_fill](#masked_fill) · [torch.softmax](#torchsoftmax) · [torch.multinomial](#torchmultinomial) · [torch.Generator and manual_seed](#torchgenerator-and-manual_seed) · [Entropy (nats)](#entropy-nats) · ["alive" tokens](#alive-tokens) · [Helper module import (import sampling as S)](#helper-module-import-import-sampling-as-s) · [sys.path.insert](#syspathinsert) · [Function with default arguments and **kw](#function-with-default-arguments-and-kw) · [List comprehension](#list-comprehension) · [Dict comprehension](#dict-comprehension) · [torch.bincount](#torchbincount) · [Boolean mask indexing](#boolean-mask-indexing) · [Temperature 0 problem](#temperature-0-problem) · [Trade-off of randomness](#trade-off-of-randomness)

---

## Sampling vs greedy decoding  `(concept)`
**1. Definition:** Greedy decoding always chooses the highest-probability token; sampling draws the next token at random according to the probabilities.
**2. In plain words:** Greedy is repeatable but can be dull or loop. Sampling gives variety. The parameters in this step (temperature, top-k, top-p) control how adventurous the randomness is.
**3. Example to quote:** > "Greedy gave 'to learn from the experience of the programmer' every time; sampling with temperature 0.7 gave different, still sensible, sentences."

**Seen in:** `sampling.py`, `sample_next_token`.

## Decoding pipeline order  `(concept)`
**1. Definition:** The usual sequence of steps turning logits into one token: repetition penalty, divide by temperature, top-k filter, top-p filter, softmax, then sample.
**2. In plain words:** Order matters because each step changes what the next one sees. Filters set removed tokens to `-inf`, so softmax gives them exactly zero probability.
**3. Example to quote:** > "I apply the penalty, then temperature, then top-k and top-p, and only then softmax and sample."

**Seen in:** `sampling.py` docstring.

## Logits (1-D tensor)  `(concept)`
**1. Definition:** The raw scores, one per vocabulary token, for the next position; here a tensor of shape `(50257,)`.
**2. In plain words:** Every function in `sampling.py` takes this one-dimensional tensor. Passing an extra batch dimension `(1, 50257)` produces errors or odd results.
**3. Example to quote:** > "`model(ids).logits[0, -1]` gives me the 50,257 scores for the next token."

**Seen in:** `main.py`, top.

## Repetition penalty  `(concept)`
**1. Definition:** For tokens already generated, a positive logit is divided by the penalty and a negative one is multiplied by it, making the token less likely when penalty > 1.
**2. In plain words:** It discourages loops. GPT-2 logits are mostly negative, so multiplying makes them even more negative. Too high a penalty stops the model using needed words like "the".
**3. Example to quote:** > "With penalty 1.5 and ' to' already used, ' by' became the top token at 0.35."
```python
logits[token_id] = score / penalty if score > 0 else score * penalty
```
**Seen in:** `sampling.py`, `apply_repetition_penalty`.

## set()  `(Python built-in)`
**1. Definition:** Creates a collection of unique items from an iterable.
**2. In plain words:** `set(generated_ids)` ensures each distinct token is penalised once even if it appeared many times.
**3. Example to quote:** > "`set([1, 1, 2])` is `{1, 2}`, so repeated tokens are only penalised once."

**Seen in:** `sampling.py`, `apply_repetition_penalty`.

## tensor.clone()  `(third-party: torch)`
**1. Definition:** Makes an independent copy of a tensor.
**2. In plain words:** The functions copy logits before changing them so the caller's original scores are never modified by accident.
**3. Example to quote:** > "I `clone()` the logits first so my penalty does not mutate the caller's tensor."

**Seen in:** `sampling.py`.

## Temperature  `(concept)`
**1. Definition:** Dividing the logits by a number T before softmax; T < 1 sharpens the distribution, T > 1 flattens it.
**2. In plain words:** Low T makes the top token dominate (safe, repetitive); high T makes unlikely tokens plausible (creative, incoherent). At T = 3.0 the distribution is nearly uniform and the text is random tokens.
**3. Example to quote:** > "At temperature 0.2 ' to' had probability 1.00; at 1.5 it dropped to 0.16."
```python
def apply_temperature(logits, temperature=1.0):
    return logits / temperature
```
**Seen in:** `sampling.py`, `apply_temperature`.

## Top-k filter  `(concept)`
**1. Definition:** Keeps only the k highest logits and sets all others to `-inf`.
**2. In plain words:** It trims the long tail of unlikely tokens. Its weakness is that k is fixed and ignores how confident the model is. `k <= 0` or `k >= vocabulary size` means no filtering.
**3. Example to quote:** > "`top_k_filter(logits, 5)` left exactly five tokens alive."

**Seen in:** `sampling.py`, `top_k_filter`.

## torch.topk  `(third-party: torch)`
**1. Definition:** Returns the k largest values of a tensor and their indices.
**2. In plain words:** `torch.topk(logits, k).values[-1]` is the k-th largest score, used as the cut-off.
**3. Example to quote:** > "The k-th value is my threshold; everything below it is masked."

**Seen in:** `sampling.py`, `main.py`.

## torch.where  `(third-party: torch)`
**1. Definition:** `torch.where(cond, a, b)` selects elements from `a` where `cond` is True, else from `b`.
**2. In plain words:** An element-wise if/else. Here: where a logit is below the cut-off, use `-inf`, otherwise keep it.
**3. Example to quote:** > "`torch.where(logits < kth_value, -inf, logits)` removes everything below the top k."

**Seen in:** `sampling.py`, `top_k_filter`.

## float("-inf") filtering  `(concept)`
**1. Definition:** Setting a logit to negative infinity so that `exp(-inf) = 0` and softmax assigns that token probability exactly 0.
**2. In plain words:** It is how every filter "deletes" a token while keeping the tensor shape. The remaining probabilities renormalise to sum to 1.
**3. Example to quote:** > "Filtered tokens get `-inf`, so after softmax they cannot be drawn."

**Seen in:** `sampling.py`.

## Top-p (nucleus) filter  `(concept)`
**1. Definition:** Keeps the smallest set of top tokens whose probabilities add up to at least p and removes the rest.
**2. In plain words:** The candidate list adapts to confidence: when the model is sure, few tokens; when unsure, hundreds. `p = 0.3` kept one token because the best token alone had 59%.
**3. Example to quote:** > "With `top_p` 0.9 only 15 tokens survived; with 0.99 about 1,556 did."

**Seen in:** `sampling.py`, `top_p_filter`.

## torch.sort  `(third-party: torch)`
**1. Definition:** Sorts a tensor and returns the sorted values and their original indices.
**2. In plain words:** Top-p needs tokens ordered from most to least likely, and the indices let it map the result back to the original token order.
**3. Example to quote:** > "`torch.sort(logits, descending=True)` gives me ordered scores and their token IDs."

**Seen in:** `sampling.py`, `top_p_filter`.

## torch.cumsum  `(third-party: torch)`
**1. Definition:** Cumulative sum along a dimension: each element is the total of all elements up to it.
**2. In plain words:** It gives a running total of probability. The cut-off is the point where the running total reaches p.
**3. Example to quote:** > "`cumsum` of sorted probabilities tells me how much probability the top n tokens cover."
```python
torch.cumsum(torch.tensor([0.5, 0.3, 0.2]), dim=0)  # [0.5, 0.8, 1.0]
```
**Seen in:** `sampling.py`, `top_p_filter`.

## masked_fill  `(third-party: torch)`
**1. Definition:** `tensor.masked_fill(mask, value)` replaces elements where the boolean mask is True with `value`.
**2. In plain words:** Used to put `-inf` at the removed token positions in one call.
**3. Example to quote:** > "`logits.masked_fill(remove, float('-inf'))` removes the tokens outside the nucleus."

**Seen in:** `sampling.py`, `top_p_filter`.

## torch.softmax  `(third-party: torch)`
**1. Definition:** Converts values along a dimension into probabilities that sum to 1.
**2. In plain words:** The last step before drawing a token. It also reveals filters: tokens with `-inf` come out with probability 0.
**3. Example to quote:** > "I call `torch.softmax(logits, dim=-1)` after filtering to get a valid distribution."

**Seen in:** `sampling.py`, `main.py`.

## torch.multinomial  `(third-party: torch)`
**1. Definition:** Draws random indices from a probability tensor, with each index chosen in proportion to its probability.
**2. In plain words:** This is the "roll the weighted dice" step. `num_samples=1` returns one token ID. It errors if the probabilities contain `inf`, `nan` or negatives.
**3. Example to quote:** > "`torch.multinomial(probs, num_samples=1)` picked ' to' about 99% of the time at temperature 0.3."

**Seen in:** `sampling.py`, `sample_next_token`.

## torch.Generator and manual_seed  `(third-party: torch)`
**1. Definition:** `torch.Generator().manual_seed(n)` creates a private random-number stream with a fixed seed, passed to sampling functions as `generator=`.
**2. In plain words:** It makes random draws reproducible without affecting other randomness in the program.
**3. Example to quote:** > "I used a seeded generator so the 1,000 draws give the same counts every run."

**Seen in:** `main.py`, sampling sections.

## Entropy (nats)  `(concept)`
**1. Definition:** `-sum(p * log p)` over the distribution; a measure of randomness, 0 when one token is certain and about 10.8 when all 50,257 are equally likely.
**2. In plain words:** It gives one number for "how undecided is the model". Raising temperature raises entropy: 0.01 at T=0.2, 6.84 at T=1.5.
**3. Example to quote:** > "Entropy went from 0.01 at T=0.2 to 10.28 at T=3.0, nearly uniform."

**Seen in:** `main.py`, `show` function.

## "alive" tokens  `(concept)`
**1. Definition:** In this demo, the number of tokens whose probability after processing is greater than zero.
**2. In plain words:** Filters reduce it deliberately. At low temperature it can fall because tiny probabilities underflow to exactly 0 in float32, not because a filter removed them.
**3. Example to quote:** > "`top_k 5` leaves alive=5, while `top_p 0.99` leaves 1,556."

**Seen in:** `main.py`, `show`.

## Helper module import (import sampling as S)  `(syntax)`
**1. Definition:** Importing your own file as a module and giving it a short alias.
**2. In plain words:** `sampling.py` sits next to `main.py`; `import sampling as S` lets you write `S.top_k_filter(...)`. The file is named `sampling.py` rather than `main.py` so imports do not clash.
**3. Example to quote:** > "I wrote the sampling code once in `sampling.py` and reuse it in steps 03 and 07."

**Seen in:** `main.py`.

## sys.path.insert  `(standard library)`
**1. Definition:** Adds a folder to the front of Python's module search path.
**2. In plain words:** Makes `import sampling` work no matter which directory you run from.
**3. Example to quote:** > "I insert the script folder into `sys.path` so the helper module is found."

**Seen in:** `main.py`.

## Function with default arguments and **kw  `(syntax)`
**1. Definition:** Parameters can have defaults (`k=0`), and `**kw` collects extra keyword arguments into a dict that can be forwarded.
**2. In plain words:** `my_sample_generate(ids, temperature=0.7, top_p=0.9)` passes those settings through to `sample_next_token`.
**3. Example to quote:** > "Calling `f(**{'temperature': 0.7})` is the same as `f(temperature=0.7)`."

**Seen in:** `sampling.py`, `main.py`.

## List comprehension  `(syntax)`
**1. Definition:** `[expr for item in iterable]` builds a list in one line.
**2. In plain words:** The script uses it to draw 1,000 tokens: `[S.sample_next_token(...) for _ in range(1000)]`.
**3. Example to quote:** > "I used a list comprehension to take 1,000 samples and count them."
```python
[x * x for x in range(3)]  # [0, 1, 4]
```
**Seen in:** `main.py`.

## Dict comprehension  `(syntax)`
**1. Definition:** `{k: v for k, v in pairs}` builds a dictionary in one line.
**2. In plain words:** It turns token IDs and counts into a readable `{token: count}` dictionary.
**3. Example to quote:** > "`{tok.decode(i): c ...}` gave `{' to': 991, ' by': 9}`."

**Seen in:** `main.py`.

## torch.bincount  `(third-party: torch)`
**1. Definition:** Counts occurrences of each non-negative integer in a tensor.
**2. In plain words:** After 1,000 draws, `bincount` tells you how many times each token ID appeared, which shows that counts follow the probabilities.
**3. Example to quote:** > "`torch.bincount(torch.tensor(draws))` showed ' to' appeared 991 times of 1,000 at T=0.3."

**Seen in:** `main.py`.

## Boolean mask indexing  `(syntax)`
**1. Definition:** Using a boolean tensor as an index (`probs[probs > 0]`) to select only the elements where it is True.
**2. In plain words:** The entropy calculation uses it to skip zero probabilities, because `log(0)` is not defined.
**3. Example to quote:** > "I compute entropy over `probs[probs > 0]` to avoid log of zero."

**Seen in:** `main.py`, `show`.

## Temperature 0 problem  `(concept)`
**1. Definition:** Dividing logits by 0 gives `inf`/`nan`, which later makes `torch.multinomial` fail.
**2. In plain words:** Hugging Face rejects `temperature=0` for that reason; use `do_sample=False` for deterministic output. The error appears in multinomial, but the cause is earlier: logits / 0.
**3. Example to quote:** > "`temperature=0` surfaced as 'probability tensor contains inf or nan', but the root cause was dividing by zero."

**Seen in:** `main.py`, BREAK IT comment.

## Trade-off of randomness  `(concept)`
**1. Definition:** More randomness raises variety and risk of errors; less randomness raises reliability and repetition.
**2. In plain words:** Choose by task: low temperature for extraction and facts, higher for brainstorming.
**3. Example to quote:** > "For JSON extraction I use low temperature; for brainstorming names I raise it."

**Seen in:** `WALKTHROUGH.md`, "Say it aloud".
