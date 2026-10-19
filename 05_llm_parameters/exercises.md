# Exercises - 05 LLM Parameters

Same rules as module 04: change one thing, predict first, break it, rebuild with the file closed, explain aloud. Worked answers in [solutions/](solutions/).

## Step 01 - Config
- **Modify:** let command-line flags override `config.json` (`--temperature 0.2 --top_k 10`) using `argparse`. -> `solutions/ex01_cli_override.py`
- **Extend:** add `"seed"` and `"stop_strings"` to the config and to validation; make `generation_kwargs` drop non-HF keys.
- **Break it:** delete a comma in the JSON; rename a section; read both errors and write friendlier messages.
- **Rebuild:** a loader with 3 validations.
- **Explain aloud:** each parameter in one sentence.

## Step 02 - Sampling
- **Modify:** implement **min-p** filtering (keep tokens with p >= min_p x top probability) and compare how many tokens survive for a confident vs an uncertain prompt. -> `solutions/ex02_min_p_filter.py`
- **Predict then check:** with `top_p=0.0001` how many tokens survive? With `temperature=0.01`?
- **Break it:** `temperature=0`; read the error, find where the `inf/nan` starts.
- **Rebuild:** temperature + top-k + top-p in <25 lines.
- **Explain aloud:** the difference between top-k and top-p with a concrete example.

## Step 03 - HF generate
- **Modify:** measure stability: how many distinct outputs do 6 seeds produce at T = 0.1, 0.5, 1.0, 1.5? -> `solutions/ex03_seed_stability.py`
- **Experiment:** try `num_beams=4` and describe what changes vs greedy.
- **Break it:** `temperature=0.0` with sampling on; `top_p=1.5`.
- **Rebuild:** a `run(label, **overrides)` helper.
- **Explain aloud:** why fix the seed when comparing settings.

## Step 04 - Tokens and context
- **Modify:** write `trim_history(messages, budget)` that always keeps the system message and drops the oldest turns. -> `solutions/ex04_chat_trimmer.py`
- **Calculate:** KV cache of a 70B-like model (80 layers, 8192 hidden) at 8k tokens, batch 8, fp16.
- **Break it:** run `generate` on a 1030-token prompt without trimming; locate the root cause from the traceback.
- **Rebuild:** `fit_to_window`.
- **Explain aloud:** two strategies besides truncation for long conversations.

## Step 05 - Quantization
- **Modify:** plot/print int4 error against group size (16 ... whole tensor) on a matrix with outliers, including the scale overhead. -> `solutions/ex05_group_size_curve.py`
- **Experiment:** quantize only attention weights, then only MLP weights of distilgpt2; which hurts loss more?
- **Break it:** remove `np.clip`; print `q.min()/q.max()` to find the wrap-around.
- **Rebuild:** `quantize_symmetric` / `dequantize`.
- **Explain aloud:** why a single outlier ruins per-tensor scaling.

## Step 06 - Training parameters
- **Modify:** a "learning-rate range test": train briefly at many learning rates and choose one. -> `solutions/ex06_lr_range_test.py`
- **Experiment:** repeat the epoch curve with `hidden=8` vs `hidden=256`; with 40 vs 300 training points.
- **Break it:** drop `zero_grad()`; batch size 0.
- **Rebuild:** the batch loop (shuffle, slice, forward, loss, backward, step, zero_grad).
- **Explain aloud:** recognise from loss numbers: LR too high, too low, overfitting.

## Step 07 - Experiment logger
- **Modify:** add rows for `top_p 0.5`, `repetition_penalty 1.3`, `temperature 0.9`; add a second prompt and average. Fill `my_notes` for each row after reading the samples.
- **Experiment:** add a column "words_per_second" and "cost_per_1k_tokens" (assume a price).
- **Break it:** `REPEATS = 0`; add an assert near the constant.
- **Rebuild:** time-and-log loop writing a CSV.
- **Explain aloud:** the three most important findings in your table and one limitation of the methodology.

## Module capstone (45 min)
Produce `my_parameter_report.md` containing: your experiment table (>= 8 rows), three conclusions ("lowering temperature to 0.3 reduced variation but didn't change latency"), one recommended config for (a) JSON extraction, (b) chat, (c) brainstorming with a one-line justification each, and one thing you could not explain (add it to your vocabulary file).
