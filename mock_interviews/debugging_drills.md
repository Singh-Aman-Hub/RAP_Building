# Debugging Drills (no AI)

For each drill: run it, read the traceback bottom-up, state the root cause in one sentence, fix, verify, then write it up with [../01_debugging_and_environment/](../01_debugging_and_environment/) journal template. Don't read the module's hints first.

| # | File to run | Expected symptom | Topic |
|---|-------------|------------------|-------|
| 1 | [01_reading_tracebacks/broken_traceback.py](../01_debugging_and_environment/01_reading_tracebacks/broken_traceback.py) | exception deep in a call chain | tracebacks |
| 2 | [02_syntax_and_runtime_errors/broken_syntax.py](../01_debugging_and_environment/02_syntax_and_runtime_errors/broken_syntax.py) | SyntaxError | syntax |
| 3 | [02_syntax_and_runtime_errors/broken_runtime.py](../01_debugging_and_environment/02_syntax_and_runtime_errors/broken_runtime.py) | runtime exception | runtime |
| 4 | [03_import_and_environment/broken_import.py](../01_debugging_and_environment/03_import_and_environment/broken_import.py) | import error | env/imports |
| 5 | [04_paths_and_cwd/broken_paths.py](../01_debugging_and_environment/04_paths_and_cwd/broken_paths.py) | file not found | paths |
| 6 | [05_types_and_validation/broken_types.py](../01_debugging_and_environment/05_types_and_validation/broken_types.py) | TypeError/ValueError | validation |
| 7 | [06_http_and_network/broken_http.py](../01_debugging_and_environment/06_http_and_network/broken_http.py) | HTTP/network failure | HTTP |
| 8 | [07_env_vars_and_config/broken_env.py](../01_debugging_and_environment/07_env_vars_and_config/broken_env.py) | missing config | env vars |
| 9-13 | Docker: [11_docker/05_healthchecks_logs_debugging/](../11_docker/05_healthchecks_logs_debugging/) `broken_*` folders | wrong port, bad CMD path, exits immediately, localhost bind, missing dependency | Docker |

Have a friend (or a script) corrupt a working example from modules 08/09 (rename an argument, break JSON, wrong port) for fresh drills.

---

## Drills for modules 15 to 21 (inject the bug, then find it cold)

These come from the `BREAK IT` and `Debugging practice` sections of the walkthroughs. Work on a copy: make the edit in the "Inject" column, run the step's `main.py` (`HF_HUB_OFFLINE=1 .venv/bin/python <step>/main.py`), then write symptom, root cause (one sentence), fix, and how you would catch it earlier. Do not read the step's debugging section first; compare afterwards.

| # | Step | Inject | Expected symptom | Topic |
|---|------|--------|------------------|-------|
| 14 | [15/02 MLP backprop](../15_neural_networks/02_mlp_backprop_from_scratch/WALKTHROUGH.md) | delete `dlogits /= N` | gradient check: same relative error (0.6 for N=4) on every parameter | loss scale, gradient check |
| 15 | [15/05 normalisation](../15_neural_networks/05_initialization_and_normalization/WALKTHROUGH.md) | run BatchNorm in `train()` with a batch of one | `ValueError: Expected more than 1 value per channel` | train/eval mode |
| 16 | [17/02 SFT](../17_finetuning_alignment_and_benchmarks/02_sft_from_scratch_with_loss_masking/WALKTHROUGH.md) | drop the one-token shift in `sft_loss` | no error, loss falls, model does not learn the task | off-by-one in labels |
| 17 | [17/05 DPO](../17_finetuning_alignment_and_benchmarks/05_dpo_from_scratch/WALKTHROUGH.md) | `LR=5e-4` | accuracy 0.87 but chosen log-prob falls from -34 to -82 | over-optimisation |
| 18 | [18/05 CRNN and CTC](../18_ocr_and_document_parsing/05_crnn_and_ctc/WALKTHROUGH.md) | skip the `permute` to `(T, B, C)` | `RuntimeError: input_lengths must be of size batch_size` | tensor layout |
| 19 | [19/01 tokens](../19_chunking_deep_dive/01_tokens_not_characters/WALKTHROUGH.md) | `fixed_token_windows(doc, size=128, overlap=128)` | `AssertionError: overlap must be smaller than size` | parameter validation |
| 20 | [19/02 recursive splitter](../19_chunking_deep_dive/02_recursive_splitter_rebuilt/WALKTHROUGH.md) | `keep_separator="end"` on Markdown | each chunk ends with the heading of the next section | separator side |
| 21 | [20/06 resilience](../20_llm_system_lld/06_resilience/WALKTHROUGH.md) | retry policy `retry_on=(Exception,)` with a permanent 401 | four attempts on an invalid key, slow failure | error classification |
| 22 | [21/01 checkpoints](../21_llm_inference_engineering/01_checkpoint_formats_and_loading/WALKTHROUGH.md) | `"<Q"` to `">Q"`, or `offset=s` instead of `data_start + s` | absurd header length and `UnicodeDecodeError`; or wrong values with no error | binary format |
| 23 | [21/08 HF inference](../21_llm_inference_engineering/08_hf_inference_batching_kvcache/WALKTHROUGH.md) | `tokenizer.padding_side = "right"` before the batching section | short prompt continues as `'Hello, my name is\n\n\n\n'`, one-line warning | padding side |
| 24 | [21/12 prefix caching](../21_llm_inference_engineering/12_continuous_batching_prefix_caching/WALKTHROUGH.md) | drop the hash chain in `BlockCache._hashes`; or drop `copy.deepcopy(prefix_cache)` in part D | hit rate looks better but wrong KV would be shared; or wrong logits from a mutated cache | cache correctness |

Self-test habit for all of them: reproduce, shrink, state the invariant that was violated (for example "batch result equals batch-1 result", "gradient equals numeric gradient", "cached output equals uncached output"), fix, add the assertion that would have caught it.
