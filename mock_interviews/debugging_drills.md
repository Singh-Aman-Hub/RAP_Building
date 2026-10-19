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
