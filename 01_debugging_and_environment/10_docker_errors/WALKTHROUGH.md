# Walkthrough 10 — Debugging Docker problems (text-first)

## Goal
Apply the same method to containers: **read logs -> compare with the laptop -> change one thing**. Docker syntax is taught in [module 11](../../11_docker/); read it again after that module.

## Run it
```bash
cd 01_debugging_and_environment/10_docker_errors
python main.py                              # offline error lookup
python main.py "bind: address already in use"
```
No Docker needed for this step.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `sys.argv[1:]` | The command-line arguments after the script name |
| `re.search(pattern, text, flags=re.IGNORECASE)` | Find a regex anywhere in text |
| `messages = sys.argv[1:] or SAMPLES` | Use samples if no arguments were given (`or` returns the first truthy value) |

## Libraries & functions
`re` (regex, standard lib) and `sys` only.

## Execution trace (real output)
```
MESSAGE : ... bind: address already in use
CAUSE   : Another process/container already uses the HOST port in -p HOST:CONTAINER.
CHECK   : lsof -i :8000   |   docker ps   (then stop the other one or change the host port)

MESSAGE : ModuleNotFoundError: No module named 'fastapi'
CAUSE   : Dependency not installed in the IMAGE ...
CHECK   : docker run --rm IMAGE pip list ; check requirements.txt + the RUN pip install line
```

## The six commands that solve most Docker issues
| Command | Use |
|---|---|
| `docker ps -a` | Is it running? Exit code? (`Exited (1)` = crashed at startup) |
| `docker logs --tail 50 NAME` | The Python traceback — same skills as steps 01-02 |
| `docker exec -it NAME sh` | Look inside: `ls`, `env`, `pwd`, `pip list` |
| `docker inspect NAME` | Ports, env vars, mounts, OOMKilled |
| `docker build --progress=plain .` | See which Dockerfile step failed and its full output |
| `docker run --rm -p 8000:8000 IMAGE` | Reproduce quickly; `--rm` removes it afterwards |

## Map each earlier skill to Docker
| Earlier step | Docker form |
|---|---|
| 03 import errors | package missing from `requirements.txt` / wrong base image / different Python version |
| 04 paths | `WORKDIR`, `COPY` destinations, case-sensitive Linux paths |
| 06 network | server bound to `127.0.0.1` instead of `0.0.0.0`; wrong `-p` mapping; container-to-container uses the **service name**, not `localhost` |
| 07 env vars | missing `-e` / `--env-file` / compose `environment:` |
| 08 logging | log to stdout/stderr so `docker logs` can see it |
| 09 RCA | change **one** Dockerfile line at a time and rebuild |

## Experiments
1. Add a regex for "no space left on device" and test it.
2. Pass three messages in one command.
3. After module 11, deliberately break your Dockerfile in each way above and check this table predicts the symptom.

## Debugging practice
Write a journal entry for one of the sample errors: symptom, hypothesis, command to test, expected result if hypothesis is true.

## Rebuild checklist
- [ ] Name the first three commands you run for a crashing container
- [ ] Explain why `localhost` inside a container is not your laptop
- [ ] Explain why the process must listen on `0.0.0.0`

## Say it aloud
"When a container fails I check `docker ps -a` for the exit code, read `docker logs` — which is just a normal traceback — then compare what exists inside the container (files, env vars, packages) with my laptop. Most issues are ports, missing files/dependencies, env vars, or the server binding to localhost."
Trade-off: containers make environments reproducible but add a layer where files, networking and permissions differ from your local machine.
