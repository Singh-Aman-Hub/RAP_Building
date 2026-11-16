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

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| image vs container | blueprint vs running instance | No | You build an image; a container is a process started from it. |
| `docker build` vs `docker run` | create image vs start container | No | Errors at build time (COPY, pip) differ from run-time errors (ports, env). |
| `EXPOSE` vs `-p` | documents port vs publishes port | No | `EXPOSE` is documentation; `-p HOST:CONTAINER` actually maps the port. |
| `127.0.0.1` vs `0.0.0.0` | loopback only vs all interfaces | No | Inside a container a server on 127.0.0.1 cannot be reached from the host. |
| OOMKilled vs Exited (1) | out of memory vs crashed | No | OOMKilled is the kernel killing the process (exit code 137); Exited (1) is your program failing. |
| regex `re.search` vs `re.match` | anywhere vs start of string | No | `match` only tests the beginning. |

## Code, step by step

### 1. Header docstring  (`main.py` lines 1–11)

```python
"""
GOAL:  A tiny "error lookup" that maps common Docker error messages to their usual cause and first command to run.
       (Docker itself is taught in module 11; this step is about the DEBUGGING MINDSET: message -> cause -> check.)

RUN:   cd 01_debugging_and_environment/10_docker_errors && python main.py
       or with your own text:  python main.py "bind: address already in use"

EXPECTED OUTPUT: for each sample message, the diagnosis and the command to confirm it.

NEEDS: nothing (standard library only). Docker is NOT required to run this script.
"""
```

- **In simple words:** Label for the file: a lookup that maps common Docker error messages to a cause and a first command. Docker is not needed to run it.
- **Exactly:** Docstring; nothing runs.
- **Why it is here:** The skill taught is the mindset message -> cause -> check.

### 2. Import re and sys  (`main.py` lines 13–14)

```python
import re
import sys
```

- **In simple words:** `re` for pattern matching on error text, `sys` for command-line arguments.
- **Exactly:** `re` is the standard-library regular-expression module; `sys.argv` lists command-line words.
- **Why it is here:** Real error text varies in wording, so we match patterns instead of exact strings.

### 3. KNOWN_ERRORS table: ports, COPY, missing module  (`main.py` lines 16–26)

```python
# (regex on the error text, usual cause, first check)
KNOWN_ERRORS = [
    (r"address already in use|port is already allocated",
     "Another process/container already uses the HOST port in -p HOST:CONTAINER.",
     "lsof -i :8000   |   docker ps   (then stop the other one or change the host port)"),
    (r"No such file or directory.*requirements|COPY failed|not found in build context",
     "COPY path is wrong or outside the build context / listed in .dockerignore.",
     "ls  from the folder you pass to `docker build`; check .dockerignore"),
    (r"ModuleNotFoundError|No module named",
     "Dependency not installed in the IMAGE (works on your laptop, not in the container).",
     "docker run --rm IMAGE pip list ; check requirements.txt + the RUN pip install line"),
```

- **In simple words:** The first three rules, each a tuple of (regex, usual cause, first check).
- **Exactly:** Port rule: `address already in use|port is already allocated` (`|` means OR). COPY rule uses `.*` (any characters) and `COPY failed`. Module rule matches `ModuleNotFoundError|No module named`. Each tuple holds three `str`.
- **Why it is here:** A data table is easy to extend: a new error is one more tuple, no new logic.

### 4. Rules: connection refused, exec format, permission denied  (`main.py` lines 27–35)

```python
    (r"Connection refused|connection reset|ERR_EMPTY_RESPONSE|Empty reply",
     "App listens on 127.0.0.1 inside the container, so the host cannot reach it.",
     "Start server with --host 0.0.0.0 ; confirm with: docker logs CONTAINER"),
    (r"exec format error",
     "Image built for a different CPU architecture (e.g. arm64 Mac -> amd64 server).",
     "docker build --platform linux/amd64 ... ; docker image inspect IMAGE | grep Architecture"),
    (r"permission denied",
     "File not executable or container user lacks rights on a mounted volume.",
     "ls -l ; check USER in Dockerfile and volume ownership"),
```

- **In simple words:** Three more common failures.
- **Exactly:** `Connection refused|...|Empty reply` -> app bound to 127.0.0.1 inside the container (use `0.0.0.0`). `exec format error` -> CPU architecture mismatch (arm64 vs amd64). `permission denied` -> file mode or user rights.
- **Why it is here:** These three are among the most frequent "it works locally" container failures.

### 5. Rules: crash/OOM, pull denied, daemon down, env var  (`main.py` lines 36–48)

```python
    (r"Exited \(1\)|exit code 1|OOMKilled|Killed",
     "Process crashed at startup (Exited 1) or ran out of memory (Killed/OOMKilled, e.g. big model).",
     "docker logs CONTAINER ; docker inspect CONTAINER --format '{{.State.OOMKilled}}'"),
    (r"pull access denied|manifest.*not found|repository does not exist",
     "Image name/tag typo, or the image is private.",
     "docker images ; check spelling and tag; docker login"),
    (r"Cannot connect to the Docker daemon",
     "Docker Desktop/daemon is not running.",
     "open -a Docker  (macOS) ; docker info"),
    (r"KeyError: '?[A-Z_]+'?|environment variable .* not set",
     "Env var exists on your laptop but was not passed to the container (-e / --env-file / compose environment:).",
     "docker exec CONTAINER env | grep NAME"),
]
```

- **In simple words:** The last four rules.
- **Exactly:** `Exited \(1\)|exit code 1|OOMKilled|Killed` (parentheses are escaped with backslashes). Pull-denied matches `manifest.*not found`. `Cannot connect to the Docker daemon` -> Docker Desktop not running. The last matches `KeyError: '?[A-Z_]+'?` (optional quotes, capital letters and underscores) or an unset variable message.
- **Why it is here:** Completes the table; the closing `]` ends the list.

### 6. diagnose(): first matching rule wins  (`main.py` lines 51–55)

```python
def diagnose(message):
    for pattern, cause, check in KNOWN_ERRORS:
        if re.search(pattern, message, flags=re.IGNORECASE):
            return cause, check
    return "Unknown. Read the FIRST error in `docker logs`, not the last.", "docker logs --tail 50 CONTAINER"
```

- **In simple words:** Look through the rules in order and return the cause and check for the first one that fits.
- **Exactly:** `re.search(pattern, message, flags=re.IGNORECASE)` looks anywhere in the text, ignoring case; returns a match or `None`. Returns a `(cause, check)` tuple. If nothing matches, the fallback advice is: read the FIRST error in `docker logs`.
- **Why it is here:** Order matters: a more specific rule must come before a more general one. The fallback prevents silence.

### 7. Five sample messages  (`main.py` lines 58–64)

```python
SAMPLES = [
    "docker: Error response from daemon: Ports are not available: listen tcp 0.0.0.0:8000: bind: address already in use",
    "ModuleNotFoundError: No module named 'fastapi'",
    "curl: (52) Empty reply from server",
    "exec /usr/local/bin/python: exec format error",
    "container exited with Killed",
]
```

- **In simple words:** Example errors that run when you give no arguments.
- **Exactly:** `SAMPLES` is a list of 5 strings: port in use, missing `fastapi`, `Empty reply from server`, `exec format error`, `container exited with Killed`.
- **Why it is here:** Gives a demo that needs no Docker.

### 8. Main guard: diagnose each message and print  (`main.py` lines 66–70)

```python
if __name__ == "__main__":
    messages = sys.argv[1:] or SAMPLES      # argv[0] is the script name; the rest are user arguments
    for m in messages:
        cause, check = diagnose(m)
        print(f"MESSAGE : {m}\nCAUSE   : {cause}\nCHECK   : {check}\n")
```

- **In simple words:** Use your own messages from the command line, or the samples.
- **Exactly:** `sys.argv[1:] or SAMPLES` takes extra arguments if given, else the samples (an empty list is falsy). For each message prints `MESSAGE`, `CAUSE`, `CHECK`. Real output example: for `ModuleNotFoundError: No module named 'fastapi'` -> cause "Dependency not installed in the IMAGE".
- **Why it is here:** Lets you paste a real error from your terminal and get a first lead.

### 9. TRY IT and BREAK IT comments  (`main.py` lines 72–73)

```python
# TRY IT: add a new (regex, cause, check) tuple for "no space left on device".
# BREAK IT: write a regex with an unbalanced "(" and read the re.error traceback.
```

- **In simple words:** Experiments for you (comments only).
- **Exactly:** An unbalanced `(` in a regex raises `re.error` (for example `missing ), unterminated subpattern`).
- **Why it is here:** Practice reading a different kind of traceback.

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
