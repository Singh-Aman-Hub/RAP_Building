# Step 05 - Logs, health checks and debugging broken containers

## The story so far
[Step 04](../04_compose_multi_container/WALKTHROUGH.md) starts a healthy stack when everything is right. Real life delivers the other case.

## The pain point
`curl` hangs, the container shows `Exited (1)`, or it is running but nothing answers. Guessing and rebuilding blindly wastes hours. The cause is almost always one of five small things.

## The spark
Debug a container like any program, in a fixed order: is it running (`ps -a`), what did it say (`logs`), is it listening where and how I think (`docker port`, bind address), is the configuration what the app expects (`exec`, `inspect`)? Health checks separate *alive* from *ready*.

## Goal
Diagnose five deliberately broken setups using only Docker's own tools, no AI.

## Run it
```bash
cd 11_docker/05_healthchecks_logs_debugging
cat DIAGNOSIS.md               # drills; try each BEFORE opening the <details> answer
./run_all_drills.sh            # optional: see all symptoms in one go (needs Docker Desktop)
```
The broken images are expected to misbehave; nothing to run locally.

## Syntax explained
| Syntax | Meaning | Example |
|---|---|---|
| `docker ps -a` | include stopped containers | STATUS `Exited (1) 3 s ago` |
| `--format '{{.State.ExitCode}}'` | Go template to pick fields | `docker inspect c --format '{{.State.ExitCode}}'` |
| `--entrypoint sh` | override the entrypoint to get a shell | `docker run --rm -it --entrypoint sh img` |
| Exit codes | 0 finished, 1 app error, 125-127 docker/command problems, 137 SIGKILL/OOM, 139 segfault | |
| `HEALTHCHECK` | Dockerfile instruction running a probe command | see debug_cheatsheet.md |

## Execution trace (diagnosis loop)
Input: "curl doesn't work". Processing: `ps -a` (alive?) -> `logs` (what happened?) -> compare listening port/host with `-p` mapping (`docker port`) -> `exec` and look inside. Output: root cause sentence + fix + verification.

## Experiments
1. Add the HEALTHCHECK from the cheat-sheet to `broken_wrong_port` - status shows `unhealthy` after the retries although the container is "Up" (health measures the app, not the process).
2. Make `/health` return 500 deliberately and watch `docker ps` status change.
3. Fix each drill and write the one-line verification (`curl` output / `ps` status).

## Debugging practice
This whole step is the debugging practice. Format to write down for each drill: symptom - command that revealed it - root cause - fix - verification.

## Rebuild checklist
- [ ] List the 6 debug commands from memory with what each tells you.
- [ ] Explain exit code 0 vs 1 vs 137.

## The new problem (cliffhanger)
Now we can diagnose problems, but our image is still large, carries build tools, and runs as root, so a bug in the app is a bug with root rights. Next: [step 06, multi-stage builds and security](../06_multistage_and_security/WALKTHROUGH.md).

## Say it aloud
> **Say it aloud (interview box):** I debug containers like any program: is it running, what do the logs say, is it listening where I think, is the port mapped how I think, and is the config what the app expects. Typical culprits are binding to 127.0.0.1, a wrong port mapping, missing dependencies in the image and a CMD that exits. The trade-off is that a health check tells you the app is unwell, but plain Docker will not fix it, which needs an orchestrator.
