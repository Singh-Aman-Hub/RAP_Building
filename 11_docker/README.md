# 11 - Docker and Docker Compose

## Why this matters
"It works on my machine" is the most common deployment failure. Docker packages your code, Python version and dependencies into one image that behaves the same everywhere. Interviewers ask you to explain a Dockerfile line by line, map ports, pass env vars, and debug a container that will not start.

## Where this fits
**Builds on**
- [../09_api_development/](../09_api_development/): the FastAPI/Flask app you containerise; `--host 0.0.0.0`, uvicorn/gunicorn commands and `/health` endpoints come from there.
- [../01_debugging_and_environment/](../01_debugging_and_environment/): virtual environments and `requirements.txt` become `RUN pip install -r requirements.txt`; reading tracebacks becomes reading `docker logs`.
- [../00_python_foundations/](../00_python_foundations/): `os.getenv`, `pathlib` and exceptions are used for configuration and health logic.

**Leads to**
- [../12_streamlit_gradio/](../12_streamlit_gradio/): the UI becomes its own container that calls the API by service name (step 04).
- [../10_ml_system_architecture/](../10_ml_system_architecture/): containers are the deployment unit for model serving, versioned image tags = model versioning, health checks = monitoring.
- [../13_capstone_project/](../13_capstone_project/): final project ships as a Compose stack.

**Swapping in the module-09 app.** Steps use a tiny self-contained app so this module runs on its own. To containerise your real module-09 API: copy its folder next to a Dockerfile, make sure it has a `requirements.txt`, change `CMD ["uvicorn", "<module>:<app_variable>", "--host", "0.0.0.0", "--port", "8000"]`, and keep `/health`. Loading a model file? Add it with `COPY` (small) or a volume (large).

## Concept map
```
 Dockerfile --build--> IMAGE --run--> CONTAINER  <--env vars (-e) / ports (-p) / volumes (-v)
    |  (layers, cache order,            |  (one main process; logs to stdout; health check)
    |   base image, multi-stage)        |
    |                                   +--- network: service names as DNS ---+
    v                                                                         |
 .dockerignore / non-root / pinned / no secrets   Compose YAML: api + ui + redis (depends_on health)
 (optimisation + security)                         |
                                                   +--> debugging: ps -a, logs, exec, inspect, exit codes
```

## Study path
Check `docker info` first: Docker Desktop must be running for build/run commands. (`docker compose config` and the `check_locally.py` scripts work without it.)

| Step | Folder | What you learn | Run | Time |
|---|---|---|---|---|
| 1 | [01_first_image](01_first_image/) | image vs container, basic Dockerfile, build/run/ports | `./commands.sh` | 60 min |
| 2 | [02_layers_and_cache](02_layers_and_cache/) | layers, cache order, slim vs alpine | `./commands.sh` | 40 min |
| 3 | [03_ports_env_volumes](03_ports_env_volumes/) | `-p`, `-e`, `--env-file`, volumes vs bind mounts | `./commands.sh` | 60 min |
| 4 | [04_compose_multi_container](04_compose_multi_container/) | Compose YAML, networking, health-ordered start-up | `docker compose up --build` | 75 min |
| 5 | [05_healthchecks_logs_debugging](05_healthchecks_logs_debugging/) | logs/exec/inspect, 5 broken drills | `DIAGNOSIS.md` | 75 min |
| 6 | [06_multistage_and_security](06_multistage_and_security/) | multi-stage, non-root, hardening | `./commands.sh` | 45 min |

Every step has `WALKTHROUGH.md`; exercises in [exercises.md](exercises.md), answers in [solutions/](solutions/).

**Verification note.** The Docker daemon was not running when this module was written, so image builds and runs were not executed. What WAS verified: the app code runs (`check_locally.py`, TestClient), `docker compose config` validates the Compose file, and all Dockerfiles were checked for syntax rules by hand. Output blocks in walkthroughs that come from Docker are labelled "typical output". Run each `commands.sh` and treat any mismatch as a debugging exercise.

## Syllabus bullets explained

**Containers vs virtual machines.** A VM virtualises hardware and runs a whole guest OS (GBs, minutes to boot). A container is a normal process isolated by the host kernel (namespaces for what it sees, cgroups for what it can use), so it starts in seconds and is MBs. Analogy: VM = a separate house; container = a locked apartment in one building. Trade-off: containers share the host kernel, so isolation is weaker than a VM.

**Images vs containers.** Image = read-only template (like a class or an installer); container = running instance (like an object). One image can run many containers; deleting a container does not delete the image. Container writable layer disappears with the container.

**Dockerfile instructions.** `FROM` base, `WORKDIR` directory, `COPY` files in, `RUN` build-time command, `ENV` default env var, `EXPOSE` documentation of the port, `USER` who runs it, `HEALTHCHECK` probe, `CMD` default run command, `ENTRYPOINT` fixed program (CMD supplies default args). Prefer exec-form JSON lists `["a","b"]` over shell-form. Walked through in steps 01 and 06.

**Base images.** `python:3.12` (full, ~1 GB, has compilers), `python:3.12-slim` (~120 MB, best default), `python:3.12-alpine` (tiny, musl libc, many ML wheels fail). Pin the tag. For GPUs you would start from an NVIDIA CUDA image (reference only, no GPU needed here).

**Dependencies and requirements files.** Pin versions (`fastapi==0.146.0`), install with `pip install --no-cache-dir -r requirements.txt`, and copy `requirements.txt` before code (step 02).

**Build and run commands.** `docker build -t name:tag .`, `docker run -d --name n -p H:C -e K=V -v vol:/path image`, `docker ps`, `docker stop/rm`, `docker rmi`.

**Port mapping.** `-p HOST:CONTAINER`. Container-internal port must match what the app binds to, on `0.0.0.0`. `EXPOSE` does not publish.

**Environment variables.** Configuration without rebuilds: `ENV` default, `-e` or `--env-file` override, Compose `environment:`. Not for real secrets in image layers.

**Volumes and bind mounts.** Named volume: Docker-managed persistence. Bind mount: host folder shared into the container. Containers' own filesystem is ephemeral.

**Container networking.** Default bridge network isolates containers from the host; published ports open a door. On a Compose (user-defined) network, containers resolve each other by service name. Inside a container, `localhost` = itself.

**Compose YAML structure.** `services:` (each: `image` or `build`, `ports`, `environment`, `volumes`, `depends_on`, `healthcheck`), top-level `volumes:` and `networks:`. One `docker compose up` replaces several long `docker run` commands.

**Multi-container applications.** One process per container: api, ui, redis. Each scales and restarts independently (step 04).

**Logs and debugging.** Apps log to stdout/stderr; `docker logs`. Then `exec`, `inspect`, exit codes (step 05).

**Health checks.** Distinguish *alive* (`/health`) from *ready* (`/ready`, dependencies reachable). Dockerfile `HEALTHCHECK` or Compose `healthcheck`; Compose can wait for `service_healthy`.

**Multi-stage builds.** Build in one stage, copy artefacts to a slim final stage. Smaller, fewer CVEs.

**Image optimisation and security basics.** See [06 SECURITY_CHECKLIST](06_multistage_and_security/SECURITY_CHECKLIST.md): slim base, `.dockerignore`, no pip cache, non-root user, pinned versions, secrets at run time only.

## Common mistakes and gotchas
- Binding to `127.0.0.1` inside a container (unreachable from outside).
- Mapping `-p 8000:8000` while the app listens on another port.
- Inline `#` comments after Dockerfile instructions (passed as arguments).
- Copying code before installing requirements (cache busting).
- Using `localhost` to reach another container.
- Storing state in the container filesystem; baking `.env` into an image.
- `latest` tags; forgetting `.dockerignore` (huge build contexts, leaked `.env`).
- Running `docker compose down -v` and wondering where the data went.

## Self-check
- [ ] Containers vs VMs, images vs containers
- [ ] Wrote a Dockerfile from memory, explained every line
- [ ] Layer cache ordering; slim vs alpine
- [ ] Build/run, port mapping, env vars, volumes vs bind mounts
- [ ] Networking and Compose multi-container app
- [ ] Logs and health checks; diagnosed all 5 drills without help
- [ ] Multi-stage build, non-root user, security checklist
- [ ] Containerised the module-09 API

## Interview questions
1. **What is the difference between an image and a container?** Image = immutable layered template; container = running (or stopped) instance with a writable layer. Trade-off: containers are disposable, so persist data elsewhere.
2. **Container vs VM?** Containers share the host kernel and isolate via namespaces/cgroups: lightweight, fast. VMs include a guest OS: stronger isolation, heavier.
3. **CMD vs ENTRYPOINT?** ENTRYPOINT is the executable, CMD default arguments; `docker run img args` replaces CMD only.
4. **Why copy requirements.txt before the code?** Layer caching: dependencies change rarely, so pip install stays cached when only code changes.
5. **Why does my app work in the container but I cannot reach it?** Likely bound to 127.0.0.1 or wrong `-p` mapping. Check logs and `docker port`.
6. **Does EXPOSE publish a port?** No; only documents. `-p` publishes.
7. **Volume vs bind mount?** Volume: Docker-managed, persistent, portable (databases). Bind mount: host path, great for development, host-dependent.
8. **How do containers in Compose talk?** Over a shared network using service names as hostnames and container ports.
9. **depends_on guarantees readiness?** Not by default (only start order); use `condition: service_healthy` with a healthcheck.
10. **How do you keep an image small and secure?** Slim base, multi-stage, `--no-cache-dir`, `.dockerignore`, non-root, pinned versions, no secrets baked in.
11. **A container exits immediately with code 0, why?** Its main process finished; the CMD must run a foreground long-lived server.
12. **How do you pass secrets?** At run time (env vars, secret managers, orchestrator secrets), never COPY or ENV them in the Dockerfile.
13. **Why is alpine a poor default for ML?** musl libc; many wheels (numpy, torch) are not available, forcing slow source builds.
14. **How would you containerise a model-serving API?** Slim base + pinned deps, COPY app (+ small model) or mount model volume, run as non-root on 0.0.0.0, health endpoint, tag image with model version.
