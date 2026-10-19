# Terms — Multi-stage builds, small images and security

Folder: `11_docker/06_multistage_and_security` · Files covered: `Dockerfile`, `Dockerfile.single`, `SECURITY_CHECKLIST.md`, `commands.sh`, `app/main.py`, `requirements.txt`, `.dockerignore`, `WALKTHROUGH.md`
Back to module: [11_docker README](../README.md)

**Terms in this folder:** [Multi-stage build (concept)](#multi-stage-build-concept) · [FROM ... AS builder (Dockerfile instruction)](#from--as-builder-dockerfile-instruction) · [COPY --from (Dockerfile instruction flag)](#copy---from-dockerfile-instruction-flag) · [pip install --prefix (command flag)](#pip-install---prefix-command-flag) · [Non-root user (useradd, USER) (security concept)](#non-root-user-useradd-user-security-concept) · [useradd --create-home --uid (command)](#useradd---create-home---uid-command) · [COPY --chown (Dockerfile instruction flag)](#copy---chown-dockerfile-instruction-flag) · [PermissionError after adding USER (error pattern)](#permissionerror-after-adding-user-error-pattern) · [ENV PYTHONDONTWRITEBYTECODE and PYTHONUNBUFFERED (environment variables)](#env-pythondontwritebytecode-and-pythonunbuffered-environment-variables) · [HEALTHCHECK (Dockerfile instruction)](#healthcheck-dockerfile-instruction) · [ENTRYPOINT vs CMD (Dockerfile instructions)](#entrypoint-vs-cmd-dockerfile-instructions) · [Shell form vs exec form (syntax)](#shell-form-vs-exec-form-syntax) · [python:3.12 vs python:3.12-slim (base images)](#python312-vs-python312-slim-base-images) · [Image size and attack surface (concept)](#image-size-and-attack-surface-concept) · [Pinning versions (concept)](#pinning-versions-concept) · [Secrets in images (security concept)](#secrets-in-images-security-concept) · [One process per container (concept)](#one-process-per-container-concept) · [COPY vs ADD (Dockerfile instructions)](#copy-vs-add-dockerfile-instructions) · [Volume vs bind mount vs image layer (concept)](#volume-vs-bind-mount-vs-image-layer-concept) · [Containers share the host kernel (concept)](#containers-share-the-host-kernel-concept) · [Image scanning (docker scout, trivy) (tool)](#image-scanning-docker-scout-trivy-tool) · [.dockerignore (file)](#dockerignore-file) · [pip install --no-cache-dir (command flag)](#pip-install---no-cache-dir-command-flag) · [docker exec whoami (command)](#docker-exec-whoami-command) · [docker ps --format (command flag)](#docker-ps---format-command-flag) · [FastAPI toy app and requirements (concept)](#fastapi-toy-app-and-requirements-concept) · [docker images (command)](#docker-images-command)

---

## Multi-stage build `(concept)`
**1. Definition:** A Dockerfile with several `FROM` stages in which later stages can copy artifacts from earlier ones, so only the final stage becomes the shipped image.
**2. In plain words:** Do the messy work (installing, compiling) in a "builder" stage, then start a clean image and copy in only the results. Compilers and caches never reach the final image, making it smaller and safer. The gain is largest when the build needs heavy tools like gcc.
**3. Example to quote:** > "I install dependencies in a builder stage and copy only `/install` into a clean slim runtime stage, so no build tools ship."
**Seen in:** `Dockerfile`

## FROM ... AS builder `(Dockerfile instruction)`
**1. Definition:** Starts a new build stage and gives it a name that later instructions can refer to.
**2. In plain words:** Each `FROM` begins a fresh image. Naming the first stage `builder` lets the second stage say `--from=builder`. Everything in the builder not explicitly copied out is thrown away.
**3. Example to quote:** > "`FROM python:3.12-slim AS builder` names stage one so the runtime stage can copy from it."
**Seen in:** `Dockerfile`

## COPY --from `(Dockerfile instruction flag)`
**1. Definition:** Copies files from a previous build stage, or another image, instead of from the build context.
**2. In plain words:** This is the bridge between stages. `COPY --from=builder /install /usr/local` moves only the installed Python packages into the final image.
**3. Example to quote:** > "`COPY --from=builder /install /usr/local` brings across the packages and nothing else from the builder."
**Seen in:** `Dockerfile`

## pip install --prefix `(command flag)`
**1. Definition:** Installs packages into a custom directory prefix such as `/install` rather than the system location.
**2. In plain words:** Putting everything in one folder makes it trivial to copy wholesale into another stage. Copied onto `/usr/local`, Python finds the packages normally.
**3. Example to quote:** > "I install with `--prefix=/install`, so the whole dependency set is one folder I can copy into the runtime image."
**Seen in:** `Dockerfile`

## Non-root user (useradd, USER) `(security concept)`
**1. Definition:** Creating an unprivileged account with `useradd` and switching to it with `USER` so the container's process doesn't run as root.
**2. In plain words:** If the app is compromised, an attacker who is "appuser" can do far less than one who is root inside the container. It is a cheap, standard hardening step. The catch is that the user may not be able to write to root-owned folders, so give it ownership of the paths it needs.
**3. Example to quote:** > "`docker exec sec whoami` printed `appuser`; without the `USER` line it printed `root`."
```dockerfile
RUN useradd --create-home --uid 10001 appuser
USER appuser
```
**Seen in:** `Dockerfile`

## useradd --create-home --uid `(command)`
**1. Definition:** A Linux command that creates a user account; `--create-home` makes a home directory and `--uid 10001` fixes the numeric user ID.
**2. In plain words:** A high, fixed UID avoids clashing with system users and makes file ownership predictable across environments.
**3. Example to quote:** > "I create `appuser` with UID 10001 so ownership on mounted volumes is consistent."
**Seen in:** `Dockerfile`

## COPY --chown `(Dockerfile instruction flag)`
**1. Definition:** Sets the owner and group of copied files at copy time.
**2. In plain words:** Files copied in as root would otherwise be unwritable by the unprivileged user. `--chown=appuser:appuser` gives the app ownership of its own code without an extra `chown` layer.
**3. Example to quote:** > "I use `COPY --chown=appuser:appuser app ./app` so the non-root user owns its code."
**Seen in:** `Dockerfile`

## PermissionError after adding USER `(error pattern)`
**1. Definition:** A `PermissionError: [Errno 13]` raised when the non-root user tries to write to a path owned by root, such as `/data`.
**2. In plain words:** It's the usual side effect of hardening. The fix is to create the directory and `chown` it to the app user in the Dockerfile before the `USER` line.
**3. Example to quote:** > "After adding `USER appuser`, writing to `/data` failed; I fixed it with `RUN mkdir /data && chown appuser /data` before `USER`."
**Seen in:** `WALKTHROUGH.md`, debugging practice

## ENV PYTHONDONTWRITEBYTECODE and PYTHONUNBUFFERED `(environment variables)`
**1. Definition:** `PYTHONDONTWRITEBYTECODE=1` stops Python writing `.pyc` files; `PYTHONUNBUFFERED=1` makes stdout and stderr unbuffered so output appears immediately.
**2. In plain words:** No `.pyc` files keeps the image tidy. Unbuffered output means `docker logs` shows each line live instead of after a delay, which matters when debugging a crash.
**3. Example to quote:** > "I set `PYTHONUNBUFFERED=1` so my print and log lines show up in `docker logs` right away."
**Seen in:** `Dockerfile`

## HEALTHCHECK `(Dockerfile instruction)`
**1. Definition:** A probe command that Docker runs periodically; exit 0 is healthy and exit 1 unhealthy.
**2. In plain words:** It reports whether the app really works. The probe uses Python's `urllib` because `curl` isn't in `python:slim`, and `--start-period` stops early failures from counting during boot.
**3. Example to quote:** > "About twelve seconds after start, `docker ps` showed `(healthy)` for the multi-stage container."
**Seen in:** `Dockerfile`, `commands.sh`

## ENTRYPOINT vs CMD `(Dockerfile instructions)`
**1. Definition:** `ENTRYPOINT` sets the fixed executable of the container; `CMD` supplies default arguments that `docker run image ARGS` replaces.
**2. In plain words:** With `ENTRYPOINT ["uvicorn"]` and `CMD ["app.main:app", ...]`, the program is always uvicorn but you can change its arguments at run time. Passing arguments after the image name replaces CMD, never ENTRYPOINT.
**3. Example to quote:** > "`docker run --rm sec-demo:multi --version` ran `uvicorn --version`, because only CMD was replaced."
```dockerfile
ENTRYPOINT ["uvicorn"]
CMD ["app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
**Seen in:** `Dockerfile`, `commands.sh`

## Shell form vs exec form `(syntax)`
**1. Definition:** Shell form (`CMD uvicorn ...`) runs through `/bin/sh -c`; exec form (`CMD ["uvicorn", ...]`) runs the program directly as PID 1.
**2. In plain words:** In shell form the shell is PID 1 and may not pass the stop signal to your app, causing slow or unclean shutdowns. Prefer the JSON list form.
**3. Example to quote:** > "I always use exec form so uvicorn receives SIGTERM from `docker stop` and shuts down cleanly."
**Seen in:** `SECURITY_CHECKLIST.md`

## python:3.12 vs python:3.12-slim `(base images)`
**1. Definition:** `python:3.12` is the full Debian-based image with compilers and many tools (around 1 GB); `-slim` omits most of them.
**2. In plain words:** `Dockerfile.single` uses the big image and runs as root as a baseline to compare against. The slim multi-stage version is far smaller, with fewer packages and therefore fewer known vulnerabilities.
**3. Example to quote:** > "`docker images sec-demo` showed `single` around a gigabyte and `multi` a fraction of that."
**Seen in:** `Dockerfile.single`, `commands.sh`

## Image size and attack surface `(concept)`
**1. Definition:** A smaller image contains fewer packages, so it has fewer known vulnerabilities (CVEs) and downloads faster.
**2. In plain words:** Every extra tool in an image is something that can be attacked. Slim bases and multi-stage builds shrink both size and risk.
**3. Example to quote:** > "I chose slim and multi-stage because fewer packages in the image means fewer things to patch."
**Seen in:** `SECURITY_CHECKLIST.md`

## Pinning versions `(concept)`
**1. Definition:** Fixing exact versions of base images and packages (`python:3.12-slim`, `fastapi==0.146.0`) rather than floating tags like `latest`.
**2. In plain words:** Floating versions can change under you and break a build that worked yesterday. Pinning makes builds reproducible.
**3. Example to quote:** > "My requirements are pinned, so the image I build next month matches today's."
**Seen in:** `SECURITY_CHECKLIST.md`, `requirements.txt`

## Secrets in images `(security concept)`
**1. Definition:** Anything baked into an image via `ENV`, `COPY .env` or `ARG` is stored in its layers and readable by anyone who can pull it; `docker history` even prints `ENV` lines.
**2. In plain words:** Never put API keys in a Dockerfile. Inject them at run time with `-e`, `--env-file`, Compose `environment` or an orchestrator's secrets feature, and exclude `.env` with `.dockerignore`.
**3. Example to quote:** > "I pass my API key with `-e` at run time; if I had written `ENV API_KEY=...`, `docker history` would reveal it."
**Seen in:** `SECURITY_CHECKLIST.md`

## One process per container `(concept)`
**1. Definition:** The practice of running a single main service in each container, composed together with others.
**2. In plain words:** It keeps logs simple, lets you scale and restart parts independently, and matches how Compose runs api, ui and redis separately.
**3. Example to quote:** > "My stack has three containers rather than one container running everything."
**Seen in:** `SECURITY_CHECKLIST.md`

## COPY vs ADD `(Dockerfile instructions)`
**1. Definition:** `COPY` copies local files; `ADD` does the same but also extracts local tar archives and can fetch URLs.
**2. In plain words:** `ADD`'s extra behaviour is surprising, so the guidance is to prefer `COPY` unless you really want that magic.
**3. Example to quote:** > "I use `COPY` everywhere; I only reach for `ADD` if I specifically need to unpack a tarball."
**Seen in:** `SECURITY_CHECKLIST.md`

## Volume vs bind mount vs image layer `(concept)`
**1. Definition:** Image layers are read-only and baked in; volumes are Docker-managed persistent storage; bind mounts map a host path into the container.
**2. In plain words:** Choosing correctly matters: put code and dependencies in layers, state in volumes, and development files in bind mounts.
**3. Example to quote:** > "I keep the app in layers, the database files in a named volume, and edit code through a bind mount in development."
**Seen in:** `SECURITY_CHECKLIST.md`

## Containers share the host kernel `(concept)`
**1. Definition:** Containers are isolated processes using kernel namespaces and cgroups, not separate operating systems.
**2. In plain words:** That is what makes them light, and also why isolation is weaker than a VM's. Hardening steps like non-root users matter because of it.
**3. Example to quote:** > "A container is not a VM; it's an isolated process on the host's kernel, so I run it as non-root."
**Seen in:** `SECURITY_CHECKLIST.md`

## Image scanning (docker scout, trivy) `(tool)`
**1. Definition:** Tools that inspect an image's packages against vulnerability databases and report known CVEs.
**2. In plain words:** They tell you which libraries in your image have published security problems so you can update the base image or packages.
**3. Example to quote:** > "Before shipping I'd run `docker scout quickview` on the image to look for known vulnerabilities."
**Seen in:** `SECURITY_CHECKLIST.md`

## .dockerignore `(file)`
**1. Definition:** Lists paths excluded from the build context.
**2. In plain words:** Keeps `.venv`, `.git` and `.env` out of the context and therefore out of the image, which helps both size and secrecy.
**3. Example to quote:** > "My `.dockerignore` stops the `.env` file from ever being copied into an image layer."
**Seen in:** `.dockerignore`

## pip install --no-cache-dir `(command flag)`
**1. Definition:** Prevents pip from keeping its download cache.
**2. In plain words:** In a multi-stage build the builder's cache is discarded anyway, but the flag keeps the builder layer lean and is a good habit everywhere.
**3. Example to quote:** > "I add `--no-cache-dir` in the builder so pip's cache doesn't bloat even the intermediate layers."
**Seen in:** `Dockerfile`

## docker exec whoami `(command)`
**1. Definition:** Runs `whoami` inside a running container to show which user the process runs as.
**2. In plain words:** It's the quick proof that your `USER` line took effect: `appuser` is good, `root` is not.
**3. Example to quote:** > "`docker exec sec whoami` printed `appuser`, confirming the container isn't running as root."
**Seen in:** `commands.sh`

## docker ps --format `(command flag)`
**1. Definition:** Formats `docker ps` output with a Go template such as `'{{.Names}} {{.Status}}'`.
**2. In plain words:** It prints just the columns you care about, making the health status easy to read at a glance.
**3. Example to quote:** > "`docker ps --format '{{.Names}} {{.Status}}'` showed `sec Up 15 seconds (healthy)`."
**Seen in:** `commands.sh`

## FastAPI toy app and requirements `(concept)`
**1. Definition:** The same keyword-sentiment FastAPI app and pinned `fastapi` and `uvicorn` requirements used across the module.
**2. In plain words:** The app is kept trivial so the lesson is about the image, not the code; the pinned versions make the size comparison fair.
**3. Example to quote:** > "Both Dockerfiles package the identical app, so any size difference comes only from the build approach."
**Seen in:** `app/main.py`, `requirements.txt`

## docker images `(command)`
**1. Definition:** Lists local images with their sizes.
**2. In plain words:** It's how you compare the single-stage and multi-stage images side by side.
**3. Example to quote:** > "`docker images sec-demo` lists `single` and `multi` so I can compare their SIZE column."
**Seen in:** `commands.sh`
