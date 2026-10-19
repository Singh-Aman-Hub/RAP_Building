# Step 06 - Multi-stage builds, small images and security basics

## The story so far
[Step 05](../05_healthchecks_logs_debugging/WALKTHROUGH.md) taught us to diagnose containers. This step makes the image worth shipping.

## The pain point
`Dockerfile.single` installs and runs in one stage: compilers and pip caches stay in the final image, it is large, it has more known vulnerabilities, and the process runs as root. If the app is compromised, the attacker is root in the container.

## The spark
Build in one stage and copy only the results into a clean slim runtime stage (**multi-stage**). Create and switch to a **non-root** user, pin versions, keep secrets out of layers (see `SECURITY_CHECKLIST.md`) and add a `HEALTHCHECK`.

## Goal
Produce a small, non-root, health-checked image and know the security habits interviewers ask about.

## Run it
```bash
cd 11_docker/06_multistage_and_security
cat Dockerfile Dockerfile.single SECURITY_CHECKLIST.md
./commands.sh                     # needs Docker Desktop
```

## Syntax explained
| Syntax | Meaning | Example |
|---|---|---|
| `FROM x AS builder` | Name a stage | second `FROM` starts a fresh image |
| `COPY --from=builder /install /usr/local` | Copy files out of an earlier stage | |
| `pip install --prefix=/install` | Install to a custom folder | |
| `useradd --create-home --uid 10001 appuser` | Make an unprivileged user | |
| `USER appuser` | Everything after (and the running process) uses it | |
| `COPY --chown=u:g` | Set ownership while copying | |
| `ENTRYPOINT [...]` + `CMD [...]` | fixed program + default args | `docker run img --version` |
| `ENV PYTHONUNBUFFERED=1` | flush prints immediately so `docker logs` is live | |

## Execution trace
1. Stage `builder`: installs deps into `/install`. Stage `runtime`: starts from a clean slim image, copies only `/install` and the code. The final image contains no builder layers.
2. `docker images sec-demo`: `single` (based on full `python:3.12`, ~1 GB) vs `multi` (slim, ~150-200 MB). (Typical values, not captured.)
3. `docker exec sec whoami` -> `appuser`. After ~10 s `docker ps` shows `(healthy)`.
Note: with only pure-Python wheels the multi-stage gain is mostly "slim vs full + no pip cache"; the pattern pays off hugely when builder needs gcc/compilers.

## Experiments
1. Add `numpy` to requirements; compare sizes again.
2. Remove `USER appuser`; run `docker exec sec whoami` -> `root`.
3. Run `docker run --rm sec-demo:multi app.main:app --port 9000` - what replaced what? (CMD args replaced; ENTRYPOINT intact.)

## Debugging practice
`PermissionError: [Errno 13] Permission denied: '/data'` after adding `USER` -> the non-root user cannot write to a root-owned folder. Fix with `RUN mkdir /data && chown appuser /data` before `USER`.

## Rebuild checklist
- [ ] Sketch a 2-stage Dockerfile from memory.
- [ ] List 5 hardening practices.

## The new problem (cliffhanger)
The service is now portable, small and safer, but its only users are programs and people who can type `curl`. A product manager cannot test it. We need a face for the model. Next: [module 12, Streamlit and Gradio](../../12_streamlit_gradio/01_streamlit_basics/WALKTHROUGH.md).

## Say it aloud
> **Say it aloud (interview box):** A multi-stage build installs or compiles in one stage and copies only the results into a clean runtime stage, so the final image is smaller and has no build tools. I also run as a non-root user, pin versions, keep secrets out of layers and add a health check. The trade-off is a more complex Dockerfile and a slower first build, in exchange for a smaller and safer image.
