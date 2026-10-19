# Image optimisation & security checklist

| Practice | Why | Where it appears in this module |
|---|---|---|
| Small base (`-slim`) | Smaller image, fewer CVEs, faster pulls | Dockerfile (step 01), compare in 06 |
| Pin versions (`fastapi==0.146.0`, `python:3.12-slim`) | Reproducible builds; `latest` changes under you | requirements.txt |
| `pip install --no-cache-dir` | pip's download cache is dead weight in a layer | every Dockerfile |
| Order layers by change frequency | Fast rebuilds through the layer cache | 02_layers_and_cache |
| `.dockerignore` | Keeps `.venv`, `.git`, `.env`, data out of the build context and the image | every step |
| Multi-stage build | Build tools never reach the final image | 06 Dockerfile |
| Run as non-root (`USER`) | Limits damage if the app is compromised | 06 Dockerfile |
| No secrets in image (`ENV API_KEY=...`, `COPY .env`) | Image layers are readable by anyone who can pull the image, and `docker history` shows ENV lines. Inject at run time: `-e`, `--env-file`, Compose `environment`, orchestrator secrets | 03 and 04 |
| One process per container | Simple logs, scaling, restarts | 04 (api / ui / redis) |
| Health checks | Detect "running but broken" | 04, 06 |
| Scan images (`docker scout quickview`, trivy) | Find known CVEs | optional |

## Common interview traps
- **CMD vs ENTRYPOINT**: ENTRYPOINT is the executable, CMD the default args. `docker run image X` replaces CMD (not ENTRYPOINT).
- **Shell form vs exec form**: `CMD uvicorn ...` (shell form) wraps in `/bin/sh -c`, so SIGTERM may not reach your app; prefer `CMD ["uvicorn", ...]`.
- **COPY vs ADD**: prefer COPY; ADD also untars and fetches URLs (surprising).
- **Volume vs bind mount vs image layer**: layers are read-only and baked; volumes are managed persistent storage; bind mounts map a host path.
- **A container is not a VM**: it shares the host kernel (isolation through namespaces and cgroups).
