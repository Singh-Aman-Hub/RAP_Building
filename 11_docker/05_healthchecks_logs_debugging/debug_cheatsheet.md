# Docker debugging cheat-sheet

| Question | Command |
|---|---|
| What is running / what died? | `docker ps -a` (look at STATUS and exit code) |
| What did it print? | `docker logs <c>`; `docker logs -f --tail 50 <c>`; Compose: `docker compose logs -f <service>` |
| Look inside a running container | `docker exec -it <c> sh` (then `ls`, `env`, `ps`, `cat file`) |
| Start a shell in an image whose CMD crashes | `docker run --rm -it --entrypoint sh <image>` |
| Full configuration (ports, env, mounts, health) | `docker inspect <c>` (use `--format '{{.State.ExitCode}}'` to filter) |
| Health status | `docker inspect --format '{{json .State.Health}}' <c>` |
| Which ports are published? | `docker port <c>` |
| CPU / memory live | `docker stats` |
| Which layer made the image big? | `docker history <image>` |
| Rebuild ignoring cache | `docker build --no-cache -t name .` |
| Compose: effective config | `docker compose config` |
| Disk cleanup | `docker system df`, `docker system prune` (careful: deletes stopped containers and dangling images) |

## Systematic order (same as debugging Python)
1. Is the container running? (`ps -a`, exit code) 2. Read logs bottom-up. 3. Is the app listening on the port/host
you think? 4. Is the port mapped the way you think? 5. Is config (env vars, files) what the app expects? (`exec ... env`)
6. Reproduce minimally (`docker run --rm -it ... sh`) and change ONE thing.

## HEALTHCHECK in a Dockerfile
```dockerfile
# python:slim has no curl, so use Python itself. Exit code 0 = healthy, 1 = unhealthy.
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')" || exit 1
```
Docker only *reports* unhealthy; plain Docker will not restart it. Compose `depends_on: condition: service_healthy`
and orchestrators (Kubernetes liveness/readiness probes, ECS) *act* on it.
