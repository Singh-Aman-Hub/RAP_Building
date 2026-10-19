# Step 04 - Docker Compose: API + UI + Redis

## The story so far
[Step 03](../03_ports_env_volumes/WALKTHROUGH.md) configured one container. Our system needs three: an API, a Streamlit UI and Redis as a cache.

## The pain point
By hand you start Redis, wait, start the API with the right Redis address, wait, start the UI with the API address. Inside a container `localhost` means that container itself, so the UI cannot reach the API through it. One typo and the stack is half-alive.

## The spark
Describe the whole stack once in a `compose.yaml`. Compose creates a shared network where each service name is a DNS name (`http://api:8000`, host `redis`), and `depends_on` with a health condition starts services only when the one they need is ready.

## Goal
Run three cooperating containers with one command and understand how they find each other.

## Run it
```bash
cd 11_docker/04_compose_multi_container
docker compose config > /dev/null && echo valid     # works without the daemon (verified when written)
./commands.sh                                        # needs Docker Desktop
```
Local (no Docker) check of the API in memory mode, real output:
```
{'label': 'negative', 'score': -1} {'predictions': 1, 'counter_backend': 'memory'} {'status': 'ready'}
```

## Syntax explained (YAML + Compose)
| Syntax | Meaning | Example |
|---|---|---|
| YAML indentation | Structure (2 spaces, no tabs) | `services:` then `  api:` |
| `- item` | list element | `- "8000:8000"` |
| `key: value` | mapping | `REDIS_HOST: redis` |
| `${VAR}` | filled from `.env` / shell | `${API_PORT}` |
| `build: ./api` vs `image: redis:7-alpine` | build locally vs pull | |
| `depends_on.condition: service_healthy` | start order based on health | |
| `volumes:` (top level) | declares named volumes | `redis-data:` |

## Commands
| Command | Does | Notes |
|---|---|---|
| `docker compose up --build -d` | build + start all | `-d` detached |
| `docker compose ps` | status incl. health | |
| `docker compose logs -f api` | follow one service | |
| `docker compose exec api sh` | shell in running service | |
| `docker compose down [-v]` | stop+remove (+volumes) | `-v` deletes data |

## Execution trace
1. Compose creates a project network (bridge, user-defined). Every service joins it and gets a DNS name = service name.
2. `redis` starts; its healthcheck (`redis-cli ping`) turns healthy.
3. `api` starts (depends_on healthy), reads `REDIS_HOST=redis`; `redis.Redis(host="redis")` resolves via Docker DNS.
4. `ui` starts once api is healthy; browser (host) -> `localhost:8501` -> ui container -> `http://api:8000` (container port, not the host-mapped one) -> redis.
5. `down` removes containers; `redis-data` volume keeps the counter.
Important distinction: the *browser* uses published host ports; *containers* talk over the network by service name and container port.

## Experiments
1. Change `API_URL` in the ui service to `http://localhost:8000`: the UI fails ("connection refused"). Why? (localhost = the ui container.)
2. `docker compose stop redis`; call `/ready` -> 503; `/health` still 200.
3. Remove the volume mapping; `down` then `up`: the counter resets.
4. Delete the `ports:` of `api`: UI still works, `curl localhost:8000` does not (internal-only service).

## Debugging practice
`dependency failed to start: container ...-redis-1 is unhealthy` -> run `docker compose logs redis` and `docker inspect --format '{{json .State.Health}}' <container>` to see the last health probe outputs.

## Rebuild checklist
- [ ] Draw the network: browser, ui, api, redis, with the hostnames and ports used on each arrow.
- [ ] Write a 2-service compose file from memory.

## The new problem (cliffhanger)
One `docker compose up` works, until it does not: a container is unhealthy, exits at once, or is up but unreachable. Now you need a method, not luck, to find out why. Next: [step 05, health checks, logs and debugging](../05_healthchecks_logs_debugging/WALKTHROUGH.md).

## Say it aloud
> **Say it aloud (interview box):** Compose describes a multi-container app declaratively, and its services share a network and reach each other by service name, so the API connects to host redis while only published ports reach my laptop. depends_on with a health condition orders start-up, because started does not mean ready. The trade-off is that Compose suits dev and single-host deploys but has no multi-node scaling or self-healing like Kubernetes.
