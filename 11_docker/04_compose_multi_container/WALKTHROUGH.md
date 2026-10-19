# Step 04 - Docker Compose: API + UI + Redis

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

## Say it aloud
"Compose describes a multi-container app declaratively. Services share a network and reach each other by service name, so the API connects to host `redis`, while only the ports I publish are reachable from my laptop. depends_on with health conditions orders start-up. Trade-off: Compose is great for dev and single-host deploys, but it does not provide multi-node scaling or self-healing like Kubernetes."
