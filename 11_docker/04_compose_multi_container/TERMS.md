# Terms — Docker Compose: API + UI + Redis

Folder: `11_docker/04_compose_multi_container` · Files covered: `compose.yaml`, `.env`, `api/Dockerfile`, `api/app/main.py`, `api/requirements.txt`, `ui/Dockerfile`, `ui/app.py`, `ui/requirements.txt`, `commands.sh`, `WALKTHROUGH.md`
Back to module: [11_docker README](../README.md)

**Terms in this folder:** [Docker Compose (concept)](#docker-compose-concept) · [compose.yaml (file)](#composeyaml-file) · [YAML syntax (indentation, mapping, list) (syntax)](#yaml-syntax-indentation-mapping-list-syntax) · [Service (Compose concept)](#service-compose-concept) · [Service-name DNS (concept)](#service-name-dns-concept) · [Compose network (concept)](#compose-network-concept) · [image vs build (compose.yaml keys)](#image-vs-build-composeyaml-keys) · [ports (HOST:CONTAINER) (compose.yaml key)](#ports-hostcontainer-composeyaml-key) · [environment (compose.yaml) (compose.yaml key)](#environment-composeyaml-composeyaml-key) · [${VARIABLE} interpolation and .env (Compose feature)](#variable-interpolation-and-env-compose-feature) · [depends_on with condition: service_healthy (compose.yaml key)](#depends_on-with-condition-service_healthy-composeyaml-key) · [healthcheck (compose.yaml) (compose.yaml key)](#healthcheck-composeyaml-composeyaml-key) · [Named volume (top-level volumes:) (compose.yaml key)](#named-volume-top-level-volumes-composeyaml-key) · [Redis (third-party service)](#redis-third-party-service) · [redis-py client (third-party: redis)](#redis-py-client-third-party-redis) · [Lazy / conditional import (syntax)](#lazy--conditional-import-syntax) · [global statement (Python built-in syntax)](#global-statement-python-built-in-syntax) · [Health vs readiness endpoints (concept)](#health-vs-readiness-endpoints-concept) · [HTTPException and 503 (third-party: fastapi)](#httpexception-and-503-third-party-fastapi) · [try / except Exception (syntax)](#try--except-exception-syntax) · [Streamlit (third-party: streamlit)](#streamlit-third-party-streamlit) · [requests.post with timeout (third-party: requests)](#requestspost-with-timeout-third-party-requests) · [raise_for_status (third-party: requests)](#raise_for_status-third-party-requests) · [requests.RequestException (third-party: requests)](#requestsrequestexception-third-party-requests) · [localhost inside a container (concept)](#localhost-inside-a-container-concept) · [streamlit run --server.address (command flag)](#streamlit-run---serveraddress-command-flag) · [docker compose up / down / ps / logs / exec (commands)](#docker-compose-up--down--ps--logs--exec-commands) · [docker compose config (command)](#docker-compose-config-command) · [Kubernetes vs Compose (concept)](#kubernetes-vs-compose-concept)

---

## Docker Compose `(concept)`
**1. Definition:** A tool that defines and runs multi-container applications from a single YAML file describing services, networks and volumes.
**2. In plain words:** Instead of typing three long `docker run` commands in the right order, you describe the whole app once and start it with one command. Compose builds images, creates a shared network and volume, and starts the containers in the order you specify. It is great for development and single-machine setups but it does not scale across many machines.
**3. Example to quote:** > "One `docker compose up --build -d` started my Redis, API and Streamlit UI together, in dependency order."
**Seen in:** `compose.yaml`, `commands.sh`

## compose.yaml `(file)`
**1. Definition:** The Compose configuration file whose top-level keys include `services`, `volumes` and optionally `networks`.
**2. In plain words:** It is the blueprint of the application. Each entry under `services:` becomes a container, and it says how to get the image, which ports to publish, which environment variables to set and what it depends on.
**3. Example to quote:** > "My `compose.yaml` has three services, `redis`, `api` and `ui`, plus one named volume `redis-data`."
**Seen in:** `compose.yaml`

## YAML syntax (indentation, mapping, list) `(syntax)`
**1. Definition:** A human-readable data format where nesting is shown by indentation with spaces (never tabs), `key: value` forms a mapping and `- item` forms a list.
**2. In plain words:** Spacing is meaning. A line indented two spaces under `services:` belongs to it. One wrong indent changes the structure or breaks the file, so a validator like `docker compose config` is very handy.
**3. Example to quote:** > "Under `ports:` I wrote `- "8000:8000"`; the dash makes it a list item, and the quotes stop YAML misreading the colon."
```yaml
services:
  api:
    ports:
      - "8000:8000"
```
**Seen in:** `compose.yaml`, `WALKTHROUGH.md`

## Service `(Compose concept)`
**1. Definition:** A named container definition in Compose; its name is also its DNS hostname on the Compose network.
**2. In plain words:** Each part of your app, such as the database, the API or the UI, is a service. Other services reach it by that name, so you never need to know its IP address.
**3. Example to quote:** > "The Redis service is just called `redis`, so my API connects to host `redis`."
**Seen in:** `compose.yaml`

## Service-name DNS `(concept)`
**1. Definition:** Docker's embedded DNS on user-defined networks resolves each service name to the container's IP address on that network.
**2. In plain words:** On the Compose network, `redis` and `api` are working hostnames. That is why `localhost` is wrong between containers: inside a container, `localhost` means that very container.
**3. Example to quote:** > "I proved it with `socket.gethostbyname('redis')` inside the api container, which returned the Redis container's IP."
**Seen in:** `commands.sh`, `WALKTHROUGH.md`

## Compose network `(concept)`
**1. Definition:** A user-defined bridge network that Compose creates per project; all services join it and can talk to each other on container ports.
**2. In plain words:** Services are placed on a private virtual network. Only ports you publish with `ports:` are reachable from your laptop; container-to-container traffic flows freely inside the network, on the container port, not on the host-mapped port.
**3. Example to quote:** > "The UI calls `http://api:8000` — the container port — even though the host maps it differently."
**Seen in:** `WALKTHROUGH.md`, execution trace

## image vs build `(compose.yaml keys)`
**1. Definition:** `image:` pulls a ready-made image from a registry; `build:` builds an image from a Dockerfile in a given folder.
**2. In plain words:** Use `image:` for off-the-shelf software like Redis, and `build:` for your own code. `build: ./api` uses `./api/Dockerfile` with `./api` as the build context.
**3. Example to quote:** > "Redis uses `image: ${REDIS_IMAGE}` while my API uses `build: ./api`."
**Seen in:** `compose.yaml`

## ports (HOST:CONTAINER) `(compose.yaml key)`
**1. Definition:** Publishes container ports on host ports, using the same `HOST:CONTAINER` order as `docker run -p`.
**2. In plain words:** It's what makes a service reachable from your browser. A service with no `ports:` is internal only: other containers can reach it but your laptop cannot.
**3. Example to quote:** > "I removed `ports:` from the API and the UI still worked, but `curl localhost:8000` stopped; the API was internal-only."
**Seen in:** `compose.yaml`

## environment (compose.yaml) `(compose.yaml key)`
**1. Definition:** Sets environment variables inside a service's container.
**2. In plain words:** It's `docker run -e` in YAML form. Here it tells the API where Redis is (`REDIS_HOST: redis`) and tells the UI where the API is (`API_URL: http://api:8000`).
**3. Example to quote:** > "`API_URL: http://api:8000` in Compose is how the UI finds the API without any code change."
**Seen in:** `compose.yaml`

## ${VARIABLE} interpolation and .env `(Compose feature)`
**1. Definition:** Compose substitutes `${NAME}` in the YAML with values from the shell environment or a `.env` file in the project folder.
**2. In plain words:** Settings such as ports and the Redis image tag live in `.env`, so you change them without editing the YAML. Note this `.env` configures Compose itself; it is not automatically passed into the containers.
**3. Example to quote:** > "I set `API_PORT=8000` in `.env`, and `compose.yaml` uses `"${API_PORT}:8000"`."
**Seen in:** `.env`, `compose.yaml`

## depends_on with condition: service_healthy `(compose.yaml key)`
**1. Definition:** Orders start-up so a service waits until the listed dependency is not just started but reports a healthy status.
**2. In plain words:** "Started" does not mean "ready". Redis may take a moment to accept connections. With the health condition, the API only starts after Redis answers its ping, and the UI waits for the API.
**3. Example to quote:** > "Without the health condition my API sometimes crashed because Redis wasn't accepting connections yet."
**Seen in:** `compose.yaml`

## healthcheck (compose.yaml) `(compose.yaml key)`
**1. Definition:** A command Docker runs periodically inside the container to decide whether it is healthy, with `interval`, `timeout`, `retries` and `start_period` settings.
**2. In plain words:** It's an automatic "are you OK?" probe. Redis uses `redis-cli ping`. The API uses a tiny Python one-liner with `urllib` because `curl` isn't installed in `python:slim`. `start_period` gives the app time to boot before failures count.
**3. Example to quote:** > "After startup `docker compose ps` showed `(healthy)` for each service once its probe passed."
```yaml
healthcheck:
  test: ["CMD", "redis-cli", "ping"]
  interval: 5s
  timeout: 3s
  retries: 5
```
**Seen in:** `compose.yaml`

## Named volume (top-level volumes:) `(compose.yaml key)`
**1. Definition:** A top-level `volumes:` entry declares a named volume that services can mount, persisting across container removal.
**2. In plain words:** `redis-data:/data` means Redis writes into a Docker-managed volume. `docker compose down` removes containers but keeps it; `down -v` deletes it too and the counter resets.
**3. Example to quote:** > "After `docker compose down` and `up`, the prediction count was still there, because it lived in the `redis-data` volume."
**Seen in:** `compose.yaml`

## Redis `(third-party service)`
**1. Definition:** An in-memory key-value data store, commonly used as a cache, counter or message broker, accessed over the network.
**2. In plain words:** It's a small, very fast database that keeps data in memory. Here it simply counts predictions, which shows how an app container talks to a second container.
**3. Example to quote:** > "Redis is a separate container; my API increments a counter named `predictions` stored inside it."
**Seen in:** `compose.yaml`, `api/app/main.py`

## redis-py client `(third-party: redis)`
**1. Definition:** The Python library `redis` providing `redis.Redis(host, port, decode_responses)` and commands like `incr`, `get` and `ping`.
**2. In plain words:** It's how Python talks to the Redis server. `decode_responses=True` returns normal strings instead of bytes. `incr` is atomic, so two simultaneous requests can't both read the same number.
**3. Example to quote:** > "`_redis.incr("predictions")` adds one to the shared counter and returns the new value."
```python
_redis = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
```
**Seen in:** `api/app/main.py`

## Lazy / conditional import `(syntax)`
**1. Definition:** Placing an `import` inside an `if` block or function so the module is loaded only when needed.
**2. In plain words:** The API imports `redis` only when `REDIS_HOST` is set, so the app still runs locally with an in-memory counter and no Redis installed.
**3. Example to quote:** > "Locally I run without Redis and it falls back to memory; in Compose, `REDIS_HOST=redis` switches on the real client."
**Seen in:** `api/app/main.py`

## global statement `(Python built-in syntax)`
**1. Definition:** Declares that a name assigned inside a function refers to the module-level variable instead of a new local one.
**2. In plain words:** Without `global`, `_memory_counter += 1` would fail because Python would treat it as a local variable. It is acceptable for a small demo, though shared mutable globals are not thread-safe.
**3. Example to quote:** > "I wrote `global _memory_counter` so the increment changes the module-level counter."
**Seen in:** `api/app/main.py`, `incr`

## Health vs readiness endpoints `(concept)`
**1. Definition:** A health (liveness) endpoint reports that the process is alive; a readiness endpoint reports whether it can actually do work, such as reach its dependencies.
**2. In plain words:** `/health` always says ok if the app runs; `/ready` also pings Redis and returns 503 if it's down. Orchestrators use the difference to decide between restarting a container and just not sending traffic to it.
**3. Example to quote:** > "I stopped Redis and `/ready` returned 503 while `/health` stayed 200."
**Seen in:** `api/app/main.py`, `ready`

## HTTPException and 503 `(third-party: fastapi)`
**1. Definition:** `HTTPException(status_code, detail)` aborts the request with an HTTP error response; 503 means Service Unavailable.
**2. In plain words:** It's the proper way to say "I can't serve this right now" from inside a FastAPI function. The client gets a clean status and message instead of a crash.
**3. Example to quote:** > "When Redis is down, `/ready` raises `HTTPException(503, ...)` with the connection error text."
**Seen in:** `api/app/main.py`

## try / except Exception `(syntax)`
**1. Definition:** Runs a block and, if any exception derived from `Exception` occurs, executes the handler instead of crashing.
**2. In plain words:** A broad `except` is usually a code smell, but it's appropriate here: any failure talking to Redis means "not ready". Convert the failure into a 503 rather than letting it propagate.
**3. Example to quote:** > "I wrap `_redis.ping()` in try/except and turn any failure into a 503."
**Seen in:** `api/app/main.py`

## Streamlit `(third-party: streamlit)`
**1. Definition:** A Python library that turns scripts into interactive web apps using functions such as `st.title`, `st.text_input`, `st.button` and `st.json`.
**2. In plain words:** You write a top-to-bottom script and Streamlit shows it in the browser, re-running it on each interaction. It is the UI container in this stack.
**3. Example to quote:** > "My UI container runs `streamlit run app.py` and shows a text box and a Predict button."
**Seen in:** `ui/app.py`

## requests.post with timeout `(third-party: requests)`
**1. Definition:** `requests.post(url, json=..., timeout=...)` sends an HTTP POST with a JSON body and raises if no response arrives within the timeout.
**2. In plain words:** It's how the UI calls the API. Always set a timeout; without one a stuck API would freeze your UI forever. `json=` encodes the dict and sets the content type for you.
**3. Example to quote:** > "`requests.post(f"{API_URL}/predict", json={"text": text}, timeout=5)` gives up after five seconds instead of hanging."
**Seen in:** `ui/app.py`

## raise_for_status `(third-party: requests)`
**1. Definition:** A response method that raises `requests.HTTPError` if the status code is 4xx or 5xx.
**2. In plain words:** Without it, an error page would be treated as a normal answer. Calling it makes failures show up as exceptions that you can catch and display.
**3. Example to quote:** > "A 503 from the API becomes an exception, so the UI shows an error box instead of garbage."
**Seen in:** `ui/app.py`

## requests.RequestException `(third-party: requests)`
**1. Definition:** The base exception class for all errors raised by the `requests` library, including connection errors and timeouts.
**2. In plain words:** Catching it handles "API unreachable", "timeout" and bad statuses in one clause, and you can show the user a friendly message.
**3. Example to quote:** > "If I set `API_URL` to localhost inside the UI container, `RequestException` fires and the UI says it could not reach the API."
**Seen in:** `ui/app.py`

## localhost inside a container `(concept)`
**1. Definition:** `localhost` (127.0.0.1) refers to the container's own network namespace, not the host or other containers.
**2. In plain words:** It's the single most common Compose mistake. The UI can't reach the API at `localhost:8000` because there's nothing listening inside the UI container itself. Use the service name.
**3. Example to quote:** > "I set `API_URL` to `http://localhost:8000` in the UI service and got connection refused; `http://api:8000` fixed it."
**Seen in:** `ui/app.py` docstring, `WALKTHROUGH.md` experiment 1

## streamlit run --server.address `(command flag)`
**1. Definition:** `streamlit run app.py --server.address=0.0.0.0 --server.port=8501` starts the Streamlit server bound to all interfaces on port 8501.
**2. In plain words:** Like uvicorn's `--host 0.0.0.0`, binding to all interfaces is required for the published port to work from outside the container.
**3. Example to quote:** > "My UI Dockerfile ends with `--server.address=0.0.0.0` for the same reason my API uses `--host 0.0.0.0`."
**Seen in:** `ui/Dockerfile`

## docker compose up / down / ps / logs / exec `(commands)`
**1. Definition:** `up --build -d` builds and starts all services detached; `down` stops and removes containers and the network (`-v` also removes volumes); `ps` shows status and health; `logs -f SERVICE` follows logs; `exec SERVICE CMD` runs a command in a running service.
**2. In plain words:** These five cover almost all day-to-day Compose work. `logs` and `exec` are your debugging tools, and `down -v` is the one that deletes data, so use it deliberately.
**3. Example to quote:** > "I used `docker compose logs -f api` to watch requests and `docker compose exec api sh` to look inside the running container."
**Seen in:** `commands.sh`

## docker compose config `(command)`
**1. Definition:** Validates `compose.yaml` and prints the final configuration with variables substituted; it does not need the Docker daemon running.
**2. In plain words:** It catches indentation and variable mistakes before you start anything. If it prints a clean file, your YAML is structurally valid.
**3. Example to quote:** > "I ran `docker compose config` and saw `${API_PORT}` replaced by 8000, so I knew the `.env` was read."
**Seen in:** `commands.sh`, `WALKTHROUGH.md`

## Kubernetes vs Compose `(concept)`
**1. Definition:** Kubernetes is a cluster orchestrator with scheduling, self-healing and multi-node scaling; Compose manages containers on a single host.
**2. In plain words:** Compose is the simple option that fits laptops and small deployments. When you need many machines, rolling updates and automatic recovery, you move to something like Kubernetes.
**3. Example to quote:** > "Compose is great for my dev stack, but it won't replace failed containers across machines the way Kubernetes would."
**Seen in:** `WALKTHROUGH.md`, "Say it aloud"
