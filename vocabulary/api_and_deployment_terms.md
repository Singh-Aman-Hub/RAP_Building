# API, Docker, Architecture and Deployment Terms

140 terms, alphabetical. Back to [index](index.md).

- [.dockerignore](#dockerignore)
- [0.0.0.0 vs 127.0.0.1 binding](#0000-vs-127001-binding)
- [422 Unprocessable Content](#422-unprocessable-content)
- [Alias (model registry)](#alias-model-registry)
- [API client (UI to API)](#api-client-ui-to-api)
- [AppTest (Streamlit testing)](#apptest-streamlit-testing)
- [ASGI](#asgi)
- [Autoscaling](#autoscaling)
- [Base image](#base-image)
- [Batch inference](#batch-inference)
- [Bind mount](#bind-mount)
- [Bounded queue](#bounded-queue)
- [Build cache (layer cache)](#build-cache-layer-cache)
- [Build context](#build-context)
- [Caching](#caching)
- [Callback (event handler)](#callback-event-handler)
- [Canary deployment](#canary-deployment)
- [CMD vs ENTRYPOINT](#cmd-vs-entrypoint)
- [Compose file](#compose-file)
- [Concurrency](#concurrency)
- [Configuration](#configuration)
- [Confirmation hook](#confirmation-hook)
- [Container](#container)
- [Container networking](#container-networking)
- [COPY (Dockerfile)](#copy-dockerfile)
- [curl](#curl)
- [Dependency injection (FastAPI)](#dependency-injection-fastapi)
- [depends_on](#depends_on)
- [Deployment (UI apps)](#deployment-ui-apps)
- [Docker Compose](#docker-compose)
- [docker exec](#docker-exec)
- [Docker logs](#docker-logs)
- [Dockerfile](#dockerfile)
- [Environment variable (Docker)](#environment-variable-docker)
- [Error handler](#error-handler)
- [Event loop](#event-loop)
- [Exec form vs shell form](#exec-form-vs-shell-form)
- [Exit code (container)](#exit-code-container)
- [Exponential backoff](#exponential-backoff)
- [EXPOSE](#expose)
- [FastAPI](#fastapi)
- [Flask](#flask)
- [FROM](#from)
- [GET vs POST](#get-vs-post)
- [gr.Blocks](#grblocks)
- [gr.Error](#grerror)
- [gr.Interface](#grinterface)
- [gr.Label](#grlabel)
- [gr.Progress](#grprogress)
- [gr.State](#grstate)
- [Gradio](#gradio)
- [Gunicorn](#gunicorn)
- [Header](#header)
- [Health check](#health-check)
- [Health check (Docker)](#health-check-docker)
- [HTTP](#http)
- [HTTP 500](#http-500)
- [HTTP method](#http-method)
- [Idempotent](#idempotent)
- [Image](#image)
- [Jitter](#jitter)
- [JSON (JavaScript Object Notation)](#json-javascript-object-notation)
- [JSON Schema](#json-schema)
- [JSONL (JSON Lines)](#jsonl-json-lines)
- [Layer (Docker)](#layer-docker)
- [Least privilege](#least-privilege)
- [Lifespan (startup/shutdown)](#lifespan-startupshutdown)
- [Little's Law](#littles-law)
- [Liveness vs readiness](#liveness-vs-readiness)
- [Load balancer](#load-balancer)
- [Load shedding](#load-shedding)
- [Load testing](#load-testing)
- [Loading state](#loading-state)
- [Micro-batching](#micro-batching)
- [Model registry](#model-registry)
- [Model serving](#model-serving)
- [Model versioning](#model-versioning)
- [Multi-stage build](#multi-stage-build)
- [Namespace and cgroup](#namespace-and-cgroup)
- [Non-root user (container)](#non-root-user-container)
- [Object storage](#object-storage)
- [OpenAPI / Swagger docs](#openapi-swagger-docs)
- [Parameterized query](#parameterized-query)
- [Path parameter](#path-parameter)
- [Path sandboxing](#path-sandboxing)
- [Path traversal](#path-traversal)
- [Permanent error](#permanent-error)
- [Port](#port)
- [Port mapping](#port-mapping)
- [Prediction log](#prediction-log)
- [Pydantic](#pydantic)
- [Query parameter](#query-parameter)
- [Rate limit (429)](#rate-limit-429)
- [Read-only mode](#read-only-mode)
- [Real-time inference](#real-time-inference)
- [Request body](#request-body)
- [Request ID](#request-id)
- [Requirements file](#requirements-file)
- [Rerun model (Streamlit)](#rerun-model-streamlit)
- [REST](#rest)
- [Rollback](#rollback)
- [Route](#route)
- [RUN (Dockerfile)](#run-dockerfile)
- [Secret](#secret)
- [Service (Compose)](#service-compose)
- [Session state](#session-state)
- [Shadow deployment](#shadow-deployment)
- [Share link (Gradio)](#share-link-gradio)
- [Slim image](#slim-image)
- [Smoke test](#smoke-test)
- [SQL injection](#sql-injection)
- [SQLite](#sqlite)
- [st.button](#stbutton)
- [st.cache_data](#stcache_data)
- [st.cache_resource](#stcache_resource)
- [st.columns](#stcolumns)
- [st.error](#sterror)
- [st.file_uploader](#stfile_uploader)
- [st.form](#stform)
- [st.set_page_config](#stset_page_config)
- [st.sidebar](#stsidebar)
- [st.spinner](#stspinner)
- [st.tabs](#sttabs)
- [Status code](#status-code)
- [Streamlit](#streamlit)
- [Streamlit theme (config.toml)](#streamlit-theme-configtoml)
- [Structured logging](#structured-logging)
- [Tail latency](#tail-latency)
- [Test client](#test-client)
- [Timeout (HTTP)](#timeout-http)
- [Transient error](#transient-error)
- [Uvicorn](#uvicorn)
- [Validation](#validation)
- [Virtual machine (vs container)](#virtual-machine-vs-container)
- [Volume (named volume)](#volume-named-volume)
- [Widget](#widget)
- [WORKDIR](#workdir)
- [Worker](#worker)
- [WSGI](#wsgi)
- [YAML](#yaml)

---

### .dockerignore
*From module: 11_docker*

- **Technical definition:** File listing patterns excluded from the build context.
- **Plain English:** Tells Docker which files to ignore.
- **Example / analogy:** Like .gitignore for Docker builds.
- **Snippet:**
```text
.venv
.git
__pycache__
.env
```
- **Snippet explained:** Keeps virtualenvs, git history and secrets out of the image.
- **Related terms:** [Build context](api_and_deployment_terms.md#build-context)

---

### 0.0.0.0 vs 127.0.0.1 binding
*From module: 11_docker*

- **Technical definition:** 127.0.0.1 listens only on loopback (inside the container); 0.0.0.0 listens on all interfaces so published ports can reach it.
- **Plain English:** Inside Docker, servers must listen on 0.0.0.0 or nobody outside can connect.
- **Example / analogy:** Speaking inside a room versus using the front door.
- **Snippet:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
- **Snippet explained:** Makes the server reachable through -p.
- **Related terms:** [Port mapping](api_and_deployment_terms.md#port-mapping)

---

### 422 Unprocessable Content
*From module: 09_api_development*

- **Technical definition:** Status code meaning the request is well-formed JSON but its values fail validation; FastAPI returns it automatically for Pydantic errors.
- **Plain English:** 'I understood your message but the content is not acceptable.'
- **Example / analogy:** Handing in a form with your age written as 'banana'.
- **Snippet:**
```bash
curl -s -o /dev/null -w '%{http_code}' -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{"text":""}'
```
- **Snippet explained:** Prints 422 because `text` fails `min_length=1`.
- **Related terms:** [Status code](api_and_deployment_terms.md#status-code), [Pydantic](api_and_deployment_terms.md#pydantic), [Validation](api_and_deployment_terms.md#validation)

---

### Alias (model registry)
*From module: 10_ml_system_architecture*

- **Technical definition:** A movable label (production, staging) pointing to a specific model version, so serving code refers to the label and ops change the target.
- **Plain English:** A nickname for a version that you can re-point.
- **Example / analogy:** The 'current edition' sticker.
- **Snippet:**
```python
aliases = {'production': 'v0001'}
```
- **Snippet explained:** Rolling back is editing this mapping.
- **Related terms:** [Model registry](api_and_deployment_terms.md#model-registry), [Rollback](api_and_deployment_terms.md#rollback)

---

### API client (UI to API)
*From module: 12_streamlit_gradio*

- **Technical definition:** Code in the UI that sends HTTP requests to a prediction API and handles responses and failures.
- **Plain English:** The UI asking the model server for answers.
- **Example / analogy:** A waiter taking orders to the kitchen.
- **Snippet:**
```python
r = requests.post(url, json={'text': t}, timeout=10)
r.raise_for_status()
r.json()
```
- **Snippet explained:** Sends JSON with a timeout; raises HTTPError on 4xx/5xx; parses the answer.
- **Related terms:** [Timeout (HTTP)](api_and_deployment_terms.md#timeout-http), [st.error](api_and_deployment_terms.md#sterror)

---

### AppTest (Streamlit testing)
*From module: 12_streamlit_gradio*

- **Technical definition:** streamlit.testing.v1.AppTest runs a Streamlit script headlessly so tests can set widget values and read outputs.
- **Plain English:** Test a Streamlit app without a browser.
- **Example / analogy:** A robot user.
- **Snippet:**
```python
from streamlit.testing.v1 import AppTest
at = AppTest.from_file('app.py').run()
assert not at.exception
```
- **Snippet explained:** Runs the script once and checks no exception was raised.
- **Related terms:** [Streamlit](api_and_deployment_terms.md#streamlit)

---

### ASGI
*From module: 09_api_development*

- **Technical definition:** Asynchronous Server Gateway Interface: async successor to WSGI where the app is `async def app(scope, receive, send)`, supporting HTTP, WebSockets and lifespan events.
- **Plain English:** The modern handshake that allows apps to wait without blocking, and supports live connections.
- **Example / analogy:** A phone line that lets you put callers on hold while helping others.
- **Snippet:**
```python
async def app(scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"hi"})
```
- **Snippet explained:** The smallest ASGI app sends two events.
- **Related terms:** [WSGI](api_and_deployment_terms.md#wsgi), [Uvicorn](api_and_deployment_terms.md#uvicorn), [FastAPI](api_and_deployment_terms.md#fastapi)

---

### Autoscaling
*From module: 10_ml_system_architecture*

- **Technical definition:** Automatically adding or removing replicas/instances based on metrics such as CPU, queue depth or latency.
- **Plain English:** The system adds more servers when busy and removes them when quiet.
- **Example / analogy:** A supermarket opening extra checkouts at rush hour.
- **Snippet:**
n/a
- **Snippet explained:** n/a
- **Related terms:** [Little's Law](api_and_deployment_terms.md#littles-law), [Load balancer](api_and_deployment_terms.md#load-balancer)

---

### Base image
*From module: 11_docker*

- **Technical definition:** The image named in FROM that provides the OS layer and runtimes (e.g., python:3.12-slim).
- **Plain English:** The foundation your image sits on.
- **Example / analogy:** The chassis a car body is mounted on.
- **Snippet:**
```dockerfile
FROM python:3.12-alpine
```
- **Snippet explained:** Alpine is tiny but uses musl libc, so many ML wheels are unavailable; slim is the safe default.
- **Related terms:** [FROM](api_and_deployment_terms.md#from), [Slim image](api_and_deployment_terms.md#slim-image)

---

### Batch inference
*From module: 10_ml_system_architecture*

- **Technical definition:** Scoring many items together on a schedule or in bulk; optimises throughput and cost via vectorized calls.
- **Plain English:** Scoring a big pile at once, nobody waiting.
- **Example / analogy:** A bus carrying many passengers.
- **Snippet:**
```python
for piece in chunks(texts, 500):
    out.extend(model.predict(piece))
```
- **Snippet explained:** One vectorized call per 500 items.
- **Related terms:** [Real-time inference](api_and_deployment_terms.md#real-time-inference), [Micro-batching](api_and_deployment_terms.md#micro-batching), [Generator](python_syntax.md#generator)

---

### Bind mount
*From module: 11_docker*

- **Technical definition:** Mounts a host directory or file into a container at a path.
- **Plain English:** Shares a folder from your computer into the container.
- **Example / analogy:** A shared network drive.
- **Snippet:**
```bash
docker run -v "$(pwd)/host-data:/data" toy-api:v1
```
- **Snippet explained:** The container sees and writes your host-data folder directly.
- **Related terms:** [Volume (named volume)](api_and_deployment_terms.md#volume-named-volume)

---

### Bounded queue
*From module: 10_ml_system_architecture*

- **Technical definition:** A FIFO queue with a maximum size that applies backpressure by blocking or rejecting producers when full.
- **Plain English:** A waiting line with a limited number of places.
- **Example / analogy:** A turnstile line that stops at capacity.
- **Snippet:**
```python
work = queue.Queue(maxsize=5)
```
- **Snippet explained:** At most 5 jobs wait.
- **Related terms:** [Load shedding](api_and_deployment_terms.md#load-shedding)

---

### Build cache (layer cache)
*From module: 11_docker*

- **Technical definition:** Docker reuses a layer when its instruction and inputs are unchanged and all previous layers were reused.
- **Plain English:** Docker remembers finished steps so rebuilds are fast.
- **Example / analogy:** Not re-baking the base of a cake when only the icing changed.
- **Snippet:**
```dockerfile
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app ./app
```
- **Snippet explained:** Dependencies first, code last: editing code does not re-run pip install.
- **Related terms:** [Layer (Docker)](api_and_deployment_terms.md#layer-docker), [Requirements file](api_and_deployment_terms.md#requirements-file)

---

### Build context
*From module: 11_docker*

- **Technical definition:** The set of files (folder path given to docker build, minus .dockerignore) sent to the Docker daemon and available to COPY.
- **Plain English:** The folder Docker is allowed to see while building.
- **Example / analogy:** The ingredients on the kitchen counter - anything not there cannot be used.
- **Snippet:**
```bash
docker build -t app .
```
- **Snippet explained:** The trailing dot is the build context: the current directory.
- **Related terms:** [Dockerfile](api_and_deployment_terms.md#dockerfile), [.dockerignore](api_and_deployment_terms.md#dockerignore)

---

### Caching
*From module: 10_ml_system_architecture*

- **Technical definition:** Storing results of expensive computations keyed by input so repeats are answered instantly; risks staleness and memory use.
- **Plain English:** Remembering answers you already computed.
- **Example / analogy:** Writing down a calculation result instead of redoing it.
- **Snippet:**
```python
@lru_cache(maxsize=1024)
def cached_predict(text): ...
```
- **Snippet explained:** Same text returns the saved result.
- **Related terms:** [Latency](llm_terms.md#latency)

---

### Callback (event handler)
*From module: 12_streamlit_gradio*

- **Technical definition:** A function registered to run when an event (click, submit, upload) occurs.
- **Plain English:** The function that runs when the user does something.
- **Example / analogy:** A doorbell wired to a chime.
- **Snippet:**
```python
btn.click(fn, inputs=[txt], outputs=[out])
```
- **Snippet explained:** On click, Gradio calls fn(txt value) and puts the return into out.
- **Related terms:** [gr.Blocks](api_and_deployment_terms.md#grblocks), [Rerun model (Streamlit)](api_and_deployment_terms.md#rerun-model-streamlit)

---

### Canary deployment
*From module: 10_ml_system_architecture*

- **Technical definition:** Releasing a new version to a small share of traffic first and promoting it only if metrics stay healthy.
- **Plain English:** Test the new model on a few users first.
- **Example / analogy:** The canary in the coal mine.
- **Snippet:**
n/a
- **Snippet explained:** n/a
- **Related terms:** [Rollback](api_and_deployment_terms.md#rollback), [Shadow deployment](api_and_deployment_terms.md#shadow-deployment)

---

### CMD vs ENTRYPOINT
*From module: 11_docker*

- **Technical definition:** ENTRYPOINT defines the fixed executable; CMD supplies default arguments (or the default command) and is replaced by arguments to docker run.
- **Plain English:** ENTRYPOINT is the program; CMD is its default options.
- **Example / analogy:** ENTRYPOINT = 'pizza oven', CMD = 'default topping'.
- **Snippet:**
```dockerfile
ENTRYPOINT ["uvicorn"]
CMD ["app.main:app", "--host", "0.0.0.0"]
```
- **Snippet explained:** docker run img app.main:app --port 9000 replaces only the CMD part.
- **Related terms:** [Exec form vs shell form](api_and_deployment_terms.md#exec-form-vs-shell-form), [RUN (Dockerfile)](api_and_deployment_terms.md#run-dockerfile)

---

### Compose file
*From module: 11_docker*

- **Technical definition:** compose.yaml describing services (image/build, ports, environment, volumes, depends_on, healthcheck) and top-level volumes/networks.
- **Plain English:** The YAML that describes your whole app stack.
- **Example / analogy:** The architecture blueprint.
- **Snippet:**
```yaml
services:
  api:
    build: ./api
    ports: ["8000:8000"]
```
- **Snippet explained:** Defines one service called api built from ./api and published on 8000.
- **Related terms:** [Docker Compose](api_and_deployment_terms.md#docker-compose), [YAML](api_and_deployment_terms.md#yaml)

---

### Concurrency
*From module: 09_api_development*

- **Technical definition:** Making progress on several requests at overlapping times, via processes, threads or async I/O.
- **Plain English:** Handling more than one thing at once (or seeming to).
- **Example / analogy:** A chef stirring one pot while another simmers.
- **Snippet:**
```python
await asyncio.sleep(1)   # lets the event loop serve others meanwhile
```
- **Snippet explained:** Async waiting does not block other requests; time.sleep would.
- **Related terms:** [Async/await](python_syntax.md#asyncawait), [Worker](api_and_deployment_terms.md#worker), GIL

---

### Configuration
*From module: 09_api_development*

- **Technical definition:** All settings that change behaviour between environments (paths, versions, limits, keys), kept out of code and read in one place.
- **Plain English:** The knobs and switches of your program.
- **Example / analogy:** A car's seat and mirror settings, separate from the car's engine.
- **Snippet:**
```python
@dataclass(frozen=True)
class Settings:
    model_path: str | None
    max_text_length: int
```
- **Snippet explained:** A frozen dataclass collects settings in one typed object.
- **Related terms:** [Environment variable](python_syntax.md#environment-variable), [.env file](python_syntax.md#env-file)

---

### Confirmation hook
*From module: 08_agent_harness*

- **Technical definition:** A function called before a risky tool runs that returns whether a human (or policy) approves it.
- **Plain English:** A 'are you sure?' step before dangerous actions.
- **Example / analogy:** Two-person approval for a payment.
- **Snippet:**
```python
def ask_human(name, args):
    return input(f"Allow {name}({args})? [y/N] ").lower() == "y"
```
- **Snippet explained:** Returns True only for an explicit yes.
- **Related terms:** [Read-only mode](api_and_deployment_terms.md#read-only-mode), [Least privilege](api_and_deployment_terms.md#least-privilege)

---

### Container
*From module: 11_docker*

- **Technical definition:** An isolated process (or group) running on the host kernel with its own filesystem, network and process namespace, started from an image.
- **Plain English:** A lightweight sealed box that runs your app with everything it needs.
- **Example / analogy:** Like a locked apartment in a shared building.
- **Snippet:**
```bash
docker run -d --name web -p 8000:8000 toy-api:v1
```
- **Snippet explained:** Starts one container from image toy-api:v1 in the background, mapping host port 8000.
- **Related terms:** [Image](api_and_deployment_terms.md#image), [Virtual machine (vs container)](api_and_deployment_terms.md#virtual-machine-vs-container), [Namespace and cgroup](api_and_deployment_terms.md#namespace-and-cgroup)

---

### Container networking
*From module: 11_docker*

- **Technical definition:** Containers attach to virtual networks; on user-defined (Compose) networks, containers resolve each other by service name via Docker DNS.
- **Plain English:** How containers talk to each other and the outside world.
- **Example / analogy:** An office intercom with names.
- **Snippet:**
```bash
docker compose exec api python -c "import socket;print(socket.gethostbyname('redis'))"
```
- **Snippet explained:** Resolves the redis service name to its container IP.
- **Related terms:** [Docker Compose](api_and_deployment_terms.md#docker-compose), [Port mapping](api_and_deployment_terms.md#port-mapping)

---

### COPY (Dockerfile)
*From module: 11_docker*

- **Technical definition:** Copies files from the build context (or another stage with --from) into the image.
- **Plain English:** Puts your files into the image.
- **Example / analogy:** Packing files into the box.
- **Snippet:**
```dockerfile
COPY app ./app
```
- **Snippet explained:** Copies the app folder into /code/app (relative to WORKDIR).
- **Related terms:** [Build context](api_and_deployment_terms.md#build-context), [Multi-stage build](api_and_deployment_terms.md#multi-stage-build), [WORKDIR](api_and_deployment_terms.md#workdir)

---

### curl
*From module: 09_api_development*

- **Technical definition:** A command-line tool for making HTTP requests; flags: -i headers, -X method, -H header, -d body, -w output format, -o output file.
- **Plain English:** Typing a web request in the terminal.
- **Example / analogy:** A hand-operated browser.
- **Snippet:**
```bash
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{"text":"great"}'
```
- **Snippet explained:** POSTs JSON; -s hides the progress meter.
- **Related terms:** [HTTP](api_and_deployment_terms.md#http), [GET vs POST](api_and_deployment_terms.md#get-vs-post)

---

### Dependency injection (FastAPI)
*From module: 09_api_development*

- **Technical definition:** FastAPI's `Depends(...)` mechanism for supplying shared resources (settings, DB sessions, auth) to route functions declaratively.
- **Plain English:** Letting the framework hand your function the tools it needs.
- **Example / analogy:** A waiter bringing you the right cutlery for each course.
- **Snippet:**
```python
from fastapi import Depends
@app.get('/x')
def x(settings=Depends(get_settings)): ...
```
- **Snippet explained:** FastAPI calls get_settings() and passes the result in.
- **Related terms:** [FastAPI](api_and_deployment_terms.md#fastapi), Config

---

### depends_on
*From module: 11_docker*

- **Technical definition:** Compose option controlling start order, optionally waiting for a dependency to be healthy.
- **Plain English:** Start B only after A.
- **Example / analogy:** Open the shop after the electricity is on.
- **Snippet:**
```yaml
depends_on:
  redis:
    condition: service_healthy
```
- **Snippet explained:** Starts the api only once redis passes its healthcheck.
- **Related terms:** [Health check (Docker)](api_and_deployment_terms.md#health-check-docker)

---

### Deployment (UI apps)
*From module: 12_streamlit_gradio*

- **Technical definition:** Hosting a UI app permanently: Docker on a server, Streamlit Community Cloud, Hugging Face Spaces, cloud VMs.
- **Plain English:** Putting the app somewhere others can reach all the time.
- **Example / analogy:** Moving from a garage prototype to a shop.
- **Snippet:**
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```
- **Snippet explained:** Bind to all interfaces and a fixed port so containers/servers can expose it.
- **Related terms:** [Share link (Gradio)](api_and_deployment_terms.md#share-link-gradio)

---

### Docker Compose
*From module: 11_docker*

- **Technical definition:** Tool and YAML format to define and run multi-container applications (services, networks, volumes).
- **Plain English:** One file and one command to run several containers.
- **Example / analogy:** A conductor's score for an orchestra.
- **Snippet:**
```bash
docker compose up --build -d
```
- **Snippet explained:** Builds images and starts all services in the background.
- **Related terms:** [Compose file](api_and_deployment_terms.md#compose-file), [Service (Compose)](api_and_deployment_terms.md#service-compose), [Container networking](api_and_deployment_terms.md#container-networking)

---

### docker exec
*From module: 11_docker*

- **Technical definition:** Runs an extra command inside a running container (often a shell).
- **Plain English:** Open a terminal inside the container.
- **Example / analogy:** Walking into the apartment.
- **Snippet:**
```bash
docker exec -it toy sh
```
- **Snippet explained:** Opens an interactive shell; then try ls, env, ps.
- **Related terms:** [Docker logs](api_and_deployment_terms.md#docker-logs)

---

### Docker logs
*From module: 11_docker*

- **Technical definition:** Stdout/stderr of the container's main process, shown by docker logs.
- **Plain English:** What your app printed.
- **Example / analogy:** The black box recorder.
- **Snippet:**
```bash
docker logs -f --tail 50 toy
```
- **Snippet explained:** Follows the last 50 lines of the container's output live.
- **Related terms:** [Exit code (container)](api_and_deployment_terms.md#exit-code-container), [docker exec](api_and_deployment_terms.md#docker-exec)

---

### Dockerfile
*From module: 11_docker*

- **Technical definition:** A text file of ordered instructions (FROM, COPY, RUN...) that docker build executes to create an image.
- **Plain English:** The step-by-step script for building your image.
- **Example / analogy:** A cooking recipe.
- **Snippet:**
```dockerfile
FROM python:3.12-slim
COPY app ./app
CMD ["python", "app/main.py"]
```
- **Snippet explained:** Start from Python, add code, define the default command.
- **Related terms:** [Image](api_and_deployment_terms.md#image), [Layer (Docker)](api_and_deployment_terms.md#layer-docker), [Build context](api_and_deployment_terms.md#build-context)

---

### Environment variable (Docker)
*From module: 11_docker*

- **Technical definition:** Key=value configuration injected into a container process via ENV, -e, --env-file or Compose environment.
- **Plain English:** Settings passed to your app without changing code.
- **Example / analogy:** Dials on a machine.
- **Snippet:**
```bash
docker run -e MODEL_NAME=v2 --env-file sample.env toy-api:v1
```
- **Snippet explained:** -e sets one variable; --env-file loads many; code reads them with os.getenv.
- **Related terms:** [Secret](api_and_deployment_terms.md#secret), [Compose file](api_and_deployment_terms.md#compose-file)

---

### Error handler
*From module: 09_api_development*

- **Technical definition:** A function registered to turn an exception or status (404, 405, any Exception) into a custom response, e.g. JSON instead of HTML, without leaking stack traces.
- **Plain English:** A safety net that decides what the user sees when something goes wrong.
- **Example / analogy:** A polite receptionist when the manager is unavailable.
- **Snippet:**
```python
@app.errorhandler(404)
def nf(e):
    return jsonify(error='route not found'), 404
```
- **Snippet explained:** Replaces Flask's HTML 404 page with JSON.
- **Related terms:** [Status code](api_and_deployment_terms.md#status-code), [HTTP 500](api_and_deployment_terms.md#http-500)

---

### Event loop
*From module: 09_api_development*

- **Technical definition:** The scheduler in async programs that runs coroutines and resumes them when their awaited I/O finishes; it runs in a single thread, so blocking calls stall everything.
- **Plain English:** The traffic controller deciding which waiting task goes next.
- **Example / analogy:** One waiter checking which tables are ready.
- **Snippet:**
```python
import asyncio
asyncio.run(main())
```
- **Snippet explained:** Creates an event loop, runs `main()` to completion, closes it.
- **Related terms:** [Concurrency](api_and_deployment_terms.md#concurrency), [ASGI](api_and_deployment_terms.md#asgi), [Async/await](python_syntax.md#asyncawait)

---

### Exec form vs shell form
*From module: 11_docker*

- **Technical definition:** Exec form is a JSON array run directly (no shell); shell form is a string run via /bin/sh -c.
- **Plain English:** JSON list = runs your program as PID 1 and receives stop signals properly.
- **Example / analogy:** Calling someone directly versus via a receptionist.
- **Snippet:**
```dockerfile
CMD ["uvicorn", "app.main:app"]   # exec form
CMD uvicorn app.main:app           # shell form
```
- **Snippet explained:** Prefer exec form so SIGTERM from docker stop reaches your process.
- **Related terms:** [CMD vs ENTRYPOINT](api_and_deployment_terms.md#cmd-vs-entrypoint)

---

### Exit code (container)
*From module: 11_docker*

- **Technical definition:** Numeric status of a stopped container's main process: 0 success, 1 app error, 137 killed (SIGKILL/OOM), 139 segfault.
- **Plain English:** A number telling you why the container stopped.
- **Example / analogy:** A report card.
- **Snippet:**
```bash
docker inspect toy --format '{{.State.ExitCode}}'
```
- **Snippet explained:** Prints the exit code of the container.
- **Related terms:** [Docker logs](api_and_deployment_terms.md#docker-logs)

---

### Exponential backoff
*From module: 08_agent_harness*

- **Technical definition:** A retry strategy that waits progressively longer between attempts (base * 2^(n-1)), usually with random jitter, and a retry cap.
- **Plain English:** Wait a bit, then longer, then longer again before retrying.
- **Example / analogy:** Knocking on a door with growing pauses.
- **Snippet:**
```python
import random, time
delay = 0.05 * 2 ** (attempt - 1) + random.uniform(0, 0.05)
time.sleep(delay)
```
- **Snippet explained:** Attempt 1 waits ~0.05s, attempt 2 ~0.10s, attempt 3 ~0.20s.
- **Related terms:** [Jitter](api_and_deployment_terms.md#jitter), [Transient error](api_and_deployment_terms.md#transient-error), [Rate limit (429)](api_and_deployment_terms.md#rate-limit-429)

---

### EXPOSE
*From module: 11_docker*

- **Technical definition:** Documents the port an app listens on; does not publish it.
- **Plain English:** A note saying 'this app uses port 8000'.
- **Example / analogy:** A sign on the door that does not unlock it.
- **Snippet:**
```dockerfile
EXPOSE 8000
```
- **Snippet explained:** Metadata only; use -p 8000:8000 to publish.
- **Related terms:** [Port mapping](api_and_deployment_terms.md#port-mapping)

---

### FastAPI
*From module: 09_api_development*

- **Technical definition:** A modern Python web framework on ASGI (Starlette + Pydantic) that uses type hints for parameter parsing, validation and automatic OpenAPI docs.
- **Plain English:** A web framework where your type hints do the checking and the docs write themselves.
- **Example / analogy:** A toolbox with built-in measuring tape and instruction manual.
- **Snippet:**
```python
from fastapi import FastAPI
app = FastAPI()
@app.get("/add")
def add(a: float, b: float):
    return {"result": a + b}
```
- **Snippet explained:** `a` and `b` are parsed and validated as floats automatically.
- **Related terms:** [ASGI](api_and_deployment_terms.md#asgi), [Pydantic](api_and_deployment_terms.md#pydantic), [Uvicorn](api_and_deployment_terms.md#uvicorn)

---

### Flask
*From module: 09_api_development*

- **Technical definition:** A lightweight Python micro-framework for web apps and APIs that speaks WSGI; provides routing and request/response helpers and leaves validation/docs to you.
- **Plain English:** A small toolkit for making web endpoints in Python.
- **Example / analogy:** A basic toolbox: few tools, you add what you need.
- **Snippet:**
```python
from flask import Flask, jsonify
app = Flask(__name__)
@app.get("/health")
def health():
    return jsonify(status="ok")
```
- **Snippet explained:** Creates an app and a health route returning JSON.
- **Related terms:** [WSGI](api_and_deployment_terms.md#wsgi), [Gunicorn](api_and_deployment_terms.md#gunicorn), [FastAPI](api_and_deployment_terms.md#fastapi)

---

### FROM
*From module: 11_docker*

- **Technical definition:** Dockerfile instruction selecting the base image (and starting a build stage).
- **Plain English:** Which image you build on top of.
- **Example / analogy:** Choosing the starting template.
- **Snippet:**
```dockerfile
FROM python:3.12-slim
```
- **Snippet explained:** Uses Debian-based Python 3.12 with minimal extras.
- **Related terms:** [Base image](api_and_deployment_terms.md#base-image), [Multi-stage build](api_and_deployment_terms.md#multi-stage-build)

---

### GET vs POST
*From module: 09_api_development*

- **Technical definition:** GET retrieves data and has no body (parameters live in the URL, results may be cached); POST sends a body to create or process something and is not cached or idempotent by default.
- **Plain English:** GET asks, POST sends something along.
- **Example / analogy:** GET is reading a notice board; POST is handing a form to the clerk.
- **Snippet:**
```bash
curl 'http://127.0.0.1:8000/add?a=1&b=2'
curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{"text":"hi"}'
```
- **Snippet explained:** First sends data in the URL; second sends a JSON body, which suits long text and keeps it out of URLs/logs.
- **Related terms:** [HTTP method](api_and_deployment_terms.md#http-method), [Idempotent](api_and_deployment_terms.md#idempotent), [Request body](api_and_deployment_terms.md#request-body)

---

### gr.Blocks
*From module: 12_streamlit_gradio*

- **Technical definition:** Low-level Gradio API for custom layouts and multiple event listeners.
- **Plain English:** Build your own layout.
- **Example / analogy:** Lego instead of a ready-made toy.
- **Snippet:**
```python
with gr.Blocks() as demo:
    t = gr.Textbox()
    b = gr.Button('Go')
    b.click(f, t, t)
```
- **Snippet explained:** Components are placed in order; click wires f to them.
- **Related terms:** [gr.Interface](api_and_deployment_terms.md#grinterface), [Callback (event handler)](api_and_deployment_terms.md#callback-event-handler)

---

### gr.Error
*From module: 12_streamlit_gradio*

- **Technical definition:** Exception that Gradio displays as a toast message to the user.
- **Plain English:** A polite error message.
- **Example / analogy:** A pop-up notice.
- **Snippet:**
```python
raise gr.Error('Please enter text')
```
- **Snippet explained:** Stops the function and shows the message in the UI.
- **Related terms:** [st.error](api_and_deployment_terms.md#sterror)

---

### gr.Interface
*From module: 12_streamlit_gradio*

- **Technical definition:** High-level Gradio class that builds a UI around one function given inputs and outputs.
- **Plain English:** The simplest Gradio app.
- **Example / analogy:** A form attached to a function.
- **Snippet:**
```python
gr.Interface(fn=f, inputs=gr.Textbox(), outputs=gr.Label())
```
- **Snippet explained:** One input component per parameter, one output per return value.
- **Related terms:** [Gradio](api_and_deployment_terms.md#gradio), [gr.Blocks](api_and_deployment_terms.md#grblocks), [Callback (event handler)](api_and_deployment_terms.md#callback-event-handler)

---

### gr.Label
*From module: 12_streamlit_gradio*

- **Technical definition:** Gradio output component for classification results given a {class: probability} dict.
- **Plain English:** Displays class probabilities as bars.
- **Example / analogy:** A scoreboard.
- **Snippet:**
```python
return {'positive': 0.8, 'negative': 0.2}
```
- **Snippet explained:** The dict shape is required by gr.Label.
- **Related terms:** [gr.Interface](api_and_deployment_terms.md#grinterface)

---

### gr.Progress
*From module: 12_streamlit_gradio*

- **Technical definition:** Helper injected into a Gradio function for updating a progress bar.
- **Plain English:** Progress bar controller.
- **Example / analogy:** A loading bar.
- **Snippet:**
```python
def f(x, progress=gr.Progress()):
    progress(0.5, desc='half')
```
- **Snippet explained:** Gradio detects the default argument and passes a tracker.
- **Related terms:** [Loading state](api_and_deployment_terms.md#loading-state)

---

### gr.State
*From module: 12_streamlit_gradio*

- **Technical definition:** Gradio component storing per-session Python values not shown in the UI.
- **Plain English:** Hidden memory for a session.
- **Example / analogy:** A pocket in the UI.
- **Snippet:**
```python
history = gr.State([])
```
- **Snippet explained:** Pass it as input and output to read and update.
- **Related terms:** [Session state](api_and_deployment_terms.md#session-state)

---

### Gradio
*From module: 12_streamlit_gradio*

- **Technical definition:** Open-source Python library that builds web UIs around functions by declaring input and output components.
- **Plain English:** Wrap a function with a web form.
- **Example / analogy:** A remote control for a function.
- **Snippet:**
```python
import gradio as gr
gr.Interface(lambda t: t[::-1], 'text', 'text').launch()
```
- **Snippet explained:** Builds a text-reversing app at http://127.0.0.1:7860.
- **Related terms:** [Callback (event handler)](api_and_deployment_terms.md#callback-event-handler), [gr.Interface](api_and_deployment_terms.md#grinterface)

---

### Gunicorn
*From module: 09_api_development*

- **Technical definition:** A production-grade pre-fork WSGI HTTP server that manages multiple worker processes (and can run Uvicorn workers for ASGI apps).
- **Plain English:** The program that actually listens on the port and runs several copies of your Flask app.
- **Example / analogy:** A restaurant manager who hires several waiters.
- **Snippet:**
```bash
gunicorn flask_app:app --workers 2 --bind 127.0.0.1:5001
```
- **Snippet explained:** Starts 2 worker processes serving the `app` object from flask_app.py.
- **Related terms:** [Worker](api_and_deployment_terms.md#worker), [WSGI](api_and_deployment_terms.md#wsgi), [Uvicorn](api_and_deployment_terms.md#uvicorn)

---

### Header
*From module: 09_api_development*

- **Technical definition:** A name/value metadata line on a request or response (Content-Type, Content-Length, Authorization...).
- **Plain English:** Notes written on the envelope, not inside the letter.
- **Example / analogy:** 'Fragile' sticker on a parcel.
- **Snippet:**
```bash
curl -H 'Content-Type: application/json' ...
```
- **Snippet explained:** Tells the server the body is JSON; Flask's get_json depends on it.
- **Related terms:** [HTTP](api_and_deployment_terms.md#http), [Request body](api_and_deployment_terms.md#request-body)

---

### Health check
*From module: 09_api_development*

- **Technical definition:** A cheap endpoint (usually GET /health) reporting that the process is alive (liveness) and optionally that the model is loaded (readiness); used by load balancers and orchestrators.
- **Plain English:** A 'are you okay?' address orchestrators ping.
- **Example / analogy:** A pulse check.
- **Snippet:**
```python
@app.get("/health")
def health():
    return {"status": "ok"}
```
- **Snippet explained:** Returns 200 when the app can answer.
- **Related terms:** [Smoke test](api_and_deployment_terms.md#smoke-test), Docker health check

---

### Health check (Docker)
*From module: 11_docker*

- **Technical definition:** A command run periodically inside a container; exit 0 = healthy, 1 = unhealthy.
- **Plain English:** A probe asking 'are you really working?'
- **Example / analogy:** A doctor's pulse check.
- **Snippet:**
```dockerfile
HEALTHCHECK --interval=10s CMD python -c "import urllib.request as u;u.urlopen('http://localhost:8000/health')" || exit 1
```
- **Snippet explained:** Calls /health every 10 s; failure marks the container unhealthy.
- **Related terms:** [depends_on](api_and_deployment_terms.md#depends_on), [Liveness vs readiness](api_and_deployment_terms.md#liveness-vs-readiness)

---

### HTTP
*From module: 09_api_development*

- **Technical definition:** HyperText Transfer Protocol: a request/response protocol where a client sends a method, path, headers and optional body and the server replies with a status code, headers and body.
- **Plain English:** The language browsers and programs use to ask a server for things and get answers.
- **Example / analogy:** Ordering at a restaurant: you ask (request), the kitchen answers (response) with a status like 'ready' or 'sold out'.
- **Snippet:**
```bash
curl -i http://127.0.0.1:8000/health
```
- **Snippet explained:** `-i` prints the status line and response headers before the body.
- **Related terms:** [HTTP method](api_and_deployment_terms.md#http-method), [Status code](api_and_deployment_terms.md#status-code), [REST](api_and_deployment_terms.md#rest)

---

### HTTP 500
*From module: 09_api_development*

- **Technical definition:** Internal Server Error: an unexpected failure in the server code; the server, not the client, must be fixed.
- **Plain English:** 'Something broke on our side.'
- **Example / analogy:** The kitchen caught fire, regardless of what you ordered.
- **Snippet:**
```python
@app.errorhandler(Exception)
def unexpected(e):
    app.logger.exception("unhandled")
    return jsonify(error="internal server error"), 500
```
- **Snippet explained:** Logs the traceback privately and returns a safe generic message to the client.
- **Related terms:** [Status code](api_and_deployment_terms.md#status-code), [Error handler](api_and_deployment_terms.md#error-handler), [Logging](python_syntax.md#logging)

---

### HTTP method
*From module: 09_api_development*

- **Technical definition:** The verb of a request (GET, POST, PUT, PATCH, DELETE) describing the intended action on the resource at the path.
- **Plain English:** What you want to DO: read, create, replace, change, delete.
- **Example / analogy:** GET = look at the menu, POST = place an order, DELETE = cancel it.
- **Snippet:**
```python
@app.get("/items")
@app.post("/items")
```
- **Snippet explained:** Same path, two routes: the method decides which function runs.
- **Related terms:** [GET vs POST](api_and_deployment_terms.md#get-vs-post), [Route](api_and_deployment_terms.md#route), [REST](api_and_deployment_terms.md#rest)

---

### Idempotent
*From module: 09_api_development*

- **Technical definition:** An operation that has the same effect whether done once or many times (GET, PUT, DELETE are; POST usually isn't). Safe-to-retry operations must be idempotent.
- **Plain English:** Doing it twice is the same as doing it once.
- **Example / analogy:** Pressing the lift's 'up' button repeatedly.
- **Snippet:**
```python
# retry only if repeating is harmless
for attempt in range(3):
    ...
```
- **Snippet explained:** Idempotency decides whether automatic retries are safe.
- **Related terms:** [Retry with backoff](python_syntax.md#retry-with-backoff), [HTTP method](api_and_deployment_terms.md#http-method)

---

### Image
*From module: 11_docker*

- **Technical definition:** A read-only, layered template containing a filesystem and metadata (default command, env, ports) used to create containers.
- **Plain English:** The recipe-plus-ingredients package your container is created from.
- **Example / analogy:** A class vs object: image = class, container = instance.
- **Snippet:**
```bash
docker build -t toy-api:v1 .
```
- **Snippet explained:** Builds an image named toy-api with tag v1 from the Dockerfile in the current folder.
- **Related terms:** [Container](api_and_deployment_terms.md#container), [Dockerfile](api_and_deployment_terms.md#dockerfile), [Layer (Docker)](api_and_deployment_terms.md#layer-docker)

---

### Jitter
*From module: 08_agent_harness*

- **Technical definition:** A small random amount added to retry delays so many clients do not retry at the same moment.
- **Plain English:** Randomness in the wait time.
- **Example / analogy:** People leaving a stadium at slightly different times.
- **Snippet:**
```python
random.uniform(0, 0.05)
```
- **Snippet explained:** Returns a random float between 0 and 0.05.
- **Related terms:** [Exponential backoff](api_and_deployment_terms.md#exponential-backoff)

---

### JSON (JavaScript Object Notation)
*From module: 07_prompt_engineering*

- **Technical definition:** A text format for nested key-value data using double-quoted keys/strings, arrays, numbers, booleans and null.
- **Plain English:** A universal way to write dicts and lists as text.
- **Example / analogy:** A labelled box-in-box packing list.
- **Snippet:**
```python
import json
print(json.dumps({"a": [1, 2]}))
print(json.loads('{"a": [1, 2]}'))
```
- **Snippet explained:** `dumps` writes JSON text; `loads` reads it (single quotes are invalid JSON).
- **Related terms:** [Structured output](llm_terms.md#structured-output), JSONL

---

### JSON Schema
*From module: 08_agent_harness*

- **Technical definition:** A standard JSON vocabulary (type, properties, required, enum) describing the structure of JSON data; used to describe tool inputs.
- **Plain English:** A machine-readable description of what the JSON must look like.
- **Example / analogy:** A blueprint for a data form.
- **Snippet:**
```json
{"type": "object", "properties": {"a": {"type": "number"}}, "required": ["a"]}
```
- **Snippet explained:** Declares an object that must contain numeric `a`.
- **Related terms:** [Native tool calling](llm_terms.md#native-tool-calling), [Pydantic model](python_syntax.md#pydantic-model)

---

### JSONL (JSON Lines)
*From module: 08_agent_harness*

- **Technical definition:** A text format with one JSON object per line; append-friendly and easy to grep or stream.
- **Plain English:** A log file where each line is a small JSON record.
- **Example / analogy:** A ledger with one entry per line.
- **Snippet:**
```python
with open("trace.jsonl", "a") as f:
    f.write(json.dumps({"event": "start"}) + "\n")
```
- **Snippet explained:** Opens in append mode and writes one object plus a newline.
- **Related terms:** [Trace (agent trace)](llm_terms.md#trace-agent-trace), [JSON (JavaScript Object Notation)](api_and_deployment_terms.md#json-javascript-object-notation)

---

### Layer (Docker)
*From module: 11_docker*

- **Technical definition:** A read-only filesystem diff created by a Dockerfile instruction; layers are stacked and cached.
- **Plain English:** One step of the build saved so it can be reused.
- **Example / analogy:** Pages in a book; changing page 3 means reprinting pages 3 onward.
- **Snippet:**
```bash
docker history toy-api:v1
```
- **Snippet explained:** Lists the layers of an image with the instruction and size of each.
- **Related terms:** [Build cache (layer cache)](api_and_deployment_terms.md#build-cache-layer-cache), [Dockerfile](api_and_deployment_terms.md#dockerfile), [Multi-stage build](api_and_deployment_terms.md#multi-stage-build)

---

### Least privilege
*From module: 08_agent_harness*

- **Technical definition:** Giving a component only the permissions and tools it needs for its task.
- **Plain English:** Don't hand over more power than needed.
- **Example / analogy:** A cleaner gets a room key, not the safe combination.
- **Snippet:**
```python
agent = Agent(tools=["calculator"], mode="read_only")
```
- **Snippet explained:** Only one read-only tool is exposed.
- **Related terms:** [Read-only mode](api_and_deployment_terms.md#read-only-mode), [Tool safety](llm_terms.md#tool-safety)

---

### Lifespan (startup/shutdown)
*From module: 09_api_development*

- **Technical definition:** FastAPI's context manager for code that runs once per worker process before serving (e.g. loading a model) and after shutdown.
- **Plain English:** A place to load heavy things once when the server starts.
- **Example / analogy:** Opening the shop and switching on the lights before the first customer.
- **Snippet:**
```python
@asynccontextmanager
async def lifespan(app):
    app.state.model = load_model()
    yield
```
- **Snippet explained:** Loads the model before the first request; code after `yield` runs at shutdown.
- **Related terms:** [FastAPI](api_and_deployment_terms.md#fastapi), Model loading, [Worker](api_and_deployment_terms.md#worker)

---

### Little's Law
*From module: 10_ml_system_architecture*

- **Technical definition:** In a stable system, average items in the system = arrival rate x average time in system (L = lambda x W); used for capacity planning.
- **Plain English:** Requests in flight = requests per second x seconds each takes.
- **Example / analogy:** A cafe: 10 customers/min x 3 min each = 30 people inside.
- **Snippet:**
```python
workers_needed = arrival_rate * latency_seconds
```
- **Snippet explained:** 200 req/s at 0.2 s needs about 40 concurrent workers.
- **Related terms:** [Throughput](llm_terms.md#throughput), [Latency](llm_terms.md#latency), [Autoscaling](api_and_deployment_terms.md#autoscaling)

---

### Liveness vs readiness
*From module: 11_docker*

- **Technical definition:** Liveness: process is alive; readiness: it can serve real work (dependencies ready).
- **Plain English:** Alive means running; ready means able to help.
- **Example / analogy:** A restaurant open (alive) vs kitchen ready.
- **Snippet:**
```python
@app.get('/ready')
def ready(): ...  # checks redis.ping()
```
- **Snippet explained:** Returns 503 until dependencies respond.
- **Related terms:** [Health check (Docker)](api_and_deployment_terms.md#health-check-docker)

---

### Load balancer
*From module: 10_ml_system_architecture*

- **Technical definition:** A component that distributes incoming requests across several service instances and routes around unhealthy ones using health checks.
- **Plain English:** A traffic cop spreading requests over servers.
- **Example / analogy:** A host seating guests at different tables.
- **Snippet:**
n/a
- **Snippet explained:** n/a
- **Related terms:** [Autoscaling](api_and_deployment_terms.md#autoscaling), [Health check](api_and_deployment_terms.md#health-check)

---

### Load shedding
*From module: 10_ml_system_architecture*

- **Technical definition:** Deliberately rejecting excess requests (429/503) when overloaded, using a bounded queue, so the service stays healthy for accepted requests.
- **Plain English:** Turning away extra work fast instead of collapsing.
- **Example / analogy:** A restaurant closing the door when full.
- **Snippet:**
```python
try:
    q.put_nowait(job)
except queue.Full:
    return 503
```
- **Snippet explained:** Rejects immediately when the queue is full.
- **Related terms:** [Bounded queue](api_and_deployment_terms.md#bounded-queue), [Timeout](python_syntax.md#timeout)

---

### Load testing
*From module: 10_ml_system_architecture*

- **Technical definition:** Sending controlled, high-volume traffic to a service to measure throughput, latency percentiles and errors and find limits.
- **Plain English:** Stress-testing the service with many fake users.
- **Example / analogy:** Walking a crowd across a new bridge.
- **Snippet:**
```python
threads = [threading.Thread(target=client_loop, args=(...)) for _ in range(clients)]
```
- **Snippet explained:** Many simulated clients hit the endpoint concurrently.
- **Related terms:** [Throughput](llm_terms.md#throughput), [Tail latency](api_and_deployment_terms.md#tail-latency)

---

### Loading state
*From module: 12_streamlit_gradio*

- **Technical definition:** UI feedback shown while an operation is in progress (spinner, progress bar, disabled button).
- **Plain English:** Telling the user to wait.
- **Example / analogy:** A 'please hold' message.
- **Snippet:**
```python
progress(0.5, desc='Calling API...')
```
- **Snippet explained:** In Gradio, gr.Progress updates a progress bar from inside the function.
- **Related terms:** [st.spinner](api_and_deployment_terms.md#stspinner), [gr.Progress](api_and_deployment_terms.md#grprogress)

---

### Micro-batching
*From module: 10_ml_system_architecture*

- **Technical definition:** Collecting requests for a few milliseconds and processing them as one batch to gain throughput with small added latency (used by LLM inference servers).
- **Plain English:** Waiting a moment so several requests share one model call.
- **Example / analogy:** An elevator that waits briefly for more riders.
- **Snippet:**
n/a
- **Snippet explained:** n/a
- **Related terms:** [Batch inference](api_and_deployment_terms.md#batch-inference), [Throughput](llm_terms.md#throughput)

---

### Model registry
*From module: 10_ml_system_architecture*

- **Technical definition:** A store of versioned model artifacts with metadata and stage labels (aliases), supporting promotion and rollback; e.g. MLflow, SageMaker, Hugging Face Hub.
- **Plain English:** A library of model versions with labels like 'production'.
- **Example / analogy:** A document archive with 'current' stamped on one version.
- **Snippet:**
```python
reg.promote('v0002', 'production')
model, meta = reg.load('production')
```
- **Snippet explained:** Moves the label then loads whichever version it points to.
- **Related terms:** [Model versioning](api_and_deployment_terms.md#model-versioning), Alias, [Model artifact](ml_terms.md#model-artifact)

---

### Model serving
*From module: 10_ml_system_architecture*

- **Technical definition:** Exposing a trained model so other programs can request predictions, typically via an HTTP API or a dedicated inference server.
- **Plain English:** Making the model available as a service.
- **Example / analogy:** A restaurant kitchen open for orders.
- **Snippet:**
```python
@app.post('/predict')
def predict(req): ...
```
- **Snippet explained:** A route that calls the model.
- **Related terms:** [Inference pipeline](ml_terms.md#inference-pipeline), [Batch inference](api_and_deployment_terms.md#batch-inference), [Real-time inference](api_and_deployment_terms.md#real-time-inference)

---

### Model versioning
*From module: 10_ml_system_architecture*

- **Technical definition:** Assigning immutable identifiers to trained models together with their data fingerprint, parameters, metrics and library versions for reproducibility and rollback.
- **Plain English:** Numbering each trained model and recording how it was made.
- **Example / analogy:** Software releases: v1.0, v1.1.
- **Snippet:**
```python
version = f'v{n:04d}'
```
- **Snippet explained:** Zero-padded version names sort correctly.
- **Related terms:** [Model registry](api_and_deployment_terms.md#model-registry), Alias

---

### Multi-stage build
*From module: 11_docker*

- **Technical definition:** A Dockerfile with several FROM stages; later stages copy only needed artefacts from earlier ones.
- **Plain English:** Build in a big workshop, ship only the finished product.
- **Example / analogy:** Cook in a big kitchen, deliver only the meal.
- **Snippet:**
```dockerfile
FROM python:3.12-slim AS builder
...
FROM python:3.12-slim
COPY --from=builder /install /usr/local
```
- **Snippet explained:** The final image contains only copied packages, not build tools.
- **Related terms:** [Slim image](api_and_deployment_terms.md#slim-image), [Layer (Docker)](api_and_deployment_terms.md#layer-docker)

---

### Namespace and cgroup
*From module: 11_docker*

- **Technical definition:** Linux kernel features: namespaces limit what a process can see (PIDs, network, mounts); cgroups limit resources (CPU, memory).
- **Plain English:** The kernel tricks that make a container feel isolated and capped.
- **Example / analogy:** Namespaces = walls; cgroups = utility meters.
- **Snippet:**
```bash
docker run --memory=256m --cpus=0.5 toy-api:v1
```
- **Snippet explained:** Limits the container to 256 MB RAM and half a CPU (enforced via cgroups).
- **Related terms:** [Container](api_and_deployment_terms.md#container)

---

### Non-root user (container)
*From module: 11_docker*

- **Technical definition:** Running the container process as an unprivileged user via USER.
- **Plain English:** The app does not run as administrator.
- **Example / analogy:** A guest account instead of admin.
- **Snippet:**
```dockerfile
RUN useradd --uid 10001 appuser
USER appuser
```
- **Snippet explained:** Limits damage if the app is compromised.
- **Related terms:** [Secret](api_and_deployment_terms.md#secret)

---

### Object storage
*From module: 10_ml_system_architecture*

- **Technical definition:** A service storing blobs (files) by key at large scale cheaply, such as S3 or GCS; the usual home for datasets and model artifacts.
- **Plain English:** A giant cloud file cabinet.
- **Example / analogy:** A warehouse with labelled boxes.
- **Snippet:**
n/a
- **Snippet explained:** n/a
- **Related terms:** [Model registry](api_and_deployment_terms.md#model-registry), [Model artifact](ml_terms.md#model-artifact)

---

### OpenAPI / Swagger docs
*From module: 09_api_development*

- **Technical definition:** OpenAPI is a machine-readable API description; FastAPI generates it at /openapi.json and renders interactive Swagger UI at /docs from your code.
- **Plain English:** Automatic, clickable documentation for your API.
- **Example / analogy:** A menu that writes itself and lets you taste-test each dish.
- **Snippet:**
```bash
curl -s localhost:8000/openapi.json | head -c 200
```
- **Snippet explained:** Prints the start of the generated machine-readable spec.
- **Related terms:** [FastAPI](api_and_deployment_terms.md#fastapi), [Pydantic](api_and_deployment_terms.md#pydantic)

---

### Parameterized query
*From module: 10_ml_system_architecture*

- **Technical definition:** A SQL statement with placeholders (?) whose values are passed separately so the database treats them strictly as data.
- **Plain English:** A safe fill-in-the-blank SQL command.
- **Example / analogy:** A form with boxes: whatever you write stays inside the box.
- **Snippet:**
```python
conn.execute('INSERT INTO t (a) VALUES (?)', (value,))
```
- **Snippet explained:** Value bound to the placeholder.
- **Related terms:** [SQL injection](api_and_deployment_terms.md#sql-injection), [SQLite](api_and_deployment_terms.md#sqlite)

---

### Path parameter
*From module: 09_api_development*

- **Technical definition:** A variable part of the URL path, e.g. /items/7, captured and passed to the handler.
- **Plain English:** A blank in the address that you fill in.
- **Example / analogy:** Room number in a hotel address.
- **Snippet:**
```python
@app.get("/items/{item_id}")
def get_item(item_id: int): ...
```
- **Snippet explained:** `item_id` receives 7 for /items/7 and FastAPI converts it to int.
- **Related terms:** [Query parameter](api_and_deployment_terms.md#query-parameter), [Route](api_and_deployment_terms.md#route)

---

### Path sandboxing
*From module: 08_agent_harness*

- **Technical definition:** Restricting file access to one directory by resolving the requested path and verifying it is inside that directory.
- **Plain English:** Only letting the agent open files in one folder.
- **Example / analogy:** A playpen with walls.
- **Snippet:**
```python
p = (SANDBOX / user_path).resolve()
if not p.is_relative_to(SANDBOX):
    raise PermissionError
```
- **Snippet explained:** `resolve()` removes `..`; `is_relative_to` checks containment.
- **Related terms:** [Path traversal](api_and_deployment_terms.md#path-traversal), [Least privilege](api_and_deployment_terms.md#least-privilege)

---

### Path traversal
*From module: 08_agent_harness*

- **Technical definition:** An attack using `..` segments or absolute paths to reach files outside an intended directory.
- **Plain English:** Sneaking out of the allowed folder with ../..
- **Example / analogy:** Using a back staircase to reach a restricted floor.
- **Snippet:**
```python
Path("sandbox") / "../../.env"
```
- **Snippet explained:** Without resolving and checking, this reaches a parent directory file.
- **Related terms:** [Path sandboxing](api_and_deployment_terms.md#path-sandboxing)

---

### Permanent error
*From module: 08_agent_harness*

- **Technical definition:** A failure that retrying cannot fix (e.g. 400 bad request, 401 unauthorised, 404 not found, bug in code).
- **Plain English:** An error that will keep happening.
- **Example / analogy:** Dialling a number that does not exist.
- **Snippet:**
```python
except ValueError:
    raise  # do not retry bugs
```
- **Snippet explained:** Non-retryable errors propagate immediately.
- **Related terms:** [Transient error](api_and_deployment_terms.md#transient-error), [HTTP status code](python_syntax.md#http-status-code)

---

### Port
*From module: 09_api_development*

- **Technical definition:** A numeric endpoint (0-65535) on a host where a server listens; two programs cannot listen on the same port.
- **Plain English:** A numbered door on the computer.
- **Example / analogy:** Apartment numbers in one building address.
- **Snippet:**
```bash
lsof -i :8000
```
- **Snippet explained:** Shows which process is using port 8000 ('Address already in use' fix).
- **Related terms:** Host, Docker port mapping

---

### Port mapping
*From module: 11_docker*

- **Technical definition:** Publishing a container port on a host port with -p HOST:CONTAINER.
- **Plain English:** Connects a door on your laptop to a door inside the container.
- **Example / analogy:** Phone extension forwarding.
- **Snippet:**
```bash
docker run -p 9000:8000 toy-api:v1
```
- **Snippet explained:** localhost:9000 on the host reaches port 8000 inside.
- **Related terms:** [EXPOSE](api_and_deployment_terms.md#expose), [0.0.0.0 vs 127.0.0.1 binding](api_and_deployment_terms.md#0000-vs-127001-binding)

---

### Prediction log
*From module: 10_ml_system_architecture*

- **Technical definition:** A stored record of each prediction (time, model version, input or its reference, output, confidence, latency) used for debugging, monitoring and future training data.
- **Plain English:** A diary of every answer the model gave.
- **Example / analogy:** A call-centre record of each call.
- **Snippet:**
```python
log.record('v0001', text, label, conf, ms)
```
- **Snippet explained:** Inserts one row per prediction.
- **Related terms:** [SQLite](api_and_deployment_terms.md#sqlite), [Model monitoring](ml_terms.md#model-monitoring)

---

### Pydantic
*From module: 09_api_development*

- **Technical definition:** A Python library that validates and converts data using type annotations on a BaseModel class, raising ValidationError with all problems.
- **Plain English:** A strict bouncer that checks every field of incoming data matches the shape you declared.
- **Example / analogy:** A form with rules: age must be a number 0-120.
- **Snippet:**
```python
from pydantic import BaseModel, Field
class Req(BaseModel):
    text: str = Field(min_length=1)
Req(text="hi")
```
- **Snippet explained:** Declares a required non-empty string; invalid data raises ValidationError.
- **Related terms:** [Validation](api_and_deployment_terms.md#validation), [FastAPI](api_and_deployment_terms.md#fastapi), [422 Unprocessable Content](api_and_deployment_terms.md#422-unprocessable-content)

---

### Query parameter
*From module: 09_api_development*

- **Technical definition:** Key=value pairs after `?` in a URL; always strings until converted.
- **Plain English:** Optional extra settings tacked onto the address.
- **Example / analogy:** Filters on a shopping site: ?color=red&size=M.
- **Snippet:**
```python
@app.get("/add")
def add(a: float, b: float):
    return {"result": a + b}
```
- **Snippet explained:** In FastAPI, non-path typed arguments become query params: /add?a=1&b=2.
- **Related terms:** [Path parameter](api_and_deployment_terms.md#path-parameter), [Request body](api_and_deployment_terms.md#request-body)

---

### Rate limit (429)
*From module: 08_agent_harness*

- **Technical definition:** An API response (HTTP 429) telling the client it sent too many requests; the client should wait and retry later.
- **Plain English:** 'Slow down, you are asking too fast.'
- **Example / analogy:** A turnstile that lets in limited people per minute.
- **Snippet:**
```python
if response.status_code == 429:
    time.sleep(int(response.headers.get("retry-after", 1)))
```
- **Snippet explained:** Honour the server's `retry-after` header if present.
- **Related terms:** [Exponential backoff](api_and_deployment_terms.md#exponential-backoff), [Transient error](api_and_deployment_terms.md#transient-error)

---

### Read-only mode
*From module: 08_agent_harness*

- **Technical definition:** A harness setting that refuses tools which modify state (write, delete, send).
- **Plain English:** The agent may look but not touch.
- **Example / analogy:** A museum rope.
- **Snippet:**
```python
if mode == "read_only" and not spec["read_only"]:
    return "ERROR: permission denied"
```
- **Snippet explained:** Checks the tool's own `read_only` flag.
- **Related terms:** [Least privilege](api_and_deployment_terms.md#least-privilege), [Confirmation hook](api_and_deployment_terms.md#confirmation-hook)

---

### Real-time inference
*From module: 10_ml_system_architecture*

- **Technical definition:** Producing a prediction per request while the caller waits; optimises latency.
- **Plain English:** Answering immediately, one at a time.
- **Example / analogy:** A taxi on demand.
- **Snippet:**
```python
[model.predict([t])[0] for t in texts]   # one call per item
```
- **Snippet explained:** One model call per request.
- **Related terms:** [Batch inference](api_and_deployment_terms.md#batch-inference), [Latency](llm_terms.md#latency)

---

### Request body
*From module: 09_api_development*

- **Technical definition:** The payload of a request (typically JSON) sent after the headers, used with POST/PUT/PATCH.
- **Plain English:** The actual data you send along with your request.
- **Example / analogy:** The contents of the envelope.
- **Snippet:**
```python
data = request.get_json(silent=True)
```
- **Snippet explained:** Flask parses the JSON body; `silent=True` returns None instead of raising on bad JSON.
- **Related terms:** [JSON](python_syntax.md#json), [GET vs POST](api_and_deployment_terms.md#get-vs-post), [Pydantic](api_and_deployment_terms.md#pydantic)

---

### Request ID
*From module: 10_ml_system_architecture*

- **Technical definition:** A unique identifier attached to a request and all its log lines so one request can be traced across logs and services.
- **Plain English:** A tracking number for a request.
- **Example / analogy:** A parcel tracking number.
- **Snippet:**
```python
request_id = uuid.uuid4().hex[:8]
```
- **Snippet explained:** Short random id.
- **Related terms:** [Structured logging](api_and_deployment_terms.md#structured-logging)

---

### Requirements file
*From module: 11_docker*

- **Technical definition:** requirements.txt: list of Python dependencies, ideally pinned (name==version), installed with pip -r.
- **Plain English:** The shopping list of Python packages.
- **Example / analogy:** A bill of materials.
- **Snippet:**
```text
fastapi==0.146.0
uvicorn[standard]==0.54.0
```
- **Snippet explained:** Pinned versions give reproducible builds.
- **Related terms:** [Build cache (layer cache)](api_and_deployment_terms.md#build-cache-layer-cache)

---

### Rerun model (Streamlit)
*From module: 12_streamlit_gradio*

- **Technical definition:** Streamlit re-executes the whole script top to bottom whenever a widget changes or a button is clicked.
- **Plain English:** The page is re-drawn by running your file again.
- **Example / analogy:** Re-reading a recipe from the top every time.
- **Snippet:**
```python
count = 0
count += 1   # always 1 after every rerun
```
- **Snippet explained:** Plain variables reset on every rerun; use session_state to remember.
- **Related terms:** [Session state](api_and_deployment_terms.md#session-state), [Streamlit](api_and_deployment_terms.md#streamlit)

---

### REST
*From module: 09_api_development*

- **Technical definition:** An API style where URLs identify resources (nouns), HTTP methods express actions, and responses use standard status codes and usually JSON.
- **Plain English:** A tidy convention for naming web endpoints so they're predictable.
- **Example / analogy:** A library catalogue: /books/7 is a book, GET reads it, DELETE removes it.
- **Snippet:**
```python
@app.get("/items/{item_id}")
@app.post("/items")
```
- **Snippet explained:** Resource path + method pairs follow REST naming.
- **Related terms:** [HTTP method](api_and_deployment_terms.md#http-method), [Route](api_and_deployment_terms.md#route), [JSON](python_syntax.md#json)

---

### Rollback
*From module: 10_ml_system_architecture*

- **Technical definition:** Reverting to a previously working model/version quickly after a bad release, ideally by moving an alias.
- **Plain English:** Going back to the last good version.
- **Example / analogy:** Undo button for a release.
- **Snippet:**
```python
reg.promote('v0001', 'production')
```
- **Snippet explained:** Points production back to v0001.
- **Related terms:** [Alias (model registry)](api_and_deployment_terms.md#alias-model-registry), [Canary deployment](api_and_deployment_terms.md#canary-deployment)

---

### Route
*From module: 09_api_development*

- **Technical definition:** A mapping from (HTTP method, URL path pattern) to a handler function.
- **Plain English:** A rule saying 'when someone asks for THIS address with THIS verb, run THAT function'.
- **Example / analogy:** A phone switchboard: dial extension 12, reach the person at desk 12.
- **Snippet:**
```python
@app.get("/health")
def health():
    return {"status": "ok"}
```
- **Snippet explained:** Registers `health` as the handler for GET /health.
- **Related terms:** [Decorator](python_syntax.md#decorator), [HTTP method](api_and_deployment_terms.md#http-method), [Path parameter](api_and_deployment_terms.md#path-parameter)

---

### RUN (Dockerfile)
*From module: 11_docker*

- **Technical definition:** Executes a command at build time and commits the result as a new layer.
- **Plain English:** A command that runs while building the image.
- **Example / analogy:** Assembling furniture before shipping.
- **Snippet:**
```dockerfile
RUN pip install --no-cache-dir -r requirements.txt
```
- **Snippet explained:** Installs dependencies; --no-cache-dir avoids storing pip's cache in the layer.
- **Related terms:** [CMD vs ENTRYPOINT](api_and_deployment_terms.md#cmd-vs-entrypoint), [Layer (Docker)](api_and_deployment_terms.md#layer-docker)

---

### Secret
*From module: 11_docker*

- **Technical definition:** Sensitive value (API key, password) that must not be baked into images or committed.
- **Plain English:** Passwords and keys you must keep out of the image.
- **Example / analogy:** A house key you never leave under the doormat.
- **Snippet:**
```bash
docker run -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" app
```
- **Snippet explained:** Injected at run time; not stored in any layer.
- **Related terms:** [Environment variable (Docker)](api_and_deployment_terms.md#environment-variable-docker)

---

### Service (Compose)
*From module: 11_docker*

- **Technical definition:** A named container definition in a Compose file; its name is its DNS hostname.
- **Plain English:** One part of your app, such as api, ui or redis.
- **Example / analogy:** A department in a company.
- **Snippet:**
```yaml
ui:
  environment:
    API_URL: http://api:8000
```
- **Snippet explained:** The ui service reaches the api service by name and container port.
- **Related terms:** [Compose file](api_and_deployment_terms.md#compose-file), [Container networking](api_and_deployment_terms.md#container-networking)

---

### Session state
*From module: 12_streamlit_gradio*

- **Technical definition:** Per-session dict-like storage (st.session_state / gr.State) that survives reruns and interactions for one user session.
- **Plain English:** Memory for one visitor's tab.
- **Example / analogy:** A name tag that stays on you during a visit.
- **Snippet:**
```python
if 'n' not in st.session_state:
    st.session_state.n = 0
st.session_state.n += 1
```
- **Snippet explained:** Initialise once, then update; value persists across reruns.
- **Related terms:** [Rerun model (Streamlit)](api_and_deployment_terms.md#rerun-model-streamlit), [gr.State](api_and_deployment_terms.md#grstate)

---

### Shadow deployment
*From module: 10_ml_system_architecture*

- **Technical definition:** Running a new model on real requests in parallel with production but discarding (only logging) its outputs to compare safely.
- **Plain English:** Let the new model watch and guess without affecting users.
- **Example / analogy:** A trainee shadowing an expert.
- **Snippet:**
n/a
- **Snippet explained:** n/a
- **Related terms:** [Canary deployment](api_and_deployment_terms.md#canary-deployment), [Model monitoring](ml_terms.md#model-monitoring)

---

### Share link (Gradio)
*From module: 12_streamlit_gradio*

- **Technical definition:** launch(share=True) creates a temporary public URL tunnelling to your local server.
- **Plain English:** Instant public link to your demo.
- **Example / analogy:** A temporary guest pass.
- **Snippet:**
```python
demo.launch(share=True)
```
- **Snippet explained:** Prints a *.gradio.live URL valid while your process runs; do not use with private data.
- **Related terms:** [gr.Interface](api_and_deployment_terms.md#grinterface), [Deployment (UI apps)](api_and_deployment_terms.md#deployment-ui-apps)

---

### Slim image
*From module: 11_docker*

- **Technical definition:** A Docker image variant (-slim) with the runtime but without most build tools and docs.
- **Plain English:** A trimmed Python image: small but still compatible.
- **Example / analogy:** A carry-on bag rather than a trunk.
- **Snippet:**
```dockerfile
FROM python:3.12-slim
```
- **Snippet explained:** About 120 MB versus about 1 GB for the full python image.
- **Related terms:** [Base image](api_and_deployment_terms.md#base-image)

---

### Smoke test
*From module: 09_api_development*

- **Technical definition:** A quick end-to-end check against a really running service that its basic endpoints respond correctly, typically after deploy.
- **Plain English:** 'Does it turn on and do the basics?'
- **Example / analogy:** Switching on a new appliance to see if smoke comes out.
- **Snippet:**
```python
assert httpx.get(BASE + '/health').status_code == 200
```
- **Snippet explained:** The simplest smoke check.
- **Related terms:** [Test client](api_and_deployment_terms.md#test-client), [Health check](api_and_deployment_terms.md#health-check)

---

### SQL injection
*From module: 10_ml_system_architecture*

- **Technical definition:** An attack where user input is concatenated into SQL text and changes the query; prevented with parameterized queries.
- **Plain English:** Sneaking database commands in through a text field.
- **Example / analogy:** Writing extra orders on a form the clerk blindly copies.
- **Snippet:**
```python
conn.execute('SELECT * FROM predictions WHERE label = ?', (user_text,))
```
- **Snippet explained:** Placeholder keeps the input as data, never as SQL.
- **Related terms:** [Parameterized query](api_and_deployment_terms.md#parameterized-query), [SQLite](api_and_deployment_terms.md#sqlite)

---

### SQLite
*From module: 10_ml_system_architecture*

- **Technical definition:** A serverless SQL database stored in a single file, included in Python's standard library (sqlite3).
- **Plain English:** A small database in one file, no server needed.
- **Example / analogy:** A notebook with tables.
- **Snippet:**
```python
import sqlite3
conn = sqlite3.connect(':memory:')
```
- **Snippet explained:** Creates a temporary in-memory database.
- **Related terms:** [Parameterized query](api_and_deployment_terms.md#parameterized-query), [Prediction log](api_and_deployment_terms.md#prediction-log)

---

### st.button
*From module: 12_streamlit_gradio*

- **Technical definition:** Streamlit widget returning True only during the rerun caused by its click.
- **Plain English:** A button that is True once after the click.
- **Example / analogy:** A doorbell: rings once.
- **Snippet:**
```python
if st.button('Go'):
    st.write('clicked')
```
- **Snippet explained:** The block runs only on the click's rerun.
- **Related terms:** [Rerun model (Streamlit)](api_and_deployment_terms.md#rerun-model-streamlit)

---

### st.cache_data
*From module: 12_streamlit_gradio*

- **Technical definition:** Decorator caching serialisable return values (DataFrames, results), returning a copy per call.
- **Plain English:** Remember answers to repeated computations.
- **Example / analogy:** A cheat sheet of results.
- **Snippet:**
```python
@st.cache_data
def load_csv(p): ...
```
- **Snippet explained:** Reruns with the same arguments reuse the stored result.
- **Related terms:** [st.cache_resource](api_and_deployment_terms.md#stcache_resource)

---

### st.cache_resource
*From module: 12_streamlit_gradio*

- **Technical definition:** Decorator caching a function's returned object (models, connections) once per process and sharing it across reruns and users.
- **Plain English:** Load the heavy thing once and reuse it.
- **Example / analogy:** Keeping the oven hot.
- **Snippet:**
```python
@st.cache_resource
def get_model():
    return load_model()
```
- **Snippet explained:** First call loads; later calls return the same object instantly.
- **Related terms:** [st.cache_data](api_and_deployment_terms.md#stcache_data), [Session state](api_and_deployment_terms.md#session-state)

---

### st.columns
*From module: 12_streamlit_gradio*

- **Technical definition:** Creates side-by-side layout containers.
- **Plain English:** Splits the page into vertical strips.
- **Example / analogy:** A newspaper layout.
- **Snippet:**
```python
left, right = st.columns(2)
left.write('A')
```
- **Snippet explained:** Returns two column objects; call widgets on them or use `with`.
- **Related terms:** [st.tabs](api_and_deployment_terms.md#sttabs), [st.sidebar](api_and_deployment_terms.md#stsidebar)

---

### st.error
*From module: 12_streamlit_gradio*

- **Technical definition:** Displays a red error message box; sister functions: st.warning, st.info, st.success.
- **Plain English:** Shows a friendly error to the user.
- **Example / analogy:** A red sign.
- **Snippet:**
```python
try:
    ...
except ValueError as e:
    st.error(str(e))
```
- **Snippet explained:** Converts an exception into a message instead of a crash page.
- **Related terms:** [gr.Error](api_and_deployment_terms.md#grerror), [st.spinner](api_and_deployment_terms.md#stspinner)

---

### st.file_uploader
*From module: 12_streamlit_gradio*

- **Technical definition:** Widget that lets users upload files, returning an UploadedFile (file-like) or None.
- **Plain English:** A file picker.
- **Example / analogy:** An attachment button.
- **Snippet:**
```python
f = st.file_uploader('CSV', type=['csv'])
if f:
    data = f.getvalue()
```
- **Snippet explained:** getvalue() returns the bytes of the uploaded file.
- **Related terms:** [st.form](api_and_deployment_terms.md#stform)

---

### st.form
*From module: 12_streamlit_gradio*

- **Technical definition:** Container grouping widgets so the script only reruns when its submit button is pressed.
- **Plain English:** Fill in everything, then press submit.
- **Example / analogy:** A paper form.
- **Snippet:**
```python
with st.form('f'):
    t = st.text_input('x')
    ok = st.form_submit_button('Go')
```
- **Snippet explained:** No rerun while typing; ok is True after submit.
- **Related terms:** [st.button](api_and_deployment_terms.md#stbutton)

---

### st.set_page_config
*From module: 12_streamlit_gradio*

- **Technical definition:** Sets page title, icon and layout; must be the first Streamlit command.
- **Plain English:** Basic page settings.
- **Example / analogy:** The page's name tag.
- **Snippet:**
```python
st.set_page_config(layout='wide')
```
- **Snippet explained:** Use the whole screen width.
- **Related terms:** [Streamlit](api_and_deployment_terms.md#streamlit)

---

### st.sidebar
*From module: 12_streamlit_gradio*

- **Technical definition:** Left panel container for global controls.
- **Plain English:** A settings drawer.
- **Example / analogy:** A car dashboard side panel.
- **Snippet:**
```python
x = st.sidebar.slider('n', 0, 10)
```
- **Snippet explained:** Slider is rendered in the sidebar; returns its value.
- **Related terms:** [st.columns](api_and_deployment_terms.md#stcolumns)

---

### st.spinner
*From module: 12_streamlit_gradio*

- **Technical definition:** Context manager displaying a spinner message while a block executes.
- **Plain English:** A 'loading...' indicator.
- **Example / analogy:** The hourglass icon.
- **Snippet:**
```python
with st.spinner('Working...'):
    result = slow()
```
- **Snippet explained:** Spinner disappears when the block finishes.
- **Related terms:** [Loading state](api_and_deployment_terms.md#loading-state), [st.error](api_and_deployment_terms.md#sterror)

---

### st.tabs
*From module: 12_streamlit_gradio*

- **Technical definition:** Creates tabbed sections; returns a list of containers.
- **Plain English:** Several pages in one.
- **Example / analogy:** Browser tabs.
- **Snippet:**
```python
t1, t2 = st.tabs(['One','Two'])
with t1:
    st.write('hi')
```
- **Snippet explained:** Only content inside each `with` block appears in that tab.
- **Related terms:** [st.columns](api_and_deployment_terms.md#stcolumns)

---

### Status code
*From module: 09_api_development*

- **Technical definition:** A three-digit number in the response: 2xx success, 3xx redirect, 4xx client error, 5xx server error.
- **Plain English:** A one-number summary of how the request went.
- **Example / analogy:** Like a traffic light on the answer: green 200, amber 4xx 'you did something wrong', red 5xx 'we broke'.
- **Snippet:**
```python
return jsonify(error="not found"), 404
```
- **Snippet explained:** Flask's (body, status) tuple sets the status code.
- **Related terms:** [HTTP](api_and_deployment_terms.md#http), [422 Unprocessable Content](api_and_deployment_terms.md#422-unprocessable-content), [HTTP 500](api_and_deployment_terms.md#http-500)

---

### Streamlit
*From module: 12_streamlit_gradio*

- **Technical definition:** Open-source Python framework that turns scripts into interactive web apps by rerunning the script on each interaction.
- **Plain English:** Write a Python script, get a web page.
- **Example / analogy:** A notebook that becomes an app.
- **Snippet:**
```python
import streamlit as st
st.title('Hi')
```
- **Snippet explained:** Run with `streamlit run app.py`; st.title draws a heading.
- **Related terms:** [Rerun model (Streamlit)](api_and_deployment_terms.md#rerun-model-streamlit), [Widget](api_and_deployment_terms.md#widget)

---

### Streamlit theme (config.toml)
*From module: 12_streamlit_gradio*

- **Technical definition:** Appearance settings in .streamlit/config.toml under [theme].
- **Plain English:** Colours and fonts for the app.
- **Example / analogy:** Paint colours.
- **Snippet:**
```toml
[theme]
primaryColor = "#0b6e4f"
```
- **Snippet explained:** Changes button and link accent colour.
- **Related terms:** [st.set_page_config](api_and_deployment_terms.md#stset_page_config)

---

### Structured logging
*From module: 10_ml_system_architecture*

- **Technical definition:** Emitting each log record as machine-readable key/value data (usually one JSON line) with fields like request_id and latency_ms.
- **Plain English:** Logs that computers can search and count.
- **Example / analogy:** A form with labelled fields instead of a paragraph.
- **Snippet:**
```python
logger.info('prediction', extra={'fields': {'request_id': rid, 'latency_ms': ms}})
```
- **Snippet explained:** Attaches fields that a JSON formatter writes out.
- **Related terms:** [Request ID](api_and_deployment_terms.md#request-id), [Model monitoring](ml_terms.md#model-monitoring)

---

### Tail latency
*From module: 10_ml_system_architecture*

- **Technical definition:** The slowest requests (p95/p99); they drive timeouts and user frustration and are hidden by averages.
- **Plain English:** How bad the worst slow answers are.
- **Example / analogy:** The longest waits in a queue.
- **Snippet:**
```python
np.percentile(lat, 99)
```
- **Snippet explained:** 99% of requests were faster than this.
- **Related terms:** [Latency](llm_terms.md#latency)

---

### Test client
*From module: 09_api_development*

- **Technical definition:** A helper that calls a web app in-process without a network server (Flask `app.test_client()`, FastAPI `TestClient`).
- **Plain English:** A pretend browser used inside tests.
- **Example / analogy:** A flight simulator instead of a real flight.
- **Snippet:**
```python
client = app.test_client()
resp = client.post('/predict', json={'text': 'hi'})
```
- **Snippet explained:** Sends a POST in-process; `resp.status_code`, `resp.get_json()` hold the result.
- **Related terms:** [pytest](python_syntax.md#pytest), [Smoke test](api_and_deployment_terms.md#smoke-test)

---

### Timeout (HTTP)
*From module: 12_streamlit_gradio*

- **Technical definition:** Maximum time to wait for a network call before raising an exception.
- **Plain English:** Do not wait forever.
- **Example / analogy:** Hanging up after 10 rings.
- **Snippet:**
```python
requests.get(url, timeout=3)
```
- **Snippet explained:** Raises requests.Timeout if no answer in 3 seconds.
- **Related terms:** [API client (UI to API)](api_and_deployment_terms.md#api-client-ui-to-api)

---

### Transient error
*From module: 08_agent_harness*

- **Technical definition:** A temporary failure (timeout, 429, 5xx, dropped connection) that may succeed if retried.
- **Plain English:** A hiccup that may go away.
- **Example / analogy:** A busy phone line.
- **Snippet:**
```python
RETRYABLE = (TimeoutError, ConnectionError)
```
- **Snippet explained:** Only these exception types are retried.
- **Related terms:** [Exponential backoff](api_and_deployment_terms.md#exponential-backoff), [Permanent error](api_and_deployment_terms.md#permanent-error)

---

### Uvicorn
*From module: 09_api_development*

- **Technical definition:** A fast ASGI server (built on uvloop/httptools when available) used to run FastAPI/Starlette apps; supports --workers and --reload.
- **Plain English:** The program that listens on the port and runs your FastAPI app.
- **Example / analogy:** The shop front that passes orders to the kitchen.
- **Snippet:**
```bash
uvicorn fastapi_app:app --port 8000 --workers 2
```
- **Snippet explained:** Serves `app` from fastapi_app.py with 2 processes.
- **Related terms:** [ASGI](api_and_deployment_terms.md#asgi), [Gunicorn](api_and_deployment_terms.md#gunicorn), [Worker](api_and_deployment_terms.md#worker)

---

### Validation
*From module: 09_api_development*

- **Technical definition:** Checking that input has the expected type, range and size before using it; rejecting it with a clear client error otherwise.
- **Plain English:** Making sure what came in is acceptable before you act on it.
- **Example / analogy:** Airport security before boarding.
- **Snippet:**
```python
if not isinstance(text, str) or not text.strip():
    return jsonify(error="text required"), 422
```
- **Snippet explained:** A hand-written validation check as done in Flask.
- **Related terms:** [Pydantic](api_and_deployment_terms.md#pydantic), [422 Unprocessable Content](api_and_deployment_terms.md#422-unprocessable-content)

---

### Virtual machine (vs container)
*From module: 11_docker*

- **Technical definition:** Full emulated hardware running a guest operating system on a hypervisor.
- **Plain English:** A whole pretend computer; heavier and slower to start than a container.
- **Example / analogy:** A separate house versus an apartment.
- **Snippet:**
n/a
- **Snippet explained:** 
- **Related terms:** [Container](api_and_deployment_terms.md#container), [Namespace and cgroup](api_and_deployment_terms.md#namespace-and-cgroup)

---

### Volume (named volume)
*From module: 11_docker*

- **Technical definition:** Docker-managed persistent storage mounted into containers, surviving container removal.
- **Plain English:** A storage drive that outlives the container.
- **Example / analogy:** An external hard drive.
- **Snippet:**
```bash
docker run -v api-data:/data toy-api:v1
```
- **Snippet explained:** Files written to /data are kept in the api-data volume.
- **Related terms:** [Bind mount](api_and_deployment_terms.md#bind-mount), [Container](api_and_deployment_terms.md#container)

---

### Widget
*From module: 12_streamlit_gradio*

- **Technical definition:** Interactive UI element (text input, slider, button) that returns its current value in Streamlit.
- **Plain English:** A control the user can touch.
- **Example / analogy:** A form field.
- **Snippet:**
```python
name = st.text_input('Name', 'Ada')
```
- **Snippet explained:** Returns the typed string on each rerun.
- **Related terms:** [Streamlit](api_and_deployment_terms.md#streamlit), [Rerun model (Streamlit)](api_and_deployment_terms.md#rerun-model-streamlit)

---

### WORKDIR
*From module: 11_docker*

- **Technical definition:** Sets the working directory for subsequent instructions and for the container at run time; creates it if needed.
- **Plain English:** The folder you are 'in' inside the image.
- **Example / analogy:** cd, but permanent.
- **Snippet:**
```dockerfile
WORKDIR /code
```
- **Snippet explained:** Later COPY . . and CMD use /code as the current directory.
- **Related terms:** [COPY (Dockerfile)](api_and_deployment_terms.md#copy-dockerfile), [Dockerfile](api_and_deployment_terms.md#dockerfile)

---

### Worker
*From module: 09_api_development*

- **Technical definition:** An operating-system process (or thread) launched by the server to handle requests; each process has its own memory, so state and models are duplicated per worker.
- **Plain English:** One copy of your app doing the work; more workers = more requests in parallel.
- **Example / analogy:** Extra checkout lanes in a shop - each needs its own cashier (and its own copy of the price list).
- **Snippet:**
```bash
curl localhost:5001/whoami   # pid differs between workers
```
- **Snippet explained:** Different `pid` values prove separate processes.
- **Related terms:** [Gunicorn](api_and_deployment_terms.md#gunicorn), [Uvicorn](api_and_deployment_terms.md#uvicorn), [Concurrency](api_and_deployment_terms.md#concurrency)

---

### WSGI
*From module: 09_api_development*

- **Technical definition:** Web Server Gateway Interface: the synchronous Python standard where a server calls app(environ, start_response) and receives an iterable of bytes.
- **Plain English:** The old agreed handshake between a Python web app and the server running it.
- **Example / analogy:** A standard plug shape so any appliance fits any socket.
- **Snippet:**
```python
def app(environ, start_response):
    start_response("200 OK", [("Content-Type", "text/plain")])
    return [b"hello"]
```
- **Snippet explained:** The smallest WSGI app.
- **Related terms:** [ASGI](api_and_deployment_terms.md#asgi), [Gunicorn](api_and_deployment_terms.md#gunicorn), [Flask](api_and_deployment_terms.md#flask)

---

### YAML
*From module: 11_docker*

- **Technical definition:** Indentation-based data format used for configuration (mappings, lists, scalars).
- **Plain English:** A readable config format where spacing matters.
- **Example / analogy:** A nested outline.
- **Snippet:**
```yaml
services:
  web:
    ports:
      - "80:80"
```
- **Snippet explained:** Mapping keys nested by 2-space indentation; '-' marks list items.
- **Related terms:** [Compose file](api_and_deployment_terms.md#compose-file)

---
