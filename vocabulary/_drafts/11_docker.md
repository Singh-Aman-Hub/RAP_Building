# Vocabulary draft - 11 Docker

### Container
<!-- cat: api -->
- **Technical definition:** An isolated process (or group) running on the host kernel with its own filesystem, network and process namespace, started from an image.
- **Plain English:** A lightweight sealed box that runs your app with everything it needs.
- **Example / analogy:** Like a locked apartment in a shared building.
- **Snippet:**
```bash
docker run -d --name web -p 8000:8000 toy-api:v1
```
- **Snippet explained:** Starts one container from image toy-api:v1 in the background, mapping host port 8000.
- **Related terms:** [[Image]], [[Virtual machine (vs container)]], [[Namespace and cgroup]]

### Image
<!-- cat: api -->
- **Technical definition:** A read-only, layered template containing a filesystem and metadata (default command, env, ports) used to create containers.
- **Plain English:** The recipe-plus-ingredients package your container is created from.
- **Example / analogy:** A class vs object: image = class, container = instance.
- **Snippet:**
```bash
docker build -t toy-api:v1 .
```
- **Snippet explained:** Builds an image named toy-api with tag v1 from the Dockerfile in the current folder.
- **Related terms:** [[Container]], [[Dockerfile]], [[Layer (Docker)]]

### Virtual machine (vs container)
<!-- cat: api -->
- **Technical definition:** Full emulated hardware running a guest operating system on a hypervisor.
- **Plain English:** A whole pretend computer; heavier and slower to start than a container.
- **Example / analogy:** A separate house versus an apartment.
- **Snippet:**
n/a
- **Snippet explained:** 
- **Related terms:** [[Container]], [[Namespace and cgroup]]

### Namespace and cgroup
<!-- cat: api -->
- **Technical definition:** Linux kernel features: namespaces limit what a process can see (PIDs, network, mounts); cgroups limit resources (CPU, memory).
- **Plain English:** The kernel tricks that make a container feel isolated and capped.
- **Example / analogy:** Namespaces = walls; cgroups = utility meters.
- **Snippet:**
```bash
docker run --memory=256m --cpus=0.5 toy-api:v1
```
- **Snippet explained:** Limits the container to 256 MB RAM and half a CPU (enforced via cgroups).
- **Related terms:** [[Container]]

### Dockerfile
<!-- cat: api -->
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
- **Related terms:** [[Image]], [[Layer (Docker)]], [[Build context]]

### Build context
<!-- cat: api -->
- **Technical definition:** The set of files (folder path given to docker build, minus .dockerignore) sent to the Docker daemon and available to COPY.
- **Plain English:** The folder Docker is allowed to see while building.
- **Example / analogy:** The ingredients on the kitchen counter - anything not there cannot be used.
- **Snippet:**
```bash
docker build -t app .
```
- **Snippet explained:** The trailing dot is the build context: the current directory.
- **Related terms:** [[Dockerfile]], [[.dockerignore]]

### Layer (Docker)
<!-- cat: api -->
- **Technical definition:** A read-only filesystem diff created by a Dockerfile instruction; layers are stacked and cached.
- **Plain English:** One step of the build saved so it can be reused.
- **Example / analogy:** Pages in a book; changing page 3 means reprinting pages 3 onward.
- **Snippet:**
```bash
docker history toy-api:v1
```
- **Snippet explained:** Lists the layers of an image with the instruction and size of each.
- **Related terms:** [[Build cache (layer cache)]], [[Dockerfile]], [[Multi-stage build]]

### Build cache (layer cache)
<!-- cat: api -->
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
- **Related terms:** [[Layer (Docker)]], [[Requirements file]]

### FROM
<!-- cat: api -->
- **Technical definition:** Dockerfile instruction selecting the base image (and starting a build stage).
- **Plain English:** Which image you build on top of.
- **Example / analogy:** Choosing the starting template.
- **Snippet:**
```dockerfile
FROM python:3.12-slim
```
- **Snippet explained:** Uses Debian-based Python 3.12 with minimal extras.
- **Related terms:** [[Base image]], [[Multi-stage build]]

### Base image
<!-- cat: api -->
- **Technical definition:** The image named in FROM that provides the OS layer and runtimes (e.g., python:3.12-slim).
- **Plain English:** The foundation your image sits on.
- **Example / analogy:** The chassis a car body is mounted on.
- **Snippet:**
```dockerfile
FROM python:3.12-alpine
```
- **Snippet explained:** Alpine is tiny but uses musl libc, so many ML wheels are unavailable; slim is the safe default.
- **Related terms:** [[FROM]], [[Slim image]]

### Slim image
<!-- cat: api -->
- **Technical definition:** A Docker image variant (-slim) with the runtime but without most build tools and docs.
- **Plain English:** A trimmed Python image: small but still compatible.
- **Example / analogy:** A carry-on bag rather than a trunk.
- **Snippet:**
```dockerfile
FROM python:3.12-slim
```
- **Snippet explained:** About 120 MB versus about 1 GB for the full python image.
- **Related terms:** [[Base image]]

### WORKDIR
<!-- cat: api -->
- **Technical definition:** Sets the working directory for subsequent instructions and for the container at run time; creates it if needed.
- **Plain English:** The folder you are 'in' inside the image.
- **Example / analogy:** cd, but permanent.
- **Snippet:**
```dockerfile
WORKDIR /code
```
- **Snippet explained:** Later COPY . . and CMD use /code as the current directory.
- **Related terms:** [[COPY (Dockerfile)]], [[Dockerfile]]

### COPY (Dockerfile)
<!-- cat: api -->
- **Technical definition:** Copies files from the build context (or another stage with --from) into the image.
- **Plain English:** Puts your files into the image.
- **Example / analogy:** Packing files into the box.
- **Snippet:**
```dockerfile
COPY app ./app
```
- **Snippet explained:** Copies the app folder into /code/app (relative to WORKDIR).
- **Related terms:** [[Build context]], [[Multi-stage build]], [[WORKDIR]]

### RUN (Dockerfile)
<!-- cat: api -->
- **Technical definition:** Executes a command at build time and commits the result as a new layer.
- **Plain English:** A command that runs while building the image.
- **Example / analogy:** Assembling furniture before shipping.
- **Snippet:**
```dockerfile
RUN pip install --no-cache-dir -r requirements.txt
```
- **Snippet explained:** Installs dependencies; --no-cache-dir avoids storing pip's cache in the layer.
- **Related terms:** [[CMD vs ENTRYPOINT]], [[Layer (Docker)]]

### CMD vs ENTRYPOINT
<!-- cat: api -->
- **Technical definition:** ENTRYPOINT defines the fixed executable; CMD supplies default arguments (or the default command) and is replaced by arguments to docker run.
- **Plain English:** ENTRYPOINT is the program; CMD is its default options.
- **Example / analogy:** ENTRYPOINT = 'pizza oven', CMD = 'default topping'.
- **Snippet:**
```dockerfile
ENTRYPOINT ["uvicorn"]
CMD ["app.main:app", "--host", "0.0.0.0"]
```
- **Snippet explained:** docker run img app.main:app --port 9000 replaces only the CMD part.
- **Related terms:** [[Exec form vs shell form]], [[RUN (Dockerfile)]]

### Exec form vs shell form
<!-- cat: api -->
- **Technical definition:** Exec form is a JSON array run directly (no shell); shell form is a string run via /bin/sh -c.
- **Plain English:** JSON list = runs your program as PID 1 and receives stop signals properly.
- **Example / analogy:** Calling someone directly versus via a receptionist.
- **Snippet:**
```dockerfile
CMD ["uvicorn", "app.main:app"]   # exec form
CMD uvicorn app.main:app           # shell form
```
- **Snippet explained:** Prefer exec form so SIGTERM from docker stop reaches your process.
- **Related terms:** [[CMD vs ENTRYPOINT]]

### EXPOSE
<!-- cat: api -->
- **Technical definition:** Documents the port an app listens on; does not publish it.
- **Plain English:** A note saying 'this app uses port 8000'.
- **Example / analogy:** A sign on the door that does not unlock it.
- **Snippet:**
```dockerfile
EXPOSE 8000
```
- **Snippet explained:** Metadata only; use -p 8000:8000 to publish.
- **Related terms:** [[Port mapping]]

### Port mapping
<!-- cat: api -->
- **Technical definition:** Publishing a container port on a host port with -p HOST:CONTAINER.
- **Plain English:** Connects a door on your laptop to a door inside the container.
- **Example / analogy:** Phone extension forwarding.
- **Snippet:**
```bash
docker run -p 9000:8000 toy-api:v1
```
- **Snippet explained:** localhost:9000 on the host reaches port 8000 inside.
- **Related terms:** [[EXPOSE]], [[0.0.0.0 vs 127.0.0.1 binding]]

### 0.0.0.0 vs 127.0.0.1 binding
<!-- cat: api -->
- **Technical definition:** 127.0.0.1 listens only on loopback (inside the container); 0.0.0.0 listens on all interfaces so published ports can reach it.
- **Plain English:** Inside Docker, servers must listen on 0.0.0.0 or nobody outside can connect.
- **Example / analogy:** Speaking inside a room versus using the front door.
- **Snippet:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
- **Snippet explained:** Makes the server reachable through -p.
- **Related terms:** [[Port mapping]]

### Environment variable (Docker)
<!-- cat: api -->
- **Technical definition:** Key=value configuration injected into a container process via ENV, -e, --env-file or Compose environment.
- **Plain English:** Settings passed to your app without changing code.
- **Example / analogy:** Dials on a machine.
- **Snippet:**
```bash
docker run -e MODEL_NAME=v2 --env-file sample.env toy-api:v1
```
- **Snippet explained:** -e sets one variable; --env-file loads many; code reads them with os.getenv.
- **Related terms:** [[Secret]], [[Compose file]]

### Volume (named volume)
<!-- cat: api -->
- **Technical definition:** Docker-managed persistent storage mounted into containers, surviving container removal.
- **Plain English:** A storage drive that outlives the container.
- **Example / analogy:** An external hard drive.
- **Snippet:**
```bash
docker run -v api-data:/data toy-api:v1
```
- **Snippet explained:** Files written to /data are kept in the api-data volume.
- **Related terms:** [[Bind mount]], [[Container]]

### Bind mount
<!-- cat: api -->
- **Technical definition:** Mounts a host directory or file into a container at a path.
- **Plain English:** Shares a folder from your computer into the container.
- **Example / analogy:** A shared network drive.
- **Snippet:**
```bash
docker run -v "$(pwd)/host-data:/data" toy-api:v1
```
- **Snippet explained:** The container sees and writes your host-data folder directly.
- **Related terms:** [[Volume (named volume)]]

### Container networking
<!-- cat: api -->
- **Technical definition:** Containers attach to virtual networks; on user-defined (Compose) networks, containers resolve each other by service name via Docker DNS.
- **Plain English:** How containers talk to each other and the outside world.
- **Example / analogy:** An office intercom with names.
- **Snippet:**
```bash
docker compose exec api python -c "import socket;print(socket.gethostbyname('redis'))"
```
- **Snippet explained:** Resolves the redis service name to its container IP.
- **Related terms:** [[Docker Compose]], [[Port mapping]]

### Docker Compose
<!-- cat: api -->
- **Technical definition:** Tool and YAML format to define and run multi-container applications (services, networks, volumes).
- **Plain English:** One file and one command to run several containers.
- **Example / analogy:** A conductor's score for an orchestra.
- **Snippet:**
```bash
docker compose up --build -d
```
- **Snippet explained:** Builds images and starts all services in the background.
- **Related terms:** [[Compose file]], [[Service (Compose)]], [[Container networking]]

### Compose file
<!-- cat: api -->
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
- **Related terms:** [[Docker Compose]], [[YAML]]

### Service (Compose)
<!-- cat: api -->
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
- **Related terms:** [[Compose file]], [[Container networking]]

### YAML
<!-- cat: api -->
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
- **Related terms:** [[Compose file]]

### depends_on
<!-- cat: api -->
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
- **Related terms:** [[Health check (Docker)]]

### Health check (Docker)
<!-- cat: api -->
- **Technical definition:** A command run periodically inside a container; exit 0 = healthy, 1 = unhealthy.
- **Plain English:** A probe asking 'are you really working?'
- **Example / analogy:** A doctor's pulse check.
- **Snippet:**
```dockerfile
HEALTHCHECK --interval=10s CMD python -c "import urllib.request as u;u.urlopen('http://localhost:8000/health')" || exit 1
```
- **Snippet explained:** Calls /health every 10 s; failure marks the container unhealthy.
- **Related terms:** [[depends_on]], [[Liveness vs readiness]]

### Liveness vs readiness
<!-- cat: api -->
- **Technical definition:** Liveness: process is alive; readiness: it can serve real work (dependencies ready).
- **Plain English:** Alive means running; ready means able to help.
- **Example / analogy:** A restaurant open (alive) vs kitchen ready.
- **Snippet:**
```python
@app.get('/ready')
def ready(): ...  # checks redis.ping()
```
- **Snippet explained:** Returns 503 until dependencies respond.
- **Related terms:** [[Health check (Docker)]]

### Docker logs
<!-- cat: api -->
- **Technical definition:** Stdout/stderr of the container's main process, shown by docker logs.
- **Plain English:** What your app printed.
- **Example / analogy:** The black box recorder.
- **Snippet:**
```bash
docker logs -f --tail 50 toy
```
- **Snippet explained:** Follows the last 50 lines of the container's output live.
- **Related terms:** [[Exit code (container)]], [[docker exec]]

### docker exec
<!-- cat: api -->
- **Technical definition:** Runs an extra command inside a running container (often a shell).
- **Plain English:** Open a terminal inside the container.
- **Example / analogy:** Walking into the apartment.
- **Snippet:**
```bash
docker exec -it toy sh
```
- **Snippet explained:** Opens an interactive shell; then try ls, env, ps.
- **Related terms:** [[Docker logs]]

### Exit code (container)
<!-- cat: api -->
- **Technical definition:** Numeric status of a stopped container's main process: 0 success, 1 app error, 137 killed (SIGKILL/OOM), 139 segfault.
- **Plain English:** A number telling you why the container stopped.
- **Example / analogy:** A report card.
- **Snippet:**
```bash
docker inspect toy --format '{{.State.ExitCode}}'
```
- **Snippet explained:** Prints the exit code of the container.
- **Related terms:** [[Docker logs]]

### Multi-stage build
<!-- cat: api -->
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
- **Related terms:** [[Slim image]], [[Layer (Docker)]]

### .dockerignore
<!-- cat: api -->
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
- **Related terms:** [[Build context]]

### Non-root user (container)
<!-- cat: api -->
- **Technical definition:** Running the container process as an unprivileged user via USER.
- **Plain English:** The app does not run as administrator.
- **Example / analogy:** A guest account instead of admin.
- **Snippet:**
```dockerfile
RUN useradd --uid 10001 appuser
USER appuser
```
- **Snippet explained:** Limits damage if the app is compromised.
- **Related terms:** [[Secret]]

### Secret
<!-- cat: api -->
- **Technical definition:** Sensitive value (API key, password) that must not be baked into images or committed.
- **Plain English:** Passwords and keys you must keep out of the image.
- **Example / analogy:** A house key you never leave under the doormat.
- **Snippet:**
```bash
docker run -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" app
```
- **Snippet explained:** Injected at run time; not stored in any layer.
- **Related terms:** [[Environment variable (Docker)]]

### Requirements file
<!-- cat: api -->
- **Technical definition:** requirements.txt: list of Python dependencies, ideally pinned (name==version), installed with pip -r.
- **Plain English:** The shopping list of Python packages.
- **Example / analogy:** A bill of materials.
- **Snippet:**
```text
fastapi==0.146.0
uvicorn[standard]==0.54.0
```
- **Snippet explained:** Pinned versions give reproducible builds.
- **Related terms:** [[Build cache (layer cache)]]
